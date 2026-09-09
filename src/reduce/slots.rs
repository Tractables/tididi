//! The value store of a marginal level, as slots.
//!
//! A marginal level holds one value per node in a store its parent's pairs
//! index. The reduction passes that rewrite such a level — the slot pruner, the
//! same-left-child pair fusion — all need the same four things: a hashable key
//! for a stored value, a way to append a value as a new slot, a dedup map from
//! value to slot, and the set of slots a parent still references. They are here
//! rather than in any one pass because all of them use all of them.

use rustc_hash::{FxHashMap, FxHashSet};

use crate::value_fold::{Count, CountRead, IntFold, STREAM_OVERFLOW};
use crate::diagram::marg::refs::ChildSide;
use crate::diagram::{BigSide, InputPair, MargSide, NodeIdx, TddLevel, ValueRef};
use crate::engine::{ApplyBudget, Engine};
use crate::error::ApplyError;

/// Append `key` as a NEW slot on a marginal store, fallibly. Returns the new
/// slot index.
///
/// One home for the store's OVERFLOW convention: `counts[i]` holds the small
/// count, or the `u128::MAX` sentinel meaning "the real value is `big`'s entry
/// for slot `i`". `big` is a sparse slot-keyed table allocated on the first
/// overflow, so a `Small` push writes nothing there (an absent entry IS the
/// "fits the fast lane" encoding) and a `Big` push records exactly one entry.
///
/// Every growth is budget-tracked (`try_push` / [`BigSide::try_insert`] under
/// [`ApplyBudget`], the same accounting the fast column uses), so an
/// over-budget store push surfaces as `ApplyError::OverBudget` rather than
/// aborting.
///
/// This is the minting half of every production slot path: a caller that wants
/// one slot per distinct count checks [`SlotInterner`]'s map first and only pushes
/// on a miss (`apply_p_fusion`).
pub(crate) fn push_count_key(
    eng: &Engine,
    counts: &mut Vec<u128>,
    big: &mut Option<BigSide>,
    key: &Count,
) -> Result<u32, ApplyError> {
    let lim = eng.limits();
    let new_idx = counts.len() as u32;
    match key {
        Count::Fast(c) => {
            // A small count landing exactly on the sentinel would be re-read as
            // OVERFLOW with no `big` entry behind it; producers must route that
            // value to `Big` (see `sum_marginal_counts` and its pinned test).
            debug_assert!(*c != u128::MAX, "small count must not alias the overflow sentinel");
            lim.try_push(counts, *c)?;
        }
        Count::Big(v) => {
            lim.try_push(counts, u128::MAX)?;
            big.get_or_insert_with(BigSide::default)
                .try_insert::<ApplyBudget>(eng, counts.len() - 1, v.clone())?;
        }
    }
    Ok(new_idx)
}

// ── SlotInterner ─────────────────────────────────────────────────────────────

/// Seeded dedup map from [`Count`] to slot index, used by the p-fusion and
/// slot-prune compaction paths to keep marginal stores at one slot per value.
pub(crate) struct SlotInterner {
    pub(super) map: FxHashMap<Count, u32>,
}

impl SlotInterner {
    /// Create an empty interner.
    pub(crate) fn new() -> Self {
        Self { map: FxHashMap::default() }
    }

    /// Seed from an existing `(counts, big)` store so that subsequent lookups
    /// reuse existing slots for equal values.
    /// Duplicate counts in the seed are collapsed to the first occurrence
    /// (same dedup semantics as `dedup_fresh_store`).
    pub(crate) fn seed(
        &mut self,
        counts: &[u128],
        big: Option<&BigSide>,
    ) {
        for i in 0..counts.len() {
            let key = count_key_at(counts, big, i);
            self.map.entry(key).or_insert(i as u32);
        }
    }
}

/// Read the marginal count at `slot` as a `Count`.
///
/// Mirrors the overflow-sentinel convention: `counts[slot] ==
/// STREAM_OVERFLOW` means the real value is `big`'s entry for `slot`.
pub(crate) fn count_key_at(
    counts: &[u128],
    big: Option<&BigSide>,
    slot: usize,
) -> Count {
    let c = counts[slot];
    if c == STREAM_OVERFLOW {
        let b = big
            .and_then(|b| b.get(slot))
            .expect("OVERFLOW sentinel requires a marginal_counts_big entry")
            .clone();
        Count::Big(b)
    } else {
        Count::Fast(c)
    }
}

/// Sum the values at `indices`, each a marginal-side reference.
///
/// A reference is EITHER an inline value (bit-30 clear: the value IS the count,
/// no array load) or a tagged slot (bit-30 set: index `counts`).
/// `ValueRef::from_raw` does that split; its bit-31 assert fires in a debug
/// build if a ZERO sentinel ever reaches here — by design, so the source is
/// localized rather than papered over with a guessed 0.
///
/// The arithmetic is [`IntFold::fold`], the crate's one two-pass integer fold,
/// driven with a constant 1 on the right: `Σ cᵢ` is `Σ (cᵢ × 1)`. That is where
/// the overflow rule lives — including the promotion of a total landing exactly
/// on the sentinel, which would otherwise be stored as "the real value is in
/// the side table" with no side-table entry to find.
pub(crate) fn sum_marginal_counts(
    counts: &[u128],
    big: Option<&BigSide>,
    indices: &[u32],
) -> Count {
    let read = |raw: usize| -> CountRead<'_> {
        match ValueRef::from_raw(MargSide(raw as u32)) {
            ValueRef::Inline(v) => CountRead::Fast(v as u128),
            ValueRef::Slot(s) => read_slot(counts, big, s as usize),
        }
    };
    let pairs = indices
        .iter()
        .map(|&raw| InputPair { left: NodeIdx(raw), right: NodeIdx(0) });
    IntFold::fold(pairs, read, |_| CountRead::Fast(1))
}

/// One stored slot, decoded against the overflow sentinel: `counts[slot] ==
/// STREAM_OVERFLOW` means the real value is `big`'s entry for `slot`.
#[inline]
fn read_slot<'a>(counts: &[u128], big: Option<&'a BigSide>, slot: usize) -> CountRead<'a> {
    if counts[slot] == STREAM_OVERFLOW {
        CountRead::Big(
            big.and_then(|b| b.get(slot))
                .expect("the overflow sentinel requires a marginal_counts_big entry"),
        )
    } else {
        CountRead::Fast(counts[slot])
    }
}

/// Caller-owned scratch for [`referenced_marg_slots`].
///
/// The pass runs once per boundary-marginal level on every slot-prune sweep,
/// and every sweep runs inside the per-merge minimize — so a freshly allocated
/// result `Vec` plus dedup `FxHashSet` per level is pure allocator churn on a
/// workload made of many tiny diagrams. One scratch, cleared per level, reused for
/// the whole sweep.
#[derive(Default)]
pub(crate) struct RefSlotScratch {
    /// The deduped, sorted slot list — the pass's result, borrowed by the caller.
    pub(crate) referenced: Vec<u32>,
    seen: FxHashSet<u32>,
}

impl RefSlotScratch {
    /// Empty both buffers, retaining their allocations. The single clear used
    /// both by [`referenced_marg_slots`] (per level) and by the sweep-lifetime
    /// pool in `minimize::slot_prune` (on take), so a pooled scratch differs
    /// from a fresh one only in capacity.
    pub(crate) fn clear(&mut self) {
        self.referenced.clear();
        self.seen.clear();
    }

    /// Drop the allocation of either buffer whose retained capacity exceeds
    /// `max_bytes`, INDEPENDENTLY per buffer — the retention policy
    /// `minimize::contract::scratch` applies field by field. Both are refilled
    /// from scratch on every use, so a released one costs the next sweep one
    /// reallocation and nothing else.
    pub(crate) fn release_oversized(&mut self, max_bytes: usize) {
        crate::engine::pool::release_if_oversized(&mut self.referenced, max_bytes);
        // `FxHashSet` has no `Vec` shape for `release_if_oversized`; its table is
        // `capacity` u32 entries plus control bytes, so the same element-count
        // bound applies.
        if self.seen.capacity().saturating_mul(std::mem::size_of::<u32>()) > max_bytes {
            self.seen = FxHashSet::default();
        }
    }
}

/// Fill `scratch.referenced` with the slots of a boundary-marginal level that
/// are referenced from `plevel`'s marg-side pair refs (deduped, sorted). Skips
/// ZERO sentinels and inline refs; OOB filtering is the caller's choice.
///
/// Dedup stays hash-based rather than push-then-sort-dedup on purpose: the
/// number of *refs* walked is unbounded (a wide parent level can hold millions
/// of pairs) while the number of *distinct slots* is bounded by the store, so
/// hashing keeps the sort at store size instead of ref-occurrence size.
pub(crate) fn referenced_marg_slots<'a>(
    plevel: &TddLevel,
    side: ChildSide,
    scratch: &'a mut RefSlotScratch,
) -> &'a [u32] {
    scratch.clear();
    let RefSlotScratch { referenced, seen } = scratch;
    for n in 0..plevel.nodes.len() {
        if plevel.nodes[n].is_leaf() {
            continue;
        }
        // Borrowed directly — the old copy-into-a-buffer step was a memcpy of
        // every pair on the level for a read-only walk.
        for p in plevel.pairs_of_idx(n) {
            let raw = match side {
                ChildSide::Right => p.right.0,
                ChildSide::Left => p.left.0,
            };
            if MargSide(raw).is_zero_sentinel() {
                continue;
            }
            if let ValueRef::Slot(s) = ValueRef::from_raw(MargSide(raw))
                && seen.insert(s) {
                    referenced.push(s);
                }
        }
    }
    referenced.sort_unstable();
    referenced
}

#[cfg(test)]
#[path = "slots_sum_tests.rs"]
mod sum_tests;
