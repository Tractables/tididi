//! Shared marginal-slot primitives.
//!
//! These are the count-keyed slot building blocks used across the
//! marginal-canonical machinery: the marginalize path (`weight.rs`), the
//! post-tagger compaction (`minimize/slot_prune.rs`), the same-left-child pair fusion
//! (`minimize/contract/p_fusion.rs`), and the marginal invariant checkers
//! (`validate/marg.rs`). One shared home, no copies.

use crate::engine::Engine;
use num_bigint::BigUint;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::counts::ApplyBudget;
use crate::error::ApplyError;
use crate::diagram::{BigSide, MargSide, ValueRef, Tdd, TddLevel};
use crate::vtree::{VtreeIdx, VtreeNode};

/// Side of a parent's vtree node at which a marginal child sits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ChildSide {
    Left,
    Right,
}

/// Counts and big counts for a marginal-level slot, hashable / orderable
/// so we can group equal-count slots.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) enum CountKey {
    Small(u128),
    Big(BigUint),
}

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
    key: &CountKey,
) -> Result<u32, ApplyError> {
    let lim = eng.limits();
    let new_idx = counts.len() as u32;
    match key {
        CountKey::Small(c) => {
            // A small count landing exactly on the sentinel would be re-read as
            // OVERFLOW with no `big` entry behind it; producers must route that
            // value to `Big` (see `sum_marginal_counts` and its pinned test).
            debug_assert!(*c != u128::MAX, "small count must not alias the overflow sentinel");
            lim.try_push(counts, *c)?;
        }
        CountKey::Big(v) => {
            lim.try_push(counts, u128::MAX)?;
            big.get_or_insert_with(BigSide::default)
                .try_insert::<ApplyBudget>(eng, counts.len() - 1, v.clone())?;
        }
    }
    Ok(new_idx)
}

/// Apply `f` to every reference the nodes of `level` hold on `side`.
///
/// This is the ONE home for the side→field mapping: `ChildSide::Left` means
/// `pair.left.0` for a multi-pair node and `node.a` for an inline one;
/// `ChildSide::Right` means `pair.right.0` / `node.b`. (An inline node stores
/// its single pair in its own `(a, b)` words — and inlining requires `b` to
/// carry no `LEAF_BIT`, so `b` is a plain index there, exactly like
/// `pair.right.0`.) Leaves and tombstones hold no refs and are skipped.
///
/// Used by every pass that rewrites one side's refs through a remap table:
/// slot-prune's parent-ref rewrite (integer and weighted) and the
/// content-twin grandparent rewrite.
///
/// NOT usable by `minimize::prune`'s node-index remap: that loop filters each
/// node on a `reachable` bitmap and rewrites BOTH sides in a single visit —
/// a different traversal, not a `side` instantiation of this one. Don't try to
/// fold it in here.
#[inline]
pub(crate) fn for_each_side_ref_mut(
    level: &mut crate::diagram::TddLevel,
    side: ChildSide,
    mut f: impl FnMut(&mut u32),
) {
    for ni in 0..level.nodes.len() {
        if level.nodes[ni].is_leaf() {
            continue;
        }
        if level.nodes[ni].is_multi() {
            for p in level.pairs_mut(ni) {
                f(match side {
                    ChildSide::Left => &mut p.left.0,
                    ChildSide::Right => &mut p.right.0,
                });
            }
        } else {
            let node = &mut level.nodes[ni];
            f(match side {
                ChildSide::Left => &mut node.a,
                ChildSide::Right => &mut node.b,
            });
        }
    }
}

/// Rewrite every reference `level` holds on `side` through `remap`, indexed by
/// the cells of the child level `view` describes.
///
/// The typed sibling of [`for_each_side_ref_mut`]: the caller says which child
/// level the refs point at and what happened to its cells, and the encoding is
/// [`SideView::remap`]'s business. Used by slot-prune's parent rewrite and by
/// the content-twin grandparent rewrite.
#[inline]
pub(crate) fn remap_side_refs(
    level: &mut crate::diagram::TddLevel,
    side: ChildSide,
    view: crate::diagram::SideView,
    remap: &[u32],
) {
    for_each_side_ref_mut(level, side, |r| {
        *r = view.remap(crate::diagram::NodeIdx(*r), remap).0;
    });
}

/// Locate the side at which `child` sits in `parent`.
fn side_of(tdd: &Tdd, parent: VtreeIdx, child: VtreeIdx) -> ChildSide {
    match tdd.vtree.node(parent) {
        VtreeNode::Internal { left, right, .. } => {
            if *left == child {
                ChildSide::Left
            } else {
                debug_assert_eq!(*right, child, "child must be left or right of parent");
                ChildSide::Right
            }
        }
        _ => panic!("parent must be internal vtree node"),
    }
}

/// The boundary-marginal test for ONE vtree node: `Some((v, parent, side))`
/// iff `v`'s level is marginal and its vtree parent's level is not. This is the
/// single definition of "boundary marginal level" — both collectors below are
/// just different traversals feeding it.
#[inline]
fn boundary_entry(tdd: &Tdd, v: VtreeIdx) -> Option<(VtreeIdx, VtreeIdx, ChildSide)> {
    if !tdd.levels[v.idx()].is_marginal() {
        return None;
    }
    let parent = tdd.vtree.node(v).parent()?; // root: no parent
    if tdd.levels[parent.idx()].is_marginal() {
        return None; // deep marginal: parent also marginal
    }
    Some((v, parent, side_of(tdd, parent, v)))
}

/// Fill `out` with every boundary marginal level and its non-marginal parent.
pub(crate) fn boundary_marginal_levels_into(
    tdd: &Tdd,
    out: &mut Vec<(VtreeIdx, VtreeIdx, ChildSide)>,
) {
    out.clear();
    out.extend((0..tdd.levels.len()).filter_map(|i| boundary_entry(tdd, VtreeIdx(i as u32))));
}

/// Iterate boundary marginal levels with their non-marginal parent.
pub(crate) fn boundary_marginal_levels(tdd: &Tdd) -> Vec<(VtreeIdx, VtreeIdx, ChildSide)> {
    let mut out = Vec::new();
    boundary_marginal_levels_into(tdd, &mut out);
    out
}

/// Fill `out` with the boundary marginal levels **whose parent is in
/// `parents`** — the same triples `boundary_marginal_levels` would yield, in
/// the same (ascending marginal-child index) order, restricted to that parent
/// set.
///
/// Why this exists: a boundary's parent is by definition the vtree parent of
/// its marginal level, so a caller that already knows the parents it cares
/// about can reach their (at most two) boundaries directly. The all-levels scan
/// costs O(levels) *per call*, and `apply_p_fusion_inner` is called once per
/// marginal-boundary parent inside the contract fixpoint — turning a per-parent
/// constant into an O(parents x levels) sweep over the whole diagram, plus a
/// throwaway `Vec` each time, to keep at most two entries.
pub(crate) fn boundary_marginal_levels_of(
    tdd: &Tdd,
    parents: &[VtreeIdx],
    out: &mut Vec<(VtreeIdx, VtreeIdx, ChildSide)>,
) {
    out.clear();
    for &p in parents {
        if let VtreeNode::Internal { left, right, .. } = tdd.vtree.node(p) {
            out.extend(boundary_entry(tdd, *left));
            out.extend(boundary_entry(tdd, *right));
        }
    }
    // Restore the all-levels traversal's ordering and its once-per-level
    // property (a repeated parent in `parents` would otherwise yield its
    // boundaries twice). A marginal level has exactly one boundary entry, so
    // keying both on the child index is exact.
    out.sort_unstable_by_key(|&(v, _, _)| v.0);
    out.dedup_by_key(|&mut (v, _, _)| v.0);
}

// ── SlotInterner ─────────────────────────────────────────────────────────────

/// Seeded dedup map from [`CountKey`] to slot index, used by the p-fusion and
/// slot-prune compaction paths to keep marginal stores at one slot per value.
pub(crate) struct SlotInterner {
    pub(super) map: FxHashMap<CountKey, u32>,
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

/// Read the marginal count at `slot` as a `CountKey`.
///
/// Mirrors the OVERFLOW-sentinel convention: `counts[slot] == u128::MAX`
/// means the real value is `big`'s entry for `slot`.
pub(crate) fn count_key_at(
    counts: &[u128],
    big: Option<&BigSide>,
    slot: usize,
) -> CountKey {
    const OVERFLOW: u128 = u128::MAX;
    let c = counts[slot];
    if c == OVERFLOW {
        let b = big
            .and_then(|b| b.get(slot))
            .expect("OVERFLOW sentinel requires a marginal_counts_big entry")
            .clone();
        CountKey::Big(b)
    } else {
        CountKey::Small(c)
    }
}

/// Sum marginal counts at `indices`. Returns a CountKey (Small or Big).
pub(crate) fn sum_marginal_counts(
    counts: &[u128],
    big: Option<&BigSide>,
    indices: &[u32],
) -> CountKey {
    use crate::diagram::ValueRef;
    const OVERFLOW: u128 = u128::MAX;
    // Decode one marg-side ref to its contributing count: a ref is EITHER an
    // inline count (bit-30 clear: the value IS the count, no array load) or a
    // tagged slot (bit-30 set: index `counts`). `ValueRef::from_raw` does this
    // split; its bit-31 assert will fire (debug) if a ZERO sentinel ever
    // reaches here — by design, so the source is localized rather than papered
    // over with a guessed 0.
    let load = |i: u32| -> u128 {
        match ValueRef::from_raw(MargSide(i)) {
            ValueRef::Inline(v) => v as u128,
            ValueRef::Slot(s) => counts[s as usize],
        }
    };
    // First pass: try all-small sum without overflow. If any contribution is the
    // OVERFLOW sentinel or the sum would overflow u128, fall through to BigUint.
    let mut acc: u128 = 0;
    let mut any_big = false;
    for &i in indices {
        let c = load(i);
        if c == OVERFLOW {
            any_big = true;
            break;
        }
        match acc.checked_add(c) {
            Some(s) => acc = s,
            None => {
                any_big = true;
                break;
            }
        }
    }
    // `acc == OVERFLOW` falls through to the BigUint path even though the sum
    // fits u128: a `Small(u128::MAX)` would later be *stored* as the OVERFLOW
    // sentinel without a `marginal_counts_big` entry, and the next
    // `count_key_at` read of that slot would panic.
    if !any_big && acc != OVERFLOW {
        return CountKey::Small(acc);
    }
    // BigUint sum. An inline ref's value (≤ MARG_INLINE_MAX) is never OVERFLOW,
    // so only tagged slots can route into `big`.
    let mut big_acc = BigUint::default();
    for &i in indices {
        match ValueRef::from_raw(MargSide(i)) {
            ValueRef::Slot(s) => {
                let idx = s as usize;
                if counts[idx] == OVERFLOW {
                    let b = big
                        .and_then(|b| b.get(idx))
                        .expect("OVERFLOW sentinel requires a marginal_counts_big entry");
                    big_acc += b;
                } else {
                    big_acc += BigUint::from(counts[idx]);
                }
            }
            ValueRef::Inline(v) => {
                big_acc += BigUint::from(v);
            }
        }
    }
    CountKey::Big(big_acc)
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
        crate::utils::release_if_oversized(&mut self.referenced, max_bytes);
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
            if raw & (1u32 << 31) != 0 {
                continue; // ZERO sentinel
            }
            if let ValueRef::Slot(s) = ValueRef::from_raw(MargSide(raw)) {
                if seen.insert(s) {
                    referenced.push(s);
                }
            }
        }
    }
    referenced.sort_unstable();
    referenced
}

#[cfg(test)]
#[path = "marg_slots_sum_tests.rs"]
mod sum_tests;
