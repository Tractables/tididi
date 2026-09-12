//! The value store of a marginal level, as slots: a hashable key for a stored
//! value, minting a value as a new slot, compaction, and the set of slots a
//! parent still references.

use std::hash::Hash;

use num_bigint::BigUint;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::value::{Count, CountRead, IntFold, WeightFold};
use crate::diagram::marginal_ref::refs::ChildSide;
use crate::diagram::semiring::{weight_key, WeightKey};
use crate::diagram::{
    CountOverflow, ChildPair, MarginalSide, NodeIdx, Tdd, TddLevel, ValueRef, WeightStore, WeightValue,
};
use crate::engine::Engine;
use crate::limits::ApplyBudget;
use crate::limits::OperationError;
use crate::vtree::VtreeIdx;

/// Append `key` to a marginal store as a freshly minted slot, never reusing an
/// existing one; returns the new slot index.
///
/// `counts[i]` holds the small count, or the `u128::MAX` sentinel meaning the
/// real value is `big`'s entry for slot `i`; `big` is allocated on the first
/// overflow, a `Fast` push writes nothing there and a `Big` push records one
/// entry. Growth is reserved under [`ApplyBudget`], so an over-budget push
/// returns `OperationError::OverBudget`.
pub(crate) fn push_count_key(
    eng: &Engine,
    counts: &mut Vec<u128>,
    big: &mut Option<CountOverflow>,
    key: &Count,
) -> Result<u32, OperationError> {
    let lim = eng.limits();
    let new_idx = counts.len() as u32;
    match key {
        Count::Fast(c) => {
            // A small count landing exactly on the sentinel would be re-read as
            // `COUNT_OVERFLOW` with no `big` entry behind it; producers must route that
            // value to `Big` (see `sum_marginal_counts` and its pinned test).
            debug_assert!(*c != u128::MAX, "small count must not alias the overflow sentinel");
            lim.try_push(counts, *c)?;
        }
        Count::Big(v) => {
            lim.try_push(counts, u128::MAX)?;
            big.get_or_insert_with(CountOverflow::default)
                .try_insert::<ApplyBudget>(eng, counts.len() - 1, v.clone())?;
        }
    }
    Ok(new_idx)
}

// ── Compaction ───────────────────────────────────────────────────────────────

/// Compact a store to the slots `kept` names, in the ascending order given,
/// merging every slot whose key an earlier kept slot already carries onto that
/// slot. Writes `remap[old]` for every kept slot and returns
/// `(new_len, values_merged)`, where `values_merged` counts the kept slots
/// that landed on an earlier equal-keyed one. The caller truncates the store
/// to `new_len` afterwards.
///
/// `key_at(store, i)` reads the key of slot `i`; `move_down(store, dst, src)`
/// moves slot `src` to `dst`, with `dst <= src`.
///
/// # Soundness
///
/// `kept` is strictly ascending, so at step `i` the source is `old >= i` and
/// the destination is `new_len <= i <= old`: every write lands at or below a
/// slot already consumed, and every later read is at a strictly larger index
/// than any write done so far. A move that swaps writes `old` as well, which
/// is also `<= old`, and that slot is never read again.
pub(crate) fn compact_slots<S: ?Sized, K: Hash + Eq>(
    store: &mut S,
    kept: impl IntoIterator<Item = usize>,
    key_at: impl Fn(&S, usize) -> K,
    mut move_down: impl FnMut(&mut S, usize, usize),
    remap: &mut [u32],
) -> (usize, usize) {
    let mut by_key: FxHashMap<K, u32> = FxHashMap::default();
    let mut values_merged = 0usize;
    let mut new_len = 0usize;
    for old in kept {
        let key = key_at(store, old);
        // A hit means `old` holds a value an earlier surviving slot already
        // carries (invariant 10 merge); a miss mints the next compacted slot.
        if let Some(&hit) = by_key.get(&key) {
            remap[old] = hit;
            values_merged += 1;
            continue;
        }
        by_key.insert(key, new_len as u32);
        move_down(store, new_len, old);
        remap[old] = new_len as u32;
        new_len += 1;
    }
    (new_len, values_merged)
}

/// Re-file an overflow table under the compacted slot indices: a slot `remap`
/// leaves at `u32::MAX` is dropped with its value, and a merged slot writes an
/// equal value over its canonical's entry. One drain, values moved not cloned.
pub(crate) fn rekey_big(big: Option<CountOverflow>, remap: &[u32]) -> Option<CountOverflow> {
    big.map(|b| {
        b.into_iter()
            .filter_map(|(slot, v)| {
                let new = remap[slot as usize];
                (new != u32::MAX).then_some((new, v))
            })
            .collect::<CountOverflow>()
    })
}

/// Truncate a store compacted in place to `new_len`, giving its slack back
/// once the store has more than halved. The capacity observed here is the
/// pre-compaction one, so shrinking at 2× is the effective ceiling on the
/// slack a store keeps for its lifetime.
pub(crate) fn truncate_with_slack<T>(store: &mut Vec<T>, new_len: usize) {
    store.truncate(new_len);
    if store.capacity() > 64 && store.capacity() > 2 * store.len() {
        store.shrink_to_fit();
    }
}

/// Map every distinct count of a store to its first slot; a duplicate count
/// collapses to its first occurrence.
fn seed_slot_map(map: &mut FxHashMap<Count, u32>, counts: &[u128], big: Option<&CountOverflow>) {
    for i in 0..counts.len() {
        let key = count_key_at(counts, big, i);
        map.entry(key).or_insert(i as u32);
    }
}

// ── SlotValues ───────────────────────────────────────────────────────────────

/// The value arithmetic of one marginal store, for the passes that compute a
/// value and need a reference carrying it: the group sum of pair fusion and the
/// run length of duplicate-pair resolution.
///
/// Implemented on the same two domains as [`ValueDomain`](crate::value::domain::ValueDomain),
/// so how a value is minted sits next to how it folds.
pub(crate) trait SlotValues {
    /// A value of the store.
    type Value: Clone;
    /// The hashable form of a value; equal keys share a slot.
    type Key: Hash + Eq;

    /// The key of `value`.
    fn key(value: &Self::Value) -> Self::Key;

    /// The sum of the values the marginal-side references `refs` name at level
    /// `v`, over the occurrence multiset: a reference that occurs twice is
    /// added twice.
    fn sum_refs(tdd: &Tdd, v: VtreeIdx, refs: &[u32]) -> Self::Value;

    /// `k` times the value the marginal-side reference `raw` names at level `v`.
    fn scaled(tdd: &Tdd, v: VtreeIdx, raw: u32, k: u32) -> Self::Value;

    /// The reference carrying `value` in the pair itself, when the domain has
    /// such an encoding for it.
    fn inline_ref(value: &Self::Value) -> Option<u32>;

    /// Enter the slots level `v` already holds into `map`, for a domain that
    /// shares an existing slot with a value equal to it.
    fn seed(tdd: &Tdd, v: VtreeIdx, map: &mut FxHashMap<Self::Key, u32>);

    /// Append `value` as a fresh slot of level `v`'s store and return its
    /// index.
    fn push_slot(eng: &Engine, tdd: &mut Tdd, v: VtreeIdx, value: Self::Value) -> Result<u32, OperationError>;

    /// Whether a marginal vtree leaf's column is pinned: never written, so
    /// the only values representable there are the ones it already holds.
    const LEAF_PINNED: bool;

    /// The reference carrying `value` at the pinned marginal leaf `v`, or
    /// `None` when its column does not hold it. Asked only where
    /// [`Self::LEAF_PINNED`].
    fn leaf_ref(tdd: &Tdd, v: VtreeIdx, value: &Self::Value) -> Option<u32>;
}

/// `value` as a marginal-side reference into level `v`: inline where the
/// domain can carry it in the pair, a fresh slot otherwise. No interning: the
/// slot pruner merges equal-valued slots on the next prune.
pub(crate) fn mint_ref<D: SlotValues>(
    eng: &Engine,
    tdd: &mut Tdd,
    v: VtreeIdx,
    value: D::Value,
) -> Result<u32, OperationError> {
    if let Some(raw) = D::inline_ref(&value) {
        return Ok(raw);
    }
    D::push_slot(eng, tdd, v, value).map(ValueRef::slot_raw)
}

impl SlotValues for IntFold {
    type Value = Count;
    type Key = Count;

    #[inline]
    fn key(value: &Count) -> Count {
        value.clone()
    }

    fn sum_refs(tdd: &Tdd, v: VtreeIdx, refs: &[u32]) -> Count {
        let level = &tdd.levels[v.idx()];
        let counts = level.marginal_counts().expect("pair fusion: the marginal level has no count store");
        sum_marginal_counts(counts, level.marginal_counts_big(), refs)
    }

    fn scaled(tdd: &Tdd, v: VtreeIdx, raw: u32, k: u32) -> Count {
        match ValueRef::from_raw(MarginalSide(raw)) {
            // c ≤ 2^30−1, k ≤ 2^32−1 → product fits u128 with room to spare.
            ValueRef::Inline(c) => Count::Fast(c as u128 * k as u128),
            ValueRef::Slot(s) => {
                let level = &tdd.levels[v.idx()];
                let counts = level
                    .marginal_counts()
                    .expect("scale: slot ref into non-marginal level");
                match CountRead::from_slot(counts, level.marginal_counts_big(), s as usize) {
                    CountRead::Big(b) => Count::Big(b * k),
                    CountRead::Fast(c) => match c.checked_mul(k as u128) {
                        Some(v) if v != u128::MAX => Count::Fast(v),
                        _ => Count::Big(BigUint::from(c) * k),
                    },
                }
            }
        }
    }

    /// A small count rides in the pair itself, which also skips slot sharing:
    /// an inline reference is cheaper than a shared slot.
    #[inline]
    fn inline_ref(value: &Count) -> Option<u32> {
        match value {
            Count::Fast(c) => ValueRef::inline_raw(*c),
            Count::Big(_) => None,
        }
    }

    /// Seeded with the whole store, so a value equal to an existing slot's
    /// reuses it and the store stays at one slot per value.
    fn seed(tdd: &Tdd, v: VtreeIdx, map: &mut FxHashMap<Count, u32>) {
        let level = &tdd.levels[v.idx()];
        let counts = level.marginal_counts().expect("pair fusion: the marginal level has no count store");
        seed_slot_map(map, counts, level.marginal_counts_big());
    }

    fn push_slot(eng: &Engine, tdd: &mut Tdd, v: VtreeIdx, value: Count) -> Result<u32, OperationError> {
        let (counts, big) = tdd.levels[v.idx()]
            .marginal_store_mut()
            .expect("push_slot: level is not marginal");
        push_count_key(eng, counts, big, &value)
    }

    /// An integer leaf's store is written like an internal one.
    const LEAF_PINNED: bool = false;

    fn leaf_ref(_: &Tdd, _: VtreeIdx, _: &Count) -> Option<u32> {
        unreachable!("an integer leaf column is not pinned")
    }
}

/// `k · v` in the store's active mode — the one place a multiplicity becomes a
/// weighted factor. Building `k` as a same-mode `WeightValue` keeps the scale a
/// same-variant `WeightValue::mul`.
pub(crate) fn scaled_weight(ws: &WeightStore, v: &WeightValue, k: u32) -> WeightValue {
    use crate::diagram::SignedLog;
    use num_bigint::BigInt;
    use num_rational::BigRational;
    let k_w = if ws.is_log() {
        WeightValue::Log(SignedLog::from_rational(&BigRational::from_integer(BigInt::from(k))))
    } else {
        // A `u32` multiplicity is always in the small exact representation.
        WeightValue::ExactSmall(i128::from(k))
    };
    v.mul(&k_w)
}

impl SlotValues for WeightFold {
    type Value = WeightValue;
    type Key = WeightKey;

    #[inline]
    fn key(value: &WeightValue) -> WeightKey {
        weight_key(value)
    }

    /// # Soundness
    ///
    /// Slots at a marginal level carry pairwise-disjoint model sets (partition
    /// invariant), so the values of a group's members are values of disjoint
    /// sets and add. Finite additivity over a disjoint union holds for signed
    /// measures, so a negative literal weight is not an obstacle; the parent's
    /// contribution `Σᵢ W(x)·W(mᵢ) = W(x)·Σᵢ W(mᵢ)` then follows from
    /// distributivity in ℚ. The reasoning is exact-domain only; the caller
    /// declines in the log domain.
    fn sum_refs(tdd: &Tdd, v: VtreeIdx, refs: &[u32]) -> WeightValue {
        let ws = tdd.weight_store();
        let values = ws.level(v.idx());
        let mut acc = ws.wzero();
        for &raw in refs {
            // The zero sentinel (bit 31) never appears in a pair list; if it
            // did, it would contribute the additive identity, so it is skipped.
            debug_assert!(
                !MarginalSide(raw).is_zero_sentinel(),
                "the zero sentinel must not reach a marginal-side pair ref"
            );
            if MarginalSide(raw).is_zero_sentinel() {
                continue;
            }
            match ValueRef::from_raw(MarginalSide(raw)) {
                ValueRef::Inline(_) => unreachable!("weighted marginal-side refs are bare slots"),
                ValueRef::Slot(s) => {
                    let v = &values.expect("weighted pair fusion: marginal level has no WeightStore")
                        [s as usize];
                    acc.add_assign(v);
                }
            }
        }
        acc
    }

    fn scaled(tdd: &Tdd, v: VtreeIdx, raw: u32, k: u32) -> WeightValue {
        // Weighted marginal-side refs reaching here are always Slot — nothing mints a
        // weighted `Inline` — and the arm below only holds the match exhaustive.
        // Zero sentinels carry no value and are not scaled here.
        match ValueRef::from_raw(MarginalSide(raw)) {
            ValueRef::Slot(s) => {
                let ws = tdd.weight_store();
                let values = ws
                    .level(v.idx())
                    .expect("scale: weighted level has no store");
                scaled_weight(ws, &values[s as usize], k)
            }
            ValueRef::Inline(_) => {
                unreachable!(
                    "a weighted marginal ref is always a store slot: no path mints an \
                     Inline ref on a weighted level, and an Inline ref here would \
                     dangle across component graft (store rebuild drops the intern \
                     table)"
                )
            }
        }
    }

    /// An inline payload is an integer count, which a weighted value has no
    /// encoding for.
    #[inline]
    fn inline_ref(_: &WeightValue) -> Option<u32> {
        None
    }

    /// Nothing: a fused value shares a slot with the plans of the same sweep
    /// only. Sharing with an existing slot is deferred to the slot pruner's
    /// value merge on the next prune.
    fn seed(_: &Tdd, _: VtreeIdx, _: &mut FxHashMap<WeightKey, u32>) {}

    /// Bumps `weight_width`, the weighted level's live slot count, which is
    /// what `slot_count()` reads and apply sizes its buffers from.
    ///
    /// Signed weights make a value of exactly 0 reachable (for instance from
    /// `+a` and `−a`). That is a value like any other and gets its own slot — it
    /// must never become the bit-31 zero sentinel, which denotes the structural
    /// false node; `slot_raw` keeps bit 31 clear by construction and the assert
    /// pins it.
    fn push_slot(_: &Engine, tdd: &mut Tdd, v: VtreeIdx, value: WeightValue) -> Result<u32, OperationError> {
        // A weighted leaf column is pinned to three label-ordered slots that
        // every diagram of the compile aliases; appending a fourth would break
        // that alias. The leaf paths resolve by lookup and never reach here.
        debug_assert!(
            !tdd.vtree.node(v).is_leaf(),
            "refusing to mint a weight slot into a pinned leaf column (level {})",
            v.0
        );
        let s = tdd.weight_store_mut().push_value(v.idx(), value);
        let s = u32::try_from(s).map_err(|_| OperationError::OverBudget)?;
        if !ValueRef::slot_is_referenceable(s) {
            // A slot index that would not fit the 30-bit marginal-ref payload cannot
            // be referenced at all — surface it as OverBudget (routed to
            // recovery) rather than truncate a ref.
            return Err(OperationError::OverBudget);
        }
        tdd.levels[v.idx()].set_weight_width(s + 1);
        debug_assert!(
            !MarginalSide(ValueRef::slot_raw(s)).is_zero_sentinel(),
            "a minted weighted marginal ref must never alias the zero sentinel",
        );
        Ok(s)
    }

    /// A weighted leaf column is the pinned, label-ordered three-slot cache
    /// every diagram of the compile aliases.
    const LEAF_PINNED: bool = true;

    /// The slot holding `value`, found by
    /// [`find_leaf_slot_by_value`](crate::diagram::find_leaf_slot_by_value),
    /// which scans ascending and so answers the canonical slot of its value
    /// class. Exact domain only: `weight_key` equality on a log value is
    /// `f64` bit equality.
    fn leaf_ref(tdd: &Tdd, v: VtreeIdx, value: &WeightValue) -> Option<u32> {
        let ws = tdd.weight_store();
        debug_assert!(
            !ws.is_log(),
            "leaf lookup reached in the log domain (the fusion gate must exclude it)"
        );
        crate::diagram::find_leaf_slot_by_value(ws, v.idx(), value).map(ValueRef::slot_raw)
    }
}

/// Read the marginal count at `slot` as an owned `Count`.
pub(crate) fn count_key_at(
    counts: &[u128],
    big: Option<&CountOverflow>,
    slot: usize,
) -> Count {
    CountRead::from_slot(counts, big, slot).to_count()
}

/// Sum the values at `indices`, each a marginal-side reference.
///
/// A reference is either an inline value (bit 30 set: the value is the count,
/// with no array load) or a bare slot (bit 30 clear: an index into `counts`).
/// `ValueRef::from_raw` does that split; its bit-31 assert fires in a debug
/// build if a zero sentinel ever reaches here.
///
/// The arithmetic is [`IntFold::fold`] driven with a constant 1 on the right,
/// so the overflow rule, including the promotion of a total landing on the
/// sentinel, is applied there.
pub(crate) fn sum_marginal_counts(
    counts: &[u128],
    big: Option<&CountOverflow>,
    indices: &[u32],
) -> Count {
    let read = |raw: usize| -> CountRead<'_> {
        match ValueRef::from_raw(MarginalSide(raw as u32)) {
            ValueRef::Inline(v) => CountRead::Fast(v as u128),
            ValueRef::Slot(s) => CountRead::from_slot(counts, big, s as usize),
        }
    };
    let pairs = indices
        .iter()
        .map(|&raw| ChildPair { left: NodeIdx(raw), right: NodeIdx(0) });
    IntFold::fold(pairs, read, |_| CountRead::Fast(1))
}

/// Caller-owned scratch for [`referenced_marginal_slots`], reused across the
/// levels of a slot-prune sweep.
#[derive(Default)]
pub(crate) struct RefSlotScratch {
    /// The deduped, sorted slot list — the pass's result, borrowed by the caller.
    pub(crate) referenced: Vec<u32>,
    seen: FxHashSet<u32>,
}

impl RefSlotScratch {
    /// Empty both buffers, retaining their allocations.
    pub(crate) fn clear(&mut self) {
        self.referenced.clear();
        self.seen.clear();
    }

    /// Drop the allocation of either buffer whose retained capacity exceeds
    /// the scratch-retention cap; each buffer is judged on its own capacity.
    pub(crate) fn release_oversized(&mut self) {
        crate::limits::pool::release_if_oversized(&mut self.referenced);
        // `FxHashSet` has no `Vec` shape for `release_if_oversized`; its table is
        // `capacity` u32 entries plus control bytes, so the same element-count
        // bound applies.
        if self.seen.capacity().saturating_mul(std::mem::size_of::<u32>()) > crate::limits::pool::SCRATCH_RETAIN_BYTES {
            self.seen = FxHashSet::default();
        }
    }
}

/// Fill `scratch.referenced` with the slots of a boundary-marginal level that
/// are referenced from `plevel`'s marginal-side pair refs (deduped, sorted). Skips
/// `ZERO` sentinels and inline refs; discarding out-of-range slots is the
/// caller's choice.
///
/// Dedup is hash-based so the sort is over distinct slots, not ref occurrences.
pub(crate) fn referenced_marginal_slots<'a>(
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
        for p in plevel.pairs_of_idx(n) {
            let raw = match side {
                ChildSide::Right => p.right.0,
                ChildSide::Left => p.left.0,
            };
            if MarginalSide(raw).is_zero_sentinel() {
                continue;
            }
            if let ValueRef::Slot(s) = ValueRef::from_raw(MarginalSide(raw))
                && seen.insert(s) {
                    referenced.push(s);
                }
        }
    }
    referenced.sort_unstable();
    referenced
}

#[cfg(test)]
mod tests;
