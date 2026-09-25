//! The value store of a marginal level, as slots: a hashable key for a stored
//! value, minting a value as a new slot, compaction, and the set of slots a
//! parent still references.

use std::hash::Hash;

use num_bigint::BigUint;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::value::{Count, CountRead, CountRef, IntFold, WeightFold};
use crate::diagram::{weight_key, ChildSide, WeightKey};
use crate::diagram::{
    EncodedChildRef, ChildDecoder, CountOverflow, ChildPair, Tdd, TddLevel, ValueRef, WeightStore, WeightValue,
};
use crate::Engine;
use crate::limits::{Limits, OperationError};
use crate::vtree::VtreeIdx;

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

/// Compact counts and rekey their overflow entries together. As with
/// [`compact_slots`], the caller truncates the column to the returned length.
/// `kept` is strictly ascending; unreferenced entries in `remap` are `u32::MAX`.
pub(crate) fn compact_count_slots(
    counts: &mut Vec<u128>,
    big: &mut Option<CountOverflow>,
    kept: impl IntoIterator<Item = usize>,
    remap: &mut [u32],
) -> (usize, usize) {
    let (new_len, merged) = compact_slots(
        counts,
        kept,
        |counts, old| count_key_at(counts, big.as_ref(), old),
        |counts, dst, src| counts[dst] = counts[src],
        remap,
    );
    // Keeping every slot without merging makes the remap the identity.
    if new_len != counts.len() {
        *big = rekey_big(big.take(), remap);
    }
    (new_len, merged)
}

/// Re-file an overflow table under the compacted slot indices: a slot `remap`
/// leaves at `u32::MAX` is dropped with its value, and a merged slot writes an
/// equal value over its canonical's entry. One drain, values moved not cloned.
fn rekey_big(big: Option<CountOverflow>, remap: &[u32]) -> Option<CountOverflow> {
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
fn seed_slot_map(
    lim: &Limits, map: &mut FxHashMap<Count, u32>, counts: CountRef<'_>,
) -> Result<(), OperationError> {
    lim.reserve_map(map, counts.len())?;
    for i in 0..counts.len() {
        map.entry(counts.get(i).to_count()).or_insert(i as u32);
    }
    Ok(())
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
    /// shares an existing slot with a value equal to it; the map's growth is
    /// charged to `lim`.
    ///
    /// # Errors
    ///
    /// `Err(OperationError::OverBudget)` when the growth is refused.
    fn seed(lim: &Limits, tdd: &Tdd, v: VtreeIdx, map: &mut FxHashMap<Self::Key, u32>) -> Result<(), OperationError>;

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

/// Check the next slot against the reference encoding before growing a store.
pub(crate) fn next_slot_index(len: usize) -> Result<u32, OperationError> {
    let slot = u32::try_from(len).map_err(|_| OperationError::IndexOverflow)?;
    ValueRef::Slot(slot).side().map_err(|_| OperationError::IndexOverflow)?;
    Ok(slot)
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
        let counts = tdd.levels[v.idx()].count_column().expect("pair fusion: the marginal level has no count store");
        sum_marginal_counts(counts, refs)
    }

    fn scaled(tdd: &Tdd, v: VtreeIdx, raw: u32, k: u32) -> Count {
        match ChildDecoder::marginal().value(EncodedChildRef::from_raw(raw)) {
            // c ≤ 2^30−1, k ≤ 2^32−1 → product fits u128 with room to spare.
            ValueRef::Inline(c) => Count::Fast(c as u128 * k as u128),
            ValueRef::Slot(s) => {
                let counts = tdd.levels[v.idx()]
                    .count_column()
                    .expect("scale: slot ref into non-marginal level");
                match counts.get(s as usize) {
                    CountRead::Big(b) => Count::Big(b * k),
                    CountRead::Fast(c) => match c.checked_mul(k as u128) {
                        Some(v) => Count::from_u128(v),
                        None => Count::Big(BigUint::from(c) * k),
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
    fn seed(lim: &Limits, tdd: &Tdd, v: VtreeIdx, map: &mut FxHashMap<Count, u32>) -> Result<(), OperationError> {
        let counts = tdd.levels[v.idx()].count_column().expect("pair fusion: the marginal level has no count store");
        seed_slot_map(lim, map, counts)
    }

    fn push_slot(eng: &Engine, tdd: &mut Tdd, v: VtreeIdx, value: Count) -> Result<u32, OperationError> {
        let (counts, big) = tdd.levels[v.idx()]
            .marginal_store_mut()
            .expect("push_slot: level is not marginal");
        let slot = next_slot_index(counts.len())?;
        super::append_count(eng, counts, big, value)?;
        Ok(slot)
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
            // The zero sentinel (bit 31) contributes the additive identity.
            if EncodedChildRef::from_raw(raw).is_reserved() {
                continue;
            }
            let slot = ChildDecoder::marginal().index(EncodedChildRef::from_raw(raw));
            acc.add_assign(&values.expect("weighted pair fusion: marginal level has no WeightStore")[slot]);
        }
        acc
    }

    fn scaled(tdd: &Tdd, v: VtreeIdx, raw: u32, k: u32) -> WeightValue {
        let slot = ChildDecoder::marginal().index(EncodedChildRef::from_raw(raw));
        let ws = tdd.weight_store();
        let values = ws.level(v.idx()).expect("scale: weighted level has no store");
        scaled_weight(ws, &values[slot], k)
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
    fn seed(_: &Limits, _: &Tdd, _: VtreeIdx, _: &mut FxHashMap<WeightKey, u32>) -> Result<(), OperationError> {
        Ok(())
    }

    /// Bumps the weighted level's live slot count, which is what
    /// `slot_count()` reads and apply sizes its buffers from.
    ///
    /// Signed weights make a value of exactly 0 reachable (for instance from
    /// `+a` and `−a`). That is a value like any other and gets its own slot — it
    /// must never become the bit-31 zero sentinel, which denotes the structural
    /// false node. The slot is checked before allocation or mutation.
    fn push_slot(eng: &Engine, tdd: &mut Tdd, v: VtreeIdx, value: WeightValue) -> Result<u32, OperationError> {
        // A weighted leaf column is pinned to three label-ordered slots that
        // every diagram of the compile aliases; appending a fourth would break
        // that alias. The leaf paths resolve by lookup and never reach here.
        debug_assert!(
            !tdd.vtree.node(v).is_leaf(),
            "refusing to mint a weight slot into a pinned leaf column (level {})",
            v.0
        );
        crate::diagram::MarginalStorage::new(&mut tdd.levels[v.idx()], tdd.weights.as_mut(), v.idx())
            .push_weight(eng, value)
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
/// The arithmetic is [`IntFold::fold`] driven with a constant 1 on the right,
/// so the overflow rule, including the promotion of a total landing on the
/// sentinel, is applied there.
pub(crate) fn sum_marginal_counts(counts: CountRef<'_>, indices: &[u32]) -> Count {
    let pairs = indices
        .iter()
        .map(|&raw| ChildPair::new(EncodedChildRef::from_raw(raw), EncodedChildRef::from_raw(0)));
    IntFold::fold(pairs, |raw| counts.read(ChildDecoder::marginal(), raw), |_| CountRead::Fast(1))
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
}

impl crate::limits::pool::Buffers for RefSlotScratch {
    fn buffers(&mut self, visit: &mut dyn FnMut(&mut dyn crate::limits::pool::Scratch)) {
        visit(&mut self.referenced);
        visit(&mut self.seen);
    }
}

impl crate::limits::pool::PooledScratch for RefSlotScratch {
    fn prepare(&mut self) { self.clear(); }
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
        for p in plevel.pairs_of_idx(n) {
            let raw = match side {
                ChildSide::Right => p.right.0,
                ChildSide::Left => p.left.0,
            };
            if EncodedChildRef::from_raw(raw).is_reserved() {
                continue;
            }
            if let ValueRef::Slot(s) = ChildDecoder::marginal().value(EncodedChildRef::from_raw(raw))
                && seen.insert(s) {
                    referenced.push(s);
                }
        }
    }
    referenced.sort_unstable();
    referenced
}

#[cfg(test)]
#[path = "tests/slots/mod.rs"]
mod tests;
