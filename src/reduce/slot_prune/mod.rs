//! Slot-prune of orphaned marginal-count slots.
//!
//! Slots become garbage at a boundary marginal level (a marginal child of a
//! structural parent) when the tagger, pair fusion or a twin merge redirects
//! the refs that named them; no other pass shrinks such a store. The
//! postcondition is `test_helpers::check::marginal::check_no_orphan_slots`.
//!
//! `prune_value_slots` compacts each boundary store to exactly the slots
//! referenced from its parent's marginal-side refs and rewrites those refs
//! through the same remap. The output level's store is exempt: it holds the
//! result. Equal-valued surviving slots are merged, which is where slot-count
//! uniqueness (invariant 10) is established for stores born at an apply emit
//! site.
//!
//! **Precondition:** parent levels must be in post-tagger form (marginal-side refs
//! decodable with `ValueRef::from_raw`) — never mid-apply.
//!
//! Runs at the end of `try_reduce` and after each `fuse_pairs_at_parents`
//! sweep; the contract-only path kills no pairs and skips it.
//!
//! Integer and weighted levels share the one skeleton
//! `prune_marginal_slots_generic`, generic over the `SlotStore` trait.

use crate::engine::Engine;


use crate::diagram::Tdd;
use crate::vtree::VtreeIdx;

use crate::value::{IntFold, WeightFold, SlotStore};
use crate::value::slots::{RefSlotScratch, referenced_marginal_slots};
use crate::diagram::{boundary_marginal_levels, remap_refs_into};
use crate::value::slots::{compact_slots, count_key_at, rekey_big, truncate_with_slack};

impl crate::limits::pool::PooledScratch for RefSlotScratch {
    fn prepare(&mut self) { self.clear(); }
    fn retain(&mut self) { self.release_oversized(); }
}

/// What a `prune_value_slots` sweep reclaimed.
#[derive(Debug, Default, Clone)]
pub(crate) struct ValueSlotPruneStats {
    /// Referenced slots eliminated by value-dedup: each mapped onto an earlier
    /// equal-valued slot, with every parent ref remapped to the survivor.
    /// Distinct from unreferenced-orphan drops. Such a merge can make two
    /// parent nodes content-equal.
    pub(crate) values_merged: usize,
    /// Marginal vtree levels where `values_merged` fired this sweep; the
    /// content-twin scan restricts itself to their boundary parents.
    pub(crate) value_merged_levels: Vec<u32>,
}


/// Collect orphaned marginal-count slots diagram-wide. Precondition: the
/// diagram is in post-tagger form (module doc).
pub(crate) fn prune_value_slots(eng: &Engine, tdd: &mut Tdd) -> ValueSlotPruneStats {
    if tdd.weights.is_some() {
        prune_marginal_slots_generic::<WeightFold>(eng, tdd)
    } else {
        prune_marginal_slots_generic::<IntFold>(eng, tdd)
    }
}


/// Integer: values are u128 counts in `TddLevel::marginal_counts`,
/// with exact `BigUint` overflow entries in the `marginal_counts_big` side
/// table.
impl SlotStore for IntFold {
    fn store_len(tdd: &Tdd, v: VtreeIdx) -> usize {
        tdd.levels[v.idx()].marginal_counts().map_or(0, |c| c.len())
    }

    fn compact_store(tdd: &mut Tdd, v: VtreeIdx, referenced: &[u32], remap: &mut [u32]) -> (usize, usize) {
        // `referenced` is strictly ascending and duplicate-free (its producer
        // `referenced_marginal_slots` sorts it), which is what `compact_slots`
        // needs. The sparse overflow table is keyed by old slot index, so it is
        // taken out, read through `count_key_at` during the loop, and rekeyed
        // once `remap` is complete.
        let (counts, big) = tdd.levels[v.idx()].marginal_store_mut().unwrap();
        let old_big = big.take();
        let (new_len, values_merged) = compact_slots(
            counts,
            referenced.iter().map(|&old| old as usize),
            |counts, old| count_key_at(counts, old_big.as_ref(), old),
            |counts, dst, src| counts[dst] = counts[src],
            remap,
        );
        *big = rekey_big(old_big, remap);
        truncate_with_slack(counts, new_len);
        (new_len, values_merged)
    }

    /// Integer: add `freed` to the level's retirement tally. The live width is
    /// `marginal_counts.len()`, already committed by the caller's compaction.
    fn update_width(tdd: &mut Tdd, v: VtreeIdx, freed: usize, new_len: usize) {
        let level = &mut tdd.levels[v.idx()];
        let before = level.retired_marginal_slots();
        level.retire_marginal_slots(freed as u32);
        debug_assert!(
            level.retired_marginal_slots() >= before,
            "the retirement tally only grows — it is never a live width"
        );
        debug_assert_eq!(
            level.slot_count(),
            new_len,
            "integer live width is marginal_counts.len(), committed before update_width"
        );
    }
}

/// Weighted: values live in the external `WeightStore`, indexed by level, and
/// the level's `weight_width` is the live slot count.
///
/// Stores are compacted even when every marginal-side ref is inline:
/// `weight_width` is what `slot_count()` returns for a weight-marginal level, and
/// the apply buffers are sized from it. Both ref walkers
/// (`referenced_marginal_slots`, `remap_refs_into`) touch only
/// `ValueRef::Slot`, so inline refs pass through verbatim.
impl SlotStore for WeightFold {
    fn store_len(tdd: &Tdd, v: VtreeIdx) -> usize {
        tdd.weights
            .as_ref()
            .and_then(|ws| ws.level(v.idx()))
            .map_or(0, |s| s.len())
    }

    /// Value-dedup keys on the semiring value directly — one uniform key type,
    /// no Small/Big `Count` split. A marginalized node is fully represented
    /// by its value, so two referenced slots with equal value are
    /// interchangeable upward and merge to one (first occurrence wins).
    fn compact_store(tdd: &mut Tdd, v: VtreeIdx, referenced: &[u32], remap: &mut [u32]) -> (usize, usize) {
        use crate::diagram::semiring::weight_key;
        // `WeightValue` is not `Copy`, so the move down is a swap; the displaced
        // value is dead from then on and dropped by the closing truncate.
        let ws = tdd.weight_store_mut();
        if !ws.is_set(v.idx()) {
            // A marginal boundary level with no store allocated gets an empty
            // one, the "weight-marginal, zero slots" state. Only reachable with
            // an empty `referenced`: the caller's out-of-range guard rejects
            // any ref into a zero-length store.
            ws.set_level(v.idx(), Vec::new());
        }
        let values = ws.level_vals_mut(v.idx()).expect("store present: ensured just above");
        let (new_len, values_merged) = compact_slots(
            values,
            referenced.iter().map(|&old| old as usize),
            |values, old| weight_key(&values[old]),
            |values, dst, src| values.swap(dst, src),
            remap,
        );
        // Drops the orphans, the merged-away duplicates, and the values
        // swapped up out of the prefix — the point of the pass.
        truncate_with_slack(values, new_len);
        (new_len, values_merged)
    }

    /// Weighted semantics: `weight_width` is itself the live slot count —
    /// `TddLevel::slot_count()` returns it for a weight-marginal level (set by
    /// `become_marginal_weighted`), and the apply/streaming buffers are
    /// sized from that. This assigns `new_len`; `freed` is stats-only here and must
    /// not be added, or the width drifts up and re-opens the oversized-buffer
    /// blowup described on the impl above.
    fn update_width(tdd: &mut Tdd, v: VtreeIdx, _freed: usize, new_len: usize) {
        tdd.levels[v.idx()].set_weight_width(new_len as u32);
        debug_assert_eq!(
            tdd.weights.as_ref().and_then(|ws| ws.level(v.idx())).map_or(0, |s| s.len()),
            new_len,
            "weight_width must equal the live WeightStore length"
        );
    }
}

/// The one prune skeleton, generic over where the values live.
fn prune_marginal_slots_generic<S: SlotStore>(eng: &Engine, tdd: &mut Tdd) -> ValueSlotPruneStats {
    // Invariant 11 (`check_leaf_columns_pinned`), checked here because this
    // pass runs after every pass that could break it.
    #[cfg(debug_assertions)]
    if let Err(e) = crate::test_helpers::check::marginal::check_leaf_columns_pinned(tdd) {
        panic!("leaf column pin: {e}");
    }
    let mut stats = ValueSlotPruneStats::default();
    // Both buffers are refilled per level, so a pooled pair differs from a
    // fresh one only in capacity.
    let mut slots = eng.reduce().slot_prune_slots.checkout();
    let mut remap = eng.reduce().slot_prune_remap.checkout();
    // The output level's store is the result; never touch it.
    let out_v = tdd.output.vtree;


    compact_boundary_stores::<S>(tdd, out_v, &mut stats, &mut slots, &mut remap);

    stats
}

mod stores;
use stores::compact_boundary_stores;

#[cfg(test)]
mod tests;
