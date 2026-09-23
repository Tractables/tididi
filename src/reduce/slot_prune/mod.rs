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
//! decodable with `ChildDecoder`) — never mid-apply.
//!
//! Runs at the end of `Engine::reduce` and after each `fuse_pairs_at_parents`
//! sweep; the contract-only path kills no pairs and skips it.
//!
//! Integer and weighted levels share the one skeleton
//! `prune_marginal_slots_generic`, generic over the `SlotStore` trait.

use crate::Engine;


use crate::diagram::{Tdd, TddLevel, WeightStore};
use crate::vtree::VtreeIdx;

use crate::value::{IntFold, WeightFold};
use crate::value::slots::{RefSlotScratch, referenced_marginal_slots};
use crate::diagram::boundary_marginal_levels;
use crate::value::slots::{compact_slots, compact_count_slots, truncate_with_slack};

impl crate::limits::pool::PooledScratch for RefSlotScratch {
    fn prepare(&mut self) { self.clear(); }
    fn retain(&mut self, lim: &crate::limits::Limits) { self.release_oversized(lim); }
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


/// Storage-specific part of pruning. Each implementation commits its values
/// and metadata together; the traversal only discovers and rewrites references.
trait SlotStore {
    fn store_len(tdd: &Tdd, v: VtreeIdx) -> usize;

    /// Keep the sorted `referenced` slots, merge equal values and fill `remap`.
    /// `remap` has the old store length. Update live width and retirement
    /// accounting; return the number merged.
    fn compact_store(level: &mut TddLevel, weights: Option<&mut WeightStore>, v: VtreeIdx, referenced: &[u32], remap: &mut [u32]) -> usize;
}

impl SlotStore for IntFold {
    fn store_len(tdd: &Tdd, v: VtreeIdx) -> usize {
        tdd.levels[v.idx()].marginal_counts().map_or(0, |c| c.len())
    }

    fn compact_store(level: &mut TddLevel, _weights: Option<&mut WeightStore>, _v: VtreeIdx, referenced: &[u32], remap: &mut [u32]) -> usize {
        let Some((counts, big)) = level.marginal_store_mut() else { return 0 };
        let old_len = counts.len();
        if old_len == 0 {
            return 0;
        }
        let (new_len, merged) = compact_count_slots(
            counts, big, referenced.iter().map(|&old| old as usize), remap,
        );
        truncate_with_slack(counts, new_len);
        // Integer live width is the column length; only the retirement tally
        // needs an explicit update after truncating it.
        level.retire_marginal_slots((old_len - new_len) as u32);
        merged
    }
}

impl SlotStore for WeightFold {
    fn store_len(tdd: &Tdd, v: VtreeIdx) -> usize {
        tdd.weights.as_ref().and_then(|ws| ws.level(v.idx())).map_or(0, |s| s.len())
    }

    fn compact_store(level: &mut TddLevel, weights: Option<&mut WeightStore>, v: VtreeIdx, referenced: &[u32], remap: &mut [u32]) -> usize {
        use crate::diagram::semiring::weight_key;
        if remap.is_empty() {
            level.set_weight_width(0);
            return 0;
        }
        let values = weights.expect("weighted compaction requires its store").level_vals_mut(v.idx())
            .expect("a nonempty weighted store has a column");
        let (new_len, merged) = compact_slots(
            values,
            referenced.iter().map(|&old| old as usize),
            |values, old| weight_key(&values[old]),
            // Displaced values are dead and dropped by truncation.
            |values, dst, src| values.swap(dst, src),
            remap,
        );
        truncate_with_slack(values, new_len);
        level.set_weight_width(new_len as u32);
        merged
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
    let mut slots = eng.reduce_scratch().slot_prune_slots.checkout(eng.limits());
    let mut remap = eng.reduce_scratch().slot_prune_remap.checkout(eng.limits());
    // The output level's store is the result; never touch it.
    let out_v = tdd.output.vtree;


    compact_boundary_stores::<S>(tdd, out_v, &mut stats, &mut slots, &mut remap);

    // It rewrites stores in place and so cannot stop partway, for the reason
    // `prune_unreachable` gives; charging keeps the walk on the work clock.
    eng.limits().charge_work(tdd.vtree.num_nodes() as u64);

    stats
}

/// Compact each boundary store to its parent-referenced set and rewrite the
/// parent's refs through the composed remap.
fn compact_boundary_stores<S: SlotStore>(
    tdd: &mut Tdd,
    out_v: VtreeIdx,
    stats: &mut ValueSlotPruneStats,
    slots: &mut RefSlotScratch,
    remap: &mut Vec<u32>,
) {
    // Among the referenced slots, equal-valued ones merge to one output slot;
    // the composed remap (reachability, then value dedup) is applied to the
    // parent refs in the same pass.
    for (v, parent, side) in boundary_marginal_levels(tdd) {
        if v == out_v {
            continue;
        }
        // Invariant 11: a weight-marginal leaf's column is the label-ordered
        // cache of `WeightStore::leaf_val`, shared by every diagram over this
        // vtree, so compacting it would corrupt every other holder.
        if tdd.vtree.node(v).is_leaf() && tdd.levels[v.idx()].is_weight_marginal() {
            continue;
        }
        // Empty stores need no parent walks. Weighted live width still needs
        // synchronization, even when no store column is installed.
        let store_len = S::store_len(tdd, v);
        if store_len == 0 {
            tdd.reindex_level(v, &[], &mut [], |level, weights, referenced, remap| {
                S::compact_store(level, weights, v, referenced, remap)
            });
            continue;
        }

        let referenced =
            referenced_marginal_slots(&tdd.levels[parent.idx()], side, slots);
        if referenced.last().is_some_and(|&s| (s as usize) >= store_len) {
            continue; // OOB ref: broken upstream (the marginal-canonicality checker's domain)
        }

        // Build the composed remap: old_slot → final_output_slot.
        // Unreferenced slots (and all slots if referenced is empty) are dropped.
        // Equal-valued referenced slots map to the same output slot (first
        // occurrence wins).
        remap.clear();
        remap.resize(store_len, u32::MAX);
        let values_merged = tdd.reindex_level(v, referenced, remap, |level, weights, referenced, remap| {
            S::compact_store(level, weights, v, referenced, remap)
        });
        stats.values_merged += values_merged;
        if values_merged > 0 {
            stats.value_merged_levels.push(v.0);
        }
    }
}

#[cfg(test)]
mod tests;
