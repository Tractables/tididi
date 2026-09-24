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
//! Runs at the end of every prune and every full reduction, and after each
//! `fuse_pairs_at_parents` sweep; the contract-only plan kills no pairs and
//! skips it.

use crate::Engine;


use crate::diagram::{Tdd, MarginalValues, MarginalStorage};

use crate::value::slots::{RefSlotScratch, referenced_marginal_slots};
use crate::diagram::boundary_marginal_levels;

impl crate::limits::pool::PooledScratch for RefSlotScratch {
    fn retained_bytes(&self) -> usize { RefSlotScratch::retained_bytes(self) }
    fn prepare(&mut self) { self.clear(); }
    fn retain(&mut self, lim: &crate::limits::Limits) { self.release_oversized(lim); }
}

/// Compact every boundary store to the slots its parent references, merging
/// equal-valued survivors, and rewrite the parent's refs through the composed
/// remap. Returns the marginal levels where a value merge fired: such a merge
/// can make two parent nodes content-equal, so the content-twin scan restricts
/// itself to their boundary parents.
///
/// Precondition: the diagram is in post-tagger form (module doc).
pub(crate) fn prune_value_slots(eng: &Engine, tdd: &mut Tdd) -> Vec<u32> {
    // Invariant 11 (`check_leaf_columns_pinned`), checked here because this
    // pass runs after every pass that could break it.
    #[cfg(debug_assertions)]
    if let Err(e) = crate::test_helpers::check::marginal::check_leaf_columns_pinned(tdd) {
        panic!("leaf column pin: {e}");
    }
    let mut merged_levels = Vec::new();
    // Both buffers are refilled per level, so a pooled pair differs from a
    // fresh one only in capacity.
    let mut slots = eng.reduce_scratch().slot_prune_slots.checkout(eng.limits());
    let mut remap = eng.reduce_scratch().slot_prune_remap.checkout(eng.limits());
    // The output level's store is the result; never touch it.
    let out_v = tdd.output.vtree;
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
        let store_len = MarginalValues::read(&tdd.levels[v.idx()], tdd.weights.as_ref(), v.idx()).map_or(0, |values| values.len());
        if store_len == 0 {
            tdd.reindex_level(v, &[], &mut [], |level, weights, referenced, remap| {
                MarginalStorage::new(level, weights, v.idx()).compact(referenced, remap)
            });
            continue;
        }

        let referenced =
            referenced_marginal_slots(&tdd.levels[parent.idx()], side, &mut slots);
        if referenced.last().is_some_and(|&s| (s as usize) >= store_len) {
            continue; // OOB ref: broken upstream (the marginal-canonicality checker's domain)
        }

        // The composed remap, old slot to final slot: unreferenced slots are
        // dropped, and equal-valued referenced slots map to one output slot
        // (first occurrence wins).
        remap.clear();
        remap.resize(store_len, u32::MAX);
        let values_merged = tdd.reindex_level(v, referenced, &mut remap, |level, weights, referenced, remap| {
            MarginalStorage::new(level, weights, v.idx()).compact(referenced, remap)
        });
        if values_merged > 0 {
            merged_levels.push(v.0);
        }
    }

    // It rewrites stores in place and so cannot stop partway, for the reason
    // `prune_unreachable` gives; charging keeps the walk on the work clock.
    eng.limits().charge_work(tdd.vtree.num_nodes() as u64);

    merged_levels
}

#[cfg(test)]
mod tests;
