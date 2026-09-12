//! The store-compacting sweep of the marginal-slot prune.

use super::*;

/// Compact each boundary store to its parent-referenced set and rewrite the
/// parent's refs through the composed remap.
pub(super) fn compact_boundary_stores<S: SlotStore>(
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
        // Empty-store fast path: an empty store has nothing to compact and
        // names no slot a parent ref could hold, so skipping the two parent
        // walks below loses nothing. Common, because the tagger inlines every
        // count that fits a ref. `update_width` still runs: a no-op for the
        // integer tally, the live-width reset for weighted.
        let store_len = S::store_len(tdd, v);
        if store_len == 0 {
            S::update_width(tdd, v, 0, 0);
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
        let (new_len, values_merged) = S::compact_store(tdd, v, referenced, remap);
        stats.values_merged += values_merged;
        if values_merged > 0 {
            stats.value_merged_levels.push(v.0);
        }
        S::update_width(tdd, v, store_len - new_len, new_len);

        // `referenced` is exactly the set of `ValueRef::Slot` refs the parent
        // holds on this side; when it is empty every ref there is an inline
        // count or a `ZERO` sentinel, which the remap leaves untouched.
        if referenced.is_empty() {
            continue;
        }

        remap_refs_into(tdd, v, remap);
    }
}
