//! The two store-rewriting sweeps of the marginal-slot prune.

use super::*;

/// Free the store of every marginal level whose parent is also marginal — the
/// parent consumed those values at cascade-marginalize time.
pub(super) fn clear_dead_deep_stores<S: SlotStore>(
    tdd: &mut Tdd,
    out_v: VtreeIdx,
    stats: &mut ValueSlotPruneStats,
) {
    // The root level (no parent) keeps its store: it holds the final count.
    for i in 0..tdd.levels.len() {
        if !tdd.levels[i].is_marginal() {
            continue;
        }
        let v = VtreeIdx(i as u32);
        if v == out_v {
            continue;
        }
        // Pin invariant (see `marginalize::marginalize_leaf_weighted`): a
        // weight-marginal leaf's column is an immutable, label-ordered, exactly
        // 3-slot cache of `WeightStore::leaf_val`. One column is shared (keyed by
        // vtree index, read by every `Tdd` this one's store reaches) and bare
        // leaf-label refs alias its slots by position. This pass can only rewrite
        // this `Tdd`'s own parent refs, so compacting or erasing a leaf column
        // silently corrupts every other holder — including the structural leaf
        // levels of fresh clause diagrams. Exempt from both walks.
        // (Integer-marginal leaves are not exempted: their store is empty, so
        // both walks below are already no-ops on them and the integer arm stays
        // bit-identical.)
        if tdd.vtree.node(v).is_leaf() && tdd.levels[i].is_weight_marginal() {
            continue;
        }
        let Some(parent) = tdd.vtree.node(v).parent() else { continue };
        if !tdd.levels[parent.idx()].is_marginal() {
            continue; // boundary level: compacted below
        }
        let freed = S::clear_dead_store(tdd, v);
        if freed == 0 {
            continue;
        }
        stats.stores_cleared += 1;
        S::update_width(tdd, v, freed, 0);
    }
}

/// Compact each boundary store to its parent-referenced set and rewrite the
/// parent's refs through the composed remap.
pub(super) fn compact_boundary_stores<S: SlotStore>(
    tdd: &mut Tdd,
    out_v: VtreeIdx,
    stats: &mut ValueSlotPruneStats,
    slots: &mut RefSlotScratch,
    remap: &mut Vec<u32>,
) {
    // Boundary stores: compact to the referenced set, remap parent refs.
    //
    // Value-dedup is also applied here (this is where slot-count uniqueness is
    // established for stores born at an apply emit site). Among the referenced slots, equal-valued slots are merged to
    // one output slot. The composed remap (reachability + value dedup) is
    // applied to parent refs in the same pass. Soundness follows from the
    // module invariant: every surviving parent ref is rewritten through the
    // remap in this same pass.
    for (v, parent, side) in boundary_marginal_levels(tdd) {
        if v == out_v {
            continue;
        }
        // Weight-marginal leaf exemption — the pin invariant, same constraint as
        // the dead-deep-stores walk above.
        if tdd.vtree.node(v).is_leaf() && tdd.levels[v.idx()].is_weight_marginal() {
            continue;
        }
        // Empty-store fast path. Read the store length before walking the
        // parent: an empty store has nothing to compact and names no slot a
        // parent ref could legally hold, so everything below collapses to
        // `update_width(0, 0)` — and skipping it skips both full parent-level
        // walks (the ref collection and the ref rewrite).
        //
        // This is the steady state, not a corner case. The end-of-apply tagger
        // rewrites every marginal-side ref whose count fits a ref into an
        // inline count, so on a diagram whose counts stay under
        // that bound the first sweep compacts each boundary store to zero and
        // every later sweep over the same level finds it already empty. The
        // per-merge minimize runs this sweep tens of times per compile.
        //
        // `update_width` is still called so the two value kinds keep their
        // (deliberately inverted) semantics: a no-op `+= 0` for integer, and the
        // load-bearing weighted-width reset.
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

        // No slot refs: `referenced` is exactly the set of `ValueRef::Slot`
        // refs the parent holds on this side, so an empty one means every ref
        // there is an inline count or a `ZERO` sentinel — both of which the
        // remap passes through untouched. Walking the level would rewrite
        // nothing. (Common: see the empty-store note above — this is the sweep
        // that first empties the store.)
        if referenced.is_empty() {
            continue;
        }

        remap_refs_into(tdd, v, remap);
    }
}
