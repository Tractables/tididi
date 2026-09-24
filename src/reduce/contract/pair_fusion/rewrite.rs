//! Phase 3: rewriting the parent level's pair lists in place.

use rustc_hash::FxHashMap;

use crate::limits::{OperationError, Transient};
use crate::diagram::{EncodedChildRef, ChildPair, Tdd, TddLevel};
use crate::Engine;
use crate::vtree::VtreeIdx;

use crate::diagram::ChildSide;

use super::PlanEntry;

/// Phase 3: rewrite the parent's pair lists in place, node by node. `plans`
/// must hold each node's entries contiguously (Phase 1 emits them in ascending
/// `node_idx`); only the nodes named in it are touched.
///
/// # Soundness
///
/// Phase 1 emits a plan only for a group of ≥2 pairs sharing one `x_idx`, and
/// Phase 3 replaces that whole group with one fused pair, so a node carrying
/// `k` plans drops ≥ 2k pairs and gains `k`: its new list fits inside its own
/// arena range. Per node a write cursor trails the read cursor over that range,
/// then the `k` fused pairs are appended at the cursor, still inside the old
/// range since `kept + k ≤ old_len − k`. The abandoned tail is charged to
/// `dead_pairs` and reclaimed by the level's arena sweep at the end.
pub(super) fn rebuild_parent_level<V>(
    eng: &Engine,
    tdd: &mut Tdd,
    parent: VtreeIdx,
    side: ChildSide,
    any_inline: bool,
    plans: &[PlanEntry<V>],
) -> Result<(), OperationError> {
    let level = &mut tdd.levels[parent.idx()];
    // Fusion-inline may mint a fresh inline marginal-side ref (bit-30 tagged) this
    // sweep; the marker for that side must be raised or the end-of-apply tagger
    // and the apply reader misread the ref as a grid coordinate. Rewriting in place
    // preserves every other flag — including the marker for a side that was
    // already inlined, and `n_tombstones` — by construction.
    if any_inline {
        level.set_has_value_refs(side, true);
    }

    // `fused_x` maps each fused x_idx to its new marginal-side ref. A plan
    // removes every pair at its x_idx (its group is the whole marginal multiset
    // there), so "this pair is fused away" is exactly `fused_x.contains_key`. A
    // node can carry thousands of plans, so membership stays a hash lookup.
    // Charged as it grows and handed back when the sweep ends.
    let mut fused_x: Transient<'_, FxHashMap<u32, u32>> = Transient::new(eng.limits(), FxHashMap::default());
    // Arena slots the shrink abandons, noted in one charge below: the counter's
    // only reader is the sweep at the end, so per-node saturating adds buy nothing.
    let mut dead_acc = 0usize;

    let mut cursor = 0usize;
    while cursor < plans.len() {
        let n = plans[cursor].node_idx;
        let plan_start = cursor;
        while cursor < plans.len() && plans[cursor].node_idx == n {
            cursor += 1;
        }
        let this_plans = &plans[plan_start..cursor];
        dead_acc += fuse_node_pairs(eng, level, n, side, this_plans, &mut fused_x)?;
    }
    level.note_dead_pairs(dead_acc);
    // Legal only now: the rewrite is done, so no pair-arena offset is held
    // across the call (the caller obligation on `compact_pairs_if_stale`).
    level.compact_pairs_if_stale();
    Ok(())
}

/// Rewrite one node's pair list in place: drop every pair whose x-side carries a
/// plan, then append one fused pair per plan. Returns the arena slots the shrink
/// abandoned.
fn fuse_node_pairs<V>(
    eng: &Engine,
    level: &mut TddLevel,
    n: usize,
    side: ChildSide,
    this_plans: &[PlanEntry<V>],
    fused_x: &mut FxHashMap<u32, u32>,
) -> Result<usize, OperationError> {
    fused_x.clear();
    // Each plan covers a distinct x_idx (Phase 1 emits one plan per
    // (node, x_idx) group), so the map holds one entry per plan — an
    // x_idx collision here would silently drop a fused pair's count.
    eng.limits().reserve_map(fused_x, this_plans.len())?;
    for plan in this_plans {
        fused_x.insert(plan.x_idx, plan.new_ref);
    }
    debug_assert_eq!(fused_x.len(), this_plans.len(), "plans must have distinct x_idx per node");

    // A plan-carrying node held ≥2 pairs (Phase 1 skips `pair_count_at < 2`),
    // so it is arena-backed — never a leaf, a tombstone, or an inline node
    // whose single pair lives in the node word.
    debug_assert!(
        level.nodes[n].kind().pairs_in_arena(),
        "rebuild_parent_level: node {n} carries a plan but owns no arena range",
    );
    let range = level.pair_range_at(n);
    let (start, old_len) = (range.start, range.len());

    // Fused away iff the x-side index carries a plan (see `fused_x` above).
    let is_fused = |p: ChildPair| {
        let x_idx = match side {
            ChildSide::Right => p.left.0,
            ChildSide::Left => p.right.0,
        };
        fused_x.contains_key(&x_idx)
    };

    // Keep the un-fused pairs, compacting them onto the front of the node's
    // own range: `write` never overtakes `read` (it advances at most once
    // per read, from the same origin), so a kept pair only ever moves down
    // onto a slot already read past.
    let mut write = start;
    for read in start..start + old_len {
        let p = level.pairs[read];
        if !is_fused(p) {
            level.pairs[write] = p;
            write += 1;
        }
    }
    // Append one fused pair per plan. Every read is done, and each plan
    // removed ≥2 pairs above, so `write + this_plans.len() ≤ start + old_len`
    // — the appends stay inside the node's own range and cannot reach the
    // next node's slots.
    for (&x_idx, &r_new) in fused_x.iter() {
        // `r_new` is the fully-encoded marginal-side ref from Phase 2 —
        // either a tagged inline count (bit-30 set) or a bare slot
        // index (bit-30 clear), self-describing. Write it verbatim;
        // `x_idx` is the non-marginal side.
        let fused = match side {
            ChildSide::Right => ChildPair::new(EncodedChildRef::from_raw(x_idx), EncodedChildRef::from_raw(r_new)),
            ChildSide::Left => ChildPair::new(EncodedChildRef::from_raw(r_new), EncodedChildRef::from_raw(x_idx)),
        };
        debug_assert!(write < start + old_len, "fusion must shrink the pair list");
        level.pairs[write] = fused;
        write += 1;
    }

    // Re-encode via the shared epilogue: shrink in place, or inline the sole
    // survivor.
    Ok(level.reencode_shrunk(n, start, old_len, write - start))
}
