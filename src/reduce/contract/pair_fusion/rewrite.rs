//! Phase 3: rewriting the parent level's pair lists in place.

use crate::engine::Engine;
use rustc_hash::FxHashMap;

use crate::error::ApplyError;
use crate::diagram::{InputPair, NodeIdx, Tdd, TddLevel};
use crate::vtree::VtreeIdx;

use crate::diagram::ChildSide;

use super::PlanEntry;

/// Phase 3: rewrite the parent's pair lists in place, node by node.
///
/// Fusion strictly shrinks every node it touches, which is what makes the
/// in-place form sound: Phase 1 emits a plan only for a group of ≥2 pairs
/// sharing one `x_idx`, and Phase 3 replaces that whole group with one fused
/// pair — never splits one. A node carrying `k` plans therefore drops ≥ 2k
/// pairs and gains exactly `k`, so its new list fits strictly inside its own
/// arena range. Per changed node: a write cursor trails the read cursor over
/// that range (dropping the pairs whose x-side carries a plan), then the `k`
/// fused pairs are appended at the cursor — still inside the old range, since
/// `kept + k ≤ old_len − k`. Nothing else on the level is touched, so
/// unchanged nodes (the majority on many instances) cost nothing at all.
///
/// The rewrite is in place because the alternatives — a full-size rebuild of
/// the level, or a per-node intermediate — cost a transient copy of the whole
/// parent level inside the minimize loop, where memory is tightest.
///
/// The shrink leaves the tail of each rewritten range unreferenced; it is
/// charged to `dead_pairs` and reclaimed by the level's own amortized arena
/// sweep at the end.
///
/// `plans` must keep all of one node's entries contiguous (Phase 1 emits them
/// in ascending `node_idx`), so we walk the plan list itself rather than the
/// whole level.
#[inline(always)]
pub(super) fn rebuild_parent_level(
    eng: &Engine,
    tdd: &mut Tdd,
    parent: VtreeIdx,
    side: ChildSide,
    any_inline: bool,
    plans: &[PlanEntry],
) -> Result<(), ApplyError> {
    let level = &mut tdd.levels[parent.idx()];
    // Fusion-inline may mint a fresh inline marginal-side ref (bit-30 tagged) this
    // sweep; the marker for that side must be raised or the end-of-apply tagger
    // and the apply reader misread the ref as a grid coordinate. Rewriting in place
    // preserves every other flag — including the marker for a side that was
    // already inlined, and `n_tombstones` — by construction.
    if any_inline {
        match side {
            ChildSide::Left => level.set_marginal_inlined_left(true),
            ChildSide::Right => level.set_marginal_inlined_right(true),
        }
    }

    // `fused_x` maps each fused x_idx -> its new marginal-side ref. Keyed on a
    // single u32 (the x-side index), not on (x_idx, marginal) tuples: a plan
    // removes every pair at its x_idx (its distinct_margs is the full marginal
    // multiset there), so "this pair is fused away" == "its x_idx has a plan"
    // == `fused_x.contains_key`, so no tuple-keyed set of removed pairs is
    // built or probed. Some nodes carry thousands of plans, so membership must
    // stay a hash lookup: a linear scan over fused entries is
    // O(old_pairs · plans).
    let mut fused_x: FxHashMap<u32, u32> = FxHashMap::default();
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
    // Reclaim the abandoned tails once they dominate the arena (the level's own
    // amortized trigger). Safe here and nowhere earlier: the rewrite is done, so
    // no pair-arena offset is held across the call — the caller obligation
    // documented on `compact_pairs_if_stale` (types/level.rs). The boundary loop
    // above holds only vtree indices, so it is unaffected.
    level.compact_pairs_if_stale();
    Ok(())
}

/// Rewrite one node's pair list in place: drop every pair whose x-side carries a
/// plan, then append one fused pair per plan. Returns the arena slots the shrink
/// abandoned.
fn fuse_node_pairs(
    eng: &Engine,
    level: &mut TddLevel,
    n: usize,
    side: ChildSide,
    this_plans: &[PlanEntry],
    fused_x: &mut FxHashMap<u32, u32>,
) -> Result<usize, ApplyError> {

    fused_x.clear();
    // Each plan covers a distinct x_idx (Phase 1 emits one plan per
    // (node, x_idx) group), so the map holds one entry per plan — an
    // x_idx collision here would silently drop a fused pair's count.
    for plan in this_plans {
        fused_x.insert(plan.x_idx, plan.new_ref);
    }
    debug_assert_eq!(fused_x.len(), this_plans.len(), "plans must have distinct x_idx per node");

    // A plan-carrying node held ≥2 pairs (Phase 1 skips `pair_count_at < 2`),
    // so it is arena-backed — never a leaf, a tombstone, or an inline node
    // whose single pair lives in the node word.
    debug_assert!(
        level.nodes[n].is_multi(),
        "rebuild_parent_level: node {n} carries a plan but owns no arena range",
    );
    let start = level.multi_start_at(n);
    let old_len = level.multi_len_at(n);

    // Keep the un-fused pairs, compacting them onto the front of the node's
    // own range: `write` never overtakes `read` (it advances at most once
    // per read, from the same origin), so a kept pair only ever moves down
    // onto a slot already read past.
    let mut write = start;
    for read in start..start + old_len {
        let p = level.pairs[read];
        let x_idx = match side {
            ChildSide::Right => p.left.0,
            ChildSide::Left => p.right.0,
        };
        // A pair is fused away iff its x-side index carries a plan: that
        // plan's distinct_margs is the full set of marginal values at this
        // x_idx (built from this node's own pairs in Phase 1), so every
        // pair at a fused x_idx is removed and replaced by one fused
        // pair. Hence membership in `fused_x` is the exact removal test
        // — no per-(x,marginal) set needed.
        if !fused_x.contains_key(&x_idx) {
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
            ChildSide::Right => InputPair {
                left: NodeIdx(x_idx),
                right: NodeIdx(r_new),
            },
            ChildSide::Left => InputPair {
                left: NodeIdx(r_new),
                right: NodeIdx(x_idx),
            },
        };
        debug_assert!(write < start + old_len, "fusion must shrink the pair list");
        level.pairs[write] = fused;
        write += 1;
    }

    let new_len = write - start;
    // Re-encode via the shared epilogue (`TddLevel::reencode_shrunk_multi`,
    // also used by `contract_leaf::rewrite_level`): shrink in place, inline
    // the sole survivor, or fall back to a length-1 extended multi aliasing
    // the node's own first slot — reusing its existing `multi_pairs` entry when the
    // node is already extended, so nothing here abandons an old `multi_pairs` slot
    // as garbage.
    level.reencode_shrunk_multi(eng, n, start, old_len, new_len)
}
