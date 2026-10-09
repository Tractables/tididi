//! Phase 3: rewriting the parent level's pair lists in place.

use rustc_hash::FxHashMap;

use crate::limits::{OperationError, Transient};
use crate::diagram::{EncodedChildRef, ChildPair, Tdd, TddLevel};
use crate::Engine;
use crate::vtree::VtreeIdx;

use crate::diagram::ChildSide;

use super::PlanEntry;

/// The fewest plans at which a node tests "fused away" on a bitmap over the
/// explicit-side refs instead of in `fused_x`: below it the map fits the cache
/// and each probe is cheap. Above it the map outgrows the cache, and the probe
/// per pair the rewrite makes misses it on nearly every pair of a wide node —
/// where a bitmap as wide as the refs it covers stays in it.
pub(super) const BITMAP_MIN_PLANS: usize = 1 << 12;

/// The widest bitmap the test takes, in bits per plan: at 64 it is at most
/// eight bytes per plan, below the map's own footprint of a `(u32, u32)`
/// entry and its control byte, so it never costs more memory than the map it
/// stands in for.
pub(super) const BITMAP_BITS_PER_PLAN: usize = 64;

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
    rebuild_parent_level_with(eng, tdd, parent, side, any_inline, plans, BITMAP_MIN_PLANS)
}

/// [`rebuild_parent_level`] with the fewest plans at which a node tests
/// membership on a bitmap given, so a test can run one level both ways.
pub(super) fn rebuild_parent_level_with<V>(
    eng: &Engine,
    tdd: &mut Tdd,
    parent: VtreeIdx,
    side: ChildSide,
    any_inline: bool,
    plans: &[PlanEntry<V>],
    bitmap_min_plans: usize,
) -> Result<(), OperationError> {
    // The fusion changes node lengths in place: an implicit level, one whose
    // child became marginal, is built stored where its pairs lie first.
    tdd.levels[parent.idx()].store_if_implicit(eng.limits())?;
    tdd.levels.mark_changed(parent);
    let level = &mut tdd.levels[parent.idx()];
    // Fusion-inline may mint a fresh inline marginal-side ref (bit-30 tagged) this
    // sweep; the marker for that side must be raised or the end-of-apply tagger
    // and the apply reader misread the ref as a grid coordinate. Rewriting in place
    // preserves every other flag, including the marker for a side that was
    // already inlined, by construction.
    if any_inline {
        level.set_has_value_refs(side, true);
    }

    // `fused_x` maps each fused x_idx to its new marginal-side ref. A plan
    // removes every pair at its x_idx (its group is the whole marginal multiset
    // there), so "this pair is fused away" is exactly `fused_x.contains_key`. A
    // node can carry thousands of plans, so membership stays a hash lookup.
    // Charged as it grows and handed back when the sweep ends.
    let mut fused_x: Transient<'_, FxHashMap<u32, u32>> = Transient::new(eng.limits(), FxHashMap::default());
    // The membership bitmap of a node with many plans (see `fuse_node_pairs`),
    // all zero between nodes, charged as it grows and handed back at the end.
    let mut bits: Transient<'_, Vec<u64>> = Transient::new(eng.limits(), Vec::new());
    // Arena slots the shrink abandons, noted in one charge below: the counter's
    // only reader is the sweep at the end, so per-node saturating adds buy nothing.
    let mut dead_acc = 0usize;
    // A stop is tested between nodes, once per stride of the pairs rewritten.
    // Each node is rewritten whole, and a node left unfused is only a redex
    // left for later, so a stop leaves a valid level behind it.
    let lim = eng.limits();
    let stride = lim.reduce_poll_stride();
    let mut read = 0u64;
    let mut stopped = Ok(());

    let mut cursor = 0usize;
    while cursor < plans.len() {
        let n = plans[cursor].node_idx;
        read += 1 + level.pair_count_at(n) as u64;
        if read >= stride {
            read = 0;
            stopped = lim.check_stop();
            if stopped.is_err() {
                break;
            }
        }
        let plan_start = cursor;
        while cursor < plans.len() && plans[cursor].node_idx == n {
            cursor += 1;
        }
        let this_plans = &plans[plan_start..cursor];
        dead_acc += fuse_node_pairs(eng, level, n, side, this_plans, &mut fused_x, &mut bits, bitmap_min_plans)?;
    }
    level.note_dead_pairs(dead_acc);
    // Legal only now: the rewrite is done, so no pair-arena offset is held
    // across the call (the caller obligation on `compact_pairs_if_stale`).
    level.compact_pairs_if_stale();
    stopped
}

/// Rewrite one node's pair list in place: drop every pair whose x-side carries a
/// plan, then append one fused pair per plan. Returns the arena slots the shrink
/// abandoned.
///
/// The kept pairs stay in their order and the fused pairs follow them in plan
/// order, the order in which the node's pairs first name each fused x-ref: a
/// function of the node's old list alone, so a builder that sums a marginal
/// child as it emits the node can write the same list
/// (`sparse::sum`).
///
/// A node with at least `bitmap_min_plans` ([`BITMAP_MIN_PLANS`]) plans whose largest planned x-ref
/// is under [`BITMAP_BITS_PER_PLAN`] bits per plan tests each pair on a bitmap
/// of the planned x-refs (`bits`, zero on entry and on return) instead of in
/// `fused_x`, which it then does not build. The bit of `x` is set exactly when
/// a plan names `x`, so the pairs kept, the pairs dropped and the node's new
/// pair list are the ones the map gives, in the same order.
#[expect(clippy::too_many_arguments)]
fn fuse_node_pairs<V>(
    eng: &Engine,
    level: &mut TddLevel,
    n: usize,
    side: ChildSide,
    this_plans: &[PlanEntry<V>],
    fused_x: &mut FxHashMap<u32, u32>,
    bits: &mut Vec<u64>,
    bitmap_min_plans: usize,
) -> Result<usize, OperationError> {
    // A plan-carrying node held ≥2 pairs (Phase 1 skips `pair_count_at < 2`),
    // so it is arena-backed, never an inline node whose single pair lives in
    // the node word.
    debug_assert!(
        level.node(n).kind().pairs_in_arena(),
        "rebuild_parent_level: node {n} carries a plan but owns no arena range",
    );
    let range = level.pair_range_at(n);
    let (start, old_len) = (range.start, range.len());

    let x_of = |p: ChildPair| match side {
        ChildSide::Right => p.left.0,
        ChildSide::Left => p.right.0,
    };
    // The bitmap's width in words, or `None` for the map: the largest planned
    // x-ref bounds it, and a pair whose x-ref is past it carries no plan.
    let words = match this_plans.len() >= bitmap_min_plans {
        false => None,
        true => {
            let widest = this_plans.iter().map(|plan| plan.x_idx as usize).max().unwrap_or(0);
            (widest / BITMAP_BITS_PER_PLAN < this_plans.len()).then_some(widest / 64 + 1)
        }
    };
    match words {
        Some(words) => {
            eng.limits().try_resize(bits, words, 0u64)?;
            for plan in this_plans {
                let x = plan.x_idx as usize;
                bits[x >> 6] |= 1u64 << (x & 63);
            }
        }
        None => {
            fused_x.clear();
            // Each plan covers a distinct x_idx (Phase 1 emits one plan per
            // (node, x_idx) group), so the map holds one entry per plan — an
            // x_idx collision here would silently drop a fused pair's count.
            eng.limits().reserve_map(fused_x, this_plans.len())?;
            for plan in this_plans {
                fused_x.insert(plan.x_idx, plan.new_ref);
            }
            debug_assert_eq!(fused_x.len(), this_plans.len(), "plans must have distinct x_idx per node");
        }
    }

    // Keep the un-fused pairs, compacting them onto the front of the node's
    // own range: `write` never overtakes `read` (it advances at most once
    // per read, from the same origin), so a kept pair only ever moves down
    // onto a slot already read past. A pair is fused away iff its x-side index
    // carries a plan (see `fused_x` above).
    let mut write = start;
    let arena = level.pairs.stored_mut();
    match words {
        None => {
            for read in start..start + old_len {
                let p = arena[read];
                if !fused_x.contains_key(&x_of(p)) {
                    arena[write] = p;
                    write += 1;
                }
            }
        }
        Some(words) => {
            let planned = &bits[..words];
            for read in start..start + old_len {
                let p = arena[read];
                let x = x_of(p) as usize;
                let fused = planned.get(x >> 6).is_some_and(|w| (w >> (x & 63)) & 1 == 1);
                if !fused {
                    arena[write] = p;
                    write += 1;
                }
            }
            // Zero again for the next node: only the planned words were set.
            for plan in this_plans {
                bits[plan.x_idx as usize >> 6] = 0;
            }
        }
    }
    // Append one fused pair per plan, in plan order. Every read is done, and
    // each plan removed ≥2 pairs above, so `write + this_plans.len() ≤ start +
    // old_len` — the appends stay inside the node's own range and cannot reach
    // the next node's slots.
    for plan in this_plans {
        // `new_ref` is the fully-encoded marginal-side ref from Phase 2 —
        // either a tagged inline count (bit-30 set) or a bare slot
        // index (bit-30 clear), self-describing. Write it verbatim;
        // `x_idx` is the non-marginal side.
        let (x_idx, r_new) = (plan.x_idx, plan.new_ref);
        let fused = match side {
            ChildSide::Right => ChildPair::new(EncodedChildRef::from_raw(x_idx), EncodedChildRef::from_raw(r_new)),
            ChildSide::Left => ChildPair::new(EncodedChildRef::from_raw(r_new), EncodedChildRef::from_raw(x_idx)),
        };
        debug_assert!(write < start + old_len, "fusion must shrink the pair list");
        arena[write] = fused;
        write += 1;
    }

    // Re-encode via the shared epilogue: shrink in place, or inline the sole
    // survivor.
    Ok(level.reencode_shrunk(n, start, old_len, write - start))
}
