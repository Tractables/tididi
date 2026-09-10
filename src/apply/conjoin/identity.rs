//! Identity/constant-true detection and the per-level identity fast paths for
//! the apply product construction.
//!
//! This module owns the leaf-identity precompute (`init_leaf_identity` +
//! `level_marginal_is_constant_true`), the shared identity-swap body
//! (`apply_identity_fast_path`), and the per-level fast-path region
//! (`take_level_fast_path` + `FastPathResult`) plus the debug-only
//! marginal-schedule assert. The driver in `mod.rs` calls the `pub(super)`
//! entries; `eng.apply().subvars`/`eng.apply().marginal_stack` scratch pools,
//! `bump_live_count`, and `LevelGrid` belong to `mod.rs` and are reached via
//! `super::`.

use crate::engine::Engine;
use crate::vtree::VtreeIdx;
use crate::diagram::{self, *};
use super::ApplyError;
use super::output::LiveCounts;
use super::grid_arena::GridArena;

/// Compute which leaf levels are "identity" (constant-true) for a diagram operand.
///
/// A leaf level is identity if only the One label (local index 0, `LeafLabel::One`)
/// is referenced by parent pairs — meaning the diagram computes constant-true there.
/// Identity levels can be skipped during product construction (x ∧ 1 = x).
///
/// Algorithm: assume all leaves are identity, then scan parent pairs bottom-up.
/// Any reference to Pos (0) or Neg (1) proves the leaf is not identity.
///
/// Marginal-level correction: `internal_inputs_iter()` returns nothing for marginal
/// levels (their pair structure was marginalized into per-node counts), so the
/// pair-scan can't refute identity through a marginal parent. A leaf under a
/// marginal subtree is identity in this operand only if the marginal subtree's
/// top level represents the constant-true function: width 1 with
/// `marginal_counts[0] == 2^subvars_t`. Any other shape (width > 1, or width 1
/// with a smaller count) means the subtree carries non-trivial constraints,
/// and every leaf below must be marked non-identity — otherwise the
/// identity-operand carry swaps in `take_level_fast_path` fire on a
/// leaf flag left true and the operand's content at `t` is silently dropped.
pub(super) fn init_leaf_identity(eng: &Engine, buf: &mut Vec<bool>, tdd: &Tdd, vtree: &crate::vtree::Vtree, num_nodes: usize) -> Result<(), ApplyError> {
    let lim = eng.limits();
    lim.try_resize(buf, num_nodes, false)?;
    for (t, _) in vtree.leaf_bottomup() {
        buf[t.idx()] = true;  // assume identity until proven otherwise
    }
    // Internal levels: computed from children, not preset.
    buf[vtree.num_leaves() as usize..num_nodes].fill(false);
    // Scan parent pairs: any reference to Pos (0) or Neg (1) means not identity.
    let mut has_any_marginal = false;
    for (t, left, right) in vtree.internal_bottomup() {
        if tdd.levels[t.idx()].is_marginal() { has_any_marginal = true; }
        let left_leaf = vtree.node(left).is_leaf();
        let right_leaf = vtree.node(right).is_leaf();
        // The only writes this level can make are `buf[left] = false` and
        // `buf[right] = false` — monotone true→false. So once neither flag is
        // still true there is nothing left to prove: skip the level at entry,
        // and stop the node/pair scan the moment both have fallen. Every
        // suppressed iteration could only have re-assigned `false` to a flag
        // already false, so the resulting `buf` is value-identical.
        // (Subsumes the old `!left_leaf && !right_leaf` skip: a non-leaf side
        // is never a candidate.)
        let mut want_left = left_leaf && buf[left.idx()];
        let mut want_right = right_leaf && buf[right.idx()];
        if !want_left && !want_right { continue; }
        let level = &tdd.levels[t.idx()];
        // Marginal levels have no structural pairs to scan — they're handled by
        // the `has_any_marginal` block below using per-node counts (integer) or
        // the marginal-forest walk. For integer-marginal levels `nodes` is also
        // cleared so the loop below is a no-op; weight-marginal levels keep
        // `nodes` (for `width()`) but clear `pairs`, so `pairs_of` would index an
        // empty `pairs`. Skip them explicitly. (Regular MC has no marginal
        // levels, so this guard is a no-op there.)
        if level.is_marginal() { continue; }
        'nodes: for node in level.nodes.iter() {
            if !node.is_internal() { continue; }
            for pair in level.pairs_of(node) {
                if want_left && pair.left != ONE_LEAF_IDX {
                    buf[left.idx()] = false;
                    want_left = false;
                }
                if want_right && pair.right != ONE_LEAF_IDX {
                    buf[right.idx()] = false;
                    want_right = false;
                }
                // Bitwise `|`: both operands are plain locals, so no branch.
                if !(want_left | want_right) { break 'nodes; }
            }
        }
    }
    // Marginal-level case: for any marginal level whose value isn't
    // constant-true at t, every leaf descendant *that isn't itself sheltered
    // by a constant-true marginal child* Must be non-identity. The pair-scan
    // above couldn't see those references — the pairs no longer exist — so
    // we correct here using the surviving per-node counts.
    //
    // Walk each marginal forest top-down from its root (a marginal level
    // whose vtree parent is not marginal). At each node:
    //   - marginal & constant-true → prune (entire subtree is identity)
    //   - marginal & non-CT        → recurse into both children
    //   - leaf reached             → mark non-identity
    // Pruning at CT intermediate levels lets us preserve fast-path
    // propagation when a marginal level's constraint is localized to a
    // sub-region.
    if has_any_marginal {
        let mut subvars = eng.apply().subvars.take();
        lim.try_resize(&mut subvars, num_nodes, 0u32)?;
        for (t, _) in vtree.leaf_bottomup() {
            subvars[t.idx()] = 1;
        }
        for (t, left, right) in vtree.internal_bottomup() {
            subvars[t.idx()] = subvars[left.idx()] + subvars[right.idx()];
        }
        let mut stack = eng.apply().marginal_stack.take();
        stack.clear();
        for (t, _, _) in vtree.internal_bottomup() {
            let level = &tdd.levels[t.idx()];
            if !level.is_marginal() { continue; }
            // Process only marginal-subtree roots (parent is non-marginal or
            // the level has no parent). Children of a marginal level reach
            // their leaves through the top-down recursion below.
            if let Some(p) = vtree.node(t).parent()
                && tdd.levels[p.idx()].is_marginal() { continue; }
            // Root-prune: if the root itself is CT, the whole subtree is
            // identity; nothing to mark.
            if level_marginal_is_constant_true(level, subvars[t.idx()]) { continue; }
            stack.push(t);
            while let Some(node) = stack.pop() {
                match vtree.node(node) {
                    crate::vtree::VtreeNode::Leaf { .. } => buf[node.idx()] = false,
                    crate::vtree::VtreeNode::Internal { left, right, .. } => {
                        // Children of a marginal level are marginal by the
                        // invariant — but a child may itself be CT, in which
                        // case its leaf descendants stay identity.
                        let left_lvl = &tdd.levels[left.idx()];
                        let right_lvl = &tdd.levels[right.idx()];
                        if !(left_lvl.is_marginal()
                            && level_marginal_is_constant_true(left_lvl, subvars[left.idx()]))
                        {
                            stack.push(*left);
                        }
                        if !(right_lvl.is_marginal()
                            && level_marginal_is_constant_true(right_lvl, subvars[right.idx()]))
                        {
                            stack.push(*right);
                        }
                    }
                }
            }
        }
        eng.apply().subvars.put(subvars);
        eng.apply().marginal_stack.put(stack);
    }
    Ok(())
}

/// True iff the marginal level represents the constant-true function over its
/// subtree variables: width 1 with model count exactly 2^subvars.
///
/// Fast path: when `subvars < 128`, the target `1u128 << subvars` fits in u128
/// and we compare directly without allocating a BigUint. When `subvars >= 128`,
/// the target exceeds u128, so the only way `counts[0]` can match is via the
/// overflow sentinel + BigUint side table.
pub(super) fn level_marginal_is_constant_true(level: &TddLevel, subvars: u32) -> bool {
    debug_assert!(level.is_marginal());
    // Weighted marginal levels carry no integer counts (values live
    // in the WeightStore), so this integer constant-true fast-path can't apply.
    // Returning false just skips the optimization — always sound (the general
    // apply path handles the marginal child structurally).
    if level.is_weight_marginal() {
        return false;
    }
    let counts = level.marginal_counts().expect("is_marginal");
    if counts.len() != 1 {
        return false;
    }
    let c0 = counts[0];
    if subvars < 128 {
        // Target fits in u128: 2^subvars <= 2^127 < `u128::MAX`. A `c0 == u128::MAX`
        // sentinel means the real value overflowed u128, which is strictly > target,
        // so they cannot be equal.
        if c0 == u128::MAX {
            return false;
        }
        c0 == (1u128 << subvars)
    } else {
        // 2^subvars >= 2^128 > `u128::MAX`. If c0 didn't overflow, it can't reach.
        if c0 != u128::MAX {
            return false;
        }
        let target = num_bigint::BigUint::from(1u32) << subvars as usize;
        match level.marginal_counts_big().and_then(|b| b.get(0)) {
            Some(b) => *b == target,
            None => false,
        }
    }
}

/// Shared body of the two identity fast-paths in `apply_and_fallible_inner`.
///
/// FP1 (`C1_IS_CARRIER = true`): g is the identity operand, f is the carrier.
/// FP2 (`C1_IS_CARRIER = false`): f is the identity operand, g is the carrier.
///
/// The guards (k==1, identity-child flags, marginal checks) are asymmetric and
/// remain inline at each call site. This function handles everything after the
/// guard is satisfied, up to (but not including) the `continue`.
///
/// `carrier_levels` is `f.levels` when `C1_IS_CARRIER` else `g.levels`.
/// `k_carrier` is `left_width` when `C1_IS_CARRIER` else `right_width` (analogously for `k_other`).
/// `carrier_identity` / `id_identity` are the identity-flag slices for the
/// carrier and identity operands respectively.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn apply_identity_fast_path<const C1_IS_CARRIER: bool>(
    eng: &Engine,
    t_idx: usize,
    left_idx: usize,
    right_idx: usize,
    k_carrier: usize,
    _k_other: usize,
    carrier_levels: &mut [TddLevel],
    levels: &mut [TddLevel],
    carrier_identity: &mut [bool],
    id_identity: &mut [bool],
    live_counts: &mut LiveCounts,
    arena: &mut GridArena,
) -> Result<(), ApplyError> {
    // Mark the identity operand's slot as identity at this level. The carrier
    // operand's slot is also identity-shaped if it has width 1 with identity
    // children — set it too so ancestors see both flags (neither should be
    // silently dropped by the early `continue`).
    id_identity[t_idx] = true;
    if k_carrier == 1 && carrier_identity[left_idx] && carrier_identity[right_idx] {
        carrier_identity[t_idx] = true;
    }

    std::mem::swap(&mut levels[t_idx], &mut carrier_levels[t_idx]);

    // When a source-marginal child remains marginal in the output, re-resolve the
    // swapped-in carrier level's bit-30-tagged refs into the output child
    // store-space. `?`: the re-resolve reserves the destination store growth it
    // needs before rewriting anything, so `OverBudget` here aborts the apply with
    // the swapped-in level untouched — never half-remapped.
    if carrier_levels[left_idx].is_marginal() && levels[left_idx].is_marginal() {
        diagram::resolve_swapped_marginal_side(
            eng,
            levels, t_idx, left_idx, &carrier_levels[left_idx], true,
        )?;
    }
    if carrier_levels[right_idx].is_marginal() && levels[right_idx].is_marginal() {
        diagram::resolve_swapped_marginal_side(
            eng,
            levels, t_idx, right_idx, &carrier_levels[right_idx], false,
        )?;
    }

    if arena.is_bump() {
        live_counts.bump(t_idx, k_carrier);
    } else {
        let output_grid_base = arena.materialized(t_idx).expect("a pre-planned layout grids every level");
        let slab = arena.slab_mut();
        for idx in 0..k_carrier {
            slab[output_grid_base.idx() + idx] = idx as u32;
        }
        arena.set_dense(t_idx, output_grid_base);
    }
    Ok(())
}


/// Return type for [`take_level_fast_path`].
///
/// `Taken` means a fast path fired and the call site should reclaim the
/// consumed child grids (`reclaim_child_grids!`) then `continue` the outer loop.
/// `NotTaken` means no fast path fired; fall through to the dense/sparse path.
#[derive(PartialEq, Eq)]
pub(super) enum FastPathResult {
    Taken,
    NotTaken,
}

/// The fast path for a level both operands made marginal with zero width.
///
/// Such a level is an orphan: a consistent diagram cannot hold a pair
/// referencing an empty level, so nothing above reads this subtree and it is
/// vacuously the identity for the ancestors' own fast paths. Flagging it keeps
/// an already-marginal ancestor from falling through to the dense route, which
/// would read pairs out of an empty level.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn try_zero_width_marginal(
    f: &Tdd,
    g: &Tdd,
    t: VtreeIdx,
    t_idx: usize,
    left_width: usize,
    right_width: usize,
    left_identity: &mut [bool],
    right_identity: &mut [bool],
    live_counts: &mut LiveCounts,
    arena: &mut GridArena,
) -> FastPathResult {
    // 0-width marginal fast-path: both operands carry a 0-width marginal level
    // at t. This happens when the marginalize cascade / `ensure_counts` processes a
    // sub-level structurally unreachable from the diagram output (0 nodes in the
    // disjoint sub-vtree). ensure_counts lacks the width==0 guard that
    // marginalize_batch has at line 701, so it emits Some(vec![]) and
    // the cascade calls become_marginal(vec![], None). The cross-product
    // 0×0=0; the output level is also a 0-width orphan. Neither identity
    // fast-path fires (both require k==1). Without this guard, the dense path
    // reaches pairs_of_idx(0) on an empty nodes Vec and panics.
    // True upstream fix: add width()==0 guard to ensure_counts
    // in the marginalize pass, but that restructuring is a separate task.
    if left_width == 0 && right_width == 0 && f.level(t).is_marginal() && g.level(t).is_marginal() {
        // A 0-width marginal is an orphan: consistent inputs cannot hold a
        // pair reference into an empty level, so no ancestor constrains or
        // reads this subtree — it is vacuously identity for the ancestor
        // fast-paths. Without these flags, the already-marginal ancestor
        // sitting above the orphan (its counts were snapshotted before the
        // orphan formed) fails both k==1 identity checks and falls through
        // to the dense path → the same empty-nodes panic one level up.
        left_identity[t_idx] = true;
        right_identity[t_idx] = true;
        if arena.is_bump() {
            live_counts.bump(t_idx, 0);
        } else {
            let output_grid_base = arena.materialized(t_idx).expect("a pre-planned layout grids every level");
            arena.set_dense(t_idx, output_grid_base);
        }
        return FastPathResult::Taken;
    }
    FastPathResult::NotTaken
}

/// Identity fast-path region for one vtree level.
///
/// Covers FP1 (`f` is carrier / `g` identity), FP2 (symmetric), the
/// zero-width orphan-marginal case, and the both-marginal-width-1 guard.
/// Any of these ends in a logical `continue` for the outer loop; this
/// function signals that by returning `FastPathResult::Taken`.
/// When no fast path matches, returns `FastPathResult::NotTaken`.
///
/// The arena is only written on the zero-width orphan path (and only when the
/// layout is pre-planned); on FP1/FP2 that write flows through
/// `apply_identity_fast_path`.
#[allow(clippy::too_many_arguments)]
pub(super) fn take_level_fast_path(
    eng: &Engine,
    f: &mut Tdd,
    g: &mut Tdd,
    t: VtreeIdx,
    left_width: usize,
    right_width: usize,
    t_idx: usize,
    left_idx: usize,
    right_idx: usize,
    levels: &mut [TddLevel],
    left_identity: &mut [bool],
    right_identity: &mut [bool],
    live_counts: &mut LiveCounts,
    arena: &mut GridArena,
) -> Result<FastPathResult, ApplyError> {
    // Identity internal: g has width 1 and both children were identity,
    // so g's single node has one pair (0,0) referencing the identity nodes
    // at each child level. Product of f[i] with g[0] = f[i] unchanged.
    //
    // A *marginal* width-1 level is not a count-neutral identity — its
    // single slot carries a model-count multiplier (the marginal sub-vtree's
    // mass). Dropping it (carrying f) loses that mass. When g's level
    // here is marginal, defer to the symmetric fast-path below, which
    // *carries* g and preserves the mass. The schedule guarantees f is
    // identity at t whenever g is marginal at t, so fast-path-2 is
    // eligible.
    //
    // Exception (the projection cofactor path): when both operands are
    // marginal width-1 with identity children on both sides, they summed out
    // the same sub-function over the same scope, so the conjunction carries
    // that count once — f's level is kept and g's absorbed. Anything else
    // reaching the both-marginal state is unsound (a summed-out scope
    // re-constrained); the debug assertion below requires the two counts to
    // be equal.
    let both_marginal_w1 = left_width == 1 && right_width == 1
        && f.levels[t_idx].is_marginal() && g.levels[t_idx].is_marginal()
        && left_identity[left_idx] && left_identity[right_idx]
        && right_identity[left_idx] && right_identity[right_idx];
    #[cfg(debug_assertions)]
    if both_marginal_w1 {
        debug_assert_eq!(
            f.levels[t_idx].marginal_counts(), g.levels[t_idx].marginal_counts(),
            "both-marginal width-1 conjunction at t={t_idx}: unequal marginal \
             masses — absorbing one side would be unsound"
        );
        debug_assert_eq!(
            f.levels[t_idx].marginal_counts_big(), g.levels[t_idx].marginal_counts_big(),
            "both-marginal width-1 conjunction at t={t_idx}: unequal marginal \
             big masses — absorbing one side would be unsound"
        );
    }
    if right_width == 1 && right_identity[left_idx] && right_identity[right_idx]
        && (!g.levels[t_idx].is_marginal() || both_marginal_w1)
        && !(!f.levels[t_idx].is_marginal()
            && (levels[left_idx].is_marginal() || levels[right_idx].is_marginal()))
    {
        // FP1: f is the carrier, g is the identity operand.
        apply_identity_fast_path::<true>(
            eng,
            t_idx, left_idx, right_idx,
            left_width, right_width,
            &mut f.levels, levels,
            left_identity, right_identity,
            live_counts, arena,
        )?;
        // No drop here: the start-of-iteration drop already released the
        // children.
        return Ok(FastPathResult::Taken);
    }

    // Symmetric identity: f is constant-true at this subtree, copy g's nodes.
    // Mirror of the fast-path-1 guard — a marginal f carries count mass
    // that this path would drop (it carries g). Defer to fast-path-1 above
    // (which carries f) when f is marginal.
    if left_width == 1 && left_identity[left_idx] && left_identity[right_idx]
        && !f.levels[t_idx].is_marginal()
        && !(!g.levels[t_idx].is_marginal()
            && (levels[left_idx].is_marginal() || levels[right_idx].is_marginal()))
    {
        // FP2: g is the carrier, f is the identity operand.
        apply_identity_fast_path::<false>(
            eng,
            t_idx, left_idx, right_idx,
            right_width, left_width,
            &mut g.levels, levels,
            right_identity, left_identity,
            live_counts, arena,
        )?;
        // No drop here: the start-of-iteration drop already released the
        // children.
        return Ok(FastPathResult::Taken);
    }

    if try_zero_width_marginal(
        f, g, t, t_idx, left_width, right_width,
        left_identity, right_identity, live_counts, arena,
    ) == FastPathResult::Taken {
        return Ok(FastPathResult::Taken);
    }

    Ok(FastPathResult::NotTaken)
}

/// The subtree rooted at `t`, one line per vtree node: both operands' widths,
/// marginal flags, identity flags and node counts.
///
/// Decorates the marginalize-schedule panic, which reports the offending node
/// but not the shape around it. A full vtree dump overflows stderr buffers on
/// large formulas, so the walk stops at `t`'s subtree.
#[cfg(debug_assertions)]
#[cold]
#[allow(clippy::too_many_arguments)]
pub(super) fn marginal_schedule_dump(
    f: &Tdd,
    g: &Tdd,
    t: VtreeIdx,
    vtree: &crate::vtree::Vtree,
    left_widths: &[usize],
    right_widths: &[usize],
    left_identity: &[bool],
    right_identity: &[bool],
) -> String {
    let mut dump = String::from("\nSubtree dump:\n");
    let mut stack: Vec<(VtreeIdx, usize)> = vec![(t, 0)];
    while let Some((node, depth)) = stack.pop() {
        let vi = node.idx();
        let indent = "  ".repeat(depth);
        let n = vtree.node(VtreeIdx(vi as u32));
        let kind = match n {
            crate::vtree::VtreeNode::Leaf { var, .. } => format!("Leaf(var={})", var.idx()),
            crate::vtree::VtreeNode::Internal { left, right, .. } => {
                format!("Internal(L={},R={})", left.idx(), right.idx())
            }
        };
        dump.push_str(&format!(
            "{indent}v{vi} {kind}: f.w={} g.w={} f.marginal={} g.marginal={} left_id={} right_id={} \
             f.nodes={} g.nodes={}\n",
            left_widths[vi], right_widths[vi],
            f.levels[vi].is_marginal(), g.levels[vi].is_marginal(),
            left_identity[vi], right_identity[vi],
            f.levels[vi].nodes.len(), g.levels[vi].nodes.len(),
        ));
        if let crate::vtree::VtreeNode::Internal { left, right, .. } = n {
            stack.push((*right, depth + 1));
            stack.push((*left, depth + 1));
        }
    }
    dump
}
