//! Identity/constant-true detection and the per-level identity fast paths for
//! the apply product construction.
//!
//! Split out of `conjoin/mod.rs` (pure code motion): leaf-identity precompute
//! (`init_leaf_identity` + `level_marginal_is_constant_true`), the shared
//! identity-swap body (`apply_identity_fast_path`), and the per-level fast-path
//! region (`try_level_fast_paths` + `FastPathResult`) plus the debug-only
//! marginal-schedule assert. The driver in `mod.rs` calls the `pub(super)`
//! entries; `eng.apply().subvars`/`eng.apply().marginal_stack` scratch pools,
//! `bump_live_count`, and `LevelGrid` stay in `mod.rs` and are reached via
//! `super::`.

use crate::engine::Engine;
use crate::vtree::VtreeIdx;
use crate::diagram::{self, *};
use crate::utils::{pool_take, pool_put};
use super::{ApplyError, LevelGrid, bump_live_count};

/// Compute which leaf levels are "identity" (constant-true) for a TDD operand.
///
/// A leaf level is identity if only the One label (local index 0, `LeafLabel::One`)
/// is referenced by parent pairs — meaning the TDD computes constant-true there.
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
/// identity-operand carry swaps in `try_level_fast_paths` fire on a
/// stale-TRUE leaf flag and the operand's content at `t` is silently dropped.
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
        // cleared so the loop below is a no-op; weight-marginal levels KEEP
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
    // by a constant-true marginal child* must be non-identity. The pair-scan
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
        let mut subvars = pool_take(&eng.apply().subvars);
        lim.try_resize(&mut subvars, num_nodes, 0u32)?;
        for (t, _) in vtree.leaf_bottomup() {
            subvars[t.idx()] = 1;
        }
        for (t, left, right) in vtree.internal_bottomup() {
            subvars[t.idx()] = subvars[left.idx()] + subvars[right.idx()];
        }
        let mut stack = pool_take(&eng.apply().marginal_stack);
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
        pool_put(&eng.apply().subvars, subvars);
        pool_put(&eng.apply().marginal_stack, stack);
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
    let counts = level.marginal_counts.as_ref().expect("is_marginal");
    if counts.len() != 1 {
        return false;
    }
    let c0 = counts[0];
    if subvars < 128 {
        // Target fits in u128: 2^subvars <= 2^127 < u128::MAX. A `c0 == u128::MAX`
        // sentinel means the real value overflowed u128, which is strictly > target,
        // so they cannot be equal.
        if c0 == u128::MAX {
            return false;
        }
        c0 == (1u128 << subvars)
    } else {
        // 2^subvars >= 2^128 > u128::MAX. If c0 didn't overflow, it can't reach.
        if c0 != u128::MAX {
            return false;
        }
        let target = num_bigint::BigUint::from(1u32) << subvars as usize;
        match level.marginal_counts_big.as_ref().and_then(|b| b.get(0)) {
            Some(b) => *b == target,
            None => false,
        }
    }
}

/// Shared body of the two identity fast-paths in `apply_and_fallible_inner`.
///
/// FP1 (`C1_IS_CARRIER = true`): c2 is the identity operand, c1 is the carrier.
/// FP2 (`C1_IS_CARRIER = false`): c1 is the identity operand, c2 is the carrier.
///
/// The guards (k==1, identity-child flags, marginal checks) are asymmetric and
/// remain inline at each call site. This function handles everything after the
/// guard is satisfied, up to (but not including) the `continue`.
///
/// `carrier_levels` is `c1.levels` when `C1_IS_CARRIER` else `c2.levels`.
/// `k_carrier` is `k1` when `C1_IS_CARRIER` else `k2` (analogously for `k_other`).
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
    might_use_sparse: bool,
    live_counts: &mut [usize],
    out_nodes_so_far: &mut u64,
    grids: &mut [LevelGrid],
    node_idx: &mut [u32],
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
        diagram::resolve_swapped_marg_side(
            eng,
            levels, t_idx, left_idx, &carrier_levels[left_idx], true,
        )?;
    }
    if carrier_levels[right_idx].is_marginal() && levels[right_idx].is_marginal() {
        diagram::resolve_swapped_marg_side(
            eng,
            levels, t_idx, right_idx, &carrier_levels[right_idx], false,
        )?;
    }

    if might_use_sparse {
        bump_live_count(live_counts, out_nodes_so_far, t_idx, k_carrier);
    } else {
        let t_base = grids[t_idx].base_unchecked();
        for idx in 0..k_carrier {
            node_idx[t_base + idx] = idx as u32;
        }
        grids[t_idx] = LevelGrid::DenseStrict { base: t_base };
    }
    Ok(())
}


/// Return type for [`try_level_fast_paths`].
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
    c1: &Tdd,
    c2: &Tdd,
    t: VtreeIdx,
    t_idx: usize,
    k1: usize,
    k2: usize,
    might_use_sparse: bool,
    c1_identity: &mut [bool],
    c2_identity: &mut [bool],
    live_counts: &mut [usize],
    out_nodes_so_far: &mut u64,
    grids: &mut [LevelGrid],
) -> FastPathResult {
    // 0-width marginal fast-path: both operands carry a 0-width marginal level
    // at t. This happens when the freeze cascade / `ensure_counts` processes a
    // sub-level structurally unreachable from the TDD output (0 nodes in the
    // disjoint sub-vtree). ensure_counts lacks the width==0 guard that
    // marginalize_batch has at line 701, so it emits Some(vec![]) and
    // the cascade calls make_marginal(vec![], None). The cross-product
    // 0×0=0; the output level is also a 0-width orphan. Neither identity
    // fast-path fires (both require k==1). Without this guard, the dense path
    // reaches pairs_view_into(0) on an empty nodes Vec and panics.
    // True upstream fix: add width()==0 guard to ensure_counts
    // (compile_marginalize.rs:878), but that restructuring is a separate task.
    if k1 == 0 && k2 == 0 && c1.level(t).is_marginal() && c2.level(t).is_marginal() {
        // A 0-width marginal is an orphan: consistent inputs cannot hold a
        // pair reference into an empty level, so no ancestor constrains or
        // reads this subtree — it is vacuously identity for the ancestor
        // fast-paths. Without these flags, the already-marginal ancestor
        // sitting above the orphan (its counts were snapshotted before the
        // orphan formed) fails both k==1 identity checks and falls through
        // to the dense path → the same empty-nodes panic one level up.
        c1_identity[t_idx] = true;
        c2_identity[t_idx] = true;
        if might_use_sparse {
            bump_live_count(live_counts, out_nodes_so_far, t_idx, 0);
        } else {
            let t_base = grids[t_idx].base_unchecked();
            grids[t_idx] = LevelGrid::DenseStrict { base: t_base };
        }
        return FastPathResult::Taken;
    }
    FastPathResult::NotTaken
}

/// Identity fast-path region for one vtree level.
///
/// Covers FP1 (`c1` is carrier / `c2` identity), FP2 (symmetric), the
/// zero-width orphan-marginal case, and the both-marginal-width-1 guard.
/// Any of these ends in a logical `continue` for the outer loop; this
/// function signals that by returning `FastPathResult::Taken`.
/// When no fast path matches, returns `FastPathResult::NotTaken`.
///
/// `grids` and `node_idx` are only mutated on the zero-width orphan path (in
/// non-sparse mode); on FP1/FP2, mutation flows through `apply_identity_fast_path`.
#[allow(clippy::too_many_arguments)]
pub(super) fn try_level_fast_paths(
    eng: &Engine,
    c1: &mut Tdd,
    c2: &mut Tdd,
    t: VtreeIdx,
    k1: usize,
    k2: usize,
    t_idx: usize,
    left_idx: usize,
    right_idx: usize,
    might_use_sparse: bool,
    levels: &mut [TddLevel],
    c1_identity: &mut [bool],
    c2_identity: &mut [bool],
    live_counts: &mut [usize],
    out_nodes_so_far: &mut u64,
    grids: &mut [LevelGrid],
    node_idx: &mut [u32],
) -> Result<FastPathResult, ApplyError> {
    // Identity internal: c2 has width 1 and both children were identity,
    // so c2's single node has one pair (0,0) referencing the identity nodes
    // at each child level. Product of c1[i] with c2[0] = c1[i] unchanged.
    //
    // A *marginal* width-1 level is NOT a count-neutral identity — its
    // single slot carries a model-count multiplier (the frozen sub-vtree's
    // mass). Dropping it (carrying c1) loses that mass. When c2's level
    // here is marginal, defer to the symmetric fast-path below, which
    // *carries* c2 and preserves the mass. The schedule guarantees c1 is
    // identity at t whenever c2 is marginal at t, so fast-path-2 is
    // eligible.
    //
    // Exception (mc-project cofactor path): when BOTH operands are marginal
    // width-1 with identity children on both sides, they froze the same
    // sub-function over the same scope (projection cofactors share the
    // frozen sub-TDD verbatim), so the conjunction carries that mass ONCE —
    // c1's level is kept and c2's is absorbed. Anything else reaching the
    // both-marginal state is unsound (a marginalized scope re-constrained);
    // the debug_assert below requires the frozen masses to be equal.
    let both_marg_w1 = k1 == 1 && k2 == 1
        && c1.levels[t_idx].is_marginal() && c2.levels[t_idx].is_marginal()
        && c1_identity[left_idx] && c1_identity[right_idx]
        && c2_identity[left_idx] && c2_identity[right_idx];
    #[cfg(debug_assertions)]
    if both_marg_w1 {
        debug_assert_eq!(
            c1.levels[t_idx].marginal_counts, c2.levels[t_idx].marginal_counts,
            "both-marginal width-1 conjunction at t={t_idx}: unequal frozen \
             masses — absorbing one side would be unsound"
        );
        debug_assert_eq!(
            c1.levels[t_idx].marginal_counts_big, c2.levels[t_idx].marginal_counts_big,
            "both-marginal width-1 conjunction at t={t_idx}: unequal frozen \
             big masses — absorbing one side would be unsound"
        );
    }
    if k2 == 1 && c2_identity[left_idx] && c2_identity[right_idx]
        && (!c2.levels[t_idx].is_marginal() || both_marg_w1)
        && !(!c1.levels[t_idx].is_marginal()
            && (levels[left_idx].is_marginal() || levels[right_idx].is_marginal()))
    {
        // FP1: c1 is the carrier, c2 is the identity operand.
        apply_identity_fast_path::<true>(
            eng,
            t_idx, left_idx, right_idx,
            k1, k2,
            &mut c1.levels, levels,
            c1_identity, c2_identity,
            might_use_sparse, live_counts, out_nodes_so_far, grids, node_idx,
        )?;
        // No drop here: the start-of-iteration drop already released the
        // children.
        return Ok(FastPathResult::Taken);
    }

    // Symmetric identity: c1 is constant-true at this subtree, copy c2's nodes.
    // Mirror of the fast-path-1 guard — a marginal c1 carries count mass
    // that this path would drop (it carries c2). Defer to fast-path-1 above
    // (which carries c1) when c1 is marginal.
    if k1 == 1 && c1_identity[left_idx] && c1_identity[right_idx]
        && !c1.levels[t_idx].is_marginal()
        && !(!c2.levels[t_idx].is_marginal()
            && (levels[left_idx].is_marginal() || levels[right_idx].is_marginal()))
    {
        // FP2: c2 is the carrier, c1 is the identity operand.
        apply_identity_fast_path::<false>(
            eng,
            t_idx, left_idx, right_idx,
            k2, k1,
            &mut c2.levels, levels,
            c2_identity, c1_identity,
            might_use_sparse, live_counts, out_nodes_so_far, grids, node_idx,
        )?;
        // No drop here: the start-of-iteration drop already released the
        // children.
        return Ok(FastPathResult::Taken);
    }

    if try_zero_width_marginal(
        c1, c2, t, t_idx, k1, k2, might_use_sparse,
        c1_identity, c2_identity, live_counts, out_nodes_so_far, grids,
    ) == FastPathResult::Taken {
        return Ok(FastPathResult::Taken);
    }

    Ok(FastPathResult::NotTaken)
}

/// Debug-only marginal-schedule assert (extraction 1).
///
/// Fires only when at least one operand's level `t` is marginal — the identity
/// fast-paths above MUST have consumed it before we reach the dense path.
/// Builds a subtree dump and asserts, then writes the dump to
/// `/tmp/tididi_crash_dump.txt` for post-mortem inspection.
#[cfg(debug_assertions)]
#[allow(clippy::too_many_arguments)]
pub(super) fn debug_assert_marg_schedule(
    c1: &Tdd,
    c2: &Tdd,
    t: VtreeIdx,
    left: VtreeIdx,
    right: VtreeIdx,
    vtree: &crate::vtree::Vtree,
    k1: usize,
    k2: usize,
    left_idx: usize,
    right_idx: usize,
    c1_widths: &[usize],
    c2_widths: &[usize],
    c1_identity: &[bool],
    c2_identity: &[bool],
) {
    if c1.level(t).is_marginal() || c2.level(t).is_marginal() {
        // Walk only the subtree rooted at t (recursively) — full vtree
        // dumps overflow stderr buffers on large CNFs. Per-node format:
        // depth-indented vtree-id with widths, marginal flags, identity flags.
        let mut subtree_dump = String::new();
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
            subtree_dump.push_str(&format!(
                "{indent}v{vi} {kind}: c1.w={} c2.w={} c1.marg={} c2.marg={} c1_id={} c2_id={} \
                 c1.nodes={} c2.nodes={}\n",
                c1_widths[vi], c2_widths[vi],
                c1.levels[vi].is_marginal(), c2.levels[vi].is_marginal(),
                c1_identity[vi], c2_identity[vi],
                c1.levels[vi].nodes.len(), c2.levels[vi].nodes.len(),
            ));
            if let crate::vtree::VtreeNode::Internal { left, right, .. } = n {
                stack.push((*right, depth + 1));
                stack.push((*left, depth + 1));
            }
        }
        // Persist to file so the full dump survives stderr truncation.
        let _ = std::fs::write("/tmp/tididi_crash_dump.txt", &subtree_dump);
        assert!(
            !c1.level(t).is_marginal(),
            "apply_and: c1 marginal at vtree node {t:?} (left={left:?} right={right:?}) \
             but c2 not identity (k1={k1}, k2={k2}, c2_id[left]={}, c2_id[right]={}). \
             Marginal pair structure cannot conjoin with a non-trivial operand. \
             Likely a stale marginalize schedule. \
             Subtree dump (also at /tmp/tididi_crash_dump.txt):\n{}",
            c2_identity[left_idx], c2_identity[right_idx], subtree_dump,
        );
        assert!(
            !c2.level(t).is_marginal(),
            "apply_and: c2 marginal at vtree node {t:?} (left={left:?} right={right:?}) \
             but c1 not identity (k1={k1}, k2={k2}, c1_id[left]={}, c1_id[right]={}). \
             Marginal pair structure cannot conjoin with a non-trivial operand. \
             Likely a stale marginalize schedule. \
             Subtree dump (also at /tmp/tididi_crash_dump.txt):\n{}",
            c1_identity[left_idx], c1_identity[right_idx], subtree_dump,
        );
    }
}
