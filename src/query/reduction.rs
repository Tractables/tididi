//! Reduction metrics for compiled TDDs.
//!
//! `reduced_tdd_size` and `r2_reduced_tdd_size` estimate how much smaller
//! the TDD would be under SDD-style reduction rules (see `docs/tdd.md`).

use num_bigint::BigUint;

use crate::marg_slots::ChildSide;
use super::compute_node_counts;
use crate::diagram::*;

/// Estimate TDD size after applying SDD-style reduction rules (r1SDD metric).
///
/// A node is **reducible** if its sub-function depends on only one child's
/// variables — it can be "short-circuited" down to the relevant child.
///
/// Two symmetric cases:
///   - **Reducible to right** (all-same-right): every pair shares the same right
///     child, and the left children's model counts sum to 2^|vars(left)|. The
///     left side covers ALL assignments → function depends only on the right child.
///   - **Reducible to left** (all-same-left): symmetric — function depends only
///     on the left child.
///
/// Returns `tdd.size() - reducible_pairs`. See `docs/tdd.md` for details.
pub fn reduced_tdd_size(tdd: &Tdd) -> usize {
    // ZERO sentinel: the TDD is UNSAT, size is 0.
    if tdd.is_zero() {
        return 0;
    }

    let vtree = &tdd.vtree;
    let num_levels = tdd.levels.len();

    // Compute per-node model counts and subtree variable counts.
    let counts = compute_node_counts(tdd);
    let mut subtree_vars = vec![0u32; num_levels];
    for (t, _var) in vtree.leaf_bottomup() {
        subtree_vars[t.idx()] = 1;
    }
    for (t, left, right) in vtree.internal_bottomup() {
        subtree_vars[t.idx()] = subtree_vars[left.idx()] + subtree_vars[right.idx()];
    }

    // Count reducible pairs (pairs belonging to reducible nodes).
    // The debug_asserts below verify two structural invariants of reducible
    // nodes in debug builds — they are compiled out in release mode.
    let mut reducible_pairs = 0usize;

    for (t, left, right) in vtree.internal_bottomup() {
        let ti = t.idx();
        let li = left.idx();
        let ri = right.idx();
        // Marg-side refs are slot-tagged (bit 30): mask to the bare slot before
        // indexing the child's count array. Non-marg child indexes verbatim.
        let li_mask = if tdd.levels[li].is_marginal() { MARG_VALUE_MASK as usize } else { usize::MAX };
        let ri_mask = if tdd.levels[ri].is_marginal() { MARG_VALUE_MASK as usize } else { usize::MAX };
        let true_t1 = BigUint::from(1u32) << subtree_vars[li] as usize;
        let true_t2 = BigUint::from(1u32) << subtree_vars[ri] as usize;
        let level = &tdd.levels[ti];

        for (node_i, node) in level.nodes.iter().enumerate() {
            if node.is_internal() {
                // Materialize the pair list (cheap relative to BigUint sums
                // computed below). Survives both packed and unpacked levels.
                let pairs: Vec<InputPair> = level.pairs_iter_of(node).collect();
                if pairs.is_empty() {
                    continue;
                }
                // Reducible to right: all right sides identical; left model counts sum to 2^|vars(t1)|
                let first_right = pairs[0].right;
                if pairs.iter().all(|p| p.right == first_right) {
                    let sum: BigUint =
                        pairs.iter().map(|p| &counts[li][p.left.idx() & li_mask]).sum();
                    if sum == true_t1 {
                        reducible_pairs += pairs.len();

                        // Structural completeness assertion removed: with implicit
                        // leaves, a node can be reducible without referencing all
                        // 3 implicit labels (e.g., One alone covers 2^1 models).

                        debug_assert_eq!(
                            counts[ti][node_i],
                            &counts[ri][first_right.idx() & ri_mask] * &true_t1,
                            "reducible-to-right product-form check failed at \
                             vtree {ti} node {node_i}"
                        );

                        continue;
                    }
                }
                // Reducible to left: all left sides identical; right model counts sum to 2^|vars(t2)|
                let first_left = pairs[0].left;
                if pairs.iter().all(|p| p.left == first_left) {
                    let sum: BigUint =
                        pairs.iter().map(|p| &counts[ri][p.right.idx() & ri_mask]).sum();
                    if sum == true_t2 {
                        reducible_pairs += pairs.len();

                        debug_assert_eq!(
                            counts[ti][node_i],
                            &counts[li][first_left.idx() & li_mask] * &true_t2,
                            "reducible-to-left product-form check failed at \
                             vtree {ti} node {node_i}"
                        );
                    }
                }
            }
        }
    }

    tdd.size().saturating_sub(reducible_pairs)
}

/// Reduced TDD size under the r2TDD rule (structural variant, always ≤ r1SDD).
///
/// Like `reduced_tdd_size`, checks for nodes that can be short-circuited to
/// one child. But instead of computing model counts (O(size × `BigUint`)), uses
/// a purely structural check: a node is r2-reducible if one side enumerates
/// *all* nodes at the child level. This is cheaper (no `BigUint` arithmetic)
/// but strictly more aggressive — every r1SDD-reducible node is also
/// r2TDD-reducible, but not vice versa.
///
/// In a purely Boolean diagram pair lists are duplicate-free,
/// so "all lefts identical AND `pairs.len()` == `child_level.width()`" implies the
/// right indices are exactly {0, 1, ..., width-1}. In a marginalized diagram pair
/// lists are multisets and a repeated pair can inflate `pairs.len()` to the level
/// width without covering it — this diagnostic size metric may then over-count
/// reducible pairs. It feeds reporting only, never a model count.
pub fn r2_reduced_tdd_size(tdd: &Tdd) -> usize {
    if tdd.is_zero() {
        return 0;
    }

    let vtree = &tdd.vtree;
    let mut reducible_pairs = 0usize;

    for (t, left, right) in vtree.internal_bottomup() {
        let ti = t.idx();
        let level = &tdd.levels[ti];

        for node in level.nodes.iter() {
            if node.is_internal() {
                let pairs: Vec<InputPair> = level.pairs_iter_of(node).collect();
                if pairs.is_empty() {
                    continue;
                }
                // Reducible to right: all rights identical AND left side covers the full child width.
                let first_right = pairs[0].right;
                if pairs.iter().all(|p| p.right == first_right)
                    && covers_child_width(tdd, left, &pairs, ChildSide::Left)
                {
                    reducible_pairs += pairs.len();
                    continue;
                }
                // Reducible to left: all lefts identical AND right side covers the full child width.
                let first_left = pairs[0].left;
                if pairs.iter().all(|p| p.left == first_left)
                    && covers_child_width(tdd, right, &pairs, ChildSide::Right)
                {
                    reducible_pairs += pairs.len();
                }
            }
        }
    }

    tdd.size().saturating_sub(reducible_pairs)
}

/// Check if a set of pairs covers the full width of a child level.
///
/// For internal children: `pairs.len() == width` (all indices 0..width).
/// For leaf children: the referenced labels must cover all 2^1 = 2 models
/// of the single leaf variable. Pos and Neg each contribute 1 model;
/// One alone contributes 2 (covers both assignments).
fn covers_child_width(tdd: &Tdd, child: crate::vtree::VtreeIdx, pairs: &[InputPair], side: ChildSide) -> bool {
    if tdd.vtree.node(child).is_leaf() {
        let mut seen = [false; LEAF_WIDTH];
        for pair in pairs {
            let idx = if side == ChildSide::Left { pair.left.idx() } else { pair.right.idx() };
            seen[idx] = true;
        }
        // Model weights: One(0)=2, Pos(1)=1, Neg(2)=1.
        const LEAF_MODEL_WEIGHT: [usize; LEAF_WIDTH] = [2, 1, 1];
        let coverage: usize = (0..LEAF_WIDTH)
            .filter(|&i| seen[i])
            .map(|i| LEAF_MODEL_WEIGHT[i])
            .sum();
        coverage >= 2 // 2^1 = total models for one leaf variable
    } else {
        pairs.len() == tdd.levels[child.idx()].width()
    }
}
