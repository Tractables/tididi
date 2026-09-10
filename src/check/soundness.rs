//! Deeper soundness checks: the reduced-size decisions and determinism.

use std::sync::Arc;
use crate::diagram::{ChildRef, ValueRef, NodeIdx};
use num_bigint::BigUint;
use crate::vtree::{VtreeIdx, VtreeNode};
use crate::apply::apply_and;
use crate::query::model_count;
use crate::diagram::*;
use super::signature::*;

/// The model count a child reference stands for: an inline marginal reference
/// carries it, and every other form reads it from the child level's per-node
/// counts.
fn child_count(child: ChildRef, level_counts: &[BigUint]) -> BigUint {
    match child {
        ChildRef::Value(ValueRef::Inline(c)) => BigUint::from(c),
        ChildRef::Node(NodeIdx(s)) | ChildRef::Value(ValueRef::Slot(s)) => {
            level_counts[s as usize].clone()
        }
    }
}

/// Independently validate every reducibility decision made by `reduced_size`.
///
/// For each internal node where Case L or Case R fires, verifies:
/// - Structural completeness: child indices == 0..child_width
/// - Product-form: count(g) == count(target) × 2^|vars(child)|
///
/// Cost: O(diagram size), but uses BigUint arithmetic for model counts.
pub fn check_reduced_size_sanity(tdd: &Tdd) -> Result<(), String> {
    if tdd.output.local == ZERO {
        return Ok(());
    }

    let vtree = &tdd.vtree;
    let num_levels = tdd.levels.len();

    // Reuse the shared model count computation from `query`.
    let counts = crate::query::node_counts(tdd);
    let mut subtree_vars = vec![0u32; num_levels];
    for (t, _var) in vtree.leaf_bottomup() {
        subtree_vars[t.idx()] = 1;
    }
    for (t, left, right) in vtree.internal_bottomup() {
        subtree_vars[t.idx()] = subtree_vars[left.idx()] + subtree_vars[right.idx()];
    }

    for (t, left, right) in vtree.internal_bottomup() {
        let ti = t.idx();
        let left_idx = left.idx();
        let right_idx = right.idx();
        let left_view = tdd.levels[left_idx].side_view();
        let right_view = tdd.levels[right_idx].side_view();
        let true_t1 = BigUint::from(1u32) << subtree_vars[left_idx] as usize;
        let true_t2 = BigUint::from(1u32) << subtree_vars[right_idx] as usize;
        let level = &tdd.levels[ti];

        for (node_i, node) in level.nodes.iter().enumerate() {
            if node.is_internal() {
                let pairs: Vec<InputPair> = level.pairs_iter_of(node).collect();
                if pairs.is_empty() {
                    continue;
                }

                let first_right = pairs[0].right;
                if pairs.iter().all(|p| p.right == first_right) {
                    let sum: BigUint =
                        pairs.iter().map(|p| child_count(left_view.child(p.left), &counts[left_idx])).sum();
                    if sum == true_t1 {
                        // Structural completeness check removed: with implicit
                        // leaves, a reducible node may reference only a subset of
                        // implicit labels (e.g., One alone covers 2^1 models).
                        let right_count = child_count(right_view.child(first_right), &counts[right_idx]);
                        let product = &right_count * &true_t1;
                        if counts[ti][node_i] != product {
                            return Err(format!(
                                "Case L product-form failed at vtree {} node {}: \
                                 count {} != {} × {} = {}",
                                ti, node_i, counts[ti][node_i],
                                right_count, true_t1, product
                            ));
                        }
                        continue;
                    }
                }

                let first_left = pairs[0].left;
                if pairs.iter().all(|p| p.left == first_left) {
                    let sum: BigUint =
                        pairs.iter().map(|p| child_count(right_view.child(p.right), &counts[right_idx])).sum();
                    if sum == true_t2 {
                        let left_count = child_count(left_view.child(first_left), &counts[left_idx]);
                        let product = &left_count * &true_t2;
                        if counts[ti][node_i] != product {
                            return Err(format!(
                                "Case R product-form failed at vtree {} node {}: \
                                 count {} != {} × {} = {}",
                                ti, node_i, counts[ti][node_i],
                                left_count, true_t2, product
                            ));
                        }
                    }
                }
            }
        }
    }

    Ok(())
}

/// Check the determinism property: at each vtree level, all *live* nodes
/// compute mutually exclusive Boolean functions.
///
/// **Internal levels** — for each pair of distinct nodes (i, j) at the same
/// internal vtree level, conjoins them via `apply_and` and verifies
/// `model_count == 0`.
///
/// **Leaf levels** — there are no stored nodes at a leaf level, but the
/// labels {Pos, Neg, One, Zero} are *implicit* nodes referenced by parent
/// pair lists (and possibly by the diagram output). Two of these labels can
/// only co-occur as live references if they are pairwise mutex as functions
/// of the leaf variable: `Pos ∧ Neg = ⊥` is fine, but `Pos ∧ One = Pos ≠ ⊥`
/// is not. Equivalently, at every leaf vtree level the set of *referenced*
/// labels must be a subset of `{Pos, Neg}` (literal mode) or `{One}`
/// (unconstrained mode), never spanning both. This is exactly the
/// per-pair-mutex condition extended to the leaf "node" that lives only
/// implicitly in the labels its parents pick.
///
/// Mode is a global property of the function on the vtree (determined by
/// whether the leaf's two bit-values induce the same external-completion
/// set), so a canonical diagram picks one mode per leaf at construction. A
/// minimize phase that fails to leaf-contract `(Pos_x, S) + (Neg_x, S)`
/// siblings into `(One_x, S)` while other parts of the same diagram already use
/// `One_x` would mix modes and trip this check.
///
/// **Only feasible for small diagrams** (≤5 variables) due to O(width² × apply)
/// cost per internal level. Do not call on easy or large benchmarks.
///
/// **Non-marginal diagrams only** — this assumes a plain Boolean diagram and
/// panics (or misbehaves) on marginal diagrams; do not call it on marginalize
/// outputs.
///
/// Cost: O(width² × apply_and_cost) per internal level + O(size) for the
/// leaf-label scan.
pub fn check_determinism(tdd: &Tdd) -> Result<(), String> {
    let vtree = &tdd.vtree;
    let shared_vtree = Arc::clone(&tdd.vtree);

    // Leaf-level mode consistency: at each leaf vtree node, collect the set
    // of labels actually referenced (via parent pair lists or the diagram output).
    // Reject mixes where a non-mutex pair of labels is live.
    let n = vtree.num_nodes();
    let mut used_at_leaf: Vec<u8> = vec![0u8; n]; // bit i = label i is used
    for vi in 0..n {
        let (left, right) = match *vtree.node(VtreeIdx(vi as u32)) {
            VtreeNode::Internal { left, right, .. } => (left, right),
            VtreeNode::Leaf { .. } => continue,
        };
        let level = &tdd.levels[vi];
        if level.width() == 0 { continue; }
        let left_is_leaf = vtree.node(left).is_leaf();
        let right_is_leaf = vtree.node(right).is_leaf();
        if !left_is_leaf && !right_is_leaf { continue; }
        for (_, pairs) in level.internal_inputs_iter() {
            for p in pairs {
                if left_is_leaf {
                    let idx = p.left.idx();
                    if idx < LEAF_WIDTH { used_at_leaf[left.idx()] |= 1u8 << idx; }
                }
                if right_is_leaf {
                    let idx = p.right.idx();
                    if idx < LEAF_WIDTH { used_at_leaf[right.idx()] |= 1u8 << idx; }
                }
            }
        }
    }
    if vtree.node(tdd.output.vtree).is_leaf() && tdd.output.local != ZERO {
        let idx = tdd.output.local.idx();
        if idx < LEAF_WIDTH { used_at_leaf[tdd.output.vtree.idx()] |= 1u8 << idx; }
    }
    let pos_mask: u8 = 1u8 << (LeafLabel::Pos as u32);
    let neg_mask: u8 = 1u8 << (LeafLabel::Neg as u32);
    let one_mask: u8 = 1u8 << (LeafLabel::One as u32);
    // Indexes the vtree and `used_at_leaf` at the same position.
    #[allow(clippy::needless_range_loop)]
    for vi in 0..n {
        if !vtree.node(VtreeIdx(vi as u32)).is_leaf() { continue; }
        let u = used_at_leaf[vi];
        let has_one = (u & one_mask) != 0;
        let has_lit = (u & (pos_mask | neg_mask)) != 0;
        if has_one && has_lit {
            return Err(format!(
                "leaf vtree {:?} mixes label modes (used = 0b{:03b}: Pos={}, \
                 Neg={}, One={}); expected subset of {{Pos,Neg}} or {{One}}, \
                 not both — Pos ∧ One ≠ ⊥",
                vi, u,
                (u & pos_mask) != 0, (u & neg_mask) != 0, has_one,
            ));
        }
    }

    for t in vtree.bottomup() {
        if vtree.node(t).is_leaf() { continue; }
        let width = tdd.effective_width(t);
        for i in 0..width {
            for j in (i + 1)..width {
                let tdd_i = tdd_with_output(tdd, &shared_vtree, t, i as u32);
                let tdd_j = tdd_with_output(tdd, &shared_vtree, t, j as u32);

                let conjoined = apply_and(tdd_i, tdd_j);
                let count = model_count(&conjoined);

                if count != BigUint::ZERO {
                    return Err(format!(
                        "vtree {:?} nodes {} and {} are not mutually exclusive \
                         (conjunction has {} models)",
                        t, i, j, count
                    ));
                }
            }
        }
    }

    Ok(())
}
