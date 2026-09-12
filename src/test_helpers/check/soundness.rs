//! The determinism check.

use std::sync::Arc;
use num_bigint::BigUint;
use crate::vtree::{VtreeIdx, VtreeNode};
use crate::apply::apply_and;
use crate::query::model_count;
use crate::diagram::*;
use super::signature::*;

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
/// set), so a canonical diagram picks one mode per leaf at construction.
///
/// Only feasible for small diagrams (≤5 variables): O(width² × apply) per
/// internal level plus O(size) for the leaf-label scan.
///
/// Non-marginal diagrams only: it assumes a plain Boolean diagram and panics
/// or misbehaves on a marginal one.
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
        if level.slot_count() == 0 { continue; }
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
        let width = tdd.reference_slot_count(t);
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
