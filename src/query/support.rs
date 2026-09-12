//! Read-only structural queries: variable support and implied literals.
//!
//! - `implied_literals` — literals forced true in every model.


use rustc_hash::FxHashMap;

use crate::diagram::{Literal, NodeIdx, Tdd};
use crate::diagram::{ONE_LEAF_IDX, POS_LEAF_IDX, NEG_LEAF_IDX};
use crate::vtree::{VarId, VtreeIdx, VtreeNode};



/// Implied literals (the **backbone**) of `f`: every literal that holds in every
/// model of `f`, sorted by variable. Cheap structural read — no conditioning, no
/// `model_count`, no clone — usable as a general diagram analysis primitive
/// anywhere a minimized diagram is in hand. A caller that wants set membership
/// builds a set from the result.
///
/// **Requires `f` minimized.** On a minimized diagram a leaf reference is reachable iff
/// it lies on a satisfying path, so for each variable we just collect which leaf
/// labels its leaf is ever referenced with: `Pos` (var=true on this path), `Neg`
/// (var=false), `One` (don't-care — var free on this path). The variable is implied
/// iff its leaf is referenced with **exactly one** of `Pos`/`Neg` and **never**
/// `One`: a lone `Pos` ⇒ implied-true, a lone `Neg` ⇒ implied-false. Seeing both
/// polarities (both values satisfiable) or any `One` (a model leaves it free) ⇒ not
/// implied. One O(size) pass over the pairs; correctness rides entirely on `f` being
/// minimized (unreachable refs would forge phantom labels). Zero / marginalized
/// variables contribute nothing (a summed-out variable is no longer a literal).
#[must_use]
pub fn implied_literals(f: &Tdd) -> Vec<Literal> {
    let mut out = Vec::new();
    if f.is_zero() {
        return out;
    }
    // Per-variable referenced-label bitmask: 1 = Pos, 2 = Neg, 4 = One (don't-care).
    let bit = |child: NodeIdx| -> u8 {
        if child == POS_LEAF_IDX {
            1
        } else if child == NEG_LEAF_IDX {
            2
        } else if child == ONE_LEAF_IDX {
            4
        } else {
            0
        }
    };
    let vt = &f.vtree;
    let mut mask: FxHashMap<VarId, u8> = FxHashMap::default();
    // Whole-diagram-is-a-single-literal case: the output sits at the leaf.
    if let VtreeNode::Leaf { var, .. } = *vt.node(f.output.vtree)
        && !f.levels[f.output.vtree.idx()].is_marginal() {
            *mask.entry(var).or_insert(0) |= bit(f.output.local);
        }
    for vi in 0..vt.num_nodes() {
        let (left, right) = match *vt.node(VtreeIdx(vi as u32)) {
            VtreeNode::Internal { left, right, .. } => (left.idx(), right.idx()),
            VtreeNode::Leaf { .. } => continue,
        };
        let level = &f.levels[vi];
        let left_marginal = f.levels[left].is_marginal();
        let right_marginal = f.levels[right].is_marginal();
        let left_var = match *vt.node(VtreeIdx(left as u32)) {
            VtreeNode::Leaf { var, .. } if !left_marginal => Some(var),
            _ => None,
        };
        let right_var = match *vt.node(VtreeIdx(right as u32)) {
            VtreeNode::Leaf { var, .. } if !right_marginal => Some(var),
            _ => None,
        };
        if left_var.is_none() && right_var.is_none() {
            continue;
        }
        for ni in 0..level.nodes.len() {
            if level.nodes[ni].is_leaf() {
                continue;
            }
            for p in level.pairs_of(&level.nodes[ni]) {
                if let Some(var) = left_var {
                    *mask.entry(var).or_insert(0) |= bit(p.left);
                }
                if let Some(var) = right_var {
                    *mask.entry(var).or_insert(0) |= bit(p.right);
                }
            }
        }
    }
    for (var, m) in mask {
        if m == 1 {
            out.push(Literal::pos(var));
        } else if m == 2 {
            out.push(Literal::neg(var));
        }
    }
    // The mask is a hash map, so the walk order is not the caller's; one variable
    // contributes at most one literal, so sorting by variable is a total order.
    out.sort_unstable_by_key(|lit| lit.var.0);
    out
}

