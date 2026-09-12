//! Read-only structural query: the literals forced true in every model.


use rustc_hash::FxHashMap;

use crate::diagram::{EncodedChildRef, Literal, Tdd};
use crate::diagram::{ONE_LEAF_IDX, POS_LEAF_IDX, NEG_LEAF_IDX};
use crate::vtree::{VarId, VtreeIdx, VtreeNode};



/// Implied literals (the backbone) of `f`: every literal that holds in every
/// model of `f`, sorted by variable. One O(size) pass over the pairs, no
/// conditioning and no counting.
///
/// Requires `f` minimized: the pass reads which leaf labels each variable's
/// leaf is referenced with, and an unreachable reference on an unminimized
/// diagram would count as a label. The zero diagram and summed-out variables
/// contribute nothing, and a variable no pair references is not implied. No
/// engine and no limit are involved.
#[must_use]
pub fn implied_literals(f: &Tdd) -> Vec<Literal> {
    let mut out = Vec::new();
    if f.is_zero() {
        return out;
    }
    // Per-variable referenced-label bitmask: 1 = Pos, 2 = Neg, 4 = One (don't-care).
    let bit = |child: EncodedChildRef| -> u8 {
        if child == POS_LEAF_IDX.into() {
            1
        } else if child == NEG_LEAF_IDX.into() {
            2
        } else if child == ONE_LEAF_IDX.into() {
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
            *mask.entry(var).or_insert(0) |= bit(f.output.local.into());
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

