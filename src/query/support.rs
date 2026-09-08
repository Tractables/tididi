//! Read-only structural queries: variable support and implied literals.
//!
//! - `implied_literals` — literals forced true in every model.


use crate::diagram::{LocalNodeIdx, Tdd};
use crate::apply::project::{POS, NEG, ONE};
use crate::vtree::{VarId, VtreeIdx, VtreeNode};



/// Implied literals (the **backbone**) of `t`: every `(var, value)` that holds in
/// EVERY model of `t`. Cheap structural read — no conditioning, no `model_count`,
/// no clone — usable as a general TDD analysis primitive anywhere a minimized
/// diagram is in hand.
///
/// **Requires `t` minimized.** On a minimized TDD a leaf reference is reachable iff
/// it lies on a satisfying path, so for each variable we just collect which leaf
/// labels its leaf is ever referenced with: `Pos` (var=true on this path), `Neg`
/// (var=false), `One` (don't-care — var free on this path). The variable is implied
/// iff its leaf is referenced with **exactly one** of `Pos`/`Neg` and **never**
/// `One`: a lone `Pos` ⇒ implied-true, a lone `Neg` ⇒ implied-false. Seeing both
/// polarities (both values satisfiable) or any `One` (a model leaves it free) ⇒ not
/// implied. One O(size) pass over the pairs; correctness rides entirely on `t` being
/// minimized (unreachable refs would forge phantom labels). Zero / marginalized
/// variables contribute nothing (a summed-out variable is no longer a literal).
pub fn implied_literals(f: &Tdd) -> std::collections::HashSet<(VarId, bool)> {
    let mut out = std::collections::HashSet::new();
    if f.is_zero() {
        return out;
    }
    // Per-variable referenced-label bitmask: 1 = Pos, 2 = Neg, 4 = One (don'f-care).
    let bit = |child: LocalNodeIdx| -> u8 {
        if child == POS {
            1
        } else if child == NEG {
            2
        } else if child == ONE {
            4
        } else {
            0
        }
    };
    let vt = &f.vtree;
    let mut mask: std::collections::HashMap<VarId, u8> = std::collections::HashMap::new();
    // Whole-diagram-is-a-single-literal case: the output sits at the leaf.
    if let VtreeNode::Leaf { var, .. } = *vt.node(f.output.vtree) {
        if !f.levels[f.output.vtree.idx()].is_marginal() {
            *mask.entry(var).or_insert(0) |= bit(f.output.local);
        }
    }
    for vi in 0..vt.num_nodes() {
        let (left, right) = match *vt.node(VtreeIdx(vi as u32)) {
            VtreeNode::Internal { left, right, .. } => (left.idx(), right.idx()),
            VtreeNode::Leaf { .. } => continue,
        };
        let level = &f.levels[vi];
        let left_marg = f.levels[left].is_marginal();
        let right_marg = f.levels[right].is_marginal();
        let left_var = match *vt.node(VtreeIdx(left as u32)) {
            VtreeNode::Leaf { var, .. } if !left_marg => Some(var),
            _ => None,
        };
        let right_var = match *vt.node(VtreeIdx(right as u32)) {
            VtreeNode::Leaf { var, .. } if !right_marg => Some(var),
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
            out.insert((var, true));
        } else if m == 2 {
            out.insert((var, false));
        }
    }
    out
}

