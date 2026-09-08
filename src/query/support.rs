//! Read-only structural queries: variable support and implied literals.
//!
//! - `support_mask` / `support_bits` — which variables a TDD depends on.
//! - `implied_literals` — literals forced true in every model.


use crate::diagram::{LocalNodeIdx, Tdd};
#[cfg(test)]
use std::sync::Arc;
#[cfg(test)]
use crate::reduce::minimize;
#[cfg(test)]
use crate::marg_slots::ChildSide;
use crate::apply::project::{POS, NEG, ONE};
use crate::vtree::{VarId, VtreeIdx, VtreeNode};

/// Structural support of `t`: `out[x]` is true iff `t` depends on variable `x`
/// (some live pair references x's leaf with a Pos/Neg label, not just One).
/// Minimizes a clone first so every scanned node is reachable. O(size).
///
/// Test-only: the exact `Vec<bool>` support oracle, kept as ground truth for the
/// `support_bits` over-approximation invariant tests (its former production
/// callers were removed).
#[cfg(test)]
pub(crate) fn support_mask(t: &Tdd) -> Vec<bool> {
    let nvars = t.vtree.num_vars() as usize;
    let mut sup = vec![false; nvars];
    if t.is_zero() {
        return sup;
    }
    let mut mt = t.clone();
    minimize(&mut mt);
    if mt.is_zero() {
        return sup;
    }
    let vtree = Arc::clone(&mt.vtree);
    for (x, sup_x) in sup.iter_mut().enumerate() {
        let leaf = vtree.leaf_of(VarId(x as u32));
        // Output sits at the leaf itself: depends on x iff the label is Pos/Neg.
        if mt.output.vtree == leaf {
            *sup_x = mt.output.local == POS || mt.output.local == NEG;
            continue;
        }
        // The leaf has exactly one parent in the (tree) vtree; find the side and
        // scan that parent level for any Pos/Neg reference to it.
        for vi in 0..vtree.num_nodes() {
            let (left, right) = match *vtree.node(VtreeIdx(vi as u32)) {
                VtreeNode::Internal { left, right, .. } => (left, right),
                VtreeNode::Leaf { .. } => continue,
            };
            let side = if left == leaf {
                ChildSide::Left
            } else if right == leaf {
                ChildSide::Right
            } else {
                continue;
            };
            let level = &mt.levels[vi];
            'scan: for ni in 0..level.nodes.len() {
                if level.nodes[ni].is_leaf() {
                    continue;
                }
                for p in level.pairs_of(&level.nodes[ni]) {
                    let child = match side {
                        ChildSide::Left => p.left,
                        ChildSide::Right => p.right,
                    };
                    if child == POS || child == NEG {
                        *sup_x = true;
                        break 'scan;
                    }
                }
            }
            break; // unique parent found; done with this var
        }
    }
    sup
}

/// Fast OVER-APPROXIMATE structural support of `t`, packed into a `u64` bitmask
/// (bit `x` set ⇒ `t` MAY depend on variable `x`). Unlike [`support_mask`] this does
/// NOT clone or `minimize` first: it walks the diagram as-is in a single O(size)
/// pass, so a dead node can set a bit for a variable `t` no longer truly depends on.
/// That one-sided error is precisely what a disjoint-support pre-skip needs — if two
/// over-approximate supports are disjoint then the TRUE supports (subsets) are too,
/// so skipping the operation is sound; a spurious overlap merely runs the op that
/// would have run anyway. (`support_mask` is the EXACT `Vec<bool>` variant that
/// minimizes first — pick by whether you need exactness or raw speed.)
///
/// Test-only: the sole non-test caller was the retired segment-compile lane; the
/// projection unit tests keep it as a fast support oracle to check `support_mask`.
#[cfg(test)]
pub(crate) fn support_bits(t: &Tdd) -> Vec<u64> {
    let vtree = &t.vtree;
    let nvars = vtree.num_vars() as usize;
    let mut bits = vec![0u64; nvars.div_ceil(64)];
    if t.is_zero() {
        return bits;
    }
    // Output sitting directly at a leaf: set that leaf's var iff labelled Pos/Neg.
    if let VtreeNode::Leaf { var, .. } = *vtree.node(t.output.vtree) {
        if t.output.local == POS || t.output.local == NEG {
            let x = var.idx();
            bits[x / 64] |= 1u64 << (x % 64);
        }
        return bits;
    }
    // Each variable is a vtree leaf whose single parent is an internal level; a Pos/Neg
    // reference to that leaf (on its side) in any of the level's pairs is a dependency.
    // Iterating levels once and reading each internal node's two leaf-children is O(size),
    // versus `support_mask`'s per-variable parent-find sweep.
    for vi in 0..vtree.num_nodes() {
        let (left, right) = match *vtree.node(VtreeIdx(vi as u32)) {
            VtreeNode::Internal { left, right, .. } => (left, right),
            VtreeNode::Leaf { .. } => continue,
        };
        let lvar = match *vtree.node(left) {
            VtreeNode::Leaf { var, .. } => Some(var.idx()),
            _ => None,
        };
        let rvar = match *vtree.node(right) {
            VtreeNode::Leaf { var, .. } => Some(var.idx()),
            _ => None,
        };
        if lvar.is_none() && rvar.is_none() {
            continue;
        }
        let level = &t.levels[vi];
        let mut need_l = lvar.is_some();
        let mut need_r = rvar.is_some();
        'scan: for ni in 0..level.nodes.len() {
            if level.nodes[ni].is_leaf() {
                continue;
            }
            for p in level.pairs_of(&level.nodes[ni]) {
                if need_l && (p.left == POS || p.left == NEG) {
                    let x = lvar.unwrap();
                    bits[x / 64] |= 1u64 << (x % 64);
                    need_l = false;
                }
                if need_r && (p.right == POS || p.right == NEG) {
                    let x = rvar.unwrap();
                    bits[x / 64] |= 1u64 << (x % 64);
                    need_r = false;
                }
                if !need_l && !need_r {
                    break 'scan;
                }
            }
        }
    }
    bits
}

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
pub fn implied_literals(t: &Tdd) -> std::collections::HashSet<(VarId, bool)> {
    let mut out = std::collections::HashSet::new();
    if t.is_zero() {
        return out;
    }
    // Per-variable referenced-label bitmask: 1 = Pos, 2 = Neg, 4 = One (don't-care).
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
    let vt = &t.vtree;
    let mut mask: std::collections::HashMap<VarId, u8> = std::collections::HashMap::new();
    // Whole-diagram-is-a-single-literal case: the output sits at the leaf.
    if let VtreeNode::Leaf { var, .. } = *vt.node(t.output.vtree) {
        if !t.levels[t.output.vtree.idx()].is_marginal() {
            *mask.entry(var).or_insert(0) |= bit(t.output.local);
        }
    }
    for vi in 0..vt.num_nodes() {
        let (left, right) = match *vt.node(VtreeIdx(vi as u32)) {
            VtreeNode::Internal { left, right, .. } => (left.idx(), right.idx()),
            VtreeNode::Leaf { .. } => continue,
        };
        let level = &t.levels[vi];
        let left_marg = t.levels[left].is_marginal();
        let right_marg = t.levels[right].is_marginal();
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

/// Total reachable input-pair count of a (preferably minimized) TDD — the honest
/// "size" for the never-larger gate (`Tdd::size` counts dead arena pairs too).
#[cfg(test)]
pub(crate) fn reachable_pairs(t: &Tdd) -> usize {
    if t.is_zero() {
        return 0;
    }
    let reach = t.reachable_nodes();
    let mut n = 0;
    for vi in 0..t.vtree.num_nodes() {
        if t.vtree.node(VtreeIdx(vi as u32)).is_leaf() {
            continue;
        }
        let level = &t.levels[vi];
        for i in 0..level.nodes.len() {
            if reach[vi][i] && level.nodes[i].is_internal() {
                n += level.pair_count_at(i);
            }
        }
    }
    n
}
