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
///
/// ```
/// use std::sync::Arc;
/// use tididi::{Tdd, Vtree};
/// use tididi::query::implied_literals;
/// use tididi::reduce::minimize;
///
/// let tree = Arc::new(Vtree::balanced(3));
/// let mut f = Tdd::clause(&tree, [1, 2]) & Tdd::clause(&tree, [1, -2]);
/// minimize(&mut f);
/// assert_eq!(implied_literals(&f), vec![1.into()]); // x1 is forced; x2 and x3 are free
/// # tididi::test_helpers::assert_canonical(&f);
/// ```
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
    let mut mask: FxHashMap<VarId, u8> = FxHashMap::default();
    for (var, label) in leaf_references(f) {
        *mask.entry(var).or_insert(0) |= bit(label);
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


/// Referenced leaf labels, shared by backbone and semantic-support queries.
pub(super) fn leaf_references(f: &Tdd) -> impl Iterator<Item = (VarId, EncodedChildRef)> + '_ {
    let output = match *f.vtree.node(f.output.vtree) {
        VtreeNode::Leaf { var, .. } if !f.levels[f.output.vtree.idx()].is_marginal() => Some((var, f.output.local.into())),
        _ => None,
    };
    output.into_iter().chain(f.vtree.internal_bottomup().flat_map(move |(t, left, right)| {
        let leaf_var = |child: VtreeIdx| match *f.vtree.node(child) {
            VtreeNode::Leaf { var, .. } if !f.levels[child.idx()].is_marginal() => Some(var),
            _ => None,
        };
        let (a, b) = (leaf_var(left), leaf_var(right));
        let level = &f.levels[t.idx()];
        let count = if a.is_some() || b.is_some() { level.nodes.len() } else { 0 };
        level.nodes[..count].iter().filter(|node| !node.is_leaf()).flat_map(move |node| {
            level.pairs_of(node).iter().flat_map(move |pair| {
                [(a, pair.left), (b, pair.right)].into_iter().filter_map(|(var, label)| var.map(|var| (var, label)))
            })
        })
    }))
}
