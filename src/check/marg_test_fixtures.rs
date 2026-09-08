use super::*;
use crate::diagram::{LocalNodeIdx, TddNodeId};
use crate::vtree::{Vtree, VtreeNode};
use std::sync::Arc;

/// Non-inlinable count: forces the slot path so the inline-discipline check
/// stays out of the way of the other invariant tests.
pub(crate) const BIG: u128 = 1u128 << 40;

/// Minimal boundary-marginal TDD: `balanced(2)` vtree, right child
/// marginal with `counts`, root holding one internal node per entry of
/// `node_pair_lists` (pairs as raw `(left, right)` values; slot refs are
/// bare indices under the bare-is-slot polarity).
pub(crate) fn toy(counts: Vec<u128>, node_pair_lists: &[&[(u32, u32)]]) -> Tdd {
    let vtree = Arc::new(Vtree::balanced(2));
    let root = vtree.root();
    let right = match vtree.node(root) {
        VtreeNode::Internal { right, .. } => *right,
        _ => panic!("balanced(2) root must be internal"),
    };
    let n = vtree.num_nodes();
    let mut levels: Vec<TddLevel> = (0..n).map(|_| TddLevel::new()).collect();
    levels[right.idx()].marginal_counts = Some(counts);
    for pl in node_pair_lists {
        let pairs: Vec<InputPair> = pl
            .iter()
            .map(|&(l, r)| InputPair { left: LocalNodeIdx(l), right: LocalNodeIdx(r) })
            .collect();
        levels[root.idx()].push_internal_node(&pairs);
    }
    let output = TddNodeId { vtree: root, local: LocalNodeIdx(0) };
    Tdd::with_levels(vtree, levels, output)
}

/// Weighted analogue of [`toy`]: the right child is a WEIGHT-marginal level
/// whose per-slot `BigRational` values live in the `WeightStore` attached to the
/// returned [`Tdd`] (not `marginal_counts`). The caller supplies the store; this
/// helper writes the values into it via `set_level` and attaches it. Parent pair
/// refs use the same bare-is-slot polarity as `toy`.
///
/// `balanced(3)`, NOT `balanced(2)` (which the integer [`toy`] still uses): its
/// root's right child is an INTERNAL node, so the marginal level here is an
/// ordinary internal one. A weight-marginal vtree LEAF is a different animal —
/// its `WeightStore` column is PINNED to the label-ordered 3-slot `leaf_val`
/// cache (`marginalize_leaf_weighted`), which no pass may compact, erase or
/// append to — so a leaf could not model an arbitrary-width, compactable
/// marginal store at all.
pub(crate) fn toy_weighted(
    mut ws: crate::weight_store::WeightStore,
    vals: Vec<num_rational::BigRational>,
    node_pair_lists: &[&[(u32, u32)]],
) -> Tdd {
    let vtree = Arc::new(Vtree::balanced(3));
    let root = vtree.root();
    let right = match vtree.node(root) {
        VtreeNode::Internal { right, .. } => *right,
        _ => panic!("balanced(3) root must be internal"),
    };
    debug_assert!(
        !vtree.node(right).is_leaf(),
        "toy_weighted's marginal side must be INTERNAL — a leaf column is pinned"
    );
    let n = vtree.num_nodes();
    let mut levels: Vec<TddLevel> = (0..n).map(|_| TddLevel::new()).collect();
    levels[right.idx()].make_marginal_weighted_with_slots(vals.len() as u32);
    for pl in node_pair_lists {
        let pairs: Vec<InputPair> = pl
            .iter()
            .map(|&(l, r)| InputPair { left: LocalNodeIdx(l), right: LocalNodeIdx(r) })
            .collect();
        levels[root.idx()].push_internal_node(&pairs);
    }
    let wvals: Vec<crate::query::WeightVal> =
        vals.into_iter().map(crate::query::WeightVal::exact).collect();
    ws.set_level(right.idx(), wvals);
    let output = TddNodeId { vtree: root, local: LocalNodeIdx(0) };
    let mut tdd = Tdd::with_levels(vtree, levels, output);
    tdd.attach_weights(ws);
    tdd
}
