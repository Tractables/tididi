//! Hand-encoded marginal diagrams too small to reach by compiling, and the
//! marginalization pass applied to one subtree of a compiled one.

use std::sync::Arc;

use crate::diagram::{ChildPair, NodeIdx, Tdd, TddLevel, TddNodeId};
use crate::vtree::{Vtree, VtreeIdx, VtreeNode};

/// Sum out of `tdd`, a diagram on a home, the variables under `root`, a node
/// of that home.
pub fn marginalize_subtree(tdd: &mut Tdd, root: VtreeIdx) {
    let home = tdd.vtree().home().expect("a diagram on a home");
    let mut under = vec![false; home.num_vars() as usize];
    for (leaf, var) in home.leaf_bottomup() {
        under[var.idx()] = home.in_subtree(leaf, root);
    }
    tdd.marginalize(|v| under.get(v.idx()).copied().unwrap_or(false)).expect("an unarmed engine refuses nothing");
}

// ── Marginal-invariant fixtures ──────────────────────────────────────────────
/// Non-inlinable count: forces the slot path so the inline-discipline check
/// stays out of the way of the other invariant tests.
pub const BIG: u128 = 1u128 << 40;

/// Minimal boundary-marginal diagram: `balanced(2)` vtree, right child
/// marginal with `counts`, root holding one internal node per entry of
/// `node_pair_lists` (pairs as raw `(left, right)` values; slot refs are
/// bare indices under the bare-is-slot polarity).
pub fn toy(counts: Vec<u128>, node_pair_lists: &[&[(u32, u32)]]) -> Tdd {
    let vtree = Arc::new(Vtree::balanced(2));
    let root = vtree.root();
    let right = match vtree.node(root) {
        VtreeNode::Internal { right, .. } => *right,
        _ => panic!("balanced(2) root must be internal"),
    };
    let n = vtree.num_nodes();
    let mut levels: Vec<TddLevel> = (0..n).map(|_| TddLevel::new()).collect();
    levels[right.idx()].set_counts_state(counts, None);
    for pl in node_pair_lists {
        let pairs: Vec<ChildPair> = pl
            .iter()
            .map(|&(l, r)| ChildPair::new(NodeIdx(l), NodeIdx(r)))
            .collect();
        levels[root.idx()].push_internal_node(&pairs);
    }
    let output = TddNodeId { vtree: root, local: NodeIdx(0) };
    Tdd::from_levels_unchecked(vtree, levels, output)
}

/// Weighted analogue of `toy`: the right child is a weight-marginal level
/// whose per-slot `BigRational` values live in the `Weights` attached to the
/// returned [`Tdd`] (not `marginal_counts`). The caller supplies the store; this
/// helper writes the values into it via `set_level` and attaches it. Parent pair
/// refs use the same bare-is-slot polarity as `toy`.
///
/// `balanced(3)`, not `balanced(2)`: its root's right child is an internal
/// node, so the marginal level is an ordinary internal one. A weight-marginal
/// vtree leaf's column is pinned to the label-ordered 3-slot `leaf_val` cache
/// (`marginalize_leaf_weighted`), so a leaf could not model an
/// arbitrary-width, compactable marginal store.
pub fn toy_weighted(
    mut ws: crate::diagram::Weights,
    values: Vec<num_rational::BigRational>,
    node_pair_lists: &[&[(u32, u32)]],
) -> Tdd {
    if ws.algebra().num_vars() < 3 {
        let weights: Vec<_> = (0..3).map(|i| {
            if i < ws.algebra().num_vars() {
                let var = crate::vtree::VarId(i as u32 + 1);
                crate::diagram::LiteralWeights { negative: ws.algebra().neg_weight(var).clone(), positive: ws.algebra().pos_weight(var).clone() }
            } else { crate::diagram::LiteralWeights { negative: super::rat(1, 1), positive: super::rat(1, 1) } }
        }).collect();
        ws = crate::diagram::Weights::new(crate::diagram::RationalWeights::from_literals(&weights), ws.arithmetic());
    }
    let vtree = Arc::new(Vtree::balanced(3).with_weights(ws).unwrap());
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
    for pl in node_pair_lists {
        let pairs: Vec<ChildPair> = pl
            .iter()
            .map(|&(l, r)| ChildPair::new(NodeIdx(l), NodeIdx(r)))
            .collect();
        levels[root.idx()].push_internal_node(&pairs);
    }
    let wvals: Vec<crate::diagram::WeightValue> =
        values.into_iter().map(crate::diagram::WeightValue::exact).collect();
    levels[right.idx()].become_marginal_weighted(wvals);
    let output = TddNodeId { vtree: root, local: NodeIdx(0) };
    Tdd::from_levels_unchecked(vtree, levels, output)
}


/// Vtree shape of the plain-level content-twin fixtures. Custom (not
/// `balanced`) so the marginal-carrying level `m` sits at an internal vtree
/// node, where an integer marginal level has a count store (a marginal leaf
/// keeps an empty one: its bare refs are leaf labels).
/// Shape (left spine root → gp → bp; each 2-leaf subtree on the right):
///   root → (gp, σ);  gp → (bp, s);  bp → (x, a leaf; m, internal);
///   m → (m_l, m_r);  s → (s_l, s_r);  σ → (sig_l, sig_r).
pub fn boundary_internal_marginal_vtree() -> Vtree {
    // 7 vars; node ids reindexed bottom-up by `from_text` (root last),
    // so callers navigate via `children()` exactly as with `balanced`.
    //   x=0(leaf)  m=(1,2)  s=(3,4)  σ=(5,6);  bp=(x,m) gp=(bp,s) root=(gp,σ)
    Vtree::from_text(
        "vtree 13\n\
         L 0 1\nL 1 2\nL 2 3\nL 3 4\nL 4 5\nL 5 6\nL 6 7\n\
         I 7 1 2\nI 8 0 7\nI 9 3 4\nI 10 8 9\nI 11 5 6\nI 12 10 11\n",
    )
    .expect("boundary_internal_marginal_vtree parse")
}


/// Replace the values of `tdd`'s weight-marginal level `t` with `values`,
/// its width with theirs.
pub fn set_weight_column(tdd: &mut Tdd, t: VtreeIdx, values: Vec<crate::diagram::WeightValue>) {
    let width = values.len() as u32;
    *tdd.levels[t.idx()].weight_values_mut().expect("a weight-marginal level") = values;
    tdd.levels[t.idx()].set_weight_width(width);
}
