//! Hand-encoded marginal diagrams too small to reach by compiling, and the
//! marginalize pass applied to one subtree of a compiled one.

use std::sync::Arc;

use crate::diagram::{
    assert_can_make_marginal, InputPair, NodeIdx, Tdd, TddLevel, TddNodeId,
};
use crate::query::node_counts;
use crate::vtree::{Vtree, VtreeIdx, VtreeNode};

use super::oracle::big_to_u128;

/// Bottom-up marginalize every internal, non-marginal, width≥1 level in
/// the subtree rooted at `root` (inclusive). Counts are derived from the
/// current diagram shape via `node_counts`. Mirrors production's
/// The marginalize pass's batch + cascade semantics for a single
/// subtree, without the streaming-marginal hooks.
pub fn marginalize_subtree(tdd: &mut Tdd, root: VtreeIdx) {
    let vtree = tdd.vtree.clone();
    let counts = node_counts(tdd);
    for &t in vtree.bottomup_slice() {
        let ti = t.idx();
        let mut under = t == root;
        if !under {
            let mut cur = t;
            while let Some(p) = vtree.node(cur).parent() {
                if p == root {
                    under = true;
                    break;
                }
                cur = p;
            }
        }
        if !under {
            continue;
        }
        if matches!(*vtree.node(VtreeIdx(ti as u32)), VtreeNode::Leaf { .. }) || tdd.levels[ti].is_marginal() {
            continue;
        }
        let w = tdd.levels[ti].width();
        if w == 0 {
            continue;
        }
        let u128_counts: Vec<u128> = (0..w).map(|i| big_to_u128(&counts[ti][i])).collect();
        assert_can_make_marginal(&tdd.levels, &vtree, t);
        tdd.levels[ti].become_marginal(u128_counts, None);
    }
    // Emulate production marginalization, which tags every persisted marginal-side
    // slot ref (bit 30) so the 0=inline decode invariant holds. Without this the
    // strict decode assert fires when a later reader hits a raw slot ref.
    crate::diagram::tag_all_marginal_side_slots(tdd, None);
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
        let pairs: Vec<InputPair> = pl
            .iter()
            .map(|&(l, r)| InputPair { left: NodeIdx(l), right: NodeIdx(r) })
            .collect();
        levels[root.idx()].push_internal_node(&pairs);
    }
    let output = TddNodeId { vtree: root, local: NodeIdx(0) };
    Tdd::from_levels_unchecked(vtree, levels, output)
}

/// Weighted analogue of [`toy`]: the right child is a WEIGHT-marginal level
/// whose per-slot `BigRational` values live in the `WeightStore` attached to the
/// returned [`Tdd`] (not `marginal_counts`). The caller supplies the store; this
/// helper writes the values into it via `set_level` and attaches it. Parent pair
/// refs use the same bare-is-slot polarity as `toy`.
///
/// `balanced(3)`, not `balanced(2)` (which the integer [`toy`] still uses): its
/// root's right child is an INTERNAL node, so the marginal level here is an
/// ordinary internal one. A weight-marginal vtree LEAF is a different animal —
/// its `WeightStore` column is PINNED to the label-ordered 3-slot `leaf_val`
/// cache (`marginalize_leaf_weighted`), which no pass may compact, erase or
/// append to — so a leaf could not model an arbitrary-width, compactable
/// marginal store at all.
pub fn toy_weighted(
    mut ws: crate::diagram::WeightStore,
    values: Vec<num_rational::BigRational>,
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
    levels[right.idx()].become_marginal_weighted(values.len() as u32);
    for pl in node_pair_lists {
        let pairs: Vec<InputPair> = pl
            .iter()
            .map(|&(l, r)| InputPair { left: NodeIdx(l), right: NodeIdx(r) })
            .collect();
        levels[root.idx()].push_internal_node(&pairs);
    }
    let wvals: Vec<crate::diagram::WeightVal> =
        values.into_iter().map(crate::diagram::WeightVal::exact).collect();
    ws.set_level(right.idx(), wvals);
    let output = TddNodeId { vtree: root, local: NodeIdx(0) };
    let mut tdd = Tdd::from_levels_unchecked(vtree, levels, output);
    tdd.set_weights(ws);
    tdd
}


/// Vtree shape shared by the fork-down fixtures below. Custom (not `balanced`)
/// so the marginal-carrying level `m` sits at an INTERNAL vtree node: an integer
/// marginal LEAF keeps an EMPTY store (bare refs are leaf-LABELS, decoded by
/// `read_marginal_count`), so it is not a legal fork-down scale target and the
/// mint that these tests exercise would be unsound there (see
/// `duplicate_pair_resolve.rs` `scale_leaf_marginal_label`). An internal marginal level exercises
/// the multiplicity-fork-down mechanics identically, with a real store to mint
/// into. Shape (left spine root → gp → bp; each 2-leaf subtree on the right):
///   root → (gp, σ);  gp → (bp, s);  bp → (x [leaf], m [INTERNAL]);
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

