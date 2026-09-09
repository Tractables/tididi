//! Regression tests for `is_self_conjunction` (A4). The structural shortcut
//! `f ∧ g = f.clone()` must fire ONLY when the operands are the same
//! function. Before A4 it compared only per-level `nodes`/`pairs`, so it
//! (a) treated two operands as equal when they agreed on every EXPLICIT level
//! but differed in marginal content (a marginal level clears `nodes`/`pairs`),
//! silently dropping one side's counts, and (b) ignored the `multi_pairs` table.
use super::is_self_conjunction;
use crate::diagram::{
    MultiPairRange, InputPair, LeafLabel, NodeIdx, Tdd, TddNodeId,
    assert_can_make_marginal, take_levels,
};
use crate::vtree::{Vtree, VtreeIdx};
use std::sync::Arc;

/// Build a small 4-leaf TDD (two width-2 internal children under the root).
/// Both operands built this way are byte-identical.
fn build_operand(vtree: &Arc<Vtree>) -> Tdd {
    let eng = &crate::engine::Engine::new();
    let root = VtreeIdx((vtree.num_nodes() - 1) as u32);
    let (v_left, v_right) = vtree.children(root);
    let one = NodeIdx(LeafLabel::One as u32);
    let pos = NodeIdx(LeafLabel::Pos as u32);
    let neg = NodeIdx(LeafLabel::Neg as u32);
    let mut levels = take_levels(eng, vtree.num_nodes());
    let a0 = levels[v_left.idx()].push_internal_node(&[InputPair { left: pos, right: one }]);
    let a1 = levels[v_left.idx()].push_internal_node(&[InputPair { left: neg, right: one }]);
    let r0 = levels[v_right.idx()].push_internal_node(&[InputPair { left: pos, right: one }]);
    let r1 = levels[v_right.idx()].push_internal_node(&[InputPair { left: one, right: pos }]);
    let root_l = levels[root.idx()].push_internal_node(&[
        InputPair { left: a0, right: r0 },
        InputPair { left: a1, right: r1 },
    ]);
    Tdd::from_levels_unchecked(vtree.clone(), levels, TddNodeId { vtree: root, local: root_l })
}

#[test]
fn identical_nonmarginal_takes_shortcut() {
    let vtree = Arc::new(Vtree::balanced(4));
    let a = build_operand(&vtree);
    let b = build_operand(&vtree);
    assert!(
        is_self_conjunction(&a, &b),
        "byte-identical non-marginal operands must still take the self-conjunction shortcut"
    );
}

#[test]
fn marginal_level_blocks_shortcut() {
    let vtree = Arc::new(Vtree::balanced(4));
    let root = VtreeIdx((vtree.num_nodes() - 1) as u32);
    let (v_left, _) = vtree.children(root);
    let mut a = build_operand(&vtree);
    assert_can_make_marginal(&a.levels, &vtree, v_left);
    a.levels[v_left.idx()].become_marginal(vec![2u128, 2u128], None);
    assert!(a.levels[v_left.idx()].is_marginal());
    let b = a.clone();
    // Byte-identical operands, but the marginal level cleared its nodes/pairs.
    // Pre-A4 the structural test compared only nodes/pairs → equal → `true`,
    // letting the shortcut drop a real operand's marginal store.
    assert!(
        !is_self_conjunction(&a, &b),
        "operands carrying a marginal level must NOT take the structural shortcut (A4)"
    );
}

#[test]
fn differing_ext_blocks_shortcut() {
    let vtree = Arc::new(Vtree::balanced(4));
    let root = VtreeIdx((vtree.num_nodes() - 1) as u32);
    let a = build_operand(&vtree);
    let mut b = build_operand(&vtree);
    // Equal nodes+pairs but a different `multi_pairs` arrangement is a different
    // function; pre-A4 the test ignored `multi_pairs` and returned `true`.
    b.levels[root.idx()].multi_pairs.push(MultiPairRange { start: 0, len: 2 });
    assert!(
        !is_self_conjunction(&a, &b),
        "operands whose `multi_pairs` tables differ must NOT be treated as self-conjunction (A4)"
    );
}
