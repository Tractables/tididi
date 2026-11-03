//! A content-twin merge whose implicit parent cannot be moved returns
//! `OverBudget` and leaves the diagram as it was.

use crate::Engine;
use crate::diagram::{ChildPair, ChildSide, NodeIdx, Tdd, ValueRef};
use crate::limits::OperationError;
use crate::reduce::contract::content_twin::merge_content_equal_nodes;
use crate::test_helpers::{sorted_pairs, x_decision_diagram};
use crate::vtree::VtreeIdx;

/// A node capacity whose byte size overflows `isize::MAX`, so storing the
/// level is refused by `try_reserve_exact` without touching the allocator.
const OVER_BUDGET_NODES: usize = usize::MAX / 8;

/// The merges at 8 nodes: `w`'s `4 * 8` content-equal nodes onto one, then
/// `v`'s 8 nodes, which that move makes content-equal, onto one.
const MERGED: usize = (4 * 8 - 1) + (8 - 1);

/// The x-decision diagram with its root's left leaf summed out, so that the
/// content-twin pass runs: the root's left sides name `Inline(1)`. Returns
/// the diagram and its implicit level `v`.
fn marginal_x_decision_diagram(n: usize) -> (Tdd, VtreeIdx) {
    let (mut tdd, v, _) = x_decision_diagram(n);
    let r = tdd.vtree.root();
    let (x1, _) = tdd.vtree.children(r);
    let one = NodeIdx(ValueRef::inline_raw(1).expect("1 fits inline"));
    let pairs: Vec<ChildPair> = (0..n).map(|i| ChildPair::new(one, NodeIdx(i as u32))).collect();
    let root = &mut tdd.levels[r.idx()];
    root.clear();
    root.push_internal_node(&pairs);
    root.set_has_value_refs(ChildSide::Left, true);
    tdd.levels[x1.idx()].become_marginal(Vec::new(), None);
    assert!(tdd.levels[v.idx()].implicit().is_some(), "the fixture's level is implicit");
    (tdd, v)
}

/// `w`'s `4n` nodes are content-equal, so the merge moves `v`'s right sides
/// onto one node. That move is not one to one and stores `v`; refusing the
/// stored level's room returns `OverBudget` with `v` still implicit and every
/// level as it was, and the same merge with the room granted keeps the count.
#[test]
fn a_refused_move_of_an_implicit_parent_is_an_error() {
    let eng = Engine::new();
    let (original, v) = marginal_x_decision_diagram(8);
    let count = original.model_count().unwrap();

    let mut tdd = original.clone();
    let capacity = tdd.levels[v.idx()].pairs.node_capacity();
    tdd.levels[v.idx()].pairs.set_node_capacity(OVER_BUDGET_NODES);
    assert_eq!(merge_content_equal_nodes(&eng, &mut tdd, None), Err(OperationError::OverBudget));
    assert_eq!(tdd.levels[v.idx()].implicit(), original.levels[v.idx()].implicit());
    assert_eq!(sorted_pairs(&tdd), sorted_pairs(&original));

    tdd.levels[v.idx()].pairs.set_node_capacity(capacity);
    assert_eq!(merge_content_equal_nodes(&eng, &mut tdd, None), Ok(MERGED));
    assert!(tdd.levels[v.idx()].implicit().is_none(), "the merging move stores v");
    assert_eq!(tdd.model_count().unwrap(), count);
}

/// Every reservation the merge makes, refused in turn, is an error that
/// leaves a valid diagram of the same count: `v` as it was until its move
/// is had, stored after it.
#[test]
fn every_refused_reservation_of_a_content_twin_merge_leaves_a_valid_diagram() {
    let (original, v) = marginal_x_decision_diagram(8);
    let count = original.model_count().unwrap();
    let mut refused = 0;
    loop {
        let eng = Engine::new();
        eng.limits().refuse_nth_reserve(refused);
        let mut tdd = original.clone();
        match merge_content_equal_nodes(&eng, &mut tdd, None) {
            Err(e) => {
                assert_eq!(e, OperationError::OverBudget);
                let level = tdd.levels[v.idx()].implicit();
                assert!(level.is_none() || level == original.levels[v.idx()].implicit(), "reserve {refused}");
                if level.is_some() {
                    assert_eq!(sorted_pairs(&tdd), sorted_pairs(&original), "reserve {refused}");
                }
                crate::test_helpers::check::validate_vtree_structure(&tdd).unwrap();
                assert_eq!(tdd.model_count().unwrap(), count, "reserve {refused}");
                refused += 1;
            }
            Ok(merged) => {
                assert_eq!(merged, MERGED);
                assert_eq!(tdd.model_count().unwrap(), count);
                break;
            }
        }
    }
    assert!(refused >= 2, "{refused} reservations");
}
