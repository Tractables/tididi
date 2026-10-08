//! `Vtree::expand_leaves`: each leaf becomes a balanced subtree over its class.

use std::sync::Arc;

use super::*;
use crate::Context;

#[test]
fn each_class_stays_together_below_the_node_that_held_its_leaf() {
    let reduced = Vtree::balanced(2);
    let classes = [vec![VarId(1), VarId(2)], vec![VarId(6)]];
    let expanded = reduced.expand_leaves(|v| classes[v.idx()].clone(), 6).unwrap();
    let mut vars: Vec<u32> = expanded.leaf_bottomup().map(|(_, var)| var.0).collect();
    vars.sort_unstable();
    assert_eq!(vars, [1, 2, 6]);
    assert_eq!(expanded.num_vars(), 6, "variables 3 to 5 are uncovered ids of the requested space");

    let (one, two) = (expanded.leaf_of(VarId(1)).unwrap(), expanded.leaf_of(VarId(2)).unwrap());
    let class_root = expanded.lca(one, two);
    let (left, right) = expanded.children(class_root);
    assert_eq!([left, right], [one, two]);
    assert_eq!(expanded.node(class_root).parent(), Some(expanded.root()));
}

#[test]
fn singleton_classes_rename_the_leaves_and_keep_the_shape() {
    let reduced = Vtree::linear(3);
    let expanded = reduced.expand_leaves(|v| [VarId(v.0 + 3)], 6).unwrap();
    let renamed = Vtree::linear_from_order(&[VarId(4), VarId(5), VarId(6)]).unwrap();
    assert_eq!(expanded.to_text(), renamed.to_text());
}

#[test]
fn a_class_is_arranged_as_a_balanced_tree_over_its_order() {
    let order = [VarId(3), VarId(1), VarId(4), VarId(2)];
    let expanded = Vtree::leaf(VarId(1)).expand_leaves(|_| order, 4).unwrap();
    assert_eq!(expanded.to_text(), Vtree::balanced_over(&order).unwrap().to_text());
}

#[test]
fn the_result_keeps_the_execution_context() {
    let context = Arc::new(Context::new());
    let reduced = context.bind(Vtree::balanced(3));
    let expanded = reduced.expand_leaves(|v| [v], 3).unwrap();
    assert!(Arc::ptr_eq(expanded.context(), &context));
}

#[test]
fn empty_overlapping_and_out_of_range_classes_are_refused() {
    let reduced = Vtree::balanced(2);
    let empty = reduced.expand_leaves(|v| if v == VarId(2) { vec![] } else { vec![v] }, 2);
    assert!(matches!(empty, Err(VtreeError::Invalid(_))), "{empty:?}");
    let overlapping = reduced.expand_leaves(|_| [VarId(1)], 2);
    assert_eq!(overlapping.unwrap_err(), VtreeError::OverlappingVariable(VarId(1)));
    let out_of_range = reduced.expand_leaves(|v| [VarId(v.0 + 1)], 2);
    assert!(matches!(out_of_range, Err(VtreeError::Invalid(_))), "{out_of_range:?}");
    let zero = reduced.expand_leaves(|v| [VarId(v.0 - 1)], 2);
    assert!(matches!(zero, Err(VtreeError::Invalid(_))), "{zero:?}");
}
