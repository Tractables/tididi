use super::*;
use crate::vtree::{Vtree, VtreeIdx};

/// An explicit recursive specification for a small tree, recording compute and release events.
fn reference(tree: &Vtree, root: VtreeIdx, held: usize, keep: Option<VtreeIdx>, events: &mut Vec<(bool, VtreeIdx)>) {
    if held & (1 << root.idx()) != 0 { return; }
    if !tree.node(root).is_leaf() {
        let (left, right) = tree.children(root);
        reference(tree, left, held, keep, events);
        reference(tree, right, held, keep, events);
    }
    events.push((true, root));
    if let Some(keep) = keep
        && !tree.node(root).is_leaf()
    {
        let (left, right) = tree.children(root);
        for child in [left, right] {
            if child != keep { events.push((false, child)); }
        }
    }
}

#[test]
fn parent_link_walk_preserves_subtree_order_caches_and_releases() {
    for tree in [Vtree::balanced(4), Vtree::linear(4)] {
        for root in tree.bottomup() {
            for held in 0..1 << tree.num_nodes() {
                for keep in [None, Some(root), Some(tree.leaf_of(crate::vtree::VarId(1)).unwrap())] {
                    let mut expected = Vec::new();
                    reference(&tree, root, held, keep, &mut expected);
                    let events = std::cell::RefCell::new(Vec::new());
                    walk_bottom_up(&tree, root, &mut [] as &mut [()], |_, i| held & (1 << i) != 0,
                        |_, t| { events.borrow_mut().push((true, t)); Ok::<_, ()>(()) },
                        |_, i| events.borrow_mut().push((false, VtreeIdx(i as u32))), keep).unwrap();
                    assert_eq!(events.into_inner(), expected, "root {root:?}, held {held}, keep {keep:?}");
                }
            }
        }
    }
}

#[test]
fn parent_link_walk_handles_deep_trees_and_stops_at_the_first_error() {
    let tree = Vtree::linear(20_000);
    let mut count = 0;
    walk_bottom_up(&tree, tree.root(), &mut [] as &mut [()], |_, _| false,
        |_, _| { count += 1; Ok::<_, ()>(()) }, |_, _| {}, None).unwrap();
    assert_eq!(count, tree.num_nodes());
    let mut count = 0;
    let result = walk_bottom_up(&tree, tree.root(), &mut [] as &mut [()], |_, _| false,
        |_, _| { count += 1; if count == 17 { Err(()) } else { Ok(()) } }, |_, _| {}, None);
    assert!(result.is_err());
    assert_eq!(count, 17);
}
