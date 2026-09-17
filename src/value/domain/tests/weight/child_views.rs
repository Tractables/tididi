use super::super::*;
use crate::diagram::{Arithmetic, LeafLabel, RationalWeights};
use crate::vtree::VarId;

#[test]
fn marginal_child_view_borrows_the_stored_column() {
    let tree = Vtree::balanced(2);
    let root = tree.root().idx();
    for arithmetic in [Arithmetic::ExactRational, Arithmetic::SignedLog] {
        let mut store = WeightStore::new(RationalWeights::unit(2), arithmetic);
        let values = vec![store.leaf_val(VarId(1), LeafLabel::One); 64];
        store.set_level(root, values);
        let mut level = TddLevel::new();
        level.become_marginal_weighted(64);
        let computed = vec![None; tree.num_nodes()];
        let view = WeightFold::child_view(root, &tree, &level, &computed, &store);
        assert!(view.is_marginal);
        assert!(std::ptr::eq(view.col.as_ref(), store.level(root).unwrap()));
    }
}

#[test]
fn structural_child_view_borrows_scratch_even_when_the_store_has_a_column() {
    let tree = Vtree::balanced(2);
    let root = tree.root().idx();
    for arithmetic in [Arithmetic::ExactRational, Arithmetic::SignedLog] {
        let mut store = WeightStore::new(RationalWeights::unit(2), arithmetic);
        store.set_level(root, vec![store.wzero()]);
        let level = TddLevel::new();
        let mut computed = vec![None; tree.num_nodes()];
        computed[root] = Some(vec![store.leaf_val(VarId(1), LeafLabel::One)]);
        let view = WeightFold::child_view(root, &tree, &level, &computed, &store);
        assert!(!view.is_marginal);
        assert!(std::ptr::eq(view.col.as_ref(), computed[root].as_ref().unwrap().as_slice()));
    }
}
