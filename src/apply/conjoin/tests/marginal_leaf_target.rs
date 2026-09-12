//! A leaf named as a marginalizing-conjunction target is summed out.
//!
//! Sibling of `marginal_level.rs`.

use std::sync::Arc;

use crate::marginal::marginalize_levels;
use crate::test_helpers::assert_same_shape;
use crate::reduce::minimize;
use crate::vtree::{VarId, Vtree};
use crate::{Engine, Tdd};

#[test]
fn a_leaf_target_is_summed_out_after_the_product() {
    let vtree = Arc::new(Vtree::balanced(4));
    let f = Tdd::clause(&vtree, [1, -2]) & Tdd::clause(&vtree, [2, 3]);
    let g = Tdd::clause(&vtree, [3, -4]) & Tdd::clause(&vtree, [-1, 4]);
    let leaf = vtree.leaf_of(VarId(2)).expect("the vtree carries x3");
    let expected = (f.clone() & g.clone()).model_count();

    let eng = Engine::new();
    let h = eng.and_marginalizing(f.clone(), g.clone(), &[leaf]).expect("no limit is armed");
    assert!(h.levels[leaf.idx()].is_marginal());
    assert_eq!(h.model_count(), expected);

    // The same diagram the pass gives when it runs after the product.
    let mut by_pass = f & g;
    marginalize_levels(&eng, &mut by_pass, &[leaf]).expect("no limit is armed");
    let (mut h, mut by_pass) = (h, by_pass);
    minimize(&mut h);
    minimize(&mut by_pass);
    assert_same_shape(&h, &by_pass, "leaf target in the product vs the pass");
}
