//! A leaf named as a marginalizing-conjunction target is summed out.
//!
//! Sibling of `marginal_level.rs`.

use std::sync::Arc;


use crate::test_helpers::assert_same_shape;

use crate::vtree::{VarId, Vtree};
use crate::{Engine, Tdd};

// Padded form: the padded sum-out.
#[test]
fn a_leaf_target_is_summed_out_after_the_product() {
    let vtree = Arc::new(Vtree::balanced(4));
    let f = Tdd::clause(&crate::vtree::Home::unchecked(&vtree), [1, -2]).unwrap() & Tdd::clause(&crate::vtree::Home::unchecked(&vtree), [2, 3]).unwrap();
    let g = Tdd::clause(&crate::vtree::Home::unchecked(&vtree), [3, -4]).unwrap() & Tdd::clause(&crate::vtree::Home::unchecked(&vtree), [-1, 4]).unwrap();
    let leaf = vtree.leaf_of(VarId(3)).expect("the vtree carries x3");
    let expected = (f.clone() & g.clone()).model_count().unwrap();

    let eng = Engine::new();
    let h = crate::test_helpers::and_marginalizing_under(&eng, f.clone(), g.clone(), &[leaf]).expect("no limit is armed");
    assert!(h.levels[leaf.idx()].is_marginal());
    assert_eq!(h.model_count().unwrap(), expected);

    // The same diagram the pass gives when it runs after the product.
    let mut by_pass = f & g;
    crate::test_helpers::marginalize_under(&eng, &mut by_pass, &[leaf]).expect("no limit is armed");
    let (mut h, mut by_pass) = (h, by_pass);
    h.minimize().unwrap();
    by_pass.minimize().unwrap();
    assert_same_shape(&h, &by_pass, "leaf target in the product vs the pass");
}
