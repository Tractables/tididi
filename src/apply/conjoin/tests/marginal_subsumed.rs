//! A marginalizing conjunction leaves no value store under a marginal parent.
//!
//! Sibling of `marginal_level.rs`.

use std::sync::Arc;

use crate::test_helpers::check::marginal::check_subsumed_stores_empty;
use crate::vtree::{Vtree, VtreeIdx};
use crate::{Engine, Tdd};

#[test]
fn a_marginalizing_conjunction_frees_the_stores_it_subsumes() {
    let vtree = Arc::new(Vtree::balanced(6));
    let f = Tdd::clause(&vtree, [1, 2, -3]) & Tdd::clause(&vtree, [-2, 4]);
    let g = Tdd::clause(&vtree, [3, 5, -6])
        & Tdd::clause(&vtree, [-1, -4, 5])
        & Tdd::clause(&vtree, [6, 2]);
    let expected = (f.clone() & g.clone()).model_count();

    // Every level a target, so each internal level is summed out over
    // children that were summed out moments earlier in the same conjunction.
    let targets: Vec<VtreeIdx> = vtree.bottomup().collect();
    let h = Engine::new()
        .and_marginalizing(f, g, &targets)
        .expect("an unarmed engine refuses nothing");

    assert_eq!(h.model_count(), expected);
    check_subsumed_stores_empty(&h).unwrap_or_else(|e| panic!("{e}"));
}
