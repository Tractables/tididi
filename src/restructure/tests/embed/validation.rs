use std::sync::Arc;

use crate::restructure::GraftError;
use crate::vtree::{VarId, Vtree, VtreeError};
use crate::{Engine, OperationError, Tdd};

use crate::test_helpers::assert_canonical;

/// `x1 ∨ x2` on a two-leaf stick.
fn pair_relation() -> (Arc<Vtree>, Tdd) {
    let vtree = Arc::new(Vtree::linear(2));
    let mut f = Tdd::clause(&vtree, [1, 2]).unwrap();
    f.minimize().unwrap();
    assert_canonical(&f);
    (vtree, f)
}

#[test]
fn two_variables_may_not_share_an_image() {
    let (_, f) = pair_relation();
    let big = Arc::new(Vtree::linear(4));
    assert_eq!(
        f.embed(&big, |_| VarId(3)).unwrap_err(),
        GraftError::Vtree(VtreeError::OverlappingVariable(VarId(3))),
    );
}

#[test]
fn an_image_variable_outside_the_destination_is_refused() {
    let (_, f) = pair_relation();
    let big = Arc::new(Vtree::linear(4));
    assert_eq!(
        f.embed(&big, |v| VarId(v.0 + 9)).unwrap_err(),
        GraftError::VariableOutOfRange { variable: VarId(10), num_vars: 4 },
    );
    assert_eq!(
        f.embed(&big, |v| VarId(v.0 + 9)).unwrap_err().to_string(),
        "grafted variable 10 is outside the variables 1 to 4",
    );
}

#[test]
fn a_renaming_that_reverses_the_leaf_order_is_refused() {
    let (small, f) = pair_relation();
    let big = Arc::new(Vtree::linear(4));
    // The relation's first variable would land to the right of its second.
    let reversed = |v: VarId| if v == VarId(1) { VarId(3) } else { VarId(1) };
    let Err(GraftError::NotIsomorphic { source }) = f.embed(&big, reversed) else {
        panic!("a reversed renaming has no level-by-level copy");
    };
    assert!(small.node(source).is_leaf());
}

#[test]
fn a_destination_grouping_the_variables_differently_is_refused() {
    let small = Arc::new(Vtree::balanced(4));
    let mut f = Tdd::clause(&small, [1, -3]).unwrap();
    f.minimize().unwrap();
    // A stick never contains a balanced four-leaf shape: the destination
    // branches where the source has a single variable.
    let big = Arc::new(Vtree::linear(8));
    let error = f.embed(&big, |v| v).unwrap_err();
    assert!(matches!(error, GraftError::NotIsomorphic { .. }));
    assert!(error.to_string().contains("does not contain the source vtree's shape"));
}

#[test]
fn a_destination_leaf_where_the_source_branches_is_refused() {
    let small = Arc::new(Vtree::linear(4));
    let mut f = Tdd::clause(&small, [1, -3]).unwrap();
    f.minimize().unwrap();
    // The other direction: the destination runs out of variables where the
    // source still has a subtree to copy.
    let big = Arc::new(Vtree::balanced(4));
    assert!(matches!(f.embed(&big, |v| v), Err(GraftError::NotIsomorphic { .. })));
}

#[test]
fn a_diagram_with_a_marginal_level_is_refused() {
    let (small, mut f) = pair_relation();
    Engine::new().marginalize_levels(&mut f, &[small.root()]).unwrap();
    let big = Arc::new(Vtree::linear(4));
    assert!(matches!(
        f.embed(&big, |v| v),
        Err(GraftError::Operation(OperationError::MarginalLevel(_))),
    ));
}
