use std::sync::Arc;

use crate::restructure::EmbedError;
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
        EmbedError::Vtree(VtreeError::OverlappingVariable(VarId(3))),
    );
}

#[test]
fn an_image_variable_outside_the_destination_is_refused() {
    let (_, f) = pair_relation();
    let big = Arc::new(Vtree::linear(4));
    assert_eq!(
        f.embed(&big, |v| VarId(v.0 + 9)).unwrap_err(),
        EmbedError::VariableOutOfRange { variable: VarId(10), num_vars: 4 },
    );
    assert_eq!(
        f.embed(&big, |v| VarId(v.0 + 9)).unwrap_err().to_string(),
        "renamed variable 10 is outside the variables 1 to 4",
    );
}

#[test]
fn a_renaming_that_reverses_the_leaf_order_is_refused() {
    let (small, f) = pair_relation();
    let big = Arc::new(Vtree::linear(4));
    // The relation's first variable would land to the right of its second.
    let reversed = |v: VarId| if v == VarId(1) { VarId(3) } else { VarId(1) };
    let Err(EmbedError::NotIsomorphic { source }) = f.embed(&big, reversed) else {
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
    assert!(matches!(error, EmbedError::NotIsomorphic { .. }));
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
    assert!(matches!(f.embed(&big, |v| v), Err(EmbedError::NotIsomorphic { .. })));
}

#[test]
fn a_diagram_with_a_marginal_level_is_refused() {
    let (small, mut f) = pair_relation();
    Engine::new().marginalize_levels(&mut f, &[small.root()]).unwrap();
    let big = Arc::new(Vtree::linear(4));
    assert!(matches!(
        f.embed(&big, |v| v),
        Err(EmbedError::Operation(OperationError::MarginalLevel(_))),
    ));
}

#[test]
fn embedding_checks_planning_work_before_the_false_shortcut() {
    use crate::limits::{LimitConfig, StopAt, StopRules};

    let small = Arc::new(Vtree::linear(2));
    let f = Tdd::zero(&small);
    let big = Arc::new(Vtree::linear(8));
    assert_canonical(&f);
    for stride in [None, Some(1)] {
        let engine = Engine::new();
        engine.limits().pin_reduce_poll_stride(stride);
        let result = {
            let _scope = engine.limits().scope(LimitConfig::none().with_stop_rules(StopRules {
                unconditional: Some(StopAt::WorkUnits(1)), ..StopRules::default()
            }));
            engine.embed(&f, &big, |var| VarId(var.0 + 2))
        };
        assert_eq!(result.unwrap_err(), EmbedError::Operation(OperationError::Stopped));
        let (g, levels) = engine.embed(&f, &big, |var| VarId(var.0 + 2)).unwrap();
        assert_canonical(&g);
        assert!(g.is_zero());
        for (leaf, var) in small.leaf_bottomup() {
            assert_eq!(levels.level_of(leaf), big.leaf_of(VarId(var.0 + 2)).unwrap());
        }
    }
}

#[test]
fn embedding_retries_after_each_refused_reservation_without_changing_the_source() {
    use crate::diagram::{Arithmetic, RationalWeights, WeightStore};

    let (_, mut f) = pair_relation();
    f.set_weights(WeightStore::new(RationalWeights::unit(2), Arithmetic::ExactRational)).unwrap();
    let before = format!("{f:?}");
    let big = Arc::new(Vtree::linear(6));
    let rename = |var: VarId| VarId(if var == VarId(1) { 2 } else { 5 });
    let expected = Tdd::clause(&big, [2, 5]).unwrap();
    assert_canonical(&f);
    assert_canonical(&expected);
    let mut completed = false;
    let mut refusals = 0;
    for cut in 0..256 {
        let engine = Engine::new();
        engine.limits().refuse_nth_reserve(cut);
        let result = engine.embed(&f, &big, rename);
        engine.limits().grant_every_reserve();
        let (g, levels) = match result {
            Ok(result) => { completed = true; result }
            Err(error) => {
                assert_eq!(error, EmbedError::Operation(OperationError::OverBudget));
                refusals += 1;
                engine.embed(&f, &big, rename).unwrap()
            }
        };
        assert_canonical(&g);
        assert!(g.equivalent(&expected).unwrap());
        assert!(g.weights().is_none());
        for (leaf, var) in f.vtree().leaf_bottomup() {
            assert_eq!(levels.level_of(leaf), big.leaf_of(rename(var)).unwrap());
        }
        assert_eq!(format!("{f:?}"), before);
        assert!(f.weights().is_some());
        assert_eq!(f.weighted_value().unwrap().unwrap().as_rational().into_owned(),
            num_rational::BigRational::from_integer(3.into()));
        assert_canonical(&f);
        if completed { break; }
    }
    assert!(completed && refusals > 0, "cover planning, assembly and cleanup reservations");
}
