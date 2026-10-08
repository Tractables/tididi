//! `RationalWeights::restated`: a weight table in a component's own numbering.

use std::sync::Arc;

use super::*;
use crate::test_helpers::assert_canonical;
use crate::vtree::Vtree;
use crate::Tdd;

fn w(n: i64) -> BigRational {
    BigRational::from_integer(n.into())
}

/// Variable `v` weighs `10 v + 1` negative and `10 v + 2` positive.
fn table(num_vars: i64) -> RationalWeights {
    let weights: Vec<_> = (1..=num_vars)
        .map(|v| LiteralWeights { negative: w(10 * v + 1), positive: w(10 * v + 2) })
        .collect();
    RationalWeights::from_literals(&weights)
}

#[test]
fn each_local_variable_takes_the_weights_of_its_global_one() {
    let global = table(4);
    let local = global.restated(&[VarId(4), VarId(2), VarId(3)]).unwrap();
    assert_eq!(local.num_vars(), 3);
    for (local_var, global_var) in [(1, 4), (2, 2), (3, 3)] {
        assert_eq!(local.neg_weight(VarId(local_var)), global.neg_weight(VarId(global_var)));
        assert_eq!(local.pos_weight(VarId(local_var)), global.pos_weight(VarId(global_var)));
    }
}

#[test]
fn an_empty_map_gives_an_empty_table_and_a_repeated_variable_is_copied() {
    let global = table(2);
    assert_eq!(global.restated(&[]).unwrap().num_vars(), 0);
    let twice = global.restated(&[VarId(2), VarId(2)]).unwrap();
    assert_eq!(twice.pos_weight(VarId(1)), &w(22));
    assert_eq!(twice.pos_weight(VarId(2)), &w(22));
}

#[test]
fn a_variable_the_table_does_not_cover_is_refused() {
    let global = table(3);
    assert_eq!(global.restated(&[VarId(1), VarId(4)]), Err(TddBuildError::MissingVariableWeight(VarId(4))));
    assert_eq!(global.restated(&[VarId(0)]), Err(TddBuildError::MissingVariableWeight(VarId(0))));
    assert_eq!(RationalWeights::unit(0).restated(&[VarId(1)]), Err(TddBuildError::MissingVariableWeight(VarId(1))));
}

/// A component compiled in its own numbering evaluates under the restated
/// table to what the same function gives in the global numbering.
#[test]
fn a_component_evaluates_as_it_does_in_the_global_numbering() {
    let global = table(5);
    let local_to_global = [VarId(5), VarId(2), VarId(4)];
    let local = global.restated(&local_to_global).unwrap();

    let local_vtree = Arc::new(Vtree::balanced(3));
    let component = Tdd::clause(&local_vtree, [1, -2, 3]).unwrap();
    assert_canonical(&component);

    let global_vtree = Arc::new(Vtree::balanced_over(&local_to_global).unwrap());
    let same = Tdd::clause(&global_vtree, [5, -2, 4]).unwrap();
    assert_canonical(&same);

    assert_eq!(component.evaluate(&local).unwrap(), same.evaluate(&global).unwrap());
}
