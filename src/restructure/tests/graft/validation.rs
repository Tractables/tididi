use std::sync::Arc;
use crate::{Engine, Tdd};
use crate::restructure::GraftError;
use crate::diagram::TddBuildError;
use crate::diagram::{Arithmetic, LiteralWeights, RationalWeights, WeightStore};

use crate::test_helpers::{assert_canonical, rat};
use crate::vtree::{VarId, Vtree};

/// A uniform table over a fixed variable space.
fn store(n: usize, weight: i64, arithmetic: Arithmetic) -> WeightStore {
    WeightStore::new(RationalWeights::from_literals(&vec![LiteralWeights {
        negative: rat(weight, 1), positive: rat(weight, 1),
    }; n]), arithmetic)
}

/// A weighted clause whose root stores its value.
fn marginal_part() -> Tdd {
    let tree = Arc::new(Vtree::balanced(2));
    let mut f = Tdd::clause(&tree, [1, 2]).unwrap();
    f.set_weights(store(2, 2, Arithmetic::ExactRational)).unwrap();
    (Engine::new()).marginalize_levels(&mut f, &[tree.root()]).unwrap();
    assert_canonical(&f);
    f
}

#[test]
fn graft_rejects_weighted_marginals_without_a_destination() {
    assert!(matches!(Tdd::graft(vec![marginal_part()], &[]), Err(GraftError::PartWeights { part: 0, source: TddBuildError::WeightedLevelWithoutStore { .. } })));
    assert!(matches!(Tdd::graft_over(&Engine::new(), vec![(marginal_part(), vec![VarId(1), VarId(2)])], &[], 2, None), Err(GraftError::PartWeights { part: 0, source: TddBuildError::WeightedLevelWithoutStore { .. } })));
}

#[test]
fn graft_rejects_changed_weights_after_marginalization() {
    for (weight, arithmetic) in [(3, Arithmetic::ExactRational), (2, Arithmetic::SignedLog)] {
        assert_eq!(Tdd::graft_over(&Engine::new(), vec![(marginal_part(), vec![VarId(1), VarId(2)])], &[], 2, Some(store(2, weight, arithmetic))).unwrap_err(), GraftError::PartWeights { part: 0, source: TddBuildError::IncompatibleWeights });
    }
}

#[test]
fn graft_rejects_integer_counts_in_a_weighted_destination() {
    let tree = Arc::new(Vtree::balanced(2));
    let mut f = Tdd::clause(&tree, [1, 2]).unwrap();
    (Engine::new()).marginalize_levels(&mut f, &[tree.root()]).unwrap();
    assert_canonical(&f);
    assert!(matches!(Tdd::graft_over(&Engine::new(), vec![(f, vec![VarId(1), VarId(2)])], &[], 2, Some(store(2, 2, Arithmetic::ExactRational))), Err(GraftError::PartWeights { part: 0, source: TddBuildError::CountLevelWithWeights { .. } })));
}

#[test]
fn a_false_graft_keeps_the_destination_weights() {
    let tree = Arc::new(Vtree::balanced(2));
    let f = Tdd::zero(&tree);
    assert_canonical(&f);
    let (result, _) = Tdd::graft_over(&Engine::new(), vec![(f, vec![VarId(1), VarId(2)])], &[], 2, Some(store(2, 2, Arithmetic::ExactRational))).unwrap();
    assert_canonical(&result);
    assert!(result.is_zero());
    assert_eq!(result.weighted_value().unwrap().expect("the false result retains weights").as_rational().into_owned(), rat(0, 1));
}

#[test]
fn graft_rejects_missing_variable_mappings() {
    let tree = Arc::new(Vtree::leaf(VarId(3)));
    let f = Tdd::one(&tree);
    assert_canonical(&f);
    assert_eq!(Tdd::graft_over(&Engine::new(), vec![(f, vec![VarId(1), VarId(2)])], &[], 3, None).unwrap_err(), GraftError::MissingVariableMapping { part: 0, variable: VarId(3) });
}

#[test]
fn graft_rejects_variables_outside_the_destination_space() {
    let tree = Arc::new(Vtree::balanced(1));
    let f = Tdd::one(&tree);
    assert_canonical(&f);
    assert_eq!(Tdd::graft_over(&Engine::new(), vec![(f, vec![VarId(3)])], &[], 2, None).unwrap_err(), GraftError::VariableOutOfRange { variable: VarId(3), num_vars: 2 });
    assert_eq!(Tdd::graft(vec![], &[VarId(0)]).unwrap_err(), GraftError::VariableOutOfRange { variable: VarId(0), num_vars: 0 });
    // A zero spine variable beside real parts names the space those parts span.
    assert_eq!(Tdd::graft(vec![Tdd::one(&Arc::new(Vtree::balanced(4)))], &[VarId(0)]).unwrap_err(), GraftError::VariableOutOfRange { variable: VarId(0), num_vars: 4 });
}

#[test]
fn graft_rejects_missing_destination_weights_before_false_shortcuts() {
    let tree = Arc::new(Vtree::balanced(2));
    for f in [Tdd::one(&tree), Tdd::zero(&tree)] {
        assert_canonical(&f);
        assert_eq!(Tdd::graft_over(&Engine::new(), vec![(f, vec![VarId(1), VarId(2)])], &[], 2, Some(store(1, 2, Arithmetic::ExactRational))).unwrap_err(), GraftError::DestinationWeights(TddBuildError::MissingVariableWeight(VarId(2))));
    }
}

#[test]
fn graft_keeps_a_marginal_root_canonical_under_a_new_parent() {
    let tree = Arc::new(Vtree::balanced(2));
    let mut f = Tdd::clause(&tree, [1, 2]).unwrap();
    (Engine::new()).marginalize_levels(&mut f, &[tree.root()]).unwrap();
    assert_canonical(&f);
    let result = Tdd::graft(vec![f], &[VarId(3)]).unwrap();
    assert_eq!(result.model_count().unwrap(), 6u32.into());
    assert_canonical(&result);
}

#[test]
fn graft_reweights_a_structural_part_in_the_destination() {
    let tree = Arc::new(Vtree::balanced(2));
    let mut f = Tdd::clause(&tree, [1]).unwrap();
    let previous = marginal_part();
    f.set_weights(previous.weights().unwrap().clone()).unwrap();
    assert_canonical(&f);
    let (mut result, _) = Tdd::graft_over(&Engine::new(), vec![(f, vec![VarId(1), VarId(2)])], &[], 2, Some(store(2, 3, Arithmetic::ExactRational))).unwrap();
    assert_canonical(&result);
    assert_eq!(result.weighted_value().unwrap().unwrap().as_rational().into_owned(), rat(18, 1));
    let root = result.vtree().root();
    (Engine::new()).marginalize_levels(&mut result, &[root]).unwrap();
    assert_canonical(&result);
    assert_eq!(result.weighted_value().unwrap().unwrap().as_rational().into_owned(), rat(18, 1));
}

#[test]
fn graft_preserves_renamed_marginal_roots_in_both_arithmetics() {
    let local = Arc::new(Vtree::balanced(2));
    let eng = Engine::new();
    for arithmetic in [Arithmetic::ExactRational, Arithmetic::SignedLog] {
        let weights = [
            LiteralWeights { negative: rat(2, 1), positive: rat(3, 1) },
            LiteralWeights { negative: rat(5, 1), positive: rat(7, 1) },
        ];
        let mut f = Tdd::clause(&local, [1, 2]).unwrap();
        f.set_weights(WeightStore::new(RationalWeights::from_literals(&weights), arithmetic)).unwrap();
        eng.marginalize_levels(&mut f, &[local.root()]).unwrap();
        assert_canonical(&f);
        let global = [weights[1].clone(), weights[0].clone(), LiteralWeights { negative: rat(11, 1), positive: rat(13, 1) }];
        let (mut result, _) = Tdd::graft_over(&eng, vec![(f, vec![VarId(2), VarId(1)])], &[VarId(3)], 3,
            Some(WeightStore::new(RationalWeights::from_literals(&global), arithmetic))).unwrap();
        assert_canonical(&result);
        result.minimize().unwrap();
        assert_canonical(&result);
        let value = result.weighted_value().unwrap().unwrap();
        // Clause weight: 5 * 12 - 2 * 5 = 50; the free variable contributes 24.
        if let Some(log) = value.as_log() { assert!((log.log10_abs() - 1200f64.log10()).abs() < 1e-12); }
        else { assert_eq!(value.as_rational().into_owned(), rat(1200, 1)); }
    }
}

#[test]
fn graft_boundary_cleanup_observes_the_engine_budget() {
    let eng = Engine::new();
    let part = marginal_part();
    let _scope = eng.limits().scope(crate::limits::LimitConfig::none().with_memory_budget_bytes(Some(0)));
    let result = Tdd::graft_over(&eng, vec![(part, vec![VarId(1), VarId(2)])], &[VarId(3)], 3,
        Some(store(3, 2, Arithmetic::ExactRational)));
    assert_eq!(result.unwrap_err(), GraftError::Operation(crate::OperationError::OverBudget));
}

#[test]
fn graft_canonicalizes_equal_weight_slots_of_a_marginal_leaf_root() {
    let eng = Engine::new();
    let tree = Arc::new(Vtree::leaf(VarId(1)));
    for arithmetic in [Arithmetic::ExactRational, Arithmetic::SignedLog] {
        let mut f = Tdd::clause(&tree, [-1]).unwrap();
        f.set_weights(WeightStore::new(RationalWeights::unit(1), arithmetic)).unwrap();
        eng.marginalize_levels(&mut f, &[tree.root()]).unwrap();
        assert_canonical(&f);
        let (mut result, _) = Tdd::graft_over(&eng, vec![(f, vec![VarId(1)])], &[VarId(2)], 2,
            Some(WeightStore::new(RationalWeights::unit(2), arithmetic))).unwrap();
        assert_canonical(&result);
        result.minimize().unwrap();
        assert_canonical(&result);
        let value = result.weighted_value().unwrap().unwrap();
        if let Some(log) = value.as_log() { assert!((log.log10_abs() - 2f64.log10()).abs() < 1e-12); }
        else { assert_eq!(value.as_rational().into_owned(), rat(2, 1)); }
    }
}

#[test]
fn an_unmapped_variable_is_named_the_way_the_input_names_it() {
    let tree = Arc::new(Vtree::leaf(VarId(1)));
    let f = Tdd::clause(&tree, [1]).unwrap();
    assert_eq!(
        Tdd::graft_over(&Engine::new(), vec![(f, vec![])], &[], 1, None).unwrap_err().to_string(),
        "part 0 has no mapping for variable 1",
    );
}
