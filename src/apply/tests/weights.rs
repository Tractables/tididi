use super::*;
use crate::apply::RestrictionOutcome;
use crate::diagram::{Arithmetic, LiteralWeights, RationalWeights, WeightStore};
use crate::limits::{LimitConfig, OperationError};


/// Attach one equal weight to both polarities of every variable.
fn weighted(mut f: Tdd, n: i64, arithmetic: Arithmetic) -> Tdd {
    let values = vec![LiteralWeights { negative: rat(n, 1), positive: rat(n, 1) }; f.vtree.num_vars() as usize];
    f.set_weights(WeightStore::new(RationalWeights::from_literals(&values), arithmetic)).unwrap();
    assert_canonical(&f);
    f
}

#[test]
fn incompatible_weights_are_rejected_before_shortcuts_and_allocations() {
    let tree = Arc::new(Vtree::balanced(4));
    let eng = Engine::new();
    for (n, arithmetic) in [(2, Arithmetic::ExactRational), (1, Arithmetic::SignedLog)] {
        for (a, b) in [
            (Tdd::one(&tree), Tdd::one(&tree)),
            (Tdd::zero(&tree), Tdd::one(&tree)),
            (Tdd::clause(&tree, [1]).unwrap(), Tdd::clause(&tree, [2, 3]).unwrap()),
        ] {
            let a = weighted(a, 1, Arithmetic::ExactRational);
            let b = weighted(b, n, arithmetic);
            let _limit = eng.limits().scope(LimitConfig::none().with_memory_budget_bytes(Some(0)));
            for (f, g) in [(&a, &b), (&b, &a)] {
                assert_eq!(eng.and(f.clone(), g.clone()).unwrap_err(), OperationError::IncompatibleWeights);
                assert_eq!(eng.or(f.clone(), g.clone()).unwrap_err(), OperationError::IncompatibleWeights);
                assert_eq!(eng.and_marginalizing(f.clone(), g.clone(), &[tree.root()]).unwrap_err(), OperationError::IncompatibleWeights);
            }
        }
    }
}

#[test]
fn incompatible_marginal_columns_are_rejected_before_merging() {
    let tree = Arc::new(Vtree::balanced(4));
    let eng = Engine::new();
    let (left, right) = tree.children(tree.root());
    let mut a = weighted(Tdd::clause(&tree, [1]).unwrap(), 1, Arithmetic::ExactRational);
    let mut b = weighted(Tdd::clause(&tree, [3]).unwrap(), 2, Arithmetic::ExactRational);
    eng.marginalize_levels(&mut a, &[left]).unwrap();
    eng.marginalize_levels(&mut b, &[right]).unwrap();
    assert_canonical(&a);
    assert_canonical(&b);
    for (f, g) in [(&a, &b), (&b, &a)] {
        assert_eq!(eng.and(f.clone(), g.clone()).unwrap_err(), OperationError::IncompatibleWeights);
    }
}

#[test]
fn integer_marginal_values_cannot_inherit_weights() {
    let tree = Arc::new(Vtree::balanced(4));
    let eng = Engine::new();
    let mut counts = Tdd::clause(&tree, [1, 3]).unwrap();
    eng.marginalize_levels(&mut counts, &[tree.children(tree.root()).0]).unwrap();
    assert_canonical(&counts);
    let weights = weighted(Tdd::one(&tree), 1, Arithmetic::ExactRational);
    for (f, g) in [(&counts, &weights), (&weights, &counts)] {
        assert_eq!(eng.and(f.clone(), g.clone()).unwrap_err(), OperationError::IncompatibleWeights);
    }
}

#[test]
fn structural_operands_inherit_weights_in_either_order() {
    let tree = Arc::new(Vtree::balanced(2));
    let eng = Engine::new();
    for arithmetic in [Arithmetic::ExactRational, Arithmetic::SignedLog] {
        for (a, b, and_value, or_value) in [
            (Tdd::one(&tree), Tdd::one(&tree), 16, 16),
            (Tdd::one(&tree), Tdd::clause(&tree, [1]).unwrap(), 8, 16),
            (Tdd::zero(&tree), Tdd::clause(&tree, [1]).unwrap(), 0, 8),
            (Tdd::clause(&tree, [1]).unwrap(), Tdd::zero(&tree), 0, 8),
        ] {
            let a = weighted(a, 2, arithmetic);
            assert_canonical(&b);
            for (f, g) in [(&a, &b), (&b, &a)] {
                for (mut result, expected) in [
                    (eng.and(f.clone(), g.clone()).unwrap(), and_value),
                    (eng.or(f.clone(), g.clone()).unwrap(), or_value),
                ] {
                    result.minimize().unwrap();
                    assert_canonical(&result);
                    let value = eng.weighted_value(&result).unwrap().unwrap();
                    if let Some(log) = value.as_log() {
                        if expected == 0 { assert!(log.is_zero()); }
                        else { assert!((log.log10_abs() - (expected as f64).log10()).abs() < 1e-12); }
                    } else {
                        assert_eq!(exact_weight(&value), rat(expected, 1));
                    }
                }
            }
        }
    }
}

#[test]
fn separately_allocated_equal_weight_tables_are_compatible() {
    let tree = Arc::new(Vtree::balanced(2));
    let eng = Engine::new();
    let a = weighted(Tdd::clause(&tree, [1]).unwrap(), 2, Arithmetic::ExactRational);
    let b = weighted(Tdd::clause(&tree, [2]).unwrap(), 2, Arithmetic::ExactRational);
    let mut result = eng.and(a, b).unwrap();
    result.minimize().unwrap();
    assert_canonical(&result);
    assert_eq!(exact_weight(&eng.weighted_value(&result).unwrap().unwrap()), rat(4, 1));
}

#[test]
fn a_structural_operand_inherits_a_marginal_operands_weight_table() {
    let tree = Arc::new(Vtree::balanced(4));
    let eng = Engine::new();
    let mut a = weighted(Tdd::clause(&tree, [1]).unwrap(), 2, Arithmetic::ExactRational);
    eng.marginalize_levels(&mut a, &[tree.children(tree.root()).0]).unwrap();
    assert_canonical(&a);
    let b = Tdd::clause(&tree, [3]).unwrap();
    assert_canonical(&b);
    for (f, g) in [(&a, &b), (&b, &a)] {
        let mut result = eng.and(f.clone(), g.clone()).unwrap();
        result.minimize().unwrap();
        assert_canonical(&result);
        assert_eq!(exact_weight(&eng.weighted_value(&result).unwrap().unwrap()), rat(64, 1));
    }
}

#[test]
fn projecting_a_leaf_output_preserves_weights() {
    let tree = Arc::new(Vtree::balanced(1));
    let eng = Engine::new();
    let f = weighted(Tdd::clause(&tree, [1]).unwrap(), 2, Arithmetic::ExactRational);
    let result = eng.exists_var(f.clone(), VarId(1)).unwrap();
    assert_canonical(&result);
    assert_eq!(exact_weight(&eng.weighted_value(&result).unwrap().unwrap()), rat(4, 1));
    let mut marginal = f;
    eng.marginalize_levels(&mut marginal, &[tree.root()]).unwrap();
    assert_canonical(&marginal);
    assert_eq!(eng.exists_var(marginal, VarId(1)).unwrap_err(), OperationError::MarginalLevel(tree.root()));
}

#[test]
fn an_empty_clause_preserves_the_accumulators_weights() {
    let tree = Arc::new(Vtree::balanced(2));
    let eng = Engine::new();
    for arithmetic in [Arithmetic::ExactRational, Arithmetic::SignedLog] {
        let f = weighted(Tdd::one(&tree), 2, arithmetic);
        let result = eng.and_clause(f, &[] as &[crate::Literal]).unwrap();
        assert_canonical(&result);
        assert!(result.is_zero());
        let value = eng.weighted_value(&result).unwrap().expect("the false result retains weights");
        if let Some(log) = value.as_log() { assert!(log.is_zero()); }
        else { assert_eq!(exact_weight(&value), rat(0, 1)); }
    }
}

#[test]
fn every_care_restriction_outcome_preserves_the_input_weights() {
    let tree = Arc::new(Vtree::balanced(2));
    let eng = Engine::new();
    for arithmetic in [Arithmetic::ExactRational, Arithmetic::SignedLog] {
        for (f, care, expected) in [
            (Tdd::clause(&tree, [1]).unwrap(), Tdd::one(&tree), 8),
            (Tdd::clause(&tree, [1, 2]).unwrap(), Tdd::clause(&tree, [1]).unwrap(), 8),
            (Tdd::clause(&tree, [1]).unwrap(), Tdd::zero(&tree), 0),
            (Tdd::clause(&tree, [1]).unwrap(), Tdd::clause(&tree, [-1]).unwrap(), 0),
        ] {
            let f = weighted(f, 2, arithmetic);
            let care = weighted(care, 7, Arithmetic::ExactRational);
            let mut result = eng.restrict_to_care(f, care).unwrap().into_tdd();
            result.minimize().unwrap();
            assert_canonical(&result);
            let value = eng.weighted_value(&result).unwrap().expect("restriction preserves weights");
            if let Some(log) = value.as_log() {
                if expected == 0 { assert!(log.is_zero()); }
                else { assert!((log.log10_abs() - (expected as f64).log10()).abs() < 1e-12); }
            } else { assert_eq!(exact_weight(&value), rat(expected, 1)); }
        }
    }
}

#[test]
fn a_care_rebuild_keeps_the_stores_of_marginal_levels() {
    let tree = Arc::new(Vtree::balanced(4));
    let eng = Engine::new();
    let mut f = weighted(Tdd::clause(&tree, [1]).unwrap() & Tdd::clause(&tree, [3, 4]).unwrap(), 2, Arithmetic::ExactRational);
    eng.marginalize_levels(&mut f, &[tree.children(tree.root()).0]).unwrap();
    f.minimize().unwrap();
    assert_canonical(&f);
    let care = Tdd::clause(&tree, [3]).unwrap();
    assert_canonical(&care);
    let outcome = eng.restrict_to_care(f, care).unwrap();
    assert!(matches!(outcome, RestrictionOutcome::Shrunk(_)));
    let mut result = outcome.into_tdd();
    result.minimize().unwrap();
    assert_canonical(&result);
    assert_eq!(exact_weight(&eng.weighted_value(&result).unwrap().unwrap()), rat(64, 1));
}

#[test]
fn ite_inherits_weights_from_every_nonempty_operand_subset() {
    let tree = Arc::new(Vtree::balanced(3));
    let eng = Engine::new();
    for arithmetic in [Arithmetic::ExactRational, Arithmetic::SignedLog] {
        for subset in 0..8 {
            let operands = [1, 2, 3].map(|literal| eng.literal(&tree, literal).unwrap());
            let [condition, yes, no] = operands.into_iter().enumerate().map(|(i, f)| {
                if subset & (1 << i) != 0 { weighted(f, 2, arithmetic) } else { f }
            }).collect::<Vec<_>>().try_into().unwrap();
            for f in [&condition, &yes, &no] { assert_canonical(f); }
            let result = eng.ite(condition, yes, no).unwrap();
            assert_canonical(&result);
            assert_eq!(result.model_count().unwrap(), 4u32.into());
            for row in 0..8 {
                let a = [row & 1 != 0, row & 2 != 0, row & 4 != 0];
                assert_eq!(eval(&result, &a), if a[0] { a[1] } else { a[2] });
            }
            if subset == 0 {
                assert!(result.weights().is_none());
            } else {
                assert_eq!(result.weights().unwrap().arithmetic(), arithmetic);
                let value = eng.weighted_value(&result).unwrap().unwrap();
                if let Some(log) = value.as_log() {
                    assert!((log.log10_abs() - 32f64.log10()).abs() < 1e-12);
                } else { assert_eq!(exact_weight(&value), rat(32, 1)); }
            }
        }
    }
}

#[test]
fn ternary_weight_conflicts_leave_unweighted_operands_unmodified() {
    let tree = Arc::new(Vtree::balanced(2));
    for missing in 0..3 {
        let mut operands = [Tdd::one(&tree), Tdd::one(&tree), Tdd::one(&tree)];
        operands[(missing + 1) % 3] = weighted(Tdd::one(&tree), 1, Arithmetic::ExactRational);
        operands[(missing + 2) % 3] = weighted(Tdd::one(&tree), 2, Arithmetic::ExactRational);
        let [a, b, c] = &mut operands;
        assert_eq!(crate::apply::prepare_weights(&mut [a, b, c]), Err(OperationError::IncompatibleWeights));
        assert!(operands[missing].weights().is_none());
        for f in &operands { assert_canonical(f); }
    }
}
