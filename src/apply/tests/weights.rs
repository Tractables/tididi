use super::*;
use crate::apply::RestrictionOutcome;
use crate::diagram::{Arithmetic, LiteralWeights, RationalWeights, Weights};
use crate::limits::{LimitConfig, OperationError};
use crate::test_helpers::bounded_log10;

/// A balanced vtree on `vars` variables with one equal weight `n` on both
/// polarities of every variable.
pub(in crate::apply) fn weighted_tree(vars: u32, n: i64, arithmetic: Arithmetic) -> crate::vtree::Home {
    let values = vec![
        LiteralWeights {
            negative: rat(n, 1),
            positive: rat(n, 1)
        };
        vars as usize
    ];
    let weights = Weights::new(RationalWeights::from_literals(&values), arithmetic);
    crate::vtree::Home::weighted(Vtree::balanced(vars), weights).unwrap()
}

fn assert_weighted_int(eng: &Engine, f: &Tdd, expected: i64) {
    let value = eng.weighted_value(f).unwrap().expect("weighted diagram");
    if let Some(log) = bounded_log10(&value) {
        if expected == 0 {
            assert_eq!(log, f64::NEG_INFINITY);
        } else {
            assert!((log - (expected as f64).log10()).abs() < 1e-12);
        }
    } else {
        assert_eq!(exact_weight(&value), rat(expected, 1));
    }
}

/// Two vtrees of one shape whose weights differ, or of which one is
/// unweighted, are different vtrees: every operation refuses the pair, also
/// where one operand is constant or summed out, and before it allocates.
#[test]
fn operands_on_differently_weighted_vtrees_are_refused_before_shortcuts_and_allocations() {
    let eng = Engine::new();
    let first = weighted_tree(4, 1, Arithmetic::ExactRational);
    for other in [
        weighted_tree(4, 2, Arithmetic::ExactRational),
        weighted_tree(4, 1, Arithmetic::SignedLog),
        weighted_tree(4, 1, Arithmetic::ScaledFloat),
        weighted_tree(4, 1, Arithmetic::ExactRational),
        crate::vtree::Home::new(Vtree::balanced(4)),
    ] {
        let summed = |tree: &crate::vtree::Home, lits: &[i32]| {
            let mut f = Tdd::clause(tree, lits.iter().copied()).unwrap();
            eng.marginalize(&mut f, |v| v.0 <= 2).unwrap();
            assert_canonical(&f);
            f
        };
        for (a, b) in [
            (Tdd::one(&first), Tdd::one(&other)),
            (Tdd::zero(&first), Tdd::one(&other)),
            (Tdd::clause(&first, [1]).unwrap(), Tdd::clause(&other, [2, 3]).unwrap()),
            (summed(&first, &[1]), summed(&other, &[3])),
            (summed(&first, &[1, 3]), Tdd::one(&other)),
        ] {
            let _limit = eng.limits().scope(LimitConfig::none().with_memory_budget_bytes(Some(0)));
            for (f, g) in [(&a, &b), (&b, &a)] {
                let refused = |r: Result<Tdd, OperationError>| assert_eq!(r.unwrap_err(), OperationError::VtreeMismatch);
                refused(eng.and(f.clone(), g.clone()));
                refused(eng.or(f.clone(), g.clone()));
                refused(eng.xor(f.clone(), g.clone()));
                refused(eng.and_marginalizing_where(f.clone(), g.clone(), |_| true));
                refused(eng.or_many(vec![f.clone(), g.clone()]));
                refused(eng.nor_many(vec![f.clone(), g.clone()]));
                refused(eng.ite(f.clone(), g.clone(), f.clone()));
                refused(eng.ite(f.clone(), f.clone(), g.clone()));
            }
        }
    }
}

#[test]
fn structural_operands_on_a_weighted_vtree_combine_in_either_order() {
    let eng = Engine::new();
    for arithmetic in [
        Arithmetic::ExactRational,
        Arithmetic::SignedLog,
        Arithmetic::ScaledFloat,
    ] {
        let tree = weighted_tree(2, 2, arithmetic);
        for (a, b, and_value, or_value) in [
            (Tdd::one(&tree), Tdd::one(&tree), 16, 16),
            (Tdd::one(&tree), Tdd::clause(&tree, [1]).unwrap(), 8, 16),
            (Tdd::zero(&tree), Tdd::clause(&tree, [1]).unwrap(), 0, 8),
            (Tdd::clause(&tree, [1]).unwrap(), Tdd::zero(&tree), 0, 8),
        ] {
            assert_canonical(&a);
            assert_canonical(&b);
            for (f, g) in [(&a, &b), (&b, &a)] {
                for (mut result, expected) in [
                    (eng.and(f.clone(), g.clone()).unwrap(), and_value),
                    (eng.or(f.clone(), g.clone()).unwrap(), or_value),
                ] {
                    result.minimize().unwrap();
                    assert_canonical(&result);
                    assert_weighted_int(&eng, &result, expected);
                }
            }
        }
    }
}

#[test]
fn a_structural_operand_conjoins_with_a_weighted_marginal_operand() {
    let tree = weighted_tree(4, 2, Arithmetic::ExactRational);
    let eng = Engine::new();
    let mut a = Tdd::clause(&tree, [1]).unwrap();
    eng.marginalize(&mut a, |v| v.0 <= 2).unwrap();
    assert_canonical(&a);
    let b = Tdd::clause(&tree, [3]).unwrap();
    assert_canonical(&b);
    for (f, g) in [(&a, &b), (&b, &a)] {
        let mut result = eng.and(f.clone(), g.clone()).unwrap();
        result.minimize().unwrap();
        assert_canonical(&result);
        assert_eq!(
            exact_weight(&eng.weighted_value(&result).unwrap().unwrap()),
            rat(64, 1)
        );
    }
}

#[test]
fn projecting_a_leaf_output_preserves_weights() {
    let tree = weighted_tree(1, 2, Arithmetic::ExactRational);
    let eng = Engine::new();
    let f = Tdd::clause(&tree, [1]).unwrap();
    assert_canonical(&f);
    let result = eng.exists_var(f.clone(), VarId(1)).unwrap();
    assert_canonical(&result);
    assert_eq!(
        exact_weight(&eng.weighted_value(&result).unwrap().unwrap()),
        rat(4, 1)
    );
    let mut marginal = f;
    eng.marginalize(&mut marginal, |_| true).unwrap();
    assert_canonical(&marginal);
    assert_eq!(exact_weight(&eng.weighted_value(&marginal).unwrap().unwrap()), rat(2, 1));
}

/// On a home, summing every variable leaves the constant on no leaf with
/// the sum's value: projecting a variable it no longer reads keeps it.
#[test]
fn a_projection_after_a_full_sum_keeps_the_sum() {
    let values = vec![LiteralWeights { negative: rat(2, 1), positive: rat(2, 1) }];
    let weights = Weights::new(RationalWeights::from_literals(&values), Arithmetic::ExactRational);
    let tree = crate::vtree::Home::weighted(Vtree::balanced(1), weights).unwrap();
    let eng = Engine::new();
    let mut marginal = Tdd::clause(&tree, [1]).unwrap();
    eng.marginalize(&mut marginal, |_| true).unwrap();
    assert_eq!(exact_weight(&eng.weighted_value(&marginal).unwrap().unwrap()), rat(2, 1));
    match eng.exists_var(marginal, VarId(1)) {
        Ok(projected) => assert_eq!(exact_weight(&eng.weighted_value(&projected).unwrap().unwrap()), rat(2, 1)),
        Err(e) => assert!(matches!(e, OperationError::MarginalLevel(_)), "{e:?}"),
    }
    let home = crate::vtree::Home::new(Vtree::balanced(2));
    for lit in [1, -1] {
        let mut counted = Tdd::clause(&home, [lit]).unwrap();
        eng.marginalize(&mut counted, |_| true).unwrap();
        assert_marginal_canonical(&counted);
        assert_eq!(counted.model_count().unwrap(), 2u32.into());
        assert!(matches!(eng.exists_var(counted.clone(), VarId(1)), Err(OperationError::MarginalLevel(_))));
        assert!(matches!(eng.and(counted, Tdd::clause(&home, [1]).unwrap()), Err(OperationError::MarginalLevel(_))));
    }
}

/// The truth table of `truth` with `var` projected out.
fn exists_row(truth: &[bool], var: u32) -> Vec<bool> {
    let bit = 1usize << (var - 1);
    (0..truth.len()).map(|m| truth[m | bit] || truth[m & !bit]).collect()
}

/// Whether `truth` reads `var`.
fn reads(truth: &[bool], var: u32) -> bool {
    exists_row(truth, var) != truth
}

/// Every weighted entry that can sum a home diagram out, against the
/// enumerated weighted sum: a diagram whose sum reads a summed variable keeps
/// a summed-out level, which projection and conjunction refuse, and one
/// that does not stays structural and projects as the oracle does.
#[test]
fn sums_on_a_home_agree_with_enumeration() {
    let mut rng = crate::test_helpers::Lcg::new(0x5eed_0011);
    for round in 0..300 {
        let n = 1 + rng.below(4) as u32;
        let values: Vec<_> = (0..n)
            .map(|_| LiteralWeights {
                negative: rat(rng.below(5) as i64 - 1, 1 + rng.below(3) as i64),
                positive: rat(rng.below(5) as i64 - 1, 1 + rng.below(3) as i64),
            })
            .collect();
        let weights = Weights::new(RationalWeights::from_literals(&values), Arithmetic::ExactRational);
        let shape = if rng.coin() { Vtree::balanced(n) } else { Vtree::random(n, rng.next_u64()) };
        let tree = crate::vtree::Home::weighted(shape, weights).unwrap();
        let eng = Engine::new();
        let cnf = |rng: &mut crate::test_helpers::Lcg| {
            crate::test_helpers::rand_cnf(rng, n, crate::test_helpers::CnfShape { clauses: 3, width: 2 })
        };
        let (fc, gc) = (cnf(&mut rng), cnf(&mut rng));
        let (ft, gt) = (truth_table(n, &fc), truth_table(n, &gc));
        let f = compile_clauses_on(&eng, &tree, &fc);
        let g = compile_clauses_on(&eng, &tree, &gc);
        let summed: Vec<bool> = (0..n).map(|_| rng.coin()).collect();
        let summable = |v: VarId| summed[v.0 as usize - 1];
        let oracle = |truth: &[bool]| weighted_sum(truth, &values).0;
        let value = |t: &Tdd| exact_weight(&eng.weighted_value(t).unwrap().unwrap());
        let keeps_sum = |truth: &[bool]| (1..=n).any(|v| summed[v as usize - 1] && reads(truth, v));
        let canonical = |t: &Tdd| {
            let t = crate::test_helpers::minimized(&eng, Ok(t.clone()));
            match t.has_marginal_level() {
                true => assert_marginal_canonical(&t),
                false => assert_canonical(&t),
            }
        };
        let what = format!("round {round}: {n} variables, f {fc:?}, g {gc:?}, summed {summed:?}");

        let mut m = f.clone();
        eng.marginalize(&mut m, summable).unwrap();
        canonical(&m);
        assert_eq!(value(&m), oracle(&ft), "marginalize, {what}");
        assert_eq!(m.has_marginal_level(), keeps_sum(&ft), "{what}");
        for v in 1..=n {
            match eng.exists_var(m.clone(), VarId(v)) {
                Ok(p) => {
                    assert!(!m.has_marginal_level(), "exists x{v} of a sum, {what}");
                    assert_eq!(value(&p), oracle(&exists_row(&ft, v)), "exists x{v}, {what}");
                }
                Err(e) => assert!(m.has_marginal_level() && matches!(e, OperationError::MarginalLevel(_)), "exists x{v}: {e:?}, {what}"),
            }
        }

        let fg: Vec<bool> = ft.iter().zip(&gt).map(|(a, b)| *a && *b).collect();
        let c = eng.and_marginalizing_where(f.clone(), g.clone(), summable).unwrap();
        canonical(&c);
        assert_eq!(value(&c), oracle(&fg), "and_marginalizing_where, {what}");
        // A summed variable an operand read may stay a region where the
        // conjunction does not read it.
        assert!(c.has_marginal_level() || !keeps_sum(&fg), "and_marginalizing_where, {what}");

        let x = 1 + rng.below(u64::from(n)) as i32;
        let split = |lit: i32| {
            let mut h = eng.and(f.clone(), Tdd::clause(&tree, [lit]).unwrap()).unwrap();
            eng.marginalize(&mut h, summable).unwrap();
            h
        };
        match eng.add(split(x), split(-x)) {
            Ok(sum) => {
                canonical(&sum);
                assert_eq!(value(&sum), oracle(&ft), "add, {what}");
            }
            Err(e) => assert!(
                matches!(e, OperationError::MarginalLevel(_) | OperationError::SharedModel | OperationError::IncompatibleWeights),
                "add: {e:?}, {what}"
            ),
        }
    }
}

#[test]
fn an_empty_clause_preserves_the_accumulators_weights() {
    let eng = Engine::new();
    for arithmetic in [
        Arithmetic::ExactRational,
        Arithmetic::SignedLog,
        Arithmetic::ScaledFloat,
    ] {
        let tree = weighted_tree(2, 2, arithmetic);
        let f = Tdd::one(&tree);
        let result = eng.and_clause(f, &[] as &[crate::Literal]).unwrap();
        assert_canonical(&result);
        assert!(result.is_zero());
        assert_weighted_int(&eng, &result, 0);
    }
}

#[test]
fn every_care_restriction_outcome_preserves_the_input_weights() {
    let eng = Engine::new();
    for arithmetic in [
        Arithmetic::ExactRational,
        Arithmetic::SignedLog,
        Arithmetic::ScaledFloat,
    ] {
        let tree = weighted_tree(2, 2, arithmetic);
        for (f, care, expected) in [
            (Tdd::clause(&tree, [1]).unwrap(), Tdd::one(&tree), 8),
            (
                Tdd::clause(&tree, [1, 2]).unwrap(),
                Tdd::clause(&tree, [1]).unwrap(),
                8,
            ),
            (Tdd::clause(&tree, [1]).unwrap(), Tdd::zero(&tree), 0),
            (
                Tdd::clause(&tree, [1]).unwrap(),
                Tdd::clause(&tree, [-1]).unwrap(),
                0,
            ),
        ] {
            assert_canonical(&f);
            assert_canonical(&care);
            let mut result = eng.restrict_to_care(f, &care).unwrap().into_tdd();
            result.minimize().unwrap();
            assert_canonical(&result);
            assert_weighted_int(&eng, &result, expected);
        }
    }
}

#[test]
fn a_care_rebuild_keeps_the_weight_values_of_marginal_levels() {
    let tree = weighted_tree(4, 2, Arithmetic::ExactRational);
    let eng = Engine::new();
    let mut f = Tdd::clause(&tree, [1]).unwrap() & Tdd::clause(&tree, [3, 4]).unwrap();
    eng.marginalize(&mut f, |v| v.0 <= 2).unwrap();
    f.minimize().unwrap();
    assert_canonical(&f);
    let care = Tdd::clause(&tree, [3]).unwrap();
    assert_canonical(&care);
    let outcome = eng.restrict_to_care(f, &care).unwrap();
    assert!(matches!(outcome, RestrictionOutcome::Shrunk(_)));
    let mut result = outcome.into_tdd();
    result.minimize().unwrap();
    assert_canonical(&result);
    assert_eq!(
        exact_weight(&eng.weighted_value(&result).unwrap().unwrap()),
        rat(64, 1)
    );
}

#[test]
fn ite_on_a_weighted_vtree_is_weighted() {
    let eng = Engine::new();
    for arithmetic in [
        Arithmetic::ExactRational,
        Arithmetic::SignedLog,
        Arithmetic::ScaledFloat,
    ] {
        for weighted in [false, true] {
            let tree = if weighted { weighted_tree(3, 2, arithmetic) } else { crate::vtree::Home::new(Vtree::balanced(3)) };
            let [condition, yes, no] = [1, 2, 3].map(|literal| eng.literal(&tree, literal).unwrap());
            for f in [&condition, &yes, &no] {
                assert_canonical(f);
            }
            let result = eng.ite(condition, yes, no).unwrap();
            assert_canonical(&result);
            assert_eq!(result.model_count().unwrap(), 4u32.into());
            for row in 0..8 {
                let a = [row & 1 != 0, row & 2 != 0, row & 4 != 0];
                assert_eq!(eval(&result, &a), if a[0] { a[1] } else { a[2] });
            }
            if weighted {
                assert_eq!(result.weights().unwrap().arithmetic(), arithmetic);
                assert_weighted_int(&eng, &result, 32);
            } else {
                assert!(result.weights().is_none());
            }
        }
    }
}

