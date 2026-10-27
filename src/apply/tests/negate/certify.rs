//! The complement of a minimized operand needs only a prune: it comes back
//! certified, and a full reduction with the certificate dropped finds nothing
//! to merge.

use std::sync::Arc;

use crate::reduce::ReductionPlan;
use crate::test_helpers::{assert_same_shape, compile_clauses_on, rand_cnf, vtree_shapes, CnfShape, Lcg};
use crate::{Engine, Tdd, Vtree};

/// `negated` is certified, and minimizing a copy without its certificate
/// leaves the same diagram.
fn assert_certified_minimal(eng: &Engine, negated: &Tdd, what: &str) {
    assert!(negated.levels.is_canonical(negated.output), "{what}: the complement is not certified");
    let mut check = negated.clone();
    check.levels.forget();
    eng.minimize(&mut check).unwrap();
    assert_same_shape(&check, negated, what);
}

#[test]
fn the_complement_of_a_minimized_operand_is_minimal_once_pruned() {
    let eng = &Engine::new();
    let mut rng = Lcg::new(0x1c3f_0a77);
    let mut cases = 0usize;
    for num_vars in [2u32, 3, 5, 8] {
        for (name, vtree) in vtree_shapes(num_vars) {
            let mut operands = vec![Tdd::one(&vtree), Tdd::zero(&vtree)];
            for round in 0..16 {
                // Few short clauses give dense functions, many long ones sparse
                // functions, so both the fill and the root complement vary.
                let shape = CnfShape { clauses: 1 + round % 7, width: 1 + round % 3 };
                operands.push(compile_clauses_on(eng, &vtree, &rand_cnf(&mut rng, num_vars, shape)));
            }
            for f in operands {
                assert!(f.levels.is_canonical(f.output), "{name}: the operand is not certified");
                for plan in [ReductionPlan::Prune, ReductionPlan::default()] {
                    let what = format!("{name}, {num_vars} variables, {plan:?}");
                    let negated = eng.negate_with(f.clone(), plan).unwrap();
                    assert_certified_minimal(eng, &negated, &what);
                    assert!(eng.equivalent(&eng.and(f.clone(), negated.clone()).unwrap(), &Tdd::zero(&vtree)).unwrap(), "{what}");
                    let back = eng.negate(negated).unwrap();
                    assert_certified_minimal(eng, &back, &what);
                    assert_same_shape(&back, &f, &format!("{what}: double complement"));
                    cases += 1;
                }
            }
        }
    }
    assert!(cases > 0, "nothing was tested");
}

#[test]
fn an_uncertified_operand_is_still_reduced_in_full() {
    let eng = &Engine::new();
    let vtree = Arc::new(Vtree::balanced(4));
    let f = Tdd::clause(&vtree, [1, 2]).unwrap();
    let g = Tdd::clause(&vtree, [-2, 3]).unwrap();
    let mut conj = eng.and(f, g).unwrap();
    // The conjunction is not minimized, so negation reduces it as before.
    conj.levels.forget();
    let negated = eng.negate(conj.clone()).unwrap();
    let mut minimized = conj;
    eng.minimize(&mut minimized).unwrap();
    let expected = eng.negate(minimized).unwrap();
    assert_same_shape(&negated, &expected, "uncertified operand");
    let pruned = eng.negate_with(expected.clone(), ReductionPlan::Contract).unwrap();
    assert!(!pruned.levels.is_canonical(pruned.output), "a contraction-only plan certified an unpruned complement");
}

#[test]
fn a_leaf_root_complement_is_certified() {
    let eng = &Engine::new();
    let vtree = Arc::new(Vtree::balanced(1));
    let x = crate::literal(&vtree, 1).unwrap();
    let negated = eng.negate_with(x, ReductionPlan::Prune).unwrap();
    assert_certified_minimal(eng, &negated, "leaf root");
    assert!(negated.equivalent(&crate::literal(&vtree, -1).unwrap()).unwrap());
}

/// The completion and root complement of a certified operand, which read
/// most levels off their pair counts, charge the work clock what they
/// charge a copy without its certificate, whose every level is marked in a
/// cover: a unit a pair, and a scan of the basis on a level short of a
/// cell and at the root. And the two build the same diagram.
#[test]
fn a_certified_complement_charges_what_a_cover_of_every_level_does() {
    let eng = &Engine::new();
    let mut rng = Lcg::new(0x5be0_cd19);
    let mut cases = 0usize;
    for num_vars in [3u32, 5, 8, 12] {
        for (name, vtree) in vtree_shapes(num_vars) {
            for round in 0..24 {
                let shape = CnfShape { clauses: 1 + round % 9, width: 1 + round % 4 };
                let f = compile_clauses_on(eng, &vtree, &rand_cnf(&mut rng, num_vars, shape));
                if f.is_zero() {
                    continue;
                }
                let what = format!("{name}, {num_vars} variables, round {round}");
                assert!(f.levels.is_canonical(f.output), "{what}: the operand is not certified");
                let mut bare = f.clone();
                bare.levels.forget();
                let start = eng.limits().work_units();
                let (certified, _) = crate::apply::negate::complement_on(eng, f.into()).unwrap();
                let read = eng.limits().work_units() - start;
                let start = eng.limits().work_units();
                let (covered, _) = crate::apply::negate::complement_on(eng, bare.into()).unwrap();
                assert_eq!(eng.limits().work_units() - start, read, "{what}: the two completions charged different work");
                assert_same_shape(&certified, &covered, &what);
                cases += 1;
            }
        }
    }
    assert!(cases > 300, "expected a corpus, got {cases} cases");
}
