//! A negation that borrows its operand builds what the negation of the same
//! operand owned builds: the same diagram, stored alike and known alike,
//! for the same work, whatever the plan; and it leaves the operand as it
//! was.

use std::borrow::Cow;
use std::sync::Arc;

use super::plan::shape;
use crate::diagram::{Arithmetic, LiteralWeights, RationalWeights, WeightStore};
use crate::reduce::{ContentTwinPolicy, ReductionPlan};
use crate::test_helpers::{compile_clauses_on, known, rand_cnf, rat, same_storage, vtree_shapes, with_floor, CnfShape, Lcg, LOW_FLOOR};
use crate::vtree::Vtree;
use crate::{Engine, Tdd};

/// The plans a negation ends with, by number.
fn plan(i: usize) -> ReductionPlan<'static> {
    match i {
        0 => ReductionPlan::Prune,
        1 => ReductionPlan::Contract,
        2 => ReductionPlan::Full(ContentTwinPolicy::Skip),
        _ => ReductionPlan::Full(ContentTwinPolicy::Fresh),
    }
}

/// `f` negated on a fresh engine under `plan`, with the work it took.
fn negated(f: Cow<'_, Tdd>, plan: ReductionPlan<'_>) -> (Tdd, u64) {
    let eng = Engine::new();
    let mark = eng.limits().mark();
    let out = eng.negate_with(f, plan).unwrap();
    (out, eng.limits().work_since(mark))
}

/// Check `f` negated borrowed against `f` negated owned under every plan.
fn check(what: &str, f: &Tdd) {
    let before = (f.clone(), known(f));
    for i in 0..4 {
        let name = format!("{what}, {:?}", plan(i));
        let (owned, owned_work) = negated(Cow::Owned(f.clone()), plan(i));
        let (borrowed, borrowed_work) = negated(Cow::Borrowed(f), plan(i));
        assert_eq!(shape(&borrowed), shape(&owned), "{name}: the diagrams differ");
        assert!(same_storage(&borrowed, &owned), "{name}: the storage differs");
        assert_eq!(known(&borrowed), known(&owned), "{name}: the results are known differently");
        assert_eq!(borrowed.weights.is_some(), owned.weights.is_some(), "{name}: the weights differ");
        assert_eq!(borrowed_work, owned_work, "{name}: the work differs");
        assert!(same_storage(f, &before.0) && known(f) == before.1, "{name}: the borrowed operand changed");
    }
}

#[test]
fn a_borrowed_negation_builds_the_owned_one() {
    let mut rng = Lcg::new(0x5eed_0b0e);
    let mut checked = 0;
    for nvars in [1u32, 3, 5, 8] {
        for (name, vtree) in vtree_shapes(nvars) {
            let eng = Engine::new();
            for case in 0..4 {
                let clauses = rand_cnf(&mut rng, nvars, CnfShape { clauses: 1 + case + nvars as usize, width: 3 });
                let mut f = compile_clauses_on(&eng, &vtree, &clauses);
                check(&format!("{name}, {nvars} variables, case {case}"), &f);
                eng.minimize(&mut f).unwrap();
                check(&format!("{name}, {nvars} variables, case {case}, minimized"), &f);
                checked += 2;
            }
            let all: Vec<i32> = (1..=nvars as i32).map(|v| if v % 2 == 0 { -v } else { v }).collect();
            check(&format!("{name}, {nvars} variables, cube"), &eng.cube(&vtree, all.iter().copied()).unwrap());
            check(&format!("{name}, {nvars} variables, clause"), &eng.clause(&vtree, all.iter().copied()).unwrap());
            check(&format!("{name}, {nvars} variables, true"), &crate::build::constant_one(&eng, &vtree));
            check(&format!("{name}, {nvars} variables, false"), &crate::build::constant_on(&eng, &vtree, false).unwrap());
            checked += 4;
        }
    }
    assert!(checked > 100, "{checked} diagrams checked");
}

#[test]
fn a_borrowed_negation_copies_described_levels_and_weights() {
    let mut rng = Lcg::new(0x5eed_0b0f);
    let mut described = 0;
    with_floor(LOW_FLOOR, || {
        for nvars in [6u32, 10] {
            for (name, vtree) in vtree_shapes(nvars) {
                let eng = Engine::new();
                for case in 0..3 {
                    let clauses = rand_cnf(&mut rng, nvars, CnfShape { clauses: nvars as usize + 2 * case, width: 3 });
                    let f = compile_clauses_on(&eng, &vtree, &clauses);
                    if f.levels.iter().any(|level| level.implicit().is_some()) {
                        check(&format!("{name}, {nvars} variables, case {case}, described"), &f);
                        described += 1;
                    }
                }
            }
        }
    });
    assert!(described > 0, "no diagram held a described level");

    let vtree = Arc::new(Vtree::balanced(4));
    let mut f = Tdd::clause(&vtree, [1, -2]).unwrap() & Tdd::clause(&vtree, [3, 4]).unwrap();
    let table = RationalWeights::from_literals(&vec![LiteralWeights { negative: rat(1, 3), positive: rat(2, 3) }; 4]);
    f.set_weights(WeightStore::new(table, Arithmetic::ExactRational)).unwrap();
    check("weighted", &f);
    let negated = Engine::new().negate(&f).unwrap();
    assert_eq!(
        negated.weighted_value().unwrap().unwrap().into_rational() + f.weighted_value().unwrap().unwrap().into_rational(),
        rat(1, 1),
        "a weighted negation's value is the rest of one",
    );
}
