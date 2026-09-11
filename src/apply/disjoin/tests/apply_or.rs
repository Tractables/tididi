use std::sync::Arc;
use crate::apply::conjoin_clause::clause_to_tdd;
use crate::build::constant_zero;
use crate::apply::apply_and;
use crate::reduce::minimize;
use crate::query::model_count;
use crate::vtree::Vtree;
use num_bigint::BigUint;

fn balanced_vtree(n: u32) -> Arc<Vtree> {
    Arc::new(Vtree::balanced(n))
}

#[test]
fn test_apply_or_basic() {
    let eng = &crate::engine::Engine::new();
    let vtree = balanced_vtree(4);
    let mut f = clause_to_tdd(eng, &vtree, &crate::test_helpers::clause(&[(0, true), (1, true)]));
    let mut g = clause_to_tdd(eng, &vtree, &crate::test_helpers::clause(&[(2, true), (3, true)]));
    minimize(&mut f);
    minimize(&mut g);

    let result = super::apply_or(f.clone(), g.clone());
    assert_eq!(model_count(&result), BigUint::from(15u32));
}

#[test]
fn test_apply_or_with_zero() {
    let eng = &crate::engine::Engine::new();
    let vtree = balanced_vtree(4);
    let mut f = clause_to_tdd(eng, &vtree, &crate::test_helpers::clause(&[(0, true)]));
    minimize(&mut f);
    let zero = constant_zero(eng, &vtree);

    let result = super::apply_or(f.clone(), zero.clone());
    assert_eq!(model_count(&result), model_count(&f));
}

#[test]
fn test_apply_or_canonical() {
    let eng = &crate::engine::Engine::new();
    use crate::check::check_all_fast;

    let vtree = balanced_vtree(4);
    let mut f = clause_to_tdd(eng, &vtree, &crate::test_helpers::clause(&[(0, true), (1, true)]));
    let mut g = clause_to_tdd(eng, &vtree, &crate::test_helpers::clause(&[(2, true), (3, true)]));
    minimize(&mut f);
    minimize(&mut g);

    let result = super::apply_or(f.clone(), g.clone());
    check_all_fast(&result, "apply_or result");
}

#[test]
fn test_apply_or_compiled_formulas() {
    let eng = &crate::engine::Engine::new();
    let vtree = balanced_vtree(4);

    let f = clause_to_tdd(eng, &vtree, &crate::test_helpers::clause(&[(0, true), (1, true)]));
    let g = clause_to_tdd(eng, &vtree, &crate::test_helpers::clause(&[(2, true), (3, true)]));
    let mut f = apply_and(f, g);
    minimize(&mut f);

    let c3 = clause_to_tdd(eng, &vtree, &crate::test_helpers::clause(&[(0, true)]));
    let c4 = clause_to_tdd(eng, &vtree, &crate::test_helpers::clause(&[(2, true)]));
    let mut g = apply_and(c3, c4);
    minimize(&mut g);

    let count_f = model_count(&f);
    let count_g = model_count(&g);

    let f_clone = f.clone();
    let g_clone = g.clone();
    let mut f_and_g = apply_and(f_clone, g_clone);
    minimize(&mut f_and_g);
    let count_f_and_g = model_count(&f_and_g);

    let expected = &count_f + &count_g - &count_f_and_g;

    let result = super::apply_or(f.clone(), g.clone());
    assert_eq!(model_count(&result), expected,
        "apply_or count {} != inclusion-exclusion {}",
        model_count(&result), expected);
}

/// Indices the sweep below arms the allocation-failure injection at, past the
/// last reserve the disjunction takes; the sweep asserts that its tail was
/// granted, so every reserve was refused once.
const RESERVES_PER_DISJUNCTION: u32 = 96;

/// The two minimizations inside the disjunction reserve through the caller's
/// engine: a refusal anywhere in `Engine::or` comes back as `Err(OverBudget)`,
/// a granted run answers the same count, and the engine stays usable.
#[test]
fn a_refused_reserve_inside_the_disjunction_returns_over_budget() {
    use crate::apply::conjoin::conjoin_owned;
    use crate::apply::negate::negate_tdd_owned;
    use crate::engine::Engine;
    use crate::limits::ApplyError;

    let eng = &Engine::new();
    let vtree = balanced_vtree(4);
    let mut f = clause_to_tdd(eng, &vtree, &crate::test_helpers::clause(&[(0, true), (1, true)]));
    let mut g = clause_to_tdd(eng, &vtree, &crate::test_helpers::clause(&[(2, true), (3, false)]));
    minimize(&mut f);
    minimize(&mut g);
    let expected = model_count(&super::apply_or(f.clone(), g.clone()));

    let mut refused_or = 0;
    for nth in 0..RESERVES_PER_DISJUNCTION {
        eng.limits().refuse_nth_reserve(nth);
        let res = eng.or(f.clone(), g.clone());
        eng.limits().grant_every_reserve();
        match res {
            Ok(h) => assert_eq!(model_count(&h), expected, "a granted run at reserve {nth}"),
            Err(e) => {
                assert_eq!(e, ApplyError::OverBudget, "refusal at reserve {nth}");
                refused_or += 1;
            }
        }
    }
    assert!(refused_or > 0, "the sweep must actually refuse something");
    assert!(refused_or < RESERVES_PER_DISJUNCTION, "the sweep must run past the last reserve");

    // The conjunction between the negated operands alone takes fewer reserves
    // than the whole disjunction, because the two minimizations that follow it
    // reserve through the same engine.
    let mut refused_and = 0;
    for nth in 0..RESERVES_PER_DISJUNCTION {
        eng.limits().refuse_nth_reserve(nth);
        let res = conjoin_owned(eng, negate_tdd_owned(f.clone()), negate_tdd_owned(g.clone()), None);
        eng.limits().grant_every_reserve();
        if res.is_err() {
            refused_and += 1;
        }
    }
    assert!(
        refused_or > refused_and,
        "the disjunction was refused {refused_or} times and its conjunction alone {refused_and}",
    );

    let h = eng.or(f, g).expect("nothing is armed any more");
    assert_eq!(model_count(&h), expected);
}
