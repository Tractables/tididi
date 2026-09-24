use std::sync::Arc;
use crate::test_helpers::clause_to_tdd;
use crate::build::constant_zero;
use crate::apply::apply_and;

use crate::vtree::Vtree;
use num_bigint::BigUint;

#[test]
fn test_apply_or_basic() {
    let vtree = Arc::new(Vtree::balanced(4));
    let mut f = clause_to_tdd(&vtree, &crate::test_helpers::clause(&[(1, true), (2, true)]));
    let mut g = clause_to_tdd(&vtree, &crate::test_helpers::clause(&[(3, true), (4, true)]));
    f.minimize().unwrap();
    g.minimize().unwrap();

    let result = super::apply_or(f.clone(), g.clone());
    assert_eq!(result.model_count().unwrap(), BigUint::from(15u32));
}

#[test]
fn test_apply_or_with_zero() {
    let eng = &crate::Engine::new();
    let vtree = Arc::new(Vtree::balanced(4));
    let mut f = clause_to_tdd(&vtree, &crate::test_helpers::clause(&[(1, true)]));
    f.minimize().unwrap();
    let zero = constant_zero(eng, &vtree);

    let result = super::apply_or(f.clone(), zero.clone());
    assert_eq!(result.model_count().unwrap(), f.model_count().unwrap());
}

#[test]
fn test_apply_or_canonical() {
    use crate::test_helpers::check::check_all_fast;

    let vtree = Arc::new(Vtree::balanced(4));
    let mut f = clause_to_tdd(&vtree, &crate::test_helpers::clause(&[(1, true), (2, true)]));
    let mut g = clause_to_tdd(&vtree, &crate::test_helpers::clause(&[(3, true), (4, true)]));
    f.minimize().unwrap();
    g.minimize().unwrap();

    let result = super::apply_or(f.clone(), g.clone());
    check_all_fast(&result, "apply_or result");
}

#[test]
fn test_apply_or_compiled_formulas() {
    let vtree = Arc::new(Vtree::balanced(4));

    let f = clause_to_tdd(&vtree, &crate::test_helpers::clause(&[(1, true), (2, true)]));
    let g = clause_to_tdd(&vtree, &crate::test_helpers::clause(&[(3, true), (4, true)]));
    let mut f = apply_and(f, g);
    f.minimize().unwrap();

    let c3 = clause_to_tdd(&vtree, &crate::test_helpers::clause(&[(1, true)]));
    let c4 = clause_to_tdd(&vtree, &crate::test_helpers::clause(&[(3, true)]));
    let mut g = apply_and(c3, c4);
    g.minimize().unwrap();

    let count_f = f.model_count().unwrap();
    let count_g = g.model_count().unwrap();

    let f_clone = f.clone();
    let g_clone = g.clone();
    let mut f_and_g = apply_and(f_clone, g_clone);
    f_and_g.minimize().unwrap();
    let count_f_and_g = f_and_g.model_count().unwrap();

    let expected = &count_f + &count_g - &count_f_and_g;

    let result = super::apply_or(f.clone(), g.clone());
    assert_eq!(result.model_count().unwrap(), expected,
        "apply_or count {} != inclusion-exclusion {}",
        result.model_count().unwrap(), expected);
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
    use crate::apply::negate::negate_on;
    use crate::Engine;
    use crate::limits::OperationError;

    let eng = &Engine::new();
    let vtree = Arc::new(Vtree::balanced(4));
    let mut f = clause_to_tdd(&vtree, &crate::test_helpers::clause(&[(1, true), (2, true)]));
    let mut g = clause_to_tdd(&vtree, &crate::test_helpers::clause(&[(3, true), (4, false)]));
    f.minimize().unwrap();
    g.minimize().unwrap();
    let expected = (super::apply_or(f.clone(), g.clone())).model_count().unwrap();

    let mut refused_or = 0;
    for nth in 0..RESERVES_PER_DISJUNCTION {
        eng.limits().refuse_nth_reserve(nth);
        let res = eng.or(f.clone(), g.clone());
        eng.limits().grant_every_reserve();
        match res {
            Ok(h) => assert_eq!(h.model_count().unwrap(), expected, "a granted run at reserve {nth}"),
            Err(e) => {
                assert_eq!(e, OperationError::OverBudget, "refusal at reserve {nth}");
                refused_or += 1;
            }
        }
    }
    assert!(refused_or > 0, "the sweep must actually refuse something");
    assert!(refused_or < RESERVES_PER_DISJUNCTION, "the sweep must run past the last reserve");

    // The conjunction between the negated operands alone takes fewer reserves
    // than the whole disjunction, because the two minimizations that follow it
    // reserve through the same engine.
    let not_f = negate_on(eng, f.clone()).unwrap();
    let not_g = negate_on(eng, g.clone()).unwrap();
    let mut refused_and = 0;
    for nth in 0..RESERVES_PER_DISJUNCTION {
        eng.limits().refuse_nth_reserve(nth);
        let res = eng.and(not_f.clone(), not_g.clone());
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
    assert_eq!(h.model_count().unwrap(), expected);
}
