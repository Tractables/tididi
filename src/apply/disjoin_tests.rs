use std::sync::Arc;
use crate::build::{clause_to_tdd, constant_zero};
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
