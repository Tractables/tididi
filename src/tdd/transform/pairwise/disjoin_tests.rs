use std::sync::Arc;
use crate::vtree::Literal;
use crate::vtree::VarId;
use crate::tdd::build::{clause_to_tdd, constant_zero};
use crate::tdd::transform::pairwise::conjoin::apply_and;
use crate::tdd::minimize::minimize;
use crate::tdd::query::model_count;
use crate::vtree::Vtree;
use num_bigint::BigUint;

fn balanced_vtree(n: u32) -> Arc<Vtree> {
    Arc::new(Vtree::balanced(n))
}

fn clause(lits: &[(u32, bool)]) -> Vec<Literal> {
    lits.iter().map(|&(v, p)| Literal { var: VarId(v), positive: p }).collect()
}

#[test]
fn test_apply_or_basic() {
    let vtree = balanced_vtree(4);
    let mut f = clause_to_tdd(&vtree, &clause(&[(0, true), (1, true)]));
    let mut g = clause_to_tdd(&vtree, &clause(&[(2, true), (3, true)]));
    minimize(&mut f);
    minimize(&mut g);

    let result = super::apply_or(&f, &g);
    assert_eq!(model_count(&result), BigUint::from(15u32));
}

#[test]
fn test_apply_or_with_zero() {
    let vtree = balanced_vtree(4);
    let mut f = clause_to_tdd(&vtree, &clause(&[(0, true)]));
    minimize(&mut f);
    let zero = constant_zero(&vtree);

    let result = super::apply_or(&f, &zero);
    assert_eq!(model_count(&result), model_count(&f));
}

#[test]
fn test_apply_or_canonical() {
    use crate::tdd::validate::check_all_fast;

    let vtree = balanced_vtree(4);
    let mut f = clause_to_tdd(&vtree, &clause(&[(0, true), (1, true)]));
    let mut g = clause_to_tdd(&vtree, &clause(&[(2, true), (3, true)]));
    minimize(&mut f);
    minimize(&mut g);

    let result = super::apply_or(&f, &g);
    check_all_fast(&result, "apply_or result");
}

#[test]
fn test_apply_or_compiled_formulas() {
    let vtree = balanced_vtree(4);

    let c1 = clause_to_tdd(&vtree, &clause(&[(0, true), (1, true)]));
    let c2 = clause_to_tdd(&vtree, &clause(&[(2, true), (3, true)]));
    let mut f = apply_and(c1, c2);
    minimize(&mut f);

    let c3 = clause_to_tdd(&vtree, &clause(&[(0, true)]));
    let c4 = clause_to_tdd(&vtree, &clause(&[(2, true)]));
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

    let result = super::apply_or(&f, &g);
    assert_eq!(model_count(&result), expected,
        "apply_or count {} != inclusion-exclusion {}",
        model_count(&result), expected);
}
