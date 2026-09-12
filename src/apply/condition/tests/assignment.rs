use super::*;
use crate::test_helpers::{assert_canonical, eval};
use crate::vtree::Vtree;

#[test]
fn mixed_assignments_match_the_truth_table_and_sequential_conditioning() {
    for n in 2..=7 {
        let tree = Arc::new(Vtree::balanced(n));
        let f = Tdd::clause(&tree, [1, -2]) & Tdd::clause(&tree, [-1, n as i32]);
        assert_canonical(&f);
        let eng = Engine::new();
        for positive in [false, true] {
            let assignment = [crate::diagram::Literal::new(VarId(0), positive), crate::diagram::Literal::new(VarId(n - 1), !positive)];
            let mixed = eng.condition(f.clone(), assignment).unwrap();
            let first = eng.condition_var(f.clone(), VarId(0), positive).unwrap();
            let sequential = eng.condition_var(first, VarId(n - 1), !positive).unwrap();
            assert_canonical(&mixed);
            assert_canonical(&sequential);
            assert_eq!(mixed.model_count(), sequential.model_count());
            for bits in 0..1u32 << n {
                let values: Vec<_> = (0..n).map(|i| bits & (1 << i) != 0).collect();
                let mut fixed = values.clone();
                fixed[0] = positive;
                fixed[n as usize - 1] = !positive;
                assert_eq!(eval(&mixed, &values), eval(&f, &fixed));
            }
        }
    }
}

#[test]
fn assignment_duplicates_and_errors_are_decided_before_rewriting() {
    let tree = Arc::new(Vtree::balanced(2));
    let eng = Engine::new();
    let f = Tdd::clause(&tree, [1, 2]);
    assert_canonical(&f);
    let repeated = eng.condition(f.clone(), [1, 1, -2, -2]).unwrap();
    assert_canonical(&repeated);
    assert_eq!(repeated.model_count(), 4u32.into());
    let contradiction = eng.condition(f.clone(), [1, -1]).unwrap();
    assert!(contradiction.is_zero());
    assert_canonical(&contradiction);
    assert!(matches!(eng.condition(f, [1, -1, 3]), Err(ApplyError::VariableNotInVtree(VarId(2)))));
    let _limited = eng.limits().scope(crate::limits::LimitSet::none().budget(Some(0)));
    assert!(matches!(eng.condition(repeated, [1]), Err(ApplyError::OverBudget)));
}

#[test]
fn conditioning_a_weighted_leaf_preserves_the_weight_configuration() {
    use crate::diagram::{Arithmetic, RationalWeights, WeightStore};
    let tree = Arc::new(Vtree::balanced(1));
    let eng = Engine::new();
    for assignment in [vec![1], vec![-1], vec![1, -1]] {
        let mut f = Tdd::clause(&tree, [1]);
        f.set_weights(WeightStore::new(RationalWeights::unit(1), Arithmetic::ExactRational)).unwrap();
        assert_canonical(&f);
        let result = eng.condition(f, assignment.clone()).unwrap();
        assert_canonical(&result);
        assert!(result.weights().is_some());
        let value = crate::query::weighted_value(&result).unwrap();
        let expected = if assignment == [1] { 2 } else { 0 };
        assert_eq!(value.as_rational().into_owned(), num_rational::BigRational::from_integer(expected.into()));
    }
}

#[test]
fn a_false_cofactor_with_a_weighted_sibling_keeps_its_store_and_no_false_nodes() {
    use crate::diagram::{Arithmetic, RationalWeights, WeightStore};
    let tree = Arc::new(Vtree::balanced(4));
    let eng = Engine::new();
    let mut f = Tdd::clause(&tree, [1]);
    f.set_weights(WeightStore::new(RationalWeights::unit(4), Arithmetic::ExactRational)).unwrap();
    let (_, right) = tree.children(tree.root());
    crate::marginal::marginalize(&eng, &mut f, &[right]).unwrap();
    assert_canonical(&f);
    let result = eng.condition(f, [-1]).unwrap();
    assert!(result.is_zero());
    assert!(result.weights().is_some());
    assert_canonical(&result);
}
