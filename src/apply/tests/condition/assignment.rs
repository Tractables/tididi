use super::*;
use crate::test_helpers::{assert_canonical, eval};
use crate::vtree::Vtree;

#[test]
fn mixed_assignments_match_the_truth_table_and_sequential_conditioning() {
    for n in 2..=7 {
        let tree = Arc::new(Vtree::balanced(n));
        let f = Tdd::clause(&tree, [1, -2]).unwrap() & Tdd::clause(&tree, [-1, n as i32]).unwrap();
        assert_canonical(&f);
        let eng = Engine::new();
        for positive in [false, true] {
            let assignment = [crate::diagram::Literal::new(VarId(1), positive), crate::diagram::Literal::new(VarId(n), !positive)];
            let mixed = eng.condition(f.clone(), assignment).unwrap();
            let first = eng.condition_var(f.clone(), VarId(1), positive).unwrap();
            let sequential = eng.condition_var(first, VarId(n), !positive).unwrap();
            assert_canonical(&mixed);
            assert_canonical(&sequential);
            assert_eq!(mixed.model_count().unwrap(), sequential.model_count().unwrap());
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
    let f = Tdd::clause(&tree, [1, 2]).unwrap();
    assert_canonical(&f);
    let repeated = eng.condition(f.clone(), [1, 1, -2, -2]).unwrap();
    assert_canonical(&repeated);
    assert_eq!(repeated.model_count().unwrap(), 4u32.into());
    let contradiction = eng.condition(f.clone(), [1, -1]).unwrap();
    assert!(contradiction.is_zero());
    assert_canonical(&contradiction);
    assert!(matches!(eng.condition(f, [1, -1, 3]), Err(OperationError::VariableNotInVtree(VarId(3)))));
    let _limited = eng.limits().scope(crate::limits::LimitConfig::none().with_memory_budget_bytes(Some(0)));
    assert!(matches!(eng.condition(repeated, [1]), Err(OperationError::OverBudget)));
}

#[test]
fn conditioning_a_weighted_leaf_preserves_the_weight_configuration() {
    use crate::diagram::{Arithmetic, RationalWeights, WeightStore};
    let tree = Arc::new(Vtree::balanced(1));
    let eng = Engine::new();
    for assignment in [vec![1], vec![-1], vec![1, -1]] {
        let mut f = Tdd::clause(&tree, [1]).unwrap();
        f.set_weights(WeightStore::new(RationalWeights::unit(1), Arithmetic::ExactRational)).unwrap();
        assert_canonical(&f);
        let result = eng.condition(f, assignment.clone()).unwrap();
        assert_canonical(&result);
        assert!(result.weights().is_some());
        let value = result.weighted_value().unwrap().unwrap();
        let expected = if assignment == [1] { 2 } else { 0 };
        assert_eq!(value.as_rational().into_owned(), num_rational::BigRational::from_integer(expected.into()));
    }
}

#[test]
fn a_false_cofactor_with_a_weighted_sibling_keeps_its_store_and_no_false_nodes() {
    use crate::diagram::{Arithmetic, RationalWeights, WeightStore};
    let tree = Arc::new(Vtree::balanced(4));
    let eng = Engine::new();
    let mut f = Tdd::clause(&tree, [1]).unwrap();
    f.set_weights(WeightStore::new(RationalWeights::unit(4), Arithmetic::ExactRational)).unwrap();
    let (_, right) = tree.children(tree.root());
    eng.marginalize_levels(&mut f, &[right]).unwrap();
    assert_canonical(&f);
    let result = eng.condition(f, [-1]).unwrap();
    assert!(result.is_zero());
    assert!(result.weights().is_some());
    assert_canonical(&result);
}

#[test]
fn sparse_assignments_on_rotated_trees_match_enumeration() {
    use crate::diagram::Literal;
    use crate::vtree::{rotate::rotate_pointers, RotationKind};
    let vars = [VarId(20), VarId(3), VarId(72), VarId(9)];
    let mut rotated = Vtree::balanced_over(&vars).unwrap();
    let root = rotated.root();
    rotate_pointers(&mut rotated, root, RotationKind::Left).unwrap().commit(&mut rotated);
    for tree in [Vtree::balanced_over(&vars).unwrap(), Vtree::linear_from_order(&vars).unwrap(), rotated] {
        let tree = Arc::new(tree);
        let eng = Engine::new();
        let mut f = eng.clause(&tree, [Literal::pos(vars[0]), Literal::neg(vars[1])]).unwrap();
        let g = eng.clause(&tree, [Literal::pos(vars[1]), Literal::pos(vars[2]), Literal::neg(vars[3])]).unwrap();
        assert_canonical(&f);
        assert_canonical(&g);
        f = eng.and(f, g).unwrap();
        assert_canonical(&f);
        for code in 0..81 {
            let mut digits = code;
            let assignment: Vec<_> = vars.iter().filter_map(|&var| {
                let state = digits % 3;
                digits /= 3;
                (state != 0).then(|| Literal::new(var, state == 2))
            }).collect();
            let result = eng.condition(f.clone(), assignment.iter().rev().copied()).unwrap();
            assert_canonical(&result);
            let mut count = 0u32;
            for bits in 0..16 {
                let mut values = vec![false; 72];
                for (i, &var) in vars.iter().enumerate() { values[var.idx()] = bits & (1 << i) != 0; }
                let mut fixed = values.clone();
                for literal in &assignment { fixed[literal.var.idx()] = literal.sign; }
                let expected = (fixed[19] || !fixed[2]) && (fixed[2] || fixed[71] || !fixed[8]);
                assert_eq!(eval(&result, &values), expected, "assignment {code}, row {bits}");
                count += u32::from(expected);
            }
            assert_eq!(result.model_count().unwrap(), count.into());
        }
    }
}

#[test]
fn falsity_cascades_preserve_marginal_sibling_values_on_either_side() {
    use crate::diagram::{Arithmetic, RationalWeights, WeightStore};
    let eng = Engine::new();
    let tree = Arc::new(Vtree::balanced(8));
    for weighted in [false, true] {
        for target_right in [false, true] {
            let (left, right) = tree.children(tree.root());
            let target = if target_right { right } else { left };
            let sibling = if target_right { left } else { right };
            let offset = if target_right { 4 } else { 0 };
            let mut f = eng.and(Tdd::clause(&tree, [1, 2]).unwrap(), Tdd::clause(&tree, [5, 6]).unwrap()).unwrap();
            assert_canonical(&f);
            if weighted {
                f.set_weights(WeightStore::new(RationalWeights::unit(8), Arithmetic::ExactRational)).unwrap();
            }
            eng.marginalize_levels(&mut f, &[sibling]).unwrap();
            assert_canonical(&f);
            assert!(!f.level(target).is_marginal());
            for all_false in [false, true] {
                let mut assignment = vec![-(offset + 1)];
                if all_false { assignment.push(-(offset + 2)); }
                let result = eng.condition(f.clone(), assignment).unwrap();
                assert_canonical(&result);
                assert_eq!(result.is_zero(), all_false);
                let expected = if all_false { 0u32 } else { 96 };
                if weighted {
                    assert_eq!(eng.weighted_value(&result).unwrap().unwrap().as_rational().into_owned(),
                        num_rational::BigRational::from_integer(expected.into()));
                } else {
                    assert_eq!(eng.model_count(&result).unwrap(), expected.into());
                }
            }
        }
    }
}

#[test]
fn conditioning_recovers_from_each_reservation_refusal() {
    use crate::diagram::{Arithmetic, RationalWeights, WeightStore};

    let vtree = Arc::new(Vtree::balanced(8));
    let setup = Engine::new();
    for marginal in [None, Some(false), Some(true)] {
        let mut f = setup.and(Tdd::clause(&vtree, [1, 5]).unwrap(), Tdd::clause(&vtree, [8]).unwrap()).unwrap();
        assert_canonical(&f);
        if let Some(weighted) = marginal {
            if weighted {
                f.set_weights(WeightStore::new(RationalWeights::unit(8), Arithmetic::ExactRational)).unwrap();
            }
            setup.marginalize_levels(&mut f, &[vtree.children(vtree.root()).0]).unwrap();
            assert_canonical(&f);
        }
        for last in [-8, 8] {
            let mut completed = false;
            let mut refusals = 0;
            for cut in 0..256 {
                let engine = Engine::new();
                engine.limits().refuse_nth_reserve(cut);
                let result = engine.condition(f.clone(), [-5, last]);
                engine.limits().grant_every_reserve();
                let g = match result {
                    Ok(g) => { completed = true; g }
                    Err(error) => {
                        assert_eq!(error, OperationError::OverBudget);
                        refusals += 1;
                        engine.condition(f.clone(), [-5, last]).unwrap()
                    }
                };
                assert_canonical(&g);
                assert_eq!(g.is_zero(), last < 0);
                let expected = if last < 0 { 0u32 } else { 128 };
                if marginal == Some(true) {
                    assert_eq!(g.weighted_value().unwrap().unwrap().as_rational().into_owned(),
                        num_rational::BigRational::from_integer(expected.into()));
                } else {
                    assert_eq!(g.model_count().unwrap(), expected.into());
                }
                if completed { break; }
            }
            assert!(completed && refusals > 0, "must refuse and then complete every variant");
            assert_canonical(&f);
        }
    }
}
