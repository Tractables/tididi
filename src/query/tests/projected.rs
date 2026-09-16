use super::*;
use crate::limits::{LimitConfig, OperationError, StopAt, StopRules};
use crate::test_helpers::{assert_canonical, stopping_engine};

#[test]
fn projected_counts_match_all_three_variable_truth_tables() {
    let vars = [VarId(19), VarId(2), VarId(8)];
    for vtree in [Vtree::balanced_over(&vars), Vtree::linear_from_order(&vars)] {
        let vtree = Arc::new(vtree);
        for bits in 0..256u16 {
            let mut f = Tdd::zero(&vtree);
            for row in 0..8 {
                if bits & (1 << row) != 0 {
                    let cube = Tdd::cube(&vtree, vars.iter().enumerate().map(|(i, &var)| {
                        Literal::new(var, row & (1 << i) != 0)
                    })).unwrap();
                    assert_canonical(&cube);
                    f = crate::or(f, cube).unwrap();
                }
            }
            assert_canonical(&f);
            for mask in 0..8 {
                let mut selected: Vec<_> = vars.iter().enumerate()
                    .filter_map(|(i, &var)| (mask & (1 << i) != 0).then_some(var)).collect();
                let expected: std::collections::BTreeSet<_> = (0..8)
                    .filter(|row| bits & (1 << row) != 0)
                    .map(|row| row & mask).collect();
                assert_eq!(f.projected_model_count(&selected).unwrap(), expected.len().into(),
                    "truth table {bits}, selection {mask}");
                selected.reverse();
                selected.extend(selected.clone());
                let count = vtree.context().run(|engine| engine.projected_model_count(&f, &selected)).unwrap();
                assert_eq!(count, expected.len().into(), "repeated/reordered selection");
            }
            assert_eq!(f.model_count().unwrap(), bits.count_ones().into());
            assert_canonical(&f);
        }
    }
}

#[test]
fn projected_counts_cover_leaf_outputs_and_large_free_domains() {
    let vtree = Arc::new(Vtree::leaf(VarId(71)));
    for f in [Tdd::zero(&vtree), Tdd::one(&vtree), crate::literal(&vtree, 72).unwrap()] {
        assert_canonical(&f);
        assert_eq!(f.projected_model_count(&[VarId(71)]).unwrap(), f.model_count().unwrap());
        assert_eq!(f.projected_model_count(&[]).unwrap(), u32::from(f.is_sat().unwrap()).into());
        assert_eq!(f.projected_model_count(&[VarId(71), VarId(0)]),
            Err(OperationError::VariableNotInVtree(VarId(0))));
    }
    let vtree = Arc::new(Vtree::balanced(132));
    let f = crate::literal(&vtree, 132).unwrap();
    assert_canonical(&f);
    let selected: Vec<_> = (0..130).map(VarId).collect();
    assert_eq!(f.projected_model_count(&selected).unwrap(), BigUint::from(1u32) << 130usize);
}

#[test]
fn projected_counts_ignore_weights_and_reject_marginal_structure() {
    use crate::diagram::{Arithmetic, LiteralWeights, RationalWeights, WeightStore};
    use crate::test_helpers::rat;
    let vtree = Arc::new(Vtree::balanced(3));
    let mut f = Tdd::clause(&vtree, [1, 2]).unwrap();
    let weights = vec![LiteralWeights { negative: rat(0, 1), positive: rat(0, 1) }; 3];
    f.set_weights(WeightStore::new(RationalWeights::from_literals(&weights), Arithmetic::ExactRational)).unwrap();
    assert_canonical(&f);
    assert_eq!(f.projected_model_count(&[VarId(0)]).unwrap(), 2u32.into());
    assert_eq!(f.model_count().unwrap(), 6u32.into());
    assert_eq!(f.weighted_value().unwrap().unwrap().into_rational(), rat(0, 1));
    f.marginalize_levels(&[vtree.root()]).unwrap();
    assert_canonical(&f);
    for selected in [vec![], vec![VarId(0)], vec![VarId(0), VarId(1), VarId(2)]] {
        assert!(matches!(f.projected_model_count(&selected), Err(OperationError::MarginalLevel(_))));
    }
    let mut f = Tdd::clause(&vtree, [1, 2]).unwrap();
    f.marginalize_levels(&[vtree.root()]).unwrap();
    assert_canonical(&f);
    assert!(matches!(f.projected_model_count(&[VarId(0)]), Err(OperationError::MarginalLevel(_))));
}

#[test]
fn projected_counts_preserve_inputs_and_recover_after_refusals() {
    let vtree = Arc::new(Vtree::balanced(3));
    let f = Tdd::clause(&vtree, [1, 2]).unwrap();
    assert_canonical(&f);
    let engine = Engine::new();
    {
        let _scope = engine.limits().scope(LimitConfig::none().with_memory_budget_bytes(Some(0)));
        assert_eq!(engine.projected_model_count(&f, &[VarId(0)]), Err(OperationError::OverBudget));
    }
    let stopped = stopping_engine();
    for selected in [vec![], vec![VarId(0)], vec![VarId(0), VarId(1), VarId(2)]] {
        assert_eq!(stopped.projected_model_count(&f, &selected), Err(OperationError::Stopped));
    }
    {
        // Stop after selection validation, during structural validation.
        let before = engine.limits().work_units();
        let _scope = engine.limits().scope(LimitConfig::none().with_stop_rules(StopRules {
            unconditional: Some(StopAt::WorkUnits(before + 2)), after_pairs: None,
        }));
        assert_eq!(engine.projected_model_count(&f, &[VarId(0)]), Err(OperationError::Stopped));
    }
    assert_eq!(engine.projected_model_count(&f, &[VarId(0)]).unwrap(), 2u32.into());
    assert_eq!(f.model_count().unwrap(), 6u32.into());
    assert!(f.equivalent(&Tdd::clause(&vtree, [1, 2]).unwrap()).unwrap());
    assert_canonical(&f);
}

#[test]
fn projected_counts_recover_from_each_allocation_failure() {
    let vtree = Arc::new(Vtree::balanced(4));
    let left = Tdd::clause(&vtree, [1, 3]).unwrap();
    let right = Tdd::clause(&vtree, [2, 4]).unwrap();
    assert_canonical(&left);
    assert_canonical(&right);
    let f = crate::and(left, right).unwrap();
    assert_canonical(&f);
    let selected = [VarId(0), VarId(1)];
    let mut completed = false;
    for cut in 0..512 {
        let engine = Engine::new();
        engine.limits().refuse_nth_reserve(cut);
        let result = engine.projected_model_count(&f, &selected);
        engine.limits().grant_every_reserve();
        assert_canonical(&f);
        assert_eq!(engine.projected_model_count(&f, &selected).unwrap(), 4u32.into());
        match result {
            Ok(count) => {
                assert_eq!(count, 4u32.into());
                assert!(cut > 10, "exercise the compound query beyond its selection buffer");
                completed = true;
                break;
            }
            Err(error) => assert_eq!(error, OperationError::OverBudget, "reserve {cut}"),
        }
    }
    assert!(completed, "reach the end of the query's allocations");
    assert_eq!(f.model_count().unwrap(), 9u32.into());
}
