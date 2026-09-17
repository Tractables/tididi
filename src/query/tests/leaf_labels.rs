use super::*;
use crate::limits::{LimitConfig, StopAt, StopRules};
use crate::query::boolean::visit_leaf_labels;
use crate::test_helpers::assert_canonical;
use crate::OperationError;

#[test]
fn support_and_backbone_match_sparse_id_truth_tables() {
    let eng = Engine::new();
    let vars = [VarId(20), VarId(3), VarId(9)];
    for tree in [Vtree::balanced_over(&vars).unwrap(), Vtree::linear_from_order(&vars).unwrap()] {
        let tree = Arc::new(tree);
        for bits in 0..256u16 {
            let mut f = Tdd::one(&tree);
            for row in 0..8 {
                if bits & (1 << row) == 0 {
                    let clause = eng.clause(&tree, vars.iter().enumerate().map(|(i, &var)| {
                        Literal::new(var, row & (1 << i) == 0)
                    })).unwrap();
                    f = eng.and(f, clause).unwrap();
                }
            }
            let checked_backbone = eng.implied_literals(&f).unwrap();
            f.minimize().unwrap();
            assert_canonical(&f);
            assert_eq!(f.model_count().unwrap(), bits.count_ones().into());
            let mut expected_support = Vec::new();
            let mut expected_backbone = Vec::new();
            for (i, &var) in vars.iter().enumerate() {
                if (0..8).any(|row| ((bits >> row) & 1) != ((bits >> (row ^ (1 << i))) & 1)) {
                    expected_support.push(var);
                }
                if bits != 0 {
                    for positive in [false, true] {
                        if (0..8).filter(|row| bits & (1 << row) != 0)
                            .all(|row| (row & (1 << i) != 0) == positive) {
                            expected_backbone.push(Literal::new(var, positive));
                        }
                    }
                }
            }
            expected_support.sort_unstable();
            expected_backbone.sort_unstable_by_key(|literal| literal.var);
            assert_eq!(eng.support(&f).unwrap(), expected_support, "truth table {bits}");
            assert_eq!(f.implied_literals().unwrap(), expected_backbone, "truth table {bits}");
            assert_eq!(checked_backbone, expected_backbone, "truth table {bits}");
        }
    }
}

#[test]
fn leaf_outputs_have_only_their_forced_literal() {
    let eng = Engine::new();
    let var = VarId(20);
    let tree = Arc::new(Vtree::leaf(var));
    for (f, expected) in [
        (Tdd::zero(&tree), vec![]),
        (Tdd::one(&tree), vec![]),
        (eng.literal(&tree, Literal::pos(var)).unwrap(), vec![Literal::pos(var)]),
        (eng.literal(&tree, Literal::neg(var)).unwrap(), vec![Literal::neg(var)]),
    ] {
        assert_canonical(&f);
        assert_eq!(f.implied_literals().unwrap(), expected);
        assert_eq!(eng.implied_literals(&f).unwrap(), expected);
        assert_eq!(eng.support(&f).unwrap(), expected.iter().map(|literal| literal.var).collect::<Vec<_>>());
    }
}

#[test]
fn leaf_scan_omits_marginal_leaves_and_public_query_rejects_them() {
    let eng = Engine::new();
    let vars = [VarId(20), VarId(3), VarId(9)];
    let tree = Arc::new(Vtree::balanced_over(&vars).unwrap());
    let mut f = eng.cube(&tree, [Literal::pos(vars[0]), Literal::neg(vars[1])]).unwrap();
    assert_canonical(&f);
    let leaf = tree.leaf_of(vars[0]).unwrap();
    eng.marginalize_levels(&mut f, &[leaf]).unwrap();
    f.minimize().unwrap();
    assert_canonical(&f);
    let mut literals = Vec::new();
    visit_leaf_labels(&f, |_| Ok(()), |var, labels| {
        literals.extend(labels.implied(var));
        Ok(())
    }).unwrap();
    assert_eq!(literals, vec![Literal::neg(vars[1])]);
    assert!(matches!(f.implied_literals(), Err(OperationError::MarginalLevel(_))));

    let tree = Arc::new(Vtree::leaf(vars[0]));
    let mut f = eng.literal(&tree, Literal::pos(vars[0])).unwrap();
    eng.marginalize_levels(&mut f, &[tree.root()]).unwrap();
    assert_canonical(&f);
    let mut visited = 0;
    visit_leaf_labels(&f, |_| Ok(()), |_, _| { visited += 1; Ok(()) }).unwrap();
    assert_eq!(visited, 0);
    assert!(matches!(f.implied_literals(), Err(OperationError::MarginalLevel(_))));
}

#[test]
fn leaf_scan_stops_before_finishing_a_parent_and_can_retry() {
    let eng = Engine::new();
    let tree = Arc::new(Vtree::balanced(2));
    let f = eng.clause(&tree, [1, 2]).unwrap();
    assert_canonical(&f);
    assert!(f.level(tree.root()).pair_count_at(0) > 1);
    let lim = eng.limits();
    let before = lim.work_units();
    let mut visited = 0;
    {
        let _limit = lim.scope(LimitConfig::none().with_stop_rules(StopRules {
            unconditional: Some(StopAt::WorkUnits(before + 2)), after_pairs: None,
        }));
        let _op = lim.begin_operation();
        let mut gate = eng.limits().gate_with(1);
        let result = visit_leaf_labels(&f, |work| gate.poll(work), |_, _| {
            visited += 1;
            Ok(())
        });
        assert_eq!(result, Err(OperationError::Stopped));
        assert_eq!(lim.work_units() - before, 2);
        assert_eq!(visited, 0, "the stop must precede completion of the parent's summaries");
    }
    assert_eq!(eng.support(&f).unwrap(), vec![VarId(1), VarId(2)]);
    assert!(f.implied_literals().unwrap().is_empty());
    assert_canonical(&f);
}

#[test]
fn checked_backbone_preserves_inputs_across_resource_refusals() {
    use crate::limits::{StopCallback, StopDecision};
    let engine = Engine::new();
    let tree = Arc::new(Vtree::balanced(3));
    let f = engine.cube(&tree, [1, -2]).unwrap();
    assert_canonical(&f);
    {
        let _scope = engine.limits().scope(LimitConfig::none().with_memory_budget_bytes(Some(0)));
        assert_eq!(engine.implied_literals(&f), Err(OperationError::OverBudget));
    }
    let zero = engine.zero(&tree);
    assert_canonical(&zero);
    {
        let _scope = engine.limits().scope(LimitConfig::none().with_stop_callback(Some(
            StopCallback::new(|_, _| StopDecision::Stop))));
        assert_eq!(engine.implied_literals(&f), Err(OperationError::Stopped));
        assert_eq!(engine.implied_literals(&zero), Err(OperationError::Stopped));
    }
    assert_eq!(engine.implied_literals(&f).unwrap(), vec![1.try_into().unwrap(), (-2).try_into().unwrap()]);
    assert_eq!(engine.model_count(&f).unwrap(), 2u32.into());
    assert_canonical(&f);
}
