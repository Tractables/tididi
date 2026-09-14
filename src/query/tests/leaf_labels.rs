use super::*;
use crate::limits::{LimitConfig, PollGate, StopAt, StopRules};
use crate::query::support::visit_leaf_labels;
use crate::test_helpers::assert_canonical;
use crate::OperationError;

#[test]
fn support_and_backbone_match_sparse_id_truth_tables() {
    let eng = Engine::new();
    let vars = [VarId(19), VarId(2), VarId(8)];
    for tree in [Vtree::balanced_over(&vars), Vtree::linear_from_order(&vars)] {
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
            minimize(&mut f);
            assert_canonical(&f);
            assert_eq!(f.model_count(), bits.count_ones().into());
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
            assert_eq!(implied_literals(&f), expected_backbone, "truth table {bits}");
        }
    }
}

#[test]
fn leaf_outputs_have_only_their_forced_literal() {
    let eng = Engine::new();
    let var = VarId(19);
    let tree = Arc::new(Vtree::leaf(var));
    for (f, expected) in [
        (Tdd::zero(&tree), vec![]),
        (Tdd::one(&tree), vec![]),
        (eng.literal(&tree, Literal::pos(var)).unwrap(), vec![Literal::pos(var)]),
        (eng.literal(&tree, Literal::neg(var)).unwrap(), vec![Literal::neg(var)]),
    ] {
        assert_canonical(&f);
        assert_eq!(implied_literals(&f), expected);
        assert_eq!(eng.support(&f).unwrap(), expected.iter().map(|literal| literal.var).collect::<Vec<_>>());
    }
}

#[test]
fn backbone_omits_marginal_leaves() {
    let eng = Engine::new();
    let vars = [VarId(19), VarId(2), VarId(8)];
    let tree = Arc::new(Vtree::balanced_over(&vars));
    let mut f = eng.cube(&tree, [Literal::pos(vars[0]), Literal::neg(vars[1])]).unwrap();
    assert_canonical(&f);
    let leaf = tree.leaf_of(vars[0]).unwrap();
    crate::marginal::marginalize_levels(&eng, &mut f, &[leaf]).unwrap();
    minimize(&mut f);
    assert_canonical(&f);
    assert_eq!(implied_literals(&f), vec![Literal::neg(vars[1])]);

    let tree = Arc::new(Vtree::leaf(vars[0]));
    let mut f = eng.literal(&tree, Literal::pos(vars[0])).unwrap();
    crate::marginal::marginalize_levels(&eng, &mut f, &[tree.root()]).unwrap();
    assert_canonical(&f);
    assert!(implied_literals(&f).is_empty());
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
        let mut gate = PollGate::new(1);
        let result = visit_leaf_labels(&f, |work| lim.poll(&mut gate, work), |_, _| {
            visited += 1;
            Ok(())
        });
        assert_eq!(result, Err(OperationError::Stopped));
        assert_eq!(lim.work_units() - before, 2);
        assert_eq!(visited, 0, "the stop must precede completion of the parent's summaries");
    }
    assert_eq!(eng.support(&f).unwrap(), vec![VarId(0), VarId(1)]);
    assert!(implied_literals(&f).is_empty());
    assert_canonical(&f);
}
