use super::*;
use crate::diagram::{Arithmetic, LiteralWeights, RationalWeights, WeightStore};
use crate::limits::{LimitConfig, StopCallback, StopDecision};
use crate::test_helpers::{assert_canonical, rat};
use crate::OperationError;

#[test]
fn checked_satisfiability_agrees_with_every_three_variable_truth_table() {
    let engine = Engine::new();
    for tree in [Vtree::balanced(3), Vtree::linear(3)] {
        let tree = Arc::new(tree);
        for truth in 0u16..256 {
            let mut f = engine.zero(&tree);
            for assignment in 0..8 {
                if truth & (1 << assignment) == 0 { continue; }
                let cube = engine.cube(&tree, (0..3).map(|var| {
                    Literal::new(VarId(var + 1), assignment & (1 << var) != 0)
                })).unwrap();
                assert_canonical(&cube);
                f = engine.or(f, cube).unwrap();
            }
            assert_canonical(&f);
            assert_eq!(engine.is_sat(&f), Ok(truth != 0), "truth table {truth}");
            let not_third = engine.literal(&tree, -3).unwrap();
            assert_canonical(&not_third);
            let mut constrained = engine.and(f, not_third).unwrap();
            assert_eq!(engine.is_sat(&constrained), Ok(truth & 0x0f != 0));
            constrained.minimize().unwrap();
            assert_canonical(&constrained);
        }
    }
}

#[test]
fn checked_satisfiability_ignores_weights_and_reads_marginal_counts() {
    let engine = Engine::new();
    let tree = Arc::new(Vtree::balanced(3));
    let mut f = engine.cube(&tree, [1, -2]).unwrap();
    assert_canonical(&f);
    let weights = RationalWeights::from_literals(&vec![
        LiteralWeights { negative: rat(0, 1), positive: rat(0, 1) }; 3
    ]);
    f.set_weights(WeightStore::new(weights, Arithmetic::ExactRational)).unwrap();
    assert_eq!(engine.is_sat(&f), Ok(true));
    assert_eq!(engine.implied_literals(&f), Ok(vec![1.try_into().unwrap(), (-2).try_into().unwrap()]));
    assert_eq!(engine.weighted_value(&f).unwrap().unwrap().into_rational(), rat(0, 1));
    engine.marginalize_levels(&mut f, &[tree.root()]).unwrap();
    assert_eq!(engine.is_sat(&f), Err(OperationError::IncompatibleWeights));
    assert!(matches!(engine.implied_literals(&f), Err(OperationError::MarginalLevel(_))));
    let mut counts = engine.one(&tree);
    assert_canonical(&counts);
    engine.marginalize_levels(&mut counts, &[tree.root()]).unwrap();
    assert_eq!(engine.is_sat(&counts), Ok(true));
    assert!(matches!(engine.implied_literals(&counts), Err(OperationError::MarginalLevel(_))));
}

#[test]
fn checked_satisfiability_returns_refusals_without_changing_the_input() {
    let engine = Engine::new();
    let tree = Arc::new(Vtree::balanced(3));
    let f = engine.clause(&tree, [1, 2]).unwrap();
    assert_canonical(&f);
    {
        let _scope = engine.limits().scope(LimitConfig::none().with_memory_budget_bytes(Some(0)));
        assert_eq!(engine.is_sat(&f), Ok(true));
    }
    let zero = engine.zero(&tree);
    assert_canonical(&zero);
    {
        let _scope = engine.limits().scope(LimitConfig::none().with_stop_callback(Some(
            StopCallback::new(|_, _| StopDecision::Stop))));
        assert_eq!(engine.is_sat(&f), Err(OperationError::Stopped));
        assert_eq!(engine.is_sat(&zero), Err(OperationError::Stopped));
    }
    assert_eq!(engine.is_sat(&f), Ok(true));
    assert_eq!(engine.model_count(&f).unwrap(), 6u32.into());
    assert_canonical(&f);
}

/// A count-marginal diagram with its reduction still pending.
fn edited_marginal_fixture(engine: &Engine) -> Tdd {
    let tree = Arc::new(Vtree::balanced(8));
    let mut g = engine.clause(&tree, [1, 5]).unwrap();
    assert_canonical(&g);
    let mut under_left = vec![false; tree.num_nodes()];
    let mut stack = vec![tree.children(tree.root()).0];
    while let Some(t) = stack.pop() {
        under_left[t.idx()] = true;
        if !tree.node(t).is_leaf() { let (l, r) = tree.children(t); stack.extend([l, r]); }
    }
    let summed: Vec<VtreeIdx> = tree.internal_bottomup_slice().iter().copied()
        .filter(|&t| under_left[t.idx()]).collect();
    engine.marginalize_levels(&mut g, &summed).unwrap();
    let f = engine.and(g, engine.literal(&tree, 8).unwrap()).unwrap();
    assert!(f.has_marginal_level() && !f.dirty.is_empty());
    f
}

/// A weight-marginal diagram with its reduction still pending.
fn edited_weighted_fixture(engine: &Engine) -> Tdd {
    let tree = Arc::new(Vtree::balanced(4));
    let mut g = engine.clause(&tree, [1, 3]).unwrap();
    assert_canonical(&g);
    let weights = RationalWeights::from_literals(&vec![
        LiteralWeights { negative: rat(0, 1), positive: rat(1, 2) }; 4
    ]);
    g.set_weights(WeightStore::new(weights, Arithmetic::ExactRational)).unwrap();
    let (left, _) = tree.children(tree.root());
    engine.marginalize_levels(&mut g, &[left]).unwrap();
    let f = engine.and(g, engine.literal(&tree, 4).unwrap()).unwrap();
    assert!(f.has_marginal_level() && !f.dirty.is_empty());
    f
}

/// A structural output above marginal levels answers from the output node,
/// edited or not, without doing any work.
#[test]
fn checked_satisfiability_reads_the_output_above_marginal_levels() {
    let engine = Engine::new();
    for mut f in [edited_marginal_fixture(&engine), edited_weighted_fixture(&engine)] {
        for _ in 0..2 {
            let before = engine.limits().work_units();
            assert_eq!(engine.is_sat(&f), Ok(true));
            assert_eq!(engine.limits().work_units(), before);
            f.minimize().unwrap();
            assert_canonical(&f);
        }
    }
}

/// Satisfiability reads summed-out variables as satisfiable whatever their
/// counts, so a hand-written zero count does not make a diagram unsatisfiable.
#[test]
fn checked_satisfiability_does_not_read_counts() {
    use crate::diagram::{ChildPair, TddNodeId, POS_LEAF_IDX};
    use crate::test_helpers::{inline_side, sum_out_leaf};
    let engine = Engine::new();
    let tree = Arc::new(Vtree::balanced(2));
    let root = tree.root();
    let mut builder = Tdd::builder(&engine, &tree).unwrap();
    sum_out_leaf(&engine, &tree, &mut builder, tree.children(root).1);
    let out = builder.push(&engine, root, &[ChildPair::new(POS_LEAF_IDX, inline_side(0))]).unwrap();
    let f = builder.finish(TddNodeId { vtree: root, local: out }).unwrap();
    assert_eq!(engine.is_sat(&f), Ok(true));
    assert_eq!(engine.model_count(&f).unwrap(), 0u32.into());
}
