use super::*;
use crate::limits::{LimitConfig, OperationError, StopAt, StopRules};

#[test]
fn filter_keeps_allocation_or_removes_the_root() {
    let tree = Arc::new(Vtree::balanced(5));
    let f = Tdd::clause(&tree, [1, 2, -3]).unwrap();
    assert_canonical(&f);
    let allocation = f.levels.as_ptr();
    let g = f.filter_nodes(|_| true).unwrap();
    assert_eq!(g.levels.as_ptr(), allocation);
    assert_canonical(&g);
    let root = g.output();
    let z = g.filter_nodes(|id| id != root).unwrap();
    assert!(z.is_zero());
    assert_canonical(&z);
}

#[test]
fn filtering_nodes_only_removes_models_and_pairs() {
    let eng = Engine::new();
    let tree = Arc::new(Vtree::balanced(6));
    let f = eng.clause(&tree, [1, 2, 3, 4, 5, 6]).unwrap();
    assert_canonical(&f);
    for divisor in 2..7 {
        let mut g = eng.filter_nodes(f.clone(), |id| (id.vtree.idx() + id.local.idx()) % divisor != 0).unwrap();
        assert!(g.pair_count() <= f.pair_count());
        eng.minimize(&mut g).unwrap();
        assert_canonical(&g);
        let conjunction = eng.and(g.clone(), f.clone()).unwrap();
        assert!(eng.equivalent(&g, &conjunction).unwrap());
    }
}

#[test]
fn node_filter_honors_cancellation_before_the_callback() {
    let tree = Arc::new(Vtree::balanced(4));
    let f = Tdd::clause(&tree, [1, 2]).unwrap();
    assert_canonical(&f);
    let eng = Engine::new();
    let _guard = eng.limits().scope(LimitConfig::none().with_stop_rules(StopRules {
        unconditional: Some(StopAt::WorkUnits(0)), ..StopRules::default()
    }));
    assert_eq!(eng.filter_nodes(f, |_| panic!("canceled before callback")).err(), Some(OperationError::Stopped));
}

#[test]
fn filter_respects_memory_refusal() {
    let tree = Arc::new(Vtree::balanced(4));
    let f = Tdd::clause(&tree, [1, 2]).unwrap();
    assert_canonical(&f);
    let eng = Engine::new();
    let _guard = eng.limits().scope(LimitConfig::none().with_memory_budget_bytes(Some(0)));
    assert_eq!(eng.filter_nodes(f, |_| true).err(), Some(OperationError::OverBudget));
}

#[test]
fn filter_retains_weighted_marginal_columns() {
    use crate::diagram::{Arithmetic, LiteralWeights, RationalWeights, WeightStore};
    let eng = Engine::new();
    let tree = Arc::new(Vtree::balanced(8));
    let mut f = Tdd::clause(&tree, [1, 3, 5, 7]).unwrap();
    let weights = vec![LiteralWeights { negative: rat(2, 1), positive: rat(3, 1) }; 8];
    f.set_weights(WeightStore::new(RationalWeights::from_literals(&weights), Arithmetic::ExactRational)).unwrap();
    let forgotten = tree.children(tree.root()).0;
    eng.marginalize_levels(&mut f, &[forgotten]).unwrap();
    assert_canonical(&f);
    let root = f.output();
    let mut rejected = None;
    let mut g = eng.filter_nodes(f.clone(), |id| {
        assert!(!f.level(id.vtree).is_marginal());
        if id != root && rejected.is_none() { rejected = Some(id); false } else { true }
    }).unwrap();
    assert!(rejected.is_some());
    eng.minimize(&mut g).unwrap();
    assert_canonical(&g);
    assert!(g.pair_count() < f.pair_count());
    // The retained circuit carries weighted marginal coefficients through a
    // rebuild. Evaluating it before and after minimization agrees exactly.
    let before = eng.weighted_value(&g).unwrap().unwrap();
    let same = eng.filter_nodes(g.clone(), |_| true).unwrap();
    assert_canonical(&same);
    assert_eq!(exact_weight(&before), exact_weight(&eng.weighted_value(&same).unwrap().unwrap()));
    assert!(eng.weighted_value(&f).unwrap().unwrap().as_log().is_none());
}
