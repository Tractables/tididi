use std::sync::Arc;
use crate::{Engine, Tdd};
use crate::diagram::{Arithmetic, LiteralWeights, RationalWeights, WeightStore, WeightValue};


use crate::test_helpers::{assert_canonical, rat};
use crate::vtree::{VarId, Vtree};

#[test]
fn requested_targets_are_completed_after_identity_and_self_conjunction() {
    let eng = Engine::new();
    for tree in [Vtree::balanced(4), Vtree::linear(4)] {
        let tree = Arc::new(tree);
        let (left, _) = tree.children(tree.root());
        let leaf = tree.leaf_of(VarId(1)).unwrap();
        for arithmetic in [None, Some(Arithmetic::ExactRational), Some(Arithmetic::SignedLog), Some(Arithmetic::ScaledFloat)] {
            let mut f = eng.clause(&tree, [1, 2]).unwrap();
            let mut disjoint = eng.clause(&tree, [3, 4]).unwrap();
            if let Some(arithmetic) = arithmetic {
                let weights = RationalWeights::from_literals(&vec![
                    LiteralWeights { negative: rat(1, 3), positive: rat(2, 3) }; 4
                ]);
                let store = WeightStore::new(weights, arithmetic);
                f.set_weights(store.clone()).unwrap();
                disjoint.set_weights(store).unwrap();
            }
            assert_canonical(&f);
            assert_canonical(&disjoint);
            for g in [f.clone(), eng.one(&tree), disjoint] {
                assert_canonical(&g);
                for targets in [vec![left], vec![tree.root()], vec![leaf, left, left], vec![tree.root(), leaf, left]] {
                    let mut expected = eng.and(f.clone(), g.clone()).unwrap();
                    eng.marginalize_levels(&mut expected, &targets).unwrap();
                    let mut actual = eng.and_marginalizing(f.clone(), g.clone(), &targets).unwrap();
                    for &target in &targets {
                        assert!(actual.level(target).is_marginal(), "target {target:?}, arithmetic {arithmetic:?}");
                    }
                    if arithmetic.is_some() {
                        let a = eng.weighted_value(&actual).unwrap().unwrap();
                        let b = eng.weighted_value(&expected).unwrap().unwrap();
                        if arithmetic == Some(Arithmetic::ExactRational) { assert_eq!(a.into_rational(), b.into_rational()); }
                        else {
                            let signed = |v: &WeightValue| match v {
                                WeightValue::Log(s) => f64::from(s.sign) * s.ln_abs.exp(),
                                WeightValue::Scaled(s) => s.to_f64(),
                                _ => panic!("expected a bounded-precision value"),
                            };
                            assert!((signed(&a) - signed(&b)).abs() < 1e-12);
                        }
                    } else { assert_eq!(actual.model_count().unwrap(), expected.model_count().unwrap()); }
                    actual.minimize().unwrap();
                    expected.minimize().unwrap();
                    assert_canonical(&actual);
                    assert_canonical(&expected);
                }
            }
        }
    }
}

#[test]
fn target_completion_preserves_false_and_empty_requests() {
    let eng = Engine::new();
    let tree = Arc::new(Vtree::balanced(4));
    let f = eng.clause(&tree, [1, 2]).unwrap();
    assert_canonical(&f);
    let zero = Tdd::zero(&tree);
    assert_canonical(&zero);
    let result = eng.and_marginalizing(f.clone(), zero, &[tree.root()]).unwrap();
    assert!(result.is_zero());
    assert_canonical(&result);
    let result = eng.and_marginalizing(f.clone(), f.clone(), &[]).unwrap();
    assert!(eng.equivalent(&result, &f).unwrap());
    assert_canonical(&result);
}

#[test]
fn collapse_target_subtrees_handles_prior_marginal_descendants_and_siblings() {
    let eng = Engine::new();
    let tree = Arc::new(Vtree::balanced(8));
    let root = tree.root();
    let (left, right) = tree.children(root);
    let (ll, lr) = tree.children(left);
    let (rl, rr) = tree.children(right);

    let mut f = eng.clause(&tree, [1, 2, 3, 4]).unwrap();
    f = eng.and(f, eng.clause(&tree, [-1, -3, 5, 6]).unwrap()).unwrap();
    f = eng.and(f, eng.clause(&tree, [2, -4, 7, 8]).unwrap()).unwrap();
    let mut g = eng.clause(&tree, [3, 5, 7]).unwrap();
    g = eng.and(g, eng.clause(&tree, [-4, -6, 8]).unwrap()).unwrap();

    // Step 1: marginalize `ll` inside `left`, leaving `lr` structural.
    let mut f_step1 = f.clone();
    eng.marginalize_levels(&mut f_step1, &[ll]).unwrap();
    assert!(f_step1.level(ll).is_marginal());
    assert!(!f_step1.level(lr).is_marginal());

    // Step 2: schedule `[left, lr]` (and `[root, left, lr, right, rl, rr]`)
    // when `ll` is already marginal in `f_step1`.
    for targets in [
        vec![left, lr],
        vec![lr],
        vec![left, lr, rl],
        vec![root, left, lr, right, rl, rr],
    ] {
        let mut expected = eng.and(f_step1.clone(), g.clone()).unwrap();
        eng.marginalize_levels(&mut expected, &targets).unwrap();
        let mut actual = eng.and_marginalizing(f_step1.clone(), g.clone(), &targets).unwrap();
        for &t in &targets {
            assert!(actual.level(t).is_marginal(), "target {t:?} in {targets:?}");
        }
        assert_eq!(actual.model_count().unwrap(), expected.model_count().unwrap());
        let counted = eng
            .and_model_count(f_step1.clone(), g.clone(), &targets)
            .unwrap();
        assert_eq!(counted, expected.model_count().unwrap());
        actual.minimize().unwrap();
        expected.minimize().unwrap();
        assert_canonical(&actual);
        assert_canonical(&expected);
        assert_eq!(actual.pair_count(), expected.pair_count());
        assert_eq!(actual.node_count(), expected.node_count());
    }
}
