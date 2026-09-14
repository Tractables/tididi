use std::sync::Arc;
use crate::{Engine, Tdd};
use crate::diagram::{Arithmetic, LiteralWeights, RationalWeights, WeightStore};
use crate::marginal::marginalize_levels;
use crate::reduce::minimize;
use crate::test_helpers::{assert_canonical, rat};
use crate::vtree::{VarId, Vtree};

#[test]
fn requested_targets_are_completed_after_identity_and_self_conjunction() {
    let eng = Engine::new();
    for tree in [Vtree::balanced(4), Vtree::linear(4)] {
        let tree = Arc::new(tree);
        let (left, _) = tree.children(tree.root());
        let leaf = tree.leaf_of(VarId(0)).unwrap();
        for arithmetic in [None, Some(Arithmetic::ExactRational), Some(Arithmetic::SignedLog)] {
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
                    marginalize_levels(&eng, &mut expected, &targets).unwrap();
                    let mut actual = eng.and_marginalizing(f.clone(), g.clone(), &targets).unwrap();
                    for &target in &targets {
                        assert!(actual.level(target).is_marginal(), "target {target:?}, arithmetic {arithmetic:?}");
                    }
                    if arithmetic.is_some() {
                        let a = eng.weighted_value(&actual).unwrap().unwrap();
                        let b = eng.weighted_value(&expected).unwrap().unwrap();
                        if arithmetic == Some(Arithmetic::ExactRational) { assert_eq!(a.into_rational(), b.into_rational()); }
                        else {
                            let a = a.as_log().unwrap();
                            let b = b.as_log().unwrap();
                            assert!((f64::from(a.sign) * a.ln_abs.exp() - f64::from(b.sign) * b.ln_abs.exp()).abs() < 1e-12);
                        }
                    } else { assert_eq!(actual.model_count(), expected.model_count()); }
                    minimize(&mut actual);
                    minimize(&mut expected);
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
