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

/// Two conjunctions over eight variables, `f` with the level `ll` already
/// summed out, and the internal levels of the balanced vtree.
fn collapse_operands(eng: &Engine) -> (Tdd, Tdd, Arc<Vtree>) {
    let tree = Arc::new(Vtree::balanced(8));
    let (left, _) = tree.children(tree.root());
    let (ll, lr) = tree.children(left);
    let mut f = eng.clause(&tree, [1, 2, 3, 4]).unwrap();
    f = eng.and(f, eng.clause(&tree, [-1, -3, 5, 6]).unwrap()).unwrap();
    f = eng.and(f, eng.clause(&tree, [2, -4, 7, 8]).unwrap()).unwrap();
    let mut g = eng.clause(&tree, [3, 5, 7]).unwrap();
    g = eng.and(g, eng.clause(&tree, [-4, -6, 8]).unwrap()).unwrap();
    eng.marginalize_levels(&mut f, &[ll]).unwrap();
    assert!(f.level(ll).is_marginal() && !f.level(lr).is_marginal());
    (f, g, tree)
}

/// Naming a closed subtree level by level, over levels an operand already
/// summed out and beside summed siblings, leaves what the conjunction
/// followed by the marginalization pass leaves, at the default sparse floor
/// and with every level over it.
#[test]
fn closed_target_subtrees_match_the_two_step_result() {
    use crate::limits::{LimitConfig, SparseRoute};
    let eng = Engine::new();
    let (f, g, tree) = collapse_operands(&eng);
    let root = tree.root();
    let (left, right) = tree.children(root);
    let (_, lr) = tree.children(left);
    let (rl, rr) = tree.children(right);
    let leaf = tree.leaf_of(VarId(7)).unwrap();
    for min_grid in [SparseRoute::DEFAULT.min_grid, 0] {
        let _scope = eng
            .limits()
            .scope(LimitConfig::none().with_sparse_route(SparseRoute::new(2, min_grid)));
        for targets in [
            vec![left, lr],
            vec![lr],
            vec![left, lr, rl],
            vec![right, rl, rr],
            vec![right, rl, rr, leaf],
            vec![right, rl],
            vec![root, left, lr, right, rl, rr],
        ] {
            let mut expected = eng.and(f.clone(), g.clone()).unwrap();
            eng.marginalize_levels(&mut expected, &targets).unwrap();
            let mut actual = eng.and_marginalizing(f.clone(), g.clone(), &targets).unwrap();
            for &t in &targets {
                assert!(actual.level(t).is_marginal(), "target {t:?} of {targets:?}, floor {min_grid}");
            }
            let count = expected.model_count().unwrap();
            assert_eq!(actual.model_count().unwrap(), count, "{targets:?}, floor {min_grid}");
            assert_eq!(
                eng.and_model_count(f.clone(), g.clone(), &targets).unwrap(),
                count,
                "counted, {targets:?}, floor {min_grid}",
            );
            actual.minimize().unwrap();
            expected.minimize().unwrap();
            assert_canonical(&actual);
            assert_canonical(&expected);
            assert_eq!(actual.pair_count(), expected.pair_count(), "{targets:?}, floor {min_grid}");
            assert_eq!(actual.node_count(), expected.node_count(), "{targets:?}, floor {min_grid}");
        }
    }
}

/// With every level over the sparse floor, the levels under the top of a
/// closed target subtree leave the mask, and a subtree with a level that is
/// neither a target nor marginal keeps its top.
#[test]
fn collapse_clears_the_levels_under_a_closed_target() {
    use crate::limits::{LimitConfig, SparseRoute};
    let eng = Engine::new();
    let (f, g, tree) = collapse_operands(&eng);
    let _scope = eng
        .limits()
        .scope(LimitConfig::none().with_sparse_route(SparseRoute::new(2, 0)));
    let (left, right) = tree.children(tree.root());
    let (ll, lr) = tree.children(left);
    let (rl, rr) = tree.children(right);
    let mask_of = |targets: &[crate::vtree::VtreeIdx]| {
        let mut mask = vec![false; tree.num_nodes()];
        for &t in targets {
            mask[t.idx()] = true;
        }
        super::super::collapse_target_subtrees(&eng, &tree, &f, &g, &mut mask).unwrap();
        mask
    };

    let closed = mask_of(&[right, rl, rr]);
    assert!(!closed[rl.idx()] && !closed[rr.idx()], "the levels under a closed target left in the mask");

    // `ll` is marginal in `f`, so `left` is closed without naming it.
    let over_marginal = mask_of(&[left, lr]);
    assert!(!over_marginal[lr.idx()] && !over_marginal[ll.idx()]);
    assert!(!over_marginal[left.idx()], "a target over an operand's marginal level is left to the pass");

    let open = mask_of(&[right, rl]);
    assert!(open[right.idx()], "a target over a level that is neither a target nor marginal left the mask");
}
