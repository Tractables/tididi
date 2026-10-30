use std::sync::Arc;

use num_rational::BigRational;

use crate::diagram::{Arithmetic, LiteralWeights, RationalWeights, WeightStore};
use crate::test_helpers::assert_canonical;
use crate::vtree::{VarId, Vtree};
use crate::{Engine, OperationError, Tdd, add, and, or};

#[test]
fn negate_marginal_subtree_commutes_with_marginalization_unweighted() {
    let eng = Engine::new();
    let vtree = Arc::new(Vtree::balanced(6));
    let f1 = Tdd::clause(&vtree, [1, 2, 4]).unwrap();
    let f2 = Tdd::clause(&vtree, [-1, 3, -5, 6]).unwrap();
    let f3 = Tdd::clause(&vtree, [2, -3, 5]).unwrap();
    let base = and(and(f1, f2).unwrap(), f3).unwrap();

    let (left, right) = vtree.children(vtree.root());
    let (ll, lr) = vtree.children(left);
    let leaf1 = vtree.leaf_of(VarId(1)).unwrap();

    for target in [leaf1, ll, lr, left, right, vtree.root()] {
        let mut marg_then_neg = base.clone();
        eng.marginalize_levels(&mut marg_then_neg, &[target]).unwrap();

        let borrowed_neg = eng.negate(&marg_then_neg).unwrap();
        let owned_neg = eng.negate(marg_then_neg.clone()).unwrap();
        assert_canonical(&borrowed_neg);
        assert_canonical(&owned_neg);

        eng.minimize(&mut marg_then_neg).unwrap();
        assert_canonical(&marg_then_neg);

        let mut neg_then_marg = eng.negate(&base).unwrap();
        eng.marginalize_levels(&mut neg_then_marg, &[target]).unwrap();
        eng.minimize(&mut neg_then_marg).unwrap();
        assert_canonical(&neg_then_marg);

        assert_eq!(borrowed_neg.model_count().unwrap(), neg_then_marg.model_count().unwrap());
        assert_eq!(owned_neg.model_count().unwrap(), neg_then_marg.model_count().unwrap());

        // Double negation recovers the original marginalized diagram's count.
        let double_neg = eng.negate(owned_neg).unwrap();
        assert_canonical(&double_neg);
        assert_eq!(double_neg.model_count().unwrap(), marg_then_neg.model_count().unwrap());
    }
}

#[test]
fn negate_marginal_constants_and_root_leaf() {
    let eng = Engine::new();
    let vtree1 = Arc::new(Vtree::balanced(1));
    let root1 = vtree1.root();
    for lit in [Tdd::zero(&vtree1), Tdd::one(&vtree1), crate::literal(&vtree1, 1).unwrap(), crate::literal(&vtree1, -1).unwrap()] {
        let mut m = lit.clone();
        eng.marginalize_levels(&mut m, &[root1]).unwrap();
        let neg_m = eng.negate(m).unwrap();
        let mut expected = eng.negate(lit).unwrap();
        eng.marginalize_levels(&mut expected, &[root1]).unwrap();
        assert_eq!(neg_m.model_count().unwrap(), expected.model_count().unwrap());
    }
}

#[test]
fn negate_marginal_subtree_commutes_with_marginalization_weighted() {
    let eng = Engine::new();
    let vtree = Arc::new(Vtree::balanced(4));
    let weights = RationalWeights::from_literals(&[
        LiteralWeights { negative: BigRational::new(2.into(), 3.into()), positive: BigRational::new(1.into(), 3.into()) },
        LiteralWeights { negative: BigRational::new(1.into(), 2.into()), positive: BigRational::new(1.into(), 2.into()) },
        LiteralWeights { negative: BigRational::new(2.into(), 5.into()), positive: BigRational::new(3.into(), 5.into()) },
        LiteralWeights { negative: BigRational::new(3.into(), 4.into()), positive: BigRational::new(1.into(), 4.into()) },
    ]);

    let c1 = Tdd::clause(&vtree, [1, 3]).unwrap();
    let c2 = Tdd::clause(&vtree, [-2, 4]).unwrap();
    let base = or(and(c1, c2).unwrap(), Tdd::cube(&vtree, [1, 2, -3]).unwrap()).unwrap();

    let (left, right) = vtree.children(vtree.root());
    let leaf1 = vtree.leaf_of(VarId(1)).unwrap();
    let leaf2 = vtree.leaf_of(VarId(2)).unwrap();

    for arithmetic in [Arithmetic::ExactRational, Arithmetic::ScaledFloat, Arithmetic::SignedLog] {
        for target in [leaf1, leaf2, left, right, vtree.root()] {
            let mut f = base.clone();
            f.set_weights(WeightStore::new(weights.clone(), arithmetic)).unwrap();
            eng.marginalize_levels(&mut f, &[target]).unwrap();

            let neg = eng.negate(&f).unwrap();
            assert_canonical(&neg);

            eng.minimize(&mut f).unwrap();
            assert_canonical(&f);

            let mut expected = eng.negate(&base).unwrap();
            expected.set_weights(WeightStore::new(weights.clone(), arithmetic)).unwrap();
            eng.marginalize_levels(&mut expected, &[target]).unwrap();
            eng.minimize(&mut expected).unwrap();
            assert_canonical(&expected);

            match arithmetic {
                Arithmetic::ExactRational => {
                    assert_eq!(
                        neg.weighted_value().unwrap().unwrap().into_rational(),
                        expected.weighted_value().unwrap().unwrap().into_rational(),
                    );
                }
                Arithmetic::ScaledFloat => {
                    let a = neg.weighted_value().unwrap().unwrap().as_scaled().unwrap().to_f64();
                    let b = expected.weighted_value().unwrap().unwrap().as_scaled().unwrap().to_f64();
                    assert!((a - b).abs() <= 1e-12, "a={a}, b={b}");
                }
                Arithmetic::SignedLog => {
                    let s1 = *neg.weighted_value().unwrap().unwrap().as_log().unwrap();
                    let s2 = *expected.weighted_value().unwrap().unwrap().as_log().unwrap();
                    let a = f64::from(s1.sign) * s1.ln_abs.exp();
                    let b = f64::from(s2.sign) * s2.ln_abs.exp();
                    assert!((a - b).abs() <= 1e-12, "a={a}, b={b}");
                }
            }
        }
    }
}

#[test]
fn negate_refuses_overflowing_unweighted_marginal_counts() {
    let eng = Engine::new();
    let vtree = Arc::new(Vtree::balanced(2));
    let left = vtree.children(vtree.root()).0;
    let mut f = Tdd::one(&vtree);
    eng.marginalize_levels(&mut f, &[left]).unwrap();
    // Adding two constant-one diagrams doubles the marginal count at `left` to 4 > 2^1.
    let doubled = add(f.clone(), f).unwrap();
    assert_eq!(eng.negate(doubled).unwrap_err(), OperationError::MarginalLevel(left));
}
