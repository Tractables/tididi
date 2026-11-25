use crate::{Engine, Tdd};
use crate::diagram::{Arithmetic, WeightValue};
use crate::apply::tests::weights::weighted_tree;
use crate::test_helpers::{assert_canonical, bounded_log10};

fn same_weight(a: Option<WeightValue>, b: Option<WeightValue>) {
    let (a, b) = (a.expect("weights retained"), b.expect("reference is weighted"));
    match (bounded_log10(&a), bounded_log10(&b)) {
        (Some(la), Some(lb)) => {
            assert_eq!(a.is_zero(), b.is_zero());
            if !a.is_zero() { assert!((la - lb).abs() < 1e-12); }
        }
        (None, None) => assert_eq!(a.as_rational(), b.as_rational()),
        _ => panic!("arithmetic changed"),
    }
}

#[test]
fn constant_operands_keep_the_weight_contract() {
    let eng = Engine::new();
    for arithmetic in [Arithmetic::ExactRational, Arithmetic::SignedLog, Arithmetic::ScaledFloat] {
        let vtree = weighted_tree(3, 2, arithmetic);
        for reverse in [false, true] {
            let zero = Tdd::zero(&crate::vtree::Home::unchecked(&vtree));
            let f = Tdd::cube(&crate::vtree::Home::unchecked(&vtree), [1]).unwrap();
            let want = eng.or(zero.clone(), f.clone()).unwrap();
            let mut operands = vec![zero, f];
            if reverse { operands.reverse(); }
            let got = eng.or_many(operands.clone()).unwrap();
            assert_canonical(&got);
            same_weight(got.weighted_value().unwrap(), want.weighted_value().unwrap());
            let got = eng.nor_many(operands).unwrap();
            let want = want.negate().unwrap();
            assert_canonical(&got);
            same_weight(got.weighted_value().unwrap(), want.weighted_value().unwrap());
        }
        let zeros = vec![Tdd::zero(&crate::vtree::Home::unchecked(&vtree)), Tdd::zero(&crate::vtree::Home::unchecked(&vtree))];
        let want = eng.or(zeros[0].clone(), zeros[1].clone()).unwrap();
        let got = eng.or_many(zeros.clone()).unwrap();
        assert_canonical(&got);
        same_weight(got.weighted_value().unwrap(), want.weighted_value().unwrap());
        let got = eng.nor_many(zeros).unwrap();
        assert_canonical(&got);
        same_weight(got.weighted_value().unwrap(), want.negate().unwrap().weighted_value().unwrap());
    }
}

#[test]
fn overlay_compositions_keep_both_weight_arithmetics() {
    let eng = Engine::new();
    for arithmetic in [Arithmetic::ExactRational, Arithmetic::SignedLog, Arithmetic::ScaledFloat] {
        let tree = weighted_tree(6, 2, arithmetic);
        let a = Tdd::clause(&crate::vtree::Home::unchecked(&tree), [1, -3, 5]).unwrap();
        let b = Tdd::cube(&crate::vtree::Home::unchecked(&tree), [-2, 4]).unwrap();
        let c = Tdd::clause(&crate::vtree::Home::unchecked(&tree), [3, 6]).unwrap();
        let reference_or = eng.negate(eng.and(eng.negate(a.clone()).unwrap(), eng.negate(b.clone()).unwrap()).unwrap()).unwrap();
        let reference_diff = eng.and(a.clone(), eng.negate(b.clone()).unwrap()).unwrap();
        let reference_or_many = eng.negate(eng.and(eng.negate(reference_or.clone()).unwrap(),eng.negate(c.clone()).unwrap()).unwrap()).unwrap();
        let reference_xor = eng.and(reference_or.clone(), eng.negate(eng.and(a.clone(), b.clone()).unwrap()).unwrap()).unwrap();
        let reference_ite = eng.negate(eng.and(
            eng.negate(eng.and(a.clone(), b.clone()).unwrap()).unwrap(),
            eng.negate(eng.and(eng.negate(a.clone()).unwrap(), c.clone()).unwrap()).unwrap(),
        ).unwrap()).unwrap();
        let pairs = [
            (eng.or(a.clone(), b.clone()).unwrap(), reference_or),
            (eng.and_not(a.clone(), b.clone()).unwrap(), reference_diff),
            (eng.or_many(vec![a.clone(), b.clone(), c.clone()]).unwrap(), reference_or_many),
            (eng.xor(a.clone(), b.clone()).unwrap(), reference_xor),
            (eng.ite(a, b, c).unwrap(), reference_ite),
        ];
        for (got, mut want) in pairs {
            want.minimize().unwrap();
            assert_canonical(&got);
            assert!(got.equivalent(&want).unwrap());
            same_weight(got.weighted_value().unwrap(), want.weighted_value().unwrap());
        }
    }
}
