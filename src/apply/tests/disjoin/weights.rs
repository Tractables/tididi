use std::sync::Arc;
use crate::{Engine, OperationError, Tdd, Vtree};
use crate::diagram::{Arithmetic, LiteralWeights, RationalWeights, WeightStore, WeightValue};
use crate::test_helpers::{assert_canonical, rat};

fn weighted(mut f: Tdd, n: i64, arithmetic: Arithmetic) -> Tdd {
    let weights = vec![LiteralWeights { negative: rat(n, 1), positive: rat(n, 1) };
        f.vtree().num_vars() as usize];
    f.set_weights(WeightStore::new(RationalWeights::from_literals(&weights), arithmetic)).unwrap();
    f
}

fn same_weight(a: Option<WeightValue>, b: Option<WeightValue>) {
    let (a, b) = (a.expect("weights retained"), b.expect("reference is weighted"));
    match (a.as_log(), b.as_log()) {
        (Some(a), Some(b)) => {
            assert_eq!(a.is_zero(), b.is_zero());
            if !a.is_zero() { assert!((a.log10_abs() - b.log10_abs()).abs() < 1e-12); }
        }
        (None, None) => assert_eq!(a.as_rational(), b.as_rational()),
        _ => panic!("arithmetic changed"),
    }
}

#[test]
fn constant_operands_keep_the_weight_contract() {
    let vtree = Arc::new(Vtree::balanced(3));
    let eng = Engine::new();
    for arithmetic in [Arithmetic::ExactRational, Arithmetic::SignedLog] {
        for reverse in [false, true] {
            let zero = weighted(Tdd::zero(&vtree), 2, arithmetic);
            let f = Tdd::cube(&vtree, [1]).unwrap();
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
        let zeros = vec![weighted(Tdd::zero(&vtree), 2, arithmetic), Tdd::zero(&vtree)];
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
fn discarded_constants_still_reject_incompatible_weights() {
    let vtree = Arc::new(Vtree::balanced(3));
    let eng = Engine::new();
    for (n, arithmetic) in [(3, Arithmetic::ExactRational), (2, Arithmetic::SignedLog)] {
        for all_false in [false, true] {
            let first = weighted(Tdd::zero(&vtree), 2, Arithmetic::ExactRational);
            let second = if all_false { Tdd::zero(&vtree) } else { Tdd::cube(&vtree, [1]).unwrap() };
            let second = weighted(second, n, arithmetic);
            for inputs in [vec![first.clone(), second.clone()], vec![second, first]] {
                assert_eq!(eng.or_many(inputs.clone()).unwrap_err(), OperationError::IncompatibleWeights);
                assert_eq!(eng.nor_many(inputs).unwrap_err(), OperationError::IncompatibleWeights);
            }
        }
    }
}
