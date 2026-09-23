use super::*;
use std::sync::Arc;
use std::cell::Cell;
use num_rational::BigRational;
use num_traits::{Zero, One};
use crate::diagram::{LeafLabel, LiteralWeights, RationalWeights};
use crate::test_helpers::assert_canonical;
use crate::Vtree;

fn weights(n: usize) -> RationalWeights {
    RationalWeights::from_literals(&(0..n).map(|i| LiteralWeights {
        negative: BigRational::from_integer((i as i32 - 1).into()),
        positive: BigRational::new((i as i32 + 1).into(), 3.into()),
    }).collect::<Vec<_>>())
}

#[test]
fn evidence_updates_match_enumerated_signed_weights() {
    let vtree = Arc::new(Vtree::balanced(4));
    let f = Tdd::clause(&vtree, [1, -2]).unwrap();
    assert_canonical(&f);
    let w = weights(4);
    let mut e = f.evaluator(w.clone()).unwrap();
    for code in 0..81 {
        let mut c = code;
        let pins: Vec<_> = (1..=4).map(|v| {
            let pin = match c % 3 { 0 => None, 1 => Some(false), _ => Some(true) };
            c /= 3;
            (VarId(v), pin)
        }).collect();
        e.set_pins(&pins).unwrap();
        let mut expected = BigRational::zero();
        for mask in 0..16 {
            if mask & 1 == 0 && mask & 2 != 0 { continue; }
            if pins.iter().any(|(var, p)| p.is_some_and(|p| p != (mask & (1 << var.idx()) != 0))) { continue; }
            let mut product = BigRational::one();
            for var in 1..=4 {
                product *= if mask & (1 << (var-1)) != 0 { w.pos_weight(VarId(var)) } else { w.neg_weight(VarId(var)) };
            }
            expected += product;
        }
        assert_eq!(e.value().unwrap(), expected, "pins {pins:?}");
        assert_eq!(e.value().unwrap(), expected);
    }
    e.clear_pins();
    assert_eq!(e.value().unwrap(), f.evaluate(&w).unwrap());
    let replacement = RationalWeights::unit(4);
    e.replace_algebra(replacement.clone());
    assert_eq!(e.value().unwrap(), f.evaluate(&replacement).unwrap());
}

#[test]
fn sparse_variables_validate_observations_atomically() {
    let vtree = Arc::new(Vtree::balanced_over(&[VarId(3), VarId(9)]).unwrap());
    let f = Tdd::one(&vtree);
    assert_canonical(&f);
    let mut e = f.evaluator(RationalWeights::unit(9)).unwrap();
    e.observe([3]).unwrap();
    let before = e.value().unwrap();
    assert_eq!(e.observe([-3, 2]), Err(OperationError::VariableNotInVtree(VarId(2))));
    assert_eq!(e.observe([-3, 0]), Err(OperationError::InvalidLiteral(0)));
    assert_eq!(e.value().unwrap(), before);
    e.observe([3, -3, 9]).unwrap();
    assert_eq!(e.value().unwrap(), BigRational::one());
}

struct Tracked {
    calls: std::rc::Rc<Cell<usize>>,
    panic: std::rc::Rc<Cell<bool>>,
}
impl EvalAlgebra for Tracked {
    type Value = u64;
    fn zero(&self) -> u64 { 0 }
    fn leaf(&self, _: VarId, label: LeafLabel) -> u64 {
        assert!(!self.panic.replace(false), "algebra interrupted");
        self.calls.set(self.calls.get() + 1);
        if label == LeafLabel::One { 2 } else { 1 }
    }
    fn add_assign(&self, a: &mut u64, b: &u64) { *a += b; }
    fn mul(&self, a: &u64, b: &u64) -> u64 { a * b }
}

#[test]
fn unchanged_branches_are_cached_and_algebra_panics_are_retryable() {
    let vtree = Arc::new(Vtree::balanced(8));
    let f = Tdd::one(&vtree);
    assert_canonical(&f);
    let calls = std::rc::Rc::new(Cell::new(0));
    let panic = std::rc::Rc::new(Cell::new(false));
    let mut e = f.evaluator(Tracked { calls: calls.clone(), panic: panic.clone() }).unwrap();
    assert_eq!(e.value().unwrap(), 256);
    calls.set(0);
    assert_eq!(e.value().unwrap(), 256);
    assert_eq!(calls.get(), 0);
    e.observe([-1]).unwrap();
    assert_eq!(e.value().unwrap(), 128);
    assert_eq!(calls.get(), 2, "only the changed leaf is evaluated");
    e.observe([2]).unwrap();
    panic.set(true);
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| e.value())).is_err());
    e.observe([-3]).unwrap();
    assert_eq!(e.value().unwrap(), 32);
    e.clear_pins();
    assert_eq!(e.value().unwrap(), 256);
}

#[test]
fn allocation_refusals_preserve_pins_and_retry_on_the_same_engine() {
    let vtree = Arc::new(Vtree::balanced(8));
    let f = Tdd::clause(&vtree, [1, 2]).unwrap();
    assert_canonical(&f);
    let engine = Engine::new();
    for incremental in [false, true] {
        let mut completed = false;
        for n in 0..80 {
            let mut e = f.evaluator(RationalWeights::unit(8)).unwrap();
            if incremental { e.value().unwrap(); }
            e.observe([-1]).unwrap();
            engine.limits().refuse_nth_reserve(n);
            let result = e.bind(&engine).value();
            engine.limits().grant_every_reserve();
            e.observe([3]).unwrap();
            assert_eq!(e.bind(&engine).value().unwrap(), BigRational::from_integer(32.into()));
            if result.is_ok() { completed = true; break; }
            assert_eq!(result.unwrap_err(), OperationError::OverBudget);
        }
        assert!(completed);
    }
}

#[test]
fn constants_marginals_and_batch_limits() {
    use crate::limits::LimitConfig;
    let vtree = Arc::new(Vtree::leaf(VarId(5)));
    for f in [Tdd::zero(&vtree), Tdd::one(&vtree)] {
        assert_canonical(&f);
        let engine = Engine::new();
        let mut e = engine.evaluator(&f, RationalWeights::unit(5)).unwrap();
        let original = e.value().unwrap();
        e.observe([-5]).unwrap();
        assert_eq!(e.value().unwrap(), original / BigRational::from_integer(2.into()));
    }
    let vtree = Arc::new(Vtree::balanced(4));
    let mut f = Tdd::one(&vtree);
    assert_canonical(&f);
    {
        let engine = Engine::new();
        let mut e = engine.evaluator(&f, RationalWeights::unit(4)).unwrap();
        e.value().unwrap();
        let _scope = engine.limits().scope(LimitConfig::none().with_stop_rules(crate::limits::StopRules { unconditional: Some(crate::limits::StopAt::WorkUnits(0)), after_pairs: None }));
        assert!(e.value().is_err());
    }
    f.marginalize_levels(&[vtree.root()]).unwrap();
    assert_canonical(&f);
    assert!(matches!(f.evaluator(RationalWeights::unit(4)), Err(OperationError::MarginalLevel(_))));
}
