use std::sync::atomic::{AtomicUsize, Ordering};
use super::*;
use std::cell::Cell;
use std::rc::Rc;
use crate::diagram::{EvalAlgebra, LeafLabel, RationalWeights};
use crate::limits::{LimitConfig, StopCallback, StopDecision};
use crate::OperationError;
use crate::test_helpers::assert_canonical;

/// A clause with a free variable, built before any resource limit is armed.
fn fixture() -> Tdd {
    let f = Tdd::clause(&Arc::new(Vtree::balanced(8)), [1, 3, -5]).unwrap();
    assert_canonical(&f);
    f
}

#[test]
fn algebra_evaluation_recovers_from_every_buffer_refusal() {
    let eng = Engine::new();
    let f = fixture();
    let algebra = RationalWeights::unit(8);
    let expected = crate::test_helpers::rat(224, 1);
    {
        let _limit = eng.limits().scope(LimitConfig::none().with_memory_budget_bytes(Some(0)));
        assert_eq!(eng.evaluate(&f, &algebra), Err(OperationError::OverBudget));
    }
    let mut completed = false;
    for reserve in 0..128 {
        eng.limits().refuse_nth_reserve(reserve);
        let result = eng.evaluate(&f, &algebra);
        eng.limits().grant_every_reserve();
        assert_eq!(eng.evaluate(&f, &algebra).unwrap(), expected);
        match result {
            Ok(value) => { assert_eq!(value, expected); completed = true; break; }
            Err(error) => assert_eq!(error, OperationError::OverBudget),
        }
    }
    assert!(completed, "the sweep must reach the first successful reserve count");
    assert_canonical(&f);
}

#[test]
fn algebra_evaluation_stops_at_entry_during_work_and_before_return() {
    let eng = Engine::new();
    eng.limits().pin_reduce_poll_stride(Some(1));
    let f = fixture();
    let algebra = RationalWeights::unit(8);
    let mut completed = false;
    for cut in 0..128 {
        let calls = Arc::new(AtomicUsize::new(0));
        let callback_calls = calls.clone();
        let result = {
            let _limit = eng.limits().scope(LimitConfig::none().with_stop_callback(Some(
                StopCallback::new(move |_, _| {
                    let call = callback_calls.fetch_add(1, Ordering::Relaxed);
                    if call == cut { StopDecision::Stop } else { StopDecision::Continue }
                }))));
            eng.evaluate(&f, &algebra)
        };
        assert_eq!(eng.evaluate(&f, &algebra).unwrap(), crate::test_helpers::rat(224, 1));
        match result {
            Ok(_) => { assert!(cut > 2); completed = true; break; }
            Err(error) => assert_eq!(error, OperationError::Stopped),
        }
    }
    assert!(completed);
    // A short walk must flush its final work even below the default stride.
    eng.limits().pin_reduce_poll_stride(None);
    for f in [Tdd::one(&Arc::new(Vtree::leaf(VarId(1)))), Tdd::zero(&Arc::new(Vtree::balanced(3)))] {
        assert_canonical(&f);
        let calls = Arc::new(AtomicUsize::new(0));
        let _limit = eng.limits().scope(LimitConfig::none().with_stop_callback(Some(
            StopCallback::new(move |_, _| {
                let call = calls.fetch_add(1, Ordering::Relaxed);
                if call == 1 { StopDecision::Stop } else { StopDecision::Continue }
            }))));
        assert_eq!(eng.evaluate(&f, &algebra), Err(OperationError::Stopped));
    }
}

/// The closing stop test charges no work of its own: evaluating the false
/// diagram walks nothing, so the work clock does not move.
#[test]
fn the_closing_stop_test_charges_no_work() {
    let eng = Engine::new();
    let f = Tdd::zero(&Arc::new(Vtree::balanced(3)));
    assert_canonical(&f);
    let mark = eng.limits().mark();
    eng.evaluate(&f, &RationalWeights::unit(3)).unwrap();
    assert_eq!(eng.limits().work_since(mark), 0);
}

/// Records live algebra values and the charged column-buffer high-water mark.
#[derive(Default)]
struct Usage {
    live: Cell<usize>,
    peak_values: Cell<usize>,
    peak_bytes: Cell<u64>,
}

struct Value {
    truth: bool,
    usage: Rc<Usage>,
}

impl Value {
    /// Create one tracked value.
    fn new(truth: bool, usage: &Rc<Usage>) -> Self {
        let live = usage.live.get() + 1;
        usage.live.set(live);
        usage.peak_values.set(usage.peak_values.get().max(live));
        Self { truth, usage: usage.clone() }
    }
}

impl Clone for Value {
    fn clone(&self) -> Self { Self::new(self.truth, &self.usage) }
}

impl Drop for Value {
    fn drop(&mut self) { self.usage.live.set(self.usage.live.get() - 1); }
}

struct TrackedBoolean<'a> {
    eng: &'a Engine,
    usage: Rc<Usage>,
}

impl EvalAlgebra for TrackedBoolean<'_> {
    type Value = Value;
    fn zero(&self) -> Value {
        self.usage.peak_bytes.set(self.usage.peak_bytes.get().max(self.eng.limits().meters().in_flight_bytes));
        Value::new(false, &self.usage)
    }
    fn leaf(&self, _: VarId, label: LeafLabel) -> Value {
        Value::new(label != LeafLabel::Zero, &self.usage)
    }
    fn add_assign(&self, a: &mut Value, b: &Value) { a.truth |= b.truth; }
    fn mul(&self, a: &Value, b: &Value) -> Value { Value::new(a.truth && b.truth, &self.usage) }
}

#[test]
fn algebra_columns_use_less_memory_than_allocating_every_level() {
    for tree in [Vtree::balanced(128), Vtree::linear(128)] {
        let eng = Engine::new();
        let f = Tdd::one(&Arc::new(tree));
        assert_canonical(&f);
        let usage = Rc::new(Usage::default());
        let algebra = TrackedBoolean { eng: &eng, usage: usage.clone() };
        let value = eng.evaluate(&f, &algebra).unwrap();
        assert!(value.truth);
        drop(value);
        assert_eq!(usage.live.get(), 0);
        let eager_slots: usize = f.vtree().bottomup().map(|t| f.reference_slot_count(t)).sum();
        let eager_bytes = (f.vtree().num_nodes() * std::mem::size_of::<Vec<Value>>()
            + eager_slots * std::mem::size_of::<Value>()) as u64;
        assert!(usage.peak_values.get() < eager_slots);
        assert!(usage.peak_bytes.get() < eager_bytes);
        eprintln!("{} levels: peak {} values / {} eager slots; {} charged bytes / {} eager bytes",
            f.vtree().num_nodes(), usage.peak_values.get(), eager_slots, usage.peak_bytes.get(), eager_bytes);
        // Reusing freed budget must allow the same pass under its measured peak.
        let _limit = eng.limits().scope(LimitConfig::none().with_memory_budget_bytes(Some(usage.peak_bytes.get())));
        assert!(eng.evaluate(&f, &algebra).unwrap().truth);
        assert_eq!(usage.live.get(), 0);
    }
}

#[test]
fn custom_algebra_panics_propagate_and_leave_the_engine_usable() {
    struct Panics;
    impl EvalAlgebra for Panics {
        type Value = usize;
        fn zero(&self) -> usize { 0 }
        fn leaf(&self, _: VarId, _: LeafLabel) -> usize { panic!("caller algebra failed") }
        fn add_assign(&self, a: &mut usize, b: &usize) { *a += b; }
        fn mul(&self, a: &usize, b: &usize) -> usize { a * b }
    }
    let eng = Engine::new();
    let f = fixture();
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| eng.evaluate(&f, &Panics))).unwrap_err();
    assert_eq!(panic.downcast_ref::<&str>(), Some(&"caller algebra failed"));
    assert_eq!(eng.evaluate(&f, &RationalWeights::unit(8)).unwrap(), crate::test_helpers::rat(224, 1));
}

#[test]
fn an_overriding_sum_of_products_is_used_and_agrees_with_the_default() {
    // Weighted model counts, a weight of its own per variable, so that a
    // product taken with the wrong partner shows. The override sums the
    // left values of the pairs that name one right child, by the value's
    // address, and multiplies each sum once.
    fn weight(var: VarId, label: LeafLabel) -> u128 {
        match label {
            LeafLabel::Zero => 0,
            LeafLabel::Neg => 1,
            LeafLabel::Pos => 2 + var.0 as u128,
            LeafLabel::One => 3 + var.0 as u128,
        }
    }
    struct Plain;
    impl EvalAlgebra for Plain {
        type Value = u128;
        fn zero(&self) -> u128 { 0 }
        fn leaf(&self, var: VarId, label: LeafLabel) -> u128 { weight(var, label) }
        fn add_assign(&self, a: &mut u128, b: &u128) { *a += b; }
        fn mul(&self, a: &u128, b: &u128) -> u128 { a * b }
    }
    struct Regrouped { pairs: Cell<usize>, products: Cell<usize> }
    impl EvalAlgebra for Regrouped {
        type Value = u128;
        fn zero(&self) -> u128 { 0 }
        fn leaf(&self, var: VarId, label: LeafLabel) -> u128 { weight(var, label) }
        fn add_assign(&self, a: &mut u128, b: &u128) { *a += b; }
        fn mul(&self, a: &u128, b: &u128) -> u128 { a * b }
        fn sum_of_products<'v>(&self, pairs: impl ExactSizeIterator<Item = (&'v u128, &'v u128)>) -> u128 {
            self.pairs.set(self.pairs.get() + pairs.len());
            let mut by_right: Vec<(&u128, u128)> = Vec::new();
            for (a, b) in pairs {
                match by_right.iter_mut().find(|(right, _)| std::ptr::eq(*right, b)) {
                    Some((_, sum)) => *sum += a,
                    None => by_right.push((b, *a)),
                }
            }
            self.products.set(self.products.get() + by_right.len());
            by_right.iter().map(|&(b, sum)| sum * b).sum()
        }
    }
    let eng = Engine::new();
    let mut rng = crate::vtree::rng::Lcg::new(29);
    let vars: Vec<VarId> = (1..=12).map(VarId).collect();
    for tree in [Vtree::balanced(14), Vtree::linear(14), Vtree::random(14, 3)] {
        let vtree = Arc::new(tree);
        for round in 0..6u64 {
            // Few values of the last variables, so that right children repeat.
            let rows: Vec<u64> = (0..200 + 150 * round)
                .map(|_| (rng.next_u64() & 0x3ff) | (rng.next_u64() % 3) << 10)
                .collect();
            let f = eng.from_models(&vtree, &vars, &rows).unwrap();
            assert_canonical(&f);
            let regrouped = Regrouped { pairs: Cell::new(0), products: Cell::new(0) };
            let value = eng.evaluate(&f, &regrouped).unwrap();
            assert!(regrouped.pairs.get() > 0, "the fold sums its products through sum_of_products");
            assert_eq!(value, eng.evaluate(&f, &Plain).unwrap(), "round {round}");
        }
    }
}

#[test]
fn an_overriding_mul_add_is_used_and_agrees_with_the_default() {
    // Model counts with a leaf `One` worth two: the fold's sum of products.
    struct Counts { fused: Cell<usize> }
    impl EvalAlgebra for Counts {
        type Value = u128;
        fn zero(&self) -> u128 { 0 }
        fn leaf(&self, _: VarId, label: LeafLabel) -> u128 {
            match label { LeafLabel::Zero => 0, LeafLabel::One => 2, _ => 1 }
        }
        fn add_assign(&self, a: &mut u128, b: &u128) { *a += b; }
        fn mul(&self, a: &u128, b: &u128) -> u128 { a * b }
        fn mul_add(&self, acc: &mut u128, a: &u128, b: &u128) {
            self.fused.set(self.fused.get() + 1);
            *acc += a * b;
        }
    }
    struct Plain;
    impl EvalAlgebra for Plain {
        type Value = u128;
        fn zero(&self) -> u128 { 0 }
        fn leaf(&self, _: VarId, label: LeafLabel) -> u128 {
            match label { LeafLabel::Zero => 0, LeafLabel::One => 2, _ => 1 }
        }
        fn add_assign(&self, a: &mut u128, b: &u128) { *a += b; }
        fn mul(&self, a: &u128, b: &u128) -> u128 { a * b }
    }
    let eng = Engine::new();
    for tree in [Vtree::balanced(8), Vtree::linear(8)] {
        let vtree = Arc::new(tree);
        let f = Tdd::clause(&vtree, [1, 3, -5]).unwrap() & Tdd::clause(&vtree, [-2, 6]).unwrap();
        assert_canonical(&f);
        let fused = Counts { fused: Cell::new(0) };
        let value = eng.evaluate(&f, &fused).unwrap();
        assert!(fused.fused.get() > 0, "the fold accumulates through mul_add");
        assert_eq!(value, eng.evaluate(&f, &Plain).unwrap());
        assert_eq!(crate::test_helpers::rat(value as i64, 1), eng.evaluate(&f, &RationalWeights::unit(8)).unwrap());
    }
}
