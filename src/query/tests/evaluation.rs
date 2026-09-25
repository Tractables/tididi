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
