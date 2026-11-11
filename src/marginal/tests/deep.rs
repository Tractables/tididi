//! The cascade on a vtree deeper than a small thread stack holds frames.
//!
//! A cascade marginalizes every structural level under a target. Where a
//! conjunction carries one operand's levels unbuilt (the other operand is true
//! over them), or a pass names only a target high above its descendants,
//! those levels form a chain as long as the vtree is deep. Each test runs on
//! a thread whose stack is far smaller than one frame per level of such a
//! chain would need.

use std::sync::Arc;

use num_bigint::BigUint;

use crate::Engine;
use crate::vtree::{Vtree, VtreeIdx};

/// Levels below the target: far more than [`STACK`] holds a frame for each.
const DEPTH: u32 = 20_000;

/// The test thread's stack.
const STACK: usize = 512 << 10;

/// Runs `test` on a thread with a stack of [`STACK`] bytes.
fn on_small_stack(test: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .stack_size(STACK)
        .spawn(test)
        .expect("the thread starts")
        .join()
        .expect("the test passes");
}

/// `2^k`.
fn pow2(k: u32) -> BigUint {
    BigUint::from(1u32) << k
}

/// A pass whose one target is the root of a right-linear vtree marginalizes
/// every level of the spine under it.
#[test]
fn marginalize_root_of_deep_spine() {
    on_small_stack(|| {
        let engine = Engine::new();
        let vtree = Arc::new(Vtree::linear(DEPTH));
        let mut f = engine.clause(&vtree, 1..=DEPTH as i32).unwrap();
        engine.marginalize_levels(&mut f, &[vtree.root()]).unwrap();
        assert!(f.level(vtree.root()).is_marginal());
        assert_eq!(engine.model_count(&f).unwrap(), pow2(DEPTH) - 1u32);
    });
}

/// A conjunction with an operand that is true below its second level carries
/// the other operand's spine there unbuilt, and summing every level out
/// marginalizes that spine from the lowest level the conjunction builds.
#[test]
fn and_marginalizing_over_deep_unbuilt_spine() {
    on_small_stack(|| {
        let engine = Engine::new();
        let vtree = Arc::new(Vtree::linear(DEPTH));
        // (x2 ∨ … ∨ xn) ∧ (x1 ∨ x2)
        let f = engine.clause(&vtree, 2..=DEPTH as i32).unwrap();
        let g = engine.clause(&vtree, [1, 2]).unwrap();
        let targets: Vec<VtreeIdx> = vtree.internal_bottomup().map(|(t, _, _)| t).collect();
        let counted = engine.and_marginalizing(&f, &g, &targets).unwrap();
        assert!(counted.level(vtree.root()).is_marginal());
        // x2 true: any x1 and the rest; x2 false: x1, and not every one of
        // x3 … xn false.
        let models = pow2(DEPTH - 1) + pow2(DEPTH - 2) - 1u32;
        assert_eq!(engine.model_count(&counted).unwrap(), models);
    });
}
