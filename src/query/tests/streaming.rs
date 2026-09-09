//! The streaming fold against the materialized count.
//!
//! Sibling of `query_tests.rs`, which holds the fixtures these read.

use super::*;

use crate::engine::Engine;
use crate::apply::conjoin::apply_and;
use crate::apply::conjoin::targets::MargTargets;
use crate::build::{clause_to_tdd, constant_one};
use crate::diagram::Literal;
use crate::vtree::{VarId, Vtree, VtreeIdx};
use crate::diagram::Tdd;
use num_bigint::BigUint;
use std::sync::Arc;


#[test]
fn streaming_fold_count_matches_materialized_randomized() {
    let eng = Engine::new();
    // The apply's streaming-marginal fold with ALL interior vtree levels as
    // targets (the terminal `#F` count discipline — the apply collapses each level
    // to Σ left_count × right_count, never keeping the full product) must equal the
    // materialize-then-count oracle. This exercises the churn-free collapse-at-source
    // path (`run_level_rows_stream_count`) in both its guises: dense lookups at the
    // lowest levels whose children are leaves, marg lookups above them.
    // Count-identity is the regression arbiter for the streaming fold; it must hold
    // by construction.
    use crate::apply::conjoin::apply_and_fallible;
    let mut state: u64 = 0x0bad_c0de_1337_f00d;
    let mut rng = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        state >> 33
    };
    let mut checked = 0u32;
    let mut nonzero = 0u32;
    for &nvars in &[2u32, 3, 4, 5, 6] {
        let vtree = Arc::new(Vtree::balanced(nvars));
        // All interior vtree levels marginalized — mirrors the fused-terminal
        // `ws_fuse_targets` (`!is_leaf(i)`), so every interior level streams.
        let targets: Vec<bool> =
            (0..vtree.num_nodes()).map(|i| !vtree.node(VtreeIdx(i as u32)).is_leaf()).collect();
        let rand_fn = |rng: &mut dyn FnMut() -> u64| -> Tdd {
            let nclauses = 1 + (rng() % 4) as usize;
            let mut acc = constant_one(&eng, &vtree);
            for _ in 0..nclauses {
                let width = 1 + (rng() % nvars as u64) as usize;
                let mut lits: Vec<Literal> = Vec::new();
                let mut seen = vec![false; nvars as usize];
                for _ in 0..width {
                    let v = (rng() % nvars as u64) as u32;
                    if seen[v as usize] {
                        continue;
                    }
                    seen[v as usize] = true;
                    let pol = rng().is_multiple_of(2);
                    lits.push(if pol {
                        Literal::pos(VarId(v))
                    } else {
                        Literal::neg(VarId(v))
                    });
                }
                if lits.is_empty() {
                    continue;
                }
                let cl = clause_to_tdd(&eng, &vtree, &lits);
                acc = apply_and(acc, cl);
            }
            acc
        };
        for _ in 0..200 {
            let a = rand_fn(&mut rng);
            let b = rand_fn(&mut rng);
            // apply_and_fallible mutates (drops) its operands — clone per call.
            let oracle = {
                let mut a_o = a.clone();
                let mut b_o = b.clone();
                model_count(&apply_and_fallible(&eng, &mut a_o, &mut b_o, MargTargets::None).unwrap())
            };
            let fold = {
                let mut a_f = a.clone();
                let mut b_f = b.clone();
                model_count(&apply_and_fallible(&eng, &mut a_f, &mut b_f, MargTargets::At(&targets)).unwrap(),
                )
            };
            assert_eq!(fold, oracle, "nvars={nvars}: streaming fold != materialized");
            checked += 1;
            if oracle != BigUint::ZERO {
                nonzero += 1;
            }
        }
    }
    assert!(checked > 0);
    assert!(
        nonzero > 0,
        "test built only UNSAT formulas — the fold path was never exercised"
    );
}

/// WEIGHTED twin of `streaming_fold_count_matches_materialized_randomized` (D2
/// design stage T.1). With a weight context
/// active (random small positive rational per-literal weights) and every
/// interior vtree level as a marginalize target, the streamed weighted output
/// value must equal the materialize-then-evaluate oracle.
///
/// Written (stage T) against the pre-stage-1 materialize+snap weighted route;
/// since D2 stage 1 the default path here is the weighted collapse-at-source
/// fold (`WeightFold` on the shared walker — `marg_stream_collapse` /
/// Route B no longer exclude weighted), so this parity check is now the
/// primary correctness evidence for weighted collapse. The snap route it
/// originally pinned stays covered via the gate-off twin below until D2
/// stage 2 deletes it.
///
/// A fresh weight store is attached immediately before each of the two
/// `apply_and_fallible` calls (oracle, fold) and read back right after, per
/// trial, so this can't leak stale per-level state from a previous trial/nvars
/// into `ensure_weights`'s `ws.is_set(i)` "already computed" check (which trusts
/// the store to exactly mirror the CURRENT diagram's marginal levels).
///
/// 50 formulas/nvars, not 200 like the integer twin: `BigRational` arithmetic
/// under the weighted marg path costs materially more per apply than the
/// integer count path, and this parity check doesn't need the larger sample to
/// be discriminating — any single fold/oracle mismatch fails it.
#[test]
fn streaming_fold_weighted_matches_materialized_randomized() {
    let eng = Engine::new();
    use crate::marginal::weighted_value;
    use crate::diagram::WeightStore;
    use crate::apply::conjoin::apply_and_fallible;
    use crate::diagram::{RationalWeights, WeightVal};
    use crate::diagram::Arithmetic;
    use num_bigint::BigInt;
    use num_rational::BigRational;
    use num_traits::Zero;

    // Every store in this test is built `Arithmetic::ExactRational` — `WeightVal::Log`
    // should never appear here.
    fn weight_to_exact(v: &WeightVal) -> BigRational {
        match v {
            WeightVal::Log(_) => panic!(
                "expected exact-mode WeightVal (log mode not installed) in this test"
            ),
            v => v.as_rational().into_owned(),
        }
    }

    let mut state: u64 = 0x5eed_5eed_c0ff_ee11;
    let mut rng = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        state >> 33
    };
    let mut checked = 0u32;
    let mut nonzero = 0u32;
    for &nvars in &[2u32, 3, 4, 5, 6] {
        let vtree = Arc::new(Vtree::balanced(nvars));
        let targets: Vec<bool> =
            (0..vtree.num_nodes()).map(|i| !vtree.node(VtreeIdx(i as u32)).is_leaf()).collect();
        let rand_fn = |rng: &mut dyn FnMut() -> u64| -> Tdd {
            let nclauses = 1 + (rng() % 4) as usize;
            let mut acc = constant_one(&eng, &vtree);
            for _ in 0..nclauses {
                let width = 1 + (rng() % nvars as u64) as usize;
                let mut lits: Vec<Literal> = Vec::new();
                let mut seen = vec![false; nvars as usize];
                for _ in 0..width {
                    let v = (rng() % nvars as u64) as u32;
                    if seen[v as usize] {
                        continue;
                    }
                    seen[v as usize] = true;
                    let pol = rng().is_multiple_of(2);
                    lits.push(if pol {
                        Literal::pos(VarId(v))
                    } else {
                        Literal::neg(VarId(v))
                    });
                }
                if lits.is_empty() {
                    continue;
                }
                let cl = clause_to_tdd(&eng, &vtree, &lits);
                acc = apply_and(acc, cl);
            }
            acc
        };
        // Random small strictly-positive rational literal weights: a weighted
        // value of 0 can then only mean model_count == 0 (no zero-cancellation
        // from a negative/zero weight muddying the fold-vs-oracle comparison).
        let rand_weight = |rng: &mut dyn FnMut() -> u64| -> BigRational {
            let n = 1 + (rng() % 9);
            let d = 1 + (rng() % 9);
            BigRational::new(BigInt::from(n), BigInt::from(d))
        };
        for _ in 0..50 {
            let a = rand_fn(&mut rng);
            let b = rand_fn(&mut rng);
            // Per-variable (w_neg, w_pos) weights, shared by both branches below
            // so oracle and fold evaluate the SAME weighted function.
            let weights: Vec<(BigRational, BigRational)> = (0..nvars)
                .map(|_| (rand_weight(&mut rng), rand_weight(&mut rng)))
                .collect();

            let store = || {
                WeightStore::new(
                    RationalWeights::from_weights(&weights),
                    Arithmetic::ExactRational,
                )
            };
            let oracle = {
                let mut a_o = a.clone();
                let mut b_o = b.clone();
                a_o.set_weights(store());
                let result = apply_and_fallible(&eng, &mut a_o, &mut b_o, MargTargets::None).unwrap();
                weight_to_exact(&weighted_value(&result).expect("store follows the result"))
            };
            let fold = {
                let mut a_f = a.clone();
                let mut b_f = b.clone();
                a_f.set_weights(store());
                let result = apply_and_fallible(&eng, &mut a_f, &mut b_f, MargTargets::At(&targets)).unwrap();
                weight_to_exact(&weighted_value(&result).expect("store follows the result"))
            };
            assert_eq!(
                fold, oracle,
                "nvars={nvars}: weighted streaming fold != materialized oracle"
            );
            checked += 1;
            if !oracle.is_zero() {
                nonzero += 1;
            }
        }
    }
    assert!(checked > 0);
    assert!(
        nonzero > 0,
        "test built only UNSAT formulas — the weighted fold path was never exercised"
    );
}

/// GATE-OFF twin of `streaming_fold_count_matches_materialized_randomized` (D2
/// design stage T.2). Forces the streaming
/// gate off. Since D2 stage 2 the knob short-circuits
/// `stream_marginal_eligible` itself — gate-off means "don't stream at all"
/// (the old materialize+snap fallback was deleted): every level materializes
/// normally and the marginalize request is simply not honored inside the
/// apply (production callers batch-marginalize after; marginalization is
/// count-preserving, so `model_count` parity still pins the configuration).
/// Count-identity with the un-fused oracle must hold — the gate is a perf
/// toggle, never a semantics toggle.
///
/// The gate is flipped through `with_bothmarg_collapse_forced`, a thread-local
/// test-only override — safe under parallel test execution since each test runs
/// on its own thread.
#[test]
fn streaming_fold_count_matches_materialized_gate_off_randomized() {
    let eng = Engine::new();
    use crate::apply::conjoin::{apply_and_fallible, with_bothmarg_collapse_forced};
    with_bothmarg_collapse_forced(false, || {
        let mut state: u64 = 0x0bad_c0de_1337_f00d ^ 0xdead_beef_dead_beef;
        let mut rng = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state >> 33
        };
        let mut checked = 0u32;
        let mut nonzero = 0u32;
        for &nvars in &[2u32, 3, 4, 5, 6] {
            let vtree = Arc::new(Vtree::balanced(nvars));
            let targets: Vec<bool> =
                (0..vtree.num_nodes()).map(|i| !vtree.node(VtreeIdx(i as u32)).is_leaf()).collect();
            let rand_fn = |rng: &mut dyn FnMut() -> u64| -> Tdd {
                let nclauses = 1 + (rng() % 4) as usize;
                let mut acc = constant_one(&eng, &vtree);
                for _ in 0..nclauses {
                    let width = 1 + (rng() % nvars as u64) as usize;
                    let mut lits: Vec<Literal> = Vec::new();
                    let mut seen = vec![false; nvars as usize];
                    for _ in 0..width {
                        let v = (rng() % nvars as u64) as u32;
                        if seen[v as usize] {
                            continue;
                        }
                        seen[v as usize] = true;
                        let pol = rng().is_multiple_of(2);
                        lits.push(if pol {
                            Literal::pos(VarId(v))
                        } else {
                            Literal::neg(VarId(v))
                        });
                    }
                    if lits.is_empty() {
                        continue;
                    }
                    let cl = clause_to_tdd(&eng, &vtree, &lits);
                    acc = apply_and(acc, cl);
                }
                acc
            };
            for _ in 0..200 {
                let a = rand_fn(&mut rng);
                let b = rand_fn(&mut rng);
                let oracle = {
                    let mut a_o = a.clone();
                    let mut b_o = b.clone();
                    model_count(&apply_and_fallible(&eng, &mut a_o, &mut b_o, MargTargets::None).unwrap())
                };
                let fold = {
                    let mut a_f = a.clone();
                    let mut b_f = b.clone();
                    model_count(&apply_and_fallible(&eng, &mut a_f, &mut b_f, MargTargets::At(&targets)).unwrap(),
                    )
                };
                assert_eq!(
                    fold, oracle,
                    "nvars={nvars}: gate-off streaming fold != materialized"
                );
                checked += 1;
                if oracle != BigUint::ZERO {
                    nonzero += 1;
                }
            }
        }
        assert!(checked > 0);
        assert!(
            nonzero > 0,
            "test built only UNSAT formulas — the gate-off snap path was never exercised"
        );
    });
}
