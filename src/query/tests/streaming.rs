//! The streaming fold against the materialized count.
//!
//! The fixtures these read are in `mod.rs`.

use super::*;

use crate::engine::Engine;
use crate::apply::conjoin::apply_and;
use crate::apply::conjoin::targets::MarginalTargets;
use crate::test_helpers::clause_to_tdd;
use crate::build::constant_one;
use crate::test_helpers::{literals, rand_cnf, CnfShape, Lcg};
use crate::vtree::{Vtree, VtreeIdx};
use crate::diagram::Tdd;
use num_bigint::BigUint;
use std::sync::Arc;


#[test]
fn streaming_fold_count_matches_materialized_randomized() {
    let eng = Engine::new();
    // The apply's streaming-marginal fold with all interior vtree levels as
    // targets (the terminal `#F` count discipline — the apply collapses each level
    // to Σ left_count × right_count, never keeping the full product) must equal the
    // materialize-then-count oracle. This exercises the churn-free collapse-at-source
    // path (`run_level_rows_stream_count`) in both its guises: dense lookups at the
    // lowest levels whose children are leaves, marginal lookups above them.
    // Count-identity is the regression arbiter for the streaming fold; it must hold
    // by construction.
    use crate::apply::conjoin::apply_and_fallible;
    let mut rng = Lcg::new(0x0bad_c0de_1337_f00d);
    let mut checked = 0u32;
    let mut nonzero = 0u32;
    for &nvars in &[2u32, 3, 4, 5, 6] {
        let vtree = Arc::new(Vtree::balanced(nvars));
        // All interior vtree levels marginalized — mirrors the fused-terminal
        // `ws_fuse_targets` (`!is_leaf(i)`), so every interior level streams.
        let targets: Vec<bool> =
            (0..vtree.num_nodes()).map(|i| !vtree.node(VtreeIdx(i as u32)).is_leaf()).collect();
        // Conjoined without an intervening minimize: the fold under test must
        // agree with the materialized count on the diagrams the apply produces,
        // not only on reduced ones.
        let rand_fn = |rng: &mut Lcg| -> Tdd {
            let shape = CnfShape { clauses: 4, width: nvars as usize };
            let mut acc = constant_one(&eng, &vtree);
            for clause in rand_cnf(rng, nvars, shape) {
                let cl = clause_to_tdd(&eng, &vtree, &literals(&clause));
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
                model_count(&apply_and_fallible(&eng, &mut a_o, &mut b_o, MarginalTargets::None).unwrap())
            };
            let fold = {
                let mut a_f = a.clone();
                let mut b_f = b.clone();
                model_count(&apply_and_fallible(&eng, &mut a_f, &mut b_f, MarginalTargets::At(&targets)).unwrap())
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

/// Weighted twin of `streaming_fold_count_matches_materialized_randomized`. With a weight context
/// active (random small positive rational per-literal weights) and every
/// interior vtree level as a marginalization target, the streamed weighted output
/// value must equal the materialize-then-evaluate oracle.
///
/// The path exercised is the weighted collapse-at-source fold (`WeightFold` on
/// the shared walker), so this parity check is the primary correctness
/// evidence for weighted collapse.
///
/// A fresh weight store is attached before each of the two
/// `apply_and_fallible` calls and read back right after, per trial, so no
/// per-level state leaks between trials.
///
/// Fewer formulas per variable count than the integer twin: `BigRational`
/// arithmetic costs more per apply, and one fold/oracle mismatch fails the
/// check.
#[test]
fn streaming_fold_weighted_matches_materialized_randomized() {
    let eng = Engine::new();
    use crate::query::weighted_value;
    use crate::diagram::WeightStore;
    use crate::apply::conjoin::apply_and_fallible;
    use crate::diagram::{LiteralWeights, RationalWeights};
    use crate::diagram::Arithmetic;
    use crate::test_helpers::{exact_weight, rat};
    use num_rational::BigRational;
    use num_traits::Zero;

    let mut rng = Lcg::new(0x5eed_5eed_c0ff_ee11);
    let mut checked = 0u32;
    let mut nonzero = 0u32;
    for &nvars in &[2u32, 3, 4, 5, 6] {
        let vtree = Arc::new(Vtree::balanced(nvars));
        let targets: Vec<bool> =
            (0..vtree.num_nodes()).map(|i| !vtree.node(VtreeIdx(i as u32)).is_leaf()).collect();
        // Conjoined without an intervening minimize: the fold under test must
        // agree with the materialized count on the diagrams the apply produces,
        // not only on reduced ones.
        let rand_fn = |rng: &mut Lcg| -> Tdd {
            let shape = CnfShape { clauses: 4, width: nvars as usize };
            let mut acc = constant_one(&eng, &vtree);
            for clause in rand_cnf(rng, nvars, shape) {
                let cl = clause_to_tdd(&eng, &vtree, &literals(&clause));
                acc = apply_and(acc, cl);
            }
            acc
        };
        // Random small strictly-positive rational literal weights: a weighted
        // value of 0 can then only mean model_count == 0 (no zero-cancellation
        // from a negative/zero weight muddying the fold-vs-oracle comparison).
        let rand_weight = |rng: &mut Lcg| -> BigRational {
            rat(1 + rng.below(9) as i64, 1 + rng.below(9) as i64)
        };
        for _ in 0..50 {
            let a = rand_fn(&mut rng);
            let b = rand_fn(&mut rng);
            // Per-variable (w_neg, w_pos) weights, shared by both branches below
            // so oracle and fold evaluate the same weighted function.
            let weights: Vec<LiteralWeights<BigRational>> = (0..nvars)
                .map(|_| LiteralWeights { negative: rand_weight(&mut rng), positive: rand_weight(&mut rng) })
                .collect();

            let store = || {
                WeightStore::new(
                    RationalWeights::from_literals(&weights),
                    Arithmetic::ExactRational,
                )
            };
            let oracle = {
                let mut a_o = a.clone();
                let mut b_o = b.clone();
                a_o.set_weights(store()).unwrap();
                let result = apply_and_fallible(&eng, &mut a_o, &mut b_o, MarginalTargets::None).unwrap();
                exact_weight(&weighted_value(&result).expect("store follows the result"))
            };
            let fold = {
                let mut a_f = a.clone();
                let mut b_f = b.clone();
                a_f.set_weights(store()).unwrap();
                let result = apply_and_fallible(&eng, &mut a_f, &mut b_f, MarginalTargets::At(&targets)).unwrap();
                exact_weight(&weighted_value(&result).expect("store follows the result"))
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
