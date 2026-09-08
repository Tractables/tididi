use super::*;
use super::sat::output_is_satisfiable;
use crate::tdd::transform::pairwise::conjoin::{apply_and, apply_and_fallible};
use crate::tdd::build::{clause_to_tdd, constant_one};
use crate::tdd::minimize::minimize;
use crate::vtree::{Literal, VarId, Vtree, VtreeIdx};
use crate::tdd::types::Tdd;
use num_bigint::BigUint;
use std::sync::Arc;

#[test]
fn test_model_count_constant_one() {
    let vtree = Arc::new(Vtree::balanced(3));
    let tdd = constant_one(&vtree);
    // 3 variables → 2^3 = 8 models
    assert_eq!(model_count(&tdd), BigUint::from(8u32));
}

#[test]
fn test_model_count_single_positive_literal() {
    let vtree = Arc::new(Vtree::balanced(3));
    let clause = vec![Literal::pos(VarId(0))];
    let tdd = clause_to_tdd(&vtree, &clause);
    // x0: satisfied when x0=1. 4 assignments for x1,x2 → 4 models
    assert_eq!(model_count(&tdd), BigUint::from(4u32));
}

#[test]
fn test_model_count_two_literal_clause() {
    let vtree = Arc::new(Vtree::balanced(3));
    // x0 ∨ ¬x1: satisfied unless x0=0 and x1=1
    let clause = vec![
        Literal::pos(VarId(0)),
        Literal::neg(VarId(1)),
    ];
    let tdd = clause_to_tdd(&vtree, &clause);
    // 8 - 2 = 6 models (2 assignments with x0=0,x1=1, times 2 for x2)
    assert_eq!(model_count(&tdd), BigUint::from(6u32));
}

#[test]
fn test_model_count_conjunction() {
    let vtree = Arc::new(Vtree::balanced(3));
    // (x0) ∧ (x1): both must be true, x2 free → 2 models
    let c1 = vec![Literal::pos(VarId(0))];
    let c2 = vec![Literal::pos(VarId(1))];
    let t1 = clause_to_tdd(&vtree, &c1);
    let t2 = clause_to_tdd(&vtree, &c2);
    let result = apply_and(t1, t2);
    assert_eq!(model_count(&result), BigUint::from(2u32));
}

#[test]
fn test_model_count_unsat() {
    let vtree = Arc::new(Vtree::balanced(1));
    // (x0) ∧ (¬x0) = UNSAT
    let c1 = vec![Literal::pos(VarId(0))];
    let c2 = vec![Literal::neg(VarId(0))];
    let t1 = clause_to_tdd(&vtree, &c1);
    let t2 = clause_to_tdd(&vtree, &c2);
    let result = apply_and(t1, t2);
    assert_eq!(model_count(&result), BigUint::ZERO);
    assert!(!is_sat(&result));
}

#[test]
fn test_model_count_single_var() {
    let vtree = Arc::new(Vtree::balanced(1));
    let tdd = constant_one(&vtree);
    assert_eq!(model_count(&tdd), BigUint::from(2u32));
}

#[test]
fn test_model_count_clause_all_vars() {
    // 4 variables, clause x0 ∨ x1 ∨ x2 ∨ x3
    // Unsatisfied only when all are 0: 2^4 - 1 = 15 models
    let vtree = Arc::new(Vtree::balanced(4));
    let clause = vec![
        Literal::pos(VarId(0)),
        Literal::pos(VarId(1)),
        Literal::pos(VarId(2)),
        Literal::pos(VarId(3)),
    ];
    let tdd = clause_to_tdd(&vtree, &clause);
    assert_eq!(model_count(&tdd), BigUint::from(15u32));
}

// --- reduced_tdd_size tests ---

#[test]
fn test_reduced_size_constant_one_all_reducible() {
    // constant_one TDD: every internal node has E = {(one_{t1}, one_{t2})} → all reducible.
    // A 3-var balanced vtree has 2 internal vtree nodes.
    // The root has 1 internal tdd-node with 1 pair; its child also has 1 with 1 pair.
    // Both are reducible → reduced_tdd_size == 0.
    let vtree = Arc::new(Vtree::balanced(3));
    let tdd = constant_one(&vtree);
    assert_eq!(reduced_tdd_size(&tdd), 0);
}

#[test]
fn test_reduced_size_irrelevant_variable() {
    // Clause (x0) over a 2-var balanced vtree with x0 and x1.
    // After building and minimizing, the root node has E = {(c_{x0}, one_{t2})} →
    // exactly 1 reducible node. reduced_size == tdd.size() - 1.
    let vtree = Arc::new(Vtree::balanced(2));
    let clause = vec![Literal::pos(VarId(0))];
    let mut tdd = clause_to_tdd(&vtree, &clause);
    minimize(&mut tdd);
    let size = tdd.size();
    assert!(size >= 1);
    assert_eq!(reduced_tdd_size(&tdd), size - 1);
}

#[test]
fn test_reduced_size_no_reducible_nodes() {
    // Pigeonhole PHP(2,1): x0 ∧ x1. Both variables are relevant on both sides
    // at the root, so no node has an unconstrained child subtree.
    let vtree = Arc::new(Vtree::balanced(2));
    let t0 = clause_to_tdd(&vtree, &vec![Literal::pos(VarId(0))]);
    let t1 = clause_to_tdd(&vtree, &vec![Literal::pos(VarId(1))]);
    let mut tdd = apply_and(t0, t1);
    minimize(&mut tdd);
    assert_eq!(reduced_tdd_size(&tdd), tdd.size());
}

#[test]
fn test_reduced_size_multi_pair_generalisation() {
    // apply_and of (x0 ∨ x1) and (¬x0 ∨ x1) on a linear(3) vtree.
    //
    // After minimize, twin contraction merges Pos_x0 and Neg_x0 into One_x0 at
    // the leaf level. The root node ends up with a single pair {(One_x0, c_t1)}
    // where One_x0 has model count 2 = 2^|vars(x0)|.
    let vtree = Arc::new(Vtree::linear(3));
    let c1 = vec![Literal::pos(VarId(0)), Literal::pos(VarId(1))];
    let c2 = vec![Literal::neg(VarId(0)), Literal::pos(VarId(1))];
    let t1 = clause_to_tdd(&vtree, &c1);
    let t2 = clause_to_tdd(&vtree, &c2);
    let mut tdd = apply_and(t1, t2);
    minimize(&mut tdd);
    // The root node is reducible: reduced size must be strictly smaller.
    assert!(reduced_tdd_size(&tdd) < tdd.size(),
        "expected reduction: size={}, rtdd={}",
        tdd.size(), reduced_tdd_size(&tdd));
}

// --- compute_node_counts ---

#[test]
fn test_compute_node_counts_basic() {
    let vtree = Arc::new(Vtree::balanced(3));
    let tdd = constant_one(&vtree);
    let counts = compute_node_counts(&tdd);
    // Output node should have count = 2^3 = 8
    let out_count = &counts[tdd.output.vtree.idx()][tdd.output.local.idx()];
    assert_eq!(*out_count, BigUint::from(8u32));
}

// --- Overflow tests ---
//
// `test_model_count_hybrid_agrees_with_biguint` moved to
// `tests/tdd_query_compile.rs` (drives compilation facilities that live only
// in the downstream driver crate, which `tididi` cannot depend on).

/// Differential invariant underpinning conditioning's false-output canonicalization:
/// `output_is_satisfiable(t)` MUST agree with `model_count(t) != 0` for every diagram,
/// including non-canonical ⊥ (structurally-false output node that still carries pairs).
/// The canonicalization relies on this equivalence to collapse a dead diagram to ZERO
/// without ever changing a live count. Covers SAT, plain UNSAT, and a multi-conjoin
/// UNSAT that produces an in-range (non-root-stale) false output.
#[test]
fn test_output_is_satisfiable_agrees_with_model_count() {
    let check = |t: &Tdd, what: &str| {
        let sat = output_is_satisfiable(t);
        let nonzero = model_count(t) != BigUint::ZERO;
        assert_eq!(sat, nonzero, "output_is_satisfiable disagrees with model_count>0 for {what}");
    };

    // SAT: tautology, single literal, satisfiable conjunction.
    check(&constant_one(&Arc::new(Vtree::balanced(3))), "constant_one");
    let vtree = Arc::new(Vtree::balanced(3));
    check(&clause_to_tdd(&vtree, &vec![Literal::pos(VarId(0))]), "single literal");
    {
        let t1 = clause_to_tdd(&vtree, &vec![Literal::pos(VarId(0))]);
        let t2 = clause_to_tdd(&vtree, &vec![Literal::pos(VarId(1))]);
        check(&apply_and(t1, t2), "x0 ∧ x1 (SAT)");
    }

    // UNSAT: direct contradiction.
    {
        let v1 = Arc::new(Vtree::balanced(1));
        let t1 = clause_to_tdd(&v1, &vec![Literal::pos(VarId(0))]);
        let t2 = clause_to_tdd(&v1, &vec![Literal::neg(VarId(0))]);
        check(&apply_and(t1, t2), "x0 ∧ ¬x0 (UNSAT)");
    }

    // UNSAT via a chain of conjoins over a wider vtree — exercises a deeper false output,
    // not just the root-stale-grid case the existing FALSE guard already caught.
    {
        let v = Arc::new(Vtree::balanced(4));
        let mut acc = clause_to_tdd(&v, &vec![Literal::pos(VarId(0))]);
        for lit in [
            Literal::pos(VarId(1)),
            Literal::pos(VarId(2)),
            Literal::neg(VarId(0)), // contradicts the seed → UNSAT
        ] {
            let step = clause_to_tdd(&v, &vec![lit]);
            acc = apply_and(acc, step);
        }
        check(&acc, "chained conjoin → UNSAT");
    }

    // Marginal level: its column comes from the summed counts (it reads no child
    // column at all), and its own parent reads it as a marginal child. Covers the
    // walk's column-release rule on both sides of a marginal level.
    {
        let v = Arc::new(Vtree::balanced(4));
        let marg_root = (0..v.num_nodes())
            .find(|&vi| !v.node(VtreeIdx(vi as u32)).is_leaf() && vi != v.root().idx())
            .map(|vi| VtreeIdx(vi as u32))
            .expect("balanced(4) has a non-root internal node");
        let t1 = clause_to_tdd(&v, &vec![Literal::pos(VarId(0)), Literal::pos(VarId(2))]);
        let t2 = clause_to_tdd(&v, &vec![Literal::neg(VarId(1)), Literal::pos(VarId(3))]);
        let mut t = apply_and(t1, t2);
        crate::tdd::test_helpers::marginalize_subtree(&mut t, marg_root);
        minimize(&mut t);
        assert!(
            t.levels[marg_root.idx()].is_marginal(),
            "fixture must carry a marginal level"
        );
        check(&t, "marginalized subtree (SAT)");
    }
}

/// T3 — operand-consumption contract of `apply_and_fallible`.
///
/// `apply_and_fallible` drains dead operand-child levels in place as its
/// bottom-up loop ascends (`drop_dead_operand_level`), so a COMPLETED
/// conjoin leaves both operands consumed — every level below each root has been
/// stolen. This is the guaranteed, observable half of the "operands are
/// consumed / unspecified after the call" contract documented on
/// `apply_and_fallible` (the `Err` path is even less specified: an early
/// budget/cap trip may ascend little and leave operands nearly intact — which is
/// exactly why callers must never reuse operands and must rebuild from a clone).
#[test]
fn test_apply_fallible_consumes_operands() {
    // Fold clauses into a TDD; every operand shares the same vtree Arc so the
    // conjoin's pointer-identical-vtree precondition holds.
    fn build(vtree: &Arc<Vtree>, clauses: &[&[i32]]) -> Tdd {
        let mut acc = constant_one(vtree);
        for lits in clauses {
            let clause: Vec<Literal> = lits
                .iter()
                .map(|&l| Literal::new(VarId(l.unsigned_abs() - 1), l > 0))
                .collect();
            let c = clause_to_tdd(vtree, &clause);
            acc = apply_and(acc, c);
        }
        acc
    }

    // Two multi-level functions over interleaved vars — a genuine multi-node
    // intermediate diagram, so the ascending loop drains many operand levels.
    let vtree = Arc::new(Vtree::balanced(14));
    let fa: &[&[i32]] = &[
        &[1, 8], &[2, 9], &[3, 10], &[4, 11], &[5, 12], &[6, 13], &[7, 14],
        &[-1, -9], &[-2, -10], &[-3, -11], &[-4, -12], &[-5, -13], &[-6, -14],
    ];
    let fb: &[&[i32]] = &[
        &[1, -8], &[2, -9], &[3, -10], &[4, -11], &[5, -12], &[6, -13], &[7, -14],
        &[8, 2], &[9, 3], &[10, 4], &[11, 5], &[12, 6], &[13, 7],
    ];

    let mut a = build(&vtree, fa);
    let mut b = build(&vtree, fb);
    let a_before = a.total_nodes();
    let b_before = b.total_nodes();
    assert!(a_before > 1 && b_before > 1, "operands should be multi-node to make consumption observable");

    // A completed (uncapped) conjoin: must succeed, and consume both operands.
    let result = apply_and_fallible(&mut a, &mut b, None);
    assert!(result.is_ok(), "uncapped conjoin should complete: {:?}", result.err());
    assert!(
        a.total_nodes() < a_before && b.total_nodes() < b_before,
        "a completed apply must consume both operands (drain levels below root) — \
         a: {a_before} -> {}, b: {b_before} -> {}",
        a.total_nodes(),
        b.total_nodes(),
    );
}

#[test]
fn streaming_fold_count_matches_materialized_randomized() {
    // The apply's streaming-marginal fold with ALL interior vtree levels as
    // targets (the terminal `#F` count discipline — the apply collapses each level
    // to Σ left_count × right_count, never keeping the full product) must equal the
    // materialize-then-count oracle. This exercises the churn-free collapse-at-source
    // path (`run_level_rows_stream_count`) in both its guises: dense lookups at the
    // lowest levels whose children are leaves, marg lookups above them.
    // Count-identity is the regression arbiter for the streaming fold; it must hold
    // by construction.
    use crate::tdd::transform::pairwise::conjoin::apply_and_fallible;
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
            let mut acc = constant_one(&vtree);
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
                    let pol = rng() % 2 == 0;
                    lits.push(if pol {
                        Literal::pos(VarId(v))
                    } else {
                        Literal::neg(VarId(v))
                    });
                }
                if lits.is_empty() {
                    continue;
                }
                let cl = clause_to_tdd(&vtree, &lits);
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
                model_count(&apply_and_fallible(&mut a_o, &mut b_o, None).unwrap())
            };
            let fold = {
                let mut a_f = a.clone();
                let mut b_f = b.clone();
                model_count(
                    &apply_and_fallible(&mut a_f, &mut b_f, Some(&targets)).unwrap(),
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
    use crate::tdd::transform::unary::marginalize::weighted_value;
    use crate::tdd::weight_store::WeightStore;
    use crate::tdd::transform::pairwise::conjoin::apply_and_fallible;
    use crate::tdd::query::{RationalSemiring, WeightVal};
    use crate::tdd::weight_store::Precision;
    use num_bigint::BigInt;
    use num_rational::BigRational;
    use num_traits::Zero;

    // Every store in this test is built `Precision::Exact` — `WeightVal::Log`
    // should never appear here.
    fn weight_to_exact(v: &WeightVal) -> BigRational {
        match v {
            WeightVal::Log(_) => panic!(
                "expected exact-mode WeightVal (log mode not installed) in this test"
            ),
            v => v.as_rational().into_owned(),
        }
    }

    let mut state: u64 = 0x5eed_5eed_c0ffee11;
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
            let mut acc = constant_one(&vtree);
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
                    let pol = rng() % 2 == 0;
                    lits.push(if pol {
                        Literal::pos(VarId(v))
                    } else {
                        Literal::neg(VarId(v))
                    });
                }
                if lits.is_empty() {
                    continue;
                }
                let cl = clause_to_tdd(&vtree, &lits);
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
                    RationalSemiring::from_weights(&weights),
                    Precision::Exact,
                )
            };
            let oracle = {
                let mut a_o = a.clone();
                let mut b_o = b.clone();
                a_o.attach_weights(store());
                let result = apply_and_fallible(&mut a_o, &mut b_o, None).unwrap();
                weight_to_exact(&weighted_value(&result).expect("store follows the result"))
            };
            let fold = {
                let mut a_f = a.clone();
                let mut b_f = b.clone();
                a_f.attach_weights(store());
                let result = apply_and_fallible(&mut a_f, &mut b_f, Some(&targets)).unwrap();
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
    use crate::tdd::transform::pairwise::conjoin::{apply_and_fallible, with_bothmarg_collapse_forced};
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
                let mut acc = constant_one(&vtree);
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
                        let pol = rng() % 2 == 0;
                        lits.push(if pol {
                            Literal::pos(VarId(v))
                        } else {
                            Literal::neg(VarId(v))
                        });
                    }
                    if lits.is_empty() {
                        continue;
                    }
                    let cl = clause_to_tdd(&vtree, &lits);
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
                    model_count(&apply_and_fallible(&mut a_o, &mut b_o, None).unwrap())
                };
                let fold = {
                    let mut a_f = a.clone();
                    let mut b_f = b.clone();
                    model_count(
                        &apply_and_fallible(&mut a_f, &mut b_f, Some(&targets)).unwrap(),
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

/// Regression ahead of the pinned-counter storage migration: `IncrementalPinnedCounter`
/// (query.rs ~438) had ZERO test coverage before this. Pins it against its two BigUint
/// oracles — `model_count_pinned_bigint` (the FREED/`fix=false` convention) and
/// `model_count_pinned_fix` (the FIX/`fix=true` convention) — confirmed by reading both
/// leaf-seed tables (`leaf_seed_big` / `leaf_seed_big_fix`, query.rs ~120-178) plus the
/// counter impl (query.rs ~438-522) before writing this test.
///
/// Exercises BOTH of the counter's entry points per formula: a `full_recompute` from a
/// freshly-pinned state (checked against the oracle called with the same pins), then a
/// sequence of incremental steps that flip exactly ONE variable's pin and call
/// `recompute_levels` on only the "dirty cone" — that variable's leaf vtree level
/// followed by its ancestors up to the root (children-before-parents, the order
/// `recompute_levels` requires) — re-checked against the oracle recomputed from scratch
/// under the updated pins each time. This pins the O(cone) incremental contract itself,
/// not just the equivalent-to-full-recompute case.
#[test]
fn incremental_pinned_counter_matches_pinned_bigint_randomized() {
    let mut state: u64 = 0xfeed_face_dead_1234;
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
        // Leaf VtreeIdx of each variable, for building each var's dirty cone below.
        let mut leaf_of = vec![VtreeIdx(0); nvars as usize];
        for (t, var) in vtree.leaf_bottomup() {
            leaf_of[var.idx()] = t;
        }
        let rand_fn = |rng: &mut dyn FnMut() -> u64| -> Tdd {
            let nclauses = 1 + (rng() % 4) as usize;
            let mut acc = constant_one(&vtree);
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
                    let pol = rng() % 2 == 0;
                    lits.push(if pol {
                        Literal::pos(VarId(v))
                    } else {
                        Literal::neg(VarId(v))
                    });
                }
                if lits.is_empty() {
                    continue;
                }
                let cl = clause_to_tdd(&vtree, &lits);
                acc = apply_and(acc, cl);
            }
            acc
        };
        for _ in 0..100 {
            let tdd = rand_fn(&mut rng);
            // Structurally-zero diagrams have a sentinel output (local == u32::MAX)
            // and NO count slot — `root_count` has an implicit `!is_zero` precondition,
            // which every production wrapper (`model_count`, `model_count_pinned*`)
            // enforces with an early return. Mirror that contract here; UNSAT-*under-
            // pins* formulas (count 0 with a real output node) are still exercised.
            if tdd.is_zero() {
                continue;
            }
            for &fix in &[false, true] {
                let mut pins: Vec<Option<bool>> = (0..nvars)
                    .map(|_| match rng() % 3 {
                        0 => None,
                        1 => Some(true),
                        _ => Some(false),
                    })
                    .collect();
                // `ColumnRetention::All`: the incremental dirty-cone half of
                // this test re-reads cached child columns.
                let mut ctr = IncrementalPinnedCounter::new_with_fix(
                    &tdd,
                    nvars as usize,
                    fix,
                    ColumnRetention::All,
                );
                for (v, &p) in pins.iter().enumerate() {
                    ctr.set_pin(VarId(v as u32), p);
                }
                ctr.full_recompute(&tdd);
                let expected = if fix {
                    model_count_pinned_fix(&tdd, &pins)
                } else {
                    model_count_pinned_bigint(&tdd, &pins)
                };
                assert_eq!(
                    ctr.root_count(&tdd),
                    expected,
                    "nvars={nvars} fix={fix}: full_recompute mismatch"
                );
                checked += 1;
                if expected != BigUint::ZERO {
                    nonzero += 1;
                }

                // Incremental steps: flip one variable's pin, recompute only its
                // dirty cone (leaf level + ancestors to the root), and re-check
                // against a from-scratch oracle call under the updated pins.
                for _ in 0..8 {
                    let v = (rng() % nvars as u64) as u32;
                    let new_pin = match rng() % 3 {
                        0 => None,
                        1 => Some(true),
                        _ => Some(false),
                    };
                    pins[v as usize] = new_pin;
                    ctr.set_pin(VarId(v), new_pin);

                    let mut levels = vec![leaf_of[v as usize]];
                    let mut cur = leaf_of[v as usize];
                    while let Some(p) = tdd.vtree.node(cur).parent() {
                        levels.push(p);
                        cur = p;
                    }
                    ctr.recompute_levels(&tdd, &levels);

                    let expected = if fix {
                        model_count_pinned_fix(&tdd, &pins)
                    } else {
                        model_count_pinned_bigint(&tdd, &pins)
                    };
                    assert_eq!(
                        ctr.root_count(&tdd),
                        expected,
                        "nvars={nvars} fix={fix}: incremental dirty-cone mismatch after flipping var {v}"
                    );
                    checked += 1;
                    if expected != BigUint::ZERO {
                        nonzero += 1;
                    }
                }
            }
        }
    }
    assert!(checked > 0);
    assert!(
        nonzero > 0,
        "test built only UNSAT-under-pins formulas — the incremental counter path was never exercised"
    );
}

/// Differential guard for the structured-count boundary readout: the u128-hybrid
/// pinned counter must equal the full-precision `BigUint` oracle on diagrams that
/// carry MARGINAL levels, not just fully-explicit ones — the regime the
/// downstream driver's structured-count component-boundary function actually
/// runs in (own-show leaves marginalized, boundary vars left Boolean and
/// pinned one assignment at a time).
/// The readout is routed through the hybrid counter instead of `model_count_pinned_fix`,
/// so this equality is the soundness contract of that routing, per convention
/// (`fix=true` = clean-fix ×1, `fix=false` = freed ×2).
///
/// Also pins the two things that routing relies on beyond value equality:
/// - `ColumnRetention::Frontier` (children freed as parents complete) yields the
///   same root count as `ColumnRetention::All`, on diagrams whose levels include a
///   marginal one (whose column comes from its summed store and whose own parent
///   reads it as a marginal child);
/// - ONE `Frontier` counter REUSED across successive pin assignments — the readout's
///   loop shape — agrees with a freshly constructed counter each time.
#[test]
fn pinned_hybrid_matches_bigint_on_marginalized_diagrams() {
    use crate::tdd::test_helpers::marginalize_subtree;

    let mut state: u64 = 0x5eed_1234_abcd_0f0f;
    let mut rng = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        state >> 33
    };
    let mut checked = 0u32;
    let mut nonzero = 0u32;
    let mut with_marginal = 0u32;

    for &nvars in &[3u32, 4, 5, 6] {
        let vtree = Arc::new(Vtree::balanced(nvars));
        // A NON-root internal node: marginalizing its subtree leaves the levels
        // above it explicit, so a fold reads a marginal child (and the pinned
        // vars split into "summed out below the marg root" and "still Boolean").
        let Some(marg_root) = (0..vtree.num_nodes())
            .find(|&vi| !vtree.node(VtreeIdx(vi as u32)).is_leaf() && vi != vtree.root().idx())
            .map(|vi| VtreeIdx(vi as u32))
        else {
            continue;
        };

        for _ in 0..40 {
            // Random conjunction of clauses, same shape as the explicit-diagram
            // differential test above.
            let mut tdd = {
                let nclauses = 1 + (rng() % 4) as usize;
                let mut acc = constant_one(&vtree);
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
                        lits.push(if rng() % 2 == 0 {
                            Literal::pos(VarId(v))
                        } else {
                            Literal::neg(VarId(v))
                        });
                    }
                    if lits.is_empty() {
                        continue;
                    }
                    let cl = clause_to_tdd(&vtree, &lits);
                    acc = apply_and(acc, cl);
                }
                acc
            };
            if tdd.is_zero() {
                continue;
            }
            // `marginalize_subtree` skips width-0 internal levels, which would
            // then trip the marginal-parent precondition above them. Skip such
            // fixtures rather than trip an assert unrelated to what's under test.
            if (0..vtree.num_nodes())
                .any(|vi| !vtree.node(VtreeIdx(vi as u32)).is_leaf() && tdd.levels[vi].width() == 0)
            {
                continue;
            }
            marginalize_subtree(&mut tdd, marg_root);
            minimize(&mut tdd);
            // `root_count` has an implicit `!is_zero` precondition (a structurally
            // zero diagram has no output count slot); every production wrapper
            // early-returns on it.
            if tdd.is_zero() {
                continue;
            }
            if tdd.levels[marg_root.idx()].is_marginal() {
                with_marginal += 1;
            }

            for &fix in &[false, true] {
                // One reused Frontier counter for the whole pin sweep — the
                // structured-count readout's exact shape.
                let mut reused = IncrementalPinnedCounter::new_with_fix(
                    &tdd,
                    nvars as usize,
                    fix,
                    ColumnRetention::Frontier,
                );
                for _ in 0..4 {
                    let pins: Vec<Option<bool>> = (0..nvars)
                        .map(|_| match rng() % 3 {
                            0 => None,
                            1 => Some(true),
                            _ => Some(false),
                        })
                        .collect();
                    let expected = if fix {
                        model_count_pinned_fix(&tdd, &pins)
                    } else {
                        model_count_pinned_bigint(&tdd, &pins)
                    };

                    for (v, &p) in pins.iter().enumerate() {
                        reused.set_pin(VarId(v as u32), p);
                    }
                    reused.full_recompute(&tdd);
                    assert_eq!(
                        reused.root_count(&tdd),
                        expected,
                        "nvars={nvars} fix={fix}: reused Frontier counter disagrees with the \
                         BigUint oracle on a marginalized diagram"
                    );

                    // Fresh counters, both retention policies: the flag must be
                    // value-neutral, and a fresh Frontier pass must match the
                    // reused one (no state carried between assignments).
                    for retain in [ColumnRetention::All, ColumnRetention::Frontier] {
                        let mut fresh = IncrementalPinnedCounter::new_with_fix(
                            &tdd,
                            nvars as usize,
                            fix,
                            retain,
                        );
                        for (v, &p) in pins.iter().enumerate() {
                            fresh.set_pin(VarId(v as u32), p);
                        }
                        fresh.full_recompute(&tdd);
                        assert_eq!(
                            fresh.root_count(&tdd),
                            expected,
                            "nvars={nvars} fix={fix} retain={retain:?}: fresh counter disagrees \
                             with the BigUint oracle on a marginalized diagram"
                        );
                    }

                    checked += 1;
                    if expected != BigUint::ZERO {
                        nonzero += 1;
                    }
                }
            }
        }
    }
    assert!(checked > 0);
    assert!(
        with_marginal > 0,
        "no fixture ended up with a marginal level — the marginal-child fold path was never exercised"
    );
    assert!(
        nonzero > 0,
        "test built only UNSAT-under-pins formulas — the counting path was never exercised"
    );
}

// `incremental_pinned_counter_overflow_promotion_and_stale_clear` moved to
// `tests/tdd_query_compile.rs` (drives compilation facilities that live only
// in the downstream driver crate, which `tididi` cannot depend on).

