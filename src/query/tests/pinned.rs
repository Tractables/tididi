//! The incremental pinned counter against its full-precision oracle.
//!
//! The fixtures these read are in `mod.rs`.

use super::*;

use crate::engine::Engine;
use crate::apply::conjoin::apply_and;
use crate::build::{clause_to_tdd, constant_one};
use crate::reduce::minimize;
use crate::test_helpers::{literals, rand_cnf, CnfShape, Lcg};
use crate::vtree::{VarId, Vtree, VtreeIdx};
use crate::diagram::Tdd;
use num_bigint::BigUint;
use std::sync::Arc;


/// One fresh counter under retention policy `R`: pin, pass, read the root.
fn fresh_root<R: Retention>(
    eng: &Engine,
    tdd: &Tdd,
    convention: SeedConvention,
    pins: &[Option<bool>],
) -> BigUint {
    let mut c = IncrementalCounter::<R, Unevaluated>::new(eng, tdd, pins.len(), convention);
    for (v, &p) in pins.iter().enumerate() {
        c.set_pin(VarId(v as u32), p);
    }
    c.compute(eng, tdd).output_count(tdd)
}

/// Pins `IncrementalCounter` against its two `BigUint` oracles: `pinned_counts`
/// under both seed conventions.
///
/// Exercises both of the counter's entry points per formula: a `compute` from a
/// freshly-pinned state (checked against the oracle called with the same pins),
/// then a sequence of incremental steps that change exactly one variable's pin
/// and call `recompute`, re-checked against the oracle recomputed from scratch
/// under the updated pins each time. This pins the incremental contract itself,
/// not just the equivalent-to-full-recompute case.
#[test]
fn incremental_pinned_counter_matches_pinned_bigint_randomized() {
    let eng = Engine::new();
    let mut rng = Lcg::new(0xfeed_face_dead_1234);
    let mut checked = 0u32;
    let mut nonzero = 0u32;
    for &nvars in &[2u32, 3, 4, 5, 6] {
        let vtree = Arc::new(Vtree::balanced(nvars));
        // Conjoined without an intervening minimize: the counter must agree
        // with its oracle on the diagrams the apply produces, not only on
        // reduced ones.
        let rand_fn = |rng: &mut Lcg| -> Tdd {
            let shape = CnfShape { clauses: 4, width: nvars as usize };
            let mut acc = constant_one(&eng, &vtree);
            for clause in rand_cnf(rng, nvars, shape) {
                let cl = clause_to_tdd(&eng, &vtree, &literals(&clause));
                acc = apply_and(acc, cl);
            }
            acc
        };
        for _ in 0..100 {
            let tdd = rand_fn(&mut rng);
            // Structurally-zero diagrams have a sentinel output (local == u32::MAX)
            // and no count slot — `output_count` has an implicit `!is_zero` precondition,
            // which every production wrapper (`model_count`, `pinned_counts`)
            // enforces with an early return. Mirror that contract here; UNSAT-*under-
            // pins* formulas (count 0 with a real output node) are still exercised.
            if tdd.is_zero() {
                continue;
            }
            for convention in [SeedConvention::Free, SeedConvention::Fixed] {
                let mut pins: Vec<Option<bool>> = (0..nvars)
                    .map(|_| match rng.below(3) {
                        0 => None,
                        1 => Some(true),
                        _ => Some(false),
                    })
                    .collect();
                // `KeepAllColumns`: the incremental dirty-cone half of this test
                // re-reads cached child columns.
                let mut ctr = IncrementalCounter::<KeepAllColumns, Unevaluated>::new(
                    &eng,
                    &tdd,
                    nvars as usize,
                    convention,
                );
                for (v, &p) in pins.iter().enumerate() {
                    ctr.set_pin(VarId(v as u32), p);
                }
                let mut ctr = ctr.compute(&eng, &tdd);
                let expected = if convention == SeedConvention::Fixed {
                    pinned_counts(&tdd, &pins, SeedConvention::Fixed)
                } else {
                    pinned_counts(&tdd, &pins, SeedConvention::Free)
                };
                assert_eq!(
                    ctr.output_count(&tdd),
                    expected,
                    "nvars={nvars} convention={convention:?}: compute mismatch"
                );
                checked += 1;
                if expected != BigUint::ZERO {
                    nonzero += 1;
                }

                // Incremental steps: change one variable's pin, recompute, and
                // re-check against a from-scratch oracle call under the updated
                // pins.
                for _ in 0..8 {
                    let v = rng.below(u64::from(nvars)) as u32;
                    let new_pin = match rng.below(3) {
                        0 => None,
                        1 => Some(true),
                        _ => Some(false),
                    };
                    pins[v as usize] = new_pin;
                    ctr.set_pin(VarId(v), new_pin);
                    ctr.recompute(&eng, &tdd);

                    let expected = if convention == SeedConvention::Fixed {
                        pinned_counts(&tdd, &pins, SeedConvention::Fixed)
                    } else {
                        pinned_counts(&tdd, &pins, SeedConvention::Free)
                    };
                    assert_eq!(
                        ctr.output_count(&tdd),
                        expected,
                        "nvars={nvars} convention={convention:?}: recompute mismatch after changing var {v}"
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

/// A diagram over six variables in which every variable matters, so that a
/// stale count anywhere between a changed leaf and the root shows at the
/// output: the chain of implications x1→x2→…→x6 with one extra clause.
fn six_var_diagram(eng: &Engine, vtree: &Arc<Vtree>) -> Tdd {
    let clauses: [Vec<i32>; 6] = [
        vec![-1, 2], vec![-2, 3], vec![-3, 4], vec![-4, 5], vec![-5, 6], vec![1, 3, 5],
    ];
    let mut acc = constant_one(eng, vtree);
    for clause in &clauses {
        acc = apply_and(acc, clause_to_tdd(eng, vtree, &literals(clause)));
    }
    minimize(&mut acc);
    assert!(!acc.is_zero(), "the fixture must be satisfiable");
    acc
}

/// Two pins changed between two recounts: the second `recompute` must fold
/// the union of both cones, and the count after each step must equal the
/// oracle under the pins then in force.
#[test]
fn recompute_after_two_pin_changes_matches_oracle() {
    let eng = Engine::new();
    let vtree = Arc::new(Vtree::balanced(6));
    let tdd = six_var_diagram(&eng, &vtree);
    for convention in [SeedConvention::Free, SeedConvention::Fixed] {
        let mut pins: Vec<Option<bool>> = vec![None; 6];
        let mut ctr = IncrementalCounter::<KeepAllColumns, Unevaluated>::new(&eng, &tdd, 6, convention)
            .compute(&eng, &tdd);
        assert_eq!(ctr.output_count(&tdd), pinned_counts(&tdd, &pins, convention));

        // Two leaves in different halves of the tree, so the cones share only
        // the root.
        pins[0] = Some(false);
        pins[5] = Some(true);
        ctr.set_pin(VarId(0), Some(false));
        ctr.set_pin(VarId(5), Some(true));
        ctr.recompute(&eng, &tdd);
        assert_eq!(
            ctr.output_count(&tdd),
            pinned_counts(&tdd, &pins, convention),
            "convention={convention:?}: two pins changed in different subtrees"
        );

        // Two leaves under one parent, and one of them changed twice: the
        // cone is folded once from the pins in force at the recompute.
        pins[1] = Some(false);
        pins[2] = Some(true);
        ctr.set_pin(VarId(1), Some(true));
        ctr.set_pin(VarId(1), Some(false));
        ctr.set_pin(VarId(2), Some(true));
        ctr.recompute(&eng, &tdd);
        assert_eq!(
            ctr.output_count(&tdd),
            pinned_counts(&tdd, &pins, convention),
            "convention={convention:?}: two pins changed under one parent"
        );
        assert!(
            ctr.output_count(&tdd) != BigUint::ZERO,
            "convention={convention:?}: the fixture must stay satisfiable under these pins"
        );
    }
}

/// A pin set to the value it already holds records no change: `recompute`
/// then folds nothing, and the count it leaves is the count it found — the
/// oracle's, since the pins did not move. Then a pin changed and changed back
/// before the recount still counts as changed and is folded, so the count
/// returns to the oracle's under the original pins.
#[test]
fn pin_reset_to_same_value_records_nothing() {
    let eng = Engine::new();
    let vtree = Arc::new(Vtree::balanced(6));
    let tdd = six_var_diagram(&eng, &vtree);
    let pins: Vec<Option<bool>> = vec![Some(true), None, Some(false), None, None, None];
    let mut ctr = IncrementalCounter::<KeepAllColumns, Unevaluated>::new(&eng, &tdd, 6, SeedConvention::Fixed);
    for (v, &p) in pins.iter().enumerate() {
        ctr.set_pin(VarId(v as u32), p);
    }
    let mut ctr = ctr.compute(&eng, &tdd);
    let expected = pinned_counts(&tdd, &pins, SeedConvention::Fixed);
    assert_eq!(ctr.output_count(&tdd), expected);
    assert!(format!("{ctr:?}").contains("changed_since_pass: 0"), "compute clears the change set: {ctr:?}");

    ctr.set_pin(VarId(0), Some(true));
    ctr.set_pin(VarId(1), None);
    assert!(format!("{ctr:?}").contains("changed_since_pass: 0"), "a same-value pin is no change: {ctr:?}");
    ctr.recompute(&eng, &tdd);
    assert_eq!(ctr.output_count(&tdd), expected, "nothing changed, nothing moves");

    ctr.set_pin(VarId(0), Some(false));
    ctr.set_pin(VarId(0), Some(true));
    assert!(format!("{ctr:?}").contains("changed_since_pass: 1"), "a changed-and-restored pin is recorded: {ctr:?}");
    ctr.recompute(&eng, &tdd);
    assert_eq!(ctr.output_count(&tdd), expected, "the original pins give the original count");
    assert!(format!("{ctr:?}").contains("changed_since_pass: 0"), "recompute clears the change set: {ctr:?}");
}

/// Differential guard for the structured-count boundary readout: the u128-hybrid
/// pinned counter must equal the full-precision `BigUint` oracle on diagrams that
/// carry MARGINAL levels, not just fully-explicit ones — the regime the
/// downstream driver's structured-count component-boundary function actually
/// runs in (own-show leaves marginalized, boundary vars left Boolean and
/// pinned one assignment at a time).
/// The readout is routed through the hybrid counter instead of `pinned_counts` under the FIX convention,
/// so this equality is the soundness contract of that routing, per convention
/// (`fix=true` = clean-fix ×1, `fix=false` = freed ×2).
///
/// Also pins the two things that routing relies on beyond value equality:
/// - `KeepFrontier` (children freed as parents complete) yields the
///   same root count as `KeepAllColumns`, on diagrams whose levels include a
///   marginal one (whose column comes from its summed store and whose own parent
///   reads it as a marginal child);
/// - one `Frontier` counter REUSED across successive pin assignments — the readout's
///   loop shape — agrees with a freshly constructed counter each time.
#[test]
fn pinned_hybrid_matches_bigint_on_marginalized_diagrams() {
    let eng = Engine::new();
    use crate::test_helpers::marginalize_subtree;

    let mut rng = Lcg::new(0x5eed_1234_abcd_0f0f);
    let mut checked = 0u32;
    let mut nonzero = 0u32;
    let mut with_marginal = 0u32;

    for &nvars in &[3u32, 4, 5, 6] {
        let vtree = Arc::new(Vtree::balanced(nvars));
        // A NON-root internal node: marginalizing its subtree leaves the levels
        // above it explicit, so a fold reads a marginal child (and the pinned
        // vars split into "summed out below the marginal root" and "still Boolean").
        let Some(marginal_root) = (0..vtree.num_nodes())
            .find(|&vi| !vtree.node(VtreeIdx(vi as u32)).is_leaf() && vi != vtree.root().idx())
            .map(|vi| VtreeIdx(vi as u32))
        else {
            continue;
        };

        for _ in 0..40 {
            // Random conjunction of clauses, same shape as the explicit-diagram
            // differential test above.
            let mut tdd = {
                let shape = CnfShape { clauses: 4, width: nvars as usize };
                let mut acc = constant_one(&eng, &vtree);
                for clause in rand_cnf(&mut rng, nvars, shape) {
                    let cl = clause_to_tdd(&eng, &vtree, &literals(&clause));
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
            marginalize_subtree(&mut tdd, marginal_root);
            minimize(&mut tdd);
            // `output_count` has an implicit `!is_zero` precondition (a structurally
            // zero diagram has no output count slot); every production wrapper
            // early-returns on it.
            if tdd.is_zero() {
                continue;
            }
            if tdd.levels[marginal_root.idx()].is_marginal() {
                with_marginal += 1;
            }

            for convention in [SeedConvention::Free, SeedConvention::Fixed] {
                // One reused Frontier counter for the whole pin sweep — the
                // structured-count readout's exact shape.
                let mut reused = IncrementalCounter::<KeepFrontier, Unevaluated>::new(
                    &eng,
                    &tdd,
                    nvars as usize,
                    convention,
                )
                .compute(&eng, &tdd);
                for _ in 0..4 {
                    let pins: Vec<Option<bool>> = (0..nvars)
                        .map(|_| match rng.below(3) {
                            0 => None,
                            1 => Some(true),
                            _ => Some(false),
                        })
                        .collect();
                    let expected = if convention == SeedConvention::Fixed {
                        pinned_counts(&tdd, &pins, SeedConvention::Fixed)
                    } else {
                        pinned_counts(&tdd, &pins, SeedConvention::Free)
                    };

                    for (v, &p) in pins.iter().enumerate() {
                        reused.set_pin(VarId(v as u32), p);
                    }
                    reused = reused.compute(&eng, &tdd);
                    assert_eq!(
                        reused.output_count(&tdd),
                        expected,
                        "nvars={nvars} convention={convention:?}: reused Frontier counter disagrees with the \
                         BigUint oracle on a marginalized diagram"
                    );

                    // Unevaluated counters, both retention policies: the policy must
                    // be value-neutral, and a fresh frontier pass must match the
                    // reused one (no state carried between assignments).
                    assert_eq!(
                        fresh_root::<KeepAllColumns>(&eng, &tdd, convention, &pins),
                        expected,
                        "nvars={nvars} convention={convention:?}: a fresh whole-array counter disagrees \
                         with the BigUint oracle on a marginalized diagram"
                    );
                    assert_eq!(
                        fresh_root::<KeepFrontier>(&eng, &tdd, convention, &pins),
                        expected,
                        "nvars={nvars} convention={convention:?}: a fresh frontier counter disagrees \
                         with the BigUint oracle on a marginalized diagram"
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
        with_marginal > 0,
        "no fixture ended up with a marginal level — the marginal-child fold path was never exercised"
    );
    assert!(
        nonzero > 0,
        "test built only UNSAT-under-pins formulas — the counting path was never exercised"
    );
}
