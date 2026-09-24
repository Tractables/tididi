//! The incremental pinned counter against its full-precision oracle.
//!
//! The fixtures these read are in `mod.rs`.

use super::*;

use crate::Engine;
use crate::apply::conjoin::apply_and;
use crate::test_helpers::clause_to_tdd;
use crate::build::constant_one;

use crate::test_helpers::{literals, rand_cnf, CnfShape, Lcg};
use crate::vtree::{VarId, Vtree, VtreeIdx};
use crate::diagram::Tdd;
use num_bigint::BigUint;
use std::sync::Arc;


/// Decide whether a variable lies beneath any summed-out level.
fn is_summed_out(tdd: &Tdd, var: VarId) -> bool {
    let mut level = tdd.vtree.leaf_of(var);
    while let Some(t) = level {
        if tdd.levels[t.idx()].is_marginal() { return true; }
        level = tdd.vtree.node(t).parent();
    }
    false
}

/// One fresh counter under retention policy `R`: pin, pass, read the root.
fn fresh_root(retention: Retention, 
    eng: &Engine,
    tdd: &Tdd,
    convention: PinSemantics,
    pins: &[Option<bool>],
) -> BigUint {
    let mut c = eng.counter_with(tdd, retention, convention).unwrap();
    for (v, &p) in pins.iter().enumerate() {
        if is_summed_out(tdd, VarId(v as u32 + 1)) { assert_eq!(p, None); continue; }
        c.set_pin(VarId(v as u32 + 1), p).unwrap();
    }
    c.model_count().unwrap()
}

/// Pins `ModelCounter` against its two `BigUint` oracles: `pinned_counts`
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
                let cl = clause_to_tdd(&vtree, &literals(&clause));
                acc = apply_and(acc, cl);
            }
            acc
        };
        for _ in 0..100 {
            let tdd = rand_fn(&mut rng);
            // Structurally-zero diagrams have a sentinel output (local == u32::MAX)
            // and no count slot — `model_count` has an implicit `!is_zero` precondition,
            // which every production wrapper (`model_count`, `pinned_counts`)
            // enforces with an early return. Mirror that contract here; UNSAT-*under-
            // pins* formulas (count 0 with a real output node) are still exercised.
            if tdd.is_zero() {
                continue;
            }
            for convention in [PinSemantics::Cofactor, PinSemantics::Evidence] {
                let mut pins: Vec<Option<bool>> = (0..nvars)
                    .map(|_| match rng.below(3) {
                        0 => None,
                        1 => Some(true),
                        _ => Some(false),
                    })
                    .collect();
                // `Retention::All`: the incremental dirty-cone half of this test
                // re-reads cached child columns.
                let mut ctr = eng.counter_with(&tdd, Retention::All, convention,
                ).unwrap();
                for (v, &p) in pins.iter().enumerate() {
                    ctr.set_pin(VarId(v as u32 + 1), p).unwrap();
                }

                let expected = if convention == PinSemantics::Evidence {
                    pinned_counts(&tdd, &pins, PinSemantics::Evidence)
                } else {
                    pinned_counts(&tdd, &pins, PinSemantics::Cofactor)
                };
                assert_eq!(
                    ctr.model_count().unwrap(),
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
                    ctr.set_pin(VarId(v + 1), new_pin).unwrap();


                    let expected = if convention == PinSemantics::Evidence {
                        pinned_counts(&tdd, &pins, PinSemantics::Evidence)
                    } else {
                        pinned_counts(&tdd, &pins, PinSemantics::Cofactor)
                    };
                    assert_eq!(
                        ctr.model_count().unwrap(),
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
        acc = apply_and(acc, clause_to_tdd(vtree, &literals(clause)));
    }
    acc.minimize().unwrap();
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
    for convention in [PinSemantics::Cofactor, PinSemantics::Evidence] {
        let mut pins: Vec<Option<bool>> = vec![None; 6];
        let mut ctr = eng.counter_with(&tdd, Retention::All, convention).unwrap()
            ;
        assert_eq!(ctr.model_count().unwrap(), pinned_counts(&tdd, &pins, convention));

        // Two leaves in different halves of the tree, so the cones share only
        // the root.
        pins[0] = Some(false);
        pins[5] = Some(true);
        ctr.set_pin(VarId(1), Some(false)).unwrap();
        ctr.set_pin(VarId(6), Some(true)).unwrap();

        assert_eq!(
            ctr.model_count().unwrap(),
            pinned_counts(&tdd, &pins, convention),
            "convention={convention:?}: two pins changed in different subtrees"
        );

        // Two leaves under one parent, and one of them changed twice: the
        // cone is folded once from the pins in force at the recompute.
        pins[1] = Some(false);
        pins[2] = Some(true);
        ctr.set_pin(VarId(2), Some(true)).unwrap();
        ctr.set_pin(VarId(2), Some(false)).unwrap();
        ctr.set_pin(VarId(3), Some(true)).unwrap();

        assert_eq!(
            ctr.model_count().unwrap(),
            pinned_counts(&tdd, &pins, convention),
            "convention={convention:?}: two pins changed under one parent"
        );
        assert!(
            ctr.model_count().unwrap() != BigUint::ZERO,
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
    let mut ctr = eng.counter_with(&tdd, Retention::All, PinSemantics::Evidence).unwrap();
    for (v, &p) in pins.iter().enumerate() {
        ctr.set_pin(VarId(v as u32 + 1), p).unwrap();
    }

    let expected = pinned_counts(&tdd, &pins, PinSemantics::Evidence);
    assert_eq!(ctr.model_count().unwrap(), expected);
    assert!(format!("{ctr:?}").contains("changed_since_pass: 0"), "compute clears the change set: {ctr:?}");

    ctr.set_pin(VarId(1), Some(true)).unwrap();
    ctr.set_pin(VarId(2), None).unwrap();
    assert!(format!("{ctr:?}").contains("changed_since_pass: 0"), "a same-value pin is no change: {ctr:?}");

    assert_eq!(ctr.model_count().unwrap(), expected, "nothing changed, nothing moves");

    ctr.set_pin(VarId(1), Some(false)).unwrap();
    ctr.set_pin(VarId(1), Some(true)).unwrap();
    assert!(format!("{ctr:?}").contains("changed_since_pass: 1"), "a changed-and-restored pin is recorded: {ctr:?}");

    assert_eq!(ctr.model_count().unwrap(), expected, "the original pins give the original count");
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
/// - `Retention::Frontier` (children freed as parents complete) yields the
///   same root count as `Retention::All`, on diagrams whose levels include a
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
                    let cl = clause_to_tdd(&vtree, &literals(&clause));
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
                .any(|vi| !vtree.node(VtreeIdx(vi as u32)).is_leaf() && tdd.levels[vi].slot_count() == 0)
            {
                continue;
            }
            marginalize_subtree(&mut tdd, marginal_root);
            tdd.minimize().unwrap();
            // `model_count` has an implicit `!is_zero` precondition (a structurally
            // zero diagram has no output count slot); every production wrapper
            // early-returns on it.
            if tdd.is_zero() {
                continue;
            }
            if tdd.levels[marginal_root.idx()].is_marginal() {
                with_marginal += 1;
            }

            for convention in [PinSemantics::Cofactor, PinSemantics::Evidence] {
                // One reused Frontier counter for the whole pin sweep — the
                // structured-count readout's exact shape.
                let mut reused = eng.counter_with(&tdd, Retention::Frontier, convention,
                ).unwrap()
                ;
                for _ in 0..4 {
                    let pins: Vec<Option<bool>> = (0..nvars)
                        .map(|v| {
                            if is_summed_out(&tdd, VarId(v + 1)) { return None; }
                            match rng.below(3) {
                                0 => None,
                                1 => Some(true),
                                _ => Some(false),
                            }
                        })
                        .collect();
                    let expected = if convention == PinSemantics::Evidence {
                        pinned_counts(&tdd, &pins, PinSemantics::Evidence)
                    } else {
                        pinned_counts(&tdd, &pins, PinSemantics::Cofactor)
                    };

                    for (v, &p) in pins.iter().enumerate() {
                        if !is_summed_out(&tdd, VarId(v as u32 + 1)) {
                            reused.set_pin(VarId(v as u32 + 1), p).unwrap();
                        }
                    }
                    assert_eq!(
                        reused.model_count().unwrap(),
                        expected,
                        "nvars={nvars} convention={convention:?}: reused Frontier counter disagrees with the \
                         BigUint oracle on a marginalized diagram"
                    );

                    // Unevaluated counters, both retention policies: the policy must
                    // be value-neutral, and a fresh frontier pass must match the
                    // reused one (no state carried between assignments).
                    assert_eq!(
                        fresh_root(Retention::All, &eng, &tdd, convention, &pins),
                        expected,
                        "nvars={nvars} convention={convention:?}: a fresh whole-array counter disagrees \
                         with the BigUint oracle on a marginalized diagram"
                    );
                    assert_eq!(
                        fresh_root(Retention::Frontier, &eng, &tdd, convention, &pins),
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

#[test]
fn interrupted_pin_refresh_recomputes_before_the_next_read() {
    use crate::query::{PinSemantics, Retention};
    use crate::limits::{LimitConfig, StopCallback, StopDecision};
    let eng = crate::Engine::new();
    let tree = std::sync::Arc::new(crate::vtree::Vtree::balanced(4));
    let f = crate::Tdd::clause(&tree, [1, 2]).unwrap();
    crate::test_helpers::assert_canonical(&f);
    let mut counter = eng.counter_with(&f, Retention::All, PinSemantics::Evidence).unwrap();
    assert_eq!(counter.model_count().unwrap(), 12u32.into());
    counter.set_pin(crate::vtree::VarId(1), Some(false)).unwrap();
    {
        let _stop = eng.limits().scope(LimitConfig::none().with_stop_callback(Some(StopCallback::new(|_, _| StopDecision::Stop))));
        assert!(counter.model_count().is_err());
    }
    assert_eq!(counter.model_count().unwrap(), 4u32.into());
    counter.set_pin(crate::vtree::VarId(1), None).unwrap();
    assert_eq!(counter.model_count().unwrap(), 12u32.into());
}
