//! The incremental pinned counter against its full-precision oracle.
//!
//! Sibling of `query_tests.rs`, which holds the fixtures these read.

use super::*;

use crate::engine::Engine;
use crate::apply::conjoin::apply_and;
use crate::build::{clause_to_tdd, constant_one};
use crate::reduce::minimize;
use crate::diagram::Literal;
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
/// Exercises both of the counter's entry points per formula: a `recompute_all` from a
/// freshly-pinned state (checked against the oracle called with the same pins), then a
/// sequence of incremental steps that flip exactly one variable's pin and call
/// `recompute_dirty` on only the "dirty cone" — that variable's leaf vtree level
/// followed by its ancestors up to the root (children-before-parents, the order
/// `recompute_dirty` requires) — re-checked against the oracle recomputed from scratch
/// under the updated pins each time. This pins the O(cone) incremental contract itself,
/// not just the equivalent-to-full-recompute case.
#[test]
fn incremental_pinned_counter_matches_pinned_bigint_randomized() {
    let eng = Engine::new();
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
            let mut acc = constant_one(&eng, &vtree);
            for _ in 0..nclauses {
                let width = 1 + (rng() % nvars as u64) as usize;
                let mut literals: Vec<Literal> = Vec::new();
                let mut seen = vec![false; nvars as usize];
                for _ in 0..width {
                    let v = (rng() % nvars as u64) as u32;
                    if seen[v as usize] {
                        continue;
                    }
                    seen[v as usize] = true;
                    let pol = rng().is_multiple_of(2);
                    literals.push(if pol {
                        Literal::pos(VarId(v))
                    } else {
                        Literal::neg(VarId(v))
                    });
                }
                if literals.is_empty() {
                    continue;
                }
                let cl = clause_to_tdd(&eng, &vtree, &literals);
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
                    .map(|_| match rng() % 3 {
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
                    "nvars={nvars} convention={convention:?}: recompute_all mismatch"
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
                    ctr.recompute_dirty(&eng, &tdd, &tdd.vtree.bottom_up_subset(levels));

                    let expected = if convention == SeedConvention::Fixed {
                        pinned_counts(&tdd, &pins, SeedConvention::Fixed)
                    } else {
                        pinned_counts(&tdd, &pins, SeedConvention::Free)
                    };
                    assert_eq!(
                        ctr.output_count(&tdd),
                        expected,
                        "nvars={nvars} convention={convention:?}: incremental dirty-cone mismatch after flipping var {v}"
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
                let nclauses = 1 + (rng() % 4) as usize;
                let mut acc = constant_one(&eng, &vtree);
                for _ in 0..nclauses {
                    let width = 1 + (rng() % nvars as u64) as usize;
                    let mut literals: Vec<Literal> = Vec::new();
                    let mut seen = vec![false; nvars as usize];
                    for _ in 0..width {
                        let v = (rng() % nvars as u64) as u32;
                        if seen[v as usize] {
                            continue;
                        }
                        seen[v as usize] = true;
                        literals.push(if rng() % 2 == 0 {
                            Literal::pos(VarId(v))
                        } else {
                            Literal::neg(VarId(v))
                        });
                    }
                    if literals.is_empty() {
                        continue;
                    }
                    let cl = clause_to_tdd(&eng, &vtree, &literals);
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
                        .map(|_| match rng() % 3 {
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
