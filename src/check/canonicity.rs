//! Canonicity checking and the minimize round-trip it backs.

use std::collections::HashSet;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use crate::reduce::minimize;
use crate::diagram::*;
use super::signature::*;

/// Check canonicity via probabilistic equivalence testing (Schwartz–Zippel).
///
/// In a canonical (minimized) diagram, no two nodes at the same vtree level compute
/// the same Boolean function. This checker evaluates the diagram bottom-up in the
/// (Z_p, +, ×) semiring with random variable assignments and looks for collisions.
///
/// - Uses prime p = 2^61 − 1
/// - Per-round collision probability per pair: ≤ n/p (Schwartz-Zippel lemma)
/// - 3 rounds gives negligible false-negative probability
///
/// Cost: O(diagram size × rounds).
///
/// Limitation: a marginal level's node signature is its stored count, so a
/// structural level both of whose children are marginal is not decided by this
/// test — distinct nodes there can share a signature (4×1 and 2×2) and the
/// check passes.
pub fn check_canonicity(tdd: &Tdd, rounds: u32) -> Result<(), String> {
    let vtree = &tdd.vtree;
    let num_vars = vtree.num_vars() as usize;

    for round in 0..rounds {
        let mut rng = SmallRng::seed_from_u64((round as u64).wrapping_mul(0x517cc1b727220a95));
        let (pos_val, neg_val) = random_var_assignments(num_vars, &mut rng);
        let signatures = eval_all_signatures(tdd, &pos_val, &neg_val);

        // Check for collisions: two nodes at the same level with equal
        // signatures likely compute the same function (non-canonical).
        for t in vtree.bottomup() {
            let level_sigs = &signatures[t.idx()];
            let mut seen = HashSet::with_capacity(level_sigs.len());
            for (i, &sig) in level_sigs.iter().enumerate() {
                if !seen.insert(sig) {
                    let j = level_sigs.iter().position(|&s| s == sig).unwrap();
                    if j != i {
                        return Err(format!(
                            "round {}: vtree {:?} nodes {} and {} have the same signature ({}) \
                             — likely equivalent (non-canonical)",
                            round, t, j, i, sig
                        ));
                    }
                }
            }
        }
    }

    Ok(())
}

/// Check that `minimize` preserves the Boolean function computed by the diagram.
///
/// Computes the output node's semiring signature before and after calling
/// `minimize`. If the signatures differ, `minimize` changed the function
/// (a soundness bug).
///
/// **Mutates `tdd`** by calling `minimize` once. Safe to call on already-
/// minimized diagrams (idempotency check) or on raw `apply_and` output.
///
/// Cost: O(diagram size × rounds) plus one full minimize pass.
pub fn check_minimize_soundness(tdd: &mut Tdd, rounds: u32) -> Result<(), String> {
    let num_vars = tdd.vtree.num_vars() as usize;

    let mut before_sigs = Vec::with_capacity(rounds as usize);
    for round in 0..rounds {
        let mut rng = SmallRng::seed_from_u64((round as u64).wrapping_mul(0x9e3779b97f4a7c15));
        let (pos_val, neg_val) = random_var_assignments(num_vars, &mut rng);
        before_sigs.push(eval_output_signature(tdd, &pos_val, &neg_val));
    }

    minimize(tdd);

    for round in 0..rounds {
        let mut rng = SmallRng::seed_from_u64((round as u64).wrapping_mul(0x9e3779b97f4a7c15));
        let (pos_val, neg_val) = random_var_assignments(num_vars, &mut rng);
        let after = eval_output_signature(tdd, &pos_val, &neg_val);
        if before_sigs[round as usize] != after {
            return Err(format!(
                "round {}: output signature changed from {} to {} \
                 — minimize altered the function",
                round, before_sigs[round as usize], after
            ));
        }
    }

    Ok(())
}
