//! Projective (ray) canonicity: invariant 3 up to positive scale.
//!
//! Two nodes i, j at a level are *ray-equivalent* (their functions are
//! proportional, w.h.p.) iff sig_i·N_j ≡ sig_j·N_i (mod p) in every round, where
//! N is the node's integer mass (`eval_mass_vector`). Equivalently, when N ≢ 0
//! the canonical ray key is sig·N^{-1} mod p; nodes with N ≡ 0 bucket separately
//! on the raw sig tuple. A genuinely-zero (dead) node cannot survive minimize, so
//! an N ≡ 0 bucket is only reachable via an accidental N ≡ 0 mod p on a nonzero
//! integer mass — probability ~2^{-61} per node, acceptable and noted here.
//!
//! On purely-Boolean levels val is 0/1-valued, so proportional ⟺ equal and the
//! ray classes coincide with the exact (Inv-3) classes: the projective check
//! strictly generalizes `check_canonicity`. On marginal levels every node is a
//! scalar count, so all nonzero-mass nodes collapse to one ray class, which is
//! why current diagrams fail the projective check.

use std::collections::HashMap;

use crate::test_helpers::Lcg;

use crate::test_helpers::check::signature::{eval_all_signatures, mod_mul, random_var_assignments, PRIME};
use crate::diagram::Tdd;
use crate::vtree::VtreeIdx;

/// One level's first ray collision, if it has one.
///
/// Only **live** nodes take part. A node is live iff it is reachable from the
/// vtree root level and is not a dead zero-node (all-rounds signature zero and
/// mass zero). Mid- and post-compile levels accumulate orphaned and retired
/// nodes — that is what `prune_value_slots` exists for — and those nodes all
/// carry signature zero and mass zero, so they collapse into one bucket and
/// would report a collision that no live node has.
struct LevelAnalysis {
    vtree: VtreeIdx,
    /// First `(j, i)` live-node pair found sharing a ray class, with the round-0
    /// signature of `i`.
    first_ray_collision: Option<(usize, usize, u64)>,
}

/// Bucket every **live** node at every non-leaf level into ray classes, running
/// `rounds` random Schwartz–Zippel rounds plus one deterministic mass pass, and
/// report the first collision each level carries. Dead nodes are excluded — see
/// [`LevelAnalysis`].
fn analyze_ray_classes(tdd: &Tdd, rounds: u32) -> Vec<LevelAnalysis> {
    let vtree = &tdd.vtree;
    let num_vars = vtree.num_vars() as usize;

    let masses = eval_mass_vector(tdd);
    let round_sigs: Vec<Vec<Vec<u64>>> = (0..rounds)
        .map(|round| {
            let mut rng = Lcg::new((round as u64).wrapping_mul(0x517cc1b727220a95));
            let (pos_val, neg_val) = random_var_assignments(num_vars, &mut rng);
            eval_all_signatures(tdd, &pos_val, &neg_val)
        })
        .collect();

    // Reachability from every node at the vtree root level (not just `output`):
    // the audit runs mid-compile where the root can hold several live candidates.
    let reachable = tdd.reachable_from_root_level();

    let mut out = Vec::new();
    for t in vtree.bottomup() {
        if vtree.node(t).is_leaf() {
            continue; // implicit leaf labels are never diagram nodes
        }
        let ti = t.idx();
        let width = masses[ti].len();

        let mut ray_owner: HashMap<(bool, Vec<u64>), usize> = HashMap::with_capacity(width);
        let mut first_ray_collision = None;

        for i in 0..width {
            let exact_key: Vec<u64> = (0..rounds as usize).map(|r| round_sigs[r][ti][i]).collect();
            let mass = masses[ti][i];

            // Liveness filter: skip nodes unreachable from the root level, and
            // dead zero-nodes (all-rounds sig zero and mass zero) — the orphan
            // and retired residue that would otherwise report a collision no
            // live node has.
            let is_reachable = reachable[ti][i];
            let is_dead_zero = mass == 0 && exact_key.iter().all(|&s| s == 0);
            if !is_reachable || is_dead_zero {
                continue;
            }

            let ray_key = if mass == 0 {
                (true, exact_key) // zero-mass bucket, keyed by raw sig tuple
            } else {
                let inv = mod_inv(mass);
                (false, (0..rounds as usize).map(|r| mod_mul(round_sigs[r][ti][i], inv)).collect())
            };

            if let Some(&j) = ray_owner.get(&ray_key) {
                if first_ray_collision.is_none() {
                    first_ray_collision = Some((j, i, round_sigs[0][ti][i]));
                }
            } else {
                ray_owner.insert(ray_key, i);
            }
        }

        out.push(LevelAnalysis { vtree: t, first_ray_collision });
    }
    out
}

/// Projective (up-to-positive-scale) invariant 3: no two nodes at the same level
/// compute *proportional* functions. This is the canonicity property the diagram
/// form aims at — strictly stronger than [`check_canonicity`], which only
/// rejects *equal* functions. Errors on the first ray collision.
///
/// Current diagrams legitimately fail this (marginal levels carry proportional
/// scalar nodes), so it is not wired into the standard invariant bundles — it is
/// test-support / the projective-canonicity goalpost, like `check_canonicity`.
///
/// Cost: O(diagram size × rounds).
pub(super) fn check_canonicity_projective(tdd: &Tdd, rounds: u32) -> Result<(), String> {
    for lv in analyze_ray_classes(tdd, rounds) {
        if let Some((j, i, sig)) = lv.first_ray_collision {
            return Err(format!(
                "vtree {:?} nodes {} and {} are ray-equivalent (proportional functions, \
                 node {} round-0 sig {}) — not projectively canonical",
                lv.vtree, j, i, i, sig,
            ));
        }
    }
    Ok(())
}

/// Per-node integer mass `N(n)` = `Σ_x` val(n)(x) mod p for every node, via the
/// shared bottom-up recurrence seeded with all-ones leaf weights (⊤→2,
/// literal→1). The recurrence is identical to the random-point signature — only
/// the leaf seeding differs — so masses and signatures share one loop
/// (`eval_all_signatures`), never a copied variant. By distributivity N(n) =
/// `Σ_pairs` N(l)·N(r) (no disjointness needed), and a marginal node's mass is its
/// stored count.
fn eval_mass_vector(tdd: &Tdd) -> Vec<Vec<u64>> {
    let num_vars = tdd.vtree.num_vars() as usize;
    let ones = vec![1u64; num_vars];
    eval_all_signatures(tdd, &ones, &ones)
}

/// Modular exponentiation base^exp mod `PRIME` (Mersenne 2^61−1).
fn mod_pow(mut base: u64, mut exp: u64) -> u64 {
    let mut result = 1u64;
    base %= PRIME as u64;
    while exp > 0 {
        if exp & 1 == 1 {
            result = mod_mul(result, base);
        }
        base = mod_mul(base, base);
        exp >>= 1;
    }
    result
}

/// Modular inverse mod `PRIME` via Fermat's little theorem (`PRIME` is prime).
/// Caller guarantees `a` is nonzero mod `PRIME`.
fn mod_inv(a: u64) -> u64 {
    mod_pow(a, (PRIME - 2) as u64)
}
