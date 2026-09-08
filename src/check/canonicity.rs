//! Canonicity checking, the gauge audit over its ray classes, and the
//! minimize round-trip it backs.

use std::collections::{HashMap, HashSet};
use std::fmt;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use crate::vtree::VtreeIdx;
use crate::reduce::minimize;
use crate::diagram::*;
use super::signature::*;

/// Check canonicity via probabilistic equivalence testing (Schwartz–Zippel).
///
/// In a canonical (minimized) TDD, no two nodes at the same vtree level compute
/// the same Boolean function. This checker evaluates the TDD bottom-up in the
/// (Z_p, +, ×) semiring with random variable assignments and looks for collisions.
///
/// - Uses prime p = 2^61 − 1
/// - Per-round collision probability per pair: ≤ n/p (Schwartz-Zippel lemma)
/// - 3 rounds gives negligible false-negative probability
///
/// Cost: O(TDD size × rounds).
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

// ── Projective (ray) canonicity — Invariant 3 up to positive scale ───────────
//
// Two nodes i, j at a level are *ray-equivalent* (their functions are
// proportional, w.h.p.) iff sig_i·N_j ≡ sig_j·N_i (mod p) in every round, where
// N is the node's integer mass (`eval_mass_vector`). Equivalently, when N ≢ 0
// the canonical ray key is sig·N^{-1} mod p; nodes with N ≡ 0 bucket separately
// on the raw sig tuple. A genuinely-zero (dead) node cannot survive minimize, so
// an N ≡ 0 bucket is only reachable via an accidental N ≡ 0 mod p on a nonzero
// integer mass — probability ~2^{-61} per node, acceptable and noted here.
//
// On purely-Boolean levels val is 0/1-valued, so proportional ⟺ equal and the
// ray classes coincide with the exact (Inv-3) classes: the projective check
// strictly generalizes `check_canonicity`. On marginal levels every node is a
// scalar count, so all nonzero-mass nodes collapse to one ray class — the gauge
// redundancy the audit surfaces.

/// Per-level ray/exact classification, shared by [`gauge_audit`] and
/// [`check_canonicity_projective`] (single source of truth for the analysis).
///
/// Statistics count only **live** nodes. A node is live iff it is reachable from
/// the vtree root level AND is not a dead zero-node (all-rounds signature ≡ 0 and
/// mass ≡ 0). Mid-/post-compile levels accumulate orphaned/retired nodes (the
/// slot-pruning machinery (`prune_marg_slots`) exists precisely because they
/// do); those dead
/// nodes all carry signature 0 / mass 0, collapse into one bucket, and would
/// otherwise inflate the reported gauge redundancy with garbage — hence the
/// liveness filter.
struct LevelAnalysis {
    vtree: VtreeIdx,
    /// Total nodes stored at this level (live + dead).
    nodes: usize,
    /// Live nodes (reachable and not a dead zero-node).
    live: usize,
    /// Distinct exact-equivalence classes among live nodes (equal sig tuple
    /// across all rounds).
    exact: usize,
    /// Distinct ray-equivalence classes among live nodes (proportional functions).
    ray: usize,
    /// First (j, i) live-node pair found sharing a ray class, with the round-0
    /// sig of `i`, if any — used to phrase the projective-canonicity violation
    /// (whose only caller, `check_canonicity_projective`, is test-only).
    #[cfg_attr(not(test), allow(dead_code))]
    first_ray_collision: Option<(usize, usize, u64)>,
}

/// Bucket every **live** node at every non-leaf level into exact and ray classes,
/// running `rounds` random Schwartz–Zippel rounds plus one deterministic mass
/// pass. Dead nodes (unreachable, or all-zero signature with zero mass) are
/// excluded from every tally — see [`LevelAnalysis`].
fn analyze_ray_classes(tdd: &Tdd, rounds: u32) -> Vec<LevelAnalysis> {
    let vtree = &tdd.vtree;
    let num_vars = vtree.num_vars() as usize;

    let masses = eval_mass_vector(tdd);
    let round_sigs: Vec<Vec<Vec<u64>>> = (0..rounds)
        .map(|round| {
            let mut rng = SmallRng::seed_from_u64((round as u64).wrapping_mul(0x517cc1b727220a95));
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

        let mut live = 0usize;
        let mut exact_set: HashSet<Vec<u64>> = HashSet::with_capacity(width);
        let mut ray_set: HashSet<(bool, Vec<u64>)> = HashSet::with_capacity(width);
        let mut ray_owner: HashMap<(bool, Vec<u64>), usize> = HashMap::with_capacity(width);
        let mut first_ray_collision = None;

        for i in 0..width {
            let exact_key: Vec<u64> = (0..rounds as usize).map(|r| round_sigs[r][ti][i]).collect();
            let mass = masses[ti][i];

            // Liveness filter: skip nodes unreachable from the root level, and
            // dead zero-nodes (all-rounds sig ≡ 0 AND mass ≡ 0) — the orphan/
            // retired residue that would otherwise pollute the class counts.
            let is_reachable = reachable[ti][i];
            let is_dead_zero = mass == 0 && exact_key.iter().all(|&s| s == 0);
            if !is_reachable || is_dead_zero {
                continue;
            }
            live += 1;

            let ray_key = if mass == 0 {
                (true, exact_key.clone()) // zero-mass bucket, keyed by raw sig tuple
            } else {
                let inv = mod_inv(mass);
                (false, (0..rounds as usize).map(|r| mod_mul(round_sigs[r][ti][i], inv)).collect())
            };

            if let Some(&j) = ray_owner.get(&ray_key) {
                if first_ray_collision.is_none() {
                    first_ray_collision = Some((j, i, round_sigs[0][ti][i]));
                }
            } else {
                ray_owner.insert(ray_key.clone(), i);
            }
            ray_set.insert(ray_key);
            exact_set.insert(exact_key);
        }

        out.push(LevelAnalysis {
            vtree: t,
            nodes: width,
            live,
            exact: exact_set.len(),
            ray: ray_set.len(),
            first_ray_collision,
        });
    }
    out
}

/// One vtree level's node/class tallies in a [`GaugeAuditReport`]. All class
/// counts are over LIVE nodes only (`nodes` is the total live + dead width for
/// context).
pub struct GaugeLevelStat {
    /// Vtree level index this row describes.
    pub vtree: VtreeIdx,
    /// Total node width at the level (live + dead).
    pub nodes: usize,
    /// Live-node count at the level.
    pub live: usize,
    /// Number of distinct exact node signatures among the live nodes.
    pub exact: usize,
    /// Number of distinct ray (gauge-equivalence) classes among the live nodes.
    pub ray: usize,
}

/// Gauge-redundancy audit result: per-level live-node exact/ray class counts plus
/// diagram totals. Gauge redundancy is defined over LIVE nodes: `live − ray`. Its
/// [`fmt::Display`] prints a compact table (only levels carrying redundancy) and a
/// totals line.
pub struct GaugeAuditReport {
    /// Per-level tallies (one entry per vtree level).
    pub levels: Vec<GaugeLevelStat>,
    /// Total node width summed over all levels (live + dead).
    pub node_count: usize,
    /// Total live nodes summed over all levels.
    pub total_live: usize,
    /// Total distinct exact-class count summed over all levels.
    pub total_exact: usize,
    /// Total distinct ray-class count summed over all levels.
    pub total_ray: usize,
}

impl fmt::Display for GaugeAuditReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for lv in &self.levels {
            // Only levels that carry live redundancy: ray-merges (ray < exact) or
            // exact-merges among live nodes (exact < live, a non-canonical
            // collision). Equivalent to ray < live since ray ≤ exact ≤ live.
            if lv.ray < lv.exact || lv.exact < lv.live {
                writeln!(
                    f,
                    "  vtree {:>4}: {} nodes ({} live), {} exact, {} ray",
                    lv.vtree.idx(), lv.nodes, lv.live, lv.exact, lv.ray,
                )?;
            }
        }
        write!(
            f,
            "gauge-audit: {} nodes, {} live, {} exact-classes, {} ray-classes (gauge redundancy {})",
            self.node_count,
            self.total_live,
            self.total_exact,
            self.total_ray,
            self.total_live - self.total_ray,
        )
    }
}

/// Audit gauge redundancy: per vtree level, how many LIVE nodes collapse to how
/// many exact-equivalence and ray-equivalence (proportional) classes. The
/// difference `live − ray_classes` is the gauge redundancy — live nodes that a
/// projective (up-to-positive-scale) canonical form would identify. Dead nodes
/// (unreachable from the root level, or zero-signature/zero-mass orphans) are
/// excluded so they cannot inflate the count. Marginal levels are the prime
/// source of true redundancy (all nonzero-mass scalar nodes are one ray class).
///
/// Cost: O(TDD size × rounds).
pub fn gauge_audit(tdd: &Tdd, rounds: u32) -> GaugeAuditReport {
    let analysis = analyze_ray_classes(tdd, rounds);
    let mut report = GaugeAuditReport {
        levels: Vec::with_capacity(analysis.len()),
        node_count: 0,
        total_live: 0,
        total_exact: 0,
        total_ray: 0,
    };
    for lv in analysis {
        report.node_count += lv.nodes;
        report.total_live += lv.live;
        report.total_exact += lv.exact;
        report.total_ray += lv.ray;
        report.levels.push(GaugeLevelStat {
            vtree: lv.vtree,
            nodes: lv.nodes,
            live: lv.live,
            exact: lv.exact,
            ray: lv.ray,
        });
    }
    report
}

/// Projective (up-to-positive-scale) Invariant 3: no two nodes at the same level
/// compute *proportional* functions. This is the ATDD-target canonicity property
/// — strictly stronger than [`check_canonicity`] (which only rejects *equal*
/// functions). Errors on the first ray collision.
///
/// Current diagrams legitimately fail this (marginal levels carry proportional
/// scalar nodes), so it is NOT wired into the standard invariant bundles — it is
/// test-support / the projective-canonicity goalpost, like `check_canonicity`.
///
/// Cost: O(TDD size × rounds).
#[cfg(test)]
pub(crate) fn check_canonicity_projective(tdd: &Tdd, rounds: u32) -> Result<(), String> {
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


/// Check that `minimize` preserves the Boolean function computed by the TDD.
///
/// Computes the output node's semiring signature before and after calling
/// `minimize`. If the signatures differ, `minimize` changed the function
/// (a soundness bug).
///
/// **Mutates `tdd`** by calling `minimize` once. Safe to call on already-
/// minimized TDDs (idempotency check) or on raw `apply_and` output.
///
/// Cost: O(TDD size × rounds) plus one full minimize pass.
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
