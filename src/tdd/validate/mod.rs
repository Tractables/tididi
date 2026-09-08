//! Invariant checkers for TDDs — test infrastructure, compiled only with
//! `debug_assertions` and hidden from the documented API.
//!
//! Every checker returns `Ok(())` or `Err(String)` naming the violation. The
//! marginal-canonical-form checks and the model-count localizer live in
//! [`marg`].
//!
//! ## Available checks
//!
//! | Checker | Cost | When to use |
//! |---------|------|-------------|
//! | [`validate_vtree_structure`] | O(size) | Always |
//! | [`check_no_false_nodes`] | O(nodes) | Always |
//! | [`check_no_false_nodes_in_levels`] | O(nodes) | Pre-minimize |
//! | [`check_canonicity`] | O(size × rounds) | After minimize |
//! | [`check_minimize_soundness`] | O(size × rounds) + minimize | Before + after minimize |
//! | [`check_reduced_size_sanity`] | O(size) + BigUint | After minimize |
//! | [`check_determinism`] | O(width² × apply / level + size) | Small TDDs only (≤5 vars). Includes leaf-level label-mode consistency. |

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::Arc;

use num_bigint::BigUint;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

use crate::vtree::{VtreeIdx, VtreeNode};

use crate::tdd::transform::pairwise::conjoin::apply_and;
use crate::tdd::minimize::minimize;
use crate::tdd::query::model_count;
use crate::tdd::query::reduced_tdd_size;
use crate::tdd::types::*;

// ── Semiring helpers (probabilistic equivalence testing) ─────────────────────
//
// To check whether two TDD nodes compute the same Boolean function without
// enumerating all 2^n assignments, we evaluate both in a finite-field semiring
// (Z_p, +, ×) with random variable weights. By the Schwartz–Zippel lemma,
// two distinct multilinear polynomials agree on a random point with probability
// ≤ degree/p. With p = 2^61 − 1 and degree ≤ num_vars, false negatives are
// astronomically unlikely (~10^{-18} per round).

/// Mersenne prime 2^61 − 1, used for probabilistic polynomial identity testing.
const PRIME: u128 = (1 << 61) - 1;

/// Modular multiplication mod PRIME, exploiting the Mersenne structure to
/// avoid full 128-bit division.
fn mod_mul(a: u64, b: u64) -> u64 {
    let prod = a as u128 * b as u128;
    let lo = prod & PRIME;
    let hi = prod >> 61;
    let result = lo + hi;
    let result = if result >= PRIME { result - PRIME } else { result };
    result as u64
}

fn mod_add(a: u64, b: u64) -> u64 {
    let sum = a as u128 + b as u128;
    (if sum >= PRIME { sum - PRIME } else { sum }) as u64
}

/// Generate random variable assignments for semiring evaluation.
fn random_var_assignments(num_vars: usize, rng: &mut SmallRng) -> (Vec<u64>, Vec<u64>) {
    let mut pos_val = vec![0u64; num_vars];
    let mut neg_val = vec![0u64; num_vars];
    for i in 0..num_vars {
        pos_val[i] = rng.random_range(1..PRIME as u64);
        neg_val[i] = rng.random_range(1..PRIME as u64);
    }
    (pos_val, neg_val)
}

/// Evaluate every TDD node bottom-up in the (`Z_p`, +, ×) semiring.
///
/// Returns per-level signature arrays: `signatures[vtree_idx][node_idx]` is the
/// fingerprint of that node's Boolean function under the given random assignment.
/// Two nodes with equal signatures very likely compute the same function
/// (collision probability ≤ degree/p ≈ 10^{-18} per pair).
fn eval_all_signatures(tdd: &Tdd, pos_val: &[u64], neg_val: &[u64]) -> Vec<Vec<u64>> {
    let vtree = &tdd.vtree;
    let mut signatures: Vec<Vec<u64>> = (0..tdd.levels.len())
        .map(|i| vec![0u64; tdd.effective_width(VtreeIdx(i as u32))])
        .collect();

    for (t, var) in vtree.leaf_bottomup() {
        let v = var.idx();
        // Leaf levels are marginal: iterate 0..LEAF_WIDTH using LeafLabel::from_idx.
        for i in 0..LEAF_WIDTH {
            let label = LeafLabel::from_idx(i);
            signatures[t.idx()][i] = match label {
                LeafLabel::Pos => pos_val[v],
                LeafLabel::Neg => neg_val[v],
                LeafLabel::One => mod_add(pos_val[v], neg_val[v]),
                LeafLabel::Zero => 0,
            };
        }
    }
    for (t, left, right) in vtree.internal_bottomup() {
        let level = tdd.level(t);
        // Marginal level: each node's signature (and mass) is its stored count
        // mod p — same value in both passes, independent of the leaf seeding.
        // Weighted-marginal levels carry no integer counts (`marginal_counts`
        // None); leave their slots zero — the audit dispatch skips weighted mode.
        if level.is_marginal() {
            if let Some(counts) = &level.marginal_counts {
                let big = level.marginal_counts_big.as_ref();
                for (i, &c) in counts.iter().enumerate() {
                    let big_i = big.and_then(|b| b.get(i));
                    signatures[t.idx()][i] = count_mod_p(c, big_i);
                }
            }
            continue;
        }
        let left_marg = tdd.level(left).is_marginal();
        let right_marg = tdd.level(right).is_marginal();
        for (i, node) in level.nodes.iter().enumerate() {
            let mut total = 0u64;
            let mut any = false;
            for pair in level.pairs_iter_of(node) {
                any = true;
                // Inline(k) contributes the scalar k mod p directly (k <=
                // MARG_INLINE_MAX < PRIME, so k mod p == k). Index(s) reads the
                // child level's already-computed signature at slot/node s.
                let l = match resolve_marg_ref(pair.left.0, left_marg) {
                    MargResolved::Index(s) => signatures[left.idx()][s],
                    MargResolved::Inline(k) => k as u64,
                };
                let r = match resolve_marg_ref(pair.right.0, right_marg) {
                    MargResolved::Index(s) => signatures[right.idx()][s],
                    MargResolved::Inline(k) => k as u64,
                };
                total = mod_add(total, mod_mul(l, r));
            }
            if any {
                signatures[t.idx()][i] = total;
            }
        }
    }

    signatures
}

/// Reduce an integer marginal count to `Z_p`. `c == u128::MAX` is the `BigUint`
/// overflow sentinel — the true value lives in the level's `marginal_counts_big`
/// side-table entry `big`.
fn count_mod_p(c: u128, big: Option<&BigUint>) -> u64 {
    if c == u128::MAX {
        let b = big.expect("marginal_counts_big entry missing for u128::MAX sentinel");
        (b % BigUint::from(PRIME as u64))
            .to_u64_digits()
            .first()
            .copied()
            .unwrap_or(0)
    } else {
        (c % PRIME) as u64
    }
}

/// Per-node integer MASS N(n) = `Σ_x` val(n)(x) mod p for every node, via the
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

/// Modular exponentiation base^exp mod PRIME (Mersenne 2^61−1).
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

/// Modular inverse mod PRIME via Fermat's little theorem (PRIME is prime).
/// Caller guarantees `a` is nonzero mod PRIME.
fn mod_inv(a: u64) -> u64 {
    mod_pow(a, (PRIME - 2) as u64)
}

/// Evaluate a TDD bottom-up in the (`Z_p`, +, ×) semiring with the given random
/// variable assignments. Returns the signature of the output node.
fn eval_output_signature(tdd: &Tdd, pos_val: &[u64], neg_val: &[u64]) -> u64 {
    if tdd.output.local == ZERO {
        return 0;
    }
    let signatures = eval_all_signatures(tdd, pos_val, neg_val);
    signatures[tdd.output.vtree.idx()][tdd.output.local.idx()]
}

/// Create a TDD sharing the same levels but with a different output node.
/// Used by [`check_determinism`] to construct per-node sub-TDDs.
fn tdd_with_output(
    tdd: &Tdd,
    vtree: &Arc<crate::vtree::Vtree>,
    vtree_node: VtreeIdx,
    local: u32,
) -> Tdd {
    Tdd::with_levels(
        Arc::clone(vtree),
        tdd.levels.clone(),
        TddNodeId { vtree: vtree_node, local: LocalNodeIdx(local) },
    )
}

// ── Public checker functions ─────────────────────────────────────────────────

/// Validate that every TDD node matches its vtree position.
///
/// Checks:
/// - Leaf vtree levels have no stored nodes (implicit representation)
/// - Internal vtree levels contain only internal TDD nodes
/// - All `InputPair` child references are in bounds
/// - Output node is at the vtree root with a valid local index
///
/// Cost: O(TDD size).
pub fn validate_vtree_structure(tdd: &Tdd) -> Result<(), String> {
    let vtree = &tdd.vtree;

    if tdd.output.vtree != vtree.root() {
        return Err(format!(
            "output vtree {:?} != root {:?}",
            tdd.output.vtree, vtree.root()
        ));
    }

    if tdd.output.local == ZERO {
        return Ok(());
    }

    let root_width = tdd.effective_width(vtree.root());
    if tdd.output.local.idx() >= root_width {
        return Err(format!(
            "output local index {} >= root width {}",
            tdd.output.local.0, root_width
        ));
    }

    for (t, _var) in vtree.leaf_bottomup() {
        if !tdd.level(t).nodes.is_empty() {
            return Err(format!(
                "vtree leaf {:?} has non-empty nodes vec (len {}) — \
                 leaf levels should be implicit",
                t, tdd.level(t).nodes.len()
            ));
        }
    }
    for (t, left, right) in vtree.internal_bottomup() {
        let left_width = tdd.effective_width(left);
        let right_width = tdd.effective_width(right);
        let level = tdd.level(t);
        let left_marg = tdd.level(left).is_marginal();
        let right_marg = tdd.level(right).is_marginal();

        for (i, node) in level.nodes.iter().enumerate() {
            if !node.is_internal() {
                return Err(format!(
                    "vtree internal {:?} has non-Internal node at index {}",
                    t, i
                ));
            }
            for (j, pair) in level.pairs_iter_of(node).enumerate() {
                let l = match resolve_marg_ref(pair.left.0, left_marg) {
                    MargResolved::Index(s) => s,
                    MargResolved::Inline(_) => unreachable!("Phase A: inline marg ref in validate_vtree_structure"),
                };
                let r = match resolve_marg_ref(pair.right.0, right_marg) {
                    MargResolved::Index(s) => s,
                    MargResolved::Inline(_) => unreachable!("Phase A: inline marg ref in validate_vtree_structure"),
                };
                if l >= left_width {
                    return Err(format!(
                        "vtree {:?} node {} input {} left index {} >= left child width {}",
                        t, i, j, l, left_width
                    ));
                }
                if r >= right_width {
                    return Err(format!(
                        "vtree {:?} node {} input {} right index {} >= right child width {}",
                        t, i, j, r, right_width
                    ));
                }
            }
        }
    }

    Ok(())
}

/// Check that no node in any level computes the constant-false function.
///
/// This invariant holds for every public TDD (even before minimize):
/// - Leaf levels are marginal (no stored nodes) — Zero never appears
/// - No internal nodes with empty pairs exist in any level
/// - If UNSAT (after minimize), all internal levels are empty
///
/// Cost: O(total nodes).
pub fn check_no_false_nodes(tdd: &Tdd) -> Result<(), String> {
    check_no_false_nodes_in_levels(tdd)?;

    if tdd.output.local == ZERO {
        for (t_idx, level) in tdd.levels.iter().enumerate() {
            if !level.nodes.is_empty() {
                return Err(format!(
                    "UNSAT TDD (ZERO output) has non-empty level at vtree index {} \
                     (width {}) — expected empty after minimize",
                    t_idx,
                    level.width()
                ));
            }
        }
    }
    Ok(())
}

/// Check that no node in any level computes constant-false.
///
/// This is the per-level subset of [`check_no_false_nodes`] — it does *not*
/// require all levels to be empty on UNSAT. Useful for checking raw
/// `apply_and` output before `minimize`.
///
/// Cost: O(total nodes).
pub fn check_no_false_nodes_in_levels(tdd: &Tdd) -> Result<(), String> {
    for t in tdd.vtree.bottomup() {
        // Skip leaf levels — they are marginal and always contain Pos, Neg, One (no Zero).
        if tdd.vtree.node(t).is_leaf() {
            continue;
        }
        let level = tdd.level(t);
        for (i, node) in level.nodes.iter().enumerate() {
            if node.is_internal() && level.pairs_iter_of(node).next().is_none() {
                return Err(format!(
                    "vtree {:?} node {}: Internal with empty inputs — no real node \
                     should compute constant-false (use ZERO sentinel instead)",
                    t, i
                ));
            }
        }
    }
    Ok(())
}

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
    pub total_nodes: usize,
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
            self.total_nodes,
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
        total_nodes: 0,
        total_live: 0,
        total_exact: 0,
        total_ray: 0,
    };
    for lv in analysis {
        report.total_nodes += lv.nodes;
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

/// Independently validate every reducibility decision made by `reduced_tdd_size`.
///
/// For each internal node where Case L or Case R fires, verifies:
/// - Structural completeness: child indices == 0..child_width
/// - Product-form: count(g) == count(target) × 2^|vars(child)|
///
/// Cost: O(TDD size), but uses BigUint arithmetic for model counts.
pub fn check_reduced_size_sanity(tdd: &Tdd) -> Result<(), String> {
    if tdd.output.local == ZERO {
        return Ok(());
    }

    let vtree = &tdd.vtree;
    let num_levels = tdd.levels.len();

    // Reuse the shared model count computation from query.rs.
    let counts = crate::tdd::query::compute_node_counts(tdd);
    let mut subtree_vars = vec![0u32; num_levels];
    for (t, _var) in vtree.leaf_bottomup() {
        subtree_vars[t.idx()] = 1;
    }
    for (t, left, right) in vtree.internal_bottomup() {
        subtree_vars[t.idx()] = subtree_vars[left.idx()] + subtree_vars[right.idx()];
    }

    for (t, left, right) in vtree.internal_bottomup() {
        let ti = t.idx();
        let li = left.idx();
        let ri = right.idx();
        let left_marg = tdd.levels[li].is_marginal();
        let right_marg = tdd.levels[ri].is_marginal();
        let true_t1 = BigUint::from(1u32) << subtree_vars[li] as usize;
        let true_t2 = BigUint::from(1u32) << subtree_vars[ri] as usize;
        let level = &tdd.levels[ti];

        for (node_i, node) in level.nodes.iter().enumerate() {
            if node.is_internal() {
                let pairs: Vec<InputPair> = level.pairs_iter_of(node).collect();
                if pairs.is_empty() {
                    continue;
                }

                let first_right = pairs[0].right;
                if pairs.iter().all(|p| p.right == first_right) {
                    let sum: BigUint = pairs.iter().map(|p| {
                        let l = match resolve_marg_ref(p.left.0, left_marg) {
                            MargResolved::Index(s) => s,
                            MargResolved::Inline(_) => unreachable!("Phase A: inline marg ref in check_reduced_size_sanity"),
                        };
                        &counts[li][l]
                    }).sum();
                    if sum == true_t1 {
                        // Structural completeness check removed: with implicit
                        // leaves, a reducible node may reference only a subset of
                        // implicit labels (e.g., One alone covers 2^1 models).
                        let first_right_slot = match resolve_marg_ref(first_right.0, right_marg) {
                            MargResolved::Index(s) => s,
                            MargResolved::Inline(_) => unreachable!("Phase A: inline marg ref in check_reduced_size_sanity"),
                        };
                        let product = &counts[ri][first_right_slot] * &true_t1;
                        if counts[ti][node_i] != product {
                            return Err(format!(
                                "Case L product-form failed at vtree {} node {}: \
                                 count {} != {} × {} = {}",
                                ti, node_i, counts[ti][node_i],
                                counts[ri][first_right_slot], true_t1, product
                            ));
                        }
                        continue;
                    }
                }

                let first_left = pairs[0].left;
                if pairs.iter().all(|p| p.left == first_left) {
                    let sum: BigUint = pairs.iter().map(|p| {
                        let r = match resolve_marg_ref(p.right.0, right_marg) {
                            MargResolved::Index(s) => s,
                            MargResolved::Inline(_) => unreachable!("Phase A: inline marg ref in check_reduced_size_sanity"),
                        };
                        &counts[ri][r]
                    }).sum();
                    if sum == true_t2 {
                        let first_left_slot = match resolve_marg_ref(first_left.0, left_marg) {
                            MargResolved::Index(s) => s,
                            MargResolved::Inline(_) => unreachable!("Phase A: inline marg ref in check_reduced_size_sanity"),
                        };
                        let product = &counts[li][first_left_slot] * &true_t2;
                        if counts[ti][node_i] != product {
                            return Err(format!(
                                "Case R product-form failed at vtree {} node {}: \
                                 count {} != {} × {} = {}",
                                ti, node_i, counts[ti][node_i],
                                counts[li][first_left_slot], true_t2, product
                            ));
                        }
                    }
                }
            }
        }
    }

    Ok(())
}

/// Check the determinism property: at each vtree level, all *live* nodes
/// compute mutually exclusive Boolean functions.
///
/// **Internal levels** — for each pair of distinct nodes (i, j) at the same
/// internal vtree level, conjoins them via `apply_and` and verifies
/// `model_count == 0`.
///
/// **Leaf levels** — there are no stored nodes at a leaf level, but the
/// labels {Pos, Neg, One, Zero} are *implicit* nodes referenced by parent
/// pair lists (and possibly by the TDD output). Two of these labels can
/// only co-occur as live references if they are pairwise mutex as functions
/// of the leaf variable: `Pos ∧ Neg = ⊥` is fine, but `Pos ∧ One = Pos ≠ ⊥`
/// is not. Equivalently, at every leaf vtree level the set of *referenced*
/// labels must be a subset of `{Pos, Neg}` (literal mode) or `{One}`
/// (unconstrained mode), never spanning both. This is exactly the
/// per-pair-mutex condition extended to the leaf "node" that lives only
/// implicitly in the labels its parents pick.
///
/// Mode is a global property of the function on the vtree (determined by
/// whether the leaf's two bit-values induce the same external-completion
/// set), so a canonical TDD picks one mode per leaf at construction. A
/// minimize phase that fails to leaf-contract `(Pos_x, S) + (Neg_x, S)`
/// siblings into `(One_x, S)` while other parts of the same TDD already use
/// `One_x` would mix modes and trip this check.
///
/// **Only feasible for small TDDs** (≤5 variables) due to O(width² × apply)
/// cost per internal level. Do not call on easy or large benchmarks.
///
/// **Non-marginal TDDs only** — this assumes a plain Boolean diagram and
/// panics (or misbehaves) on marginal TDDs; do not call it on marginalize
/// outputs.
///
/// Cost: O(width² × apply_and_cost) per internal level + O(size) for the
/// leaf-label scan.
pub fn check_determinism(tdd: &Tdd) -> Result<(), String> {
    let vtree = &tdd.vtree;
    let shared_vtree = Arc::clone(&tdd.vtree);

    // Leaf-level mode consistency: at each leaf vtree node, collect the set
    // of labels actually referenced (via parent pair lists or the TDD output).
    // Reject mixes where a non-mutex pair of labels is live.
    let n = vtree.num_nodes();
    let mut used_at_leaf: Vec<u8> = vec![0u8; n]; // bit i = label i is used
    for vi in 0..n {
        let (left, right) = match *vtree.node(VtreeIdx(vi as u32)) {
            VtreeNode::Internal { left, right, .. } => (left, right),
            VtreeNode::Leaf { .. } => continue,
        };
        let level = &tdd.levels[vi];
        if level.width() == 0 { continue; }
        let left_is_leaf = vtree.node(left).is_leaf();
        let right_is_leaf = vtree.node(right).is_leaf();
        if !left_is_leaf && !right_is_leaf { continue; }
        for (_, pairs) in level.internal_inputs_iter() {
            for p in pairs {
                if left_is_leaf {
                    let idx = p.left.idx();
                    if idx < LEAF_WIDTH { used_at_leaf[left.idx()] |= 1u8 << idx; }
                }
                if right_is_leaf {
                    let idx = p.right.idx();
                    if idx < LEAF_WIDTH { used_at_leaf[right.idx()] |= 1u8 << idx; }
                }
            }
        }
    }
    if vtree.node(tdd.output.vtree).is_leaf() && tdd.output.local != ZERO {
        let idx = tdd.output.local.idx();
        if idx < LEAF_WIDTH { used_at_leaf[tdd.output.vtree.idx()] |= 1u8 << idx; }
    }
    let pos_mask: u8 = 1u8 << (LeafLabel::Pos as u32);
    let neg_mask: u8 = 1u8 << (LeafLabel::Neg as u32);
    let one_mask: u8 = 1u8 << (LeafLabel::One as u32);
    for vi in 0..n {
        if !vtree.node(VtreeIdx(vi as u32)).is_leaf() { continue; }
        let u = used_at_leaf[vi];
        let has_one = (u & one_mask) != 0;
        let has_lit = (u & (pos_mask | neg_mask)) != 0;
        if has_one && has_lit {
            return Err(format!(
                "leaf vtree {:?} mixes label modes (used = 0b{:03b}: Pos={}, \
                 Neg={}, One={}); expected subset of {{Pos,Neg}} or {{One}}, \
                 not both — Pos ∧ One ≠ ⊥",
                vi, u,
                (u & pos_mask) != 0, (u & neg_mask) != 0, has_one,
            ));
        }
    }

    for t in vtree.bottomup() {
        if vtree.node(t).is_leaf() { continue; }
        let width = tdd.effective_width(t);
        for i in 0..width {
            for j in (i + 1)..width {
                let tdd_i = tdd_with_output(tdd, &shared_vtree, t, i as u32);
                let tdd_j = tdd_with_output(tdd, &shared_vtree, t, j as u32);

                let conjoined = apply_and(tdd_i, tdd_j);
                let count = model_count(&conjoined);

                if count != BigUint::ZERO {
                    return Err(format!(
                        "vtree {:?} nodes {} and {} are not mutually exclusive \
                         (conjunction has {} models)",
                        t, i, j, count
                    ));
                }
            }
        }
    }

    Ok(())
}

/// Run all fast invariant checks (structure + no_false_nodes + canonicity).
///
/// Convenience wrapper that runs the three cheapest checks in sequence.
/// Suitable for use on any compiled TDD, including large easy benchmarks.
///
/// Cost: O(TDD size).
pub fn check_all_fast(tdd: &Tdd, label: &str) {
    validate_vtree_structure(tdd)
        .unwrap_or_else(|e| panic!("{}: vtree structure: {}", label, e));
    check_no_false_nodes(tdd)
        .unwrap_or_else(|e| panic!("{}: no_false_nodes: {}", label, e));
    check_canonicity(tdd, 3)
        .unwrap_or_else(|e| panic!("{}: canonicity: {}", label, e));
}

/// Run all invariant checks including minimize soundness and reduced size sanity.
///
/// **Mutates `tdd`** (calls minimize once via `check_minimize_soundness`).
/// Suitable only for moderately-sized TDDs — see individual checker docs for costs.
pub fn check_all_deep(tdd: &mut Tdd, label: &str) {
    validate_vtree_structure(tdd)
        .unwrap_or_else(|e| panic!("{}: vtree structure: {}", label, e));
    check_no_false_nodes(tdd)
        .unwrap_or_else(|e| panic!("{}: no_false_nodes: {}", label, e));
    check_canonicity(tdd, 3)
        .unwrap_or_else(|e| panic!("{}: canonicity: {}", label, e));
    check_minimize_soundness(tdd, 3)
        .unwrap_or_else(|e| panic!("{}: minimize_soundness: {}", label, e));
    check_reduced_size_sanity(tdd)
        .unwrap_or_else(|e| panic!("{}: reduced_size_sanity: {}", label, e));
    // Also call reduced_tdd_size to trigger its inline debug_assert!s
    let _ = reduced_tdd_size(tdd);
}


pub mod marg;

#[cfg(test)]
mod invariants_tests;
