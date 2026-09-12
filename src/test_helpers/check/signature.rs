//! Probabilistic equivalence testing: random weights in Z_p and the
//! bottom-up signature evaluation the canonicity checks compare.
use std::sync::Arc;
use crate::diagram::{ChildRef, ValueRef};

use num_bigint::BigUint;
use rand::RngExt;
use rand::rngs::SmallRng;
use crate::vtree::VtreeIdx;
use crate::diagram::*;


// ── EvalAlgebra helpers (probabilistic equivalence testing) ─────────────────────
//
// To check whether two diagram nodes compute the same Boolean function without
// enumerating all 2^n assignments, we evaluate both in a finite-field semiring
// (Z_p, +, ×) with random variable weights. By the Schwartz–Zippel lemma,
// two distinct multilinear polynomials agree on a random point with probability
// ≤ degree/p. With p = 2^61 − 1 and degree ≤ num_vars, false negatives are
// astronomically unlikely (~10^{-18} per round).

/// Mersenne prime 2^61 − 1, used for probabilistic polynomial identity testing.
pub(super) const PRIME: u128 = (1 << 61) - 1;

/// Modular multiplication mod `PRIME`, exploiting the Mersenne structure to
/// avoid full 128-bit division.
pub(super) fn mod_mul(a: u64, b: u64) -> u64 {
    let product = a as u128 * b as u128;
    let lo = product & PRIME;
    let hi = product >> 61;
    let result = lo + hi;
    let result = if result >= PRIME { result - PRIME } else { result };
    result as u64
}

pub(super) fn mod_add(a: u64, b: u64) -> u64 {
    let sum = a as u128 + b as u128;
    (if sum >= PRIME { sum - PRIME } else { sum }) as u64
}

/// Generate random variable assignments for semiring evaluation.
pub(crate) fn random_var_assignments(num_vars: usize, rng: &mut SmallRng) -> (Vec<u64>, Vec<u64>) {
    let mut pos_val = vec![0u64; num_vars];
    let mut neg_val = vec![0u64; num_vars];
    for i in 0..num_vars {
        pos_val[i] = rng.random_range(1..PRIME as u64);
        neg_val[i] = rng.random_range(1..PRIME as u64);
    }
    (pos_val, neg_val)
}

/// The factor one pair side contributes: an inline value is the scalar `k`
/// itself (`k <= MARGINAL_INLINE_MAX < PRIME`, so `k mod p == k`); an indexing
/// ref reads the child level's already-computed signature at that cell.
fn side_signature(child: ChildRef, child_signatures: &[u64]) -> u64 {
    match child {
        ChildRef::Value(ValueRef::Inline(k)) => k as u64,
        indexing => child_signatures[indexing.index().expect("an indexing ref names a cell")],
    }
}

/// Evaluate every diagram node bottom-up in the (`Z_p`, +, ×) semiring.
///
/// Returns per-level signature arrays: `signatures[vtree_idx][node_idx]` is the
/// fingerprint of that node's Boolean function under the given random assignment.
/// Two nodes with equal signatures very likely compute the same function
/// (collision probability ≤ degree/p ≈ 10^{-18} per pair).
pub(crate) fn eval_all_signatures(tdd: &Tdd, pos_val: &[u64], neg_val: &[u64]) -> Vec<Vec<u64>> {
    let vtree = &tdd.vtree;
    let mut signatures: Vec<Vec<u64>> = (0..tdd.levels.len())
        .map(|i| vec![0u64; tdd.effective_width(VtreeIdx(i as u32))])
        .collect();

    for (t, var) in vtree.leaf_bottomup() {
        let v = var.idx();
        // Leaf levels are marginal: iterate 0..`LEAF_WIDTH` using `LeafLabel::from_idx`.
        // The index is a leaf-label ordinal, not a position in one array.
        #[allow(clippy::needless_range_loop)]
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
        // A weight-marginal level keeps no integer counts; its per-node value
        // lives in the diagram's `WeightStore`, and that value is the node's
        // whole identity, so the signature is a fingerprint of the weight row
        // (`weight_mod_p`).
        if level.is_marginal() {
            if let Some(counts) = level.marginal_counts() {
                let big = level.marginal_counts_big();
                for (i, &c) in counts.iter().enumerate() {
                    let big_i = big.and_then(|b| b.get(i));
                    signatures[t.idx()][i] = count_mod_p(c, big_i);
                }
            } else if let Some(values) = tdd.weights().and_then(|ws| ws.level(t.idx())) {
                for (i, value) in values.iter().enumerate() {
                    signatures[t.idx()][i] = weight_mod_p(value);
                }
            }
            continue;
        }
        let left_view = tdd.level(left).side_view();
        let right_view = tdd.level(right).side_view();
        for (i, node) in level.nodes.iter().enumerate() {
            let mut total = 0u64;
            let mut any = false;
            for pair in level.pairs_iter_of(node) {
                any = true;
                let l = side_signature(left_view.child(pair.left), &signatures[left.idx()]);
                let r = side_signature(right_view.child(pair.right), &signatures[right.idx()]);
                total = mod_add(total, mod_mul(l, r));
            }
            if any {
                signatures[t.idx()][i] = total;
            }
        }
    }

    signatures
}

/// Fingerprint a weight-marginal node's value as a nonzero element of `Z_p`.
///
/// A weight-marginal level's per-node payload is a semiring value, not a model
/// count, so there is no arithmetic that carries it into the signature
/// recurrence the way `count_mod_p` carries an integer count. What the
/// canonicity check needs of such a node is only an identity: two nodes on one
/// level are the same node exactly when they carry the same value. So the
/// signature is a hash of the value's content — the numerator and denominator
/// of a rational in its canonical (reduced, positively-signed-denominator)
/// form, or the sign and bit pattern of a log-domain magnitude — mixed into the
/// field.
///
/// Distinct values reach distinct fingerprints with the same probability the
/// rest of the check rests on, and equal values always reach the same one,
/// which is the direction soundness needs: a collision reported here is a
/// genuine pair of equal rows, never an artifact of an unread payload. Two
/// mathematically equal log-domain values that were reached by different
/// roundings fingerprint apart, which can only hide a collision, never invent
/// one.
///
/// The result is never zero, so a fingerprinted slot is distinguishable from
/// one no pass has written.
pub(super) fn weight_mod_p(value: &WeightVal) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = rustc_hash::FxHasher::default();
    match value {
        WeightVal::ExactSmall(n) => {
            0u8.hash(&mut hasher);
            n.hash(&mut hasher);
        }
        WeightVal::Exact(r) => {
            1u8.hash(&mut hasher);
            r.numer().to_signed_bytes_le().hash(&mut hasher);
            r.denom().to_signed_bytes_le().hash(&mut hasher);
        }
        WeightVal::Log(l) => {
            2u8.hash(&mut hasher);
            l.sign.hash(&mut hasher);
            // Every zero is one zero, whatever magnitude bits it carries.
            if l.sign != 0 {
                l.ln_abs.to_bits().hash(&mut hasher);
            }
        }
    }
    // FxHash mixes weakly in its high bits; one splitmix round spreads the
    // whole word before the reduction takes it modulo the field.
    let mut h = hasher.finish();
    h ^= h >> 30;
    h = h.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    h ^= h >> 27;
    h = h.wrapping_mul(0x94d0_49bb_1331_11eb);
    h ^= h >> 31;
    h % (PRIME as u64 - 1) + 1
}

/// Reduce an integer marginal count to `Z_p`. `c == u128::MAX` is the `BigUint`
/// overflow sentinel — the true value lives in the level's `marginal_counts_big`
/// side-table entry `big`.
pub(super) fn count_mod_p(c: u128, big: Option<&BigUint>) -> u64 {
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

/// Create a diagram sharing the same levels but with a different output node.
/// Used by [`check_determinism`](crate::test_helpers::check::check_determinism) to construct per-node sub-diagrams.
pub(super) fn tdd_with_output(
    tdd: &Tdd,
    vtree: &Arc<crate::vtree::Vtree>,
    vtree_node: VtreeIdx,
    local: u32,
) -> Tdd {
    Tdd::from_levels_unchecked(
        Arc::clone(vtree),
        tdd.levels.clone(),
        TddNodeId { vtree: vtree_node, local: NodeIdx(local) },
    )
}
