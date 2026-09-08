//! Model counting on compiled TDDs.
//!
//! Bottom-up hybrid u128/BigUint semiring evaluation: uses native u128
//! arithmetic for most nodes, falling back to `BigUint` only where overflow
//! occurs.

mod hybrid;

pub use hybrid::IncrementalPinnedCounter;

use num_bigint::BigUint;

use crate::vtree::VtreeIdx;

// The overflow sentinel and the hybrid column live in `counts` — ONE
// discipline shared with the in-apply streaming and finished-Tdd marginalize
// contexts. `OVERFLOW` is a local alias, not a second definition.
use crate::counts::STREAM_OVERFLOW as OVERFLOW;
use crate::diagram::*;

// The column-lifetime policy is shared with `counts::ensure_fold_walk` — one
// definition for "when does a bottom-up pass's column die". Re-exported so
// external callers of the `pub` counter constructors can name it (`counts` is
// a crate-private module).
pub use crate::counts::ColumnRetention;

// ── Model counting ───────────────────────────────────────────────────────────

/// Count the number of satisfying assignments (models) of a TDD.
///
/// Uses hybrid u128/BigUint arithmetic: u128 for most nodes (no heap
/// allocation), `BigUint` only where overflow occurs. `compute_node_counts`
/// provides a full `BigUint` fallback for callers that need per-node counts.
///
/// ```
/// use std::sync::Arc;
/// use num_bigint::BigUint;
/// use tididi::Tdd;
/// use tididi::query::model_count;
/// use tididi::vtree::Vtree;
///
/// let vtree = Arc::new(Vtree::balanced(3));
/// let f = Tdd::clause(&vtree, [1, 2, 3]); // x1 ∨ x2 ∨ x3
/// assert_eq!(model_count(&f), BigUint::from(7u32)); // 2^3 − 1
/// ```
///
/// # Panics
///
/// Panics if `tdd` is poisoned (a mid-rewrite `OverBudget` left it in an
/// inconsistent state); the caller must drop and recover instead of counting it.
pub fn model_count(tdd: &Tdd) -> BigUint {
    // A poisoned diagram carries an unreliable count: `contract_twins` hit an
    // OverBudget mid parent-rewrite (W2) and left the structure inconsistent.
    // Every count consumer must have bailed to its recovery path before reaching
    // here; counting a poisoned diagram is a soundness bug, so trip loudly.
    assert!(
        !tdd.scratch.poisoned,
        "model_count called on a poisoned TDD (contract_twins W2 mid-rewrite OverBudget); \
         the caller must drop the diagram and recover instead of counting it"
    );
    if tdd.is_zero() {
        return BigUint::ZERO;
    }
    model_count_hybrid(tdd)
}

/// BigUint reference for freed-convention (×2) pinned counting — a full-precision
/// pass with no u128
/// fast path. Used as the differential-test oracle for the u128-hybrid counter,
/// including by the downstream compiler crate's tests — so `pub` and not
/// `#[cfg(test)]`-gated (dependency crates never see `cfg(test)`).
#[cfg(test)]
pub fn model_count_pinned_bigint(tdd: &Tdd, pins: &[Option<bool>]) -> BigUint {
    if tdd.is_zero() {
        return BigUint::ZERO;
    }
    let counts = compute_node_counts_pinned(tdd, pins);
    let (out_t, out_i) = (tdd.output.vtree.idx(), tdd.output.local.idx());
    counts[out_t][out_i].clone()
}

/// Like [`model_count_pinned_bigint`] but with the CLEAN-FIX seed convention (pinned vars
/// counted ×1, not freed ×2). See `leaf_seed_big_fix`.
///
/// FIX-convention differential-test oracle for the u128-hybrid counter
/// ([`IncrementalPinnedCounter`] with `fix = true`), plus the reference
/// spelling of the pinned readout: with the own-show leaves marginalized and
/// the boundary vars left Boolean, pinning a boundary assignment and counting
/// yields that assignment's boundary-function entry, marginal tagging decoded
/// internally (never read `marginal_counts` raw).
///
/// COST: one full-precision `BigUint` bottom-up pass with no u128 fast path,
/// allocating a per-node count column for every level, per call. The
/// structured-count boundary readout — a component-boundary counting pass
/// that lives in the downstream driver crate — calls a pinned count once per
/// boundary assignment — `2^|boundary|` times on the same diagram —
/// therefore runs on the hybrid counter instead, with this function as its
/// oracle.
#[cfg(test)]
pub fn model_count_pinned_fix(tdd: &Tdd, pins: &[Option<bool>]) -> BigUint {
    if tdd.is_zero() {
        return BigUint::ZERO;
    }
    let counts = compute_node_counts_pinned_mode(tdd, pins, true);
    let (out_t, out_i) = (tdd.output.vtree.idx(), tdd.output.local.idx());
    counts[out_t][out_i].clone()
}

/// Compute per-node model counts using `BigUint` arithmetic (arbitrary precision).
///
/// Returns a 2D array `counts[vtree_idx][node_idx]` = number of satisfying
/// assignments for each TDD node. Used by `model_count`, `reduced_tdd_size`,
/// and `check_reduced_size_sanity` in `invariants.rs`.
pub fn compute_node_counts(tdd: &Tdd) -> Vec<Vec<BigUint>> {
    compute_node_counts_pinned(tdd, &[])
}

// ── Leaf-seed tables ─────────────────────────────────────────────────────────
// The four functions below form a 2×2 matrix that's easy to confuse:
//   precision { BigUint, u128 } × convention { FREED (×2), FIX (×1) }.
// FIX (`*_fix`) is the production pinned-count convention — exact even for
// coupled copies. FREED is retained only as the differential-test reference.
// Each u128 variant mirrors its BigUint twin exactly (seeds are always 0/1/2).

/// `BigUint` seed for a leaf node `label` under an optional pin on its variable —
/// the FREED (×2) convention.
///
/// A pin reproduces EXACTLY what `condition_var` does to a leaf, but via a seed
/// override instead of a pair rewrite + minimize: the inconsistent polarity branch
/// is dropped (→0) and the consistent branch is FREED (→×2, i.e. set to `One`),
/// not fixed to ×1. Conditioning frees the kept copy and the caller divides by
/// `2^(#conditioned)` to correct.
///
/// This is no longer the production eval convention — that is now FIX
/// (×1, [`leaf_seed_big_fix`]), which is exact even for coupled copies and divides
/// out only the genuinely-free vars (`2^(n_hubs + n_free_copies)`). A copy's ×2
/// freedom can live inside a *marginalized* node, out of reach of any leaf-level
/// override; the FIX path divides out exactly those, so it does not under-count
/// them. The freed seed is retained only as the
/// differential-test reference (the conditioning recovery fallback was removed).
///
/// - `None` (free): `One`→2, `Pos`/`Neg`→1 (the literal determines the var).
/// - `Some(v)`: `Pos`/`Neg`→2 if the literal agrees with `v` (freed), else 0
///   (dropped); `One`→2 (already free).
fn leaf_seed_big(label: LeafLabel, pin: Option<bool>) -> BigUint {
    match (label, pin) {
        (LeafLabel::Zero, _) => BigUint::ZERO,
        (LeafLabel::One, _) => BigUint::from(2u32),
        (LeafLabel::Pos, None) => BigUint::from(1u32),
        (LeafLabel::Neg, None) => BigUint::from(1u32),
        (LeafLabel::Pos, Some(false)) => BigUint::ZERO,
        (LeafLabel::Pos, Some(true)) => BigUint::from(2u32),
        (LeafLabel::Neg, Some(true)) => BigUint::ZERO,
        (LeafLabel::Neg, Some(false)) => BigUint::from(2u32),
    }
}

/// Native-u128 mirror of [`leaf_seed_big`] (leaf seeds are 0/1/2, always in range).
#[inline]
pub(super) fn leaf_seed_u128(label: LeafLabel, pin: Option<bool>) -> u128 {
    match (label, pin) {
        (LeafLabel::Zero, _) => 0,
        (LeafLabel::One, _) => 2,
        (LeafLabel::Pos, None) | (LeafLabel::Neg, None) => 1,
        (LeafLabel::Pos, Some(true)) | (LeafLabel::Neg, Some(false)) => 2,
        (LeafLabel::Pos, Some(false)) | (LeafLabel::Neg, Some(true)) => 0,
    }
}

/// Native-u128 mirror of [`leaf_seed_big_fix`] (FIX convention; seeds are 0/1/2).
#[inline]
pub(super) fn leaf_seed_u128_fix(label: LeafLabel, pin: Option<bool>) -> u128 {
    match (label, pin) {
        (LeafLabel::Zero, _) => 0,
        (LeafLabel::One, None) => 2,
        (LeafLabel::One, Some(_)) => 1,
        (LeafLabel::Pos, None) | (LeafLabel::Neg, None) => 1,
        (LeafLabel::Pos, Some(true)) | (LeafLabel::Neg, Some(false)) => 1,
        (LeafLabel::Pos, Some(false)) | (LeafLabel::Neg, Some(true)) => 0,
    }
}

/// CLEAN-FIX leaf seed (the production pinned-count convention since the FIX
/// switch-over): a pinned variable is *fixed* (counted ×1), not *freed* (×2).
/// Agree→1, disagree→0, true-constant under a pin→1 (the var is determined). Free
/// (unpinned) leaves are unchanged from `leaf_seed_big`. Recovery with this seed is
/// the plain restricted model count: divide the diagonal sum only by `2^n_hubs`
/// (free original hub vars) and `2^n_free_copies` (copies that ended up free —
/// eliminated/marginalized), NOT by `2^n_copies`. Pinning ×1 is exact even when a
/// copy is coupled, so co-occurring hubs recover exactly instead of being refused.
#[cfg(test)]
fn leaf_seed_big_fix(label: LeafLabel, pin: Option<bool>) -> BigUint {
    match (label, pin) {
        (LeafLabel::Zero, _) => BigUint::ZERO,
        (LeafLabel::One, None) => BigUint::from(2u32),
        (LeafLabel::One, Some(_)) => BigUint::from(1u32),
        (LeafLabel::Pos, None) => BigUint::from(1u32),
        (LeafLabel::Neg, None) => BigUint::from(1u32),
        (LeafLabel::Pos, Some(false)) => BigUint::ZERO,
        (LeafLabel::Pos, Some(true)) => BigUint::from(1u32),
        (LeafLabel::Neg, Some(true)) => BigUint::ZERO,
        (LeafLabel::Neg, Some(false)) => BigUint::from(1u32),
    }
}

/// Like [`compute_node_counts`] but with optional per-variable pins indexed by
/// `VarId::idx()`. A pin fixes that variable's value during the upward count pass
/// (see [`leaf_seed_big`]); out-of-range or `None` entries leave the variable
/// free. Used by evaluation-based pinned cofactor recovery, which sums the
/// count over diagonal assignments with no circuit conditioning.
pub(crate) fn compute_node_counts_pinned(tdd: &Tdd, pins: &[Option<bool>]) -> Vec<Vec<BigUint>> {
    let mut counts = alloc_count_array(tdd);
    for (t, var) in tdd.vtree.leaf_bottomup() {
        let pin = pins.get(var.idx()).copied().flatten();
        seed_leaf_level(&mut counts, t.idx(), pin);
    }
    for (t, _left, _right) in tdd.vtree.internal_bottomup() {
        recompute_internal_level(tdd, &mut counts, t);
    }
    counts
}

/// `compute_node_counts_pinned` with a seed-convention switch. `fix=false` is the
/// FREED seed (`leaf_seed_big`, the differential-test reference); `fix=true` is the
/// clean-fix seed (`leaf_seed_big_fix`), reached in production through
/// [`model_count_pinned_fix`]. Reuses the shared
/// `alloc_count_array`/`recompute_internal_level` helpers.
#[cfg(test)]
pub(crate) fn compute_node_counts_pinned_mode(
    tdd: &Tdd,
    pins: &[Option<bool>],
    fix: bool,
) -> Vec<Vec<BigUint>> {
    if !fix {
        return compute_node_counts_pinned(tdd, pins);
    }
    let mut counts = alloc_count_array(tdd);
    for (t, var) in tdd.vtree.leaf_bottomup() {
        let pin = pins.get(var.idx()).copied().flatten();
        let ti = t.idx();
        for i in 0..LEAF_WIDTH {
            counts[ti][i] = leaf_seed_big_fix(LeafLabel::from_idx(i), pin);
        }
    }
    for (t, _left, _right) in tdd.vtree.internal_bottomup() {
        recompute_internal_level(tdd, &mut counts, t);
    }
    counts
}

/// Allocate the zero-filled per-node count array `counts[vtree_idx][node_idx]`.
fn alloc_count_array(tdd: &Tdd) -> Vec<Vec<BigUint>> {
    (0..tdd.vtree.num_nodes())
        .map(|i| vec![BigUint::ZERO; tdd.effective_width(VtreeIdx(i as u32))])
        .collect()
}

/// Seed one leaf vtree level `ti` under variable pin `pin` (the leaf pass of
/// [`compute_node_counts_pinned`], factored out so the incremental counter can
/// re-seed a single flipped leaf). See [`leaf_seed_big`].
#[inline]
fn seed_leaf_level(counts: &mut [Vec<BigUint>], ti: usize, pin: Option<bool>) {
    for i in 0..LEAF_WIDTH {
        counts[ti][i] = leaf_seed_big(LeafLabel::from_idx(i), pin);
    }
}

/// Recompute one internal vtree level `t` from its children's (already-computed)
/// counts — the internal pass of [`compute_node_counts_pinned`], factored out so the
/// incremental counter can recompute just the dirty cone. Marginal levels copy their
/// pin-independent precomputed counts. Reads `counts[left]`/`counts[right]`, writes
/// `counts[t]`; callers must have computed both children first (bottom-up order).
fn recompute_internal_level(tdd: &Tdd, counts: &mut [Vec<BigUint>], t: VtreeIdx) {
    let ti = t.idx();
    let level = &tdd.levels[ti];
    if level.is_marginal() {
        let ic = level.marginal_counts.as_ref().unwrap();
        for (i, &c) in ic.iter().enumerate() {
            if c != OVERFLOW {
                counts[ti][i] = BigUint::from(c);
            } else if let Some(bv) = level.marginal_counts_big.as_ref().and_then(|b| b.get(i)) {
                counts[ti][i].clone_from(bv);
            }
        }
        return;
    }
    let (left_child, right_child) = tdd.vtree.children(t);
    let li = left_child.idx();
    let ri = right_child.idx();
    // A marg-side ref may carry an inline count (bit-30 tag) instead of a
    // slot index; decode per side. Non-marginal children index verbatim.
    let li_marg = tdd.levels[li].is_marginal();
    let ri_marg = tdd.levels[ri].is_marginal();
    for (i, pairs) in level.internal_inputs_iter() {
        let mut total = BigUint::ZERO;
        for pair in pairs {
            let lc = match resolve_marg_ref(pair.left.0, li_marg) {
                MargResolved::Inline(c) => BigUint::from(c),
                MargResolved::Index(idx) => counts[li][idx].clone(),
            };
            let rc = match resolve_marg_ref(pair.right.0, ri_marg) {
                MargResolved::Inline(c) => BigUint::from(c),
                MargResolved::Index(idx) => counts[ri][idx].clone(),
            };
            total += &lc * &rc;
        }
        counts[ti][i] = total;
    }
}

/// Hybrid model counting: u128 for most nodes, `BigUint` only where overflow occurs.
///
/// Most TDD nodes (especially at lower vtree levels) have model counts that fit
/// in u128. Only nodes near the root may overflow. This avoids creating `BigUint`
/// objects for the vast majority of nodes — on c1908 (590K nodes, 125M pairs),
/// this reduces model counting from ~3.5s to a fraction of that by keeping
/// 99%+ of arithmetic in native u128.
///
/// This is [`IncrementalPinnedCounter`] with zero pins under the freed
/// convention: an unpinned leaf seeds identically (`One`→2, `Pos`/`Neg`→1,
/// `Zero`→0) and the internal pass is the same hybrid discipline (with the
/// zero-operand short-circuit and the borrow-based mixed-magnitude repass) —
/// there is deliberately ONE counting engine, not a second whole-diagram copy.
/// Precondition (as before): `!tdd.is_zero()` — enforced by [`model_count`].
///
/// Only the root value is read, so the pass runs under
/// [`ColumnRetention::Frontier`]: each child column is freed as its parent's
/// completes, and the live set is the frontier rather than a u128 column for
/// every level at once.
pub(crate) fn model_count_hybrid(tdd: &Tdd) -> BigUint {
    let mut ctr = IncrementalPinnedCounter::new(tdd, 0, ColumnRetention::Frontier);
    ctr.full_recompute(tdd);
    ctr.root_count(tdd)
}

/// Per-node u128 model counts (`counts[vtree_idx][node_idx]`), the hybrid-
/// evaluator counterpart of [`compute_node_counts`]'s `BigUint` array. Runs the
/// SAME single bottom-up pass as `model_count_hybrid` (zero pins, freed
/// convention, identical leaf seeds / `resolve_marg_ref` / marginal handling)
/// but keeps every column instead of only the root, then drops the `BigUint` side
/// table: an overflowed slot saturates to `OVERFLOW` (`u128::MAX`), while ZERO
/// stays exact (the u128 array is authoritative for zero). Structurally it is
/// [`compute_node_counts`] with u128-primary arithmetic — no new traversal, so
/// it matches the `BigUint` pass node-for-node on every non-overflowing slot.
///
/// Used by sat-prune MC-priority, whose consumers need only monotone ordering,
/// a small-threshold flip, and exact-zero kills — none read an overflowed
/// node's exact magnitude — so this avoids the per-slot `BigUint` allocation and
/// per-pair heap multiply the `BigUint` pass pays every firing.
pub fn node_counts_u128(tdd: &Tdd) -> Vec<Vec<u128>> {
    // `ColumnRetention::All`: this caller's whole product IS the per-level
    // column array, so no column may be released mid-pass.
    let mut ctr = IncrementalPinnedCounter::new(tdd, 0, ColumnRetention::All);
    ctr.full_recompute(tdd);
    ctr.into_fast_counts()
}

