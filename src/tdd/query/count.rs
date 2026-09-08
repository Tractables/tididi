//! Model counting on compiled TDDs.
//!
//! Bottom-up hybrid u128/BigUint semiring evaluation: uses native u128
//! arithmetic for most nodes, falling back to `BigUint` only where overflow
//! occurs.

use num_bigint::BigUint;

use crate::vtree::{VarId, VtreeIdx};

// The overflow sentinel and the hybrid column live in `counts` — ONE
// discipline shared with the in-apply streaming and finished-Tdd marginalize
// contexts. `OVERFLOW` is a local alias, not a second definition.
use crate::tdd::counts::{Count, CountRead, CountVec, RecoveryPanic, STREAM_OVERFLOW as OVERFLOW};
use crate::tdd::types::*;

// The column-lifetime policy is shared with `counts::ensure_fold_walk` — one
// definition for "when does a bottom-up pass's column die". Re-exported so
// external callers of the `pub` counter constructors can name it (`counts` is
// a crate-private module).
pub use crate::tdd::counts::ColumnRetention;

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
/// use tididi::tdd::Tdd;
/// use tididi::tdd::query::model_count;
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
#[doc(hidden)]
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
#[doc(hidden)]
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
#[doc(hidden)]
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
fn leaf_seed_u128(label: LeafLabel, pin: Option<bool>) -> u128 {
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
fn leaf_seed_u128_fix(label: LeafLabel, pin: Option<bool>) -> u128 {
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

/// Seed one leaf vtree level `ti` (hybrid counts) under variable pin `pin`. `fix`
/// selects the FIX convention (pinned var counted ×1, see [`leaf_seed_u128_fix`])
/// over the freed convention (×2, [`leaf_seed_u128`]).
#[inline]
fn hybrid_seed_leaf(cols: &mut [CountVec<RecoveryPanic>], ti: usize, pin: Option<bool>, fix: bool) {
    for i in 0..LEAF_WIDTH {
        let seed = if fix {
            leaf_seed_u128_fix(LeafLabel::from_idx(i), pin)
        } else {
            leaf_seed_u128(LeafLabel::from_idx(i), pin)
        };
        cols[ti].set_i(i, Count::from_u128(seed));
    }
}

/// Recompute one internal vtree level `t` with u128-primary arithmetic, spilling a node
/// to the `BigUint` side-table only on overflow — the pinned mirror of
/// [`model_count_hybrid`]'s internal pass. Reads both children's columns (already
/// computed), writes `cols[t]`. The sentinel ⟺ big-slot invariant, the exact-max
/// promotion, and the stale-overflow clear on recompute (a node may stop overflowing
/// when pins change) are all owned by [`CountVec::set`]/[`Count::from_u128`].
fn hybrid_recompute_internal(tdd: &Tdd, cols: &mut [CountVec<RecoveryPanic>], t: VtreeIdx) {
    let ti = t.idx();
    let level = &tdd.levels[ti];
    if level.is_marginal() {
        let ic = level.marginal_counts.as_ref().unwrap();
        for (i, &c) in ic.iter().enumerate() {
            if c == OVERFLOW {
                let bv = level
                    .marginal_counts_big
                    .as_ref()
                    .and_then(|m| m.get(i).cloned())
                    .expect("marginal OVERFLOW slot without a big entry — level invariant violated");
                cols[ti].set_i(i, Count::Big(bv));
            } else {
                cols[ti].set_i(i, Count::from_u128(c));
            }
        }
        return;
    }
    let (left_child, right_child) = tdd.vtree.children(t);
    let li = left_child.idx();
    let ri = right_child.idx();
    let li_marg = tdd.levels[li].is_marginal();
    let ri_marg = tdd.levels[ri].is_marginal();
    for (i, pairs) in level.internal_inputs_iter() {
        let mut total: u128 = 0;
        let mut overflowed = false;
        for pair in pairs {
            let lc = match resolve_marg_ref(pair.left.0, li_marg) {
                MargResolved::Inline(c) => c as u128,
                MargResolved::Index(idx) => cols[li].fast_val(idx),
            };
            // Zero-operand pairs (left subfunction UNSAT under the pins) contribute
            // 0·rc = 0: skip without even resolving rc. On the pinned cofactor eval these
            // dominate — pinning the relaxed vars leaves 50–90% of pairs with a zero
            // operand — so this short-circuit avoids the bulk of the multiplies.
            if lc == 0 {
                continue;
            }
            let rc = match resolve_marg_ref(pair.right.0, ri_marg) {
                MargResolved::Inline(c) => c as u128,
                MargResolved::Index(idx) => cols[ri].fast_val(idx),
            };
            if rc == 0 {
                continue;
            }
            // OVERFLOW × 1 wouldn't trip checked_mul, so test the sentinel explicitly.
            if lc == OVERFLOW || rc == OVERFLOW {
                overflowed = true;
                break;
            }
            match lc.checked_mul(rc).and_then(|p| total.checked_add(p)) {
                Some(v) => total = v,
                None => {
                    overflowed = true;
                    break;
                }
            }
        }
        if !overflowed {
            // `from_u128` owns the exact-max promotion (a natural total of exactly
            // u128::MAX routes to Big so parents reading the sentinel find a big
            // entry); `set` owns the stale-overflow clear.
            cols[ti].set_i(i, Count::from_u128(total));
        } else {
            let mut bt = BigUint::ZERO;
            for pair in level.pairs_iter_of_idx(i) {
                // Resolve each operand to (u128 view, Some(&big) iff it overflowed).
                // Skip zero operands before any allocation — zero is always a clean
                // u128 (only OVERFLOW forces a big read). Then dispatch by width:
                //   both small  → u128 mul (no BigUint operand allocs at all);
                //   mixed       → scalar mul `&big * u128` (no small-operand alloc,
                //                 faster than promoting to BigUint + general mul);
                //   both big    → `&big * &big`, operands borrowed not cloned.
                let (lu, lbig) = match resolve_marg_ref(pair.left.0, li_marg) {
                    MargResolved::Inline(0) => continue,
                    MargResolved::Inline(c) => (c as u128, None),
                    MargResolved::Index(idx) => match cols[li].fast_val(idx) {
                        0 => continue,
                        OVERFLOW => (OVERFLOW, Some(sentinel_big(&cols[li], idx))),
                        v => (v, None),
                    },
                };
                let (ru, rbig) = match resolve_marg_ref(pair.right.0, ri_marg) {
                    MargResolved::Inline(0) => continue,
                    MargResolved::Inline(c) => (c as u128, None),
                    MargResolved::Index(idx) => match cols[ri].fast_val(idx) {
                        0 => continue,
                        OVERFLOW => (OVERFLOW, Some(sentinel_big(&cols[ri], idx))),
                        v => (v, None),
                    },
                };
                match (lbig, rbig) {
                    (None, None) => match lu.checked_mul(ru) {
                        Some(p) => bt += p,
                        None => bt += BigUint::from(lu) * BigUint::from(ru),
                    },
                    (Some(lb), None) => bt += lb * ru,
                    (None, Some(rb)) => bt += rb * lu,
                    (Some(lb), Some(rb)) => bt += lb * rb,
                }
            }
            cols[ti].set_i(i, Count::Big(bt));
        }
    }
}

/// Borrow the big value of a slot known to hold the overflow sentinel.
/// Precondition: `cols.fast_val(node) == OVERFLOW` (then `big_val` is `Some`
/// by the `CountVec` invariant).
#[inline]
fn sentinel_big(col: &CountVec<RecoveryPanic>, node: usize) -> &BigUint {
    col.big_val(node)
        .expect("CountVec: sentinel fast slot without a big value — invariant violated")
}

/// Incremental pinned model counter for a Gray-code cofactor sum.
///
/// One hybrid `CountVec` column per vtree level (u128-primary, `BigUint` side
/// table on overflow — keeps 99%+ of arithmetic off the heap; the discipline
/// shared with the apply/marginalize contexts, see `tididi/src/tdd/counts.rs`). Holds
/// the full per-node count array; after one [`full_recompute`](Self::full_recompute), flipping a
/// few variables' pins and calling [`recompute_levels`](Self::recompute_levels) on just
/// the affected vtree levels (the "dirty cone" from those leaves to the root) updates the
/// root count in `O(cone)` instead of the `O(|D|)` of a fresh full pass — every unaffected
/// node's cached count is reused verbatim. The result equals [`model_count_pinned_bigint`].
pub struct IncrementalPinnedCounter {
    cols: Vec<CountVec<RecoveryPanic>>,
    pins: Vec<Option<bool>>,
    /// FIX convention (pinned var ×1) when true; freed convention (×2) when false.
    fix: bool,
    /// Column-lifetime policy for [`full_recompute`](Self::full_recompute); see
    /// [`ColumnRetention`]. `Frontier` makes the counter root-read-only.
    retain: ColumnRetention,
}

impl IncrementalPinnedCounter {
    /// Allocate the count array with pin slots `0..n_pins`. No pass run yet.
    /// Uses the freed (×2) convention; see [`new_with_fix`](Self::new_with_fix). The
    /// production pinned-count path uses `new_with_fix(.., true, ..)`; the freed default
    /// serves [`model_count_hybrid`] (zero pins — the conventions coincide there) and
    /// the differential-test reference counter.
    pub(crate) fn new(tdd: &Tdd, n_pins: usize, retain: ColumnRetention) -> Self {
        Self::new_with_fix(tdd, n_pins, false, retain)
    }

    /// Like `new` but selects the seed convention: `fix=true` counts a
    /// pinned variable ×1 (the production pinned-count convention), `fix=false`
    /// frees it ×2.
    ///
    /// `retain` is the column-lifetime policy ([`ColumnRetention`]):
    /// - `All` allocates every level's column up front and keeps it. Required by
    ///   [`recompute_levels`](Self::recompute_levels) (the Gray-code dirty-cone
    ///   update re-reads cached columns) and by
    ///   `into_fast_counts` (which hands the whole array
    ///   out). Both fail fast under `Frontier`.
    /// - `Frontier` allocates each column only when the pass writes it and frees
    ///   each child as its parent completes, so peak is the pass frontier rather
    ///   than the whole diagram. ROOT-ONLY: the only legal read afterwards is
    ///   [`root_count`](Self::root_count).
    ///
    /// The counter OWNS its `cols`/`pins` arrays (sized from `tdd` here) and does
    /// not borrow `tdd` — every method takes `tdd` as an argument. This lets a caller keep
    /// one counter alive across many evaluations of the SAME diagram (re-pinning + a dirty-
    /// cone [`recompute_levels`](Self::recompute_levels) under `All`, or re-pinning + a
    /// fresh [`full_recompute`](Self::full_recompute) under `Frontier`, instead of a new
    /// allocation each time). Callers MUST pass the same `tdd` the counter was sized from;
    /// passing a structurally different diagram is a logic error (the arrays would be
    /// mis-sized).
    pub fn new_with_fix(tdd: &Tdd, n_pins: usize, fix: bool, retain: ColumnRetention) -> Self {
        let cols = (0..tdd.vtree.num_nodes())
            .map(|i| match retain {
                ColumnRetention::All => {
                    CountVec::with_width(tdd.effective_width(VtreeIdx(i as u32)))
                }
                // Frontier: allocate on write (`ensure_col`), free on parent
                // completion — pre-sizing here would commit the whole-diagram
                // array this policy exists to avoid.
                ColumnRetention::Frontier => CountVec::with_width(0),
            })
            .collect();
        Self {
            cols,
            pins: vec![None; n_pins],
            fix,
            retain,
        }
    }

    /// Set one variable's pin (does not recompute). `var.idx()` must be `< n_pins`.
    #[inline]
    pub fn set_pin(&mut self, var: VarId, val: Option<bool>) {
        self.pins[var.idx()] = val;
    }

    /// (Re)allocate `cols[ti]` to level `ti`'s effective width when it is not
    /// already that size. A no-op under [`ColumnRetention::All`] (the
    /// constructor pre-sized every column and nothing shrinks them), so this is
    /// one length compare per level on that path; under `Frontier` it is the
    /// allocate-on-write step for a column that starts — or was freed — empty.
    /// A freshly allocated column is all-zero, which is exactly what a fresh
    /// counter's column holds, so slots no pass writes (tombstones, which
    /// `internal_inputs_iter` skips) read the same under both policies.
    #[inline]
    fn ensure_col(&mut self, tdd: &Tdd, ti: usize) {
        let w = tdd.effective_width(VtreeIdx(ti as u32));
        if self.cols[ti].len() != w {
            self.cols[ti] = CountVec::with_width(w);
        }
    }

    /// Full bottom-up pass under the current pins (every leaf + every internal level).
    /// Call once for the starting Gray-code state — or once per pin assignment when
    /// the counter is `Frontier` (which has no incremental path).
    pub fn full_recompute(&mut self, tdd: &Tdd) {
        let out_t = tdd.output.vtree.idx();
        if self.retain == ColumnRetention::Frontier {
            // Free-before-rebuild: drop the previous pass's surviving column
            // (the root's, plus any level this pass will not revisit) BEFORE
            // allocating anything new, so two passes' peaks never overlap.
            for c in &mut self.cols {
                *c = CountVec::with_width(0);
            }
        }
        for (t, var) in tdd.vtree.leaf_bottomup() {
            let pin = self.pins.get(var.idx()).copied().flatten();
            self.ensure_col(tdd, t.idx());
            hybrid_seed_leaf(&mut self.cols, t.idx(), pin, self.fix);
        }
        for (t, l, r) in tdd.vtree.internal_bottomup() {
            self.ensure_col(tdd, t.idx());
            hybrid_recompute_internal(tdd, &mut self.cols, t);
            if self.retain == ColumnRetention::Frontier {
                // The vtree is a tree: `t` is the ONE parent of `l`/`r`, so
                // their columns are dead now that `t`'s is complete. `out_t` is
                // the single column read after the pass — it is the root under
                // the output-at-root invariant (hence never a child here), but
                // an all-backbone compile can collapse the output onto a LEAF
                // level that IS a child, so the guard is load-bearing.
                for c in [l.idx(), r.idx()] {
                    if c != out_t {
                        self.cols[c] = CountVec::with_width(0);
                    }
                }
            }
        }
    }

    /// Recompute exactly `levels`, in the given order — which MUST be children-before-
    /// parents (a `bottomup_topo`-ordered subset). Leaf levels are re-seeded from the
    /// current pins; internal levels are re-summed from their (already-updated) children.
    ///
    /// # Panics
    ///
    /// Panics unless the counter was built with [`ColumnRetention::All`] — the
    /// dirty-cone update reads cached columns outside `levels`, which
    /// `Frontier` frees as parents complete.
    pub fn recompute_levels(&mut self, tdd: &Tdd, levels: &[VtreeIdx]) {
        assert_eq!(
            self.retain,
            ColumnRetention::All,
            "recompute_levels requires ColumnRetention::All: the dirty-cone update re-reads \
             cached child columns, which ColumnRetention::Frontier frees as parents complete"
        );
        for &t in levels {
            if tdd.vtree.node(t).is_leaf() {
                let var = tdd.vtree.leaf_var(t);
                let pin = self.pins.get(var.idx()).copied().flatten();
                hybrid_seed_leaf(&mut self.cols, t.idx(), pin, self.fix);
            } else {
                hybrid_recompute_internal(tdd, &mut self.cols, t);
            }
        }
    }

    /// The current root (output) model count.
    #[inline]
    pub fn root_count(&self, tdd: &Tdd) -> BigUint {
        let (t, i) = (tdd.output.vtree.idx(), tdd.output.local.idx());
        match self.cols[t].get(i) {
            CountRead::Fast(v) => BigUint::from(v),
            CountRead::Big(b) => b.clone(),
        }
    }

    /// Consume the counter, returning the per-node u128 count columns
    /// (`fast[t][i]`) and discarding the `BigUint` side table. A slot that
    /// counted past `u128` saturates to `OVERFLOW` (`u128::MAX`) and its exact
    /// magnitude is dropped. ZERO is exact: the u128 array is authoritative for
    /// zero — only a *non-zero* overflow ever spills to the Big side table — so
    /// `fast[t][i] == 0` iff node `(t,i)` has no models. For callers that need
    /// only monotone ordering, a small-threshold compare, and exact-zero
    /// detection (sat-prune MC-priority), never an overflowed node's exact value.
    ///
    /// # Panics
    ///
    /// Panics unless the counter was built with [`ColumnRetention::All`] —
    /// `Frontier` keeps only the root column, so there is no per-node array to
    /// hand out.
    pub(crate) fn into_fast_counts(self) -> Vec<Vec<u128>> {
        assert_eq!(
            self.retain,
            ColumnRetention::All,
            "into_fast_counts requires ColumnRetention::All: ColumnRetention::Frontier keeps \
             only the root column, so the per-node array does not exist"
        );
        self.cols.into_iter().map(|c| c.into_parts().0).collect()
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
#[doc(hidden)]
pub fn node_counts_u128(tdd: &Tdd) -> Vec<Vec<u128>> {
    // `ColumnRetention::All`: this caller's whole product IS the per-level
    // column array, so no column may be released mid-pass.
    let mut ctr = IncrementalPinnedCounter::new(tdd, 0, ColumnRetention::All);
    ctr.full_recompute(tdd);
    ctr.into_fast_counts()
}

