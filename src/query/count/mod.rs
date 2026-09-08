//! Model counting on compiled TDDs.
//!
//! Bottom-up hybrid u128/BigUint semiring evaluation: uses native u128
//! arithmetic for most nodes, falling back to `BigUint` only where overflow
//! occurs.

mod hybrid;

use crate::engine::{Engine, PollGate};
use crate::error::ApplyError;
pub use hybrid::{AllColumns, Computed, CounterState, Fresh, FrontierOnly, PinnedCounter, Retention};

use num_bigint::BigUint;

use crate::vtree::{VarId, VtreeIdx};
use crate::diagram::PairsIter;
use super::fold::{fold_bottom_up_unpolled, LevelFold, PairAlgebra, Side};

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
pub fn model_count(f: &Tdd) -> BigUint {
    Engine::new()
        .try_model_count(f)
        .expect("a fresh engine arms no stop axis")
}

/// Which leaf-seed convention a pinned count uses for a pinned variable.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum SeedConvention {
    /// The pinned variable is freed: its consistent branch counts x2. The
    /// differential-test reference.
    Freed,
    /// The pinned variable is fixed: its consistent branch counts x1. The
    /// production pinned-count convention, exact even for coupled copies.
    Fix,
}

/// Full-precision pinned model count of `tdd` under `convention`.
///
/// The oracle the u128-hybrid pinned counter ([`PinnedCounter`]) is
/// differentially tested against: one `BigUint` bottom-up pass with no u128
/// fast path, allocating a per-node count column for every level, per call. A
/// caller counting many pinned assignments of one diagram wants the hybrid
/// counter instead.
///
/// Under [`SeedConvention::Fix`] this is also the reference spelling of the
/// pinned readout: with the own-show leaves marginalized and the boundary vars
/// left Boolean, pinning a boundary assignment and counting yields that
/// assignment's boundary-function entry, marginal tagging decoded internally
/// (never read `marginal_counts` raw).
#[cfg(test)]
pub(crate) fn pinned_counts(
    tdd: &Tdd,
    pins: &[Option<bool>],
    convention: SeedConvention,
) -> BigUint {
    if tdd.is_zero() {
        return BigUint::ZERO;
    }
    let counts = compute_node_counts_pinned_mode(tdd, pins, convention);
    let (out_t, out_i) = (tdd.output.vtree.idx(), tdd.output.local.idx());
    counts[out_t][out_i].clone()
}

/// Compute per-node model counts using `BigUint` arithmetic (arbitrary precision).
///
/// Returns a 2D array `counts[vtree_idx][node_idx]` = number of satisfying
/// assignments for each TDD node. Used by `model_count`, `reduced_size`,
/// and `check_reduced_size_sanity` in `invariants.rs`.
pub fn compute_node_counts(tdd: &Tdd) -> Vec<Vec<BigUint>> {
    compute_node_counts_pinned(tdd, &[])
}

// ── The leaf seed ────────────────────────────────────────────────────────────

/// The count a leaf `label` seeds with, for a variable pinned to `pin`.
///
/// Three values, always: a leaf is a constant, a literal, or dropped. `One`
/// counts both assignments of its variable, a literal counts one, and `Zero`
/// counts none.
///
/// A pin reproduces exactly what conditioning does to a leaf, but by overriding
/// the seed instead of rewriting pairs and re-minimizing: the branch that
/// disagrees with the pin is dropped. What the agreeing branch is worth is the
/// [`SeedConvention`] — `Fix` counts the pinned variable as determined (×1),
/// which is exact even when a copy is coupled; `Freed` counts it as still free
/// (×2), leaving the caller to divide by `2^(#pinned)`.
pub(super) fn leaf_seed(label: LeafLabel, pin: Option<bool>, convention: SeedConvention) -> u128 {
    let agreeing = match convention {
        SeedConvention::Freed => 2,
        SeedConvention::Fix => 1,
    };
    let Some(v) = pin else {
        // Unpinned: the literal determines its variable, the constant does not.
        return match label {
            LeafLabel::Zero => 0,
            LeafLabel::One => 2,
            LeafLabel::Pos | LeafLabel::Neg => 1,
        };
    };
    match label {
        LeafLabel::Zero => 0,
        // Already free of the variable, so the pin only decides whether the
        // variable is still counted.
        LeafLabel::One => agreeing,
        LeafLabel::Pos => u128::from(v) * agreeing,
        LeafLabel::Neg => u128::from(!v) * agreeing,
    }
}

/// Per-node model counts in exact `BigUint`, with optional per-variable pins
/// indexed by `VarId::idx()`; out-of-range or `None` entries leave the variable
/// free.
///
/// This is the full-precision oracle: no u128 fast path, one `BigUint` per
/// node. It shares the WALK with [`PinnedCounter`] and nothing else
/// — its arithmetic is independent, which is what makes the differential test
/// between the two worth running.
pub(crate) fn compute_node_counts_pinned(tdd: &Tdd, pins: &[Option<bool>]) -> Vec<Vec<BigUint>> {
    count_big(tdd, pins, SeedConvention::Freed)
}

/// [`compute_node_counts_pinned`] under an explicit seed convention.
#[cfg(test)]
pub(crate) fn compute_node_counts_pinned_mode(
    tdd: &Tdd,
    pins: &[Option<bool>],
    convention: SeedConvention,
) -> Vec<Vec<BigUint>> {
    count_big(tdd, pins, convention)
}

fn count_big(tdd: &Tdd, pins: &[Option<bool>], convention: SeedConvention) -> Vec<Vec<BigUint>> {
    let eng = Engine::new();
    let fold = BigCounts { pins, convention };
    let mut cols: Vec<Vec<BigUint>> = (0..tdd.vtree.num_nodes())
        .map(|i| fold.alloc(&eng, tdd.effective_width(VtreeIdx(i as u32))))
        .collect();
    fold_bottom_up_unpolled(&fold, &eng, tdd, &mut cols, ColumnRetention::All, |_, _| {});
    cols
}

/// The exact-`BigUint` counting fold.
struct BigCounts<'a> {
    pins: &'a [Option<bool>],
    convention: SeedConvention,
}

impl LevelFold for BigCounts<'_> {
    type Value = BigUint;
    type Col = Vec<BigUint>;

    fn alloc(&self, _eng: &Engine, width: usize) -> Vec<BigUint> {
        vec![BigUint::ZERO; width]
    }

    fn set(&self, _eng: &Engine, col: &mut Vec<BigUint>, i: usize, v: BigUint) {
        col[i] = v;
    }

    fn leaf(&self, var: VarId, label: LeafLabel) -> BigUint {
        let pin = self.pins.get(var.idx()).copied().flatten();
        BigUint::from(leaf_seed(label, pin, self.convention))
    }

    /// A frozen level's counts are pin-independent: they were summed out before
    /// any pin existed, so they are read across verbatim.
    fn frozen_column(&self, _eng: &Engine, tdd: &Tdd, t: VtreeIdx, col: &mut Vec<BigUint>) {
        let level = &tdd.levels[t.idx()];
        let counts = level.marginal_counts().expect("a frozen level carries counts");
        for (i, &c) in counts.iter().enumerate() {
            if c != OVERFLOW {
                col[i] = BigUint::from(c);
            } else if let Some(bv) = level.marginal_counts_big().and_then(|b| b.get(i)) {
                col[i].clone_from(bv);
            }
        }
    }

    fn fold_node(
        &self,
        pairs: PairsIter<'_>,
        left: Side<'_, Vec<BigUint>>,
        right: Side<'_, Vec<BigUint>>,
    ) -> BigUint {
        self.sum_over_pairs(pairs, left, right)
    }
}

impl PairAlgebra for BigCounts<'_> {
    fn zero(&self) -> BigUint {
        BigUint::ZERO
    }
    fn read(&self, col: &Vec<BigUint>, i: usize) -> BigUint {
        col[i].clone()
    }
    fn inline(&self, count: u32) -> BigUint {
        BigUint::from(count)
    }
    fn add_assign(&self, acc: &mut BigUint, v: &BigUint) {
        *acc += v;
    }
    fn mul(&self, a: &BigUint, b: &BigUint) -> BigUint {
        a * b
    }
}

/// The model count of `tdd` under `eng`'s stop axis.
///
/// Hybrid arithmetic: u128 for most nodes, `BigUint` only where one overflows.
/// Most nodes — especially at the lower vtree levels — count well inside a
/// u128 and only nodes near the root overflow, so this keeps almost all of the
/// arithmetic off the heap.
///
/// It is a [`PinnedCounter`] with zero pins under the freed
/// convention: an unpinned leaf seeds identically (`One`→2, `Pos`/`Neg`→1,
/// `Zero`→0) and the internal pass is the same hybrid discipline. There is
/// deliberately ONE counting engine, not a second whole-diagram copy of it.
///
/// Only the root value is read, so the pass runs under
/// [`ColumnRetention::Frontier`]: each child column is freed as its parent's
/// completes, and the live set is the walk frontier rather than a column per
/// level.
///
/// # Errors
///
/// Propagates the armed stop, polled at every level boundary. Nothing has been
/// read at the cut, so the partial columns are simply dropped.
///
/// # Panics
///
/// Panics if `tdd` is poisoned: a mid parent-rewrite `OverBudget` left the
/// structure inconsistent, so its count is unreliable and every consumer must
/// have bailed to its recovery path before reaching here.
pub(crate) fn try_model_count(eng: &Engine, tdd: &Tdd) -> Result<BigUint, ApplyError> {
    assert!(
        !tdd.poisoned,
        "model count of a poisoned TDD (a mid-rewrite OverBudget left it inconsistent); \
         the caller must drop the diagram and recover instead of counting it"
    );
    if tdd.is_zero() {
        return Ok(BigUint::ZERO);
    }
    let ctr = PinnedCounter::<FrontierOnly, Fresh>::new(eng, tdd, 0, SeedConvention::Freed);
    let mut gate = PollGate::new(eng.limits().reduce_poll_stride());
    Ok(ctr.try_compute(eng, tdd, Some(&mut gate))?.root_count(tdd))
}

/// Per-node u128 model counts (`counts[vtree_idx][node_idx]`), the hybrid-
/// evaluator counterpart of [`compute_node_counts`]'s `BigUint` array. Runs the
/// SAME single bottom-up pass as [`try_model_count`] (zero pins, freed
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
    let eng = Engine::new();
    // `ColumnRetention::All`: this caller's whole product IS the per-level
    // column array, so no column may be released mid-pass.
    let ctr = PinnedCounter::<AllColumns, Fresh>::new(&eng, tdd, 0, SeedConvention::Freed);
    ctr.compute(&eng, tdd).into_fast_counts()
}

