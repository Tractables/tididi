//! Model counting on compiled diagrams.
//!
//! Bottom-up hybrid u128/BigUint semiring evaluation: uses native u128
//! arithmetic for most nodes, falling back to `BigUint` only where overflow
//! occurs.

mod incremental;

use crate::engine::Engine;
use crate::limits::PollGate;
use crate::limits::OperationError;
pub use incremental::{KeepAllColumns, KeepFrontier, ModelCounter, Retention};

use num_bigint::BigUint;

use crate::diagram::*;

// The column-lifetime policy is shared with `value::walk_bottom_up` — one
// definition for "when does a bottom-up pass's column die". Re-exported so
// external callers of the `pub` counter constructors can name it (`counts` is
// a crate-private module).
pub use crate::value::ColumnRetention;

// ── Model counting ───────────────────────────────────────────────────────────

/// Count the number of satisfying assignments (models) of a diagram.
///
/// Uses hybrid u128/BigUint arithmetic: u128 for most nodes (no heap
/// allocation), `BigUint` only where overflow occurs.
///
/// The two spellings a caller has are
/// [`Tdd::model_count`](crate::Tdd::model_count), which is this, and
/// [`Engine::model_count`](crate::Engine::model_count), which is this under a
/// caller's limits.
pub(crate) fn model_count(f: &Tdd) -> BigUint {
    Engine::new()
        .model_count(f)
        .expect("a fresh engine arms no stop axis")
}

/// The counting entry point on a diagram.
impl Tdd {
    /// Exact unweighted model count of this diagram, as an arbitrary-precision integer.
    ///
    /// Sugar over [`Engine::model_count`](crate::Engine::model_count) on a
    /// transient engine, which arms no stop, so the count cannot be cut; the
    /// contract is stated there.
    ///
    /// # Panics
    ///
    /// As [`Engine::model_count`](crate::Engine::model_count).
    ///
    /// ```
    /// use std::sync::Arc;
    /// use num_bigint::BigUint;
    /// use tididi::Tdd;
    /// use tididi::vtree::Vtree;
    ///
    /// let vtree = Arc::new(Vtree::balanced(3));
    /// let f = Tdd::clause(&vtree, [1, 2, 3]); // x1 ∨ x2 ∨ x3
    /// assert_eq!(f.model_count(), BigUint::from(7u32)); // 2^3 − 1
    /// ```
    pub fn model_count(&self) -> num_bigint::BigUint {
        crate::query::model_count(self)
    }
}

/// Whether pins count as evidence or as substitution over the unchanged vtree.
///
/// Evidence counts assignments consistent with the pins. A cofactor counts
/// the conditioned function over all variables of the vtree, including the
/// substituted variables, which are now free.
///
/// ```
/// use std::sync::Arc;
/// use tididi::{Engine, Tdd};
/// use tididi::query::{KeepAllColumns, ModelCounter, PinSemantics};
/// use tididi::vtree::{VarId, Vtree};
///
/// let engine = Engine::new();
/// let tree = Arc::new(Vtree::balanced(2));
/// let f = Tdd::clause(&tree, [1]) & Tdd::clause(&tree, [2]);
/// # tididi::test_helpers::assert_canonical(&f);
/// for (semantics, expected) in [(PinSemantics::Evidence, 1u32), (PinSemantics::Cofactor, 2)] {
///     let mut counter = ModelCounter::<KeepAllColumns>::new(&engine, &f, 2, semantics);
///     counter.set_pin(VarId(0), Some(true));
///     assert_eq!(counter.model_count(&engine), expected.into());
/// }
/// let cofactor = engine.condition_var(f, VarId(0), true).unwrap();
/// # tididi::test_helpers::assert_canonical(&cofactor);
/// assert_eq!(cofactor.model_count(), 2u32.into());
/// ```
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
#[non_exhaustive]
pub enum PinSemantics {
    /// Count the conditioned function over the unchanged vtree; each pinned
    /// variable contributes a factor of two on its agreeing branch.
    Cofactor,
    /// Count assignments consistent with the pins; each pinned variable
    /// contributes one on its agreeing branch.
    Evidence,
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
/// [`PinSemantics`] — `Evidence` counts the pinned variable as determined (×1),
/// which is exact even when a copy is coupled; `Cofactor` counts it as still free
/// (×2), leaving the caller to divide by `2^(#pinned)`.
pub(crate) fn leaf_seed(label: LeafLabel, pin: Option<bool>, convention: PinSemantics) -> u128 {
    let agreeing = match convention {
        PinSemantics::Cofactor => 2,
        PinSemantics::Evidence => 1,
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

/// The model count of `tdd` under `eng`'s stop axis.
///
/// Hybrid arithmetic: u128 per node, `BigUint` only where one overflows, which
/// keeps most of the arithmetic off the heap. It is an [`ModelCounter`]
/// with zero pins under the freed convention (`One`→2, `Pos`/`Neg`→1,
/// `Zero`→0).
///
/// Only the root value is read, so the pass runs under
/// [`ColumnRetention::Frontier`]: each child column is freed as its parent's
/// completes.
///
/// # Errors
///
/// Propagates the armed stop, polled at every level boundary. Nothing has been
/// read at the cut, so the partial columns are simply dropped.
pub(crate) fn try_model_count(eng: &Engine, tdd: &Tdd) -> Result<BigUint, OperationError> {
    let _op = eng.limits().begin_operation();
    if tdd.is_zero() {
        return Ok(BigUint::ZERO);
    }
    let mut ctr = ModelCounter::<KeepFrontier>::new(eng, tdd, 0, PinSemantics::Cofactor);
    let mut gate = PollGate::new(eng.limits().reduce_poll_stride());
    ctr.try_count(eng, Some(&mut gate))
}

/// Per-node model counts in `u128` (`counts[vtree_idx][node_idx]`), saturating a
/// slot too large for the width to `u128::MAX`; a zero count stays exact, so the
/// array is authoritative for zero.
///
/// The same bottom-up pass as [`Engine::model_count`](crate::Engine::model_count),
/// on a transient engine, keeping every column and dropping the `BigUint`
/// side table, so every non-saturating slot equals the exact count. For a
/// caller that needs ordering, a small-threshold compare or exact-zero
/// detection and never an overflowed node's magnitude. A count-marginal
/// level's column is its stored counts, one per slot; a leaf level's column
/// has three entries, one per label; every internal column of ⊥ is empty.
///
/// # Panics
///
/// As [`Engine::model_count`](crate::Engine::model_count).
#[must_use]
pub fn node_counts_u128(tdd: &Tdd) -> Vec<Vec<u128>> {
    let eng = Engine::new();
    // `ColumnRetention::All`: what this caller returns is exactly the per-level
    // column array, so no column may be released mid-pass.
    let ctr = ModelCounter::<KeepAllColumns>::new(&eng, tdd, 0, PinSemantics::Cofactor);
    ctr.into_fast_counts(&eng)
}

/// The counting entry point on a caller's engine.
impl crate::engine::Engine {
    /// The number of satisfying assignments of `tdd`, under this engine's
    /// limits.
    ///
    /// [`Tdd::model_count`](crate::Tdd::model_count) is the same count with
    /// nothing armed to interrupt it. The diagram need not be canonical; every
    /// variable of the vtree is counted, a free one contributing a factor of
    /// two. A count-marginal level is read from its stored counts; ⊥ counts
    /// zero.
    ///
    /// # Errors
    ///
    /// [`OperationError::Stopped`] when the armed
    /// deadline passes or a stop decision fires, polled at every level of the
    /// bottom-up pass. No byte budget is charged.
    ///
    /// # Panics
    ///
    /// Panics on a diagram with a weight-marginal level, whose values live in
    /// the weight store; its value is
    /// [`weighted_value`](crate::query::weighted_value).
    ///
    /// ```
    /// # use std::sync::Arc;
    /// # use std::time::Instant;
    /// # use tididi::{OperationError, Engine, Tdd};
    /// # use tididi::limits::LimitConfig;
    /// # use tididi::vtree::Vtree;
    /// # let vtree = Arc::new(Vtree::balanced(4));
    /// let engine = Engine::new();
    /// let f = Tdd::clause(&vtree, [1, -2]) & Tdd::clause(&vtree, [2, 3]);
    /// assert_eq!(engine.model_count(&f).unwrap(), f.model_count());
    ///
    /// // The pass polls on a stride, so the stop is observed once the walk has
    /// // covered enough levels to reach a poll point.
    /// let wide = Arc::new(Vtree::balanced(20_000));
    /// let g = Tdd::clause(&wide, [1, -2]);
    /// let _armed = engine.limits().scope(LimitConfig::none().with_deadline(Some(Instant::now())));
    /// match engine.model_count(&g) {
    ///     Ok(_) => unreachable!("the deadline has passed"),
    ///     Err(e) => assert_eq!(e, OperationError::Stopped),
    /// }
    /// ```
    pub fn model_count(&self, tdd: &crate::Tdd) -> Result<num_bigint::BigUint, crate::limits::OperationError> {
        crate::query::count::try_model_count(self, tdd)
    }
}
