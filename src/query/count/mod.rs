//! Model counting on compiled diagrams.
//!
//! Bottom-up hybrid u128/BigUint semiring evaluation: uses native u128
//! arithmetic for most nodes, falling back to `BigUint` only where overflow
//! occurs.

mod incremental;

use crate::engine::Engine;
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
/// let f = Tdd::clause(&tree, [1])? & Tdd::clause(&tree, [2])?;
/// # tididi::test_helpers::assert_canonical(&f);
/// for (semantics, expected) in [(PinSemantics::Evidence, 1u32), (PinSemantics::Cofactor, 2)] {
///     let mut counter = ModelCounter::<KeepAllColumns>::new(&f, semantics)?;
///     counter.set_pin(VarId(0), Some(true)).unwrap();
///     assert_eq!(counter.model_count()?, expected.into());
/// }
/// let cofactor = engine.condition_var(f, VarId(0), true).unwrap();
/// # tididi::test_helpers::assert_canonical(&cofactor);
/// assert_eq!(cofactor.model_count()?, 2u32.into());
/// # Ok::<(), Box<dyn std::error::Error>>(())
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
pub(crate) fn model_count(eng: &Engine, tdd: &Tdd) -> Result<BigUint, OperationError> {
    let _op = eng.limits().begin_operation();
    if tdd.is_zero() {
        if eng.limits().should_stop() { return Err(OperationError::Stopped); }
        return Ok(BigUint::ZERO);
    }
    ModelCounter::<KeepFrontier>::allocate(eng, tdd, 0, PinSemantics::Cofactor)?.model_count_on(eng)
}

impl Tdd {
    /// Return per-node counts, saturating values above `u128::MAX`.
    ///
    /// Indexed by vtree level and local node index. Zero and all values below
    /// the saturation sentinel are exact. Leaf columns contain three labels;
    /// count-marginal columns contain their stored values. Internal columns of
    /// a false diagram are empty. Uses the diagram's context and retains all
    /// columns during the fold.
    ///
    /// # Errors
    ///
    /// Returns [`OperationError::IncompatibleWeights`] for weighted marginal
    /// levels, or [`OperationError::OverBudget`] if an allocation is refused.
    pub fn node_counts_u128(&self) -> Result<Vec<Vec<u128>>, OperationError> {
        self.context().run(|eng| eng.node_counts_u128(self))
    }
}

impl Engine {
    /// Run [`Tdd::node_counts_u128`] under this engine's allocation and stop limits.
    ///
    /// Returns the query's errors or [`OperationError::Stopped`] on cancellation.
    pub fn node_counts_u128(&self, tdd: &Tdd) -> Result<Vec<Vec<u128>>, OperationError> {
        ModelCounter::<KeepAllColumns>::allocate(self, tdd, 0, PinSemantics::Cofactor)?
            .into_fast_counts(self)
    }
}

/// The counting entry point on a caller's engine.
impl crate::engine::Engine {
    /// Run [`Tdd::model_count`](crate::Tdd::model_count) using this batch's scratch and resource limits.
    ///
    /// Operand requirements, ownership and result semantics follow the diagram method.
    ///
    /// # Errors
    ///
    /// Returns the operation's errors or [`OperationError::Stopped`]
    /// on cancellation. Allocation refusals return
    /// [`OperationError::OverBudget`].
    ///
    /// Buffer growth is charged to the best-effort byte budget; allocations inside
    /// big-integer arithmetic are outside that budget. The input is unchanged.
    pub fn model_count(&self, tdd: &crate::Tdd) -> Result<num_bigint::BigUint, crate::limits::OperationError> {
        crate::query::count::model_count(self, tdd)
    }
}
