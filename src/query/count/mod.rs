//! Model counting on compiled diagrams.
//!
//! Bottom-up hybrid u128/BigUint semiring evaluation: uses native u128
//! arithmetic for most nodes, falling back to `BigUint` only where overflow
//! occurs.

mod incremental;

use crate::Engine;
use crate::limits::OperationError;
pub use incremental::{ModelCounter, BoundModelCounter};
pub use crate::value::Retention;

use num_bigint::BigUint;

use crate::diagram::{LeafLabel, Tdd};
use crate::vtree::{VarId, VtreeNode};

/// Whether pins count as evidence or as substitution over the unchanged vtree.
///
/// Evidence counts assignments consistent with the pins. A cofactor counts
/// the conditioned function over all variables of the vtree, including the
/// substituted variables, which are now free.
///
/// ```
/// use std::sync::Arc;
/// use tididi::{literal, and};
/// use tididi::query::{PinSemantics, Retention};
/// use tididi::vtree::{VarId, Vtree};
///
/// let vtree = Arc::new(Vtree::balanced(2));
/// let f = and(literal(&vtree, 1)?, literal(&vtree, 2)?)?;
/// # tididi::test_helpers::assert_canonical(&f);
/// for (semantics, expected) in [(PinSemantics::Evidence, 1u32), (PinSemantics::Cofactor, 2)] {
///     let mut counter = f.counter_with(Retention::All, semantics)?;
///     counter.set_pin(VarId(1), Some(true))?;
///     assert_eq!(counter.model_count()?, expected.into());
/// }
/// let cofactor = f.condition_var(VarId(1), true)?;
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

/// Count a leaf label under an optional pin and its chosen semantics.
///
/// An unpinned free variable contributes two assignments; a literal contributes
/// one. A disagreeing pin contributes zero, and an agreeing pin contributes one
/// for evidence or two for cofactoring over the unchanged variable universe.
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

/// Count with exact integer arithmetic, releasing child columns after use.
///
/// Uses the shared counter fold without allocating pin storage. Allocation
/// refusals and cancellation propagate through the engine's limits.
pub(crate) fn model_count(eng: &Engine, tdd: &Tdd) -> Result<BigUint, OperationError> {
    let _op = eng.limits().begin_operation();
    if tdd.is_zero() {
        eng.limits().check_stop()?;
        return Ok(BigUint::ZERO);
    }
    ModelCounter::allocate(eng, tdd, 0, Retention::Frontier, PinSemantics::Cofactor)?.count_with(eng)
}

impl Tdd {
    /// Count distinct assignments to `vars` that have a satisfying extension.
    ///
    /// Each assignment is counted once, even when several assignments to the
    /// other variables satisfy the function. Selected variables that the function
    /// leaves free still contribute a factor of two. Order and duplicates do not
    /// matter; an empty selection counts one for a satisfiable function and zero
    /// for an unsatisfiable one. Every selected variable must belong to the vtree.
    ///
    /// Borrows a structural diagram, ignores attached weights, and returns an
    /// exact integer. The query quantifies unselected variables on a copy before
    /// counting, so it can require more work and memory than [`model_count`](Self::model_count).
    /// Selecting every vtree variable uses ordinary counting without copying.
    ///
    /// # Errors
    ///
    /// Returns [`OperationError::VariableNotInVtree`] for an absent variable,
    /// [`OperationError::MarginalLevel`] for discarded structure, or
    /// [`OperationError::OverBudget`] if an allocation is refused.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{Tdd, Vtree};
    /// use tididi::vtree::VarId;
    /// let vtree = Arc::new(Vtree::balanced(3));
    /// let f = Tdd::clause(&vtree, [1, 2])?;
    /// # tididi::test_helpers::assert_canonical(&f);
    /// assert_eq!(f.model_count()?, 6u32.into());
    /// // Either value of x1 can be extended to a satisfying assignment.
    /// assert_eq!(f.projected_model_count(&[VarId(1)])?, 2u32.into());
    /// assert_eq!(f.projected_model_count(&[VarId(1), VarId(2)])?, 3u32.into());
    /// # Ok::<(), tididi::OperationError>(())
    /// ```
    pub fn projected_model_count(&self, vars: &[VarId]) -> Result<BigUint, OperationError> {
        self.context().run(|eng| eng.projected_model_count(self, vars))
    }

    /// Return per-node counts, saturating values above `u128::MAX`.
    ///
    /// Indexed by vtree level and local node index. Zero and all values below
    /// `u128::MAX` are exact. A value of `u128::MAX` means the count is at least
    /// that large; it cannot distinguish an exact maximum from overflow.
    /// [`model_count`](Self::model_count) returns the exact total. Leaf columns contain three labels;
    /// count-marginal columns contain their stored values. Internal columns of
    /// a false diagram are empty. Uses the diagram's context and retains all
    /// columns during the fold.
    ///
    /// # Errors
    ///
    /// Returns [`OperationError::IncompatibleWeights`] for weighted marginal
    /// levels, [`OperationError::OverBudget`] if an allocation is refused,
    /// or [`OperationError::Stopped`] on cancellation.
    pub fn node_counts_u128(&self) -> Result<Vec<Vec<u128>>, OperationError> {
        self.context().run(|eng| eng.node_counts_u128(self))
    }
}

impl Engine {
    /// Run [`Tdd::projected_model_count`] under this batch's resource limits.
    ///
    /// Returns the query's errors, [`OperationError::Stopped`] on cancellation,
    /// or [`OperationError::OutputCap`] if quantification exceeds the node cap.
    /// Copying, quantification and counting share one operation scope; the input
    /// remains unchanged on success and error. Big-integer arithmetic allocations
    /// are outside the best-effort byte budget, as with [`Tdd::model_count`].
    pub fn projected_model_count(&self, tdd: &Tdd, vars: &[VarId]) -> Result<BigUint, OperationError> {
        let lim = self.limits();
        let _op = lim.begin_operation();
        lim.check_stop()?;
        let vtree = tdd.vtree();
        let mut gate = lim.gate();
        for &var in vars {
            gate.poll(1)?;
            vtree.leaf_of(var).ok_or(OperationError::VariableNotInVtree(var))?;
        }
        gate.flush()?;
        tdd.require_structure()?;
        let satisfiable = self.is_sat(tdd)?;
        if vars.is_empty() || !satisfiable {
            return Ok(u32::from(satisfiable).into());
        }

        let mut selected = Vec::new();
        lim.try_resize(&mut selected, vtree.num_nodes(), false)?;
        for &var in vars {
            gate.poll(1)?;
            selected[vtree.leaf_of(var).expect("validated variable").idx()] = true;
        }
        let mut eliminated = Vec::new();
        for level in vtree.bottomup() {
            gate.poll(1)?;
            if let VtreeNode::Leaf { var, .. } = *vtree.node(level) && !selected[level.idx()] {
                lim.try_push(&mut eliminated, var)?;
            }
        }
        gate.flush()?;
        if eliminated.is_empty() { return self.model_count(tdd); }
        let mut copy = tdd.try_clone_on(self)?;
        copy.weights = None;
        let projected = self.exists_vars(copy, &eliminated)?;
        Ok(self.model_count(&projected)? >> eliminated.len())
    }

    /// Run [`Tdd::node_counts_u128`] under this engine's allocation and stop limits.
    ///
    /// Returns the query's errors or [`OperationError::Stopped`] on cancellation.
    pub fn node_counts_u128(&self, tdd: &Tdd) -> Result<Vec<Vec<u128>>, OperationError> {
        let _op = self.limits().begin_operation();
        ModelCounter::allocate(self, tdd, 0, Retention::All, PinSemantics::Cofactor)?
            .into_fast_counts(self)
    }
}

/// The counting entry point on a caller's engine.
impl crate::Engine {
    /// Run [`Tdd::model_count`](crate::Tdd::model_count) using this batch's scratch and resource limits.
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
