//! Model counting on compiled diagrams.
//!
//! Bottom-up hybrid u128/BigUint semiring evaluation: uses native u128
//! arithmetic for most nodes, falling back to `BigUint` only where overflow
//! occurs.

mod incremental;

use crate::Engine;
use crate::limits::OperationError;
use super::cache::QueryCache;
pub use incremental::{Counter, ModelCounter, OwnedModelCounter, BoundCounter, BoundModelCounter};
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
        let _op = lim.enter()?;
        tdd.require_structure()?;
        let vtree = tdd.vtree();
        let mut gate = lim.gate();
        for &var in vars {
            gate.poll(1)?;
            vtree.leaf_of(var).ok_or(OperationError::VariableNotInVtree(var))?;
        }
        gate.flush()?;
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
        let lim = self.limits();
        let _op = lim.enter()?;
        // The shared counter fold with no pin storage, keeping every column;
        // the fast half of each column holds the saturated counts.
        let mut cache = QueryCache::new(self, tdd, PinSemantics::Cofactor, 0, Retention::All)?;
        let mut gate = lim.gate();
        cache.refresh(self, tdd, &mut gate)?;
        let columns = cache.into_columns();
        let mut counts = Vec::new();
        lim.reserve_exact(&mut counts, columns.len())?;
        for column in columns {
            gate.poll(1)?;
            counts.push(column.into_parts().0);
        }
        gate.flush()?;
        Ok(counts)
    }

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
    pub fn model_count(&self, tdd: &Tdd) -> Result<BigUint, OperationError> {
        let _op = self.limits().enter()?;
        if tdd.is_zero() { return Ok(BigUint::ZERO); }
        // The shared counter fold with no pin storage, releasing each child
        // column once its parent has read it.
        QueryCache::new(self, tdd, PinSemantics::Cofactor, 0, Retention::Frontier)?.read(self, tdd)
    }
}
