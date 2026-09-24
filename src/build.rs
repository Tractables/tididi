//! Constant, literal and cube construction over a shared vtree.

use std::sync::Arc;

use crate::vtree::Vtree;
use crate::Engine;
use crate::limits::{OperationError};

use crate::diagram::{self, *};

pub(crate) mod models;

/// Build a diagram computing the constant-false function (no assignment satisfies it).
/// Output points to the `ZERO` sentinel (`u32::MAX`) — no actual nodes are created.
pub(crate) fn constant_zero(eng: &Engine, vtree: &Arc<Vtree>) -> Tdd {
    let levels = diagram::take_levels(eng, vtree.num_nodes());
    Tdd::from_levels_unchecked(
        Arc::clone(vtree),
        levels,
        TddNodeId { vtree: vtree.root(), local: ZERO },
    )
}

/// Build a diagram computing the constant-true function (all assignments satisfy it).
/// Width 1 at every internal vtree level (only one node at index 0).
/// Leaf levels are implicit (no stored nodes); One is at index 0 (`ONE_LEAF_IDX`).
pub(crate) fn constant_one(eng: &Engine, vtree: &Arc<Vtree>) -> Tdd {
    let mut levels = diagram::take_levels(eng, vtree.num_nodes());

    // Internal levels: each has one node pairing the child's "true" node.
    // Leaf children reference One (`ONE_LEAF_IDX`); internal children reference
    // their single node at index 0.
    for (t, left, right) in vtree.internal_bottomup() {
        let left_child_idx = if vtree.node(left).is_leaf() {
            ONE_LEAF_IDX
        } else {
            NodeIdx(0)
        };
        let right_child_idx = if vtree.node(right).is_leaf() {
            ONE_LEAF_IDX
        } else {
            NodeIdx(0)
        };
        let pair = ChildPair::new(left_child_idx, right_child_idx);
        levels[t.idx()].push_internal_node(&[pair]);
    }

    // Output: One (for single-variable vtrees) or the sole internal node (index 0).
    let out_local = if vtree.node(vtree.root()).is_leaf() {
        ONE_LEAF_IDX
    } else {
        NodeIdx(0)
    };
    Tdd::from_levels_unchecked(
        Arc::clone(vtree),
        levels,
        TddNodeId { vtree: vtree.root(), local: out_local },
    )
}

/// Build a constant with the operand's vtree and weight configuration, without its computed columns.
pub(crate) fn constant_like(eng: &Engine, source: &Tdd, value: bool) -> Tdd {
    let mut result = if value { constant_one(eng, &source.vtree) } else { constant_zero(eng, &source.vtree) };
    result.weights = source.weights.as_ref().map(WeightStore::empty_like);
    result
}

/// Build the cube diagram: one width-1 node per internal vtree node, whose
/// pair names the leaf label the cube assigns to each side's subtree.
///
/// Bottom-up, so the pair a node writes names children that already exist.
fn cube_to_tdd(
    eng: &Engine,
    vtree: &Arc<Vtree>,
    literals: impl IntoIterator<Item = impl TryInto<Literal, Error: Into<OperationError>>>,
) -> Result<Tdd, OperationError> {
    let lim = eng.limits();
    let _op = lim.begin_operation();
    lim.check_stop()?;
    let mut gate = lim.gate();
    let mut label = Vec::new();
    for lit in literals {
        gate.poll(1)?;
        let lit: Literal = lit.try_into().map_err(Into::into)?;
        let leaf = vtree.leaf_of(lit.var).ok_or(OperationError::VariableNotInVtree(lit.var))?;
        if label.is_empty() { lim.try_resize(&mut label, vtree.num_nodes(), ONE_LEAF_IDX)?; }
        if label[leaf.idx()] != ONE_LEAF_IDX {
            return Err(OperationError::DuplicateVariable(lit.var));
        }
        label[leaf.idx()] = if lit.sign { POS_LEAF_IDX } else { NEG_LEAF_IDX };
    }
    let label_at = |t: crate::vtree::VtreeIdx| {
        if label.is_empty() { ONE_LEAF_IDX } else { label[t.idx()] }
    };
    let mut levels = diagram::try_take_levels(eng, vtree.num_nodes())?;
    for (emitted, (t, left, right)) in vtree.internal_bottomup().enumerate() {
        gate.poll(1)?;
        let index = levels[t.idx()].push_node(eng.limits(), &[ChildPair::new(label_at(left), label_at(right))])?;
        // A cleared internal level receives exactly one node, at the free label's index.
        debug_assert_eq!(index, ONE_LEAF_IDX);
        lim.check_output_cap(emitted as u64 + 1)?;
    }
    gate.flush()?;
    let root = vtree.root();
    Tdd::try_from_levels_on(eng, Arc::clone(vtree), levels, TddNodeId { vtree: root, local: label_at(root) })
}

/// Build a canonical diagram for one literal, leaving other variables free.
///
/// Integers are signed and one-based; typed [`Literal`] values also work.
/// Uses the execution context attached to the vtree.
/// Use [`Engine::literal`] inside a batch with resource limits.
///
/// # Errors
///
/// Returns [`OperationError::InvalidLiteral`] for integer zero,
/// [`OperationError::VariableNotInVtree`] for an absent variable, or
/// [`OperationError::OverBudget`] if an allocation is refused.
///
/// ```
/// use std::sync::Arc;
/// use tididi::{and, literal, Vtree};
/// let vtree = Arc::new(Vtree::balanced(3));
/// let f = and(literal(&vtree, 1)?, literal(&vtree, -2)?)?;
/// assert_eq!(u64::try_from(f.model_count()?)?, 2);
/// # tididi::test_helpers::assert_canonical(&f);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn literal(vtree: &Arc<Vtree>, literal: impl TryInto<Literal, Error: Into<OperationError>>) -> Result<Tdd, OperationError> {
    vtree.context().run(|eng| eng.literal(vtree, literal))
}

impl Tdd {
    /// Build a canonical conjunction of literals, leaving other variables free.
    ///
    /// Integers are signed and one-based; typed [`Literal`] values also work.
    /// Each variable must appear at most once, even with the same polarity.
    /// An empty cube is true. Uses the execution context attached to the vtree.
    ///
    /// # Errors
    ///
    /// Returns [`OperationError::InvalidLiteral`] for integer zero,
    /// [`OperationError::VariableNotInVtree`] for an absent variable,
    /// [`OperationError::DuplicateVariable`] for a repeated variable, or
    /// [`OperationError::OverBudget`] if an allocation is refused.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{Tdd, Vtree};
    /// let vtree = Arc::new(Vtree::balanced(3));
    /// let f = Tdd::cube(&vtree, [1, -2])?;
    /// assert_eq!(f.model_count()?, 2u32.into());
    /// # tididi::test_helpers::assert_canonical(&f);
    /// # Ok::<(), tididi::OperationError>(())
    /// ```
    pub fn cube(vtree: &Arc<Vtree>, literals: impl IntoIterator<Item = impl TryInto<Literal, Error: Into<OperationError>>>) -> Result<Tdd, OperationError> {
        vtree.context().run(|eng| eng.cube(vtree, literals))
    }

    /// The constant-true function over `vtree`: every assignment satisfies it.
    ///
    /// One node at every internal vtree level and no marginal level; the
    /// result is canonical. [`Engine::one`] is the same diagram built in a
    /// caller's engine.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{Tdd, Vtree};
    ///
    /// let vtree = Arc::new(Vtree::balanced(3));
    /// let all = Tdd::one(&vtree);
    /// let none = Tdd::zero(&vtree);
    /// assert_eq!(all.model_count()?, 8u32.into());
    /// assert!(none.is_zero());
    /// # tididi::test_helpers::assert_canonical(&all);
    /// # tididi::test_helpers::assert_canonical(&none);
    /// # Ok::<(), tididi::OperationError>(())
    /// ```
    pub fn one(vtree: &Arc<Vtree>) -> Tdd {
        vtree.context().run(|eng| eng.one(vtree))
    }

    /// The constant-false function over `vtree`: no assignment satisfies it.
    ///
    /// The output points at the `ZERO` sentinel, so no nodes are created and
    /// [`Tdd::is_zero`] is true. [`Engine::zero`] is the same diagram built in
    /// a caller's engine.
    pub fn zero(vtree: &Arc<Vtree>) -> Tdd {
        vtree.context().run(|eng| eng.zero(vtree))
    }
}

/// The construction entry points on a caller's engine.
impl crate::Engine {
    /// Run [`crate::literal`] using this batch's scratch and resource limits.
    ///
    /// # Errors
    ///
    /// Returns the operation's errors or [`OperationError::Stopped`]
    /// on cancellation. Allocation refusals return
    /// [`OperationError::OverBudget`]. An exceeded output-node cap returns
    /// [`OperationError::OutputCap`].
    pub fn literal(&self, vtree: &Arc<Vtree>, literal: impl TryInto<Literal, Error: Into<OperationError>>) -> Result<Tdd, OperationError> {
        self.cube(vtree, [literal])
    }

    /// The constant-true function over `vtree`, built in this engine's pools.
    ///
    /// Does not check resource limits; use an empty [`Engine::cube`] for checked construction.
    #[must_use]
    pub fn one(&self, vtree: &Arc<Vtree>) -> Tdd {
        crate::build::constant_one(self, vtree)
    }

    /// The constant-false function over `vtree`, built in this engine's pools.
    ///
    /// Does not check resource limits; use an empty [`Engine::clause`] for checked construction.
    #[must_use]
    pub fn zero(&self, vtree: &Arc<Vtree>) -> Tdd {
        crate::build::constant_zero(self, vtree)
    }

    /// Run [`Tdd::cube`](crate::Tdd::cube) using this batch's scratch and resource limits.
    ///
    /// # Errors
    ///
    /// Returns the operation's errors or [`OperationError::Stopped`]
    /// on cancellation. Allocation refusals return
    /// [`OperationError::OverBudget`]. An exceeded output-node cap returns
    /// [`OperationError::OutputCap`].
    pub fn cube(
        &self,
        vtree: &Arc<Vtree>,
        literals: impl IntoIterator<Item = impl TryInto<Literal, Error: Into<OperationError>>>,
    ) -> Result<Tdd, OperationError> {
        crate::build::cube_to_tdd(self, vtree, literals)
    }
}

#[cfg(test)]
#[path = "tests/build/mod.rs"]
mod tests;
