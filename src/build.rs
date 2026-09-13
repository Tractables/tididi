//! Constants and cubes as diagrams.
//!
//! These are the leaves of every compilation: everything else is built by
//! combining them with [`crate::apply`] and reducing with [`crate::reduce`].
//!
//! Entry points: [`Tdd::one`] and [`Tdd::zero`] are the two constants;
//! [`Engine::cube`] builds a conjunction of literals. A single clause is
//! [`Tdd::clause`], the clause conjoined into ⊤ by
//! [`crate::apply::apply_and_clause`].

use std::sync::Arc;

use crate::vtree::Vtree;
use crate::engine::Engine;
use crate::limits::{OperationError, PollGate};

use crate::diagram::{self, *};

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
    literals: impl IntoIterator<Item = impl Into<Literal>>,
) -> Result<Tdd, OperationError> {
    let lim = eng.limits();
    let _op = lim.begin_operation();
    if lim.should_stop() { return Err(OperationError::Stopped); }
    let mut gate = PollGate::new(lim.reduce_poll_stride());
    let mut label = Vec::new();
    lim.try_resize(&mut label, vtree.num_nodes(), ONE_LEAF_IDX)?;
    for lit in literals {
        lim.poll(&mut gate, 1)?;
        let lit: Literal = lit.into();
        let leaf = vtree.leaf_of(lit.var).ok_or(OperationError::VariableNotInVtree(lit.var))?;
        if label[leaf.idx()] != ONE_LEAF_IDX {
            return Err(OperationError::DuplicateVariable(lit.var));
        }
        label[leaf.idx()] = if lit.positive { POS_LEAF_IDX } else { NEG_LEAF_IDX };
    }
    let mut levels = diagram::try_take_levels(eng, vtree.num_nodes())?;
    for (emitted, (t, left, right)) in vtree.internal_bottomup().enumerate() {
        lim.poll(&mut gate, 1)?;
        label[t.idx()] = levels[t.idx()].push_node_on(eng, &[ChildPair::new(label[left.idx()], label[right.idx()])])?;
        lim.level_done(emitted as u64 + 1)?;
    }
    lim.flush_poll(&mut gate)?;
    let root = vtree.root();
    Tdd::try_from_levels_on(eng, Arc::clone(vtree), levels, TddNodeId { vtree: root, local: label[root.idx()] })
}

impl Tdd {
    /// A diagram for one literal, with every other vtree variable free.
    ///
    /// Uses a temporary engine; see [`Engine::literal`] for a checked constructor.
    ///
    /// # Panics
    ///
    /// Panics on an absent variable, allocation failure or a conversion to
    /// [`Literal`] that panics (including integer zero).
    pub fn literal(vtree: &Arc<Vtree>, literal: impl Into<Literal>) -> Tdd {
        Engine::new().literal(vtree, literal).expect("literal: use Engine::literal to handle errors")
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
    /// let tree = Arc::new(Vtree::balanced(3));
    /// let all = Tdd::one(&tree);
    /// let none = Tdd::zero(&tree);
    /// assert_eq!(all.model_count(), 8u32.into());
    /// assert!(none.is_zero());
    /// # tididi::test_helpers::assert_canonical(&all);
    /// # tididi::test_helpers::assert_canonical(&none);
    /// ```
    pub fn one(vtree: &Arc<Vtree>) -> Tdd {
        Engine::new().one(vtree)
    }

    /// The constant-false function over `vtree`: no assignment satisfies it.
    ///
    /// The output points at the `ZERO` sentinel, so no nodes are created and
    /// [`Tdd::is_zero`] is true. [`Engine::zero`] is the same diagram built in
    /// a caller's engine.
    pub fn zero(vtree: &Arc<Vtree>) -> Tdd {
        Engine::new().zero(vtree)
    }
}

/// The construction entry points on a caller's engine.
impl crate::engine::Engine {
    /// A minimized diagram for one literal, with other vtree variables free.
    ///
    /// Integer literals use the signed, 1-based DIMACS convention; [`Literal`]
    /// also accepts a typed, 0-based variable identifier. Delegates to [`Engine::cube`].
    ///
    /// # Errors
    ///
    /// An absent variable or the allocation, cancellation and output-cap errors
    /// of [`Engine::cube`].
    ///
    /// # Panics
    ///
    /// If conversion to [`Literal`] panics, including integer zero.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{Engine, Vtree};
    /// let engine = Engine::new();
    /// let tree = Arc::new(Vtree::balanced(3));
    /// let x = engine.literal(&tree, 1)?;
    /// let not_y = engine.literal(&tree, -2)?;
    /// let f = engine.and(x, not_y)?;
    /// assert_eq!(f.model_count(), 2u32.into()); // x AND NOT y, with z free
    /// # Ok::<(), tididi::OperationError>(())
    /// ```
    pub fn literal(&self, vtree: &Arc<Vtree>, literal: impl Into<Literal>) -> Result<Tdd, OperationError> {
        self.cube(vtree, [literal.into()])
    }

    /// The constant-true function over `vtree`, built in this engine's pools.
    #[must_use]
    pub fn one(&self, vtree: &Arc<Vtree>) -> Tdd {
        crate::build::constant_one(self, vtree)
    }

    /// The constant-false function over `vtree`, built in this engine's pools.
    #[must_use]
    pub fn zero(&self, vtree: &Arc<Vtree>) -> Tdd {
        crate::build::constant_zero(self, vtree)
    }

    /// The conjunction of `literals` over `vtree`: one width-1 node per
    /// internal vtree node, so the whole diagram is one path.
    ///
    /// A variable no literal mentions is free — the cube says nothing about
    /// it, so both of its values satisfy the result. Each item is converted
    /// with [`Into<Literal>`], so plain integers use the 1-based DIMACS sign
    /// convention. The result is canonical. Allocation, cancellation, and the
    /// output-node cap are checked during construction.
    ///
    /// # Errors
    ///
    /// Returns [`OperationError::VariableNotInVtree`] for an absent variable,
    /// [`OperationError::DuplicateVariable`] for a repeated variable, or the
    /// resource error that stopped construction.
    ///
    /// # Panics
    ///
    /// If an item's conversion to [`Literal`] panics, including a zero integer.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::Engine;
    /// use tididi::vtree::Vtree;
    ///
    /// let eng = Engine::new();
    /// let vtree = Arc::new(Vtree::balanced(3));
    /// let f = eng.cube(&vtree, [1, -2]).unwrap(); // x1 ∧ ¬x2, with x3 free
    /// assert_eq!(f.model_count(), 2u32.into());
    /// ```
    ///
    /// Each cube variable must appear once, even when the repeated polarity agrees:
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{Engine, OperationError, Vtree};
    /// use tididi::vtree::VarId;
    ///
    /// let tree = Arc::new(Vtree::balanced(2));
    /// let result = Engine::new().cube(&tree, [1, 1]);
    /// assert_eq!(result.err(), Some(OperationError::DuplicateVariable(VarId(0))));
    /// ```
    pub fn cube(
        &self,
        vtree: &Arc<Vtree>,
        literals: impl IntoIterator<Item = impl Into<Literal>>,
    ) -> Result<Tdd, OperationError> {
        crate::build::cube_to_tdd(self, vtree, literals)
    }
}

#[cfg(test)]
mod tests;
