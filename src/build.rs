//! Constant, literal and cube construction over a shared vtree.

use std::sync::Arc;

use crate::vtree::{Vtree, VtreeIdx};
use crate::Engine;
use crate::limits::OperationError;

use crate::diagram::{self, *};

pub(crate) mod models;

/// The constant-false diagram over `vtree`, built outside the limits.
///
/// The output points at the `ZERO` sentinel, so no node is stored.
pub(crate) fn constant_zero(eng: &Engine, vtree: &Arc<Vtree>) -> Tdd {
    seat_canonical(eng, vtree, diagram::take_levels(eng, vtree.num_nodes()), TddNodeId { vtree: vtree.root(), local: ZERO })
}

/// The constant-true diagram over `vtree`, built outside the limits: the empty
/// cube, one node per internal level.
pub(crate) fn constant_one(eng: &Engine, vtree: &Arc<Vtree>) -> Tdd {
    let levels = cube_levels(eng, vtree, |_| ONE_LEAF_IDX, false).expect("an untracked build cannot be refused");
    seat_canonical(eng, vtree, levels, TddNodeId { vtree: vtree.root(), local: ONE_LEAF_IDX })
}

/// A constant over `vtree`, built under the engine's limits.
pub(crate) fn constant_on(eng: &Engine, vtree: &Arc<Vtree>, value: bool) -> Result<Tdd, OperationError> {
    eng.limits().check_stop()?;
    let (levels, local) = if value {
        (cube_levels(eng, vtree, |_| ONE_LEAF_IDX, true)?, ONE_LEAF_IDX)
    } else {
        (diagram::try_take_levels(eng, vtree.num_nodes())?, ZERO)
    };
    Ok(seat_canonical(eng, vtree, levels, TddNodeId { vtree: vtree.root(), local }))
}

/// A constant with the operand's vtree and weight configuration, without its
/// computed columns, built under the engine's limits.
pub(crate) fn constant_like(eng: &Engine, source: &Tdd, value: bool) -> Result<Tdd, OperationError> {
    let mut result = constant_on(eng, &source.vtree, value)?;
    result.weights = source.weights.as_ref().map(WeightStore::empty_like);
    Ok(result)
}

/// The levels of a cube: one node per internal vtree node, whose pair names
/// the leaf label the cube assigns to each side's subtree, bottom-up so the
/// pair a node writes names children that already exist. Every node lands at
/// the free label's index, so a side naming an internal child and a side
/// naming a free leaf are written the same way.
///
/// `charged` builds through the engine's limits, which may refuse; otherwise
/// the levels grow through `Vec`.
fn cube_levels(
    eng: &Engine,
    vtree: &Arc<Vtree>,
    label_at: impl Fn(VtreeIdx) -> NodeIdx,
    charged: bool,
) -> Result<Vec<TddLevel>, OperationError> {
    let lim = eng.limits();
    let mut levels = if charged {
        diagram::try_take_levels(eng, vtree.num_nodes())?
    } else {
        diagram::take_levels(eng, vtree.num_nodes())
    };
    let mut gate = lim.gate();
    for (emitted, (t, left, right)) in vtree.internal_bottomup().enumerate() {
        let pair = ChildPair::new(label_at(left), label_at(right));
        let index = if charged {
            gate.poll(1)?;
            let index = levels[t.idx()].push_node(eng.limits(), &[pair])?;
            lim.check_output_cap(emitted as u64 + 1)?;
            index
        } else {
            levels[t.idx()].push_internal_node(&[pair])
        };
        debug_assert_eq!(index, ONE_LEAF_IDX, "a cleared internal level receives its one node at the free label's index");
    }
    if charged { gate.flush()?; }
    Ok(levels)
}

/// Seat levels that are canonical as built: no level owes a contraction pass,
/// and the output is certified so the next reduction returns at once.
pub(crate) fn seat_canonical(eng: &Engine, vtree: &Arc<Vtree>, levels: Vec<TddLevel>, output: TddNodeId) -> Tdd {
    let mut tdd = Tdd::try_with_levels_dirty(eng, Arc::clone(vtree), levels, output, Dirty::default(), &[])
        .expect("seeding no worklist cannot be refused");
    tdd.levels.certify(output);
    tdd
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
        let lim = self.limits();
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
        gate.flush()?;
        let label_at = |t: VtreeIdx| {
            if label.is_empty() { ONE_LEAF_IDX } else { label[t.idx()] }
        };
        let levels = cube_levels(self, vtree, label_at, true)?;
        Ok(seat_canonical(self, vtree, levels, TddNodeId { vtree: vtree.root(), local: label_at(vtree.root()) }))
    }
}

#[cfg(test)]
#[path = "tests/build/mod.rs"]
mod tests;
