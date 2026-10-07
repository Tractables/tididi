//! Model counting on compiled diagrams.
//!
//! Bottom-up hybrid u128/BigUint semiring evaluation: uses native u128
//! arithmetic for most nodes, falling back to `BigUint` only where overflow
//! occurs.

mod incremental;
mod column;
mod prepared;

use crate::Engine;
use crate::limits::OperationError;
use super::cache::QueryCache;
use incremental::CountQuery;
use crate::value::CountVec;
pub use incremental::{Counter, ModelCounter, OwnedModelCounter, BoundCounter, BoundModelCounter, MAX_COUNT_TABLE_VARS};
pub use crate::value::Retention;

use num_bigint::BigUint;

use std::sync::Arc;

use crate::diagram::{LeafLabel, LevelCounts, Tdd};
use crate::value::CountRead;
use crate::vtree::{VarId, VtreeIdx, VtreeNode};

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
        let mut cache = QueryCache::new(self, tdd, CountQuery::<CountVec>::new(PinSemantics::Cofactor), 0, Retention::All)?;
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

    /// Count every internal level's nodes once and keep the counts with the
    /// diagram, so that later counts read them instead of folding the levels
    /// again.
    ///
    /// Level `t`'s counts are those [`Engine::model_count`] folds on its way
    /// to the output: node `i`'s model count over the variables under `t`.
    /// Once kept, [`Engine::model_count`] reads the output's count, and a
    /// conjunction carries the counts of every level it moves from this
    /// diagram untouched, a level under which the other operand is
    /// constant-true: [`Engine::and_model_count`] then folds only the levels
    /// it builds, and [`Engine::and`] keeps the moved levels' counts with its
    /// result. The counts describe the levels as they are, so any operation
    /// that changes the diagram's levels drops them;
    /// [`Tdd::has_level_counts`](crate::Tdd::has_level_counts) says whether
    /// they are kept.
    ///
    /// The fold is the model count's, keeping each level's column instead of
    /// releasing it: one `u128` per node of every internal structural level,
    /// with an exact side table for counts that do not fit. A marginal level
    /// stores its counts already and keeps no column.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{Engine, Vtree};
    /// use tididi::vtree::VarId;
    ///
    /// let engine = Engine::new();
    /// let vtree = Arc::new(Vtree::balanced(4));
    /// let mut f = engine.clause(&vtree, [1, 2])?;
    /// let g = engine.clause(&vtree, [3, 4])?;
    /// engine.attach_level_counts(&mut f)?;
    /// assert!(f.has_level_counts());
    /// assert_eq!(engine.model_count(&f)?, 12u32.into());
    /// assert_eq!(engine.and_model_count(f.clone(), g, &[])?, 9u32.into());
    /// // A change to the levels drops the counts.
    /// let f1 = f.condition_var(VarId(1), false)?;
    /// assert!(!f1.has_level_counts());
    /// assert_eq!(engine.model_count(&f1)?, 8u32.into());
    /// # Ok::<(), tididi::OperationError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`OperationError::IncompatibleWeights`] for weighted marginal
    /// levels, [`OperationError::OverBudget`] when a column's allocation is
    /// refused, and [`OperationError::Stopped`] on cancellation. On error the
    /// diagram keeps whatever counts it kept before.
    pub fn attach_level_counts(&self, tdd: &mut Tdd) -> Result<(), OperationError> {
        let lim = self.limits();
        let _op = lim.enter()?;
        let vtree = Arc::clone(&tdd.vtree);
        let mut counts = LevelCounts::none(self, vtree.num_nodes())?;
        if !tdd.is_zero() {
            // The model count's fold, keeping every column.
            let mut cache = QueryCache::new(self, tdd, CountQuery::<CountVec>::new(PinSemantics::Cofactor), 0, Retention::All)?;
            let mut gate = lim.gate();
            cache.refresh(self, tdd, &mut gate)?;
            gate.flush()?;
            for (i, column) in cache.into_columns().into_iter().enumerate() {
                let t = VtreeIdx(i as u32);
                if !vtree.node(t).is_leaf() && !tdd.levels[i].is_marginal() {
                    counts.set(t, Arc::new(column));
                }
            }
        }
        tdd.levels.keep_counts(counts);
        Ok(())
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
        // Counts kept with the diagram hold the output's already.
        if let Some(column) = tdd.levels.counts().and_then(|counts| counts.column(tdd.output.vtree)) {
            self.limits().check_stop()?;
            return Ok(match column.get(tdd.output.local.idx()) {
                CountRead::Fast(value) => BigUint::from(value),
                CountRead::Big(value) => value.clone(),
            });
        }
        // The shared counter fold with no pin storage, releasing each child
        // column once its parent has read it.
        QueryCache::new(self, tdd, CountQuery::<CountVec>::new(PinSemantics::Cofactor), 0, Retention::Frontier)?.read(self, tdd)
    }
}
