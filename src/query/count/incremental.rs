//! The hybrid u128/`BigUint` counting engine and the incremental pinned counter.

use crate::Engine;
use crate::diagram::{ChildRef, EncodedChildRef, LeafLabel, NodeIdx, PairsIter, Tdd, ValueRef};
use num_bigint::BigUint;

use super::{leaf_seed, PinSemantics};
use super::super::fold::{LevelFold, Side};
use super::super::cache::{Observations, PinState, refresh_columns, BoundState};
use crate::limits::{OperationError, PollGate};
use crate::value::{Retention, Count, CountRead, CountVec, IntFold};
use crate::vtree::{VarId, VtreeIdx};

/// The u128-primary counting fold: native arithmetic for the vast majority of
/// nodes, spilling a node to the exact `BigUint` side table only where it
/// overflows.
pub(super) struct OverflowingCounts<'a> {
    pins: &'a [PinState],
    pub(super) convention: PinSemantics,
}

impl LevelFold for OverflowingCounts<'_> {
    const NODE_WORK: bool = true;
    type Value = Count;
    type Col = CountVec;

    fn alloc(&self, eng: &Engine, width: usize) -> Result<CountVec, OperationError> {
        CountVec::try_with_width(eng, width)
    }

    fn release(&self, eng: &Engine, col: &mut Self::Col) {
        eng.limits().discard(std::mem::take(col));
    }

    fn set(&self, eng: &Engine, col: &mut CountVec, i: usize, v: Count) -> Result<(), OperationError> {
        col.set(eng, i, v)
    }

    fn leaf(&self, leaf: VtreeIdx, _var: VarId, label: LeafLabel) -> Count {
        let pin = self.pins.get(leaf.idx()).and_then(|pin| pin.value);
        Count::from_u128(leaf_seed(label, pin, self.convention))
    }

    /// A marginal level's counts are pin-independent — summed out before any pin
    /// existed — so they are read across verbatim.
    fn marginal_column(
        &self,
        eng: &Engine,
        tdd: &Tdd,
        t: VtreeIdx,
        col: &mut CountVec,
    ) -> Result<(), OperationError> {
        let level = &tdd.levels[t.idx()];
        let values = crate::diagram::MarginalValues::read(level, None, t.idx()).expect("marginal count column");
        for i in 0..values.len() {
            col.set(eng, i, values.count(i).to_count())?;
        }
        Ok(())
    }

    /// The shared two-pass integer fold, with this query's child readers.
    ///
    /// Reading a child is the only thing that differs from any other integer
    /// fold: a pinned counter resolves through a [`ChildDecoder`], which knows
    /// whether the ref is a node index or a marginal-side value.
    fn fold_node(
        &self,
        pairs: PairsIter<'_>,
        left: Side<'_, CountVec>,
        right: Side<'_, CountVec>,
    ) -> Count {
        IntFold::fold(pairs, |k| read_side(left, k), |k| read_side(right, k))
    }
}

/// Resolve one child ref of a pinned counter to a count read.
///
/// The sentinel ⟺ big-slot invariant, the exact-max promotion, and the
/// stale-overflow clear on recompute (a node may stop overflowing when pins
/// change) are all owned by [`CountVec::set`] / [`Count::from_u128`].
#[inline]
fn read_side<'a>(side: Side<'a, CountVec>, k: EncodedChildRef) -> CountRead<'a> {
    let idx = match side.view.child(k) {
        ChildRef::Value(ValueRef::Inline(c)) => return CountRead::Fast(c as u128),
        ChildRef::Node(NodeIdx(idx)) | ChildRef::Value(ValueRef::Slot(idx)) => idx as usize,
    };
    side.col.get(idx)
}

/// Count repeatedly under changing observations without modifying the diagram.
///
/// Create one with [`Tdd::counter`], set observations with [`Self::observe`],
/// and read the count with [`Self::model_count`]. The default counter counts
/// assignments consistent with the observations and reuses cached counts;
/// changing an observation refreshes only the affected ancestor levels.
///
/// ```
/// use std::sync::Arc;
/// use tididi::{Tdd, Vtree};
/// let vtree = Arc::new(Vtree::balanced(4));
/// let f = Tdd::clause(&vtree, [1, -2])?;
/// let mut counter = f.counter()?;
/// assert_eq!(counter.model_count()?, 12u32.into());
/// counter.observe([1])?;
/// assert_eq!(counter.model_count()?, 8u32.into());
/// counter.observe([-1])?;
/// assert_eq!(counter.model_count()?, 4u32.into());
/// counter.clear_pins();
/// assert_eq!(counter.model_count()?, 12u32.into());
/// # tididi::test_helpers::assert_canonical(&f);
/// # Ok::<(), tididi::OperationError>(())
/// ```
///
/// A pin assigns a variable true or false; clearing it makes the variable
/// unobserved again. [`PinSemantics::Evidence`] counts original assignments
/// consistent with the pins. [`PinSemantics::Cofactor`] instead counts the
/// substituted function with the pinned variables free in the vtree.
///
/// [`Tdd::counter`] selects [`Retention::All`] and evidence semantics; use
/// [`Tdd::counter_with`] to choose [`Retention::Frontier`] or [`PinSemantics::Cofactor`].
/// Updates are deferred until [`model_count`](Self::model_count).
///
/// The diagram need not be minimized. Attached literal weights are ignored on
/// structural levels; weighted marginal levels are rejected. Count-marginal
/// levels retain their stored counts, and [`set_pin`](Self::set_pin) rejects
/// variables whose structure has already been summed out.
///
/// The counter borrows its diagram, so the diagram cannot change while the
/// counter remains in use:
///
/// ```compile_fail
/// use std::sync::Arc;
/// use tididi::Tdd;
/// use tididi::vtree::Vtree;
/// let vtree = Arc::new(Vtree::balanced(2));
/// let mut f = Tdd::clause(&vtree, [1]).unwrap();
/// let mut counter = f.counter().unwrap();
/// f.minimize().unwrap();
/// counter.model_count().unwrap();
/// ```
pub struct ModelCounter<'a> {
    tdd: &'a Tdd,
    cols: Vec<CountVec>,
    observations: Observations,
    convention: PinSemantics,
}

impl Tdd {
    /// Create an unpinned counter that retains columns for repeated counts under evidence.
    ///
    /// Pins restrict the assignments counted without changing this diagram.
    /// After a pin changes, the next count refreshes only its ancestor levels.
    /// The counter borrows this diagram and uses its shared execution context.
    /// See [`ModelCounter`] for pin semantics, storage and examples.
    /// Use [`Tdd::counter_with`] to choose another retention policy or pin convention.
    ///
    /// # Errors
    ///
    /// Returns [`OperationError::IncompatibleWeights`] for weighted marginal levels
    /// or [`OperationError::OverBudget`] if counter storage cannot be reserved.
    pub fn counter(&self) -> Result<ModelCounter<'_>, OperationError> {
        self.counter_with(Retention::All, PinSemantics::Evidence)
    }

    /// Create an unpinned counter with the chosen retention policy and pin semantics.
    ///
    /// The counter borrows this diagram and uses its shared execution context.
    /// [`Retention::All`] retains counts for incremental updates;
    /// [`Retention::Frontier`] frees child columns after their parent is computed.
    /// [`PinSemantics`] controls how observed variables contribute to counts.
    /// Pin storage is proportional to the vtree's size, including for sparse
    /// variable IDs. Value columns are allocated on the first count, and
    /// pin changes require no further allocation.
    ///
    /// # Errors
    ///
    /// Returns [`OperationError::IncompatibleWeights`] for weighted marginal
    /// levels, [`OperationError::OverBudget`] for a refused buffer reservation,
    /// or [`OperationError::Stopped`] for an armed stop.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{Tdd, Vtree};
    /// use tididi::query::{PinSemantics, Retention};
    /// use tididi::vtree::VarId;
    /// let vtree = Arc::new(Vtree::balanced(3));
    /// let f = Tdd::clause(&vtree, [1, 2])?;
    /// # tididi::test_helpers::assert_canonical(&f);
    /// let mut counter = f.counter_with(Retention::Frontier, PinSemantics::Cofactor)?;
    /// counter.set_pin(VarId(1), Some(false))?;
    /// assert_eq!(counter.model_count()?, 4u32.into());
    /// # Ok::<(), tididi::OperationError>(())
    /// ```
    pub fn counter_with(&self, retention: Retention, convention: PinSemantics) -> Result<ModelCounter<'_>, OperationError> {
        self.vtree().context().run(|eng| ModelCounter::allocate(eng, self, self.vtree.num_leaves() as usize, retention, convention))
    }
}

/// A counter whose reads use the borrowed engine's limits.
///
/// [`Engine::counter`] creates a counter for a batch; [`ModelCounter::bind`]
/// lends an existing counter to one. Both retain pins and cached columns after
/// a refused read, invalidating the cache so the next read recomputes it.
/// Dropping a borrowed binding leaves the original counter available with its
/// updated pins and columns. Dropping an owned counter releases that storage.
///
/// The engine must remain borrowed throughout the batch. A counter cannot
/// escape the engine checkout that created it:
///
/// ```compile_fail
/// use std::sync::Arc;
/// use tididi::{Tdd, Vtree};
/// let vtree = Arc::new(Vtree::balanced(2));
/// let f = Tdd::one(&vtree);
/// let mut counter = vtree.context().run(|engine| engine.counter(&f)).unwrap();
/// counter.model_count().unwrap();
/// ```
pub struct BoundModelCounter<'a, 'batch> {
    counter: BoundState<'batch, ModelCounter<'a>>,
    engine: &'batch Engine,
}

impl std::fmt::Debug for BoundModelCounter<'_, '_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let counter = match &self.counter {
            BoundState::Owned(counter) => counter,
            BoundState::Borrowed(counter) => counter,
        };
        f.debug_struct("BoundModelCounter").field("counter", counter).finish_non_exhaustive()
    }
}

impl BoundModelCounter<'_, '_> {
    /// Set or clear a pin with the validation and deferred refresh of [`ModelCounter::set_pin`].
    pub fn set_pin(&mut self, var: VarId, val: Option<bool>) -> Result<(), OperationError> {
        self.counter.get_mut().set_pin(var, val)
    }

    /// Apply a validated group of pin updates with [`ModelCounter::set_pins`] semantics.
    pub fn set_pins(&mut self, pins: &[(VarId, Option<bool>)]) -> Result<(), OperationError> {
        self.counter.get_mut().set_pins(pins)
    }

    /// Apply literal observations with [`ModelCounter::observe`] semantics.
    pub fn observe<L: crate::LiteralInput>(&mut self, literals: impl AsRef<[L]>) -> Result<(), OperationError> {
        self.counter.get_mut().observe(literals)
    }

    /// Clear all observations with [`ModelCounter::clear_pins`] semantics.
    pub fn clear_pins(&mut self) {
        self.counter.get_mut().clear_pins();
    }

    /// Count with [`ModelCounter::model_count`] semantics under the borrowed engine's limits.
    ///
    /// Every read checks the engine's current limits, including cached and
    /// constant answers. A refusal preserves pins and invalidates the cache;
    /// reading again after the limit is relaxed recomputes the result.
    ///
    /// # Errors
    ///
    /// Returns [`OperationError::Stopped`] for an armed stop or
    /// [`OperationError::OverBudget`] when a buffer reservation is refused.
    pub fn model_count(&mut self) -> Result<BigUint, OperationError> {
        self.counter.get_mut().count_with(self.engine)
    }
}

impl Engine {
    /// Create a counter with [`Tdd::counter`] semantics bound to this engine.
    ///
    /// Construction and every subsequent count use this engine's limits.
    /// The counter borrows the engine and diagram for its lifetime.
    /// Use [`Self::counter_with`] to choose a retention policy or pin convention.
    ///
    /// # Errors
    ///
    /// Returns the errors from [`Tdd::counter`], or [`OperationError::Stopped`]
    /// for an armed stop.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{Tdd, Vtree};
    /// use tididi::limits::LimitConfig;
    /// use tididi::vtree::VarId;
    /// let vtree = Arc::new(Vtree::balanced(3));
    /// let f = Tdd::clause(&vtree, [1, 2])?;
    /// # tididi::test_helpers::assert_canonical(&f);
    /// vtree.context().with_limits(
    ///     LimitConfig::none().with_memory_budget_bytes(Some(1_000_000)),
    ///     |engine| {
    ///         let mut counter = engine.counter(&f)?;
    ///         counter.set_pin(VarId(1), Some(false))?;
    ///         assert_eq!(counter.model_count()?, 2u32.into());
    ///         Ok::<(), tididi::OperationError>(())
    ///     },
    /// )?;
    /// # Ok::<(), tididi::OperationError>(())
    /// ```
    pub fn counter<'a, 'batch>(&'batch self, tdd: &'a Tdd) -> Result<BoundModelCounter<'a, 'batch>, OperationError> {
        self.counter_with(tdd, Retention::All, PinSemantics::Evidence)
    }

    /// Create a counter with [`Tdd::counter_with`] semantics bound to this engine.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`Self::counter`], using this engine for
    /// allocation and stop checks during construction and every count.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{Engine, Tdd, Vtree};
    /// use tididi::query::{PinSemantics, Retention};
    /// let engine = Engine::new();
    /// let f = Tdd::one(&Arc::new(Vtree::balanced(3)));
    /// # tididi::test_helpers::assert_canonical(&f);
    /// let mut counter = engine.counter_with(&f, Retention::Frontier, PinSemantics::Evidence)?;
    /// assert_eq!(counter.model_count()?, 8u32.into());
    /// # Ok::<(), tididi::OperationError>(())
    /// ```
    pub fn counter_with<'a, 'batch>(&'batch self, tdd: &'a Tdd, retention: Retention, convention: PinSemantics) -> Result<BoundModelCounter<'a, 'batch>, OperationError> {
        let counter = ModelCounter::allocate(self, tdd, tdd.vtree.num_leaves() as usize, retention, convention)?;
        Ok(BoundModelCounter { counter: BoundState::Owned(counter), engine: self })
    }
}

impl std::fmt::Debug for ModelCounter<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModelCounter")
            .field("retention", &self.observations.retention)
            .field("evaluated", &self.observations.evaluated)
            .field("pins", &self.observations.pins)
            .field("changed_since_pass", &self.observations.changed.len())
            .finish()
    }
}

impl<'a> ModelCounter<'a> {
    /// Borrow this counter for a batch whose reads use the supplied engine's limits.
    ///
    /// Binding does not allocate or evaluate. Pins and cached columns stay in
    /// this counter, including updates made through the binding. Once the
    /// binding ends, ordinary reads use the diagram's context again.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{Tdd, Vtree};
    /// use tididi::vtree::VarId;
    /// let vtree = Arc::new(Vtree::balanced(3));
    /// let f = Tdd::clause(&vtree, [1, 2])?;
    /// # tididi::test_helpers::assert_canonical(&f);
    /// let mut counter = f.counter()?;
    /// vtree.context().run(|engine| {
    ///     let mut batch = counter.bind(engine);
    ///     batch.set_pin(VarId(1), Some(false))?;
    ///     assert_eq!(batch.model_count()?, 2u32.into());
    ///     Ok::<(), tididi::OperationError>(())
    /// })?;
    /// assert_eq!(counter.model_count()?, 2u32.into());
    /// # Ok::<(), tididi::OperationError>(())
    /// ```
    pub fn bind<'batch>(&'batch mut self, engine: &'batch Engine) -> BoundModelCounter<'a, 'batch> {
        BoundModelCounter { counter: BoundState::Borrowed(self), engine }
    }

    /// Allocate leaf-indexed pin slots, or zero slots for an internal unpinned query.
    pub(super) fn allocate(eng: &Engine, tdd: &'a Tdd, pin_slots: usize, retention: Retention, convention: PinSemantics) -> Result<Self, OperationError> {
        let lim = eng.limits();
        let _op = lim.begin_operation();
        if tdd.levels.iter().any(|level| level.is_weight_marginal()) {
            return Err(OperationError::IncompatibleWeights);
        }
        lim.check_stop()?;
        let mut cols = Vec::new();
        lim.reserve_exact(&mut cols, tdd.vtree.num_nodes())?;
        cols.resize_with(tdd.vtree.num_nodes(), CountVec::default);
        let observations = Observations::new(eng, pin_slots, retention)?;
        lim.check_stop()?;
        Ok(Self { tdd, cols, observations, convention })
    }

    /// Set or clear a vtree variable's pin, deferring affected counts until the next read.
    ///
    /// `Some(value)` pins the variable and `None` removes its pin. A variable
    /// absent from the function's support can still be pinned if the vtree
    /// carries it. Repeating a pin leaves the cached counts valid.
    ///
    /// # Errors
    ///
    /// Returns [`OperationError::VariableNotInVtree`] for an absent variable,
    /// or [`OperationError::MarginalLevel`] for a variable already summed out.
    /// An error leaves pins and cached counts unchanged, including when clearing a pin.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{OperationError, Tdd, Vtree};
    /// use tididi::vtree::VarId;
    /// let vtree = Arc::new(Vtree::leaf(VarId(8)));
    /// let f = Tdd::one(&vtree);
    /// # tididi::test_helpers::assert_canonical(&f);
    /// let mut counter = f.counter()?;
    /// counter.set_pin(VarId(8), Some(true))?;
    /// assert_eq!(counter.model_count()?, 1u32.into());
    /// assert_eq!(counter.set_pin(VarId(1), Some(true)), Err(OperationError::VariableNotInVtree(VarId(1))));
    /// counter.set_pin(VarId(8), None)?;
    /// assert_eq!(counter.model_count()?, 2u32.into());
    /// # Ok::<(), OperationError>(())
    /// ```
    pub fn set_pin(&mut self, var: VarId, val: Option<bool>) -> Result<(), OperationError> {
        self.observations.set_pin(self.tdd, var, val)
    }

    /// Set observations using signed integers or named [`Literal`](crate::Literal) values.
    ///
    /// `2` observes the second variable as true; `-2` observes it as false.
    /// Only listed variables change, and the last occurrence of a variable wins.
    /// An empty input has no effect. Updates allocate no storage and defer
    /// counting until the next read. Use [`Self::clear_pins`] to remove all
    /// observations, or [`Self::set_pin`] to clear one.
    ///
    /// # Errors
    ///
    /// Returns [`OperationError::InvalidLiteral`] for zero, or the variable
    /// errors of [`Self::set_pin`]. Every input is validated before applying
    /// any update; an error preserves all pins, cached counts and pending changes.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{Literal, Tdd, Vtree};
    /// let vtree = Arc::new(Vtree::balanced(3));
    /// let f = Tdd::clause(&vtree, [1, 2])?;
    /// # tididi::test_helpers::assert_canonical(&f);
    /// let mut counter = f.counter()?;
    /// let enabled = Literal::try_from(1)?;
    /// let disabled = Literal::try_from(-3)?;
    /// counter.observe([enabled, disabled])?;
    /// assert_eq!(counter.model_count()?, 2u32.into());
    /// counter.observe([-1])?;
    /// assert_eq!(counter.model_count()?, 1u32.into());
    /// counter.clear_pins();
    /// assert_eq!(counter.model_count()?, 6u32.into());
    /// # Ok::<(), tididi::OperationError>(())
    /// ```
    pub fn observe<L: crate::LiteralInput>(&mut self, literals: impl AsRef<[L]>) -> Result<(), OperationError> {
        self.observations.observe(self.tdd, literals)
    }

    /// Apply a group of pin changes after validating every variable.
    ///
    /// Only listed variables change; `Some(value)` pins a variable and `None`
    /// clears it. If a variable appears more than once, its last value wins.
    /// An empty slice has no effect. Updates allocate no storage and defer
    /// affected counts until the next read, as in [`Self::set_pin`].
    ///
    /// # Errors
    ///
    /// Returns [`OperationError::VariableNotInVtree`] for an absent variable,
    /// or [`OperationError::MarginalLevel`] for a variable already summed out,
    /// including entries that clear pins. An error leaves all pins, cached
    /// counts and previously pending updates unchanged.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{Tdd, Vtree};
    /// use tididi::vtree::VarId;
    /// let vtree = Arc::new(Vtree::balanced(3));
    /// let f = Tdd::clause(&vtree, [1, 2])?;
    /// # tididi::test_helpers::assert_canonical(&f);
    /// let mut counter = f.counter()?;
    /// counter.set_pins(&[(VarId(1), Some(false)), (VarId(3), Some(true))])?;
    /// assert_eq!(counter.model_count()?, 1u32.into());
    /// counter.set_pins(&[(VarId(1), None)])?;
    /// assert_eq!(counter.model_count()?, 3u32.into());
    /// # Ok::<(), tididi::OperationError>(())
    /// ```
    pub fn set_pins(&mut self, pins: &[(VarId, Option<bool>)]) -> Result<(), OperationError> {
        self.observations.set_pins(self.tdd, pins)
    }

    /// Clear every pin without allocating, deferring affected counts until the next read.
    ///
    /// The next successful count returns the unobserved diagram's model count.
    /// Existing count storage is retained for reuse; an already unpinned
    /// counter is unchanged.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{Tdd, Vtree};
    /// use tididi::vtree::VarId;
    /// let vtree = Arc::new(Vtree::balanced(3));
    /// let f = Tdd::clause(&vtree, [1, 2])?;
    /// # tididi::test_helpers::assert_canonical(&f);
    /// let mut counter = f.counter()?;
    /// counter.set_pins(&[(VarId(1), Some(false)), (VarId(3), Some(true))])?;
    /// counter.clear_pins();
    /// assert_eq!(counter.model_count()?, 6u32.into());
    /// # Ok::<(), tididi::OperationError>(())
    /// ```
    pub fn clear_pins(&mut self) {
        self.observations.clear_pins()
    }

    /// Refresh the current pins and count under the diagram context's allocation and stop rules.
    ///
    /// Stops are checked even for a cached or constant answer, during a refresh,
    /// and before return. A refusal invalidates the cached result; a later call
    /// recomputes before reading it. Pins are retained.
    /// The best-effort byte budget covers buffer growth during this call, not
    /// retained columns or allocations inside big-integer arithmetic.
    ///
    /// # Errors
    ///
    /// [`OperationError::OverBudget`] for a refused scratch or overflow-table
    /// reservation, or [`OperationError::Stopped`] for an armed stop.
    pub fn model_count(&mut self) -> Result<BigUint, OperationError> {
        let tdd = self.tdd;
        tdd.vtree().context().run(|eng| self.count_with(eng))
    }

    /// Refresh pins and count under the supplied engine's limits for every entry point.
    pub(super) fn count_with(&mut self, eng: &Engine) -> Result<BigUint, OperationError> {
        let lim = eng.limits();
        let _op = lim.begin_operation();
        let result = (|| {
            lim.check_stop()?;
            let mut gate = lim.gate();
            let count = if self.tdd.is_zero() {
                BigUint::ZERO
            } else {
                self.refresh(eng, &mut gate)?;
                let out = self.tdd.output;
                match self.cols[out.vtree.idx()].get(out.local.idx()) {
                    CountRead::Fast(value) => BigUint::from(value),
                    CountRead::Big(value) => value.clone(),
                }
            };
            gate.poll(1)?;
            gate.flush()?;
            Ok(count)
        })();
        if result.is_err() {
            self.observations.invalidate();
        }
        result
    }

    fn refresh(&mut self, eng: &Engine, gate: &mut PollGate) -> Result<(), OperationError> {
        let (tdd, cols, convention) = (self.tdd, &mut self.cols, self.convention);
        self.observations.refresh(eng, tdd, gate, |pins, changed, gate| {
            let fold = OverflowingCounts { pins, convention };
            refresh_columns(&fold, eng, tdd, cols, changed, gate, |fold, col, width| {
                if col.len() != width { *col = fold.alloc(eng, width)?; }
                Ok(())
            })
        })
    }

}

impl ModelCounter<'_> {
    /// Compute and return every fast count slot, preserving overflow sentinels.
    ///
    /// Only a [`Retention::All`] counter holds every slot after the pass.
    pub(crate) fn into_fast_counts(mut self, eng: &Engine) -> Result<Vec<Vec<u128>>, OperationError> {
        debug_assert_eq!(self.observations.retention, Retention::All, "a frontier counter frees the columns this reads");
        let lim = eng.limits();
        let mut gate = lim.gate();
        self.refresh(eng, &mut gate)?;
        let mut counts = Vec::new();
        lim.reserve_exact(&mut counts, self.cols.len())?;
        for column in self.cols {
            gate.poll(1)?;
            counts.push(column.into_parts().0);
        }
        gate.flush()?;
        Ok(counts)
    }
}

#[cfg(test)]
#[path = "tests/incremental.rs"]
mod tests;
