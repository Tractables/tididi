//! The hybrid u128/`BigUint` counting engine and the incremental pinned counter.

use crate::Engine;
use std::borrow::Borrow;
use std::marker::PhantomData;
use std::sync::Arc;
use crate::diagram::{ChildRef, EncodedChildRef, LeafLabel, NodeIdx, PairsIter, Tdd, ValueRef};
use num_bigint::BigUint;

use super::{leaf_seed, PinSemantics};
use super::super::fold::{LevelFold, Side};
use super::super::cache::{BoundState, CachedQuery, PinState, QueryCache};
use crate::limits::OperationError;
use crate::value::{Retention, Count, CountRead, IntFold};
use crate::vtree::{VarId, VtreeIdx};
use super::column::{CountColumn, QueryCounts};
use super::prepared::Prepared;

/// The most variables [`ModelCounter::count_table`] lists: its table holds
/// one count per assignment of them.
pub const MAX_COUNT_TABLE_VARS: usize = 30;

/// The u128-primary counting fold: native arithmetic for the vast majority of
/// nodes, spilling a node to the exact `BigUint` side table only where it
/// overflows.
pub(crate) struct OverflowingCounts<'a, C> {
    pins: &'a [PinState],
    prepared: &'a Prepared,
    convention: PinSemantics,
    column: PhantomData<C>,
}

impl<C: CountColumn> LevelFold for OverflowingCounts<'_, C> {
    const NODE_WORK: bool = true;
    type Value = Count;
    type Col = C;

    #[inline(always)]
    fn alloc(&self, eng: &Engine, width: usize) -> Result<C, OperationError> {
        C::try_with_width(eng, width)
    }

    fn release(&self, eng: &Engine, col: &mut Self::Col) {
        eng.limits().discard(std::mem::take(col));
    }

    fn set(&self, eng: &Engine, col: &mut C, i: usize, v: Count) -> Result<(), OperationError> {
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
        col: &mut C,
    ) -> Result<(), OperationError> {
        let level = &tdd.levels[t.idx()];
        let values = crate::diagram::MarginalValues::read(level, None, t.idx()).expect("marginal count column");
        for i in 0..values.len() {
            col.set(eng, i, values.count(i).to_count())?;
        }
        Ok(())
    }

    fn prepared_level(
        &self, eng: &Engine, tdd: &Tdd, cols: &mut [C],
        t: VtreeIdx, gate: &mut crate::limits::PollGate,
    ) -> Result<bool, OperationError> {
        if C::PREPARED_READS { self.prepared.fold_level(eng, tdd, cols, t, self.pins, gate) } else { Ok(false) }
    }

    /// The shared two-pass integer fold, with this query's child readers.
    ///
    /// Reading a child is the only thing that differs from any other integer
    /// fold: a pinned counter resolves through a [`ChildDecoder`], which knows
    /// whether the ref is a node index or a marginal-side value.
    fn fold_node(
        &self,
        pairs: PairsIter<'_>,
        left: Side<'_, C>,
        right: Side<'_, C>,
    ) -> Count {
        if !left.view.is_marginal() && !right.view.is_marginal()
            && let Some(stored) = pairs.as_slice()
            && let Some(total) = C::fold_structural(stored.iter().copied(), left.col, right.col)
        {
            return Count::from_u128(total);
        }
        IntFold::fold(pairs, |k| read_side(left, k), |k| read_side(right, k))
    }
}

/// Exported columns start in their destination format; retained counts begin narrow.
pub(super) struct CountQuery<C>(PinSemantics, PhantomData<C>, Prepared);

impl<C> CountQuery<C> {
    pub(super) fn new(convention: PinSemantics) -> Self { Self(convention, PhantomData, Prepared::default()) }
}

impl<C: CountColumn> CachedQuery for CountQuery<C> {
    type Col = C;
    type Output = BigUint;
    type Fold<'a> = OverflowingCounts<'a, C> where C: 'a;

    fn admit(tdd: &Tdd) -> Result<(), OperationError> {
        if tdd.levels.iter().any(|level| level.is_weight_marginal()) {
            return Err(OperationError::IncompatibleWeights);
        }
        Ok(())
    }

    fn fold<'a>(&'a self, pins: &'a [PinState]) -> OverflowingCounts<'a, C> {
        OverflowingCounts { pins, prepared: &self.2, convention: self.0, column: PhantomData }
    }

    fn false_value(&self) -> BigUint {
        BigUint::ZERO
    }

    fn output(&self, col: &C, i: usize) -> BigUint {
        let i = if C::PREPARED_READS { self.2.output.unwrap_or(i) } else { i };
        match col.get(i) {
            CountRead::Fast(value) => BigUint::from(value),
            CountRead::Big(value) => value.clone(),
        }
    }
}

/// Resolve one child ref of a pinned counter to a count read.
///
/// The sentinel ⟺ big-slot invariant, the exact-max promotion, and the
/// stale-overflow clear on recompute (a node may stop overflowing when pins
/// change) are all owned by [`CountColumn::set`] / [`Count::from_u128`].
#[inline]
fn read_side<'a, C: CountColumn>(side: Side<'a, C>, k: EncodedChildRef) -> CountRead<'a> {
    match side.view.child(k) {
        ChildRef::Value(ValueRef::Inline(c)) => CountRead::Fast(c as u128),
        ChildRef::Node(NodeIdx(i)) | ChildRef::Value(ValueRef::Slot(i)) => side.col.get(i as usize),
    }
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
/// [`ModelCounter`] borrows its diagram; [`OwnedModelCounter`] owns it. Neither
/// permits mutation while cached values are in use. A borrowed diagram cannot change while the
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
pub struct Counter<D: Borrow<Tdd>> {
    tdd: D,
    cache: QueryCache<CountQuery<QueryCounts>>,
}

/// A counter borrowing its circuit. Created by [`Tdd::counter`].
pub type ModelCounter<'a> = Counter<&'a Tdd>;
/// A counter owning its circuit. Created by [`Tdd::into_counter`].
pub type OwnedModelCounter = Counter<Tdd>;

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

    /// Move this circuit into a reusable counter without copying it.
    ///
    /// Observations and errors follow [`Self::counter`]. The returned counter
    /// has no lifetime tied to a separate circuit; [`Counter::into_inner`]
    /// returns the original circuit, discarding its query cache and observations.
    /// On construction failure, the consumed circuit is dropped.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{Tdd, Vtree};
    /// let vtree = Arc::new(Vtree::balanced(2));
    /// let mut counter = Tdd::clause(&vtree, [1, 2])?.into_counter()?;
    /// counter.observe([-1])?;
    /// assert_eq!(counter.model_count()?, 1u32.into());
    /// let circuit = counter.into_inner();
    /// assert_eq!(circuit.model_count()?, 3u32.into());
    /// # tididi::test_helpers::assert_canonical(&circuit);
    /// # Ok::<(), tididi::OperationError>(())
    /// ```
    pub fn into_counter(self) -> Result<OwnedModelCounter, OperationError> {
        self.into_counter_with(Retention::All, PinSemantics::Evidence)
    }

    /// [`Self::into_counter`] with the storage and pin semantics of [`Self::counter_with`].
    pub fn into_counter_with(self, retention: Retention, convention: PinSemantics) -> Result<OwnedModelCounter, OperationError> {
        let context = Arc::clone(self.context());
        context.run(|eng| Counter::new(eng, self, retention, convention))
    }

    /// Create an unpinned counter with the chosen retention policy and pin semantics.
    ///
    /// The counter borrows this diagram and uses its shared execution context.
    /// [`Retention::All`] retains counts for incremental updates;
    /// [`Retention::Frontier`] frees child columns after their parent is computed.
    /// [`PinSemantics`] controls how observed variables contribute to counts.
    /// Pin storage is proportional to the vtree's size, including for sparse
    /// variable IDs. Value columns are allocated on the first count and reused
    /// while retained. Refreshing counts can allocate when a column widens or
    /// a count overflows.
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
        self.context().run(|eng| ModelCounter::new(eng, self, retention, convention))
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
pub struct BoundCounter<'batch, D: Borrow<Tdd>> {
    counter: BoundState<'batch, Counter<D>>,
    engine: &'batch Engine,
}

/// A borrowed diagram counter bound to an engine.
pub type BoundModelCounter<'a, 'batch> = BoundCounter<'batch, &'a Tdd>;

impl<D: Borrow<Tdd>> std::fmt::Debug for BoundCounter<'_, D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BoundModelCounter").field("counter", self.counter.get()).finish_non_exhaustive()
    }
}

impl<D: Borrow<Tdd>> BoundCounter<'_, D> {
    /// Prepare compact reads with [`Counter::prepare`] semantics under this engine's limits.
    pub fn prepare(&mut self) -> Result<(), OperationError> {
        self.counter.get_mut().prepare_with(self.engine)
    }

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
        let counter = self.counter.get_mut();
        counter.cache.read(self.engine, counter.tdd.borrow())
    }

    /// Count every assignment of `vars` with [`ModelCounter::count_table`]
    /// semantics under the borrowed engine's limits.
    ///
    /// Each count is charged to the byte budget as a read of its own.
    ///
    /// # Errors
    ///
    /// As [`ModelCounter::count_table`].
    pub fn count_table(&mut self, vars: &[VarId]) -> Result<Vec<BigUint>, OperationError> {
        self.counter.get_mut().count_table_on(self.engine, vars)
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
        let counter = ModelCounter::new(self, tdd, retention, convention)?;
        Ok(BoundCounter { counter: BoundState::Owned(counter), engine: self })
    }
}

impl<D: Borrow<Tdd>> std::fmt::Debug for Counter<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModelCounter")
            .field("retention", &self.cache.observations.retention)
            .field("evaluated", &self.cache.observations.evaluated)
            .field("pins", &self.cache.observations.pins)
            .field("changed_since_pass", &self.cache.observations.changed.len())
            .finish()
    }
}

impl<D: Borrow<Tdd>> Counter<D> {
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
    pub fn bind<'batch>(&'batch mut self, engine: &'batch Engine) -> BoundCounter<'batch, D> {
        BoundCounter { counter: BoundState::Borrowed(self), engine }
    }

    /// Prepare compact read storage for repeated counts under changing observations.
    ///
    /// Preparation scans stored structural levels and retains an additional compact
    /// copy of sufficiently large levels. Its narrower references and cached
    /// implications can reduce subsequent refresh work. The original circuit stays
    /// available through [`Self::circuit`]; preparation therefore increases total
    /// retained memory and is intended for long-lived counters, not a single count.
    /// Small, implicit and marginal levels keep their ordinary readers.
    ///
    /// Pins are preserved and cached counts are invalidated on success. Repeating
    /// preparation is a no-op apart from checking the current limits. Dropping the
    /// counter or calling [`Self::into_inner`] releases the prepared storage.
    ///
    /// # Errors
    ///
    /// Returns [`OperationError::OverBudget`] for a refused buffer reservation or
    /// [`OperationError::Stopped`] for an armed stop. A refusal preserves the
    /// existing pins, cached values and preparation. Bind the counter to an engine
    /// with [`Self::bind`] to use that engine's limits for preparation.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{Tdd, Vtree};
    /// let vtree = Arc::new(Vtree::balanced(4));
    /// let f = Tdd::clause(&vtree, [1, -2])?;
    /// # tididi::test_helpers::assert_canonical(&f);
    /// let mut counter = f.counter()?;
    /// counter.prepare()?;
    /// counter.observe([-1])?;
    /// assert_eq!(counter.model_count()?, 4u32.into());
    /// # Ok::<(), tididi::OperationError>(())
    /// ```
    pub fn prepare(&mut self) -> Result<(), OperationError> {
        let context = Arc::clone(self.tdd.borrow().context());
        context.run(|eng| self.prepare_with(eng))
    }

    fn prepare_with(&mut self, eng: &Engine) -> Result<(), OperationError> {
        let _op = eng.limits().enter()?;
        eng.limits().check_stop()?;
        if self.cache.query().2.ready { return Ok(()); }
        let prepared = Prepared::new(eng, self.tdd.borrow())?;
        let convention = self.cache.query().0;
        self.cache.replace_query(CountQuery(convention, PhantomData, prepared));
        Ok(())
    }

    /// Reserve one pin slot per vtree leaf.
    fn new(eng: &Engine, tdd: D, retention: Retention, convention: PinSemantics) -> Result<Self, OperationError> {
        let slots = tdd.borrow().vtree.num_leaves() as usize;
        let cache = QueryCache::new(eng, tdd.borrow(), CountQuery::<QueryCounts>::new(convention), slots, retention)?;
        Ok(Self { tdd, cache })
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
        self.cache.observations.set_pin(self.tdd.borrow(), var, val)
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
        self.cache.observations.observe(self.tdd.borrow(), literals)
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
        self.cache.observations.set_pins(self.tdd.borrow(), pins)
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
        self.cache.observations.clear_pins()
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
        let tdd = self.tdd.borrow();
        tdd.context().run(|eng| self.cache.read(eng, tdd))
    }

    /// Count once for every assignment of `vars`, under the other pins.
    ///
    /// Entry `a` of the table is the count with each `vars[k]` pinned to bit
    /// `k` of `a`, so the table has `2^vars.len()` entries, and an empty list
    /// gives the one count [`model_count`](Self::model_count) would. Pins on
    /// other variables apply to every entry. Consecutive assignments differ
    /// in the bits a binary increment changes, and only those pins change
    /// between counts, so with [`Retention::All`] each count refreshes only
    /// their ancestors. Afterwards the listed variables have the pins they
    /// had before, also after an error, and the next read refreshes them.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use num_bigint::BigUint;
    /// use tididi::{Tdd, Vtree};
    /// use tididi::vtree::VarId;
    /// let vtree = Arc::new(Vtree::balanced(3));
    /// let f = Tdd::clause(&vtree, [1, -2, 3])?;
    /// # tididi::test_helpers::assert_canonical(&f);
    /// let mut counter = f.counter()?;
    /// // Entry 2 has variable 1 false and variable 2 true: only x3 is left.
    /// let table = counter.count_table(&[VarId(1), VarId(2)])?;
    /// assert_eq!(table, [2u32, 2, 1, 2].map(BigUint::from));
    /// counter.observe([3])?;
    /// assert_eq!(counter.count_table(&[VarId(1), VarId(2)])?, [1u32; 4].map(BigUint::from));
    /// assert_eq!(counter.model_count()?, 4u32.into());
    /// # Ok::<(), tididi::OperationError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// [`OperationError::TableTooWide`] for more than [`MAX_COUNT_TABLE_VARS`]
    /// variables, [`OperationError::DuplicateVariable`] for a variable listed
    /// twice, and the variable errors of [`Self::set_pin`]; these are checked
    /// before anything is counted. [`OperationError::OverBudget`] for a
    /// refused reservation, the table's included, or
    /// [`OperationError::Stopped`] for an armed stop.
    pub fn count_table(&mut self, vars: &[VarId]) -> Result<Vec<BigUint>, OperationError> {
        let context = Arc::clone(self.tdd.borrow().context());
        context.run(|eng| self.count_table_on(eng, vars))
    }

    /// [`Self::count_table`] under `eng`'s limits.
    ///
    /// The checks and the table's reservation are one operation, and each
    /// count is a read of its own, as [`Self::model_count`] would make it.
    fn count_table_on(&mut self, eng: &Engine, vars: &[VarId]) -> Result<Vec<BigUint>, OperationError> {
        let lim = eng.limits();
        let tdd = self.tdd.borrow();
        let mut before = [(VarId(0), None); MAX_COUNT_TABLE_VARS];
        let mut table = Vec::new();
        {
            let _op = lim.enter()?;
            if vars.len() > MAX_COUNT_TABLE_VARS {
                return Err(OperationError::TableTooWide { vars: vars.len() });
            }
            if let Some(k) = (1..vars.len()).find(|&k| vars[..k].contains(&vars[k])) {
                return Err(OperationError::DuplicateVariable(vars[k]));
            }
            let observations = &self.cache.observations;
            for (pin, &var) in before.iter_mut().zip(vars) {
                let leaf = observations.validate_pin(tdd, var)?;
                *pin = (var, observations.pins[leaf.idx()].value);
            }
            lim.try_resize(&mut table, 1usize << vars.len(), BigUint::ZERO)?;
        }
        let before = &before[..vars.len()];
        let mut pins = [(VarId(0), None); MAX_COUNT_TABLE_VARS];
        let pins = &mut pins[..vars.len()];
        pins.copy_from_slice(before);
        let counted = (|| {
            for (assignment, count) in table.iter_mut().enumerate() {
                // The first assignment sets every pin, and each later one the
                // bits its increment carries through.
                let changed = if assignment == 0 { vars.len() } else { assignment.trailing_zeros() as usize + 1 };
                for (bit, (_, value)) in pins[..changed].iter_mut().enumerate() {
                    *value = Some((assignment >> bit) & 1 == 1);
                }
                self.cache.observations.set_pins(tdd, &pins[..changed])?;
                *count = self.cache.read(eng, tdd)?;
            }
            Ok(())
        })();
        self.cache.observations.set_pins(tdd, before).expect("the listed variables were validated");
        counted.map(|()| table)
    }

    /// The unchanged circuit, available for read-only queries.
    pub fn circuit(&self) -> &Tdd { self.tdd.borrow() }

    /// Discard cached values and observations and recover the circuit owner or borrow.
    pub fn into_inner(self) -> D { self.tdd }
}

#[cfg(test)]
#[path = "tests/incremental.rs"]
mod tests;
