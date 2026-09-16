//! The hybrid u128/`BigUint` counting engine and the incremental pinned counter.

use crate::Engine;
use crate::diagram::{EncodedChildRef, ChildRef, ValueRef, NodeIdx};
use num_bigint::BigUint;

use super::{leaf_seed, PinSemantics};
use super::super::fold::{fold_bottom_up, fold_level, LevelFold, Side};
use crate::limits::PollGate;
use crate::limits::OperationError;
use crate::diagram::PairsIter;
use crate::value::{ColumnRetention, Count, CountRead, CountVec, IntFold};
use crate::diagram::*;
use crate::vtree::{VarId, VtreeIdx};
use std::marker::PhantomData;

/// A leaf observation and its membership in the pending ancestor traversal.
#[derive(Clone, Copy, Debug, Default)]
struct PinState {
    value: Option<bool>,
    dirty: bool,
}

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
        let bytes = col.buffer_bytes();
        *col = CountVec::default();
        eng.limits().release_bytes(bytes);
    }

    #[inline(always)]
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
        let counts = level.marginal_counts().expect("a marginal level carries counts");
        let big = level.marginal_counts_big();
        for i in 0..counts.len() {
            col.set(eng, i, CountRead::from_slot(counts, big, i).to_count())?;
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

/// The column retention used by a counter: [`KeepAllColumns`] or [`KeepFrontier`].
pub trait Retention: sealed::Sealed {}

/// Keep every column so reads after pin changes need only their ancestor cone.
#[derive(Debug, Clone, Copy)]
pub struct KeepAllColumns;

/// Keep only columns still needed by the bottom-up walk, freeing each child
/// column after its parent is computed. A changed query repeats the full fold.
#[derive(Debug, Clone, Copy)]
pub struct KeepFrontier;

impl Retention for KeepAllColumns {}
impl Retention for KeepFrontier {}

mod sealed {
    pub trait Sealed {
        /// Whether counts survive a completed pass for incremental updates.
        const KEEP_ALL_COLUMNS: bool;
    }
    impl Sealed for super::KeepAllColumns {
        const KEEP_ALL_COLUMNS: bool = true;
    }
    impl Sealed for super::KeepFrontier {
        const KEEP_ALL_COLUMNS: bool = false;
    }
}

/// Count repeatedly under changing observations without modifying the diagram.
///
/// A pin assigns a variable true or false; clearing it makes the variable
/// unobserved again. [`PinSemantics::Evidence`] counts original assignments
/// consistent with the pins. [`PinSemantics::Cofactor`] instead counts the
/// substituted function with the pinned variables free in the vtree.
///
/// Use [`KeepAllColumns`] to retain per-node counts between queries: after a
/// pin changes, only levels on its path to the root need recomputation.
/// [`KeepFrontier`] uses less retained storage by freeing child counts after
/// their parent is computed, and performs a fresh fold for each changed query.
/// Updates are deferred until [`model_count`](Self::model_count).
/// [`Tdd::counter`] selects retained columns and evidence semantics; use
/// [`Tdd::counter_with`] to choose another policy or convention.
///
/// ```
/// use std::sync::Arc;
/// use tididi::Tdd;
/// use tididi::vtree::{VarId, Vtree};
/// let vtree = Arc::new(Vtree::balanced(4));
/// let f = Tdd::clause(&vtree, [1, -2])?;
/// let mut counter = f.counter()?;
/// assert_eq!(counter.model_count()?, 12u32.into());
/// counter.set_pin(VarId(0), Some(true))?;
/// assert_eq!(counter.model_count()?, 8u32.into());
/// counter.set_pin(VarId(0), Some(false))?;
/// assert_eq!(counter.model_count()?, 4u32.into());
/// counter.set_pin(VarId(0), None)?;
/// assert_eq!(counter.model_count()?, 12u32.into());
/// # tididi::test_helpers::assert_canonical(&f);
/// # Ok::<(), tididi::OperationError>(())
/// ```
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
pub struct ModelCounter<'a, R: Retention = KeepAllColumns> {
    tdd: &'a Tdd,
    cols: Vec<CountVec>,
    pins: Vec<PinState>,
    changed: Vec<VtreeIdx>,
    convention: PinSemantics,
    evaluated: bool,
    _marker: PhantomData<R>,
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
        self.counter_with(PinSemantics::Evidence)
    }

    /// Create an unpinned counter with the chosen retention policy and pin semantics.
    ///
    /// The counter borrows this diagram and uses its shared execution context.
    /// [`KeepAllColumns`] retains counts for incremental updates;
    /// [`KeepFrontier`] frees child columns after their parent is computed.
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
    /// use tididi::query::{KeepFrontier, PinSemantics};
    /// use tididi::vtree::VarId;
    /// let vtree = Arc::new(Vtree::balanced(3));
    /// let f = Tdd::clause(&vtree, [1, 2])?;
    /// # tididi::test_helpers::assert_canonical(&f);
    /// let mut counter = f.counter_with::<KeepFrontier>(PinSemantics::Cofactor)?;
    /// counter.set_pin(VarId(0), Some(false))?;
    /// assert_eq!(counter.model_count()?, 4u32.into());
    /// # Ok::<(), tididi::OperationError>(())
    /// ```
    pub fn counter_with<R: Retention>(&self, convention: PinSemantics) -> Result<ModelCounter<'_, R>, OperationError> {
        self.vtree().context().run(|eng| ModelCounter::allocate(eng, self, self.vtree.num_leaves() as usize, convention))
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
pub struct BoundModelCounter<'a, 'batch, R: Retention = KeepAllColumns> {
    counter: CounterStorage<'a, 'batch, R>,
    engine: &'batch Engine,
}

/// Store a batch's counter or borrow the state of a persistent counter.
enum CounterStorage<'a, 'batch, R: Retention> {
    Owned(ModelCounter<'a, R>),
    Borrowed(&'batch mut ModelCounter<'a, R>),
}

impl<'a, R: Retention> CounterStorage<'a, '_, R> {
    /// Borrow the counter state used by either kind of batch binding.
    fn get_mut(&mut self) -> &mut ModelCounter<'a, R> {
        match self {
            Self::Owned(counter) => counter,
            Self::Borrowed(counter) => counter,
        }
    }
}

impl<R: Retention> std::fmt::Debug for BoundModelCounter<'_, '_, R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let counter = match &self.counter {
            CounterStorage::Owned(counter) => counter,
            CounterStorage::Borrowed(counter) => counter,
        };
        f.debug_struct("BoundModelCounter").field("counter", counter).finish_non_exhaustive()
    }
}

impl<R: Retention> BoundModelCounter<'_, '_, R> {
    /// Set or clear a pin with the validation and deferred refresh of [`ModelCounter::set_pin`].
    pub fn set_pin(&mut self, var: VarId, val: Option<bool>) -> Result<(), OperationError> {
        self.counter.get_mut().set_pin(var, val)
    }

    /// Apply a validated group of pin updates with [`ModelCounter::set_pins`] semantics.
    pub fn set_pins(&mut self, pins: &[(VarId, Option<bool>)]) -> Result<(), OperationError> {
        self.counter.get_mut().set_pins(pins)
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
    ///         counter.set_pin(VarId(0), Some(false))?;
    ///         assert_eq!(counter.model_count()?, 2u32.into());
    ///         Ok::<(), tididi::OperationError>(())
    ///     },
    /// )?;
    /// # Ok::<(), tididi::OperationError>(())
    /// ```
    pub fn counter<'a, 'batch>(&'batch self, tdd: &'a Tdd) -> Result<BoundModelCounter<'a, 'batch>, OperationError> {
        self.counter_with(tdd, PinSemantics::Evidence)
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
    /// use tididi::query::{KeepFrontier, PinSemantics};
    /// let engine = Engine::new();
    /// let f = Tdd::one(&Arc::new(Vtree::balanced(3)));
    /// # tididi::test_helpers::assert_canonical(&f);
    /// let mut counter = engine.counter_with::<KeepFrontier>(&f, PinSemantics::Evidence)?;
    /// assert_eq!(counter.model_count()?, 8u32.into());
    /// # Ok::<(), tididi::OperationError>(())
    /// ```
    pub fn counter_with<'a, 'batch, R: Retention>(&'batch self, tdd: &'a Tdd, convention: PinSemantics) -> Result<BoundModelCounter<'a, 'batch, R>, OperationError> {
        let counter = ModelCounter::allocate(self, tdd, tdd.vtree.num_leaves() as usize, convention)?;
        Ok(BoundModelCounter { counter: CounterStorage::Owned(counter), engine: self })
    }
}

impl<R: Retention> std::fmt::Debug for ModelCounter<'_, R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let retention = if R::KEEP_ALL_COLUMNS { ColumnRetention::All } else { ColumnRetention::Frontier };
        f.debug_struct("ModelCounter")
            .field("retention", &retention)
            .field("evaluated", &self.evaluated)
            .field("pins", &self.pins)
            .field("changed_since_pass", &self.changed.len())
            .finish()
    }
}

impl<'a, R: Retention> ModelCounter<'a, R> {
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
    ///     batch.set_pin(VarId(0), Some(false))?;
    ///     assert_eq!(batch.model_count()?, 2u32.into());
    ///     Ok::<(), tididi::OperationError>(())
    /// })?;
    /// assert_eq!(counter.model_count()?, 2u32.into());
    /// # Ok::<(), tididi::OperationError>(())
    /// ```
    pub fn bind<'batch>(&'batch mut self, engine: &'batch Engine) -> BoundModelCounter<'a, 'batch, R> {
        BoundModelCounter { counter: CounterStorage::Borrowed(self), engine }
    }

    /// Allocate leaf-indexed pin slots, or zero slots for an internal unpinned query.
    pub(super) fn allocate(eng: &Engine, tdd: &'a Tdd, pin_slots: usize, convention: PinSemantics) -> Result<Self, OperationError> {
        let lim = eng.limits();
        let _op = lim.begin_operation();
        if tdd.levels.iter().any(|level| level.is_weight_marginal()) {
            return Err(OperationError::IncompatibleWeights);
        }
        if lim.should_stop() { return Err(OperationError::Stopped); }
        let mut cols = Vec::new();
        lim.reserve_exact(&mut cols, tdd.vtree.num_nodes())?;
        cols.resize_with(tdd.vtree.num_nodes(), CountVec::default);
        let mut pins = Vec::new();
        lim.try_resize(&mut pins, pin_slots, PinState::default())?;
        let mut changed = Vec::new();
        if pin_slots != 0 && R::KEEP_ALL_COLUMNS {
            lim.reserve_exact(&mut changed, pin_slots)?;
        }
        if lim.should_stop() { return Err(OperationError::Stopped); }
        Ok(Self { tdd, cols, pins, changed, convention, evaluated: false, _marker: PhantomData })
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
    /// let vtree = Arc::new(Vtree::leaf(VarId(7)));
    /// let f = Tdd::one(&vtree);
    /// # tididi::test_helpers::assert_canonical(&f);
    /// let mut counter = f.counter()?;
    /// counter.set_pin(VarId(7), Some(true))?;
    /// assert_eq!(counter.model_count()?, 1u32.into());
    /// assert_eq!(counter.set_pin(VarId(0), Some(true)), Err(OperationError::VariableNotInVtree(VarId(0))));
    /// counter.set_pin(VarId(7), None)?;
    /// assert_eq!(counter.model_count()?, 2u32.into());
    /// # Ok::<(), OperationError>(())
    /// ```
    pub fn set_pin(&mut self, var: VarId, val: Option<bool>) -> Result<(), OperationError> {
        let leaf = self.validate_pin(var)?;
        self.set_leaf_pin(leaf, val);
        Ok(())
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
    /// counter.set_pins(&[(VarId(0), Some(false)), (VarId(2), Some(true))])?;
    /// assert_eq!(counter.model_count()?, 1u32.into());
    /// counter.set_pins(&[(VarId(0), None)])?;
    /// assert_eq!(counter.model_count()?, 3u32.into());
    /// # Ok::<(), tididi::OperationError>(())
    /// ```
    pub fn set_pins(&mut self, pins: &[(VarId, Option<bool>)]) -> Result<(), OperationError> {
        for &(var, _) in pins { self.validate_pin(var)?; }
        for &(var, val) in pins {
            let leaf = self.tdd.vtree.leaf_of(var).expect("validated pin variable");
            self.set_leaf_pin(leaf, val);
        }
        Ok(())
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
    /// counter.set_pins(&[(VarId(0), Some(false)), (VarId(2), Some(true))])?;
    /// counter.clear_pins();
    /// assert_eq!(counter.model_count()?, 6u32.into());
    /// # Ok::<(), tididi::OperationError>(())
    /// ```
    pub fn clear_pins(&mut self) {
        for leaf in 0..self.pins.len() {
            if self.pins[leaf].value.is_some() { self.set_leaf_pin(VtreeIdx(leaf as u32), None); }
        }
    }

    /// Resolve a variable to its structural leaf without changing counter state.
    fn validate_pin(&self, var: VarId) -> Result<VtreeIdx, OperationError> {
        let leaf = self.tdd.vtree.leaf_of(var).ok_or(OperationError::VariableNotInVtree(var))?;
        // An implicit integer leaf can remain below a marginal parent.
        for level in std::iter::once(leaf).chain(self.tdd.vtree.node(leaf).parent()) {
            if self.tdd.levels[level.idx()].is_marginal() {
                return Err(OperationError::MarginalLevel(level));
            }
        }
        Ok(leaf)
    }

    /// Update a validated leaf's pin and record its deferred refresh once.
    fn set_leaf_pin(&mut self, leaf: VtreeIdx, val: Option<bool>) {
        if !R::KEEP_ALL_COLUMNS {
            if self.pins[leaf.idx()].value != val {
                self.pins[leaf.idx()].value = val;
                self.evaluated = false;
            }
            return;
        }
        if !self.evaluated {
            self.clear_changed();
            self.pins[leaf.idx()].value = val;
            return;
        }
        if self.pins[leaf.idx()].value == val { return; }
        self.pins[leaf.idx()].value = val;
        if !self.pins[leaf.idx()].dirty {
            self.changed.push(leaf);
            self.pins[leaf.idx()].dirty = true;
        }
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
            if lim.should_stop() { return Err(OperationError::Stopped); }
            let mut gate = PollGate::new(lim.reduce_poll_stride());
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
            lim.poll(&mut gate, 1)?;
            lim.flush_poll(&mut gate)?;
            Ok(count)
        })();
        if result.is_err() {
            self.evaluated = false;
            self.clear_changed();
        }
        result
    }

    /// Refresh all columns or the dirty ancestor cone, leaving a failed pass invalidated.
    fn refresh(&mut self, eng: &Engine, gate: &mut PollGate) -> Result<(), OperationError> {
        if self.evaluated && self.changed.is_empty() { return Ok(()); }
        let incremental = self.evaluated && R::KEEP_ALL_COLUMNS;
        self.evaluated = false;
        let tdd = self.tdd;
        if incremental {
            eng.limits().try_resize(&mut self.pins, tdd.vtree.num_nodes(), PinState::default())?;
            let leaves = self.changed.len();
            for i in 0..leaves {
                let mut current = self.changed[i];
                while let Some(parent) = tdd.vtree.node(current).parent() {
                    if self.pins[parent.idx()].dirty { break; }
                    eng.limits().try_push(&mut self.changed, parent)?;
                    self.pins[parent.idx()].dirty = true;
                    eng.limits().poll(gate, 1)?;
                    current = parent;
                }
                eng.limits().poll(gate, 1)?;
            }
            tdd.vtree.sort_bottom_up(&mut self.changed);
            let fold = OverflowingCounts { pins: &self.pins, convention: self.convention };
            for &level in &self.changed {
                fold_level(&fold, eng, tdd, &mut self.cols, level, Some(gate))?;
            }
        } else {
            let fold = OverflowingCounts { pins: &self.pins, convention: self.convention };
            if !R::KEEP_ALL_COLUMNS {
                for col in &mut self.cols { *col = CountVec::default(); }
            }
            let retention = if R::KEEP_ALL_COLUMNS { ColumnRetention::All } else { ColumnRetention::Frontier };
            fold_bottom_up(&fold, eng, tdd, &mut self.cols, retention, Some(gate), |cols, ti| {
                let width = tdd.reference_slot_count(VtreeIdx(ti as u32));
                if cols[ti].len() != width { cols[ti] = fold.alloc(eng, width)?; }
                Ok(())
            })?;
        }
        self.clear_changed();
        self.evaluated = true;
        Ok(())
    }

    /// Clear pending traversal membership while retaining its allocated storage.
    fn clear_changed(&mut self) {
        for &level in &self.changed { self.pins[level.idx()].dirty = false; }
        self.changed.clear();
    }
}

impl ModelCounter<'_, KeepAllColumns> {
    /// Compute and return every fast count slot, preserving overflow sentinels.
    pub(crate) fn into_fast_counts(mut self, eng: &Engine) -> Result<Vec<Vec<u128>>, OperationError> {
        let _op = eng.limits().begin_operation();
        let mut gate = PollGate::new(eng.limits().reduce_poll_stride());
        self.refresh(eng, &mut gate)?;
        Ok(self.cols.into_iter().map(|c| c.into_parts().0).collect())
    }
}

#[cfg(test)]
#[path = "tests/incremental.rs"]
mod tests;
