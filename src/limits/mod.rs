//! Resource limits, cancellation and work measurements.
//!
//! Configure a batch with [`Context::with_limits`](crate::Context::with_limits),
//! or use [`Limits::scope`] to install a [`LimitConfig`] on an existing engine.
//! [`Limits::meters`] reports the work charged to that engine.
//!
//! The memory budget covers tracked reservations within one operation. It is
//! not a process-memory limit: untracked allocations can exceed it and may abort
//! on allocator failure. Nested operations share their parent's charges.
//! Output-node limits apply to the operations listed on
//! [`LimitConfig::with_output_node_cap`]. Cancellation takes effect at poll points.

pub(crate) mod pool;
mod error;
pub(crate) mod growth;
mod memory;
mod meters;
mod poll;
mod stop;

use std::cell::{Cell, RefCell};
use std::sync::Arc;
use std::time::Instant;

pub use error::OperationError;
pub use memory::MemoryHooks;
pub use meters::{OperationMetrics, ConjunctionProgress};
pub use stop::{StopDecision, StopRules, StopAt};

pub(crate) use growth::PAIR_ELEM_BYTES;
pub(crate) use growth::{Charged, Transient};

pub(crate) use poll::PollGate;


/// A cancellation callback receiving the current measurements and poll time.
/// See [`LimitConfig::stop_callback`] for invocation order.
#[derive(Clone)]
pub struct StopCallback(Arc<StopCallbackFn>);

type StopCallbackFn = dyn Fn(&OperationMetrics, Instant) -> StopDecision + Send + Sync;

impl StopCallback {
    /// Own a callback and its captured state, which must support transfer between threads.
    ///
    /// Share a cancellation flag with the caller:
    ///
    /// ```
    /// use std::sync::{Arc, atomic::{AtomicBool, Ordering}};
    /// use tididi::limits::{StopCallback, StopDecision};
    ///
    /// let cancelled = Arc::new(AtomicBool::new(false));
    /// let flag = Arc::clone(&cancelled);
    /// let callback = StopCallback::new(move |_, _| {
    ///     if flag.load(Ordering::Relaxed) { StopDecision::Stop }
    ///     else { StopDecision::Continue }
    /// });
    /// cancelled.store(true, Ordering::Relaxed);
    /// let meters = tididi::Engine::new().limits().meters();
    /// assert_eq!(callback.decide(&meters, std::time::Instant::now()), StopDecision::Stop);
    /// ```
    pub fn new(decide: impl Fn(&OperationMetrics, Instant) -> StopDecision + Send + Sync + 'static) -> Self {
        Self(Arc::new(decide))
    }

    /// Ask the installed policy at the current meters and clock reading.
    pub fn decide(&self, meters: &OperationMetrics, now: Instant) -> StopDecision {
        (self.0)(meters, now)
    }
}

impl std::fmt::Debug for StopCallback {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("StopCallback")
    }
}

/// Resource limits and callbacks to install together on an engine.
///
/// Start with [`LimitConfig::none`], set the bounds needed by the application,
/// then install them with [`Limits::scope`] for a block or [`Limits::install`]
/// until explicitly replaced. Installing a configuration replaces every setting;
/// [`Limits::edit`] changes selected settings while preserving the rest.
///
/// ```
/// use tididi::Engine;
/// use tididi::limits::LimitConfig;
///
/// let engine = Engine::new();
/// let config = LimitConfig::none()
///     .with_memory_budget_bytes(Some(64 * 1024 * 1024))
///     .with_output_node_cap(Some(100_000));
/// let _limits = engine.limits().scope(config);
/// // Checked operations here use these limits; dropping _limits restores the prior set.
/// ```
///
/// The byte budget covers charged allocation growth within one operation;
/// it is not a process-memory cap. A deadline is cooperative: it is observed
/// at operation poll points, not by preempting running code. The output-node
/// cap applies only to the operations listed on [`Self::with_output_node_cap`].
#[derive(Clone, Debug, Default)]
pub struct LimitConfig {
    memory_budget_bytes: Option<u64>,
    output_node_cap: Option<u64>,
    stop: StopRules,
    stop_callback: Option<StopCallback>,
    memory_hooks: MemoryHooks,
    conjunction_progress: bool,
}

impl LimitConfig {
    /// No limits or callbacks.
    #[must_use]
    pub fn none() -> LimitConfig {
        LimitConfig::default()
    }

    /// Set the soft byte budget. Best effort: only the reservations routed
    /// through the engine are charged, so an operation can run past it. `None`
    /// arms none; an allocator refusal is still [`OperationError::OverBudget`].
    #[must_use]
    pub fn with_memory_budget_bytes(mut self, bytes: Option<u64>) -> LimitConfig {
        self.memory_budget_bytes = bytes;
        self
    }

    /// Set the cap on emitted nodes for conjunction, checked construction,
    /// structural projection, and care rebuilding; `None` arms none.
    ///
    /// Each operation states which intermediate nodes it counts; exceeding
    /// the cap returns [`OperationError::OutputCap`]. Complementation and the
    /// operations built on it (`or`, `xor`, `ite`, `implies`) are bounded by
    /// the byte budget and the stop rules, not by this cap.
    #[must_use]
    pub fn with_output_node_cap(mut self, cap: Option<u64>) -> LimitConfig {
        self.output_node_cap = cap;
        self
    }

    /// Set cancellation thresholds; `StopRules::default()` disables them.
    #[must_use]
    pub fn with_stop_rules(mut self, stop: StopRules) -> LimitConfig {
        self.stop = stop;
        self
    }

    /// Set a deadline independently of output size, preserving the other stop rules
    /// and callback. [`Self::without_stop_rules`] clears all cancellation settings.
    #[must_use]
    pub fn with_deadline(mut self, deadline: Option<Instant>) -> LimitConfig {
        self.stop.unconditional = deadline.map(StopAt::Time);
        self
    }

    /// Remove the stop bounds and callback, leaving memory and output limits unchanged.
    #[must_use]
    pub fn without_stop_rules(mut self) -> LimitConfig {
        self.stop = StopRules::NONE;
        self.stop_callback = None;
        self
    }

    /// Set the cancellation callback; `None` removes it.
    /// See [`Self::stop_callback`] for invocation order.
    #[must_use]
    pub fn with_stop_callback(mut self, s: Option<StopCallback>) -> LimitConfig {
        self.stop_callback = s;
        self
    }

    /// Set the host's memory hooks; [`MemoryHooks::NONE`] disables them.
    #[must_use]
    pub fn with_memory_hooks(mut self, m: MemoryHooks) -> LimitConfig {
        self.memory_hooks = m;
        self
    }

    /// Enable progress reporting through [`OperationMetrics::conjunction`].
    /// Disabling it leaves the last recorded progress unchanged.
    #[must_use]
    pub fn with_conjunction_progress(mut self, on: bool) -> LimitConfig {
        self.conjunction_progress = on;
        self
    }


    /// The soft budget, in bytes, that one operation may grow its storage by
    /// before it fails with [`OperationError::OverBudget`]. `None` disables the
    /// predictive check; the fallible reserves still catch an allocator refusal.
    #[must_use]
    #[inline]
    pub fn memory_budget_bytes(&self) -> Option<u64> {
        self.memory_budget_bytes
    }

    /// The emitted-node cap for the operations listed by [`Self::with_output_node_cap`].
    /// Exceeding it returns [`OperationError::OutputCap`].
    #[must_use]
    #[inline]
    pub fn output_node_cap(&self) -> Option<u64> {
        self.output_node_cap
    }

    /// The installed stop bounds, independently of the callback.
    ///
    /// Read them to change one bound while preserving the other:
    /// `let stop = s.stop_rules().after_pairs(n, at); s.with_stop_rules(stop)`.
    #[must_use]
    #[inline]
    pub fn stop_rules(&self) -> StopRules {
        self.stop
    }

    /// The callback invoked before cancellation thresholds are checked at each poll.
    ///
    /// It receives the current measurements and poll time, even when no threshold
    /// is configured. Returning [`StopDecision::ReplaceRules`] changes the active
    /// thresholds, which [`Limits::armed`] reports.
    #[must_use]
    #[inline]
    pub fn stop_callback(&self) -> Option<StopCallback> {
        self.stop_callback.clone()
    }

    /// The host's memory hooks.
    #[must_use]
    #[inline]
    pub fn memory_hooks(&self) -> MemoryHooks {
        self.memory_hooks.clone()
    }

    /// Whether conjunctions update [`OperationMetrics::conjunction`].
    #[must_use]
    #[inline]
    pub fn conjunction_progress_enabled(&self) -> bool {
        self.conjunction_progress
    }
}

/// Active resource limits and work measurements for one engine.
///
/// Callbacks may inspect or change settings without holding an internal borrow.
pub struct Limits {
    pub(crate) retained_scratch: Cell<usize>,
    budget: Cell<Option<u64>>,
    in_flight_bytes: Cell<u64>,
    pairs_in_flight: Cell<u64>,
    pairs_level_charge: Cell<u64>,
    work_clock: Cell<u64>,
    stop: Cell<StopRules>,
    stop_callback: RefCell<Option<StopCallback>>,
    output_node_cap: Cell<Option<u64>>,
    bounded_growth: Cell<bool>,
    conjunction_progress: Cell<bool>,
    conjunction: Cell<Option<ConjunctionProgress>>,
    memory_hooks: RefCell<MemoryHooks>,
    /// The address-space ceiling, answered once per install: it is stable for
    /// the life of the probes, and the growth machinery asks per huge level.
    vas_limit: Cell<Option<Option<u64>>>,
    #[cfg(test)]
    poll_stride_pin: Cell<Option<u64>>,
    /// Operations in flight on this engine; the meters are zeroed when it
    /// goes from zero to one.
    op_depth: Cell<u32>,
    #[cfg(test)]
    refuse_after: Cell<Option<u32>>,
    #[cfg(test)]
    width_cap_pin: Cell<Option<usize>>,
    /// Bytes asked for by the most recent reserve the allocator turned down.
    /// An allocator refusal and a soft-budget refusal both arrive as
    /// [`OperationError::OverBudget`]; the size tells a caller which it was.
    refused_bytes: Cell<Option<u64>>,
}

/// A work-clock reading to compare with [`Limits::work_since`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WorkMark(u64);

impl std::fmt::Debug for Limits {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Limits")
            .field("armed", &self.armed())
            .field("meters", &self.meters())
            .finish()
    }
}

impl Limits {
    /// No configured limits and zeroed measurements.
    #[must_use]
    pub(crate) const fn new() -> Limits {
        Limits {
            retained_scratch: Cell::new(0),
            budget: Cell::new(None),
            in_flight_bytes: Cell::new(0),
            pairs_in_flight: Cell::new(0),
            pairs_level_charge: Cell::new(0),
            work_clock: Cell::new(0),
            stop: Cell::new(StopRules::NONE),
            stop_callback: RefCell::new(None),
            output_node_cap: Cell::new(None),
            bounded_growth: Cell::new(false),
            conjunction_progress: Cell::new(false),
            conjunction: Cell::new(None),
            memory_hooks: RefCell::new(MemoryHooks::NONE),
            vas_limit: Cell::new(None),
            #[cfg(test)]
            poll_stride_pin: Cell::new(None),
            op_depth: Cell::new(0),
            #[cfg(test)]
            refuse_after: Cell::new(None),
            #[cfg(test)]
            width_cap_pin: Cell::new(None),
            refused_bytes: Cell::new(None),
        }
    }


    /// Snapshot the active configuration.
    #[must_use]
    pub fn armed(&self) -> LimitConfig {
        LimitConfig {
            memory_budget_bytes: self.budget.get(),
            output_node_cap: self.output_node_cap.get(),
            stop: self.stop.get(),
            stop_callback: self.stop_callback.borrow().clone(),
            memory_hooks: self.memory_hooks.borrow().clone(),
            conjunction_progress: self.conjunction_progress.get(),
        }
    }

    /// Replace the configuration and return the previous settings without resetting measurements.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{OperationError, Engine, Tdd};
    /// use tididi::limits::LimitConfig;
    /// use tididi::vtree::Vtree;
    ///
    /// let vtree = Arc::new(Vtree::balanced(4));
    /// let engine = Engine::new();
    ///
    /// // Nothing is armed on a fresh engine, so the conjunction runs.
    /// let f = Tdd::clause(&vtree, [1, -2])?;
    /// let g = Tdd::clause(&vtree, [2, 3])?;
    /// assert!(engine.and(f, g).is_ok());
    ///
    /// // Arm a byte budget of zero; the next conjunction is refused.
    /// let prior = engine.limits().install(LimitConfig::none().with_memory_budget_bytes(Some(0)));
    /// let (f, g) = (Tdd::clause(&vtree, [1, -2])?, Tdd::clause(&vtree, [2, 3])?);
    /// match engine.and(f, g) {
    ///     Ok(_) => unreachable!("no reservation can be granted"),
    ///     Err(e) => assert_eq!(e, OperationError::OverBudget),
    /// }
    ///
    /// // Put back what was armed before and the engine runs freely again.
    /// let refused = engine.limits().install(prior);
    /// assert_eq!(refused.memory_budget_bytes(), Some(0));
    /// let (f, g) = (Tdd::clause(&vtree, [1, -2])?, Tdd::clause(&vtree, [2, 3])?);
    /// assert!(engine.and(f, g).is_ok());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[must_use = "install returns the prior set; bind it or use scope/edit"]
    pub fn install(&self, set: LimitConfig) -> LimitConfig {
        let prior = self.armed();
        self.budget.set(set.memory_budget_bytes);
        self.output_node_cap.set(set.output_node_cap);
        self.stop.set(set.stop);
        self.stop_callback.replace(set.stop_callback);
        self.conjunction_progress.set(set.conjunction_progress);
        self.memory_hooks.replace(set.memory_hooks);
        self.vas_limit.set(None);
        prior
    }

    /// Install `set` until the returned guard drops, then restore the previous
    /// configuration, including during panic unwinding.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{Engine, OperationError, Vtree};
    /// use tididi::limits::LimitConfig;
    ///
    /// let engine = Engine::new();
    /// let vtree = Arc::new(Vtree::balanced(3));
    /// {
    ///     let _limit = engine.limits().scope(LimitConfig::none().with_memory_budget_bytes(Some(0)));
    ///     assert_eq!(engine.clause(&vtree, [1, 2]).err(), Some(OperationError::OverBudget));
    /// }
    /// let f = engine.clause(&vtree, [1, 2]).unwrap(); // the previous limits are restored
    /// assert_eq!(f.model_count()?, 6u32.into());
    /// # tididi::test_helpers::assert_canonical(&f);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[must_use = "the scope restores the prior set when dropped; bind it to a name"]
    pub fn scope(&self, set: LimitConfig) -> LimitScope<'_> {
        LimitScope { lim: self, prior: self.install(set) }
    }

    /// Temporarily change selected settings, preserving the rest.
    /// Dropping the guard restores the previous configuration.
    ///
    /// ```
    /// use tididi::Engine;
    /// use tididi::limits::LimitConfig;
    ///
    /// let engine = Engine::new();
    /// let _outer = engine.limits().scope(LimitConfig::none().with_memory_budget_bytes(Some(4096)));
    /// {
    ///     let _inner = engine.limits().edit(|config| config.with_output_node_cap(Some(10)));
    ///     assert_eq!(engine.limits().armed().memory_budget_bytes(), Some(4096));
    ///     assert_eq!(engine.limits().armed().output_node_cap(), Some(10));
    /// }
    /// assert_eq!(engine.limits().armed().output_node_cap(), None);
    /// ```
    #[must_use = "the scope restores the prior set when dropped; bind it to a name"]
    pub fn edit(&self, edit: impl FnOnce(LimitConfig) -> LimitConfig) -> LimitScope<'_> {
        self.scope(edit(self.armed()))
    }

    /// The armed soft byte budget.
    #[must_use]
    #[inline]
    pub(crate) fn budget(&self) -> Option<u64> {
        self.budget.get()
    }

    /// Set or clear the soft budget alone, for a caller that re-derives it as it
    /// goes (a compile loop refreshing `budget − live` after every step).
    pub fn set_budget(&self, remaining_bytes: Option<u64>) {
        self.budget.set(remaining_bytes);
    }


    /// Snapshot the meters. What is armed reads back through
    /// [`Limits::armed`].
    #[must_use]
    pub fn meters(&self) -> OperationMetrics {
        OperationMetrics {
            in_flight_bytes: self.in_flight_bytes.get(),
            pairs_in_flight: self.pairs_in_flight.get(),
            work_units: self.work_clock.get(),
            refused_reserve_bytes: self.refused_bytes.get(),
            conjunction: self.conjunction.get(),
        }
    }

    /// Zero the in-flight byte meter and forget any recorded allocator refusal.
    ///
    /// Every operation zeroes the meter at entry; this is for a caller that
    /// charges reservations of its own between operations and wants them
    /// metered from zero.
    pub fn reset_meters(&self) {
        self.in_flight_bytes.set(0);
        self.refused_bytes.set(None);
    }

    /// Enter an operation: zero every per-operation meter unless another
    /// operation on this engine is already in flight, and hold the depth
    /// until the guard drops.
    #[must_use = "the guard marks the operation in flight until it drops; bind it to a name"]
    pub(crate) fn begin_operation(&self) -> OperationScope<'_> {
        if self.op_depth.get() == 0 {
            self.in_flight_bytes.set(0);
            self.pairs_in_flight.set(0);
            self.pairs_level_charge.set(0);
            self.bounded_growth.set(false);
        }
        self.op_depth.set(self.op_depth.get() + 1);
        OperationScope { lim: self }
    }

    /// The post-conjunction walks' poll stride.
    #[inline]
    pub(crate) fn reduce_poll_stride(&self) -> u64 {
        #[cfg(test)]
        if let Some(stride) = self.poll_stride_pin.get() { return stride; }
        poll::REDUCE_POLL_STRIDE
    }

    /// The work clock: units the operations run on these limits have polled
    /// through. Monotone and never reset, so an interval is a subtraction of
    /// two reads.
    #[must_use]
    #[inline]
    pub fn work_units(&self) -> u64 {
        self.work_clock.get()
    }

    /// Record the work clock for a later call to [`Self::work_since`].
    /// The clock is never reset, so a mark remains valid across operations.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{Engine, Vtree};
    ///
    /// let engine = Engine::new();
    /// let vtree = Arc::new(Vtree::balanced(3));
    /// let start = engine.limits().mark();
    /// let f = engine.clause(&vtree, [1, 2, 3]).unwrap();
    /// let work = engine.limits().work_since(start);
    /// assert!(work > 0);
    /// # tididi::test_helpers::assert_canonical(&f);
    /// ```
    #[must_use]
    #[inline]
    pub fn mark(&self) -> WorkMark {
        WorkMark(self.work_clock.get())
    }

    /// Units the clock has run since `mark`.
    #[must_use]
    #[inline]
    pub fn work_since(&self, mark: WorkMark) -> u64 {
        self.work_clock.get().saturating_sub(mark.0)
    }

    /// Add `units` to the work clock.
    #[inline]
    pub(crate) fn charge_work(&self, units: u64) {
        self.work_clock
            .set(self.work_clock.get().saturating_add(units));
    }


    /// Charge `bytes` of newly reserved storage against the soft budget.
    #[inline(always)]
    pub(crate) fn charge_bytes(&self, bytes: u64) -> Result<(), OperationError> {
        if bytes == 0 {
            return Ok(());
        }
        let total = self.in_flight_bytes.get().saturating_add(bytes);
        self.in_flight_bytes.set(total);
        match self.budget.get() {
            Some(rem) if total > rem => Err(OperationError::OverBudget),
            _ => Ok(()),
        }
    }

    /// Give `bytes` back when a transient's backing allocation is actually
    /// freed. The in-flight meter is otherwise monotone within one operation
    /// (persistent storage only grows, and the meter resets at entry), so a
    /// per-level scratch that charged itself and then drops mid-operation must
    /// un-charge here or it permanently consumes headroom it no longer uses.
    /// Only pair with a real free of the exact charged capacity.
    #[inline]
    pub(crate) fn release_bytes(&self, bytes: u64) {
        if bytes == 0 {
            return;
        }
        self.in_flight_bytes
            .set(self.in_flight_bytes.get().saturating_sub(bytes));
    }
}

/// Marks an operation in flight on its engine; see [`Limits::begin_operation`].
#[must_use = "the guard marks the operation in flight until it drops; bind it to a name"]
pub(crate) struct OperationScope<'a> {
    lim: &'a Limits,
}

impl Drop for OperationScope<'_> {
    fn drop(&mut self) {
        self.lim.op_depth.set(self.lim.op_depth.get() - 1);
    }
}

/// Restores the previous configuration when dropped.
/// Created by [`Limits::scope`] or [`Limits::edit`].
#[must_use = "the scope restores the prior set when dropped; bind it to a name"]
pub struct LimitScope<'a> {
    lim: &'a Limits,
    prior: LimitConfig,
}

impl std::fmt::Debug for LimitScope<'_> {
    /// The set that will be restored on drop.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LimitScope").field("restores", &self.prior).finish()
    }
}

impl Drop for LimitScope<'_> {
    fn drop(&mut self) {
        let _restored = self.lim.install(std::mem::take(&mut self.prior));
    }
}

impl Limits {
    /// Add a synthetic byte charge for tests without allocating memory.
    #[doc(hidden)]
    pub fn charge_in_flight(&self, bytes: u64) {
        self.in_flight_bytes
            .set(self.in_flight_bytes.get().saturating_add(bytes));
    }
}

#[cfg(test)]
mod tests;
