//! What an operation runs under and what it parks between calls: the byte
//! budget, the output-node cap, the stop axis, the host's memory probes, the
//! meters they are checked against, and the pool a scratch buffer waits in
//! between operations.
//!
//! Everything here hangs off one [`Limits`] value owned by the
//! [`Engine`](crate::Engine). A caller describes the axes it wants with a
//! [`LimitConfig`], arms them with [`Limits::install`], [`Limits::scope`] or
//! [`Limits::edit`], and reads what the operations spent as [`OperationMetrics`].
//! An operation charges the reservations it routes through the engine against
//! the budget and polls the stop axis as it runs.
//!
//! The byte budget is best effort. Only the reservations routed through the
//! engine are charged, so an operation can run past the budget by whatever it
//! allocates elsewhere, and an allocation outside the charged path that the
//! operating system refuses aborts the process as any Rust allocation does.
//! The meter the budget is checked against is zeroed when an operation starts
//! and by [`Limits::reset_meters`], so a budget bounds one operation at a
//! time; an operation another one runs as a step keeps the outer meter. The
//! output-node cap and the stop axis are exact.

pub(crate) mod policy;
pub(crate) mod pool;
mod error;
pub(crate) mod growth;
mod memory;
mod meters;
mod poll;
mod stop;

use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::time::Instant;

pub use error::OperationError;
pub use memory::MemoryHooks;
pub use meters::{OperationMetrics, ConjunctionProgress};
pub use stop::{StopDecision, StopRules, StopAt};

pub(crate) use growth::PAIR_ELEM_BYTES;
pub(crate) use meters::ByteCharge;
pub(crate) use policy::{unwrap_infallible, ApplyBudget, RecoveryPanic, ReservePolicy};
pub(crate) use poll::PollGate;


/// The decision callback a stop poll asks, handed the meters and the instant
/// the poll read; see [`LimitConfig::stop_callback`].
#[derive(Clone)]
pub struct StopCallback(Rc<ScheduleFn>);

type ScheduleFn = dyn Fn(&OperationMetrics, Instant) -> StopDecision;

impl StopCallback {
    /// Own a callback and any caller state it captures.
    pub fn new(decide: impl Fn(&OperationMetrics, Instant) -> StopDecision + 'static) -> Self {
        Self(Rc::new(decide))
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

/// The scalar limits and shared callback handles a caller arms together.
///
/// Installing a set replaces every axis; there is no per-axis install, and no
/// axis is left over from whatever ran before. A caller that wants to change
/// one axis reads the current set, arms the axis it wants on the value it read,
/// and installs the result — which is also how it restores what it found. The
/// axes are read back one at a time, so a set can gain an axis without any
/// caller having to name the ones it does not care about.
#[derive(Clone, Debug, Default)]
pub struct LimitConfig {
    memory_budget_bytes: Option<u64>,
    output_node_cap: Option<u64>,
    stop: StopRules,
    schedule: Option<StopCallback>,
    mem_pressure: MemoryHooks,
    watch: bool,
}

impl LimitConfig {
    /// Nothing armed.
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
    /// the cap returns [`OperationError::OutputCap`].
    #[must_use]
    pub fn with_output_node_cap(mut self, cap: Option<u64>) -> LimitConfig {
        self.output_node_cap = cap;
        self
    }

    /// Set the stop axis. `StopRules::default()` arms none.
    #[must_use]
    pub fn with_stop_rules(mut self, stop: StopRules) -> LimitConfig {
        self.stop = stop;
        self
    }

    /// Stop unconditionally at `deadline`, leaving the size-conditional bound
    /// and the schedule alone. [`LimitConfig::without_stop_rules`] is the verb that clears the
    /// whole axis.
    #[must_use]
    pub fn with_deadline(mut self, deadline: Option<Instant>) -> LimitConfig {
        self.stop.unconditional = deadline.map(StopAt::Time);
        self
    }

    /// Remove the stop bounds and callback, leaving memory and output limits unchanged.
    #[must_use]
    pub fn without_stop_rules(mut self) -> LimitConfig {
        self.stop = StopRules::NONE;
        self.schedule = None;
        self
    }

    /// Arm the decision callback the stop polls ask; see
    /// [`LimitConfig::stop_callback`]. `None` arms none.
    #[must_use]
    pub fn with_stop_callback(mut self, s: Option<StopCallback>) -> LimitConfig {
        self.schedule = s;
        self
    }

    /// Install the host's memory probes. [`MemoryHooks::NONE`], the default,
    /// is every probe a no-op.
    #[must_use]
    pub fn with_memory_hooks(mut self, m: MemoryHooks) -> LimitConfig {
        self.mem_pressure = m;
        self
    }

    /// Publish where each pairwise conjunction stands, as
    /// [`OperationMetrics::conjunction`]. Off, `conjunction` is never written.
    #[must_use]
    pub fn with_conjunction_progress(mut self, on: bool) -> LimitConfig {
        self.watch = on;
        self
    }

    // ── reading an axis back ───────────────────────────────────────────────

    /// The soft budget, in bytes, that one operation may grow its storage by
    /// before it fails with [`OperationError::OverBudget`]. `None` disables the
    /// predictive check; the fallible reserves still catch an allocator refusal.
    #[must_use]
    #[inline]
    pub fn memory_budget_bytes(&self) -> Option<u64> {
        self.memory_budget_bytes
    }

    /// The cap on the output nodes one pairwise conjunction may produce before
    /// it fails with [`OperationError::OutputCap`]. A deliberate size cut rather
    /// than a memory guard, which is why it is its own error variant.
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

    /// The decision callback the in-operation polls ask, handed the meters and
    /// the clock reading the poll has already taken. It is asked before the
    /// stop bounds on every poll, whether or not a bound is armed, so a caller
    /// with decision points of its own tests them itself and answers
    /// [`StopDecision::Continue`] until one arrives. A [`StopDecision::ReplaceRules`] answer
    /// rewrites the armed stop axis, which [`Limits::armed`] then reads back.
    #[must_use]
    #[inline]
    pub fn stop_callback(&self) -> Option<StopCallback> {
        self.schedule.clone()
    }

    /// The host's memory probes.
    #[must_use]
    #[inline]
    pub fn memory_hooks(&self) -> MemoryHooks {
        self.mem_pressure.clone()
    }

    /// Whether a conjunction in flight publishes where it stands, for
    /// [`Limits::meters`] to read as [`OperationMetrics::conjunction`]. An unwatched
    /// operation pays one `Cell` load and nothing else.
    #[must_use]
    #[inline]
    pub fn conjunction_progress_enabled(&self) -> bool {
        self.watch
    }
}

/// The armed limits and meters, with scalar charging in `Cell`s and owned callbacks.
///
/// Callback handles are cloned before invocation, so a callback holds no borrow of the settings.
pub struct Limits {
    budget_remaining: Cell<Option<u64>>,
    in_flight_bytes: Cell<u64>,
    pairs_in_flight: Cell<u64>,
    pairs_level_charge: Cell<u64>,
    work_clock: Cell<u64>,
    stop: Cell<StopRules>,
    schedule: RefCell<Option<StopCallback>>,
    output_node_cap: Cell<Option<u64>>,
    bounded_growth: Cell<bool>,
    watched: Cell<bool>,
    conjunction: Cell<Option<ConjunctionProgress>>,
    mem: RefCell<MemoryHooks>,
    /// The address-space ceiling, answered once per install: it is stable for
    /// the life of the probes, and the growth machinery asks per huge level.
    vas_limit: Cell<Option<Option<u64>>>,
    /// Pin for the post-conjunction walks' poll stride, so the amortization
    /// itself is observable without lowering the production cadence. `None`
    /// leaves the production cadence in force, which is what production runs on.
    poll_stride_pin: Cell<Option<u64>>,
    /// Operations in flight on this engine; the meters are zeroed when it
    /// goes from zero to one.
    op_depth: Cell<u32>,
    /// Consults left before the allocation-failure injection fires once.
    ///
    /// `None` — the production state — never fires. Armed by the tests that
    /// assert a refused reserve leaves the diagram exactly as it was: every
    /// fallible growth in the crate funnels through the reserve entries, so a
    /// count chooses which one is refused without any injection point being
    /// written into the algorithms themselves.
    refuse_after: Cell<Option<u32>>,
    /// Bytes asked for by the most recent reserve the allocator turned down.
    /// An allocator refusal and a soft-budget refusal both arrive as
    /// [`OperationError::OverBudget`]; the size tells a caller which it was.
    refused_bytes: Cell<Option<u64>>,
}

/// A reading of a [`Limits`] work clock, for measuring an interval of work
/// against.
///
/// Opaque: the only thing a caller does with a mark is hand it back to
/// [`Limits::work_since`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WorkMark(u64);

impl Default for Limits {
    fn default() -> Self {
        Limits::new()
    }
}

impl std::fmt::Debug for Limits {
    /// What is armed and what the armed axes are being checked against — the
    /// two reads the public surface already offers, side by side.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Limits")
            .field("armed", &self.armed())
            .field("meters", &self.meters())
            .finish()
    }
}

impl Limits {
    /// Nothing armed, every meter at zero.
    #[must_use]
    pub(crate) const fn new() -> Limits {
        Limits {
            budget_remaining: Cell::new(None),
            in_flight_bytes: Cell::new(0),
            pairs_in_flight: Cell::new(0),
            pairs_level_charge: Cell::new(0),
            work_clock: Cell::new(0),
            stop: Cell::new(StopRules::NONE),
            schedule: RefCell::new(None),
            output_node_cap: Cell::new(None),
            bounded_growth: Cell::new(false),
            watched: Cell::new(false),
            conjunction: Cell::new(None),
            mem: RefCell::new(MemoryHooks::NONE),
            vas_limit: Cell::new(None),
            poll_stride_pin: Cell::new(None),
            op_depth: Cell::new(0),
            refuse_after: Cell::new(None),
            refused_bytes: Cell::new(None),
        }
    }

    // ── what is armed ──────────────────────────────────────────────────────

    /// The armed set.
    #[must_use]
    pub fn armed(&self) -> LimitConfig {
        LimitConfig {
            memory_budget_bytes: self.budget_remaining.get(),
            output_node_cap: self.output_node_cap.get(),
            stop: self.stop.get(),
            schedule: self.schedule.borrow().clone(),
            mem_pressure: self.mem.borrow().clone(),
            watch: self.watched.get(),
        }
    }

    /// Arm `set`, returning what was armed before. The meters are untouched.
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
    /// let f = Tdd::clause(&vtree, [1, -2]);
    /// let g = Tdd::clause(&vtree, [2, 3]);
    /// assert!(engine.and(f, g).is_ok());
    ///
    /// // Arm a byte budget of zero; the next conjunction is refused.
    /// let prior = engine.limits().install(LimitConfig::none().with_memory_budget_bytes(Some(0)));
    /// let (f, g) = (Tdd::clause(&vtree, [1, -2]), Tdd::clause(&vtree, [2, 3]));
    /// match engine.and(f, g) {
    ///     Ok(_) => unreachable!("no reservation can be granted"),
    ///     Err(e) => assert_eq!(e, OperationError::OverBudget),
    /// }
    ///
    /// // Put back what was armed before and the engine runs freely again.
    /// let refused = engine.limits().install(prior);
    /// assert_eq!(refused.memory_budget_bytes(), Some(0));
    /// let (f, g) = (Tdd::clause(&vtree, [1, -2]), Tdd::clause(&vtree, [2, 3]));
    /// assert!(engine.and(f, g).is_ok());
    /// ```
    #[must_use = "install returns the prior set; bind it or use scope/edit"]
    pub fn install(&self, set: LimitConfig) -> LimitConfig {
        let prior = self.armed();
        self.budget_remaining.set(set.memory_budget_bytes);
        self.output_node_cap.set(set.output_node_cap);
        self.stop.set(set.stop);
        self.schedule.replace(set.schedule);
        self.watched.set(set.watch);
        self.mem.replace(set.mem_pressure);
        self.vas_limit.set(None);
        prior
    }

    /// Arm `set` for a lexical scope, restoring what was armed before when the
    /// returned guard drops.
    ///
    /// The restore happens on every exit path, an unwind included, which is
    /// what a caller that catches a panic and carries on needs: installing a
    /// set replaces every axis, so a limit armed for the work that panicked
    /// would otherwise still be armed for whatever runs next.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{Engine, OperationError, Vtree};
    /// use tididi::limits::LimitConfig;
    ///
    /// let engine = Engine::new();
    /// let tree = Arc::new(Vtree::balanced(3));
    /// {
    ///     let _limit = engine.limits().scope(LimitConfig::none().with_memory_budget_bytes(Some(0)));
    ///     assert_eq!(engine.clause(&tree, [1, 2]).err(), Some(OperationError::OverBudget));
    /// }
    /// let f = engine.clause(&tree, [1, 2]).unwrap(); // the previous limits are restored
    /// assert_eq!(f.model_count(), 6u32.into());
    /// # tididi::test_helpers::assert_canonical(&f);
    /// ```
    #[must_use = "the scope restores the prior set when dropped; bind it to a name"]
    pub fn scope(&self, set: LimitConfig) -> LimitScope<'_> {
        LimitScope { lim: self, prior: self.install(set) }
    }

    /// Arm the armed set with `edit` applied to it, for a lexical scope.
    ///
    /// The form for changing one axis and leaving the rest of the set where it
    /// is: `edit(|s| s.with_deadline(Some(t)))`.
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
        self.budget_remaining.get()
    }

    /// Set or clear the soft budget alone, for a caller that re-derives it as it
    /// goes (a compile loop refreshing `budget − live` after every step).
    pub fn set_budget(&self, remaining_bytes: Option<u64>) {
        self.budget_remaining.set(remaining_bytes);
    }

    // ── the meters ─────────────────────────────────────────────────────────

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
        poll::reduce_poll_stride(self.poll_stride_pin.get())
    }

    /// Whether the armed injection refuses this reserve. Disarmed — always, in
    /// production — this is one load of a cell that is `None`.
    #[inline(always)]
    pub(crate) fn refuses_reserve(&self) -> bool {
        match self.refuse_after.get() {
            None => false,
            Some(n) => self.count_down_refusal(n),
        }
    }

    /// Countdown arm of [`Limits::refuses_reserve`], reached only while the
    /// injection is armed.
    #[cold]
    #[inline(never)]
    fn count_down_refusal(&self, n: u32) -> bool {
        self.refuse_after.set(n.checked_sub(1));
        n == 0
    }

    /// The work clock: units the operations run on these limits have polled
    /// through. Monotone and never reset, so an interval is a subtraction of
    /// two reads.
    #[must_use]
    #[inline]
    pub fn work_units(&self) -> u64 {
        self.work_clock.get()
    }

    /// Mark the work clock here.
    ///
    /// The clock is monotone and never reset, so a mark stays valid across any
    /// operation boundary and scoping the clock to a caller's own unit of work
    /// — one attempt, one step — is [`Limits::work_since`] against a mark taken
    /// at its door.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{Engine, Vtree};
    ///
    /// let engine = Engine::new();
    /// let tree = Arc::new(Vtree::balanced(3));
    /// let start = engine.limits().mark();
    /// let f = engine.clause(&tree, [1, 2, 3]).unwrap();
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

    // ── the four verbs ─────────────────────────────────────────────────────

    /// Charge `bytes` of newly reserved storage against the soft budget.
    #[inline(always)]
    pub(crate) fn charge_bytes(&self, bytes: u64) -> Result<(), OperationError> {
        if bytes == 0 {
            return Ok(());
        }
        let total = self.in_flight_bytes.get().saturating_add(bytes);
        self.in_flight_bytes.set(total);
        match self.budget_remaining.get() {
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

/// Restores the [`LimitConfig`] that was armed when it was made.
///
/// Made by [`Limits::scope`] and [`Limits::edit`]; see those for what the
/// restore is for.
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
    /// Charge the in-flight meter as an aborted operation would have, without
    /// allocating the bytes: the seam the ownership rule on
    /// [`Limits::reset_meters`] is tested through, here and in a test suite
    /// built on the crate.
    #[cfg(any(test, debug_assertions))]
    #[doc(hidden)]
    pub fn charge_in_flight(&self, bytes: u64) {
        self.in_flight_bytes
            .set(self.in_flight_bytes.get().saturating_add(bytes));
    }
}

#[cfg(test)]
mod tests;
