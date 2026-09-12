//! What an operation runs under and what it parks between calls: the byte
//! budget, the output-node cap, the stop axis, the host's memory probes, the
//! meters they are checked against, and the pool a scratch buffer waits in
//! between operations.
//!
//! Everything here hangs off one [`Limits`] value owned by the
//! [`Engine`](crate::Engine). A caller describes the axes it wants with a
//! [`LimitSet`], arms them with `install` or `scope`, and reads what the last
//! operation spent as [`ApplyMeters`]. An operation charges the arenas and
//! scratch it reserves through the engine against the budget and polls the
//! stop axis as it runs, so [`ApplyError::OverBudget`] is minted in one place.
//!
//! The byte budget is best effort. What is charged is most of an operation's
//! growth, not all of it: a run can exceed the budget by an amount bounded by
//! the size of the diagram it builds, and an allocation outside the charged
//! path that the operating system refuses aborts the process as any Rust
//! allocation does. The output-node cap and the stop axis are exact.

pub(crate) mod policy;
pub(crate) mod pool;
mod error;
pub(crate) mod growth;
mod memory;
mod meters;
mod poll;
mod stop;

use std::cell::Cell;
use std::time::Instant;

pub use error::ApplyError;
pub use memory::MemPressure;
pub use meters::{ApplyMeters, MergeProgress};
pub use stop::{Scheduled, Stop, StopAt};

pub(crate) use growth::PAIR_ELEM_BYTES;
pub(crate) use meters::ByteCharge;
pub(crate) use policy::{unwrap_infallible, ApplyBudget, RecoveryPanic, ReservePolicy};
pub(crate) use poll::PollGate;


/// Poll hook consulted for a scheduled stop: sees the meters and the apply start instant.
pub type ScheduleHook = fn(&ApplyMeters, Instant) -> Scheduled;

/// Everything a caller arms, as one plain `Copy` value.
///
/// Installing a set replaces every axis; there is no per-axis install, and no
/// axis is left over from whatever ran before. A caller that wants to change
/// one axis reads the current set, arms the axis it wants on the value it read,
/// and installs the result — which is also how it restores what it found. The
/// axes are read back one at a time, so a set can gain an axis without any
/// caller having to name the ones it does not care about.
#[derive(Clone, Copy, Debug, Default)]
pub struct LimitSet {
    budget_bytes: Option<u64>,
    output_node_cap: Option<u64>,
    stop: Stop,
    schedule: Option<ScheduleHook>,
    mem_pressure: MemPressure,
    watch: bool,
}

impl LimitSet {
    /// Nothing armed.
    #[must_use]
    pub fn none() -> LimitSet {
        LimitSet::default()
    }

    /// Set the soft byte budget, which an operation may exceed by up to the
    /// size of the diagram it builds, since not every allocation is charged.
    #[must_use]
    pub fn budget(mut self, bytes: Option<u64>) -> LimitSet {
        self.budget_bytes = bytes;
        self
    }

    /// Set the output-node cap.
    #[must_use]
    pub fn output_cap(mut self, cap: Option<u64>) -> LimitSet {
        self.output_node_cap = cap;
        self
    }

    /// Set the stop axis.
    #[must_use]
    pub fn stop(mut self, stop: Stop) -> LimitSet {
        self.stop = stop;
        self
    }

    /// Stop unconditionally at `deadline`, leaving the size-conditional bound
    /// and the schedule alone. [`LimitSet::uncut`] is the verb that clears the
    /// whole axis.
    #[must_use]
    pub fn deadline(mut self, deadline: Option<Instant>) -> LimitSet {
        self.stop.wall = deadline.map(StopAt::Wall);
        self
    }

    /// Remove every bound on when the operation gives up: no stop, and no
    /// schedule to answer one.
    ///
    /// This is the verb for "run this to completion". [`LimitSet::deadline`]
    /// clears the unconditional wall alone, so a size-conditional bound armed
    /// by whatever ran before survives it, and an armed schedule can still
    /// answer [`Scheduled::Stop`]. The budget and the output cap guard memory
    /// on a different axis and are left where they are.
    #[must_use]
    pub fn uncut(mut self) -> LimitSet {
        self.stop = Stop::NONE;
        self.schedule = None;
        self
    }

    /// Arm the decision callback.
    #[must_use]
    pub fn schedule(mut self, s: Option<ScheduleHook>) -> LimitSet {
        self.schedule = s;
        self
    }

    /// Install the host's memory probes.
    #[must_use]
    pub fn mem_pressure(mut self, m: MemPressure) -> LimitSet {
        self.mem_pressure = m;
        self
    }

    /// Watch the conjunctions run under this set.
    #[must_use]
    pub fn watch(mut self, on: bool) -> LimitSet {
        self.watch = on;
        self
    }

    // ── reading an axis back ───────────────────────────────────────────────

    /// The soft budget, in bytes, that one operation may grow its storage by
    /// before it fails with [`ApplyError::OverBudget`]. `None` disables the
    /// predictive check; the fallible reserves still catch an OS-level refusal.
    #[must_use]
    #[inline]
    pub fn budget_bytes(&self) -> Option<u64> {
        self.budget_bytes
    }

    /// The cap on the output nodes one conjunction may produce before it fails
    /// with [`ApplyError::OutputCap`]. A deliberate size cut rather than a
    /// memory guard, which is why it is its own error variant.
    #[must_use]
    #[inline]
    pub fn output_node_cap(&self) -> Option<u64> {
        self.output_node_cap
    }

    /// When the operation gives up. `Stop::default()` is every operation nobody
    /// walled in. Read it to arm one of its bounds and leave the other alone:
    /// `s.stop(s.stop_axis().after_pairs(n, at))`.
    #[must_use]
    #[inline]
    pub fn stop_axis(&self) -> Stop {
        self.stop
    }

    /// The decision callback the in-operation polls ask, handed the clock
    /// reading the poll has already taken. The stop says when the operation
    /// must end; what this adds is the asking. The callback is asked on every
    /// poll: this crate holds no view on when a decision is due, so a caller
    /// with decision points of its own tests them itself and answers
    /// [`Scheduled::Carry`] until one arrives. What the poll provides is the one
    /// thing the caller cannot — a place to stand inside an operation, on a poll
    /// the operation was already paying for.
    #[must_use]
    #[inline]
    pub fn schedule_hook(&self) -> Option<ScheduleHook> {
        self.schedule
    }

    /// The host's memory probes.
    #[must_use]
    #[inline]
    pub fn memory_probes(&self) -> MemPressure {
        self.mem_pressure
    }

    /// Whether a conjunction in flight publishes where it stands, for
    /// [`Limits::meters`] to read as [`ApplyMeters::merge`]. An unwatched
    /// operation pays one `Cell` load and nothing else.
    #[must_use]
    #[inline]
    pub fn watching(&self) -> bool {
        self.watch
    }
}

/// The armed limits and the meters they are checked against, as one value of
/// plain `Cell`s.
///
/// `Cell`, not `RefCell`: the byte-charge path runs once per emitted node and
/// must stay a bare load and store.
pub struct Limits {
    budget_remaining: Cell<Option<u64>>,
    in_flight_bytes: Cell<u64>,
    pairs_in_flight: Cell<u64>,
    pairs_level_charge: Cell<u64>,
    work_clock: Cell<u64>,
    stop: Cell<Stop>,
    schedule: Cell<Option<ScheduleHook>>,
    output_node_cap: Cell<Option<u64>>,
    bounded_growth: Cell<bool>,
    watched: Cell<bool>,
    merge: Cell<Option<MergeProgress>>,
    mem: Cell<MemPressure>,
    /// The address-space ceiling, answered once per install: it is stable for
    /// the life of the probes, and the growth machinery asks per huge level.
    vas_limit: Cell<Option<Option<u64>>>,
    /// Pin for the post-conjunction walks' poll stride, so the amortization
    /// itself is observable without lowering the production cadence. `None`
    /// leaves the production cadence in force, which is what production runs on.
    poll_stride_pin: Cell<Option<u64>>,
    /// Consults left before the allocation-failure injection fires once.
    ///
    /// `None` — the production state — never fires. Armed by the tests that
    /// assert a refused reserve leaves the diagram exactly as it was: every
    /// fallible growth in the crate funnels through the reserve entries, so a
    /// count chooses which one is refused without any injection point being
    /// written into the algorithms themselves.
    refuse_after: Cell<Option<u32>>,
    /// Bytes asked for by the most recent reserve the allocator turned down.
    ///
    /// "The allocator said no" and "the soft budget said no" both arrive at the
    /// caller as [`ApplyError::OverBudget`], and the two demand opposite
    /// responses: a refused 300 GB grid is a size no diagram can have on
    /// any machine, while a refused 20 GB one is a machine that is currently
    /// full.
    refused_bytes: Cell<Option<u64>>,
}

/// A reading of a [`Limits`] work clock, for measuring an interval of work
/// against.
///
/// Opaque on purpose: what the number counts is this crate's business, and the
/// only thing a caller does with a mark is hand it back to
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
    pub const fn new() -> Limits {
        Limits {
            budget_remaining: Cell::new(None),
            in_flight_bytes: Cell::new(0),
            pairs_in_flight: Cell::new(0),
            pairs_level_charge: Cell::new(0),
            work_clock: Cell::new(0),
            stop: Cell::new(Stop::NONE),
            schedule: Cell::new(None),
            output_node_cap: Cell::new(None),
            bounded_growth: Cell::new(false),
            watched: Cell::new(false),
            merge: Cell::new(None),
            mem: Cell::new(MemPressure::NONE),
            vas_limit: Cell::new(None),
            poll_stride_pin: Cell::new(None),
            refuse_after: Cell::new(None),
            refused_bytes: Cell::new(None),
        }
    }

    // ── what is armed ──────────────────────────────────────────────────────

    /// The armed set.
    #[must_use]
    pub fn armed(&self) -> LimitSet {
        LimitSet {
            budget_bytes: self.budget_remaining.get(),
            output_node_cap: self.output_node_cap.get(),
            stop: self.stop.get(),
            schedule: self.schedule.get(),
            mem_pressure: self.mem.get(),
            watch: self.watched.get(),
        }
    }

    /// Arm `set`, returning what was armed before.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{ApplyError, Engine, Tdd};
    /// use tididi::limits::LimitSet;
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
    /// let prior = engine.limits().install(LimitSet::none().budget(Some(0)));
    /// let (f, g) = (Tdd::clause(&vtree, [1, -2]), Tdd::clause(&vtree, [2, 3]));
    /// match engine.and(f, g) {
    ///     Ok(_) => unreachable!("no reservation can be granted"),
    ///     Err(e) => assert_eq!(e, ApplyError::OverBudget),
    /// }
    ///
    /// // Put back what was armed before and the engine runs freely again.
    /// let refused = engine.limits().install(prior);
    /// assert_eq!(refused.budget_bytes(), Some(0));
    /// let (f, g) = (Tdd::clause(&vtree, [1, -2]), Tdd::clause(&vtree, [2, 3]));
    /// assert!(engine.and(f, g).is_ok());
    /// ```
    #[must_use = "install returns the prior set; bind it or use scope/edit"]
    pub fn install(&self, set: LimitSet) -> LimitSet {
        let prior = self.armed();
        self.budget_remaining.set(set.budget_bytes);
        self.output_node_cap.set(set.output_node_cap);
        self.stop.set(set.stop);
        self.schedule.set(set.schedule);
        self.watched.set(set.watch);
        self.mem.set(set.mem_pressure);
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
    #[must_use = "the scope restores the prior set when dropped; bind it to a name"]
    pub fn scope(&self, set: LimitSet) -> LimitScope<'_> {
        LimitScope { lim: self, prior: self.install(set) }
    }

    /// Arm the armed set with `edit` applied to it, for a lexical scope.
    ///
    /// The form for changing one axis and leaving the rest of the set where it
    /// is: `edit(|s| s.deadline(Some(t)))`.
    #[must_use = "the scope restores the prior set when dropped; bind it to a name"]
    pub fn edit(&self, edit: impl FnOnce(LimitSet) -> LimitSet) -> LimitScope<'_> {
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
    pub fn meters(&self) -> ApplyMeters {
        ApplyMeters {
            in_flight_bytes: self.in_flight_bytes.get(),
            pairs_in_flight: self.pairs_in_flight.get(),
            work_units: self.work_clock.get(),
            refused_reserve_bytes: self.refused_bytes.get(),
            merge: self.merge.get(),
        }
    }

    /// Zero the in-flight byte meter and forget any recorded allocator refusal.
    ///
    /// Both are per-operation state that a conjunction clears at entry, but
    /// tracked reserves also happen between conjunctions, so a traversal that
    /// ended inside a huge one leaves a large total behind. A traversal calls
    /// this at entry so it only ever measures bytes it charged itself.
    pub fn reset_meters(&self) {
        self.in_flight_bytes.set(0);
        self.refused_bytes.set(None);
    }

    /// Zero every per-operation meter. Called once at conjunction entry.
    pub(crate) fn begin_operation(&self) {
        self.in_flight_bytes.set(0);
        self.pairs_in_flight.set(0);
        self.pairs_level_charge.set(0);
        self.bounded_growth.set(false);
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
    pub(crate) fn charge_bytes(&self, bytes: u64) -> Result<(), ApplyError> {
        if bytes == 0 {
            return Ok(());
        }
        let total = self.in_flight_bytes.get().saturating_add(bytes);
        self.in_flight_bytes.set(total);
        match self.budget_remaining.get() {
            Some(rem) if total > rem => Err(ApplyError::OverBudget),
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

/// Restores the [`LimitSet`] that was armed when it was made.
///
/// Made by [`Limits::scope`] and [`Limits::edit`]; see those for what the
/// restore is for.
#[must_use = "the scope restores the prior set when dropped; bind it to a name"]
pub struct LimitScope<'a> {
    lim: &'a Limits,
    prior: LimitSet,
}

impl std::fmt::Debug for LimitScope<'_> {
    /// The set that will be restored on drop.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LimitScope").field("restores", &self.prior).finish()
    }
}

impl Drop for LimitScope<'_> {
    fn drop(&mut self) {
        let _restored = self.lim.install(self.prior);
    }
}

// Test support.
impl Limits {
    /// Arm the allocation-failure injection to refuse the `(n+1)`-th reserve
    /// this `Limits` is asked for: the next `n` are granted, the one after is
    /// refused, and the injection disarms itself.
    #[cfg(test)]
    pub(crate) fn refuse_nth_reserve(&self, n: u32) {
        self.refuse_after.set(Some(n));
    }

    /// Disarm the allocation-failure injection.
    #[cfg(test)]
    pub(crate) fn grant_every_reserve(&self) {
        self.refuse_after.set(None);
    }

    /// Pin the post-conjunction walks' poll stride, returning the prior pin.
    #[cfg(test)]
    pub(crate) fn pin_reduce_poll_stride(&self, stride: Option<u64>) -> Option<u64> {
        self.poll_stride_pin.replace(stride)
    }

    /// Charge the in-flight meter as an aborted operation would have, without
    /// allocating the bytes: the seam the ownership rule on
    /// [`Limits::reset_meters`] is tested through.
    #[cfg(test)]
    pub(crate) fn charge_in_flight(&self, bytes: u64) {
        self.in_flight_bytes
            .set(self.in_flight_bytes.get().saturating_add(bytes));
    }
}

#[cfg(test)]
mod tests;
