//! The limits an operation runs under: the byte budget, the output-node cap,
//! the stop axis, the host's memory probes, and the meters they are checked
//! against.
//!
//! Everything here hangs off ONE value owned by the [`Engine`]. The four verbs
//! are [`Limits::charge_bytes`], [`Limits::release_bytes`], [`Limits::poll`]
//! and [`Limits::headroom`]; the level-shaped state an operation accumulates
//! goes in and out through [`Limits::begin_level`] and [`Limits::level_done`].
//! The allocation helpers (`try_push`, `try_resize`, `reserve`) are ergonomics
//! over `charge_bytes`, so [`ApplyError::OverBudget`] is minted in one place.

use std::cell::Cell;
use std::time::Instant;

use crate::error::ApplyError;

use super::memory::MemPressure;
use super::meters::{ApplyMeters, MergePosition};
use super::stop::{Scheduled, Stop, StopAt};

/// Poll hook consulted for a scheduled stop: sees the meters and the apply start instant.
pub type ScheduleHook = fn(&ApplyMeters, Instant) -> Scheduled;

/// Everything a caller arms, as one plain `Copy` value.
///
/// Installing a set replaces EVERY axis; there is no per-axis install, and no
/// axis is left over from whatever ran before. A caller that wants to change
/// one axis reads the current set, edits the field, and installs the result —
/// which is also how it restores what it found.
#[derive(Clone, Copy, Debug, Default)]
pub struct LimitSet {
    /// Soft budget, in bytes, that one operation may grow its storage by before
    /// it fails with [`ApplyError::OverBudget`]. `None` disables the predictive
    /// check; the fallible reserves still catch an OS-level refusal.
    pub budget_bytes: Option<u64>,
    /// Cap on the output nodes one conjunction may produce before it fails with
    /// [`ApplyError::OutputCap`]. A deliberate size cut rather than a memory
    /// guard, which is why it is its own error variant.
    pub output_node_cap: Option<u64>,
    /// When the operation gives up. [`Stop::NONE`] is every operation nobody
    /// walled in.
    pub stop: Stop,
    /// A decision callback the in-operation polls ask, handed the clock reading
    /// the poll has already taken. The stop says when the operation MUST end;
    /// this says that it must be ASKED. The callback is asked on every poll:
    /// this crate holds no view on when a decision is due, so a caller with
    /// decision points of its own tests them itself and answers
    /// [`Scheduled::Carry`] until one arrives. What the poll provides is the one
    /// thing the caller cannot — a place to stand inside an operation, on a poll
    /// the operation was already paying for.
    pub schedule: Option<ScheduleHook>,
    /// The host's memory probes.
    pub mem_pressure: MemPressure,
    /// Publish where a conjunction in flight stands, for [`Limits::meters`] to
    /// read as [`ApplyMeters::merge`]. An unwatched operation pays one `Cell`
    /// load and nothing else.
    pub watch: bool,
}

impl LimitSet {
    /// Nothing armed.
    #[must_use]
    pub fn none() -> LimitSet {
        LimitSet::default()
    }

    /// Set the soft byte budget.
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

    /// Stop unconditionally at `deadline` (leaving any size-conditional bound
    /// alone).
    #[must_use]
    pub fn deadline(mut self, deadline: Option<Instant>) -> LimitSet {
        self.stop.wall = deadline.map(StopAt::Wall);
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
    merge: Cell<Option<MergePosition>>,
    mem: Cell<MemPressure>,
    /// The address-space ceiling, answered once per install: it is stable for
    /// the life of the probes, and the growth machinery asks per huge level.
    vas_limit: Cell<Option<Option<u64>>>,
    /// Test-only pin for the post-conjunction walks' poll stride, so the
    /// amortization itself is testable without lowering the production cadence.
    #[cfg(test)]
    poll_stride_pin: Cell<Option<u64>>,
    /// Bytes asked for by the most recent reserve the allocator REFUSED.
    ///
    /// "The allocator said no" and "the soft budget said no" both arrive at the
    /// caller as [`ApplyError::OverBudget`], and the two demand opposite
    /// responses: a refused 300 GB grid is a size the compile can never have on
    /// any machine, while a refused 20 GB one is a machine that is currently
    /// full.
    refused_bytes: Cell<Option<u64>>,
}

impl Default for Limits {
    fn default() -> Self {
        Limits::new()
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
            #[cfg(test)]
            poll_stride_pin: Cell::new(None),
            refused_bytes: Cell::new(None),
        }
    }

    /// A fresh set of limits with `stop` armed.
    #[must_use]
    pub fn with_stop(stop: Stop) -> Limits {
        let lim = Limits::new();
        lim.install(LimitSet::none().stop(stop));
        lim
    }

    /// A fresh set of limits with an output-node cap armed.
    #[must_use]
    pub fn with_output_cap(cap: u64) -> Limits {
        let lim = Limits::new();
        lim.install(LimitSet::none().output_cap(Some(cap)));
        lim
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

    /// Snapshot the armed limits and the meters.
    #[must_use]
    pub fn meters(&self) -> ApplyMeters {
        ApplyMeters {
            budget_remaining: self.budget_remaining.get(),
            in_flight_bytes: self.in_flight_bytes.get(),
            pairs_in_flight: self.pairs_in_flight.get(),
            work_units: self.work_clock.get(),
            refused_reserve_bytes: self.refused_bytes.get(),
            stop: self.stop.get(),
            schedule: self.schedule.get(),
            output_node_cap: self.output_node_cap.get(),
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
        #[cfg(test)]
        let pinned = self.poll_stride_pin.get();
        #[cfg(not(test))]
        let pinned = None;
        super::poll::reduce_poll_stride(pinned)
    }

    /// Pin the post-conjunction walks' poll stride for one test.
    #[cfg(test)]
    pub(crate) fn pin_reduce_poll_stride(&self, stride: Option<u64>) -> Option<u64> {
        self.poll_stride_pin.replace(stride)
    }

    /// The work clock: units the operations run on these limits have polled
    /// through. Monotone and never reset, so an interval is a subtraction of
    /// two reads.
    #[must_use]
    #[inline]
    pub fn work_units(&self) -> u64 {
        self.work_clock.get()
    }

    /// Add `units` to the work clock.
    #[inline]
    pub(crate) fn charge_work(&self, units: u64) {
        self.work_clock
            .set(self.work_clock.get().saturating_add(units));
    }

    /// Charge the in-flight meter as an aborted operation would have, without
    /// allocating the bytes: the test seam for the ownership rule on
    /// [`Limits::reset_meters`].
    #[cfg(any(test, debug_assertions))]
    pub fn charge_in_flight_for_test(&self, bytes: u64) {
        self.in_flight_bytes
            .set(self.in_flight_bytes.get().saturating_add(bytes));
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

    /// Add `work` units to `gate` and, once it comes due, charge the work clock
    /// and test the stop axis.
    ///
    /// The one amortized cut every in-operation loop makes: the dense between-
    /// cell loops, the sparse scatter and collapse collectors, the intra-cell
    /// N×M arm, and the walks that run between two conjunctions of one step.
    #[inline(always)]
    pub(crate) fn poll(&self, gate: &mut PollGate, work: u64) -> Result<(), ApplyError> {
        gate.work += work;
        if gate.work < gate.stride {
            return Ok(());
        }
        let done = std::mem::replace(&mut gate.work, 0);
        self.poll_now(done)
    }

    /// The cold half of [`Limits::poll`].
    ///
    /// `done` is what the gate actually held, NOT the stride it crossed: a
    /// single poll can carry a whole dense row, which may be many strides wide
    /// on its own, and charging one stride per poll would price that row the
    /// same as the narrowest one that trips the gate.
    #[cold]
    fn poll_now(&self, done: u64) -> Result<(), ApplyError> {
        self.charge_work(done);
        if self.should_stop() {
            return Err(ApplyError::Deadline);
        }
        Ok(())
    }
}

mod growth;

/// Emitted-pair bound above which [`Limits::begin_level`] runs the growth-mode
/// decision at all. Fixed at 128 M pairs: below it, doubling pays a few hundred
/// MiB of transient peak, so small levels skip the decision and never pay the
/// headroom read (whose address-space fallback does a microsecond-scale
/// epoch-advance read). Above it, doubling from capacity N to 2N transients 3N,
/// which on a level of that size is tens of GiB — exactly what the bounded mode
/// protects against.
pub(crate) const DENSE_GROWTH_DECISION_THRESHOLD: u128 = 128 * 1024 * 1024;

/// Bytes one output pair occupies in a level's arena — the unit the emit
/// growth policy and the level's doubling-transient estimate are stated in.
pub(crate) const PAIR_ELEM_BYTES: u64 = std::mem::size_of::<crate::diagram::InputPair>() as u64;

/// The accumulator an in-operation loop polls through: one poll per `stride`
/// units of work, so the check amortizes to nothing.
///
/// The compile path is sequential — there is no cross-thread cancellation, so
/// the stop axis is the only mid-level cut.
pub(crate) struct PollGate {
    work: u64,
    stride: u64,
}

impl PollGate {
    #[inline]
    pub(crate) fn new(stride: u64) -> PollGate {
        PollGate { work: 0, stride }
    }
}
