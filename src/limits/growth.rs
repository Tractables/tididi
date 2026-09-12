//! The memory half of the limits: how much room is left, which growth mode a
//! level runs in, and the allocation helpers that charge against the budget.

use std::time::Instant;

use crate::limits::OperationError;

use crate::limits::memory::{VAS_UNLIMITED_HEADROOM, vas_headroom_with_margin};
use crate::limits::meters::ConjunctionProgress;
use crate::limits::stop::{StopDecision, StopAt};

use super::Limits;

/// Emitted-pair bound above which [`Limits::begin_level`] runs the growth-mode
/// decision at all; a level bounded below it doubles without a headroom read.
pub(crate) const DENSE_GROWTH_DECISION_THRESHOLD: u128 = 128 * 1024 * 1024;

/// Bytes one output pair occupies in a level's arena — the unit the emit
/// growth policy and the level's doubling-transient estimate are stated in.
pub(crate) const PAIR_ELEM_BYTES: u64 = std::mem::size_of::<crate::diagram::ChildPair>() as u64;

impl Limits {
    /// Room the growth machinery may still take, with an address-space fallback
    /// when no soft budget is armed; always answers, where
    /// [`Limits::budget_headroom`] answers only under a soft budget.
    ///
    /// - **Soft budget armed**: exactly [`Limits::budget_headroom`] unwrapped;
    ///   no address space is consulted.
    /// - **No soft budget**: `RLIMIT_AS − margin − mapped`, through the
    ///   installed [`MemoryHooks`](super::MemoryHooks) probes. The margin holds room back below the
    ///   ceiling so the guarded path never consumes the last of the address
    ///   space, leaving somewhere for the unguarded transients that would
    ///   otherwise abort the process uncatchably.
    /// - **`RLIMIT_AS` unlimited**: nothing for a doubling transient to trip, so
    ///   a large finite figure.
    ///
    /// Conservative by construction: `mapped_bytes` is a high-water figure, so
    /// it can only over-count live usage, which only ever shrinks the answer.
    #[inline]
    pub(crate) fn headroom(&self) -> u64 {
        if let Some(h) = self.budget_headroom() {
            return h;
        }
        match self.address_space_limit() {
            Some(limit) => {
                let mem = self.mem.borrow().clone();
                vas_headroom_with_margin(limit, mem.mapped_bytes())
            },
            None => VAS_UNLIMITED_HEADROOM,
        }
    }

    /// Remaining soft-budget headroom: `budget − in flight`, saturating, or
    /// `None` when no soft budget is armed. A single reserve of at most this
    /// many bytes is guaranteed not to trip the soft trigger.
    #[inline]
    pub(crate) fn budget_headroom(&self) -> Option<u64> {
        self.budget_remaining
            .get()
            .map(|rem| rem.saturating_sub(self.in_flight_bytes.get()))
    }

    // ── the level boundary ─────────────────────────────────────────────────

    /// Enter a level whose emitted pairs are bounded by `pair_bound`.
    ///
    /// The bound picks this level's growth mode, with no counting walk in
    /// either: past [`DENSE_GROWTH_DECISION_THRESHOLD`], when the worst-case
    /// `Vec`-doubling transient of the level's pair arena (allocate the new
    /// block, copy, free the old — three times the arena, live at once) is not
    /// provably affordable, growth goes through bounded, headroom-aware
    /// increments instead. Below the threshold the level never pays the
    /// headroom read.
    ///
    /// `None` is a level whose caller offers no bound — a streaming target that
    /// truncates pairs per cell, or a route that never emits into the arena at
    /// all — which is plain doubling. Every level calls this exactly once, so a
    /// near-cap decision can never leak into the next one.
    ///
    /// `pair_bound` must be an upper bound that genuinely holds: the dense walk passes
    /// `|f.pairs| × |g.pairs|` (every product pair emits at most once), the
    /// clause conjunction its own per-level worst case.
    #[inline]
    pub(crate) fn begin_level(&self, pair_bound: Option<u128>) {
        let bounded = match pair_bound {
            Some(bound) if bound > DENSE_GROWTH_DECISION_THRESHOLD => {
                bound
                    .saturating_mul(3)
                    .saturating_mul(u128::from(PAIR_ELEM_BYTES))
                    >= u128::from(self.headroom())
            }
            _ => false,
        };
        self.bounded_growth.set(bounded);
    }

    /// Is this level growing under the bounded-increment mode?
    #[inline]
    pub(crate) fn bounded_growth(&self) -> bool {
        self.bounded_growth.get()
    }

    /// The per-level-boundary cut check: the stop axis, then the output-node
    /// cap, in that order — which is load-bearing.
    ///
    /// This is the only stop poll in the per-level orchestration. Without it a
    /// level wide enough to grind for minutes is a level the caller's stop
    /// cannot cut, because the finer polls sit inside the cell loops the
    /// orchestration wraps. `out_nodes` is the running sum of the output nodes
    /// every finished level built.
    #[inline]
    pub(crate) fn level_done(&self, out_nodes: u64) -> Result<(), OperationError> {
        if self.should_stop() {
            return Err(OperationError::Stopped);
        }
        if let Some(cap) = self.output_node_cap.get()
            && out_nodes > cap
        {
            return Err(OperationError::OutputCap);
        }
        Ok(())
    }

    /// Charge `delta` more slots of capacity in an output level's pair arena.
    ///
    /// The single writer of the output-pair meter. Capacity and not length:
    /// length is bumped by the emit walk's bare push, roughly a billion times
    /// per conjunction-heavy compile, and a store there is not affordable.
    /// Capacity changes only on a growth event, which is already cold, so the
    /// charge amortizes to nothing. The meter therefore reads high by at most
    /// the arena's doubling slack and never low, which is the direction a size
    /// floor can tolerate.
    #[inline]
    pub(crate) fn charge_output_pairs(&self, delta: usize) {
        if delta == 0 {
            return;
        }
        let delta = delta as u64;
        self.pairs_in_flight
            .set(self.pairs_in_flight.get().saturating_add(delta));
        self.pairs_level_charge
            .set(self.pairs_level_charge.get().saturating_add(delta));
    }

    /// Swap the level's charged capacity for the pairs it actually holds, so
    /// only the level in flight is ever an estimate and the arena's slack
    /// cannot accumulate over the thousands of levels one conjunction walks.
    ///
    /// The early-exit routes that skip the per-level tail never settle, so what
    /// they charged comes off at the next boundary instead: the meter reads low
    /// there, which is the direction a size floor tolerates.
    #[inline]
    pub(crate) fn level_settled(&self, exact_pairs: u64) {
        let charged = self.pairs_level_charge.replace(0);
        let total = self.pairs_in_flight.get().saturating_sub(charged);
        self.pairs_in_flight.set(total.saturating_add(exact_pairs));
    }

    // ── conjunction_progress_enabled ───────────────────────────────────────────────────────────

    /// Is anyone conjunction_progress_enabled?
    #[inline]
    pub(crate) fn watched(&self) -> bool {
        self.watched.get()
    }

    /// A conjunction beginning, over `levels` vtree levels. Clears whatever the
    /// last one left, so a watcher can tell two apart by the instant alone.
    pub(crate) fn merge_began(&self, levels: u32) {
        self.conjunction.set(Some(ConjunctionProgress {
            started_at: Instant::now(),
            level: 0,
            levels,
        }));
    }

    /// A conjunction reaching `level`. One store, no clock — the watcher reads
    /// the clock it was already reading.
    pub(crate) fn merge_reached(&self, level: u32) {
        if let Some(m) = self.conjunction.get() {
            self.conjunction.set(Some(ConjunctionProgress { level, ..m }));
        }
    }

    // ── the stop axis ──────────────────────────────────────────────────────

    /// Has the operation in flight reached something that stops it?
    ///
    /// The schedule is asked before the bounds, and the order is load-bearing: a
    /// schedule may conclude that the operation deserves the rest of the wall,
    /// and asking a stale, shorter bound first would cut an operation the
    /// schedule has already committed to.
    #[inline]
    pub(crate) fn should_stop(&self) -> bool {
        let stop = self.stop.get();
        let schedule = self.schedule.borrow().clone();
        if !stop.armed() && schedule.is_none() {
            return false;
        }
        let now = Instant::now();
        let stop = match schedule {
            Some(decide) => match decide.decide(&self.meters(), now) {
                StopDecision::Stop => return true,
                StopDecision::Continue => stop,
                StopDecision::ReplaceRules(next) => {
                    self.stop.set(next);
                    next
                }
            },
            None => stop,
        };
        // The size-conditional bound first: it is the cheaper half (the clock is
        // already read) and before it falls the pair meter does not matter.
        if let Some((floor_pairs, at)) = stop.after_pairs
            && self.reached(at, now)
            && self.pairs_in_flight.get() >= floor_pairs
        {
            return true;
        }
        stop.unconditional.is_some_and(|at| self.reached(at, now))
    }

    #[inline]
    fn reached(&self, at: StopAt, now: Instant) -> bool {
        match at {
            StopAt::Time(t) => now >= t,
            StopAt::WorkUnits(units) => self.work_clock.get() >= units,
        }
    }

    // ── host memory probes ─────────────────────────────────────────────────

    /// Pre-allocation release notice for a growth of `request_bytes`.
    #[inline(always)]
    pub(crate) fn preflight_alloc(&self, request_bytes: u64) {
        let mem = self.mem.borrow().clone();
        mem.preflight_alloc(request_bytes);
    }

    /// Once-per-operation eager-reclaim nudge.
    #[inline(always)]
    pub(crate) fn eager_reclaim(&self) {
        let mem = self.mem.borrow().clone();
        mem.eager_reclaim();
    }

    /// The installed address-space ceiling, answered once per install.
    fn address_space_limit(&self) -> Option<u64> {
        match self.vas_limit.get() {
            Some(v) => v,
            None => {
                let mem = self.mem.borrow().clone();
                let v = mem.address_space_limit();
                self.vas_limit.set(Some(v));
                v
            }
        }
    }

    // ── allocation ergonomics over `charge_bytes` ──────────────────────────

    /// Record an allocator refusal's request size. Cold: only ever reached on
    /// the error path of a fallible reserve.
    #[cold]
    #[inline(never)]
    fn note_refused(&self, bytes: u64) -> OperationError {
        self.refused_bytes.set(Some(bytes));
        OperationError::OverBudget
    }

    /// Tracked `try_reserve`/`try_reserve_exact`: preflight the host, map an
    /// allocator refusal to [`OperationError::OverBudget`], and charge the capacity
    /// delta.
    ///
    /// `EXACT` picks the `Vec` method and, with it, the size the preflight and
    /// the refusal report: exactly `additional` for the exact form, and the
    /// doubled estimate `capacity().max(additional)` for the doubling form,
    /// whose actual grab is up to twice the current capacity.
    ///
    /// An armed allocation-failure injection refuses here.
    #[inline(always)]
    fn reserve_impl<T, const EXACT: bool>(
        &self,
        v: &mut Vec<T>,
        additional: usize,
    ) -> Result<(), OperationError> {
        if self.refuses_reserve() {
            return Err(OperationError::OverBudget);
        }
        let pre_cap = v.capacity();
        let elem = std::mem::size_of::<T>() as u64;
        let grab = if EXACT { additional } else { v.capacity().max(additional) };
        let grab_bytes = (grab as u64).saturating_mul(elem);
        // Release notice only on actual growth: a zero-byte notice is the host's
        // entry heartbeat, throttled separately.
        if additional > v.capacity() - v.len() {
            self.preflight_alloc(grab_bytes);
        }
        let grown = if EXACT { v.try_reserve_exact(additional) } else { v.try_reserve(additional) };
        grown.map_err(|_| self.note_refused(grab_bytes))?;
        self.charge_bytes((v.capacity().saturating_sub(pre_cap) as u64).saturating_mul(elem))
    }

    /// Tracked `try_reserve_exact`. Preferred for known-size grows.
    #[inline(always)]
    pub(crate) fn reserve_exact<T>(&self, v: &mut Vec<T>, additional: usize) -> Result<(), OperationError> {
        self.reserve_impl::<T, true>(v, additional)
    }

    /// Tracked `try_reserve`, with `Vec`'s doubling growth. Use when the caller
    /// is genuinely amortizing many small pushes.
    #[inline(always)]
    pub(crate) fn reserve<T>(&self, v: &mut Vec<T>, additional: usize) -> Result<(), OperationError> {
        self.reserve_impl::<T, false>(v, additional)
    }

    /// Reserve hash-table entries, charging their capacity and control-byte estimate.
    pub(crate) fn reserve_map<K: Eq + std::hash::Hash, V, S: std::hash::BuildHasher>(
        &self, map: &mut std::collections::HashMap<K, V, S>, additional: usize,
    ) -> Result<(), OperationError> {
        if self.refuses_reserve() { return Err(OperationError::OverBudget); }
        let before = map.capacity();
        let bytes = (std::mem::size_of::<(K, V)>() + 1) as u64;
        let request = (additional as u64).saturating_mul(bytes);
        if additional > before - map.len() { self.preflight_alloc(request); }
        map.try_reserve(additional).map_err(|_| self.note_refused(request))?;
        self.charge_bytes((map.capacity().saturating_sub(before) as u64).saturating_mul(bytes))
    }

    /// Fallible `push`: reserve one slot before the push so allocation failure
    /// returns `Err(OverBudget)` instead of aborting the process.
    ///
    /// The `len < capacity` fast path stores the element and nothing else; the
    /// reserve-and-account body lives in [`Limits::push_grow`], `#[inline(never)]`,
    /// so the push loop keeps `len`, `capacity` and the base pointer in registers.
    #[inline(always)]
    pub(crate) fn try_push<T>(&self, v: &mut Vec<T>, x: T) -> Result<(), OperationError> {
        if v.len() < v.capacity() {
            v.push(x);
            return Ok(());
        }
        self.push_grow(v, x)
    }

    /// Growth arm of [`Limits::try_push`]. Reached only when `len == capacity`,
    /// once per doubling event, so the out-of-line call amortizes to nothing.
    #[cold]
    #[inline(never)]
    fn push_grow<T>(&self, v: &mut Vec<T>, x: T) -> Result<(), OperationError> {
        self.reserve(v, 1)?;
        v.push(x);
        Ok(())
    }

    /// Fallible analogue of `Vec::resize` for grow-only callers. No-op when
    /// `v.len() >= new_len`. Uses `try_reserve_exact` so address space is not
    /// over-reserved.
    #[inline]
    pub(crate) fn try_resize<T: Clone>(
        &self,
        v: &mut Vec<T>,
        new_len: usize,
        val: T,
    ) -> Result<(), OperationError> {
        if v.len() >= new_len {
            return Ok(());
        }
        let additional = new_len - v.len();
        self.reserve_exact(v, additional)?;
        v.resize(new_len, val);
        Ok(())
    }
}
