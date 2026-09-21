//! The memory half of the limits: how much room is left, which growth mode a
//! level runs in, and the allocation helpers that charge against the budget.
//!
//! The allocation helpers are `#[inline(always)]` because they are called once
//! per reserve on paths that reserve constantly; the same measurement that
//! kept the attribute on `CountVec::try_with_capacity` puts `reserve_impl`
//! among the handful of functions whose instruction count moves when the
//! forcing goes away.

use crate::limits::OperationError;

use crate::limits::memory::{VAS_UNLIMITED_HEADROOM, vas_headroom_with_margin};

use super::Limits;

/// Emitted-pair bound above which [`Limits::begin_level`] runs the growth-mode
/// decision at all; a level bounded below it doubles without a headroom read.
pub(crate) const DENSE_GROWTH_DECISION_THRESHOLD: u128 = 128 * 1024 * 1024;

/// Bytes per pair in the level arena, used to estimate allocation growth.
pub(crate) const PAIR_ELEM_BYTES: u64 = std::mem::size_of::<crate::diagram::ChildPair>() as u64;

impl Limits {
    /// Available bytes under the soft budget, or the host's address-space
    /// ceiling minus its mapped bytes and a safety margin. An unlimited host
    /// ceiling yields [`VAS_UNLIMITED_HEADROOM`].
    #[inline]
    pub(crate) fn headroom(&self) -> u64 {
        if let Some(h) = self.budget_headroom() {
            return h;
        }
        match self.address_space_limit() {
            Some(limit) => {
                let mem = self.memory_hooks.borrow().clone();
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
        self.budget
            .get()
            .map(|rem| rem.saturating_sub(self.in_flight_bytes.get()))
    }


    /// Enter a level whose emitted pairs are bounded by `pair_bound`.
    ///
    /// The bound picks this level's growth mode, with no counting walk in
    /// either: past [`DENSE_GROWTH_DECISION_THRESHOLD`], when the worst-case
    /// `Vec`-doubling transient of the level's pair arena (allocate the new
    /// block, copy, then free the old: three times the arena, live at once) is not
    /// provably affordable, growth goes through bounded, headroom-aware
    /// increments instead. Below the threshold the level never pays the
    /// headroom read.
    ///
    /// `None` is a level whose caller offers no bound: a streaming target that
    /// truncates pairs per cell, or a route that never emits into the arena at
    /// all. Such levels use plain doubling. Every level calls this exactly once, so a
    /// near-cap decision can never leak into the next one.
    ///
    /// `pair_bound` must be an upper bound for the actual output: the dense walk passes
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

    /// Check cancellation before the output-node cap at a level boundary.
    /// `out_nodes` counts the nodes emitted by all completed levels.
    #[inline]
    pub(crate) fn level_done(&self, out_nodes: u64) -> Result<(), OperationError> {
        self.check_stop()?;
        self.check_output_cap(out_nodes)
    }

    /// The most node slots one level may hold, which is what a `NodeIdx` can
    /// address. A level that has filled it answers
    /// [`OperationError::IndexOverflow`], which no budget makes succeed.
    #[inline]
    pub(crate) fn level_width_cap(&self) -> usize {
        #[cfg(test)]
        if let Some(cap) = self.width_cap_pin.get() { return cap; }
        crate::diagram::NodeIdx::MAX_LIVE
    }

    /// Refuse an emitted-node total above the installed cap.
    #[inline]
    pub(crate) fn check_output_cap(&self, out_nodes: u64) -> Result<(), OperationError> {
        if let Some(cap) = self.output_node_cap.get()
            && out_nodes > cap
        {
            return Err(OperationError::OutputCap);
        }
        Ok(())
    }


    /// Pre-allocation release notice for a growth of `request_bytes`.
    #[inline(always)]
    pub(crate) fn preflight_alloc(&self, request_bytes: u64) {
        let mem = self.memory_hooks.borrow().clone();
        mem.preflight_alloc(request_bytes);
    }

    /// Once-per-operation eager-reclaim nudge.
    #[inline(always)]
    pub(crate) fn eager_reclaim(&self) {
        let mem = self.memory_hooks.borrow().clone();
        mem.eager_reclaim();
    }

    /// The installed address-space ceiling, answered once per install.
    fn address_space_limit(&self) -> Option<u64> {
        match self.vas_limit.get() {
            Some(v) => v,
            None => {
                let mem = self.memory_hooks.borrow().clone();
                let v = mem.address_space_limit();
                self.vas_limit.set(Some(v));
                v
            }
        }
    }


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
    #[inline(always)]
    fn reserve_impl<T, const EXACT: bool>(
        &self,
        v: &mut Vec<T>,
        additional: usize,
    ) -> Result<(), OperationError> {
        #[cfg(test)]
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
    /// is amortizing many small pushes.
    #[inline(always)]
    pub(crate) fn reserve<T>(&self, v: &mut Vec<T>, additional: usize) -> Result<(), OperationError> {
        self.reserve_impl::<T, false>(v, additional)
    }

    /// Reserve hash-table entries, charging their capacity and control-byte estimate.
    pub(crate) fn reserve_map<K: Eq + std::hash::Hash, V, S: std::hash::BuildHasher>(
        &self, map: &mut std::collections::HashMap<K, V, S>, additional: usize,
    ) -> Result<(), OperationError> {
        #[cfg(test)]
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

/// A buffer whose capacity was charged to a [`Limits`] as it grew.
pub(crate) trait Charged {
    /// The bytes the meter holds for this buffer: its capacity, not its length.
    fn charged_bytes(&self) -> u64;
}

impl<K, V, S> Charged for std::collections::HashMap<K, V, S> {
    fn charged_bytes(&self) -> u64 {
        (self.capacity() as u64).saturating_mul((std::mem::size_of::<(K, V)>() + 1) as u64)
    }
}

impl<T> Charged for Vec<T> {
    #[inline]
    fn charged_bytes(&self) -> u64 {
        (self.capacity() as u64).saturating_mul(std::mem::size_of::<T>() as u64)
    }
}

impl Limits {
    /// Drop a charged buffer and hand its charge back.
    #[inline]
    pub(crate) fn discard<B: Charged>(&self, buf: B) {
        self.release_bytes(buf.charged_bytes());
        drop(buf);
    }
}

/// A charged buffer that is [discarded](Limits::discard) when dropped, unless
/// it is [kept](Transient::keep). Wrap a buffer that is built and then either
/// installed or thrown away, so every early exit hands its charge back.
pub(crate) struct Transient<'a, B: Charged> {
    lim: &'a Limits,
    buf: Option<B>,
}

impl<'a, B: Charged> Transient<'a, B> {
    pub(crate) fn new(lim: &'a Limits, buf: B) -> Self {
        Transient { lim, buf: Some(buf) }
    }

    /// Take the buffer out; its charge stays with the caller.
    pub(crate) fn keep(mut self) -> B {
        self.buf.take().expect("a transient holds its buffer until it is kept")
    }
}

impl<B: Charged> std::ops::Deref for Transient<'_, B> {
    type Target = B;
    #[inline]
    fn deref(&self) -> &B {
        self.buf.as_ref().expect("a transient holds its buffer until it is kept")
    }
}

impl<B: Charged> std::ops::DerefMut for Transient<'_, B> {
    #[inline]
    fn deref_mut(&mut self) -> &mut B {
        self.buf.as_mut().expect("a transient holds its buffer until it is kept")
    }
}

impl<B: Charged> Drop for Transient<'_, B> {
    fn drop(&mut self) {
        if let Some(buf) = self.buf.take() {
            self.lim.discard(buf);
        }
    }
}
