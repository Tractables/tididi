//! The scoped-install builder and the guard that restores the prior limits.

use super::memory::MemPressure;
use super::stop::{RopeLimit, Scheduled};
use super::APPLY_LIMITS;

/// Builder for installing apply limits (deadline / decision callback / budget /
/// output-node cap / stall rope / memory probes) for a lexical scope. Axes not named (the method never called) are
/// left completely untouched — no snapshot taken, nothing restored. `apply()`
/// snapshots the prior value of each NAMED axis and returns an RAII guard
/// whose Drop restores exactly those axes — panic-safe by construction: a
/// panic inside the wrapped scope, caught by the recovery cascade's
/// `catch_unwind`, must not leave a stale limit armed on the thread (a later
/// fallback compile meant to run unbudgeted could otherwise spuriously trip).
/// Passing `None` to a named axis INSTALLS `None` for it (e.g. `.deadline(None)`
/// = shield semantics: clears the deadline for the scope, restoring the outer
/// one on drop) — this is the axis-touched-with-value-None case, distinct from
/// never calling the method at all.
#[derive(Default)]
#[must_use = "call .apply() to install the limits — dropping the builder installs nothing"]
pub struct ApplyLimitsInstall {
    deadline: Option<Option<std::time::Instant>>,
    schedule: Option<Option<fn(std::time::Instant) -> Scheduled>>,
    budget: Option<Option<u64>>,
    output_cap: Option<Option<u64>>,
    stall_rope: Option<Option<(u64, RopeLimit)>>,
    mem_pressure: Option<MemPressure>,
    watch: Option<bool>,
}

/// Start building a scoped apply-limits install. See [`ApplyLimitsInstall`].
#[inline]
pub fn apply_limits() -> ApplyLimitsInstall {
    ApplyLimitsInstall::default()
}

impl ApplyLimitsInstall {
    /// Set the wall-clock deadline after which apply bails (`None` = no deadline).
    #[inline]
    pub fn deadline(mut self, d: Option<std::time::Instant>) -> Self {
        self.deadline = Some(d);
        self
    }
    /// Arm a decision callback the in-operation polls ask (`None` = nothing
    /// asked). It is handed the clock reading the poll has already taken, so it
    /// need not read the clock again, and its answer ([`Scheduled`]) either lets
    /// the operation carry on, cuts it, or replaces the deadline it runs under.
    ///
    /// The callback is asked on EVERY poll: this crate holds no view on when a
    /// decision is due, so a caller with decision points of its own tests them
    /// itself and answers [`Scheduled::Carry`] until one arrives. What the poll
    /// provides is the only thing the caller cannot — a place to stand INSIDE an
    /// operation, on a poll the operation was already paying for.
    #[inline]
    pub fn schedule(mut self, s: Option<fn(std::time::Instant) -> Scheduled>) -> Self {
        self.schedule = Some(s);
        self
    }
    /// Set the soft allocation budget, in bytes, that one apply may grow its
    /// scratch by before it bails with [`ApplyError::OverBudget`] (`None` = no
    /// budget). Same axis as [`set_apply_budget`], but scoped to the guard
    /// returned by [`apply`](Self::apply) rather than installed open-endedly.
    /// Installing `None` explicitly claims the axis for this scope: the
    /// downstream compiler's bottom-up traversal does exactly that to own the
    /// lifetime of the budget its own per-merge refresh writes.
    #[inline]
    pub fn budget(mut self, b: Option<u64>) -> Self {
        self.budget = Some(b);
        self
    }
    /// Cap how many output nodes one apply may produce before it bails with
    /// [`ApplyError::OutputCap`] (`None` = uncapped). A deliberate size cut
    /// rather than a memory guard, so it is reported as its own error variant;
    /// handlers that do not care about the distinction treat `OutputCap`
    /// exactly like `OverBudget`.
    #[inline]
    pub fn output_cap(mut self, c: Option<u64>) -> Self {
        self.output_cap = Some(c);
        self
    }
    /// Arm a rope that comes into force once the apply has built at least
    /// `floor_pairs` output pairs (`None` = no rope; a floor of `0` = in force
    /// from the door). Cuts with [`ApplyError::Deadline`], because it IS a
    /// deadline — one whose caller has made its eligibility conditional on what
    /// the step BUILDS rather than on the size of the operands it was handed, and
    /// whose fall is either an instant or a count on the compile work clock
    /// ([`RopeLimit`]). The `u64` is a floor in output PAIRS, not bytes. See
    /// the `stall_rope` limit, the read side of this knob.
    #[inline]
    pub fn stall_rope(mut self, r: Option<(u64, RopeLimit)>) -> Self {
        self.stall_rope = Some(r);
        self
    }
    /// Install the host's memory probes for the scope ([`MemPressure::NONE`]
    /// = no host, plain doubling growth). A host installs this once, around
    /// everything it compiles.
    #[inline]
    pub fn mem_pressure(mut self, m: MemPressure) -> Self {
        self.mem_pressure = Some(m);
        self
    }

    /// Watch the applies in the scope: each publishes where it stands, one
    /// store per level, for [`apply_meters`] to read as
    /// [`ApplyMeters::merge`]. An unwatched apply pays one `Cell` load and
    /// nothing else.
    #[inline]
    pub fn watch(mut self, on: bool) -> Self {
        self.watch = Some(on);
        self
    }

    /// Install every named axis (snapshotting its prior value) and return the
    /// RAII guard that restores them on drop.
    #[must_use = "the guard restores the prior apply limits when dropped; bind it to a name"]
    #[inline]
    pub fn apply(self) -> ApplyLimitsGuard {
        APPLY_LIMITS.with(|l| ApplyLimitsGuard {
            deadline: self.deadline.map(|d| l.deadline.replace(d)),
            schedule: self.schedule.map(|s| l.schedule.replace(s)),
            // Touches `budget_remaining` only — NOT `budget_in_flight`, which
            // is reset separately, at apply entry.
            budget: self.budget.map(|b| l.budget_remaining.replace(b)),
            output_cap: self.output_cap.map(|c| l.output_node_cap.replace(c)),
            stall_rope: self.stall_rope.map(|r| l.stall_rope.replace(r)),
            mem_pressure: self.mem_pressure.map(|m| {
                l.address_space_limit_cache.set(None);
                l.mem_pressure.replace(m)
            }),
            watch: self.watch.map(|w| l.merges_watched.replace(w)),
        })
    }
}

/// RAII guard returned by [`ApplyLimitsInstall::apply`]. Restores exactly the
/// axes that were named on the builder (the others are `None` here and are
/// left alone) — panic-safe by construction, including on an unwind caught by
/// the recovery cascade's `catch_unwind`.
#[must_use = "the guard restores the prior apply limits when dropped; bind it to a name"]
pub struct ApplyLimitsGuard {
    deadline: Option<Option<std::time::Instant>>,
    schedule: Option<Option<fn(std::time::Instant) -> Scheduled>>,
    budget: Option<Option<u64>>,
    output_cap: Option<Option<u64>>,
    stall_rope: Option<Option<(u64, RopeLimit)>>,
    mem_pressure: Option<MemPressure>,
    watch: Option<bool>,
}

impl Drop for ApplyLimitsGuard {
    #[inline]
    fn drop(&mut self) {
        APPLY_LIMITS.with(|l| {
            if let Some(prior) = self.deadline {
                l.deadline.set(prior);
            }
            if let Some(prior) = self.schedule {
                l.schedule.set(prior);
            }
            if let Some(prior) = self.budget {
                l.budget_remaining.set(prior);
            }
            if let Some(prior) = self.output_cap {
                l.output_node_cap.set(prior);
            }
            if let Some(prior) = self.stall_rope {
                l.stall_rope.set(prior);
            }
            if let Some(prior) = self.mem_pressure {
                l.address_space_limit_cache.set(None);
                l.mem_pressure.set(prior);
            }
            if let Some(prior) = self.watch {
                l.merges_watched.set(prior);
            }
        });
    }
}
