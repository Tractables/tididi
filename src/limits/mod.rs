//! Limits on the engine's work — deadlines, memory budgets, output caps,
//! stall ropes, host memory probes — and the meters they are checked against.
//!
//! [`ApplyError`] is the one error every fallible operation in this crate
//! returns; the allocation helpers here (`try_push`, `try_resize`,
//! `budget_reserve*`) are how the data model and the reduction passes grow
//! storage without aborting the process. The limits are per-thread state,
//! installed for a lexical scope by [`apply_limits`] and polled by the
//! engine's amortized tickers (`PollTicker`).

use std::cell::Cell;

mod alloc;
mod error;
pub(crate) mod install;
mod memory;
pub(crate) mod meters;
mod poll;
mod stop;

pub use error::ApplyError;
pub use install::{apply_limits, ApplyLimitsInstall};
pub use memory::MemPressure;
pub use meters::{
    apply_meters, charge_apply_in_flight_for_test, reset_apply_meters, set_apply_budget,
    ApplyMeters, MergePosition,
};
pub use stop::{RopeLimit, Scheduled};

pub(crate) use alloc::{
    budget_reserve, budget_reserve_exact, try_push, try_resize, unaccount_transient_bytes,
};
pub(crate) use memory::{apply_headroom_bytes_or_vas, mem_eager_reclaim, mem_preflight_alloc};
pub(crate) use poll::{reduce_poll_stride, PollTicker};
pub(crate) use stop::{any_stop_armed, apply_output_node_cap, charge_compile_work, deadline_expired};

#[cfg(test)]
pub(crate) use alloc::apply_budget_headroom_bytes;

#[cfg(test)]
pub(crate) use poll::with_reduce_poll_stride;

#[cfg(test)]
pub(crate) use memory::{vas_headroom_with_margin, SOFT_HEADROOM_MARGIN_BYTES};

#[cfg(test)]
#[path = "../limits_headroom_tests.rs"]
mod headroom_tests;

/// All apply-limit state for the thread, as ONE TLS struct-of-Cells (one TLS
/// address resolution + field offsets, instead of six TLS slots). Plain
/// `Cell`s, NOT `RefCell<struct>` — the byte-budget charge path (`try_push`
/// per emitted node) must stay a bare load/store.
pub(crate) struct ApplyLimits {
    /// Soft-budget remaining for the upcoming apply (the heap cap minus the
    /// caller's estimate of live bytes at step entry). `None` disables the
    /// predictive check; `try_reserve_exact` still catches OS-level OOM
    /// (e.g. under `ulimit -v`) regardless. A caller that re-derives its budget
    /// as it goes sets this before each step and clears it after; see
    /// `set_apply_budget` below.
    pub(crate) budget_remaining: Cell<Option<u64>>,

    /// Bytes allocated so far in the current `apply_and_fallible` call,
    /// summed across `try_reserve(_exact)` capacity deltas in tracked
    /// Vecs. Reset to 0 at apply entry; compared against
    /// `budget_remaining` after every grow. Lets the soft trigger
    /// fire *during* a conjoin without waiting for the OS allocator —
    /// the apply-internal transient peak (e.g. `node_idx` grids, output
    /// pair vectors) is what blows past the per-step estimate.
    pub(crate) budget_in_flight: Cell<u64>,

    /// Output pairs this apply has built so far: capacity slots claimed in an
    /// OUTPUT level's `pairs` arena, summed over the apply, reset at apply
    /// entry beside `budget_in_flight`. The one meter the amortized intra-level
    /// poll can read that is denominated in the same unit `Tdd::size()` counts
    /// and the give-up rule's floor is stated in.
    ///
    /// Why capacity and not length: length is bumped by the emit walk's bare
    /// `v.push` fast path, ~1 G times per apply-heavy compile, and a TLS store
    /// there is not affordable. Capacity changes only on a growth event, which
    /// is already cold and out-of-line, so the charge is amortized to nothing.
    /// It therefore reads high by at most the arena's doubling slack (< 2×) and
    /// never low, which is the direction a size FLOOR can tolerate.
    ///
    /// Charged in exactly the three places an output pair arena grows —
    /// [`push_pair_grow`] (dense emit walk), the per-level bulk pre-reserve in
    /// `conjoin::mod`, and the sparse builder's `try_push_internal_node` — all
    /// through [`account_output_pairs`], which is the only writer. Transient
    /// `Vec<InputPair>` tables (the hoisted C2 column arena) are NOT output and
    /// are not charged here, though `budget_in_flight` does charge their bytes.
    ///
    /// Capacity is only the IN-FLIGHT level's estimate: at each level boundary
    /// [`settle_output_pairs`] swaps that level's charge for its exact
    /// `pairs.len()`, so a finished level contributes the truth and the arena's
    /// slack cannot accumulate across the thousands of levels one apply walks.
    /// (The bulk seed is up to `LEVEL_RESERVE_PAIRS_CAP` pairs a low-survival
    /// level may never use; left unsettled it would have made the meter grow
    /// with the LEVEL COUNT rather than the diagram — the same failure, in a
    /// new unit, that re-metering was fixing.)
    pub(crate) pairs_in_flight: Cell<u64>,

    /// Capacity charged into `pairs_in_flight` for the level currently being
    /// built — what [`settle_output_pairs`] takes back off the meter when that
    /// level finishes and its exact size is known. Zero between levels.
    ///
    /// The three early-exit routes that skip the per-level tail never settle,
    /// so whatever they charged is taken off at the NEXT boundary instead: the
    /// meter reads low there, which is the direction a size floor tolerates.
    pub(crate) pairs_level_charge: Cell<u64>,

    /// Optional wallclock deadline for the operation in flight. Setting it is
    /// what arms the cut: the vtree-level loop and the amortized dense/sparse
    /// cell-loop polls ([`deadline_expired`]) return `Err(ApplyError::Deadline)`
    /// once it passes. Installed via `apply_limits().deadline(..).apply()`.
    pub(crate) deadline: Cell<Option<std::time::Instant>>,

    /// Optional decision callback over the compile in flight, installed by
    /// [`ApplyLimitsInstall::schedule`] and asked beside the deadline by
    /// [`limits_reached`], which hands it the clock reading it has already
    /// taken. The deadline says when the compile must STOP; this says that it
    /// must be ASKED. `None` (default) is every compile nobody scheduled.
    pub(crate) schedule: Cell<Option<fn(std::time::Instant) -> Scheduled>>,

    /// Optional cap on the cumulative *output* node count of the apply currently
    /// in flight. When `Some(cap)`, `apply_and_fallible`'s vtree-level loop sums
    /// the output nodes built so far across all finalized levels and returns
    /// `Err(ApplyError::OutputCap)` the moment they exceed `cap`. This bounds
    /// the RESULT of a conjunction — a product whose output balloons far past the
    /// segment's intended size is doomed and not worth grinding to the wall
    /// deadline or the much looser byte budget (4 GiB ≈ 128 M nodes). `None`
    /// (default) = no node cap; install/restore with a set→clear pair like
    /// `deadline`. Read amortized at each vtree level via
    /// `apply_output_node_cap()`.
    pub(crate) output_node_cap: Cell<Option<u64>>,

    /// Cumulative work the applies on this thread have polled through, in the
    /// stride units the amortized polls already count ([`PollTicker::poll`], the
    /// `nxm_deadline_check!` cadence). Monotone, and deliberately NOT in
    /// [`reset_meters`]: it is a process-lifetime CLOCK, not a per-apply meter,
    /// so a caller scopes it by MARK-AND-SUBTRACT — read it at the door, read it
    /// again later, take the difference — exactly as the deterministic front
    /// end's own work meter is scoped. Resetting it would make a mark taken
    /// before an apply boundary read as a negative interval.
    ///
    /// It exists because the give-up rule needs a work signal and
    /// `pairs_in_flight` is not one: that meter is a SIZE, subtracted back at
    /// every level boundary and zeroed at every apply entry, and a quarter of the
    /// give-up rule's real cuts were on steps that had built no output pairs at
    /// all.
    pub(crate) work_clock: Cell<u64>,

    /// Optional `(floor_pairs, rope)` pair for a step the give-up rule declined
    /// to rope at the door and will rope if it OUTGROWS that decision: past
    /// `rope`, an apply whose `pairs_in_flight` has reached `floor_pairs` stops
    /// with `Err(ApplyError::Deadline)`. Under the floor the rope is not
    /// consulted at all, so the apply runs under whatever deadline the enclosing
    /// scopes installed.
    ///
    /// A floor of ZERO is the rope for a step already eligible at the door —
    /// `pairs_in_flight >= 0` before anything is built — which is how BOTH halves
    /// of the rule ride this one axis instead of the held half borrowing the
    /// deadline cell it shares with the program wall.
    ///
    /// The meter is [`ApplyLimits::pairs_in_flight`] — output pairs, the unit
    /// the caller's input floor is already stated in, so both ways of meeting
    /// one floor are measured in one currency. It is deliberately NOT
    /// `budget_in_flight`: that counts every byte the apply charged (nodes,
    /// ext, grids, live scratch, hoisted transients), so an 8 MB floor was met
    /// by a step that had built no diagram at all.
    ///
    /// This is deliberately NOT a second output cap: the cap says "this output
    /// is too big", while this says "this is a big diagram, and the step
    /// building it has spent long enough". The caller is the compile's
    /// progress-based give-up rule in the downstream driver,
    /// whose floor is ONE number in pairs; a step meets it with the diagram
    /// it was HANDED or with the one it has BUILT, and this cell carries both. `None` (default) —
    /// nothing armed, which is every apply the give-up rule is not running over.
    /// Install/restore with a set→clear pair like `deadline`.
    pub(crate) stall_rope: Cell<Option<(u64, RopeLimit)>>,

    /// Emit-growth mode for the dense level currently being built: `false`
    /// (default) ⇒ plain `Vec`-doubling growth in `try_push_pair_into`;
    /// `true` ⇒ near-cap level whose doubling transient is not provably
    /// affordable — growth goes through `grow_pairs_bounded` (bounded,
    /// headroom-aware increments). Disarmed exactly once per level:
    /// `decide_emit_growth_mode` disarms on entry (then arms on its near-cap
    /// decision) for the routes that call it, and the cell-build loop disarms
    /// for the routes that skip it. Apply entry (`reset_meters`) clears it
    /// again — so it can never leak across levels or applies.
    pub(crate) pairs_bounded_growth: Cell<bool>,

    /// Whether anyone is WATCHING the applies on this thread — set by the caller
    /// that wants to know where a long one has got to, and read by the apply
    /// before it pays for publishing anything ([`MergePosition`]).
    pub(crate) merges_watched: Cell<bool>,

    /// Where the apply in flight has got to: when it BEGAN, the vtree level it is
    /// standing on, and how many that apply has in all. Published by the apply
    /// itself — one store per level, no clock — and read by the watcher on the
    /// polls it is already making.
    ///
    /// This crate does not interpret it: the same division of labour as
    /// [`ApplyLimitsInstall::schedule`], where what this crate provides is the
    /// place to stand inside
    /// an operation and the caller provides everything else. `None` outside a
    /// watched apply.
    pub(crate) merge: Cell<Option<MergePosition>>,

    /// The host's memory probes ([`MemPressure`]); `MemPressure::NONE` until a
    /// scope installs one.
    pub(crate) mem_pressure: Cell<MemPressure>,

    /// `address_space_limit` answered once per install: the limit is stable for
    /// the life of an install and the emit-growth machinery asks per huge
    /// level. Cleared whenever the probes change.
    pub(crate) address_space_limit_cache: Cell<Option<Option<u64>>>,
}

thread_local! {
    pub(crate) static APPLY_LIMITS: ApplyLimits = const {
        ApplyLimits {
            budget_remaining: Cell::new(None),
            budget_in_flight: Cell::new(0),
            pairs_in_flight: Cell::new(0),
            pairs_level_charge: Cell::new(0),
            work_clock: Cell::new(0),
            deadline: Cell::new(None),
            schedule: Cell::new(None),
            output_node_cap: Cell::new(None),
            stall_rope: Cell::new(None),
            pairs_bounded_growth: Cell::new(false),
            merges_watched: Cell::new(false),
            merge: Cell::new(None),
            mem_pressure: Cell::new(MemPressure::NONE),
            address_space_limit_cache: Cell::new(None),
        }
    };
}

// Bytes asked for by the most recent fallible reserve the allocator REFUSED
// (`Vec::try_reserve*` → `TryReserveError`), or `None` if no reserve has been
// refused on this thread.
//
// Exists because "the allocator said no" and "the soft budget said no" arrive at
// the driver as the same `ApplyError::OverBudget`, and the two demand opposite
// responses: a refused 300 GB grid is a size the compile can never have on any
// machine, while a refused 20 GB one is a machine that is currently full. Without
// the size the driver could only report "unknown bytes" (which it did, for exactly
// as long as this cell did not exist) and every instant `OverBudget` would look
// like memory pressure. Written on the cold error path only; read by a caller
// reporting why it stopped.
thread_local! {
    static LAST_REFUSED_RESERVE_BYTES: Cell<Option<u64>> = const { Cell::new(None) };
}
