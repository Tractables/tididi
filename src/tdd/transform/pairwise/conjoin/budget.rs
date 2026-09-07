//! Budget, error, and flag primitives for the apply algorithm.
//!
//! Houses the soft-budget tracking (`MC_BUDGET_*`), fallible allocation
//! helpers (`budget_reserve_exact`, `try_push`, …), the deadline
//! thread-local (`APPLY_DEADLINE`), and all the cached env-flag readers
//! used throughout `apply_inner`.

use std::cell::Cell;
use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;

use crate::vtree::VarId;

/// Programmatic override for the apply-deadline check, set by callers that need
/// mid-apply deadline cutting WITHOUT the `TIDIDI_APPLY_DEADLINE_CHECK` env var
/// (e.g. the projected reactive single-shot, which must cut a giant un-yielding
/// conjoin on a bad vtree before its wall budget). Checked before the env gate
/// in `apply_deadline_check_enabled`; once set it stays on for the process. A
/// plain `AtomicBool` (not a `OnceLock`) so it can be flipped after first read.
pub(crate) static APPLY_DEADLINE_CHECK_OVERRIDE: AtomicBool = AtomicBool::new(false);

/// Force the apply-deadline check ON for the rest of the process (see
/// `APPLY_DEADLINE_CHECK_OVERRIDE`). Idempotent.
#[doc(hidden)]
pub fn enable_apply_deadline_check() {
    APPLY_DEADLINE_CHECK_OVERRIDE.store(true, Ordering::Relaxed);
}

/// Test-only: clear the sticky override so a regression test can assert the
/// arming transition deterministically (the flag is process-global, so a prior
/// test may have set it). Production has no disable path by design. `pub` (not
/// `#[cfg(test)]`) so downstream crates' tests can reach it across the crate
/// boundary (dependency crates are never compiled with `cfg(test)`).
#[doc(hidden)]
pub fn reset_apply_deadline_check_for_test() {
    APPLY_DEADLINE_CHECK_OVERRIDE.store(false, Ordering::Relaxed);
}

/// Programmatic arming for the deadline poll in the walks that run BETWEEN two
/// applies of one bottom-up step and, unlike the applies themselves, had no
/// preemption point of their own: the twin contraction
/// (`minimize::contract::contract_all_twins_topdown`), the ∃-forget batch
/// (`transform::unary::marginalize::marginalize_batch`) and the mid-compile
/// clustering rotation pass
/// (`restructure::search::cluster_marginal_rotations_in_subtree`).
///
/// ONE cell for all three because they are armed by the same thing for the same
/// reason — a canopy leaf's grant — and splitting it would make
/// `TIDIDI_CANOPY_DEADLINE=apply` mean three different amounts of coverage
/// depending on which cell a site happened to read.
///
/// A separate cell from [`APPLY_DEADLINE_CHECK_OVERRIDE`] because the two are
/// armed by different things and must stay independent: the apply gate is on for
/// every `--mc` run (the compile driver's give-up rule arms it), while this one is
/// armed per leaf compile by the DPLL-canopy driver alone
/// (`TIDIDI_CANOPY_DEADLINE=apply`). Folding them together would change the cut
/// behaviour of every `--mc` compile in the program, which the knob exists
/// precisely not to do. Same `AtomicBool`-not-`OnceLock` reason as above.
pub(crate) static REDUCE_DEADLINE_CHECK: AtomicBool = AtomicBool::new(false);

/// Force the reduce-deadline poll ON for the rest of the process (see
/// [`REDUCE_DEADLINE_CHECK`]). Idempotent.
#[doc(hidden)]
pub fn enable_reduce_deadline_check() {
    REDUCE_DEADLINE_CHECK.store(true, Ordering::Relaxed);
}

/// Test-only counterpart of [`reset_apply_deadline_check_for_test`], `pub` for
/// the same reason: the flag is process-global, so a test that asserts the
/// disarmed path has to be able to put it back.
#[doc(hidden)]
pub fn reset_reduce_deadline_check_for_test() {
    REDUCE_DEADLINE_CHECK.store(false, Ordering::Relaxed);
}

/// `true` iff the reduce poll is armed AND the compile has reached a limit —
/// the installed `APPLY_LIMITS.deadline`, or a decision point of the armed
/// [`Schedule`] that concluded the compile should stop.
///
/// Reads the SAME cells the apply's own poll reads — the deadline the streaming
/// compile installs from its caller's budget, and the schedule its caller armed
/// over it — so a leaf's grant is enforced by one number wherever the compile
/// happens to be standing. Cheap when off: a relaxed load that reads `false`
/// short-circuits before any TLS read or `Instant::now()`.
#[inline]
pub(crate) fn reduce_deadline_expired() -> bool {
    limits_reached(REDUCE_DEADLINE_CHECK.load(Ordering::Relaxed))
}

/// What a [`Schedule`] concludes when the compile reaches one of its decision
/// points.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
#[doc(hidden)]
pub enum Scheduled {
    /// Carry on. The compile is never interrupted and never re-pays anything —
    /// the decision cost it one poll it was making anyway.
    Carry,
    /// Stop here. Surfaces to the caller as [`ApplyError::Deadline`], which is
    /// the unwind path a mid-operation cut already has.
    Stop,
    /// Carry on, under this deadline from here on — a commitment, which replaces
    /// whatever deadline the compile was running under.
    Until(std::time::Instant),
}

/// A schedule of decision points armed over the compile running on this thread.
///
/// The caller that arms one owns the rule and all its state; nothing here
/// interprets anything. What this crate provides is the only thing the caller
/// cannot: a place to stand INSIDE an operation, on the poll the operation was
/// already paying for.
#[derive(Copy, Clone)]
#[doc(hidden)]
pub struct Schedule {
    /// The next decision point. Consulted only once it has passed, so a schedule
    /// whose next point is still ahead costs one load and one compare on top of
    /// the deadline check standing beside it.
    pub at: std::time::Instant,
    /// What that point concludes. Called at most once per poll, and only past
    /// `at` — the caller is expected to move `at` on (via [`arm_schedule`]) as
    /// part of deciding, or to leave it in the past deliberately.
    pub decide: fn() -> Scheduled,
}

/// Arm (or clear, with `None`) the schedule the in-operation polls consult,
/// returning the prior one so the caller can restore it.
///
/// Unscoped on purpose, like the deadline's own arm/refresh sites: the schedule
/// outlives the individual applies it is meant to interrupt, and the caller that
/// armed it is the one that knows when the compile it belongs to is over.
#[doc(hidden)]
pub fn arm_schedule(schedule: Option<Schedule>) -> Option<Schedule> {
    APPLY_LIMITS.with(|l| l.schedule.replace(schedule))
}

/// Compile work polled through on this thread so far (the apply limits' work clock).
///
/// Monotone and never reset, so an interval is a subtraction between two reads.
/// This crate takes no view on what a caller does with it: what it provides is a
/// count of the work the applies actually did, in a unit that advances whether or
/// not the step is producing output — the same division of labour as [`Schedule`].
#[inline]
#[doc(hidden)]
pub fn compile_work_units() -> u64 {
    APPLY_LIMITS.with(|l| l.work_clock.get())
}

/// Add `units` to the compile work clock.
#[inline]
pub(super) fn charge_compile_work(units: u64) {
    APPLY_LIMITS.with(|l| l.work_clock.set(l.work_clock.get().saturating_add(units)));
}

/// When a stall rope falls — the same rule in whichever currency the run is
/// budgeted in.
///
/// The give-up rule prices a step against the wall it began with, and a wall is
/// what a loaded box makes unreproducible: two runs of one formula reach
/// different steps before the same fraction is gone. `Work` prices it against
/// the compile's own work clock instead, so the cut lands at the same PLACE in
/// the compile on every box. Nothing else about the rule changes, which is why
/// this is one enum on one axis rather than a second rope.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopeLimit {
    /// The rope falls at this instant — the wall-clock rule as it shipped.
    Wall(std::time::Instant),
    /// The rope falls once [`compile_work_units`] reaches this many units.
    Work(u64),
}

impl RopeLimit {
    /// The instant this rope falls at, and `None` for a work-shaped one — for the
    /// callers that can only plan on the clock (the compile's ladder), which must
    /// see nothing rather than a converted guess.
    pub fn wall(self) -> Option<std::time::Instant> {
        match self {
            RopeLimit::Wall(at) => Some(at),
            RopeLimit::Work(_) => None,
        }
    }
}

/// The shared body of the two poll gates: has the compile reached something that
/// stops it?
///
/// `armed` is the caller's own gate cell, already loaded, so a disarmed poll
/// short-circuits before touching TLS or the clock. Past that the cost is one
/// TLS resolution, two `Cell` loads and — only if either is armed — one
/// `Instant::now()` and two compares.
///
/// The SCHEDULE is asked before the deadline, and the order is load-bearing: a
/// schedule's decision point may CONCLUDE that the compile deserves the rest of
/// the wall ([`Scheduled::Until`]), and asking the deadline first would let a
/// stale, shorter deadline — one an inner scope restored underneath the
/// commitment — cut a compile the schedule has already committed to.
#[inline]
fn limits_reached(armed: bool) -> bool {
    if !armed {
        return false;
    }
    let (deadline, schedule, stall) =
        APPLY_LIMITS.with(|l| (l.deadline.get(), l.schedule.get(), l.stall_rope.get()));
    if deadline.is_none() && schedule.is_none() && stall.is_none() {
        return false;
    }
    let now = std::time::Instant::now();
    if let Some(schedule) = schedule
        && now >= schedule.at
    {
        match (schedule.decide)() {
            Scheduled::Stop => return true,
            Scheduled::Carry => {}
            Scheduled::Until(wall) => {
                APPLY_LIMITS.with(|l| l.deadline.set(Some(wall)));
                return now >= wall;
            }
        }
    }
    // The stall rope ([`ApplyLimits::stall_rope`]): a cut whose caller made it
    // conditional on the apply having BUILT enough output PAIRS to be priced as
    // diagram growth — the same unit the caller's input floor is stated in — and
    // a floor of zero for the step that was already eligible at the door. It is
    // asked here, beside the deadline, because here is the only place inside a
    // level anything gets asked — an apply that never finishes a level is an
    // apply the level-boundary checks never reach, which is exactly the shape it
    // exists for. The rope is tested first: it is the cheaper half (the clock is
    // already read, the work clock is one TLS load), and before the rope falls
    // the meter does not matter.
    if let Some((floor_pairs, rope)) = stall
        && match rope {
            RopeLimit::Wall(at) => now >= at,
            RopeLimit::Work(at) => compile_work_units() >= at,
        }
        && APPLY_LIMITS.with(|l| l.pairs_in_flight.get()) >= floor_pairs
    {
        return true;
    }
    deadline.is_some_and(|deadline| now >= deadline)
}

/// Sentinel for dead product cells: c1[i] ∧ c2[j] = ⊥ (no output node created).
///
/// Same bit pattern as `ZERO` in types.rs but semantically distinct:
/// - `ZERO` marks a TDD whose output is UNSAT (top-level concept)
/// - `DEAD` marks a single product grid cell that produced no live pairs (local to apply)
pub(crate) const DEAD: u32 = u32::MAX;

/// Error returned by the fallible apply chain.
///
/// Allows the vsplit driver to catch over-budget conditions inside
/// `apply_and` (where a single product-grid resize can request many GiB)
/// and case-split before the allocator OOM-aborts the process.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApplyError {
    /// The OS allocator refused, or the soft apply budget
    /// would be exceeded by a hot scratch resize. Catchable by the
    /// vsplit driver; infallible wrappers panic.
    OverBudget,
    /// The wallclock deadline set via `APPLY_DEADLINE` expired during the
    /// vtree-level iteration or an amortized cell-loop poll. The per-iteration
    /// check is gated by `apply_deadline_check_enabled()` (a programmatic
    /// process-wide flag — no env gate).
    /// Recovery cascade should treat this the same as `OverBudget`.
    Deadline,
    /// The `APPLY_OUTPUT_NODE_CAP` on produced output nodes tripped — a
    /// deliberate size cut, not an OOM. Handlers that don't care treat it
    /// exactly like `OverBudget`.
    OutputCap,
}

/// A blow-up hint for out-of-memory recovery: the variables that live under the
/// vtree node whose apply level ran out of budget.
///
/// When an apply trips [`ApplyError::OverBudget`] and the caller responds by
/// case-splitting (Shannon expansion) on a variable, that variable is normally
/// chosen by a *global* occurrence ranking over the whole formula — blind to the
/// fact that the blow-up is localized to one dense cell of one vtree node. This
/// hint names exactly the variables feeding that node, so a selector can branch
/// on one that actually shrinks the exploding cross-product.
///
/// Install one with [`seed_recovery_hint`]; read it with
/// [`peek_recovery_hint_vars`] (non-consuming) or [`take_recovery_hint`]
/// (consuming). The slot is thread-local and last-writer-wins.
pub struct RecoveryHint {
    /// Leaf [`VarId`]s under the blown-up vtree node, in the variable space of
    /// the diagram that was being compiled. A caller that compiles one connected
    /// component at a time therefore receives component-LOCAL ids; translating
    /// them back to its own numbering is the caller's job.
    pub vars: Vec<VarId>,
}

thread_local! {
    /// The most recent blow-up hint (last-writer-wins), consumed exactly once by
    /// `take_recovery_hint`. Only `seed_recovery_hint` writes it, so it stays
    /// `None` unless a caller installs one.
    pub(crate) static RECOVERY_HINT: RefCell<Option<RecoveryHint>> = const { RefCell::new(None) };
}

/// Consume (`take`) the current recovery hint, leaving `None` behind. Consuming
/// is deliberate: a stale hint from a previous failed attempt can never leak
/// into an unrelated split. Returns `None` when nothing has been seeded since
/// the last take.
pub fn take_recovery_hint() -> Option<RecoveryHint> {
    RECOVERY_HINT.with(|h| h.borrow_mut().take())
}

/// Peek the pending recovery hint's variable list WITHOUT consuming it — a
/// clone, not a `take`, so the consume-once contract of [`take_recovery_hint`]
/// is untouched (the hint is still `Some` afterwards and the next take still
/// fires normally). Returns `None` when no hint is pending.
pub fn peek_recovery_hint_vars() -> Option<Vec<VarId>> {
    RECOVERY_HINT.with(|h| h.borrow().as_ref().map(|hint| hint.vars.clone()))
}

/// Seed the recovery hint with `vars`, as if an apply had just run out of budget
/// under a vtree node covering them. The next [`take_recovery_hint`] consumes
/// it. Use this to steer a case-split selector from outside the apply engine, or
/// to reinstate a hint recorded during an earlier run.
pub fn seed_recovery_hint(vars: Vec<VarId>) {
    RECOVERY_HINT.with(|h| *h.borrow_mut() = Some(RecoveryHint { vars }));
}

thread_local! {
    /// The `2^k` free-variable correction exponent a caller will apply to a
    /// recovery result to lift it from the compiled sub-formula's count up to
    /// the original instance's count.
    ///
    /// It is NOT part of the (stateless) recovery machinery — a case-split
    /// cascade returns the exact count of the formula it received — so the
    /// multiplier is recorded separately, for callers that must report how a
    /// partial result relates to the whole instance. `None` = nothing set it on
    /// this path, which is distinct from an explicit exponent of 0.
    static RECOVERY_LIFT_POW2: Cell<Option<u32>> = const { Cell::new(None) };
}

/// Record the exponent of the `2^exp` multiplier that lifts a recovery result
/// from the compiled sub-formula's count to the original instance's count, so a
/// caller inspecting a partial run can state the exact relationship between the
/// two instead of implying they are comparable. Read back with
/// [`recovery_lift_pow2`].
pub fn set_recovery_lift_pow2(exp: u32) {
    RECOVERY_LIFT_POW2.with(|c| c.set(Some(exp)));
}

/// The post-recovery lift exponent recorded by [`set_recovery_lift_pow2`], or
/// `None` if nothing set it on the current path (the multiplier is then
/// unknown, NOT `2^0 = 1`).
pub fn recovery_lift_pow2() -> Option<u32> {
    RECOVERY_LIFT_POW2.with(|c| c.get())
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
// as long as this cell did not exist) and every instant `OverBudget` looked like
// memory pressure — a whole session's diagnosis went down that wrong path. Written
// on the cold error path only; read by the compile driver when it reports the stop.
thread_local! {
    static LAST_REFUSED_RESERVE_BYTES: Cell<Option<u64>> = const { Cell::new(None) };
}

/// Bytes the most recently REFUSED fallible reserve asked for; `None` until an
/// allocator refusal happens.
/// Sticky (last-writer-wins): the reader is a failure path that runs immediately
/// after the refusal it is reporting.
pub fn last_refused_reserve_bytes() -> Option<u64> {
    LAST_REFUSED_RESERVE_BYTES.with(|c| c.get())
}

/// Forget any recorded refusal, so a report can only ever describe a refusal from
/// the run that is reporting it. Called at the entry of the bottom-up traversal —
/// the one consumer's own unit of work.
pub fn clear_last_refused_reserve() {
    LAST_REFUSED_RESERVE_BYTES.with(|c| c.set(None));
}

/// Record an allocator refusal's request size. Cold: only ever called on the
/// error path of a fallible reserve.
#[cold]
#[inline(never)]
fn note_refused_reserve(bytes: u64) {
    LAST_REFUSED_RESERVE_BYTES.with(|c| c.set(Some(bytes)));
}

/// Tracked `try_reserve_exact`: account capacity growth against the
/// soft budget and trip `OverBudget` as soon as cumulative in-flight
/// bytes (this apply's allocations) exceed the remaining envelope.
///
/// Wraps `Vec::try_reserve_exact` and additionally:
///   1. Maps `TryReserveError` (OS-level allocator failure under
///      `RLIMIT_AS` or similar) → `ApplyError::OverBudget`.
///   2. Measures the capacity delta and adds `delta × sizeof::<T>()`
///      to `MC_BUDGET_IN_FLIGHT` (a thread-local reset at apply entry).
///   3. If the updated in-flight total exceeds `MC_BUDGET_REMAINING`
///      (set per-step by `set_apply_budget`), returns `OverBudget`.
///
/// This is what lets the soft trigger fire *during* a conjoin instead
/// of only at the apply-entry predictive check or the OS-OOM backstop.
#[inline(always)]
pub(crate) fn budget_reserve_exact<T>(v: &mut Vec<T>, additional: usize) -> Result<(), ApplyError> {
    let pre_cap = v.capacity();
    // Lazy→eager jemalloc decay preflight: purge dirty pages before the
    // kernel charges this request against RLIMIT_AS. Only on actual
    // growth — `request_bytes == 0` is reserved for the apply-entry
    // heartbeat poll, which the check throttles separately. See mem.rs.
    if additional > v.capacity() - v.len() {
        crate::tdd::mem_pressure::preflight_alloc_decay(
            (additional as u64).saturating_mul(std::mem::size_of::<T>() as u64),
        );
    }
    v.try_reserve_exact(additional).map_err(|_| {
        note_refused_reserve((additional as u64).saturating_mul(std::mem::size_of::<T>() as u64));
        ApplyError::OverBudget
    })?;
    account_capacity_delta::<T>(v.capacity(), pre_cap)
}

/// Tracked `try_reserve`. Like `budget_reserve_exact` but uses the
/// doubling growth strategy of `Vec::try_reserve`. Use when the caller
/// is genuinely amortizing many small pushes; `budget_reserve_exact` is
/// preferred for known-size grows.
#[inline(always)]
pub(crate) fn budget_reserve<T>(v: &mut Vec<T>, additional: usize) -> Result<(), ApplyError> {
    let pre_cap = v.capacity();
    // Doubling growth: the actual grab is up to 2× current capacity, not
    // `additional`. Preflight with the doubled estimate so a multi-GiB
    // realloc still gets a pre-purge. See mem.rs.
    if additional > v.capacity() - v.len() {
        crate::tdd::mem_pressure::preflight_alloc_decay(
            ((v.capacity().max(additional)) as u64)
                .saturating_mul(std::mem::size_of::<T>() as u64),
        );
    }
    v.try_reserve(additional).map_err(|_| {
        // Doubling growth: the grab is up to 2× capacity, so report what the
        // allocator was actually asked for, not the caller's `additional`.
        note_refused_reserve(
            (v.capacity().max(additional) as u64).saturating_mul(std::mem::size_of::<T>() as u64),
        );
        ApplyError::OverBudget
    })?;
    account_capacity_delta::<T>(v.capacity(), pre_cap)
}

/// Shared bookkeeping for `budget_reserve(_exact)`. Updates the in-flight
/// thread-local and trips `OverBudget` if it crosses `MC_BUDGET_REMAINING`.
/// Enforcement is unconditional — there is no opt-out.
#[inline(always)]
pub(super) fn account_capacity_delta<T>(new_cap: usize, pre_cap: usize) -> Result<(), ApplyError> {
    let delta = new_cap.saturating_sub(pre_cap);
    if delta == 0 { return Ok(()); }
    let bytes = (delta as u64).saturating_mul(std::mem::size_of::<T>() as u64);
    let (total, rem) = APPLY_LIMITS.with(|l| {
        let next = l.budget_in_flight.get().saturating_add(bytes);
        l.budget_in_flight.set(next);
        (next, l.budget_remaining.get())
    });
    if let Some(rem) = rem {
        if total > rem {
            return Err(ApplyError::OverBudget);
        }
    }
    Ok(())
}

/// Charge the output-pair meter ([`ApplyLimits::pairs_in_flight`]): `delta`
/// more slots of capacity in an OUTPUT level's pair arena. The single writer —
/// every site that grows such an arena routes its capacity delta here, and
/// nothing else touches the cell.
///
/// Separate from [`account_capacity_delta`] on purpose, and not derivable from
/// it: that one is typed `T` and sees a `Vec<InputPair>` it cannot tell apart
/// from the hoisted C2 column table, which is a transient scratch and not
/// output. The distinction is the whole point of the meter.
#[inline]
pub(super) fn account_output_pairs(delta: usize) {
    if delta == 0 {
        return;
    }
    APPLY_LIMITS.with(|l| {
        l.pairs_in_flight.set(l.pairs_in_flight.get().saturating_add(delta as u64));
        l.pairs_level_charge.set(l.pairs_level_charge.get().saturating_add(delta as u64));
    });
}

/// Settle the level that just finished: replace the capacity
/// [`account_output_pairs`] charged for it with the `pairs` it actually holds.
/// Called once per level from the per-level tail, which is the one place that
/// knows both. Keeps [`ApplyLimits::pairs_in_flight`] exact over finished
/// levels, so only the level in flight is ever an estimate.
#[inline]
pub(super) fn settle_output_pairs(exact_pairs: usize) {
    APPLY_LIMITS.with(|l| {
        let charged = l.pairs_level_charge.replace(0);
        let total = l.pairs_in_flight.get().saturating_sub(charged);
        l.pairs_in_flight.set(total.saturating_add(exact_pairs as u64));
    });
}

/// Release a TRANSIENT's bytes from the in-flight accounting when the backing
/// allocation is actually freed. `MC_BUDGET_IN_FLIGHT` is otherwise monotone
/// within one apply (persistent structures — nodes/pairs/ext/grids — only
/// grow, and the counter resets at apply entry), so a per-level scratch that
/// charged itself via `budget_reserve(_exact)` and then drops mid-apply must
/// un-charge here or it permanently consumes soft headroom it no longer uses.
/// Only pair with a real free of the exact accounted capacity (see
/// `cell::PreparedC2`'s Drop); never call for still-live allocations.
#[inline]
pub(super) fn unaccount_transient_bytes(bytes: u64) {
    if bytes == 0 { return; }
    APPLY_LIMITS.with(|l| l.budget_in_flight.set(l.budget_in_flight.get().saturating_sub(bytes)));
}

/// Remaining apply-byte headroom for the current `apply_and_fallible` call:
/// `MC_BUDGET_REMAINING − MC_BUDGET_IN_FLIGHT` (saturating). `None` when no soft
/// budget is installed. Mirrors the `total > rem` test in `account_capacity_delta`
/// — a single reserve of ≤ this many bytes is guaranteed not to trip the soft
/// trigger. Consumed (through [`apply_headroom_bytes_or_vas`]) by
/// `decide_emit_growth_mode` and `grow_pairs_bounded`'s increment policy.
#[inline]
pub(super) fn apply_budget_headroom_bytes() -> Option<u64> {
    APPLY_LIMITS.with(|l| {
        l.budget_remaining.get().map(|rem| rem.saturating_sub(l.budget_in_flight.get()))
    })
}

/// Generous finite headroom returned by [`apply_headroom_bytes_or_vas`] when
/// `RLIMIT_AS` is unlimited (plain dev runs — the bench harness and the MCC
/// competition always install a cap). With no address-space ceiling there is
/// nothing for the emit's `Vec`-doubling transient to trip, so plain doubling
/// is unconditionally safe; 1 TiB dwarfs any real reservation while staying
/// finite so the u128 `3 × bound × pair_bytes < h` comparison never overflows.
const VAS_UNLIMITED_HEADROOM: u64 = 1 << 40; // 1 TiB

/// Safety margin of address space the *guarded* apply path refuses to consume,
/// subtracted from the `RLIMIT_AS − mapped` headroom in
/// [`apply_headroom_bytes_or_vas`] branch (2, the default-production
/// no-soft-budget path).
///
/// **Abort class it protects against.** Rust's infallible allocations abort the
/// process on failure (`memory allocation of N bytes failed`, SIGABRT) via a
/// `#[rustc_nounwind]` handler — the panic cannot unwind, so the handled-OOM →
/// Shannon-recovery cascade never runs. Our fallible reserves
/// (`budget_reserve*`) and the dense-precount gates surface `OverBudget`
/// cleanly, but if they let the process consume address space right up to
/// `RLIMIT_AS`, any moderate *unguarded* transient — a count-walk level vec, a
/// projection row buffer, a recovery child's raw alloc — lands on a full
/// address space and aborts uncatchably. Measured: MCC-2026 085 at a 3000 MiB
/// ceiling died this way inside a recovery child on a 485,609,608-byte raw
/// alloc; 139/143 die the same way on first compile. Holding this much room
/// back below the ceiling keeps the guarded path from ever reaching the wall,
/// so those transients have somewhere to land and the *handled* failure fires
/// first (recovery gets its chance).
///
/// 1.5 GiB: the observed aborting transient was ~0.46 GiB; this leaves several
/// such transients of room. It is NOT sized to cover the large *guarded* apply
/// transients (the ~5.9 GiB unwind-dropped delta seen on 177 was guarded apply
/// memory, which the budget/precount gates already handle) — only the small
/// unguarded strays.
const SOFT_HEADROOM_MARGIN_BYTES: u64 = 1536 * 1024 * 1024; // 1.5 GiB

/// Cached `RLIMIT_AS` for the emit-growth VAS fallback. Captured on first
/// consult, which happens deep in a compile — long after startup lowers the
/// address-space cap to mirror the cgroup/harness ceiling
/// (`driver::mod.rs`), the ONLY writer (`set_rlimit_as_bytes` only ever lowers,
/// at startup). So the limit is stable by the time the emit-growth machinery
/// first reads it, and caching avoids a `getrlimit` syscall per huge level. Reads through the shared
/// `crate::tdd::mem_pressure::rlimit_as_bytes()` — a memoized *value*, NOT a second
/// RLIMIT reader.
fn cached_rlimit_as_bytes() -> Option<u64> {
    static RL: OnceLock<Option<u64>> = OnceLock::new();
    *RL.get_or_init(crate::tdd::mem_pressure::rlimit_as_bytes)
}

/// Headroom for the emit-growth machinery (`decide_emit_growth_mode` and
/// `grow_pairs_bounded`'s increment policy), WITH a VAS-derived fallback
/// when no soft budget is armed. Unlike [`apply_budget_headroom_bytes`] this
/// always returns a value — the whole point is a real headroom figure in
/// default production, where no soft budget exists.
///
/// - **Soft budget armed** (segmented compile): returns exactly
///   `apply_budget_headroom_bytes()` unwrapped — IDENTICAL to the old
///   soft-budget semantics that path relies on; no VAS is consulted.
/// - **No soft budget** (default production): the soft-budget accounting is
///   inert, so derive real address-space room directly —
///   `RLIMIT_AS − SOFT_HEADROOM_MARGIN_BYTES − current VAS usage`, via the same
///   `crate::tdd::mem_pressure::{rlimit_as_bytes, mapped_bytes}` handle the memory-adaptive
///   machinery keys off. The [`SOFT_HEADROOM_MARGIN_BYTES`] subtraction holds a
///   safety margin back below the ceiling so the guarded path never consumes
///   the last of the address space — leaving room for unguarded transients that
///   would otherwise abort the process uncatchably (see that constant's doc).
///   Plentiful room ⇒ plain doubling; near-cap levels (canary `mc2020_131` under
///   the 31 GiB MCC cap) get a small headroom ⇒ bounded-increment growth.
/// - **`RLIMIT_AS` unlimited** (plain dev runs): no ceiling for the doubling
///   transient to trip, so return [`VAS_UNLIMITED_HEADROOM`] (doubling always
///   safe).
///
/// Conservative by construction: `mapped_bytes()` is the mapped+retained
/// high-water figure `RLIMIT_AS` actually charges against, so it can only
/// OVER-count live usage — which only ever SHRINKS the returned headroom
/// (⇒ bounded growth / smaller increments), never reporting more room than
/// truly exists.
#[inline]
pub(super) fn apply_headroom_bytes_or_vas() -> u64 {
    if let Some(h) = apply_budget_headroom_bytes() {
        return h;
    }
    match cached_rlimit_as_bytes() {
        Some(limit) => vas_headroom_with_margin(limit, crate::tdd::mem_pressure::mapped_bytes()),
        None => VAS_UNLIMITED_HEADROOM,
    }
}

/// Pure branch-(2) arithmetic for [`apply_headroom_bytes_or_vas`] (factored out
/// for unit tests): `limit − SOFT_HEADROOM_MARGIN_BYTES − mapped`, saturating.
///
/// Holds [`SOFT_HEADROOM_MARGIN_BYTES`] back below the `RLIMIT_AS` ceiling so the
/// guarded path refuses the last margin of address space — see that constant's
/// doc for the uncatchable-abort class this prevents. Saturating: a small or
/// already-exhausted address space just reports zero headroom (⇒ most bounded
/// growth), never wraps.
#[inline]
fn vas_headroom_with_margin(limit: u64, mapped: u64) -> u64 {
    limit
        .saturating_sub(SOFT_HEADROOM_MARGIN_BYTES)
        .saturating_sub(mapped)
}

/// Returns `true` when the fine-grained apply-deadline check is enabled.
///
/// Enabled programmatically via `enable_apply_deadline_check()` (set by a
/// downstream driver performing a deadline-bounded compile). When ON,
/// `apply_and_fallible`'s vtree-level loop checks `APPLY_DEADLINE` at the top of
/// each iteration and returns `Err(ApplyError::Deadline)` on expire. Default OFF.
#[doc(hidden)]
pub fn apply_deadline_check_enabled() -> bool {
    APPLY_DEADLINE_CHECK_OVERRIDE.load(Ordering::Relaxed)
}

/// `true` iff the apply deadline check is enabled AND the compile has reached
/// something that stops it — the installed `APPLY_DEADLINE`, or a decision point
/// of an armed [`Schedule`] that concluded it should stop ([`limits_reached`]).
///
/// Intended for amortized calls from the dense cell-build row loops (e.g. once
/// per ~65k cells) so a single un-yielding wide node — whose product can run
/// minutes between vtree-level boundaries — can still be cut mid-build and
/// surfaced as `Err(Deadline)`. This is also the ONLY place a schedule armed
/// over a long step gets to stand: without it a decision point inside a step
/// that never ends is a decision point that is never reached.
///
/// Cheap when off: an `AtomicBool` relaxed load that reads `false`
/// short-circuits before any TLS read or `Instant::now()`.
#[inline]
pub(super) fn apply_deadline_expired() -> bool {
    limits_reached(apply_deadline_check_enabled())
}

/// Which arming cell a [`PollTicker`] consults when its meter comes due.
///
/// The two are separate gates, not two names for one (see
/// [`REDUCE_DEADLINE_CHECK`]), and the ticker carries the choice as data so both
/// walks share ONE amortization implementation instead of growing a second copy
/// of the counter/stride/poll trio.
#[derive(Copy, Clone, PartialEq, Eq)]
pub(crate) enum PollGate {
    /// The apply's own cell/scatter loops.
    Apply,
    /// The post-apply walks between two applies of one bottom-up step: the
    /// reduce walk, the ∃-forget batch and the clustering rotation pass (see
    /// [`REDUCE_DEADLINE_CHECK`], the cell all three consult).
    Reduce,
}

/// Amortized cut ticker shared by the dense between-cell loops (stride 1<<16),
/// the sparse-scatter/collapse-collector loops (stride 1<<20 =
/// `APPLY_POLL_STRIDE`) and the reduce walk (`REDUCE_POLL_STRIDE`). One poll per
/// `stride` units of accumulated work: a gated wall-deadline check. See
/// `nxm_deadline_check!` for the deliberate intra-cell exception. The compile
/// path is sequential — there is no cross-thread cancellation, so the wall
/// deadline is the only mid-level cut.
pub(crate) struct PollTicker {
    work: u64,
    stride: u64,
    gate: PollGate,
}

impl PollTicker {
    #[inline]
    pub(super) fn new(stride: u64) -> Self {
        Self { work: 0, stride, gate: PollGate::Apply }
    }

    /// A ticker for the post-apply walks — same amortization, the other arming
    /// cell ([`PollGate::Reduce`]). `stride` is a parameter (not the constant) so
    /// a test can pin the counter's cadence without lowering the production one.
    #[inline]
    pub(crate) fn reduce(stride: u64) -> Self {
        Self { work: 0, stride, gate: PollGate::Reduce }
    }

    /// Add `inc` units of accumulated work; once `work >= stride`, reset to 0
    /// and poll. (The former one-unit `tick()` had no callers left once the
    /// dense row loop moved to a per-ROW `tick_by(k2)`.)
    #[inline(always)]
    pub(crate) fn tick_by(&mut self, inc: u64) -> Result<(), ApplyError> {
        self.work += inc;
        if self.work >= self.stride {
            let done = self.work;
            self.work = 0;
            self.poll(done)
        } else {
            Ok(())
        }
    }

    #[cold]
    fn poll(&self, done: u64) -> Result<(), ApplyError> {
        // A TEE of the work `tick_by` already counted, not a second counter:
        // this is the amortization point that already exists for reading it,
        // which is why the compile work clock costs the hot path nothing.
        //
        // `done` is what the meter actually held, NOT the stride it crossed. A
        // single `tick_by` carries a whole dense row (`k2`), which can be many
        // strides wide on its own — charging one stride per poll would price
        // that row the same as the narrowest one that trips the meter, and the
        // undercount would fall hardest on exactly the wide-row applies the
        // give-up rule is trying to measure.
        charge_compile_work(done);
        let expired = match self.gate {
            PollGate::Apply => apply_deadline_expired(),
            PollGate::Reduce => reduce_deadline_expired(),
        };
        if expired {
            return Err(ApplyError::Deadline);
        }
        Ok(())
    }
}

/// Amortization stride for the sparse-scatter/collapse-collector [`PollTicker`]s
/// — one poll per ~1M units of inner work, matching `nxm_deadline_check`'s
/// `1 << 20` cadence.
pub(super) const APPLY_POLL_STRIDE: u64 = 1 << 20;

/// Amortization stride for the dense between-cell [`PollTicker`]s — one poll
/// per ~65536 cell iterations, carrying over the prior `CancelPoll`'s cadence.
pub(super) const DENSE_CELL_POLL_STRIDE: u64 = 1 << 16;

/// Amortization stride for the post-apply walks' [`PollTicker`] — one poll per
/// ~16384 units, where a unit is one node of the level the walk is standing on
/// (a contracted parent's level, a forget batch's target level, a clustering
/// pivot's pair count).
///
/// Smaller than the apply strides because the unit is coarser: the apply counts
/// individual product pairs, these walks count whole levels' node widths, and a
/// worklist of narrow parents would otherwise run millions of pops between two
/// polls. At this cadence the poll (one relaxed load, one TLS read, one
/// `Instant::now`) is under a thousandth of the work it amortizes over even when
/// every popped parent is as small as it can be.
const REDUCE_POLL_STRIDE: u64 = 1 << 14;

#[cfg(test)]
thread_local! {
    /// Test-only override for [`reduce_poll_stride`]. Thread-local, so a test that
    /// pins the cadence cannot race the rest of the suite — the same shape as
    /// `cell::BOTHMARG_COLLAPSE_OVERRIDE`.
    static REDUCE_POLL_STRIDE_OVERRIDE: Cell<Option<u64>> = const { Cell::new(None) };
}

/// The post-apply walks' amortization stride: [`REDUCE_POLL_STRIDE`], or a
/// test's pinned value.
///
/// The hook exists so the amortization itself is testable — a diagram big enough
/// to accumulate 16384 units of contract work before the meter comes due is not a
/// unit test — WITHOUT lowering the production cadence, which is the number the
/// overhead argument is made about.
#[inline]
pub(crate) fn reduce_poll_stride() -> u64 {
    #[cfg(test)]
    if let Some(stride) = REDUCE_POLL_STRIDE_OVERRIDE.with(|c| c.get()) {
        return stride;
    }
    REDUCE_POLL_STRIDE
}

/// Run `body` with the post-apply walks' stride pinned to `stride` on this
/// thread, restoring the prior setting on return. Test-only.
#[cfg(test)]
pub(crate) fn with_reduce_poll_stride<T>(stride: u64, body: impl FnOnce() -> T) -> T {
    let prev = REDUCE_POLL_STRIDE_OVERRIDE.with(|c| c.replace(Some(stride)));
    let out = body();
    REDUCE_POLL_STRIDE_OVERRIDE.with(|c| c.set(prev));
    out
}

/// Grow `v` up to `new_len`, filling with `DEAD`. Returns `OverBudget`
/// if the allocator refuses (e.g. under `ulimit -v`) or the soft budget
/// would be exceeded.
#[inline]
pub(super) fn try_resize_dead(v: &mut Vec<u32>, new_len: usize) -> Result<(), ApplyError> {
    if v.len() >= new_len { return Ok(()); }
    let additional = new_len - v.len();
    budget_reserve_exact(v, additional)?;
    v.resize(new_len, DEAD);
    Ok(())
}

/// Interleaved-map variant of `try_resize_dead`: grows a `[ct, dt]`-entry
/// vec, filling with `[DEAD, DEAD]`.
#[inline]
pub(crate) fn try_resize_dead2(v: &mut Vec<[u32; 2]>, new_len: usize) -> Result<(), ApplyError> {
    if v.len() >= new_len { return Ok(()); }
    let additional = new_len - v.len();
    budget_reserve_exact(v, additional)?;
    v.resize(new_len, [DEAD, DEAD]);
    Ok(())
}

/// Fallible `push`: reserve one slot before the push so allocation failure
/// returns `Err(OverBudget)` instead of aborting the process. Same doubling
/// growth as `Vec::push` when the vec is full. Use on every `push` in
/// `apply_and_fallible`'s reach that could grow under adversarial CNF inputs.
///
/// Shape: an explicit `len < capacity` fast path that stores the element and
/// nothing else, with the ENTIRE reserve-and-account body exiled to
/// [`try_push_grow`]. The two are observationally identical because with spare
/// capacity the old monolithic body was already inert: `budget_reserve(v, 1)`
/// skipped its `preflight_alloc_decay` (guarded on `1 > cap − len`),
/// `try_reserve(1)` found `needs_to_grow == false`, and
/// `account_capacity_delta` saw `delta == 0` and charged nothing. Splitting
/// them is a codegen fix, not a semantic one: the reserve path's
/// `OnceLock`-indirect preflight probe and its TLS accounting store are joins
/// LLVM will not keep `len`/`capacity`/the vec base live across, so rejoining
/// them re-loaded all three and re-tested `len == capacity` three more times
/// per pushed element inside the apply kernel. `#[inline(never)]` on the grow
/// helper is load-bearing — it is what removes the join.
#[inline(always)]
pub(crate) fn try_push<T>(v: &mut Vec<T>, x: T) -> Result<(), ApplyError> {
    if v.len() < v.capacity() {
        // Spare capacity: `Vec::push` cannot reallocate, so there is nothing to
        // preflight, refuse, or charge.
        v.push(x);
        return Ok(());
    }
    try_push_grow(v, x)
}

/// Growth arm of [`try_push`] — the original body, verbatim. Reached only when
/// `len == capacity`, i.e. once per doubling event, so the out-of-line call is
/// amortized to nothing. All budget accounting lives here (and only here):
/// `budget_reserve` runs the allocator preflight, maps a refusal to
/// `OverBudget`, and charges the capacity delta through
/// `account_capacity_delta`.
#[cold]
#[inline(never)]
fn try_push_grow<T>(v: &mut Vec<T>, x: T) -> Result<(), ApplyError> {
    budget_reserve(v, 1)?;
    v.push(x);
    Ok(())
}

/// Fallible pair push: stores into `level.pairs`, routing growth through
/// `try_push`. Packing paths (packed-pairs, Phase F, Lever 14) have been deleted.
///
/// This is the single choke point for `level.pairs` growth on the dense emit
/// walk. When `decide_emit_growth_mode` has flagged the level as
/// near-cap ([`ApplyLimits::pairs_bounded_growth`]), a growth event (len ==
/// capacity) routes through [`grow_pairs_bounded`] — bounded, headroom-aware
/// increments instead of `Vec`'s 2× doubling — so the reallocation transient
/// stays `current + increment` rather than doubling's 3×current. In the
/// default mode (flag off) the extra branch is one predicted-false TLS check
/// on growth events only; the capacity-available fast path is unchanged.
///
/// Split like [`try_push`]: a bare `len < capacity` store here, everything else
/// in [`push_pair_grow`]. See `try_push`'s doc for why the spare-capacity path
/// was already inert (no preflight, no TLS read, zero budget charged) and why
/// exiling the rest is what lets LLVM keep `len`/`capacity`/the arena base in
/// registers across the emit walk's pushes.
#[inline(always)]
pub(super) fn try_push_pair_into(level: &mut crate::tdd::types::TddLevel, pair: crate::tdd::types::InputPair) -> Result<(), ApplyError> {
    let v = &mut level.pairs;
    if v.len() < v.capacity() {
        v.push(pair);
        return Ok(());
    }
    push_pair_grow(v, pair)
}

/// Growth arm of [`try_push_pair_into`] — the original body, verbatim (the
/// `pairs_bounded_growth` TLS check gating [`grow_pairs_bounded`], then
/// [`try_push`], which supplies the `budget_reserve` → `account_capacity_delta`
/// → push accounting). Reached only when `len == capacity`, so the
/// `len == v.capacity()` re-test is trivially true and kept only so this body
/// stays a literal copy of the pre-split one.
#[cold]
#[inline(never)]
fn push_pair_grow(
    v: &mut Vec<crate::tdd::types::InputPair>,
    pair: crate::tdd::types::InputPair,
) -> Result<(), ApplyError> {
    let pre_cap = v.capacity();
    if v.len() == v.capacity() && APPLY_LIMITS.with(|l| l.pairs_bounded_growth.get()) {
        grow_pairs_bounded(v)?;
    }
    let out = try_push(v, pair);
    // Output-pair meter: charged here rather than in `try_push_pair_into`
    // because this is the arm a growth event reaches, and growth is the only
    // thing that moves capacity. See [`ApplyLimits::pairs_in_flight`].
    account_output_pairs(v.capacity().saturating_sub(pre_cap));
    out
}

/// Minimum bounded-growth increment for `level.pairs`, in bytes (1 M pairs at
/// 8 B). Floors the increment so growth never degenerates to per-push
/// reallocation. It cannot cause O(n²) memcpy in practice: the floor only
/// binds when half the remaining headroom is below it (headroom
/// < 2 × this), i.e. within one or two growth events of `OverBudget` —
/// everywhere else the increment is half-headroom (geometric) or full
/// doubling.
const PAIRS_GROW_MIN_CHUNK_BYTES: u64 = 8 * 1024 * 1024;

/// Byte size of one `level.pairs` element — the single element-size constant
/// for every pair-arena growth computation.
const PAIR_ELEM_BYTES: u64 = std::mem::size_of::<crate::tdd::types::InputPair>() as u64;

/// Pure increment policy for [`grow_pairs_bounded`] (factored out for unit
/// tests): grow a full `cap`-capacity Vec by
/// `min(cap, max(min_chunk, headroom/2))` elements, i.e.
/// `next_cap = min(2×cap, cap + max(min_chunk, half-headroom))`.
///
/// - Plentiful headroom (`headroom/2 ≥ cap × elem_bytes`): increment = `cap`
///   — plain doubling, amortized O(n) pushes.
/// - Shrinking headroom: increment = half the remaining headroom, so the
///   reallocation transient (`old cap + increment`) always fits, and
///   successive increments halve geometrically — O(log) growth events.
/// - Near exhaustion: the `min_chunk` floor; at most a couple of events
///   before the budget/allocator trips `OverBudget` (see the constant doc).
fn bounded_grow_increment(cap: usize, headroom_bytes: u64, elem_bytes: u64) -> usize {
    let eb = elem_bytes.max(1);
    let half_room = usize::try_from(headroom_bytes / 2 / eb).unwrap_or(usize::MAX);
    let min_chunk = (PAIRS_GROW_MIN_CHUNK_BYTES / eb) as usize;
    half_room.max(min_chunk).min(cap)
}

/// Bounded-growth increment for a `cap`-capacity `level.pairs`: the ONE place
/// that feeds the pair element size and the current headroom into
/// [`bounded_grow_increment`]. Shared by [`grow_pairs_bounded`] (per-push choke
/// point) and [`reserve_pairs_for_emit`] (bulk twin), so both grow by the same
/// policy.
#[inline]
fn bounded_pairs_increment(cap: usize) -> usize {
    bounded_grow_increment(cap, apply_headroom_bytes_or_vas(), PAIR_ELEM_BYTES)
}

/// Bounded, headroom-aware growth for a full `level.pairs` Vec (the near-cap
/// emit mode — see [`try_push_pair_into`]). Reserves the
/// [`bounded_grow_increment`] via `budget_reserve_exact`, so the chunk goes
/// through the ONE existing accounting path (`account_capacity_delta`,
/// preflight decay check, OS-failure → `OverBudget`) — no second budget
/// mechanism. Cold: called once per growth event, never per push.
#[cold]
#[inline(never)]
fn grow_pairs_bounded(v: &mut Vec<crate::tdd::types::InputPair>) -> Result<(), ApplyError> {
    let cap = v.capacity();
    if cap == 0 {
        // Fresh vec: `try_push`'s doubling from empty is trivially transient-safe.
        return Ok(());
    }
    budget_reserve_exact(v, bounded_pairs_increment(cap))
}

/// Bulk twin of [`try_push_pair_into`]: guarantee room for `additional` more
/// pairs in `level.pairs`, so the caller can then emit them with plain
/// `Vec::push` / `extend_from_slice` instead of a per-pair reserve.
///
/// Growth obeys the SAME per-level mode as the per-push choke point — bounded,
/// headroom-aware [`bounded_grow_increment`] chunks when
/// `decide_emit_growth_mode` armed this level, plain `Vec` doubling otherwise —
/// so `level.pairs` has one growth policy, not two. A refusal maps to
/// `OverBudget`, the same failure the caller's other reserves return, so the
/// recovery cascade sees no new error class.
///
/// The ONE intentional divergence from the accounted reserves is the budget
/// CHARGE: this serves the clause apply, which runs outside [`reset_meters`]
/// (only `apply_and_fallible` entry clears that counter), so charging its pair
/// arena would accumulate across clauses and trip the soft budget spuriously.
/// The allocator preflight and the `OverBudget` refusal channel are the same.
#[inline]
pub(crate) fn reserve_pairs_for_emit(
    level: &mut crate::tdd::types::TddLevel,
    additional: usize,
) -> Result<(), ApplyError> {
    let v = &mut level.pairs;
    if additional <= v.capacity() - v.len() {
        return Ok(());
    }
    if APPLY_LIMITS.with(|l| l.pairs_bounded_growth.get()) {
        let inc = bounded_pairs_increment(v.capacity()).max(additional);
        // Same lazy→eager jemalloc decay preflight `budget_reserve_exact` runs
        // before an exact grow: purge dirty pages so the realloc is not refused
        // under RLIMIT_AS.
        crate::tdd::mem_pressure::preflight_alloc_decay(
            (inc as u64).saturating_mul(PAIR_ELEM_BYTES),
        );
        return v.try_reserve_exact(inc).map_err(|_| ApplyError::OverBudget);
    }
    // Doubling growth: preflight with the doubled estimate, exactly as
    // `budget_reserve` does.
    crate::tdd::mem_pressure::preflight_alloc_decay(
        (v.capacity().max(additional) as u64).saturating_mul(PAIR_ELEM_BYTES),
    );
    v.try_reserve(additional).map_err(|_| ApplyError::OverBudget)
}

/// Fallible analogue of `Vec::resize` for grow-only callers. No-op when
/// `v.len() >= new_len`. Uses `try_reserve_exact` so VAS is not over-reserved
/// (unlike the doubling strategy of `try_reserve`).
#[inline]
pub(crate) fn try_resize<T: Clone>(v: &mut Vec<T>, new_len: usize, val: T) -> Result<(), ApplyError> {
    if v.len() >= new_len { return Ok(()); }
    let additional = new_len - v.len();
    budget_reserve_exact(v, additional)?;
    v.resize(new_len, val);
    Ok(())
}

/// All apply-limit state for the thread, as ONE TLS struct-of-Cells (one TLS
/// address resolution + field offsets, instead of six TLS slots). Plain
/// `Cell`s, NOT `RefCell<struct>` — the byte-budget charge path (`try_push`
/// per emitted node) must stay a bare load/store.
pub(super) struct ApplyLimits {
    /// Soft-budget remaining for the upcoming apply (the heap cap
    /// minus `live_bytes_estimate` at step entry). `None` disables the
    /// predictive check; `try_reserve_exact` still catches OS-level OOM
    /// (e.g. under `ulimit -v`) regardless. Set by the vsplit driver
    /// before each `run_internal_step` and cleared after; see
    /// `set_apply_budget` below.
    pub(super) budget_remaining: Cell<Option<u64>>,

    /// Bytes allocated so far in the current `apply_and_fallible` call,
    /// summed across `try_reserve(_exact)` capacity deltas in tracked
    /// Vecs. Reset to 0 at apply entry; compared against
    /// `budget_remaining` after every grow. Lets the soft trigger
    /// fire *during* a conjoin without waiting for the OS allocator —
    /// the apply-internal transient peak (e.g. `node_idx` grids, output
    /// pair vectors) is what blows past the per-step estimate.
    pub(super) budget_in_flight: Cell<u64>,

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
    pub(super) pairs_in_flight: Cell<u64>,

    /// Capacity charged into `pairs_in_flight` for the level currently being
    /// built — what [`settle_output_pairs`] takes back off the meter when that
    /// level finishes and its exact size is known. Zero between levels.
    ///
    /// The three early-exit routes that skip the per-level tail never settle,
    /// so whatever they charged is taken off at the NEXT boundary instead: the
    /// meter reads low there, which is the direction a size floor tolerates.
    pub(super) pairs_level_charge: Cell<u64>,

    /// Optional wallclock deadline for the current apply call. When set AND the
    /// deadline check is enabled (via `enable_apply_deadline_check()`, which
    /// production callers fire — see `apply_deadline_check_enabled`),
    /// `apply_and_fallible`'s vtree-level loop and the amortized dense/sparse
    /// cell-loop polls (`apply_deadline_expired`) return `Err(ApplyError::Deadline)`
    /// once it passes. Installed via `apply_limits().deadline(..).apply()`.
    /// A field of `ApplyLimits`, itself `pub(super)` — nothing outside
    /// `apply_inner` may touch this raw cell.
    pub(super) deadline: Cell<Option<std::time::Instant>>,

    /// Optional schedule of decision points over the compile in flight, armed by
    /// [`arm_schedule`] and read beside the deadline by [`limits_reached`]. The
    /// deadline says when the compile must STOP; this says when it must be
    /// ASKED. `None` (default) is every compile nobody scheduled.
    pub(super) schedule: Cell<Option<Schedule>>,

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
    pub(super) output_node_cap: Cell<Option<u64>>,

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
    pub(super) work_clock: Cell<u64>,

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
    /// progress-based give-up rule (the downstream driver's `StallRope`),
    /// whose floor is ONE number in pairs; a step meets it with the diagram
    /// it was HANDED or with the one it has BUILT, and this cell carries both. `None` (default) —
    /// nothing armed, which is every apply the give-up rule is not running over.
    /// Install/restore with a set→clear pair like `deadline`.
    pub(super) stall_rope: Cell<Option<(u64, RopeLimit)>>,

    /// Emit-growth mode for the dense level currently being built: `false`
    /// (default) ⇒ plain `Vec`-doubling growth in `try_push_pair_into`;
    /// `true` ⇒ near-cap level whose doubling transient is not provably
    /// affordable — growth goes through `grow_pairs_bounded` (bounded,
    /// headroom-aware increments). Disarmed exactly once per level:
    /// `decide_emit_growth_mode` disarms on entry (then arms on its near-cap
    /// decision) for the routes that call it, and the cell-build loop disarms
    /// for the routes that skip it. Apply entry (`reset_meters`) clears it
    /// again — so it can never leak across levels or applies.
    pub(super) pairs_bounded_growth: Cell<bool>,

    /// Whether anyone is WATCHING the applies on this thread — set by the caller
    /// that wants to know where a long one has got to, and read by the apply
    /// before it pays for publishing anything ([`merge_position`]).
    pub(super) merges_watched: Cell<bool>,

    /// Where the apply in flight has got to: when it BEGAN, the vtree level it is
    /// standing on, and how many that apply has in all. Published by the apply
    /// itself — one store per level, no clock — and read by the watcher on the
    /// polls it is already making.
    ///
    /// This crate does not interpret it: the same division of labour as
    /// [`Schedule`], where what this crate provides is the place to stand inside
    /// an operation and the caller provides everything else. `None` outside a
    /// watched apply.
    pub(super) merge: Cell<Option<(std::time::Instant, u32, u32)>>,
}

thread_local! {
    pub(super) static APPLY_LIMITS: ApplyLimits = const {
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
        }
    };
}

/// Watch (or stop watching, with `false`) the applies on this thread, returning
/// the prior setting so the caller can restore it.
///
/// Unscoped for the same reason [`arm_schedule`] is: the watcher outlives the
/// individual applies it is watching, and the caller that armed it is the one
/// that knows when the compile they belong to is over.
#[doc(hidden)]
pub fn watch_merges(on: bool) -> bool {
    APPLY_LIMITS.with(|l| l.merges_watched.replace(on))
}

/// Whether to publish where this apply has got to — one `Cell` load, asked once
/// per apply and once before each level, so an unwatched compile pays that and
/// nothing else.
pub(super) fn merges_watched() -> bool {
    APPLY_LIMITS.with(|l| l.merges_watched.get())
}

/// An apply BEGINNING, over `levels` vtree levels. Clears whatever the last one
/// left, so a watcher can tell two applies apart by the instant alone.
pub(super) fn merge_began(levels: u32) {
    APPLY_LIMITS.with(|l| l.merge.set(Some((std::time::Instant::now(), 0, levels))));
}

/// An apply REACHING level `level`. One store, no clock — the watcher reads the
/// clock it was already reading.
pub(super) fn merge_reached(level: u32) {
    APPLY_LIMITS.with(|l| {
        if let Some((began, _, levels)) = l.merge.get() {
            l.merge.set(Some((began, level, levels)));
        }
    });
}

/// Where the apply in flight has got to — `(began, level, levels)`, or `None`
/// outside a watched apply (see `watch_merges`).
#[doc(hidden)]
pub fn merge_position() -> Option<(std::time::Instant, u32, u32)> {
    APPLY_LIMITS.with(|l| l.merge.get())
}

/// Reset the per-apply meters — the in-flight byte counters that accumulate
/// across the apply currently in flight — in one TLS touch. Called once, at the
/// top of `apply_and_fallible_inner`.
pub(super) fn reset_meters() {
    APPLY_LIMITS.with(|l| {
        l.budget_in_flight.set(0);
        l.pairs_in_flight.set(0);
        l.pairs_level_charge.set(0);
        // Hygiene: the per-level emit-growth mode must never survive into a
        // new apply (it is also disarmed once per level).
        l.pairs_bounded_growth.set(false);
    });
}

/// Set the per-level bounded-emit-growth mode consulted by
/// `try_push_pair_into` (see [`ApplyLimits::pairs_bounded_growth`]). Disarmed
/// (false) once per level — by `decide_emit_growth_mode` on entry, or by the
/// cell-build loop on the routes that skip it; armed (true) only by
/// `decide_emit_growth_mode`'s near-cap decision.
#[inline]
pub(super) fn set_pairs_bounded_growth(on: bool) {
    APPLY_LIMITS.with(|l| l.pairs_bounded_growth.set(on));
}

/// Set or clear the per-thread soft budget consulted by
/// `apply_and_fallible`'s pre-reserve check. Pass `Some(remaining)` =
/// total budget minus current live bytes, or `None` to disable.
///
/// The ONE writer is the compiler's per-merge refresh in the downstream
/// driver's batch-build step, which recomputes
/// `budget − live` after every merge of a budgeted compile. That value is
/// derived per-traversal state, NOT a caller-supplied input: the traversal
/// (`bottomup_compile`) owns its lifetime and scopes it with
/// `apply_limits().budget(None).apply()`, so nothing it writes can leak onto
/// the thread and make a LATER, unbudgeted compile trip `OverBudget` on its
/// first tracked allocation. Callers that install a budget for a lexical scope
/// should use that RAII builder rather than this raw setter.
pub fn set_apply_budget(remaining_bytes: Option<u64>) {
    APPLY_LIMITS.with(|l| l.budget_remaining.set(remaining_bytes));
}

/// The soft budget currently armed on this thread — the read side of
/// [`set_apply_budget`], and the only one outside the apply hot path.
///
/// Exists so the invariant that makes sequential in-process compiles trustworthy
/// is ASSERTABLE, not merely documented: a budgeted traversal writes this cell
/// once per merge and `bottomup_compile` scopes its lifetime, so a compile that
/// ended at its memory ceiling must leave `None` behind. If it ever leaves a
/// near-zero `Some`, the next compile on the thread dies `OverBudget` on its
/// first tracked allocation — a phantom failure indistinguishable from a formula
/// that genuinely cannot compile in the memory available (see
/// `tests/compile_budget_isolation.rs`, which is the reader that keeps this
/// accessor alive). Plain `pub` because dependency crates never see
/// `cfg(test)`.
pub fn apply_budget_remaining() -> Option<u64> {
    APPLY_LIMITS.with(|l| l.budget_remaining.get())
}

/// Zero the in-flight byte meter.
///
/// The meter is per-APPLY state (`reset_meters` clears it at apply entry) but it
/// is compared against a budget that a *traversal* arms, and tracked reserves also
/// happen between applies (building the next clause batch). A traversal that ended
/// inside a huge apply therefore leaves a large total behind, and the next
/// traversal on the thread can charge ITS first between-apply reserve against that
/// stale total the moment its own per-merge refresh arms a budget — an
/// `OverBudget` in milliseconds, on a compile that has barely allocated. So the
/// bottom-up traversal zeroes the meter at entry, alongside the `budget(None)`
/// scope that owns `budget_remaining`: a traversal only ever measures bytes it
/// charged itself. Also used by apply unit tests that charge the meter directly
/// without going through `apply_and_fallible`.
pub fn reset_apply_in_flight() {
    APPLY_LIMITS.with(|l| l.budget_in_flight.set(0));
}

/// Charge the in-flight meter as a previous traversal's aborted apply would have,
/// WITHOUT allocating the bytes. The test seam for the ownership rule on
/// [`reset_apply_in_flight`]: a regression test needs "a compile inherits a
/// multi-GiB charge" to be cheap and exact, and the honest way to get there —
/// aborting a real multi-GiB compile — costs seconds and pins nothing precisely.
/// Same role as [`seed_recovery_hint`], and like it, never called in production.
#[doc(hidden)]
pub fn charge_apply_in_flight_for_test(bytes: u64) {
    APPLY_LIMITS.with(|l| l.budget_in_flight.set(l.budget_in_flight.get().saturating_add(bytes)));
}

/// Bytes charged to the in-flight meter so far — the read side of
/// [`reset_apply_in_flight`], for the tests that pin the ownership rule above
/// (`tests/compile_budget_isolation.rs`). Not used on any hot path.
pub fn apply_in_flight_bytes() -> u64 {
    APPLY_LIMITS.with(|l| l.budget_in_flight.get())
}

/// Output pairs this apply has built so far — the read side of
/// the per-apply output-pair meter. Used by the give-up rule's trace line to
/// report what the cut actually saw (`built=<pairs>`), and by the tests that
/// pin the rope's eligibility test. Not on any hot path.
#[doc(hidden)]
pub fn apply_pairs_in_flight() -> u64 {
    APPLY_LIMITS.with(|l| l.pairs_in_flight.get())
}

/// Current per-thread cumulative-output-node cap, or `None` when unset.
/// Cheap (one TLS read); the per-level loop only sums output levels when this
/// returns `Some`, so the no-cap hot path is unchanged. Also the read side of
/// [`ApplyLimitsInstall::output_cap`], for the callers whose tests pin that a
/// scope really did arm the cap it says it arms — the same role
/// [`apply_in_flight_bytes`] plays for the byte meter.
pub fn apply_output_node_cap() -> Option<u64> {
    APPLY_LIMITS.with(|l| l.output_node_cap.get())
}

/// Current per-thread stall rope, or `None` when unset. The enforcement itself
/// is in the poll gate, which reads the cell alongside the deadline it
/// already reads; this is the read side of [`ApplyLimitsInstall::stall_rope`],
/// for the tests that pin what a scope armed.
pub fn apply_stall_rope() -> Option<(u64, RopeLimit)> {
    APPLY_LIMITS.with(|l| l.stall_rope.get())
}

/// Builder for installing apply limits (deadline / budget / output-node cap /
/// stall rope) for a lexical scope. Axes not named (the method never called) are
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
    budget: Option<Option<u64>>,
    output_cap: Option<Option<u64>>,
    stall_rope: Option<Option<(u64, RopeLimit)>>,
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
    /// [`apply_stall_rope`], the read side of this knob.
    #[inline]
    pub fn stall_rope(mut self, r: Option<(u64, RopeLimit)>) -> Self {
        self.stall_rope = Some(r);
        self
    }

    /// Install every named axis (snapshotting its prior value) and return the
    /// RAII guard that restores them on drop.
    #[must_use = "the guard restores the prior apply limits when dropped; bind it to a name"]
    #[inline]
    pub fn apply(self) -> ApplyLimitsGuard {
        APPLY_LIMITS.with(|l| ApplyLimitsGuard {
            deadline: self.deadline.map(|d| l.deadline.replace(d)),
            // Touches `budget_remaining` only — NOT `budget_in_flight`, which
            // is reset separately, at apply entry.
            budget: self.budget.map(|b| l.budget_remaining.replace(b)),
            output_cap: self.output_cap.map(|c| l.output_node_cap.replace(c)),
            stall_rope: self.stall_rope.map(|r| l.stall_rope.replace(r)),
        })
    }
}

/// RAII guard returned by [`ApplyLimitsInstall::apply`]. Restores exactly the
/// axes that were named on the builder (the others are `None` here and are
/// left alone) — panic-safe by construction, including on an unwind caught by
/// the recovery cascade's `catch_unwind`.
#[must_use = "the guard restores the prior apply limits when dropped; bind it to a name"]
#[doc(hidden)]
pub struct ApplyLimitsGuard {
    deadline: Option<Option<std::time::Instant>>,
    budget: Option<Option<u64>>,
    output_cap: Option<Option<u64>>,
    stall_rope: Option<Option<(u64, RopeLimit)>>,
}

impl Drop for ApplyLimitsGuard {
    #[inline]
    fn drop(&mut self) {
        APPLY_LIMITS.with(|l| {
            if let Some(prior) = self.deadline {
                l.deadline.set(prior);
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
        });
    }
}

/// The per-level-boundary cut check, in fixed order: (gated) wall deadline,
/// then output-node cap. Called once per vtree level by
/// `apply_and_fallible_inner`. Order and short-circuit behavior are
/// load-bearing — do not reorder.
///
/// `out_nodes_so_far` / `live_counts` are the running output-node-cap inputs
/// (the full re-sum is only formed, for the `debug_assert`, when the cap is
/// armed).
pub(super) fn check_level_boundary(
    out_nodes_so_far: u64,
    live_counts: &[usize],
) -> Result<(), ApplyError> {
    // Per-iteration `ApplyLimits::deadline` poll, gated by the
    // `enable_apply_deadline_check()` override (default-OFF). This is the ONLY
    // wall-deadline cut for the per-level apply orchestration
    // (plan/finalize). Without it a
    // stuck branch grinds wide levels for minutes/hours and conditioning can
    // never deepen it. When the env var is unset `apply_deadline_expired()`
    // folds to `false` (OnceLock) → byte-identical hot path, so the
    // submission's no-env default is unchanged.
    if apply_deadline_expired() {
        return Err(ApplyError::Deadline);
    }

    // Output-size bail: when the segment-conjoin driver arms an output-node
    // cap (`ApplyLimits::output_node_cap`), abort the moment the output nodes built
    // so far exceed the cap. This bounds the conjunction's RESULT — a product
    // whose output balloons far past the segment's intended size is doomed
    // and not worth grinding to the wall deadline or the looser byte budget.
    // `out_nodes_so_far` is the running `sum(live_counts)`, maintained in O(1)
    // by `bump_live_count` at every build path (was an O(levels²) re-sum here);
    // the `debug_assert_eq!` cross-checks it against the full sum. Gated behind
    // the cap being `Some`, so the no-env hot path pays one TLS read only.
    if let Some(cap) = apply_output_node_cap() {
        debug_assert_eq!(
            out_nodes_so_far,
            live_counts.iter().map(|&c| c as u64).sum::<u64>(),
            "out_nodes_so_far desynced from live_counts sum — a live_counts \
             write bypassed bump_live_count",
        );
        if out_nodes_so_far > cap {
            return Err(ApplyError::OutputCap);
        }
    }

    Ok(())
}

/// Threshold (in scatter iterations = `c1.pairs.len() * c2.pairs.len()`)
/// above which the dense path runs the emit-growth mode decision
/// (`decide_emit_growth_mode`): compare the worst-case Vec-doubling transient
/// of the level's pair-product bound against the byte headroom and arm the
/// bounded-increment growth mode when it doesn't provably fit.
///
/// Fixed at 128 M iterations. Below this, doubling pays ~hundreds of MiB of
/// transient peak — acceptable — so small levels skip the decision entirely
/// and never pay the `apply_headroom_bytes_or_vas()` read (whose VAS fallback
/// does a ~µs `mapped_bytes()` epoch-advance read). Above, doubling from cap
/// N→2N transients 3N (e.g. canary 131's 1.25 G→2.5 G entries = 30 GiB peak,
/// trips the 31 GiB MCC cap) — exactly what the bounded mode protects.
pub(super) const DENSE_GROWTH_DECISION_THRESHOLD: usize = 128 * 1024 * 1024;

#[cfg(test)]
#[path = "budget_headroom_tests.rs"]
mod headroom_tests;

#[cfg(test)]
#[path = "budget_bounded_growth_tests.rs"]
mod bounded_growth_tests;

#[cfg(test)]
#[path = "budget_recovery_hint_gate_tests.rs"]
mod recovery_hint_gate_tests;

#[cfg(test)]
mod stall_rope_tests {
    use super::{
        account_output_pairs, apply_deadline_check_enabled, apply_deadline_expired, apply_limits,
        apply_pairs_in_flight, apply_stall_rope, charge_apply_in_flight_for_test,
        charge_compile_work, check_level_boundary, compile_work_units, enable_apply_deadline_check,
        reset_apply_deadline_check_for_test, reset_apply_in_flight, settle_output_pairs,
        ApplyError, RopeLimit, APPLY_LIMITS,
    };
    use std::time::{Duration, Instant};

    fn reset_pairs() {
        APPLY_LIMITS.with(|l| {
            l.pairs_in_flight.set(0);
            l.pairs_level_charge.set(0);
        });
    }

    /// **The stall rope needs BOTH halves, and it is asked at the INTRA-LEVEL
    /// poll — not only at the level boundary.**
    ///
    /// It is the give-up rule's answer to a step whose inputs were small and
    /// whose output is not: the caller arms the rule's own pair floor — in
    /// PAIRS, the unit that floor is stated in — as the trigger, and the step's
    /// usual share of the wall as the rope. The meter is `pairs_in_flight`,
    /// which is what lets the check ride `limits_reached` — the poll the wall
    /// deadline already uses from inside the dense loops. An apply that never
    /// finishes a level never reaches a level boundary, and that is precisely
    /// the apply this is for.
    #[test]
    fn a_stall_rope_cuts_at_the_intra_level_poll_once_built_and_out_of_time() {
        let was_armed = apply_deadline_check_enabled();
        enable_apply_deadline_check();
        reset_pairs();
        let spent = RopeLimit::Wall(Instant::now() - Duration::from_secs(1));
        let unspent = RopeLimit::Wall(Instant::now() + Duration::from_secs(60));

        // Nothing armed: the poll is inert, whatever the apply has built.
        account_output_pairs(1 << 20);
        assert!(apply_stall_rope().is_none());
        assert!(!apply_deadline_expired());
        assert!(check_level_boundary(0, &[]).is_ok());

        reset_pairs();
        {
            let _g = apply_limits().stall_rope(Some((1_000, spent))).apply();
            assert_eq!(apply_stall_rope(), Some((1_000, spent)));
            // Share spent, output still under the floor: this is a step whose
            // long run is search, which is the whole reason the rule has a size
            // factor at all. It is not cut.
            account_output_pairs(999);
            assert!(!apply_deadline_expired(), "under the floor, not at it");
            // At the floor with the share spent: a big diagram, and out of time.
            account_output_pairs(1);
            assert_eq!(apply_pairs_in_flight(), 1_000);
            assert!(apply_deadline_expired());
            // …and the level boundary reports it as what it is, because it asks
            // that same poll rather than keeping a second meter of its own.
            assert!(matches!(check_level_boundary(0, &[]), Err(ApplyError::Deadline)));
        }

        {
            // Built past the floor, but the share it was given is still on the
            // clock — the floor moves who is ELIGIBLE, never when the rope falls.
            let _g = apply_limits().stall_rope(Some((1_000, unspent))).apply();
            assert!(!apply_deadline_expired());
        }

        // …and the guard put the axis back, so nothing outside the step it
        // belonged to can be cut by it.
        assert!(apply_stall_rope().is_none());
        assert!(!apply_deadline_expired());
        reset_pairs();
        if !was_armed {
            reset_apply_deadline_check_for_test();
        }
    }

    /// **A step that is heavy in BYTES but light in built PAIRS is not
    /// eligible.**
    ///
    /// The floor asks how big a diagram this step holds. `budget_in_flight`
    /// answers a different question — how much memory the apply has claimed,
    /// including grids, live scratch and hoisted transients — and metering the
    /// rule with it made a tiny node that touched 8 MB of scratch look like a
    /// million-pair diagram. That is the miss this pins: the byte meter is over
    /// any plausible floor, the pair meter is one pair under, and the rope holds.
    /// **A level's unused arena slack does not survive the level.**
    ///
    /// Within a level the meter can only estimate — it reads the arena's
    /// CAPACITY, which is seeded up to `LEVEL_RESERVE_PAIRS_CAP` before a
    /// single pair is emitted. Left standing, that seed would accumulate once
    /// per level over the thousands one apply walks, and the meter would grow
    /// with the LEVEL COUNT instead of the diagram. So each boundary swaps the
    /// estimate for the truth.
    #[test]
    fn a_finished_level_contributes_its_pairs_not_its_capacity() {
        reset_pairs();
        // Level 1: seeded for 8192, emitted 3.
        account_output_pairs(8192);
        assert_eq!(apply_pairs_in_flight(), 8192, "in flight, capacity is all there is");
        settle_output_pairs(3);
        assert_eq!(apply_pairs_in_flight(), 3);
        // Level 2 does the same: the slack does not compound.
        account_output_pairs(8192);
        settle_output_pairs(3);
        assert_eq!(apply_pairs_in_flight(), 6, "two low-survival levels, six pairs");
        // A level that genuinely builds keeps what it built.
        account_output_pairs(2_000_000);
        settle_output_pairs(1_500_000);
        assert_eq!(apply_pairs_in_flight(), 1_500_006);
        reset_pairs();
    }

    #[test]
    fn bytes_are_not_pairs_a_big_apply_that_built_little_is_not_cut() {
        let was_armed = apply_deadline_check_enabled();
        enable_apply_deadline_check();
        reset_apply_in_flight();
        reset_pairs();
        let spent = RopeLimit::Wall(Instant::now() - Duration::from_secs(1));

        let _g = apply_limits().stall_rope(Some((1_000_000, spent))).apply();
        // A gigabyte of charged apply memory, and a diagram of 999_999 pairs.
        charge_apply_in_flight_for_test(1 << 30);
        account_output_pairs(999_999);
        assert!(
            !apply_deadline_expired(),
            "eligibility is the pair count, not the byte count",
        );
        // One more pair — the same rope, now genuinely met.
        account_output_pairs(1);
        assert!(apply_deadline_expired());

        drop(_g);
        reset_apply_in_flight();
        reset_pairs();
        if !was_armed {
            reset_apply_deadline_check_for_test();
        }
    }

    /// **A work rope falls on the work clock and on nothing else.**
    ///
    /// The whole point of the currency: no deadline is installed, no time passes,
    /// and the cut happens anyway — because the compile did the work. Its
    /// counterpart is the wall rope above, which the clock decides and charging
    /// cannot move.
    #[test]
    fn a_work_rope_falls_on_the_work_clock_and_not_on_the_wall() {
        let was_armed = apply_deadline_check_enabled();
        enable_apply_deadline_check();
        reset_pairs();

        let stride = 1u64 << 20;
        let at = compile_work_units().saturating_add(4 * stride);
        let _g = apply_limits().stall_rope(Some((0, RopeLimit::Work(at)))).apply();
        assert_eq!(apply_stall_rope(), Some((0, RopeLimit::Work(at))));
        // Nothing has a wall here: an unbudgeted apply, and a rope that is not
        // due. The floor is zero, so what holds the rope back is the clock alone.
        assert!(!apply_deadline_expired(), "a work rope fell before its work was done");
        charge_compile_work(3 * stride);
        assert!(!apply_deadline_expired(), "one stride short is short");
        charge_compile_work(stride);
        assert!(apply_deadline_expired());
        assert!(matches!(check_level_boundary(0, &[]), Err(ApplyError::Deadline)));

        drop(_g);
        assert!(!apply_deadline_expired(), "the guard left a work rope armed");
        reset_pairs();
        if !was_armed {
            reset_apply_deadline_check_for_test();
        }
    }
}
