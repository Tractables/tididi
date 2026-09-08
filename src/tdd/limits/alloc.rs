//! Fallible allocation: the tracked reserves, the byte accounting they feed,
//! and the push/resize wrappers built on them.

use super::error::ApplyError;
use super::memory::mem_preflight_alloc;
use super::{APPLY_LIMITS, LAST_REFUSED_RESERVE_BYTES};

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
///      to [`ApplyLimits::budget_in_flight`] (reset at apply entry).
///   3. If the updated in-flight total exceeds [`ApplyLimits::budget_remaining`]
///      (set per-step by `set_apply_budget`), returns `OverBudget`.
///
/// This is what lets the soft trigger fire *during* a conjoin instead
/// of only at the apply-entry predictive check or the OS-OOM backstop.
#[inline(always)]
pub(crate) fn budget_reserve_exact<T>(v: &mut Vec<T>, additional: usize) -> Result<(), ApplyError> {
    let pre_cap = v.capacity();
    // Release notice only on actual growth: a zero-byte notice is the host's
    // apply-entry heartbeat, throttled separately.
    if additional > v.capacity() - v.len() {
        mem_preflight_alloc(
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
    // `additional`, so the notice carries the doubled estimate.
    if additional > v.capacity() - v.len() {
        mem_preflight_alloc(
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
/// thread-local and trips `OverBudget` if it crosses [`ApplyLimits::budget_remaining`].
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

/// Release a TRANSIENT's bytes from the in-flight accounting when the backing
/// allocation is actually freed. [`ApplyLimits::budget_in_flight`] is otherwise monotone
/// within one apply (persistent structures — nodes/pairs/ext/grids — only
/// grow, and the counter resets at apply entry), so a per-level scratch that
/// charged itself via `budget_reserve(_exact)` and then drops mid-apply must
/// un-charge here or it permanently consumes soft headroom it no longer uses.
/// Only pair with a real free of the exact accounted capacity (see
/// `conjoin::cell::PreparedC2`'s Drop); never call for still-live allocations.
#[inline]
pub(crate) fn unaccount_transient_bytes(bytes: u64) {
    if bytes == 0 { return; }
    APPLY_LIMITS.with(|l| l.budget_in_flight.set(l.budget_in_flight.get().saturating_sub(bytes)));
}

/// Remaining apply-byte headroom for the current `apply_and_fallible` call:
/// `budget_remaining − budget_in_flight` (saturating). `None` when no soft
/// budget is installed. Mirrors the `total > rem` test in `account_capacity_delta`
/// — a single reserve of ≤ this many bytes is guaranteed not to trip the soft
/// trigger. Consumed (through [`apply_headroom_bytes_or_vas`]) by
/// `decide_emit_growth_mode` and `grow_pairs_bounded`'s increment policy.
#[inline]
pub(crate) fn apply_budget_headroom_bytes() -> Option<u64> {
    APPLY_LIMITS.with(|l| {
        l.budget_remaining.get().map(|rem| rem.saturating_sub(l.budget_in_flight.get()))
    })
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
/// skipped its `mem_preflight_alloc` (guarded on `1 > cap − len`),
/// `try_reserve(1)` found `needs_to_grow == false`, and
/// `account_capacity_delta` saw `delta == 0` and charged nothing. Splitting
/// them is a codegen fix, not a semantic one: the reserve path's
/// TLS-indirect preflight probe and its TLS accounting store are joins
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
