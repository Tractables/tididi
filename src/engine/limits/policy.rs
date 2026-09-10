//! How a growing buffer asks for memory, as a type parameter.
//!
//! Two contexts allocate the same count and weight columns and want opposite
//! things from a refused allocation. Inside an apply the caller is running
//! under a byte budget it can back away from, so the refusal is a value:
//! [`ApplyBudget`] returns [`ApplyError::OverBudget`] and the operation
//! unwinds through its normal error path. After a compile, the marginal
//! cascade has no such path — so [`RecoveryPanic`] raises a controlled panic
//! the caller's recovery catches.
//!
//! The policy is a type parameter rather than a flag because the column types
//! are monomorphized on it: a buffer that cannot fail carries no `Result`
//! through its inner loop.

use crate::engine::Engine;
use crate::error::ApplyError;

/// Fallible-allocation strategy for a count or weight column's backing `Vec`s
/// — the one axis that intentionally differs between the in-apply and
/// finished-diagram contexts. `reserve`/`reserve_exact` mirror `Vec::try_reserve`/`Vec::try_reserve_exact`'s growth strategies
/// (amortized-doubling vs. exact), mapped to the policy's own error type.
pub(crate) trait ReservePolicy {
    type Err;
    fn reserve<T>(eng: &Engine, v: &mut Vec<T>, additional: usize) -> Result<(), Self::Err>;
    fn reserve_exact<T>(eng: &Engine, v: &mut Vec<T>, additional: usize) -> Result<(), Self::Err>;
}

/// [`ReservePolicy`] for the in-apply streaming counts path
/// (`conjoin::streaming_marginal`). Delegates to the soft apply-budget tracker in
/// `conjoin::budget` (armed as `LimitSet::budget_bytes`) — the one place it is read — so a
/// resize that would exceed the remaining envelope returns a cooperative
/// `Err(ApplyError::OverBudget)` instead of allocating. No accounting is
/// re-implemented here; both methods are pure delegation.
pub(crate) struct ApplyBudget;

impl ReservePolicy for ApplyBudget {
    type Err = ApplyError;

    #[inline(always)]
    fn reserve<T>(eng: &Engine, v: &mut Vec<T>, additional: usize) -> Result<(), Self::Err> {
        let lim = eng.limits();
        lim.reserve(v, additional)
    }

    #[inline(always)]
    fn reserve_exact<T>(eng: &Engine, v: &mut Vec<T>, additional: usize) -> Result<(), Self::Err> {
        let lim = eng.limits();
        lim.reserve_exact(v, additional)
    }
}

/// [`ReservePolicy`] for post-compile marginalization scratch
/// (`marginal`'s `marginalize_batch`/`ensure_counts` and friends).
///
/// On pathological levels a marginal-count buffer can require a single
/// 10–17 GiB allocation. The infallible `vec![0u128; width]` (or a plain
/// `.clone()`) invokes Rust's alloc-error handler on failure, which
/// **aborts** (SIGABRT, rc=-6) when the heap is at the `RLIMIT_AS` ceiling:
/// the unwind machinery's own allocation also fails, double-faulting past the
/// recovery cascade's `catch_unwind`.
///
/// `try_reserve`/`try_reserve_exact` instead return `Err` *without*
/// committing the allocation or touching the abort handler, leaving the heap
/// at its pre-attempt level. We then raise a controlled panic from normal
/// code, which unwinds cleanly into the `catch_unwind` of the caller's
/// memory-budget recovery path, triggering a Shannon-split retry instead of
/// killing the process. These
/// panics must remain ordinary unwinding panics — no abort, no panic hooks —
/// since recovery depends on catching them. Mirrors the already-fallible
/// apply-stream counts path (`ApplyBudget`, above), which instead maps the
/// same failure to a cooperative `Err`.
pub(crate) struct RecoveryPanic;

impl ReservePolicy for RecoveryPanic {
    type Err = std::convert::Infallible;

    fn reserve<T>(_eng: &Engine, v: &mut Vec<T>, additional: usize) -> Result<(), Self::Err> {
        if v.try_reserve(additional).is_err() {
            panic_over_budget::<T>(additional);
        }
        Ok(())
    }

    fn reserve_exact<T>(_eng: &Engine, v: &mut Vec<T>, additional: usize) -> Result<(), Self::Err> {
        if v.try_reserve_exact(additional).is_err() {
            panic_over_budget::<T>(additional);
        }
        Ok(())
    }
}

/// Raise the controlled recovery-split panic for `RecoveryPanic` (see its doc
/// comment). `#[cold]`/`#[inline(never)]` since this only ever runs on the
/// already-doomed over-budget branch.
#[cold]
#[inline(never)]
fn panic_over_budget<T>(additional: usize) -> ! {
    panic!(
        "count buffer of {additional} entries ({:.1} GiB) over budget — triggering recovery split",
        (additional as f64 * std::mem::size_of::<T>() as f64) / (1u64 << 30) as f64,
    );
}

