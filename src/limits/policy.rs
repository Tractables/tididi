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

use crate::Engine;
use crate::limits::ApplyError;

/// Fallible-allocation strategy for a count or weight column's backing `Vec`s.
/// `reserve`/`reserve_exact` mirror `Vec::try_reserve`/`Vec::try_reserve_exact`
/// (amortized doubling vs. exact), mapped to the policy's own error type.
pub(crate) trait ReservePolicy {
    type Err;
    fn reserve<T>(eng: &Engine, v: &mut Vec<T>, additional: usize) -> Result<(), Self::Err>;
    fn reserve_exact<T>(eng: &Engine, v: &mut Vec<T>, additional: usize) -> Result<(), Self::Err>;
}

/// [`ReservePolicy`] for the in-apply streaming counts path: delegates to the
/// engine's tracked reserves, so a resize past the armed budget returns
/// `Err(ApplyError::OverBudget)` instead of allocating.
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

/// [`ReservePolicy`] for the marginalization scratch: a refused
/// `try_reserve`/`try_reserve_exact` leaves the heap at its pre-attempt level
/// and raises an ordinary unwinding panic, which a caller's `catch_unwind`
/// catches. It must stay an unwinding panic (no abort, no panic hook), since a
/// caller's recovery depends on catching it.
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

/// Unwrap a `Result` whose error type is uninhabited, so a `RecoveryPanic`
/// caller stays a one-liner.
#[inline]
pub(crate) fn unwrap_infallible<T>(r: Result<T, std::convert::Infallible>) -> T {
    match r {
        Ok(v) => v,
        Err(e) => match e {},
    }
}
