//! The last step of every public operation.

use crate::{Engine, OperationError, Tdd};

/// `f` as the operation built it, minimized when the engine's limits ask for
/// canonical results ([`LimitConfig::with_canonical_results`](crate::limits::LimitConfig::with_canonical_results)).
/// Composite operations call it once, at their own exit.
#[inline(always)]
pub(crate) fn result(eng: &Engine, mut f: Tdd) -> Result<Tdd, OperationError> {
    if eng.limits().canonical_results() {
        minimize(eng, &mut f)?;
    }
    Ok(f)
}

/// [`result`] of an operation's outcome, passed through unchanged when the
/// flag is off, so the diagram is not moved out of the `Result` and back.
#[inline(always)]
pub(crate) fn outcome(eng: &Engine, r: Result<Tdd, OperationError>) -> Result<Tdd, OperationError> {
    if eng.limits().canonical_results() {
        return minimized(eng, r);
    }
    r
}

#[cold]
#[inline(never)]
fn minimized(eng: &Engine, r: Result<Tdd, OperationError>) -> Result<Tdd, OperationError> {
    let mut f = r?;
    eng.minimize(&mut f)?;
    Ok(f)
}

#[cold]
#[inline(never)]
fn minimize(eng: &Engine, f: &mut Tdd) -> Result<(), OperationError> {
    eng.minimize(f)
}
