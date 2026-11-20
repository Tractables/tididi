//! The last step of every public operation.

use crate::{Engine, OperationError, Tdd};

/// The operation's outcome as it built it, minimized when the engine's limits
/// ask for canonical results ([`LimitConfig::with_canonical_results`](crate::limits::LimitConfig::with_canonical_results)).
/// Composite operations call it once, at their own exit. Taking the `Result`
/// whole lets it pass through unmoved when the flag is off: unwrapping the
/// diagram and wrapping it again copies the `Tdd` twice.
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
