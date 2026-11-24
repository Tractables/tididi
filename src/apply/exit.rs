//! The last step of the public operations of `apply`.

use crate::{Engine, OperationError, Tdd};

/// The operation's outcome as it built it, minimized when the engine's limits
/// ask for canonical results ([`LimitConfig::with_canonical_results`](crate::limits::LimitConfig::with_canonical_results)).
/// A composite operation calls it at its own exit. One built on another
/// public entry passes through it there too, and that intermediate is then
/// minimized: `substitute` (a negated replacement), `and_restoring` of an
/// operand with a summed-out level, and the queries `implies`,
/// `projected_model_count` and `and_model_count`. Taking the `Result`
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
