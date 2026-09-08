//! Disjunction (OR) of two TDDs.
//!
//! Implemented by De Morgan over the sibling `unary::negate` complement and
//! `conjoin`'s AND: `f v g = !(!f ^ !g)`. `apply_or` shares the make-full work
//! across the three negations so only the two boundary results are minimized.
//!
//! **Cost.** Negation fills a diagram out to its full structure before
//! complementing it, which can grow it substantially; a disjunction runs three
//! such fills, so `|` is the expensive operator here, not the cheap one.

use crate::engine::Limits;
use crate::diagram::*;
use crate::error::ApplyError;
use crate::apply::negate::negate_tdd_owned;

/// Disjunction by De Morgan: `f v g = !(!f ^ !g)`.
///
/// Consumes both operands, as [`apply_and`](crate::apply::apply_and) does. The
/// two operand negations skip minimization, because the conjunction between
/// them canonicalizes its output anyway; only the intermediate and the final
/// negation are minimized. Each negation fills its operand out to full
/// structure first, so this can grow the diagram — see the module doc.
///
/// # Panics
/// Panics if the conjunction runs out of memory. Use [`try_apply_or`] to
/// recover from that instead.
pub fn apply_or(f: Tdd, g: Tdd) -> Tdd {
    let lim = Limits::new();
    try_apply_or(&lim, f, g)
        .expect("apply_or: allocator OOM in infallible entry — use try_apply_or to recover")
}

/// Fallible [`apply_or`]: the same disjunction, with the memory refusal handed
/// back instead of panicked on.
///
/// The infallible entry above is this function on unarmed limits plus an
/// `expect` — one implementation, two contracts, the same pairing
/// `apply_and` / `try_apply_and` already has on the AND side. A caller that
/// drives the apply primitives directly and owns its own give-up policy (the
/// grove driver's DPLL TDD fold, which disjoins the two sides of every branch
/// node) needs the `Err`: a panic there would land in the cascade's
/// panic-as-control-flow recovery, which that driver is specified never to
/// reach.
///
/// # Errors
///
/// Returns the conjunction's [`ApplyError`] — a refused buffer reservation
/// (allocator failure or the configured soft budget), the output-node cap, or
/// the scoped apply deadline.
pub fn try_apply_or(lim: &Limits, f: Tdd, g: Tdd) -> Result<Tdd, ApplyError> {
    use crate::apply::try_apply_and;

    if f.is_zero() { return Ok(g); }
    if g.is_zero() { return Ok(f); }

    // Negate without minimize — the AND step handles canonicalization.
    let not_f = negate_tdd_owned(f);
    let not_g = negate_tdd_owned(g);

    let mut and_result = try_apply_and(lim, not_f, not_g, None)?;
    crate::reduce::minimize(&mut and_result);

    let mut result = negate_tdd_owned(and_result);
    crate::reduce::minimize(&mut result);
    Ok(result)
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "disjoin_tests.rs"]
mod tests;
