//! Disjunction (OR) of two TDDs.
//!
//! Implemented by De Morgan over the sibling `unary::negate` complement and
//! `conjoin`'s AND: `f ∨ g = ¬(¬f ∧ ¬g)`. `apply_or` shares the make-full work
//! across the three negations so only the two boundary results are minimized.
//!
//! **Cost.** Negation fills a diagram out to its full structure before
//! complementing it, which can grow it substantially; a disjunction runs three
//! such fills, so `|` is the expensive operator here, not the cheap one.

use crate::diagram::*;
use crate::limits::ApplyError;
use crate::apply::negate::{negate_tdd, negate_tdd_owned};

/// Disjunction by De Morgan: `f ∨ g = ¬(¬f ∧ ¬g)`.
///
/// The two operand negations skip minimization, because the conjunction between
/// them canonicalizes its output anyway; only the intermediate and the final
/// negation are minimized. Each negation fills its operand out to full
/// structure first, so this can grow the diagram — see the module doc.
pub fn apply_or(f: &Tdd, g: &Tdd) -> Tdd {
    use crate::apply::conjoin::apply_and;

    if f.is_zero() { return g.clone(); }
    if g.is_zero() { return f.clone(); }

    // Negate without minimize — the AND step handles canonicalization.
    let not_f = negate_tdd(f);
    let not_g = negate_tdd(g);

    // AND the raw negations.
    let mut and_result = apply_and(not_f, not_g);
    crate::reduce::minimize(&mut and_result);

    // Final negation + minimize. `and_result` is a local we own and discard, so
    // negate it owned — eliding the defensive full-diagram clone `negate_tdd`
    // would make.
    let mut result = negate_tdd_owned(and_result);
    crate::reduce::minimize(&mut result);
    result
}

/// Fallible [`apply_or`]: the same disjunction, with the memory refusal handed
/// back instead of panicked on.
///
/// # Errors
///
/// Returns the underlying conjunction's [`ApplyError`] — a refused buffer
/// reservation (allocator failure or the configured soft budget), the
/// output-node cap, or the scoped apply deadline.
pub fn try_apply_or(f: &Tdd, g: &Tdd) -> Result<Tdd, ApplyError> {
    if f.is_zero() {
        return Ok(g.clone());
    }
    if g.is_zero() {
        return Ok(f.clone());
    }
    // The clones are the ones `apply_or`'s borrowed negations make anyway.
    try_apply_or_owned(f.clone(), g.clone())
}

/// Owned-operand [`apply_or`]: consumes `f` and `g`, eliding the two full-diagram
/// operand clones the borrowed form makes inside `negate_tdd`. Use when the caller
/// holds the only references and discards both operands after disjoining (e.g. the
/// ∃-forget cofactor fold in `project_var` / the concat merge). Same result as
/// `apply_or(&f, &g)`, just without copying either operand.
pub(crate) fn apply_or_owned(f: Tdd, g: Tdd) -> Tdd {
    use crate::limits::apply_limits;

    let _shield = apply_limits().deadline(None).apply();
    try_apply_or_owned(f, g).expect(
        "apply_or_owned: allocator OOM in infallible entry — use try_apply_or_owned to recover",
    )
}

/// Fallible `apply_or_owned`: the same disjunction, with the memory refusal
/// handed back instead of panicked on.
///
/// The infallible entry above is this function under a deadline shield plus an
/// `expect` — one implementation, two contracts, the same pairing
/// `apply_and` / `try_apply_and` already has on the AND
/// side. A caller that drives the apply primitives directly and owns its own
/// give-up policy (the grove driver's DPLL TDD fold, which disjoins the two sides
/// of every branch node) needs the `Err`: a panic there would land in the
/// cascade's panic-as-control-flow recovery, which that driver is specified never
/// to reach.
///
/// # Errors
///
/// Returns the conjunction's [`ApplyError`] — a refused buffer reservation
/// (allocator failure or the configured soft budget), the output-node cap, or
/// the scoped apply deadline.
pub(crate) fn try_apply_or_owned(f: Tdd, g: Tdd) -> Result<Tdd, ApplyError> {
    use crate::apply::conjoin::try_apply_and;

    if f.is_zero() { return Ok(g); }
    if g.is_zero() { return Ok(f); }

    // Negate without minimize — the AND step handles canonicalization.
    let not_f = negate_tdd_owned(f);
    let not_g = negate_tdd_owned(g);

    let mut and_result = try_apply_and(not_f, not_g, None)?;
    crate::reduce::minimize(&mut and_result);

    let mut result = negate_tdd_owned(and_result);
    crate::reduce::minimize(&mut result);
    Ok(result)
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "disjoin_tests.rs"]
mod tests;
