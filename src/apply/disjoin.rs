//! Disjunction (OR) of two diagrams.
//!
//! Implemented by De Morgan over `negate` and `conjoin`: `f v g = !(!f ^ !g)`.
//! Each negation fills its operand out to full structure before complementing
//! it, and a disjunction runs three such fills, so `|` can grow a diagram
//! where `&` would not.

use crate::engine::Engine;
use crate::diagram::*;
use crate::limits::ApplyError;
use crate::apply::negate::negate_tdd_owned;
use crate::reduce::{try_minimize, MinimizeOptions};

/// Disjunction by De Morgan: `f v g = !(!f ^ !g)`.
///
/// Consumes both operands, as [`apply_and`](crate::apply::apply_and) does. The
/// two operand negations skip minimization, because the conjunction between
/// them canonicalizes its output anyway; only the intermediate and the final
/// negation are minimized. Each negation fills its operand out to full
/// structure first, so this can grow the diagram — see the module doc.
///
/// # Panics
/// Panics if an allocation is refused. Use `disjoin_owned` to recover from
/// that instead.
pub(crate) fn apply_or(f: Tdd, g: Tdd) -> Tdd {
    Engine::new()
        .or(f, g)
        .expect("apply_or: allocator OOM in infallible entry — use Engine::or to recover")
}

/// Fallible [`apply_or`]: the same disjunction, with the memory refusal handed
/// back instead of panicked on.
///
/// # Errors
///
/// Returns the conjunction's or a minimization's [`ApplyError`] — a refused
/// buffer reservation (allocator failure or the configured soft budget), the
/// output-node cap, or the scoped apply deadline.
pub(crate) fn disjoin_owned(eng: &Engine, f: Tdd, g: Tdd) -> Result<Tdd, ApplyError> {
    use crate::apply::conjoin::conjoin_owned;

    if f.is_zero() { return Ok(g); }
    if g.is_zero() { return Ok(f); }

    // Negate without minimize; the conjunction's result is minimized below.
    let not_f = negate_tdd_owned(f);
    let not_g = negate_tdd_owned(g);

    let mut and_result = conjoin_owned(eng, not_f, not_g, None)?;
    try_minimize(eng, &mut and_result, MinimizeOptions::default())?;

    let mut result = negate_tdd_owned(and_result);
    try_minimize(eng, &mut result, MinimizeOptions::default())?;
    Ok(result)
}

/// The disjunction entry point on a caller's engine.
impl crate::engine::Engine {
    /// Disjoin two diagrams over the same vtree, by De Morgan over
    /// [`Engine::and`].
    ///
    /// Both operands are consumed on `Err` as well as on `Ok`, as in
    /// [`Engine::and`]. Neither may have a marginal level: the negations have
    /// no structure to complement there. Each negation fills its operand out
    /// to full structure first, so this can grow the diagram. The result is
    /// canonical; a ⊥ operand returns the other operand as it is.
    ///
    /// # Errors
    ///
    /// As [`Engine::and`].
    ///
    /// # Panics
    ///
    /// As [`Engine::and`].
    ///
    /// ```
    /// # use std::sync::Arc;
    /// # use std::time::Instant;
    /// # use tididi::{ApplyError, Engine, Tdd};
    /// # use tididi::limits::LimitSet;
    /// # use tididi::vtree::Vtree;
    /// # let vtree = Arc::new(Vtree::balanced(4));
    /// let engine = Engine::new();
    /// let f = Tdd::clause(&vtree, [1]);
    /// let g = Tdd::clause(&vtree, [2]);
    /// let h = engine.or(f, g).expect("nothing is armed on a fresh engine");
    /// assert_eq!(h.model_count(), 12u32.into()); // x1 ∨ x2 over four variables
    ///
    /// let _armed = engine.limits().scope(LimitSet::none().deadline(Some(Instant::now())));
    /// let (f, g) = (Tdd::clause(&vtree, [1, -2]), Tdd::clause(&vtree, [2, 3]));
    /// match engine.or(f, g) {
    ///     Ok(_) => unreachable!("the deadline has passed"),
    ///     Err(e) => assert_eq!(e, ApplyError::Deadline),
    /// }
    /// ```
    pub fn or(&self, f: Tdd, g: Tdd) -> Result<Tdd, ApplyError> {
        crate::apply::disjoin::disjoin_owned(self, f, g)
    }
}

#[cfg(test)]
mod tests;
