//! Disjunction (OR) of two diagrams.
//!
//! Implemented by De Morgan over `negate` and `conjoin`: `f v g = !(!f ^ !g)`.
//! Each negation fills its operand out to full structure before complementing
//! it, and a disjunction runs three such fills, so `|` can grow a diagram
//! where `&` would not.

use crate::engine::Engine;
use crate::diagram::*;
use crate::limits::OperationError;
use crate::apply::negate::negate_tdd_owned;
use crate::reduce::{try_reduce, ReductionPlan};

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
/// Returns the conjunction's or a minimization's [`OperationError`] — a refused
/// buffer reservation (allocator failure or the configured soft budget), the
/// output-node cap, or the scoped apply deadline.
pub(crate) fn disjoin_owned(eng: &Engine, f: Tdd, g: Tdd) -> Result<Tdd, OperationError> {
    use crate::apply::conjoin::conjoin_owned;

    crate::apply::check_vtree(&f, &g)?;
    let _op = eng.limits().begin_operation();
    if f.is_zero() { return Ok(g); }
    if g.is_zero() { return Ok(f); }

    // Negate without minimize; the conjunction's result is minimized below.
    let not_f = negate_tdd_owned(eng, f)?;
    let not_g = negate_tdd_owned(eng, g)?;

    let mut and_result = conjoin_owned(eng, not_f, not_g, None)?;
    try_reduce(eng, &mut and_result, ReductionPlan::default())?;

    let mut result = negate_tdd_owned(eng, and_result)?;
    try_reduce(eng, &mut result, ReductionPlan::default())?;
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
    /// [`OperationError::VtreeMismatch`] before any work if the vtree allocations
    /// differ; allocation and stop errors propagate from the component operations.
    ///
    /// # Panics
    ///
    /// Panics when negation encounters a marginal level.
    ///
    /// ```
    /// # use std::sync::Arc;
    /// # use std::time::Instant;
    /// # use tididi::{OperationError, Engine, Tdd};
    /// # use tididi::limits::LimitConfig;
    /// # use tididi::vtree::Vtree;
    /// # let vtree = Arc::new(Vtree::balanced(4));
    /// let engine = Engine::new();
    /// let f = Tdd::clause(&vtree, [1]);
    /// let g = Tdd::clause(&vtree, [2]);
    /// let h = engine.or(f, g).expect("nothing is armed on a fresh engine");
    /// assert_eq!(h.model_count(), 12u32.into()); // x1 ∨ x2 over four variables
    ///
    /// let _armed = engine.limits().scope(LimitConfig::none().with_deadline(Some(Instant::now())));
    /// let (f, g) = (Tdd::clause(&vtree, [1, -2]), Tdd::clause(&vtree, [2, 3]));
    /// match engine.or(f, g) {
    ///     Ok(_) => unreachable!("the deadline has passed"),
    ///     Err(e) => assert_eq!(e, OperationError::Stopped),
    /// }
    /// ```
    pub fn or(&self, f: Tdd, g: Tdd) -> Result<Tdd, OperationError> {
        crate::apply::disjoin::disjoin_owned(self, f, g)
    }
}

#[cfg(test)]
mod tests;
