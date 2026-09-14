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
/// two operand negations skip minimization; the conjunction's result and the
/// final complement are minimized. Each negation fills its operand out to full
/// structure first, so this can grow the diagram — see the module doc.
///
/// # Panics
/// Panics on invalid operands or allocation refusal; [`Engine::or`] returns the error.
pub(crate) fn apply_or(f: Tdd, g: Tdd) -> Tdd {
    Engine::new()
        .or(f, g)
        .expect("apply_or: operation refused; use Engine::or to handle errors")
}

/// Fallible [`apply_or`]: the same disjunction, with the memory refusal handed
/// back instead of panicked on.
///
/// # Errors
///
/// Returns the conjunction's or a minimization's [`OperationError`] — a refused
/// buffer reservation (allocator failure or the configured soft budget), the
/// output-node cap, or the scoped apply deadline.
pub(crate) fn disjoin_owned(eng: &Engine, mut f: Tdd, mut g: Tdd) -> Result<Tdd, OperationError> {
    use crate::apply::conjoin::conjoin_owned;

    crate::apply::check_vtree(&f, &g)?;
    crate::apply::prepare_weights([&mut f, &mut g])?;
    f.require_structure()?;
    g.require_structure()?;
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
    /// Return the disjunction of two structural diagrams sharing a vtree allocation.
    ///
    /// Both operands are consumed on success and on error. The result is minimized,
    /// except that a false operand returns the other operand without minimization.
    /// Weight compatibility and inheritance follow [`Engine::and`].
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{Engine, Vtree};
    ///
    /// let engine = Engine::new();
    /// let tree = Arc::new(Vtree::balanced(3));
    /// let first_two = engine.cube(&tree, [1, 2])?;
    /// let third = engine.literal(&tree, 3)?;
    /// let f = engine.or(first_two, third)?;
    /// assert_eq!(engine.model_count(&f)?, 5u32.into());
    /// # Ok::<(), tididi::OperationError>(())
    /// ```
    ///
    /// Disjunction uses complementation and conjunction, so intermediate diagrams
    /// can be larger than either operand.
    ///
    /// # Errors
    ///
    /// [`OperationError::VtreeMismatch`] for different vtree allocations,
    /// [`OperationError::IncompatibleWeights`] for different weight interpretations,
    /// or [`OperationError::MarginalLevel`] if either operand has discarded structure,
    /// even when the other operand is false. Allocation, output-cap, and stop
    /// refusals propagate from the component operations.
    pub fn or(&self, f: Tdd, g: Tdd) -> Result<Tdd, OperationError> {
        crate::apply::disjoin::disjoin_owned(self, f, g)
    }
}

#[cfg(test)]
mod tests;
