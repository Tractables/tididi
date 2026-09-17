//! Disjunction (OR) of two diagrams.
//!
//! Implemented by De Morgan over `negate` and `conjoin`: `f v g = !(!f ^ !g)`.
//! Each negation fills its operand out to full structure before complementing
//! it, and a disjunction runs three such fills, so `|` can grow a diagram
//! where `&` would not.

use crate::Engine;
use crate::diagram::*;
use crate::limits::OperationError;
use crate::apply::negate::negate_tdd_owned;
use crate::reduce::{ReductionPlan};

/// Panicking disjunction used by `BitOr` and test fixtures.
/// The checked entry point is [`or`].
pub(crate) fn apply_or(f: Tdd, g: Tdd) -> Tdd {
    or(f, g)
        .expect("apply_or: operation refused; use tididi::or to handle errors")
}

/// Disjoin owned operands by De Morgan, minimizing the product and final complement.
/// Operand and resource errors follow [`or`] and the engine's installed limits.
pub(crate) fn disjoin_owned(eng: &Engine, mut f: Tdd, mut g: Tdd) -> Result<Tdd, OperationError> {
    use crate::apply::conjoin::conjoin_owned;

    crate::apply::check_vtree(&f, &g)?;
    crate::apply::prepare_weights([&mut f, &mut g])?;
    f.require_structure()?;
    g.require_structure()?;
    let _op = eng.limits().begin_operation();
    eng.limits().check_stop()?;
    if f.is_zero() { return Ok(g); }
    if g.is_zero() { return Ok(f); }

    // Negate without minimize; the conjunction's result is minimized below.
    let not_f = negate_tdd_owned(eng, f)?;
    let not_g = negate_tdd_owned(eng, g)?;

    let mut and_result = conjoin_owned(eng, not_f, not_g, None)?;
    eng.reduce(&mut and_result, ReductionPlan::default())?;

    let mut result = negate_tdd_owned(eng, and_result)?;
    eng.reduce(&mut result, ReductionPlan::default())?;
    Ok(result)
}

/// Return the disjunction of two structural diagrams sharing a vtree allocation.
///
/// Uses the shared vtree's execution context automatically.
///
/// Both operands are consumed on success and on error. The result is minimized,
/// except that a false operand returns the other operand without minimization.
/// Weight compatibility and inheritance follow [`and`](crate::and).
///
/// ```
/// use std::sync::Arc;
/// use tididi::{literal, or, Tdd, Vtree};
///
/// let vtree = Arc::new(Vtree::balanced(3));
/// let first_two = Tdd::cube(&vtree, [1, 2])?;
/// let third = literal(&vtree, 3)?;
/// let f = or(first_two, third)?;
/// assert_eq!(f.model_count()?, 5u32.into());
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
/// even when the other operand is false. Allocation refusals propagate from
/// the component operations.
pub fn or(f: Tdd, g: Tdd) -> Result<Tdd, OperationError> {
    let context = std::sync::Arc::clone(f.context());
    context.run(|eng| eng.or(f, g))
}

impl crate::Engine {
    /// Run [`or`] using this batch's scratch and resource limits.
    ///
    /// # Errors
    ///
    /// Returns the operation's errors, plus [`OperationError::Stopped`] or
    /// [`OperationError::OutputCap`] when an installed limit refuses the work.
    pub fn or(&self, f: Tdd, g: Tdd) -> Result<Tdd, OperationError> {
        crate::apply::disjoin::disjoin_owned(self, f, g)
    }
}

#[cfg(test)]
#[path = "tests/disjoin/mod.rs"]
mod tests;
