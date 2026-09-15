//! Combine and transform Boolean functions.
//!
//! Combine [`Tdd`](crate::Tdd) values with [`and`], [`or`], [`xor`] and [`ite`].
//! Each call reuses scratch from the vtree's execution context.
//!
//! [`Tdd::negate`](crate::Tdd::negate) complements a function.
//! [`Tdd::condition`](crate::Tdd::condition) fixes an assignment;
//! [`Tdd::exists_vars`](crate::Tdd::exists_vars) quantifies variables;
//! [`Tdd::substitute`](crate::Tdd::substitute) replaces them with functions.
//! [`Tdd::rename_vars`](crate::Tdd::rename_vars) handles swaps and renames,
//! and [`and_exists`] constructs an existential conjunction.
//!
//! These operations return errors; the `&`, `|` and `!` operators panic on refusal.
//! Use [`Context::with_limits`](crate::Context::with_limits) for a bounded batch.
//! Find a task and its examples in the [task guide](crate::guide::api).

pub(crate) mod conjoin;
pub(crate) mod scoped_flags;
pub(crate) mod conjoin_clause;
pub(crate) mod leaf;
mod grid;
pub(crate) mod disjoin;
pub(crate) mod negate;
pub(crate) mod condition;
pub(crate) mod project;
pub(crate) mod restrict_to_care;
mod operators;
mod compose;
mod substitute;

pub(crate) use conjoin::apply_and;
pub use conjoin::and;
pub use conjoin_clause::ClauseLiteral;
pub(crate) use disjoin::apply_or;
pub use disjoin::or;
pub use compose::{xor, ite, and_exists, and_exists_with_strategy};
pub use project::QuantificationStrategy;
pub use restrict_to_care::RestrictionOutcome;

#[cfg(test)]
mod tests;

/// Require both operands to share their vtree allocation before reading either one.
pub(crate) fn check_vtree(f: &crate::Tdd, g: &crate::Tdd) -> Result<(), crate::OperationError> {
    if !std::sync::Arc::ptr_eq(f.vtree(), g.vtree()) {
        return Err(crate::OperationError::VtreeMismatch);
    }
    Ok(())
}

/// Require a shared vtree and output level for a pairwise conjunction.
pub(crate) fn check_conjunction_operands(f: &crate::Tdd, g: &crate::Tdd) -> Result<(), crate::OperationError> {
    check_vtree(f, g)?;
    if f.output().vtree != g.output().vtree {
        return Err(crate::OperationError::RootMismatch);
    }
    Ok(())
}

/// Validate one weight interpretation for all operands, then install it where absent.
pub(crate) fn prepare_weights<const N: usize>(mut operands: [&mut crate::Tdd; N]) -> Result<(), crate::OperationError> {
    let Some(source) = operands.iter().position(|f| f.weights.is_some()) else { return Ok(()) };
    let (before, rest) = operands.split_at_mut(source);
    let (source, after) = rest.split_first_mut().unwrap();
    let weights = source.weights.as_ref().unwrap();
    for f in before.iter().chain(after.iter()) {
        match &f.weights {
            Some(other) if !weights.compatible(other) => return Err(crate::OperationError::IncompatibleWeights),
            None if f.has_marginal_level() => return Err(crate::OperationError::IncompatibleWeights),
            _ => {}
        }
    }
    for f in before.iter_mut().chain(after.iter_mut()) {
        if f.weights.is_none() { f.weights = Some(weights.empty_like()); }
    }
    Ok(())
}
