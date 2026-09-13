//! Combine and transform Boolean functions.
//!
//! Use [`Engine::and`](crate::Engine::and), [`Engine::or`](crate::Engine::or),
//! [`Engine::negate`](crate::Engine::negate), [`Engine::xor`](crate::Engine::xor)
//! and [`Engine::ite`](crate::Engine::ite) to compose functions.
//! [`Engine::condition`](crate::Engine::condition) fixes an assignment;
//! [`Engine::exists_vars`](crate::Engine::exists_vars) quantifies variables;
//! [`Engine::substitute`](crate::Engine::substitute) replaces them with functions.
//! [`Engine::rename_vars`](crate::Engine::rename_vars) handles swaps and renames,
//! and [`Engine::and_exists`](crate::Engine::and_exists) constructs an existential conjunction.
//!
//! These methods use a caller's engine and return errors under its limits.
//! The `&`, `|` and `!` operators use a temporary engine and panic on refusal.
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
pub use conjoin_clause::apply_and_clause;
pub(crate) use disjoin::apply_or;
pub use negate::negate;
pub use condition::{condition_var, condition_vars};
pub use project::{exists_var, exists_vars, QuantificationStrategy};
pub use restrict_to_care::{restrict_to_care, RestrictionOutcome};

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

/// Give structural operands one compatible weight interpretation before shortcuts or merging.
pub(crate) fn prepare_weights(f: &mut crate::Tdd, g: &mut crate::Tdd) -> Result<(), crate::OperationError> {
    match (f.weights.as_ref(), g.weights.as_ref()) {
        (Some(a), Some(b)) => {
            if !a.compatible(b) { return Err(crate::OperationError::IncompatibleWeights); }
        }
        (Some(weights), None) => {
            if g.has_marginal_level() { return Err(crate::OperationError::IncompatibleWeights); }
            g.weights = Some(weights.empty_like());
        }
        (None, Some(weights)) => {
            if f.has_marginal_level() { return Err(crate::OperationError::IncompatibleWeights); }
            f.weights = Some(weights.empty_like());
        }
        (None, None) => {}
    }
    Ok(())
}
