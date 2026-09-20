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
//! The [API overview](crate::guide::api) connects circuit operations with worked examples.

pub(crate) mod conjoin;
pub(crate) mod conjoin_clause;
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
pub(crate) use disjoin::apply_or;
pub use disjoin::{or, or_many};
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

/// Static 3×3 conjunction grid for implicit leaf product.
///
/// `CONJOIN_GRID[i][j]` = output label index when conjoining leaf label `i`
/// with leaf label `j`, or `u32::MAX` if the conjunction is Zero.
///
/// ```text
///        j=One(0)  j=Pos(1)  j=Neg(2)
/// i=One:    0         1        2
/// i=Pos:    1         1       Zero
/// i=Neg:    2        Zero     2
/// ```
pub(crate) const CONJOIN_GRID: [[u32; 3]; 3] = [
    [0,    1,    2   ],  // One ∧ {One, Pos, Neg}
    [1,    1,    conjoin::budget::NO_PRODUCT],  // Pos ∧ {One, Pos, Neg}
    [2,    conjoin::budget::NO_PRODUCT, 2   ],  // Neg ∧ {One, Pos, Neg}
];
