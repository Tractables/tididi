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
mod falsity;
pub(crate) mod project;
pub(crate) mod restrict_to_care;
mod filter_nodes;
mod operators;
mod compose;
mod substitute;

pub(crate) use conjoin::apply_and;
pub use conjoin::and;
pub(crate) use disjoin::apply_or;
pub use disjoin::{nor_many, or, or_many};
pub use compose::{xor, ite, and_exists, Quantification};
pub use restrict_to_care::RestrictionOutcome;
pub use filter_nodes::{FilterOutcome, FilterStats};

#[cfg(test)]
mod tests;

/// Require both operands to share their vtree allocation before reading either one.
pub(crate) fn check_vtree(f: &crate::Tdd, g: &crate::Tdd) -> Result<(), crate::OperationError> {
    if !std::sync::Arc::ptr_eq(f.vtree(), g.vtree()) {
        return Err(crate::OperationError::VtreeMismatch);
    }
    Ok(())
}

/// Require every operand to share the first one's vtree allocation.
pub(crate) fn check_same_vtree(operands: &[crate::Tdd]) -> Result<(), crate::OperationError> {
    let Some((first, rest)) = operands.split_first() else { return Ok(()) };
    rest.iter().try_for_each(|g| check_vtree(first, g))
}

/// Validate one weight interpretation for all operands, then install it where absent.
pub(crate) fn prepare_weights<T: std::borrow::BorrowMut<crate::Tdd>>(operands: &mut [T]) -> Result<(), crate::OperationError> {
    let Some(source) = operands.iter().position(|f| f.borrow().weights.is_some()) else { return Ok(()) };
    let (before, rest) = operands.split_at_mut(source);
    let (source, after) = rest.split_first_mut().unwrap();
    let weights = source.borrow().weights.as_ref().unwrap();
    for operand in before.iter().chain(after.iter()) {
        let f = operand.borrow();
        match &f.weights {
            Some(other) if !weights.compatible(other) => return Err(crate::OperationError::IncompatibleWeights),
            None if f.has_marginal_level() => return Err(crate::OperationError::IncompatibleWeights),
            _ => {}
        }
    }
    for operand in before.iter_mut().chain(after.iter_mut()) {
        let f = operand.borrow_mut();
        if f.weights.is_none() { f.weights = Some(weights.empty_like()); }
    }
    Ok(())
}

/// The pair of a ⊤ node: the constant-true node sits at local index 0 of every
/// level, leaf or internal, so both sides name it.
pub(crate) const TRUE_PAIR: crate::diagram::ChildPair = crate::diagram::ChildPair {
    left: crate::diagram::EncodedChildRef::from_raw(crate::diagram::ONE_LEAF_IDX.0),
    right: crate::diagram::EncodedChildRef::from_raw(crate::diagram::ONE_LEAF_IDX.0),
};

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
