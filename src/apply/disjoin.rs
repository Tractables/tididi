//! Disjunction (OR) of diagrams.
//!
//! Implemented by De Morgan over `negate` and `conjoin`: `f v g = !(!f ^ !g)`.
//! Each negation fills its operand out to full structure before complementing
//! it, and a disjunction runs three such fills, so `|` can grow a diagram
//! where `&` would not.
//!
//! That is why [`or_many`] exists and why a caller with more than two operands
//! should reach for it. Folding [`or`] over `n` operands runs `3(n - 1)`
//! fills, two of every three on the accumulator, which is the largest diagram
//! present. One `!(!f_1 ^ ... ^ !f_n)` runs `n + 1`, none of them on a partial
//! disjunction, because the complement is postponed to the end.

use crate::Engine;
use crate::diagram::Tdd;
use crate::limits::OperationError;
use crate::apply::negate::negate_tdd_owned;
use crate::reduce::ReductionPlan;

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

/// Disjoin owned operands by De Morgan with one final complement.
///
/// `!f_1 ^ ... ^ !f_n` is built first and complemented once. The conjunction
/// folds as a balanced tree so that no single fold step meets the whole
/// accumulated conjunction.
pub(crate) fn disjoin_many_owned(eng: &Engine, operands: Vec<Tdd>) -> Result<Tdd, OperationError> {
    use crate::apply::conjoin::conjoin_owned;

    let _op = eng.limits().begin_operation();
    eng.limits().check_stop()?;

    // A false operand contributes no model; every operand false is false.
    let mut complements: Vec<Tdd> = Vec::with_capacity(operands.len());
    let mut a_false_one: Option<Tdd> = None;
    for mut f in operands {
        crate::apply::prepare_weights([&mut f])?;
        f.require_structure()?;
        if f.is_zero() {
            a_false_one = Some(f);
            continue;
        }
        complements.push(negate_tdd_owned(eng, f)?);
    }
    if complements.is_empty() {
        return a_false_one.ok_or(OperationError::EmptyOperands);
    }
    if complements.len() == 1 {
        let mut result = negate_tdd_owned(eng, complements.pop().expect("one operand"))?;
        eng.reduce(&mut result, ReductionPlan::default())?;
        return Ok(result);
    }
    // Balanced fold: a chain would touch the growing conjunction once per
    // operand and so cost the square of the operand count.
    while complements.len() > 1 {
        let mut next = Vec::with_capacity(complements.len().div_ceil(2));
        let mut it = complements.into_iter();
        while let Some(a) = it.next() {
            match it.next() {
                Some(b) => {
                    let mut c = conjoin_owned(eng, a, b, None)?;
                    eng.reduce(&mut c, ReductionPlan::default())?;
                    next.push(c);
                }
                None => next.push(a),
            }
        }
        complements = next;
    }
    let mut result = negate_tdd_owned(eng, complements.pop().expect("one operand"))?;
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

/// Return the disjunction of any number of diagrams sharing a vtree allocation.
///
/// **Prefer this to folding [`or`]** whenever more than two operands are in
/// hand. A disjunction is `!(!f ^ !g)`, so a fold of `n` operands complements
/// `3(n - 1)` times and two of every three of those complements fall on the
/// running disjunction, the largest diagram in the fold. This builds
/// `!f_1 ^ ... ^ !f_n` and complements once, which is `n + 1` complements on
/// the operands alone. The conjunction in the middle is where a disjunction
/// of diagrams is cheap, and it is folded as a balanced tree.
///
/// All operands are consumed. The result is minimized. A false operand is
/// dropped; if every operand is false, so is the result.
///
/// ```
/// use std::sync::Arc;
/// use tididi::{or_many, Tdd, Vtree};
///
/// let vtree = Arc::new(Vtree::balanced(4));
/// let clauses = [
///     Tdd::cube(&vtree, [1, 2])?,
///     Tdd::cube(&vtree, [3])?,
///     Tdd::cube(&vtree, [-4])?,
/// ];
/// let f = or_many(clauses)?;
/// assert_eq!(f.model_count()?, 13u32.into());
/// # Ok::<(), tididi::OperationError>(())
/// ```
///
/// # Errors
///
/// [`OperationError::EmptyOperands`] when no operand is given, since the
/// vtree of the result would be unknown; the other errors follow [`or`].
pub fn or_many(operands: impl IntoIterator<Item = Tdd>) -> Result<Tdd, OperationError> {
    let operands: Vec<Tdd> = operands.into_iter().collect();
    let Some(first) = operands.first() else {
        return Err(OperationError::EmptyOperands);
    };
    for g in &operands[1..] {
        crate::apply::check_vtree(first, g)?;
    }
    let context = std::sync::Arc::clone(first.context());
    context.run(|eng| eng.or_many(operands))
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

    /// Run [`or_many`] using this batch's scratch and resource limits.
    ///
    /// # Errors
    ///
    /// Returns the operation's errors, plus [`OperationError::Stopped`] or
    /// [`OperationError::OutputCap`] when an installed limit refuses the work.
    pub fn or_many(&self, operands: Vec<Tdd>) -> Result<Tdd, OperationError> {
        let Some(first) = operands.first() else {
            return Err(OperationError::EmptyOperands);
        };
        for g in &operands[1..] {
            crate::apply::check_vtree(first, g)?;
        }
        crate::apply::disjoin::disjoin_many_owned(self, operands)
    }
}

#[cfg(test)]
#[path = "tests/disjoin/mod.rs"]
mod tests;
