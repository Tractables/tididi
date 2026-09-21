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
    crate::apply::prepare_weights(&mut [&mut f, &mut g])?;
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
/// `!f_1 ^ ... ^ !f_n` is [`nor_many_owned`]; this complements it once.
pub(crate) fn disjoin_many_owned(eng: &Engine, operands: Vec<Tdd>) -> Result<Tdd, OperationError> {
    let _op = eng.limits().begin_operation();
    eng.limits().check_stop()?;
    let (complements, a_false_one) = complements_of(eng, operands)?;
    if complements.is_empty() {
        // Every operand is false, so the disjunction is: the operand itself,
        // which carries the vtree and the weights.
        return a_false_one.ok_or(OperationError::EmptyOperands);
    }
    let mut result = negate_tdd_owned(eng, fold_conjunction(eng, complements)?)?;
    eng.reduce(&mut result, ReductionPlan::default())?;
    Ok(result)
}

/// `!f_1 ^ ... ^ !f_n` over owned operands: the conjunction of the complements,
/// which is [`disjoin_many_owned`] without its final complement.
pub(crate) fn nor_many_owned(eng: &Engine, operands: Vec<Tdd>) -> Result<Tdd, OperationError> {
    let _op = eng.limits().begin_operation();
    eng.limits().check_stop()?;
    let (complements, a_false_one) = complements_of(eng, operands)?;
    if complements.is_empty() {
        // Every operand is false, so every complement is true.
        let f = a_false_one.ok_or(OperationError::EmptyOperands)?;
        let mut out = crate::build::constant_one(eng, &f.vtree);
        out.weights = f.weights;
        return Ok(out);
    }
    fold_conjunction(eng, complements)
}

/// Complement every operand that is not false, without minimizing; a false
/// operand is returned separately, since its complement is the constant true
/// and contributes nothing to the conjunction.
fn complements_of(
    eng: &Engine,
    mut operands: Vec<Tdd>,
) -> Result<(Vec<Tdd>, Option<Tdd>), OperationError> {
    crate::apply::prepare_weights(&mut operands)?;
    for f in &operands { f.require_structure()?; }
    let mut complements: Vec<Tdd> = Vec::new();
    eng.limits().reserve_exact(&mut complements, operands.len())?;
    let mut a_false_one: Option<Tdd> = None;
    for f in operands {
        if f.is_zero() {
            a_false_one = Some(f);
            continue;
        }
        complements.push(negate_tdd_owned(eng, f)?);
    }
    Ok((complements, a_false_one))
}

/// Conjoin a non-empty operand list as a balanced tree, minimizing each
/// result. A chain would touch the growing conjunction once per operand and so
/// repeatedly combine a large intermediate with a small operand.
fn fold_conjunction(eng: &Engine, mut operands: Vec<Tdd>) -> Result<Tdd, OperationError> {
    use crate::apply::conjoin::conjoin_owned;
    use crate::limits::Charged;
    debug_assert!(!operands.is_empty());
    while operands.len() > 1 {
        let mut next = Vec::new();
        eng.limits().reserve_exact(&mut next, operands.len().div_ceil(2))?;
        let operand_bytes = operands.charged_bytes();
        let mut it = operands.into_iter();
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
        drop(it);
        eng.limits().release_bytes(operand_bytes);
        operands = next;
    }
    let mut result = operands.pop().expect("a non-empty operand list");
    eng.limits().discard(operands);
    eng.reduce(&mut result, ReductionPlan::default())?;
    Ok(result)
}

/// Collect the caller's operands without turning allocator refusal into a panic.
fn collect_operands(operands: impl IntoIterator<Item = Tdd>) -> Result<Vec<Tdd>, OperationError> {
    let mut out = Vec::new();
    for operand in operands {
        out.try_reserve(1).map_err(|_| OperationError::OverBudget)?;
        out.push(operand);
    }
    Ok(out)
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
/// Complements each nonfalse operand, conjoins those complements in a balanced
/// tree, then complements the result. This avoids repeatedly complementing a
/// running disjunction, as a fold of [`or`] would. Intermediate sizes still
/// depend on the operands and their grouping.
///
/// All operands are consumed. The result is minimized. A false operand is
/// dropped after checking weight compatibility; if every operand is false,
/// so is the result. The result retains the agreed weights, as in [`or`].
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
    let operands = collect_operands(operands)?;
    let Some(first) = operands.first() else {
        return Err(OperationError::EmptyOperands);
    };
    for g in &operands[1..] {
        crate::apply::check_vtree(first, g)?;
    }
    let context = std::sync::Arc::clone(first.context());
    context.run(|eng| eng.or_many(operands))
}

/// The conjunction of the operands' complements, `!f_1 ^ ... ^ !f_n`.
///
/// True exactly when none of the operands is true. Uses the same balanced
/// conjunction as [`or_many`], without its final complement.
///
/// All operands are consumed and must share a vtree. The result is minimized.
/// A false operand is dropped, its complement being the constant true; if
/// every operand is false, the result is the constant true. Weight compatibility
/// and inheritance follow [`or`], including for false operands.
///
/// ```
/// use std::sync::Arc;
/// use tididi::{nor_many, Tdd, Vtree};
///
/// let vtree = Arc::new(Vtree::balanced(4));
/// let bodies = [Tdd::cube(&vtree, [1, 2])?, Tdd::cube(&vtree, [3])?];
/// let f = nor_many(bodies)?;
/// // 16 assignments, 4 with x1 ^ x2, 8 with x3, 2 with both.
/// assert_eq!(f.model_count()?, 6u32.into());
/// # Ok::<(), tididi::OperationError>(())
/// ```
///
/// # Errors
///
/// [`OperationError::EmptyOperands`] when no operand is given, since the
/// vtree of the result would be unknown; the other errors follow [`or`].
pub fn nor_many(operands: impl IntoIterator<Item = Tdd>) -> Result<Tdd, OperationError> {
    let operands = collect_operands(operands)?;
    let Some(first) = operands.first() else {
        return Err(OperationError::EmptyOperands);
    };
    for g in &operands[1..] {
        crate::apply::check_vtree(first, g)?;
    }
    let context = std::sync::Arc::clone(first.context());
    context.run(|eng| eng.nor_many(operands))
}

impl crate::Engine {
    /// Run [`nor_many`] using this batch's scratch and resource limits.
    ///
    /// # Errors
    ///
    /// Returns the operation's errors, plus [`OperationError::Stopped`] or
    /// [`OperationError::OutputCap`] when an installed limit refuses the work.
    pub fn nor_many(&self, operands: Vec<Tdd>) -> Result<Tdd, OperationError> {
        let Some(first) = operands.first() else {
            return Err(OperationError::EmptyOperands);
        };
        for g in &operands[1..] {
            crate::apply::check_vtree(first, g)?;
        }
        crate::apply::disjoin::nor_many_owned(self, operands)
    }

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
