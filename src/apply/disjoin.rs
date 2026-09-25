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
use crate::apply::negate::negate_on;
use crate::reduce::ReductionPlan;

/// Panicking disjunction used by `BitOr` and test fixtures.
/// The checked entry point is [`or`].
pub(crate) fn apply_or(f: Tdd, g: Tdd) -> Tdd {
    or(f, g)
        .expect("apply_or: operation refused; use tididi::or to handle errors")
}

/// Disjoin owned operands by De Morgan with one final complement.
///
/// `!f_1 ^ ... ^ !f_n` is [`Engine::nor_many`]; this complements it once. When
/// at most one operand is left once the false ones are set aside there is
/// nothing to complement: that operand is the disjunction, and it comes back
/// minimized without a fill.
///
/// The operand list is the caller's and is not charged to the engine; the
/// complements and products built from it are.
pub(crate) fn disjoin_many_on(eng: &Engine, operands: Vec<Tdd>) -> Result<Tdd, OperationError> {
    let _op = eng.limits().enter()?;
    let (mut live, a_false_one) = live_operands(operands)?;
    if live.len() <= 1 {
        // The operand itself carries the vtree and the weights; when every
        // operand is false, so is the disjunction.
        let mut result = live.pop().or(a_false_one).ok_or(OperationError::EmptyOperands)?;
        eng.reduce(&mut result, ReductionPlan::default())?;
        return Ok(result);
    }
    let mut result = negate_on(eng, fold_conjunction(eng, complements_of(eng, live)?)?)?;
    eng.reduce(&mut result, ReductionPlan::default())?;
    Ok(result)
}

/// Validate the operands and set the false ones aside, keeping their order.
/// A false operand contributes nothing to a disjunction, and its complement,
/// the constant true, nothing to a conjunction. One of them is kept, since it
/// carries the vtree and the agreed weights when no other operand does.
fn live_operands(mut operands: Vec<Tdd>) -> Result<(Vec<Tdd>, Option<Tdd>), OperationError> {
    crate::apply::prepare_weights(&mut operands)?;
    for f in &operands { f.require_structure()?; }
    let mut kept = 0;
    for i in 0..operands.len() {
        if !operands[i].is_zero() {
            operands.swap(kept, i);
            kept += 1;
        }
    }
    let mut false_ones = operands.drain(kept..);
    let a_false_one = false_ones.next();
    drop(false_ones);
    Ok((operands, a_false_one))
}

/// Complement every operand without minimizing, into a list charged to the engine.
fn complements_of(eng: &Engine, operands: Vec<Tdd>) -> Result<Vec<Tdd>, OperationError> {
    let mut complements: Vec<Tdd> = Vec::new();
    eng.limits().reserve_exact(&mut complements, operands.len())?;
    for f in operands {
        complements.push(negate_on(eng, f)?);
    }
    Ok(complements)
}

/// Conjoin a non-empty operand list as a balanced tree, minimizing each
/// result. A chain would touch the growing conjunction once per operand and so
/// repeatedly combine a large intermediate with a small operand.
fn fold_conjunction(eng: &Engine, mut operands: Vec<Tdd>) -> Result<Tdd, OperationError> {
    use crate::limits::Charged;
    debug_assert!(!operands.is_empty());
    let lone = operands.len() == 1;
    while operands.len() > 1 {
        let mut next = Vec::new();
        eng.limits().reserve_exact(&mut next, operands.len().div_ceil(2))?;
        let operand_bytes = operands.charged_bytes();
        let mut it = operands.into_iter();
        while let Some(a) = it.next() {
            match it.next() {
                Some(b) => {
                    let mut c = eng.and(a, b)?;
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
    // A round minimizes each product as it builds it, so only a lone
    // complement still owes its reduction.
    if lone {
        eng.reduce(&mut result, ReductionPlan::default())?;
    }
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
/// Both operands are consumed on success and on error. The result is minimized.
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
    let context = std::sync::Arc::clone(first.context());
    context.run(|eng| eng.nor_many(operands))
}

impl crate::Engine {
    /// Run [`nor_many`] using this batch's scratch and resource limits:
    /// the conjunction of the complements, which [`Self::or_many`]
    /// complements once more.
    ///
    /// # Errors
    ///
    /// Returns the operation's errors, plus [`OperationError::Stopped`] or
    /// [`OperationError::OutputCap`] when an installed limit refuses the work.
    pub fn nor_many(&self, operands: Vec<Tdd>) -> Result<Tdd, OperationError> {
        let _op = self.limits().enter()?;
        crate::apply::check_same_vtree(&operands)?;
        let (live, a_false_one) = live_operands(operands)?;
        if live.is_empty() {
            // Every operand is false, so every complement is true.
            let f = a_false_one.ok_or(OperationError::EmptyOperands)?;
            return crate::build::constant_like(self, &f, true);
        }
        fold_conjunction(self, complements_of(self, live)?)
    }

    /// Run [`or`] using this batch's scratch and resource limits.
    ///
    /// # Errors
    ///
    /// Returns the operation's errors, plus [`OperationError::Stopped`] or
    /// [`OperationError::OutputCap`] when an installed limit refuses the work.
    pub fn or(&self, f: Tdd, g: Tdd) -> Result<Tdd, OperationError> {
        let _op = self.limits().enter()?;
        crate::apply::check_vtree(&f, &g)?;
        disjoin_many_on(self, collect_operands([f, g])?)
    }

    /// Run [`or_many`] using this batch's scratch and resource limits.
    ///
    /// # Errors
    ///
    /// Returns the operation's errors, plus [`OperationError::Stopped`] or
    /// [`OperationError::OutputCap`] when an installed limit refuses the work.
    pub fn or_many(&self, operands: Vec<Tdd>) -> Result<Tdd, OperationError> {
        let _op = self.limits().enter()?;
        crate::apply::check_same_vtree(&operands)?;
        disjoin_many_on(self, operands)
    }
}

#[cfg(test)]
#[path = "tests/disjoin/mod.rs"]
mod tests;
