//! Boolean combinations built from the shared apply and quantification kernels.

use crate::vtree::VarId;
use crate::{Engine, OperationError, Tdd};

/// Exclusive disjunction: exactly one operand holds.
///
/// Uses the shared vtree's execution context automatically.
///
/// Both operands are consumed, must be structural, and must share a vtree
/// allocation. The result is minimized; weight handling and intermediate
/// storage follow [`ite`](crate::ite).
///
/// # Errors
///
/// The structural-input, compatibility and resource errors of [`ite`](crate::ite).
///
/// ```
/// use std::sync::Arc;
/// use tididi::{literal, xor, Vtree};
/// let vtree = Arc::new(Vtree::balanced(2));
/// let parity = xor(literal(&vtree, 1)?, literal(&vtree, 2)?)?;
/// assert_eq!(parity.model_count()?, 2u32.into());
/// # Ok::<(), tididi::OperationError>(())
/// ```
pub fn xor(f: Tdd, g: Tdd) -> Result<Tdd, OperationError> {
    let context = std::sync::Arc::clone(f.context());
    context.run(|eng| eng.xor(f, g))
}

/// If `condition` holds, use `then_branch`; otherwise use `else_branch`.
///
/// Uses the shared vtree's execution context automatically.
///
/// The condition is itself a Boolean function, evaluated on each assignment:
/// `(condition ∧ then_branch) ∨ (¬condition ∧ else_branch)`. All three operands
/// must be structural and share a vtree allocation and compatible weights;
/// an unweighted operand inherits the agreed weights. The result is minimized.
///
/// Operands are consumed on success or error. Composition copies the condition
/// and builds intermediate diagrams using the shared vtree context; temporary
/// storage can exceed the result's size.
///
/// # Errors
///
/// Returns a vtree, root, weight or marginal-level error before composition,
/// or a resource error from a component operation.
///
/// ```
/// use std::sync::Arc;
/// use tididi::{literal, ite, Vtree};
/// let vtree = Arc::new(Vtree::balanced(3));
/// let select = literal(&vtree, 1)?;
/// let yes = literal(&vtree, 2)?;
/// let no = literal(&vtree, 3)?;
/// let choice = ite(select, yes, no)?;
/// assert_eq!(choice.model_count()?, 4u32.into());
/// # Ok::<(), tididi::OperationError>(())
/// ```
pub fn ite(condition: Tdd, then_branch: Tdd, else_branch: Tdd) -> Result<Tdd, OperationError> {
    let context = std::sync::Arc::clone(condition.context());
    context.run(|eng| eng.ite(condition, then_branch, else_branch))
}

/// Existential conjunction: `exists vars. (f AND g)`.
///
/// Uses the shared vtree's execution context automatically.
///
/// Both operands are structural and consumed; the result is minimized and
/// keeps their shared vtree and agreed weights. Quantified variables remain
/// free in that universe, as in [`Tdd::exists_vars`]. This composes
/// conjunction and quantification: it materializes the intermediate product.
/// Summing counts with [`Engine::and_marginalizing`] is a different operation.
///
/// # Errors
///
/// Validates operand compatibility, structure and every variable before
/// applying; component resource errors propagate.
///
/// ```
/// use std::sync::Arc;
/// use tididi::{literal, and_exists, xor, Vtree};
/// use tididi::vtree::VarId;
/// let vtree = Arc::new(Vtree::balanced(2));
/// let current = literal(&vtree, -1)?; // current state x is false
/// let transition = xor(literal(&vtree, 1)?, literal(&vtree, 2)?)?;
/// // The relation flips x to next-state y; forget the current-state variable.
/// let next = and_exists(current, transition, &[VarId(1)])?;
/// assert!(next.equivalent(&literal(&vtree, 2)?)?);
/// # Ok::<(), tididi::OperationError>(())
/// ```
pub fn and_exists(f: Tdd, g: Tdd, vars: &[VarId]) -> Result<Tdd, OperationError> {
    let context = std::sync::Arc::clone(f.context());
    context.run(|eng| eng.and_exists(f, g, vars))
}

impl Engine {
    /// Run [`ite`] using this batch's scratch and resource limits.
    ///
    /// # Errors
    ///
    /// Returns the operation's errors, plus [`OperationError::Stopped`] or
    /// [`OperationError::OutputCap`] when an installed limit refuses the work.
    pub fn ite(
        &self,
        mut condition: Tdd,
        mut then_branch: Tdd,
        mut else_branch: Tdd,
    ) -> Result<Tdd, OperationError> {
        super::check_vtree(&condition, &then_branch)?;
        super::check_vtree(&condition, &else_branch)?;
        for f in [&condition, &then_branch, &else_branch] {
            f.require_structure()?;
        }
        super::prepare_weights([&mut condition, &mut then_branch, &mut else_branch])?;
        let _op = self.limits().begin_operation();
        self.limits().check_stop()?;
        let otherwise = self.negate(condition.try_clone_on(self)?)?;
        let yes = self.and(condition, then_branch)?;
        let no = self.and(otherwise, else_branch)?;
        // Disjunction minimizes unless a false operand selects its identity shortcut.
        let identity = yes.is_zero() || no.is_zero();
        let mut result = self.or(yes, no)?;
        if identity { self.minimize(&mut result)?; }
        Ok(result)
    }

    /// Run [`xor`] using this batch's scratch and resource limits.
    ///
    /// # Errors
    ///
    /// Returns the operation's errors, plus [`OperationError::Stopped`] or
    /// [`OperationError::OutputCap`] when an installed limit refuses the work.
    pub fn xor(&self, mut f: Tdd, mut g: Tdd) -> Result<Tdd, OperationError> {
        super::check_vtree(&f, &g)?;
        f.require_structure()?;
        g.require_structure()?;
        super::prepare_weights([&mut f, &mut g])?;
        let _op = self.limits().begin_operation();
        self.limits().check_stop()?;
        let not_g = self.negate(g.try_clone_on(self)?)?;
        self.ite(f, not_g, g)
    }

    /// Run [`and_exists`] using this batch's scratch and resource limits.
    ///
    /// # Errors
    ///
    /// Returns the operation's errors, plus [`OperationError::Stopped`] or
    /// [`OperationError::OutputCap`] when an installed limit refuses the work.
    pub fn and_exists(&self, mut f: Tdd, mut g: Tdd, vars: &[VarId]) -> Result<Tdd, OperationError> {
        super::check_vtree(&f, &g)?;
        f.require_structure()?;
        g.require_structure()?;
        super::prepare_weights([&mut f, &mut g])?;
        let _op = self.limits().begin_operation();
        self.limits().check_stop()?;
        let targets = super::project::quantification_targets(self, f.vtree(), vars)?;
        let product = self.and(f, g)?;
        // A nonempty quantification minimizes a non-false product.
        let identity = targets.is_empty() || product.is_zero();
        let mut result = super::project::exists_targets_on(self, product, &targets)?;
        if identity { self.minimize(&mut result)?; }
        Ok(result)
    }
}
