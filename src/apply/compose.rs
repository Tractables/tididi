//! Boolean combinations built from the shared apply and quantification kernels.

use super::QuantificationStrategy;
use crate::vtree::VarId;
use crate::{Engine, OperationError, Tdd};

impl Engine {
    /// If `condition` holds, use `then_branch`; otherwise use `else_branch`.
    ///
    /// The condition is itself a Boolean function, evaluated on each assignment:
    /// `(condition ∧ then_branch) ∨ (¬condition ∧ else_branch)`. All three operands
    /// must be structural and share a vtree allocation and compatible weights;
    /// an unweighted operand inherits the agreed weights. The result is minimized.
    ///
    /// Operands are consumed on success or error. Composition copies the condition
    /// and builds intermediate diagrams under this engine's limits; temporary
    /// storage can exceed the result's size.
    ///
    /// # Errors
    ///
    /// Returns a vtree, root, weight or marginal-level error before composition,
    /// or a resource error from a component operation.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{Engine, Vtree};
    /// let engine = Engine::new();
    /// let tree = Arc::new(Vtree::balanced(3));
    /// let select = engine.literal(&tree, 1)?;
    /// let yes = engine.literal(&tree, 2)?;
    /// let no = engine.literal(&tree, 3)?;
    /// let choice = engine.ite(select, yes, no)?;
    /// assert_eq!(choice.model_count(), 4u32.into());
    /// # Ok::<(), tididi::OperationError>(())
    /// ```
    pub fn ite(
        &self,
        mut condition: Tdd,
        mut then_branch: Tdd,
        mut else_branch: Tdd,
    ) -> Result<Tdd, OperationError> {
        super::check_conjunction_operands(&condition, &then_branch)?;
        super::check_conjunction_operands(&condition, &else_branch)?;
        for f in [&condition, &then_branch, &else_branch] {
            f.require_structure()?;
        }
        super::prepare_weights([&mut condition, &mut then_branch, &mut else_branch])?;
        let _op = self.limits().begin_operation();
        if self.limits().should_stop() {
            return Err(OperationError::Stopped);
        }
        let otherwise = self.negate(condition.try_clone_on(self)?)?;
        let yes = self.and(condition, then_branch)?;
        let no = self.and(otherwise, else_branch)?;
        // Disjunction minimizes unless a false operand selects its identity shortcut.
        let identity = yes.is_zero() || no.is_zero();
        let mut result = self.or(yes, no)?;
        if identity { crate::reduce::try_minimize(self, &mut result)?; }
        Ok(result)
    }

    /// Exclusive disjunction: exactly one operand holds.
    ///
    /// Both operands are consumed, must be structural, and must share a vtree
    /// allocation. The result is minimized; weight handling and intermediate
    /// storage follow [`Engine::ite`].
    ///
    /// # Errors
    ///
    /// The structural-input, compatibility and resource errors of [`Engine::ite`].
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{Engine, Vtree};
    /// let engine = Engine::new();
    /// let tree = Arc::new(Vtree::balanced(2));
    /// let parity = engine.xor(engine.literal(&tree, 1)?, engine.literal(&tree, 2)?)?;
    /// assert_eq!(parity.model_count(), 2u32.into());
    /// # Ok::<(), tididi::OperationError>(())
    /// ```
    pub fn xor(&self, mut f: Tdd, mut g: Tdd) -> Result<Tdd, OperationError> {
        super::check_conjunction_operands(&f, &g)?;
        f.require_structure()?;
        g.require_structure()?;
        super::prepare_weights([&mut f, &mut g])?;
        let _op = self.limits().begin_operation();
        if self.limits().should_stop() {
            return Err(OperationError::Stopped);
        }
        let not_g = self.negate(g.try_clone_on(self)?)?;
        self.ite(f, not_g, g)
    }

    /// Existential conjunction: `exists vars. (f AND g)`.
    ///
    /// Both operands are structural and consumed; the result is minimized and
    /// keeps their shared vtree and agreed weights. Quantified variables remain
    /// free in that universe, as in [`Engine::exists_vars`]. This composes
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
    /// use tididi::{Engine, Vtree};
    /// use tididi::apply::QuantificationStrategy;
    /// use tididi::vtree::VarId;
    /// let engine = Engine::new();
    /// let tree = Arc::new(Vtree::balanced(2));
    /// let current = engine.literal(&tree, -1)?; // current state x is false
    /// let transition = engine.xor(engine.literal(&tree, 1)?, engine.literal(&tree, 2)?)?;
    /// // The relation flips x to next-state y; forget the current-state variable.
    /// let next = engine.and_exists(current, transition, &[VarId(0)], QuantificationStrategy::Automatic)?;
    /// assert!(engine.equivalent(&next, &engine.literal(&tree, 2)?)?);
    /// # Ok::<(), tididi::OperationError>(())
    /// ```
    pub fn and_exists(
        &self,
        mut f: Tdd,
        mut g: Tdd,
        vars: &[VarId],
        how: QuantificationStrategy,
    ) -> Result<Tdd, OperationError> {
        super::check_conjunction_operands(&f, &g)?;
        f.require_structure()?;
        g.require_structure()?;
        super::prepare_weights([&mut f, &mut g])?;
        let _op = self.limits().begin_operation();
        if self.limits().should_stop() {
            return Err(OperationError::Stopped);
        }
        let targets = super::project::quantification_targets(self, f.vtree(), vars)?;
        let product = self.and(f, g)?;
        // A nonempty quantification minimizes a non-false product.
        let identity = targets.is_empty() || product.is_zero();
        let mut result = super::project::exists_targets_on(self, product, &targets, how)?;
        if identity { crate::reduce::try_minimize(self, &mut result)?; }
        Ok(result)
    }
}
