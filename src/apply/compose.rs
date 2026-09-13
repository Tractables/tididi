//! Boolean combinations built from the shared apply and quantification kernels.

use super::QuantificationStrategy;
use crate::vtree::VarId;
use crate::{Engine, OperationError, Tdd};

impl Engine {
    /// If `condition` holds, use `then_branch`; otherwise use `else_branch`.
    ///
    /// Computes `(condition AND then_branch) OR (NOT condition AND else_branch)`
    /// using the existing Boolean operations and minimizes the result. All three
    /// operands must be structural and share a vtree and compatible weights;
    /// an unweighted operand inherits the agreed weights. The condition is
    /// copied once through this engine's allocation policy. Operands are consumed
    /// on success or refusal; intermediate diagrams can exceed the result's size.
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
        super::prepare_weights(&mut condition, &mut then_branch)?;
        super::prepare_weights(&mut condition, &mut else_branch)?;
        super::prepare_weights(&mut condition, &mut then_branch)?;
        let _op = self.limits().begin_operation();
        if self.limits().should_stop() {
            return Err(OperationError::Stopped);
        }
        let otherwise = self.negate(condition.try_clone_on(self)?)?;
        let yes = self.and(condition, then_branch)?;
        let no = self.and(otherwise, else_branch)?;
        let mut result = self.or(yes, no)?;
        crate::reduce::try_minimize(self, &mut result)?;
        Ok(result)
    }

    /// Exclusive disjunction: exactly one operand holds.
    ///
    /// Delegates to [`Engine::ite`] with the negation of `g` as its true branch;
    /// the result is minimized and retains compatible operand weights.
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
        super::prepare_weights(&mut f, &mut g)?;
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
    /// let f = engine.clause(&tree, [1, 2])?;
    /// let g = engine.literal(&tree, -1)?;
    /// let projected = engine.and_exists(f, g, &[VarId(0)], QuantificationStrategy::Automatic)?;
    /// assert!(engine.equivalent(&projected, &engine.literal(&tree, 2)?)?);
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
        super::prepare_weights(&mut f, &mut g)?;
        let _op = self.limits().begin_operation();
        let mut gate = crate::limits::PollGate::new(self.limits().reduce_poll_stride());
        if self.limits().should_stop() {
            return Err(OperationError::Stopped);
        }
        for &var in vars {
            if f.vtree().leaf_of(var).is_none() {
                return Err(OperationError::VariableNotInVtree(var));
            }
            self.limits().poll(&mut gate, 1)?;
        }
        self.limits().flush_poll(&mut gate)?;
        let mut result = self.exists_vars(self.and(f, g)?, vars, how)?;
        crate::reduce::try_minimize(self, &mut result)?;
        Ok(result)
    }
}
