//! Diagram operations using the execution context attached to the vtree.

use std::sync::Arc;

use crate::{Context, Literal, OperationError, Tdd};
use crate::apply::{QuantificationStrategy, RestrictionOutcome};
use crate::diagram::{EvalAlgebra, WeightValue};
use crate::vtree::{VarId, VtreeIdx};

impl Tdd {
    /// The reusable execution context shared by this diagram's vtree.
    ///
    /// Ordinary operations use it automatically. [`Context::with_limits`] lends
    /// one execution session for a group of operations under explicit limits;
    /// [`Context::clear_scratch`] releases its idle working buffers.
    pub fn context(&self) -> &Arc<Context> {
        self.vtree().context()
    }

    /// Conjoin two diagrams sharing the same vtree allocation.
    ///
    /// Both operands are consumed, including on error. The result need not be minimal.
    /// See [`Engine::and`](crate::Engine::and)
    /// for input requirements and error behavior.
    pub fn and(self, other: Tdd) -> Result<Tdd, OperationError> {
        let context = Arc::clone(self.context());
        context.run(|eng| eng.and(self, other))
    }

    /// Disjoin two structural diagrams sharing the same vtree allocation.
    ///
    /// Both operands are consumed, including on error; the result is minimized.
    /// See [`Engine::or`](crate::Engine::or)
    /// for input requirements and error behavior.
    pub fn or(self, other: Tdd) -> Result<Tdd, OperationError> {
        let context = Arc::clone(self.context());
        context.run(|eng| eng.or(self, other))
    }

    /// Complement a structural diagram, returning minimized form.
    ///
    /// The operand is consumed on success and on error.
    /// See [`Engine::negate`](crate::Engine::negate)
    /// for input requirements and error behavior.
    pub fn negate(self) -> Result<Tdd, OperationError> {
        let context = Arc::clone(self.context());
        context.run(|eng| eng.negate(self))
    }

    /// Exclusive OR of two structural diagrams on the same vtree.
    ///
    /// Both operands are consumed, including on error.
    /// See [`Engine::xor`](crate::Engine::xor)
    /// for input requirements and error behavior.
    pub fn xor(self, other: Tdd) -> Result<Tdd, OperationError> {
        let context = Arc::clone(self.context());
        context.run(|eng| eng.xor(self, other))
    }

    /// Choose between two branches according to this Boolean condition.
    ///
    /// All three structural operands must share a vtree and are consumed.
    /// See [`Engine::ite`](crate::Engine::ite)
    /// for input requirements and error behavior.
    pub fn ite(self, then_branch: Tdd, else_branch: Tdd) -> Result<Tdd, OperationError> {
        let context = Arc::clone(self.context());
        context.run(|eng| eng.ite(self, then_branch, else_branch))
    }

    /// Conjoin a disjunction of literals with this diagram.
    ///
    /// Consumes the diagram; an empty clause makes the result false.
    /// See [`Engine::and_clause`](crate::Engine::and_clause)
    /// for input requirements and error behavior.
    pub fn and_clause(self, clause: &[Literal]) -> Result<Tdd, OperationError> {
        let context = Arc::clone(self.context());
        context.run(|eng| eng.and_clause(self, clause))
    }

    /// Substitute fixed values for variables in this diagram.
    ///
    /// The vtree stays fixed: substituted variables become free in model counts.
    /// See [`Engine::condition`](crate::Engine::condition)
    /// for input requirements and error behavior.
    pub fn condition(self, assignment: impl IntoIterator<Item = impl TryInto<Literal, Error: Into<OperationError>>>) -> Result<Tdd, OperationError> {
        let context = Arc::clone(self.context());
        context.run(|eng| eng.condition(self, assignment))
    }

    /// Substitute one variable with a fixed Boolean value.
    ///
    /// Consumes the diagram; the variable remains in the counting universe.
    /// See [`Engine::condition_var`](crate::Engine::condition_var)
    /// for input requirements and error behavior.
    pub fn condition_var(self, var: VarId, value: bool) -> Result<Tdd, OperationError> {
        let context = Arc::clone(self.context());
        context.run(|eng| eng.condition_var(self, var, value))
    }

    /// Substitute the same Boolean value for several variables.
    ///
    /// Consumes the diagram and retains the original vtree.
    /// See [`Engine::condition_vars`](crate::Engine::condition_vars)
    /// for input requirements and error behavior.
    pub fn condition_vars(self, vars: &[VarId], value: bool) -> Result<Tdd, OperationError> {
        let context = Arc::clone(self.context());
        context.run(|eng| eng.condition_vars(self, vars, value))
    }

    /// Existentially quantify one variable using the automatic strategy.
    ///
    /// The quantified variable remains free in the unchanged vtree.
    /// See [`Engine::exists_var`](crate::Engine::exists_var)
    /// for input requirements and error behavior.
    pub fn exists_var(self, var: VarId) -> Result<Tdd, OperationError> {
        let context = Arc::clone(self.context());
        context.run(|eng| eng.exists_var(self, var))
    }

    /// Existentially quantify variables using the automatic strategy.
    ///
    /// Each distinct variable becomes free; full-vtree counts include those choices.
    /// See [`Engine::exists_vars`](crate::Engine::exists_vars)
    /// for input requirements and error behavior.
    pub fn exists_vars(self, vars: &[VarId]) -> Result<Tdd, OperationError> {
        let context = Arc::clone(self.context());
        context.run(|eng| eng.exists_vars(self, vars))
    }

    /// Quantify one variable with an explicit algorithm choice.
    ///
    /// Use `exists_var` unless this workload needs a particular rewrite.
    /// See [`Engine::exists_var_with_strategy`](crate::Engine::exists_var_with_strategy)
    /// for input requirements and error behavior.
    pub fn exists_var_with_strategy(self, var: VarId, strategy: QuantificationStrategy) -> Result<Tdd, OperationError> {
        let context = Arc::clone(self.context());
        context.run(|eng| eng.exists_var_with_strategy(self, var, strategy))
    }

    /// Quantify several variables with an explicit algorithm choice.
    ///
    /// Use `exists_vars` for automatic selection.
    /// See [`Engine::exists_vars_with_strategy`](crate::Engine::exists_vars_with_strategy)
    /// for input requirements and error behavior.
    pub fn exists_vars_with_strategy(self, vars: &[VarId], strategy: QuantificationStrategy) -> Result<Tdd, OperationError> {
        let context = Arc::clone(self.context());
        context.run(|eng| eng.exists_vars_with_strategy(self, vars, strategy))
    }

    /// Conjoin two diagrams and existentially quantify variables from the result.
    ///
    /// Both operands are consumed; strategy selection is automatic.
    /// See [`Engine::and_exists`](crate::Engine::and_exists)
    /// for input requirements and error behavior.
    pub fn and_exists(self, other: Tdd, vars: &[VarId]) -> Result<Tdd, OperationError> {
        let context = Arc::clone(self.context());
        context.run(|eng| eng.and_exists(self, other, vars))
    }

    /// Conjoin and quantify with an explicit quantification strategy.
    ///
    /// Use `and_exists` for automatic selection.
    /// See [`Engine::and_exists_with_strategy`](crate::Engine::and_exists_with_strategy)
    /// for input requirements and error behavior.
    pub fn and_exists_with_strategy(self, other: Tdd, vars: &[VarId], strategy: QuantificationStrategy) -> Result<Tdd, OperationError> {
        let context = Arc::clone(self.context());
        context.run(|eng| eng.and_exists_with_strategy(self, other, vars, strategy))
    }

    /// Rename variables simultaneously within the existing vtree.
    ///
    /// Targets must already occur in the tree; swaps and identification are supported.
    /// See [`Engine::rename_vars`](crate::Engine::rename_vars)
    /// for input requirements and error behavior.
    pub fn rename_vars(self, renames: &[(VarId, VarId)]) -> Result<Tdd, OperationError> {
        let context = Arc::clone(self.context());
        context.run(|eng| eng.rename_vars(self, renames))
    }

    /// Simultaneously replace variables with structural Boolean functions.
    ///
    /// All replacement diagrams must share this diagram's vtree.
    /// See [`Engine::substitute`](crate::Engine::substitute)
    /// for input requirements and error behavior.
    pub fn substitute(self, replacements: &[(VarId, &Tdd)]) -> Result<Tdd, OperationError> {
        let context = Arc::clone(self.context());
        context.run(|eng| eng.substitute(self, replacements))
    }

    /// Simplify a function on the assignments described by a care set.
    ///
    /// Both diagrams are consumed; the result agrees with this function on the care set.
    /// See [`Engine::restrict_to_care`](crate::Engine::restrict_to_care)
    /// for input requirements and error behavior.
    pub fn restrict_to_care(self, care: Tdd) -> Result<RestrictionOutcome, OperationError> {
        let context = Arc::clone(self.context());
        context.run(|eng| eng.restrict_to_care(self, care))
    }

    /// Whether two structural diagrams on the same vtree represent the same function.
    ///
    /// Borrows both diagrams; their node numbering and minimality need not agree.
    /// See [`Engine::equivalent`](crate::Engine::equivalent)
    /// for input requirements and error behavior.
    pub fn equivalent(&self, other: &Tdd) -> Result<bool, OperationError> {
        let context = Arc::clone(self.context());
        context.run(|eng| eng.equivalent(self, other))
    }

    /// Whether every model of this structural diagram satisfies another.
    ///
    /// Both diagrams are borrowed and must share a vtree.
    /// See [`Engine::implies`](crate::Engine::implies)
    /// for input requirements and error behavior.
    pub fn implies(&self, other: &Tdd) -> Result<bool, OperationError> {
        let context = Arc::clone(self.context());
        context.run(|eng| eng.implies(self, other))
    }

    /// Return the variables that can affect this structural function.
    ///
    /// Borrows the diagram and returns zero-based variable identifiers.
    /// See [`Engine::support`](crate::Engine::support)
    /// for input requirements and error behavior.
    pub fn support(&self) -> Result<Vec<VarId>, OperationError> {
        let context = Arc::clone(self.context());
        context.run(|eng| eng.support(self))
    }

    /// Return literals that hold in every satisfying assignment.
    ///
    /// Borrows the diagram; see the batch operation for the unsatisfiable-input convention.
    /// See [`Engine::implied_literals`](crate::Engine::implied_literals)
    /// for input requirements and error behavior.
    pub fn implied_literals(&self) -> Result<Vec<Literal>, OperationError> {
        let context = Arc::clone(self.context());
        context.run(|eng| eng.implied_literals(self))
    }

    /// Count satisfying assignments, returning an error on refusal.
    ///
    /// The count includes every variable in the vtree, including free variables.
    /// See [`Engine::model_count`](crate::Engine::model_count)
    /// for input requirements and error behavior.
    pub fn try_model_count(&self) -> Result<num_bigint::BigUint, OperationError> {
        let context = Arc::clone(self.context());
        context.run(|eng| eng.model_count(self))
    }

    /// Check structural satisfiability, returning an error on refusal.
    ///
    /// Borrows the diagram and ignores literal weights.
    /// See [`Engine::is_sat`](crate::Engine::is_sat)
    /// for input requirements and error behavior.
    pub fn try_is_sat(&self) -> Result<bool, OperationError> {
        let context = Arc::clone(self.context());
        context.run(|eng| eng.is_sat(self))
    }

    /// Find a complete satisfying assignment, returning errors to the caller.
    ///
    /// Returns `None` for a false structural function; the chosen model is unspecified.
    /// See [`Engine::satisfying_assignment`](crate::Engine::satisfying_assignment)
    /// for input requirements and error behavior.
    pub fn try_satisfying_assignment(&self) -> Result<Option<Vec<Literal>>, OperationError> {
        let context = Arc::clone(self.context());
        context.run(|eng| eng.satisfying_assignment(self))
    }

    /// Evaluate a structural diagram with the supplied algebra or literal weights.
    ///
    /// Borrows the diagram; each call uses the supplied values without changing structure.
    /// See [`Engine::evaluate`](crate::Engine::evaluate)
    /// for input requirements and error behavior.
    pub fn evaluate<S: EvalAlgebra>(&self, algebra: &S) -> Result<S::Value, OperationError> {
        let context = Arc::clone(self.context());
        context.run(|eng| eng.evaluate(self, algebra))
    }

    /// Read the value under weights attached to this diagram.
    ///
    /// Returns `None` when no weight store is attached.
    /// See [`Engine::weighted_value`](crate::Engine::weighted_value)
    /// for input requirements and error behavior.
    pub fn weighted_value(&self) -> Result<Option<WeightValue>, OperationError> {
        let context = Arc::clone(self.context());
        context.run(|eng| eng.weighted_value(self))
    }

    /// Minimize this diagram under its current vtree.
    ///
    /// Keeps the represented function and may retain completed edits after a refusal.
    /// See [`reduce::try_minimize`](crate::reduce::try_minimize)
    /// for input requirements and error behavior.
    pub fn minimize(&mut self) -> Result<(), OperationError> {
        let context = Arc::clone(self.context());
        context.run(|eng| crate::reduce::try_minimize(eng, self))
    }

    /// Run a selected reduction plan on this diagram.
    ///
    /// A partial plan does not necessarily establish canonical form.
    /// See [`reduce::try_reduce`](crate::reduce::try_reduce)
    /// for input requirements and error behavior.
    pub fn reduce(&mut self, plan: crate::reduce::ReductionPlan<'_>) -> Result<(), OperationError> {
        let context = Arc::clone(self.context());
        context.run(|eng| crate::reduce::try_reduce(eng, self, plan))
    }

    /// Replace selected subtrees with counts or attached weighted values.
    ///
    /// Discarded Boolean structure cannot be recovered or evaluated under new weights.
    /// See [`marginal::marginalize_levels`](crate::marginal::marginalize_levels)
    /// for input requirements and error behavior.
    pub fn marginalize_levels(&mut self, levels: &[VtreeIdx]) -> Result<(), OperationError> {
        let context = Arc::clone(self.context());
        context.run(|eng| crate::marginal::marginalize_levels(eng, self, levels))
    }

    /// Search nearby vtree shapes while preserving this function.
    ///
    /// The resulting tree keeps its execution context; other diagrams keep their own shape.
    /// See [`Engine::rotation_search`](crate::Engine::rotation_search)
    /// for input requirements and error behavior.
    pub fn rotation_search<O: crate::restructure::search::RotationObjective>(&mut self, objective: &mut O, config: &crate::restructure::search::RotationSearchConfig) -> Result<crate::restructure::search::RotationSearchStats, OperationError> {
        let context = Arc::clone(self.context());
        context.run(|eng| eng.rotation_search(self, objective, config))
    }
}
