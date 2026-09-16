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
    /// Ordinary methods have no configured resource limits and do not inherit
    /// limits from an enclosing batch; use the supplied engine to bound its work.
    pub fn context(&self) -> &Arc<Context> {
        self.vtree().context()
    }

    /// Return the Boolean complement of this structural diagram in minimized form.
    ///
    /// Consumes the operand on success and error, retaining its vtree and attached
    /// weights. Complementation can grow the diagram. If only an unweighted count
    /// is needed, subtract the original count from `2^n`, where `n` is the number
    /// of vtree variables.
    ///
    /// # Errors
    ///
    /// Returns [`OperationError::MarginalLevel`] if any level has discarded its
    /// structure, or [`OperationError::OverBudget`] if an allocation is refused.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{Tdd, Vtree};
    /// let vtree = Arc::new(Vtree::balanced(3));
    /// let neither = Tdd::clause(&vtree, [1, 2])?.negate()?;
    /// assert!(neither.equivalent(&Tdd::cube(&vtree, [-1, -2])?)?);
    /// assert_eq!(neither.model_count()?, 2u32.into());
    /// # Ok::<(), tididi::OperationError>(())
    /// ```
    pub fn negate(self) -> Result<Tdd, OperationError> {
        let context = Arc::clone(self.context());
        context.run(|eng| eng.negate(self))
    }

    /// Conjoin a disjunction of literals without building a separate diagram.
    ///
    /// Accepts arrays, slices and vectors of signed, one-based integers or typed
    /// [`Literal`] values. Collect an iterator into a vector before passing it.
    /// Typed slices are borrowed directly; integer conversion uses temporary storage.
    ///
    /// Consumes the diagram on success and error; the result retains its vtree and
    /// weights. Repeated literals are ignored, opposite polarities make a tautology,
    /// and an empty clause makes the result false. A false operand stays false.
    /// Only paths from clause variables to the root are rebuilt; marginal levels
    /// outside those paths pass through. The result counts correctly but may need
    /// [`minimize`](Self::minimize) to establish canonical form.
    ///
    /// # Errors
    ///
    /// Returns [`OperationError::InvalidLiteral`] for integer zero,
    /// [`OperationError::VariableNotInVtree`] for an absent variable,
    /// [`OperationError::MarginalLevel`] if a required level has discarded its
    /// structure, or [`OperationError::OverBudget`] if an allocation is refused.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{Tdd, Vtree};
    /// let vtree = Arc::new(Vtree::balanced(3));
    /// let f = Tdd::one(&vtree).and_clause([-1, 2])?;
    /// assert_eq!(f.model_count()?, 6u32.into()); // remote implies encrypted
    /// # Ok::<(), tididi::OperationError>(())
    /// ```
    pub fn and_clause<L: crate::apply::ClauseLiteral>(self, clause: impl AsRef<[L]>) -> Result<Tdd, OperationError> {
        let context = Arc::clone(self.context());
        context.run(|eng| eng.and_clause(self, clause.as_ref()))
    }

    /// Substitute an assignment into this function and return its minimized cofactor.
    ///
    /// Positive literals set variables to true, negative literals to false. Integers
    /// are signed and one-based; typed [`Literal`] values also work. The substituted
    /// variables become free in the unchanged vtree: each contributes a factor of
    /// two to the resulting model count. Conjoin a cube instead to count only the
    /// original assignments consistent with an observation; [`counter`](Self::counter)
    /// provides repeated evidence counts without building a new diagram each time.
    ///
    /// Consumes the diagram on success and error and retains its weights. Equal
    /// literals are ignored. Opposite literals produce false after all variables
    /// are validated; an empty assignment returns the operand unchanged. For a
    /// consistent assignment on a nonfalse input, each target leaf and its parent
    /// must be structural; other levels may be marginal. Conditioning does not
    /// grow the diagram. With weighted marginal values, a zero evaluation may
    /// reflect zero weights or cancellation rather than Boolean falsity.
    ///
    /// # Errors
    ///
    /// Returns [`OperationError::InvalidLiteral`] for integer zero,
    /// [`OperationError::VariableNotInVtree`] for an absent variable,
    /// [`OperationError::MarginalLevel`] for required discarded structure, or
    /// [`OperationError::OverBudget`] if an allocation is refused.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{and, literal, Tdd, Vtree};
    /// let vtree = Arc::new(Vtree::balanced(3));
    /// let f = Tdd::clause(&vtree, [1, 2, 3])?;
    /// let cofactor = f.clone().condition([-1, -2])?;
    /// assert!(cofactor.equivalent(&literal(&vtree, 3)?)?);
    /// assert_eq!(cofactor.model_count()?, 4u32.into());
    /// let observed = and(f, Tdd::cube(&vtree, [-1, -2])?)?;
    /// assert_eq!(observed.model_count()?, 1u32.into());
    /// # Ok::<(), tididi::OperationError>(())
    /// ```
    pub fn condition(self, assignment: impl IntoIterator<Item = impl TryInto<Literal, Error: Into<OperationError>>>) -> Result<Tdd, OperationError> {
        let context = Arc::clone(self.context());
        context.run(|eng| eng.condition(self, assignment))
    }

    /// Substitute `value` for one variable and return its minimized cofactor.
    ///
    /// Consumes the diagram, retaining its weights and vtree. The variable becomes
    /// free, so the result's model count includes both of its values. Only its leaf
    /// and parent need to be structural; other levels may be marginal. A false
    /// operand remains false. The variable is validated even for a false operand.
    /// [`condition`](Self::condition) accepts mixed assignments and explains
    /// cofactor counts versus evidence counts.
    ///
    /// # Errors
    ///
    /// Returns [`OperationError::VariableNotInVtree`] for an absent variable,
    /// [`OperationError::MarginalLevel`] when a nonfalse operand needs a leaf or
    /// parent whose structure was discarded, or [`OperationError::OverBudget`]
    /// if an allocation is refused. The operand is also consumed on error.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{Tdd, Vtree};
    /// let vtree = Arc::new(Vtree::balanced(3));
    /// let f = Tdd::clause(&vtree, [1, 2])?;
    /// let g = f.condition_var(tididi::vtree::VarId(0), false)?;
    /// assert_eq!(g.model_count()?, 4u32.into()); // x2, with x1 and x3 free
    /// # Ok::<(), tididi::OperationError>(())
    /// ```
    pub fn condition_var(self, var: VarId, value: bool) -> Result<Tdd, OperationError> {
        let context = Arc::clone(self.context());
        context.run(|eng| eng.condition_var(self, var, value))
    }

    /// Substitute the same Boolean value for every variable in `vars`.
    ///
    /// Consumes the diagram and returns its minimized cofactor using one propagation
    /// and reduction pass. Repeated variables are ignored; an empty slice returns
    /// the operand unchanged. The vtree and weights are retained, and each distinct
    /// target becomes free in model counts. For a nonfalse input, each target leaf
    /// and its parent must be structural; other levels may be marginal.
    /// Use [`condition`](Self::condition) for mixed polarities.
    ///
    /// # Errors
    ///
    /// All variables are validated before rewriting. Returns
    /// [`OperationError::VariableNotInVtree`] for an absent variable,
    /// [`OperationError::MarginalLevel`] for required discarded structure, or
    /// [`OperationError::OverBudget`] if an allocation is refused.
    /// The operand is also consumed on error.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{Tdd, Vtree};
    /// let vtree = Arc::new(Vtree::balanced(3));
    /// use tididi::vtree::VarId;
    /// let f = Tdd::clause(&vtree, [1, 2, 3])?;
    /// let g = f.condition_vars(&[VarId(0), VarId(1)], false)?;
    /// assert_eq!(g.model_count()?, 4u32.into()); // x3, with x1 and x2 free
    /// # Ok::<(), tididi::OperationError>(())
    /// ```
    pub fn condition_vars(self, vars: &[VarId], value: bool) -> Result<Tdd, OperationError> {
        let context = Arc::clone(self.context());
        context.run(|eng| eng.condition_vars(self, vars, value))
    }

    /// Existentially quantify one variable using the automatic rewrite strategy.
    ///
    /// The result holds whenever either value of `var` satisfies this function.
    /// Consumes the diagram, retaining its weights and vtree, and minimizes a
    /// nonfalse result. A false operand remains false. The quantified variable
    /// becomes free, so divide the full-vtree model count by two to count assignments
    /// to the remaining variables. [`exists_vars`](Self::exists_vars) accepts a batch.
    ///
    /// # Errors
    ///
    /// Returns [`OperationError::VariableNotInVtree`] for an absent variable,
    /// [`OperationError::MarginalLevel`] if the target leaf, a rewritten ancestor,
    /// or an ancestor's grandchild is marginal, or [`OperationError::OverBudget`]
    /// if an allocation is refused. The operand is also consumed on error.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{Tdd, Vtree};
    /// let vtree = Arc::new(Vtree::balanced(3));
    /// let f = Tdd::cube(&vtree, [1, 2])?;
    /// let g = f.exists_var(tididi::vtree::VarId(1))?;
    /// assert_eq!(g.model_count()?, 4u32.into()); // x1, with x2 and x3 free
    /// # Ok::<(), tididi::OperationError>(())
    /// ```
    pub fn exists_var(self, var: VarId) -> Result<Tdd, OperationError> {
        let context = Arc::clone(self.context());
        context.run(|eng| eng.exists_var(self, var))
    }

    /// Remove dependence on variables by allowing either value of each one.
    ///
    /// An assignment to the remaining variables satisfies the result when at least
    /// one extension satisfies this function. Multiple satisfying extensions count
    /// as one remaining assignment. Uses [`QuantificationStrategy::Automatic`];
    /// [`exists_vars_with_strategy`](Self::exists_vars_with_strategy) selects a rewrite.
    ///
    /// Consumes the diagram on success and error, retaining its vtree and weights.
    /// Each distinct variable is processed once, in first-occurrence order. An
    /// empty slice returns the operand unchanged; a nonempty request minimizes a
    /// nonfalse operand. A false operand remains false.
    ///
    /// The quantified variables remain free in the unchanged vtree. Divide the
    /// result's full model count by `2^k` to count over the remaining variables,
    /// where `k` is the number of distinct quantified variables. In contrast,
    /// [`marginalize_levels`](Self::marginalize_levels) sums extension counts and
    /// preserves the original total.
    ///
    /// # Errors
    ///
    /// Every variable is validated before quantification. Returns
    /// [`OperationError::VariableNotInVtree`] for an absent variable,
    /// [`OperationError::MarginalLevel`] if a target leaf, rewritten ancestor, or
    /// an ancestor's grandchild is marginal, or [`OperationError::OverBudget`]
    /// if an allocation is refused.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{Tdd, Vtree};
    /// let vtree = Arc::new(Vtree::balanced(2));
    /// use tididi::vtree::VarId;
    /// let f = Tdd::clause(&vtree, [1, 2])?;
    /// let projected = f.exists_vars(&[VarId(0)])?;
    /// assert!(projected.equivalent(&Tdd::one(&vtree))?);
    /// let remaining_count = projected.model_count()? >> 1usize;
    /// assert_eq!(remaining_count, 2u32.into());
    /// # Ok::<(), tididi::OperationError>(())
    /// ```
    pub fn exists_vars(self, vars: &[VarId]) -> Result<Tdd, OperationError> {
        let context = Arc::clone(self.context());
        context.run(|eng| eng.exists_vars(self, vars))
    }

    /// Existentially quantify one variable with an explicit rewrite strategy.
    ///
    /// Ownership, counting semantics and errors follow [`exists_var`](Self::exists_var).
    /// The strategy applies to each distinct variable. The structural rewrite
    /// retains marginal levels off the rewritten paths unchanged; cofactor
    /// rewriting requires fully structural input. Use the automatic form unless
    /// a workload needs a specific algorithm.
    pub fn exists_var_with_strategy(self, var: VarId, strategy: QuantificationStrategy) -> Result<Tdd, OperationError> {
        let context = Arc::clone(self.context());
        context.run(|eng| eng.exists_var_with_strategy(self, var, strategy))
    }

    /// Existentially quantify variables with an explicit rewrite strategy.
    ///
    /// Ownership, counting semantics and errors follow [`exists_vars`](Self::exists_vars).
    /// The strategy applies to each distinct variable. The structural rewrite
    /// retains marginal levels off the rewritten paths unchanged; cofactor
    /// rewriting requires fully structural input. Use the automatic form unless
    /// a workload needs a specific algorithm.
    pub fn exists_vars_with_strategy(self, vars: &[VarId], strategy: QuantificationStrategy) -> Result<Tdd, OperationError> {
        let context = Arc::clone(self.context());
        context.run(|eng| eng.exists_vars_with_strategy(self, vars, strategy))
    }

    /// Rename variable occurrences simultaneously within the existing vtree.
    ///
    /// Each `(source, target)` uses zero-based [`VarId`]s. Omitted sources keep their
    /// meaning. Swaps and cycles happen simultaneously; distinct sources may share
    /// a target, but each source may appear only once. Every source and target must
    /// occur in the vtree.
    ///
    /// Consumes the diagram on success and error. A nonempty map returns a minimized
    /// function; an empty map returns the operand unchanged after validation. The
    /// vtree allocation, shape, variable ids and literal weights stay fixed, so
    /// renaming can change a weighted value even for a permutation. Rebuilding can
    /// require intermediates larger than the input or result.
    ///
    /// # Errors
    ///
    /// Returns [`OperationError::VariableNotInVtree`] for an absent source or target,
    /// [`OperationError::DuplicateVariable`] for a repeated source,
    /// [`OperationError::MarginalLevel`] for discarded structure, or
    /// [`OperationError::OverBudget`] if an allocation is refused.
    /// All map entries are validated before rebuilding.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{Tdd, Vtree};
    /// let vtree = Arc::new(Vtree::balanced(2));
    /// use tididi::vtree::VarId;
    /// let f = Tdd::cube(&vtree, [1, -2])?;
    /// let swapped = f.rename_vars(&[(VarId(0), VarId(1)), (VarId(1), VarId(0))])?;
    /// assert!(swapped.equivalent(&Tdd::cube(&vtree, [-1, 2])?)?);
    /// # Ok::<(), tididi::OperationError>(())
    /// ```
    pub fn rename_vars(self, renames: &[(VarId, VarId)]) -> Result<Tdd, OperationError> {
        let context = Arc::clone(self.context());
        context.run(|eng| eng.rename_vars(self, renames))
    }

    /// Simultaneously replace variables with structural Boolean functions.
    ///
    /// Each source variable appears once; omitted sources keep their meaning.
    /// Replacements are used as given, without recursively substituting inside
    /// them. All diagrams must be structural and share the same vtree allocation
    /// and output level. The result retains this diagram's literal weights;
    /// replacement weights are ignored.
    ///
    /// Consumes this diagram on success and error and borrows replacements without
    /// changing them. A nonempty map produces a minimized result; an empty map
    /// returns the operand unchanged after checking structure. Rebuilding can
    /// require intermediate diagrams larger than the inputs or result.
    ///
    /// # Errors
    ///
    /// Rejects absent or duplicate source variables, replacement vtree or root
    /// mismatches, and marginal levels before rebuilding. Returns
    /// [`OperationError::OverBudget`] if an allocation is refused.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{Tdd, Vtree};
    /// let vtree = Arc::new(Vtree::balanced(3));
    /// let f = Tdd::cube(&vtree, [1, -2])?;
    /// let replacement = Tdd::clause(&vtree, [2, 3])?;
    /// let g = f.substitute(&[(tididi::vtree::VarId(0), &replacement)])?;
    /// assert!(g.equivalent(&Tdd::cube(&vtree, [-2, 3])?)?);
    /// # Ok::<(), tididi::OperationError>(())
    /// ```
    pub fn substitute(self, replacements: &[(VarId, &Tdd)]) -> Result<Tdd, OperationError> {
        let context = Arc::clone(self.context());
        context.run(|eng| eng.substitute(self, replacements))
    }

    /// Simplify this function where only assignments satisfying `care` matter.
    ///
    /// The result `g` is no larger than this function `f` and satisfies
    /// `g ∧ care == f ∧ care`; outside the care set its value is unspecified.
    /// Both operands must share a vtree allocation, but their output levels may
    /// differ. Both are consumed on success and error.
    ///
    /// [`RestrictionOutcome`] distinguishes an unchanged operand, a shrunk diagram,
    /// and a false result; [`RestrictionOutcome::into_tdd`] extracts the diagram.
    /// A shrunk result may need minimization. A false operand is returned unchanged;
    /// otherwise false `care` produces an unsatisfiable outcome.
    ///
    /// Marginal levels in this function are retained; marginal levels in `care`
    /// place no constraint on their discarded subtrees. Every outcome retains this
    /// function's weights and arithmetic; the care set's weights are ignored.
    ///
    /// # Errors
    ///
    /// Returns [`OperationError::VtreeMismatch`] for different vtree allocations,
    /// or [`OperationError::OverBudget`] if an allocation is refused.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{and, literal, Tdd, Vtree};
    /// let vtree = Arc::new(Vtree::balanced(3));
    /// let f = Tdd::clause(&vtree, [1, 2])?;
    /// let care = literal(&vtree, 1)?;
    /// let g = f.clone().restrict_to_care(care.clone())?.into_tdd();
    /// assert!(and(g, care.clone())?.equivalent(&and(f, care)?)?);
    /// # Ok::<(), tididi::OperationError>(())
    /// ```
    pub fn restrict_to_care(self, care: Tdd) -> Result<RestrictionOutcome, OperationError> {
        let context = Arc::clone(self.context());
        context.run(|eng| eng.restrict_to_care(self, care))
    }

    /// Whether two structural diagrams represent the same Boolean function.
    ///
    /// Borrows both operands without changing them; they must share the same vtree
    /// allocation and output level. Weights are ignored. The inputs need not be
    /// minimal or use the same node numbering or pair order: checked copies are
    /// minimized and compared. This needs temporary storage proportional to the
    /// diagrams, plus sorting of each node's pairs.
    ///
    /// # Errors
    ///
    /// Returns a vtree or root mismatch, [`OperationError::MarginalLevel`] for
    /// discarded structure, or [`OperationError::OverBudget`] for refused allocation.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{and, literal, or, Vtree};
    /// let vtree = Arc::new(Vtree::balanced(2));
    /// let x = literal(&vtree, 1)?;
    /// let y = literal(&vtree, 2)?;
    /// let absorbed = or(x.clone(), and(x.clone(), y.clone())?)?;
    /// assert!(x.equivalent(&absorbed)?); // x OR (x AND y) = x
    /// assert!(!x.equivalent(&y)?); // equal counts do not imply equivalence
    /// # Ok::<(), tididi::OperationError>(())
    /// ```
    pub fn equivalent(&self, other: &Tdd) -> Result<bool, OperationError> {
        let context = Arc::clone(self.context());
        context.run(|eng| eng.equivalent(self, other))
    }

    /// Whether every model of this structural function satisfies `other`.
    ///
    /// Both diagrams are borrowed and unchanged. They must share the same vtree
    /// allocation and output level; attached weights are ignored. Checks whether
    /// `self ∧ ¬other` is false using copies, so intermediate storage can exceed
    /// both operand sizes. The inputs need not be minimized.
    ///
    /// # Errors
    ///
    /// Returns a vtree or root mismatch, [`OperationError::MarginalLevel`] for
    /// discarded structure, or [`OperationError::OverBudget`] for refused allocation.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{literal, Tdd, Vtree};
    /// let vtree = Arc::new(Vtree::balanced(2));
    /// let both = Tdd::cube(&vtree, [1, 2])?;
    /// let x = literal(&vtree, 1)?;
    /// assert!(both.implies(&x)?);
    /// assert!(!x.implies(&both)?);
    /// # Ok::<(), tididi::OperationError>(())
    /// ```
    pub fn implies(&self, other: &Tdd) -> Result<bool, OperationError> {
        let context = Arc::clone(self.context());
        context.run(|eng| eng.implies(self, other))
    }

    /// Return variables that can change this Boolean function, sorted by identifier.
    ///
    /// Borrows a structural diagram and minimizes a copy; the input need not be
    /// minimal and stays unchanged. Constants have empty support. Free variables
    /// in the vtree are excluded, and attached weights are ignored.
    /// [`implied_literals`](Self::implied_literals) instead finds literals true
    /// in every model.
    ///
    /// # Errors
    ///
    /// Returns [`OperationError::MarginalLevel`] for discarded structure, or
    /// [`OperationError::OverBudget`] if an allocation is refused.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{Tdd, Vtree};
    /// let vtree = Arc::new(Vtree::balanced(3));
    /// let f = Tdd::clause(&vtree, [1, 2])?;
    /// assert_eq!(f.support()?, vec![tididi::vtree::VarId(0), tididi::vtree::VarId(1)]);
    /// # Ok::<(), tididi::OperationError>(())
    /// ```
    pub fn support(&self) -> Result<Vec<VarId>, OperationError> {
        let context = Arc::clone(self.context());
        context.run(|eng| eng.support(self))
    }

    /// Return literals true in every model, sorted by variable identifier.
    ///
    /// This is the function's backbone. Both constant functions return an empty
    /// list, including the unsatisfiable function. Borrows a structural diagram
    /// and minimizes a copy; the input need not be minimal and remains unchanged.
    /// Literal weights are ignored.
    ///
    /// # Errors
    ///
    /// Returns [`OperationError::MarginalLevel`] for discarded structure, or
    /// [`OperationError::OverBudget`] if an allocation is refused.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{and, Literal, Tdd, Vtree};
    /// let vtree = Arc::new(Vtree::balanced(3));
    /// let f = and(Tdd::clause(&vtree, [1, 2])?, Tdd::clause(&vtree, [1, -2])?)?;
    /// assert_eq!(f.implied_literals()?, vec![Literal::try_from(1)?]);
    /// # Ok::<(), tididi::OperationError>(())
    /// ```
    pub fn implied_literals(&self) -> Result<Vec<Literal>, OperationError> {
        let context = Arc::clone(self.context());
        context.run(|eng| eng.implied_literals(self))
    }

    /// Count satisfying assignments over every variable in this diagram's vtree.
    ///
    /// Returns an exact arbitrary-precision integer. Each free variable contributes
    /// a factor of two; the diagram need not be minimized. Literal weights on
    /// structural levels are ignored, count-marginal levels use their stored counts,
    /// and the false diagram counts zero. The diagram is borrowed and unchanged.
    /// For repeated counts under observations, use [`counter`](Self::counter).
    ///
    /// # Errors
    ///
    /// Returns [`OperationError::IncompatibleWeights`] for weighted marginal values,
    /// or [`OperationError::OverBudget`] if a buffer allocation is refused.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{Tdd, Vtree};
    /// let vtree = Arc::new(Vtree::balanced(3));
    /// let f = Tdd::clause(&vtree, [1, 2])?;
    /// assert_eq!(f.model_count()?, 6u32.into()); // three choices for x1,x2; two for x3
    /// # Ok::<(), tididi::OperationError>(())
    /// ```
    pub fn model_count(&self) -> Result<num_bigint::BigUint, OperationError> {
        let context = Arc::clone(self.context());
        context.run(|eng| eng.model_count(self))
    }

    /// Whether this structural function has at least one satisfying assignment.
    ///
    /// Borrows the diagram without changing it and accepts nonminimal input.
    /// Literal weights are ignored, so a satisfiable function remains satisfiable
    /// when its weighted value is zero. Checks all levels for discarded structure,
    /// then reads the false sentinel; no minimization or scratch buffers are needed.
    /// Use [`satisfying_assignment`](Self::satisfying_assignment) to obtain a model.
    ///
    /// # Errors
    ///
    /// Returns [`OperationError::MarginalLevel`] for discarded structure.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{and, Tdd, Vtree};
    /// let vtree = Arc::new(Vtree::balanced(2));
    /// let either = Tdd::clause(&vtree, [1, 2])?;
    /// assert!(either.is_sat()?);
    /// let impossible = and(either, Tdd::cube(&vtree, [-1, -2])?)?;
    /// assert!(!impossible.is_sat()?);
    /// # Ok::<(), tididi::OperationError>(())
    /// ```
    pub fn is_sat(&self) -> Result<bool, OperationError> {
        let context = Arc::clone(self.context());
        context.run(|eng| eng.is_sat(self))
    }

    /// Return a complete satisfying assignment, or `None` if the function is false.
    ///
    /// The assignment contains one literal per vtree variable, sorted by identifier,
    /// including free variables. The current traversal assigns false to free leaves;
    /// the chosen model may change after minimization or restructuring. The diagram
    /// must be structural but need not be minimized. It is borrowed and unchanged;
    /// attached weights are ignored.
    ///
    /// The walk and temporary space are linear in the vtree size. Sorting the
    /// returned literals takes O(n log n) for n variables.
    ///
    /// # Errors
    ///
    /// Returns [`OperationError::MarginalLevel`] for discarded structure, or
    /// [`OperationError::OverBudget`] if an allocation is refused.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{Tdd, Vtree};
    /// let vtree = Arc::new(Vtree::balanced(3));
    /// let f = Tdd::cube(&vtree, [1, -2])?;
    /// let model = f.satisfying_assignment()?.unwrap();
    /// assert_eq!(model.len(), 3);
    /// assert!(Tdd::cube(&vtree, model)?.implies(&f)?);
    /// # Ok::<(), tididi::OperationError>(())
    /// ```
    pub fn satisfying_assignment(&self) -> Result<Option<Vec<Literal>>, OperationError> {
        let context = Arc::clone(self.context());
        context.run(|eng| eng.satisfying_assignment(self))
    }

    /// Evaluate this structural diagram in a caller-supplied algebra.
    ///
    /// For literal weights, this sums the weight of every satisfying assignment,
    /// where an assignment's weight is the product of its literal weights.
    /// Nonnegative weights summing to one for each variable define independent
    /// Bernoulli probabilities; other tables give an unnormalized weighted sum.
    ///
    /// The diagram is borrowed, unchanged, and need not be minimized. Every vtree
    /// variable contributes, including free variables through its `One` value.
    /// All values come from `algebra`, independently of attached weights, so later
    /// calls can use new tables without rebuilding. Each table must cover every
    /// variable named by the tree. [`EvalAlgebra`] describes other arithmetic.
    ///
    /// For conditional probabilities, evaluate query conjoined with evidence and
    /// divide by the evidence value under the same weights. A zero evidence value
    /// leaves the conditional probability undefined. Zero values alone do not
    /// establish unsatisfiability: weights can be zero or cancel.
    ///
    /// # Errors
    ///
    /// Returns [`OperationError::MarginalLevel`] for discarded structure, or
    /// [`OperationError::OverBudget`] if a buffer allocation is refused.
    ///
    /// # Panics
    ///
    /// Panics from the caller's algebra propagate, including out-of-range lookups
    /// when a weight table does not cover a vtree variable.
    ///
    /// The example uses `BigRational` from `num-rational`; add that crate as a direct
    /// dependency when using this type in your application.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{Tdd, Vtree};
    /// let vtree = Arc::new(Vtree::balanced(2));
    /// use num_rational::BigRational;
    /// use tididi::diagram::{LiteralWeights, RationalWeights};
    /// let f = Tdd::clause(&vtree, [1, 2])?;
    /// let half = BigRational::new(1.into(), 2.into());
    /// let weights = RationalWeights::from_literals(&vec![
    ///     LiteralWeights { negative: half.clone(), positive: half }; 2
    /// ]);
    /// assert_eq!(f.evaluate(&weights)?, BigRational::new(3.into(), 4.into()));
    /// # Ok::<(), tididi::OperationError>(())
    /// ```
    pub fn evaluate<S: EvalAlgebra>(&self, algebra: &S) -> Result<S::Value, OperationError> {
        let context = Arc::clone(self.context());
        context.run(|eng| eng.evaluate(self, algebra))
    }

    /// Evaluate this diagram using its attached weight store, if present.
    ///
    /// Structural levels are folded from literal weights; weighted marginal levels
    /// contribute their stored values. The returned [`WeightValue`] uses the store's
    /// arithmetic. Returns `Ok(None)` without a weight store. The borrowed diagram
    /// is unchanged. A zero value may come from zero or cancelling weights and does
    /// not establish Boolean unsatisfiability.
    ///
    /// # Errors
    ///
    /// Returns [`OperationError::OverBudget`] if a scratch allocation is refused.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{Tdd, Vtree};
    /// let vtree = Arc::new(Vtree::balanced(2));
    /// use num_rational::BigRational;
    /// use tididi::diagram::{Arithmetic, LiteralWeights, RationalWeights, WeightStore};
    /// let mut f = Tdd::clause(&vtree, [1, 2])?;
    /// let half = BigRational::new(1.into(), 2.into());
    /// let weights = RationalWeights::from_literals(&vec![
    ///     LiteralWeights { negative: half.clone(), positive: half }; 2
    /// ]);
    /// f.set_weights(WeightStore::new(weights, Arithmetic::ExactRational))?;
    /// assert_eq!(f.weighted_value()?.unwrap().into_rational(),
    ///            BigRational::new(3.into(), 4.into()));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn weighted_value(&self) -> Result<Option<WeightValue>, OperationError> {
        let context = Arc::clone(self.context());
        context.run(|eng| eng.weighted_value(self))
    }

    /// Minimize this diagram under its current vtree, preserving its function and count.
    ///
    /// For a structural diagram, the result is canonical and minimal up to node
    /// numbering and pair order. Removes unreachable nodes and contracts nodes
    /// with identical parent contexts. Marginal levels stay marginal, with their
    /// applicable value and pair cleanup passes applied.
    ///
    /// This is fixed-vtree minimization as described in
    /// [Section 5 of the TDD paper](https://arxiv.org/html/2604.05537v1#S5).
    /// [`rotation_search`](Self::rotation_search) searches other tree shapes.
    /// Counting, satisfiability and witness queries accept nonminimal diagrams;
    /// minimize when canonical form or removal of redundant storage is needed.
    ///
    /// # Errors
    ///
    /// Returns [`OperationError::OverBudget`] if an allocation is refused.
    /// Completed passes remain applied: the diagram is well-formed and count-correct
    /// at the last completed pass boundary, and can still be queried or retried.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{and, literal, Tdd, Vtree};
    /// let vtree = Arc::new(Vtree::balanced(3));
    /// let mut f = and(Tdd::clause(&vtree, [1, 2])?, literal(&vtree, 3)?)?;
    /// let before = f.model_count()?;
    /// f.minimize()?;
    /// assert_eq!(f.model_count()?, before);
    /// # tididi::test_helpers::assert_canonical(&f);
    /// # Ok::<(), tididi::OperationError>(())
    /// ```
    pub fn minimize(&mut self) -> Result<(), OperationError> {
        let context = Arc::clone(self.context());
        context.run(|eng| eng.minimize(self))
    }

    /// Run the reduction passes selected by `plan`, preserving function and count.
    ///
    /// A partial plan need not establish canonical form; [`minimize`](Self::minimize)
    /// runs the full default plan. Marginal levels remain marginal. The plan selects
    /// pruning, contraction, or a full pass with its content-twin policy.
    ///
    /// # Errors
    ///
    /// Returns [`OperationError::OverBudget`] if an allocation is refused. Each pass
    /// reserves growth before mutation, so completed passes remain applied and the
    /// diagram stays well-formed and count-correct at the last pass boundary.
    /// It can still be queried or retried.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{and, literal, Tdd, Vtree};
    /// let vtree = Arc::new(Vtree::balanced(3));
    /// use tididi::reduce::ReductionPlan;
    /// let mut f = and(Tdd::clause(&vtree, [1, 2])?, literal(&vtree, 3)?)?;
    /// let before = f.model_count()?;
    /// f.reduce(ReductionPlan::Prune)?;
    /// assert_eq!(f.model_count()?, before);
    /// # Ok::<(), tididi::OperationError>(())
    /// ```
    pub fn reduce(&mut self, plan: crate::reduce::ReductionPlan<'_>) -> Result<(), OperationError> {
        let context = Arc::clone(self.context());
        context.run(|eng| eng.reduce(self, plan))
    }

    /// Replace selected subtrees with per-node counts or fixed weighted values.
    ///
    /// Targets are vtree node indices, not variable ids. An internal target also
    /// marginalizes its descendants; already marginal levels are skipped and a
    /// false diagram stays false. Without attached weights, the values are exact
    /// model counts. With a [`WeightStore`](crate::diagram::WeightStore) attached
    /// through [`set_weights`](Self::set_weights), they use that store's arithmetic
    /// and remain readable through [`weighted_value`](Self::weighted_value).
    /// Attach weights before marginalizing: counts cannot later be converted to
    /// arbitrary weights.
    ///
    /// The discarded structure cannot be recovered. Later operations cannot
    /// constrain its variables or recover their assignments; structural serialization
    /// and Boolean queries may reject the result. [`exists_vars`](Self::exists_vars)
    /// instead retains a Boolean function and merges satisfying extensions.
    ///
    /// Completed levels retain their values and valid parent references. The pass
    /// also fuses eligible marginal pairs and removes unused or duplicate value
    /// slots. Rounded log arithmetic skips fusion that changes arithmetic order.
    ///
    /// # Errors
    ///
    /// Returns [`OperationError::LevelNotInVtree`] for an invalid target before
    /// mutation. [`OperationError::OverBudget`] may leave a completed prefix whose
    /// counts or weighted values remain readable and preserved. An unfinished
    /// column is discarded before installation; retrying finishes remaining levels
    /// and cleanup.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{and, Tdd, Vtree};
    /// let vtree = Arc::new(Vtree::balanced(4));
    /// let (left, _) = vtree.children(vtree.root());
    /// let mut f = and(Tdd::clause(&vtree, [1, 2])?, Tdd::clause(&vtree, [3, 4])?)?;
    /// let before = f.model_count()?;
    /// f.marginalize_levels(&[left])?;
    /// assert!(f.level(left).is_marginal());
    /// assert_eq!(f.model_count()?, before);
    /// # Ok::<(), tididi::OperationError>(())
    /// ```
    pub fn marginalize_levels(&mut self, levels: &[VtreeIdx]) -> Result<(), OperationError> {
        let context = Arc::clone(self.context());
        context.run(|eng| eng.marginalize_levels(self, levels))
    }

    /// Search nearby vtree shapes while preserving this function and its count.
    ///
    /// Each sweep probes left and right rotations at internal nodes, accepting
    /// a move when the objective delta is negative. Sweeps stop after one accepts
    /// no move or the configured maximum is reached. The returned
    /// [`RotationSearchStats`](crate::restructure::search::RotationSearchStats)
    /// records probes, accepted rotations and sweeps. Rotations retain variable ids
    /// and the execution context. If the vtree is shared, the diagram gets a private
    /// copy, so other diagrams keep their shape and no longer share its allocation.
    ///
    /// Structural diagrams are minimized first. A diagram with marginal levels must
    /// arrive canonical; trials touching marginal levels are skipped. Accepted
    /// rotations preserve canonical form and counts, including marginal values.
    ///
    /// # Errors
    ///
    /// The initial minimization can return [`OperationError::OverBudget`]; its
    /// partial-result contract is described by [`minimize`](Self::minimize).
    ///
    /// # Panics
    ///
    /// If the objective panics, its current trial is rolled back before unwinding;
    /// earlier accepted rotations remain committed.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{and, literal, Tdd, Vtree};
    /// let vtree = Arc::new(Vtree::balanced(4));
    /// use tididi::diagram::TddLevel;
    /// use tididi::restructure::search::{RotationObjective, RotationSearchConfig};
    /// struct MinSize;
    /// impl RotationObjective for MinSize {
    ///     fn delta(&mut self, b: (&TddLevel, &TddLevel), a: (&TddLevel, &TddLevel)) -> i64 {
    ///         (a.0.slot_count() + a.1.slot_count()) as i64
    ///             - (b.0.slot_count() + b.1.slot_count()) as i64
    ///     }
    /// }
    /// let mut f = and(Tdd::clause(&vtree, [1, 2])?, Tdd::clause(&vtree, [3, 4])?)?;
    /// let before = f.model_count()?;
    /// f.rotation_search(&mut MinSize, &RotationSearchConfig::default())?;
    /// assert_eq!(f.model_count()?, before);
    /// let extra = literal(f.vtree(), 1)?;
    /// let constrained = and(f, extra)?;
    /// assert!(constrained.is_sat()?);
    /// # Ok::<(), tididi::OperationError>(())
    /// ```
    pub fn rotation_search<O: crate::restructure::search::RotationObjective>(&mut self, objective: &mut O, config: &crate::restructure::search::RotationSearchConfig) -> Result<crate::restructure::search::RotationSearchStats, OperationError> {
        let context = Arc::clone(self.context());
        context.run(|eng| eng.rotation_search(self, objective, config))
    }
}
