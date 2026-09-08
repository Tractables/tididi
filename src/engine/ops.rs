//! The engine's operations: every entry point that consumes diagrams, honours
//! the engine's limits, and hands a refusal back instead of panicking on it.
//!
//! These are the implementations. The free functions and operator sugar
//! elsewhere in the crate build a transient engine, call one of these, and
//! `expect` the result.

use std::sync::Arc;

use crate::apply::{BatchMerge, CareCanonical, Restricted, Spine};
use crate::diagram::{Literal, Tdd};
use crate::engine::Engine;
use crate::error::ApplyError;
use crate::restructure::search::{RotationObjective, RotationSearchConfig, RotationSearchStats};
use crate::vtree::{VarId, Vtree};

impl Engine {
    /// Conjoin two diagrams over the same vtree.
    ///
    /// Both operands are consumed on `Err` as well as on `Ok`: the product
    /// construction drains their level arenas as it walks bottom-up and
    /// recycles the storage into the result. Clone one first if you need to
    /// keep it, and never reuse an operand after a call.
    ///
    /// # Errors
    ///
    /// [`ApplyError::OverBudget`] when a buffer reservation is refused (the
    /// allocator or the armed soft budget), [`ApplyError::OutputCap`] on the
    /// output-node cap, [`ApplyError::Deadline`] on the armed deadline or a
    /// stop decision.
    ///
    /// # Panics
    ///
    /// Panics if the operands do not share a vtree, or their outputs sit at
    /// different vtree nodes.
    pub fn and(&self, f: Tdd, g: Tdd) -> Result<Tdd, ApplyError> {
        crate::apply::conjoin::conjoin_owned(self, f, g, None)
    }

    /// [`Engine::and`], emitting the named vtree levels as streaming-marginal
    /// instead of explicit — the levels are summed out as the product is
    /// built rather than in a pass after it.
    ///
    /// `targets` is indexed by [`VtreeIdx`]: `true` at index `t` marginalizes
    /// the output's level `t`.
    ///
    /// # Errors
    ///
    /// As [`Engine::and`].
    pub fn and_marginalizing(&self, f: Tdd, g: Tdd, targets: &[bool]) -> Result<Tdd, ApplyError> {
        crate::apply::conjoin::conjoin_owned(self, f, g, Some(targets))
    }

    /// Conjoin one clause into a diagram without building the clause as a
    /// diagram of its own: only the levels on the clause's spine are rebuilt.
    ///
    /// The operand is consumed either way, as in [`Engine::and`].
    ///
    /// # Errors
    ///
    /// As [`Engine::and`].
    pub fn and_clause(&self, f: Tdd, clause: &[Literal]) -> Result<Tdd, ApplyError> {
        crate::apply::conjoin_clause::conjoin_clause_owned(self, f, clause)
    }

    /// Conjoin a small batch into a large accumulator by rebuilding only the
    /// levels the batch can reach — the ancestor closure of `spine`.
    ///
    /// Declines rather than fails when the shape does not suit the restricted
    /// merge, returning both operands untouched in
    /// [`BatchMerge::Declined`] for the caller to conjoin the ordinary way.
    ///
    /// # Errors
    ///
    /// As [`Engine::and`].
    pub fn and_batch(
        &self,
        acc: Tdd,
        batch: Tdd,
        spine: &Spine<'_>,
    ) -> Result<BatchMerge, ApplyError> {
        crate::apply::conjoin::conjoin_batch(self, acc, batch, spine)
    }

    /// Disjoin two diagrams over the same vtree, by De Morgan over
    /// [`Engine::and`].
    ///
    /// Both operands are consumed, as in [`Engine::and`]. Each negation fills
    /// its operand out to full structure first, so this can grow the diagram.
    ///
    /// # Errors
    ///
    /// As [`Engine::and`].
    pub fn or(&self, f: Tdd, g: Tdd) -> Result<Tdd, ApplyError> {
        crate::apply::disjoin::disjoin_owned(self, f, g)
    }

    /// Condition `x` to a constant `value`, removing it from the result (cofactor).
    /// Marginal-safe: unlike `project_var`, this only rewrites x's leaf-parent level
    /// (drops the opposite-polarity pairs, fixes the kept side to One) and never calls
    /// `apply_or`, so it is sound when sibling levels are marginal (mc mode). Restriction
    /// is monotone non-increasing in size — it can never blow up like a general apply.
    #[must_use]
    pub fn condition_var(&self, f: &Tdd, x: VarId, value: bool) -> Tdd {
        crate::apply::condition::condition_var_on(self, f, x, value)
    }

    /// Condition a SET of variables to the same constant `value`, removing them all,
    /// with a SINGLE `minimize` at the end (vs one per var in `condition_var`). Much
    /// cheaper when conditioning many copies of one hub on a large diagram. Marginal-safe
    /// for the same reason as `condition_var`. Like `condition_var`, the kept side is set
    /// to One (free) — the caller must divide the final count by 2^(#vars conditioned).
    #[must_use]
    pub fn condition_vars(&self, f: &Tdd, vars: &[VarId], value: bool) -> Tdd {
        crate::apply::condition::condition_vars_on(self, f, vars, value)
    }

    /// Returns a fully minimized canonical TDD representing ∃x. t.
    ///
    /// Precondition: `x` must be a leaf in `t.vtree`, and no ancestor of x's leaf
    /// may be a marginal level (i.e., must be called on a full/non-mc TDD).
    ///
    /// Count convention: the result keeps `t.vtree` unchanged, so `x` remains a
    /// (now don't-care) variable and [`Tdd::model_count`] still ranges over it —
    /// each satisfying assignment of ∃x. t over the remaining variables is counted
    /// twice (once per value of `x`). To count over the remaining variables only,
    /// divide by 2 (by 2^k after projecting k variables).
    ///
    /// ```
    /// use std::sync::Arc;
    /// use num_bigint::BigUint;
    /// use tididi::Tdd;
    /// use tididi::vtree::{VarId, Vtree};
    /// use tididi::Engine;
    ///
    /// let eng = Engine::new();
    /// let vtree = Arc::new(Vtree::balanced(3));
    /// let f = Tdd::clause(&vtree, [1]) & Tdd::clause(&vtree, [2]); // x1 ∧ x2
    /// assert_eq!(f.model_count(), BigUint::from(2u32));
    /// // ∃x2. (x1 ∧ x2) == x1: forgetting x2 frees it, doubling the count.
    /// let g = eng.project_var(&f, VarId(1));
    /// assert_eq!(g.model_count(), BigUint::from(4u32));
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `x` is not a variable present in `t.vtree`.
    #[must_use]
    pub fn project_var(&self, f: &Tdd, x: VarId) -> Tdd {
        crate::apply::project::project_var_on(self, f, x)
    }

    /// Sum every variable in `vars` out of the structure, one at a time.
    #[must_use]
    pub fn project_vars(&self, f: &Tdd, vars: &[VarId]) -> Tdd {
        crate::apply::project::project_vars_on(self, f, vars)
    }

    /// Restriction (generalized cofactor) by dead-marking: see
    /// [`crate::apply::restrict`] for the contract and the algorithm.
    ///
    /// Takes `care` BY VALUE (it may minimize it in place); callers that hand over a
    /// discardable clone lose nothing. `care_canonical` selects the prologue: `Yes`
    /// skips `minimize(care)` when the caller guarantees canonical care (see
    /// [`CareCanonical`]). Returns a [`Restricted`] so the caller can skip the dead
    /// epilogue on `Unchanged`; `.into_tdd(f)` collapses it to a plain `Tdd`.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::Tdd;
    /// use tididi::apply::CareCanonical;
    /// use tididi::vtree::Vtree;
    /// use tididi::Engine;
    ///
    /// let eng = Engine::new();
    /// let vtree = Arc::new(Vtree::balanced(3));
    /// let f = Tdd::clause(&vtree, [1, 2]); // x1 ∨ x2
    /// let care = Tdd::clause(&vtree, [1]); // x1
    /// let g = eng.restrict(&f, care, CareCanonical::No).into_tdd(&f);
    /// // Contract: g agrees with f wherever care holds, i.e. g ∧ x1 == f ∧ x1.
    /// let lhs = g & Tdd::clause(&vtree, [1]);
    /// let rhs = Tdd::clause(&vtree, [1, 2]) & Tdd::clause(&vtree, [1]);
    /// assert_eq!(lhs.model_count(), rhs.model_count());
    /// ```
    #[must_use]
    pub fn restrict(&self, f: &Tdd, care: Tdd, care_canonical: CareCanonical) -> Restricted {
        crate::apply::restrict::restrict_on(self, f, care, care_canonical)
    }

    /// Descend the diagram's vtree greedily by rotation, keeping every probe
    /// the objective scores as an improvement.
    ///
    /// Rotations are pure variable reorders, so the model count is preserved
    /// under any objective. The armed stop is polled once per pivot, which is
    /// what lets a caller bound a search that would otherwise run to a local
    /// minimum.
    ///
    /// # Errors
    ///
    /// [`ApplyError::Deadline`] when the armed deadline passes or a stop
    /// decision concludes the search should end. The diagram is left canonical
    /// and count-correct at whatever local point the search had reached.
    pub fn rotation_search<O: RotationObjective>(
        &self,
        tdd: &mut Tdd,
        objective: &mut O,
        config: &RotationSearchConfig,
    ) -> Result<RotationSearchStats, ApplyError> {
        crate::restructure::search::rotation_search_on(self, tdd, objective, config)
    }


    /// A TDD for one clause over `vtree`, built in this engine's pools.
    ///
    /// The engine-owned form of [`Tdd::clause`]; identical result, and the
    /// per-level buffers stay warm for the next clause.
    #[must_use]
    pub fn clause(
        &self,
        vtree: &Arc<Vtree>,
        lits: impl IntoIterator<Item = impl Into<Literal>>,
    ) -> Tdd {
        let clause: Vec<Literal> = lits.into_iter().map(Into::into).collect();
        crate::build::clause_to_tdd(self, vtree, &clause)
    }

    /// The constant-true function over `vtree`, built in this engine's pools.
    #[must_use]
    pub fn one(&self, vtree: &Arc<Vtree>) -> Tdd {
        crate::build::constant_one(self, vtree)
    }

    /// The constant-false function over `vtree`, built in this engine's pools.
    #[must_use]
    pub fn zero(&self, vtree: &Arc<Vtree>) -> Tdd {
        crate::build::constant_zero(self, vtree)
    }
}
