//! Specialized diagram × clause conjunction.
//!
//! The clause is never built as a diagram. At each vtree level `t` it stands for
//! exactly two functions over that subtree's variables:
//!
//! - **`c_t`** — the clause restricted to subtree `t`: at least one of its
//!   literals inside `t` is satisfied;
//! - **`d_t`** — the complement of `c_t` within the subtree: none of them is.
//!
//! Together they partition the subtree's assignments, which is what lets one
//! bottom-up walk carry both branches of the clause at once: a level's output
//! for an accumulator node is its conjunction with `c_t` and with `d_t`, kept
//! side by side in `cd_map`. The whole file is written in terms of this pair.

use crate::engine::Engine;
use crate::limits::pool::Pool;
use crate::apply::scoped_flags::{ScopedFlags, FlagBuffer};
use std::sync::Arc;

use crate::diagram::Literal;
use crate::vtree::{Vtree, VtreeIdx};
use crate::apply::leaf::CONJOIN_GRID;
use crate::diagram::{self, *};

use crate::limits::OperationError;
use crate::apply::conjoin::budget::{reserve_pairs_for_emit, NO_PRODUCT};

mod spine;
use spine::*;
mod emit;
use emit::*;
mod pairs;
use pairs::*;
mod rebuild;
use rebuild::*;

/// An element accepted by [`Tdd::and_clause`] and [`Engine::and_clause`].
///
/// Implemented for [`Literal`] and signed, one-based `i32` literals. Pass an
/// array, slice or vector of either type. Typed literals are borrowed directly;
/// integers are validated and converted under the operation's allocation limits.
/// This trait is sealed.
pub trait ClauseLiteral: input::Sealed {}

impl ClauseLiteral for Literal {}
impl ClauseLiteral for i32 {}

mod input {
    use super::*;

    pub trait Sealed: Sized {
        /// Convert the clause when necessary, then invoke the typed conjunction.
        fn conjoin(eng: &Engine, f: Tdd, clause: &[Self]) -> Result<Tdd, OperationError>;
    }

    impl Sealed for Literal {
        #[inline]
        fn conjoin(eng: &Engine, f: Tdd, clause: &[Self]) -> Result<Tdd, OperationError> {
            conjoin_clause_owned(eng, f, clause)
        }
    }

    impl Sealed for i32 {
        fn conjoin(eng: &Engine, f: Tdd, clause: &[Self]) -> Result<Tdd, OperationError> {
            let lim = eng.limits();
            let _op = lim.begin_operation();
            if lim.should_stop() { return Err(OperationError::Stopped); }
            let mut gate = crate::limits::PollGate::new(lim.reduce_poll_stride());
            let mut literals = Vec::new();
            for &value in clause {
                lim.poll(&mut gate, 1)?;
                let literal = Literal::try_from(value)?;
                if f.vtree().leaf_of(literal.var).is_none() {
                    return Err(OperationError::VariableNotInVtree(literal.var));
                }
                lim.try_push(&mut literals, literal)?;
            }
            lim.flush_poll(&mut gate)?;
            conjoin_clause_owned(eng, f, &literals)
        }
    }
}

/// Every buffer one engine's clause conjunctions reuse between calls.
///
/// The two flag arrays hold an all-false invariant between calls: only spine
/// entries are ever set, and the rollback log resets them on every exit.
#[derive(Default)]
pub(crate) struct ClauseScratch {
    /// Maps accumulator node index → `[ct, dt]` output indices for conjunction
    /// with the clause's `c_t` / `d_t` virtual nodes, interleaved so one
    /// per-pair lookup serves both lanes; the `dt` lane is written iff
    /// `need_dt` for the level.
    cd_map: Pool<Vec<[u32; 2]>>,
    /// Cumulative offsets into `cd_map`, one per vtree level.
    level_base: Pool<Vec<usize>>,
    /// Per-level flags: `on_spine[t]` = clause has variables in subtree t.
    on_spine: Pool<FlagBuffer>,
    /// Per-level flags: `need_dt[t]` = must compute complement conjunction at t.
    need_dt: Pool<FlagBuffer>,
    /// Spine internal nodes in bottom-up (post-order) order.
    spine_internal: Pool<Vec<VtreeIdx>>,
    /// Work stack for the post-order spine walk (node, processed?).
    dfs_stack: Pool<Vec<(VtreeIdx, bool)>>,
}

impl ClauseScratch {
    /// Release every retained buffer, leaving the pools empty.
    pub(crate) fn drain(&self) {
        self.cd_map.drain();
        self.level_base.drain();
        self.on_spine.drain();
        self.need_dt.drain();
        self.spine_internal.drain();
        self.dfs_stack.drain();
    }
}

/// Conjoin `clause` into `f`, moving `f`'s levels and weights into the result;
/// `f` is left empty on `Err` as well as on `Ok`.
///
/// # Errors
/// Returns the [`OperationError`] the conjunction stopped on.
pub(crate) fn conjoin_clause_into(eng: &Engine, f: &mut Tdd, clause: &[Literal]) -> Result<Tdd, OperationError> {
    let lim = eng.limits();
    let _op = lim.begin_operation();
    if lim.should_stop() { return Err(OperationError::Stopped); }
    let mut gate = crate::limits::PollGate::new(lim.reduce_poll_stride());
    let pool = eng.clause_pool();
    let vtree = &f.vtree;
    let num_nodes = vtree.num_nodes();
    for lit in clause {
        lim.poll(&mut gate, 1)?;
        let leaf = vtree.leaf_of(lit.var).ok_or(OperationError::VariableNotInVtree(lit.var))?;
        f.require_structure_at(leaf)?;
    }

    lim.flush_poll(&mut gate)?;
    if f.is_zero() {
        let levels = diagram::try_take_levels(eng, num_nodes)?;
        let mut out = Tdd::try_from_levels_on(eng,
            Arc::clone(vtree),
            levels,
            TddNodeId { vtree: f.output.vtree, local: ZERO },
        )?;
        out.weights = f.weights.take();
        return Ok(out);
    }

    // The empty clause is false, so conjoining it gives ⊥ whatever `f` is.
    if clause.is_empty() {
        let levels = diagram::try_take_levels(eng, num_nodes)?;
        let mut out = Tdd::try_from_levels_on(eng, Arc::clone(vtree), levels, TddNodeId { vtree: vtree.root(), local: ZERO })?;
        out.weights = f.weights.as_ref().map(WeightStore::empty_like);
        return Ok(out);
    }

    // A variable named in both polarities satisfies the clause whatever its
    // value, so conjoining it is the identity. The rebuild below keeps one
    // column per variable of the clause and cannot say that.
    if crate::diagram::is_tautological(lim, clause)? {
        let vtree = Arc::clone(vtree);
        let output = f.output;
        let mut out =
            Tdd::try_from_levels_on(eng, vtree, std::mem::take(&mut f.levels), output)?;
        out.weights = f.weights.take();
        return Ok(out);
    }

    // The clause spine — the Steiner tree of its variables' leaves — and the
    // `need_dt` flag propagated top-down over it.
    let mut on_spine = ScopedFlags::take(lim, &pool.on_spine, num_nodes)?;
    let mut spine_internal = pool.spine_internal.take();
    let mut dfs_stack = pool.dfs_stack.take();
    build_clause_spine(lim, vtree, clause, &mut on_spine, &mut spine_internal, &mut dfs_stack)?;
    let mut need_dt = ScopedFlags::take(lim, &pool.need_dt, num_nodes)?;
    propagate_need_dt(vtree, &spine_internal, &on_spine, &mut need_dt);

    // Take ownership of f's levels: off-spine levels pass through as the
    // identity, spine internal levels are rebuilt in place below, and f is
    // left with empty levels.
    let out_vtree = f.output.vtree;
    let out_local_in = f.output.local;
    let mut levels = std::mem::take(&mut f.levels);
    // The accumulator's marginal values move to the output along with its levels:
    // a clause carries none of its own, and the output is nothing but the
    // accumulator, one clause further on.
    let f_weights = f.weights.take();

    // Per-level base offsets into `cd_map`: only spine levels get storage, so
    // the map is `O(Σ spine widths)` rather than `O(|f|)`; off-spine levels
    // are read through raw pair indices. See `plan_cd_map_bases`.
    let mut level_base = pool.level_base.take();
    lim.try_resize(&mut level_base, num_nodes, 0usize)?;
    let total = plan_cd_map_bases(vtree, clause, &spine_internal, &levels, &mut level_base)?;

    // The base blocks partition `[0, total)` with no gaps and every `c_t`
    // entry is written once below, so no bulk `NO_PRODUCT` fill is needed. A
    // `d_t` lane is written iff `need_dt[t]`, and a read of one implies
    // `need_dt` on that child, so a stale lane is never read.
    let mut cd_map = pool.cd_map.take();
    lim.try_resize(&mut cd_map, total, [NO_PRODUCT, NO_PRODUCT])?;

    fill_leaf_maps(vtree, clause, &level_base, &need_dt, &mut cd_map);

    // Pair buffers reused across the per-level and per-node loops.
    let mut clause_dt_pairs: Vec<ChildPair> = Vec::new();  // f × d_t pairs
    // "Type 3" pairs (dt_L, ct_R) of the both-relevant case have larger left
    // indices than type 1/2 pairs, so they are buffered and flushed after
    // them to keep the sorted order.
    let mut clause_t3_buf: Vec<ChildPair> = Vec::new();

    // Rebuild each spine internal level bottom-up. Children's maps are fully
    // written before any parent reads them. The stop axis and the output cap
    // are checked after each level, the cap against the rebuilt levels' nodes.
    let mut out_nodes = 0u64;
    let mut tables = ClauseTables {
        cd_map: &mut cd_map,
        level_base: &level_base,
        need_dt: &need_dt,
        on_spine: &on_spine,
        t3_buf: &mut clause_t3_buf,
        dt_pairs: &mut clause_dt_pairs,
    };
    for &t in &spine_internal {
        rebuild_spine_level(eng, t, vtree, &mut levels, &mut tables)?;
        out_nodes += levels[t.idx()].slot_count() as u64;
        lim.level_done(out_nodes)?;
    }

    // Output: conjunction of f's output with c_t at the root. `out_vtree` is
    // an ancestor of every clause leaf, so it is on the spine and its ct_map
    // block is filled.
    let out_base = level_base[out_vtree.idx()];
    let ct_out = cd_map[out_base + out_local_in.idx()][0];
    let out_local = if ct_out != NO_PRODUCT { NodeIdx(ct_out) } else { ZERO };

    // Contract seed: this clause's spine, which is ancestor-closed, so an
    // off-spine level and its whole subtree are the accumulator's own bytes;
    // whatever the accumulator still owed is carried over, which keeps this
    // exact for a caller that does not minimize between clauses.
    let vtree = Arc::clone(vtree);
    let carried = f.take_worklists();
    let mut out = Tdd::try_with_levels_dirty(
        eng,
        vtree,
        levels,
        TddNodeId { vtree: out_vtree, local: out_local },
        carried,
        &spine_internal,
    )?;
    out.weights = f_weights;

    // `level_base` needs no reset — every spine entry is rewritten each call
    // and irrelevant entries are never read.
    pool.cd_map.put_bounded(cd_map);
    pool.level_base.put(level_base);
    pool.spine_internal.put(spine_internal);
    pool.dfs_stack.put(dfs_stack);

    Ok(out)
}

/// Conjoin `clause` into `f` under `eng`'s limits, recycling the accumulator's
/// levels when it still owns any. The implementation behind
/// [`Engine::and_clause`](crate::Engine::and_clause).
///
/// # Errors
///
/// Returns `Err(OperationError::OverBudget)` if any internal allocation is refused.
pub(crate) fn conjoin_clause_owned(eng: &Engine, mut f: Tdd, clause: &[Literal]) -> Result<Tdd, OperationError> {
    let result = conjoin_clause_into(eng, &mut f, clause);
    // Recycle what is left of `f` only if it is a real level array: on every
    // path but the zero early-out `f` is left empty, and parking an empty Vec
    // would evict the warm entry the one-slot pool holds.
    let spent = std::mem::take(&mut f.levels);
    if !spent.is_empty() {
        diagram::return_levels(eng, diagram::PoolSlot::First, spent);
    }
    result
}

impl Tdd {
    /// Build a canonical disjunction of literals over the shared vtree.
    ///
    /// Integers are signed and one-based; typed [`Literal`] values also work.
    /// Repeated literals are ignored. Both polarities of a variable make the
    /// clause true; an empty clause is false. Unmentioned variables remain free.
    /// Uses the execution context attached to the vtree.
    ///
    /// # Errors
    ///
    /// Returns [`OperationError::InvalidLiteral`] for integer zero,
    /// [`OperationError::VariableNotInVtree`] for an absent variable, or
    /// [`OperationError::OverBudget`] if an allocation is refused.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{Literal, Tdd, Vtree};
    /// use tididi::vtree::VarId;
    /// let vtree = Arc::new(Vtree::balanced(3));
    /// let f = Tdd::clause(&vtree, [Literal::pos(VarId(0)), Literal::neg(VarId(1))])?;
    /// assert_eq!(f.model_count()?, 6u32.into());
    /// # tididi::test_helpers::assert_canonical(&f);
    /// # Ok::<(), tididi::OperationError>(())
    /// ```
    pub fn clause(vtree: &Arc<Vtree>, literals: impl IntoIterator<Item = impl TryInto<Literal, Error: Into<OperationError>>>) -> Result<Tdd, OperationError> {
        vtree.context().run(|eng| eng.clause(vtree, literals))
    }
}

/// The clause entry points on a caller's engine.
impl crate::engine::Engine {
    /// Run [`Tdd::clause`](crate::Tdd::clause) using this batch's scratch and resource limits.
    ///
    /// Operand requirements, ownership and result semantics follow the diagram method.
    ///
    /// # Errors
    ///
    /// Returns the operation's errors or [`OperationError::Stopped`]
    /// on cancellation. Allocation refusals return
    /// [`OperationError::OverBudget`]. An exceeded output-node cap returns
    /// [`OperationError::OutputCap`].
    ///
    /// The output cap applies to the initial true diagram and rebuilt spine nodes.
    pub fn clause(
        &self,
        vtree: &Arc<Vtree>,
        literals: impl IntoIterator<Item = impl TryInto<Literal, Error: Into<OperationError>>>,
    ) -> Result<Tdd, OperationError> {
        let lim = self.limits();
        let _op = lim.begin_operation();
        if lim.should_stop() { return Err(OperationError::Stopped); }
        let mut gate = crate::limits::PollGate::new(lim.reduce_poll_stride());
        let mut clause = Vec::new();
        for lit in literals {
            lim.poll(&mut gate, 1)?;
            let lit: Literal = lit.try_into().map_err(Into::into)?;
            if vtree.leaf_of(lit.var).is_none() { return Err(OperationError::VariableNotInVtree(lit.var)); }
            lim.try_push(&mut clause, lit)?;
        }
        lim.flush_poll(&mut gate)?;
        let one = self.cube(vtree, std::iter::empty::<Literal>())?;
        conjoin_clause_owned(self, one, &clause)
    }

    /// Run [`Tdd::and_clause`](crate::Tdd::and_clause) using this batch's scratch and resource limits.
    ///
    /// Operand requirements, ownership and result semantics follow the diagram method.
    ///
    /// # Errors
    ///
    /// Returns the operation's errors or [`OperationError::Stopped`]
    /// on cancellation. Allocation refusals return
    /// [`OperationError::OverBudget`]. An exceeded output-node cap returns
    /// [`OperationError::OutputCap`].
    ///
    /// Borrow a slice of signed integers or typed literals.
    /// Stop and output limits are checked once per rebuilt level; the output cap
    /// counts nodes in the levels rebuilt so far.
    pub fn and_clause<L: ClauseLiteral>(&self, f: Tdd, clause: &[L]) -> Result<Tdd, OperationError> {
        L::conjoin(self, f, clause)
    }
}

#[cfg(test)]
mod tests;
