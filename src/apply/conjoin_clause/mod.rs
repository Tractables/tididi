//! Specialized diagram × clause conjunction, and its dual.
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
//!
//! The same walk disjoins a cube: `¬M` is a clause whose `d_t` is `M_t`, so
//! [`cube`] rides the lanes this one already carries. See that module for the
//! derivation.

use crate::Engine;
use crate::limits::pool::Pool;
use std::sync::Arc;

use crate::diagram::Literal;
use crate::vtree::{Vtree, VtreeIdx};
use super::CONJOIN_GRID;
use crate::diagram::{self, *};

use crate::limits::OperationError;
use crate::apply::conjoin::budget::{reserve_pairs_for_emit, NO_PRODUCT};

mod spine;
use spine::*;
mod pairs;
use pairs::*;
mod rebuild;
use rebuild::*;
mod cube;
use cube::CubeChain;
pub(crate) use cube::disjoin_cube_owned;

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
    on_spine: Pool<MarkBuffer>,
    /// Per-level flags: `need_dt[t]` = must compute complement conjunction at t.
    need_dt: Pool<MarkBuffer>,
    /// Spine internal nodes in bottom-up (post-order) order.
    spine_internal: Pool<Vec<VtreeIdx>>,
    /// Work stack for the post-order spine walk (node, processed?).
    dfs_stack: Pool<Vec<(VtreeIdx, bool)>>,
}

impl ClauseScratch {
    /// Release every retained buffer, leaving the pools empty.
    pub(crate) fn drain(&self, lim: &crate::limits::Limits) {
        self.cd_map.drain(lim);
        self.level_base.drain(lim);
        self.on_spine.drain(lim);
        self.need_dt.drain(lim);
        self.spine_internal.drain(lim);
        self.dfs_stack.drain(lim);
    }
}

/// Conjoin `clause` into `f`, moving `f`'s levels and weights into the result;
/// `f` is left empty on `Err` as well as on `Ok`.
///
/// # Errors
/// Returns the [`OperationError`] the conjunction stopped on.
pub(crate) fn conjoin_clause_into(eng: &Engine, f: &mut Tdd, clause: &[Literal]) -> Result<Tdd, OperationError> {
    spine_walk(eng, f, clause, false)
}

/// The shared bottom-up walk: conjoin `clause` into `f`, or — with `disjoin` —
/// disjoin the cube `clause` negates, by carrying the [`CubeChain`] alongside.
///
/// A disjunction's caller has already handled the operands the two modes read
/// differently; only the tautological clause, which is the false cube, is the
/// identity for both.
fn spine_walk(eng: &Engine, f: &mut Tdd, clause: &[Literal], disjoin: bool) -> Result<Tdd, OperationError> {
    debug_assert!(!disjoin || (!f.is_zero() && !clause.is_empty()),
        "a disjunction's caller answers the false accumulator and the true cube");
    let lim = eng.limits();
    let _op = lim.begin_operation();
    lim.check_stop()?;
    let mut gate = lim.gate();
    let pool = eng.clause_pool();
    let vtree = &f.vtree;
    let num_nodes = vtree.num_nodes();
    for lit in clause {
        gate.poll(1)?;
        let leaf = vtree.leaf_of(lit.var).ok_or(OperationError::VariableNotInVtree(lit.var))?;
        f.require_structure_at(leaf)?;
    }

    gate.flush()?;
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
            Tdd::try_from_levels_on(eng, vtree, std::mem::take(&mut f.levels).into_vec(), output)?;
        out.weights = f.weights.take();
        return Ok(out);
    }

    // The clause spine — the Steiner tree of its variables' leaves — and the
    // `need_dt` flag propagated top-down over it.
    let mut on_spine = SpineMarks::take(lim, &pool.on_spine, num_nodes)?;
    let mut spine_internal = pool.spine_internal.checkout_preserving(lim);
    let mut dfs_stack = pool.dfs_stack.checkout_preserving(lim);
    build_clause_spine(lim, vtree, clause, &mut on_spine, &mut spine_internal, &mut dfs_stack)?;
    let mut need_dt = SpineMarks::take(lim, &pool.need_dt, num_nodes)?;
    propagate_need_dt(vtree, &spine_internal, &on_spine, &mut need_dt);

    // Take ownership of f's levels: off-spine levels pass through as the
    // identity, spine internal levels are rebuilt in place below, and f is
    // left with empty levels.
    let out_vtree = f.output.vtree;
    let out_local_in = f.output.local;
    let mut levels = std::mem::take(&mut f.levels).into_vec();
    // The accumulator's marginal values move to the output along with its levels:
    // a clause carries none of its own, and the output is nothing but the
    // accumulator, one clause further on.
    let f_weights = f.weights.take();

    // Per-level base offsets into `cd_map`: only spine levels get storage, so
    // the map is `O(Σ spine widths)` rather than `O(|f|)`; off-spine levels
    // are read through raw pair indices. See `plan_cd_map_bases`.
    let mut level_base = pool.level_base.checkout_preserving(lim);
    lim.try_resize(&mut level_base, num_nodes, 0usize)?;
    let total = plan_cd_map_bases(vtree, clause, &spine_internal, &levels, &mut level_base)?;

    // The base blocks partition `[0, total)` with no gaps and every `c_t`
    // entry is written once below, so no bulk `NO_PRODUCT` fill is needed. A
    // `d_t` lane is written iff `need_dt[t]`, and a read of one implies
    // `need_dt` on that child, so a stale lane is never read.
    let mut cd_map = pool.cd_map.checkout_preserving(lim);
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
    let mut chain = if disjoin { Some(CubeChain::new(lim, vtree, clause)?) } else { None };
    let mut tables = ClauseTables {
        cd_map: &mut cd_map,
        level_base: &level_base,
        need_dt: &need_dt,
        on_spine: &on_spine,
        t3_buf: &mut clause_t3_buf,
        dt_pairs: &mut clause_dt_pairs,
        output_cube_pair: None,
    };
    for &t in spine_internal.iter() {
        // The `cd_map` block of a level is sized at its width before the
        // rebuild, which is also the range the chain reads back.
        let old_width = levels[t.idx()].slot_count();
        if let Some(chain) = chain.as_ref() {
            // The output level is last in the bottom-up order, so the chain
            // is complete below it and the cube joins the output node here.
            tables.output_cube_pair = (t == out_vtree)
                .then(|| (level_base[t.idx()] + out_local_in.idx(), chain.pair_at(vtree, t)));
        }
        rebuild_spine_level(eng, t, vtree, &mut levels, &mut tables)?;
        if let Some(chain) = chain.as_mut()
            && t != out_vtree
        {
            let base = tables.level_base[t.idx()];
            let lanes = &tables.cd_map[base..base + old_width];
            chain.close_level(eng, t, vtree, &mut levels[t.idx()], lanes)?;
        }
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
    let spent = std::mem::take(&mut f.levels).into_vec();
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
    /// let f = Tdd::clause(&vtree, [Literal::pos(VarId(1)), Literal::neg(VarId(2))])?;
    /// assert_eq!(f.model_count()?, 6u32.into());
    /// # tididi::test_helpers::assert_canonical(&f);
    /// # Ok::<(), tididi::OperationError>(())
    /// ```
    pub fn clause(vtree: &Arc<Vtree>, literals: impl IntoIterator<Item = impl TryInto<Literal, Error: Into<OperationError>>>) -> Result<Tdd, OperationError> {
        vtree.context().run(|eng| eng.clause(vtree, literals))
    }
}

impl crate::Engine {
    /// Run [`Tdd::clause`](crate::Tdd::clause) using this batch's scratch and resource limits.
    ///
    /// # Errors
    ///
    /// Returns the linked operation's errors; cancellation, allocation refusal and
    /// the output-node cap return [`OperationError::Stopped`],
    /// [`OperationError::OverBudget`] and [`OperationError::OutputCap`], respectively.
    ///
    /// The output cap applies to the initial true diagram and rebuilt spine nodes.
    pub fn clause(
        &self,
        vtree: &Arc<Vtree>,
        literals: impl IntoIterator<Item = impl TryInto<Literal, Error: Into<OperationError>>>,
    ) -> Result<Tdd, OperationError> {
        let lim = self.limits();
        let _op = lim.begin_operation();
        lim.check_stop()?;
        let mut gate = lim.gate();
        let mut clause = Vec::new();
        for lit in literals {
            gate.poll(1)?;
            let lit: Literal = lit.try_into().map_err(Into::into)?;
            if vtree.leaf_of(lit.var).is_none() { return Err(OperationError::VariableNotInVtree(lit.var)); }
            lim.try_push(&mut clause, lit)?;
        }
        gate.flush()?;
        let one = self.cube(vtree, std::iter::empty::<Literal>())?;
        conjoin_clause_owned(self, one, &clause)
    }

    /// Run [`Tdd::and_clause`](crate::Tdd::and_clause) using this batch's scratch and resource limits.
    ///
    /// # Errors
    ///
    /// Returns the linked operation's errors; cancellation, allocation refusal and
    /// the output-node cap return [`OperationError::Stopped`],
    /// [`OperationError::OverBudget`] and [`OperationError::OutputCap`], respectively.
    ///
    /// Borrow a slice of signed integers or typed literals.
    /// Stop and output limits are checked once per rebuilt level; the output cap
    /// counts nodes in the levels rebuilt so far.
    pub fn and_clause<L: crate::LiteralInput>(&self, f: Tdd, clause: &[L]) -> Result<Tdd, OperationError> {
        L::conjoin(self, f, clause)
    }

    /// Run [`Tdd::or_cube`](crate::Tdd::or_cube) using this batch's scratch and resource limits.
    ///
    /// # Errors
    ///
    /// Returns the linked operation's errors; cancellation, allocation refusal and
    /// the output-node cap return [`OperationError::Stopped`],
    /// [`OperationError::OverBudget`] and [`OperationError::OutputCap`], respectively.
    ///
    /// Borrow a slice of signed integers or typed literals.
    /// Stop and output limits are checked once per rebuilt level; the output cap
    /// counts nodes in the levels rebuilt so far.
    pub fn or_cube<L: crate::LiteralInput>(&self, f: Tdd, cube: &[L]) -> Result<Tdd, OperationError> {
        L::disjoin(self, f, cube)
    }
}

#[cfg(test)]
mod tests;
