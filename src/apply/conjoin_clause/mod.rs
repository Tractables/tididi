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
//! side by side in `cd_map`. The whole module is written in terms of this pair.
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

use crate::limits::{Limits, OperationError};
use crate::apply::conjoin::budget::{finish_node, reserve_pairs_for_emit, NO_PRODUCT};

mod spine;
use spine::*;
mod pairs;
use pairs::*;
mod rebuild;
use rebuild::*;
mod cube;
use cube::{CubeChain, disjoin_cube_by_complement};
pub(crate) use cube::disjoin_cube_on;

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

/// A clause's literals resolved to their vtree leaves, one per variable. The
/// walk indexes one `cd_map` column per literal, which is what needs this.
enum Normalized {
    /// A variable named in both polarities: the clause is true.
    Tautology,
    /// One literal per variable, in first-occurrence order, with its leaf.
    Clause(Vec<(Literal, VtreeIdx)>),
}

/// Resolve `lits` against `vtree` and drop the repeats. With `negate` each
/// literal is complemented first, which turns a cube into the clause the walk
/// carries: a cube naming a variable in both polarities is false, and its
/// negation a tautology.
///
/// # Errors
///
/// [`OperationError::VariableNotInVtree`] for a variable the vtree lacks,
/// whatever else the clause says, and the limits' errors.
fn normalize(lim: &Limits, vtree: &Vtree, lits: &[Literal], negate: bool) -> Result<Normalized, OperationError> {
    let mut gate = lim.gate();
    // The polarity each leaf was first seen with, indexed by vtree node.
    let mut seen: Vec<Option<bool>> = Vec::new();
    lim.try_resize(&mut seen, vtree.num_nodes(), None)?;
    let mut clause = Vec::new();
    // Every literal is resolved before a tautology is answered, so an absent
    // variable errors whatever else the clause says.
    let mut tautology = false;
    for lit in lits {
        gate.poll(1)?;
        let leaf = vtree.leaf_of(lit.var).ok_or(OperationError::VariableNotInVtree(lit.var))?;
        match seen[leaf.idx()].replace(lit.sign) {
            Some(sign) => tautology |= sign != lit.sign,
            None => lim.try_push(&mut clause, (if negate { lit.negated() } else { *lit }, leaf))?,
        }
    }
    gate.flush()?;
    Ok(if tautology { Normalized::Tautology } else { Normalized::Clause(clause) })
}

/// The shared bottom-up walk: conjoin the clause `lits` into `f`, or — with
/// `disjoin` — disjoin the cube `lits`, by carrying the [`CubeChain`]
/// alongside. `f` is consumed on every outcome; a level array it no longer
/// needs goes back to the engine's pool.
fn spine_walk(eng: &Engine, mut f: Tdd, lits: &[Literal], disjoin: bool) -> Result<Tdd, OperationError> {
    let lim = eng.limits();
    let _op = lim.enter()?;
    let vtree = Arc::clone(&f.vtree);
    let clause = match normalize(lim, &vtree, lits, disjoin)? {
        // A variable named in both polarities satisfies the clause whatever
        // its value, and the cube it negates is false: the identity in both
        // modes, so the accumulator is the answer as it stands, worklists
        // included.
        Normalized::Tautology => return Ok(f),
        Normalized::Clause(clause) => clause,
    };

    // The empty clause is false, so conjoining it gives ⊥ whatever `f` is.
    if !disjoin && clause.is_empty() {
        let out = crate::build::constant_like(eng, &f, false)?;
        diagram::return_levels(eng, diagram::PoolSlot::First, std::mem::take(&mut f.levels).into_vec());
        return Ok(out);
    }

    let mut gate = lim.gate();
    for &(_, leaf) in &clause {
        gate.poll(1)?;
        f.require_structure_at(leaf)?;
    }
    gate.flush()?;

    if !disjoin {
        return conjoin_normalized(eng, f, &clause);
    }

    // The empty cube is true, and `⊥ ∨ M` is the cube itself: both are the
    // cube built on its own, with the accumulator's marginal values.
    if clause.is_empty() || f.is_zero() {
        let mut out = eng.cube(&vtree, clause.iter().map(|&(lit, _)| lit.negated()))?;
        out.weights = f.weights.as_ref().map(WeightStore::empty_like);
        return Ok(out);
    }

    // The chain needs a node at every level, which the two lanes supply only
    // where the cube constrains the subtree; see the `cube` module. A
    // one-variable vtree has no internal level to hang the chain on, and an
    // output below the root leaves levels the walk would not reach.
    let root = vtree.root();
    if clause.len() != vtree.num_leaves() as usize || f.output.vtree != root || vtree.node(root).is_leaf() {
        return disjoin_cube_by_complement(eng, f, &clause);
    }

    rebuild_along_spine(eng, &mut f, &clause, true)
}

/// Conjoin a normalized, non-empty clause into `f`, which has structure at
/// every leaf of it.
fn conjoin_normalized(eng: &Engine, mut f: Tdd, clause: &[(Literal, VtreeIdx)]) -> Result<Tdd, OperationError> {
    // `⊥ ∧ c = ⊥`: the accumulator is the answer as it stands, worklists
    // included.
    if f.is_zero() {
        return Ok(f);
    }

    // The rebuild takes `f`'s levels for the output; on an error they are
    // dropped with the partial result, so `f` owns nothing worth recycling.
    rebuild_along_spine(eng, &mut f, clause, false)
}

/// Rebuild every level on the clause's spine, moving `f`'s levels into the
/// result. The clause is normalized and non-empty; `f` has structure at every
/// leaf of it and is not false.
fn rebuild_along_spine(eng: &Engine, f: &mut Tdd, clause: &[(Literal, VtreeIdx)], disjoin: bool) -> Result<Tdd, OperationError> {
    let lim = eng.limits();
    let pool = eng.clause_pool();
    let vtree = &f.vtree;
    let num_nodes = vtree.num_nodes();

    // The clause spine — the Steiner tree of its variables' leaves — and the
    // `need_dt` flag propagated top-down over it.
    let mut on_spine = pool.on_spine.checkout(lim);
    on_spine.cover(lim, num_nodes)?;
    let mut spine_internal = pool.spine_internal.checkout_preserving(lim);
    let mut dfs_stack = pool.dfs_stack.checkout_preserving(lim);
    build_clause_spine(lim, vtree, clause, &mut on_spine, &mut spine_internal, &mut dfs_stack)?;
    let mut need_dt = pool.need_dt.checkout(lim);
    need_dt.cover(lim, num_nodes)?;
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
    let total = plan_cd_map_bases(clause, &spine_internal, &levels, &mut level_base)?;

    // The base blocks partition `[0, total)` with no gaps and every `c_t`
    // entry is written once below, so no bulk `NO_PRODUCT` fill is needed. A
    // `d_t` lane is written iff `need_dt[t]`, and a read of one implies
    // `need_dt` on that child, so a stale lane is never read.
    let mut cd_map = pool.cd_map.checkout_preserving(lim);
    lim.try_resize(&mut cd_map, total, [NO_PRODUCT, NO_PRODUCT])?;

    fill_leaf_maps(clause, &level_base, &need_dt, &mut cd_map);

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
    // an ancestor of every clause leaf, so it is on the spine and its `cd_map`
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
    diagram::Assembly::from_levels(eng, vtree, levels, f_weights)
        .finish_with(TddNodeId { vtree: out_vtree, local: out_local }, carried, &spine_internal)
}

/// Conjoin `clause` into `f` under `eng`'s limits. The implementation behind
/// [`Engine::and_clause`](crate::Engine::and_clause).
///
/// # Errors
///
/// The errors of [`Engine::and_clause`](crate::Engine::and_clause).
pub(crate) fn conjoin_clause_on(eng: &Engine, f: Tdd, clause: &[Literal]) -> Result<Tdd, OperationError> {
    spine_walk(eng, f, clause, false)
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
        let _op = lim.enter()?;
        let mut gate = lim.gate();
        let mut clause = Vec::new();
        for lit in literals {
            gate.poll(1)?;
            let lit: Literal = lit.try_into().map_err(Into::into)?;
            lim.try_push(&mut clause, lit)?;
        }
        gate.flush()?;
        let one = self.cube(vtree, std::iter::empty::<Literal>())?;
        conjoin_clause_on(self, one, &clause)
    }

    /// Run [`Tdd::and_clause`](crate::Tdd::and_clause) using this batch's scratch and resource limits.
    /// Borrow a slice of signed integers or typed literals. Stop and output
    /// limits are checked once per rebuilt level; the output cap counts nodes
    /// in the levels rebuilt so far.
    ///
    /// # Errors
    ///
    /// Returns the linked operation's errors; cancellation, allocation refusal and
    /// the output-node cap return [`OperationError::Stopped`],
    /// [`OperationError::OverBudget`] and [`OperationError::OutputCap`], respectively.
    ///
    pub fn and_clause<L: crate::LiteralInput>(&self, f: Tdd, clause: &[L]) -> Result<Tdd, OperationError> {
        L::conjoin(self, f, clause)
    }

    /// Run [`Tdd::or_cube`](crate::Tdd::or_cube) using this batch's scratch and resource limits.
    /// Borrow a slice of signed integers or typed literals. Stop and output
    /// limits are checked once per rebuilt level; the output cap counts nodes
    /// in the levels rebuilt so far.
    ///
    /// # Errors
    ///
    /// Returns the linked operation's errors; cancellation, allocation refusal and
    /// the output-node cap return [`OperationError::Stopped`],
    /// [`OperationError::OverBudget`] and [`OperationError::OutputCap`], respectively.
    ///
    pub fn or_cube<L: crate::LiteralInput>(&self, f: Tdd, cube: &[L]) -> Result<Tdd, OperationError> {
        L::disjoin(self, f, cube)
    }
}

#[cfg(test)]
mod tests;
