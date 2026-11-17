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
use crate::execution::pool::{Buffers, PooledScratch, Scratch};
use std::sync::Arc;

use super::CONJOIN_GRID;
use crate::diagram::Literal;
use crate::diagram::{self, *};
use crate::vtree::{Vtree, VtreeIdx};

use crate::apply::conjoin::budget::{NO_PRODUCT, finish_node, reserve_pairs_for_emit};
use crate::limits::{Limits, OperationError};

mod spine;
use spine::*;
mod pairs;
use pairs::*;
mod rebuild;
use rebuild::*;
mod cube;
pub(crate) use cube::disjoin_cube_on;
use cube::{CubeChain, disjoin_cube_by_complement};

/// The working set of one clause walk, pooled by the engine between calls.
#[derive(Default)]
pub(crate) struct ClauseWork {
    /// Maps accumulator node index → `[ct, dt]` output indices for conjunction
    /// with the clause's `c_t` / `d_t` virtual nodes, interleaved so one
    /// per-pair lookup serves both lanes; the `dt` lane is written iff
    /// `need_dt` for the level.
    cd_map: Vec<[u32; 2]>,
    /// Cumulative offsets into `cd_map`, one per vtree level.
    level_base: Vec<usize>,
    /// Per-level flags: `on_spine[t]` = clause has variables in subtree t.
    on_spine: MarkBuffer,
    /// Per-level flags: `need_dt[t]` = must compute complement conjunction at t.
    need_dt: MarkBuffer,
    /// Spine internal nodes, children first.
    spine_internal: Vec<VtreeIdx>,
    /// The `d_t` pairs of the node being rebuilt.
    dt_pairs: Vec<ChildPair>,
    /// The both-relevant type-3 pairs of the node being rebuilt, held back
    /// until its left index changes so the pair list stays sorted.
    t3_buf: Vec<ChildPair>,
}

impl Buffers for ClauseWork {
    fn buffers(&mut self, visit: &mut dyn FnMut(&mut dyn Scratch)) {
        visit(&mut self.cd_map);
        visit(&mut self.level_base);
        self.on_spine.buffers(visit);
        self.need_dt.buffers(visit);
        visit(&mut self.spine_internal);
        visit(&mut self.dt_pairs);
        visit(&mut self.t3_buf);
    }
}

impl PooledScratch for ClauseWork {
    /// `cd_map` and `level_base` keep their entries: the walk writes every
    /// entry it reads first.
    fn prepare(&mut self) {
        self.on_spine.debug_assert_clear();
        self.need_dt.debug_assert_clear();
        self.spine_internal.clear();
        self.dt_pairs.clear();
        self.t3_buf.clear();
    }

    /// The marks are cleared on every exit, an error's too, so the parked
    /// flags are all false.
    fn retain(&mut self, lim: &crate::limits::Limits) -> usize {
        self.on_spine.unmark_all();
        self.need_dt.unmark_all();
        self.release_oversized(lim)
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

/// Convert `lits`, resolve them against `vtree` and drop the repeats. With
/// `negate` each literal is complemented first, which turns a cube into the
/// clause the walk carries: a cube naming a variable in both polarities is
/// false, and its negation a tautology.
///
/// # Errors
///
/// Per literal in order, [`OperationError::InvalidLiteral`] for integer zero,
/// then [`OperationError::VariableNotInVtree`] for a variable the vtree
/// lacks, whatever else the clause says; and the limits' errors.
fn normalize<L: crate::LiteralInput>(
    lim: &Limits,
    vtree: &Vtree,
    lits: &[L],
    negate: bool,
) -> Result<Normalized, OperationError> {
    let mut gate = lim.gate();
    // Every literal is resolved before a tautology is answered, so an absent
    // variable errors whatever else the clause says.
    let mut resolved: Vec<(Literal, VtreeIdx)> = Vec::new();
    lim.reserve_exact(&mut resolved, lits.len())?;
    for &lit in lits {
        gate.poll(1)?;
        let lit = lit.literal()?;
        let leaf = vtree.leaf_of(lit.var).ok_or(OperationError::VariableNotInVtree(lit.var))?;
        resolved.push((lit, leaf));
    }
    gate.flush()?;
    // A literal is kept where its leaf first occurs; a later one of the other
    // sign makes the clause true. The first occurrence is found by sorting
    // the literals' positions by leaf. An array over the vtree, which it used
    // before, cost a pass over the whole vtree per clause, which is most of
    // the clause's cost where it touches few levels of a large vtree.
    let mut first: Vec<u32> = Vec::new();
    lim.try_resize(&mut first, lits.len(), 0)?;
    let mut order: Vec<(usize, u32)> = Vec::new();
    lim.reserve_exact(&mut order, lits.len())?;
    order.extend(
        resolved
            .iter()
            .enumerate()
            .map(|(i, &(_, leaf))| (leaf.idx(), i as u32)),
    );
    order.sort_unstable();
    let mut run_start = 0;
    for j in 0..order.len() {
        if order[j].0 != order[run_start].0 {
            run_start = j;
        }
        first[order[j].1 as usize] = order[run_start].1;
    }
    let mut clause: Vec<(Literal, VtreeIdx)> = Vec::new();
    let mut tautology = false;
    for (i, &(lit, leaf)) in resolved.iter().enumerate() {
        let f = first[i] as usize;
        if f == i {
            lim.try_push(&mut clause, (if negate { lit.negated() } else { lit }, leaf))?;
        } else {
            tautology |= resolved[f].0.sign != lit.sign;
        }
    }
    Ok(if tautology {
        Normalized::Tautology
    } else {
        Normalized::Clause(clause)
    })
}

/// The shared bottom-up walk: conjoin the clause `lits` into `f`, or — with
/// `disjoin` — disjoin the cube `lits`, by carrying the [`CubeChain`]
/// alongside. `f` is consumed on every outcome; a level array it no longer
/// needs goes back to the engine's pool.
fn spine_walk<L: crate::LiteralInput>(
    eng: &Engine,
    mut f: Tdd,
    lits: &[L],
    disjoin: bool,
) -> Result<Tdd, OperationError> {
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
        diagram::return_levels(eng, std::mem::take(&mut f.levels).into_vec());
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
    if clause.len() != vtree.num_leaves() as usize
        || f.output.vtree != root
        || vtree.node(root).is_leaf()
    {
        return disjoin_cube_by_complement(eng, f, &clause);
    }

    rebuild_along_spine(eng, &mut f, &clause, true)
}

/// Conjoin a normalized, non-empty clause into `f`, which has structure at
/// every leaf of it.
fn conjoin_normalized(
    eng: &Engine,
    mut f: Tdd,
    clause: &[(Literal, VtreeIdx)],
) -> Result<Tdd, OperationError> {
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
fn rebuild_along_spine(
    eng: &Engine,
    f: &mut Tdd,
    clause: &[(Literal, VtreeIdx)],
    disjoin: bool,
) -> Result<Tdd, OperationError> {
    let lim = eng.limits();
    let vtree = &f.vtree;
    let num_nodes = vtree.num_nodes();
    // Only a test fixture puts the output below the root; the walk would read
    // the output's `cd_map` block unplanned where the clause leaves its subtree.
    debug_assert_eq!(f.output.vtree, vtree.root(), "the clause walk needs the output at the root level");
    let mut work = eng.scratch.clause.checkout(eng);
    let work = &mut *work;

    // The clause spine — the Steiner tree of its variables' leaves — and the
    // `need_dt` flag propagated top-down over it.
    work.on_spine.cover(lim, num_nodes)?;
    build_clause_spine(lim, vtree, clause, &mut work.on_spine, &mut work.spine_internal)?;
    work.need_dt.cover(lim, num_nodes)?;
    propagate_need_dt(vtree, &work.spine_internal, &work.on_spine, &mut work.need_dt);

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
    lim.try_resize(&mut work.level_base, num_nodes, 0usize)?;
    let total = plan_cd_map_bases(clause, &work.spine_internal, &levels, &mut work.level_base)?;

    // The base blocks partition `[0, total)` with no gaps and every `c_t`
    // entry is written once below, so no bulk `NO_PRODUCT` fill is needed. A
    // `d_t` lane is written iff `need_dt[t]`, and a read of one implies
    // `need_dt` on that child, so a stale lane is never read.
    lim.try_resize(&mut work.cd_map, total, [NO_PRODUCT, NO_PRODUCT])?;

    fill_leaf_maps(clause, &work.level_base, &work.need_dt, &mut work.cd_map);

    // Rebuild each spine internal level bottom-up. Children's maps are fully
    // written before any parent reads them. The stop axis and the output cap
    // are checked after each level, the cap against the rebuilt levels' nodes.
    let mut out_nodes = 0u64;
    let mut chain = if disjoin {
        Some(CubeChain::new(lim, vtree, clause)?)
    } else {
        None
    };
    let mut tables = ClauseTables {
        cd_map: &mut work.cd_map,
        level_base: &work.level_base,
        need_dt: &work.need_dt,
        on_spine: &work.on_spine,
        t3_buf: &mut work.t3_buf,
        dt_pairs: &mut work.dt_pairs,
        output_cube_pair: None,
    };
    for &t in work.spine_internal.iter() {
        // The `cd_map` block of a level is sized at its width before the
        // rebuild, which is also the range the chain reads back.
        let old_width = levels[t.idx()].slot_count();
        if let Some(chain) = chain.as_ref() {
            // The output level is last in the bottom-up order, so the chain
            // is complete below it and the cube joins the output node here.
            tables.output_cube_pair = (t == out_vtree).then(|| {
                (
                    tables.level_base[t.idx()] + out_local_in.idx(),
                    chain.pair_at(vtree, t),
                )
            });
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

    // Output: conjunction of f's output with c_t at the root, which is on
    // every spine, so its `cd_map` block is filled.
    let out_base = work.level_base[out_vtree.idx()];
    let ct_out = work.cd_map[out_base + out_local_in.idx()][0];
    let out_local = if ct_out != NO_PRODUCT {
        NodeIdx(ct_out)
    } else {
        ZERO
    };

    // Contract seed: this clause's spine, which is ancestor-closed, so an
    // off-spine level and its whole subtree are the accumulator's own bytes;
    // whatever the accumulator still owed is carried over, which keeps this
    // exact for a caller that does not minimize between clauses. The spine
    // is also all the walk changed, and so all the seat closes.
    // The pair fusion records (`Dirty::is_fused`) off the spine are the
    // accumulator's too.
    let vtree = Arc::clone(vtree);
    let mut carried = f.take_worklists();
    let fused = carried.take_fused();
    let mut out = diagram::Assembly::from_levels(eng, vtree, levels, f_weights).finish_with(
        TddNodeId {
            vtree: out_vtree,
            local: out_local,
        },
        carried,
        &work.spine_internal,
        Some(&work.spine_internal[..]),
    )?;
    out.dirty.restore_fused(fused, &work.spine_internal);
    #[cfg(debug_assertions)]
    {
        let recorded: Vec<VtreeIdx> = (0..out.vtree.num_nodes() as u32)
            .map(VtreeIdx)
            .filter(|t| out.dirty.is_fused(t.0))
            .collect();
        crate::test_helpers::check::marginal::debug_assert_pair_fusion_saturated(
            &out,
            Some(&recorded),
            if disjoin { "or_cube" } else { "and_clause" },
        );
    }
    Ok(out)
}

/// Conjoin `clause` into `f` under `eng`'s limits. The implementation behind
/// [`Engine::and_clause`](crate::Engine::and_clause).
///
/// # Errors
///
/// The errors of [`Engine::and_clause`](crate::Engine::and_clause).
pub(crate) fn conjoin_clause_on<L: crate::LiteralInput>(
    eng: &Engine,
    f: Tdd,
    clause: &[L],
) -> Result<Tdd, OperationError> {
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
    pub fn clause(
        vtree: &Arc<Vtree>,
        literals: impl IntoIterator<Item = impl TryInto<Literal, Error: Into<OperationError>>>,
    ) -> Result<Tdd, OperationError> {
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
    /// Borrow a slice of signed integers or typed literals. The output cap is
    /// checked after each rebuilt level, against the nodes of the levels
    /// rebuilt so far; a stop is also seen within a level as its work is
    /// polled.
    ///
    /// # Errors
    ///
    /// Returns the linked operation's errors; cancellation, allocation refusal and
    /// the output-node cap return [`OperationError::Stopped`],
    /// [`OperationError::OverBudget`] and [`OperationError::OutputCap`], respectively.
    ///
    pub fn and_clause<L: crate::LiteralInput>(
        &self,
        f: Tdd,
        clause: &[L],
    ) -> Result<Tdd, OperationError> {
        conjoin_clause_on(self, f, clause)
    }

    /// Run [`Tdd::or_cube`](crate::Tdd::or_cube) using this batch's scratch and resource limits.
    /// Borrow a slice of signed integers or typed literals. A cube naming
    /// every variable is disjoined by the walk of
    /// [`and_clause`](Self::and_clause) and checks the limits as it does; any
    /// other cube by a negation, a clause conjunction and a negation, each
    /// checking its own.
    ///
    /// # Errors
    ///
    /// Returns the linked operation's errors; cancellation, allocation refusal and
    /// the output-node cap return [`OperationError::Stopped`],
    /// [`OperationError::OverBudget`] and [`OperationError::OutputCap`], respectively.
    ///
    pub fn or_cube<L: crate::LiteralInput>(
        &self,
        f: Tdd,
        cube: &[L],
    ) -> Result<Tdd, OperationError> {
        disjoin_cube_on(self, f, cube)
    }
}

#[cfg(test)]
mod tests;
