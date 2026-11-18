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
pub(crate) enum Normalized {
    /// A variable named in both polarities: as a clause true, as a cube false.
    Tautology,
    /// One literal per variable, with its leaf, in leaf order.
    Clause(Vec<(Literal, VtreeIdx)>),
}

/// Convert `lits`, resolve them against `vtree` and drop the repeats.
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
) -> Result<Normalized, OperationError> {
    let mut gate = lim.gate();
    // Every literal is resolved before a tautology is answered, so an absent
    // variable errors whatever else the clause says.
    let mut clause: Vec<(Literal, VtreeIdx)> = Vec::new();
    lim.reserve_exact(&mut clause, lits.len())?;
    for &lit in lits {
        gate.poll(1)?;
        let lit = lit.literal()?;
        let leaf = vtree.leaf_of(lit.var).ok_or(OperationError::VariableNotInVtree(lit.var))?;
        clause.push((lit, leaf));
    }
    gate.flush()?;
    // Sorted by leaf, a variable's literals are adjacent: one is kept, and
    // two of opposite sign make the clause true. An array over the vtree
    // cost a pass over the whole vtree per clause, which is most of the
    // clause's cost where it touches few levels of a large vtree.
    clause.sort_unstable_by_key(|&(_, leaf)| leaf.idx());
    let mut tautology = false;
    clause.dedup_by(|later, kept| {
        let same = later.1 == kept.1;
        tautology |= same && later.0.sign != kept.0.sign;
        same
    });
    Ok(if tautology {
        Normalized::Tautology
    } else {
        Normalized::Clause(clause)
    })
}

/// Conjoin the clause `lits` into `f`. `f` is consumed on every outcome; a
/// level array it no longer needs goes back to the engine's pool.
fn spine_walk<L: crate::LiteralInput>(
    eng: &Engine,
    f: Tdd,
    lits: &[L],
) -> Result<Tdd, OperationError> {
    let lim = eng.limits();
    let _op = lim.enter()?;
    let (mut f, clause) = extend_to(eng, f, lits)?;
    let clause = match clause {
        // A variable named in both polarities satisfies the clause whatever
        // its value: the accumulator is the answer as it stands, worklists
        // included.
        Normalized::Tautology => return Ok(f),
        Normalized::Clause(clause) => clause,
    };

    // The empty clause is false, so conjoining it gives ⊥ whatever `f` is.
    if clause.is_empty() {
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

    // `⊥ ∧ c = ⊥`: the accumulator is the answer as it stands, worklists
    // included.
    if f.is_zero() {
        return Ok(f);
    }
    // The rebuild takes `f`'s levels for the output; on an error they are
    // dropped with the partial result, so `f` owns nothing worth recycling.
    rebuild_along_spine(eng, &mut f, &clause)
}

/// `f` on the projection of its home on its own leaves and those of `lits`,
/// when a literal's variable has a leaf in the home and none in `f`'s
/// vtree, and `lits` [`normalize`]d against the vtree `f` ends on. Its levels
/// move to their images, no node or pair copied. Literals naming a variable
/// both ways leave `f` where it is: as a cube false and as a clause true, they
/// add no variable to the answer.
///
/// # Errors
///
/// [`Vtree::leaf_sets`]' refusal of a variable inside an extent, the
/// placement's, then [`normalize`]'s.
pub(crate) fn extend_to<L: crate::LiteralInput>(
    eng: &Engine,
    f: Tdd,
    lits: &[L],
) -> Result<(Tdd, Normalized), OperationError> {
    let lim = eng.limits();
    let f = match widened(&f.vtree, lits)? {
        None => f,
        Some(_) if names_both_ways(lim, &f.vtree, lits)? => return Ok((f, Normalized::Tautology)),
        Some(into) => place_on(eng, f, &into)?,
    };
    let clause = normalize(lim, &f.vtree, lits)?;
    Ok((f, clause))
}

/// The projection [`extend_to`] places a diagram on `vtree` on, if any. A
/// literal that does not convert is left to [`normalize`] to refuse.
fn widened<L: crate::LiteralInput>(vtree: &Arc<Vtree>, lits: &[L]) -> Result<Option<Arc<Vtree>>, OperationError> {
    let Some(home) = vtree.home() else { return Ok(None) };
    let vars = || lits.iter().filter_map(|lit| lit.literal().ok()).map(|lit| lit.var);
    if vtree.has_regions() {
        return widened_with_regions(vtree, vars());
    }
    if !vars().any(|var| vtree.home_has(var)) {
        return Ok(None);
    }
    let own = vtree.leaf_bottomup().map(|(_, var)| var);
    let added = vars().filter(|&var| home.rank(var).is_some());
    Ok(Some(Vtree::projection(home, own.chain(added)).expect("every variable has a leaf in the home")))
}

/// [`widened`] for a vtree with regions, `vars` the literals' variables:
/// refused with [`OperationError::MarginalLevel`] where one is inside an
/// extent.
#[cold]
fn widened_with_regions(
    vtree: &Arc<Vtree>,
    vars: impl Iterator<Item = crate::vtree::VarId>,
) -> Result<Option<Arc<Vtree>>, OperationError> {
    let home = vtree.home().expect("a vtree with a region is a projection");
    let added = vars.filter(|&var| home.rank(var).is_some());
    let lits = Vtree::projection(home, added).expect("every variable has a leaf in the home");
    Ok(match Vtree::leaf_sets(vtree, &lits)? {
        crate::vtree::LeafSets::Union(into) => Some(into),
        _ => None,
    })
}

/// Whether `lits` name a variable both ways, with [`normalize`]'s errors in
/// its order, a variable resolved against `vtree` or its home.
#[cold]
fn names_both_ways<L: crate::LiteralInput>(lim: &Limits, vtree: &Vtree, lits: &[L]) -> Result<bool, OperationError> {
    let home = vtree.home().expect("only a projection is widened");
    let mut seen: Vec<Literal> = Vec::new();
    lim.reserve_exact(&mut seen, lits.len())?;
    let mut gate = lim.gate();
    for &lit in lits {
        gate.poll(1)?;
        let lit = lit.literal()?;
        if vtree.leaf_of(lit.var).is_none() && home.rank(lit.var).is_none() {
            return Err(OperationError::VariableNotInVtree(lit.var));
        }
        seen.push(lit);
    }
    gate.flush()?;
    seen.sort_unstable_by_key(|lit| lit.var.0);
    Ok(seen.windows(2).any(|w| w[0].var == w[1].var && w[0].sign != w[1].sign))
}

/// `f` placed on `into`, a projection of its home holding its leaves.
fn place_on(eng: &Engine, f: Tdd, into: &Arc<Vtree>) -> Result<Tdd, OperationError> {
    if f.vtree.num_nodes() == 0 {
        let mut constant = crate::build::constant_padded_on(eng, into, f.is_one())?;
        constant.weights = f.weights;
        return Ok(constant);
    }
    match crate::restructure::embed::place_moving(eng, f, into, |var| var, crate::restructure::embed::Free::Build) {
        Ok((extended, _)) => Ok(extended),
        Err(refused) => Err(match refused.error {
            crate::restructure::EmbedError::Operation(error) => error,
            _ => OperationError::VtreeMismatch,
        }),
    }
}

/// Rebuild every level on the clause's spine, moving `f`'s levels into the
/// result. The clause is normalized and non-empty; `f` has structure at every
/// leaf of it and is not false.
fn rebuild_along_spine(
    eng: &Engine,
    f: &mut Tdd,
    clause: &[(Literal, VtreeIdx)],
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
    let mut tables = ClauseTables {
        cd_map: &mut work.cd_map,
        level_base: &work.level_base,
        need_dt: &work.need_dt,
        on_spine: &work.on_spine,
        t3_buf: &mut work.t3_buf,
        dt_pairs: &mut work.dt_pairs,
        spare_nodes: &mut Vec::new(),
        spare_pairs: &mut Vec::new(),
    };
    for &t in work.spine_internal.iter() {
        rebuild_spine_level(eng, t, vtree, &mut levels, &mut tables)?;
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
            "and_clause",
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
    let mut out = spine_walk(eng, f, clause)?;
    crate::reduce::close_constant(eng, &mut out);
    Ok(out)
}

impl Tdd {
    /// Build a canonical disjunction of literals over the shared vtree.
    ///
    /// Integers are signed and one-based; typed [`Literal`] values also work.
    /// Repeated literals are ignored. Both polarities of a variable make the
    /// clause true; an empty clause is false. Unmentioned variables remain free.
    /// On a home or a projection of one, the result is on the projection of
    /// the clause's variables, and an empty clause on the home's empty
    /// projection. Uses the execution context attached to the vtree.
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
}

#[cfg(test)]
mod tests;
