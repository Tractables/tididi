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
mod widen;

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
    /// A widening's added variables with their home ranks, for a clause
    /// longer than the stack holds.
    widen_added: Vec<(u32, crate::vtree::VarId)>,
    /// A widening's image of each node of the accumulator's vtree.
    widen_map: Vec<VtreeIdx>,
    /// A widening's nodes that are no image, children first.
    widen_fresh: Vec<(VtreeIdx, crate::vtree::Reading)>,
    /// Kept across walks, as the pool keeps the rest.
    drop_yield: DropYield,
    /// The spine levels whose rebuild dropped a pair naming a node of their
    /// off-spine child ([`ClauseTables::dropped_at`]).
    dropped_at: Vec<VtreeIdx>,
}

/// The nodes the drop of unnamed nodes has removed against the parent pairs
/// it read, which decides whether a walk runs it.
#[derive(Default)]
struct DropYield {
    /// Nodes dropped, plus one for each walk that skipped the drop.
    dropped: u64,
    read: u64,
}

impl DropYield {
    /// Past this many pairs read per node dropped, the drop costs more than
    /// the later work on the nodes it removes, which the next reduction
    /// prunes anyway. What a dropped node saves later varies by a hundred
    /// times between inputs that read 50 pairs per node, so the bound only
    /// cuts the walks that read hundreds of pairs per node.
    const MAX_PAIRS_PER_NODE: u64 = 128;

    /// Whether this walk runs the drop. A walk that does not adds one node
    /// to `dropped`, so a run of skipped walks tries again.
    fn drops(&mut self) -> bool {
        let drops = self.dropped * Self::MAX_PAIRS_PER_NODE >= self.read;
        self.dropped += u64::from(!drops);
        drops
    }
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
        visit(&mut self.widen_added);
        visit(&mut self.widen_map);
        visit(&mut self.widen_fresh);
        visit(&mut self.dropped_at);
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
        self.widen_added.clear();
        self.widen_fresh.clear();
        self.dropped_at.clear();
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
    let f = match widening(&f.vtree, lits)? {
        Widening::None => f,
        _ if names_both_ways(lim, &f.vtree, lits)? => return Ok((f, Normalized::Tautology)),
        Widening::Onto(into) => place_on(eng, f, &into)?,
        Widening::Added => widen::widen(eng, f, lits)?,
    };
    let clause = normalize(lim, &f.vtree, lits)?;
    Ok((f, clause))
}

/// How [`extend_to`] widens a diagram's vtree.
enum Widening {
    None,
    /// Onto this union, from a vtree with regions.
    Onto(Arc<Vtree>),
    /// By the literals' variables with a leaf in the home and none here.
    Added,
}

/// How [`extend_to`] widens `vtree`. A literal that does not convert is
/// left to [`normalize`] to refuse.
fn widening<L: crate::LiteralInput>(vtree: &Arc<Vtree>, lits: &[L]) -> Result<Widening, OperationError> {
    if vtree.link_home().is_none() {
        return Ok(Widening::None);
    }
    let vars = || lits.iter().filter_map(|lit| lit.literal().ok()).map(|lit| lit.var);
    if vtree.has_regions() {
        return widened_with_regions(vtree, vars());
    }
    Ok(match vars().any(|var| vtree.home_has(var)) {
        true => Widening::Added,
        false => Widening::None,
    })
}

/// [`widening`] for a vtree with regions, `vars` the literals' variables:
/// refused with [`OperationError::MarginalLevel`] where one is inside an
/// extent.
#[cold]
fn widened_with_regions(
    vtree: &Arc<Vtree>,
    vars: impl Iterator<Item = crate::vtree::VarId>,
) -> Result<Widening, OperationError> {
    let home = vtree.link_home().expect("a vtree with a region is a projection");
    let added = vars.filter(|&var| home.rank(var).is_some());
    let lits = Vtree::projection(home, added).expect("every variable has a leaf in the home");
    Ok(match Vtree::leaf_sets(vtree, &lits)? {
        crate::vtree::LeafSets::Union(into) => Widening::Onto(into),
        _ => Widening::None,
    })
}

/// Whether `lits` name a variable both ways, with [`normalize`]'s errors in
/// its order, a variable resolved against `vtree` or its home.
#[cold]
fn names_both_ways<L: crate::LiteralInput>(lim: &Limits, vtree: &Vtree, lits: &[L]) -> Result<bool, OperationError> {
    let home = vtree.link_home().expect("only a projection is widened");
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
        return crate::build::constant_padded_on(eng, into, f.is_one());
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
    let canonical = f.levels.is_canonical(f.output);
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
    let top_fork = build_clause_spine(lim, vtree, clause, &mut work.on_spine, &mut work.spine_internal)?;
    debug_assert_eq!(
        top_fork,
        work.spine_internal.iter().rev().copied().find(|&t| {
            let (left, right) = vtree.children(t);
            work.on_spine[left.idx()] && work.on_spine[right.idx()]
        }),
        "the spine build's topmost fork"
    );
    work.need_dt.cover(lim, num_nodes)?;
    propagate_need_dt(vtree, &work.spine_internal, &work.on_spine, &mut work.need_dt);

    // Take ownership of f's levels: off-spine levels pass through as the
    // identity, spine internal levels are rebuilt in place below, and f is
    // left with empty levels.
    let out_vtree = f.output.vtree;
    let out_local_in = f.output.local;
    // The walk changes no level's kind. Only a known summary is carried: a
    // caller that never reads it would pay a scan of every level per clause.
    let first_marginal = f.levels.known_first_marginal();
    let mut levels = std::mem::take(&mut f.levels).into_vec();
    // The accumulator's marginal values move to the output along with its levels:
    // a clause carries none of its own, and the output is nothing but the
    // accumulator, one clause further on.
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

    let vtree = Arc::clone(vtree);
    let mut carried = f.take_worklists();
    let fused = carried.take_fused();
    let carried_loose = carried.take_loose();
    let mut loose = match canonical {
        true => {
            let mut none = carried_loose.unwrap_or_default();
            none.clear();
            Some(none)
        }
        // Once half the internal levels are listed the list spares a reader
        // little, and a run of clauses would list every spine again.
        false => carried_loose.filter(|loose| loose.len() < vtree.internal_bottomup_slice().len() / 2),
    };
    let track = loose.is_some();
    if track {
        lim.reserve(&mut work.dropped_at, work.spine_internal.len())?;
    }
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
        top: None,
        dropped_at: &mut work.dropped_at,
    };
    for &t in work.spine_internal.iter() {
        rebuild_spine_level(eng, t, &vtree, &mut levels, &mut tables, track)?;
        out_nodes += levels[t.idx()].slot_count() as u64;
        lim.level_done(out_nodes)?;
    }
    // A right `d` half skipped beside a left node with no `c` half is mostly
    // named by another pair unless the skips are a good share of the pairs:
    // below one in four, few such reads dropped a node.
    let top = tables
        .top
        .map(|(t, [left, right], pairs)| (t, [left != 0, right * 4 >= pairs && right != 0]))
        .filter(|&(_, sides)| sides != [false; 2]);
    if let Some(loose) = loose.as_mut() {
        list_rebuilt(loose, &vtree, &work.spine_internal, &work.on_spine, &work.dropped_at, top_fork);
    }

    // Output: conjunction of f's output with c_t at the root, which is on
    // every spine, so its `cd_map` block is filled.
    let out_base = work.level_base[out_vtree.idx()];
    let ct_out = work.cd_map[out_base + out_local_in.idx()][0];
    let out_local = if ct_out == NO_PRODUCT {
        ZERO
    } else if vtree.node(out_vtree).is_leaf()
        || (levels[out_vtree.idx()].slot_count() == 1 && top.is_none())
        || !work.drop_yield.drops()
    {
        NodeIdx(ct_out)
    } else {
        let (out, [dropped, read]) = crate::reduce::prune::drop_clause_orphans(
            eng,
            &vtree,
            &mut levels,
            &work.on_spine,
            top,
            (out_vtree, ct_out),
            loose.as_deref_mut(),
        )?;
        work.drop_yield.dropped += dropped;
        work.drop_yield.read += read;
        NodeIdx(out)
    };

    // Contract seed: this clause's spine, which is ancestor-closed, so an
    // off-spine level and its whole subtree are the accumulator's own bytes;
    // whatever the accumulator still owed is carried over, which keeps this
    // exact for a caller that does not minimize between clauses. The spine
    // is also all the walk changed, and so all the seat closes.
    // The pair fusion records (`Dirty::is_fused`) off the spine are the
    // accumulator's too.
    let mut out = diagram::Assembly::from_levels(eng, vtree, levels).finish_with(
        TddNodeId {
            vtree: out_vtree,
            local: out_local,
        },
        carried,
        &work.spine_internal,
        Some(&work.spine_internal[..]),
    )?;
    out.dirty.restore_fused(fused, &work.spine_internal);
    if let Some(first) = first_marginal {
        out.levels.keep_first_marginal(first);
    }
    out.dirty.restore_loose(loose);
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

/// List the levels a clause's rebuild may leave holding a node no pair of
/// their parent level names: the off-spine children of the levels in
/// `dropped_at`, the root, which may hold products the output is not, and
/// the internal children of the topmost level with both children on the
/// spine ([`build_clause_spine`]). A rebuilt child's nodes are its nodes' products, each named where
/// the node it came from was, since a node other than false has a `c_t` or a
/// `d_t` product and a pair keeps the product of either that the parent
/// computes; but under a level with both children relevant and no `d_t`, the
/// type-3 and `(c_L, d_R)` pairs name a `d_t` product only where the other
/// side's `c_t` is not false, and only the topmost such level computes no
/// `d_t` (`propagate_need_dt`). Every other rebuilt level keeps its entry,
/// its count forgotten; an off-spine level keeps its nodes and, unless a pair
/// naming one was dropped, the nodes named, so its entry and count hold.
/// The drop of unnamed nodes after it
/// ([`drop_clause_orphans`](crate::reduce::prune::drop_clause_orphans))
/// then unlists the levels it clears and lists the off-spine levels it may
/// leave loose.
fn list_rebuilt(
    loose: &mut crate::diagram::LooseLevels,
    vtree: &crate::vtree::Vtree,
    spine_internal: &[VtreeIdx],
    on_spine: &[bool],
    dropped_at: &[VtreeIdx],
    top_fork: Option<VtreeIdx>,
) {
    for &t in dropped_at {
        let (left, right) = vtree.children(t);
        let off = if on_spine[left.idx()] { right } else { left };
        if !vtree.node(off).is_leaf() {
            loose.insert(off.0);
        }
    }
    if loose.has_counts() {
        for &t in spine_internal {
            loose.forget_count(t.0);
        }
    }
    loose.insert(vtree.root().0);
    if let Some(p) = top_fork {
        let (left, right) = vtree.children(p);
        for c in [left, right] {
            if !vtree.node(c).is_leaf() {
                loose.insert(c.0);
            }
        }
    }
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
    /// Build a disjunction of literals over the shared vtree. The result is
    /// not certified canonical ([`is_canonical`](Self::is_canonical)).
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
        crate::apply::exit::outcome(self, conjoin_clause_on(self, one, &clause))
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
        crate::apply::exit::outcome(self, conjoin_clause_on(self, f, clause))
    }
}

#[cfg(test)]
mod tests;
