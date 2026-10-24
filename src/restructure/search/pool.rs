//! One vtree move applied to several diagrams at once, and a descent of such
//! moves.
//!
//! Operands of a binary operation must share one vtree allocation, so a host
//! that holds many diagrams on one vtree — the live formulas of a compiler —
//! can change the vtree only for all of them together. [`Engine::rotate_pool_if`]
//! applies one move to every member and keeps it for all or for none;
//! [`Engine::pool_search`] runs a greedy descent of such moves on the members'
//! total live pairs.
//!
//! A move is a rotation, optionally *crossed*: the promoted child's two
//! children are swapped first. A swap alone changes no diagram's size, since a
//! level is symmetric in its two sides, but rotations alone keep the
//! left-to-right order of the leaves. Crossing is what lets the search change
//! that order: at a node with children `A` and `(B, C)` the plain left
//! rotation groups `A` with `B`, the crossed one `A` with `C`.

use std::sync::Arc;

use crate::Engine;
use crate::diagram::{Dirty, Tdd, TddLevel, TddNodeId};
use crate::limits::{Charged, Limits, OperationError, Transient};
use crate::restructure::relevel::{Prepared, finish_levels, prepare_levels, rebuild_levels, score_levels, score_levels_crossed_too};
use crate::restructure::scratch::RestructureScratch;
use crate::vtree::rotate::{PendingTopo, RotationInfo, rotate_pointers};
use crate::vtree::{RotationKind, Vtree, VtreeIdx, VtreeNode};

use super::RotationMove;

/// A rotation, with the promoted child's children swapped first if
/// `crossed`.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct PoolMove {
    /// The rotation.
    pub rotation: RotationMove,
    /// Swap the children of the node the rotation promotes before rotating.
    pub crossed: bool,
}

/// What a probed move did to each member, shown to the decision that keeps
/// or reverts it.
#[derive(Debug)]
pub struct PoolProbe<'a> {
    before: &'a [usize],
    after: &'a [usize],
}

impl PoolProbe<'_> {
    /// Live pairs of each member's two rebuilt levels before the move.
    pub fn before(&self) -> &[usize] {
        self.before
    }

    /// Live pairs of each member's two rebuilt levels after the move.
    pub fn after(&self) -> &[usize] {
        self.after
    }

    /// Live pairs over all members after the move, minus before. Negative
    /// means the move shrank the pool; no other level moved.
    pub fn live_pairs_delta(&self) -> i64 {
        self.after.iter().map(|&p| p as i64).sum::<i64>() - self.before.iter().map(|&p| p as i64).sum::<i64>()
    }
}

/// Bounds of a [`pool_search`](Engine::pool_search).
#[derive(Clone, Debug)]
pub struct PoolSearchConfig {
    /// Sweeps over the internal nodes; a sweep that keeps nothing ends the
    /// search earlier.
    pub max_sweeps: usize,
    /// A move whose rebuild of one member's levels would pass this many
    /// pairs is abandoned. `usize::MAX` is no bound.
    pub max_inner_pairs: usize,
    /// Also try the crossed moves.
    pub crossed: bool,
    /// The search ends, keeping what it kept, once it has charged this many
    /// units to the work clock, its opening reduction included. `u64::MAX`
    /// is no bound.
    pub max_work_units: u64,
}

impl Default for PoolSearchConfig {
    fn default() -> Self {
        PoolSearchConfig { max_sweeps: 4, max_inner_pairs: usize::MAX, crossed: true, max_work_units: u64::MAX }
    }
}

/// What a [`pool_search`](Engine::pool_search) did.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PoolSearchStats {
    /// Sweeps started.
    pub sweeps: usize,
    /// Moves probed, abandoned ones included.
    pub probes: usize,
    /// Moves kept.
    pub accepts: usize,
    /// Live pairs over all members before the search, after the opening
    /// reduction.
    pub pairs_before: usize,
    /// Live pairs over all members after the search and its closing
    /// reduction.
    pub pairs_after: usize,
    /// Work charged by the search, its two reductions included.
    pub work_units: u64,
}

/// Work units a member's rebuild charges beyond the pairs of its two old and
/// two new levels: building two levels, and keeping them or putting the old
/// ones back, costs a few allocations and the member's saved state however
/// small the levels are. In units of the pairs the rebuild reads.
const REBUILD_UNITS: u64 = 64;

/// Work units a probe charges beyond what its members charge: the move made
/// on the tree and undone, and the probe's tables.
const PROBE_UNITS: u64 = 16;

impl Engine {
    /// Apply `mv` to every member, show the change to `accept`, and keep it
    /// for all members or for none.
    ///
    /// The members must share one vtree allocation. A kept move leaves them
    /// sharing a new one, the rotated tree; a declined or abandoned one leaves
    /// every member and the old allocation as they were. A move that does not
    /// apply at its pivot, a member with summed-out levels, or a member whose
    /// rebuild would pass `bound` pairs abandons the move and returns
    /// `Ok(false)` without calling `accept`. An empty pool keeps nothing.
    /// The members are only read until `accept` keeps the move: each
    /// member's rebuild is prepared first, which gives the pairs its two
    /// levels would hold, and the levels are written from it once the move is
    /// kept. The copy of the tree charges its nodes to the work clock, and
    /// each member the pairs of its two old and two new levels and the fixed
    /// cost of a rebuild; a member whose two levels hold a pair each, or
    /// none, holds as many after the move and is not prepared.
    ///
    /// The members should be canonical: a rebuilt level is then
    /// interchangeable with the one it replaced. A kept move can leave twins
    /// in a rebuilt level, which a later minimize contracts.
    ///
    /// # Errors
    ///
    /// [`OperationError::VtreeMismatch`] when the members do not share one
    /// vtree allocation, [`OperationError::LevelNotInVtree`] for an invalid
    /// pivot, [`OperationError::OverBudget`] for refused storage, or
    /// [`OperationError::Stopped`] for cancellation. An error leaves every
    /// member as it was; storage refused once `accept` has kept the move
    /// does too.
    pub fn rotate_pool_if(
        &self,
        members: &mut [&mut Tdd],
        mv: PoolMove,
        bound: usize,
        accept: impl FnOnce(&PoolProbe<'_>) -> bool,
    ) -> Result<bool, OperationError> {
        let _op = self.limits().enter()?;
        if members.is_empty() {
            return Ok(false);
        }
        let shared = shared_vtree(members)?;
        if mv.rotation.pivot.idx() >= shared.num_nodes() {
            return Err(OperationError::LevelNotInVtree(mv.rotation.pivot));
        }
        if members.iter().any(|m| m.has_marginal_level()) {
            return Ok(false);
        }
        let mut tree = Vtree::clone(&shared);
        let Some(pending) = move_in_place(&mut tree, mv) else { return Ok(false) };
        let info = pending.commit(&mut tree);
        let mut scratch = self.scratch.restructure.checkout(self);
        let copy = PROBE_UNITS.saturating_add(tree.num_nodes() as u64);
        let kept = probe(self, members, info, mv, bound, copy, &mut scratch, false, Twin::Alone, accept)?;
        if kept {
            install(members, tree);
        }
        Ok(kept)
    }

    /// A greedy descent over the members' shared vtree: at every internal
    /// node, the rotations both ways, crossed and not if `config.crossed`,
    /// each kept when it lowers the members' total live pairs. Sweeps repeat
    /// until one keeps nothing, `config.max_sweeps` is reached, or the
    /// search has charged `config.max_work_units`. Every member is reduced
    /// before the search and after it.
    ///
    /// The members keep their functions and end up sharing one vtree
    /// allocation, a new one if any move was kept. Stops are polled once per
    /// pivot and within each probe; an error keeps the moves kept so far and
    /// returns without the closing reduction. A probe scores the move
    /// member by member and stops once the scores refuse it, and charges
    /// every member the pairs of its two levels all the same; a move the
    /// scores keep then charges what [`rotate_pool_if`](Self::rotate_pool_if)
    /// does, less the copy of the tree, and the nodes whose place in the
    /// bottom-up order it changes: the descent moves one copy in place.
    ///
    /// # Errors
    ///
    /// As [`rotate_pool_if`](Self::rotate_pool_if), and the errors of the
    /// reductions.
    pub fn pool_search(&self, members: &mut [&mut Tdd], config: &PoolSearchConfig) -> Result<PoolSearchStats, OperationError> {
        let _op = self.limits().enter()?;
        let mut stats = PoolSearchStats::default();
        if members.is_empty() {
            return Ok(stats);
        }
        shared_vtree(members)?;
        let start = self.limits().work_units();
        let result = self.pool_search_from(members, config, start, &mut stats);
        stats.work_units = self.limits().work_units() - start;
        result.map(|()| stats)
    }

    /// [`pool_search`](Self::pool_search) on a pool that shares one vtree,
    /// its work counted from the clock's `start`.
    fn pool_search_from(
        &self,
        members: &mut [&mut Tdd],
        config: &PoolSearchConfig,
        start: u64,
        stats: &mut PoolSearchStats,
    ) -> Result<(), OperationError> {
        for m in members.iter_mut() {
            self.reduce(m, crate::reduce::ReductionPlan::default())?;
        }
        stats.pairs_before = members.iter().map(|m| m.pair_count()).sum();
        let mut scratch = self.scratch.restructure.checkout(self);
        // The members' tree, moved in place as moves are kept. The members
        // keep the old allocation until the descent ends, which hands them
        // this one if it kept a move: their levels follow the moves, and
        // nothing in a probe reads a member's tree.
        let mut tree = Vtree::clone(&members[0].vtree);
        let result = descend(self, members, &mut tree, config, start, &mut scratch, stats);
        if stats.accepts > 0 {
            install(members, tree);
        }
        result?;
        for m in members.iter_mut() {
            self.reduce(m, crate::reduce::ReductionPlan::default())?;
        }
        stats.pairs_after = members.iter().map(|m| m.pair_count()).sum();
        Ok(())
    }
}

/// The sweeps of [`Engine::pool_search`], between its two reductions, on
/// `tree`, the members' tree with the moves kept so far, from the work
/// clock's `start`.
fn descend(
    eng: &Engine,
    members: &mut [&mut Tdd],
    tree: &mut Vtree,
    config: &PoolSearchConfig,
    start: u64,
    scratch: &mut RestructureScratch,
    stats: &mut PoolSearchStats,
) -> Result<(), OperationError> {
    let crossings: &[bool] = if config.crossed { &[false, true] } else { &[false] };
    let spent = || eng.limits().work_units() - start >= config.max_work_units;
    // The pivots whose every move was probed and refused since the last kept
    // move that could change their outcome: a kept move clears its pivot,
    // the pivot's two children and its parent. A refused pivot is passed
    // over.
    let mut refused = vec![false; tree.num_nodes()];
    // The charge of each move's last refusal, kept while the two levels the
    // move reads are unchanged. A move reads its pivot's level and its
    // promoted child's; a kept move rewrites its own pivot's and promoted
    // child's. So it changes every move at those two nodes and, at the
    // pivot's parent, the moves that promote the pivot, and no other. A
    // move refused before is refused again at the same charge, so it is
    // charged and not probed: the descent decides and charges what it would
    // probing every move.
    let mut refusals: Vec<[Option<u64>; 4]> = vec![[None; 4]; tree.num_nodes()];
    // Asked once: a kept move rebuilds every member's two levels and so
    // forgets what each member knows of its levels, and asking at every
    // probe read every level of every member again, most of a search over
    // many small members. A rotation rebuilds structural levels as
    // structural levels, so the answer holds for the whole search.
    let marginal = members.iter().any(|m| m.has_marginal_level());
    // The scores of the crossed move at the pivot and rotation under way,
    // one per member, taken with the plain move's ([`Twin`]).
    let mut crossed_scores = Transient::new(eng.limits(), Vec::new());
    eng.limits().reserve_exact(&mut crossed_scores, members.len())?;
    while stats.sweeps < config.max_sweeps && !spent() {
        stats.sweeps += 1;
        let internals: Vec<VtreeIdx> = tree.internal_bottomup().map(|(v, _, _)| v).collect();
        let mut kept = 0usize;
        for v in internals {
            eng.limits().check_stop()?;
            if spent() {
                return Ok(());
            }
            if refused[v.idx()] {
                continue;
            }
            let (left, right) = tree.children(v);
            let parent = tree.node(v).parent();
            let mut moved = false;
            'pivot: for kind in [RotationKind::Left, RotationKind::Right] {
                let mut twin_scored = false;
                for &crossed in crossings {
                    if spent() {
                        return Ok(());
                    }
                    let slot = move_slot(kind, crossed);
                    stats.probes += 1;
                    if let Some(units) = refusals[v.idx()][slot]
                        && replay_refusals()
                    {
                        eng.limits().charge_work(units);
                        continue;
                    }
                    let mv = PoolMove { rotation: RotationMove { pivot: v, kind }, crossed };
                    // A plain move whose crossing is to be probed scores the
                    // crossing too; refused, it leaves the levels as they
                    // were, and the crossing reads its scores.
                    let twin = if crossed {
                        if twin_scored { Twin::Given(&crossed_scores) } else { Twin::Alone }
                    } else if config.crossed && refusals[v.idx()][move_slot(kind, true)].is_none() {
                        crossed_scores.clear();
                        twin_scored = true;
                        Twin::Fill(&mut crossed_scores)
                    } else {
                        Twin::Alone
                    };
                    let before = eng.limits().work_units();
                    if marginal || !probe_in_place(eng, members, tree, mv, config.max_inner_pairs, scratch, twin)? {
                        refusals[v.idx()][slot] = Some(eng.limits().work_units() - before);
                        continue;
                    }
                    debug_assert_eq!(members.iter().any(|m| m.has_marginal_level()), marginal);
                    stats.accepts += 1;
                    kept += 1;
                    moved = true;
                    let promoted = match kind {
                        RotationKind::Left => right,
                        RotationKind::Right => left,
                    };
                    refusals[v.idx()] = [None; 4];
                    refusals[promoted.idx()] = [None; 4];
                    if let Some(parent) = parent {
                        let at_parent = if tree.children(parent).1 == v { RotationKind::Left } else { RotationKind::Right };
                        refusals[parent.idx()][move_slot(at_parent, false)] = None;
                        refusals[parent.idx()][move_slot(at_parent, true)] = None;
                    }
                    break 'pivot;
                }
            }
            if moved {
                for t in [Some(v), Some(left), Some(right), parent].into_iter().flatten() {
                    refused[t.idx()] = false;
                }
            } else {
                refused[v.idx()] = true;
            }
        }
        if kept == 0 {
            break;
        }
    }
    Ok(())
}

/// Whether [`descend`] charges a cached refusal rather than probing again.
#[inline]
fn replay_refusals() -> bool {
    #[cfg(test)]
    if !tests::replays_refusal() { return false; }
    true
}

/// The slot of a move among its pivot's four in [`descend`]'s refusals.
fn move_slot(kind: RotationKind, crossed: bool) -> usize {
    2 * usize::from(kind == RotationKind::Right) + usize::from(crossed)
}

/// One probe of the descent: `mv` made on `tree` in place and kept when it
/// lowers the members' live pairs, else undone.
fn probe_in_place(
    eng: &Engine,
    members: &mut [&mut Tdd],
    tree: &mut Vtree,
    mv: PoolMove,
    bound: usize,
    scratch: &mut RestructureScratch,
    twin: Twin<'_>,
) -> Result<bool, OperationError> {
    let Some(pending) = move_in_place(tree, mv) else { return Ok(false) };
    let info = pending.info();
    match probe(eng, members, info, mv, bound, PROBE_UNITS, scratch, true, twin, |p| p.live_pairs_delta() < 0) {
        Ok(true) => {
            // The bottom-up order's repair moves the promoted node past the
            // subtree it now needs below it, and every node between.
            let misplaced = match mv.rotation.kind {
                RotationKind::Left => info.a_idx,
                RotationKind::Right => info.c_idx,
            };
            let moved = tree.topo_pos(misplaced).saturating_sub(tree.topo_pos(info.w_idx));
            eng.limits().charge_work(u64::from(moved));
            pending.commit(tree);
            Ok(true)
        }
        outcome => {
            pending.revert(tree);
            if mv.crossed {
                tree.swap_children(info.w_idx);
            }
            outcome
        }
    }
}

/// Make `mv` on `tree`, its children swapped first if crossed, with the
/// bottom-up order's repair owed; `None`, the tree untouched, where the move
/// does not apply. A crossed move that is reverted owes the swap back.
fn move_in_place(tree: &mut Vtree, mv: PoolMove) -> Option<PendingTopo> {
    let RotationMove { pivot, kind } = mv.rotation;
    let VtreeNode::Internal { left, right, .. } = *tree.node(pivot) else { return None };
    let promoted = match kind {
        RotationKind::Left => right,
        RotationKind::Right => left,
    };
    if tree.node(promoted).is_leaf() {
        return None;
    }
    if mv.crossed {
        tree.swap_children(promoted);
    }
    rotate_pointers(tree, pivot, kind)
}

/// Hand every member `tree`, one allocation for all.
fn install(members: &mut [&mut Tdd], tree: Vtree) {
    let tree = Arc::new(tree);
    for m in members.iter_mut() {
        m.vtree = Arc::clone(&tree);
    }
}

/// The members' one vtree allocation, or the mismatch error.
fn shared_vtree(members: &[&mut Tdd]) -> Result<Arc<Vtree>, OperationError> {
    let shared = Arc::clone(&members[0].vtree);
    if members.iter().any(|m| !Arc::ptr_eq(&m.vtree, &shared)) {
        return Err(OperationError::VtreeMismatch);
    }
    Ok(shared)
}

/// What one member needs to be put back as it was.
struct Saved<'a> {
    output: TddNodeId,
    canonical: bool,
    closed: bool,
    dirty: Dirty,
    /// The levels the rebuild replaced, outer then inner.
    levels: Option<(Transient<'a, TddLevel>, Transient<'a, TddLevel>)>,
}

/// Member `i`'s two levels at the move `info` describes rebuilt within
/// `bound`, from `prepared` where the probe prepared them, and what it needs
/// to be put back saved: the new levels' pairs, or `None` where the rebuild
/// would pass `bound`.
#[allow(clippy::too_many_arguments)]
fn rebuild_member<'a>(
    lim: &'a Limits,
    i: usize,
    m: &mut Tdd,
    info: &RotationInfo,
    mv: PoolMove,
    bound: usize,
    prepared: Option<&mut Prepared>,
    saved: &mut Vec<(usize, Saved<'a>)>,
    scratch: &mut RestructureScratch,
) -> Result<Option<usize>, OperationError> {
    saved.push((i, Saved {
        output: m.output,
        canonical: m.levels.is_canonical(m.output),
        closed: m.levels.is_closed(),
        dirty: std::mem::take(&mut m.dirty),
        levels: None,
    }));
    m.levels.forget();
    // No member has a marginal level here.
    let rebuilt = match prepared {
        Some(prepared) => finish_levels(lim, m, info, mv.rotation.kind, false, scratch, prepared).map(Some),
        None => rebuild_levels(lim, m, info, mv.rotation.kind, mv.crossed, false, scratch, bound),
    };
    let Some((outer, inner)) = rebuilt? else {
        return Ok(None);
    };
    saved.last_mut().expect("pushed above").1.levels = Some((Transient::new(lim, outer), Transient::new(lim, inner)));
    Ok(Some(m.levels[info.v_idx.idx()].live_pairs() + m.levels[info.w_idx.idx()].live_pairs()))
}

/// The rebuilds a probe prepared, one for each member but those it left
/// out, charged with their buffers.
#[derive(Default)]
struct PreparedMembers(Vec<Option<Prepared>>);

impl Charged for PreparedMembers {
    fn charged_bytes(&self) -> u64 {
        self.0.iter().flatten().fold(self.0.charged_bytes(), |bytes, p| bytes + p.charged_bytes())
    }
}

/// The members' two levels at the move `info` describes prepared, the change
/// shown to `accept`, and if kept, rebuilt for all members; `extra` units
/// charged for the move itself. The members' tree is the caller's: a kept
/// move leaves their levels on the tree `info` was taken from and their
/// pointers on the old one.
///
/// Each member's levels are only read until the move is kept. The probe
/// prepares each member's rebuild ([`prepare_levels`]), which gives the pairs
/// its two levels would hold, and writes the levels from it once the move is
/// kept ([`finish_levels`]). A member whose two levels hold one pair each, or
/// none, holds as many after the move and is not prepared; a kept move
/// rebuilds it. The charge of a member is the pairs of its two old and two
/// new levels and [`REBUILD_UNITS`], taken when it is prepared, or for a
/// member not prepared, by its rebuild once the move is kept.
///
/// Where `accept` keeps exactly the moves that lower the members' total
/// (`shrinks`), [`probe_scored`] decides instead, from scores, with `twin`.
#[allow(clippy::too_many_arguments)]
fn probe(
    eng: &Engine,
    members: &mut [&mut Tdd],
    info: RotationInfo,
    mv: PoolMove,
    bound: usize,
    extra: u64,
    scratch: &mut RestructureScratch,
    shrinks: bool,
    twin: Twin<'_>,
    accept: impl FnOnce(&PoolProbe<'_>) -> bool,
) -> Result<bool, OperationError> {
    if shrinks {
        return probe_scored(eng, members, info, mv, bound, extra, scratch, twin);
    }
    let lim = eng.limits();
    // A level of one pair rebuilds into one of one pair, two pairs for the
    // two levels, which a bound above two lets through.
    let defer = bound > 2;
    let mut gate = lim.gate();
    gate.poll(extra)?;
    let mut before = Transient::new(lim, Vec::new());
    lim.reserve_exact(&mut before, members.len())?;
    let mut after = Transient::new(lim, Vec::new());
    lim.reserve_exact(&mut after, members.len())?;
    // Each member's prepared rebuild, held until the move is decided; a
    // member prepared was charged its rebuild.
    let mut prepared = Transient::new(lim, PreparedMembers::default());
    lim.reserve_exact(&mut prepared.0, members.len())?;
    for m in members.iter() {
        let (outer, inner) = (m.levels[info.v_idx.idx()].live_pairs(), m.levels[info.w_idx.idx()].live_pairs());
        let old = outer + inner;
        before.push(old);
        if defer && outer <= 1 && inner == outer {
            gate.poll(old as u64 + 1)?;
            after.push(old);
            prepared.0.push(None);
            continue;
        }
        prepared.0.push(Some(Prepared::default()));
        let slot = prepared.0.last_mut().and_then(Option::as_mut).expect("pushed above");
        let Some(new) = prepare_levels(lim, m, &info, mv.rotation.kind, mv.crossed, false, scratch, slot, bound)? else {
            return Ok(false);
        };
        gate.poll((old + new) as u64 + REBUILD_UNITS)?;
        after.push(new);
    }
    // The gate's residue charged and the stop tested before `accept`
    // decides.
    gate.flush()?;
    lim.check_stop()?;
    if !accept(&PoolProbe { before: &before, after: &after }) {
        return Ok(false);
    }
    // Kept: every member rebuilt, each put back if a later one is refused.
    let mut saved = Transient::new(lim, Vec::<(usize, Saved<'_>)>::new());
    lim.reserve_exact(&mut saved, members.len())?;
    let mut outcome: Result<bool, OperationError> = Ok(true);
    for (i, m) in members.iter_mut().enumerate() {
        let rebuilt = rebuild_member(lim, i, m, &info, mv, bound, prepared.0[i].as_mut(), &mut saved, scratch);
        debug_assert!(!matches!(rebuilt, Ok(None)), "a rebuild passes the bound its preparation passed");
        outcome = match rebuilt {
            Ok(Some(pairs)) => {
                debug_assert_eq!(pairs, after[i], "a rebuild leaves the pairs its preparation gave");
                if prepared.0[i].is_some() { Ok(true) } else { gate.poll((before[i] + pairs) as u64 + REBUILD_UNITS).map(|()| true) }
            }
            other => other.map(|_| false),
        };
        if !matches!(outcome, Ok(true)) {
            break;
        }
    }
    settle(lim, members, &info, &mut saved, outcome)
}

/// [`probe`] for a move kept exactly when it lowers the members' total.
///
/// Each member is scored ([`score_levels`]), which is never above the pairs
/// its rebuild holds and almost always equal to them. A score reads the
/// member's two levels and builds nothing, so the probe charges each
/// member their pairs and [`REBUILD_UNITS`]. The members are scored in
/// turn, each under a cap: the old total less the scores before it and the
/// least the members after it can hold, a pair for each node of their outer
/// level. A score that reaches its cap refuses the move, and the members
/// after it are not read, though charged: the charge depends on the levels
/// alone. A move whose scores do not lower the total is refused, the
/// members only read. Otherwise every member is rebuilt in turn, charged
/// what preparing it costs, and the move kept if their pairs lower the total
/// after all; if not, or if a rebuild passes `bound` or fails, every member
/// is put back. So the scores decide, and no member's rebuild waits for the
/// others'.
#[allow(clippy::too_many_arguments)]
fn probe_scored(
    eng: &Engine,
    members: &mut [&mut Tdd],
    info: RotationInfo,
    mv: PoolMove,
    bound: usize,
    extra: u64,
    scratch: &mut RestructureScratch,
    mut twin: Twin<'_>,
) -> Result<bool, OperationError> {
    debug_assert!(match twin {
        Twin::Alone => true,
        Twin::Fill(_) => !mv.crossed,
        Twin::Given(_) => mv.crossed,
    });
    let lim = eng.limits();
    let defer = bound > 2;
    let mut gate = lim.gate();
    gate.poll(extra)?;
    let (v, w) = (info.v_idx.idx(), info.w_idx.idx());
    let mut before = Transient::new(lim, Vec::new());
    lim.reserve_exact(&mut before, members.len())?;
    // Whether each member is scored: one with at most one pair on each
    // level is not, its pairs unchanged.
    let mut scored = Transient::new(lim, Vec::new());
    lim.reserve_exact(&mut scored, members.len())?;
    // The least pairs each member can end with: a member scored keeps a
    // pair for each node of its outer level ([`score_levels`]), one not
    // scored its pairs.
    let mut floors = Transient::new(lim, Vec::new());
    lim.reserve_exact(&mut floors, members.len())?;
    for m in members.iter() {
        let ((outer, owners), inner) = (pairs_and_owners(&m.levels[v]), m.levels[w].live_pairs());
        let old = outer + inner;
        before.push(old);
        let scores = !(defer && outer <= 1 && inner == outer);
        scored.push(scores);
        floors.push(if scores { owners } else { old });
    }
    let total = before.iter().sum::<usize>();
    let mut after = Transient::new(lim, Vec::new());
    lim.reserve_exact(&mut after, members.len())?;
    after.extend_from_slice(&before);
    // The largest members first: a move they refuse is refused with less
    // of the pool read.
    let mut order = Transient::new(lim, Vec::new());
    lim.reserve_exact(&mut order, members.len())?;
    order.extend(0..members.len());
    order.sort_by_key(|&i| std::cmp::Reverse(before[i]));
    if let Twin::Fill(slots) = &mut twin {
        slots.resize(members.len(), None);
    }
    // The floors of the members still to score, and the pairs of those
    // scored, of the move and of its crossing where it is scored too.
    let mut rest = floors.iter().sum::<usize>();
    let (mut sum, mut sum_crossed) = (0usize, Some(0usize));
    let cap = |sum: usize, rest: usize| total.saturating_sub(sum + rest);
    for (at, &i) in order.iter().enumerate() {
        let m = &members[i];
        rest -= floors[i];
        let old = before[i];
        if !scored[i] {
            gate.poll(old as u64 + 1)?;
            sum += old;
            sum_crossed = sum_crossed.map(|s| s + old);
            continue;
        }
        #[cfg(test)]
        tests::note_score();
        let (kind, score) = (mv.rotation.kind, &mut scratch.score);
        let new = match &mut twin {
            Twin::Alone => score_levels(lim, m, &info, kind, mv.crossed, score, bound, cap(sum, rest))?,
            Twin::Fill(slots) => {
                // A crossing refused by an earlier member's score is capped
                // at nothing.
                let crossed_cap = sum_crossed.map_or(0, |s| cap(s, rest));
                let [plain, crossed] = score_levels_crossed_too(lim, m, &info, kind, score, bound, [cap(sum, rest), crossed_cap])?;
                slots[i] = Some(crossed);
                sum_crossed = sum_crossed.zip(crossed).map(|(s, n)| s + n);
                plain
            }
            Twin::Given(slots) => match slots.get(i) {
                Some(&Some(given)) => given,
                _ => score_levels(lim, m, &info, kind, mv.crossed, score, bound, cap(sum, rest))?,
            },
        };
        let Some(new) = new else {
            // Refused, by the cap or the bound: this member and the ones
            // after it charged as if scored.
            for &j in &order[at..] {
                gate.poll(before[j] as u64 + if scored[j] { REBUILD_UNITS } else { 1 })?;
            }
            #[cfg(debug_assertions)]
            debug_check_refusal(lim, members, &info, mv, bound, total, scratch)?;
            gate.flush()?;
            lim.check_stop()?;
            return Ok(false);
        };
        gate.poll(old as u64 + REBUILD_UNITS)?;
        after[i] = new;
        sum += new;
    }
    // A score is at most the pairs: a total no lower than before is a move
    // the search refuses.
    if sum >= total {
        #[cfg(debug_assertions)]
        debug_check_scores(lim, members, &info, mv, bound, &after, scratch)?;
        gate.flush()?;
        lim.check_stop()?;
        return Ok(false);
    }
    let mut saved = Transient::new(lim, Vec::<(usize, Saved<'_>)>::new());
    lim.reserve_exact(&mut saved, members.len())?;
    let mut outcome: Result<bool, OperationError> = Ok(true);
    for (i, m) in members.iter_mut().enumerate() {
        let rebuilt = rebuild_member(lim, i, m, &info, mv, bound, None, &mut saved, scratch);
        outcome = match rebuilt {
            Ok(Some(pairs)) => {
                debug_assert!(!scored[i] || pairs >= after[i], "a score is at most the pairs: {} > {pairs}", after[i]);
                after[i] = pairs;
                gate.poll((before[i] + pairs) as u64 + REBUILD_UNITS).map(|()| true)
            }
            other => other.map(|_| false),
        };
        if !matches!(outcome, Ok(true)) {
            break;
        }
    }
    // The gate's residue charged and the stop tested before the move is
    // kept.
    let outcome = outcome.and_then(|kept| {
        gate.flush()?;
        lim.check_stop()?;
        Ok(kept && after.iter().sum::<usize>() < before.iter().sum::<usize>())
    });
    settle(lim, members, &info, &mut saved, outcome)
}

/// A level's live pairs and the nodes that hold them.
fn pairs_and_owners(level: &TddLevel) -> (usize, usize) {
    match level.implicit() {
        Some(d) => (d.pairs(), if d.pairs_per_node() == 0 { 0 } else { d.nodes() }),
        None => {
            let (pairs, live, _) = level.pair_census();
            (pairs as usize, live as usize)
        }
    }
}

/// What a scored probe ([`probe_scored`]) knows of the crossing of its move:
/// the same rotation at the same pivot with the promoted node's children
/// swapped, which reads the same two levels.
enum Twin<'s> {
    /// Nothing: the move scored alone.
    Alone,
    /// The move is plain and its crossing is still to be probed: each
    /// member's score of the crossing taken with the move's, filed at the
    /// member's index, `None` for a member not scored or not reached.
    Fill(&'s mut Vec<Option<Option<usize>>>),
    /// The move is the crossing of a plain move refused just before, on the
    /// same levels, which filed these scores.
    Given(&'s [Option<Option<usize>>]),
}

/// The end of a probe's rebuilds, `saved` holding what each member rebuilt
/// needs to be put back: where `outcome` keeps the move, every member's two
/// rebuilt levels closed and its obligations merged back; otherwise every
/// member rebuilt put back as it was. Returns `outcome`.
fn settle<'a>(
    lim: &'a Limits,
    members: &mut [&mut Tdd],
    info: &RotationInfo,
    saved: &mut Vec<(usize, Saved<'a>)>,
    outcome: Result<bool, OperationError>,
) -> Result<bool, OperationError> {
    let (v, w) = (info.v_idx.idx(), info.w_idx.idx());
    if matches!(outcome, Ok(true)) {
        // The preimages hand their charge back as they drop; the obligations
        // taken from each member go back under what the rebuild recorded.
        // The two levels each member rebuilt are built stored; the kept move
        // is the operation's end, which closes them.
        for (i, s) in saved.drain(..) {
            let m = &mut members[i];
            m.close_changed_levels(&[info.v_idx, info.w_idx]);
            m.dirty.merge_under(s.dirty);
        }
        return Ok(true);
    }
    // Put back every member the rebuild reached, in any order: each has its
    // own levels.
    for (i, s) in saved.drain(..) {
        let m = &mut members[i];
        if let Some((outer, inner)) = s.levels {
            let rebuilt_outer = std::mem::replace(&mut m.levels[v], outer.keep());
            let rebuilt_inner = std::mem::replace(&mut m.levels[w], inner.keep());
            lim.discard(rebuilt_outer);
            lim.discard(rebuilt_inner);
        }
        m.output = s.output;
        m.dirty = s.dirty;
        if s.canonical {
            m.levels.certify(s.output);
        }
        if s.closed {
            m.reinstate_closed_levels();
        }
    }
    outcome
}

/// A debug build's check of a move its scores refused: every member's
/// preparation gives at least its score.
#[cfg(debug_assertions)]
fn debug_check_scores(
    lim: &Limits,
    members: &[&mut Tdd],
    info: &RotationInfo,
    mv: PoolMove,
    bound: usize,
    scores: &[usize],
    scratch: &mut RestructureScratch,
) -> Result<(), OperationError> {
    let (v, w) = (info.v_idx.idx(), info.w_idx.idx());
    for (m, &score) in members.iter().zip(scores) {
        if m.levels[v].live_pairs() <= 1 && m.levels[w].live_pairs() == m.levels[v].live_pairs() {
            continue;
        }
        let mut prepared = Prepared::default();
        let new = prepare_levels(lim, m, info, mv.rotation.kind, mv.crossed, false, scratch, &mut prepared, bound)?;
        lim.discard(prepared);
        if let Some(new) = new {
            debug_assert!(new >= score, "a score is at most the pairs: {score} > {new}");
        }
    }
    Ok(())
}

/// The check, in debug builds, that a move whose scores were refused short
/// of the last member is one whose rebuilt members would hold at least
/// `total` pairs.
#[cfg(debug_assertions)]
fn debug_check_refusal(
    lim: &Limits,
    members: &[&mut Tdd],
    info: &RotationInfo,
    mv: PoolMove,
    bound: usize,
    total: usize,
    scratch: &mut RestructureScratch,
) -> Result<(), OperationError> {
    let (v, w) = (info.v_idx.idx(), info.w_idx.idx());
    let mut pairs = 0usize;
    for m in members {
        let (outer, inner) = (m.levels[v].live_pairs(), m.levels[w].live_pairs());
        if outer <= 1 && inner == outer {
            pairs += outer + inner;
            continue;
        }
        let mut prepared = Prepared::default();
        let new = prepare_levels(lim, m, info, mv.rotation.kind, mv.crossed, false, scratch, &mut prepared, bound)?;
        lim.discard(prepared);
        let Some(new) = new else { return Ok(()) };
        pairs += new;
    }
    debug_assert!(pairs >= total, "a capped score refused a move to {pairs} pairs from {total}");
    Ok(())
}

#[cfg(test)]
#[path = "tests/pool.rs"]
mod tests;
