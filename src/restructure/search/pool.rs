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
use crate::limits::{OperationError, Transient};
use crate::restructure::relevel::{rebuild_crossed_levels, rebuild_rotated_levels};
use crate::restructure::scratch::RestructureScratch;
use crate::vtree::rotate::{RotationInfo, rotate_pointers, swap_children};
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
}

impl Default for PoolSearchConfig {
    fn default() -> Self {
        PoolSearchConfig { max_sweeps: 4, max_inner_pairs: usize::MAX, crossed: true }
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
}

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
    ///
    /// The members should be canonical: a rebuilt level is then
    /// interchangeable with the one it replaced. A kept move can leave twins
    /// in a rebuilt level, which a later minimize contracts.
    ///
    /// # Errors
    ///
    /// [`OperationError::VtreeMismatch`] when the members do not share one
    /// vtree allocation; [`OperationError::OverBudget`] from a rebuild the
    /// byte budget refuses, with every member as it was.
    pub fn rotate_pool_if(
        &self,
        members: &mut [&mut Tdd],
        mv: PoolMove,
        bound: usize,
        accept: impl FnOnce(&PoolProbe<'_>) -> bool,
    ) -> Result<bool, OperationError> {
        let _op = self.limits().enter()?;
        let mut scratch = self.restructure().checkout(self.limits());
        rotate_pool_on(self, members, mv, bound, &mut scratch, accept)
    }

    /// A greedy descent over the members' shared vtree: at every internal
    /// node, the rotations both ways, crossed and not if `config.crossed`,
    /// each kept when it lowers the members' total live pairs. Sweeps repeat
    /// until one keeps nothing or `config.max_sweeps` is reached. Every member
    /// is reduced before the search and after it.
    ///
    /// The members keep their functions and end up sharing one vtree
    /// allocation, a new one if any move was kept. Stops are polled once per
    /// pivot; a stop keeps the moves kept so far, with the members reduced.
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
        for m in members.iter_mut() {
            self.reduce(m, crate::reduce::ReductionPlan::default())?;
        }
        stats.pairs_before = members.iter().map(|m| m.pair_count()).sum();
        let mut scratch = self.restructure().checkout(self.limits());
        let result = descend(self, members, config, &mut scratch, &mut stats);
        for m in members.iter_mut() {
            self.reduce(m, crate::reduce::ReductionPlan::default())?;
        }
        result?;
        stats.pairs_after = members.iter().map(|m| m.pair_count()).sum();
        Ok(stats)
    }
}

/// The sweeps of [`Engine::pool_search`], between its two reductions.
fn descend(
    eng: &Engine,
    members: &mut [&mut Tdd],
    config: &PoolSearchConfig,
    scratch: &mut RestructureScratch,
    stats: &mut PoolSearchStats,
) -> Result<(), OperationError> {
    let crossings: &[bool] = if config.crossed { &[false, true] } else { &[false] };
    while stats.sweeps < config.max_sweeps {
        stats.sweeps += 1;
        let internals: Vec<VtreeIdx> = members[0].vtree.internal_bottomup().map(|(v, _, _)| v).collect();
        let mut kept = 0usize;
        for v in internals {
            eng.limits().check_stop()?;
            'pivot: for kind in [RotationKind::Left, RotationKind::Right] {
                for &crossed in crossings {
                    let mv = PoolMove { rotation: RotationMove { pivot: v, kind }, crossed };
                    stats.probes += 1;
                    if rotate_pool_on(eng, members, mv, config.max_inner_pairs, scratch, |p| p.live_pairs_delta() < 0)? {
                        stats.accepts += 1;
                        kept += 1;
                        break 'pivot;
                    }
                }
            }
        }
        if kept == 0 {
            break;
        }
    }
    Ok(())
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
    dirty: Dirty,
    /// The levels the rebuild replaced, outer then inner.
    levels: Option<(Transient<'a, TddLevel>, Transient<'a, TddLevel>)>,
}

fn rotate_pool_on(
    eng: &Engine,
    members: &mut [&mut Tdd],
    mv: PoolMove,
    bound: usize,
    scratch: &mut RestructureScratch,
    accept: impl FnOnce(&PoolProbe<'_>) -> bool,
) -> Result<bool, OperationError> {
    if members.is_empty() {
        return Ok(false);
    }
    let shared = shared_vtree(members)?;
    if members.iter().any(|m| m.has_marginal_level()) {
        return Ok(false);
    }
    let Some((rotated, info)) = rotated_tree(&shared, mv) else { return Ok(false) };
    let lim = eng.limits();
    let (v, w) = (info.v_idx.idx(), info.w_idx.idx());
    let before: Vec<usize> = members.iter().map(|m| m.levels[v].live_pairs() + m.levels[w].live_pairs()).collect();
    let mut saved: Vec<Saved<'_>> = Vec::with_capacity(members.len());
    let mut outcome: Result<bool, OperationError> = Ok(true);
    for m in members.iter_mut() {
        saved.push(Saved {
            output: m.output,
            canonical: m.levels.is_canonical(m.output),
            dirty: std::mem::take(&mut m.dirty),
            levels: None,
        });
        m.levels.forget();
        m.vtree = Arc::clone(&rotated);
        let rebuilt = if mv.crossed {
            rebuild_crossed_levels(lim, m, &info, mv.rotation.kind, scratch, bound)
        } else {
            rebuild_rotated_levels(lim, m, &info, mv.rotation.kind, scratch, bound)
        };
        match rebuilt {
            Ok(Some((outer, inner))) => {
                saved.last_mut().expect("pushed above").levels = Some((Transient::new(lim, outer), Transient::new(lim, inner)));
            }
            Ok(None) => {
                outcome = Ok(false);
                break;
            }
            Err(e) => {
                outcome = Err(e);
                break;
            }
        }
    }
    if matches!(outcome, Ok(true)) {
        let after: Vec<usize> = members.iter().map(|m| m.levels[v].live_pairs() + m.levels[w].live_pairs()).collect();
        outcome = Ok(accept(&PoolProbe { before: &before, after: &after }));
    }
    if matches!(outcome, Ok(true)) {
        // The preimages hand their charge back as they drop; the obligations
        // taken from each member go back under what the rebuild recorded.
        for (m, s) in members.iter_mut().zip(saved) {
            m.dirty.merge_under(s.dirty);
        }
        return Ok(true);
    }
    // Put back every member the loop reached, in any order: each has its own
    // levels, and all of them go back to the one old allocation.
    for (m, s) in members.iter_mut().zip(saved) {
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
        m.vtree = Arc::clone(&shared);
    }
    outcome.map(|_| false)
}

/// A copy of `shared` with `mv` applied and its bottom-up order repaired, and
/// the rotation's information; `None` where the move does not apply.
fn rotated_tree(shared: &Arc<Vtree>, mv: PoolMove) -> Option<(Arc<Vtree>, RotationInfo)> {
    let RotationMove { pivot, kind } = mv.rotation;
    let VtreeNode::Internal { left, right, .. } = *shared.node(pivot) else { return None };
    let promoted = match kind {
        RotationKind::Left => right,
        RotationKind::Right => left,
    };
    if shared.node(promoted).is_leaf() {
        return None;
    }
    let mut tree = Vtree::clone(shared);
    if mv.crossed {
        swap_children(&mut tree, promoted);
    }
    let pending = rotate_pointers(&mut tree, pivot, kind)?;
    let info = pending.commit(&mut tree);
    Some((Arc::new(tree), info))
}

#[cfg(test)]
#[path = "tests/pool.rs"]
mod tests;
