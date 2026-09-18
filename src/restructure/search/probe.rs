//! The rotation trial: rebuild the levels a sequence of rotations changes,
//! score them, then commit the sequence or restore what it replaced.
//!
//! A rotation rebuilds exactly two levels and keeps the node indices at its
//! pivot, so nothing above the pivot has to be rewritten and a trial that is
//! not kept puts the two levels back by assignment. That is what lets a trial
//! run several rotations before anything is scored: it keeps the first
//! preimage of every level the sequence touches, and a revert restores those.

use std::sync::Arc;

use smallvec::SmallVec;

use crate::Engine;
use crate::diagram::{Dirty, Tdd, TddLevel, TddNodeId};
use crate::limits::{Limits, OperationError, Transient};
use crate::restructure::relevel::{RestructureScratch, restructure_inner_search};
use crate::vtree::rotate::{PendingTopo, RotationInfo, rotate_pointers};
use crate::vtree::{RotationKind, Vtree, VtreeIdx};

/// One rotation: which internal vtree node it turns, and which way.
///
/// A left rotation at `pivot` promotes `pivot`'s right child, a right rotation
/// its left child, so the two are each other's inverse at the same pivot —
/// which is what [`inverse`](Self::inverse) returns.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct RotationMove {
    /// The internal vtree node the rotation turns.
    pub pivot: VtreeIdx,
    /// Which way it turns.
    pub kind: RotationKind,
}

impl RotationMove {
    /// The move that undoes this one.
    ///
    /// A rotation leaves its pivot's index naming the same node, so the undo
    /// turns the same pivot the other way. Applying a move and then its
    /// inverse returns the vtree to its original shape.
    #[inline]
    pub fn inverse(self) -> RotationMove {
        RotationMove { pivot: self.pivot, kind: self.kind.inverse() }
    }
}

/// The levels a probed rotation sequence rebuilt, before it ran and as they
/// are now.
///
/// Handed to the decision a trial is waiting on — the closure of
/// [`Tdd::rotate_if`], a [`RotationObjective`](super::RotationObjective) or an
/// [`AcceptancePolicy`](super::AcceptancePolicy) — while the sequence is
/// applied but not yet kept. Reading it costs nothing: the preimages are the
/// trial's own, not copies.
///
/// `before` is the state ahead of the *first* move, not the move that last
/// touched the level. A pair of rotations whose first move grows the diagram
/// and whose second more than pays for it is therefore scored as the single
/// step it is meant to be.
pub struct RotationProbe<'a> {
    tdd: &'a Tdd,
    moves: &'a [RotationMove],
    changed: &'a [VtreeIdx],
    preimages: &'a [Transient<'a, TddLevel>],
}

impl std::fmt::Debug for RotationProbe<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RotationProbe")
            .field("moves", &self.moves)
            .field("changed", &self.changed)
            .finish_non_exhaustive()
    }
}

impl RotationProbe<'_> {
    /// The sequence that was applied, in the order it was applied.
    #[inline]
    pub fn moves(&self) -> &[RotationMove] {
        self.moves
    }

    /// The diagram with the sequence applied, on its rotated vtree.
    #[inline]
    pub fn diagram(&self) -> &Tdd {
        self.tdd
    }

    /// The levels the sequence rebuilt, each listed once, in the order its
    /// moves first reached them: the outer then the inner level of the first
    /// move, then whichever of the next move's two levels is new, and so on.
    #[inline]
    pub fn changed(&self) -> &[VtreeIdx] {
        self.changed
    }

    /// A changed level as it was before the first move, or `None` for a level
    /// the sequence did not rebuild.
    #[inline]
    pub fn before(&self, level: VtreeIdx) -> Option<&TddLevel> {
        self.position(level).map(|i| &*self.preimages[i])
    }

    /// A changed level as it is now, or `None` for a level the sequence did
    /// not rebuild.
    #[inline]
    pub fn after(&self, level: VtreeIdx) -> Option<&TddLevel> {
        self.position(level).map(|_| self.tdd.level(level))
    }

    /// Live pairs over the changed levels after the sequence, minus live pairs
    /// before it. Negative means the sequence shrank the diagram; no other
    /// level's pair count moved, so this is the whole difference.
    pub fn live_pairs_delta(&self) -> i64 {
        let mut delta = 0i64;
        for (i, &level) in self.changed.iter().enumerate() {
            delta += self.tdd.level(level).live_pairs() as i64;
            delta -= self.preimages[i].live_pairs() as i64;
        }
        delta
    }

    /// Where `level` sits in the parallel `changed` and `preimages` arrays.
    #[inline]
    fn position(&self, level: VtreeIdx) -> Option<usize> {
        self.changed.iter().position(|&t| t == level)
    }
}

/// What a caller of [`probe_moves`] adds to the shared protocol.
///
/// The search supplies admission, size bounds, the decision and an
/// accepted-rotation callback.
pub(super) trait ProbeRule {
    /// A last gate before the expensive restructure, read on the rotated vtree
    /// with the levels still untouched. `false` reverts the pointers and
    /// declines the probe. Consulted once per move of the sequence.
    fn admits(&mut self, _tdd: &Tdd, _info: &RotationInfo) -> bool {
        true
    }

    /// The pair bound the restructure bails past, given the caller's default.
    fn bound(&mut self, _tdd: &Tdd, _info: &RotationInfo, default_bound: usize) -> usize {
        default_bound
    }

    /// Keep this sequence? `probe` shows the levels it rebuilt, and `info`
    /// is the last move's rotation information.
    fn keeps(&mut self, probe: &RotationProbe<'_>, info: &RotationInfo) -> bool;

    /// Run after the rotation is committed. Its `Err` propagates with the
    /// rotation kept: what it leaves unfinished is an optimization, never the
    /// diagram's correctness.
    fn on_accept(
        &mut self,
        _eng: &Engine,
        _tdd: &mut Tdd,
        _info: &RotationInfo,
    ) -> Result<(), OperationError> {
        Ok(())
    }
}

impl Tdd {
    /// Apply `moves` in order, show the levels they changed to `accept`, and
    /// keep the sequence only if it says so.
    ///
    /// This is the trial every rotation search runs on. One move is a single
    /// rotation; two and three are the connected pair and triple neighborhoods
    /// a [rotation search](Self::rotation_search) probes.
    /// The sequence is scored as one step: [`RotationProbe::before`] is the
    /// state ahead of the first move, so a pair whose first half grows the
    /// diagram and whose second half more than pays for it is seen as the
    /// improvement it is.
    ///
    /// Returns whether the moves were kept. On a decline the diagram and its
    /// vtree are exactly what they were. A move that does not apply at its
    /// pivot, a level that has been summed out, or a rebuild that would exceed
    /// `bound` input pairs abandons the sequence and returns `Ok(false)`
    /// without calling `accept`; `usize::MAX` is no bound, and a probe under
    /// it may ask for more memory than the host has.
    ///
    /// The diagram must be canonical, which is what makes a rebuilt level
    /// interchangeable with the one it replaced. A diagram with summed-out
    /// levels is left alone rather than rebuilt.
    ///
    /// # Errors
    ///
    /// [`OperationError::OverBudget`] from a rebuild that does not fit the
    /// armed byte budget, with the diagram at its pre-probe state.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{and, Tdd, Vtree};
    /// use tididi::restructure::search::RotationMove;
    /// use tididi::vtree::RotationKind;
    /// let vtree = Arc::new(Vtree::balanced(4));
    /// let mut f = and(Tdd::clause(&vtree, [1, 2])?, Tdd::clause(&vtree, [3, 4])?)?;
    /// f.minimize()?;
    /// let before = f.pair_count();
    /// let turn = RotationMove { pivot: vtree.root(), kind: RotationKind::Left };
    /// // Keep the rotation only where it does not cost storage.
    /// let kept = f.rotate_if(&[turn], 1 << 20, |probe| probe.live_pairs_delta() <= 0)?;
    /// if !kept {
    ///     assert_eq!(f.pair_count(), before);
    /// }
    /// # Ok::<(), tididi::OperationError>(())
    /// ```
    pub fn rotate_if<F>(
        &mut self,
        moves: &[RotationMove],
        bound: usize,
        accept: F,
    ) -> Result<bool, OperationError>
    where
        F: FnOnce(&RotationProbe<'_>) -> bool,
    {
        let context = Arc::clone(self.context());
        context.run(|eng| {
            let _op = eng.limits().begin_operation();
            let mut scratch = eng.restructure().checkout(eng.limits());
            let mut rule = Closure { accept: Some(accept) };
            probe_moves(eng, self, moves, &mut rule, &mut scratch, bound)
        })
    }
}

/// The rule a [`Tdd::rotate_if`] call probes under: the caller's closure,
/// consulted once.
struct Closure<F> {
    accept: Option<F>,
}

impl<F: FnOnce(&RotationProbe<'_>) -> bool> ProbeRule for Closure<F> {
    fn keeps(&mut self, probe: &RotationProbe<'_>, _info: &RotationInfo) -> bool {
        (self.accept.take().expect("a probe is scored once"))(probe)
    }
}

/// Probe the `kind` rotation at pivot `v` and keep it iff `rule` scores it an
/// improvement. Returns whether the rotation was kept; on a decline the diagram
/// is restored bit-for-bit, vtree included.
///
/// Rotation locality restricts rebuilding, scoring and rollback to two levels.
pub(super) fn probe<R: ProbeRule>(
    eng: &Engine,
    tdd: &mut Tdd,
    v: VtreeIdx,
    kind: RotationKind,
    rule: &mut R,
    scratch: &mut RestructureScratch,
    default_bound: usize,
) -> Result<bool, OperationError> {
    probe_moves(eng, tdd, &[RotationMove { pivot: v, kind }], rule, scratch, default_bound)
}

/// Apply `moves` in order, score the levels they rebuilt as one step, and keep
/// the sequence iff `rule` says so. Returns whether it was kept; on a decline
/// the diagram is restored bit-for-bit, vtree included.
///
/// A move that does not apply at its pivot, a level that is marginal, a gate
/// the rule closes or a rebuild past its bound abandons the sequence without
/// scoring anything.
///
/// # Errors
///
/// [`OperationError::OverBudget`] from a refused rebuild, with the diagram at
/// its pre-probe state, or whatever [`ProbeRule::on_accept`] returns, with the
/// sequence kept.
pub(super) fn probe_moves<R: ProbeRule>(
    eng: &Engine,
    tdd: &mut Tdd,
    moves: &[RotationMove],
    rule: &mut R,
    scratch: &mut RestructureScratch,
    default_bound: usize,
) -> Result<bool, OperationError> {
    if moves.is_empty() {
        return Ok(false);
    }
    let mut trial = RotationTrial::new(tdd, eng.limits());
    for mv in moves {
        let Some(info) = trial.rotate(*mv) else { return Ok(false) };
        // The two rebuilt levels need explicit pairs; marginal grandchildren are
        // allowed because the restructure preserves their contribution multiset.
        if trial.tdd.levels[info.v_idx.idx()].is_marginal()
            || trial.tdd.levels[info.w_idx.idx()].is_marginal()
            || !rule.admits(trial.tdd, &info)
        {
            return Ok(false);
        }
        let bound = rule.bound(trial.tdd, &info, default_bound);
        let rebuilt =
            restructure_inner_search(eng.limits(), trial.tdd, &info, mv.kind, scratch, bound)?;
        let Some(old) = rebuilt else { return Ok(false) };
        trial.record(&info, old);
    }
    let info = trial.last_info();
    #[cfg(debug_assertions)]
    crate::test_helpers::check::debug_assert_rotation_locality(eng, trial.tdd, info.w_idx);
    let keep = rule.keeps(
        &RotationProbe {
            tdd: trial.tdd,
            moves,
            changed: &trial.changed,
            preimages: &trial.preimages,
        },
        &info,
    );
    if keep {
        trial.commit();
        rule.on_accept(eng, tdd, &info)?;
        Ok(true)
    } else {
        Ok(false)
    }
}

/// Own a sequence's preimage until its topology and levels are committed together.
struct RotationTrial<'a> {
    tdd: &'a mut Tdd,
    /// Where the levels the trial builds and then drops give their bytes back.
    lim: &'a Limits,
    /// One entry per applied move, in the order they were applied.
    pending: SmallVec<[PendingTopo; 3]>,
    /// The levels the sequence rebuilt, each recorded the first time a move
    /// reached it, parallel to `preimages`.
    changed: SmallVec<[VtreeIdx; 2]>,
    preimages: SmallVec<[Transient<'a, TddLevel>; 2]>,
    old_output: TddNodeId,
    shared_tree: Option<Arc<Vtree>>,
    old_dirty: Option<Dirty>,
    /// Set by [`commit`](RotationTrial::commit), so the rollback in `Drop` knows
    /// there is nothing left to roll back.
    committed: bool,
}

impl<'a> RotationTrial<'a> {
    /// Detach a shared vtree and take the state a rollback restores.
    fn new(tdd: &'a mut Tdd, lim: &'a Limits) -> Self {
        let old_output = tdd.output;
        let shared_tree = (Arc::strong_count(&tdd.vtree) > 1 || Arc::weak_count(&tdd.vtree) > 0)
            .then(|| Arc::clone(&tdd.vtree));
        let old_dirty = Some(std::mem::take(&mut tdd.dirty));
        RotationTrial {
            tdd,
            lim,
            pending: SmallVec::new(),
            changed: SmallVec::new(),
            preimages: SmallVec::new(),
            old_output,
            shared_tree,
            old_dirty,
            committed: false,
        }
    }

    /// Rotate the pointers for one move, retaining what a rollback needs.
    fn rotate(&mut self, mv: RotationMove) -> Option<RotationInfo> {
        let pending = rotate_pointers(Arc::make_mut(&mut self.tdd.vtree), mv.pivot, mv.kind)?;
        let info = pending.info();
        self.pending.push(pending);
        Some(info)
    }

    /// Take the levels one move replaced. A level rebuilt a second time keeps
    /// its earliest preimage, so the whole sequence is scored and reverted
    /// against the state it started from.
    fn record(&mut self, info: &RotationInfo, old: (TddLevel, TddLevel)) {
        for (level, preimage) in [(info.v_idx, old.0), (info.w_idx, old.1)] {
            if self.changed.contains(&level) {
                self.lim.discard(preimage);
            } else {
                self.changed.push(level);
                self.preimages.push(Transient::new(self.lim, preimage));
            }
        }
    }

    /// The last applied move's rotation information.
    fn last_info(&self) -> RotationInfo {
        self.pending.last().expect("a scored trial has applied a move").info()
    }

    /// Repair topology and release the preimage before any accepted-rotation callback.
    fn commit(mut self) {
        self.committed = true;
        // Dropping the preimages hands their charge back, and the rebuild
        // charged the levels that replaced them, so the operation's in-flight
        // total ends up carrying the difference rather than both.
        self.preimages.clear();
        self.changed.clear();
        for pending in self.pending.drain(..) {
            pending.commit(Arc::make_mut(&mut self.tdd.vtree));
        }
        // Both exits end with the pre-probe obligations still present: `Drop`
        // restores them wholesale on reject, and the accept path puts them back
        // underneath what the rotation itself recorded. Clearing here instead
        // dropped both, which left a later `minimize` skipping levels it still
        // owed work on.
        let carried = self.old_dirty.take().expect("a trial takes the worklists when it is created");
        self.tdd.dirty.merge_under(carried);
    }
}

impl Drop for RotationTrial<'_> {
    fn drop(&mut self) {
        if self.committed {
            return;
        }
        while let Some(pending) = self.pending.pop() {
            pending.revert(Arc::make_mut(&mut self.tdd.vtree));
        }
        for (level, preimage) in self.changed.drain(..).zip(self.preimages.drain(..)) {
            // The rebuilt level is the one being dropped here, so its charge is
            // what the trial hands back.
            let rebuilt = std::mem::replace(&mut self.tdd.levels[level.idx()], preimage.keep());
            self.lim.discard(rebuilt);
        }
        self.tdd.output = self.old_output;
        if let Some(dirty) = self.old_dirty.take() {
            self.tdd.dirty = dirty;
        }
        if let Some(tree) = self.shared_tree.take() {
            self.tdd.vtree = tree;
        }
    }
}
