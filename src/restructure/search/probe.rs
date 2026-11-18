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
use crate::restructure::relevel::{RotationStep, rebuild_levels};
use crate::restructure::scratch::RestructureScratch;
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
    preimages: &'a [Preimage<'a>],
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

    /// The diagram with the sequence applied, on its rotated vtree. Not
    /// public: on a projection that vtree keeps the link of the shape before
    /// the sequence until the search seats the diagram on its lifted home, so
    /// a clone taken here would be a projection whose link misdescribes it.
    #[inline]
    pub(crate) fn diagram(&self) -> &Tdd {
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
        self.position(level).map(|i| &*self.preimages[i].level)
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
            delta -= self.preimages[i].level.live_pairs() as i64;
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
/// The search supplies admission, size bounds and the decision.
pub(in crate::restructure) trait ProbeRule {
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
    /// interchangeable with the one it replaced.
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
        context.run(|eng| rotate_if_on(eng, self, moves, bound, accept))
    }
}

/// Run a standalone trial using the caller's scratch and resource limits.
pub(super) fn rotate_if_on(
    eng: &Engine,
    tdd: &mut Tdd,
    moves: &[RotationMove],
    bound: usize,
    accept: impl FnOnce(&RotationProbe<'_>) -> bool,
) -> Result<bool, OperationError> {
    let _op = eng.limits().enter()?;
    let mut scratch = eng.scratch.restructure.checkout(eng);
    let mut rule = Closure { accept: Some(accept) };
    super::SearchTree::new(eng, tdd)?.probe_moves(eng, moves, &mut rule, &mut scratch, bound)
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

/// The rule that keeps whatever it is shown: the rewind and the multistart
/// kicks probe under it.
pub(super) struct Forced;

impl ProbeRule for Forced {
    fn keeps(&mut self, _probe: &RotationProbe<'_>, _info: &RotationInfo) -> bool {
        true
    }
}

/// Apply `moves` in order, score the levels they rebuilt as one step, and keep
/// the sequence iff `rule` says so. Returns whether it was kept; on a decline
/// the diagram's levels and its vtree's shape are restored bit-for-bit. The
/// vtree must be the diagram's own: [`SearchTree`](super::SearchTree) detaches
/// a shared one and restores its identity when nothing is kept.
///
/// A move that does not apply at its pivot, a level that is marginal, a gate
/// the rule closes or a rebuild past its bound abandons the sequence without
/// scoring anything.
///
/// # Errors
///
/// [`OperationError::OverBudget`] from a refused rebuild, with the diagram at
/// its pre-probe state.
pub(in crate::restructure) fn probe_moves<R: ProbeRule>(
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
    let mut trial = RotationTrial::new(tdd, eng.limits(), moves.len())?;
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
        let marginal = trial.tdd.has_marginal_level();
        let step = RotationStep { info: &info, dir: mv.kind, crossed: false, max_pairs: bound };
        let rebuilt = rebuild_levels(eng.limits(), trial.tdd, step, marginal, scratch)?;
        let Some(old) = rebuilt else { return Ok(false) };
        trial.record(&info, old);
    }
    let info = trial.last_info();
    #[cfg(any(test, feature = "testing"))]
    crate::test_helpers::check::assert_rotation_locality(&Engine::new(), trial.tdd, info.w_idx);
    let keep = rule.keeps(
        &RotationProbe {
            tdd: trial.tdd,
            moves,
            changed: &trial.changed,
            preimages: &trial.pre.as_ref().expect("a trial holds its preimages until it ends").levels,
        },
        &info,
    );
    if keep {
        trial.commit();
    }
    Ok(keep)
}

/// The live pairs of the two levels a rotation rebuilds.
pub(super) fn pivot_pairs(tdd: &Tdd, info: &RotationInfo) -> usize {
    tdd.levels[info.v_idx.idx()].live_pairs() + tdd.levels[info.w_idx.idx()].live_pairs()
}

/// A diagram's output, certification, closed-levels state, and pending
/// reduction obligations.
struct DiagramHeader {
    output: TddNodeId,
    canonical: bool,
    closed: bool,
    dirty: Dirty,
}

impl DiagramHeader {
    /// The header of `tdd`, with `dirty` for its pending obligations.
    fn new(tdd: &Tdd, dirty: Dirty) -> Self {
        DiagramHeader {
            output: tdd.output,
            canonical: tdd.levels.is_canonical(tdd.output),
            closed: tdd.levels.is_closed(),
            dirty,
        }
    }

    /// Put the header back on `tdd`.
    fn restore(self, tdd: &mut Tdd) {
        tdd.output = self.output;
        tdd.dirty = self.dirty;
        if self.canonical {
            tdd.levels.certify(self.output);
        }
        if self.closed {
            tdd.reinstate_closed_levels();
        }
    }
}

/// A level a rotation replaced at `node` of diagram `diagram`.
pub(super) struct Preimage<'a> {
    pub(super) diagram: usize,
    pub(super) node: VtreeIdx,
    level: Transient<'a, TddLevel>,
}

/// The headers of some diagrams and the levels rotations replaced in them,
/// for the rotations to be kept or put back together. A diagram is named by
/// its index in the slice handed back to [`keep`](Self::keep) or
/// [`put_back`](Self::put_back). Room for every header and level is reserved
/// up front, so filing one never fails.
pub(super) struct Preimages<'a> {
    lim: &'a Limits,
    headers: Transient<'a, Vec<(usize, DiagramHeader)>>,
    pub(super) levels: Transient<'a, Vec<Preimage<'a>>>,
}

impl<'a> Preimages<'a> {
    pub(super) fn new(lim: &'a Limits, headers: usize, levels: usize) -> Result<Self, OperationError> {
        let mut pre = Preimages {
            lim,
            headers: Transient::new(lim, Vec::new()),
            levels: Transient::new(lim, Vec::new()),
        };
        lim.reserve_exact(&mut pre.headers, headers)?;
        lim.reserve_exact(&mut pre.levels, levels)?;
        Ok(pre)
    }

    /// File diagram `i`'s header, its pending obligations moved out of
    /// `tdd`, or copied if `copy`.
    pub(super) fn file_header(&mut self, i: usize, tdd: &mut Tdd, copy: bool) {
        let dirty = if copy {
            tdd.dirty.clone()
        } else {
            std::mem::take(&mut tdd.dirty)
        };
        self.headers.push((i, DiagramHeader::new(tdd, dirty)));
    }

    /// File `level`, the level a rotation replaced at `node` of diagram `i`.
    pub(super) fn file(&mut self, i: usize, node: VtreeIdx, level: TddLevel) {
        let level = Transient::new(self.lim, level);
        self.levels.push(Preimage { diagram: i, node, level });
    }

    /// Keep the rotations: each diagram with a header has its `changed`
    /// levels closed and its pending obligations put back under what the
    /// rotations recorded. Returns the replaced levels, whose charge is handed
    /// back as they drop.
    pub(super) fn keep(
        self,
        diagrams: &mut [&mut Tdd],
        changed: &[VtreeIdx],
    ) -> Transient<'a, Vec<Preimage<'a>>> {
        let Preimages { mut headers, levels, .. } = self;
        for (i, header) in headers.drain(..) {
            // The rebuilt levels are built stored; the kept rotations are the
            // operation's end, which closes them.
            diagrams[i].close_changed_levels(changed);
            diagrams[i].dirty.merge_under_kept(header.dirty, changed);
        }
        levels
    }

    /// [`keep`](Self::keep) for diagrams that each rebuilt two levels, filed
    /// in turn after the diagram's header.
    pub(super) fn keep_rebuilt(self, diagrams: &mut [&mut Tdd]) -> Transient<'a, Vec<Preimage<'a>>> {
        let Preimages { mut headers, levels, .. } = self;
        debug_assert_eq!(levels.len(), 2 * headers.len());
        for ((i, header), pair) in headers.drain(..).zip(levels.chunks_exact(2)) {
            let changed = [pair[0].node, pair[1].node];
            diagrams[i].close_changed_levels(&changed);
            diagrams[i].dirty.merge_under_kept(header.dirty, &changed);
        }
        levels
    }

    /// Put every filed level and header back, the levels that replaced them
    /// discarded.
    pub(super) fn put_back(self, diagrams: &mut [&mut Tdd]) {
        self.put_back_then(diagrams, |_| {});
    }

    /// [`put_back`](Self::put_back), with `then` run on the diagrams between
    /// the levels and the headers.
    pub(super) fn put_back_then(self, diagrams: &mut [&mut Tdd], then: impl FnOnce(&mut [&mut Tdd])) {
        let Preimages { lim, mut headers, mut levels } = self;
        for p in levels.drain(..) {
            let rebuilt = std::mem::replace(&mut diagrams[p.diagram].levels[p.node.idx()], p.level.keep());
            lim.discard(rebuilt);
        }
        then(diagrams);
        for (i, header) in headers.drain(..) {
            header.restore(diagrams[i]);
        }
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
    /// reached it, in the order of their preimages in `pre`.
    changed: SmallVec<[VtreeIdx; 2]>,
    /// Taken by [`commit`](RotationTrial::commit); what `Drop` puts back.
    pre: Option<Preimages<'a>>,
}

impl<'a> RotationTrial<'a> {
    /// Take the state a rollback restores, with room for the levels of
    /// `moves` moves.
    fn new(tdd: &'a mut Tdd, lim: &'a Limits, moves: usize) -> Result<Self, OperationError> {
        let mut pre = Preimages::new(lim, 1, moves.saturating_mul(2))?;
        pre.file_header(0, tdd, false);
        Ok(RotationTrial {
            tdd,
            lim,
            pending: SmallVec::new(),
            changed: SmallVec::new(),
            pre: Some(pre),
        })
    }

    /// Rotate the pointers for one move, retaining what a rollback needs.
    fn rotate(&mut self, mv: RotationMove) -> Option<RotationInfo> {
        self.tdd.levels.forget();
        let pending = rotate_pointers(Vtree::make_mut(&mut self.tdd.vtree), mv.pivot, mv.kind)?;
        let info = pending.info();
        self.pending.push(pending);
        Some(info)
    }

    /// Take the levels one move replaced. A level rebuilt a second time keeps
    /// its earliest preimage, so the whole sequence is scored and reverted
    /// against the state it started from.
    fn record(&mut self, info: &RotationInfo, old: (TddLevel, TddLevel)) {
        let pre = self.pre.as_mut().expect("a trial holds its preimages until it ends");
        for (level, preimage) in [(info.v_idx, old.0), (info.w_idx, old.1)] {
            if self.changed.contains(&level) {
                self.lim.discard(preimage);
            } else {
                self.changed.push(level);
                pre.file(0, level, preimage);
            }
        }
    }

    /// The last applied move's rotation information.
    fn last_info(&self) -> RotationInfo {
        self.pending.last().expect("a scored trial has applied a move").info()
    }

    /// Repair topology and release the preimage.
    fn commit(mut self) {
        let pre = self.pre.take().expect("a trial holds its preimages until it ends");
        // Both exits end with the pre-probe obligations still present: `Drop`
        // restores them wholesale on reject, and the accept path puts them back
        // underneath what the rotation itself recorded. Clearing here instead
        // dropped both, which left a later `minimize` skipping levels it still
        // owed work on. The pair fusion records stand where no move rebuilt
        // or pushed a level.
        drop(pre.keep(std::slice::from_mut(&mut self.tdd), &self.changed));
        for pending in self.pending.drain(..) {
            pending.commit_in_probe(Vtree::make_mut(&mut self.tdd.vtree));
        }
    }
}

impl Drop for RotationTrial<'_> {
    fn drop(&mut self) {
        let Some(pre) = self.pre.take() else { return };
        while let Some(pending) = self.pending.pop() {
            pending.revert(Vtree::make_mut(&mut self.tdd.vtree));
        }
        pre.put_back(std::slice::from_mut(&mut self.tdd));
    }
}
