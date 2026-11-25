//! Move a diagram onto a given vtree over the same variables.
//!
//! The move is a sequence of the two local edits that keep a diagram
//! canonical: a rotation, which rebuilds the two levels it turns, and a
//! mirror, which reads one level's pairs the other way round. Neither changes
//! any other level, because a level's nodes are the classes of assignments to
//! its node's variables and only a rotation changes the variables under a
//! node: the node it demotes.
//!
//! The sequence is planned top-down. At a pair of corresponding nodes, the
//! diagram's node is rotated until one child holds exactly the variables of
//! the target node's left child, and the two pairs of children are then
//! matched on their own. A subtree whose variables all fall on one side is
//! never rotated, so a vtree that differs from the target only above some
//! shared subtrees is moved without touching the levels inside them. Mirrors are free in the plan: they cost one pass over a level and never
//! change a size. A last pass mirrors whatever orientation still differs and
//! moves every level to its index in the target's node array.

use std::sync::Arc;

use crate::Engine;
use crate::diagram::{Tdd, TddLevel, TddNodeId};
use crate::limits::{OperationError, Transient};
use crate::restructure::scratch::RestructureScratch;
use crate::restructure::search::{ProbeRule, RotationMove, RotationProbe, probe_moves};
use crate::vtree::rotate::RotationInfo;
use crate::vtree::{RotationKind, VarId, Vtree, VtreeIdx, VtreeNode};

use super::RestructureError;

/// How [`Engine::transport`] or [`Tdd::restructure_to`] moved a diagram's
/// vtree: the rotations and mirrors of the tree, not which levels were
/// rewritten.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct TransportStats {
    /// Rotations applied, each rebuilding two levels.
    pub rotations: usize,
    /// Levels whose pairs were read the other way round.
    pub mirrors: usize,
    /// The largest pair count the diagram reached on the way, the input's and
    /// the output's included.
    pub peak_pairs: usize,
}

impl Tdd {
    /// This diagram on `target`, a vtree over the same variables.
    ///
    /// Rotations and mirrors move the diagram from its vtree to `target`'s
    /// shape, and its levels are then reseated on `target` itself, so the
    /// result shares `target`'s allocation and can be combined with diagrams
    /// built on it. The function is unchanged, and a canonical diagram stays
    /// canonical. When `target` is on a [home](Vtree::home), the variables
    /// compared are those of that home and of the diagram's home (its own
    /// vtree's when it has none), as a function on a home is over every
    /// home variable; the diagram, unless it has a summed-out level, is
    /// minimized first and the variables it does not depend on leave: it
    /// ends on the projection of `target`'s home on the variables left,
    /// which is `target` when they are its leaves, and a constant ends on no
    /// leaf of that home. A variable left that is not a leaf of `target` is
    /// refused.
    ///
    /// A rotation rebuilds the two levels it turns from the pairs of the
    /// outer level expanded through the inner one; a mirror reads one level's
    /// pairs the other way round; no other level is touched. The largest
    /// subtrees the two vtrees share are kept whole, so the cost falls on the
    /// levels above them, and it depends on the diagram's size on each vtree
    /// the rotations pass through, which neither end bounds. A rebuild is refused when its triples (one per old outer
    /// pair and old inner pair) or the new inner level's nodes plus pairs
    /// reach `bound`; `usize::MAX` is no bound.
    ///
    /// Diagrams with summed-out levels move as long as no rotation turns a
    /// summed-out level. Weighted diagrams are refused.
    ///
    /// A diagram without a summed-out level that moves is minimized first, and
    /// the result is certified ([`is_canonical`](Self::is_canonical)).
    ///
    /// # Errors
    ///
    /// [`RestructureError::Variables`] when the two vtrees (the two homes,
    /// as above) do not have the same variables, or `target`, on a home,
    /// lacks a variable the diagram depends on; [`RestructureError::Weighted`] for a diagram with a
    /// weight table; [`RestructureError::Bound`] when a rotation would expand
    /// more than `bound` pairs, and [`RestructureError::Operation`] with
    /// [`OperationError::MarginalLevel`] when it would turn a summed-out
    /// level or either vtree has a region, or with a refused allocation or an
    /// armed stop. After an error a diagram on a root vtree denotes the same
    /// function on its own copy of a vtree between the two. A diagram on a projection is moved as a copy on
    /// private trees over its own leaves ([`Engine::transport`]'s rotation
    /// path), so an error leaves it as it was.
    ///
    /// ```
    /// use tididi::vtree::Home;
    /// use std::sync::Arc;
    /// use tididi::{Tdd, Vtree};
    /// use tididi::vtree::VarId;
    ///
    /// let order = [VarId(1), VarId(2), VarId(3), VarId(4)];
    /// let stick = Home::new(Vtree::linear_from_order(&order)?);
    /// let f = Tdd::clause(&stick, [1, -3])? & Tdd::clause(&stick, [2, 4])?;
    /// let before = f.model_count()?;
    ///
    /// let balanced = Home::new(Vtree::balanced_over(&[VarId(3), VarId(1), VarId(4), VarId(2)])?);
    /// let mut g = f.clone();
    /// let stats = g.restructure_to(&balanced, usize::MAX)?;
    /// assert!(Arc::ptr_eq(g.vtree(), &balanced));
    /// assert_eq!(g.model_count()?, before);
    /// assert!(stats.rotations > 0);
    /// # tididi::test_helpers::assert_canonical(&g);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn restructure_to(&mut self, target: &Arc<Vtree>, bound: usize) -> Result<TransportStats, RestructureError> {
        let context = Arc::clone(target.context());
        context.run(|eng| eng.restructure_to(self, target, bound))
    }
}

impl Engine {
    /// [`Tdd::restructure_to`] under this batch's scratch and resource limits.
    ///
    /// # Errors
    ///
    /// As [`Tdd::restructure_to`].
    pub fn restructure_to(
        &self,
        tdd: &mut Tdd,
        target: &Arc<Vtree>,
        bound: usize,
    ) -> Result<TransportStats, RestructureError> {
        let _op = self.limits().enter()?;
        match target.home() {
            Some(home) => {
                let source = tdd.vtree().home();
                same_variables(source.as_ref().map_or(tdd.vtree(), |h| &**h), &home)?;
            }
            None => same_variables(tdd.vtree(), target)?,
        }
        if let Some(region) = tdd.vtree().regions().chain(target.regions()).next() {
            return Err(OperationError::MarginalLevel(region).into());
        }
        if tdd.holds_weights() {
            return Err(RestructureError::Weighted);
        }
        let narrowed;
        let mut target = target;
        if let Some(home) = target.home()
            && !Arc::ptr_eq(tdd.vtree(), target)
        {
            if !tdd.has_marginal_level() {
                self.reduce(tdd, crate::reduce::ReductionPlan::default())?;
                if tdd.vtree().num_nodes() == 0 {
                    if !tdd.vtree().home().is_some_and(|h| Arc::ptr_eq(&h, &home)) {
                        *tdd = crate::build::constant_on(self, &Vtree::empty_projection(&home), !tdd.is_zero())?;
                    }
                    return Ok(TransportStats::default());
                }
            }
            let own = tdd.vtree();
            if let Some((_, variable)) = own.leaf_bottomup().find(|&(_, var)| target.leaf_of(var).is_none()) {
                return Err(RestructureError::Variables { variable });
            }
            if own.num_leaves() != target.num_leaves() {
                narrowed = Vtree::projection(&home, own.leaf_bottomup().map(|(_, var)| var))
                    .expect("the target's home has every variable of the diagram");
                target = &narrowed;
            }
        }
        if Arc::ptr_eq(tdd.vtree(), target) {
            return Ok(TransportStats { peak_pairs: tdd.pair_count(), ..TransportStats::default() });
        }
        if tdd.is_zero() {
            *tdd = crate::build::constant_zero(self, target);
            return Ok(TransportStats::default());
        }
        if tdd.vtree.link_home().is_some() {
            return self.restructure_projection(tdd, target, bound);
        }
        self.move_to(tdd, target, bound)
    }

    /// [`restructure_to`](Self::restructure_to) of a diagram on a
    /// projection: a copy on its own tree with each leaf named by its
    /// target's place in `target`'s leaf order, moved onto `target`'s tree
    /// named the same way.
    #[cold]
    #[inline(never)]
    fn restructure_projection(&self, tdd: &mut Tdd, target: &Arc<Vtree>, bound: usize) -> Result<TransportStats, RestructureError> {
        let k = target.num_leaves();
        let mut name = Transient::new(self.limits(), Vec::new());
        self.limits().try_resize(&mut name, target.num_nodes(), VarId(0))?;
        for (i, (leaf, _)) in target.leaf_bottomup().enumerate() {
            name[leaf.idx()] = VarId(i as u32 + 1);
        }
        let own = tdd.vtree();
        let at_target = |leaf: VtreeIdx| name[target.leaf_of(own.leaf_var(leaf)).expect("same variables").idx()];
        let mut copy = tdd.try_clone_on(self)?;
        copy.vtree = Arc::new(own.local_twin(at_target, k));
        let twin = Arc::new(target.local_twin(|leaf| name[leaf.idx()], k));
        let stats = self.move_to(&mut copy, &twin, bound)?;
        // The twin is `target` with its leaves renamed: the node indices agree.
        copy.vtree = Arc::clone(target);
        *tdd = copy;
        Ok(stats)
    }

    fn move_to(&self, tdd: &mut Tdd, target: &Arc<Vtree>, bound: usize) -> Result<TransportStats, RestructureError> {
        // A rotation's rebuilt levels are interchangeable with the ones they
        // replace only on a canonical diagram. With summed-out levels the
        // rotation keeps every contribution instead, and a minimize would
        // merge them, so those are moved as they are.
        let structural = !tdd.has_marginal_level();
        if structural {
            self.reduce_plain(tdd, crate::reduce::ReductionPlan::default())?;
        }
        self.rearrange(tdd, target, bound, structural)
    }

    /// `tdd`, on a root vtree over `target`'s variables and minimized when
    /// `structural`, rotated and mirrored to `target`'s shape and seated on
    /// it.
    pub(super) fn rearrange(
        &self,
        tdd: &mut Tdd,
        target: &Arc<Vtree>,
        bound: usize,
        structural: bool,
    ) -> Result<TransportStats, RestructureError> {
        debug_assert!(tdd.vtree.link_home().is_none(), "a projection moves on its private twin");
        let mut mover = Mover {
            eng: self,
            tdd,
            target: target.as_ref(),
            bound,
            scratch: self.scratch.restructure.checkout(self),
            stats: TransportStats::default(),
            pairs: 0,
        };
        mover.pairs = mover.tdd.pair_count();
        mover.stats.peak_pairs = mover.pairs;
        Vtree::make_mut(&mut mover.tdd.vtree);
        let mut work = Transient::new(self.limits(), Vec::new());
        self.limits().try_push(&mut work, (mover.tdd.vtree.root(), target.root()))?;
        while let Some((a, b)) = work.pop() {
            self.limits().check_stop()?;
            mover.arrange(a, b, &mut work)?;
        }
        let Mover { mut stats, .. } = mover;
        if structural {
            // A rotation of a minimized diagram leaves it minimized (rotation
            // locality, `relevel`), so the outer levels the rotations queued
            // for contraction owe nothing and no level is loose.
            tdd.dirty = crate::diagram::Dirty::default();
            tdd.dirty.set_loose(Some(Vec::new()));
        }
        stats.mirrors += reseat(self, tdd, target, structural)?;
        self.limits().check_stop()?;
        Ok(stats)
    }
}

/// Refuse two vtrees whose leaves carry different variables.
fn same_variables(source: &Vtree, target: &Vtree) -> Result<(), RestructureError> {
    let mut seen = 0u32;
    for (_, var) in source.leaf_bottomup() {
        if target.leaf_of(var).is_none() {
            return Err(RestructureError::Variables { variable: var });
        }
        seen += 1;
    }
    if seen != target.num_leaves() {
        let missing = target.leaf_bottomup().map(|(_, var)| var).find(|&var| source.leaf_of(var).is_none());
        return Err(RestructureError::Variables { variable: missing.unwrap_or(VarId(0)) });
    }
    Ok(())
}

/// The move in progress: the diagram on its own vtree, and what it cost.
struct Mover<'a, 'e> {
    eng: &'e Engine,
    tdd: &'a mut Tdd,
    target: &'a Vtree,
    bound: usize,
    scratch: crate::execution::pool::PoolGuard<'e, RestructureScratch>,
    stats: TransportStats,
    /// The diagram's live pairs now.
    pairs: usize,
}

impl Mover<'_, '_> {
    /// Make the diagram's node `a` split its variables as the target's node
    /// `b` does, then queue the two pairs of children.
    fn arrange(&mut self, a: VtreeIdx, b: VtreeIdx, work: &mut Vec<(VtreeIdx, VtreeIdx)>) -> Result<(), RestructureError> {
        if self.target.node(b).is_leaf() {
            return Ok(());
        }
        let (left, right) = self.target.children(b);
        let wanted = self.partition(a, left)?;
        let (l, r) = self.tdd.vtree.children(a);
        let other = if l == wanted { r } else { l };
        let lim = self.eng.limits();
        lim.try_push(work, (wanted, left))?;
        lim.try_push(work, (other, right))?;
        Ok(())
    }

    /// Group the variables below `target` into one child of `root`. Tasks
    /// replace recursion on a long vtree; counts change only at the two
    /// levels a rotation rebuilds.
    fn partition(&mut self, root: VtreeIdx, target: VtreeIdx) -> Result<VtreeIdx, RestructureError> {
        #[derive(Clone, Copy)]
        enum Task { Split(VtreeIdx), Lift(VtreeIdx, VtreeIdx), Join(VtreeIdx, VtreeIdx), Both(VtreeIdx, VtreeIdx) }
        let lim = self.eng.limits();
        let mut counts = Transient::new(lim, Vec::<(u32, u32)>::new());
        lim.try_resize(&mut counts, self.tdd.vtree.num_nodes(), (0, 0))?;
        let mut order = Transient::new(lim, Vec::new());
        for t in self.tdd.vtree.subtree(root) {
            lim.try_push(&mut order, t)?;
        }
        // The target's node over each node's leaves, and each target node's
        // leaf count: a pure child moves across only where the two together
        // are a node of the target.
        let mut tl = Transient::new(lim, Vec::<VtreeIdx>::new());
        lim.try_resize(&mut tl, self.tdd.vtree.num_nodes(), VtreeIdx(0))?;
        let mut tsize = Transient::new(lim, Vec::<u32>::new());
        lim.try_resize(&mut tsize, self.target.num_nodes(), 0)?;
        for t in self.target.bottomup() {
            tsize[t.idx()] = match self.target.node(t) {
                VtreeNode::Leaf { var: crate::vtree::REGION_VAR, .. } => crate::vtree::region_unsupported("restructuring"),
                VtreeNode::Leaf { .. } => 1,
                VtreeNode::Internal { left, right, .. } => tsize[left.idx()] + tsize[right.idx()],
            };
        }
        for &t in order.iter().rev() {
            lim.check_stop()?;
            counts[t.idx()] = match self.tdd.vtree.node(t) {
                VtreeNode::Leaf { var: crate::vtree::REGION_VAR, .. } => crate::vtree::region_unsupported("restructuring"),
                VtreeNode::Leaf { var, .. } => {
                    tl[t.idx()] = self.target.leaf_of(*var).expect("same variables");
                    (u32::from(self.target.in_subtree(tl[t.idx()], target)), 1)
                }
                VtreeNode::Internal { left, right, .. } => {
                    tl[t.idx()] = self.target.lca(tl[left.idx()], tl[right.idx()]);
                    let (l, r) = (counts[left.idx()], counts[right.idx()]);
                    (l.0 + r.0, l.1 + r.1)
                }
            };
        }
        let mut tasks = Transient::new(lim, Vec::new());
        lim.try_push(&mut tasks, Task::Split(root))?;
        let wanted_child = |vtree: &Vtree, counts: &[(u32, u32)], v: VtreeIdx| {
            let (l, r) = vtree.children(v);
            if counts[l.idx()].0 == counts[l.idx()].1 { l } else { r }
        };
        while let Some(task) = tasks.pop() {
            lim.check_stop()?;
            let (v, w, join) = match task {
                Task::Split(v) => {
                    let (l, r) = self.tdd.vtree.children(v);
                    let (lc, rc) = (counts[l.idx()], counts[r.idx()]);
                    let all = |c: (u32, u32)| c.0 == c.1;
                    if (all(lc) && rc.0 == 0) || (all(rc) && lc.0 == 0) { continue; }
                    // One side is pure and the mixed side `w` has a child `c`
                    // pure the same way: move `c` across now, which keeps
                    // the pure side whole, and look at `v` again.
                    let pure = |c: (u32, u32)| if all(c) { Some(true) } else if c.0 == 0 { Some(false) } else { None };
                    let across = match (pure(lc), pure(rc)) {
                        (Some(p), None) => Some((r, p, l)),
                        (None, Some(p)) => Some((l, p, r)),
                        _ => None,
                    }
                    .and_then(|(w, p, side)| {
                        let (wl, wr) = self.tdd.vtree.children(w);
                        [wl, wr]
                            .into_iter()
                            .find(|c| {
                                pure(counts[c.idx()]) == Some(p)
                                    && tsize[self.target.lca(tl[side.idx()], tl[c.idx()]).idx()] == counts[side.idx()].1 + counts[c.idx()].1
                            })
                            .map(|c| (w, c))
                    });
                    if let Some((w, c)) = across {
                        lim.try_push(&mut tasks, Task::Split(v))?;
                        (v, w, c)
                    } else {
                        let (w, next) = if lc.0 == 0 { (r, Task::Lift(v, r)) }
                            else if rc.0 == 0 { (l, Task::Lift(v, l)) }
                            else if all(lc) { (r, Task::Join(v, r)) }
                            else if all(rc) { (l, Task::Join(v, l)) }
                            else { (l, Task::Both(v, l)) };
                        lim.try_push(&mut tasks, next)?;
                        lim.try_push(&mut tasks, Task::Split(w))?;
                        continue;
                    }
                }
                Task::Join(v, w) => (v, w, wanted_child(&self.tdd.vtree, &counts, w)),
                Task::Lift(v, w) | Task::Both(v, w) => {
                    let selected = wanted_child(&self.tdd.vtree, &counts, w);
                    let (l, r) = self.tdd.vtree.children(w);
                    if matches!(task, Task::Both(..)) {
                        lim.try_push(&mut tasks, Task::Join(v, w))?;
                        lim.try_push(&mut tasks, Task::Split(w))?;
                    }
                    (v, w, if l == selected { r } else { l })
                }
            };
            self.rotate_join(v, w, join)?;
            for t in [w, v] {
                let (l, r) = self.tdd.vtree.children(t);
                let (lc, rc) = (counts[l.idx()], counts[r.idx()]);
                counts[t.idx()] = (lc.0 + rc.0, lc.1 + rc.1);
                tl[t.idx()] = self.target.lca(tl[l.idx()], tl[r.idx()]);
            }
        }
        Ok(wanted_child(&self.tdd.vtree, &counts, root))
    }

    /// Rotate `w` below `v`, joining `joined` with `w`'s sibling.
    fn rotate_join(&mut self, v: VtreeIdx, w: VtreeIdx, joined: VtreeIdx) -> Result<(), RestructureError> {
        self.eng.limits().check_stop()?;
        for level in [v, w] {
            if self.tdd.level(level).is_marginal() {
                return Err(OperationError::MarginalLevel(level).into());
            }
        }
        let (_, v_right) = self.tdd.vtree.children(v);
        let kind = if w == v_right { RotationKind::Left } else { RotationKind::Right };
        let (w_left, w_right) = self.tdd.vtree.children(w);
        let faces = match kind { RotationKind::Left => w_left, RotationKind::Right => w_right };
        if faces != joined { self.mirror(w); }
        let mut rule = Kept { delta: 0 };
        let moves = [RotationMove { pivot: v, kind }];
        if !probe_moves(self.eng, self.tdd, &moves, &mut rule, &mut self.scratch, self.bound)? {
            return Err(RestructureError::Bound { pivot: v });
        }
        self.stats.rotations += 1;
        self.pairs = (self.pairs as i64 + rule.delta) as usize;
        self.stats.peak_pairs = self.stats.peak_pairs.max(self.pairs);
        Ok(())
    }

    /// Exchange node `t`'s children and read its level's pairs the other way.
    fn mirror(&mut self, t: VtreeIdx) {
        mirror_at(self.tdd, t);
        self.stats.mirrors += 1;
    }
}

/// Exchange node `t`'s children and read its level's pairs the other way.
fn mirror_at(tdd: &mut Tdd, t: VtreeIdx) {
    Vtree::make_mut(&mut tdd.vtree).swap_children(t);
    tdd.levels[t.idx()].swap_sides();
}

/// Keep every rotation, and remember what it did to the pair count.
struct Kept {
    delta: i64,
}

impl ProbeRule for Kept {
    fn keeps(&mut self, probe: &RotationProbe<'_>, _info: &RotationInfo) -> bool {
        self.delta = probe.live_pairs_delta();
        true
    }
}

/// Mirror what still faces the other way, then move every level to its node's
/// index in `target` and seat the diagram on it.
fn reseat(eng: &Engine, tdd: &mut Tdd, target: &Arc<Vtree>, structural: bool) -> Result<usize, RestructureError> {
    let n = tdd.vtree.num_nodes();
    if n != target.num_nodes() {
        return Err(RestructureError::Variables { variable: VarId(0) });
    }
    let mut map = Transient::new(eng.limits(), Vec::new());
    eng.limits().try_resize(&mut map, n, VtreeIdx(0))?;
    let mut mirrors = 0;
    let mut stack = Transient::new(eng.limits(), Vec::new());
    eng.limits().try_push(&mut stack, (tdd.vtree.root(), target.root()))?;
    while let Some((s, t)) = stack.pop() {
        eng.limits().check_stop()?;
        map[s.idx()] = t;
        match (tdd.vtree.node(s), target.node(t)) {
            (VtreeNode::Leaf { var: a, .. }, VtreeNode::Leaf { var: b, .. }) if a == b => {}
            (VtreeNode::Internal { .. }, VtreeNode::Internal { .. }) => {
                let (tl, tr) = target.children(t);
                let (mut sl, mut sr) = tdd.vtree.children(s);
                let some_leaf = first_leaf(&tdd.vtree, sl);
                if !target.in_subtree(target.leaf_of(some_leaf).expect("same variables"), tl) {
                    mirror_at(tdd, s);
                    mirrors += 1;
                    std::mem::swap(&mut sl, &mut sr);
                }
                eng.limits().try_push(&mut stack, (sl, tl))?;
                eng.limits().try_push(&mut stack, (sr, tr))?;
            }
            _ => return Err(RestructureError::Variables { variable: VarId(0) }),
        }
    }
    let mut levels: Vec<TddLevel> = Vec::new();
    eng.limits().try_resize(&mut levels, n, TddLevel::default())?;
    for (from, &to) in map.iter().enumerate() {
        levels[to.idx()] = std::mem::take(&mut tdd.levels[from]);
    }
    tdd.dirty.remap(&map);
    tdd.levels = levels.into();
    tdd.vtree = Arc::clone(target);
    tdd.output = TddNodeId { vtree: target.root(), local: tdd.output.local };
    if structural && tdd.dirty.is_empty() {
        tdd.levels.certify(tdd.output);
    }
    Ok(mirrors)
}

/// The variable of the leftmost leaf under `t`.
fn first_leaf(vtree: &Vtree, mut t: VtreeIdx) -> VarId {
    loop {
        match vtree.node(t) {
            VtreeNode::Leaf { var: crate::vtree::REGION_VAR, .. } => crate::vtree::region_unsupported("restructuring"),
            VtreeNode::Leaf { var, .. } => return *var,
            VtreeNode::Internal { left, .. } => t = *left,
        }
    }
}

#[cfg(test)]
#[path = "tests/target.rs"]
mod tests;
