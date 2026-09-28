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
//! largest subtrees whose variable sets the two trees share are fixed units,
//! and only the part of the tree above them is rearranged; each unit is then
//! matched on its own. A vtree that differs from the target only above some
//! shared subtrees is therefore moved without touching the levels inside them.
//! Above the units the rearrangement is a shortest rotation sequence for up to
//! `EXACT_ATOMS` units, and a split-by-split construction beyond that.
//! Mirrors are free in the plan: they cost one pass over a level and never
//! change a size. A last pass mirrors whatever orientation still differs and
//! moves every level to its index in the target's node array.

use std::sync::Arc;

use rustc_hash::FxHashMap;

use crate::Engine;
use crate::diagram::{Tdd, TddLevel, TddNodeId};
use crate::limits::{Limits, OperationError, Transient};
use crate::restructure::scratch::RestructureScratch;
use crate::restructure::search::{ProbeRule, RotationMove, RotationProbe, probe_moves};
use crate::vtree::rotate::RotationInfo;
use crate::vtree::{RotationKind, VarId, Vtree, VtreeIdx, VtreeNode};

use super::RestructureError;

/// The most units a rearrangement above shared subtrees searches exactly:
/// the rooted binary trees over seven labelled units number 10 395.
pub(crate) const EXACT_ATOMS: usize = 7;

/// What a [`Tdd::restructure_to`] call did.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct RestructureStats {
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
    /// canonical.
    ///
    /// A rotation rebuilds the two levels it turns from the pairs of the
    /// outer level expanded through the inner one; a mirror reads one level's
    /// pairs the other way round; no other level is touched. The largest
    /// subtrees the two vtrees share are kept whole, so the cost falls on the
    /// levels above them, and it depends on the diagram's size on each vtree
    /// the rotations pass through, which neither end bounds. `bound` caps the
    /// input pairs any one rebuild may expand; `usize::MAX` is no bound.
    ///
    /// Diagrams with summed-out levels move as long as no rotation turns a
    /// summed-out level. Weighted diagrams are refused.
    ///
    /// # Errors
    ///
    /// [`RestructureError::Variables`] when the two vtrees do not have the
    /// same variables; [`RestructureError::Weighted`] for a diagram with a
    /// weight table; [`RestructureError::Bound`] when a rotation would expand
    /// more than `bound` pairs, and [`RestructureError::Operation`] with
    /// [`OperationError::MarginalLevel`] when it would turn a summed-out
    /// level, or with a refused allocation or an armed stop. After an error
    /// the diagram denotes the same function on its own copy of a vtree
    /// between the two.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{Tdd, Vtree};
    /// use tididi::vtree::VarId;
    ///
    /// let order = [VarId(1), VarId(2), VarId(3), VarId(4)];
    /// let stick = Arc::new(Vtree::linear_from_order(&order)?);
    /// let f = Tdd::clause(&stick, [1, -3])? & Tdd::clause(&stick, [2, 4])?;
    /// let before = f.model_count()?;
    ///
    /// let balanced = Arc::new(Vtree::balanced_over(&[VarId(3), VarId(1), VarId(4), VarId(2)])?);
    /// let mut g = f.clone();
    /// let stats = g.restructure_to(&balanced, usize::MAX)?;
    /// assert!(Arc::ptr_eq(g.vtree(), &balanced));
    /// assert_eq!(g.model_count()?, before);
    /// assert!(stats.rotations > 0);
    /// # tididi::test_helpers::assert_canonical(&g);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn restructure_to(&mut self, target: &Arc<Vtree>, bound: usize) -> Result<RestructureStats, RestructureError> {
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
    ) -> Result<RestructureStats, RestructureError> {
        let _op = self.limits().enter()?;
        same_variables(tdd.vtree(), target)?;
        if tdd.weights.is_some() {
            return Err(RestructureError::Weighted);
        }
        if Arc::ptr_eq(tdd.vtree(), target) {
            return Ok(RestructureStats { peak_pairs: tdd.pair_count(), ..RestructureStats::default() });
        }
        if tdd.is_zero() {
            *tdd = crate::build::constant_zero(self, target);
            return Ok(RestructureStats::default());
        }
        // A rotation's rebuilt levels are interchangeable with the ones they
        // replace only on a canonical diagram. With summed-out levels the
        // rotation keeps every contribution instead, and a minimize would
        // merge them, so those are moved as they are.
        let structural = !tdd.has_marginal_level();
        if structural {
            self.reduce(tdd, crate::reduce::ReductionPlan::default())?;
        }
        let mut mover = Mover {
            eng: self,
            tdd,
            target: target.as_ref(),
            bound,
            scratch: self.scratch.restructure.checkout(self),
            stats: RestructureStats::default(),
            pairs: 0,
        };
        mover.pairs = mover.tdd.pair_count();
        mover.stats.peak_pairs = mover.pairs;
        Arc::make_mut(&mut mover.tdd.vtree);
        let mut work = Transient::new(self.limits(), Vec::new());
        self.limits().try_push(&mut work, (mover.tdd.vtree.root(), target.root()))?;
        while let Some((a, b)) = work.pop() {
            self.limits().check_stop()?;
            mover.arrange(a, b, &mut work)?;
        }
        let Mover { mut stats, .. } = mover;
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

/// A deterministic 64-bit image of a variable, summed over a subtree to name
/// its variable set. The planner checks candidate shared sets exactly.
fn scramble(var: VarId) -> u64 {
    let mut z = u64::from(var.0).wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The variable-set fingerprint and leaf count of every node under `root`.
fn fingerprints(lim: &Limits, vtree: &Vtree, root: VtreeIdx) -> Result<FxHashMap<VtreeIdx, (u64, u32)>, OperationError> {
    let mut out = Transient::new(lim, FxHashMap::default());
    let mut order = Transient::new(lim, Vec::new());
    let mut gate = lim.gate();
    for t in vtree.subtree(root) {
        lim.try_push(&mut order, t)?;
        gate.poll(1)?;
    }
    lim.reserve_map(&mut out, order.len())?;
    for &t in order.iter().rev() {
        let entry = match vtree.node(t) {
            VtreeNode::Leaf { var, .. } => (scramble(*var), 1),
            VtreeNode::Internal { left, right, .. } => {
                let (l, r): ((u64, u32), (u64, u32)) = (out[left], out[right]);
                (l.0.wrapping_add(r.0), l.1 + r.1)
            }
        };
        out.insert(t, entry);
        gate.poll(1)?;
    }
    gate.finish()?;
    Ok(out.keep())
}

/// Index the proper subtrees by their variable-set fingerprint and size.
fn index_prints(lim: &Limits, prints: &FxHashMap<VtreeIdx, (u64, u32)>, root: VtreeIdx) -> Result<FxHashMap<(u64, u32), VtreeIdx>, OperationError> {
    let mut out = Transient::new(lim, FxHashMap::default());
    lim.reserve_map(&mut out, prints.len())?;
    out.extend(prints.iter().filter(|&(&t, _)| t != root).map(|(&t, &p)| (p, t)));
    Ok(out.keep())
}

/// The maximal shared proper subtrees, in left-first order.
fn units(lim: &Limits, vtree: &Vtree, root: VtreeIdx, prints: &FxHashMap<VtreeIdx, (u64, u32)>, shared: &FxHashMap<(u64, u32), VtreeIdx>) -> Result<Vec<VtreeIdx>, OperationError> {
    let mut out = Transient::new(lim, Vec::new());
    let mut stack = Transient::new(lim, Vec::new());
    lim.try_push(&mut stack, root)?;
    let mut gate = lim.gate();
    while let Some(t) = stack.pop() {
        gate.poll(1)?;
        if t != root && shared.contains_key(&prints[&t]) {
            lim.try_push(&mut out, t)?;
            continue;
        }
        if let VtreeNode::Internal { left, right, .. } = vtree.node(t) {
            lim.try_push(&mut stack, *right)?;
            lim.try_push(&mut stack, *left)?;
        }
    }
    gate.finish()?;
    Ok(out.keep())
}

/// The compact clusters above at most seven units, and their vtree nodes.
fn clusters(lim: &Limits, vtree: &Vtree, root: VtreeIdx, bit_of: &FxHashMap<VtreeIdx, u32>) -> Result<(Vec<u32>, FxHashMap<u32, VtreeIdx>), OperationError> {
    let mut node_of = Transient::new(lim, FxHashMap::default());
    lim.reserve_map(&mut node_of, bit_of.len())?;
    fn walk(vtree: &Vtree, t: VtreeIdx, bit_of: &FxHashMap<VtreeIdx, u32>, node_of: &mut FxHashMap<u32, VtreeIdx>) -> u32 {
        if let Some(&bit) = bit_of.get(&t) {
            return bit;
        }
        let (l, r) = vtree.children(t);
        let set = walk(vtree, l, bit_of, node_of) | walk(vtree, r, bit_of, node_of);
        node_of.insert(set, t);
        set
    }
    walk(vtree, root, bit_of, &mut node_of);
    let mut set = Transient::new(lim, Vec::new());
    lim.reserve_exact(&mut set, node_of.len())?;
    set.extend(node_of.keys().copied());
    set.sort_unstable();
    Ok((set.keep(), node_of.keep()))
}

/// One rotation above the units, read without orientation: the cluster `w`,
/// a child of `x | w`, is replaced by `x | y`, where `x` is `w`'s sibling and
/// `y` one of `w`'s children. The other child of `w` moves up beside `x | y`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Turn {
    pub(crate) w: u32,
    pub(crate) x: u32,
    pub(crate) y: u32,
}

/// The two children of cluster `c` in a tree given by its clusters `set`:
/// the largest clusters or single units strictly inside it.
pub(crate) fn children_of(set: &[u32], c: u32) -> (u32, u32) {
    let candidates = || set.iter().copied().filter(|&d| d != c && d & c == d)
        .chain((0..32).map(|i| 1u32 << i).filter(|&bit| c & bit != 0));
    let mut maximal = candidates().filter(|&d| !candidates().any(|e| e != d && e & d == d));
    let left = maximal.next().expect("a binary cluster has two children");
    let right = maximal.next().expect("a binary cluster has two children");
    debug_assert!(maximal.next().is_none());
    (left, right)
}

/// The smallest cluster of `set` strictly containing `c`.
fn parent_of(set: &[u32], c: u32) -> u32 {
    set.iter()
        .copied()
        .filter(|&d| d != c && d & c == c)
        .min_by_key(|d| d.count_ones())
        .expect("only the root has no parent")
}

/// A shortest sequence over at most seven units, with charged search states.
pub(crate) fn shortest_turns(lim: &Limits, from: &[u32], to: &[u32]) -> Result<Vec<Turn>, OperationError> {
    type State = [u32; EXACT_ATOMS - 1];
    debug_assert!(from.len() < EXACT_ATOMS && from.len() == to.len());
    let len = from.len();
    let root = *from.iter().max_by_key(|c| c.count_ones()).expect("a tree has a root");
    let mut first: State = [0; EXACT_ATOMS - 1];
    first[..len].copy_from_slice(from);
    let mut seen = Transient::new(lim, FxHashMap::<State, (usize, Turn)>::default());
    let mut queue = Transient::new(lim, Vec::new());
    lim.reserve_map(&mut seen, 1)?;
    seen.insert(first, (usize::MAX, Turn { w: 0, x: 0, y: 0 }));
    lim.try_push(&mut queue, first)?;
    let mut cursor = 0;
    let mut gate = lim.gate();
    while cursor < queue.len() {
        gate.poll(1)?;
        let state = queue[cursor];
        let set = &state[..len];
        if set == to {
            let mut path = Transient::new(lim, Vec::new());
            let mut at = state;
            loop {
                let &(prev, turn) = seen.get(&at).expect("a queued state was recorded");
                if prev == usize::MAX { break; }
                lim.try_push(&mut path, turn)?;
                at = queue[prev];
            }
            path.reverse();
            gate.finish()?;
            return Ok(path.keep());
        }
        for (i, &w) in set.iter().enumerate() {
            if w == root { continue; }
            let x = parent_of(set, w) & !w;
            let (y0, y1) = children_of(set, w);
            for y in [y0, y1] {
                let mut next = state;
                next[i] = x | y;
                next[..len].sort_unstable();
                if !seen.contains_key(&next) {
                    lim.reserve_map(&mut seen, 1)?;
                    seen.insert(next, (cursor, Turn { w, x, y }));
                    lim.try_push(&mut queue, next)?;
                }
            }
        }
        cursor += 1;
    }
    unreachable!("rotations connect every pair of trees over the same units")
}

/// The move in progress: the diagram on its own vtree, and what it cost.
struct Mover<'a, 'e> {
    eng: &'e Engine,
    tdd: &'a mut Tdd,
    target: &'a Vtree,
    bound: usize,
    scratch: crate::execution::pool::PoolGuard<'e, RestructureScratch>,
    stats: RestructureStats,
    /// The diagram's live pairs now.
    pairs: usize,
}

impl Mover<'_, '_> {
    /// Make the diagram's node `a` split its variables as the target's node
    /// `b` does, then queue each pair of shared subtrees below them.
    fn arrange(&mut self, a: VtreeIdx, b: VtreeIdx, work: &mut Vec<(VtreeIdx, VtreeIdx)>) -> Result<(), RestructureError> {
        if self.target.node(b).is_leaf() {
            return Ok(());
        }
        let lim = self.eng.limits();
        let prints_a = Transient::new(lim, fingerprints(lim, &self.tdd.vtree, a)?);
        let prints_b = Transient::new(lim, fingerprints(lim, self.target, b)?);
        let in_b = Transient::new(lim, index_prints(lim, &prints_b, b)?);
        let units_a = Transient::new(lim, units(lim, &self.tdd.vtree, a, &prints_a, &in_b)?);
        // Fingerprints select candidates; only equal leaf sets are shared units.
        let mut exact = units_a.len() <= EXACT_ATOMS;
        for &u in units_a.iter() {
            if !exact { break; }
            let partner = in_b[&prints_a[&u]];
            for t in self.tdd.vtree.subtree(u) {
                lim.check_stop()?;
                if let VtreeNode::Leaf { var, .. } = self.tdd.vtree.node(t) {
                    exact &= under(self.target, self.target.leaf_of(*var).expect("same variables"), partner);
                }
            }
        }
        if !exact {
            let (left, right) = self.target.children(b);
            let wanted = self.partition(a, left)?;
            let (l, r) = self.tdd.vtree.children(a);
            let other = if l == wanted { r } else { l };
            lim.try_push(work, (wanted, left))?;
            lim.try_push(work, (other, right))?;
            return Ok(());
        }
        let mut bit_a = Transient::new(lim, FxHashMap::default());
        let mut bit_b = Transient::new(lim, FxHashMap::default());
        lim.reserve_map(&mut bit_a, units_a.len())?;
        lim.reserve_map(&mut bit_b, units_a.len())?;
        for (i, &u) in units_a.iter().enumerate() {
            bit_a.insert(u, 1u32 << i);
            let partner = in_b[&prints_a[&u]];
            bit_b.insert(partner, 1u32 << i);
            lim.try_push(work, (u, partner))?;
        }
        if units_a.len() <= 2 {
            return Ok(());
        }
        let (from, node_of) = clusters(lim, &self.tdd.vtree, a, &bit_a)?;
        let from = Transient::new(lim, from);
        let mut node_of = Transient::new(lim, node_of);
        let (to, target_nodes) = clusters(lim, self.target, b, &bit_b)?;
        let to = Transient::new(lim, to);
        lim.discard(target_nodes);
        let turns = Transient::new(lim, shortest_turns(lim, &from, &to)?);
        for &turn in turns.iter() {
            self.turn(turn, &mut node_of, &bit_a)?;
        }
        Ok(())
    }

    /// Group the variables below `target` into one child of `root`, without
    /// encoding units in a fixed-width integer. Tasks replace recursion on a
    /// long vtree; counts change only at the two levels a rotation rebuilds.
    fn partition(&mut self, root: VtreeIdx, target: VtreeIdx) -> Result<VtreeIdx, RestructureError> {
        #[derive(Clone, Copy)]
        enum Task { Split(VtreeIdx), Lift(VtreeIdx, VtreeIdx), Join(VtreeIdx, VtreeIdx), Both(VtreeIdx, VtreeIdx) }
        let lim = self.eng.limits();
        let mut counts = Transient::new(lim, Vec::<(u32, u32)>::new());
        lim.try_resize(&mut counts, self.tdd.vtree.num_nodes(), (0, 0))?;
        for t in self.tdd.vtree.bottomup() {
            lim.check_stop()?;
            counts[t.idx()] = match self.tdd.vtree.node(t) {
                VtreeNode::Leaf { var, .. } => (u32::from(under(self.target, self.target.leaf_of(*var).expect("same variables"), target)), 1),
                VtreeNode::Internal { left, right, .. } => {
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
                    let (w, next) = if lc.0 == 0 { (r, Task::Lift(v, r)) }
                        else if rc.0 == 0 { (l, Task::Lift(v, l)) }
                        else if all(lc) { (r, Task::Join(v, r)) }
                        else if all(rc) { (l, Task::Join(v, l)) }
                        else { (l, Task::Both(v, l)) };
                    lim.try_push(&mut tasks, next)?;
                    lim.try_push(&mut tasks, Task::Split(w))?;
                    continue;
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
            }
        }
        Ok(wanted_child(&self.tdd.vtree, &counts, root))
    }

    /// Apply one unoriented turn: mirror the demoted node if its wrong child
    /// faces the sibling, then rotate.
    fn turn(&mut self, turn: Turn, node_of: &mut FxHashMap<u32, VtreeIdx>, bit_of: &FxHashMap<VtreeIdx, u32>) -> Result<(), RestructureError> {
        let v = node_of[&(turn.w | turn.x)];
        let w = node_of[&turn.w];
        let (left, right) = self.tdd.vtree.children(w);
        let set_of = |node: VtreeIdx| bit_of.get(&node).copied().unwrap_or_else(|| {
            node_of.iter().find(|&(_, &t)| t == node).map(|(&c, _)| c).expect("a child is a unit or cluster")
        });
        let joined = if set_of(left) == turn.y { left } else { right };
        self.rotate_join(v, w, joined)?;
        node_of.remove(&turn.w);
        node_of.insert(turn.x | turn.y, w);
        Ok(())
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
        Arc::make_mut(&mut self.tdd.vtree).swap_children(t);
        self.tdd.levels[t.idx()].swap_sides();
        self.stats.mirrors += 1;
    }
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
                if !under(target, target.leaf_of(some_leaf).expect("same variables"), tl) {
                    Arc::make_mut(&mut tdd.vtree).swap_children(s);
                    tdd.levels[s.idx()].swap_sides();
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
            VtreeNode::Leaf { var, .. } => return *var,
            VtreeNode::Internal { left, .. } => t = *left,
        }
    }
}

/// Whether `node` lies in the subtree of `root`.
fn under(vtree: &Vtree, mut node: VtreeIdx, root: VtreeIdx) -> bool {
    loop {
        if node == root {
            return true;
        }
        match vtree.node(node).parent() {
            Some(parent) => node = parent,
            None => return false,
        }
    }
}

#[cfg(test)]
#[path = "tests/target.rs"]
mod tests;
