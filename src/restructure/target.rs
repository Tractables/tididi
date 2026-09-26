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

use std::collections::VecDeque;
use std::sync::Arc;

use rustc_hash::FxHashMap;

use crate::Engine;
use crate::diagram::{Tdd, TddLevel, TddNodeId};
use crate::limits::OperationError;
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
        let mut work = vec![(mover.tdd.vtree.root(), target.root())];
        while let Some((a, b)) = work.pop() {
            self.limits().check_stop()?;
            mover.arrange(a, b, &mut work)?;
        }
        let Mover { stats, .. } = mover;
        reseat(self, tdd, target, structural)?;
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
/// its variable set. A collision can only make the plan wrong, never the
/// diagram: the reseat checks the final shape exactly.
fn scramble(var: VarId) -> u64 {
    let mut z = u64::from(var.0).wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The variable-set fingerprint and leaf count of every node under `root`.
fn fingerprints(vtree: &Vtree, root: VtreeIdx) -> FxHashMap<VtreeIdx, (u64, u32)> {
    let mut out: FxHashMap<VtreeIdx, (u64, u32)> = FxHashMap::default();
    let order: Vec<VtreeIdx> = vtree.subtree(root).collect();
    for &t in order.iter().rev() {
        let entry = match vtree.node(t) {
            VtreeNode::Leaf { var, .. } => (scramble(*var), 1),
            VtreeNode::Internal { left, right, .. } => {
                let (l, r) = (out[left], out[right]);
                (l.0.wrapping_add(r.0), l.1 + r.1)
            }
        };
        out.insert(t, entry);
    }
    out
}

/// The largest subtrees strictly below `root` whose fingerprint is in `shared`,
/// in the order a left-first walk meets them.
fn units(vtree: &Vtree, root: VtreeIdx, prints: &FxHashMap<VtreeIdx, (u64, u32)>, shared: &FxHashMap<(u64, u32), VtreeIdx>) -> Vec<VtreeIdx> {
    let mut out = Vec::new();
    let mut stack = vec![root];
    while let Some(t) = stack.pop() {
        if t != root && shared.contains_key(&prints[&t]) {
            out.push(t);
            continue;
        }
        if let VtreeNode::Internal { left, right, .. } = vtree.node(t) {
            stack.push(*right);
            stack.push(*left);
        }
    }
    out
}

/// The tree above a node's units, as the set of its internal clusters: each a
/// bitmask over unit positions, the root's all of them.
fn clusters(vtree: &Vtree, root: VtreeIdx, bit_of: &FxHashMap<VtreeIdx, u32>) -> (Vec<u32>, FxHashMap<u32, VtreeIdx>) {
    let mut node_of = FxHashMap::default();
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
    let mut set: Vec<u32> = node_of.keys().copied().collect();
    set.sort_unstable();
    (set, node_of)
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
    let mut inside: Vec<u32> = set.iter().copied().filter(|&d| d != c && d & c == d).collect();
    let mut bits = c;
    while bits != 0 {
        let low = bits & bits.wrapping_neg();
        inside.push(low);
        bits &= bits - 1;
    }
    let maximal: Vec<u32> = inside
        .iter()
        .copied()
        .filter(|&d| !inside.iter().any(|&e| e != d && e & d == d))
        .collect();
    debug_assert_eq!(maximal.len(), 2, "a cluster of a binary tree has two children");
    (maximal[0], maximal[1])
}

/// The smallest cluster of `set` strictly containing `c`.
fn parent_of(set: &[u32], c: u32) -> u32 {
    set.iter()
        .copied()
        .filter(|&d| d != c && d & c == c)
        .min_by_key(|d| d.count_ones())
        .expect("only the root has no parent")
}

/// `set` after `turn`.
fn turned(set: &[u32], turn: Turn) -> Vec<u32> {
    let mut out: Vec<u32> = set.iter().copied().filter(|&c| c != turn.w).collect();
    out.push(turn.x | turn.y);
    out.sort_unstable();
    out
}

/// A shortest sequence of turns from `from` to `to`, two trees over the same
/// units given by their clusters, by breadth-first search over the trees.
pub(crate) fn shortest_turns(from: &[u32], to: &[u32]) -> Vec<Turn> {
    let root = *from.iter().max_by_key(|c| c.count_ones()).expect("a tree has a root");
    let mut seen: FxHashMap<Vec<u32>, Option<(Vec<u32>, Turn)>> = FxHashMap::default();
    seen.insert(from.to_vec(), None);
    let mut queue = VecDeque::from([from.to_vec()]);
    while let Some(set) = queue.pop_front() {
        if set == to {
            let mut path = Vec::new();
            let mut at = set;
            while let Some(Some((prev, turn))) = seen.get(&at) {
                path.push(*turn);
                at = prev.clone();
            }
            path.reverse();
            return path;
        }
        for &w in &set {
            if w == root {
                continue;
            }
            let x = parent_of(&set, w) & !w;
            let (y0, y1) = children_of(&set, w);
            for y in [y0, y1] {
                let turn = Turn { w, x, y };
                let next = turned(&set, turn);
                if !seen.contains_key(&next) {
                    seen.insert(next.clone(), Some((set.clone(), turn)));
                    queue.push_back(next);
                }
            }
        }
    }
    unreachable!("rotations connect every pair of trees over the same units")
}

/// A sequence of turns from `from` to the tree whose children are given by
/// `split`, made one split at a time from the root down: not shortest, but
/// linear in the units per split.
pub(crate) fn split_turns(from: &[u32], split: &dyn Fn(u32) -> (u32, u32)) -> Vec<Turn> {
    let mut set = from.to_vec();
    let mut turns = Vec::new();
    let root = *set.iter().max_by_key(|c| c.count_ones()).expect("a tree has a root");
    let mut work = vec![root];
    while let Some(v) = work.pop() {
        if v.count_ones() < 2 {
            continue;
        }
        let (s1, s2) = split(v);
        make_child(&mut set, v, s1, &mut turns);
        work.push(s1);
        work.push(s2);
    }
    turns
}

/// Turn the tree `set` until cluster `v` has `s` as a child.
fn make_child(set: &mut Vec<u32>, v: u32, s: u32, turns: &mut Vec<Turn>) {
    let (l, r) = children_of(set, v);
    if l == s || r == s {
        return;
    }
    for (w, x) in [(l, r), (r, l)] {
        if w.count_ones() >= 2 {
            let (y0, y1) = children_of(set, w);
            for y in [y0, y1] {
                if x | y == s {
                    apply_turn(set, Turn { w, x, y }, turns);
                    return;
                }
            }
        }
    }
    let (sl, sr) = (s & l, s & r);
    if sr == 0 || sl == 0 {
        let (w, x) = if sr == 0 { (l, r) } else { (r, l) };
        make_child(set, w, s, turns);
        apply_turn(set, Turn { w, x, y: w & !s }, turns);
        return;
    }
    let (w, x, sw, sx) = if sl != l { (l, r, sl, sr) } else { (r, l, sr, sl) };
    if sx == x {
        // One side is whole: bring the other side's part up beside it.
        make_child(set, w, sw, turns);
        apply_turn(set, Turn { w, x, y: sw }, turns);
        return;
    }
    make_child(set, w, sw, turns);
    let rest = w & !sw;
    apply_turn(set, Turn { w, x, y: rest }, turns);
    let wide = rest | x;
    make_child(set, wide, sx, turns);
    apply_turn(set, Turn { w: wide, x: sw, y: sx }, turns);
}

/// Record `turn` and apply it to `set`.
fn apply_turn(set: &mut Vec<u32>, turn: Turn, turns: &mut Vec<Turn>) {
    *set = turned(set, turn);
    turns.push(turn);
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
        let prints_a = fingerprints(&self.tdd.vtree, a);
        let prints_b = fingerprints(self.target, b);
        let in_b: FxHashMap<(u64, u32), VtreeIdx> =
            prints_b.iter().filter(|&(&t, _)| t != b).map(|(&t, &p)| (p, t)).collect();
        let in_a: FxHashMap<(u64, u32), VtreeIdx> =
            prints_a.iter().filter(|&(&t, _)| t != a).map(|(&t, &p)| (p, t)).collect();
        let units_a = units(&self.tdd.vtree, a, &prints_a, &in_b);
        let units_b = units(self.target, b, &prints_b, &in_a);
        debug_assert_eq!(units_a.len(), units_b.len(), "shared subtrees pair up");
        let mut bit_a = FxHashMap::default();
        let mut bit_b = FxHashMap::default();
        for (i, &u) in units_a.iter().enumerate() {
            bit_a.insert(u, 1u32 << i);
            let partner = in_b[&prints_a[&u]];
            bit_b.insert(partner, 1u32 << i);
            work.push((u, partner));
        }
        if units_a.len() <= 2 {
            return Ok(());
        }
        assert!(units_a.len() <= 32, "restructure_to: more than 32 units above shared subtrees");
        let (from, mut node_of) = clusters(&self.tdd.vtree, a, &bit_a);
        let (to, target_node_of) = clusters(self.target, b, &bit_b);
        let turns = if units_a.len() <= EXACT_ATOMS {
            shortest_turns(&from, &to)
        } else {
            let split = |c: u32| {
                let (l, r) = self.target.children(target_node_of[&c]);
                (cluster_of(self.target, l, &bit_b), cluster_of(self.target, r, &bit_b))
            };
            split_turns(&from, &split)
        };
        for turn in turns {
            self.turn(turn, &mut node_of, &bit_a)?;
        }
        Ok(())
    }

    /// Apply one unoriented turn: mirror the demoted node if its wrong child
    /// faces the sibling, then rotate.
    fn turn(&mut self, turn: Turn, node_of: &mut FxHashMap<u32, VtreeIdx>, bit_of: &FxHashMap<VtreeIdx, u32>) -> Result<(), RestructureError> {
        let v = node_of[&(turn.w | turn.x)];
        let w = node_of[&turn.w];
        let (_, v_right) = self.tdd.vtree.children(v);
        // A left rotation joins the sibling with w's left child, a right
        // rotation with w's right child.
        let kind = if w == v_right { RotationKind::Left } else { RotationKind::Right };
        let (w_left, w_right) = self.tdd.vtree.children(w);
        let joined = match kind {
            RotationKind::Left => w_left,
            RotationKind::Right => w_right,
        };
        let joined_set = match bit_of.get(&joined) {
            Some(&bit) => bit,
            None => node_of.iter().find(|&(_, &t)| t == joined).map(|(&c, _)| c).expect("a child is a unit or a cluster"),
        };
        if joined_set != turn.y {
            self.mirror(w);
        }
        for level in [v, w] {
            if self.tdd.level(level).is_marginal() {
                return Err(OperationError::MarginalLevel(level).into());
            }
        }
        let mut rule = Kept { delta: 0 };
        let moves = [RotationMove { pivot: v, kind }];
        if !probe_moves(self.eng, self.tdd, &moves, &mut rule, &mut self.scratch, self.bound)? {
            return Err(RestructureError::Bound { pivot: v });
        }
        self.stats.rotations += 1;
        self.pairs = (self.pairs as i64 + rule.delta) as usize;
        self.stats.peak_pairs = self.stats.peak_pairs.max(self.pairs);
        node_of.remove(&turn.w);
        node_of.insert(turn.x | turn.y, w);
        Ok(())
    }

    /// Exchange node `t`'s children and read its level's pairs the other way.
    fn mirror(&mut self, t: VtreeIdx) {
        Arc::make_mut(&mut self.tdd.vtree).swap_children(t);
        self.tdd.levels[t.idx()].swap_sides();
        self.stats.mirrors += 1;
    }
}

/// The bitmask of units under node `t`.
fn cluster_of(vtree: &Vtree, t: VtreeIdx, bit_of: &FxHashMap<VtreeIdx, u32>) -> u32 {
    if let Some(&bit) = bit_of.get(&t) {
        return bit;
    }
    let (l, r) = vtree.children(t);
    cluster_of(vtree, l, bit_of) | cluster_of(vtree, r, bit_of)
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
fn reseat(eng: &Engine, tdd: &mut Tdd, target: &Arc<Vtree>, structural: bool) -> Result<(), RestructureError> {
    let n = tdd.vtree.num_nodes();
    if n != target.num_nodes() {
        return Err(RestructureError::Variables { variable: VarId(0) });
    }
    let mut map = Vec::new();
    eng.limits().try_resize(&mut map, n, VtreeIdx(0))?;
    let mut stack = vec![(tdd.vtree.root(), target.root())];
    while let Some((s, t)) = stack.pop() {
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
                    std::mem::swap(&mut sl, &mut sr);
                }
                stack.push((sl, tl));
                stack.push((sr, tr));
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
    Ok(())
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
