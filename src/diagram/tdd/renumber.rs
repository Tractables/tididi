//! Renumbering each level's nodes in the order a walk down from the output
//! first reaches them ([`Tdd::renumber_top_down`]).
//!
//! Two passes. The first walks the vtree from the root down and numbers each
//! structural internal level's nodes as its parent level's pairs first name
//! them, the parent's nodes read in their new order; nodes no pair names
//! follow in their stored order. The second copies every level bottom-up into
//! a fresh diagram, node by node in the new order, each pair's sides
//! rewritten through its children's numbering. Leaf and marginal levels keep
//! their slots, so references into them pass through unchanged.

use std::sync::Arc;

use crate::Engine;
use crate::diagram::{Assembly, ChildDecoder, ChildPair, ChildSide, EncodedChildRef, LevelCounts, NodeIdx, Tdd, TddNodeId, WeightStore, ZERO};
use crate::limits::OperationError;
use crate::value::{Count, CountRead, CountVec};
use crate::vtree::VtreeIdx;

/// A node of a renumbered level that no pair has named yet.
const UNSEEN: u32 = u32::MAX;

/// Each level's numbering: `new[t][i]` is the new index of stored node `i`
/// of level `t`, and `old[t][j]` the stored index of new node `j`. Both rows
/// are empty for a leaf or marginal level, which keeps its slots.
struct Numbering {
    new: Vec<Vec<u32>>,
    old: Vec<Vec<u32>>,
}

impl Numbering {
    /// Give node `i` of level `t` the level's next new index, unless a pair
    /// named it before.
    #[inline]
    fn reach(&mut self, t: VtreeIdx, i: usize) {
        if self.new[t.idx()][i] == UNSEEN {
            self.new[t.idx()][i] = self.old[t.idx()].len() as u32;
            self.old[t.idx()].push(i as u32);
        }
    }
}

impl Engine {
    /// Run [`Tdd::renumber_top_down`] with this engine's allocation,
    /// cancellation and output limits.
    ///
    /// # Errors
    ///
    /// Cancellation, allocation refusal and the output-node cap return
    /// [`OperationError::Stopped`], [`OperationError::OverBudget`] and
    /// [`OperationError::OutputCap`], respectively. `f` is not changed.
    pub fn renumber_top_down(&self, f: &Tdd) -> Result<Tdd, OperationError> {
        let lim = self.limits();
        let _op = lim.enter()?;
        if f.is_zero() {
            return f.try_clone_on(self);
        }
        let numbering = number(self, f)?;
        let result = copy(self, f, &numbering);
        for row in numbering.new.into_iter().chain(numbering.old) {
            lim.discard(row);
        }
        result
    }
}

impl Tdd {
    /// The same diagram with each level's nodes numbered in the order a walk
    /// down from the output first reaches them.
    ///
    /// The walk starts at the output and descends the vtree a level at a
    /// time. It reads a level's nodes in their new order and each node's
    /// pairs in their stored order, and numbers a node of a child level when
    /// a pair first names it. Nodes the walk does not reach, which a
    /// minimized diagram does not have, follow in their stored order. A pass
    /// that reads a level's nodes in order and their pairs in order then
    /// meets each child level's nodes in ascending order the first time it
    /// meets them.
    ///
    /// The result denotes the same function with the same nodes, each with
    /// the same pairs in the same order; only the numbers change. Leaf and
    /// marginal levels keep their slots, and references into them are
    /// unchanged. A canonical diagram stays canonical, its level counts
    /// ([`attach_level_counts`](Self::attach_level_counts)) and weights stay
    /// with their nodes, and a level is held as the description of its pairs
    /// ([`ImplicitLevel`](crate::diagram::ImplicitLevel)) where its new
    /// numbering lets it be one. Renumbering the result again changes
    /// nothing.
    ///
    /// # Errors
    ///
    /// [`OperationError::OverBudget`] if an allocation is refused. Runs on
    /// this diagram's execution context; use
    /// [`Engine::renumber_top_down`](crate::Engine::renumber_top_down)
    /// inside a batch with resource limits.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{Tdd, Vtree};
    /// use tididi::diagram::{ChildDecoder, ChildRef, NodeIdx};
    ///
    /// let vtree = Arc::new(Vtree::balanced(4));
    /// let f = Tdd::clause(&vtree, [1, 3])? & Tdd::clause(&vtree, [-2, 4])?;
    /// let g = f.renumber_top_down()?;
    /// assert!(g.equivalent(&f)?);
    /// assert_eq!((g.node_count(), g.pair_count()), (f.node_count(), f.pair_count()));
    ///
    /// // The output is the first node of its level, and its first pair
    /// // names the first node of each child level.
    /// assert_eq!(g.output().local, NodeIdx(0));
    /// let first = g.level(vtree.root()).pairs_iter_of_idx(0).next().expect("a pair");
    /// let child = |side| ChildDecoder::structural().child(side);
    /// assert_eq!(child(first.left), ChildRef::Node(NodeIdx(0)));
    /// assert_eq!(child(first.right), ChildRef::Node(NodeIdx(0)));
    /// # tididi::test_helpers::assert_canonical(&g);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn renumber_top_down(&self) -> Result<Tdd, OperationError> {
        let context = Arc::clone(self.context());
        context.run(|eng| eng.renumber_top_down(self))
    }
}

/// Whether level `t` is renumbered: a structural internal level.
fn renumbered(f: &Tdd, t: VtreeIdx) -> bool {
    !f.vtree.node(t).is_leaf() && !f.levels[t.idx()].is_marginal()
}

/// Number every renumbered level, the root's first and each level before
/// its children ([`Tdd::renumber_top_down`]).
fn number(eng: &Engine, f: &Tdd) -> Result<Numbering, OperationError> {
    let lim = eng.limits();
    let levels = f.levels.len();
    let mut numbering = Numbering { new: Vec::new(), old: Vec::new() };
    lim.reserve_exact(&mut numbering.new, levels)?;
    lim.reserve_exact(&mut numbering.old, levels)?;
    numbering.new.resize_with(levels, Vec::new);
    numbering.old.resize_with(levels, Vec::new);
    let mut top_down = Vec::new();
    lim.reserve_exact(&mut top_down, levels)?;
    for (t, left, right) in f.vtree.internal_bottomup() {
        top_down.push((t, left, right));
        if renumbered(f, t) {
            let width = f.levels[t.idx()].slot_count();
            lim.try_resize(&mut numbering.new[t.idx()], width, UNSEEN)?;
            lim.reserve_exact(&mut numbering.old[t.idx()], width)?;
        }
    }
    top_down.reverse();
    let output = f.output;
    if renumbered(f, output.vtree) {
        numbering.reach(output.vtree, output.local.idx());
    }
    let structural = ChildDecoder::structural();
    let mut poll = lim.gate();
    for &(t, left, right) in &top_down {
        if !renumbered(f, t) {
            continue;
        }
        let level = &f.levels[t.idx()];
        // The nodes no parent named, after those it did.
        for i in 0..level.slot_count() {
            numbering.reach(t, i);
        }
        let (into_left, into_right) = (renumbered(f, left), renumbered(f, right));
        if !into_left && !into_right {
            continue;
        }
        for j in 0..level.slot_count() {
            let pairs = level.pairs_iter_of_idx(numbering.old[t.idx()][j] as usize);
            poll.poll(pairs.len() as u64)?;
            for pair in pairs {
                if into_left && pair.left != ZERO.into() {
                    numbering.reach(left, structural.node(pair.left).idx());
                }
                if into_right && pair.right != ZERO.into() {
                    numbering.reach(right, structural.node(pair.right).idx());
                }
            }
        }
    }
    poll.flush()?;
    lim.discard(top_down);
    Ok(numbering)
}

/// Copy `f` into a fresh diagram under `numbering`: each renumbered level's
/// nodes in their new order with their pairs' sides renumbered, each
/// marginal level whole.
fn copy(eng: &Engine, f: &Tdd, numbering: &Numbering) -> Result<Tdd, OperationError> {
    let lim = eng.limits();
    let mut out = Assembly::new(eng, &f.vtree)?;
    *out.parts_mut().1 = f.weights.as_ref().map(WeightStore::empty_like);
    for (i, source) in f.levels.iter().enumerate() {
        let t = VtreeIdx(i as u32);
        if source.is_marginal() {
            out.replace_level(eng, t, f.level_view(t))?;
        }
    }
    let structural = ChildDecoder::structural();
    let side = |child: VtreeIdx, raw: EncodedChildRef| -> EncodedChildRef {
        let row = &numbering.new[child.idx()];
        if row.is_empty() || raw == ZERO.into() { raw } else { NodeIdx(row[structural.node(raw).idx()]).into() }
    };
    let mut pairs = Vec::new();
    let mut poll = lim.gate();
    let mut emitted = 0u64;
    for (t, left, right) in f.vtree.internal_bottomup() {
        if !renumbered(f, t) {
            continue;
        }
        let source = &f.levels[t.idx()];
        // A node of one pair holds it inline; the others' go to the arena.
        let arena: usize = numbering.old[t.idx()].iter().map(|&i| source.pairs_iter_of_idx(i as usize).len()).filter(|&n| n > 1).sum();
        out.reserve(eng, t, source.slot_count(), arena)?;
        for &i in &numbering.old[t.idx()] {
            pairs.clear();
            for pair in source.pairs_iter_of_idx(i as usize) {
                lim.try_push(&mut pairs, ChildPair::new(side(left, pair.left), side(right, pair.right)))?;
            }
            poll.poll(pairs.len() as u64)?;
            out.push(eng, t, &pairs)?;
            emitted += 1;
        }
        lim.level_done(emitted)?;
        let level = &mut out.parts_mut().0[t.idx()];
        for side in [ChildSide::Left, ChildSide::Right] {
            level.set_has_value_refs(side, source.has_value_refs(side));
        }
    }
    poll.flush()?;
    lim.discard(pairs);
    let local = match numbering.new[f.output.vtree.idx()].as_slice() {
        [] => f.output.local,
        row => NodeIdx(row[f.output.local.idx()]),
    };
    let output = TddNodeId { vtree: f.output.vtree, local };
    let counts = match f.levels.counts() {
        Some(counts) => Some(renumber_counts(eng, f, counts, numbering)?),
        None => None,
    };
    // The levels are the operand's up to numbering, so its outstanding
    // reduction work carries over as it is, and its canonical form holds.
    let mut tdd = out.finish_with(output, f.dirty.clone(), &[], None)?;
    if f.levels.is_canonical(f.output) {
        tdd.levels.certify(output);
    }
    if let Some(counts) = counts {
        tdd.levels.keep_counts(counts);
    }
    Ok(tdd)
}

/// The level counts `f` keeps, each column's values in its level's new order.
fn renumber_counts(eng: &Engine, f: &Tdd, counts: &LevelCounts, numbering: &Numbering) -> Result<LevelCounts, OperationError> {
    let mut out = LevelCounts::none(eng, f.levels.len())?;
    for i in 0..f.levels.len() {
        let t = VtreeIdx(i as u32);
        let Some(column) = counts.column(t) else { continue };
        let order = &numbering.old[i];
        if order.is_empty() {
            out.set(t, Arc::clone(column));
            continue;
        }
        let mut moved = CountVec::try_with_capacity(eng, order.len())?;
        for &old in order {
            let count = match column.get(old as usize) {
                CountRead::Fast(v) => Count::Fast(v),
                CountRead::Big(v) => Count::Big(v.clone()),
            };
            moved.push(eng, count)?;
        }
        out.set(t, Arc::new(moved));
    }
    Ok(out)
}

#[cfg(test)]
#[path = "tests/renumber.rs"]
mod tests;
