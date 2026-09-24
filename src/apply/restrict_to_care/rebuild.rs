//! Re-emitting the live subgraph as a new diagram.

use crate::Engine;
use crate::limits::{OperationError, PollGate};
use crate::reduce::ReductionPlan;
use crate::diagram::{ChildDecoder, ChildPair, ChildSide, NodeIdx, Tdd, TddLevel, TddNodeId, ZERO, Assembly};
use crate::diagram::sort_pairs;
use crate::vtree::VtreeIdx;

use super::Marking;

impl Marking {
    /// Re-emit the live subgraph under the engine's limits, carrying marginal levels and weights into the pruned result.
    pub(super) fn rebuild(self, eng: &Engine, mut f: Tdd) -> Result<Tdd, OperationError> {
        let lim = eng.limits();
        lim.check_stop()?;
        let mut gate = lim.gate();
        let nlev = f.vtree.num_nodes();
        let v0 = f.output.vtree;
        let mut memo = Vec::new();
        lim.try_resize(&mut memo, nlev, Vec::new())?;
        for (row, level) in memo.iter_mut().zip(&f.levels) {
            gate.poll(1)?;
            if !level.is_marginal() {
                lim.try_resize(row, level.nodes.len(), DeadRebuilder::UNVISITED)?;
            }
        }
        let mut assembly = Assembly::new(eng, &f.vtree)?;
        let (out, weights) = assembly.parts_mut();
        let mut rb = DeadRebuilder {
            f: &f,
            alive: self.alive,
            pair_alive: self.pair_alive,
            out,
            memo,
        };
        let root = rb.rebuild(eng, &mut gate, v0, f.output.local)?;
        drop(rb);
        for (vi, level) in out.iter_mut().enumerate() {
            gate.poll(1)?;
            if f.levels[vi].is_marginal() {
                *level = std::mem::take(&mut f.levels[vi]);
            } else {
                for side in [ChildSide::Left, ChildSide::Right] {
                    level.set_has_value_refs(side, f.levels[vi].has_value_refs(side));
                }
            }
        }
        gate.flush()?;
        *weights = f.weights.take();
        let mut g = assembly.finish(TddNodeId { vtree: v0, local: root })?;
        // A child emitted before its pair partner collapses can become an orphan.
        eng.reduce(&mut g, ReductionPlan::Prune)?;
        Ok(g)
    }
}

/// The live subgraph's output arena and old-to-new node map.
struct DeadRebuilder<'a> {
    f: &'a Tdd,
    alive: Vec<Vec<bool>>,
    /// Bit k marks a live pair; all bits set means no pair-level information.
    pair_alive: Vec<Vec<u64>>,
    out: &'a mut [TddLevel],
    memo: Vec<Vec<u32>>,
}

/// A suspended node rebuild, resumed after its next pair's children are ready.
struct Frame {
    v: VtreeIdx,
    local: NodeIdx,
    next_pair: usize,
    pairs: Vec<ChildPair>,
}

impl Frame {
    /// Start a node with no processed pairs.
    fn new(v: VtreeIdx, local: NodeIdx) -> Self {
        Self { v, local, next_pair: 0, pairs: Vec::new() }
    }
}

impl DeadRebuilder<'_> {
    /// Unvisited memo entries differ from both emitted indices and the false sentinel.
    const UNVISITED: u32 = u32::MAX - 1;

    /// Whether a structural child is a nonzero leaf label or an alive internal node.
    fn alive_child(&self, v: VtreeIdx, local: NodeIdx) -> bool {
        local != ZERO && (self.f.vtree.node(v).is_leaf() || self.alive[v.idx()][local.idx()])
    }

    /// Return a rebuilt structural child, or None when its node still needs traversal.
    fn rebuilt_child(&self, v: VtreeIdx, local: NodeIdx) -> Option<NodeIdx> {
        if self.f.vtree.node(v).is_leaf() || local == ZERO { return Some(local); }
        let cached = self.memo[v.idx()][local.idx()];
        (cached != Self::UNVISITED).then_some(NodeIdx(cached))
    }

    /// Rebuild reachable nodes in depth-first order using a fallibly grown stack.
    fn rebuild(&mut self, eng: &Engine, gate: &mut PollGate, v: VtreeIdx, local: NodeIdx) -> Result<NodeIdx, OperationError> {
        if let Some(cached) = self.rebuilt_child(v, local) { return Ok(cached); }
        let lim = eng.limits();
        let mut stack = Vec::new();
        lim.try_push(&mut stack, Frame::new(v, local))?;
        let mut emitted = 0u64;
        while let Some(frame) = stack.last() {
            gate.poll(1)?;
            let (v, local, k) = (frame.v, frame.local, frame.next_pair);
            let source = self.f.levels[v.idx()].pairs_of_idx(local.idx());
            if k == source.len() {
                let mut frame = stack.pop().expect("the current frame exists");
                let result = if frame.pairs.is_empty() {
                    ZERO
                } else {
                    // Equal pairs carry multiplicity when a child level is marginal.
                    sort_pairs(&mut frame.pairs);
                    let result = self.out[v.idx()].push_node(eng.limits(), &frame.pairs)?;
                    if result.0 >= Self::UNVISITED { return Err(OperationError::OverBudget); }
                    emitted += 1;
                    lim.level_done(emitted)?;
                    result
                };
                self.memo[v.idx()][local.idx()] = result.0;
                continue;
            }
            let mask = self.pair_alive[v.idx()][local.idx()];
            if source.len() <= 64 && mask != u64::MAX && (mask >> k) & 1 == 0 {
                stack.last_mut().expect("the current frame exists").next_pair += 1;
                continue;
            }
            let pair = source[k];
            let (left, right) = self.f.vtree.children(v);
            let children = [(left, pair.left), (right, pair.right)];
            let dead = children.iter().any(|&(child, value)| {
                !self.f.levels[child.idx()].is_marginal()
                    && !self.alive_child(child, ChildDecoder::structural().node(value))
            });
            if dead {
                stack.last_mut().expect("the current frame exists").next_pair += 1;
                continue;
            }
            let mut refs = [pair.left, pair.right];
            let mut pending = None;
            for (side, (child, value)) in children.into_iter().enumerate() {
                if self.f.levels[child.idx()].is_marginal() { continue; }
                let node = ChildDecoder::structural().node(value);
                match self.rebuilt_child(child, node) {
                    Some(rebuilt) => refs[side] = rebuilt.into(),
                    None => { pending = Some(Frame::new(child, node)); break; }
                }
            }
            if let Some(child) = pending {
                lim.try_push(&mut stack, child)?;
                continue;
            }
            let frame = stack.last_mut().expect("the current frame exists");
            frame.next_pair += 1;
            if refs.iter().all(|&r| r != ZERO.into()) {
                lim.try_push(&mut frame.pairs, ChildPair::new(refs[0], refs[1]))?;
            }
        }
        Ok(self.rebuilt_child(v, local).expect("the root has been rebuilt"))
    }
}

#[cfg(test)]
#[path = "tests/rebuild.rs"]
mod tests;
