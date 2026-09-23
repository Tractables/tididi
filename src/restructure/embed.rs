//! Copy a diagram onto a larger vtree under a renaming of its variables.
//!
//! The destination must contain the source vtree's shape under the renaming.
//! A destination node with renamed variables on both sides is the image of a
//! source node and receives that source level unchanged, node indices and all.
//! A node with renamed variables on one side only passes that side's nodes
//! through, one node per node at the same index, so the copied level above
//! still finds each child where it left it. A subtree the renaming does not
//! reach becomes constant true. Nothing is applied.

use std::sync::Arc;

use super::GraftError;

use crate::Engine;
use crate::diagram::{
    Assembly, ChildPair, LevelView, NodeIdx, Tdd, TddBuilder, TddNodeId, LEAF_WIDTH, ONE_LEAF_IDX,
};
use crate::limits::{Limits, OperationError};
use crate::reduce::ReductionPlan;
use crate::vtree::{VarId, Vtree, VtreeError, VtreeIdx};

/// Where each level of an embedded diagram landed in the destination vtree.
///
/// [`Tdd::embed`] returns one beside the result. Index it by a node of the
/// source diagram's vtree to find the level holding its copy in the result.
/// Use this map to relocate side tables keyed by source vtree index.
#[derive(Clone, Debug)]
pub struct Embedding {
    /// The destination node each source node was copied to, indexed by the
    /// source node's index.
    levels: Vec<VtreeIdx>,
}

impl Embedding {
    /// The destination level that `source`'s level was copied to.
    ///
    /// # Panics
    ///
    /// Panics if `source` is not a node of the source diagram's vtree.
    #[must_use]
    pub fn level_of(&self, source: VtreeIdx) -> VtreeIdx {
        self.levels[source.idx()]
    }

    /// The whole map, indexed by the source vtree's node index.
    #[must_use]
    pub fn as_slice(&self) -> &[VtreeIdx] {
        &self.levels
    }
}

impl Tdd {
    /// This diagram on a larger vtree, with each of its variables renamed
    /// through `map` and every other variable of `into` left free.
    ///
    /// `map` must be injective, and `into` must contain a copy of this
    /// diagram's vtree under it: restricting `into` to the renamed variables,
    /// as [`Vtree::project_to_vars`] does, must give [`vtree`](Self::vtree)'s
    /// shape with each leaf renamed. `into` may group the renamed variables
    /// into a subtree of its own, or spread them along a spine with free
    /// leaves in between; what it may not do is split or regroup them.
    ///
    /// The levels are copied: no apply runs, the result shares `into`, and it
    /// is canonical when this diagram is. The cost is the size of `into` plus
    /// the size of this diagram plus, at each destination level with renamed
    /// variables on one side only, the width of that populated side. Structural
    /// diagrams only; any weights are dropped.
    ///
    /// The returned [`Embedding`] says which level of the result each level of
    /// this diagram became. Runs on `into`'s execution context; use
    /// [`Engine::embed`] inside a batch with resource limits.
    ///
    /// # Errors
    ///
    /// [`GraftError::NotIsomorphic`] when the shapes do not match, naming the
    /// node of this diagram's vtree where the match failed;
    /// [`GraftError::VariableOutOfRange`] when a renamed variable is not a leaf
    /// of `into`; [`GraftError::Vtree`] with
    /// [`VtreeError::OverlappingVariable`] when two variables share an image;
    /// [`GraftError::Operation`] with [`OperationError::MarginalLevel`] for a
    /// diagram that has discarded the structure at a level, and for a refused
    /// allocation or an armed stop.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{and, Tdd, Vtree};
    /// use tididi::vtree::VarId;
    ///
    /// // One relation over a variable pair, compiled once and placed twice.
    /// let pair = Arc::new(Vtree::linear(2));
    /// let r = Tdd::clause(&pair, [1, 2])?;                   // x1 ∨ x2
    ///
    /// let wide = Arc::new(Vtree::linear(4));
    /// let (first, _) = r.embed(&wide, |v| v)?;               // x1 ∨ x2
    /// let (second, levels) = r.embed(&wide, |v| VarId(v.0 + 2))?;  // x3 ∨ x4
    /// assert_eq!(second.model_count()?, 12u32.into());       // 3 · 2^2
    ///
    /// let both = and(first, second)?;
    /// assert_eq!(both.model_count()?, 9u32.into());          // 3 · 3
    ///
    /// // The second placement put the relation's root level on the node of
    /// // `wide` that groups x3 with x4.
    /// let x3 = wide.leaf_of(VarId(3)).unwrap();
    /// assert_eq!(levels.level_of(r.vtree().root()), wide.node(x3).parent().unwrap());
    /// # tididi::test_helpers::assert_canonical(&both);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn embed(
        &self,
        into: &Arc<Vtree>,
        map: impl Fn(VarId) -> VarId,
    ) -> Result<(Tdd, Embedding), GraftError> {
        into.context().run(|eng| eng.embed(self, into, map))
    }
}

impl Engine {
    /// [`Tdd::embed`] under this batch's scratch and resource limits.
    ///
    /// # Errors
    ///
    /// As [`Tdd::embed`].
    pub fn embed(
        &self,
        tdd: &Tdd,
        into: &Arc<Vtree>,
        map: impl Fn(VarId) -> VarId,
    ) -> Result<(Tdd, Embedding), GraftError> {
        let lim = self.limits();
        let _op = lim.begin_operation();
        lim.check_stop()?;
        let plan = Plan::build(lim, tdd, into, map)?;
        let result = assemble(self, tdd, into, &plan)?;
        Ok((result, Embedding { levels: plan.embedding }))
    }
}

/// What each destination node does in the copy, and where each source level went.
struct Plan {
    /// Whether each destination subtree contains a renamed source variable.
    mapped: Vec<bool>,
    /// The source node a destination node is the image of, where there is one.
    covered_by: Vec<Option<VtreeIdx>>,
    /// The destination node each source node maps to.
    embedding: Vec<VtreeIdx>,
}

impl Plan {
    /// Check the renaming against both trees and record the correspondence.
    ///
    /// Matches the two trees top-down: a destination node with renamed
    /// variables on one side only is a pass-through and the walk descends
    /// into the other side without advancing the source, which is exactly the
    /// node [`Vtree::project_to_vars`] splices out. Every other node must
    /// correspond to the current source node. `O(nodes of into)`.
    fn build(
        lim: &Limits,
        tdd: &Tdd,
        into: &Vtree,
        map: impl Fn(VarId) -> VarId,
    ) -> Result<Plan, GraftError> {
        if let Some(level) = tdd.levels().iter().position(|level| level.is_marginal()) {
            return Err(OperationError::MarginalLevel(VtreeIdx(level as u32)).into());
        }
        let source = tdd.vtree();
        let mut mapped = Vec::new();
        lim.try_resize(&mut mapped, into.num_nodes(), false)?;
        let mut embedding = Vec::new();
        lim.try_resize(&mut embedding, source.num_nodes(), into.root())?;
        let mut gate = lim.gate();
        for (leaf, var) in source.leaf_bottomup() {
            gate.poll(1)?;
            let image = map(var);
            let target = into.leaf_of(image).ok_or(GraftError::VariableOutOfRange {
                variable: image,
                num_vars: into.num_vars(),
            })?;
            if mapped[target.idx()] {
                return Err(VtreeError::OverlappingVariable(image).into());
            }
            mapped[target.idx()] = true;
            embedding[leaf.idx()] = target;
        }
        for (t, left, right) in into.internal_bottomup() {
            gate.poll(1)?;
            mapped[t.idx()] = mapped[left.idx()] || mapped[right.idx()];
        }

        let mut covered_by = Vec::new();
        lim.try_resize(&mut covered_by, into.num_nodes(), None)?;
        let mut stack = Vec::new();
        lim.try_push(&mut stack, (into.root(), source.root()))?;
        while let Some((d, s)) = stack.pop() {
            gate.poll(1)?;
            if !into.node(d).is_leaf() {
                let (left, right) = into.children(d);
                if !mapped[left.idx()] || !mapped[right.idx()] {
                    let carries = if !mapped[left.idx()] { right } else { left };
                    lim.try_push(&mut stack, (carries, s))?;
                    continue;
                }
            }
            match (into.node(d).is_leaf(), source.node(s).is_leaf()) {
                // A source leaf meets the destination leaf carrying its image.
                (true, true) if embedding[s.idx()] == d => {}
                (false, false) => {
                    let (left, right) = into.children(d);
                    let (source_left, source_right) = source.children(s);
                    lim.try_push(&mut stack, (left, source_left))?;
                    lim.try_push(&mut stack, (right, source_right))?;
                }
                _ => return Err(GraftError::NotIsomorphic { source: s }),
            }
            embedding[s.idx()] = d;
            covered_by[d.idx()] = Some(s);
        }
        debug_assert_eq!(
            covered_by.iter().filter(|source| source.is_some()).count(),
            source.num_nodes(),
            "a completed match gives every source level an image",
        );
        gate.flush()?;
        Ok(Plan { mapped, covered_by, embedding })
    }
}

/// Fill the destination's levels, then drop the pass-through nodes no copied
/// level names.
fn assemble(
    eng: &Engine,
    tdd: &Tdd,
    into: &Arc<Vtree>,
    plan: &Plan,
) -> Result<Tdd, GraftError> {
    if tdd.is_zero() {
        return Ok(crate::build::constant_zero(eng, into));
    }
    let mut builder = Assembly::new(eng, into)?;
    let over_a_renamed_leaf = fill(eng, tdd, into, plan, &mut builder)?;
    // Index preservation carries the source's output index through the
    // pass-through levels above it, so the result is seated at the same index.
    let output = TddNodeId { vtree: into.root(), local: tdd.output().local };
    let mut result = builder.finish_checked(output).map_err(|error| GraftError::Operation(error.into()))?;
    if over_a_renamed_leaf {
        eng.reduce(&mut result, ReductionPlan::Prune)?;
    }
    Ok(result)
}

/// One level per destination node, children before parents. Returns whether
/// a pass-through level sat directly over a renamed leaf, which is what the
/// prune after assembly is for.
fn fill(
    eng: &Engine,
    tdd: &Tdd,
    into: &Vtree,
    plan: &Plan,
    builder: &mut TddBuilder,
) -> Result<bool, GraftError> {
    let lim = eng.limits();
    let mut gate = lim.gate();
    let mut over_a_renamed_leaf = false;
    for t in into.bottomup() {
        if into.node(t).is_leaf() {
            continue;
        }
        gate.poll(1)?;
        let (left, right) = into.children(t);
        if !plan.mapped[t.idx()] {
            builder.push(eng, t, &[ChildPair::new(true_node(into, left), true_node(into, right))])?;
        } else if let Some(source) = plan.covered_by[t.idx()] {
            let view = LevelView::unweighted(tdd.level(source))
                .expect("a diagram with no marginal level has no weighted level");
            builder.replace_level(eng, t, view)?;
        } else {
            let free_is_left = !plan.mapped[left.idx()];
            let (free, carries) = if free_is_left { (left, right) } else { (right, left) };
            let one = true_node(into, free);
            let carries_leaf = into.node(carries).is_leaf();
            let width = if carries_leaf { LEAF_WIDTH } else { builder.level(carries).slot_count() };
            // A renamed leaf gets a pass-through node for each of its three
            // labels, so the copied level above finds the label it names at
            // that label's own index. It may name only some of them, which is
            // what the prune after assembly is for.
            over_a_renamed_leaf |= carries_leaf;
            for i in 0..width {
                let child = NodeIdx(i as u32);
                let pair = if free_is_left {
                    ChildPair::new(one, child)
                } else {
                    ChildPair::new(child, one)
                };
                builder.push(eng, t, &[pair])?;
            }
        }
    }
    gate.flush()?;
    Ok(over_a_renamed_leaf)
}

/// The index of the constant-true node on `child`'s level: the `One` label on
/// a leaf level, and the one node a free internal level holds otherwise.
fn true_node(vtree: &Vtree, child: VtreeIdx) -> NodeIdx {
    if vtree.node(child).is_leaf() { ONE_LEAF_IDX } else { NodeIdx(0) }
}

#[cfg(test)]
#[path = "tests/embed/mod.rs"]
mod tests;
