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

use super::{GraftError, placement::CopyPlacement};

use crate::Engine;
use crate::diagram::Tdd;
use crate::limits::{Limits, OperationError};
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

/// A validated placement of a source vtree into a destination vtree.
///
/// Prepare once with [`Self::new`], then [`Self::apply`] to any structural
/// circuit sharing the source vtree allocation. Variable renaming and shape
/// validation happen once; each application only copies and prunes the diagram.
/// Both vtrees are retained, so their structure cannot change behind the plan.
/// The placement and weight semantics are those of [`Tdd::embed`].
///
/// ```
/// use std::sync::Arc;
/// use tididi::{Tdd, Vtree};
/// use tididi::vtree::VarId;
/// use tididi::restructure::EmbeddingPlan;
/// let small = Arc::new(Vtree::linear(2));
/// let wide = Arc::new(Vtree::linear(4));
/// let place = EmbeddingPlan::new(&small, &wide, |v| VarId(v.0 + 2))?;
/// for literals in [[1, 2], [-1, 2]] {
///     let circuit = Tdd::clause(&small, literals)?;
///     let copy = place.apply(&circuit)?;
///     assert_eq!(copy.model_count()?, 12u32.into());
///     # tididi::test_helpers::assert_canonical(&circuit);
///     # tididi::test_helpers::assert_canonical(&copy);
/// }
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug)]
pub struct EmbeddingPlan {
    source: Arc<Vtree>,
    destination: Arc<Vtree>,
    layout: Plan,
}

impl EmbeddingPlan {
    /// Validate an injective, shape-preserving renaming, using the destination context.
    ///
    /// Returns the variable, shape and resource errors of [`Tdd::embed`].
    /// `map` is called once per source variable, during construction only.
    pub fn new(source: &Arc<Vtree>, destination: &Arc<Vtree>, map: impl Fn(VarId) -> VarId) -> Result<Self, GraftError> {
        destination.context().run(|eng| eng.embedding_plan(source, destination, map))
    }

    /// Source vtree allocation required by [`Self::apply`].
    pub fn source(&self) -> &Arc<Vtree> { &self.source }

    /// Destination vtree shared by every result.
    pub fn destination(&self) -> &Arc<Vtree> { &self.destination }

    /// Destination level for each source level, for relocating side tables.
    pub fn levels(&self) -> &Embedding { &self.layout.embedding }

    /// Place a circuit using this plan, leaving the source unchanged.
    ///
    /// Runs on the destination's context. A different source vtree allocation
    /// returns [`OperationError::VtreeMismatch`] wrapped in [`GraftError::Operation`].
    /// Other errors and weight semantics are those of [`Tdd::embed`].
    pub fn apply(&self, circuit: &Tdd) -> Result<Tdd, GraftError> {
        self.destination.context().run(|eng| eng.embed_with(circuit, self))
    }

    /// Compose this placement with a placement of its destination.
    ///
    /// The result places the original source directly into `next`'s destination,
    /// without building an intermediate circuit. Intermediate free variables
    /// remain free. `self.destination()` and `next.source()` must share an
    /// allocation, otherwise this returns [`OperationError::VtreeMismatch`].
    /// Construction uses the final destination context and can return the
    /// resource errors of [`Self::new`].
    pub fn then(&self, next: &Self) -> Result<Self, GraftError> {
        next.destination.context().run(|eng| eng.compose_embeddings(self, next))
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
    /// For repeated placements, prepare an [`EmbeddingPlan`].
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
    /// Prepare an [`EmbeddingPlan`] under this engine's limits.
    pub fn embedding_plan(&self, source: &Arc<Vtree>, destination: &Arc<Vtree>, map: impl Fn(VarId) -> VarId) -> Result<EmbeddingPlan, GraftError> {
        let lim = self.limits();
        let _op = lim.begin_operation();
        lim.check_stop()?;
        let layout = Plan::build(lim, source, destination, map)?;
        Ok(EmbeddingPlan { source: source.clone(), destination: destination.clone(), layout })
    }

    /// [`EmbeddingPlan::apply`] under this engine's limits.
    pub fn embed_with(&self, circuit: &Tdd, plan: &EmbeddingPlan) -> Result<Tdd, GraftError> {
        let _op = self.limits().begin_operation();
        self.limits().check_stop()?;
        if !Arc::ptr_eq(circuit.vtree(), &plan.source) { return Err(OperationError::VtreeMismatch.into()); }
        circuit.require_structure()?;
        assemble(self, circuit, &plan.destination, &plan.layout)
    }

    /// [`EmbeddingPlan::then`] under this engine's limits.
    pub fn compose_embeddings(&self, first: &EmbeddingPlan, next: &EmbeddingPlan) -> Result<EmbeddingPlan, GraftError> {
        if !Arc::ptr_eq(&first.destination, &next.source) { return Err(OperationError::VtreeMismatch.into()); }
        self.embedding_plan(&first.source, &next.destination, |var| {
            let leaf = first.source.leaf_of(var).expect("source leaf");
            let intermediate = first.levels().level_of(leaf);
            next.destination.leaf_var(next.levels().level_of(intermediate))
        })
    }

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
        tdd.require_structure()?;
        let plan = Plan::build(lim, tdd.vtree(), into, map)?;
        let result = assemble(self, tdd, into, &plan)?;
        Ok((result, plan.embedding))
    }
}

/// What each destination node does in the copy, and where each source level went.
#[derive(Debug)]
struct Plan {
    /// Whether each destination subtree contains a renamed source variable.
    mapped: Vec<bool>,
    /// The source node a destination node is the image of, where there is one.
    covered_by: Vec<Option<VtreeIdx>>,
    /// The destination node each source node maps to.
    embedding: Embedding,
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
        source: &Vtree,
        into: &Vtree,
        map: impl Fn(VarId) -> VarId,
    ) -> Result<Plan, GraftError> {
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
        Ok(Plan { mapped, covered_by, embedding: Embedding { levels: embedding } })
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
    let mut placement = CopyPlacement::new(eng, into)?;
    let mut gate = eng.limits().gate();
    for t in into.bottomup() {
        if into.node(t).is_leaf() { continue; }
        gate.poll(1)?;
        let (left, right) = into.children(t);
        if !plan.mapped[t.idx()] {
            placement.join(t, placement.true_node(left), placement.true_node(right))?;
        } else if let Some(source) = plan.covered_by[t.idx()] {
            placement.copy_level(tdd, source, t)?;
        } else {
            placement.pass_through(t, !plan.mapped[left.idx()])?;
        }
    }
    gate.flush()?;
    Ok(placement.finish(tdd.output().local)?)
}

#[cfg(test)]
#[path = "tests/embed/mod.rs"]
mod tests;
