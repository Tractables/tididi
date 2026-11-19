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

use super::{
    EmbedError, EmbedRefused, Embedding,
    graft::check_part_weights,
    placement::MovePlacement,
};

use crate::Engine;
use crate::diagram::{
    ChildSide, LeafLabel, NEG_LEAF_IDX, NodeIdx, ONE_LEAF_IDX, POS_LEAF_IDX, Tdd, WeightStore,
    WeightValue, return_levels,
};
use crate::limits::OperationError;
use crate::vtree::{VarId, Vtree, VtreeError, VtreeIdx};

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
    pub fn new(
        source: &Arc<Vtree>,
        destination: &Arc<Vtree>,
        map: impl Fn(VarId) -> VarId,
    ) -> Result<Self, EmbedError> {
        destination
            .context()
            .run(|eng| eng.embedding_plan(source, destination, map))
    }

    /// Source vtree allocation required by [`Self::apply`].
    pub fn source(&self) -> &Arc<Vtree> {
        &self.source
    }

    /// Destination vtree shared by every result.
    pub fn destination(&self) -> &Arc<Vtree> {
        &self.destination
    }

    /// Destination level for each source level, for relocating side tables.
    pub fn levels(&self) -> &Embedding {
        &self.layout.embedding
    }

    /// Place a circuit using this plan, leaving the source unchanged.
    ///
    /// Runs on the destination's context. A different source vtree allocation
    /// returns [`OperationError::VtreeMismatch`] wrapped in [`EmbedError::Operation`].
    /// Other errors and weight semantics are those of [`Tdd::embed`].
    pub fn apply(&self, circuit: &Tdd) -> Result<Tdd, EmbedError> {
        self.destination
            .context()
            .run(|eng| eng.embed_with(circuit, self))
    }

    /// Compose this placement with a placement of its destination.
    ///
    /// The result places the original source directly into `next`'s destination,
    /// without building an intermediate circuit. Intermediate free variables
    /// remain free. `self.destination()` and `next.source()` must share an
    /// allocation, otherwise this returns [`OperationError::VtreeMismatch`].
    /// Construction uses the final destination context and can return the
    /// resource errors of [`Self::new`].
    pub fn then(&self, next: &Self) -> Result<Self, EmbedError> {
        next.destination
            .context()
            .run(|eng| eng.compose_embeddings(self, next))
    }
}

impl Tdd {
    /// This diagram on a larger vtree, with each of its variables renamed
    /// through `map` and every other variable of `into` left free.
    ///
    /// `map` must be injective, and `into` must contain a copy of this
    /// diagram's vtree under it: restricting `into` to the renamed variables,
    /// as [`Vtree::project_to_vars`] does, must give [`vtree`](Self::vtree)'s
    /// shape with each leaf renamed, **up to mirrors**: the children of any
    /// node may come in the other order. `into` may group the renamed
    /// variables into a subtree of its own, or spread them along a spine with
    /// free leaves in between; what it may not do is split or regroup them.
    ///
    /// The levels are copied: no apply runs, the result shares `into`, and it
    /// is canonical when this diagram is. A level whose image has its
    /// children the other way round is copied with the two sides of every
    /// pair exchanged, which denotes the same function over the mirrored
    /// node, since a level's nodes are classes of assignments to the node's
    /// variables whichever child is called left; a binary relation's diagram
    /// placed with its two blocks the other way round is its transpose's.
    /// The cost is the size of `into` plus
    /// the size of this diagram plus, at each destination level with renamed
    /// variables on one side only, the width of that populated side. Structural
    /// diagrams only; any weights are dropped. [`Engine::embed_over`] places a
    /// diagram that has summed levels out.
    ///
    /// The returned [`Embedding`] says which level of the result each level of
    /// this diagram became. Runs on `into`'s execution context; use
    /// [`Engine::embed`] inside a batch with resource limits.
    /// For repeated placements, prepare an [`EmbeddingPlan`].
    ///
    /// # Errors
    ///
    /// [`EmbedError::NotIsomorphic`] when the shapes do not match, naming the
    /// node of this diagram's vtree where the match failed;
    /// [`EmbedError::VariableOutOfRange`] when a renamed variable is not a leaf
    /// of `into`; [`EmbedError::Vtree`] with
    /// [`VtreeError::OverlappingVariable`] when two variables share an image;
    /// [`EmbedError::UnsummedRegion`] when `into` holds a region this
    /// diagram has not summed out;
    /// [`EmbedError::Operation`] with [`OperationError::MarginalLevel`] for a
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
    ///
    /// // x1 ∧ ¬x2 placed with x1 on the right leaf: the level is mirrored.
    /// let f = Tdd::cube(&pair, [1, -2])?;
    /// let into = Arc::new(Vtree::linear(2));
    /// let (g, _) = f.embed(&into, |v| VarId(3 - v.0))?;
    /// assert!(g.equivalent(&Tdd::cube(&into, [2, -1])?)?);
    /// # tididi::test_helpers::assert_canonical(&both);
    /// # tididi::test_helpers::assert_canonical(&g);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn embed(
        &self,
        into: &Arc<Vtree>,
        map: impl Fn(VarId) -> VarId,
    ) -> Result<(Tdd, Embedding), EmbedError> {
        into.context().run(|eng| eng.embed(self, into, map))
    }

}

impl Engine {
    /// Prepare an [`EmbeddingPlan`] under this engine's limits.
    pub fn embedding_plan(
        &self,
        source: &Arc<Vtree>,
        destination: &Arc<Vtree>,
        map: impl Fn(VarId) -> VarId,
    ) -> Result<EmbeddingPlan, EmbedError> {
        let _op = self.limits().enter()?;
        let layout = Plan::build(self, source, destination, map, true, Free::Build)?;
        Ok(EmbeddingPlan {
            source: source.clone(),
            destination: destination.clone(),
            layout,
        })
    }

    /// [`EmbeddingPlan::apply`] under this engine's limits.
    pub fn embed_with(&self, circuit: &Tdd, plan: &EmbeddingPlan) -> Result<Tdd, EmbedError> {
        let _op = self.limits().enter()?;
        if !Arc::ptr_eq(circuit.vtree(), &plan.source) {
            return Err(OperationError::VtreeMismatch.into());
        }
        circuit.require_structure()?;
        assemble(self, circuit, &plan.destination, &plan.layout)
    }

    /// [`EmbeddingPlan::then`] under this engine's limits.
    pub fn compose_embeddings(
        &self,
        first: &EmbeddingPlan,
        next: &EmbeddingPlan,
    ) -> Result<EmbeddingPlan, EmbedError> {
        let _op = self.limits().enter()?;
        if !Arc::ptr_eq(&first.destination, &next.source) {
            return Err(OperationError::VtreeMismatch.into());
        }
        self.embedding_plan(
            &first.source,
            &next.destination,
            |var| {
                let leaf = first.source.leaf_of(var).expect("source leaf");
                let intermediate = first.levels().level_of(leaf);
                next.destination
                    .leaf_var(next.levels().level_of(intermediate))
            },
        )
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
    ) -> Result<(Tdd, Embedding), EmbedError> {
        let _op = self.limits().enter()?;
        tdd.require_structure()?;
        let plan = Plan::build(self, tdd.vtree(), into, map, true, Free::Build)?;
        let result = assemble(self, tdd, into, &plan)?;
        Ok((result, plan.embedding))
    }

    /// [`Engine::embed`] under its removed name, until callers move to it.
    #[doc(hidden)]
    pub fn embed_mirrored(
        &self,
        tdd: &Tdd,
        into: &Arc<Vtree>,
        map: impl Fn(VarId) -> VarId,
    ) -> Result<(Tdd, Embedding), EmbedError> {
        self.embed(tdd, into, map)
    }

    /// [`Engine::embedding_plan`] under its removed name, until callers move
    /// to it.
    #[doc(hidden)]
    pub fn embedding_plan_mirrored(
        &self,
        source: &Arc<Vtree>,
        destination: &Arc<Vtree>,
        map: impl Fn(VarId) -> VarId,
    ) -> Result<EmbeddingPlan, EmbedError> {
        self.embedding_plan(source, destination, map)
    }

    /// [`Engine::embed`] that moves the diagram's levels into the result
    /// instead of copying them, and gives the diagram back unchanged when it
    /// is refused.
    ///
    /// The source's pairs are not touched, except that a level whose image
    /// has its children the other way round has the two sides of every pair
    /// exchanged in place: the cost is the size of `into` plus, at each
    /// destination level with renamed variables on one side only, the width
    /// of that populated side, the pairs of a level that reads a single
    /// variable through such levels, and the pairs of each mirrored level. The result is the
    /// diagram [`Engine::embed`] returns, up to the numbering of its nodes and
    /// a node for a literal of such a variable that nothing reads, which
    /// [`Engine::minimize`] removes. The levels move into storage the engine
    /// supplies and are not charged again; the level array and every level
    /// the embedding builds are charged to the engine's limits.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{Engine, Vtree};
    /// use tididi::vtree::VarId;
    ///
    /// let engine = Engine::new();
    /// let pair = Arc::new(Vtree::linear(2));
    /// let wide = Arc::new(Vtree::linear(4));
    /// let r = engine.clause(&pair, [1, 2])?;
    /// let (copied, _) = engine.embed(&r, &wide, |v| VarId(v.0 + 2))?;
    /// let (mut moved, _) = engine.embed_moving(r, &wide, |v| VarId(v.0 + 2)).map_err(|r| r.error)?;
    /// engine.minimize(&mut moved)?;
    /// assert!(engine.equivalent(&moved, &copied)?);
    /// assert_eq!(engine.model_count(&moved)?, 12u32.into());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Those of [`Tdd::embed`], each with the diagram as it was given.
    #[expect(
        clippy::result_large_err,
        reason = "the refusal hands back what it was given"
    )]
    pub fn embed_moving(
        &self,
        tdd: Tdd,
        into: &Arc<Vtree>,
        map: impl Fn(VarId) -> VarId,
    ) -> Result<(Tdd, Embedding), EmbedRefused> {
        let _op = match self.limits().enter() {
            Ok(op) => op,
            Err(e) => {
                return Err(EmbedRefused {
                    error: e.into(),
                    tdd,
                });
            }
        };
        place_moving(self, tdd, into, map, Free::Build)
            .map(|(result, plan)| (result, plan.embedding))
    }

    /// [`Tdd::embed`] for a diagram whose levels may hold weighted marginal
    /// values, with `weights` as the result's store.
    ///
    /// A weight-marginal level ([`Tdd::marginalize_levels`] under a
    /// [`WeightStore`]) is placed with its values, at the level `map` sends
    /// it to. The result sums over the same assignments as the source and
    /// over every free variable of `into`, so a variable `into` adds below
    /// the image of a marginal level multiplies that level's values by the
    /// variable's total weight (the sum of its two literal weights), and a level `into` adds below it holds no
    /// values of its own. A structural diagram is placed as by
    /// [`Tdd::embed`] and takes `weights` as its store when one is given;
    /// a diagram with a marginal level needs one, whose table must give
    /// each renamed variable the weights the source's gives the original,
    /// and needs `into` to hold its vtree's shape with every node's children
    /// in their order, as no level of it is mirrored.
    ///
    /// Copies the levels and their weight columns into storage reserved
    /// through the engine's limits, so a copy the budget or the allocator
    /// refuses comes back as [`OperationError::OverBudget`] instead of
    /// aborting; the reduction that seats a marginal level under a new
    /// parent runs under them as well.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use num_rational::BigRational;
    /// use tididi::{Engine, Tdd, Vtree};
    /// use tididi::diagram::{Arithmetic, LiteralWeights, RationalWeights, WeightStore};
    /// use tididi::vtree::VarId;
    ///
    /// let third = || BigRational::new(1.into(), 3.into());
    /// let table = |n| RationalWeights::from_literals(&vec![
    ///     LiteralWeights { negative: third(), positive: third() }; n
    /// ]);
    /// let engine = Engine::new();
    /// let small = Arc::new(Vtree::balanced(2));
    /// let mut f = engine.clause(&small, [1, 2])?;
    /// f.set_weights(WeightStore::new(table(2), Arithmetic::ExactRational))?;
    /// engine.marginalize_levels(&mut f, &[small.root()])?;
    /// let value = f.weighted_value()?.expect("weighted").into_rational();
    ///
    /// // On a larger vtree the free variables 2 and 4 each weigh 2/3.
    /// let big = Arc::new(Vtree::linear(4));
    /// let store = WeightStore::new(table(4), Arithmetic::ExactRational);
    /// let (g, _) = engine.embed_over(&f, &big, |v| VarId(2 * v.0 - 1), Some(store))?;
    /// let free = BigRational::new(4.into(), 9.into());
    /// assert_eq!(g.weighted_value()?.expect("weighted").into_rational(), value * free);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Those of [`Tdd::embed`], with [`OperationError::MarginalLevel`] naming
    /// a level that holds integer counts rather than weighted values;
    /// [`EmbedError::SourceWeights`] for a weight-marginal level without
    /// `weights` or a table that disagrees with the source's under the
    /// renaming; [`EmbedError::DestinationWeights`] for a table that does
    /// not cover `into`'s variables; [`OperationError::OverBudget`] where
    /// the copy is refused.
    pub fn embed_over(
        &self,
        tdd: &Tdd,
        into: &Arc<Vtree>,
        map: impl Fn(VarId) -> VarId,
        weights: Option<WeightStore>,
    ) -> Result<(Tdd, Embedding), EmbedError> {
        let _op = self.limits().enter()?;
        let plan = Plan::build(self, tdd.vtree(), into, &map, !tdd.has_marginal_level(), Free::Build)?;
        // Integer counts reach a parent through inline references, which
        // do not survive a new parent; only weighted values are placed.
        if let Some(t) = tdd
            .vtree()
            .bottomup()
            .find(|&t| tdd.level(t).is_marginal() && !tdd.level(t).is_weight_marginal())
        {
            return Err(OperationError::MarginalLevel(t).into());
        }
        if let Some(store) = &weights {
            store
                .check_variables(into.leaf_bottomup().map(|(_, var)| var))
                .map_err(EmbedError::DestinationWeights)?;
        }
        check_part_weights(tdd, weights.as_ref(), &map).map_err(EmbedError::SourceWeights)?;
        let result = if tdd.has_marginal_level() {
            let store = weights.expect("a marginal level was checked to have a destination store");
            assemble_marginal(self, tdd, into, &plan, store)?
        } else {
            let mut result = assemble(self, tdd, into, &plan)?;
            if let Some(store) = weights {
                result.set_weights(store).map_err(OperationError::from)?;
            }
            result
        };
        Ok((result, plan.embedding))
    }
}

/// What each destination node does in the copy, and where each source level went.
#[derive(Debug)]
pub(crate) struct Plan {
    /// Per destination node, what its level is.
    targets: Vec<Target>,
    /// The destination node each source node maps to.
    pub(crate) embedding: Embedding,
    /// Whether any image is mirrored.
    mirrors: bool,
    /// The number of pass-throughs.
    pub(crate) pass_throughs: usize,
}

/// What a destination node's level is under a [`Plan`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Target {
    /// No renamed source variable is under the node: a free level, constant
    /// true.
    Free,
    /// The image of `source`; `mirrored` when its children are the source
    /// node's swapped, so the level is copied with every pair read the other
    /// way round.
    Image { source: VtreeIdx, mirrored: bool },
    /// An internal node with renamed variables on one side only, carrying
    /// that side's source node.
    PassThrough(VtreeIdx),
}

impl Plan {
    /// Whether no renamed source variable is under destination node `t`.
    #[inline]
    pub(crate) fn is_free(&self, t: VtreeIdx) -> bool {
        self.targets[t.idx()] == Target::Free
    }

    /// The source node destination node `t` is the image of, if any.
    #[inline]
    pub(crate) fn covered_by(&self, t: VtreeIdx) -> Option<VtreeIdx> {
        match self.targets[t.idx()] {
            Target::Image { source, .. } => Some(source),
            _ => None,
        }
    }

    /// Whether destination node `t` is the image of a source node with its
    /// children swapped.
    #[inline]
    pub(crate) fn is_mirrored(&self, t: VtreeIdx) -> bool {
        matches!(self.targets[t.idx()], Target::Image { mirrored: true, .. })
    }

    /// The destination nodes that are mirrored images.
    pub(crate) fn mirrored_images(&self) -> impl Iterator<Item = VtreeIdx> + '_ {
        let images: &[VtreeIdx] = if self.mirrors { &self.embedding.levels } else { &[] };
        images.iter().copied().filter(|&t| self.is_mirrored(t))
    }

    /// The pass-throughs of the plan, `into`'s internal nodes with renamed
    /// variables on one side only, bottom-up.
    pub(crate) fn pass_throughs<'a>(&'a self, into: &'a Vtree) -> impl Iterator<Item = &'a VtreeIdx> {
        into.internal_bottomup_slice()
            .iter()
            .filter(|&&t| matches!(self.targets[t.idx()], Target::PassThrough(_)))
            .take(self.pass_throughs)
    }

    /// Check the renaming against both trees and record the correspondence.
    ///
    /// Matches the two trees in one bottom-up pass over `into`, carrying to
    /// each node the source node whose leaves are the renamed variables under
    /// it. A node with renamed variables on one side only is a pass-through
    /// carrying that side's source node, which is exactly the node
    /// [`Vtree::project_to_vars`] splices out. A node with renamed variables
    /// on both sides is the image of the parent of the two source nodes its
    /// children carry, which must be siblings, and is mirrored when the left
    /// child carries the right sibling, which an exact match refuses.
    /// `O(nodes of into)`.
    ///
    /// The work clock is charged one unit for each source leaf, one for each
    /// internal node of `into`, and one for each internal node of `into` with
    /// a renamed variable under it, in bottom-up order up to the one the
    /// match fails at, that one included.
    ///
    /// `mirror` is unset where the assembly cannot mirror a level: a
    /// marginal one, or a projection's. With [`Free::Build`], a region of
    /// `into` that is not the image of one of `source`'s is refused
    /// ([`EmbedError::UnsummedRegion`]).
    pub(crate) fn build(
        eng: &Engine,
        source: &Vtree,
        into: &Vtree,
        map: impl Fn(VarId) -> VarId,
        mirror: bool,
        free: Free,
    ) -> Result<Plan, EmbedError> {
        let lim = eng.limits();
        let (mut targets, mut embedding) = (Vec::new(), Vec::new());
        lim.try_resize(&mut targets, into.num_nodes(), Target::Free)?;
        lim.try_resize(&mut embedding, source.num_nodes(), into.root())?;
        let mut gate = lim.gate();
        for (leaf, var) in source.leaf_bottomup() {
            gate.poll(1)?;
            let image = map(var);
            let target = into.leaf_of(image).ok_or(EmbedError::VariableOutOfRange {
                variable: image,
                num_vars: into.num_vars(),
            })?;
            if targets[target.idx()] != Target::Free {
                return Err(VtreeError::OverlappingVariable(image).into());
            }
            embedding[leaf.idx()] = target;
            targets[target.idx()] = Target::Image { source: leaf, mirrored: false };
        }
        if source.has_regions() {
            map_regions(source, into, &mut targets, &mut embedding)?;
        }
        if free == Free::Build && into.has_regions() {
            refuse_free_region(into, &targets)?;
        }
        gate.poll_each(into.internal_bottomup_slice().len() as u64)?;

        let carried = |target: Target| match target {
            Target::Free => None,
            Target::Image { source, .. } | Target::PassThrough(source) => Some(source),
        };
        let mut visited = source.num_leaves() as u64;
        let (mut mirrors, mut pass_throughs) = (false, 0);
        let mut refused = None;
        for &t in into.internal_bottomup_slice() {
            let (left, right) = into.children(t);
            targets[t.idx()] = match (carried(targets[left.idx()]), carried(targets[right.idx()])) {
                (None, None) => continue,
                (Some(s), None) | (None, Some(s)) => {
                    visited += 1;
                    pass_throughs += 1;
                    Target::PassThrough(s)
                }
                (Some(a), Some(b)) => {
                    visited += 1;
                    let p = source.node(a).parent().expect("a source node beside another is not the root");
                    let swapped = source.children(p).1 == a;
                    if source.node(b).parent() != Some(p) || (swapped && !mirror) {
                        refused = Some(p);
                        break;
                    }
                    embedding[p.idx()] = t;
                    mirrors |= swapped;
                    Target::Image { source: p, mirrored: swapped }
                }
            };
        }
        gate.poll_each(visited)?;
        if let Some(s) = refused {
            return Err(EmbedError::NotIsomorphic { source: s });
        }
        debug_assert_eq!(
            targets
                .iter()
                .filter(|target| matches!(target, Target::Image { .. }))
                .count(),
            source.num_nodes(),
            "a completed match gives every source level an image",
        );
        gate.flush()?;
        Ok(Plan {
            targets,
            embedding: Embedding { levels: embedding },
            mirrors,
            pass_throughs,
        })
    }
}

/// Map each region of `source` to its image in [`Plan::build`]: the region of
/// `into` at its anchor.
#[cold]
fn map_regions(source: &Vtree, into: &Vtree, targets: &mut [Target], embedding: &mut [VtreeIdx]) -> Result<(), EmbedError> {
    for region in source.regions() {
        let target = into
            .node_at_home(source.home_node(region))
            .filter(|&t| into.node(t).is_region() && into.region_at(t) == source.region_at(region))
            .ok_or(EmbedError::NotIsomorphic { source: region })?;
        targets[target.idx()] = Target::Image { source: region, mirrored: false };
        embedding[region.idx()] = target;
    }
    Ok(())
}

/// Refuse a region of `into` no source region maps to: the copy would have
/// to give it a value, and the diagram has none for variables it has not
/// summed out.
#[cold]
fn refuse_free_region(into: &Vtree, targets: &[Target]) -> Result<(), EmbedError> {
    match into.regions().find(|t| targets[t.idx()] == Target::Free) {
        Some(region) => Err(EmbedError::UnsummedRegion { region }),
        None => Ok(()),
    }
}

/// What [`place_moving`] does with the free levels, those of the destination
/// nodes no renamed variable is under.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Free {
    /// Build each: one node, true on both sides. A free region is refused.
    Build,
    /// Build each internal one and leave a free region empty, for an
    /// operation that takes another operand's region there.
    Operand,
    /// Leave each empty, for a conjunction that reads it as the level it
    /// stands for and takes the other operand's level there; a projection,
    /// whose free nodes are those it drops, builds none either.
    Leave,
}

/// [`Engine::embed_moving`] under the caller's entry, with the plan it placed
/// `tdd` by; with [`Free::Leave`], a diagram whose free levels are empty. A
/// false diagram is the false diagram on `into`, every level empty, whatever
/// `free` says.
#[expect(
    clippy::result_large_err,
    reason = "the refusal hands back what it was given"
)]
pub(crate) fn place_moving(
    eng: &Engine,
    tdd: Tdd,
    into: &Arc<Vtree>,
    map: impl Fn(VarId) -> VarId,
    free: Free,
) -> Result<(Tdd, Plan), EmbedRefused> {
    place(eng, std::borrow::Cow::Owned(tdd), into, map, free).map_err(|(error, tdd)| {
        EmbedRefused {
            error,
            tdd: tdd.into_owned(),
        }
    })
}

/// [`place_moving`] of an owned or borrowed diagram: a borrowed one has its
/// levels copied into the destination's and its weight store copied. On a
/// projection, a summed-out level may sit at a leaf: a region, or a summed
/// variable whose parent stays its parent. Refused, the diagram comes back
/// as it was given.
#[expect(
    clippy::result_large_err,
    reason = "the refusal hands back what it was given"
)]
pub(crate) fn place<'a>(
    eng: &Engine,
    tdd: std::borrow::Cow<'a, Tdd>,
    into: &Arc<Vtree>,
    map: impl Fn(VarId) -> VarId,
    free: Free,
) -> Result<(Tdd, Plan), (EmbedError, std::borrow::Cow<'a, Tdd>)> {
    if let Err(e) = require_placeable(&tdd) {
        return Err((e.into(), tdd));
    }
    let plan = match Plan::build(eng, tdd.vtree(), into, map, true, free) {
        Ok(plan) => plan,
        Err(error) => return Err((error, tdd)),
    };
    place_by(eng, tdd, into, &plan, free).map(|result| (result, plan))
}

/// Refuse a summed-out level [`place`] does not move: any on a root vtree,
/// an internal one on a projection.
fn require_placeable(tdd: &Tdd) -> Result<(), OperationError> {
    match tdd.vtree().home() {
        None => tdd.require_structure(),
        Some(_) if !tdd.has_marginal_level() => Ok(()),
        Some(_) => match tdd.vtree().bottomup().find(|&t| !tdd.vtree().node(t).is_leaf() && tdd.level(t).is_marginal()) {
            Some(t) => Err(OperationError::MarginalLevel(t)),
            None => Ok(()),
        },
    }
}

/// [`place`] by a `plan` built for `tdd`, a structural diagram or one whose
/// summed-out levels are leaves of a projection.
#[expect(
    clippy::result_large_err,
    reason = "the refusal hands back what it was given"
)]
pub(crate) fn place_by<'a>(
    eng: &Engine,
    tdd: std::borrow::Cow<'a, Tdd>,
    into: &Arc<Vtree>,
    plan: &Plan,
    free: Free,
) -> Result<Tdd, (EmbedError, std::borrow::Cow<'a, Tdd>)> {
    if tdd.is_zero() || tdd.is_one() {
        return match crate::build::constant_on(eng, into, tdd.is_one()) {
            Ok(mut constant) => {
                constant.weights = tdd.weights.as_ref().map(WeightStore::empty_like);
                Ok(constant)
            }
            Err(e) => Err((e.into(), tdd)),
        };
    }
    assemble_moving(eng, tdd, into, plan, free, true).map_err(|(e, tdd)| (e.into(), tdd))
}

/// Copy a structural diagram's levels onto the destination through
/// [`assemble_moving`], without its weights, and drop the pass-through nodes
/// over a leaf that no level names.
fn assemble(eng: &Engine, tdd: &Tdd, into: &Arc<Vtree>, plan: &Plan) -> Result<Tdd, EmbedError> {
    if tdd.is_zero() || tdd.is_one() {
        return Ok(crate::build::constant_on(eng, into, tdd.is_one())?);
    }
    let mut result = assemble_moving(eng, std::borrow::Cow::Borrowed(tdd), into, plan, Free::Build, false)
        .map_err(|(e, _)| e)?;
    // One unit per internal node and one per slot of its level, less the
    // units `assemble_moving` polled: a large diagram's copy, charged a unit
    // a level, would go a whole embedding without a test.
    let internal = into.internal_bottomup_slice();
    let built = internal.iter().filter(|&&t| plan.is_free(t)).count() + plan.pass_throughs;
    let units: u64 = internal
        .iter()
        .map(|&t| 1 + result.level(t).slot_count() as u64)
        .sum();
    let mut gate = eng.limits().gate();
    gate.poll(units.saturating_sub(built as u64))?;
    gate.flush()?;
    // Every internal level owes the contraction passes, as an assembly's seat
    // queues them, though the copies are as reduced as `tdd`: the vtree
    // search races its candidates by the work this queues, and was tuned on it.
    result.dirty = crate::diagram::Dirty::default().seeded(into, internal.iter().copied(), Some(eng))?;
    if plan.pass_throughs(into).any(|&t| {
        let (left, right) = into.children(t);
        into.node(if plan.is_free(left) { right } else { left }).is_leaf()
    }) {
        eng.reduce(&mut result, crate::reduce::ReductionPlan::Prune)?;
    }
    Ok(result)
}

/// Move the levels of a structural diagram onto the destination and build
/// what it adds; the diagram comes back, levels and worklists in place, when
/// the result's storage is refused. A borrowed diagram has its levels copied
/// into the destination's instead, and a refusal drops the copies. The
/// result takes a moved diagram's worklists, and with `weights` its weight
/// store, and copies of a borrowed one's, once nothing
/// is left to refuse.
#[expect(
    clippy::result_large_err,
    reason = "the refusal hands back what it was given"
)]
fn assemble_moving<'a>(
    eng: &Engine,
    mut tdd: std::borrow::Cow<'a, Tdd>,
    into: &Arc<Vtree>,
    plan: &Plan,
    free: Free,
    weights: bool,
) -> Result<Tdd, (OperationError, std::borrow::Cow<'a, Tdd>)> {
    use std::borrow::Cow;
    // Room for the inner pass's share of the leaf pass's worklist (below),
    // taken while a refusal has nothing to undo.
    if let Cow::Owned(moved) = &mut tdd
        && let Err(e) = moved.dirty.reserve_leaf_owes_inner(eng)
    {
        return Err((e, tdd));
    }
    let mut placement = match MovePlacement::new(eng, into, None) {
        Ok(placement) => placement,
        Err(e) => return Err((e, tdd)),
    };
    // Moving the levels out and back forgets what `tdd` is known to be: a
    // refusal takes it back with them. A copy leaves `tdd` as it is.
    let mut known = match &mut tdd {
        Cow::Owned(tdd) => {
            let known = tdd.levels.known();
            placement.move_part(tdd, &plan.embedding.levels);
            Some(known)
        }
        Cow::Borrowed(source) => {
            if let Err(e) = placement.copy_part(source, &plan.embedding.levels) {
                return Err((e, tdd));
            }
            None
        }
    };
    // Undo a refused placement: a moved diagram takes its levels back.
    let mut refuse = |e: OperationError, mut placement: MovePlacement<'_>, mut tdd: Cow<'a, Tdd>| {
        if let (Cow::Owned(source), Some(known)) = (&mut tdd, known.take()) {
            placement.mirror(plan);
            placement.move_back(source, &plan.embedding.levels);
            source.levels.reinstate_known(known);
        }
        (e, tdd)
    };
    // A mirrored image reads its pairs the other way round, before anything
    // reads its sides (`literal_chain`). A refusal undoes it by the same
    // call, so anything that reorders a mirrored level's pairs belongs on
    // the seated result, past the last refusal.
    placement.mirror(plan);
    let mut literal_tops =
        match build_embedded_levels(eng, into, plan, free, tdd.output().local, &mut placement) {
            Ok(tops) => tops,
            Err(e) => return Err(refuse(e, placement, tdd)),
        };
    let chains = !literal_tops.is_empty();
    // Room for the result's loose levels where `tdd`'s are known and some
    // may be listed, and a borrowed diagram's worklists copied with room
    // for the inner pass's share of the leaf pass's, taken before the seat:
    // `unseat` does not undo the renumbering below.
    let mut listed = Vec::new();
    let listing = tdd.dirty.loose().is_some_and(|loose| !loose.is_empty() || chains);
    let copied = match listing {
        true => eng.limits().reserve_exact(&mut listed, into.num_nodes()),
        false => Ok(()),
    }
    .and_then(|()| match &tdd {
        Cow::Owned(_) => Ok(None),
        Cow::Borrowed(source) => {
            source.dirty.clone_on(eng).and_then(|mut copied| copied.reserve_leaf_owes_inner(eng).map(|()| Some(copied)))
        }
    });
    let copied = match copied {
        Ok(copied) => copied,
        Err(e) => return Err(refuse(e, placement, tdd)),
    };
    // Seated owing nothing: `tdd` keeps its worklists until the last
    // refusal is behind, which leaves nothing to put back.
    let mut result = match placement.seat(tdd.output().local, crate::diagram::Dirty::default()) {
        Ok(result) => result,
        Err((e, placement)) => return Err(refuse(e, placement, tdd)),
    };
    // `Pos` and `Neg` are the chain's nodes 0 and 1; nothing reads `One`.
    // The renumbering is planned whole, so a refusal finds `result` as
    // seated and gives a moved diagram its levels back. The refusals after
    // it come where nothing was renumbered (a region alone) or from a
    // borrowed diagram, which `unseat` leaves as it was.
    let closed = result.levels.is_closed();
    if !literal_tops.is_empty()
        && let Err(e) = crate::diagram::remap_refs_into_all(eng.limits(), &mut result, &literal_tops, &[u32::MAX, 0, 1])
    {
        return Err((e, unseat(eng, result, tdd, plan, known.take())));
    }
    // A region at the root holds the output's count in a slot; under a
    // parent a small count is an inline reference instead.
    if tdd.vtree().num_nodes() == 1
        && let Err(e) = inline_root_region(eng, &tdd, into, &mut result)
    {
        return Err((e, unseat(eng, result, tdd, plan, known.take())));
    }
    // Taking a moved diagram's store cannot be refused, so a refusal here
    // has moved nothing.
    if weights && tdd.weights.is_some() {
        match placed_weights(eng, &mut tdd, plan) {
            Ok(weights) => result.weights = Some(weights),
            Err(e) => return Err((e, unseat(eng, result, tdd, plan, known.take()))),
        }
    }
    let mut carried = match (&mut tdd, copied) {
        (Cow::Owned(moved), _) => moved.take_worklists(),
        (Cow::Borrowed(_), copied) => copied.expect("a borrowed diagram's worklists are copied"),
    };
    // Every level of a moved `tdd` is now one the placement took from the
    // pool, empty: they go back for the takes that follow.
    if let Cow::Owned(moved) = &mut tdd {
        return_levels(eng, std::mem::take(&mut moved.levels).into_vec());
    }
    if chains {
        result.close_marked_levels(closed);
    }
    // A moved level keeps its nodes and, through the pass-throughs and the
    // renumbered literal chains, the identities of its children's, so it
    // owes the contraction passes what it owed in `tdd`, at its image. A
    // leaf read through a literal chain is the exception: its two literals
    // are two nodes of a pass-through level, so a leaf twin over it is an
    // inner twin, which only the inner pass contracts: the inner pass is
    // owed where the leaf pass is, over a chain. New pass-through nodes have
    // distinct child identities,
    // and free levels have one node, so neither introduces twins. A later
    // child contraction queues its parents. The lists are bounded as a seat
    // bounds an assembly's.
    let loose = carried.take_loose();
    carried.remap(&plan.embedding.levels);
    if chains {
        literal_tops.sort_unstable();
        carried.leaf_owes_inner(|t| {
            // A region is a leaf the leaf pass may be owed at.
            if into.node(VtreeIdx(t)).is_leaf() {
                return false;
            }
            let (left, right) = into.children(VtreeIdx(t));
            literal_tops.binary_search(&left).is_ok() || literal_tops.binary_search(&right).is_ok()
        });
    }
    carried.dedup_above(into.num_nodes());
    debug_assert!(result.dirty.is_empty(), "a seat that rebuilt nothing owes nothing");
    result.dirty = carried;
    if let Some(mut loose) = loose {
        loose.sort_unstable();
        moved_loose(tdd.vtree(), into, plan, &loose, &literal_tops, &mut listed);
        result.dirty.set_loose(Some(listed));
    }
    Ok(result)
}

/// Inline the small counts of `result`, `tdd` placed under a parent where
/// `tdd` is an unweighted region alone.
#[cold]
fn inline_root_region(eng: &Engine, tdd: &Tdd, into: &Vtree, result: &mut Tdd) -> Result<(), OperationError> {
    if tdd.weights.is_some() || !tdd.vtree().node(tdd.vtree().root()).is_region() || into.num_nodes() == 1 {
        return Ok(());
    }
    crate::diagram::inline_small_marginal_refs(eng.limits(), result, None)
        .and_then(|()| crate::reduce::slot_prune::prune_value_slots(eng, result).map(drop))
}

/// The weight store of [`assemble_moving`]'s result: a moved diagram's own,
/// a borrowed one's copied, its columns keyed by the levels' images.
#[cold]
fn placed_weights(eng: &Engine, tdd: &mut std::borrow::Cow<'_, Tdd>, plan: &Plan) -> Result<WeightStore, OperationError> {
    let mut weights = match tdd {
        std::borrow::Cow::Owned(moved) => moved.weights.take().expect("a weighted diagram"),
        std::borrow::Cow::Borrowed(source) => source.weights.as_ref().expect("a weighted diagram").try_clone_on(eng.limits())?,
    };
    weights.rekey_levels(|level| Some(plan.embedding.levels[level].idx()));
    Ok(weights)
}

/// Give a moved diagram back the levels a seated `result` took from it, as
/// the refusal before the seat does: each mirrored image read the right way
/// round again, every level back in its place, and what was known of them.
/// The seat's close may have described a level again, which holds the same
/// pairs. The rest of `result`'s levels go back to the pool. A copied
/// diagram was never touched.
fn unseat<'a>(
    eng: &Engine,
    mut result: Tdd,
    mut tdd: std::borrow::Cow<'a, Tdd>,
    plan: &Plan,
    known: Option<crate::diagram::Known>,
) -> std::borrow::Cow<'a, Tdd> {
    let mut levels = std::mem::take(&mut result.levels).into_vec();
    if let (std::borrow::Cow::Owned(source), Some(known)) = (&mut tdd, known) {
        for t in plan.mirrored_images() {
            levels[t.idx()].swap_sides();
        }
        let source_levels = &mut source.levels[..];
        for (from, &to) in plan.embedding.levels.iter().enumerate() {
            std::mem::swap(&mut source_levels[from], &mut levels[to.idx()]);
        }
        source.levels.reinstate_known(known);
    }
    return_levels(eng, levels);
    tdd
}

/// Mark mirrored images as changed and build the free and pass-through levels
/// of `plan` into `placement`, returning the tops of the pass-through chains
/// over a leaf read as `Pos`/`Neg` whose readers must be renumbered once seated.
fn build_embedded_levels(
    eng: &Engine,
    into: &Vtree,
    plan: &Plan,
    free: Free,
    output_local: NodeIdx,
    placement: &mut MovePlacement<'_>,
) -> Result<Vec<VtreeIdx>, OperationError> {
    let mut gate = eng.limits().gate();
    let mut literal_tops = Vec::new();
    // A mirrored image was changed in place by `placement.mirror`, and its
    // seat closes it.
    if plan.mirrors {
        for &t in into.internal_bottomup_slice() {
            if plan.is_mirrored(t) {
                placement.changed_at(t);
            }
        }
    }
    // The free levels when they are built, then the pass-throughs, each
    // after the levels under it. A free level left empty costs nothing
    // here, and a conjunction that builds it carries it as an identity
    // level, which costs nothing there either.
    let built: &[VtreeIdx] = match free {
        Free::Build | Free::Operand => into.internal_bottomup_slice(),
        Free::Leave => &[],
    };
    for &t in built
        .iter()
        .filter(|&&t| plan.is_free(t))
        .chain(plan.pass_throughs(into))
    {
        gate.poll(1)?;
        if plan.is_free(t) {
            placement.free(t);
            continue;
        }
        let (left, right) = into.children(t);
        let (free_side, carried) = if plan.is_free(left) {
            (ChildSide::Left, right)
        } else {
            (ChildSide::Right, left)
        };
        // A refused join is undone with the placement by the caller.
        if !into.node(carried).is_leaf() {
            placement.pass_through(t, free_side)?;
        } else if into.node(carried).is_region() {
            match image_above(into, plan, t) {
                Some((parent, side)) if placement.holds_counts(carried) => placement.pass_over_region(t, free_side, parent, side)?,
                _ => placement.pass_through(t, free_side)?,
            }
        } else if placement.is_marginal(carried) {
            // A summed variable's values live in its parent's references,
            // which a new parent between them does not take.
            return Err(OperationError::MarginalLevel(plan.covered_by(carried).unwrap_or(carried)));
        } else if let Some(top) = literal_chain(into, plan, |p, side| placement.first_ref(p, side), output_local, t) {
            literal_tops.push(top);
            placement.pass_over_leaf(t, free_side, &[POS_LEAF_IDX, NEG_LEAF_IDX])?;
        } else {
            placement.pass_over_leaf(t, free_side, &[ONE_LEAF_IDX])?;
        }
    }
    gate.flush()?;
    Ok(literal_tops)
}

/// The nearest ancestor of `t` that is the image of a source level, and the
/// side of it `t` is under.
fn image_above(into: &Vtree, plan: &Plan, mut t: VtreeIdx) -> Option<(VtreeIdx, ChildSide)> {
    while let Some(parent) = into.node(t).parent() {
        if plan.covered_by(parent).is_some() {
            return Some((parent, ChildSide::of(into, parent, t)));
        }
        t = parent;
    }
    None
}

/// The levels of [`assemble_moving`]'s result that may hold a node no pair
/// of their parent level names, pushed to `out` bottom-up, given `loose`,
/// the source's such levels, and `literal_tops`, the tops of the
/// pass-through chains that read a leaf as `Pos` and `Neg`, both sorted.
/// `source` is the vtree the levels were moved from; the walk visits the
/// images of its internal nodes and the chains under them, not the levels
/// built with no source variable under them.
///
/// Only a level whose parent is the image of a source level can hold one. A
/// level built under a built level is named whole: a pass-through has a
/// node for each slot of the level it carries, and the single node of a
/// level with no source variable is the true node every node of its parent
/// names, of which there is one at least, as on every level of a diagram
/// that is not false. Under an image, an image is named as its source level
/// was, and the top of a pass-through chain as the level at the chain's
/// foot, whose slots the chain carries one for one; a chain over a leaf
/// holds `One`, which the image reads, or `Pos` and `Neg`, of which it may
/// read one.
pub(crate) fn moved_loose(
    source: &Vtree,
    into: &Vtree,
    plan: &Plan,
    loose: &[u32],
    literal_tops: &[VtreeIdx],
    out: &mut Vec<u32>,
) {
    // With neither, every level is named whole.
    if loose.is_empty() && literal_tops.is_empty() {
        return;
    }
    for &s in source.internal_bottomup_slice() {
        let p = plan.embedding.levels[s.idx()];
        let (left, right) = into.children(p);
        for top in [left, right] {
            if into.node(top).is_leaf() {
                continue;
            }
            let mut foot = top;
            while plan.covered_by(foot).is_none() && !into.node(foot).is_leaf() {
                let (l, r) = into.children(foot);
                foot = if plan.is_free(l) { r } else { l };
            }
            let named_whole = match into.node(foot).is_leaf() {
                true => literal_tops.binary_search(&top).is_err(),
                false => {
                    let from = plan.covered_by(foot).expect("a chain's foot off a leaf is an image");
                    loose.binary_search(&from.0).is_err()
                }
            };
            if !named_whole {
                out.push(top.0);
            }
        }
    }
}

/// For a pass-through over a leaf at `t`, the top of the chain of
/// pass-throughs it starts when the level above that chain, or the output
/// when the chain reaches the root, names the leaf as `Pos`/`Neg`; `None`
/// when it names it as `One`. Determinism makes the first reference the
/// level's form.
pub(crate) fn literal_chain(
    into: &Vtree,
    plan: &Plan,
    first_ref: impl Fn(VtreeIdx, ChildSide) -> Option<NodeIdx>,
    output: NodeIdx,
    t: VtreeIdx,
) -> Option<VtreeIdx> {
    let mut top = t;
    loop {
        match into.node(top).parent() {
            Some(p) if plan.covered_by(p).is_none() => top = p,
            Some(p) => {
                let named = first_ref(p, ChildSide::of(into, p, top))?;
                return (named != ONE_LEAF_IDX).then_some(top);
            }
            None => return (output != ONE_LEAF_IDX).then_some(top),
        }
    }
}

/// Move a copy of the levels, marginal ones with their values, onto the
/// destination, and fill what the destination adds: a level under the image
/// of a marginal level is marginal without values of its own, and the free
/// variables below that image scale its values.
fn assemble_marginal(
    eng: &Engine,
    tdd: &Tdd,
    into: &Arc<Vtree>,
    plan: &Plan,
    weights: WeightStore,
) -> Result<Tdd, EmbedError> {
    if tdd.is_zero() {
        let mut result = crate::build::constant_zero(eng, into);
        result.weights = Some(weights.empty_like());
        return Ok(result);
    }
    // The levels move out of a copy, reserved through the limits so a
    // source too large to copy refuses the embed instead of aborting.
    let mut source = tdd.try_clone_on(eng)?;
    let mut placement = MovePlacement::new(eng, into, Some(weights.empty_like()))?;
    placement.move_part(&mut source, &plan.embedding.levels);
    let (inside, boundaries) = marginal_boundaries(tdd, into, plan);
    let mut gate = eng.limits().gate();
    for t in into.bottomup() {
        if into.node(t).is_leaf() || plan.covered_by(t).is_some() {
            continue;
        }
        gate.poll(1)?;
        if inside[t.idx()] {
            placement.subsume(t);
            continue;
        }
        let (left, right) = into.children(t);
        if plan.is_free(t) {
            placement.join(t, placement.true_node(left), placement.true_node(right))?;
        } else {
            placement.pass_through(
                t,
                if plan.is_free(left) {
                    ChildSide::Left
                } else {
                    ChildSide::Right
                },
            )?;
        }
    }
    gate.flush()?;
    scale_marginal_boundaries(&mut placement, into, plan, &weights, &boundaries);
    placement
        .check_weights()
        .map_err(EmbedError::DestinationWeights)?;
    Ok(placement.finish(tdd.output().local)?)
}

/// Find destination nodes strictly below the image of a marginal level and
/// the highest such images (`boundaries`).
fn marginal_boundaries(tdd: &Tdd, into: &Vtree, plan: &Plan) -> (Vec<bool>, Vec<VtreeIdx>) {
    let mut inside = vec![false; into.num_nodes()];
    let mut boundaries = Vec::new();
    for t in into.bottomup().rev() {
        if into.node(t).is_leaf() {
            continue;
        }
        let marginal = plan.covered_by(t).is_some_and(|s| tdd.level(s).is_marginal());
        if marginal && !inside[t.idx()] {
            boundaries.push(t);
        }
        let (left, right) = into.children(t);
        inside[left.idx()] = inside[t.idx()] || marginal;
        inside[right.idx()] = inside[t.idx()] || marginal;
    }
    (inside, boundaries)
}

/// Scale each highest marginal boundary by the product of the free leaves
/// in its destination subtree.
fn scale_marginal_boundaries(
    placement: &mut MovePlacement<'_>,
    into: &Vtree,
    plan: &Plan,
    weights: &WeightStore,
    boundaries: &[VtreeIdx],
) {
    let mut stack = Vec::new();
    for &boundary in boundaries {
        let mut factor: Option<WeightValue> = None;
        stack.push(boundary);
        while let Some(t) = stack.pop() {
            if into.node(t).is_leaf() {
                if plan.is_free(t) {
                    let free = weights.leaf_val(into.leaf_var(t), LeafLabel::One);
                    factor = Some(match factor {
                        None => free,
                        Some(f) => f.mul(&free),
                    });
                }
                continue;
            }
            let (left, right) = into.children(t);
            stack.push(left);
            stack.push(right);
        }
        if let Some(factor) = factor {
            placement.scale(boundary, &factor);
        }
    }
}

#[cfg(test)]
#[path = "tests/embed/mod.rs"]
mod tests;
