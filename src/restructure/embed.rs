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

use super::{EmbedError, EmbedRefused, Embedding, graft::check_part_weights, placement::{CopyPlacement, MovePlacement}};

use crate::Engine;
use crate::diagram::{return_levels, ChildSide, LeafLabel, NodeIdx, PoolSlot, Tdd, WeightStore, WeightValue, NEG_LEAF_IDX, ONE_LEAF_IDX, POS_LEAF_IDX};
use crate::limits::{Limits, OperationError};
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
    pub fn new(source: &Arc<Vtree>, destination: &Arc<Vtree>, map: impl Fn(VarId) -> VarId) -> Result<Self, EmbedError> {
        destination.context().run(|eng| eng.embedding_plan(source, destination, map))
    }

    /// [`Self::new`], accepting a destination that holds the source's shape
    /// with the children of some nodes swapped, as [`Tdd::embed_mirrored`]
    /// does.
    pub fn new_mirrored(source: &Arc<Vtree>, destination: &Arc<Vtree>, map: impl Fn(VarId) -> VarId) -> Result<Self, EmbedError> {
        destination.context().run(|eng| eng.embedding_plan_mirrored(source, destination, map))
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
    /// returns [`OperationError::VtreeMismatch`] wrapped in [`EmbedError::Operation`].
    /// Other errors and weight semantics are those of [`Tdd::embed`].
    pub fn apply(&self, circuit: &Tdd) -> Result<Tdd, EmbedError> {
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
    pub fn then(&self, next: &Self) -> Result<Self, EmbedError> {
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
    /// # tididi::test_helpers::assert_canonical(&both);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn embed(
        &self,
        into: &Arc<Vtree>,
        map: impl Fn(VarId) -> VarId,
    ) -> Result<(Tdd, Embedding), EmbedError> {
        into.context().run(|eng| eng.embed(self, into, map))
    }

    /// [`embed`](Self::embed), where `into` may hold this diagram's vtree
    /// shape **up to mirrors**: the children of any node may come in the
    /// other order under the renaming. Such a level is copied with the two
    /// sides of every pair exchanged, which denotes the same function over
    /// the mirrored node — a level's nodes are classes of assignments to the
    /// node's variables, whichever child is called left — so the result is
    /// canonical when this diagram is, and costs what [`embed`](Self::embed)
    /// costs. A binary relation's diagram placed with its two blocks the
    /// other way round is its transpose's.
    ///
    /// # Errors
    ///
    /// As [`embed`](Self::embed).
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{Tdd, Vtree};
    /// use tididi::vtree::VarId;
    ///
    /// // x1 ∧ ¬x2 on (x1 x2), placed so that x1 lands on the right leaf.
    /// let pair = Arc::new(Vtree::linear(2));
    /// let f = Tdd::cube(&pair, [1, -2])?;
    /// let into = Arc::new(Vtree::linear(2));
    /// assert!(f.embed(&into, |v| VarId(3 - v.0)).is_err());
    /// let (g, _) = f.embed_mirrored(&into, |v| VarId(3 - v.0))?;
    /// assert!(g.equivalent(&Tdd::cube(&into, [2, -1])?)?);
    /// # tididi::test_helpers::assert_canonical(&g);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn embed_mirrored(
        &self,
        into: &Arc<Vtree>,
        map: impl Fn(VarId) -> VarId,
    ) -> Result<(Tdd, Embedding), EmbedError> {
        into.context().run(|eng| eng.embed_mirrored(self, into, map))
    }
}

impl Engine {
    /// Prepare an [`EmbeddingPlan`] under this engine's limits.
    pub fn embedding_plan(&self, source: &Arc<Vtree>, destination: &Arc<Vtree>, map: impl Fn(VarId) -> VarId) -> Result<EmbeddingPlan, EmbedError> {
        self.plan_embedding(source, destination, map, false)
    }

    /// Prepare an [`EmbeddingPlan`] up to mirrors, as
    /// [`Tdd::embed_mirrored`] places, under this engine's limits.
    pub fn embedding_plan_mirrored(&self, source: &Arc<Vtree>, destination: &Arc<Vtree>, map: impl Fn(VarId) -> VarId) -> Result<EmbeddingPlan, EmbedError> {
        self.plan_embedding(source, destination, map, true)
    }

    /// [`EmbeddingPlan::apply`] under this engine's limits.
    pub fn embed_with(&self, circuit: &Tdd, plan: &EmbeddingPlan) -> Result<Tdd, EmbedError> {
        let _op = self.limits().enter()?;
        if !Arc::ptr_eq(circuit.vtree(), &plan.source) { return Err(OperationError::VtreeMismatch.into()); }
        circuit.require_structure()?;
        assemble(self, circuit, &plan.destination, &plan.layout)
    }

    /// [`EmbeddingPlan::then`] under this engine's limits. The composition
    /// matches up to mirrors when either plan does.
    pub fn compose_embeddings(&self, first: &EmbeddingPlan, next: &EmbeddingPlan) -> Result<EmbeddingPlan, EmbedError> {
        let _op = self.limits().enter()?;
        if !Arc::ptr_eq(&first.destination, &next.source) { return Err(OperationError::VtreeMismatch.into()); }
        let mirror = first.layout.mirror || next.layout.mirror;
        self.plan_embedding(&first.source, &next.destination, |var| {
            let leaf = first.source.leaf_of(var).expect("source leaf");
            let intermediate = first.levels().level_of(leaf);
            next.destination.leaf_var(next.levels().level_of(intermediate))
        }, mirror)
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
        self.embed_on(tdd, into, map, false)
    }

    /// [`Tdd::embed_mirrored`] under this batch's scratch and resource limits.
    ///
    /// # Errors
    ///
    /// As [`Tdd::embed`].
    pub fn embed_mirrored(
        &self,
        tdd: &Tdd,
        into: &Arc<Vtree>,
        map: impl Fn(VarId) -> VarId,
    ) -> Result<(Tdd, Embedding), EmbedError> {
        self.embed_on(tdd, into, map, true)
    }

    /// Match the two vtrees, up to mirrors when `mirror` is set.
    fn plan_embedding(
        &self,
        source: &Arc<Vtree>,
        destination: &Arc<Vtree>,
        map: impl Fn(VarId) -> VarId,
        mirror: bool,
    ) -> Result<EmbeddingPlan, EmbedError> {
        let lim = self.limits();
        let _op = lim.enter()?;
        let layout = Plan::build(lim, source, destination, map, mirror)?;
        Ok(EmbeddingPlan { source: source.clone(), destination: destination.clone(), layout })
    }

    /// Match the two vtrees, up to mirrors when `mirror` is set, and copy.
    fn embed_on(
        &self,
        tdd: &Tdd,
        into: &Arc<Vtree>,
        map: impl Fn(VarId) -> VarId,
        mirror: bool,
    ) -> Result<(Tdd, Embedding), EmbedError> {
        let lim = self.limits();
        let _op = lim.enter()?;
        tdd.require_structure()?;
        let plan = Plan::build(lim, tdd.vtree(), into, map, mirror)?;
        let result = assemble(self, tdd, into, &plan)?;
        Ok((result, plan.embedding))
    }

    /// [`Engine::embed`] that moves the diagram's levels into the result
    /// instead of copying them, and gives the diagram back unchanged when it
    /// is refused.
    ///
    /// The source's pairs are not touched: the cost is the size of `into`
    /// plus, at each destination level with renamed variables on one side
    /// only, the width of that populated side, and the pairs of a level that
    /// reads a single variable through such levels. The result is the
    /// diagram [`Engine::embed`] returns, up to the numbering of its nodes and
    /// a node for a literal of such a variable that nothing reads, which
    /// [`Engine::minimize`] removes. The levels move into storage the engine
    /// supplies without charging its limits, as [`Tdd::graft_over`] does.
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
    #[expect(clippy::result_large_err, reason = "the refusal hands back what it was given")]
    pub fn embed_moving(
        &self,
        tdd: Tdd,
        into: &Arc<Vtree>,
        map: impl Fn(VarId) -> VarId,
    ) -> Result<(Tdd, Embedding), EmbedRefused> {
        let _op = match self.limits().enter() {
            Ok(op) => op,
            Err(e) => return Err(EmbedRefused { error: e.into(), tdd }),
        };
        place_moving(self, tdd, into, map, Free::Build).map(|(result, plan)| (result, plan.embedding))
    }

    /// [`Tdd::embed`] for a diagram whose levels may hold weighted marginal
    /// values, with `weights` as the result's store.
    ///
    /// A weight-marginal level ([`Tdd::marginalize_levels`] under a
    /// [`WeightStore`]) is placed with its values, at the level `map` sends
    /// it to. The result sums over the same assignments as the source and
    /// over every free variable of `into`, so a variable `into` adds below
    /// the image of a marginal level multiplies that level's values by the
    /// variable's weight of `true`, and a level `into` adds below it holds no
    /// values of its own. A structural diagram is placed as by
    /// [`Tdd::embed`] and takes `weights` as its store when one is given;
    /// a diagram with a marginal level needs one, whose table must give
    /// each renamed variable the weights the source's gives the original.
    ///
    /// Copies the levels into storage the engine supplies without charging
    /// its limits, as [`Tdd::graft_over`] does; the reduction that seats a
    /// marginal level under a new parent runs under them.
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
    /// not cover `into`'s variables.
    pub fn embed_over(
        &self,
        tdd: &Tdd,
        into: &Arc<Vtree>,
        map: impl Fn(VarId) -> VarId,
        weights: Option<WeightStore>,
    ) -> Result<(Tdd, Embedding), EmbedError> {
        let lim = self.limits();
        let _op = lim.enter()?;
        let plan = Plan::build(lim, tdd.vtree(), into, &map, false)?;
        // Integer counts reach a parent through inline references, which
        // do not survive a new parent; only weighted values are placed.
        if let Some(t) = tdd.vtree().bottomup().find(|&t| tdd.level(t).is_marginal() && !tdd.level(t).is_weight_marginal()) {
            return Err(OperationError::MarginalLevel(t).into());
        }
        if let Some(store) = &weights {
            store.check_variables(into.leaf_bottomup().map(|(_, var)| var)).map_err(EmbedError::DestinationWeights)?;
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
    /// Whether no renamed source variable is under each destination node:
    /// the nodes whose levels are constant true, the free levels.
    pub(crate) free: Vec<bool>,
    /// The source node a destination node is the image of, where there is one.
    covered_by: Vec<Option<VtreeIdx>>,
    /// The destination node each source node maps to.
    embedding: Embedding,
    /// Whether children may be matched swapped.
    mirror: bool,
    /// Per source node: its image has its children swapped, so its level is
    /// copied with every pair read the other way round. Empty unless
    /// `mirror`.
    mirrored: Vec<bool>,
}

impl Plan {
    /// Check the renaming against both trees and record the correspondence.
    ///
    /// Matches the two trees top-down: a destination node with renamed
    /// variables on one side only is a pass-through and the walk descends
    /// into the other side without advancing the source, which is exactly the
    /// node [`Vtree::project_to_vars`] splices out. Every other node must
    /// correspond to the current source node. `O(nodes of into)`; up to
    /// mirrors, each source node's orientation is read off where the image of
    /// one of its left child's leaves lies, `O(depth of into)` more per node.
    ///
    /// The work clock is charged one unit for each source leaf, one for each
    /// internal node of `into`, and one for each node the match visits, the
    /// one it fails at included.
    fn build(
        lim: &Limits,
        source: &Vtree,
        into: &Vtree,
        map: impl Fn(VarId) -> VarId,
        mirror: bool,
    ) -> Result<Plan, EmbedError> {
        let mut free = Vec::new();
        lim.try_resize(&mut free, into.num_nodes(), true)?;
        let mut embedding = Vec::new();
        lim.try_resize(&mut embedding, source.num_nodes(), into.root())?;
        let mut gate = lim.gate();
        for (leaf, var) in source.leaf_bottomup() {
            gate.poll(1)?;
            let image = map(var);
            let target = into.leaf_of(image).ok_or(EmbedError::VariableOutOfRange {
                variable: image,
                num_vars: into.num_vars(),
            })?;
            if !free[target.idx()] {
                return Err(VtreeError::OverlappingVariable(image).into());
            }
            free[target.idx()] = false;
            embedding[leaf.idx()] = target;
            // The image's ancestors have a renamed variable under them, up to
            // the one where an earlier image's path joins.
            let mut node = target;
            while let Some(parent) = into.node(node).parent()
                && free[parent.idx()]
            {
                free[parent.idx()] = false;
                node = parent;
            }
        }
        gate.poll_each(into.internal_bottomup_slice().len() as u64)?;

        // A leaf under each source node, and whether each source node's
        // image has its children the other way round.
        let mut some_leaf = Vec::new();
        let mut mirrored = Vec::new();
        if mirror {
            lim.try_resize(&mut some_leaf, source.num_nodes(), source.root())?;
            lim.try_resize(&mut mirrored, source.num_nodes(), false)?;
            for s in source.bottomup() {
                some_leaf[s.idx()] = match source.node(s).is_leaf() {
                    true => s,
                    false => some_leaf[source.children(s).0.idx()],
                };
            }
        }

        let mut covered_by = Vec::new();
        lim.try_resize(&mut covered_by, into.num_nodes(), None)?;
        let mut plan = Plan { free, covered_by, embedding: Embedding { levels: embedding }, mirror, mirrored };
        let mut visited = 0;
        let matched = plan.match_down(lim, source, into, &some_leaf, &mut visited);
        gate.poll_each(visited)?;
        matched?;
        debug_assert_eq!(
            plan.covered_by.iter().filter(|source| source.is_some()).count(),
            source.num_nodes(),
            "a completed match gives every source level an image",
        );
        gate.flush()?;
        Ok(plan)
    }

    /// The top-down match of [`build`](Self::build), once the free nodes
    /// are marked, counting in `visited` the nodes it visits.
    fn match_down(
        &mut self,
        lim: &Limits,
        source: &Vtree,
        into: &Vtree,
        some_leaf: &[VtreeIdx],
        visited: &mut u64,
    ) -> Result<(), EmbedError> {
        let Plan { free, covered_by, embedding: Embedding { levels: embedding }, mirror, mirrored } = self;
        // An image pushes one entry more than it pops and a leaf one fewer,
        // so the stack holds one entry at most for each source leaf.
        let mut stack = Vec::new();
        lim.reserve_exact(&mut stack, source.num_nodes() / 2 + 1)?;
        lim.try_push(&mut stack, (into.root(), source.root()))?;
        while let Some((d, s)) = stack.pop() {
            *visited += 1;
            if !into.node(d).is_leaf() {
                let (left, right) = into.children(d);
                if free[left.idx()] || free[right.idx()] {
                    let carries = if free[left.idx()] { right } else { left };
                    lim.try_push(&mut stack, (carries, s))?;
                    continue;
                }
            }
            match (into.node(d).is_leaf(), source.node(s).is_leaf()) {
                // A source leaf meets the destination leaf carrying its image.
                (true, true) if embedding[s.idx()] == d => {}
                (false, false) => {
                    let (left, right) = into.children(d);
                    let (mut source_left, mut source_right) = source.children(s);
                    if *mirror && lies_under(into, embedding[some_leaf[source_left.idx()].idx()], right, d) {
                        mirrored[s.idx()] = true;
                        std::mem::swap(&mut source_left, &mut source_right);
                    }
                    lim.try_push(&mut stack, (left, source_left))?;
                    lim.try_push(&mut stack, (right, source_right))?;
                }
                _ => return Err(EmbedError::NotIsomorphic { source: s }),
            }
            embedding[s.idx()] = d;
            covered_by[d.idx()] = Some(s);
        }
        Ok(())
    }
}

/// What [`place_moving`] does with the free levels, those of the destination
/// nodes no renamed variable is under.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Free {
    /// Build each: one node, true on both sides.
    Build,
    /// Leave each empty, for a conjunction that reads it as the level it
    /// stands for and takes the other operand's level there.
    Leave,
}

/// [`Engine::embed_moving`] under the caller's entry, with the plan it placed
/// `tdd` by; with [`Free::Leave`], a diagram whose free levels are empty. A
/// false diagram is the false diagram on `into`, every level empty, whatever
/// `free` says.
#[expect(clippy::result_large_err, reason = "the refusal hands back what it was given")]
pub(crate) fn place_moving(
    eng: &Engine,
    tdd: Tdd,
    into: &Arc<Vtree>,
    map: impl Fn(VarId) -> VarId,
    free: Free,
) -> Result<(Tdd, Plan), EmbedRefused> {
    if let Err(e) = tdd.require_structure() {
        return Err(EmbedRefused { error: e.into(), tdd });
    }
    let plan = match Plan::build(eng.limits(), tdd.vtree(), into, map, false) {
        Ok(plan) => plan,
        Err(error) => return Err(EmbedRefused { error, tdd }),
    };
    if tdd.is_zero() {
        return Ok((crate::build::constant_zero(eng, into), plan));
    }
    match assemble_moving(eng, tdd, into, &plan, free) {
        Ok(result) => Ok((result, plan)),
        Err((e, tdd)) => Err(EmbedRefused { error: e.into(), tdd }),
    }
}

/// Whether the ancestor path from `node` meets `target` before `stop`, for a
/// `node` below `stop`.
fn lies_under(into: &Vtree, mut node: VtreeIdx, target: VtreeIdx, stop: VtreeIdx) -> bool {
    loop {
        if node == target {
            return true;
        }
        if node == stop {
            return false;
        }
        match into.node(node).parent() {
            Some(parent) => node = parent,
            None => return false,
        }
    }
}

/// Fill the destination's levels, then drop the pass-through nodes no copied
/// level names.
fn assemble(
    eng: &Engine,
    tdd: &Tdd,
    into: &Arc<Vtree>,
    plan: &Plan,
) -> Result<Tdd, EmbedError> {
    if tdd.is_zero() {
        return Ok(crate::build::constant_zero(eng, into));
    }
    let mut placement = CopyPlacement::new(eng, into)?;
    let mut gate = eng.limits().gate();
    for t in into.bottomup() {
        if into.node(t).is_leaf() { continue; }
        gate.poll(1)?;
        let (left, right) = into.children(t);
        if plan.free[t.idx()] {
            placement.join(t, placement.true_node(left), placement.true_node(right))?;
        } else if let Some(source) = plan.covered_by[t.idx()] {
            match plan.mirrored.get(source.idx()).copied().unwrap_or(false) {
                true => placement.copy_level_mirrored(tdd, source, t)?,
                false => placement.copy_level(tdd, source, t)?,
            }
        } else {
            placement.pass_through(t, if plan.free[left.idx()] { ChildSide::Left } else { ChildSide::Right })?;
        }
    }
    gate.flush()?;
    Ok(placement.finish(tdd.output().local)?)
}

/// Move the levels of a structural diagram onto the destination and build
/// what it adds, as [`assemble`] does with copies; the diagram comes back,
/// levels in place, when the result's storage is refused.
#[expect(clippy::result_large_err, reason = "the refusal hands back what it was given")]
fn assemble_moving(
    eng: &Engine,
    mut tdd: Tdd,
    into: &Arc<Vtree>,
    plan: &Plan,
    free: Free,
) -> Result<Tdd, (OperationError, Tdd)> {
    // A moved level keeps its nodes and, through the pass-throughs and the
    // renumbered literal chains, the identities of its children's, so it
    // owes the contraction passes what it owed in `tdd`. New pass-through
    // nodes have distinct child identities, and free levels have one node,
    // so neither introduces twins. A later child contraction queues its parents.
    let mut listed = Vec::new();
    let carried = eng.limits().reserve_exact(&mut listed, into.num_nodes()).and_then(|()| tdd.dirty.clone_on(eng));
    let mut carried = match carried {
        Ok(carried) => carried,
        Err(e) => return Err((e, tdd)),
    };
    carried.remap(&plan.embedding.levels);
    // The images of the levels that may hold a node no pair of their parent
    // level names, from which `moved_loose` lists the result's.
    let loose = carried.loose().map(<[u32]>::to_vec);
    let mut placement = match MovePlacement::new(eng, into, None) {
        Ok(placement) => placement,
        Err(e) => return Err((e, tdd)),
    };
    placement.move_part(&mut tdd, &plan.embedding.levels);
    let mut gate = eng.limits().gate();
    let mut stopped = Ok(());
    // The tops of the pass-through chains over a leaf read as `Pos`/`Neg`,
    // whose readers are renumbered once the result stands.
    let mut literal_tops = Vec::new();
    for t in into.bottomup() {
        if into.node(t).is_leaf() || plan.covered_by[t.idx()].is_some() {
            continue;
        }
        // A free level left empty costs nothing here, and a conjunction
        // that builds it carries it as an identity level, which costs
        // nothing there either.
        if plan.free[t.idx()] && free == Free::Leave {
            continue;
        }
        stopped = gate.poll(1);
        if stopped.is_err() {
            break;
        }
        if plan.free[t.idx()] {
            placement.free(t);
            continue;
        }
        let (left, right) = into.children(t);
        let (free_side, carried) = if plan.free[left.idx()] { (ChildSide::Left, right) } else { (ChildSide::Right, left) };
        if !into.node(carried).is_leaf() {
            placement.pass_through(t, free_side);
        } else if let Some(top) = literal_chain(into, plan, &placement, tdd.output().local, t) {
            placement.pass_over_leaf(t, free_side, &[POS_LEAF_IDX, NEG_LEAF_IDX]);
            literal_tops.push(top);
        } else {
            placement.pass_over_leaf(t, free_side, &[ONE_LEAF_IDX]);
        }
    }
    if let Err(e) = stopped.and_then(|()| gate.flush()) {
        placement.move_back(&mut tdd, &plan.embedding.levels);
        return Err((e, tdd));
    }
    let mut result = match placement.seat(tdd.output().local, carried) {
        Ok(result) => result,
        Err((e, placement)) => {
            placement.move_back(&mut tdd, &plan.embedding.levels);
            return Err((e, tdd));
        }
    };
    // Every level of `tdd` is now one the placement took from the pool,
    // empty: they go back for the takes that follow.
    return_levels(eng, PoolSlot::Vacant, std::mem::take(&mut tdd.levels).into_vec());
    // `Pos` and `Neg` are the chain's nodes 0 and 1; nothing reads `One`.
    let closed = result.levels.is_closed();
    for &top in &literal_tops {
        crate::diagram::remap_refs_into(&mut result, top, &[u32::MAX, 0, 1]);
    }
    if !literal_tops.is_empty() {
        result.close_marked_levels(closed);
    }
    if let Some(mut loose) = loose {
        loose.sort_unstable();
        literal_tops.sort_unstable();
        moved_loose(into, plan, &loose, &literal_tops, &mut listed);
        result.dirty.set_loose(Some(listed));
    }
    Ok(result)
}

/// The levels of [`assemble_moving`]'s result that may hold a node no pair
/// of their parent level names, pushed to `out` bottom-up, given `loose`,
/// the images of the source's such levels, and `literal_tops`, the tops of
/// the pass-through chains that read a leaf as `Pos` and `Neg`, both sorted.
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
fn moved_loose(into: &Vtree, plan: &Plan, loose: &[u32], literal_tops: &[VtreeIdx], out: &mut Vec<u32>) {
    for (p, left, right) in into.internal_bottomup() {
        if plan.covered_by[p.idx()].is_none() {
            continue;
        }
        for top in [left, right] {
            if into.node(top).is_leaf() {
                continue;
            }
            let mut foot = top;
            while plan.covered_by[foot.idx()].is_none() && !into.node(foot).is_leaf() {
                let (l, r) = into.children(foot);
                foot = if plan.free[l.idx()] { r } else { l };
            }
            let named_whole = match into.node(foot).is_leaf() {
                true => literal_tops.binary_search(&top).is_err(),
                false => loose.binary_search(&foot.0).is_err(),
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
fn literal_chain(into: &Vtree, plan: &Plan, placement: &MovePlacement<'_>, output: NodeIdx, t: VtreeIdx) -> Option<VtreeIdx> {
    let mut top = t;
    loop {
        match into.node(top).parent() {
            Some(p) if plan.covered_by[p.idx()].is_none() => top = p,
            Some(p) => {
                let named = placement.first_ref(p, ChildSide::of(into, p, top))?;
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
    let mut source = tdd.clone();
    let mut placement = MovePlacement::new(eng, into, Some(weights.empty_like()))?;
    placement.move_part(&mut source, &plan.embedding.levels);
    // The destination nodes strictly below the image of a marginal level,
    // and the highest such images, whose values the free leaves scale.
    let mut inside = vec![false; into.num_nodes()];
    let mut boundaries = Vec::new();
    for t in into.bottomup().rev() {
        if into.node(t).is_leaf() {
            continue;
        }
        let marginal = plan.covered_by[t.idx()].is_some_and(|s| tdd.level(s).is_marginal());
        if marginal && !inside[t.idx()] {
            boundaries.push(t);
        }
        let (left, right) = into.children(t);
        inside[left.idx()] = inside[t.idx()] || marginal;
        inside[right.idx()] = inside[t.idx()] || marginal;
    }
    let mut gate = eng.limits().gate();
    for t in into.bottomup() {
        if into.node(t).is_leaf() || plan.covered_by[t.idx()].is_some() {
            continue;
        }
        gate.poll(1)?;
        if inside[t.idx()] {
            placement.subsume(t);
            continue;
        }
        let (left, right) = into.children(t);
        if plan.free[t.idx()] {
            placement.join(t, placement.true_node(left), placement.true_node(right));
        } else {
            placement.pass_through(t, if plan.free[left.idx()] { ChildSide::Left } else { ChildSide::Right });
        }
    }
    gate.flush()?;
    let mut stack = Vec::new();
    for boundary in boundaries {
        let mut factor: Option<WeightValue> = None;
        stack.push(boundary);
        while let Some(t) = stack.pop() {
            if into.node(t).is_leaf() {
                if plan.free[t.idx()] {
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
    placement.check_weights().map_err(EmbedError::DestinationWeights)?;
    Ok(placement.finish(tdd.output().local)?)
}

#[cfg(test)]
#[path = "tests/embed/mod.rs"]
mod tests;
