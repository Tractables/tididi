//! Move a diagram onto the projection of a home vtree on its renamed
//! variables: level for level where the two trees have one shape, by
//! rotations on private trees where they do not.

use std::borrow::Cow;
use std::sync::Arc;

use super::{RestructureError, TransportError, TransportRefused, TransportStats};

use crate::Engine;
use crate::build::constant_without_leaf;
use crate::diagram::{LevelCounts, Tdd, TddBuildError, TddLevel, TddNodeId, return_levels};
use crate::limits::{Limits, OperationError};
use crate::value::CountVec;
use crate::vtree::{coalesce, Home, VarId, Vtree, VtreeIdx};

impl Engine {
    /// `f` with each variable renamed through `map`, on
    /// [`Vtree::projection`] of `home` on the renamed variables.
    ///
    /// Where that projection has the shape of `f`'s vtree up to mirrors (the
    /// renamed leaves of each node of `f`'s vtree are the leaves of one node
    /// of the projection, its children in either order), each level of `f`
    /// becomes the level of its image as it is, node indices and all, and at
    /// an image whose children come the other way round every pair has its
    /// two sides exchanged. The cost is the number of `f`'s vtree nodes times
    /// the logarithm of `home`'s size, plus the pairs of the mirrored levels;
    /// an owned `f` is moved, a borrowed one copied first. The result keeps
    /// what `f` knows of itself: canonical and trimmed ([`Engine::trim`])
    /// when `f` is.
    ///
    /// Elsewhere, with `bound` above 0, a copy of `f` is minimized and
    /// rotated to the projection's shape as [`Engine::restructure_to`] does,
    /// on two private trees over `f`'s leaves, so no part of the cost grows
    /// with `home`'s size. Each rotation is refused when its triples or the
    /// new inner level's nodes plus pairs reach `bound`; `usize::MAX` is no
    /// bound, and mirrors are never bounded. The result is canonical whether
    /// `f` was or not, on the projection on the images of the variables the
    /// minimized copy keeps. The rotations rebuild levels, so this path costs
    /// the copy and every diagram size the rotations pass through, which
    /// neither end bounds.
    ///
    /// A region ([`VtreeNode::is_region`](crate::vtree::VtreeNode::is_region))
    /// moves with its level: its extent becomes the renamed variables of its
    /// extent, and it sits where the renamed variable of its anchor does. A
    /// diagram with a region takes no rotation path.
    ///
    /// [`TransportStats`] counts the rotations and mirrors of the tree, so it
    /// says which path ran, not which levels were rewritten. The result is
    /// read under `home`'s weights ([`Home::weighted`]), if any. A constant
    /// on no leaf, false, and true named `One` at a one-leaf root become that
    /// constant on [`Vtree::empty_projection`] of `home`.
    ///
    /// The result is certified when `f` is
    /// ([`Tdd::is_canonical`](crate::Tdd::is_canonical)).
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{Engine, Vtree};
    /// use tididi::vtree::{Home, VarId};
    ///
    /// let engine = Engine::new();
    /// let home = Home::new(Vtree::balanced(8));
    /// let f = engine.clause(&home, [1, -2])?;
    /// // x1 ∨ ¬x2 becomes x6 ∨ ¬x5; x6 is right of x5 in `home`.
    /// let (g, stats) = engine.transport(&f, &home, |v| VarId(7 - v.0), 0).map_err(|r| r.error)?;
    /// assert_eq!((stats.rotations, stats.mirrors), (0, 1));
    /// assert!(Arc::ptr_eq(&g.vtree().home().unwrap(), &home));
    /// assert_eq!(engine.model_count(&g)?, 192u32.into()); // 3/4 of 2^8
    ///
    /// // (x1 x2) and x3 on a balanced home; x1 and (x2 x3) on a linear one.
    /// let h = engine.clause(&home, [1, 2, 3])?;
    /// let linear = Home::new(Vtree::linear(8));
    /// let (moved, stats) = engine.transport(&h, &linear, |v| v, usize::MAX).map_err(|r| r.error)?;
    /// assert_eq!(stats.rotations, 1);
    /// assert_eq!(engine.model_count(&moved)?, 224u32.into());
    /// # tididi::test_helpers::assert_canonical(&g);
    /// # tididi::test_helpers::assert_canonical(&moved);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Each with `f` as it was given, an owned `f` with all it knows of
    /// itself. Checked in this order before anything moves:
    /// [`TransportError::Operation`] with
    /// [`OperationError::MarginalLevel`](crate::OperationError::MarginalLevel)
    /// for a summed-out level that is not a region; [`TransportError::Variable`]
    /// for an image, of a leaf or of an extent, that is not a leaf of `home`;
    /// [`TransportError::Overlap`] for two variables with one image;
    /// [`TransportError::Weights`] for a summed-out region whose values `home` would read
    /// under other weights: [`TddBuildError::CountLevelWithWeights`] for
    /// counts onto a weighted home, [`TddBuildError::WeightedLevelWithoutStore`]
    /// for weight values of `f`'s home onto an unweighted one, and
    /// [`TddBuildError::IncompatibleWeights`] where the arithmetic differs or
    /// `home` does not weigh the image of a variable of the region as `f` is
    /// weighed at the variable; with `bound` 0 or a region,
    /// [`TransportError::Bound`] naming the lowest node of `f`'s vtree whose
    /// image is not one node of the projection
    /// ([`transport_projection`](Self::transport_projection) checks the
    /// vtrees alone). Then, on the rotation path, [`TransportError::Bound`]
    /// naming the node a refused rotation turns, and
    /// [`TransportError::Operation`] for a refused allocation or an armed
    /// stop.
    #[expect(clippy::result_large_err, reason = "the refusal hands back what it was given")]
    pub fn transport<'a>(
        &self,
        f: impl Into<Cow<'a, Tdd>>,
        home: &Home,
        map: impl Fn(VarId) -> VarId,
        bound: usize,
    ) -> Result<(Tdd, TransportStats), TransportRefused<'a>> {
        let f = f.into();
        let _op = match self.limits().enter() {
            Ok(op) => op,
            Err(e) => return Err(TransportRefused { error: e.into(), tdd: f }),
        };
        let lim = self.limits();
        let ranked = match checked(lim, &f, home, map) {
            Ok(ranked) => ranked,
            Err(error) => return Err(TransportRefused { error, tdd: f }),
        };
        let route = match Shape::of(lim, &f, home, &ranked).and_then(|shape| Route::with_levels(lim, &f, shape)) {
            Ok(route) => route,
            Err(TransportError::Bound { .. }) if bound > 0 && !f.vtree().has_regions() => return self.transport_rotating(f, home, ranked, bound),
            Err(error) => return Err(TransportRefused { error, tdd: f }),
        };
        let stats = TransportStats { rotations: 0, mirrors: route.mirrored.len(), peak_pairs: f.pair_count() };
        let mut moved = match f {
            Cow::Owned(f) => f,
            Cow::Borrowed(g) => match g.try_clone_on(self) {
                Ok(copy) => copy,
                Err(e) => return Err(TransportRefused { error: e.into(), tdd: Cow::Borrowed(g) }),
            },
        };
        route.seat(self, &mut moved);
        Ok((moved, stats))
    }

    /// The vtree [`transport`](Self::transport) with `bound` 0 would put `f`
    /// onto under `map`: [`Vtree::projection`] of `home` on the renamed
    /// variables, or [`Vtree::empty_projection`] of `home` for the constants
    /// `transport` sends there. It reads `f`'s vtree and output, not its
    /// levels.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{Engine, Tdd, Vtree};
    /// use tididi::restructure::TransportError;
    /// use tididi::vtree::{Home, VarId};
    ///
    /// let engine = Engine::new();
    /// let home = Home::new(Vtree::balanced(8));
    /// let pair = Home::new(Vtree::balanced(2));
    /// let f = engine.clause(&pair, [1, -2])?;
    /// let onto = engine.transport_projection(&f, &home, |v| VarId(v.0 + 4))?;
    /// assert_eq!(onto.num_leaves(), 2);
    /// let none = engine.transport_projection(&Tdd::zero(&pair), &home, |v| VarId(v.0 + 4))?;
    /// assert_eq!(none.num_nodes(), 0);
    /// // (x1 (x2 x3)) onto x1, x3 and x5, whose projection is ((x1 x3) x5).
    /// let three = engine.clause(&Home::new(Vtree::balanced(3)), [1, 2, 3])?;
    /// let skew = |v: VarId| [VarId(1), VarId(3), VarId(5)][v.0 as usize - 1];
    /// assert!(matches!(engine.transport_projection(&three, &home, skew), Err(TransportError::Bound { .. })));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// [`transport`](Self::transport)'s refusals that depend on the vtrees:
    /// [`TransportError::Variable`], [`TransportError::Overlap`],
    /// [`TransportError::Bound`], and [`TransportError::Operation`] for a
    /// refused allocation or an armed stop. `transport` may still refuse `f`
    /// for its weights or a summed-out level.
    pub fn transport_projection(
        &self,
        f: &Tdd,
        home: &Home,
        map: impl Fn(VarId) -> VarId,
    ) -> Result<Arc<Vtree>, TransportError> {
        let _op = self.limits().enter()?;
        let ranked = Ranked::find(self.limits(), f.vtree(), home, map)?;
        Ok(Shape::of(self.limits(), f, home, &ranked)?.into)
    }

    /// The rotation path of [`transport`](Self::transport), on a copy of `f`
    /// that a refusal drops.
    #[cold]
    #[inline(never)]
    #[expect(clippy::result_large_err, reason = "the refusal hands back what it was given")]
    fn transport_rotating<'a>(
        &self,
        f: Cow<'a, Tdd>,
        home: &Home,
        ranked: Ranked,
        bound: usize,
    ) -> Result<(Tdd, TransportStats), TransportRefused<'a>> {
        let mut copy = match f.try_clone_on(self) {
            Ok(copy) => copy,
            Err(e) => return Err(TransportRefused { error: e.into(), tdd: f }),
        };
        match self.rotate_copy(&mut copy, home, ranked, bound) {
            Ok(stats) => Ok((copy, stats)),
            Err(error) => Err(TransportRefused { error, tdd: f }),
        }
    }

    /// `copy` seated on a root tree over the local ids `1..=k`, leaf `i` in
    /// the home's order named `i`, minimized, and moved by the restructure
    /// kernel onto the same kind of twin of its target projection.
    fn rotate_copy(
        &self,
        copy: &mut Tdd,
        home: &Home,
        Ranked { into, order }: Ranked,
        bound: usize,
    ) -> Result<TransportStats, TransportError> {
        let lim = self.limits();
        let k = order.len() as u32;
        let mut local = Vec::new();
        lim.try_resize(&mut local, copy.vtree().num_nodes(), VarId(0))?;
        for (i, &(_, leaf)) in order.iter().enumerate() {
            local[leaf.idx()] = VarId(i as u32 + 1);
        }
        copy.vtree = Arc::new(copy.vtree().local_twin(|leaf| local[leaf.idx()], k));
        lim.discard(local);
        if copy.is_zero() {
            *copy = crate::build::constant_zero(self, &into);
            return Ok(TransportStats::default());
        }
        self.reduce(copy, crate::reduce::ReductionPlan::default())?;
        let mut kept = Vec::new();
        lim.reserve_exact(&mut kept, copy.vtree().num_leaves() as usize)?;
        kept.extend(copy.vtree().leaf_bottomup().map(|(_, var)| var.0));
        kept.sort_unstable();
        let into = match kept.len() == order.len() {
            true => into,
            false => Vtree::projection_on_ranks(home, kept.iter().map(|&i| order[i as usize - 1].0).collect()),
        };
        let mut name = Vec::new();
        lim.try_resize(&mut name, into.num_nodes(), VarId(0))?;
        for (&i, leaf) in kept.iter().zip(into.projected_leaves()) {
            name[leaf.idx()] = VarId(i);
        }
        let twin = Arc::new(into.local_twin(|leaf| name[leaf.idx()], k));
        let stats = self.rearrange(copy, &twin, bound, true).map_err(|error| match error {
            RestructureError::Bound { pivot } => TransportError::Bound { pivot },
            RestructureError::Operation(error) => TransportError::Operation(error),
            RestructureError::Variables { .. } | RestructureError::Weighted => unreachable!("the twins have one leaf set and no table"),
        })?;
        // The twin is `into` with its leaves renamed: the node indices agree.
        copy.vtree = into;
        // The reduction ran on a bare root, which names a leaf the copy does
        // not depend on `One` instead of dropping it.
        crate::reduce::trim(self, copy, home, None, &[])?;
        Ok(stats)
    }
}

/// The checks of [`Engine::transport`] before the shape, in its order.
fn checked(lim: &Limits, f: &Tdd, home: &Home, map: impl Fn(VarId) -> VarId) -> Result<Ranked, TransportError> {
    if let Some(level) = f.levels.first_marginal() {
        match f.vtree().has_regions() {
            false => return Err(OperationError::MarginalLevel(level).into()),
            true => summed_only_at_regions(f)?,
        }
    }
    let ranked = Ranked::find(lim, f.vtree(), home, &map)?;
    if f.vtree().has_regions() {
        check_summed_weights(f, home, &map).map_err(TransportError::Weights)?;
    }
    Ok(ranked)
}

/// Refuse `f` onto `home` when a value it stores would be read under other
/// weights. Only the summed-out regions store values that depend on the
/// weights, and only through the weights of their variables.
#[cold]
#[inline(never)]
fn check_summed_weights(f: &Tdd, home: &Home, map: &impl Fn(VarId) -> VarId) -> Result<(), TddBuildError> {
    let vtree = f.vtree();
    let mut summed = vtree.regions().filter(|t| f.levels[t.idx()].is_marginal());
    let Some(level) = summed.next() else { return Ok(()) };
    match (f.weight_table(), home.weights()) {
        (None, None) => Ok(()),
        (None, Some(_)) => Err(TddBuildError::CountLevelWithWeights { level }),
        (Some(_), None) => Err(TddBuildError::WeightedLevelWithoutStore { level }),
        (Some(from), Some(to)) => {
            let vars: Vec<VarId> = std::iter::once(level).chain(summed).flat_map(|t| vtree.region_vars(t).1).collect();
            from.check_variables(vars.iter().copied())?;
            match from.compatible_after_rename(to, vars.iter().map(|&var| (var, map(var)))) {
                true => Ok(()),
                false => Err(TddBuildError::IncompatibleWeights),
            }
        }
    }
}

/// `f`, on a projection that rotations and swaps moved since its link was
/// made, seated level for level on the projection of
/// [`Vtree::lifted_home`] on its leaves; nothing on a root vtree.
pub(super) fn seat_on_lifted_home(eng: &Engine, f: &mut Tdd) {
    let Some(home) = f.vtree.lifted_home() else { return };
    // Nothing here is the caller's to refuse: a search keeps its moves
    // before this runs, and the diagram is only valid once seated.
    // Not `checked`, which refuses a summed-out level: the lifted home has
    // the variables and shape of the moved tree, so every level keeps its
    // meaning.
    let lim = Limits::new();
    let ranked = Ranked::find(&lim, f.vtree(), &home, |v| v).expect("the lifted home has the moved tree's leaves");
    let shape = Shape::along(&lim, f.vtree(), &ranked).expect("the lifted home has the moved tree's shape");
    let route = Route::with_levels(&lim, f, shape).expect("unlimited");
    debug_assert!(route.mirrored.is_empty());
    route.seat(eng, f);
}

/// The leaves of a vtree by the home ranks of their images, and the
/// projection of the home on those images.
struct Ranked {
    into: Arc<Vtree>,
    /// Each childless node of the vtree with the rank of its image, a
    /// region's the image of its anchor, by rank.
    order: Vec<(u32, VtreeIdx)>,
}

impl Ranked {
    fn find(lim: &Limits, source: &Vtree, home: &Home, map: impl Fn(VarId) -> VarId) -> Result<Self, TransportError> {
        if source.has_regions() {
            return Self::with_regions(source, home, map);
        }
        if source.num_nodes() == 0 {
            return Ok(Self { into: Vtree::empty_projection(home), order: Vec::new() });
        }
        let mut order = Vec::new();
        lim.reserve_exact(&mut order, source.num_leaves() as usize)?;
        for (leaf, var) in source.leaf_bottomup() {
            let image = map(var);
            let Some(rank) = home.rank(image) else { return Err(TransportError::Variable { variable: image }) };
            order.push((rank, leaf));
        }
        order.sort_unstable();
        if let Some(w) = order.windows(2).find(|w| w[0].0 == w[1].0) {
            return Err(TransportError::Overlap { variable: map(source.leaf_var(w[0].1)) });
        }
        let into = Vtree::projection_on_ranks(home, order.iter().map(|&(rank, _)| rank).collect());
        Ok(Self { into, order })
    }

    /// [`Ranked::find`] of a vtree with a region: each region's extent
    /// becomes the ranks of its variables' images.
    #[cold]
    #[inline(never)]
    fn with_regions(source: &Vtree, home: &Home, map: impl Fn(VarId) -> VarId) -> Result<Self, TransportError> {
        let rank = |var: VarId| {
            let image = map(var);
            home.rank(image).ok_or(TransportError::Variable { variable: image })
        };
        let mut order = Vec::new();
        let mut ranks = Vec::new();
        for (leaf, var) in source.leaf_bottomup() {
            let r = rank(var)?;
            order.push((r, leaf));
            ranks.push(r);
        }
        let mut regions = Vec::new();
        let mut taken = ranks.clone();
        for t in source.regions() {
            let (anchor, vars) = source.region_vars(t);
            let at = rank(anchor)?;
            let extent: Vec<(u32, u32)> = vars.into_iter().map(|var| rank(var).map(|r| (r, r + 1))).collect::<Result<_, _>>()?;
            taken.extend(extent.iter().flat_map(|&(lo, hi)| lo..hi));
            order.push((at, t));
            regions.push((at, coalesce(extent)));
        }
        taken.sort_unstable();
        if let Some(w) = taken.windows(2).find(|w| w[0] == w[1]) {
            let variable = home.leaf_bottomup().find(|&(_, var)| home.rank(var) == Some(w[0])).expect("a home leaf").1;
            return Err(TransportError::Overlap { variable });
        }
        order.sort_unstable();
        Ok(Self { into: Vtree::project_regions(home, ranks, regions), order })
    }
}

/// [`OperationError::MarginalLevel`] for a summed-out level of `f` that is
/// not a region.
#[cold]
fn summed_only_at_regions(f: &Tdd) -> Result<(), OperationError> {
    let vtree = f.vtree();
    match vtree.bottomup().find(|t| f.levels[t.idx()].is_marginal() && !vtree.node(*t).is_region()) {
        Some(level) => Err(OperationError::MarginalLevel(level)),
        None => Ok(()),
    }
}

/// The image of each node of a vtree in the projection of a home on its
/// renamed leaves.
#[derive(Clone)]
struct Shape {
    into: Arc<Vtree>,
    to: Vec<VtreeIdx>,
    /// The nodes whose images have their children the other way round.
    mirrored: Vec<VtreeIdx>,
}

impl Shape {
    /// [`along`](Self::along) `f`'s vtree, the projection replaced by the
    /// empty one for false and for true named `One` at a one-leaf root, which
    /// refuse as `along` does.
    fn of(lim: &Limits, f: &Tdd, home: &Home, ranked: &Ranked) -> Result<Self, TransportError> {
        let shape = Self::along(lim, f.vtree(), ranked)?;
        Ok(match (f.is_zero() || f.is_one_leaf_one()) && !f.vtree().has_regions() {
            true => Self { into: Vtree::empty_projection(home), to: Vec::new(), mirrored: Vec::new() },
            false => shape,
        })
    }

    /// The shape of `source` on `ranked`'s projection, or
    /// [`TransportError::Bound`] at the lowest node with no image.
    fn along(lim: &Limits, source: &Vtree, ranked: &Ranked) -> Result<Self, TransportError> {
        Self::of_children(lim, ranked, source.num_nodes(), source.internal_bottomup_slice(), |s| source.children(s))
    }

    /// [`along`](Self::along) a vtree of `nodes` nodes whose internal nodes
    /// are `bottom_up`, in that order, with the children `children`.
    fn of_children(
        lim: &Limits,
        ranked: &Ranked,
        nodes: usize,
        bottom_up: &[VtreeIdx],
        children: impl Fn(VtreeIdx) -> (VtreeIdx, VtreeIdx),
    ) -> Result<Self, TransportError> {
        let mut shape = Self { into: Arc::clone(&ranked.into), to: Vec::new(), mirrored: Vec::new() };
        if nodes == 0 {
            return Ok(shape);
        }
        lim.try_resize(&mut shape.to, nodes, VtreeIdx(0))?;
        for (&(_, leaf), image) in ranked.order.iter().zip(ranked.into.projected_leaves()) {
            shape.to[leaf.idx()] = image;
        }
        let mut gate = lim.gate();
        for &s in bottom_up {
            gate.poll(1)?;
            let (left, right) = children(s);
            let (l, r) = (shape.to[left.idx()], shape.to[right.idx()]);
            let into = &shape.into;
            let Some(d) = into.node(l).parent().filter(|&d| into.node(r).parent() == Some(d)) else {
                return Err(TransportError::Bound { pivot: s });
            };
            if into.children(d).0 != l {
                lim.try_push(&mut shape.mirrored, s)?;
            }
            shape.to[s.idx()] = d;
        }
        gate.flush()?;
        Ok(shape)
    }
}

/// Members `which` of `members`, which share a projection, seated level for
/// level on one projection of `home` on its leaves. Their levels are on the
/// projection as rotations and swaps moved it while `home` was lifted from
/// it: the internal nodes `bottom_up`, in that order, with the children
/// `kids` (by node, left first).
pub(super) fn seat_group(
    eng: &Engine,
    members: &mut [&mut Tdd],
    which: &[usize],
    kids: &[(VtreeIdx, VtreeIdx)],
    bottom_up: &[VtreeIdx],
    home: &Home,
) {
    let lim = Limits::new();
    let source = Arc::clone(&members[which[0]].vtree);
    let ranked = Ranked::find(&lim, &source, home, |v| v).expect("a home has its projections' leaves");
    let shape = Shape::of_children(&lim, &ranked, source.num_nodes(), bottom_up, |s| kids[s.idx()])
        .expect("the home has the moved tree's shape");
    debug_assert!(shape.mirrored.is_empty());
    for &i in which {
        let f = &mut *members[i];
        Route::with_levels(&lim, f, shape.clone()).expect("unlimited").seat(eng, f);
    }
}

/// Where each level of the diagram goes, with every allocation the move
/// needs, so that nothing after the match is refused.
struct Route {
    into: Arc<Vtree>,
    /// The image of each node of the diagram's vtree.
    to: Vec<VtreeIdx>,
    /// The nodes of the diagram's vtree whose images have their children the
    /// other way round.
    mirrored: Vec<VtreeIdx>,
    levels: Vec<TddLevel>,
    columns: Vec<Option<Arc<CountVec>>>,
}

impl Route {
    fn with_levels(lim: &Limits, f: &Tdd, Shape { into, to, mirrored }: Shape) -> Result<Self, TransportError> {
        let mut route = Self { into, to, mirrored, levels: Vec::new(), columns: Vec::new() };
        let n = f.vtree().num_nodes();
        if n == 0 {
            return Ok(route);
        }
        lim.try_resize(&mut route.levels, n, TddLevel::default())?;
        if f.levels.counts().is_some() {
            lim.try_resize(&mut route.columns, n, None)?;
        }
        Ok(route)
    }

    /// Move `f`'s levels to their images and seat `f` on the projection.
    fn seat(self, eng: &Engine, f: &mut Tdd) {
        let Self { into, to, mirrored, mut levels, mut columns } = self;
        if into.num_nodes() == 0 {
            let old = std::mem::replace(f, constant_without_leaf(&into, !f.is_zero()));
            return_levels(eng, old.levels.into_vec());
            return;
        }
        let canonical = f.levels.is_canonical(f.output);
        let closed = f.levels.is_closed();
        let counts = f.levels.counts().map(|counts| {
            for (s, &d) in to.iter().enumerate() {
                columns[d.idx()] = counts.column(VtreeIdx(s as u32)).cloned();
            }
            LevelCounts::from_columns(columns)
        });
        let mut old = std::mem::take(&mut f.levels).into_vec();
        for (s, &d) in to.iter().enumerate() {
            levels[d.idx()] = std::mem::take(&mut old[s]);
        }
        return_levels(eng, old);
        for &s in &mirrored {
            levels[to[s.idx()].idx()].swap_sides();
        }
        f.dirty.remap_keeping(&to, |_| mirrored.is_empty());
        f.vtree = into;
        f.levels = levels.into();
        f.levels.hold_counts(closed);
        f.output = TddNodeId { vtree: f.vtree.root(), local: f.output.local };
        for &s in &mirrored {
            f.levels.mark_changed(to[s.idx()]);
        }
        f.close_marked_levels();
        if canonical {
            f.levels.certify(f.output);
        }
        if let Some(counts) = counts {
            f.levels.keep_counts(counts);
        }
    }
}

#[cfg(test)]
#[path = "tests/transport.rs"]
mod tests;
