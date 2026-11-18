//! Move a diagram onto the projection of a home vtree on its renamed
//! variables, level for level.

use std::borrow::Cow;
use std::sync::Arc;

use super::{EmbedError, TransportRefused};

use crate::Engine;
use crate::build::constant_without_leaf;
use crate::diagram::{LevelCounts, Tdd, TddLevel, TddNodeId, return_levels};
use crate::value::CountVec;
use crate::limits::OperationError;
use crate::vtree::{coalesce, Extent, Home, VarId, Vtree, VtreeError, VtreeIdx};

impl Engine {
    /// `f` with each variable renamed through `map`, on
    /// [`Vtree::projection`] of `home` on the renamed variables.
    ///
    /// That projection must have the shape of `f`'s vtree up to mirrors: the
    /// renamed leaves of each node of `f`'s vtree are the leaves of one node
    /// of the projection, its children in either order. Each level of `f`
    /// becomes the level of its image as it is, node indices and all; at an
    /// image whose children come the other way round, every pair has its two
    /// sides exchanged, which denotes the same functions over the image. No
    /// level is built for a variable of `home` outside the image, so the cost
    /// is the number of `f`'s vtree nodes times the logarithm of `home`'s
    /// size, plus the pairs of the mirrored levels; an owned `f` is moved, a
    /// borrowed one copied first. The result is canonical when `f` is, and
    /// trimmed when `f` is ([`Engine::trim`]).
    ///
    /// A weight table moves with the diagram, read by the renamed variables'
    /// ids. A constant on no leaf becomes the same constant on
    /// [`Vtree::empty_projection`] of `home`.
    ///
    /// A region ([`VtreeNode::is_region`](crate::vtree::VtreeNode::is_region))
    /// moves with its level: its extent becomes the renamed variables of its
    /// extent, and it sits where the renamed variable of its anchor does.
    ///
    /// To put `f` on a vtree that does not have its shape under `map`,
    /// conjoin its clauses or factors there ([`Engine::and`]).
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{Engine, Vtree};
    /// use tididi::vtree::{Home, VarId};
    ///
    /// let engine = Engine::new();
    /// let home = Home::new(Vtree::balanced(8));
    /// let own = Vtree::projection(&home, [VarId(1), VarId(2)])?;
    /// let f = engine.clause(&own, [1, -2])?;
    /// // x1 ∨ ¬x2 becomes x6 ∨ ¬x5; x6 is right of x5 in `home`.
    /// let g = engine.transport(&f, &home, |v| VarId(7 - v.0)).map_err(|r| r.error)?;
    /// assert_eq!(g.vtree().num_leaves(), 2);
    /// assert!(Arc::ptr_eq(g.vtree().home().unwrap().vtree(), home.vtree()));
    /// assert_eq!(engine.model_count(&g)?, 192u32.into()); // 3/4 of 2^8
    /// # tididi::test_helpers::assert_canonical(&g);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Each with `f` as it was given: [`EmbedError::NotIsomorphic`] naming
    /// the node of `f`'s vtree whose image is not one node of the projection;
    /// [`EmbedError::VariableOutOfRange`] for an image, of a leaf or of an
    /// extent, that is not a leaf of `home`; [`EmbedError::Vtree`] with
    /// [`VtreeError::OverlappingVariable`] when two variables share an image;
    /// [`EmbedError::DestinationWeights`] when `f`'s weight table has no
    /// weights for an image of a leaf; [`EmbedError::Operation`] with
    /// [`OperationError::MarginalLevel`] for a summed-out level that is not a
    /// region, and for a refused allocation or an armed stop.
    #[expect(clippy::result_large_err, reason = "the refusal hands back what it was given")]
    pub fn transport<'a>(
        &self,
        f: impl Into<Cow<'a, Tdd>>,
        home: &Home,
        map: impl Fn(VarId) -> VarId,
    ) -> Result<Tdd, TransportRefused<'a>> {
        let f = f.into();
        let _op = match self.limits().enter() {
            Ok(op) => op,
            Err(e) => return Err(TransportRefused { error: e.into(), tdd: f }),
        };
        let route = match Route::find(self, &f, home, map) {
            Ok(route) => route,
            Err(error) => return Err(TransportRefused { error, tdd: f }),
        };
        let mut moved = match f {
            Cow::Owned(f) => f,
            Cow::Borrowed(g) => match g.try_clone_on(self) {
                Ok(copy) => copy,
                Err(e) => return Err(TransportRefused { error: e.into(), tdd: Cow::Borrowed(g) }),
            },
        };
        route.seat(self, &mut moved);
        Ok(moved)
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
    fn find(eng: &Engine, f: &Tdd, home: &Home, map: impl Fn(VarId) -> VarId) -> Result<Self, EmbedError> {
        let lim = eng.limits();
        let source = f.vtree();
        if let Some(level) = f.levels.first_marginal() {
            match source.has_regions() {
                false => return Err(OperationError::MarginalLevel(level).into()),
                true => summed_only_at_regions(f)?,
            }
        }
        let empty = Self { into: Arc::clone(source), to: Vec::new(), mirrored: Vec::new(), levels: Vec::new(), columns: Vec::new() };
        if source.num_nodes() == 0 {
            return Ok(Self { into: Vtree::empty_projection(home), ..empty });
        }
        let mut images = Vec::new();
        lim.reserve_exact(&mut images, source.num_leaves() as usize)?;
        for (_, var) in source.leaf_bottomup() {
            let image = map(var);
            if home.rank(image).is_none() {
                return Err(EmbedError::VariableOutOfRange { variable: image, num_vars: home.vtree().num_vars() });
            }
            images.push(image);
        }
        if let Some(store) = &f.weights {
            store.check_variables(images.iter().copied()).map_err(EmbedError::DestinationWeights)?;
        }
        let regions = match source.has_regions() {
            false => Vec::new(),
            true => region_images(source, home, &map)?,
        };
        let into = match regions.is_empty() {
            true => Vtree::projection(home, images.iter().copied())?,
            false => project_with_regions(home, &images, &regions)?,
        };
        if into.num_leaves() != source.num_leaves() {
            images.sort_unstable();
            let shared = images.windows(2).find(|w| w[0] == w[1]).expect("fewer leaves than images");
            return Err(VtreeError::OverlappingVariable(shared[0]).into());
        }

        let mut route = Self { into, ..empty };
        let n = source.num_nodes();
        lim.try_resize(&mut route.to, n, VtreeIdx(0))?;
        for ((leaf, _), &image) in source.leaf_bottomup().zip(&images) {
            route.to[leaf.idx()] = route.into.leaf_of(image).expect("a projected leaf");
        }
        for (t, (anchor, _)) in source.regions().zip(&regions) {
            let at = home.vtree().leaf_of(*anchor).expect("a home variable");
            route.to[t.idx()] = route.into.node_at_home(at).expect("a projected region");
        }
        lim.discard(images);
        let mut gate = lim.gate();
        for &s in source.internal_bottomup_slice() {
            gate.poll(1)?;
            let (left, right) = source.children(s);
            let (l, r) = (route.to[left.idx()], route.to[right.idx()]);
            let into = &route.into;
            let Some(d) = into.node(l).parent().filter(|&d| into.node(r).parent() == Some(d)) else {
                return Err(EmbedError::NotIsomorphic { source: s });
            };
            if into.children(d).0 != l {
                lim.try_push(&mut route.mirrored, s)?;
            }
            route.to[s.idx()] = d;
        }
        gate.flush()?;
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
            let (weights, one) = (f.weights.take(), f.is_one());
            let old = std::mem::replace(f, constant_without_leaf(&into, one));
            return_levels(eng, old.levels.into_vec());
            f.weights = weights;
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
        if let Some(weights) = f.weights.as_mut().filter(|_| f.vtree.has_regions()) {
            weights.rekey_levels(|s| Some(to[s].idx()));
        }
        f.dirty.remap_keeping(&to, |_| mirrored.is_empty());
        f.vtree = into;
        f.levels = levels.into();
        f.output = TddNodeId { vtree: f.vtree.root(), local: f.output.local };
        for &s in &mirrored {
            f.levels.mark_changed(to[s.idx()]);
        }
        f.close_marked_levels(closed);
        if canonical {
            f.levels.certify(f.output);
        }
        if let Some(counts) = counts {
            f.levels.keep_counts(counts);
        }
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

/// By region of `source`, in [`Vtree::regions`] order, the image of its
/// anchor variable and its extent's images as ranks of `home`.
#[cold]
fn region_images(source: &Vtree, home: &Home, map: &impl Fn(VarId) -> VarId) -> Result<Vec<(VarId, Extent)>, EmbedError> {
    let rank = |var: VarId| {
        let image = map(var);
        home.rank(image).ok_or(EmbedError::VariableOutOfRange { variable: image, num_vars: home.vtree().num_vars() })
    };
    let mut regions = Vec::new();
    for t in source.regions() {
        let (anchor, vars) = source.region_vars(t);
        rank(anchor)?;
        let ranks = vars.into_iter().map(|var| rank(var).map(|r| (r, r + 1))).collect::<Result<_, _>>()?;
        regions.push((map(anchor), coalesce(ranks)));
    }
    Ok(regions)
}

/// The projection of `home` on the variable leaves `images` and on
/// `regions`; [`VtreeError::OverlappingVariable`] when two of them share a
/// variable.
#[cold]
fn project_with_regions(home: &Home, images: &[VarId], regions: &[(VarId, Extent)]) -> Result<Arc<Vtree>, EmbedError> {
    let ranks: Vec<u32> = images.iter().map(|&var| home.rank(var).expect("a home variable")).collect();
    let mut taken: Vec<u32> = ranks.iter().copied().chain(regions.iter().flat_map(|(_, extent)| extent.iter().flat_map(|&(lo, hi)| lo..hi))).collect();
    taken.sort_unstable();
    if let Some(pair) = taken.windows(2).find(|w| w[0] == w[1]) {
        let var = home.vtree().leaf_bottomup().find(|&(_, var)| home.rank(var) == Some(pair[0])).expect("a home leaf").1;
        return Err(VtreeError::OverlappingVariable(var).into());
    }
    let regions = regions.iter().map(|(anchor, extent)| (home.rank(*anchor).expect("a home variable"), extent.clone())).collect();
    Ok(Vtree::project_regions(home, ranks, regions))
}

#[cfg(test)]
#[path = "tests/transport.rs"]
mod tests;
