//! Move a diagram onto the projection of a home vtree on its renamed
//! variables, level for level.

use std::borrow::Cow;
use std::sync::Arc;

use super::{EmbedError, RehomeRefused, RestructureError, TransportRefused};

use crate::Engine;
use crate::build::constant_without_leaf;
use crate::diagram::{LevelCounts, Tdd, TddLevel, TddNodeId, return_levels};
use crate::limits::Limits;
use crate::value::CountVec;
use crate::vtree::{Home, VarId, Vtree, VtreeError, VtreeIdx};

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
    /// [`EmbedError::VariableOutOfRange`] for an image that is not a leaf of
    /// `home`; [`EmbedError::Vtree`] with [`VtreeError::OverlappingVariable`]
    /// when two variables share an image; [`EmbedError::DestinationWeights`]
    /// when `f`'s weight table has no weights for an image;
    /// [`EmbedError::Operation`] with
    /// [`OperationError::MarginalLevel`](crate::OperationError::MarginalLevel)
    /// for a summed-out level, and for a refused allocation or an armed stop.
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
        let route = match Route::find(self.limits(), &f, home, map) {
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

    /// `f`, a diagram on a projection, on [`Vtree::projection`] of `home` on
    /// the same variables, so that its counts and evaluations are over
    /// `home`'s variables.
    ///
    /// Where that projection has the shape of `f`'s vtree up to mirrors, this
    /// is [`Engine::transport`] with every variable kept, which costs no
    /// rotation. Elsewhere it is [`Engine::restructure_to`] onto the
    /// projection under `bound`, which refuses weighted diagrams; a `bound`
    /// of 0 allows no rotation.
    ///
    /// A projection of a home that a rotation search moved keeps its shape on
    /// the search's new home when every kept move had its leaves under at
    /// most two of the three subtrees it turned, so such a diagram rehomes
    /// with `bound` 0.
    ///
    /// ```
    /// use tididi::{Engine, Vtree};
    /// use tididi::vtree::{Home, VarId};
    ///
    /// let engine = Engine::new();
    /// let home = Home::new(Vtree::balanced(4));
    /// let own = Vtree::projection(&home, [VarId(1), VarId(3)])?;
    /// let f = engine.clause(&own, [1, 3])?;
    /// let other = Home::new(Vtree::linear(4));
    /// let g = engine.rehome(&f, &other, 0).map_err(|r| r.error)?;
    /// assert!(g.vtree().home().is_some_and(|h| std::sync::Arc::ptr_eq(h.vtree(), other.vtree())));
    /// assert_eq!(engine.model_count(&g)?, 12u32.into());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Each with the diagram ([`RehomeRefused`]):
    /// [`RestructureError::Variables`] for a variable of `f` with no leaf in
    /// `home`; with `bound` 0, [`RestructureError::Bound`] naming the node of
    /// `f`'s vtree with no image in the projection; otherwise those of
    /// [`Engine::restructure_to`], and [`RestructureError::Operation`] for a
    /// summed-out level.
    #[expect(clippy::result_large_err, reason = "the refusal hands back what it was given")]
    pub fn rehome<'a>(&self, f: impl Into<Cow<'a, Tdd>>, home: &Home, bound: usize) -> Result<Tdd, RehomeRefused<'a>> {
        let TransportRefused { error, tdd } = match self.transport(f, home, |v| v) {
            Ok(moved) => return Ok(moved),
            Err(refused) => refused,
        };
        let error = match error {
            EmbedError::NotIsomorphic { source } if bound == 0 => RestructureError::Bound { pivot: source },
            EmbedError::NotIsomorphic { .. } => return self.restructure_onto(tdd, home, bound),
            EmbedError::VariableOutOfRange { variable, .. } => RestructureError::Variables { variable },
            error => crate::apply::extend::operation_error(error).into(),
        };
        Err(RehomeRefused { error, tdd })
    }

    /// The general case of [`rehome`](Self::rehome): `f` restructured onto
    /// the projection of `home` on its variables.
    #[expect(clippy::result_large_err, reason = "the refusal hands back what it was given")]
    fn restructure_onto<'a>(&self, f: Cow<'a, Tdd>, home: &Home, bound: usize) -> Result<Tdd, RehomeRefused<'a>> {
        let vars: Vec<VarId> = f.vtree().leaf_bottomup().map(|(_, var)| var).collect();
        let target = Vtree::projection(home, vars).expect("the transport found every variable in the home");
        let mut moved = match f {
            Cow::Owned(f) => f,
            Cow::Borrowed(g) => match g.try_clone_on(self) {
                Ok(copy) => copy,
                Err(e) => return Err(RehomeRefused { error: e.into(), tdd: Cow::Borrowed(g) }),
            },
        };
        match self.restructure_to(&mut moved, &target, bound) {
            Ok(_) => Ok(moved),
            Err(error) => Err(RehomeRefused { error, tdd: Cow::Owned(moved) }),
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
    let route = Route::find(&Limits::new(), f, &home, |v| v).expect("the lifted home has the moved tree's shape");
    debug_assert!(route.mirrored.is_empty());
    route.seat(eng, f);
}

/// Members `which` of `members`, whose levels are on `own`, a projection
/// that rotations and swaps moved as `home` was lifted from it, seated level
/// for level on one projection of `home` on `own`'s leaves.
pub(super) fn seat_group(eng: &Engine, members: &mut [&mut Tdd], which: &[usize], own: Vtree, home: &Home) {
    let own = Arc::new(own);
    let vars: Vec<VarId> = own.leaf_bottomup().map(|(_, var)| var).collect();
    let into = match vars.is_empty() {
        true => Vtree::empty_projection(home),
        false => Vtree::projection(home, vars.iter().copied()).expect("a home has its projections' leaves"),
    };
    let lim = Limits::new();
    for &i in which {
        let f = &mut *members[i];
        f.vtree = Arc::clone(&own);
        let route = Route::along(&lim, f, Arc::clone(&into), &vars).expect("the home has the moved tree's shape");
        debug_assert!(route.mirrored.is_empty());
        route.seat(eng, f);
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
    fn find(lim: &Limits, f: &Tdd, home: &Home, map: impl Fn(VarId) -> VarId) -> Result<Self, EmbedError> {
        f.require_structure()?;
        let source = f.vtree();
        if source.num_nodes() == 0 {
            return Self::along(lim, f, Vtree::empty_projection(home), &[]);
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
        let into = Vtree::projection(home, images.iter().copied())?;
        if into.num_leaves() != source.num_leaves() {
            images.sort_unstable();
            let shared = images.windows(2).find(|w| w[0] == w[1]).expect("fewer leaves than images");
            return Err(VtreeError::OverlappingVariable(shared[0]).into());
        }
        let route = Self::along(lim, f, into, &images);
        lim.discard(images);
        route
    }

    /// The route of `f` onto `into`, a projection whose leaves are
    /// `images`, the images of `f`'s leaves in bottom-up order.
    fn along(lim: &Limits, f: &Tdd, into: Arc<Vtree>, images: &[VarId]) -> Result<Self, EmbedError> {
        let source = f.vtree();
        let mut route = Self { into, to: Vec::new(), mirrored: Vec::new(), levels: Vec::new(), columns: Vec::new() };
        let n = source.num_nodes();
        if n == 0 {
            return Ok(route);
        }
        lim.try_resize(&mut route.to, n, VtreeIdx(0))?;
        for ((leaf, _), &image) in source.leaf_bottomup().zip(images) {
            route.to[leaf.idx()] = route.into.leaf_of(image).expect("a projected leaf");
        }
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

#[cfg(test)]
#[path = "tests/transport.rs"]
mod tests;
