//! Walking the marginal-side references a parent level holds.
//!
//! A pair on a marginal child's side is a [`ValueRef`](super::ValueRef),
//! decoded through the child's [`ChildDecoder`](super::ChildDecoder). The mapping from
//! a side to the word of a node that holds it lives here.

use crate::diagram::{EncodedChildRef, ChildDecoder, NodeIdx, PlannedMove, Tdd, TddLevel};
use crate::limits::{Limits, OperationError};
use crate::vtree::{Vtree, VtreeIdx, VtreeNode};

/// One of a parent's two child sides.
///
/// Only per-level code names a side at runtime; the apply's cell kernel
/// reaches the halves of a [`Sides`] as `.left` / `.right` so the choice is
/// made at compile time (indexing by a runtime side inside the walk would be
/// a branch per access).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ChildSide {
    Left,
    Right,
}

impl ChildSide {
    /// The side at which `child` sits under `parent`.
    ///
    /// # Panics
    ///
    /// Panics if `parent` is a leaf.
    pub(crate) fn of(vtree: &Vtree, parent: VtreeIdx, child: VtreeIdx) -> Self {
        let (left, right) = vtree.children(parent);
        if left == child {
            ChildSide::Left
        } else {
            debug_assert_eq!(right, child, "child must be left or right of parent");
            ChildSide::Right
        }
    }

    /// Return `left` when `self == ChildSide::Left`, and `right` otherwise.
    #[inline(always)]
    pub(crate) fn pick<T>(self, left: T, right: T) -> T {
        match self {
            ChildSide::Left => left,
            ChildSide::Right => right,
        }
    }
}

/// One value per child side.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Sides<T> {
    pub(crate) left: T,
    pub(crate) right: T,
}

impl<T> Sides<T> {
    /// Apply `f` to each side, told which side it is.
    pub(crate) fn map<U>(self, mut f: impl FnMut(ChildSide, T) -> U) -> Sides<U> {
        Sides { left: f(ChildSide::Left, self.left), right: f(ChildSide::Right, self.right) }
    }
}

/// Apply `f` to every reference the nodes of `level` hold on `side`, in
/// node order and, within a node, in pair order.
///
/// `ChildSide::Left` means `pair.left.0` for a multi-pair node and `node.a`
/// for an inline one; `ChildSide::Right` means `pair.right.0` / `node.b`.
///
/// An implicit level has its description's slots moved by `f`
/// ([`TddLevel::move_described`]), which keeps it implicit when the move is
/// one to one and the moved pairs are affine, and stores them otherwise;
/// `f` must then give the same word for the same word. What that move reads
/// and the room it is stored in are reserved through `lim`.
///
/// # Errors
///
/// `Err(OperationError::OverBudget)` when a reservation of an implicit
/// level's move is refused; the level is then as it was. A stored level is
/// rewritten in place and never refused.
#[inline]
pub(crate) fn for_each_side_ref_mut(
    lim: &Limits,
    level: &mut TddLevel,
    side: ChildSide,
    f: impl FnMut(&mut u32),
) -> Result<(), OperationError> {
    if level.pairs.implicit().is_some() {
        let f = std::cell::RefCell::new(f);
        let moved = |x: i64| {
            let mut w = x as u32;
            (f.borrow_mut())(&mut w);
            i64::from(w)
        };
        return level.move_described(lim, side, moved);
    }
    rewrite_stored_side(level, side, f);
    Ok(())
}

/// [`for_each_side_ref_mut`] on a stored level, in place.
#[inline]
fn rewrite_stored_side(level: &mut TddLevel, side: ChildSide, mut f: impl FnMut(&mut u32)) {
    for ni in 0..level.nodes().len() {
        match level.arena_range(level.node(ni).kind()) {
            Some(range) => match side {
                ChildSide::Left => level.pairs.stored_mut()[range].iter_mut().for_each(|p| f(&mut p.left.0)),
                ChildSide::Right => level.pairs.stored_mut()[range].iter_mut().for_each(|p| f(&mut p.right.0)),
            },
            None => {
                let node = &mut level.nodes.stored_mut()[ni];
                f(match side {
                    ChildSide::Left => &mut node.a,
                    ChildSide::Right => &mut node.b,
                });
            }
        }
    }
}

/// Read-only [`for_each_side_ref_mut`]: `f` sees every reference on `side`
/// in the same order.
#[inline]
pub(crate) fn for_each_side_ref(level: &TddLevel, side: ChildSide, mut f: impl FnMut(u32)) {
    for node in level.nodes() {
        for p in level.pairs_iter_of(&node) {
            f(match side {
                ChildSide::Left => p.left.0,
                ChildSide::Right => p.right.0,
            });
        }
    }
}

/// Point every reference into `child_v` — its parent's pair sides on that
/// side, and the diagram output when it sits there — at the cells `remap`
/// names. Answers whether anything moved: an identity remap is skipped
/// outright.
///
/// The sides are decoded through `child_v`'s own [`ChildDecoder`](super::ChildDecoder),
/// so the one walk serves a marginal child (slot refs after a store rewrite)
/// and a structural one (node refs after a merge).
///
/// # Errors
///
/// `Err(OperationError::OverBudget)` when `lim` refuses the move of an
/// implicit parent ([`plan_refs_remap`]); nothing is changed then.
pub(crate) fn remap_refs_into(lim: &Limits, tdd: &mut Tdd, child_v: VtreeIdx, remap: &[u32]) -> Result<bool, OperationError> {
    let view = tdd.levels[child_v.idx()].child_decoder();
    let plan = plan_refs_remap(lim, tdd, child_v, view, remap)?;
    Ok(finish_refs_remap(tdd, plan, remap))
}

/// [`remap_refs_into`] decided before anything changes, for a pass that
/// changes `child_v` before it points the references into it at the cells
/// `remap` names: the parent's sides are decoded through `view`, the decoder
/// `child_v` has once changed, and an implicit parent's move is planned and
/// its room reserved through `lim` ([`TddLevel::plan_move`]), so that
/// [`finish_refs_remap`] cannot be refused.
///
/// # Errors
///
/// `Err(OperationError::OverBudget)` when a reservation of an implicit
/// parent's move is refused.
pub(crate) fn plan_refs_remap(
    lim: &Limits,
    tdd: &Tdd,
    child_v: VtreeIdx,
    view: ChildDecoder,
    remap: &[u32],
) -> Result<RefsRemap, OperationError> {
    if remap.iter().enumerate().all(|(i, &r)| r == i as u32) {
        return Ok(RefsRemap { child_v, parent: None, identity: true });
    }
    let Some(parent) = tdd.vtree.node(child_v).parent() else {
        return Ok(RefsRemap { child_v, parent: None, identity: false });
    };
    let side = ChildSide::of(&tdd.vtree, parent, child_v);
    let level = &tdd.levels[parent.idx()];
    let moved = match level.pairs.implicit() {
        Some(_) => Some(level.plan_move(lim, side, |x| remap_word(view, remap, x))?),
        None => None,
    };
    Ok(RefsRemap { child_v, parent: Some((parent, side, view, moved)), identity: false })
}

/// Point the references into a child level at their cells as `plan` planned
/// it ([`plan_refs_remap`]), with nothing left to reserve. Answers whether
/// anything moved.
pub(crate) fn finish_refs_remap(tdd: &mut Tdd, plan: RefsRemap, remap: &[u32]) -> bool {
    let RefsRemap { child_v, parent, identity } = plan;
    if identity {
        return false;
    }
    // The output names a cell of `child_v` only through an index below the
    // level's width; a leaf label indexes nothing.
    if tdd.output.vtree == child_v && (tdd.output.local.0 as usize) < remap.len() {
        tdd.output.local = NodeIdx(remap[tdd.output.local.idx()]);
    }
    if let Some((parent, side, view, moved)) = parent {
        let level = &mut tdd.levels[parent.idx()];
        match moved {
            Some(moved) => level.finish_move(moved, side, |x| remap_word(view, remap, x)),
            None => rewrite_stored_side(level, side, |r| *r = view.remap(EncodedChildRef::from_raw(*r), remap).0),
        }
        tdd.levels.mark_changed(parent);
    }
    true
}

/// [`remap_refs_into`] of every child level in `children` through the one
/// `remap`, all or nothing: the moves of the implicit parents, of both
/// sides at once where both children are named, are planned and reserved
/// through `lim` before any reference moves.
///
/// # Errors
///
/// `Err(OperationError::OverBudget)` when a reservation is refused; nothing
/// is changed then.
pub(crate) fn remap_refs_into_all(
    lim: &Limits,
    tdd: &mut Tdd,
    children: &[VtreeIdx],
    remap: &[u32],
) -> Result<(), OperationError> {
    // Each parent once, with the sides on which it reads a child named.
    let mut parents: Vec<(VtreeIdx, Sides<bool>)> = Vec::new();
    lim.reserve_exact(&mut parents, children.len())?;
    for &child in children {
        let Some(parent) = tdd.vtree.node(child).parent() else { continue };
        let left = ChildSide::of(&tdd.vtree, parent, child) == ChildSide::Left;
        parents.push((parent, Sides { left, right: !left }));
    }
    parents.sort_unstable_by_key(|&(parent, _)| parent.0);
    parents.dedup_by(|later, kept| {
        let same = later.0 == kept.0;
        if same {
            kept.1 = Sides { left: kept.1.left || later.1.left, right: kept.1.right || later.1.right };
        }
        same
    });
    let views = |tdd: &Tdd, parent: VtreeIdx| {
        let (left, right) = tdd.vtree.children(parent);
        Sides { left: tdd.levels[left.idx()].child_decoder(), right: tdd.levels[right.idx()].child_decoder() }
    };
    let side_map = |moving: bool, view: ChildDecoder| move |x: i64| match moving {
        true => remap_word(view, remap, x),
        false => x,
    };
    let mut planned: Vec<Option<PlannedMove>> = Vec::new();
    lim.reserve_exact(&mut planned, parents.len())?;
    for &(parent, moving) in &parents {
        let level = &tdd.levels[parent.idx()];
        let view = views(tdd, parent);
        planned.push(match level.pairs.implicit() {
            Some(_) => Some(level.plan_moves(
                lim,
                moving,
                side_map(moving.left, view.left),
                side_map(moving.right, view.right),
            )?),
            None => None,
        });
    }
    for (&(parent, moving), planned) in parents.iter().zip(planned) {
        let view = views(tdd, parent);
        let level = &mut tdd.levels[parent.idx()];
        match planned {
            Some(planned) => level.finish_moves(planned, side_map(moving.left, view.left), side_map(moving.right, view.right)),
            None => {
                for (side, moving, view) in [(ChildSide::Left, moving.left, view.left), (ChildSide::Right, moving.right, view.right)] {
                    if moving {
                        rewrite_stored_side(level, side, |r| *r = view.remap(EncodedChildRef::from_raw(*r), remap).0);
                    }
                }
            }
        }
        tdd.levels.mark_changed(parent);
    }
    if children.contains(&tdd.output.vtree) && (tdd.output.local.0 as usize) < remap.len() {
        tdd.output.local = NodeIdx(remap[tdd.output.local.idx()]);
    }
    Ok(())
}

/// The word a parent's side `x` names once remapped, as an implicit level's
/// move reads it.
#[inline]
fn remap_word(view: ChildDecoder, remap: &[u32], x: i64) -> i64 {
    i64::from(view.remap(EncodedChildRef::from_raw(x as u32), remap).0)
}

/// A remap of the references into a child level, decided before anything
/// changes ([`plan_refs_remap`]).
pub(crate) struct RefsRemap {
    child_v: VtreeIdx,
    /// The parent, the side `child_v` sits on, the decoder of its words, and
    /// the planned move when the parent is implicit.
    parent: Option<(VtreeIdx, ChildSide, ChildDecoder, Option<PlannedMove>)>,
    identity: bool,
}

/// The boundary-marginal test for one vtree node: `Some((v, parent, side))`
/// iff `v`'s level is marginal and its vtree parent's level is not.
#[inline]
fn boundary_entry(tdd: &Tdd, v: VtreeIdx) -> Option<(VtreeIdx, VtreeIdx, ChildSide)> {
    if !tdd.levels[v.idx()].is_marginal() {
        return None;
    }
    let parent = tdd.vtree.node(v).parent()?; // root: no parent
    if tdd.levels[parent.idx()].is_marginal() {
        return None; // deep marginal: parent also marginal
    }
    Some((v, parent, ChildSide::of(&tdd.vtree, parent, v)))
}

/// Fill `out` with the boundary marginal levels and their non-marginal
/// parents, in ascending marginal-child index.
///
/// `parents` restricts the result to boundaries under those vtree nodes, and
/// costs two lookups per parent instead of a scan over every level — each
/// parent has at most two boundaries, reached directly. `None` is every level.
pub(crate) fn boundary_marginal_levels_into(
    tdd: &Tdd,
    parents: Option<&[VtreeIdx]>,
    out: &mut Vec<(VtreeIdx, VtreeIdx, ChildSide)>,
) {
    out.clear();
    let Some(parents) = parents else {
        out.extend((0..tdd.levels.len()).filter_map(|i| boundary_entry(tdd, VtreeIdx(i as u32))));
        return;
    };
    for &p in parents {
        if let VtreeNode::Internal { left, right, .. } = tdd.vtree.node(p) {
            out.extend(boundary_entry(tdd, *left));
            out.extend(boundary_entry(tdd, *right));
        }
    }
    // Restore the all-levels ordering and drop the duplicates a repeated
    // parent would yield; a marginal level has exactly one boundary entry.
    out.sort_unstable_by_key(|&(v, _, _)| v.0);
    out.dedup_by_key(|&mut (v, _, _)| v.0);
}

/// Every boundary marginal level with its non-marginal parent, in a fresh
/// `Vec`, for a caller with no buffer to reuse.
pub(crate) fn boundary_marginal_levels(tdd: &Tdd) -> Vec<(VtreeIdx, VtreeIdx, ChildSide)> {
    let mut out = Vec::new();
    boundary_marginal_levels_into(tdd, None, &mut out);
    out
}
