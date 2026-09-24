//! Walking the marginal-side references a parent level holds.
//!
//! A pair on a marginal child's side is a [`ValueRef`](super::ValueRef),
//! decoded through the child's [`ChildDecoder`](super::ChildDecoder). The mapping from
//! a side to the word of a node that holds it lives here.

use crate::diagram::{EncodedChildRef, ChildDecoder, NodeIdx, Tdd, TddLevel};
use crate::vtree::{VtreeIdx, VtreeNode};

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
#[inline]
pub(crate) fn for_each_side_ref_mut(
    level: &mut TddLevel,
    side: ChildSide,
    mut f: impl FnMut(&mut u32),
) {
    for ni in 0..level.nodes.len() {
        match level.arena_range(level.nodes[ni].kind()) {
            Some(range) => match side {
                ChildSide::Left => level.pairs[range].iter_mut().for_each(|p| f(&mut p.left.0)),
                ChildSide::Right => level.pairs[range].iter_mut().for_each(|p| f(&mut p.right.0)),
            },
            None => {
                let node = &mut level.nodes[ni];
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
    for node in &level.nodes {
        for p in level.pairs_of(node) {
            f(match side {
                ChildSide::Left => p.left.0,
                ChildSide::Right => p.right.0,
            });
        }
    }
}

/// Rewrite every reference `level` holds on `side` through `remap`, indexed by
/// the cells of the child level `view` describes.
///
/// [`for_each_side_ref_mut`] with [`ChildDecoder::remap`](super::ChildDecoder::remap)
/// as the rewrite.
#[inline]
pub(crate) fn remap_side_refs(
    level: &mut crate::diagram::TddLevel,
    side: ChildSide,
    view: ChildDecoder,
    remap: &[u32],
) {
    for_each_side_ref_mut(level, side, |r| {
        *r = view.remap(EncodedChildRef::from_raw(*r), remap).0;
    });
}

/// Point every reference into `child_v` — its parent's pair sides on that
/// side, and the diagram output when it sits there — at the cells `remap`
/// names. Answers whether anything moved: an identity remap is skipped
/// outright.
///
/// The sides are decoded through `child_v`'s own [`ChildDecoder`](super::ChildDecoder),
/// so the one walk serves a marginal child (slot refs after a store rewrite)
/// and a structural one (node refs after a merge).
pub(crate) fn remap_refs_into(tdd: &mut Tdd, child_v: VtreeIdx, remap: &[u32]) -> bool {
    if remap.iter().enumerate().all(|(i, &r)| r == i as u32) {
        return false;
    }
    // The output names a cell of `child_v` only through an index below the
    // level's width; a leaf label indexes nothing.
    if tdd.output.vtree == child_v && (tdd.output.local.0 as usize) < remap.len() {
        tdd.output.local = NodeIdx(remap[tdd.output.local.idx()]);
    }
    if let Some(parent) = tdd.vtree.node(child_v).parent() {
        let side = side_of(tdd, parent, child_v);
        let view = tdd.levels[child_v.idx()].child_decoder();
        remap_side_refs(&mut tdd.levels[parent.idx()], side, view, remap);
    }
    true
}

/// Locate the side at which `child` sits in `parent`.
fn side_of(tdd: &Tdd, parent: VtreeIdx, child: VtreeIdx) -> ChildSide {
    match tdd.vtree.node(parent) {
        VtreeNode::Internal { left, right, .. } => {
            if *left == child {
                ChildSide::Left
            } else {
                debug_assert_eq!(*right, child, "child must be left or right of parent");
                ChildSide::Right
            }
        }
        _ => panic!("parent must be internal vtree node"),
    }
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
    Some((v, parent, side_of(tdd, parent, v)))
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
