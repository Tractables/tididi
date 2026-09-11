//! Walking the marginal-side references a parent level holds.
//!
//! A pair on a marginal child's side is not a plain node index; it is a
//! [`ValueRef`](super::ValueRef), decoded through the child's
//! [`SideView`](super::SideView). Two kinds of pass
//! need to read or rewrite those refs in bulk — the slot pruner, which
//! compacts a child's value store and repoints its parent, and the content-twin
//! merge, which repoints a grandparent at a survivor — and both need the same
//! answer to "which word of a node is this side?". That mapping lives here,
//! once.

use crate::diagram::{NodeIdx, Tdd};
use crate::vtree::{VtreeIdx, VtreeNode};

/// Side of a parent's vtree node at which a marginal child sits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ChildSide {
    Left,
    Right,
}

/// Apply `f` to every reference the nodes of `level` hold on `side`.
///
/// This is the one home for the side→field mapping: `ChildSide::Left` means
/// `pair.left.0` for a multi-pair node and `node.a` for an inline one;
/// `ChildSide::Right` means `pair.right.0` / `node.b`. (An inline node stores
/// its single pair in its own `(a, b)` words — and inlining requires `b` to
/// carry no `LEAF_BIT`, so `b` is a plain index there, exactly like
/// `pair.right.0`.) Leaves and tombstones hold no refs and are skipped.
///
/// Used by every pass that rewrites one side's refs: the slot-prune and
/// content-twin remaps go through [`remap_side_refs`], and the leaf-marginal
/// rewrites (label to inline count, and equal-value canonicalization) pass
/// their own closure.
///
/// Not usable by `reduce::prune`'s node-index remap: that loop filters each
/// node on a `reachable` bitmap and rewrites both sides in a single visit —
/// a different traversal, not a `side` instantiation of this one. Don't try to
/// fold it in here.
#[inline]
pub(crate) fn for_each_side_ref_mut(
    level: &mut crate::diagram::TddLevel,
    side: ChildSide,
    mut f: impl FnMut(&mut u32),
) {
    for ni in 0..level.nodes.len() {
        if level.nodes[ni].is_leaf() {
            continue;
        }
        if level.nodes[ni].is_multi() {
            for p in level.pairs_mut(ni) {
                f(match side {
                    ChildSide::Left => &mut p.left.0,
                    ChildSide::Right => &mut p.right.0,
                });
            }
        } else {
            let node = &mut level.nodes[ni];
            f(match side {
                ChildSide::Left => &mut node.a,
                ChildSide::Right => &mut node.b,
            });
        }
    }
}

/// Rewrite every reference `level` holds on `side` through `remap`, indexed by
/// the cells of the child level `view` describes.
///
/// The typed sibling of [`for_each_side_ref_mut`]: the caller says which child
/// level the refs point at and what happened to its cells, and the encoding is
/// [`SideView::remap`](super::SideView::remap)'s business. Used by slot-prune's parent rewrite and by
/// the content-twin grandparent rewrite.
#[inline]
pub(crate) fn remap_side_refs(
    level: &mut crate::diagram::TddLevel,
    side: ChildSide,
    view: crate::diagram::SideView,
    remap: &[u32],
) {
    for_each_side_ref_mut(level, side, |r| {
        *r = view.remap(crate::diagram::NodeIdx(*r), remap).0;
    });
}

/// Point every reference into `child_v` — its parent's pair sides on that
/// side, and the diagram output when it sits there — at the cells `remap`
/// names. Answers whether anything moved: an identity remap is skipped
/// outright.
///
/// The sides are decoded through `child_v`'s own [`SideView`](super::SideView),
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
        let view = tdd.levels[child_v.idx()].side_view();
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
/// iff `v`'s level is marginal and its vtree parent's level is not. This is the
/// single definition of "boundary marginal level" — both collectors below are
/// just different traversals feeding it.
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

/// Fill `out` with every boundary marginal level and its non-marginal parent.
pub(crate) fn boundary_marginal_levels_into(
    tdd: &Tdd,
    out: &mut Vec<(VtreeIdx, VtreeIdx, ChildSide)>,
) {
    out.clear();
    out.extend((0..tdd.levels.len()).filter_map(|i| boundary_entry(tdd, VtreeIdx(i as u32))));
}

/// Iterate boundary marginal levels with their non-marginal parent.
pub(crate) fn boundary_marginal_levels(tdd: &Tdd) -> Vec<(VtreeIdx, VtreeIdx, ChildSide)> {
    let mut out = Vec::new();
    boundary_marginal_levels_into(tdd, &mut out);
    out
}

/// Fill `out` with the boundary marginal levels **whose parent is in
/// `parents`** — the same triples `boundary_marginal_levels` would yield, in
/// the same (ascending marginal-child index) order, restricted to that parent
/// set.
///
/// Why this exists: a boundary's parent is by definition the vtree parent of
/// its marginal level, so a caller that already knows the parents it cares
/// about can reach their (at most two) boundaries directly. The all-levels scan
/// costs O(levels) *per call*, and `fuse_pairs_inner` is called once per
/// marginal-boundary parent inside the contract fixpoint — turning a per-parent
/// constant into an O(parents x levels) sweep over the whole diagram, plus a
/// throwaway `Vec` each time, to keep at most two entries.
pub(crate) fn boundary_marginal_levels_of(
    tdd: &Tdd,
    parents: &[VtreeIdx],
    out: &mut Vec<(VtreeIdx, VtreeIdx, ChildSide)>,
) {
    out.clear();
    for &p in parents {
        if let VtreeNode::Internal { left, right, .. } = tdd.vtree.node(p) {
            out.extend(boundary_entry(tdd, *left));
            out.extend(boundary_entry(tdd, *right));
        }
    }
    // Restore the all-levels traversal's ordering and its once-per-level
    // property (a repeated parent in `parents` would otherwise yield its
    // boundaries twice). A marginal level has exactly one boundary entry, so
    // keying both on the child index is exact.
    out.sort_unstable_by_key(|&(v, _, _)| v.0);
    out.dedup_by_key(|&mut (v, _, _)| v.0);
}
