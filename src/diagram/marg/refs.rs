//! Walking the marginal-side references a parent level holds.
//!
//! A pair on a marginal child's side is not a plain node index; it is a
//! [`ValueRef`], decoded through the child's [`SideView`]. Two kinds of pass
//! need to read or rewrite those refs in bulk — the slot pruner, which
//! compacts a child's value store and repoints its parent, and the content-twin
//! merge, which repoints a grandparent at a survivor — and both need the same
//! answer to "which word of a node is this side?". That mapping lives here,
//! once.

use crate::diagram::Tdd;
use crate::vtree::{VtreeIdx, VtreeNode};

/// Side of a parent's vtree node at which a marginal child sits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ChildSide {
    Left,
    Right,
}

/// Apply `f` to every reference the nodes of `level` hold on `side`.
///
/// This is the ONE home for the side→field mapping: `ChildSide::Left` means
/// `pair.left.0` for a multi-pair node and `node.a` for an inline one;
/// `ChildSide::Right` means `pair.right.0` / `node.b`. (An inline node stores
/// its single pair in its own `(a, b)` words — and inlining requires `b` to
/// carry no `LEAF_BIT`, so `b` is a plain index there, exactly like
/// `pair.right.0`.) Leaves and tombstones hold no refs and are skipped.
///
/// Used by every pass that rewrites one side's refs through a remap table:
/// slot-prune's parent-ref rewrite (integer and weighted) and the
/// content-twin grandparent rewrite.
///
/// NOT usable by `minimize::prune`'s node-index remap: that loop filters each
/// node on a `reachable` bitmap and rewrites BOTH sides in a single visit —
/// a different traversal, not a `side` instantiation of this one. Don't try to
/// fold it in here.
#[inline]
pub(super) fn for_each_side_ref_mut(
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
/// [`SideView::remap`]'s business. Used by slot-prune's parent rewrite and by
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

/// The boundary-marginal test for ONE vtree node: `Some((v, parent, side))`
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
/// costs O(levels) *per call*, and `apply_p_fusion_inner` is called once per
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
