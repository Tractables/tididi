//! The end-of-apply pass that inlines small counts into the references
//! toward a marginal child.

use crate::diagram::level::TddLevel;
use super::refs::ChildSide;
use crate::diagram::tdd::Tdd;

/// Replace every bare slot reference toward a marginal child whose count is
/// small enough with the count itself (bit 30 set); a larger count keeps
/// its slot reference.
///
/// Called once after each apply completes, before any persisted count decode.
/// Idempotent: a side already through this pass is left alone.
pub(crate) fn inline_small_marginal_refs(
    tdd: &mut Tdd,
    // `Some(snapshot)`: `snapshot[i]` is whether level `i` was already marginal
    // at the enclosing `marginalize_batch` entry, and only sides whose child
    // became marginal in this batch are emitted (an already inline ref
    // re-read as a slot index would index out of bounds). `None` falls back to
    // the level's `marginal_inlined` markers.
    was_marginal: Option<&[bool]>,
) {
    // Disjoint-field borrow: vtree (shape) immutable, levels (data) mutable.
    let vtree = &tdd.vtree;
    let levels = &mut tdd.levels;
    for (t, left, right) in vtree.internal_bottomup() {
        inline_small_marginal_refs_at_level(levels, was_marginal, t, left, right);
    }
}

/// One internal level's share of [`inline_small_marginal_refs`].
fn inline_small_marginal_refs_at_level(
    levels: &mut [TddLevel],
    was_marginal: Option<&[bool]>,
    t: crate::vtree::VtreeIdx,
    left: crate::vtree::VtreeIdx,
    right: crate::vtree::VtreeIdx,
) {
    let ti = t.idx();
    if levels[ti].is_marginal() {
        return;
    }
    let left_idx = left.idx();
    let right_idx = right.idx();
    let tag_left = levels[left_idx].is_marginal();
    let tag_right = levels[right_idx].is_marginal();
    if !tag_left && !tag_right {
        return;
    }
    // Emit a side iff its child became marginal in this batch; the
    // snapshot is preferred since a rebuild clobbers the marker.
    let do_left = tag_left
        && match was_marginal {
            Some(wm) => !wm[left_idx],
            None => !levels[ti].marginal_inlined(ChildSide::Left),
        };
    let do_right = tag_right
        && match was_marginal {
            Some(wm) => !wm[right_idx],
            None => !levels[ti].marginal_inlined(ChildSide::Right),
        };
    if do_left || do_right {
        let [p, l, r] = levels.get_disjoint_mut([ti, left_idx, right_idx]).expect("distinct");
        p.inline_small_refs(
            do_left.then(|| l.marginal_counts()).flatten(),
            do_right.then(|| r.marginal_counts()).flatten(),
        );
    }
    // Mark the sides now carrying inline counts. Keyed off tag_left/tag_right
    // (the marginal-child predicate), not do_*: an already-inline side stays
    // marked so a later re-tag still skips it.
    if tag_left {
        levels[ti].set_marginal_inlined(ChildSide::Left, true);
    }
    if tag_right {
        levels[ti].set_marginal_inlined(ChildSide::Right, true);
    }
}
