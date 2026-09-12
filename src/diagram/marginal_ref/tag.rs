//! The end-of-apply tagger: making every persisted marginal-side reference
//! self-describing.

use crate::diagram::level::TddLevel;
use crate::diagram::tdd::Tdd;

/// Set the slot tag on every persisted marginal-side reference in the diagram
/// whose child level is marginal.
///
/// Called once after each apply completes, before any persisted count decode.
/// Idempotent: an already inline side is left alone.
pub(crate) fn tag_all_marginal_side_slots(
    tdd: &mut Tdd,
    // `Some(snapshot)`: `snapshot[i]` is whether level `i` was already marginal
    // at the enclosing `marginalize_batch` entry, and only sides whose child
    // became marginal in this batch are emitted (an already inline ref
    // re-read as a slot index would index out of bounds). `None` falls back to
    // the level's `marginal_inlined_left/right` markers.
    was_marginal: Option<&[bool]>,
) {
    // Disjoint-field borrow: vtree (shape) immutable, levels (data) mutable.
    let vtree = &tdd.vtree;
    let levels = &mut tdd.levels;
    for (t, left, right) in vtree.internal_bottomup() {
        tag_marginal_side_slots_at_level(levels, was_marginal, t, left, right);
    }
}

/// One internal level's share of [`tag_all_marginal_side_slots`].
fn tag_marginal_side_slots_at_level(
    levels: &mut [TddLevel],
    was_marginal: Option<&[bool]>,
    t: crate::vtree::VtreeIdx,
    left: crate::vtree::VtreeIdx,
    right: crate::vtree::VtreeIdx,
) {
    {
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
                None => !levels[ti].marginal_inlined_left(),
            };
        let do_right = tag_right
            && match was_marginal {
                Some(wm) => !wm[right_idx],
                None => !levels[ti].marginal_inlined_right(),
            };
        match (do_left, do_right) {
            (true, true) => {
                let [p, l, r] = levels.get_disjoint_mut([ti, left_idx, right_idx]).expect("distinct");
                p.emit_marginal_side_slots(l.marginal_counts(), r.marginal_counts());
            }
            (true, false) => {
                let [p, l] = levels.get_disjoint_mut([ti, left_idx]).expect("distinct");
                p.emit_marginal_side_slots(l.marginal_counts(), None);
            }
            (false, true) => {
                let [p, r] = levels.get_disjoint_mut([ti, right_idx]).expect("distinct");
                p.emit_marginal_side_slots(None, r.marginal_counts());
            }
            (false, false) => {} // both sides already inline — nothing to emit
        }
        // Mark the sides now carrying inline counts. Keyed off tag_left/tag_right
        // (the marginal-child predicate), not do_*: an already-inline side stays
        // marked so a later re-tag still skips it.
        if tag_left {
            levels[ti].set_marginal_inlined_left(true);
        }
        if tag_right {
            levels[ti].set_marginal_inlined_right(true);
        }
    }
}
