//! The end-of-apply tagger: making every persisted marg-side reference
//! self-describing.

use crate::diagram::level::TddLevel;
use crate::diagram::tdd::Tdd;

/// End-of-apply production tagger (Phase A): set the slot tag on every persisted
/// marg-side ref across the whole TDD whose child level is marginal.
///
/// Called once after each apply completes, before minimize/canon/query — the
/// boundary at which all intra-apply structural reads (which use raw indices)
/// are done and the first persisted count-decode is about to happen. Path-
/// independent (covers every level-build fast path) and idempotent across the
/// repeated applies of an accumulating compile. This is the single writer
/// chokepoint the strict decode assert in `resolve_marg_ref` audits.
pub(crate) fn tag_all_marg_side_slots(
    tdd: &mut Tdd,
    // No-re-expand: `Some(snapshot)` where `snapshot[i]` is whether level `i`
    // was ALREADY marginal at the enclosing `marginalize_batch` entry. When
    // present, it replaces the lossy `marg_inlined_left/right` marker as the
    // discriminator for which child sides to (re)emit: only sides whose child
    // became marginal *in this batch* hold bare-coord refs needing resolution;
    // already-marginal children carry inline counts from a prior end-sweep and
    // must be skipped (re-resolving an inline value as a slot index → OOB). The
    // reexpand baseline passes `None` and keeps the marker (byte-identical).
    was_marginal: Option<&[bool]>,
) {
    tag_all_marg_side_slots_at(tdd, was_marginal, None);
}

/// [`tag_all_marg_side_slots`] over a caller-chosen subset of internal levels.
///
/// `only` is the spine-bounded apply's rebuild set. Restricting the sweep is
/// result-identical there, not merely sound: the body below does work ONLY at a
/// structural level with at least one marginal child, every such level is in the
/// rebuild set by construction, and off the set neither the level nor its
/// children were touched by the apply — so the skipped iterations would
/// re-derive tags the accumulator already carries. `None` = every internal level
/// (the unrestricted end-of-apply sweep).
pub(crate) fn tag_all_marg_side_slots_at(
    tdd: &mut Tdd,
    was_marginal: Option<&[bool]>,
    only: Option<&[crate::vtree::VtreeIdx]>,
) {
    // Disjoint-field borrow: vtree (shape) immutable, levels (data) mutable.
    let vtree = &tdd.vtree;
    let levels = &mut tdd.levels;
    match only {
        Some(sel) => {
            for &t in sel {
                let (left, right) = vtree.children(t);
                tag_marg_side_slots_at_level(levels, was_marginal, t, left, right);
            }
        }
        None => {
            for (t, left, right) in vtree.internal_bottomup() {
                tag_marg_side_slots_at_level(levels, was_marginal, t, left, right);
            }
        }
    }
}

/// One internal level's share of [`tag_all_marg_side_slots_at`] — the whole
/// per-level body, so the full and restricted sweeps run the identical code.
fn tag_marg_side_slots_at_level(
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
        let li = left.idx();
        let ri = right.idx();
        let tag_left = levels[li].is_marginal();
        let tag_right = levels[ri].is_marginal();
        if !tag_left && !tag_right {
            return;
        }
        // Discriminator: process (resolve bare coords / emit inline) a side iff
        // its child became marginal in THIS batch. Prefer the reliable
        // `was_marginal` snapshot (the marker is clobbered by rebuilds);
        // otherwise fall back to the per-level marker.
        let do_left = tag_left
            && match was_marginal {
                Some(wm) => !wm[li],
                None => !levels[ti].marg_inlined_left(),
            };
        let do_right = tag_right
            && match was_marginal {
                Some(wm) => !wm[ri],
                None => !levels[ti].marg_inlined_right(),
            };
        match (do_left, do_right) {
            (true, true) => {
                let [p, l, r] = levels.get_disjoint_mut([ti, li, ri]).expect("distinct");
                p.emit_marg_side_slots(l.marginal_counts.as_deref(), r.marginal_counts.as_deref());
            }
            (true, false) => {
                let [p, l] = levels.get_disjoint_mut([ti, li]).expect("distinct");
                p.emit_marg_side_slots(l.marginal_counts.as_deref(), None);
            }
            (false, true) => {
                let [p, r] = levels.get_disjoint_mut([ti, ri]).expect("distinct");
                p.emit_marg_side_slots(None, r.marginal_counts.as_deref());
            }
            (false, false) => {} // both sides already inline — nothing to emit
        }
        // Mark the sides now carrying inline counts. Keyed off tag_left/tag_right
        // (the marginal-child predicate), not do_*: an already-inline side stays
        // marked so a later re-tag still skips it.
        if tag_left {
            levels[ti].set_marg_inlined_left(true);
        }
        if tag_right {
            levels[ti].set_marg_inlined_right(true);
        }
    }
}
