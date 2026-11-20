//! The end-of-apply pass that inlines small counts into the references
//! toward a marginal child.

use crate::diagram::level::TddLevel;
use crate::limits::{Limits, OperationError};
use super::refs::ChildSide;
use crate::diagram::tdd::Tdd;
use crate::vtree::{Vtree, VtreeIdx};

/// Replace every bare slot reference toward a marginal child whose count is
/// small enough with the count itself (bit 30 set); a larger count keeps
/// its slot reference.
///
/// Called once after each apply completes, before any persisted count decode.
/// Idempotent: a side already through this pass is left alone.
///
/// # Errors
///
/// `Err(OperationError::OverBudget)` when `lim` refuses the move of an
/// implicit level's side ([`TddLevel::inline_small_refs`]). The levels
/// visited before it are through the pass; that side and the ones after it
/// keep bare slot references, which still read the same counts, and are not
/// marked as carrying inline ones.
pub(crate) fn inline_small_marginal_refs(
    lim: &Limits,
    tdd: &mut Tdd,
    // `Some(made)`: `made` lists the levels a marginalization pass made
    // marginal, and only the references into them are rewritten, at the
    // parents that pass left structural. `None` visits every internal level,
    // bottom-up, and skips a side by the level's `has_value_refs` marker.
    made_marginal: Option<&[VtreeIdx]>,
) -> Result<(), OperationError> {
    // Disjoint-field borrow: vtree (shape) immutable, levels (data) mutable.
    let vtree = &tdd.vtree;
    let levels = &mut tdd.levels;
    let Some(made) = made_marginal else {
        return inline_small_marginal_refs_in(lim, vtree, levels, vtree.internal_bottomup_slice(), |_| {});
    };
    for &u in made {
        let Some(t) = vtree.node(u).parent() else { continue };
        if levels[t.idx()].is_marginal() {
            continue;
        }
        let (left, right) = vtree.children(t);
        let tags = (levels[left.idx()].is_marginal(), levels[right.idx()].is_marginal());
        // A parent with both children in `made` is visited once per side.
        inline_small_marginal_refs_at_level(lim, levels, t, (left, right), tags, (u == left, u == right))?;
    }
    Ok(())
}

/// [`inline_small_marginal_refs`] on levels not yet seated as a diagram, at
/// the internal levels `at`, bottom-up: every level with a marginal child.
/// `rewrote` is handed each level whose references the pass rewrote.
///
/// # Errors
///
/// As [`inline_small_marginal_refs`].
pub(crate) fn inline_small_marginal_refs_in(
    lim: &Limits,
    vtree: &Vtree,
    levels: &mut [TddLevel],
    at: &[VtreeIdx],
    mut rewrote: impl FnMut(VtreeIdx),
) -> Result<(), OperationError> {
    // The tests stay in the loop: most levels fail them, and a call per
    // level cost more than the pass's work.
    for &t in at {
        if levels[t.idx()].is_marginal() {
            continue;
        }
        let (left, right) = vtree.children(t);
        let tag_left = levels[left.idx()].is_marginal();
        let tag_right = levels[right.idx()].is_marginal();
        if !(tag_left || tag_right) {
            continue;
        }
        let level = &levels[t.idx()];
        let emit = (
            tag_left && !level.has_value_refs(ChildSide::Left),
            tag_right && !level.has_value_refs(ChildSide::Right),
        );
        if inline_small_marginal_refs_at_level(lim, levels, t, (left, right), (tag_left, tag_right), emit)? {
            rewrote(t);
        }
    }
    Ok(())
}

/// One internal level's share of [`inline_small_marginal_refs`], at a level
/// that is not marginal and has a marginal child (`tag_left`, `tag_right`):
/// rewrite the sides `emit` names, mark every marginal side, and return
/// whether it rewrote.
fn inline_small_marginal_refs_at_level(
    lim: &Limits,
    levels: &mut [TddLevel],
    t: VtreeIdx,
    (left, right): (VtreeIdx, VtreeIdx),
    (tag_left, tag_right): (bool, bool),
    (do_left, do_right): (bool, bool),
) -> Result<bool, OperationError> {
    let ti = t.idx();
    if do_left || do_right {
        let [p, l, r] = levels.get_disjoint_mut([ti, left.idx(), right.idx()]).expect("distinct");
        p.inline_small_refs(
            lim,
            do_left.then(|| l.marginal_counts()).flatten(),
            do_right.then(|| r.marginal_counts()).flatten(),
        )?;
    }
    // Marked off the marginal-child tags, not the emitted sides: a side
    // already inline stays marked so a later pass still skips it.
    if tag_left {
        levels[ti].set_has_value_refs(ChildSide::Left, true);
    }
    if tag_right {
        levels[ti].set_has_value_refs(ChildSide::Right, true);
    }
    Ok(do_left || do_right)
}
