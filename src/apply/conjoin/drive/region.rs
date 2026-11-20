//! The free regions of a conjunction whose operands have free levels
//! ([`ApplyRun::cone`]).
//!
//! A level free in one operand stands for the constant true level, and so
//! does every level under it, so the result's level at each is the other
//! operand's: the sweep would carry each one as an identity level. A region
//! is the subtree under a free level whose parent is not free, its top. The
//! conjunction takes every level of a region into the result before the
//! sweep, publishes each top's products as the identity mapping its parent
//! reads, and the sweep builds only the levels outside the regions: on a
//! small operand placed on a large vtree, the levels with one of its
//! variables under them.

use std::sync::Arc;

use crate::diagram::{ONE_LEAF_IDX, TddLevel};
use crate::limits::{Limits, OperationError};
use crate::vtree::{Vtree, VtreeIdx};

use super::super::held::Held;
use super::super::identity::publish_identity_level;
use super::super::setup::{ApplyRun, Cone, Operands};

/// The operand whose level the result takes at the free level `t`: `f`
/// where `g` is free, as the first identity fast path carries it, also where
/// both are; `g` where only `f` is.
#[inline]
pub(super) fn carried_from_f(cone: Cone<'_>, t: usize) -> bool {
    cone.free_in_g(t)
}

/// Take every region into the result: set each top's identity flags as the
/// sweep would have accreted them (true in both operands where the setup
/// found both constant true there, [`Cone::constant`]), publish its
/// products, and carry the
/// carrier's levels of every free internal node into the output
/// ([`Held::carry`]), building a free level where both operands are free.
/// Returns the live pairs of the levels taken, `closed` saying which
/// operands' levels were closed.
///
/// The tops are read before any level moves: a top's flag reads the
/// carrier's levels under it. A borrowed carrier's levels are copied before
/// any owned carrier's level moves, so that a refused copy leaves both
/// operands as they were.
///
/// # Errors
///
/// [`OperationError::OverBudget`] where a borrowed carrier's copy is
/// refused.
pub(super) fn take_regions(
    vtree: &Arc<Vtree>,
    lim: &Limits,
    run: &mut ApplyRun,
    f: &mut Held<'_>,
    g: &mut Held<'_>,
    closed: Operands<bool>,
) -> Result<usize, OperationError> {
    let cone = run.cone;
    for &t in cone.tops {
        let at = t.idx();
        let stack = &mut run.lists.stack;
        let width = if carried_from_f(cone, at) {
            run.g_identity[at] = true;
            // An adopting conjunction walks no region: a top it does not find
            // constant true costs a dense level at most.
            run.f_identity[at] = cone.constant || (!run.adopted && constant_true(vtree, f, cone, true, t, stack));
            run.f_widths[at]
        } else {
            run.f_identity[at] = true;
            run.g_identity[at] = cone.constant || constant_true(vtree, g, cone, false, t, stack);
            run.g_widths[at]
        };
        publish_identity_level(run.products, at, width);
        run.products.note_complete(at);
    }
    // The copies first, then the moves: a refused copy leaves every owned
    // operand's levels in place.
    if f.is_borrowed() || g.is_borrowed() {
        for &t in cone.free {
            let at = t.idx();
            let out = &mut run.levels[at];
            match carried_from_f(cone, at) {
                true if f.is_borrowed() => f.carry(lim, at, out)?,
                false if g.is_borrowed() => g.carry(lim, at, out)?,
                _ => {}
            }
        }
    }
    // An adopting conjunction leaves `f`'s regions in `f`'s array, which
    // becomes the result's, and takes only each top, which the levels
    // above read.
    let moved = if run.adopted { cone.tops } else { cone.free };
    let mut pairs = 0;
    for &t in moved {
        let at = t.idx();
        let both = cone.free_in_f(at) && cone.free_in_g(at);
        let from_f = carried_from_f(cone, at);
        match from_f {
            true => move_region_level(lim, f, t, both, &mut run.levels[at])?,
            false => move_region_level(lim, g, t, both, &mut run.levels[at])?,
        }
        let level = &run.levels[at];
        pairs += match if from_f { closed.f } else { closed.g } {
            true => level.live_pairs_closed(),
            false => level.live_pairs(),
        };
    }
    Ok(pairs)
}

/// Move an owned carrier's level at `t`, a free level of the other
/// operand, into `out`, built first where it is free in both.
// Forced: with `Held::carry` inlined here, the compiler otherwise calls this
// once per free level.
#[inline(always)]
fn move_region_level(lim: &Limits, carrier: &mut Held<'_>, t: VtreeIdx, both: bool, out: &mut TddLevel) -> Result<(), OperationError> {
    if carrier.is_borrowed() {
        return Ok(());
    }
    if both {
        carrier.build_free(t);
    }
    carrier.carry(lim, t.idx(), out)
}

/// Move the levels [`take_regions`] took back to their owned operands, for
/// a refused conjunction that restores them ([`Held::uncarry`]).
pub(super) fn give_back_regions(
    levels: &mut [TddLevel],
    cone: Cone<'_>,
    f: &mut Held<'_>,
    g: &mut Held<'_>,
) {
    for &t in cone.free {
        let at = t.idx();
        match carried_from_f(cone, at) {
            true => f.uncarry(at, &mut levels[at]),
            false => g.uncarry(at, &mut levels[at]),
        }
    }
}

/// Whether `d`, the operand `f` when `is_f` and `g` otherwise, is constant
/// true under the internal node `t` as the sweep would have found it
/// carrying `d` there: one node at `t` and at every internal level under it,
/// whose pairs read each leaf below as true only, down to the levels free in
/// `d`, which are constant true. `stack` is scratch for the walk.
fn constant_true(
    vtree: &Vtree,
    d: &Held<'_>,
    cone: Cone<'_>,
    is_f: bool,
    t: VtreeIdx,
    stack: &mut Vec<VtreeIdx>,
) -> bool {
    let free_in_d = |t: usize| {
        if is_f {
            cone.free_in_f(t)
        } else {
            cone.free_in_g(t)
        }
    };
    // Most tops hold more than one node: decided without the stack.
    if !free_in_d(t.idx()) && d.level(t.idx()).slot_count() != 1 {
        return false;
    }
    stack.clear();
    stack.push(t);
    while let Some(t) = stack.pop() {
        if free_in_d(t.idx()) {
            continue;
        }
        let level = &d.level(t.idx());
        if level.slot_count() != 1 {
            return false;
        }
        let (left, right) = vtree.children(t);
        for (side, child) in [(false, left), (true, right)] {
            if !vtree.node(child).is_leaf() {
                stack.push(child);
                continue;
            }
            for node in level.nodes().iter() {
                for pair in level.pairs_iter_of(&node) {
                    let read = if side { pair.right } else { pair.left };
                    if read != ONE_LEAF_IDX.into() {
                        return false;
                    }
                }
            }
        }
    }
    true
}
