//! The free regions of a conjunction whose operands have free levels
//! ([`ApplyRun::free`]).
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

use crate::diagram::{Tdd, TddLevel, ONE_LEAF_IDX};
use crate::restructure::placement::push_free_level;
use crate::vtree::{Vtree, VtreeIdx};

use super::super::identity::publish_identity_level;
use super::super::setup::{ApplyRun, Regions};

/// The operand whose level the result takes at the free level `t`: `f`
/// where `g` is free, as the first identity fast path carries it, also where
/// both are; `g` where only `f` is.
#[inline]
fn carried_from_f(free: Regions<'_>, t: usize) -> bool {
    free.free_in_g(t)
}

/// Take every region into the result: set each top's identity flags as the
/// sweep would have accreted them, publish its products, and move the
/// carrier's levels of every free internal node into the output, building a
/// free level where both operands are free.
///
/// One pass from the root down: a top is reached before any level under it,
/// so its operands' levels are read before any of them moves.
pub(super) fn take_regions(vtree: &Arc<Vtree>, run: &mut ApplyRun, f: &mut Tdd, g: &mut Tdd) {
    let free = run.free;
    for (t, _, _) in vtree.internal_bottomup().rev() {
        let at = t.idx();
        if !free.free_at(at) {
            continue;
        }
        let from_f = carried_from_f(free, at);
        if !free.under_free(at) {
            let width = if from_f {
                run.g_identity[at] = true;
                run.f_identity[at] = constant_true(vtree, f, free, true, t);
                run.f_widths[at]
            } else {
                run.f_identity[at] = true;
                run.g_identity[at] = constant_true(vtree, g, free, false, t);
                run.g_widths[at]
            };
            publish_identity_level(run.products, at, width);
            run.products.note_complete(at);
        }
        let carrier = if from_f { &mut *f } else { &mut *g };
        if free.free_in_f(at) && free.free_in_g(at) {
            push_free_level(&mut carrier.levels[at], vtree, t);
        }
        std::mem::swap(&mut run.levels[at], &mut carrier.levels[at]);
    }
}

/// Move the levels [`take_regions`] took back to their operands, for a
/// refused conjunction that restores them.
pub(super) fn give_back_regions(vtree: &Vtree, levels: &mut [TddLevel], free: Regions<'_>, f: &mut Tdd, g: &mut Tdd) {
    for (t, _, _) in vtree.internal_bottomup() {
        let at = t.idx();
        if free.free_at(at) {
            let carrier = if carried_from_f(free, at) { &mut *f } else { &mut *g };
            std::mem::swap(&mut levels[at], &mut carrier.levels[at]);
        }
    }
}

/// Which operand each free internal level came from, in the encoding of
/// the result's carrier map: 1 for `f`, 2 for `g`.
pub(super) fn mark_carriers(vtree: &Vtree, free: Regions<'_>, carrier: &mut [u8]) {
    for (t, _, _) in vtree.internal_bottomup() {
        let at = t.idx();
        if free.free_at(at) {
            carrier[at] = if carried_from_f(free, at) { 1 } else { 2 };
        }
    }
}

/// Whether `d`, the operand `f` when `is_f` and `g` otherwise, is constant
/// true under the internal node `t` as the sweep would have found it
/// carrying `d` there: one node at `t` and at every internal level under it,
/// whose pairs read each leaf below as true only, down to the levels free in
/// `d`, which are constant true.
fn constant_true(vtree: &Vtree, d: &Tdd, free: Regions<'_>, is_f: bool, t: VtreeIdx) -> bool {
    let free_in_d = |t: usize| if is_f { free.free_in_f(t) } else { free.free_in_g(t) };
    // Most tops hold more than one node: decided without the stack.
    if !free_in_d(t.idx()) && d.levels[t.idx()].slot_count() != 1 {
        return false;
    }
    let mut stack = vec![t];
    while let Some(t) = stack.pop() {
        if free_in_d(t.idx()) {
            continue;
        }
        let level = &d.levels[t.idx()];
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
