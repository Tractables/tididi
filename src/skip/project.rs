//! A skip-form diagram restricted to some of its variables.
//!
//! The inverse of [`super::embed`], as [`crate::Tdd::project_to_vars`] is of
//! [`crate::Tdd::embed`]. In a canonical diagram that does not depend on the
//! dropped variables no node does, so a node of a level with no kept
//! variable is `true`, and a node of a level with kept variables on one side
//! only is one pair with `true` on the other side, read as its kept side.
//! The levels with kept variables on both sides are the images of the
//! restriction's levels and are copied there, each side read through those
//! maps: `true` over a child is built as normalization writes it, and a side
//! whose node lies further down is lifted to the child as the embedding
//! lifts it ([`Placer`]). The skipped variables between a side's child level
//! and its node are `true`, the dropped ones and the kept ones alike, so the
//! lift's `true` on the kept ones is the restriction of the side.
//!
//! A diagram that depends on a dropped variable shows it at a level of
//! either kind that is not of that form; it is quantified first.

use std::sync::Arc;

use super::embed::Placer;
use super::{discard_nested, filled, ix, mk, vt, Ref, Shape, SkipForm, NONE, ONE, ZREF};
use crate::limits::Limits;
use crate::restructure::embed::Plan;
use crate::OperationError;

/// A side that is `true` over every variable it is read over.
const TOP: Ref = ZREF - 1;

/// `f` restricted to `d` along `plan`, the match of `d`'s vtree in `f`'s;
/// `None` where `f` depends on a variable `d` drops.
pub(crate) fn project(f: &SkipForm, d: &Arc<Shape>, plan: &Plan, lim: &Limits) -> Result<Option<SkipForm>, OperationError> {
    let src = &*f.shape;
    let mut cp = Placer::new(d, filled(lim, d.num_nodes(), NONE)?, lim)?;
    let mut at: Vec<Vec<Ref>> = Vec::new();
    lim.try_resize(&mut at, src.num_nodes(), Vec::new())?;
    let mut pairs = Vec::new();
    let out = walk(f, d, plan, &mut cp, &mut at, &mut pairs, lim);
    lim.discard(pairs);
    discard_nested(lim, at);
    match out? {
        Some(out) => Ok(Some(cp.finish(out)?)),
        None => {
            cp.release().discard(lim);
            Ok(None)
        }
    }
}

/// The restricted output, with `at` the image of each node of `f` (a
/// reference of `d`, or [`TOP`]); `None` where `f` reads a dropped
/// variable.
fn walk(
    f: &SkipForm,
    d: &Shape,
    plan: &Plan,
    cp: &mut Placer<'_>,
    at: &mut [Vec<Ref>],
    pairs: &mut Vec<(Ref, Ref)>,
    lim: &Limits,
) -> Result<Option<Ref>, OperationError> {
    let src = &*f.shape;
    // A reference of `f`'s as its restriction: the same label on a kept
    // leaf, `TOP` for `true`.
    let image = |at: &[Vec<Ref>], r: Ref| -> Option<Ref> {
        let v = vt(r);
        if !src.is_leaf(v) {
            return Some(at[v as usize][ix(r) as usize]);
        }
        match (ix(r), plan.covered_by[v as usize]) {
            (ONE, _) => Some(TOP),
            (label, Some(leaf)) => Some(mk(leaf.0, label)),
            (_, None) => None,
        }
    };
    // That restriction as a side at child level `c` of `d`.
    let side = |cp: &mut Placer<'_>, c: u32, x: Ref| -> Result<Ref, OperationError> {
        if x == TOP {
            return cp.top(c);
        }
        debug_assert!(d.within(vt(x), c), "a side's restriction lies under its child level");
        cp.lift(c, x)
    };
    let mut gate = lim.gate();
    for &t in &src.bottomup {
        let lvl = &f.levels[t as usize];
        if lvl.width() == 0 {
            continue;
        }
        let mut refs = Vec::new();
        lim.reserve_exact(&mut refs, lvl.width())?;
        let (l, r) = src.kids[t as usize];
        for i in 0..lvl.width() {
            let node = lvl.node(i as u32);
            gate.poll(node.len() as u64)?;
            let made = match plan.covered_by[t as usize] {
                // Kept variables on both sides: a level of the restriction.
                Some(u) => {
                    let u = u.0;
                    let (lc, rc) = d.kids[u as usize];
                    pairs.clear();
                    lim.reserve(pairs, node.len())?;
                    for &(a, b) in node {
                        let (Some(a), Some(b)) = (image(at, a), image(at, b)) else { return Ok(None) };
                        pairs.push((side(cp, lc, a)?, side(cp, rc, b)?));
                    }
                    if !d.is_leaf(lc) && !d.is_leaf(rc) {
                        pairs.sort_unstable();
                    }
                    cp.node(u, pairs)?
                }
                // No kept variable: `true` alone.
                None if plan.free[t as usize] => match node {
                    &[(a, b)] if image(at, a) == Some(TOP) && image(at, b) == Some(TOP) => TOP,
                    _ => return Ok(None),
                },
                // Kept variables on one side: that side, `true` on the other.
                None => {
                    let &[(a, b)] = node else { return Ok(None) };
                    let (kept, other) = if plan.free[r as usize] { (a, b) } else { (b, a) };
                    debug_assert!(plan.free[l as usize] != plan.free[r as usize]);
                    if image(at, other) != Some(TOP) {
                        return Ok(None);
                    }
                    match image(at, kept) {
                        Some(x) => x,
                        None => return Ok(None),
                    }
                }
            };
            refs.push(made);
        }
        at[t as usize] = refs;
    }
    gate.flush()?;
    if f.out == ZREF {
        return Ok(Some(ZREF));
    }
    match image(at, f.out) {
        Some(x) => Ok(Some(side(cp, d.root, x)?)),
        None => Ok(None),
    }
}
