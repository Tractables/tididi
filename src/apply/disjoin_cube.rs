//! Disjoining a cube: the overlay with an operand of one node a level.
//!
//! A cube `M` restricted to subtree `t` is one function `M_t`, held by one
//! node with the one pair `(M_l, M_r)`; over a subtree the cube leaves free,
//! `M_t` is true. Overlaid with it, a node `a` of `f` has at most two pieces,
//! `a ∩ M_t` and `a ∖ M_t`, and the cube's node one more, the fill
//! `M_t ∖ ⋃a`. A cell of `a`'s pair `(l, r)` is a piece of `l` by a piece of
//! `r`, and lies in `M_t` exactly when both of its components lie in the
//! cube at their levels: one flag per piece, where the general overlay looks
//! each cell up among the other operand's pairs. The fill is the cells of the
//! children's pieces in the cube, less the cells inside the cube that the
//! walk over `f` has just produced.
//!
//! For a cube naming every variable, `M_t` is one value: at most one node of
//! a level meets it, and the fill is a single pair when none does, so each
//! level gains at most one node and one pair beyond the pairs its split node's
//! parents gain. A free subtree's fill is the part of its full level that `f`
//! does not hold, which the result needs whatever builds it.
//!
//! The walk edits `f`'s levels in place. A node keeps its index for its
//! first piece, so a node none of whose children split and whose cells lie
//! all inside or all outside the cube is read and left as it is; only split
//! nodes and the parents of split children are rewritten. The result is not
//! reduced.

use std::sync::Arc;

use crate::Engine;
use crate::apply::conjoin_clause::{Normalized, extend_to};
use crate::diagram::{
    Assembly, ChildPair, EncodedChildRef, NEG_LEAF_IDX, ONE_LEAF_IDX, POS_LEAF_IDX, PairsIter,
    Tdd, TddLevel, TddNodeId, WeightStore,
};
use crate::limits::{Limits, OperationError};
use crate::vtree::VtreeIdx;

/// No piece.
const NONE: u32 = u32::MAX;

/// One level of the result as its parent reads it.
#[derive(Default)]
struct Split {
    /// Per node of `f`, its pieces: the first, and the second or [`NONE`].
    pieces: Vec<[u32; 2]>,
    /// Per output node, whether it lies in the cube.
    in_cube: Vec<bool>,
    /// The output nodes in the cube, ascending: the cube node's pieces.
    of_cube: Vec<u32>,
}

impl Split {
    fn clear(&mut self) {
        self.pieces.clear();
        self.in_cube.clear();
        self.of_cube.clear();
    }

    /// The pieces of a leaf child the parent level of `f` names as `⊤` when
    /// `f_one`, and the cube labels `label`. The result names `⊤` only when
    /// both do; otherwise a `⊤` reference splits into the two literals.
    fn leaf(&mut self, f_one: bool, label: u32) {
        let (one, pos, neg) = (ONE_LEAF_IDX.0, POS_LEAF_IDX.0, NEG_LEAF_IDX.0);
        self.clear();
        if f_one && label == one {
            self.pieces.extend([[one, NONE], [NONE, NONE], [NONE, NONE]]);
            self.in_cube.extend([true, false, false]);
            self.of_cube.push(one);
            return;
        }
        self.pieces.extend([[pos, neg], [pos, NONE], [neg, NONE]]);
        self.in_cube.extend([false, label != neg, label != pos]);
        match label == one {
            true => self.of_cube.extend([pos, neg]),
            false => self.of_cube.push(label),
        }
    }
}

/// Buffers reused from level to level.
#[derive(Default)]
struct Work {
    /// The cells of the node being split outside the cube.
    outside: Vec<ChildPair>,
    /// Its cells inside the cube.
    inside: Vec<ChildPair>,
    /// The inside pieces of split nodes, pushed after every first piece.
    deferred: Vec<ChildPair>,
    /// Per split node: its index in `f` and the end of its run in `deferred`.
    deferred_at: Vec<(u32, u32)>,
    /// The level's cells inside the cube, as keys.
    cells: Vec<u64>,
    /// The fill's pairs.
    fill: Vec<ChildPair>,
}

#[inline]
fn key(p: ChildPair) -> u64 {
    (u64::from(p.left.0) << 32) | u64::from(p.right.0)
}

#[inline]
fn pair(x: u32, y: u32) -> ChildPair {
    ChildPair::new(EncodedChildRef::from_raw(x), EncodedChildRef::from_raw(y))
}

/// Split one node of `f` by the cube: its cells into `w.outside` and
/// `w.inside`, which the caller has emptied. Returns the cells produced.
fn split_node(
    lim: &Limits,
    l: &Split,
    r: &Split,
    pairs: PairsIter<'_>,
    w: &mut Work,
) -> Result<usize, OperationError> {
    let n = pairs.len();
    lim.reserve(&mut w.outside, 4 * n)?;
    lim.reserve(&mut w.inside, 4 * n)?;
    let mut emit = |x: u32, y: u32| {
        let p = pair(x, y);
        if l.in_cube[x as usize] && r.in_cube[y as usize] {
            w.inside.push(p);
        } else {
            w.outside.push(p);
        }
    };
    for p in pairs {
        let [l0, l1] = l.pieces[p.left.0 as usize];
        let [r0, r1] = r.pieces[p.right.0 as usize];
        emit(l0, r0);
        if r1 != NONE {
            emit(l0, r1);
        }
        if l1 != NONE {
            emit(l1, r0);
            if r1 != NONE {
                emit(l1, r1);
            }
        }
    }
    Ok(w.outside.len() + w.inside.len())
}

/// Into `w.fill`, the cells of `l`'s and `r`'s cube pieces that are not in
/// `w.cells`, which it sorts.
fn fill(lim: &Limits, l: &Split, r: &Split, w: &mut Work) -> Result<(), OperationError> {
    w.fill.clear();
    w.cells.sort_unstable();
    let mut next = 0;
    for &x in &l.of_cube {
        for &y in &r.of_cube {
            let k = (u64::from(x) << 32) | u64::from(y);
            if next < w.cells.len() && w.cells[next] == k {
                next += 1;
                continue;
            }
            lim.try_push(&mut w.fill, pair(x, y))?;
        }
    }
    debug_assert_eq!(next, w.cells.len(), "a cell inside the cube outside its pieces");
    Ok(())
}

/// Whether node `a`'s cells are its own pairs, all inside the cube (`true`)
/// or all outside it; `None` when the node has to be rewritten.
fn as_is(l: &Split, r: &Split, pairs: PairsIter<'_>) -> Option<bool> {
    let mut inside = None;
    for p in pairs {
        let [l0, l1] = l.pieces[p.left.0 as usize];
        let [r0, r1] = r.pieces[p.right.0 as usize];
        if l1 != NONE || r1 != NONE || l0 != p.left.0 || r0 != p.right.0 {
            return None;
        }
        let c = l.in_cube[l0 as usize] && r.in_cube[r0 as usize];
        match inside {
            None => inside = Some(c),
            Some(i) if i != c => return None,
            Some(_) => {}
        }
    }
    inside
}

/// Turn one internal, non-root level of `f` into the level of `f ∨ M`, in
/// place, and its pieces into `s`. A node keeps its index for its first
/// piece; the inside pieces of split nodes are pushed after every node, the
/// fill last. A node whose cells are its own pairs, all on one side of the
/// cube, is read and left as it is.
fn split_level(
    lim: &Limits,
    level: &mut TddLevel,
    l: &Split,
    r: &Split,
    w: &mut Work,
    s: &mut Split,
) -> Result<(), OperationError> {
    let mut poll = lim.gate();
    s.clear();
    w.deferred.clear();
    w.deferred_at.clear();
    w.cells.clear();
    let width = level.slot_count();
    lim.reserve_exact(&mut s.pieces, width)?;
    lim.reserve(&mut s.in_cube, width)?;
    for a in 0..width {
        let pairs = level.pairs_iter_of_idx(a);
        let n = pairs.len();
        if let Some(inside) = as_is(l, r, pairs) {
            poll.poll(n as u64 + 1)?;
            s.pieces.push([a as u32, NONE]);
            s.in_cube.push(inside);
            if inside {
                lim.try_push(&mut s.of_cube, a as u32)?;
                lim.reserve(&mut w.cells, n)?;
                w.cells.extend(level.pairs_iter_of_idx(a).map(key));
            }
            continue;
        }
        w.outside.clear();
        w.inside.clear();
        let cells = split_node(lim, l, r, level.pairs_iter_of_idx(a), w)?;
        poll.poll(cells as u64 + 1)?;
        let (first, inside) = if !w.outside.is_empty() {
            level.replace_node_pairs(lim, a, &w.outside)?;
            if !w.inside.is_empty() {
                lim.reserve(&mut w.deferred, w.inside.len())?;
                w.deferred.extend_from_slice(&w.inside);
                lim.try_push(&mut w.deferred_at, (a as u32, w.deferred.len() as u32))?;
            }
            (a as u32, false)
        } else if !w.inside.is_empty() {
            level.replace_node_pairs(lim, a, &w.inside)?;
            lim.try_push(&mut s.of_cube, a as u32)?;
            (a as u32, true)
        } else {
            (NONE, false)
        };
        lim.reserve(&mut w.cells, w.inside.len())?;
        w.cells.extend(w.inside.iter().map(|&p| key(p)));
        s.pieces.push([first, NONE]);
        s.in_cube.push(inside);
    }
    let mut start = 0;
    for &(a, end) in &w.deferred_at {
        let o = level.push_node(lim, &w.deferred[start..end as usize])?.0;
        s.pieces[a as usize][1] = o;
        lim.try_push(&mut s.in_cube, true)?;
        lim.try_push(&mut s.of_cube, o)?;
        start = end as usize;
    }
    fill(lim, l, r, w)?;
    poll.poll(w.fill.len() as u64)?;
    if !w.fill.is_empty() {
        let o = level.push_node(lim, &w.fill)?.0;
        lim.try_push(&mut s.in_cube, true)?;
        lim.try_push(&mut s.of_cube, o)?;
    }
    poll.flush()
}

/// The root's one node: every cell of `f`'s output node, and the fill.
fn root_pairs<'w>(
    lim: &Limits,
    fl: &TddLevel,
    f_out: usize,
    l: &Split,
    r: &Split,
    w: &'w mut Work,
) -> Result<&'w [ChildPair], OperationError> {
    w.outside.clear();
    w.inside.clear();
    w.cells.clear();
    let cells = split_node(lim, l, r, fl.pairs_iter_of_idx(f_out), w)?;
    lim.check_stop()?;
    lim.reserve(&mut w.cells, w.inside.len())?;
    w.cells.extend(w.inside.iter().map(|&p| key(p)));
    fill(lim, l, r, w)?;
    lim.reserve(&mut w.outside, w.inside.len() + w.fill.len())?;
    w.outside.extend_from_slice(&w.inside);
    w.outside.extend_from_slice(&w.fill);
    debug_assert!(cells + w.fill.len() == w.outside.len());
    Ok(&w.outside)
}

/// `f ∨ M` for a structural, nonzero `f` whose output is at the internal
/// root of its vtree, and the cube `M` given by its leaf labels; `f`'s levels
/// become the result's.
fn overlay_cube(eng: &Engine, f: &mut Tdd, label: &[u32]) -> Result<Tdd, OperationError> {
    let lim = eng.limits();
    let vtree = Arc::clone(&f.vtree);
    let root = vtree.root();
    let out_local = f.output.local;
    let mut levels = std::mem::take(&mut f.levels).into_vec();
    let mut built: Vec<Option<Split>> = Vec::new();
    lim.reserve_exact(&mut built, vtree.num_nodes())?;
    built.resize_with(vtree.num_nodes(), || None);
    let (mut leaf_l, mut leaf_r) = (Split::default(), Split::default());
    let mut spare: Vec<Split> = Vec::new();
    let mut w = Work::default();
    let mut out_nodes = 0u64;
    for (t, left, right) in vtree.internal_bottomup() {
        let level = &mut levels[t.idx()];
        let side = |child: VtreeIdx, is_left: bool, leaf: &mut Split, built: &mut [Option<Split>]| {
            if vtree.node(child).is_leaf() {
                leaf.leaf(crate::apply::overlay::names_one(level, is_left), label[child.idx()]);
                None
            } else {
                Some(built[child.idx()].take().expect("a child level is visited before its parent"))
            }
        };
        let lo = side(left, true, &mut leaf_l, &mut built);
        let ro = side(right, false, &mut leaf_r, &mut built);
        let l = lo.as_ref().unwrap_or(&leaf_l);
        let r = ro.as_ref().unwrap_or(&leaf_r);
        if t == root {
            let pairs = root_pairs(lim, level, out_local.idx(), l, r, &mut w)?;
            // The root may hold nodes beside the output (a placement's literal
            // chain), each still naming a cube leaf as `One`: nothing reads
            // them, so the output is rebuilt as the root's only node.
            level.clear();
            let out = level.push_node(lim, pairs)?;
            lim.level_done(out_nodes + 1)?;
            let weights = f.weights.take();
            return Assembly::from_levels(eng, Arc::clone(&vtree), levels, weights)
                .finish(TddNodeId { vtree: root, local: out });
        }
        let mut s = spare.pop().unwrap_or_default();
        split_level(lim, level, l, r, &mut w, &mut s)?;
        out_nodes += s.in_cube.len() as u64;
        lim.level_done(out_nodes)?;
        built[t.idx()] = Some(s);
        spare.extend(lo);
        spare.extend(ro);
    }
    unreachable!("the bottom-up walk ends at the internal root")
}

/// Disjoin the cube `lits` into `f`, consuming it. The implementation behind
/// [`Engine::or_cube`](crate::Engine::or_cube).
///
/// # Errors
///
/// Returns the [`OperationError`] the disjunction stopped on.
pub(crate) fn disjoin_cube_on<L: crate::LiteralInput>(
    eng: &Engine,
    f: Tdd,
    lits: &[L],
) -> Result<Tdd, OperationError> {
    let lim = eng.limits();
    let _op = lim.enter()?;
    let mut out = disjoin_cube(eng, f, lits)?;
    crate::reduce::settle(eng, &mut out, &mut Vec::new())?;
    Ok(out)
}

fn disjoin_cube<L: crate::LiteralInput>(eng: &Engine, f: Tdd, lits: &[L]) -> Result<Tdd, OperationError> {
    let lim = eng.limits();
    let (mut f, cube) = extend_to(eng, f, lits)?;
    let vtree = Arc::clone(&f.vtree);
    let cube = match cube {
        // A variable in both polarities: the cube is false.
        Normalized::Tautology => return Ok(f),
        Normalized::Clause(cube) => cube,
    };
    let mut gate = lim.gate();
    for &(_, leaf) in &cube {
        gate.poll(1)?;
        f.require_structure_at(leaf)?;
    }
    gate.flush()?;
    let literals = || cube.iter().map(|&(lit, _)| lit);
    // The empty cube is true, and `⊥ ∨ M` is the cube itself: both are the
    // cube built on its own, with the accumulator's marginal values.
    if cube.is_empty() || f.is_zero() {
        let mut out = eng.cube(&vtree, literals())?;
        out.weights = f.weights.as_ref().map(WeightStore::empty_like);
        return Ok(out);
    }
    let root = vtree.root();
    if f.output.vtree != root || vtree.node(root).is_leaf() {
        let g = eng.cube(&vtree, literals())?;
        return eng.or(f, g);
    }
    f.require_structure()?;
    // The fill reads every node of a level, so an unreachable one would
    // overlap the reachable cells it duplicates. The loose levels leave out
    // the root, whose other nodes are unreachable.
    if f.levels[root.idx()].slot_count() > 1 || f.dirty.loose().is_none_or(|loose| !loose.is_empty()) {
        let closed = f.levels.is_closed();
        crate::reduce::prune::prune_unreachable(eng, &mut f, crate::reduce::prune::PruneScope::Whole)?;
        f.close_marked_levels(closed);
    }
    let mut label: Vec<u32> = Vec::new();
    lim.try_resize(&mut label, vtree.num_nodes(), ONE_LEAF_IDX.0)?;
    for &(lit, leaf) in &cube {
        label[leaf.idx()] = if lit.sign { POS_LEAF_IDX.0 } else { NEG_LEAF_IDX.0 };
    }
    overlay_cube(eng, &mut f, &label)
}

impl Engine {
    /// Run [`Tdd::or_cube`](crate::Tdd::or_cube) using this batch's scratch
    /// and resource limits. Borrow a slice of signed integers or typed
    /// literals. The output cap is checked after each level, and a stop is
    /// also seen within a level as its work is polled.
    ///
    /// # Errors
    ///
    /// Returns the linked operation's errors; cancellation, allocation refusal and
    /// the output-node cap return [`OperationError::Stopped`],
    /// [`OperationError::OverBudget`] and [`OperationError::OutputCap`], respectively.
    pub fn or_cube<L: crate::LiteralInput>(&self, f: Tdd, cube: &[L]) -> Result<Tdd, OperationError> {
        disjoin_cube_on(self, f, cube)
    }
}

#[cfg(test)]
#[path = "tests/disjoin_cube.rs"]
mod tests;
