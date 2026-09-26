//! Disjunction and difference by a direct apply over the operands' common
//! refinement, which never complements a level.
//!
//! The nodes of one level are pairwise disjoint but need not cover their
//! variables, so the complement of a level is its fill: every cell of the
//! children's `lefts x rights` basis that no node holds. De Morgan pays that
//! fill three times per disjunction, and on a level with thousands of lefts
//! and rights it is quadratic where the operands are not.
//!
//! The overlay instead splits each operand node by the other operand's nodes
//! at the same level. For `f`'s node `a` and `g`'s node `b` there are three
//! kinds of piece: the product `a ∧ b`, the residue `a ∧ ¬G` of `a` outside
//! every node of `g`, and the residue `¬F ∧ b`. The pieces of one level are
//! disjoint, and each operand node is the disjoint union of its own pieces, so
//! a pair `(l, r)` of `f` is the disjoint union of the cells `(x, y)` with `x`
//! a piece of `l` and `y` a piece of `r`. Such a cell is also inside exactly
//! one node of `g`, or inside none: the one whose pair is the cell's
//! `g`-components, which a lookup in `g`'s pairs at that level answers. That
//! lookup decides which piece of the level above the cell belongs to, so a
//! residue is decided from its cells and is never computed as a complement.
//!
//! The work at a level is the cells enumerated: for each pair of `f`, the
//! product of how many pieces its two children split into, and the same over
//! `g` for a disjunction. A child that the other operand does not split keeps
//! one piece, so where the operands overlap little this is the size of the
//! operands; the cost that remains is the local complement inside one
//! operand's pair of the other operand's cells, which the result holds before
//! it is minimized.
//!
//! A leaf child is read the same way: its pieces are `{⊤}` when both operands
//! name it as `⊤`, and `{x, ¬x}` otherwise, a `⊤` reference then standing for
//! both literals.

use std::sync::Arc;

use rustc_hash::{FxHashMap, FxHashSet};

use crate::Engine;
use crate::diagram::{
    Assembly, ChildPair, EncodedChildRef, LeafLabel, NodeIdx, Tdd, TddLevel, TddNodeId,
    NEG_LEAF_IDX, ONE_LEAF_IDX, POS_LEAF_IDX,
};
use crate::limits::OperationError;
use crate::reduce::ReductionPlan;
use crate::vtree::{Vtree, VtreeIdx};

/// Which Boolean combination the overlay emits at the root.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum Overlay {
    /// `f ∨ g`.
    Or,
    /// `f ∧ ¬g`.
    AndNot,
}

/// A piece with no component in one of the operands.
const ABSENT: u32 = u32::MAX;

/// A key for a pair of child references, for the per-level lookups.
#[inline]
fn key(l: u32, r: u32) -> u64 {
    (u64::from(l) << 32) | u64::from(r)
}

/// One level of the overlay as its parent reads it: which output nodes are
/// the pieces of each operand node, and which operand nodes each output node
/// is a piece of.
#[derive(Default)]
struct Pieces {
    /// `f_items[f_start[a]..f_start[a + 1]]` are the pieces of `f`'s node `a`.
    f_start: Vec<u32>,
    /// The output nodes, grouped by the `f` node they are pieces of.
    f_items: Vec<u32>,
    /// As `f_start`, for `g`'s nodes; empty when the overlay never walks `g`.
    g_start: Vec<u32>,
    /// As `f_items`, for `g`'s nodes.
    g_items: Vec<u32>,
    /// Output node to the `f` node it lies in, or [`ABSENT`].
    f_part: Vec<u32>,
    /// Output node to the `g` node it lies in, or [`ABSENT`].
    g_part: Vec<u32>,
}

impl Pieces {
    /// The pieces of `f`'s node `a`.
    #[inline]
    fn of_f(&self, a: u32) -> &[u32] {
        let a = a as usize;
        &self.f_items[self.f_start[a] as usize..self.f_start[a + 1] as usize]
    }

    /// The pieces of `g`'s node `b`.
    #[inline]
    fn of_g(&self, b: u32) -> &[u32] {
        let b = b as usize;
        &self.g_items[self.g_start[b] as usize..self.g_start[b + 1] as usize]
    }

    /// Empty every list, keeping the allocations for the next level.
    fn clear(&mut self) {
        self.f_start.clear();
        self.f_items.clear();
        self.g_start.clear();
        self.g_items.clear();
        self.f_part.clear();
        self.g_part.clear();
    }

    /// The four leaf pieces, indexed by [`leaf_index`].
    fn leaves() -> [Pieces; 4] {
        [Pieces::leaf(false, false), Pieces::leaf(false, true), Pieces::leaf(true, false), Pieces::leaf(true, true)]
    }

    /// The pieces of a leaf child, from how each operand's parent level names
    /// it: `⊤` alone when both use `⊤`, else the two literals, each lying in
    /// the operand's `⊤` or in the literal itself.
    fn leaf(f_one: bool, g_one: bool) -> Pieces {
        let (one, pos, neg) = (ONE_LEAF_IDX.0, POS_LEAF_IDX.0, NEG_LEAF_IDX.0);
        if f_one && g_one {
            return Pieces {
                f_start: vec![0, 1, 1, 1],
                f_items: vec![one],
                g_start: vec![0, 1, 1, 1],
                g_items: vec![one],
                f_part: vec![one, ABSENT, ABSENT],
                g_part: vec![one, ABSENT, ABSENT],
            };
        }
        // Literal pieces: a `⊤` reference splits into both, a literal
        // reference is its own piece.
        let side = |is_one: bool| -> (Vec<u32>, Vec<u32>, Vec<u32>) {
            if is_one {
                (vec![0, 2, 2, 2], vec![pos, neg], vec![ABSENT, one, one])
            } else {
                (vec![0, 0, 1, 2], vec![pos, neg], vec![ABSENT, pos, neg])
            }
        };
        let (f_start, f_items, f_part) = side(f_one);
        let (g_start, g_items, g_part) = side(g_one);
        Pieces { f_start, f_items, g_start, g_items, f_part, g_part }
    }
}

/// Whether `level` names its leaf child on `left` (else right) as `⊤`.
///
/// Determinism keeps a level to one form per leaf child, so the first
/// reference decides it. A level with no pairs names nothing and does not
/// constrain the output's form.
fn names_one(level: &TddLevel, left: bool) -> bool {
    level
        .internal_inputs_iter()
        .flat_map(|(_, pairs)| pairs)
        .next()
        .is_none_or(|p| if left { p.left.0 } else { p.right.0 } == ONE_LEAF_IDX.0)
}

/// Which of [`Pieces::leaves`] a leaf child on `left` (else right) of the
/// level `t` has: read off the two operands' forms at `t`.
fn leaf_index(f_level: &TddLevel, g_level: &TddLevel, left: bool) -> usize {
    usize::from(names_one(f_level, left)) * 2 + usize::from(names_one(g_level, left))
}

/// The pieces of an internal `child`, built by its own visit and taken out of
/// `built`; `None` for a leaf, whose pieces are one of [`Pieces::leaves`].
fn take_child(vtree: &Vtree, child: VtreeIdx, built: &mut [Option<Pieces>]) -> Option<Pieces> {
    if vtree.node(child).is_leaf() {
        return None;
    }
    Some(built[child.idx()].take().expect("a child level is visited before its parent"))
}

/// The overlay of two structural, nonzero diagrams on one internal-rooted
/// vtree, before its reduction. Weights are the caller's.
fn overlay_levels(eng: &Engine, op: Overlay, f: &Tdd, g: &Tdd) -> Result<Tdd, OperationError> {
    let vtree = Arc::clone(&f.vtree);
    let root = vtree.root();
    let mut assembly = Assembly::new(eng, &vtree)?;
    let mut built: Vec<Option<Pieces>> = Vec::new();
    eng.limits().reserve_exact(&mut built, vtree.num_nodes())?;
    built.resize_with(vtree.num_nodes(), || None);
    let mut scratch = Scratch::default();
    let mut out_nodes = 0u64;
    // A level's pieces are read once, by its parent; their lists then carry
    // the next level's, so a walk allocates about as many as the vtree is deep.
    let leaves = Pieces::leaves();
    let mut spare: Vec<Pieces> = Vec::new();

    for (t, left, right) in vtree.internal_bottomup() {
        let (fl, gl) = (&f.levels[t.idx()], &g.levels[t.idx()]);
        let lo = take_child(&vtree, left, &mut built);
        let ro = take_child(&vtree, right, &mut built);
        let lp = lo.as_ref().unwrap_or(&leaves[leaf_index(fl, gl, true)]);
        let rp = ro.as_ref().unwrap_or(&leaves[leaf_index(fl, gl, false)]);
        let (levels, _) = assembly.parts_mut();
        let out = &mut levels[t.idx()];
        if t == root {
            let pairs = root_cells(eng, op, fl, gl, f.output.local, g.output.local, lp, rp, &mut scratch)?;
            if pairs.is_empty() {
                drop(assembly);
                return crate::build::constant_on(eng, &vtree, false);
            }
            let local = out.push_node(eng.limits(), pairs)?;
            eng.limits().level_done(out_nodes + 1)?;
            return assembly.finish(TddNodeId { vtree: root, local });
        }
        let mut pieces = spare.pop().unwrap_or_default();
        pieces.clear();
        inner_level(eng, op, fl, gl, lp, rp, out, &mut scratch, &mut pieces)?;
        out_nodes += pieces.f_part.len() as u64;
        eng.limits().level_done(out_nodes)?;
        built[t.idx()] = Some(pieces);
        spare.extend(lo);
        spare.extend(ro);
    }
    unreachable!("the bottom-up walk ends at the internal root")
}

/// Buffers one overlay reuses from level to level.
#[derive(Default)]
struct Scratch {
    /// One operand node's cells, each tagged with the other operand's node it
    /// lies in.
    tagged: Vec<(u32, u32, u32)>,
    /// A node's pair list as it is pushed.
    pairs: Vec<ChildPair>,
    /// Pair of the other operand at this level to the node that holds it.
    owner: FxHashMap<u64, u32>,
    /// Pairs of one operand at this level.
    held: FxHashSet<u64>,
    /// Products of the level, as `(g node, output node)`.
    products: Vec<(u32, u32)>,
}

/// Fill `owner` with every pair of `level` and the node holding it.
fn index_owners(eng: &Engine, level: &TddLevel, owner: &mut FxHashMap<u64, u32>) -> Result<(), OperationError> {
    owner.clear();
    eng.limits().reserve_map(owner, level.live_pairs())?;
    for (n, pairs) in level.internal_inputs_iter() {
        for p in pairs {
            owner.insert(key(p.left.0, p.right.0), n as u32);
        }
    }
    Ok(())
}

/// Fill `held` with the pairs of `nodes` at `level`.
fn index_pairs(
    eng: &Engine,
    level: &TddLevel,
    nodes: impl Iterator<Item = usize>,
    held: &mut FxHashSet<u64>,
) -> Result<(), OperationError> {
    held.clear();
    for n in nodes {
        let pairs = level.pairs_of_idx(n);
        eng.limits().reserve_set(held, pairs.len())?;
        for p in pairs {
            held.insert(key(p.left.0, p.right.0));
        }
    }
    Ok(())
}

/// Build one internal, non-root level of the overlay into `out` and its
/// pieces into the empty `pieces`.
///
/// Every node of `f` is split by `g`: its cells are grouped by the node of
/// `g` they lie in, each group a product node and the ungrouped rest the
/// residue. A disjunction also walks `g` and keeps the cells no node of `f`
/// holds, as `g`'s residues; the cells both hold are the products already
/// built.
#[allow(clippy::too_many_arguments)]
fn inner_level(
    eng: &Engine,
    op: Overlay,
    fl: &TddLevel,
    gl: &TddLevel,
    lp: &Pieces,
    rp: &Pieces,
    out: &mut TddLevel,
    s: &mut Scratch,
    pieces: &mut Pieces,
) -> Result<(), OperationError> {
    let lim = eng.limits();
    let mut poll = lim.gate();
    let walk_g = op == Overlay::Or;
    let f_width = fl.slot_count();
    let g_width = gl.slot_count();
    index_owners(eng, gl, &mut s.owner)?;
    lim.reserve_exact(&mut pieces.f_start, f_width + 1)?;
    pieces.f_start.push(0);
    s.products.clear();

    for a in 0..f_width {
        s.tagged.clear();
        for p in fl.pairs_of_idx(a) {
            let xs = lp.of_f(p.left.0);
            let ys = rp.of_f(p.right.0);
            for &x in xs {
                let gx = lp.g_part[x as usize];
                for &y in ys {
                    let gy = rp.g_part[y as usize];
                    let b = if gx == ABSENT || gy == ABSENT {
                        ABSENT
                    } else {
                        s.owner.get(&key(gx, gy)).copied().unwrap_or(ABSENT)
                    };
                    lim.try_push(&mut s.tagged, (b, x, y))?;
                }
            }
            poll.poll((xs.len() * ys.len()) as u64 + 1)?;
        }
        s.tagged.sort_unstable();
        let mut i = 0;
        while i < s.tagged.len() {
            let b = s.tagged[i].0;
            s.pairs.clear();
            while i < s.tagged.len() && s.tagged[i].0 == b {
                let (_, x, y) = s.tagged[i];
                lim.try_push(&mut s.pairs, ChildPair::new(EncodedChildRef::from_raw(x), EncodedChildRef::from_raw(y)))?;
                i += 1;
            }
            let o = out.push_node(lim, &s.pairs)?.0;
            lim.try_push(&mut pieces.f_items, o)?;
            lim.try_push(&mut pieces.f_part, a as u32)?;
            lim.try_push(&mut pieces.g_part, b)?;
            if walk_g && b != ABSENT {
                lim.try_push(&mut s.products, (b, o))?;
            }
        }
        lim.try_push(&mut pieces.f_start, pieces.f_items.len() as u32)?;
    }

    if walk_g {
        index_pairs(eng, fl, 0..f_width, &mut s.held)?;
        // Stable by `g` node, so each node's products stay in output order.
        s.products.sort_by_key(|&(b, _)| b);
        lim.reserve_exact(&mut pieces.g_start, g_width + 1)?;
        pieces.g_start.push(0);
        let mut next = 0;
        for b in 0..g_width {
            while next < s.products.len() && s.products[next].0 == b as u32 {
                lim.try_push(&mut pieces.g_items, s.products[next].1)?;
                next += 1;
            }
            s.pairs.clear();
            for p in gl.pairs_of_idx(b) {
                let xs = lp.of_g(p.left.0);
                let ys = rp.of_g(p.right.0);
                for &x in xs {
                    let fx = lp.f_part[x as usize];
                    for &y in ys {
                        let fy = rp.f_part[y as usize];
                        if fx != ABSENT && fy != ABSENT && s.held.contains(&key(fx, fy)) {
                            continue;
                        }
                        lim.try_push(&mut s.pairs, ChildPair::new(EncodedChildRef::from_raw(x), EncodedChildRef::from_raw(y)))?;
                    }
                }
                poll.poll((xs.len() * ys.len()) as u64 + 1)?;
            }
            if !s.pairs.is_empty() {
                crate::diagram::sort_pairs(&mut s.pairs);
                let o = out.push_node(lim, &s.pairs)?.0;
                lim.try_push(&mut pieces.g_items, o)?;
                lim.try_push(&mut pieces.f_part, ABSENT)?;
                lim.try_push(&mut pieces.g_part, b as u32)?;
            }
            lim.try_push(&mut pieces.g_start, pieces.g_items.len() as u32)?;
        }
    }
    poll.flush()?;
    Ok(())
}

/// The root's one node: the cells of `f_out`, and for a disjunction the cells
/// of `g_out` outside `f_out`; for a difference, the cells of `f_out` outside
/// `g_out`. Returned sorted, in `s.pairs`, and empty when the result is false.
#[allow(clippy::too_many_arguments)]
fn root_cells<'s>(
    eng: &Engine,
    op: Overlay,
    fl: &TddLevel,
    gl: &TddLevel,
    f_out: NodeIdx,
    g_out: NodeIdx,
    lp: &Pieces,
    rp: &Pieces,
    s: &'s mut Scratch,
) -> Result<&'s [ChildPair], OperationError> {
    let lim = eng.limits();
    let mut poll = lim.gate();
    s.pairs.clear();
    // The operand whose pairs a cell is tested against.
    let excluded = match op {
        Overlay::Or => fl,
        Overlay::AndNot => gl,
    };
    let excluded_out = match op {
        Overlay::Or => f_out,
        Overlay::AndNot => g_out,
    };
    index_pairs(eng, excluded, std::iter::once(excluded_out.idx()), &mut s.held)?;
    if op == Overlay::Or {
        for p in fl.pairs_of_idx(f_out.idx()) {
            let xs = lp.of_f(p.left.0);
            let ys = rp.of_f(p.right.0);
            for &x in xs {
                for &y in ys {
                    lim.try_push(&mut s.pairs, ChildPair::new(EncodedChildRef::from_raw(x), EncodedChildRef::from_raw(y)))?;
                }
            }
            poll.poll((xs.len() * ys.len()) as u64 + 1)?;
        }
    }
    // The walked operand: `g` for a disjunction, `f` for a difference; a cell
    // is kept when its components in the other operand are not a pair of that
    // operand's output.
    let (walked, walked_out) = match op {
        Overlay::Or => (gl, g_out),
        Overlay::AndNot => (fl, f_out),
    };
    for p in walked.pairs_of_idx(walked_out.idx()) {
        let (xs, ys, x_part, y_part) = match op {
            Overlay::Or => (lp.of_g(p.left.0), rp.of_g(p.right.0), &lp.f_part, &rp.f_part),
            Overlay::AndNot => (lp.of_f(p.left.0), rp.of_f(p.right.0), &lp.g_part, &rp.g_part),
        };
        for &x in xs {
            let cx = x_part[x as usize];
            for &y in ys {
                let cy = y_part[y as usize];
                if cx != ABSENT && cy != ABSENT && s.held.contains(&key(cx, cy)) {
                    continue;
                }
                lim.try_push(&mut s.pairs, ChildPair::new(EncodedChildRef::from_raw(x), EncodedChildRef::from_raw(y)))?;
            }
        }
        poll.poll((xs.len() * ys.len()) as u64 + 1)?;
    }
    poll.flush()?;
    crate::diagram::sort_pairs(&mut s.pairs);
    Ok(&s.pairs)
}

/// The two-cell truth table of a leaf label: `(x, ¬x)`.
fn leaf_truth(local: NodeIdx) -> (bool, bool) {
    match LeafLabel::from_idx(local.idx()) {
        LeafLabel::One => (true, true),
        LeafLabel::Pos => (true, false),
        LeafLabel::Neg => (false, true),
        LeafLabel::Zero => (false, false),
    }
}

/// The overlay on a one-variable vtree, whose root is a leaf.
fn overlay_leaf_root(eng: &Engine, op: Overlay, f: &Tdd, g: &Tdd) -> Result<Tdd, OperationError> {
    let (fp, fn_) = leaf_truth(f.output.local);
    let (gp, gn) = leaf_truth(g.output.local);
    let (p, n) = match op {
        Overlay::Or => (fp || gp, fn_ || gn),
        Overlay::AndNot => (fp && !gp, fn_ && !gn),
    };
    let local = match (p, n) {
        (true, true) => ONE_LEAF_IDX,
        (true, false) => POS_LEAF_IDX,
        (false, true) => NEG_LEAF_IDX,
        (false, false) => return crate::build::constant_on(eng, &f.vtree, false),
    };
    let levels = crate::diagram::try_take_levels(eng, f.vtree.num_nodes())?;
    Assembly::from_levels(eng, Arc::clone(&f.vtree), levels, None)
        .finish(TddNodeId { vtree: f.vtree.root(), local })
}

/// `f ∨ g` or `f ∧ ¬g` of two owned operands on one vtree, minimized.
///
/// Validates both operands, aligns their weights, and returns the result with
/// them. A false operand short-cuts: the other one, or false.
pub(crate) fn overlay_on(eng: &Engine, op: Overlay, mut f: Tdd, mut g: Tdd) -> Result<Tdd, OperationError> {
    crate::apply::check_vtree(&f, &g)?;
    f.require_structure()?;
    g.require_structure()?;
    crate::apply::prepare_weights(&mut [&mut f, &mut g])?;
    eng.limits().check_stop()?;
    let weights = f.weights.take();
    let mut result = match (f.is_zero(), g.is_zero(), op) {
        (true, _, Overlay::AndNot) => crate::build::constant_on(eng, &f.vtree, false)?,
        (true, _, Overlay::Or) => g,
        (false, true, _) => f,
        (false, false, _) if f.vtree.node(f.vtree.root()).is_leaf() => overlay_leaf_root(eng, op, &f, &g)?,
        (false, false, _) => overlay_levels(eng, op, &f, &g)?,
    };
    result.weights = weights;
    eng.reduce(&mut result, ReductionPlan::default())?;
    Ok(result)
}

/// Return `f ∧ ¬g`, the models of `f` that are not models of `g`, for two
/// structural diagrams sharing a vtree allocation.
///
/// Uses the shared vtree's execution context automatically. Unlike
/// `f & !g`, it never builds the complement of `g`: each node of `f` is split
/// by the nodes of `g` at its level and the pieces outside `g` are kept.
///
/// Both operands are consumed on success and on error. The result is minimized.
/// Weight compatibility and inheritance follow [`and`](crate::and).
///
/// ```
/// use std::sync::Arc;
/// use tididi::{and_not, literal, Tdd, Vtree};
///
/// let vtree = Arc::new(Vtree::balanced(3));
/// let first = literal(&vtree, 1)?;
/// let first_two = Tdd::cube(&vtree, [1, 2])?;
/// let f = and_not(first, first_two)?;
/// assert_eq!(f.model_count()?, 2u32.into());
/// # Ok::<(), tididi::OperationError>(())
/// ```
///
/// # Errors
///
/// [`OperationError::VtreeMismatch`] for different vtree allocations,
/// [`OperationError::IncompatibleWeights`] for different weight interpretations,
/// or [`OperationError::MarginalLevel`] if either operand has discarded structure,
/// even when the other operand is false. Allocation refusals and installed
/// limits propagate as [`OperationError::OverBudget`],
/// [`OperationError::Stopped`] and [`OperationError::OutputCap`].
pub fn and_not(f: Tdd, g: Tdd) -> Result<Tdd, OperationError> {
    let context = Arc::clone(f.context());
    context.run(|eng| eng.and_not(f, g))
}

impl Engine {
    /// Run [`and_not`] using this batch's scratch and resource limits.
    ///
    /// # Errors
    ///
    /// Returns the operation's errors, plus [`OperationError::Stopped`] or
    /// [`OperationError::OutputCap`] when an installed limit refuses the work.
    pub fn and_not(&self, f: Tdd, g: Tdd) -> Result<Tdd, OperationError> {
        let _op = self.limits().enter()?;
        overlay_on(self, Overlay::AndNot, f, g)
    }
}

#[cfg(test)]
#[path = "tests/overlay.rs"]
mod tests;
