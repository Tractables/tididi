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
//! operand's pair of the other operand's cells, which the result holds until
//! it is minimized.
//!
//! A leaf child is read the same way: its pieces are `{⊤}` when both operands
//! name it as `⊤`, and `{x, ¬x}` otherwise, a `⊤` reference then standing for
//! both literals.
//!
//! Operands on different leaf sets run on the projection on their union,
//! each read where it is: at a node where an operand has no level of its
//! own, its nodes are those of the one child holding its leaves, or one node
//! over the whole space where it has none ([`Placed`]), and no level is
//! built for it.

use std::borrow::Cow;
use std::sync::Arc;

use rustc_hash::{FxHashMap, FxHashSet};

use crate::Engine;
use crate::diagram::{
    Assembly, ChildPair, EncodedChildRef, LeafLabel, NEG_LEAF_IDX, NodeIdx, ONE_LEAF_IDX,
    POS_LEAF_IDX, Tdd, TddLevel, TddNodeId,
};
use crate::limits::{Limits, OperationError};
use crate::vtree::{LeafSets, Reading, UnionReader, Vtree, VtreeIdx};

use super::runs::Runs;

/// Which Boolean combination the overlay emits at the root.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum Overlay {
    /// `f ∨ g`.
    Or,
    /// `f ∧ ¬g`.
    AndNot,
    /// `f ∨ g` where `f ∧ g` is false: a shared model is refused with
    /// [`OperationError::SharedModel`], the sum `f + g` of two structural
    /// operands ([`add`](crate::add)).
    Add,
}

/// A piece with no component in one of the operands.
pub(super) const ABSENT: u32 = u32::MAX;

/// A key for a pair of child references, for the per-level lookups.
#[inline]
fn key(l: u32, r: u32) -> u64 {
    (u64::from(l) << 32) | u64::from(r)
}

/// One level of the overlay as its parent reads it: which output nodes are
/// the pieces of each operand node, and which operand nodes each output node
/// is a piece of.
#[derive(Default)]
pub(super) struct Pieces {
    /// The pieces of `f`'s node `a`, as run `a`: output nodes.
    pub(super) f: Runs<u32>,
    /// As `f`, for `g`'s nodes; empty when the overlay never walks `g`.
    pub(super) g: Runs<u32>,
    /// Output node to the `f` node it lies in, or [`ABSENT`].
    pub(super) f_part: Vec<u32>,
    /// Output node to the `g` node it lies in, or [`ABSENT`].
    pub(super) g_part: Vec<u32>,
}

impl Pieces {
    /// The pieces of `f`'s node `a`.
    #[inline]
    pub(super) fn of_f(&self, a: u32) -> &[u32] {
        self.f.run(a as usize)
    }

    /// The pieces of `g`'s node `b`.
    #[inline]
    pub(super) fn of_g(&self, b: u32) -> &[u32] {
        self.g.run(b as usize)
    }

    /// The pieces of `g`'s nodes when `of_g`, else of `f`'s.
    #[inline]
    pub(super) fn side_mut(&mut self, of_g: bool) -> &mut Runs<u32> {
        match of_g {
            true => &mut self.g,
            false => &mut self.f,
        }
    }

    /// Record output node `o` as a piece of `f_node` and `g_node`, pushing `o`
    /// onto `g`'s open run when `of_g` is true and onto `f`'s otherwise.
    #[inline]
    pub(super) fn push_piece(
        &mut self,
        lim: &crate::limits::Limits,
        o: u32,
        f_node: u32,
        g_node: u32,
        of_g: bool,
    ) -> Result<(), OperationError> {
        self.side_mut(of_g).push(lim, o)?;
        lim.try_push(&mut self.f_part, f_node)?;
        lim.try_push(&mut self.g_part, g_node)
    }

    /// Empty every list, keeping the allocations for the next level.
    pub(super) fn clear(&mut self) {
        self.f.clear();
        self.g.clear();
        self.f_part.clear();
        self.g_part.clear();
    }

    /// The four leaf pieces, indexed by [`leaf_index`].
    pub(super) fn leaves() -> [Pieces; 4] {
        [
            Pieces::leaf(false, false),
            Pieces::leaf(false, true),
            Pieces::leaf(true, false),
            Pieces::leaf(true, true),
        ]
    }

    /// The pieces of a leaf child, from how each operand's parent level names
    /// it: `⊤` alone when both use `⊤`, else the two literals, each lying in
    /// the operand's `⊤` or in the literal itself.
    fn leaf(f_one: bool, g_one: bool) -> Pieces {
        let (one, pos, neg) = (ONE_LEAF_IDX.0, POS_LEAF_IDX.0, NEG_LEAF_IDX.0);
        if f_one && g_one {
            return Pieces {
                f: Runs::from_parts(vec![one], vec![0, 1, 1, 1]),
                g: Runs::from_parts(vec![one], vec![0, 1, 1, 1]),
                f_part: vec![one, ABSENT, ABSENT],
                g_part: vec![one, ABSENT, ABSENT],
            };
        }
        // Literal pieces: a `⊤` reference splits into both, a literal
        // reference is its own piece.
        let side = |is_one: bool| -> (Runs<u32>, Vec<u32>) {
            if is_one {
                (Runs::from_parts(vec![pos, neg], vec![0, 2, 2, 2]), vec![ABSENT, one, one])
            } else {
                (Runs::from_parts(vec![pos, neg], vec![0, 0, 1, 2]), vec![ABSENT, pos, neg])
            }
        };
        let (f, f_part) = side(f_one);
        let (g, g_part) = side(g_one);
        Pieces { f, g, f_part, g_part }
    }
}

/// Whether `level` names its leaf child on `left` (else right) as `⊤`.
///
/// Determinism keeps a level to one form per leaf child, so the first
/// reference decides it. A level with no pairs names nothing and does not
/// constrain the output's form.
pub(super) fn names_one(level: &TddLevel, left: bool) -> bool {
    level
        .internal_inputs_iter()
        .flat_map(|(_, pairs)| pairs)
        .next()
        .is_none_or(|p| if left { p.left.0 } else { p.right.0 } == ONE_LEAF_IDX.0)
}

/// Which of [`Pieces::leaves`] a leaf child on `left` (else right) of the
/// level `t` has: read off the two operands' forms at `t`.
pub(super) fn leaf_index(f_level: &TddLevel, g_level: &TddLevel, left: bool) -> usize {
    usize::from(names_one(f_level, left)) * 2 + usize::from(names_one(g_level, left))
}

/// An operand's nodes at a node of the vtree the overlay runs on.
pub(super) trait Reads {
    type Pairs<'s>: ExactSizeIterator<Item = ChildPair>
    where
        Self: 's;
    /// How many nodes, indexed from zero.
    fn width(&self) -> usize;
    fn pairs_of(&self, a: usize) -> Self::Pairs<'_>;
    /// `visit(n, p)` for each pair `p` of each node `n`.
    fn for_each_pair(&self, visit: impl FnMut(u32, ChildPair));
    /// An upper bound on the pairs of every node, for a reservation.
    fn pair_bound(&self) -> usize;
}

impl Reads for &TddLevel {
    type Pairs<'s>
        = crate::diagram::PairsIter<'s>
    where
        Self: 's;
    #[inline]
    fn width(&self) -> usize {
        self.slot_count()
    }
    #[inline]
    fn pairs_of(&self, a: usize) -> Self::Pairs<'_> {
        self.pairs_iter_of_idx(a)
    }
    #[inline]
    fn for_each_pair(&self, mut visit: impl FnMut(u32, ChildPair)) {
        for (n, pairs) in self.internal_inputs_iter() {
            for p in pairs {
                visit(n as u32, p);
            }
        }
    }
    #[inline]
    fn pair_bound(&self) -> usize {
        self.live_pairs()
    }
}

/// An operand read at a node of the union where it stores no level: free
/// there, one node over the whole space whose pair names node `0` of both
/// children (a free internal child's one node, or `One` at a leaf); or
/// passing, its leaves all under the child `left` (else right), each node
/// `a` the pair of `a` there and the other child's node `0`.
#[derive(Clone, Copy, Debug)]
pub(super) struct Placed {
    pub(super) width: u32,
    /// The passing child, `None` when free.
    pub(super) pass_left: Option<bool>,
    /// For a passing leaf, bit `l` set when the operand names it by label
    /// `l`; every bit for an internal node.
    pub(super) labels: u8,
}

impl Placed {
    const FREE: Placed = Placed { width: 1, pass_left: None, labels: u8::MAX };

    fn names_one(&self, left: bool) -> bool {
        match self.pass_left {
            Some(side) if side == left => self.labels & 1 << ONE_LEAF_IDX.0 != 0,
            _ => true,
        }
    }
}

impl Reads for Placed {
    type Pairs<'s> = std::option::IntoIter<ChildPair>;
    #[inline]
    fn width(&self) -> usize {
        self.width as usize
    }
    #[inline]
    fn pairs_of(&self, a: usize) -> Self::Pairs<'_> {
        let one = EncodedChildRef::from_raw(0);
        let named = self.labels & 1 << a.min(7) != 0;
        let a = EncodedChildRef::from_raw(a as u32);
        named
            .then(|| match self.pass_left {
                None => ChildPair::new(one, one),
                Some(true) => ChildPair::new(a, one),
                Some(false) => ChildPair::new(one, a),
            })
            .into_iter()
    }
    fn for_each_pair(&self, mut visit: impl FnMut(u32, ChildPair)) {
        for n in 0..self.width() {
            self.pairs_of(n).for_each(|p| visit(n as u32, p));
        }
    }
    #[inline]
    fn pair_bound(&self) -> usize {
        self.width as usize
    }
}

/// An operand at one node of the vtree the overlay runs on.
#[derive(Clone, Copy)]
pub(super) enum Read<'a> {
    Own(&'a TddLevel),
    Placed(Placed),
}

impl Read<'_> {
    fn names_one(&self, left: bool) -> bool {
        match self {
            Read::Own(level) => names_one(level, left),
            Read::Placed(placed) => placed.names_one(left),
        }
    }
}

/// The pieces of an internal `child`, built by its own visit and taken out of
/// `built`; `None` for a leaf, whose pieces are one of [`Pieces::leaves`].
fn take_child(vtree: &Vtree, child: VtreeIdx, built: &mut [Option<Pieces>]) -> Option<Pieces> {
    if vtree.node(child).is_leaf() {
        return None;
    }
    Some(
        built[child.idx()]
            .take()
            .expect("a child level is visited before its parent"),
    )
}

/// The two operands' levels at one internal vtree node and the pieces of
/// their left and right children.
pub(super) struct LevelOverlay<'a, F, G> {
    pub(super) fl: F,
    pub(super) gl: G,
    pub(super) lp: &'a Pieces,
    pub(super) rp: &'a Pieces,
}

/// The overlay of two structural, nonzero diagrams on one internal-rooted
/// vtree, before its reduction. Weights are the caller's.
fn overlay_levels(eng: &Engine, op: Overlay, f: &Tdd, g: &Tdd) -> Result<Tdd, OperationError> {
    overlay_reading(eng, op, &f.vtree, (f.output.local, g.output.local), |t| {
        (Read::Own(&f.levels[t.idx()]), Read::Own(&g.levels[t.idx()]))
    })
}

/// One level of the overlay into `out`: the root's one node when `outs`
/// names the operands' outputs, `None` when it has no cell; else the level's
/// pieces into `pieces`.
fn visit<F: Reads, G: Reads>(
    step: &LevelOverlay<'_, F, G>,
    eng: &Engine,
    op: Overlay,
    outs: Option<(NodeIdx, NodeIdx)>,
    out: &mut TddLevel,
    s: &mut Scratch,
    pieces: &mut Pieces,
) -> Result<Option<NodeIdx>, OperationError> {
    match outs {
        Some((f_out, g_out)) => {
            let pairs = step.root_cells(eng, op, f_out, g_out, s)?;
            match pairs.is_empty() {
                true => Ok(None),
                false => out.push_node(eng.limits(), pairs).map(Some),
            }
        }
        None => step.inner_level(eng, op, out, s, pieces).map(|()| None),
    }
}

/// The overlay on `vtree`, internal-rooted, of operands with outputs `outs`,
/// read at each internal node by `read`, before its reduction.
fn overlay_reading<'a>(
    eng: &Engine,
    op: Overlay,
    vtree: &Arc<Vtree>,
    outs: (NodeIdx, NodeIdx),
    read: impl Fn(VtreeIdx) -> (Read<'a>, Read<'a>),
) -> Result<Tdd, OperationError> {
    let vtree = Arc::clone(vtree);
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
        let (fr, gr) = read(t);
        let leaf_index = |left: bool| usize::from(fr.names_one(left)) * 2 + usize::from(gr.names_one(left));
        let lo = take_child(&vtree, left, &mut built);
        let ro = take_child(&vtree, right, &mut built);
        let lp = lo.as_ref().unwrap_or(&leaves[leaf_index(true)]);
        let rp = ro.as_ref().unwrap_or(&leaves[leaf_index(false)]);
        let levels = assembly.levels_mut();
        let out = &mut levels[t.idx()];
        let at_root = (t == root).then_some(outs);
        let mut pieces = spare.pop().unwrap_or_default();
        pieces.clear();
        let s = &mut scratch;
        let p = &mut pieces;
        let local = match (fr, gr) {
            (Read::Own(fl), Read::Own(gl)) => visit(&LevelOverlay { fl, gl, lp, rp }, eng, op, at_root, out, s, p),
            (Read::Own(fl), Read::Placed(gl)) => visit(&LevelOverlay { fl, gl, lp, rp }, eng, op, at_root, out, s, p),
            (Read::Placed(fl), Read::Own(gl)) => visit(&LevelOverlay { fl, gl, lp, rp }, eng, op, at_root, out, s, p),
            (Read::Placed(fl), Read::Placed(gl)) => visit(&LevelOverlay { fl, gl, lp, rp }, eng, op, at_root, out, s, p),
        }?;
        if t == root {
            let Some(local) = local else {
                drop(assembly);
                return crate::build::constant_on(eng, &vtree, false);
            };
            eng.limits().level_done(out_nodes + 1)?;
            return assembly.finish(TddNodeId { vtree: root, local });
        }
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
pub(super) struct Scratch {
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
fn index_owners(
    eng: &Engine,
    level: &impl Reads,
    owner: &mut FxHashMap<u64, u32>,
) -> Result<(), OperationError> {
    owner.clear();
    eng.limits().reserve_map(owner, level.pair_bound())?;
    level.for_each_pair(|n, p| {
        owner.insert(key(p.left.0, p.right.0), n);
    });
    Ok(())
}

/// Fill `held` with the pairs of `nodes` at `level`.
fn index_pairs(
    eng: &Engine,
    level: &impl Reads,
    nodes: impl Iterator<Item = usize>,
    held: &mut FxHashSet<u64>,
) -> Result<(), OperationError> {
    held.clear();
    for n in nodes {
        let pairs = level.pairs_of(n);
        eng.limits().reserve_set(held, pairs.len())?;
        for p in pairs {
            held.insert(key(p.left.0, p.right.0));
        }
    }
    Ok(())
}

impl<F: Reads, G: Reads> LevelOverlay<'_, F, G> {
    /// Build one internal, non-root level of the overlay into `out` and its
    /// pieces into the empty `pieces`.
    ///
    /// Every node of `f` is split by `g`: its cells are grouped by the node of
    /// `g` they lie in, each group a product node and the ungrouped rest the
    /// residue. A disjunction also walks `g` and keeps the cells no node of `f`
    /// holds, as `g`'s residues; the cells both hold are the products already
    /// built.
    pub(super) fn inner_level(
        &self,
        eng: &Engine,
        op: Overlay,
        out: &mut TddLevel,
        s: &mut Scratch,
        pieces: &mut Pieces,
    ) -> Result<(), OperationError> {
        let LevelOverlay { fl, gl, lp, rp } = self;
        let lim = eng.limits();
        let mut poll = lim.gate();
        let walk_g = op != Overlay::AndNot;
        let f_width = fl.width();
        let g_width = gl.width();
        index_owners(eng, gl, &mut s.owner)?;
        pieces.f.begin_exact(lim, f_width)?;
        s.products.clear();

        for a in 0..f_width {
            s.tagged.clear();
            for p in fl.pairs_of(a) {
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
                    lim.try_push(
                        &mut s.pairs,
                        ChildPair::new(EncodedChildRef::from_raw(x), EncodedChildRef::from_raw(y)),
                    )?;
                    i += 1;
                }
                let o = out.push_node(lim, &s.pairs)?.0;
                pieces.push_piece(lim, o, a as u32, b, false)?;
                if walk_g && b != ABSENT {
                    lim.try_push(&mut s.products, (b, o))?;
                }
            }
            pieces.f.close(lim)?;
        }

        if walk_g {
            index_pairs(eng, fl, 0..f_width, &mut s.held)?;
            // Stable by `g` node, so each node's products stay in output order.
            s.products.sort_by_key(|&(b, _)| b);
            pieces.g.begin_exact(lim, g_width)?;
            let mut next = 0;
            for b in 0..g_width {
                while next < s.products.len() && s.products[next].0 == b as u32 {
                    pieces.g.push(lim, s.products[next].1)?;
                    next += 1;
                }
                s.pairs.clear();
                for p in gl.pairs_of(b) {
                    let xs = lp.of_g(p.left.0);
                    let ys = rp.of_g(p.right.0);
                    for &x in xs {
                        let fx = lp.f_part[x as usize];
                        for &y in ys {
                            let fy = rp.f_part[y as usize];
                            if fx != ABSENT && fy != ABSENT && s.held.contains(&key(fx, fy)) {
                                continue;
                            }
                            lim.try_push(
                                &mut s.pairs,
                                ChildPair::new(
                                    EncodedChildRef::from_raw(x),
                                    EncodedChildRef::from_raw(y),
                                ),
                            )?;
                        }
                    }
                    poll.poll((xs.len() * ys.len()) as u64 + 1)?;
                }
                if !s.pairs.is_empty() {
                    crate::diagram::sort_pairs(&mut s.pairs);
                    let o = out.push_node(lim, &s.pairs)?.0;
                    pieces.push_piece(lim, o, ABSENT, b as u32, true)?;
                }
                pieces.g.close(lim)?;
            }
        }
        poll.flush()?;
        Ok(())
    }

    /// The root's one node: the cells of `f_out`, and for a disjunction the cells
    /// of `g_out` outside `f_out`; for a difference, the cells of `f_out` outside
    /// `g_out`. A sum is the disjunction, refusing a cell of `g_out` inside
    /// `f_out`. Returned sorted, in `s.pairs`, and empty when the result is false.
    ///
    /// # Errors
    ///
    /// [`OperationError::SharedModel`] for a sum whose operands share a cell.
    /// Allocation refusals and installed limits propagate.
    fn root_cells<'s>(
        &self,
        eng: &Engine,
        op: Overlay,
        f_out: NodeIdx,
        g_out: NodeIdx,
        s: &'s mut Scratch,
    ) -> Result<&'s [ChildPair], OperationError> {
        let LevelOverlay { fl, gl, lp, rp } = self;
        let lim = eng.limits();
        let mut poll = lim.gate();
        s.pairs.clear();
        // The operand whose pairs a cell is tested against.
        match op {
            Overlay::Or | Overlay::Add => index_pairs(eng, fl, std::iter::once(f_out.idx()), &mut s.held)?,
            Overlay::AndNot => index_pairs(eng, gl, std::iter::once(g_out.idx()), &mut s.held)?,
        }
        if op != Overlay::AndNot {
            for p in fl.pairs_of(f_out.idx()) {
                let xs = lp.of_f(p.left.0);
                let ys = rp.of_f(p.right.0);
                for &x in xs {
                    for &y in ys {
                        lim.try_push(
                            &mut s.pairs,
                            ChildPair::new(
                                EncodedChildRef::from_raw(x),
                                EncodedChildRef::from_raw(y),
                            ),
                        )?;
                    }
                }
                poll.poll((xs.len() * ys.len()) as u64 + 1)?;
            }
        }
        // The walked operand: `g` for a disjunction, `f` for a difference; a cell
        // is kept when its components in the other operand are not a pair of that
        // operand's output.
        match op {
            Overlay::Or | Overlay::Add => walk_cells(gl, g_out, (lp, rp), true, op, &s.held, lim, &mut poll, &mut s.pairs)?,
            Overlay::AndNot => walk_cells(fl, f_out, (lp, rp), false, op, &s.held, lim, &mut poll, &mut s.pairs)?,
        }
        poll.flush()?;
        crate::diagram::sort_pairs(&mut s.pairs);
        Ok(&s.pairs)
    }
}

/// The cells of the walked operand's `out`, its nodes' pieces those of `g`
/// when `of_g`, onto `pairs`: each cell whose components in the other
/// operand are not a pair of `held`.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn walk_cells(
    walked: &impl Reads,
    out: NodeIdx,
    (lp, rp): (&Pieces, &Pieces),
    of_g: bool,
    op: Overlay,
    held: &FxHashSet<u64>,
    lim: &Limits,
    poll: &mut crate::limits::PollGate<'_>,
    pairs: &mut Vec<ChildPair>,
) -> Result<(), OperationError> {
    let (x_part, y_part) = if of_g { (&lp.f_part, &rp.f_part) } else { (&lp.g_part, &rp.g_part) };
    for p in walked.pairs_of(out.idx()) {
        let (xs, ys) = if of_g { (lp.of_g(p.left.0), rp.of_g(p.right.0)) } else { (lp.of_f(p.left.0), rp.of_f(p.right.0)) };
        for &x in xs {
            let cx = x_part[x as usize];
            for &y in ys {
                let cy = y_part[y as usize];
                if cx != ABSENT && cy != ABSENT && held.contains(&key(cx, cy)) {
                    // A cell of both operands: a model `f + g` counts twice.
                    if op == Overlay::Add {
                        return Err(OperationError::SharedModel);
                    }
                    continue;
                }
                lim.try_push(pairs, ChildPair::new(EncodedChildRef::from_raw(x), EncodedChildRef::from_raw(y)))?;
            }
        }
        poll.poll((xs.len() * ys.len()) as u64 + 1)?;
    }
    Ok(())
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

/// The overlay on a one-variable vtree, whose root is a leaf; a sum whose
/// operands share a literal is [`OperationError::SharedModel`].
fn overlay_leaf_root(eng: &Engine, op: Overlay, f: &Tdd, g: &Tdd) -> Result<Tdd, OperationError> {
    let (fp, fn_) = leaf_truth(f.output.local);
    let (gp, gn) = leaf_truth(g.output.local);
    let (p, n) = match op {
        Overlay::Or => (fp || gp, fn_ || gn),
        Overlay::AndNot => (fp && !gp, fn_ && !gn),
        Overlay::Add if (fp && gp) || (fn_ && gn) => return Err(OperationError::SharedModel),
        Overlay::Add => (fp || gp, fn_ || gn),
    };
    let local = match (p, n) {
        (true, true) => ONE_LEAF_IDX,
        (true, false) => POS_LEAF_IDX,
        (false, true) => NEG_LEAF_IDX,
        (false, false) => return crate::build::constant_on(eng, &f.vtree, false),
    };
    let levels = crate::diagram::try_take_levels(eng, f.vtree.num_nodes())?;
    Assembly::from_levels(eng, Arc::clone(&f.vtree), levels).finish(TddNodeId {
        vtree: f.vtree.root(),
        local,
    })
}

/// `f ∨ g`, `f ∧ ¬g` or the sum `f + g` (a disjunction refusing a shared
/// model) of two owned operands on one vtree, closed and not reduced: the
/// overlay's pieces are disjoint and each node is created with its cells, so
/// no node is empty, and only a constant needs the close.
///
/// Validates both operands, aligns their weights, and returns the result with
/// them. A false operand short-cuts: the other one, or false.
pub(crate) fn overlay_on(
    eng: &Engine,
    op: Overlay,
    f: Tdd,
    g: Tdd,
) -> Result<Tdd, OperationError> {
    crate::apply::extend::refuse_other_homes([&f, &g])?;
    f.require_structure()?;
    g.require_structure()?;
    if crate::apply::extend::needed([&f, &g]) && readable(&f) && readable(&g) {
        return across(eng, op, f, g);
    }
    let [f, g] = crate::apply::extend::on_one(eng, [Cow::Owned(f), Cow::Owned(g)])?.map(Cow::into_owned);
    eng.limits().check_stop()?;
    let mut result = match (f.is_zero(), g.is_zero(), op) {
        (true, _, Overlay::AndNot) => crate::build::constant_on(eng, &f.vtree, false)?,
        (true, _, Overlay::Or | Overlay::Add) => g,
        (false, true, _) => f,
        (false, false, _) if f.vtree.node(f.vtree.root()).is_leaf() => {
            overlay_leaf_root(eng, op, &f, &g)?
        }
        (false, false, _) => overlay_levels(eng, op, &f, &g)?,
    };
    eng.close(&mut result, &[])?;
    crate::apply::extend::trimmed(eng, result)
}

/// Whether the overlay reads `tdd` where it is when the other operand's
/// leaf set differs: not false, a node, no region or weights.
fn readable(tdd: &Tdd) -> bool {
    !tdd.is_zero() && tdd.vtree.num_nodes() != 0 && !tdd.vtree.has_regions() && tdd.weight_table().is_none()
}

/// `n` copies of `value`, charged.
fn table<T: Clone>(lim: &Limits, n: usize, value: T) -> Result<Vec<T>, OperationError> {
    let mut table = Vec::new();
    lim.try_resize(&mut table, n, value)?;
    Ok(table)
}

/// What [`Vtree::leaf_sets_reading`] gives both operands' readings to.
struct Readings<'l>(&'l Limits, [Vec<Reading>; 2]);

impl UnionReader for Readings<'_> {
    fn begin(&mut self, k: usize) -> Result<(), OperationError> {
        self.1 = [table(self.0, 2 * k - 1, Reading::Free)?, table(self.0, 2 * k - 1, Reading::Free)?];
        Ok(())
    }

    fn read(&mut self, u: VtreeIdx, [rf, rg]: [Reading; 2]) {
        self.1[0][u.idx()] = rf;
        self.1[1][u.idx()] = rg;
    }
}

/// The readings on `into` of `inner`, whose leaves are among `into`'s, from
/// `leaves` ([`Vtree::nested_leaves`]).
fn nested_readings(lim: &Limits, into: &Vtree, inner: &Vtree, leaves: &[(VtreeIdx, VtreeIdx)]) -> Result<Vec<Reading>, OperationError> {
    let mut readings = table(lim, into.num_nodes(), Reading::Free)?;
    for &(s, u) in leaves {
        readings[u.idx()] = Reading::Own(s);
    }
    let mut carried = table(lim, into.num_nodes(), VtreeIdx(0))?;
    for &(s, u) in leaves {
        carried[u.idx()] = s;
    }
    for (t, l, r) in into.internal_bottomup() {
        readings[t.idx()] = match (readings[l.idx()], readings[r.idx()]) {
            (Reading::Free, Reading::Free) => continue,
            (_, Reading::Free) => {
                carried[t.idx()] = carried[l.idx()];
                Reading::Pass(l)
            }
            (Reading::Free, _) => {
                carried[t.idx()] = carried[r.idx()];
                Reading::Pass(r)
            }
            _ => {
                let s = inner.node(carried[l.idx()]).parent().expect("two carried nodes have a parent");
                carried[t.idx()] = s;
                Reading::Own(s)
            }
        };
    }
    Ok(readings)
}

/// The labels by which `level` names its leaf child on `left` (else right),
/// as bits.
fn leaf_labels(level: &TddLevel, left: bool) -> u8 {
    level.leaf_label_bits().0[usize::from(!left)]
}

/// How the overlay reads `f` at each internal node of `into` from its
/// `readings` there.
fn reads_of<'a>(lim: &Limits, f: &'a Tdd, into: &Vtree, readings: &[Reading]) -> Result<Vec<Read<'a>>, OperationError> {
    let mut reads = table(lim, into.num_nodes(), Read::Placed(Placed::FREE))?;
    let mut carried = table(lim, into.num_nodes(), VtreeIdx(0))?;
    for (u, reading) in readings.iter().enumerate().filter(|&(u, _)| into.node(VtreeIdx(u as u32)).is_leaf()) {
        if let Reading::Own(s) = *reading {
            carried[u] = s;
        }
    }
    for (t, l, _) in into.internal_bottomup() {
        reads[t.idx()] = match readings[t.idx()] {
            Reading::Own(s) => {
                carried[t.idx()] = s;
                Read::Own(&f.levels[s.idx()])
            }
            Reading::Pass(c) => {
                let s = carried[c.idx()];
                carried[t.idx()] = s;
                let (width, labels) = match f.vtree.node(s).parent() {
                    _ if !f.vtree.node(s).is_leaf() => (f.levels[s.idx()].slot_count() as u32, u8::MAX),
                    Some(p) => (3, leaf_labels(&f.levels[p.idx()], f.vtree.children(p).0 == s)),
                    None => (3, 1 << f.output.local.0),
                };
                Read::Placed(Placed { width, pass_left: Some(c == l), labels })
            }
            Reading::Free => continue,
        };
    }
    Ok(reads)
}

/// The overlay of operands on different leaf sets, neither false and both
/// with a node, each read where it is on the projection on their union.
#[cold]
fn across(eng: &Engine, op: Overlay, f: Tdd, g: Tdd) -> Result<Tdd, OperationError> {
    let lim = eng.limits();
    let mut readings = Readings(lim, [Vec::new(), Vec::new()]);
    let mut leaves = Vec::new();
    let into = match Vtree::leaf_sets_reading(&f.vtree, &g.vtree, Some(&mut readings))? {
        LeafSets::AInB if g.vtree.nested_leaves(&f.vtree, &mut leaves) => {
            readings.1[0] = nested_readings(lim, &g.vtree, &f.vtree, &leaves)?;
            Arc::clone(&g.vtree)
        }
        LeafSets::BInA if f.vtree.nested_leaves(&g.vtree, &mut leaves) => {
            readings.1[1] = nested_readings(lim, &f.vtree, &g.vtree, &leaves)?;
            Arc::clone(&f.vtree)
        }
        LeafSets::Union(into) => into,
        _ => {
            let [f, g] = crate::apply::extend::on_one(eng, [Cow::Owned(f), Cow::Owned(g)])?.map(Cow::into_owned);
            return overlay_on(eng, op, f, g);
        }
    };
    lim.check_stop()?;
    let own = |h: &Tdd| Arc::ptr_eq(&h.vtree, &into);
    let [rf, rg] = &readings.1;
    let fr = if own(&f) { Vec::new() } else { reads_of(lim, &f, &into, rf)? };
    let gr = if own(&g) { Vec::new() } else { reads_of(lim, &g, &into, rg)? };
    fn at<'a>(h: &'a Tdd, reads: &[Read<'a>], t: VtreeIdx) -> Read<'a> {
        match reads.get(t.idx()) {
            Some(&r) => r,
            None => Read::Own(&h.levels[t.idx()]),
        }
    }
    let mut result = overlay_reading(eng, op, &into, (f.output.local, g.output.local), |t| {
        (at(&f, &fr, t), at(&g, &gr, t))
    })?;
    eng.close(&mut result, &[])?;
    crate::apply::extend::trimmed(eng, result)
}

/// Return `f ∧ ¬g`, the models of `f` that are not models of `g`, for two
/// structural diagrams on one vtree or on projections of one home.
///
/// Uses the shared vtree's execution context automatically. Unlike
/// `f & !g`, it never builds the complement of `g`: each node of `f` is split
/// by the nodes of `g` at its level and the pieces outside `g` are kept.
///
/// Both operands are consumed on success and on error. The result is not
/// minimized: it may hold twins and nodes its output does not reach.
/// Weight compatibility and inheritance follow [`and`](crate::and).
///
/// The result is not certified
/// ([`Tdd::is_canonical`](crate::Tdd::is_canonical));
/// [`Engine::minimize`](crate::Engine::minimize) certifies it.
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
/// [`OperationError::VtreeMismatch`] for operands neither on one vtree nor on
/// projections of one home ([`Vtree::projection`](crate::Vtree::projection)),
/// or [`OperationError::MarginalLevel`] if either operand has discarded structure,
/// even when the other operand is false; [`OperationError::IndexOverflow`] when
/// a level's pieces outgrow a 32-bit index. Allocation refusals and installed
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
