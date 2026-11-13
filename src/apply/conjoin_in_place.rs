//! Conjunction written into the left operand's own levels.
//!
//! `f ∧ g` where `g` is local — a cube, a set of codes over one block of
//! variables, a conjunction of such constraints on disjoint subtrees — is `f`
//! with some pairs dropped and a few nodes rebuilt: the levels where `g` is
//! true over their whole subtree keep `f`'s nodes as they are, and those are
//! most of a large diagram. The conjunction is written there, in place.
//!
//! Read from `g`'s output down, `g`'s nodes split the vtree three ways:
//!
//! - where `g`'s node is the true node over the level's variables, `f`'s
//!   level is left as it is;
//! - where `g`'s node has one pair (its spine), every node of `f` at the level
//!   meets that one node, so each keeps its index and its pairs are rewritten
//!   through the children's results: a pair whose child came out false is
//!   dropped, and a node left with none is false in turn;
//! - where `g` branches (a node of two pairs or more, or a literal leaf), the
//!   products of `f`'s nodes there with `g`'s node are found from the levels
//!   under it up, each appended to `f`'s level, or `a` itself where the
//!   product leaves `f`'s node `a` as it was, and a literal leaf's labels are
//!   mapped through the literal.
//!
//! Every node of `f` at a level of `g`'s spine meets `g`'s one node there
//! because each of `f`'s nodes is named from its parent level, all of whose
//! nodes meet the parent's one node of `g`; so the rewrite in place is the
//! product. Below a branching level the products of one node of `f` with two
//! nodes of `g` are disjoint, as `g`'s nodes on a level are, so they may stand
//! beside each other. A node of `f` that no product names is unreachable once
//! the rewrite is done, and the prune at the end removes it, or, where the
//! caller asks for none ([`Engine::and_in_place_loose`]), it stays, and the
//! levels that may hold one are listed loose for the next prune. A spine
//! node left with no pair is dropped in the pass either way, its level
//! renumbered as its parent level is rewritten. Without the prune, the
//! levels from a branching level down are still pruned of what its products
//! do not reach ([`prune_under`]): a node a product replaced overlaps it,
//! where every other node left unreached is disjoint from those reached, as
//! the nodes of one level are.
//!
//! The work is the pairs of `g`'s spine levels in `f`, the products below the
//! levels where `g` branches, and the prune: never the levels where `g` is
//! true, unless a dropped pair leaves their nodes to the prune.
//!
//! The products below a branching level are found in one pass over its
//! levels, children before parents ([`block_products`]), that leaves each
//! node of `f` there with its products with every node of `g` it meets: a
//! list of `(g's node, product)`, short, since `g`'s nodes at a level are
//! disjoint. A pair `(l, r)` of `f` meets `g`'s pair `(bl, br)` exactly
//! where `l` meets `bl` and `r` meets `br`, and the product is the pair of
//! those products; so each pair of `f` is read once, its sides' lists are
//! crossed, and each combination is looked up among `g`'s pairs at the
//! level, a table keyed by the pair. The hits are grouped by `g`'s node: a
//! node of `f` whose pairs all came back whole in one node of `g` lies
//! inside it and is its own product there, no node added (a filter on a
//! column whose nodes hold one code each rebuilds none); any other group is
//! a node of the product pairs, appended to `f`'s level. A level where `g`
//! has only its true node is not read: its nodes are their own products
//! there. The work is the pairs of `f` under the branching level, a unit
//! each and one per probe of `g`'s table, with nothing memoized per pair of
//! nodes; the products no node at the branching level reaches are made too,
//! and the prune removes them.

use std::sync::Arc;

use rustc_hash::FxHashMap;

use crate::Engine;
use crate::diagram::{ChildPair, NodeIdx, Tdd};
use crate::diagram::{NEG_LEAF_IDX, ONE_LEAF_IDX, POS_LEAF_IDX};
use crate::limits::{Limits, OperationError};
use crate::reduce::ReductionPlan;
use crate::vtree::{Vtree, VtreeIdx};

use super::falsity::rewrite_level_pairs;
use super::runs::Runs;

/// A product that is false: no node, and the pair that names it dropped.
const DEAD: u32 = u32::MAX;

/// `n` values made by `make`, reserved through `lim` as the operation's
/// other buffers are: charged, and a refusal an error rather than an abort.
fn charged<T>(lim: &Limits, n: usize, make: impl FnMut() -> T) -> Result<Vec<T>, OperationError> {
    let mut v = Vec::new();
    lim.reserve_exact(&mut v, n)?;
    v.resize_with(n, make);
    Ok(v)
}

/// The true node of `g` at each level, `None` where `g` has none: a leaf's
/// constant label, and an internal level's node of one pair whose sides are
/// the children's true nodes.
pub(super) fn true_nodes(lim: &Limits, g: &Tdd) -> Result<Vec<Option<u32>>, OperationError> {
    let vtree = g.vtree();
    let mut top = charged(lim, vtree.num_nodes(), || None)?;
    for t in vtree.bottomup() {
        if vtree.node(t).is_leaf() {
            top[t.idx()] = Some(ONE_LEAF_IDX.0);
            continue;
        }
        let (l, r) = vtree.children(t);
        let (Some(tl), Some(tr)) = (top[l.idx()], top[r.idx()]) else {
            continue;
        };
        let level = g.level(t);
        top[t.idx()] = (0..level.node_count()).find_map(|i| {
            let mut pairs = level.pairs_iter_of_idx(i);
            let first = pairs.next()?;
            (pairs.next().is_none() && first.left.raw() == tl && first.right.raw() == tr)
                .then_some(i as u32)
        });
    }
    Ok(top)
}

/// How `g` reads a level, found from its output down.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Role {
    /// `g` is true over the level's variables, or the walk never got there.
    True,
    /// `g` has one node of one pair here, this one.
    Spine(u32),
    /// `g` branches here at this node, or this leaf is one of its literals.
    Branch(u32),
}

/// Where each node of `f` at a level went: what a parent's pair names in
/// its place.
enum Moved {
    /// Every node kept its index.
    Same,
    /// The products of a branching level, by the original node, or a leaf's
    /// three labels.
    To(Vec<u32>),
}

impl Moved {
    fn at(&self, x: u32) -> u32 {
        match self {
            Moved::Same => x,
            Moved::To(to) => to[x as usize],
        }
    }
}

/// The product of a leaf label of `f` with a literal label of `g`.
fn leaf_product(a: u32, b: u32) -> u32 {
    if a == ONE_LEAF_IDX.0 || a == b {
        b
    } else {
        DEAD
    }
}

/// How [`block_products`] reads the nodes of `f` at one level under a
/// branching level: what each one's products with `g`'s nodes there are.
enum Split {
    /// `g`'s only node at the level is its true node: every node of `f`
    /// lies in it whole, and its product there is itself.
    Whole(u32),
    /// A leaf level: a label's products are labels ([`label_pieces`]).
    Leaf,
    /// Per original node `x` of `f`, its products with the nodes of `g` it
    /// meets, `(g's node, product)`, as run `x`: none where it meets no
    /// node of `g`.
    Listed(Runs<(u32, u32)>),
}

/// The products of `f`'s leaf label `x` with each of `g`'s labels at a leaf
/// level, `(g's label, product)`: with `⊤` the label itself, and with a
/// literal the literal, where `x` is `⊤` or that literal.
#[inline]
fn label_pieces(x: u32, buf: &mut [(u32, u32); 3]) -> usize {
    buf[0] = (ONE_LEAF_IDX.0, x);
    match x == ONE_LEAF_IDX.0 {
        true => {
            buf[1] = (POS_LEAF_IDX.0, POS_LEAF_IDX.0);
            buf[2] = (NEG_LEAF_IDX.0, NEG_LEAF_IDX.0);
            3
        }
        false => {
            buf[1] = (x, x);
            2
        }
    }
}

/// The products of `f`'s node (or label) `x` at a level split as `split`
/// with the nodes of `g` it meets there.
#[inline]
fn pieces_of<'s>(split: &'s Split, x: u32, buf: &'s mut [(u32, u32); 3]) -> &'s [(u32, u32)] {
    match split {
        Split::Whole(top) => {
            buf[0] = (*top, x);
            &buf[..1]
        }
        Split::Leaf => {
            let n = label_pieces(x, buf);
            &buf[..n]
        }
        Split::Listed(pieces) => pieces.run(x as usize),
    }
}

/// One level's split as [`block_products`] reads it: each node's products
/// with `g`'s nodes, and the nodes those products add to the level.
struct Settle {
    /// The level's nodes before any product is added: a product made here
    /// is `width` plus its rank among those made.
    width: u32,
    /// The node being read's pairs' products, `(g's node, the pair's
    /// position, the product pair, whether it is the pair itself)`.
    hits: Vec<(u32, u32, ChildPair, bool)>,
    /// Per node of the level, its products, as [`Split::Listed`] holds them.
    pieces: Runs<(u32, u32)>,
    /// The products made, their pairs as run `k` for the `k`-th.
    made: Runs<ChildPair>,
}

impl Settle {
    /// Node `a`'s products, from its `n` pairs' hits: per node of `g` its
    /// pairs hit, `a` itself where each of its pairs met that node in one
    /// pair and came back whole, else a node of the products' pairs; the
    /// nodes before `a` that no pair named (none in a stored level) meet
    /// nothing. A node of more pairs than the hits' 32-bit positions number
    /// ([`Limits::list_index`]) is refused with
    /// [`OperationError::IndexOverflow`]: its positions would have wrapped.
    fn node(&mut self, lim: &Limits, a: usize, n: usize) -> Result<(), OperationError> {
        lim.list_index(n)?;
        self.pieces.close_until(lim, a)?;
        // Within a node of `g` the pairs stay in their order; unstable, so
        // that the sort asks for no buffer.
        self.hits.sort_unstable_by_key(|h| (h.0, h.1));
        let mut i = 0;
        while i < self.hits.len() {
            let b = self.hits[i].0;
            let k = i + self.hits[i..].iter().take_while(|h| h.0 == b).count();
            let group = &self.hits[i..k];
            let whole = group.len() == n
                && group
                    .iter()
                    .enumerate()
                    .all(|(q, h)| h.1 == q as u32 && h.3);
            let product = match whole {
                true => a as u32,
                false => {
                    lim.reserve_exact(&mut self.made.items, group.len())?;
                    self.made.items.extend(group.iter().map(|h| h.2));
                    crate::diagram::sort_pairs(self.made.open_run_mut());
                    self.made.close(lim)?;
                    // The node the product is pushed as, past the level's.
                    let node = self.width as usize + self.made.runs() - 1;
                    if node >= lim.level_width_cap() {
                        return Err(OperationError::IndexOverflow);
                    }
                    node as u32
                }
            };
            self.pieces.push(lim, (b, product))?;
            i = k;
        }
        self.hits.clear();
        self.pieces.close(lim)
    }
}

/// The products of every node of `f` at branching level `t` with `g`'s
/// node `b` there, as `to` maps them, found bottom-up: one pass over
/// the levels under `t`, children before parents, reads each pair of `f`
/// once against its children's products and finds the pairs of `g` they
/// meet in a table of `g`'s pairs at the level, so that each node of `f`
/// comes out with its products with every node of `g` it meets (`g`'s nodes
/// at a level are disjoint). A node that each of its pairs carries whole
/// into one node of `g` is its own product there, and no node is added for
/// it; a level where `g` has only its true node is not read, its nodes
/// their own products. The work is a unit per pair of `f` under `t` and
/// one per probe of `g`'s table, a probe per pair of its sides' products.
/// Each product made is
/// appended to `f`'s level, and `grown` marks the level.
fn block_products(
    lim: &Limits,
    f: &mut Tdd,
    g: &Tdd,
    top: &[Option<u32>],
    t: VtreeIdx,
    b: u32,
    grown: &mut [bool],
) -> Result<Vec<u32>, OperationError> {
    let vtree = g.vtree();
    let whole = |u: VtreeIdx| top[u.idx()].filter(|_| g.level(u).node_count() == 1);
    let order = internal_subtree_order(lim, vtree, t, |u| whole(u).is_some())?;
    let mut split: Vec<Option<Split>> = charged(lim, vtree.num_nodes(), || None)?;
    let mut gate = lim.gate();
    let mut hits = Vec::new();
    let mut to = Vec::new();
    for &u in order.iter().rev() {
        if u != t
            && let Some(top) = whole(u)
        {
            split[u.idx()] = Some(Split::Whole(top));
            continue;
        }
        let (l, r) = vtree.children(u);
        let side = |split: &mut [Option<Split>], c: VtreeIdx| match vtree.node(c).is_leaf() {
            true => Split::Leaf,
            false => split[c.idx()]
                .take()
                .expect("a level's children are split before it"),
        };
        let (sl, sr) = (side(&mut split, l), side(&mut split, r));
        // `g`'s pairs at `u` by their sides; at `t`, `b`'s alone.
        let gl = g.level(u);
        let mut table: FxHashMap<u64, u32> = FxHashMap::default();
        lim.reserve_map(&mut table, gl.live_pairs())?;
        for i in 0..gl.node_count() {
            if u == t && i as u32 != b {
                continue;
            }
            for p in gl.pairs_iter_of_idx(i) {
                table.insert(p.key(), i as u32);
            }
        }
        let level = &f.levels[u.idx()];
        let width = level.node_count();
        let mut st = Settle {
            width: width as u32,
            hits: std::mem::take(&mut hits),
            pieces: Runs::default(),
            made: Runs::default(),
        };
        st.pieces.begin_exact(lim, width)?;
        st.made.begin(lim)?;
        let mut refused = None;
        let mut cur: Option<usize> = None;
        let mut read = 0usize;
        let mut work = 0u64;
        let (mut lb, mut rb) = ([(0u32, 0u32); 3], [(0u32, 0u32); 3]);
        level.for_each_node_pair(|x, p| {
            if refused.is_some() {
                return;
            }
            if cur != Some(x) {
                if let Some(a) = cur
                    && let Err(e) = st.node(lim, a, read)
                {
                    refused = Some(e);
                    return;
                }
                cur = Some(x);
                read = 0;
            }
            let (pl, pr) = (p.left.raw(), p.right.raw());
            let (lefts, rights) = (pieces_of(&sl, pl, &mut lb), pieces_of(&sr, pr, &mut rb));
            for &(bl, cl) in lefts {
                for &(br, cr) in rights {
                    if let Some(&bn) = table.get(&((u64::from(bl) << 32) | u64::from(br)))
                        && let Err(e) = lim.try_push(
                            &mut st.hits,
                            (
                                bn,
                                read as u32,
                                ChildPair::new(NodeIdx(cl), NodeIdx(cr)),
                                cl == pl && cr == pr,
                            ),
                        )
                    {
                        refused = Some(e);
                        return;
                    }
                }
            }
            read += 1;
            // The pair read, and a probe of `g`'s table per pair of its
            // sides' products.
            work += 1 + (lefts.len() * rights.len()) as u64;
        });
        if let Some(e) = refused {
            return Err(e);
        }
        if let Some(a) = cur {
            st.node(lim, a, read)?;
        }
        st.pieces.close_until(lim, width)?;
        gate.poll(work)?;
        drop((sl, sr));
        let made = st.made.runs();
        for k in 0..made {
            let idx = f.levels[u.idx()].push_node(lim, st.made.run(k))?;
            debug_assert_eq!(
                idx.0 as usize,
                width + k,
                "a product is numbered as its rank among those made"
            );
        }
        grown[u.idx()] |= made > 0;
        hits = std::mem::take(&mut st.hits);
        match u == t {
            true => {
                lim.reserve_exact(&mut to, width)?;
                to.extend((0..width).map(|x| st.pieces.run(x).first().map_or(DEAD, |p| p.1)));
            }
            false => split[u.idx()] = Some(Split::Listed(st.pieces)),
        }
    }
    gate.flush()?;
    Ok(to)
}

/// The nodes of branching level `t` and of the levels under it that no node
/// The internal levels under `t`, parents first, stopping descent beneath any
/// node `u != t` where `stop(u)` holds.
fn internal_subtree_order(
    lim: &Limits,
    vtree: &Vtree,
    t: VtreeIdx,
    stop: impl Fn(VtreeIdx) -> bool,
) -> Result<Vec<VtreeIdx>, OperationError> {
    let mut order = Vec::new();
    lim.try_push(&mut order, t)?;
    let mut at = 0;
    while at < order.len() {
        let u = order[at];
        at += 1;
        if u != t && stop(u) {
            continue;
        }
        let (l, r) = vtree.children(u);
        for c in [l, r] {
            if !vtree.node(c).is_leaf() {
                lim.try_push(&mut order, c)?;
            }
        }
    }
    Ok(order)
}

/// Prune `f` below a branching level `t`: every node under `t` not reached
/// from `to`'s live products, and on `t` itself every node neither `to` nor a
/// node `to` names reaches, removed, and `to` renumbered through what is left.
/// `t`'s products stand beside the nodes they replace, which overlap them,
/// and so do the products under them; the prune would remove those, and
/// [`Engine::and_in_place_loose`] runs none. The work is the levels under
/// `t`. A level held as the description of its pairs is stored first, so
/// that its nodes are dropped and renumbered as a stored level's are; the
/// close describes it again where it fits.
fn prune_under(
    eng: &Engine,
    f: &mut Tdd,
    vtree: &Vtree,
    t: VtreeIdx,
    to: &mut [u32],
) -> Result<(), OperationError> {
    let lim = eng.limits();
    let order = internal_subtree_order(lim, vtree, t, |_| false)?;
    for &s in &order {
        if f.levels[s.idx()].pairs.implicit().is_some() {
            f.levels[s.idx()].store_if_implicit(lim)?;
            f.levels.mark_changed(s);
        }
    }
    let mut gate = lim.gate();
    // What the products reach, from `t` down.
    let mut reached: Vec<Vec<bool>> = charged(lim, vtree.num_nodes(), Vec::new)?;
    for &s in &order {
        lim.try_resize(&mut reached[s.idx()], f.levels[s.idx()].node_count(), false)?;
    }
    for &x in to.iter().filter(|&&x| x != DEAD) {
        reached[t.idx()][x as usize] = true;
    }
    for &s in &order {
        let (l, r) = vtree.children(s);
        let (inner_l, inner_r) = (!vtree.node(l).is_leaf(), !vtree.node(r).is_leaf());
        let level = &f.levels[s.idx()];
        for i in 0..level.node_count() {
            if !reached[s.idx()][i] {
                continue;
            }
            for p in level.pairs_iter_of_idx(i) {
                if inner_l {
                    reached[l.idx()][p.left.raw() as usize] = true;
                }
                if inner_r {
                    reached[r.idx()][p.right.raw() as usize] = true;
                }
            }
        }
        gate.poll(level.node_count() as u64)?;
    }
    // Children before parents: each level's references rewritten through
    // its children's new indices, and its unreached nodes dropped.
    let mut renumbered: Vec<Moved> = charged(lim, vtree.num_nodes(), || Moved::Same)?;
    for &s in order.iter().rev() {
        let keep = &reached[s.idx()];
        let mut new = Vec::new();
        lim.reserve_exact(&mut new, keep.len())?;
        let mut next = 0u32;
        for &k in keep {
            new.push(match k {
                true => {
                    next += 1;
                    next - 1
                }
                false => DEAD,
            });
        }
        let (l, r) = vtree.children(s);
        if !matches!(renumbered[l.idx()], Moved::Same)
            || !matches!(renumbered[r.idx()], Moved::Same)
        {
            let (ml, mr) = (&renumbered[l.idx()], &renumbered[r.idx()]);
            f.rewrite_level(s, |level| {
                rewrite_level_pairs(lim, level, |_, _, _, p| {
                    let (cl, cr) = (ml.at(p.left.raw()), mr.at(p.right.raw()));
                    (cl != DEAD && cr != DEAD).then(|| ChildPair::new(NodeIdx(cl), NodeIdx(cr)))
                })
            })?;
        }
        if (next as usize) < keep.len() {
            f.levels.mark_changed(s);
            f.try_invalidate(eng, s)?;
            let level = &mut f.levels[s.idx()];
            let mut dead = 0usize;
            for (i, &k) in keep.iter().enumerate() {
                if !k {
                    dead += level.arena_pairs_at(i);
                }
            }
            let mut i = 0;
            level.nodes.stored_mut().retain(|_| {
                i += 1;
                keep[i - 1]
            });
            level.note_dead_pairs(dead);
            level.compact_pairs_if_stale();
            renumbered[s.idx()] = Moved::To(new);
        }
        gate.poll(keep.len() as u64)?;
    }
    gate.flush()?;
    let moved = &renumbered[t.idx()];
    for x in to.iter_mut().filter(|x| **x != DEAD) {
        *x = moved.at(*x);
    }
    Ok(())
}

/// The implementation behind [`Engine::and_in_place`]: the products below
/// each branching level found bottom-up ([`block_products`]).
pub(crate) fn and_in_place_on(
    eng: &Engine,
    mut f: Tdd,
    g: &Tdd,
    prune: bool,
) -> Result<Tdd, OperationError> {
    let lim = eng.limits();
    let _op = lim.enter()?;
    super::check_vtree(&f, g)?;
    f.require_structure()?;
    g.require_structure()?;
    if f.weights.is_some() || g.weights.is_some() {
        return Err(OperationError::InertOption {
            option: "and_in_place",
            needs: "unweighted operands",
        });
    }
    if g.is_zero() || f.is_zero() {
        return crate::build::constant_like(eng, &f, false);
    }
    let vtree = Arc::clone(f.vtree());
    let n = vtree.num_nodes();
    let top = true_nodes(lim, g)?;
    let root = vtree.root();
    debug_assert!(
        f.output.vtree == root && g.output.vtree == root,
        "a structural output sits at the root"
    );

    let role = classify_g_roles(lim, g, &top)?;
    if role[root.idx()] == Role::True {
        return Ok(f);
    }
    // What a later prune may walk: levels whose nodes the rewrite may leave
    // unnamed by their parent level, read before any edit forgets them.
    // Whether every level was closed before the edits, which mark the
    // levels they change for the close at the end.
    let closed = f.levels.is_closed();
    let mut loose: Option<Vec<u32>> = match f.levels.is_canonical(f.output) {
        true => Some(Vec::new()),
        false => match f.dirty.loose() {
            Some(listed) => {
                let mut copy = Vec::new();
                lim.reserve_exact(&mut copy, listed.len())?;
                copy.extend_from_slice(listed);
                Some(copy)
            }
            None => None,
        },
    };

    // The products where `g` branches: every original node of `f` there.
    let mut moved: Vec<Moved> = charged(lim, n, || Moved::Same)?;
    let mut grown = charged(lim, n, || false)?;
    let mut branched: Vec<VtreeIdx> = Vec::new();
    for t in vtree.bottomup() {
        let Role::Branch(b) = role[t.idx()] else {
            continue;
        };
        if vtree.node(t).is_leaf() {
            let mut to = Vec::new();
            lim.reserve_exact(&mut to, 3)?;
            to.extend(
                [ONE_LEAF_IDX, POS_LEAF_IDX, NEG_LEAF_IDX]
                    .iter()
                    .map(|a| leaf_product(a.0, b)),
            );
            moved[t.idx()] = Moved::To(to);
            continue;
        }
        moved[t.idx()] = Moved::To(block_products(lim, &mut f, g, &top, t, b, &mut grown)?);
        if let Some(loose) = loose.as_mut() {
            lim.try_push(loose, t.0)?;
        }
        if !prune {
            lim.try_push(&mut branched, t)?;
        }
    }
    for (t, grew) in grown.iter().enumerate() {
        if *grew {
            let t = VtreeIdx(t as u32);
            f.levels.mark_changed(t);
            f.try_invalidate(eng, t)?;
        }
    }
    // Unpruned, each branching level's region is pruned on its own.
    for t in branched {
        let Moved::To(to) = &mut moved[t.idx()] else {
            unreachable!("a branching level maps its nodes")
        };
        prune_under(eng, &mut f, &vtree, t, to)?;
    }

    // The spine, children before parents: each pair rewritten through its
    // children's moves, a pair naming a false child dropped, and a node left
    // with none dropped and the level renumbered. A level held as a
    // description is stored first, as `prune_under` stores one, and the
    // close describes it again where it fits.
    let mut gate = lim.gate();
    for (t, _, _) in vtree.internal_bottomup() {
        if matches!(role[t.idx()], Role::Spine(_)) {
            rewrite_spine_level(lim, &mut f, &vtree, t, &mut moved, &mut loose, &mut gate)?;
        }
    }
    gate.flush()?;
    let out = moved[root.idx()].at(f.output.local.0);
    if out == DEAD {
        // Unsatisfiable: the canonical false, not a root over dead levels.
        return crate::build::constant_like(eng, &f, false);
    }
    f.output.local = NodeIdx(out);
    if let Some(mut loose) = loose {
        loose.sort_unstable();
        loose.dedup();
        f.dirty.set_loose(Some(loose));
    }
    match prune {
        true => eng.reduce(&mut f, ReductionPlan::Prune)?,
        false => f.close_marked_levels(closed),
    }
    Ok(f)
}

/// Classify `g`'s levels into [`Role::True`], [`Role::Spine`], and
/// [`Role::Branch`] from its output down.
fn classify_g_roles(
    lim: &Limits,
    g: &Tdd,
    top: &[Option<u32>],
) -> Result<Vec<Role>, OperationError> {
    let vtree = g.vtree();
    let mut role = charged(lim, vtree.num_nodes(), || Role::True)?;
    let mut stack = Vec::new();
    lim.try_push(&mut stack, (vtree.root(), g.output.local.0))?;
    while let Some((t, b)) = stack.pop() {
        if top[t.idx()] == Some(b) {
            continue;
        }
        if vtree.node(t).is_leaf() {
            role[t.idx()] = Role::Branch(b);
            continue;
        }
        let level = g.level(t);
        let mut pairs = level.pairs_iter_of_idx(b as usize);
        let first = pairs.next().expect("a stored node has a pair");
        if pairs.next().is_some() {
            role[t.idx()] = Role::Branch(b);
            continue;
        }
        role[t.idx()] = Role::Spine(b);
        let (l, r) = vtree.children(t);
        lim.try_push(&mut stack, (l, first.left.raw()))?;
        lim.try_push(&mut stack, (r, first.right.raw()))?;
    }
    Ok(role)
}

/// Rewrite one spine level `t` of `f` through its children's [`Moved`] maps,
/// compacting any emptied nodes.
fn rewrite_spine_level(
    lim: &Limits,
    f: &mut Tdd,
    vtree: &Vtree,
    t: VtreeIdx,
    moved: &mut [Moved],
    loose: &mut Option<Vec<u32>>,
    gate: &mut crate::limits::PollGate,
) -> Result<(), OperationError> {
    let (l, r) = vtree.children(t);
    let (ml, mr) = (&moved[l.idx()], &moved[r.idx()]);
    let emptied = f.rewrite_level(t, |level| {
        rewrite_level_pairs(lim, level, |_, _, _, p| {
            let cl = ml.at(p.left.raw());
            if cl == DEAD {
                return None;
            }
            let cr = mr.at(p.right.raw());
            if cr == DEAD {
                return None;
            }
            Some(ChildPair::new(NodeIdx(cl), NodeIdx(cr)))
        })
    })?;
    gate.poll(f.levels[t.idx()].node_count() as u64)?;
    if emptied {
        let level = &mut f.levels[t.idx()];
        level.store_if_implicit(lim)?;
        let width = level.node_count();
        let mut to = Vec::new();
        lim.reserve_exact(&mut to, width)?;
        let mut next = 0u32;
        for i in 0..width {
            to.push(match level.pair_count_at(i) {
                0 => DEAD,
                _ => {
                    next += 1;
                    next - 1
                }
            });
        }
        let mut i = 0;
        level.nodes.stored_mut().retain(|_| {
            i += 1;
            to[i - 1] != DEAD
        });
        moved[t.idx()] = Moved::To(to);
    }
    if let Some(loose) = loose.as_mut() {
        for c in [l, r] {
            if !vtree.node(c).is_leaf() {
                lim.try_push(loose, c.0)?;
            }
        }
    }
    Ok(())
}

impl Engine {
    /// `f ∧ g`, written into `f`'s own levels: where `g` is true over a
    /// level's variables, `f`'s nodes there are kept as they are.
    ///
    /// Meant for a small, local `g` — a cube, a set of codes over a few
    /// variables, a conjunction of such constraints over disjoint subtrees —
    /// and a large `f`: below the levels where `g` has one node of one pair,
    /// `f`'s pairs are rewritten in place, a pair whose child became false
    /// dropped; where `g` branches, the products with `f`'s nodes there are
    /// found in one pass up the levels under it, each pair of `f` there read
    /// once against a table of `g`'s pairs, and appended to `f`'s levels;
    /// and the levels under them where `g` is true are not read at all. Any
    /// `g` on `f`'s vtree gives the conjunction; one that branches near the
    /// root gives it at the cost of building most of the product beside `f`.
    ///
    /// The result is pruned of every node it no longer reaches, so its pairs
    /// are a subset of `f`'s and the products', but it may hold twins:
    /// [`Engine::minimize`] makes it canonical. Both operands must be
    /// structural and unweighted, and share their vtree.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{Engine, Vtree};
    ///
    /// let engine = Engine::new();
    /// let vtree = Arc::new(Vtree::balanced(4));
    /// let f = engine.clause(&vtree, [1, 2, 3])?;
    /// let g = engine.cube(&vtree, [-1, 4])?;
    /// let both = engine.and_in_place(f.clone(), &g)?;
    /// assert!(engine.equivalent(&both, &engine.and(f, g)?)?);
    /// # Ok::<(), tididi::OperationError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// [`OperationError::VtreeMismatch`] when the operands do not share a
    /// vtree, [`OperationError::MarginalLevel`] when either has a marginal
    /// level, [`OperationError::InertOption`] when either is weighted,
    /// [`OperationError::IndexOverflow`] when a level's products outgrow a
    /// 32-bit index, and the engine's refusals ([`OperationError::Stopped`],
    /// [`OperationError::OverBudget`]). A refusal consumes `f`.
    pub fn and_in_place(&self, f: Tdd, g: &Tdd) -> Result<Tdd, OperationError> {
        and_in_place_on(self, f, g, true)
    }

    /// [`Engine::and_in_place`] without the prune: a node of `f` the
    /// conjunction no longer reaches stays where it is, and the levels that
    /// may hold one are listed for the next prune, which a conjunction with
    /// the result or [`Engine::minimize`] runs, or which never runs where the
    /// result is read only from its output. The pass then costs the pairs of
    /// `g`'s spine levels in `f` and the products where `g` branches, however
    /// much of `f` it cuts away. The result is the conjunction, but its
    /// sizes ([`Tdd::pair_count`], [`Tdd::node_count`]) count what the
    /// prune would remove.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{Engine, Vtree};
    ///
    /// let engine = Engine::new();
    /// let vtree = Arc::new(Vtree::balanced(4));
    /// let f = engine.clause(&vtree, [1, 2, 3])?;
    /// let g = engine.cube(&vtree, [-1, 4])?;
    /// let mut both = engine.and_in_place_loose(f.clone(), &g)?;
    /// assert!(engine.equivalent(&both, &engine.and(f, g)?)?);
    /// engine.minimize(&mut both)?;
    /// # Ok::<(), tididi::OperationError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Those of [`Engine::and_in_place`].
    pub fn and_in_place_loose(&self, f: Tdd, g: &Tdd) -> Result<Tdd, OperationError> {
        and_in_place_on(self, f, g, false)
    }

    /// `f` conjoined with a cube, the conjunction of `literals`, in `f`'s
    /// own levels: [`Engine::and_in_place`] with the cube's diagram, which
    /// keeps the literals where [`Engine::condition`] removes their
    /// variables. Each literal's leaf has its parent level's pairs of the
    /// other polarity dropped and those true over the leaf set to the
    /// literal; the rest of the work is the falsity that leaves and the
    /// prune.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{Engine, Vtree};
    ///
    /// let engine = Engine::new();
    /// let vtree = Arc::new(Vtree::balanced(3));
    /// let f = engine.clause(&vtree, [1, 2])?;
    /// let kept = engine.and_cube(f, [-1, 3])?;
    /// assert_eq!(engine.model_count(&kept)?, 1u32.into());
    /// # Ok::<(), tididi::OperationError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Those of [`Engine::cube`] and [`Engine::and_in_place`].
    pub fn and_cube(
        &self,
        f: Tdd,
        literals: impl IntoIterator<
            Item = impl TryInto<crate::diagram::Literal, Error: Into<OperationError>>,
        >,
    ) -> Result<Tdd, OperationError> {
        let cube = self.cube(f.vtree(), literals)?;
        and_in_place_on(self, f, &cube, true)
    }
}

#[cfg(test)]
#[path = "tests/settle.rs"]
mod tests;
