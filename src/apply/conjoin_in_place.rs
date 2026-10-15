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
//!
//! The tests compare the same products found top-down, one memoized
//! product per pair of nodes reached from the branching level
//! (`Products`), which decides a node of `f` lying inside one node of `g`
//! without its product (`Products::inside`) and finds the pairs of a wide
//! node of `g` that a product can meet by their sides (`GSides`).

use std::rc::Rc;
use std::sync::Arc;

use rustc_hash::FxHashMap;

use crate::Engine;
use crate::diagram::{ChildPair, NodeIdx, Tdd};
use crate::diagram::{NEG_LEAF_IDX, ONE_LEAF_IDX, POS_LEAF_IDX};
use crate::limits::{Limits, OperationError};
use crate::reduce::ReductionPlan;
use crate::vtree::{Vtree, VtreeIdx};

use super::falsity::rewrite_level_pairs;

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
        let (Some(tl), Some(tr)) = (top[l.idx()], top[r.idx()]) else { continue };
        let level = g.level(t);
        top[t.idx()] = (0..level.node_count()).find_map(|i| {
            let mut pairs = level.pairs_iter_of_idx(i);
            let first = pairs.next()?;
            (pairs.next().is_none() && first.left.raw() == tl && first.right.raw() == tr).then_some(i as u32)
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

/// The top-down products below the levels where `g` branches.
struct Products<'a> {
    lim: &'a Limits,
    vtree: &'a Vtree,
    f: &'a mut Tdd,
    g: &'a Tdd,
    top: &'a [Option<u32>],
    /// Per level, the product of `(a, b)` keyed `a << 32 | b`.
    memo: Vec<FxHashMap<u64, u32>>,
    /// Per level, `g`'s node holding each pair, by the pair's key; built
    /// on the level's first read.
    g_pairs: Vec<Option<FxHashMap<u64, u32>>>,
    /// Per level, where each original node of `f` lies among `g`'s nodes
    /// there ([`Products::inside`]); sized on the level's first read.
    inside: Vec<Vec<u32>>,
    /// The levels a product was appended to.
    grown: Vec<bool>,
    /// The pairs read, for the stop and the work clock.
    work: u64,
    /// Past this many pairs of pairs a product finds `g`'s pairs by their
    /// sides ([`GRID_PAIRS`]).
    grid: usize,
    /// `g`'s nodes so indexed, by level and node.
    sides: FxHashMap<(u32, u32), Rc<GSides>>,
}

impl Products<'_> {
    /// `f`'s node `a` conjoined with `g`'s node `b` at level `t`: `a` where
    /// the product is `a`, a node appended to `f`'s level `t` otherwise, and
    /// [`DEAD`] where it is false.
    fn product(&mut self, t: VtreeIdx, a: u32, b: u32) -> Result<u32, OperationError> {
        if self.top[t.idx()] == Some(b) {
            return Ok(a);
        }
        if self.vtree.node(t).is_leaf() {
            return Ok(leaf_product(a, b));
        }
        let key = (u64::from(a) << 32) | u64::from(b);
        if let Some(&done) = self.memo[t.idx()].get(&key) {
            return Ok(done);
        }
        // Where `f`'s node lies inside one node of `g`, the product is `f`'s
        // node there and false at every other; where it meets none, false.
        match self.inside(t, a)? {
            x if x == b => return Ok(a),
            UNKNOWN => {}
            _ => return Ok(DEAD),
        }
        let (l, r) = self.vtree.children(t);
        let fa: smallvec::SmallVec<[ChildPair; 8]> = self.f.levels[t.idx()].pairs_iter_of_idx(a as usize).collect();
        // Past a few pairs of pairs, `b`'s pairs are found by their sides:
        // a side of `a`'s pair that lies inside one node of `g` meets no
        // other node there, so only `b`'s pairs on that node can give a
        // product, and every other would come back false. The index is
        // built once per node of `g`, which meets many of `f`'s.
        let wide = fa.len() * self.g.level(t).pair_count_at(b as usize) > self.grid;
        let sides = match wide {
            true => Some(self.sides_of(t, b)?),
            false => None,
        };
        let few: smallvec::SmallVec<[ChildPair; 8]> = match wide {
            true => smallvec::SmallVec::new(),
            false => self.g.level(t).pairs_iter_of_idx(b as usize).collect(),
        };
        let gb: &[ChildPair] = match &sides {
            Some(by) => &by.pairs,
            None => &few,
        };
        let mut out: smallvec::SmallVec<[ChildPair; 8]> = smallvec::SmallVec::new();
        let mut same = true;
        for pa in &fa {
            let mut hits = 0;
            let cands: smallvec::SmallVec<[u32; 8]> = match &sides {
                Some(by) => match self.meeting(l, r, pa, by)? {
                    Some(js) => js.iter().copied().collect(),
                    None => (0..gb.len() as u32).collect(),
                },
                None => (0..gb.len() as u32).collect(),
            };
            self.work += cands.len() as u64;
            for &j in &cands {
                let pb = &gb[j as usize];
                let cl = self.product(l, pa.left.raw(), pb.left.raw())?;
                if cl == DEAD {
                    continue;
                }
                let cr = self.product(r, pa.right.raw(), pb.right.raw())?;
                if cr == DEAD {
                    continue;
                }
                hits += 1;
                same &= cl == pa.left.raw() && cr == pa.right.raw();
                out.push(ChildPair::new(NodeIdx(cl), NodeIdx(cr)));
            }
            same &= hits == 1;
        }
        let done = if out.is_empty() {
            DEAD
        } else if same {
            a
        } else {
            crate::diagram::sort_pairs(&mut out);
            self.grown[t.idx()] = true;
            self.f.levels[t.idx()].push_node(self.lim, &out)?.0
        };
        self.memo[t.idx()].insert(key, done);
        Ok(done)
    }

    /// `g`'s node `b` at level `t` indexed by its pairs' sides, built on its
    /// first wide product.
    fn sides_of(&mut self, t: VtreeIdx, b: u32) -> Result<Rc<GSides>, OperationError> {
        if let Some(by) = self.sides.get(&(t.0, b)) {
            return Ok(Rc::clone(by));
        }
        let g = self.g;
        let by = Rc::new(GSides::new(self.lim, g.level(t).pairs_iter_of_idx(b as usize))?);
        self.lim.reserve_map(&mut self.sides, 1)?;
        self.sides.insert((t.0, b), Rc::clone(&by));
        Ok(by)
    }

    /// The pairs of `g`'s node, indexed by their sides in `by`, that `f`'s
    /// pair `pa` (at a level with children `l`, `r`) can meet. A side of
    /// `pa` at an internal child that lies inside one node of `g` there
    /// meets no other node of that level, and one that lies outside every
    /// node meets none: so the pair on both sides' nodes where both lie
    /// inside one (at most one pair, as a node's pairs differ), those on
    /// the one side's node where one does, none where one lies outside, and
    /// `None`, every pair, where neither side says.
    fn meeting<'b>(&mut self, l: VtreeIdx, r: VtreeIdx, pa: &ChildPair, by: &'b GSides) -> Result<Option<&'b [u32]>, OperationError> {
        let mut at = [UNKNOWN; 2];
        for (s, child, x) in [(0, l, pa.left.raw()), (1, r, pa.right.raw())] {
            if self.vtree.node(child).is_leaf() {
                continue;
            }
            match self.inside(child, x)? {
                OUTSIDE => return Ok(Some(&[])),
                y => at[s] = y,
            }
        }
        Ok(match at {
            [UNKNOWN, UNKNOWN] => None,
            [y, UNKNOWN] => Some(by.on(0, y)),
            [UNKNOWN, z] => Some(by.on(1, z)),
            [y, z] => Some(by.both(y, z)),
        })
    }

    /// Where original node `a` of `f` at internal level `t` lies among
    /// `g`'s nodes there: the one holding it whole, [`OUTSIDE`] where it
    /// meets none, [`UNKNOWN`] where it meets some but no one holds it. One
    /// pass over `f`'s nodes below, memoized: a pair of `a` lies in `g`'s
    /// node holding the pair of the nodes its sides lie in, since `g`'s
    /// nodes at a level are disjoint; a leaf side is its label, which the
    /// label itself and `⊤` hold.
    fn inside(&mut self, t: VtreeIdx, a: u32) -> Result<u32, OperationError> {
        if let Some(x) = self.top[t.idx()] {
            return Ok(x);
        }
        let width = self.f.levels[t.idx()].node_count();
        if self.inside[t.idx()].is_empty() {
            self.lim.try_resize(&mut self.inside[t.idx()], width, UNSET)?;
        }
        match self.inside[t.idx()].get(a as usize) {
            Some(&x) if x != UNSET => return Ok(x),
            Some(_) => {}
            None => return Ok(UNKNOWN),
        }
        if self.g_pairs[t.idx()].is_none() {
            let level = self.g.level(t);
            let mut map = FxHashMap::default();
            for i in 0..level.node_count() {
                for p in level.pairs_iter_of_idx(i) {
                    map.insert(p.key(), i as u32);
                }
            }
            self.g_pairs[t.idx()] = Some(map);
        }
        let (l, r) = self.vtree.children(t);
        let pairs: smallvec::SmallVec<[ChildPair; 8]> = self.f.levels[t.idx()].pairs_iter_of_idx(a as usize).collect();
        self.work += pairs.len() as u64;
        let mut verdict = UNSET;
        for p in &pairs {
            let (sl, sr) = (self.side(l, p.left.raw())?, self.side(r, p.right.raw())?);
            let here = self.pair_inside(t, sl, sr);
            verdict = match verdict {
                UNSET => here,
                x if x == here => x,
                _ => UNKNOWN,
            };
            if verdict == UNKNOWN {
                break;
            }
        }
        self.inside[t.idx()][a as usize] = verdict;
        Ok(verdict)
    }

    /// The side `x` of a pair whose child is level `s`, as
    /// [`pair_inside`](Self::pair_inside) reads it.
    fn side(&mut self, s: VtreeIdx, x: u32) -> Result<Side, OperationError> {
        Ok(match self.vtree.node(s).is_leaf() {
            true => Side::Label(x),
            false => Side::Node(self.inside(s, x)?),
        })
    }

    /// Where the pair of sides `sl`, `sr` at level `t` lies among `g`'s
    /// nodes there, as [`inside`](Self::inside) says of a node.
    fn pair_inside(&self, t: VtreeIdx, sl: Side, sr: Side) -> u32 {
        let (Some(wl), Some(wr)) = (sl.within(), sr.within()) else { return OUTSIDE };
        let (Some(pl), Some(pr)) = (sl.partly(), sr.partly()) else { return UNKNOWN };
        let map = self.g_pairs[t.idx()].as_ref().expect("built by inside");
        let at = |x: u32, y: u32| map.get(&ChildPair::new(NodeIdx(x), NodeIdx(y)).key()).copied();
        for &x in wl.iter().flatten() {
            for &y in wr.iter().flatten() {
                if let Some(node) = at(x, y) {
                    return node;
                }
            }
        }
        let meets = wl.iter().chain(&pl).flatten().any(|&x| wr.iter().chain(&pr).flatten().any(|&y| at(x, y).is_some()));
        match meets {
            true => UNKNOWN,
            false => OUTSIDE,
        }
    }
}

/// A product of two nodes with more pairs of pairs than this finds `g`'s
/// pairs by their sides ([`GSides`], [`Products::meeting`]) instead of
/// trying each against each.
const GRID_PAIRS: usize = 64;

/// A node of `g`: its pairs, and their indices by their left side, by
/// their right side and by both, each in ascending order of its key.
struct GSides {
    pairs: Vec<ChildPair>,
    keys: [Vec<u32>; 2],
    by_side: [Vec<u32>; 2],
    both: Vec<u64>,
    by_both: Vec<u32>,
}

impl GSides {
    fn new(lim: &Limits, gb: impl Iterator<Item = ChildPair>) -> Result<GSides, OperationError> {
        let mut pairs = Vec::new();
        for p in gb {
            lim.try_push(&mut pairs, p)?;
        }
        let n = pairs.len();
        let order = |key: &dyn Fn(&ChildPair) -> u64| -> Result<(Vec<u64>, Vec<u32>), OperationError> {
            let mut by: Vec<(u64, u32)> = Vec::new();
            lim.reserve_exact(&mut by, n)?;
            by.extend(pairs.iter().enumerate().map(|(j, p)| (key(p), j as u32)));
            by.sort_unstable();
            Ok(by.into_iter().unzip())
        };
        let (kl, pl) = order(&|p| u64::from(p.left.raw()))?;
        let (kr, pr) = order(&|p| u64::from(p.right.raw()))?;
        let (both, by_both) = order(&|p| p.key())?;
        let narrow = |k: Vec<u64>| k.into_iter().map(|x| x as u32).collect();
        Ok(GSides { keys: [narrow(kl), narrow(kr)], by_side: [pl, pr], both, by_both, pairs })
    }

    /// The pair whose sides are `g`'s nodes `y` (left) and `z` (right), if
    /// there is one.
    fn both(&self, y: u32, z: u32) -> &[u32] {
        let key = (u64::from(y) << 32) | u64::from(z);
        match self.both.binary_search(&key) {
            Ok(i) => &self.by_both[i..i + 1],
            Err(_) => &[],
        }
    }

    /// The pairs whose side `s` (`0` left, `1` right) is `g`'s node `y`.
    fn on(&self, s: usize, y: u32) -> &[u32] {
        let keys = &self.keys[s];
        let lo = keys.partition_point(|&k| k < y);
        let hi = lo + keys[lo..].partition_point(|&k| k == y);
        &self.by_side[s][lo..hi]
    }
}

/// [`Products::inside`]'s answers beside `g`'s node indices.
const UNSET: u32 = u32::MAX;
const OUTSIDE: u32 = u32::MAX - 1;
const UNKNOWN: u32 = u32::MAX - 2;

/// A side of `f`'s pair, as [`Products::pair_inside`] matches it with
/// `g`'s pairs: a leaf label, or where an internal node lies.
#[derive(Clone, Copy)]
enum Side {
    Label(u32),
    Node(u32),
}

impl Side {
    /// `g`'s sides that hold this side whole; `None` where it lies outside
    /// every node of `g`'s level.
    fn within(self) -> Option<[Option<u32>; 2]> {
        match self {
            Side::Label(x) if x == ONE_LEAF_IDX.0 => Some([Some(ONE_LEAF_IDX.0), None]),
            Side::Label(x) => Some([Some(x), Some(ONE_LEAF_IDX.0)]),
            Side::Node(OUTSIDE) => None,
            Side::Node(x) => Some([(x != UNKNOWN).then_some(x), None]),
        }
    }

    /// `g`'s sides that meet this side without holding it; `None` where
    /// that is not known (an internal node that no one node holds).
    fn partly(self) -> Option<[Option<u32>; 2]> {
        match self {
            Side::Label(x) if x == ONE_LEAF_IDX.0 => Some([Some(POS_LEAF_IDX.0), Some(NEG_LEAF_IDX.0)]),
            Side::Label(_) => Some([None, None]),
            Side::Node(UNKNOWN) => None,
            Side::Node(_) => Some([None, None]),
        }
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
    /// meets, `(g's node, product)`, at `pieces[start[x]..start[x + 1]]`:
    /// none where it meets no node of `g`.
    Listed { start: Vec<u32>, pieces: Vec<(u32, u32)> },
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
        Split::Listed { start, pieces } => &pieces[start[x as usize] as usize..start[x as usize + 1] as usize],
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
    start: Vec<u32>,
    pieces: Vec<(u32, u32)>,
    /// The products made, their pairs back to back, at
    /// `made[made_at[k]..made_at[k + 1]]`.
    made: Vec<ChildPair>,
    made_at: Vec<u32>,
}

impl Settle {
    /// Node `a`'s products, from its `n` pairs' hits: per node of `g` its
    /// pairs hit, `a` itself where each of its pairs met that node in one
    /// pair and came back whole, else a node of the products' pairs; the
    /// nodes before `a` that no pair named (none in a stored level) meet
    /// nothing.
    fn node(&mut self, lim: &Limits, a: usize, n: u32) -> Result<(), OperationError> {
        while self.start.len() < a + 1 {
            lim.try_push(&mut self.start, self.pieces.len() as u32)?;
        }
        // Within a node of `g` the pairs stay in their order; unstable, so
        // that the sort asks for no buffer.
        self.hits.sort_unstable_by_key(|h| (h.0, h.1));
        let mut i = 0;
        while i < self.hits.len() {
            let b = self.hits[i].0;
            let k = i + self.hits[i..].iter().take_while(|h| h.0 == b).count();
            let group = &self.hits[i..k];
            let whole = group.len() == n as usize && group.iter().enumerate().all(|(q, h)| h.1 == q as u32 && h.3);
            let product = match whole {
                true => a as u32,
                false => {
                    let at = self.made.len();
                    lim.reserve_exact(&mut self.made, group.len())?;
                    self.made.extend(group.iter().map(|h| h.2));
                    crate::diagram::sort_pairs(&mut self.made[at..]);
                    lim.try_push(&mut self.made_at, self.made.len() as u32)?;
                    self.width + (self.made_at.len() - 2) as u32
                }
            };
            lim.try_push(&mut self.pieces, (b, product))?;
            i = k;
        }
        self.hits.clear();
        lim.try_push(&mut self.start, self.pieces.len() as u32)
    }
}

/// [`Products::product`] of every node of `f` at branching level `t` with
/// `g`'s node `b` there, as `to` maps them, found bottom-up: one pass over
/// the levels under `t`, children before parents, reads each pair of `f`
/// once against its children's products and finds the pairs of `g` they
/// meet in a table of `g`'s pairs at the level, so that each node of `f`
/// comes out with its products with every node of `g` it meets (`g`'s nodes
/// at a level are disjoint). A node that each of its pairs carries whole
/// into one node of `g` is its own product there, and no node is added for
/// it; a level where `g` has only its true node is not read, its nodes
/// their own products. The work is a unit per pair of `f` under `t` and
/// one per probe of `g`'s table, a probe per pair of its sides' products,
/// against a memoized product per pair of nodes. Each product made is
/// appended to `f`'s level, and `grown` marks the level.
#[allow(clippy::too_many_arguments)]
fn block_products(
    lim: &Limits,
    vtree: &Vtree,
    f: &mut Tdd,
    g: &Tdd,
    top: &[Option<u32>],
    t: VtreeIdx,
    b: u32,
    grown: &mut [bool],
) -> Result<Vec<u32>, OperationError> {
    let whole = |u: VtreeIdx| top[u.idx()].filter(|_| g.level(u).node_count() == 1);
    // The internal levels under `t`, parents first; the descent ends where
    // `g` has only its true node.
    let mut order = Vec::new();
    lim.try_push(&mut order, t)?;
    let mut at = 0;
    while at < order.len() {
        let u = order[at];
        at += 1;
        if u != t && whole(u).is_some() {
            continue;
        }
        let (l, r) = vtree.children(u);
        for c in [l, r] {
            if !vtree.node(c).is_leaf() {
                lim.try_push(&mut order, c)?;
            }
        }
    }
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
            false => split[c.idx()].take().expect("a level's children are split before it"),
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
        let mut st = Settle { width: width as u32, hits: std::mem::take(&mut hits), start: Vec::new(), pieces: Vec::new(), made: Vec::new(), made_at: Vec::new() };
        lim.reserve_exact(&mut st.start, width + 1)?;
        lim.try_push(&mut st.made_at, 0)?;
        let mut refused = None;
        let mut cur: Option<usize> = None;
        let mut read = 0u32;
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
                        && let Err(e) = lim.try_push(&mut st.hits, (bn, read, ChildPair::new(NodeIdx(cl), NodeIdx(cr)), cl == pl && cr == pr))
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
        while st.start.len() < width + 1 {
            lim.try_push(&mut st.start, st.pieces.len() as u32)?;
        }
        gate.poll(work)?;
        drop((sl, sr));
        let made = st.made_at.len() - 1;
        for k in 0..made {
            let pairs = &st.made[st.made_at[k] as usize..st.made_at[k + 1] as usize];
            let idx = f.levels[u.idx()].push_node(lim, pairs)?;
            debug_assert_eq!(idx.0 as usize, width + k, "a product is numbered as its rank among those made");
        }
        grown[u.idx()] |= made > 0;
        hits = std::mem::take(&mut st.hits);
        match u == t {
            true => {
                lim.reserve_exact(&mut to, width)?;
                to.extend((0..width).map(|x| match st.start[x] < st.start[x + 1] {
                    true => st.pieces[st.start[x] as usize].1,
                    false => DEAD,
                }));
            }
            false => split[u.idx()] = Some(Split::Listed { start: st.start, pieces: st.pieces }),
        }
    }
    gate.flush()?;
    Ok(to)
}

/// The nodes of branching level `t` and of the levels under it that no node
/// `to` names reaches, removed, and `to` renumbered through what is left.
/// `t`'s products stand beside the nodes they replace, which overlap them,
/// and so do the products under them; the prune would remove those, and
/// [`Engine::and_in_place_loose`] runs none. The work is the levels under
/// `t`. A level held as the description of its pairs is stored first, so
/// that its nodes are dropped and renumbered as a stored level's are; the
/// close describes it again where it fits.
fn prune_under(eng: &Engine, f: &mut Tdd, vtree: &Vtree, t: VtreeIdx, to: &mut [u32]) -> Result<(), OperationError> {
    let lim = eng.limits();
    // The internal levels from `t` down, parents first.
    let mut order = Vec::new();
    lim.try_push(&mut order, t)?;
    let mut at = 0;
    while at < order.len() {
        let (l, r) = vtree.children(order[at]);
        for c in [l, r] {
            if !vtree.node(c).is_leaf() {
                lim.try_push(&mut order, c)?;
            }
        }
        at += 1;
    }
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
        if !matches!(renumbered[l.idx()], Moved::Same) || !matches!(renumbered[r.idx()], Moved::Same) {
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
pub(crate) fn and_in_place_on(eng: &Engine, f: Tdd, g: &Tdd, prune: bool) -> Result<Tdd, OperationError> {
    and_in_place_way(eng, f, g, prune, None)
}

/// [`and_in_place_on`], the products below each branching level found
/// bottom-up ([`block_products`]) without `grid`; with it, top-down
/// instead, one memoized product per pair of nodes ([`Products`]), `grid`
/// their threshold for finding `g`'s pairs by their sides: `0` always,
/// `usize::MAX` never (each pair against each). The tests compare the two.
pub(crate) fn and_in_place_way(eng: &Engine, mut f: Tdd, g: &Tdd, prune: bool, grid: Option<usize>) -> Result<Tdd, OperationError> {
    let lim = eng.limits();
    let _op = lim.enter()?;
    super::check_vtree(&f, g)?;
    f.require_structure()?;
    g.require_structure()?;
    if f.weights.is_some() || g.weights.is_some() {
        return Err(OperationError::InertOption { option: "and_in_place", needs: "unweighted operands" });
    }
    if g.is_zero() || f.is_zero() {
        return crate::build::constant_like(eng, &f, false);
    }
    let vtree = Arc::clone(f.vtree());
    let n = vtree.num_nodes();
    let top = true_nodes(lim, g)?;
    let root = vtree.root();
    debug_assert!(f.output.vtree == root && g.output.vtree == root, "a structural output sits at the root");

    // `g`'s spine and branching levels, from its output down.
    let mut role = charged(lim, n, || Role::True)?;
    let mut stack = Vec::new();
    lim.try_push(&mut stack, (root, g.output.local.0))?;
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
    let mut gate = lim.gate();
    let mut branched: Vec<VtreeIdx> = Vec::new();
    let grown = {
        let mut products = Products {
            lim,
            vtree: &vtree,
            f: &mut f,
            g,
            top: &top,
            memo: charged(lim, n, FxHashMap::default)?,
            g_pairs: charged(lim, n, || None)?,
            inside: charged(lim, n, Vec::new)?,
            grown: charged(lim, n, || false)?,
            work: 0,
            grid: grid.unwrap_or(GRID_PAIRS),
            sides: FxHashMap::default(),
        };
        for t in vtree.bottomup() {
            let Role::Branch(b) = role[t.idx()] else { continue };
            if vtree.node(t).is_leaf() {
                let mut to = Vec::new();
                lim.reserve_exact(&mut to, 3)?;
                to.extend([ONE_LEAF_IDX, POS_LEAF_IDX, NEG_LEAF_IDX].iter().map(|a| leaf_product(a.0, b)));
                moved[t.idx()] = Moved::To(to);
                continue;
            }
            let to = match grid {
                None => block_products(lim, &vtree, products.f, g, &top, t, b, &mut products.grown)?,
                Some(_) => {
                    let width = products.f.levels[t.idx()].node_count() as u32;
                    let mut to = Vec::new();
                    lim.reserve_exact(&mut to, width as usize)?;
                    for x in 0..width {
                        to.push(products.product(t, x, b)?);
                        gate.poll(std::mem::take(&mut products.work))?;
                    }
                    to
                }
            };
            moved[t.idx()] = Moved::To(to);
            if let Some(loose) = loose.as_mut() {
                lim.try_push(loose, t.0)?;
            }
            if !prune {
                lim.try_push(&mut branched, t)?;
            }
        }
        products.grown
    };
    gate.flush()?;
    for (t, grew) in grown.iter().enumerate() {
        if *grew {
            let t = VtreeIdx(t as u32);
            f.levels.mark_changed(t);
            f.try_invalidate(eng, t)?;
        }
    }
    // Unpruned, each branching level's region is pruned on its own.
    for t in branched {
        let Moved::To(to) = &mut moved[t.idx()] else { unreachable!("a branching level maps its nodes") };
        prune_under(eng, &mut f, &vtree, t, to)?;
    }

    // The spine, children before parents: each pair rewritten through its
    // children's moves, a pair naming a false child dropped, and a node left
    // with none dropped and the level renumbered. A level held as a
    // description is stored first, as `prune_under` stores one, and the
    // close describes it again where it fits.
    for (t, l, r) in vtree.internal_bottomup() {
        if !matches!(role[t.idx()], Role::Spine(_)) {
            continue;
        }
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
    /// level, [`OperationError::InertOption`] when either is weighted, and
    /// the engine's refusals ([`OperationError::Stopped`],
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
        literals: impl IntoIterator<Item = impl TryInto<crate::diagram::Literal, Error: Into<OperationError>>>,
    ) -> Result<Tdd, OperationError> {
        let cube = self.cube(f.vtree(), literals)?;
        and_in_place_on(self, f, &cube, true)
    }
}

