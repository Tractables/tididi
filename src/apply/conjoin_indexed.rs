//! Conjunction with a diagram whose pairs are indexed by their sides.
//!
//! `f ∧ g` is a diagram whose nodes are products: a node `a` of `f` and a
//! node `b` of `g` at one level, standing for the conjunction of their
//! functions. A product is *live* when that conjunction is satisfiable. At
//! a leaf the live products are read off the labels. At an internal level
//! `t`, the product of `a` and `b` is live exactly where a pair `(pl, pr)`
//! of `a` and a pair `(bl, br)` of `b` have live products on both sides:
//! `(pl, bl)` at the left child and `(pr, br)` at the right. So the live
//! products at `t` are found from the children's: for each pair of `f`'s
//! node, the live products of its two sides are combined and each
//! combination `(bl, br)` is looked up in [`PairIndex`], which names the
//! node of `g` that owns that pair, if any. Nothing of `g` is read but the
//! index's entries for the combinations tried.
//!
//! Each pair of `f`'s node meets `g` in one of three ways, whichever reads
//! fewer entries when both sides list their products: every combination of
//! the two sides' products looked up as a pair; every pair of `g` with a
//! left side among the left products, its right side checked among the
//! right products; or the mirror of that.
//!
//! Where `f` is true over a whole subtree its node there is the constant
//! true node, and its product with a node of `g` is that node of `g`: the
//! walk lists no products there. The parent meets `g`'s pairs through the
//! index's runs by the other side, and the nodes of `g` that the result
//! reaches are copied, those only, once the products above are known.
//!
//! The products are built in three passes over the levels under the top:
//! the live products bottom-up, as lists of pairs naming the children's
//! products, `g`'s nodes or labels; the nodes of `g` each subtree where `f`
//! is true must copy, top-down; and the output, bottom-up, as nodes
//! ([`Engine::and_indexed`]) or as an algebra's columns
//! ([`Engine::sum_indexed`]).
//!
//! # The bound
//!
//! Let the leaves of a set of variables κ form one subtree of the vtree,
//! with every vtree node outside it a leaf or an ancestor of it, so that
//! an ancestor's other child is a leaf. Let every variable outside κ that
//! `f` or `g` depends on be determined by κ in `f` or in `g`, and let
//! `keys(f)` be the assignments to κ that extend to a model of `f`.
//!
//! - **Products.** A live product at an internal level `t` holds an
//!   assignment `s` to the variables under `t` that extends to a model of
//!   `f` and to one of `g`. The nodes of each operand at `t` are disjoint,
//!   so `s` names the product, and only its values on the variables either
//!   operand depends on matter. Inside κ's subtree those values are part of
//!   a key of `f`; at an ancestor, the key fixes all the others. So a level
//!   holds at most `|keys(f)|` live products, and as many copies.
//! - **Reads.** A combination tried for a pair of `f`'s node holds an
//!   assignment in that pair's product, and these are distinct, so a level
//!   whose children are both internal tries at most `|keys(f)|`, and reads
//!   at most twice that in run lengths to choose the way. A level with a
//!   leaf child crosses at most three labels with the other side's live
//!   products, each reached from at most two pairs of `f` (the labels `f`
//!   uses at a leaf are disjoint): at most `6 · |keys(f)|`, and 81 where
//!   both children are leaves. A run read under a subtree where `f` is
//!   true yields a product pair at each entry. In all, at most
//!   `9 · |keys(f)| + 81` index reads per internal vtree node.
//!
//! The work is `f`'s pairs read once, plus these, whatever `g`'s size: each
//! read is a binary search in one run at most.

use std::sync::Arc;

use smallvec::SmallVec;

use crate::Engine;
use crate::diagram::{Assembly, ChildPair, ColumnAlgebra, InlineSlots, LeafLabel, NodeIdx, PairIndex, PairsIter, SlotPairs, Tdd, TddNodeId, LEAF_WIDTH};
use crate::limits::{Limits, OperationError, PollGate};
use crate::vtree::{Vtree, VtreeIdx};

use super::CONJOIN_GRID;
use super::conjoin::budget::NO_PRODUCT;
use super::conjoin_in_place::true_nodes;

/// The work an indexed walk did, as the bound in the module notes counts it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct IndexedWork {
    /// Live products built at internal levels.
    pub(crate) products: u64,
    /// Index reads: a pair looked up by both sides, or one entry of a run
    /// read, including the run lengths read to choose a way.
    pub(crate) probes: u64,
    /// Nodes of `g` copied under subtrees where `f` is true.
    pub(crate) copies: u64,
    /// Pairs of `f` read.
    pub(crate) f_pairs: u64,
}

/// One product found: `f`'s node `a`, `g`'s node `b`, and a pair of the
/// product, its sides named in the children's terms.
#[derive(Clone, Copy)]
struct Hit {
    a: u32,
    b: u32,
    pair: ChildPair,
}

/// The products of one internal level.
struct Products {
    /// `f`'s node `x`'s products as `(g's node, product)`, ascending:
    /// `items[start[x]..start[x + 1]]`. Empty when grouped by `f`'s node.
    start: Vec<u32>,
    items: Vec<(u32, u32)>,
    /// Product `k`'s pairs, `made[made_at[k]..made_at[k + 1]]`: each side a
    /// product, a node of `g` or a label of the child level, as the child's
    /// [`Level`] says.
    made_at: Vec<u32>,
    made: Vec<ChildPair>,
}

impl Products {
    fn count(&self) -> usize {
        self.made_at.len().saturating_sub(1)
    }

    fn pairs(&self, k: usize) -> &[ChildPair] {
        &self.made[self.made_at[k] as usize..self.made_at[k + 1] as usize]
    }
}

/// How the walk holds one level.
enum Level {
    /// A leaf: a product is the label [`CONJOIN_GRID`] gives.
    Leaf,
    /// `f` is the true node over the level: the product with `g`'s node `y`
    /// is `y`, copied where reached.
    True,
    /// The live products.
    Listed(Products),
}

/// The products of one side of `f`'s pair.
enum Pieces<'a> {
    /// These, as `(g's node, product)` ascending.
    List(&'a [(u32, u32)]),
    /// One for every node of `g`: the side is the true node of a [`Level::True`] level.
    All,
}

/// The three passes' state.
struct Walk<'a> {
    lim: &'a Limits,
    vtree: &'a Vtree,
    f: &'a Tdd,
    g: &'a Tdd,
    index: &'a PairIndex,
    work: IndexedWork,
}

/// The walk's plan: the levels under the top, parents first, each held as
/// its [`Level`], and the nodes of `g` each [`Level::True`] level copies,
/// ascending.
struct Plan {
    order: Vec<VtreeIdx>,
    levels: Vec<Option<Level>>,
    copies: Vec<Vec<u32>>,
}

impl Walk<'_> {
    /// The internal levels under `top` and its leaves, parents first.
    fn order(&self, top: VtreeIdx) -> Result<Vec<VtreeIdx>, OperationError> {
        let mut order = Vec::new();
        self.lim.try_push(&mut order, top)?;
        let mut at = 0;
        while at < order.len() {
            let t = order[at];
            at += 1;
            if !self.vtree.node(t).is_leaf() {
                let (l, r) = self.vtree.children(t);
                self.lim.try_push(&mut order, l)?;
                self.lim.try_push(&mut order, r)?;
            }
        }
        Ok(order)
    }

    /// The nodes of `f` reachable from `roots` at `top`, per level, ascending.
    fn reach(&mut self, order: &[VtreeIdx], top: VtreeIdx, roots: Vec<u32>) -> Result<Vec<Vec<u32>>, OperationError> {
        let mut reach: Vec<Vec<u32>> = Vec::new();
        self.lim.reserve_exact(&mut reach, self.vtree.num_nodes())?;
        reach.resize_with(self.vtree.num_nodes(), Vec::new);
        reach[top.idx()] = roots;
        let mut gate = self.lim.gate();
        for &t in order {
            if self.vtree.node(t).is_leaf() {
                continue;
            }
            let (l, r) = self.vtree.children(t);
            let f = self.f;
            let level = f.level(t);
            let (mut ls, mut rs) = (Vec::new(), Vec::new());
            for &x in &reach[t.idx()] {
                let pairs = level.pairs_iter_of_idx(x as usize);
                gate.poll(pairs.len() as u64)?;
                for p in pairs {
                    self.lim.try_push(&mut ls, p.left.raw())?;
                    self.lim.try_push(&mut rs, p.right.raw())?;
                }
            }
            for side in [&mut ls, &mut rs] {
                side.sort_unstable();
                side.dedup();
            }
            reach[l.idx()] = ls;
            reach[r.idx()] = rs;
        }
        gate.flush()?;
        Ok(reach)
    }

    /// The products of `f`'s pair side `x` at child level `c`.
    fn pieces<'p>(level: &'p Level, x: u32, buf: &'p mut SmallVec<[(u32, u32); 3]>) -> Pieces<'p> {
        match level {
            Level::Leaf => {
                buf.clear();
                for y in 0..LEAF_WIDTH as u32 {
                    let p = CONJOIN_GRID[x as usize][y as usize];
                    if p != NO_PRODUCT {
                        buf.push((y, p));
                    }
                }
                Pieces::List(buf)
            }
            Level::True => Pieces::All,
            Level::Listed(p) => {
                let (from, to) = (p.start[x as usize] as usize, p.start[x as usize + 1] as usize);
                Pieces::List(&p.items[from..to])
            }
        }
    }

    /// The products of `f`'s node `a` at `t` through one pair of it, whose
    /// sides' products are `left` and `right`, pushed onto `hits`.
    fn meet(
        &mut self,
        t: VtreeIdx,
        a: u32,
        left: Pieces<'_>,
        right: Pieces<'_>,
        leafy: bool,
        hits: &mut Vec<Hit>,
    ) -> Result<(), OperationError> {
        let index = self.index;
        let lim = self.lim;
        let mut hit = |b: u32, l: u32, r: u32| lim.try_push(hits, Hit { a, b, pair: ChildPair::new(NodeIdx(l), NodeIdx(r)) });
        let probes = &mut self.work.probes;
        match (left, right) {
            (Pieces::All, Pieces::All) => {
                for (l, r, b) in index.all_at(t) {
                    *probes += 1;
                    hit(b, l, r)?;
                }
            }
            (Pieces::All, Pieces::List(rs)) => {
                for &(br, rp) in rs {
                    let run = index.right_run(t, br);
                    *probes += 1 + run.len() as u64;
                    for &(bl, b) in run {
                        hit(b, bl, rp)?;
                    }
                }
            }
            (Pieces::List(ls), Pieces::All) => {
                for &(bl, lp) in ls {
                    let run = index.left_run(t, bl);
                    *probes += 1 + run.len() as u64;
                    for &(br, b) in run {
                        hit(b, lp, br)?;
                    }
                }
            }
            (Pieces::List(ls), Pieces::List(rs)) => {
                if ls.is_empty() || rs.is_empty() {
                    return Ok(());
                }
                let cross = ls.len() as u64 * rs.len() as u64;
                // With a leaf side the crossing is at most three times the
                // other side, and the runs are not read to compare.
                let (by_left, by_right) = match leafy {
                    true => (u64::MAX, u64::MAX),
                    false => {
                        *probes += (ls.len() + rs.len()) as u64;
                        (
                            ls.iter().map(|&(bl, _)| index.left_run(t, bl).len() as u64).sum(),
                            rs.iter().map(|&(br, _)| index.right_run(t, br).len() as u64).sum(),
                        )
                    }
                };
                if cross <= by_left && cross <= by_right {
                    *probes += cross;
                    for &(bl, lp) in ls {
                        for &(br, rp) in rs {
                            if let Some(b) = index.owner_of(t, bl, br) {
                                hit(b, lp, rp)?;
                            }
                        }
                    }
                } else if by_left <= by_right {
                    *probes += by_left;
                    for &(bl, lp) in ls {
                        for &(br, b) in index.left_run(t, bl) {
                            if let Ok(k) = rs.binary_search_by_key(&br, |e| e.0) {
                                hit(b, lp, rs[k].1)?;
                            }
                        }
                    }
                } else {
                    *probes += by_right;
                    for &(br, rp) in rs {
                        for &(bl, b) in index.right_run(t, br) {
                            if let Ok(k) = ls.binary_search_by_key(&bl, |e| e.0) {
                                hit(b, ls[k].1, rp)?;
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// The live products at internal level `t` of `f`'s nodes `nodes`,
    /// from the children's levels; grouped by `f`'s node alone when `by_f`,
    /// one product per node of `f` there.
    fn products(
        &mut self,
        t: VtreeIdx,
        nodes: &[u32],
        levels: &[Option<Level>],
        by_f: bool,
        gate: &mut PollGate<'_>,
    ) -> Result<Products, OperationError> {
        let (l, r) = self.vtree.children(t);
        let (ll, rl) = (
            levels[l.idx()].as_ref().expect("a level's children come first"),
            levels[r.idx()].as_ref().expect("a level's children come first"),
        );
        let leafy = self.vtree.node(l).is_leaf() || self.vtree.node(r).is_leaf();
        let f = self.f;
        let level = f.level(t);
        let mut hits: Vec<Hit> = Vec::new();
        let (mut lbuf, mut rbuf) = (SmallVec::new(), SmallVec::new());
        for &a in nodes {
            let before = self.work.probes;
            for p in level.pairs_iter_of_idx(a as usize) {
                self.work.f_pairs += 1;
                let left = Self::pieces(ll, p.left.raw(), &mut lbuf);
                let right = Self::pieces(rl, p.right.raw(), &mut rbuf);
                self.meet(t, a, left, right, leafy, &mut hits)?;
            }
            gate.poll(1 + self.work.probes - before)?;
        }
        hits.sort_unstable_by_key(|h| (h.a, h.b, h.pair.key()));
        let width = self.f.reference_slot_count(t);
        let mut out = Products { start: Vec::new(), items: Vec::new(), made_at: Vec::new(), made: Vec::new() };
        self.lim.reserve_exact(&mut out.made, hits.len())?;
        let mut last: Option<(u32, u32)> = None;
        match by_f {
            true => {
                // One product per node of `f`, empty where nothing met it.
                self.lim.reserve_exact(&mut out.made_at, width + 1)?;
                let mut h = 0;
                for x in 0..width as u32 {
                    out.made_at.push(out.made.len() as u32);
                    while h < hits.len() && hits[h].a == x {
                        if last != Some((hits[h].a, hits[h].b)) {
                            last = Some((hits[h].a, hits[h].b));
                            self.work.products += 1;
                        }
                        out.made.push(hits[h].pair);
                        h += 1;
                    }
                }
                out.made_at.push(out.made.len() as u32);
            }
            false => {
                self.lim.try_resize(&mut out.start, width + 1, 0)?;
                for h in &hits {
                    if last != Some((h.a, h.b)) {
                        last = Some((h.a, h.b));
                        let k = out.made_at.len() as u32;
                        self.lim.try_push(&mut out.made_at, out.made.len() as u32)?;
                        self.lim.try_push(&mut out.items, (h.b, k))?;
                        out.start[h.a as usize + 1] += 1;
                    }
                    out.made.push(h.pair);
                }
                self.lim.try_push(&mut out.made_at, out.made.len() as u32)?;
                for x in 0..width {
                    out.start[x + 1] += out.start[x];
                }
                self.work.products += out.items.len() as u64;
            }
        }
        if u32::try_from(out.made.len()).is_err() {
            return Err(OperationError::IndexOverflow);
        }
        Ok(out)
    }

    /// The three passes' first two: the levels under `top` with their live
    /// products, `roots` the nodes of `f` at `top` to begin from, and the
    /// nodes of `g` to copy.
    fn plan(&mut self, top: VtreeIdx, roots: Vec<u32>, by_f: bool) -> Result<Plan, OperationError> {
        let order = self.order(top)?;
        let reach = self.reach(&order, top, roots)?;
        let tops = true_nodes(self.f);
        let mut levels: Vec<Option<Level>> = Vec::new();
        self.lim.reserve_exact(&mut levels, self.vtree.num_nodes())?;
        levels.resize_with(self.vtree.num_nodes(), || None);
        let mut gate = self.lim.gate();
        for &t in order.iter().rev() {
            let level = match self.vtree.node(t).is_leaf() {
                true => Level::Leaf,
                false if t != top && tops[t.idx()].is_some_and(|x| reach[t.idx()] == [x]) => Level::True,
                false => Level::Listed(self.products(t, &reach[t.idx()], &levels, by_f && t == top, &mut gate)?),
            };
            levels[t.idx()] = Some(level);
        }
        drop(reach);
        // The nodes of `g` each true level copies: those its parent's
        // products name, and under them their pairs' sides.
        let mut copies: Vec<Vec<u32>> = Vec::new();
        self.lim.reserve_exact(&mut copies, self.vtree.num_nodes())?;
        copies.resize_with(self.vtree.num_nodes(), Vec::new);
        for &t in &order {
            if self.vtree.node(t).is_leaf() {
                continue;
            }
            let (l, r) = self.vtree.children(t);
            let true_side = |c: VtreeIdx| matches!(levels[c.idx()], Some(Level::True));
            match levels[t.idx()].as_ref().expect("every level under the top is planned") {
                Level::Listed(p) => {
                    for (c, left) in [(l, true), (r, false)] {
                        if true_side(c) {
                            for pair in &p.made {
                                let side = if left { pair.left.raw() } else { pair.right.raw() };
                                self.lim.try_push(&mut copies[c.idx()], side)?;
                            }
                        }
                    }
                }
                Level::True => {
                    let mut mine = std::mem::take(&mut copies[t.idx()]);
                    mine.sort_unstable();
                    mine.dedup();
                    let g = self.g;
                    let level = g.level(t);
                    for &y in &mine {
                        self.work.copies += 1;
                        let pairs = level.pairs_iter_of_idx(y as usize);
                        gate.poll(pairs.len() as u64)?;
                        for pair in pairs {
                            if true_side(l) {
                                self.lim.try_push(&mut copies[l.idx()], pair.left.raw())?;
                            }
                            if true_side(r) {
                                self.lim.try_push(&mut copies[r.idx()], pair.right.raw())?;
                            }
                        }
                    }
                    copies[t.idx()] = mine;
                }
                Level::Leaf => {}
            }
        }
        gate.flush()?;
        Ok(Plan { order, levels, copies })
    }
}

impl Plan {
    /// The slot or index a side `x` at child level `c` names in the
    /// output: a label, a product, or a copy's rank among `c`'s copies.
    fn rank(&self, c: VtreeIdx, x: u32) -> u32 {
        match self.levels[c.idx()].as_ref().expect("a planned child") {
            Level::True => self.copies[c.idx()].binary_search(&x).expect("a named node of g is copied") as u32,
            Level::Leaf | Level::Listed(_) => x,
        }
    }

    /// The pairs of level `t`'s `k`th node in the output, in the children's
    /// ranks ([`rank`](Self::rank)), into `buf`: a product's or a copy's.
    fn pairs_into(&self, g: &Tdd, t: VtreeIdx, l: VtreeIdx, r: VtreeIdx, k: usize, buf: &mut SmallVec<[ChildPair; 8]>) {
        buf.clear();
        match self.levels[t.idx()].as_ref().expect("a planned level") {
            Level::True => {
                let y = self.copies[t.idx()][k];
                buf.extend(g.level(t).pairs_iter_of_idx(y as usize).map(|p| {
                    ChildPair::new(NodeIdx(self.rank(l, p.left.raw())), NodeIdx(self.rank(r, p.right.raw())))
                }));
            }
            Level::Listed(p) => {
                buf.extend(p.pairs(k).iter().map(|p| {
                    ChildPair::new(NodeIdx(self.rank(l, p.left.raw())), NodeIdx(self.rank(r, p.right.raw())))
                }));
            }
            Level::Leaf => {}
        }
    }

    /// The nodes level `t` emits: its products or its copies.
    fn width(&self, t: VtreeIdx) -> usize {
        match self.levels[t.idx()].as_ref().expect("a planned level") {
            Level::True => self.copies[t.idx()].len(),
            Level::Listed(p) => p.count(),
            Level::Leaf => LEAF_WIDTH,
        }
    }
}

/// Refuse operands an indexed walk does not read.
fn check_operands(f: &Tdd, g: &Tdd, index: &PairIndex, what: &'static str) -> Result<(), OperationError> {
    super::check_vtree(f, g)?;
    f.require_structure()?;
    g.require_structure()?;
    if f.weights.is_some() || g.weights.is_some() {
        return Err(OperationError::InertOption { option: what, needs: "unweighted operands" });
    }
    index.check(g)
}

/// [`Engine::and_indexed`] with the work its walk did.
pub(crate) fn and_indexed_counted(eng: &Engine, f: &Tdd, g: &Tdd, index: &PairIndex) -> Result<(Tdd, IndexedWork), OperationError> {
    let lim = eng.limits();
    let _op = lim.enter()?;
    check_operands(f, g, index, "and_indexed")?;
    if f.is_zero() || g.is_zero() {
        return Ok((crate::build::constant_like(eng, f, false)?, IndexedWork::default()));
    }
    let vtree = Arc::clone(f.vtree());
    let root = vtree.root();
    if vtree.node(root).is_leaf() {
        // One variable: nothing to index.
        return Ok((eng.and(f.clone(), g.clone())?, IndexedWork::default()));
    }
    let mut walk = Walk { lim, vtree: &vtree, f, g, index, work: IndexedWork::default() };
    let plan = walk.plan(root, vec![f.output().local.0], false)?;
    let work = walk.work;
    let Some(Level::Listed(top)) = &plan.levels[root.idx()] else { unreachable!("the top is listed") };
    let (from, to) = (top.start[f.output().local.idx()] as usize, top.start[f.output().local.idx() + 1] as usize);
    let b = g.output().local.0;
    let Ok(k) = top.items[from..to].binary_search_by_key(&b, |e| e.0) else {
        return Ok((crate::build::constant_like(eng, f, false)?, work));
    };
    let made = top.items[from + k].1 as usize;
    let mut out = Assembly::new(eng, &vtree)?;
    let mut ids: Vec<Vec<u32>> = (0..vtree.num_nodes()).map(|_| Vec::new()).collect();
    let mut buf: SmallVec<[ChildPair; 8]> = SmallVec::new();
    let mut gate = lim.gate();
    for &t in plan.order.iter().rev() {
        if vtree.node(t).is_leaf() {
            continue;
        }
        let (l, r) = vtree.children(t);
        let width = plan.width(t);
        let mut mine = Vec::new();
        lim.reserve_exact(&mut mine, width)?;
        for k in 0..width {
            plan.pairs_into(g, t, l, r, k, &mut buf);
            for p in buf.iter_mut() {
                let at = |c: VtreeIdx, x: u32| match vtree.node(c).is_leaf() {
                    true => x,
                    false => ids[c.idx()][x as usize],
                };
                *p = ChildPair::new(NodeIdx(at(l, p.left.raw())), NodeIdx(at(r, p.right.raw())));
            }
            crate::diagram::sort_pairs(&mut buf);
            gate.poll(buf.len() as u64)?;
            mine.push(out.push(eng, t, &buf)?.0);
        }
        ids[l.idx()] = Vec::new();
        ids[r.idx()] = Vec::new();
        ids[t.idx()] = mine;
    }
    gate.flush()?;
    let output = TddNodeId { vtree: root, local: NodeIdx(ids[root.idx()][made]) };
    Ok((out.finish(output)?, work))
}

/// [`Engine::sum_indexed`] with the work its walk did.
pub(crate) fn sum_indexed_counted<A: ColumnAlgebra>(
    eng: &Engine,
    f: &Tdd,
    g: &Tdd,
    index: &PairIndex,
    at: VtreeIdx,
    algebra: &A,
) -> Result<(A::Column, IndexedWork), OperationError> {
    let lim = eng.limits();
    let _op = lim.enter()?;
    check_operands(f, g, index, "sum_indexed")?;
    let vtree = Arc::clone(f.vtree());
    if at.idx() >= vtree.num_nodes() {
        return Err(OperationError::LevelNotInVtree(at));
    }
    if vtree.node(at).is_leaf() {
        return Err(OperationError::InertOption { option: "sum_indexed at a leaf", needs: "an internal vtree node" });
    }
    let width = f.reference_slot_count(at);
    let mut walk = Walk { lim, vtree: &vtree, f, g, index, work: IndexedWork::default() };
    if g.is_zero() {
        // `g` reaches no node at `at`: every node's value is the empty sum.
        let mut col = algebra.column(at, width);
        let none: [ChildPair; 0] = [];
        let (l, r) = vtree.children(at);
        for k in 0..width {
            let pairs = SlotPairs::new(PairsIter::slice(&none), InlineSlots::default(), InlineSlots::default());
            algebra.fold(at, k, pairs, &algebra.column(l, 0), &algebra.column(r, 0), &mut col);
        }
        return Ok((col, walk.work));
    }
    let plan = walk.plan(at, (0..width as u32).collect(), true)?;
    let work = walk.work;
    let mut cols: Vec<A::Column> = (0..vtree.num_nodes()).map(|_| A::Column::default()).collect();
    let mut buf: SmallVec<[ChildPair; 8]> = SmallVec::new();
    let mut gate = lim.gate();
    for &t in plan.order.iter().rev() {
        if vtree.node(t).is_leaf() {
            let mut col = algebra.column(t, LEAF_WIDTH);
            let var = vtree.leaf_var(t);
            for label in [LeafLabel::One, LeafLabel::Pos, LeafLabel::Neg] {
                algebra.leaf(t, var, label, &mut col);
            }
            cols[t.idx()] = col;
            continue;
        }
        let (l, r) = vtree.children(t);
        let n = plan.width(t);
        let mut col = algebra.column(t, n);
        for k in 0..n {
            plan.pairs_into(g, t, l, r, k, &mut buf);
            gate.poll(buf.len() as u64)?;
            let pairs = SlotPairs::new(PairsIter::slice(&buf), InlineSlots::default(), InlineSlots::default());
            algebra.fold(t, k, pairs, &cols[l.idx()], &cols[r.idx()], &mut col);
        }
        cols[l.idx()] = A::Column::default();
        cols[r.idx()] = A::Column::default();
        cols[t.idx()] = col;
    }
    gate.flush()?;
    Ok((std::mem::take(&mut cols[at.idx()]), work))
}

impl Engine {
    /// The conjunction `f ∧ g`, reading `g` only through `index`, a
    /// [`PairIndex`] built from `g`: where `f` selects a small part of a
    /// large `g`, the work follows `f` and the part of `g` it meets, not
    /// `g`'s size.
    ///
    /// The walk builds, level by level from the leaves up, the products of
    /// a node of `f` and a node of `g` whose conjunction is satisfiable,
    /// each found from the children's products through one probe of the
    /// index (the module notes give the details). Where `f` is true over a
    /// whole subtree, the nodes of `g` the result reaches there are copied,
    /// and only those. Both operands are only read; the result is
    /// deterministic and not minimized ([`Engine::minimize`] makes it
    /// canonical).
    ///
    /// # Work
    ///
    /// Always: `f`'s pairs are read once, and each live product costs at
    /// most the probes its pairs' sides combine to. The bound: let the
    /// leaves of a set of variables κ form one subtree of the vtree, with
    /// every vtree node outside that subtree a leaf or an ancestor of it
    /// (the other variables on a spine above κ), and let every variable
    /// outside κ that `f` or `g` depends on be determined by κ in `f` or in
    /// `g` (for instance `g` a column `a` keyed by κ, and `f` a set of
    /// keys). Then each internal vtree node holds at most `|keys(f)|` live
    /// products and copies and costs at most `9 · |keys(f)| + 81` index
    /// reads, where `keys(f)` is the set of assignments to κ that extend to
    /// a model of `f`: the work is `O(|f| + |keys(f)| · |vtree|)`, whatever
    /// the size of `g`. Without the precondition the result is the same
    /// function; only the bound is lost.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{Engine, Tdd, Vtree};
    /// use tididi::diagram::PairIndex;
    /// use tididi::vtree::VarId;
    ///
    /// let engine = Engine::new();
    /// // A two-bit value (x1, x2) above a three-bit key (x3, x4, x5), on a spine.
    /// let vtree = Arc::new(Vtree::linear(5));
    /// let vars: Vec<VarId> = (1..=5).map(VarId).collect();
    /// // g: one value per key, bit i of a row assigning vars[i].
    /// let rows: Vec<u64> = (0..8u64).map(|key| key << 2 | (key * 3) % 4).collect();
    /// let g = Tdd::from_models(&vtree, &vars, &rows)?;
    /// let index = PairIndex::new(&g)?;
    /// // f: two keys, any value.
    /// let f = engine.or(engine.cube(&vtree, [3, -4, -5])?, engine.cube(&vtree, [-3, 4, 5])?)?;
    /// let both = engine.and_indexed(&f, &g, &index)?;
    /// assert_eq!(both.model_count()?, 2u32.into());
    /// assert!(engine.equivalent(&both, &engine.and(f, g)?)?);
    /// # Ok::<(), tididi::OperationError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// [`OperationError::VtreeMismatch`] when `f`, `g` and the index do not
    /// share one vtree, [`OperationError::IndexMismatch`] when the index was
    /// built from another diagram than `g`, [`OperationError::MarginalLevel`]
    /// when either operand has a marginal level,
    /// [`OperationError::InertOption`] when either is weighted,
    /// [`OperationError::IndexOverflow`] when a level's products outgrow a
    /// 32-bit index, and the engine's refusals ([`OperationError::Stopped`],
    /// [`OperationError::OverBudget`]).
    pub fn and_indexed(&self, f: &Tdd, g: &Tdd, index: &PairIndex) -> Result<Tdd, OperationError> {
        and_indexed_counted(self, f, g, index).map(|(tdd, _)| tdd)
    }

    /// For each node of `f` at vtree node `at`, the value under `algebra` of
    /// that node conjoined with `∃Z. g`, `Z` the variables outside `at`'s
    /// subtree: the sum, over the assignments to the subtree's variables
    /// that the node holds and that extend to a model of `g`, of the
    /// product of their leaf values. `g` is read only through `index`, a
    /// [`PairIndex`] built from `g`, and `f ∧ g` is never built.
    ///
    /// Slot `i` of the column is node `i` of `f`'s level at `at`, one per
    /// node, as [`Tdd::evaluate_columns_at`] numbers them; a node that
    /// meets no model of `g` holds the empty sum. Where `g` depends on no
    /// variable outside `at`'s subtree, `∃Z. g` is `g`, and folding `f`'s
    /// levels above `at` over these values evaluates `f ∧ g`, which is how a
    /// grouped sum reads them: the groups above `at`, the summed columns
    /// under it. The walk below `at` is [`and_indexed`](Self::and_indexed)'s,
    /// with the same work bound, evaluating each product in `algebra`
    /// instead of building it; under the same laws as
    /// [`ColumnAlgebra`]'s, each level's column has one slot per product.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{Engine, Tdd, Vtree};
    /// use tididi::diagram::{ColumnAlgebra, LeafLabel, PairIndex, SlotPairs};
    /// use tididi::vtree::{VarId, VtreeIdx};
    ///
    /// // Model counts, one flat buffer per level.
    /// struct Counts;
    /// impl ColumnAlgebra for Counts {
    ///     type Column = Vec<u128>;
    ///     type Value = u128;
    ///     fn zero(&self) -> u128 { 0 }
    ///     fn column(&self, _: VtreeIdx, width: usize) -> Vec<u128> { vec![0; width] }
    ///     fn leaf(&self, _: VtreeIdx, _: VarId, label: LeafLabel, col: &mut Vec<u128>) {
    ///         col[label as usize] = if label == LeafLabel::One { 2 } else { 1 };
    ///     }
    ///     fn fold(&self, _: VtreeIdx, slot: usize, pairs: SlotPairs<'_>, left: &Vec<u128>, right: &Vec<u128>, out: &mut Vec<u128>) {
    ///         out[slot] = pairs.map(|(l, r)| left[l] * right[r]).sum();
    ///     }
    ///     fn read(&self, _: VtreeIdx, col: Vec<u128>, slot: usize) -> u128 { col[slot] }
    /// }
    ///
    /// let engine = Engine::new();
    /// // A group bit x1 above a two-bit value (x2, x3) and a two-bit key (x4, x5).
    /// let vtree = Arc::new(Vtree::linear(5));
    /// let (_, under) = vtree.children(vtree.root());
    /// let vars: Vec<VarId> = (2..=5).map(VarId).collect();
    /// // g: the value of each key; bit i of a row assigns vars[i].
    /// let g = Tdd::from_models(&vtree, &vars, &[0b00_01, 0b01_10, 0b10_10, 0b11_11])?;
    /// let index = PairIndex::new(&g)?;
    /// // f: keys 0 and 1 in group x1, keys 2 and 3 in group ¬x1.
    /// let f = engine.or(engine.cube(&vtree, [1, -4])?, engine.cube(&vtree, [-1, 4])?)?;
    /// let counts = engine.sum_indexed(&f, &g, &index, under, &Counts)?;
    /// // f's two nodes under the root, one per group, each meets two keys.
    /// assert_eq!(counts.len(), 2);
    /// assert!(counts.iter().all(|&c| c == 2));
    /// # Ok::<(), tididi::OperationError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// [`OperationError::LevelNotInVtree`] for an `at` outside the vtree,
    /// [`OperationError::InertOption`] for an `at` that is a leaf or for a
    /// weighted operand, and [`and_indexed`](Self::and_indexed)'s errors.
    ///
    /// # Panics
    ///
    /// Panics from the caller's algebra propagate.
    pub fn sum_indexed<A: ColumnAlgebra>(
        &self,
        f: &Tdd,
        g: &Tdd,
        index: &PairIndex,
        at: VtreeIdx,
        algebra: &A,
    ) -> Result<A::Column, OperationError> {
        sum_indexed_counted(self, f, g, index, at, algebra).map(|(col, _)| col)
    }
}
