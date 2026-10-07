//! The liveness walk over `f × care`.
//!
//! A *product* is a pair of references at one vtree level, one into each
//! operand: a node, or `None` where the operand is `⊤` there (not yet rooted,
//! or `care` marginal). It is *live* when some assignment satisfies both. The
//! walk keeps an `f`-pair when a live product it reaches from the root pairs
//! it with a `care` pair whose two child products are live; those are exactly
//! the pairs that the models of `f ∧ care` use.
//!
//! Below the lower of the two operand roots both operands hold nodes, and
//! whether a pair of nodes is live is decided bottom-up before the walk
//! ([`Products`]). A level whose two children are listed or leaves joins its
//! pairs on both, as the conjunction's sparse route does: for each `f` child on
//! one side, the `care` pairs whose child there is live with it are gathered by
//! their other child, and each `f` pair under it is looked up among them, so
//! every pair of pairs it meets is live. A level with one such child joins on
//! it and tests the other side, which a marginal operand decides at once. A
//! level whose join would cost too much, or that has no such child, is decided
//! pair by pair when the walk asks. The walk then enumerates a product's live
//! pairs from the listing, or for an undecided level through the same grouping. Above that root one
//! operand is `⊤`, so a product pairs one node with `⊤`: those are discovered,
//! decided bottom-up and marked top-down directly.

use rustc_hash::{FxHashMap, FxHashSet};

use crate::Engine;
use crate::limits::{OperationError, PollGate};

use crate::apply::CONJOIN_GRID;
use crate::apply::conjoin::budget::NO_PRODUCT;
use crate::diagram::{ChildDecoder, ChildPair, EncodedChildRef, NodeIdx, Tdd, LEAF_WIDTH, ZERO};
use crate::vtree::VtreeIdx;

use super::Marking;
use super::pairs::PairMarks;

/// One operand's reference into a vtree level: `None` = the operand is `⊤` here
/// (not yet rooted, or `care` marginal at this level), `Some(l)` = its node `l`.
type Ref = Option<NodeIdx>;
/// A product: `f`'s reference and `care`'s reference at one vtree level.
type Key = (Ref, Ref);
/// A level's live `(f node, care node, f pair, care pair)`, as a join finds them.
type Found = Vec<(u32, u32, u32, u32)>;
/// A `care` node's pairs keyed by their child on one side (see [`Walk::care_index`]).
type CareIndex = FxHashMap<(u32, u8), Vec<(u32, u32)>>;

/// What a child reference pair resolves to before it becomes a product.
enum Child {
    Dead,
    Live,
    Pair(Key),
}

/// Why the walk ended before its marks were complete.
enum Halt {
    /// The probe allowance is spent: the restriction is abandoned.
    Spent,
    /// An engine limit or cancellation.
    Error(OperationError),
}

impl From<OperationError> for Halt {
    fn from(e: OperationError) -> Self {
        Halt::Error(e)
    }
}

/// A pair of nodes with at most this many pairs of pairs is joined by trying
/// each; a wider one goes through an index on one child.
const GRID_PROBES: usize = 64;
/// A level is listed when its join takes at most this many probes, or
/// [`LIST_FACTOR`] times its two operands' pairs if that is more; otherwise
/// its products are decided when asked.
const LIST_PROBES: u64 = 1 << 24;
/// See [`LIST_PROBES`].
const LIST_FACTOR: u64 = 16;

/// The live products of one level below the lower root, where both operands
/// store nodes.
enum Products {
    /// A leaf, a level marginal in either operand, or one above the lower root.
    Absent,
    /// Every live pair of nodes, found by the bottom-up join.
    Listed(Listed),
    /// Decided pair by pair when asked, and remembered.
    Lazy(FxHashMap<(u32, u32), bool>),
}

/// The live pairs of nodes at one level, each with its live pairs of pairs.
struct Listed {
    /// `pack(a, b)` → the position of `(a, b)` in `products`.
    index: FxHashMap<u64, u32>,
    /// Every live `(a, b)`, ascending.
    products: Vec<(u32, u32)>,
    /// `products[offsets[a]..offsets[a + 1]]`: those of `f` node `a`.
    offsets: Vec<u32>,
    /// `combos[starts[p]..starts[p + 1]]`: product `p`'s live `(f pair, care
    /// pair)` indices, each with both child products live.
    starts: Vec<u32>,
    combos: Vec<(u32, u32)>,
}

impl Listed {
    fn contains(&self, a: u32, b: u32) -> bool {
        self.index.contains_key(&pack(a, b))
    }

    /// The products of `f` node `a`, by ascending `care` node.
    fn of(&self, a: u32) -> &[(u32, u32)] {
        &self.products[self.offsets[a as usize] as usize..self.offsets[a as usize + 1] as usize]
    }

    /// The live pairs of pairs of `(a, b)`; empty when it is dead.
    fn combos_of(&self, a: u32, b: u32) -> &[(u32, u32)] {
        match self.index.get(&pack(a, b)) {
            Some(&p) => &self.combos[self.starts[p as usize] as usize..self.starts[p as usize + 1] as usize],
            None => &[],
        }
    }
}

fn pack(a: u32, b: u32) -> u64 {
    (u64::from(a) << 32) | u64::from(b)
}

/// What a level's child on one side is, for joining the level's pairs on it.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Side {
    /// `f` marginal: live whatever `care` holds there.
    FMarginal,
    /// `care` marginal, a leaf or not: live where `f`'s node has a model.
    CareMarginal,
    /// A vtree leaf where both are structural: live unless the two labels
    /// conflict.
    Leaf,
    /// Both structural, products listed.
    Listed,
    /// Both structural, products decided when asked.
    Lazy,
}

impl Side {
    /// Whether the product this side hands down depends on the `care` pair,
    /// so every live `care` pair of an `f` pair has to be enumerated.
    fn keyed(self) -> bool {
        matches!(self, Side::Listed | Side::Lazy)
    }
}

/// `f`'s or `care`'s pairs at one level grouped by their child on one side:
/// group `x` is `entries[offsets[x]..offsets[x + 1]]`, each entry the parent
/// node, the pair's index in it and the pair's other side.
struct Groups {
    offsets: Vec<u32>,
    entries: Vec<(u32, u32, EncodedChildRef)>,
}

impl Groups {
    fn of(&self, x: u32) -> &[(u32, u32, EncodedChildRef)] {
        &self.entries[self.offsets[x as usize] as usize..self.offsets[x as usize + 1] as usize]
    }
}

/// The pairs walked at one level above the lower root, in discovery order,
/// with their liveness (filled bottom-up).
#[derive(Default)]
struct LevelPairs {
    index: FxHashMap<Key, u32>,
    keys: Vec<Key>,
    live: Vec<bool>,
    marked: Vec<bool>,
}

impl LevelPairs {
    /// Record `k` if new; true iff it was.
    fn push(&mut self, eng: &Engine, k: Key) -> Result<bool, OperationError> {
        if self.index.len() == self.index.capacity() { eng.limits().reserve_map(&mut self.index, 1)?; }
        let std::collections::hash_map::Entry::Vacant(slot) = self.index.entry(k) else {
            return Ok(false);
        };
        slot.insert(self.keys.len() as u32);
        eng.limits().try_push(&mut self.keys, k)?;
        eng.limits().try_push(&mut self.live, false)?;
        eng.limits().try_push(&mut self.marked, false)?;
        Ok(true)
    }
}

/// The marks being made, and the state of the walk that makes them.
struct Walk<'a> {
    eng: &'a Engine,
    f: &'a Tdd,
    care: &'a Tdd,
    /// Whether a vtree node lies in the lower operand root's subtree.
    inner: Vec<bool>,
    /// `sat[v][a]`: `f` node `a` at an inner level has a model.
    sat: Vec<Vec<bool>>,
    products: Vec<Products>,
    /// Per level, a `care` node's pairs keyed by their child on one side, as
    /// `(child, pair index)` ascending, for the nodes the walk joined on it.
    care_index: Vec<CareIndex>,
    remaining: u64,
    poll: PollGate<'a>,
    alive: Vec<Vec<bool>>,
    pair_alive: PairMarks,
}

impl Marking {
    /// Mark every `f`-node and `f`-pair that a model of `f ∧ care` uses, from
    /// vtree node `r` (a root of one operand).
    ///
    /// `remaining` is the allowance of product-pair probes across the
    /// bottom-up decision and the walk; spending it returns `None` with the
    /// marks abandoned.
    pub(super) fn walk(eng: &Engine, f: &Tdd, care: &Tdd, r: VtreeIdx, remaining: u64) -> Result<Option<Marking>, OperationError> {
        let root_key = match child(f, care, r, None, None) {
            Child::Pair(k) => k,
            // A compatible leaf or a marginal scalar: nothing died.
            Child::Live => return Marking::trivial(eng, f, true).map(Some),
            Child::Dead => return Marking::trivial(eng, f, false).map(Some),
        };
        let mut walk = Walk::new(eng, f, care, r, remaining)?;
        match walk.run(r, root_key) {
            Ok(root_live) => {
                walk.poll.flush()?;
                Ok(Some(Marking { alive: walk.alive, pair_alive: walk.pair_alive, root_live }))
            }
            Err(Halt::Spent) => {
                walk.poll.flush()?;
                Ok(None)
            }
            Err(Halt::Error(e)) => Err(e),
        }
    }

    /// Marks for a walk that never examined a pair: nothing dies (every reachable
    /// node is reported alive, so `nothing_reachable_died` holds).
    pub(super) fn trivial(eng: &Engine, f: &Tdd, root_live: bool) -> Result<Marking, OperationError> {
        Ok(Marking {
            alive: mark_rows(eng, f, true)?,
            pair_alive: PairMarks::all(),
            root_live,
        })
    }

    /// Is every node and pair reachable from `f`'s root marked live? Then the
    /// rebuild would reproduce `f` pair-for-pair, so `g == f` and the caller can
    /// reuse `f` verbatim. Stack-driven traversal of `f`'s reachable subgraph.
    pub(super) fn nothing_reachable_died(&self, eng: &Engine, f: &Tdd) -> Result<bool, OperationError> {
        let mut poll = eng.limits().gate();
        let vtree = &f.vtree;
        let mut seen = mark_rows(eng, f, false)?;
        let mut stack = Vec::new();
        eng.limits().try_push(&mut stack, (f.output.vtree, f.output.local))?;
        while let Some((v, l)) = stack.pop() {
            if l == ZERO || vtree.node(v).is_leaf() || f.levels[v.idx()].is_marginal() {
                continue;
            }
            if std::mem::replace(&mut seen[v.idx()][l.idx()], true) {
                continue;
            }
            if !self.alive[v.idx()][l.idx()] {
                return Ok(false);
            }
            let level = &f.levels[v.idx()];
            if !self.pair_alive.complete(v, l, level.pair_count_at(l.idx())) {
                return Ok(false);
            }
            let (lc, rc) = vtree.children(v);
            for p in level.pairs_iter_of_idx(l.idx()) {
                for (child, side) in [(lc, p.left), (rc, p.right)] {
                    let decoder = f.levels[child.idx()].child_decoder();
                    if !decoder.is_marginal() {
                        eng.limits().try_push(&mut stack, (child, decoder.node(side)))?;
                    }
                }
                poll.poll(1)?;
            }
        }
        poll.flush()?;
        Ok(true)
    }
}

impl<'a> Walk<'a> {
    fn new(eng: &'a Engine, f: &'a Tdd, care: &'a Tdd, r: VtreeIdx, remaining: u64) -> Result<Self, OperationError> {
        let lim = eng.limits();
        let vtree = &f.vtree;
        let nlev = vtree.num_nodes();
        let lower = if f.output.vtree == r { care.output.vtree } else { f.output.vtree };
        let mut inner = Vec::new();
        lim.try_resize(&mut inner, nlev, false)?;
        let mut stack = vec![lower];
        while let Some(v) = stack.pop() {
            inner[v.idx()] = true;
            if !vtree.node(v).is_leaf() {
                let (lv, rv) = vtree.children(v);
                lim.try_push(&mut stack, lv)?;
                lim.try_push(&mut stack, rv)?;
            }
        }
        let mut sat = Vec::new();
        lim.reserve_exact(&mut sat, nlev)?;
        sat.resize_with(nlev, Vec::new);
        let mut products = Vec::new();
        lim.reserve_exact(&mut products, nlev)?;
        products.resize_with(nlev, || Products::Absent);
        let mut care_index = Vec::new();
        lim.reserve_exact(&mut care_index, nlev)?;
        care_index.resize_with(nlev, FxHashMap::default);
        Ok(Walk {
            eng,
            f,
            care,
            inner,
            sat,
            products,
            care_index,
            remaining,
            poll: lim.gate(),
            alive: mark_rows(eng, f, false)?,
            pair_alive: PairMarks::new(eng, f)?,
        })
    }

    /// Decide the products below the lower root, then mark from the root.
    /// Returns whether the root product is live.
    fn run(&mut self, r: VtreeIdx, root_key: Key) -> Result<bool, Halt> {
        self.decide()?;
        let mut seeds = Vec::new();
        let root_live = if self.inner[r.idx()] {
            let live = self.key_live(r, root_key)?;
            if live {
                seeds.push((r, root_key));
            }
            live
        } else {
            self.top(r, root_key, &mut seeds)?
        };
        self.descend(seeds)?;
        Ok(root_live)
    }

    /// Take one probe from the allowance.
    #[inline]
    fn spend(&mut self) -> Result<(), Halt> {
        if self.remaining == 0 {
            return Err(Halt::Spent);
        }
        self.remaining -= 1;
        self.poll.poll(1)?;
        Ok(())
    }

    /// Resolve child references at `cv` (see [`child`]).
    #[inline]
    fn child(&self, cv: VtreeIdx, fo: Ref, co: Ref) -> Child {
        child(self.f, self.care, cv, fo, co)
    }

    /// The kind of child `cv` is, for a level below the lower root.
    fn side(&self, cv: VtreeIdx) -> Side {
        if self.f.levels[cv.idx()].is_marginal() {
            Side::FMarginal
        } else if self.care.levels[cv.idx()].is_marginal() {
            Side::CareMarginal
        } else if self.f.vtree.node(cv).is_leaf() {
            Side::Leaf
        } else if matches!(self.products[cv.idx()], Products::Listed(_)) {
            Side::Listed
        } else {
            Side::Lazy
        }
    }

    /// Decide `sat` and the products of every level below the lower root,
    /// children before parents.
    fn decide(&mut self) -> Result<(), Halt> {
        let (eng, f, care) = (self.eng, self.f, self.care);
        for (u, lc, rc) in f.vtree.internal_bottomup() {
            if !self.inner[u.idx()] || f.levels[u.idx()].is_marginal() {
                continue;
            }
            let level = &f.levels[u.idx()];
            let n = level.nodes().len();
            let mut row = Vec::new();
            eng.limits().try_resize(&mut row, n, false)?;
            for (a, slot) in row.iter_mut().enumerate() {
                *slot = level.pairs_iter_of_idx(a).any(|p| self.f_sat(lc, p.left) && self.f_sat(rc, p.right));
            }
            self.sat[u.idx()] = row;
            if care.levels[u.idx()].is_marginal() {
                continue;
            }
            self.products[u.idx()] = self.list(u, lc, rc)?;
        }
        Ok(())
    }

    /// Whether `f`'s side `side` into child `cv` has a model, `care` being `⊤`.
    fn f_sat(&self, cv: VtreeIdx, side: EncodedChildRef) -> bool {
        let decoder = self.f.levels[cv.idx()].child_decoder();
        if decoder.is_marginal() {
            return true;
        }
        let node = decoder.node(side);
        node != ZERO && (self.f.vtree.node(cv).is_leaf() || self.sat[cv.idx()][node.idx()])
    }

    /// The products of level `u`, where both operands are structural, by a
    /// join of its pairs on its children's live products: listed when some
    /// child is listed or a leaf and the join is cheap enough, else decided
    /// when asked.
    fn list(&mut self, u: VtreeIdx, lc: VtreeIdx, rc: VtreeIdx) -> Result<Products, Halt> {
        let (eng, f, care) = (self.eng, self.f, self.care);
        let kids = [lc, rc];
        let sides = [self.side(lc), self.side(rc)];
        let width = (f.levels[u.idx()].live_pairs() + care.levels[u.idx()].live_pairs()) as u64;
        let cap = LIST_PROBES.max(LIST_FACTOR.saturating_mul(width));
        let joinable = |k: Side| matches!(k, Side::Listed | Side::Leaf);
        let found = if joinable(sides[0]) && joinable(sides[1]) {
            self.join_both(u, kids, cap)?
        } else if let Some(s) = (0..2).find(|&s| joinable(sides[s])) {
            self.join_one(u, kids, s, sides[1 - s], cap)?
        } else {
            None
        };
        Ok(match found {
            Some(mut found) => {
                found.sort_unstable();
                Products::Listed(listed(eng, f.levels[u.idx()].nodes().len(), &found)?)
            }
            None => {
                Products::Lazy(FxHashMap::default())
            }
        })
    }

    /// The live products of joinable child `cs`, taken out of the walk until
    /// [`Self::restore`] puts them back: its listing, or the leaf table.
    fn take(&mut self, cs: VtreeIdx) -> Listed {
        if self.f.vtree.node(cs).is_leaf() {
            return leaf_live();
        }
        match std::mem::replace(&mut self.products[cs.idx()], Products::Absent) {
            Products::Listed(l) => l,
            _ => unreachable!("a joinable child is listed or a leaf"),
        }
    }

    /// Put back what [`Self::take`] took.
    fn restore(&mut self, cs: VtreeIdx, l: Listed) {
        if !self.f.vtree.node(cs).is_leaf() {
            self.products[cs.idx()] = Products::Listed(l);
        }
    }

    /// Join level `u`'s pairs on the live products of both children, each
    /// listed or a leaf. One child is the outer one: for each `f` child
    /// there, the `care` pairs whose child there is live with it are gathered
    /// by their inner child, and each `f` pair under it meets those whose
    /// inner child is live with its own. Every pair of pairs found is live,
    /// so the work is the gathering and the lookups; the outer child is the
    /// one where those cost less. `None` when even that exceeds `cap`.
    fn join_both(&mut self, u: VtreeIdx, kids: [VtreeIdx; 2], cap: u64) -> Result<Option<Found>, Halt> {
        let (eng, f, care) = (self.eng, self.f, self.care);
        let lives = [self.take(kids[0]), self.take(kids[1])];
        let mut best: Option<(usize, Groups, Groups, u64)> = None;
        for o in 0..2 {
            let fg = groups(eng, f, u, o, kids[o])?;
            let cg = groups(eng, care, u, o, kids[o])?;
            let cost = join_cost(&lives[o], &lives[1 - o], &fg, &cg);
            if best.as_ref().is_none_or(|b| cost < b.3) {
                best = Some((o, fg, cg, cost));
            }
        }
        let (o, fg, cg, cost) = best.expect("two orientations were costed");
        if cost > cap {
            let [l0, l1] = lives;
            self.restore(kids[0], l0);
            self.restore(kids[1], l1);
            return Ok(None);
        }
        let (outer, inner) = (&lives[o], &lives[1 - o]);
        let mut gathered: Vec<(u32, u32, u32)> = Vec::new();
        let mut found = Vec::new();
        for x in 0..fg.offsets.len() as u32 - 1 {
            let fpairs = fg.of(x);
            if fpairs.is_empty() {
                continue;
            }
            gathered.clear();
            for &(_, y) in outer.of(x) {
                for &(b, j, cin) in cg.of(y) {
                    self.spend()?;
                    let yi = ChildDecoder::structural().node(cin);
                    if yi != ZERO {
                        eng.limits().try_push(&mut gathered, (yi.0, b, j))?;
                    }
                }
            }
            if gathered.is_empty() {
                continue;
            }
            gathered.sort_unstable();
            for &(a, i, fin) in fpairs {
                let xi = ChildDecoder::structural().node(fin);
                if xi == ZERO {
                    continue;
                }
                let partners = inner.of(xi.0);
                if partners.len() < gathered.len() {
                    for &(_, y) in partners {
                        self.spend()?;
                        let lo = gathered.partition_point(|e| e.0 < y);
                        for &(_, b, j) in gathered[lo..].iter().take_while(|e| e.0 == y) {
                            eng.limits().try_push(&mut found, (a, b, i, j))?;
                        }
                    }
                } else {
                    for &(y, b, j) in &gathered {
                        self.spend()?;
                        if inner.contains(xi.0, y) {
                            eng.limits().try_push(&mut found, (a, b, i, j))?;
                        }
                    }
                }
            }
        }
        let [l0, l1] = lives;
        self.restore(kids[0], l0);
        self.restore(kids[1], l1);
        Ok(Some(found))
    }

    /// Join level `u`'s pairs on the live products of child `kids[s]`, listed
    /// or a leaf, testing each pair of pairs on the other child: one marginal
    /// in an operand decides by one side's pair alone, a lazy one asks. `None`
    /// when a lazy test would take more than `cap` probes.
    fn join_one(&mut self, u: VtreeIdx, kids: [VtreeIdx; 2], s: usize, other_kind: Side, cap: u64) -> Result<Option<Found>, Halt> {
        let (eng, f, care) = (self.eng, self.f, self.care);
        let (cs, other) = (kids[s], kids[1 - s]);
        let live = self.take(cs);
        let fg = groups(eng, f, u, s, cs)?;
        let cg = groups(eng, care, u, s, cs)?;
        if other_kind == Side::Lazy {
            let mut est = 0u64;
            for &(x, y) in &live.products {
                est = est.saturating_add((fg.of(x).len() as u64).saturating_mul(cg.of(y).len() as u64));
            }
            if est > cap {
                self.restore(cs, live);
                return Ok(None);
            }
        }
        let mut found = Vec::new();
        for &(x, y) in &live.products {
            let cands = cg.of(y);
            if cands.is_empty() {
                continue;
            }
            for &(a, i, fside) in fg.of(x) {
                let fo = decode(f, other, fside);
                if other_kind == Side::CareMarginal && !self.live(other, fo, None)? {
                    continue;
                }
                for &(b, j, cside) in cands {
                    self.spend()?;
                    let co = decode(care, other, cside);
                    let ok = match other_kind {
                        Side::CareMarginal => true,
                        Side::FMarginal => co != Some(ZERO),
                        _ => self.live(other, fo, co)?,
                    };
                    if ok {
                        eng.limits().try_push(&mut found, (a, b, i, j))?;
                    }
                }
            }
        }
        self.restore(cs, live);
        Ok(Some(found))
    }

    /// Whether child references `(fo, co)` at child level `cv` lead to a model.
    fn live(&mut self, cv: VtreeIdx, fo: Ref, co: Ref) -> Result<bool, Halt> {
        match self.child(cv, fo, co) {
            Child::Dead => Ok(false),
            Child::Live => Ok(true),
            Child::Pair(k) => self.key_live(cv, k),
        }
    }

    /// Whether product `k` at a level below the lower root is live.
    fn key_live(&mut self, v: VtreeIdx, k: Key) -> Result<bool, Halt> {
        match k {
            (Some(a), Some(b)) => self.pair_live(v, a.0, b.0),
            (Some(a), None) => Ok(self.sat[v.idx()][a.idx()]),
            _ => unreachable!("below the lower root both operands are rooted"),
        }
    }

    /// Whether `f` node `a` and `care` node `b` at `v` share a model.
    fn pair_live(&mut self, v: VtreeIdx, a: u32, b: u32) -> Result<bool, Halt> {
        match &self.products[v.idx()] {
            Products::Listed(l) => return Ok(l.contains(a, b)),
            Products::Lazy(memo) => {
                if let Some(&x) = memo.get(&(a, b)) {
                    return Ok(x);
                }
            }
            Products::Absent => unreachable!("a pair of nodes at a level without products"),
        }
        let mut one = Vec::new();
        self.combos(v, a, b, true, true, &mut one)?;
        let x = !one.is_empty();
        let eng = self.eng;
        if let Products::Lazy(memo) = &mut self.products[v.idx()] {
            if memo.len() == memo.capacity() { eng.limits().reserve_map(memo, 1)?; }
            memo.insert((a, b), x);
        }
        Ok(x)
    }

    /// The live pairs of `f` node `a` × `care` node `b` at `v`, as `(f pair,
    /// care pair)` indices, into `out`: only the first with `first`, and one
    /// `care` pair per `f` pair with `one_each`.
    fn combos(&mut self, v: VtreeIdx, a: u32, b: u32, first: bool, one_each: bool, out: &mut Vec<(u32, u32)>) -> Result<(), Halt> {
        let (eng, f, care) = (self.eng, self.f, self.care);
        let (mut fbuf, mut cbuf) = (Vec::new(), Vec::new());
        let fp = f.levels[v.idx()].pairs_read(a as usize, &mut fbuf);
        let cp = care.levels[v.idx()].pairs_read(b as usize, &mut cbuf);
        let (lc, rc) = f.vtree.children(v);
        let kids = [lc, rc];
        let sides = [self.side(lc), self.side(rc)];
        let pick = if fp.len().saturating_mul(cp.len()) <= GRID_PROBES {
            None
        } else {
            self.index_side(kids, sides)
        };
        let Some(s) = pick else {
            'f: for (i, p) in fp.iter().enumerate() {
                let (fl, fr) = (decode(f, lc, p.left), decode(f, rc, p.right));
                for (j, q) in cp.iter().enumerate() {
                    self.spend()?;
                    if self.live(lc, fl, decode(care, lc, q.left))? && self.live(rc, fr, decode(care, rc, q.right))? {
                        eng.limits().try_push(out, (i as u32, j as u32))?;
                        if first {
                            return Ok(());
                        }
                        if one_each {
                            continue 'f;
                        }
                    }
                }
            }
            return Ok(());
        };
        let (cs, other) = (kids[s], kids[1 - s]);
        let mut cands: Vec<u32> = Vec::new();
        if sides[s] == Side::Leaf {
            let mut by_label: [Vec<u32>; LEAF_WIDTH] = Default::default();
            for (j, q) in cp.iter().enumerate() {
                let y = ChildDecoder::structural().node(side_of(*q, s));
                if y != ZERO {
                    eng.limits().try_push(&mut by_label[y.idx()], j as u32)?;
                }
            }
            for (i, p) in fp.iter().enumerate() {
                let x = ChildDecoder::structural().node(side_of(*p, s));
                if x == ZERO {
                    continue;
                }
                cands.clear();
                for (y, js) in by_label.iter().enumerate() {
                    if !leaf_dead(Some(x), Some(NodeIdx(y as u32))) {
                        cands.extend_from_slice(js);
                    }
                }
                if self.probe(other, s, *p, cp, &cands, i as u32, first, one_each, out)? {
                    return Ok(());
                }
            }
            return Ok(());
        }
        let held = std::mem::replace(&mut self.products[cs.idx()], Products::Absent);
        let Products::Listed(below) = &held else { unreachable!("an index side is listed") };
        let index = match self.care_index[v.idx()].remove(&(b, s as u8)) {
            Some(index) => index,
            None => {
                let mut index = Vec::new();
                eng.limits().reserve_exact(&mut index, cp.len())?;
                for (j, q) in cp.iter().enumerate() {
                    let y = ChildDecoder::structural().node(side_of(*q, s));
                    if y != ZERO {
                        index.push((y.0, j as u32));
                    }
                }
                index.sort_unstable();
                index
            }
        };
        for (i, p) in fp.iter().enumerate() {
            let x = ChildDecoder::structural().node(side_of(*p, s));
            if x == ZERO {
                continue;
            }
            cands.clear();
            let partners = below.of(x.0);
            if partners.len().saturating_mul(8) < index.len() {
                for &(_, y) in partners {
                    let lo = index.partition_point(|e| e.0 < y);
                    cands.extend(index[lo..].iter().take_while(|e| e.0 == y).map(|e| e.1));
                }
            } else if index.len().saturating_mul(8) < partners.len() {
                cands.extend(index.iter().filter(|e| below.contains(x.0, e.0)).map(|e| e.1));
            } else {
                let (mut m, mut n) = (0, 0);
                while m < partners.len() && n < index.len() {
                    match partners[m].1.cmp(&index[n].0) {
                        std::cmp::Ordering::Less => m += 1,
                        std::cmp::Ordering::Greater => n += 1,
                        std::cmp::Ordering::Equal => {
                            cands.push(index[n].1);
                            n += 1;
                        }
                    }
                }
            }
            if self.probe(other, s, *p, cp, &cands, i as u32, first, one_each, out)? {
                break;
            }
        }
        self.products[cs.idx()] = held;
        if self.care_index[v.idx()].len() == self.care_index[v.idx()].capacity() {
            eng.limits().reserve_map(&mut self.care_index[v.idx()], 1)?;
        }
        self.care_index[v.idx()].insert((b, s as u8), index);
        Ok(())
    }

    /// Try `care` pairs `cands` against `f` pair `p` (index `i`) on the side
    /// other than `s`, whose child is `other`; the side `s` is live for each.
    /// Returns whether `first` was met.
    #[expect(clippy::too_many_arguments)]
    fn probe(&mut self, other: VtreeIdx, s: usize, p: ChildPair, cp: &[ChildPair], cands: &[u32], i: u32,
             first: bool, one_each: bool, out: &mut Vec<(u32, u32)>) -> Result<bool, Halt> {
        let (f, care) = (self.f, self.care);
        let fo = decode(f, other, side_of(p, 1 - s));
        for &j in cands {
            self.spend()?;
            let co = decode(care, other, side_of(cp[j as usize], 1 - s));
            if self.live(other, fo, co)? {
                self.eng.limits().try_push(out, (i, j))?;
                if first {
                    return Ok(true);
                }
                if one_each {
                    return Ok(false);
                }
            }
        }
        Ok(false)
    }

    /// The child to join a pair of nodes at a level with children `kids` on:
    /// a listed one, the one with fewer partners per node if both are, else a
    /// leaf; `None` for a grid.
    fn index_side(&self, kids: [VtreeIdx; 2], sides: [Side; 2]) -> Option<usize> {
        let density = |s: usize| match &self.products[kids[s].idx()] {
            Products::Listed(l) => l.products.len() as f64 / (l.offsets.len().max(2) - 1) as f64,
            _ => f64::INFINITY,
        };
        match sides {
            [Side::Listed, Side::Listed] => Some(usize::from(density(1) < density(0))),
            [Side::Listed, _] => Some(0),
            [_, Side::Listed] => Some(1),
            [Side::Leaf, _] => Some(0),
            [_, Side::Leaf] => Some(1),
            _ => None,
        }
    }

    /// Above the lower root: discover the products the root reaches, decide
    /// them bottom-up, and mark top-down along live pairs from the root.
    /// Returns whether the root is live; the products below the lower root
    /// that live pairs reach are left in `seeds`.
    fn top(&mut self, r: VtreeIdx, root_key: Key, seeds: &mut Vec<(VtreeIdx, Key)>) -> Result<bool, Halt> {
        let (eng, f, care) = (self.eng, self.f, self.care);
        let vtree = &f.vtree;
        let mut levels: Vec<LevelPairs> = Vec::new();
        eng.limits().reserve_exact(&mut levels, vtree.num_nodes())?;
        levels.resize_with(vtree.num_nodes(), LevelPairs::default);
        levels[r.idx()].push(eng, root_key)?;
        let mut stack = Vec::new();
        eng.limits().try_push(&mut stack, (r, root_key))?;
        while let Some((v, (fo, co))) = stack.pop() {
            let (lc, rc) = vtree.children(v);
            for (fl, fr) in refs(f, v, fo) {
                for (cl, cr) in refs(care, v, co) {
                    self.spend()?;
                    for (cv, x, y) in [(lc, fl, cl), (rc, fr, cr)] {
                        if let Child::Pair(k) = self.child(cv, x, y)
                            && !self.inner[cv.idx()]
                            && levels[cv.idx()].push(eng, k)? {
                                eng.limits().try_push(&mut stack, (cv, k))?;
                            }
                    }
                }
            }
        }
        for (v, lc, rc) in vtree.internal_bottomup() {
            if self.inner[v.idx()] {
                continue;
            }
            for i in 0..levels[v.idx()].keys.len() {
                let (fo, co) = levels[v.idx()].keys[i];
                let mut any = false;
                'pairs: for (fl, fr) in refs(f, v, fo) {
                    for (cl, cr) in refs(care, v, co) {
                        self.spend()?;
                        if self.top_live(&levels, lc, fl, cl)? && self.top_live(&levels, rc, fr, cr)? {
                            any = true;
                            break 'pairs;
                        }
                    }
                }
                levels[v.idx()].live[i] = any;
            }
        }
        let root_live = levels[r.idx()].live[0];
        if !root_live {
            return Ok(false);
        }
        levels[r.idx()].marked[0] = true;
        eng.limits().try_push(&mut stack, (r, root_key))?;
        while let Some((v, (fo, co))) = stack.pop() {
            let (lc, rc) = vtree.children(v);
            for (k, (fl, fr)) in refs(f, v, fo).enumerate() {
                for (cl, cr) in refs(care, v, co) {
                    self.spend()?;
                    if !(self.top_live(&levels, lc, fl, cl)? && self.top_live(&levels, rc, fr, cr)?) {
                        continue;
                    }
                    if let Some(a) = fo {
                        self.mark(v, a, k)?;
                    }
                    for (cv, x, y) in [(lc, fl, cl), (rc, fr, cr)] {
                        let Child::Pair(key) = self.child(cv, x, y) else { continue };
                        if self.inner[cv.idx()] {
                            eng.limits().try_push(seeds, (cv, key))?;
                        } else {
                            let at = levels[cv.idx()].index[&key] as usize;
                            if !std::mem::replace(&mut levels[cv.idx()].marked[at], true) {
                                eng.limits().try_push(&mut stack, (cv, key))?;
                            }
                        }
                    }
                }
            }
        }
        Ok(true)
    }

    /// Liveness of child references at `cv` for the walk above the lower root:
    /// from `levels` above it, from the decided products below.
    fn top_live(&mut self, levels: &[LevelPairs], cv: VtreeIdx, fo: Ref, co: Ref) -> Result<bool, Halt> {
        match self.child(cv, fo, co) {
            Child::Dead => Ok(false),
            Child::Live => Ok(true),
            Child::Pair(k) if self.inner[cv.idx()] => self.key_live(cv, k),
            Child::Pair(k) => {
                let lp = &levels[cv.idx()];
                Ok(lp.live[lp.index[&k] as usize])
            }
        }
    }

    /// Walk the live products below the lower root from `seeds`, each live,
    /// and mark the `f` pairs of their live pairs.
    fn descend(&mut self, seeds: Vec<(VtreeIdx, Key)>) -> Result<(), Halt> {
        let (eng, f, care) = (self.eng, self.f, self.care);
        let vtree = &f.vtree;
        let mut seen: Vec<FxHashSet<Key>> = Vec::new();
        eng.limits().reserve_exact(&mut seen, vtree.num_nodes())?;
        seen.resize_with(vtree.num_nodes(), FxHashSet::default);
        let mut stack = Vec::new();
        for (v, k) in seeds {
            if visit(eng, &mut seen[v.idx()], k)? {
                eng.limits().try_push(&mut stack, (v, k))?;
            }
        }
        let mut out = Vec::new();
        let (mut fbuf, mut cbuf) = (Vec::new(), Vec::new());
        while let Some((v, (fo, co))) = stack.pop() {
            let a = fo.expect("below the lower root f is rooted");
            let (lc, rc) = vtree.children(v);
            let fp = f.levels[v.idx()].pairs_read(a.idx(), &mut fbuf);
            let cp = match co {
                Some(b) => care.levels[v.idx()].pairs_read(b.idx(), &mut cbuf),
                None => &[],
            };
            out.clear();
            match co {
                Some(b) => match &self.products[v.idx()] {
                    Products::Listed(l) => {
                        let combos = l.combos_of(a.0, b.0);
                        eng.limits().reserve(&mut out, combos.len())?;
                        out.extend_from_slice(combos);
                    }
                    _ => {
                        let one_each = !(self.side(lc).keyed() || self.side(rc).keyed());
                        self.combos(v, a.0, b.0, false, one_each, &mut out)?;
                    }
                },
                None => {
                    for (i, p) in fp.iter().enumerate() {
                        self.spend()?;
                        if self.live(lc, decode(f, lc, p.left), None)? && self.live(rc, decode(f, rc, p.right), None)? {
                            eng.limits().try_push(&mut out, (i as u32, 0))?;
                        }
                    }
                }
            }
            for &(i, j) in &out {
                self.mark(v, a, i as usize)?;
                let p = fp[i as usize];
                let (cl, cr) = match co {
                    Some(_) => {
                        let q = cp[j as usize];
                        (decode(care, lc, q.left), decode(care, rc, q.right))
                    }
                    None => (None, None),
                };
                for (cv, x, y) in [(lc, decode(f, lc, p.left), cl), (rc, decode(f, rc, p.right), cr)] {
                    if let Child::Pair(k) = self.child(cv, x, y)
                        && visit(eng, &mut seen[cv.idx()], k)? {
                            eng.limits().try_push(&mut stack, (cv, k))?;
                        }
                }
            }
        }
        Ok(())
    }

    /// Mark `f` pair `k` of node `a` at `v`, and the node, live.
    fn mark(&mut self, v: VtreeIdx, a: NodeIdx, k: usize) -> Result<(), OperationError> {
        self.alive[v.idx()][a.idx()] = true;
        let count = self.f.levels[v.idx()].pair_count_at(a.idx());
        self.pair_alive.mark(self.eng, v, a, k, count)
    }
}

/// Insert `k` into `seen`; true iff it was new.
fn visit(eng: &Engine, seen: &mut FxHashSet<Key>, k: Key) -> Result<bool, OperationError> {
    if seen.len() == seen.capacity() { eng.limits().reserve_set(seen, 1)?; }
    Ok(seen.insert(k))
}

/// `f`'s or `care`'s pairs at level `u` grouped by their child on side `s`
/// (`0` left, `1` right), which is level `cs`; a `ZERO` child is left out.
fn groups(eng: &Engine, t: &Tdd, u: VtreeIdx, s: usize, cs: VtreeIdx) -> Result<Groups, OperationError> {
    let keys = if t.vtree.node(cs).is_leaf() { LEAF_WIDTH } else { t.levels[cs.idx()].nodes().len() };
    let level = &t.levels[u.idx()];
    let mut offsets = Vec::new();
    eng.limits().try_resize(&mut offsets, keys + 1, 0u32)?;
    let mut total = 0usize;
    for a in 0..level.nodes().len() {
        for p in level.pairs_iter_of_idx(a) {
            let x = ChildDecoder::structural().node(side_of(p, s));
            if x != ZERO {
                offsets[x.idx() + 1] += 1;
                total += 1;
            }
        }
    }
    for x in 0..keys {
        offsets[x + 1] += offsets[x];
    }
    let mut entries = Vec::new();
    eng.limits().try_resize(&mut entries, total, (0u32, 0u32, EncodedChildRef::from_raw(0)))?;
    let mut cursor = offsets.clone();
    for a in 0..level.nodes().len() {
        for (i, p) in level.pairs_iter_of_idx(a).enumerate() {
            let x = ChildDecoder::structural().node(side_of(p, s));
            if x != ZERO {
                entries[cursor[x.idx()] as usize] = (a as u32, i as u32, side_of(p, 1 - s));
                cursor[x.idx()] += 1;
            }
        }
    }
    Ok(Groups { offsets, entries })
}

/// What [`Walk::join_both`] does with `outer` as the outer child: per `f`
/// child there, gather the `care` pairs live with it, then look each `f` pair
/// under it up among them by its inner child.
fn join_cost(outer: &Listed, inner: &Listed, fg: &Groups, cg: &Groups) -> u64 {
    let mut cost = 0u64;
    for x in 0..fg.offsets.len() as u32 - 1 {
        let fpairs = fg.of(x);
        if fpairs.is_empty() {
            continue;
        }
        let gathered: u64 = outer.of(x).iter().map(|&(_, y)| cg.of(y).len() as u64).sum();
        cost = cost.saturating_add(gathered);
        if gathered == 0 {
            continue;
        }
        for &(_, _, fin) in fpairs {
            let xi = ChildDecoder::structural().node(fin);
            if xi != ZERO {
                cost = cost.saturating_add((inner.of(xi.0).len() as u64).min(gathered));
            }
        }
    }
    cost
}

/// The live products of a leaf: every pair of labels that does not conflict.
fn leaf_live() -> Listed {
    let mut products = Vec::new();
    let mut offsets = vec![0u32; LEAF_WIDTH + 1];
    for x in 0..LEAF_WIDTH as u32 {
        for y in 0..LEAF_WIDTH as u32 {
            if !leaf_dead(Some(NodeIdx(x)), Some(NodeIdx(y))) {
                products.push((x, y));
                offsets[x as usize + 1] += 1;
            }
        }
    }
    for x in 0..LEAF_WIDTH {
        offsets[x + 1] += offsets[x];
    }
    let index = products.iter().enumerate().map(|(p, &(x, y))| (pack(x, y), p as u32)).collect();
    Listed { index, products, offsets, starts: Vec::new(), combos: Vec::new() }
}

/// The listing of a level of `nodes` `f` nodes from its live `(a, b, f pair,
/// care pair)`, ascending.
fn listed(eng: &Engine, nodes: usize, found: &[(u32, u32, u32, u32)]) -> Result<Listed, OperationError> {
    let lim = eng.limits();
    let mut products = Vec::new();
    let mut starts = Vec::new();
    let mut combos = Vec::new();
    lim.reserve_exact(&mut combos, found.len())?;
    let mut offsets = Vec::new();
    lim.try_resize(&mut offsets, nodes + 1, 0u32)?;
    for (n, &(a, b, i, j)) in found.iter().enumerate() {
        if n == 0 || (found[n - 1].0, found[n - 1].1) != (a, b) {
            lim.try_push(&mut products, (a, b))?;
            lim.try_push(&mut starts, combos.len() as u32)?;
            offsets[a as usize + 1] += 1;
        }
        combos.push((i, j));
    }
    lim.try_push(&mut starts, combos.len() as u32)?;
    for k in 0..nodes {
        offsets[k + 1] += offsets[k];
    }
    let mut index = FxHashMap::default();
    lim.reserve_map(&mut index, products.len())?;
    for (p, &(a, b)) in products.iter().enumerate() {
        index.insert(pack(a, b), p as u32);
    }
    Ok(Listed { index, products, offsets, starts, combos })
}

/// Side `s` of a pair: `0` the left, `1` the right.
#[inline]
fn side_of(p: ChildPair, s: usize) -> EncodedChildRef {
    if s == 0 { p.left } else { p.right }
}

/// The reference a pair side of `t` makes into child level `cv`: `None` toward
/// a marginal level.
#[inline]
fn decode(t: &Tdd, cv: VtreeIdx, side: EncodedChildRef) -> Ref {
    let decoder = t.levels[cv.idx()].child_decoder();
    if decoder.is_marginal() { None } else { Some(decoder.node(side)) }
}

/// Resolve the references `(fo, co)` that a parent product hands to child
/// level `cv`: root an operand whose root is `cv`, then apply the terminal
/// rules (`ZERO` dead; `f` marginal live; `care` marginal ⊤; leaf table), else
/// it is a product.
fn child(f: &Tdd, care: &Tdd, cv: VtreeIdx, fo: Ref, co: Ref) -> Child {
    let fo = if fo.is_none() && cv == f.output.vtree { Some(f.output.local) } else { fo };
    let co = if co.is_none() && cv == care.output.vtree { Some(care.output.local) } else { co };
    if fo == Some(ZERO) || co == Some(ZERO) {
        return Child::Dead;
    }
    if f.levels[cv.idx()].is_marginal() {
        // A marginal f level is a count (> 0), never a node: always live.
        return Child::Live;
    }
    // A marginal care level is a satisfiability indicator: ⊤ for liveness.
    let co = if care.levels[cv.idx()].is_marginal() { None } else { co };
    if f.vtree.node(cv).is_leaf() {
        return if leaf_dead(fo, co) { Child::Dead } else { Child::Live };
    }
    Child::Pair((fo, co))
}

/// Whether two leaf references conflict: only `{Pos, Neg}` does, and `⊤`
/// (`None`) is a wildcard. `ZERO` never reaches here.
fn leaf_dead(fo: Ref, co: Ref) -> bool {
    match (fo, co) {
        (Some(a), Some(b)) => {
            debug_assert!(a.idx() < LEAF_WIDTH && b.idx() < LEAF_WIDTH, "a leaf reference names a label");
            CONJOIN_GRID[a.idx()][b.idx()] == NO_PRODUCT
        }
        _ => false,
    }
}

/// The (left, right) child references of one operand at level `v`: its node's
/// pairs, or the single `(⊤, ⊤)` pair when the operand is `⊤` there.
fn refs(t: &Tdd, v: VtreeIdx, o: Ref) -> impl Iterator<Item = (Ref, Ref)> + '_ {
    let pairs = o.map(|l| t.levels[v.idx()].pairs_iter_of_idx(l.idx()));
    let top = o.is_none();
    let (left, right) = t.vtree.children(v);
    pairs
        .into_iter()
        .flatten()
        .map(move |p| (decode(t, left, p.left), decode(t, right, p.right)))
        .chain(std::iter::once((None, None)).filter(move |_| top))
}

/// Allocate one initialized marking row per diagram level through the engine.
fn mark_rows<T: Clone>(eng: &Engine, f: &Tdd, value: T) -> Result<Vec<Vec<T>>, OperationError> {
    let mut rows = Vec::new();
    eng.limits().reserve_exact(&mut rows, f.levels.len())?;
    for level in &f.levels {
        let mut row = Vec::new();
        eng.limits().try_resize(&mut row, level.nodes().len(), value.clone())?;
        rows.push(row);
    }
    Ok(rows)
}
