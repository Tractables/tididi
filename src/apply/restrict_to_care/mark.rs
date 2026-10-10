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
//! decided bottom-up and marked top-down directly, except an `f` node beside
//! a `care` that is `⊤` over its whole subtree, which is kept whole unwalked.
//! A listed product's position is found in its `f` node's row of the
//! listing, never hashed, and marks it visited by a bit.
//! Where each `f` node has at most one live product, the listings are the
//! marks and nothing is walked down ([`Walk::functional`]); a level where
//! `care` holds one node of few pairs is joined pair by pair in storage
//! order ([`Walk::join_spine`]), and one whose `f` nodes have one live
//! product each at its children by looking each `f` pair's partners up
//! ([`Walk::join_paired`], [`Walk::join_keyed`]).

use rustc_hash::{FxHashMap, FxHashSet};

use crate::Engine;
use crate::limits::{OperationError, PollGate};

use crate::apply::CONJOIN_GRID;
use crate::apply::conjoin::budget::NO_PRODUCT;
use crate::diagram::{ChildDecoder, ChildPair, EncodedChildRef, NodeIdx, Tdd, LEAF_WIDTH, ONE_LEAF_IDX, ZERO};
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

/// The two passes of [`Walk::path_top`]: liveness up the path, marks down.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Pass {
    Up,
    Down,
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
/// A level where `care` holds one node of at most this many pairs is joined
/// by trying each `f` pair against each of them ([`Walk::join_spine`]).
const SPINE_PAIRS: usize = 8;

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
/// A product is found by its position in `products`, from `f`'s node's
/// run there ([`Listed::position`]): no table keyed by the pair, so the
/// walk's lookups and its marks of what it has visited are arrays.
struct Listed {
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
    /// The position of product `(a, b)` in `products`, where it is live:
    /// a search in `f` node `a`'s run, which is short.
    fn position(&self, a: u32, b: u32) -> Option<u32> {
        let (lo, hi) = (self.offsets[a as usize] as usize, self.offsets[a as usize + 1] as usize);
        let run = &self.products[lo..hi];
        let at = match run.len() {
            0..=8 => run.iter().position(|e| e.1 == b)?,
            _ => run.binary_search_by_key(&b, |e| e.1).ok()?,
        };
        Some((lo + at) as u32)
    }

    fn contains(&self, a: u32, b: u32) -> bool {
        self.position(a, b).is_some()
    }

    /// The products of `f` node `a`, by ascending `care` node.
    fn of(&self, a: u32) -> &[(u32, u32)] {
        &self.products[self.offsets[a as usize] as usize..self.offsets[a as usize + 1] as usize]
    }

    /// The live pairs of pairs of `(a, b)`; empty when it is dead.
    fn combos_of(&self, a: u32, b: u32) -> &[(u32, u32)] {
        match self.position(a, b) {
            Some(p) => self.combos_at(p),
            None => &[],
        }
    }

    /// The live pairs of pairs of the product at position `p`.
    fn combos_at(&self, p: u32) -> &[(u32, u32)] {
        &self.combos[self.starts[p as usize] as usize..self.starts[p as usize + 1] as usize]
    }
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

/// What [`Walk::join_spine`] asks of an `f` pair's side for one `care`
/// pair ([`Walk::side_test`]).
enum SideTest {
    /// `care`'s side is `ZERO`: no `f` pair is live with it.
    Dead,
    /// `f` is marginal there: a count, always live.
    Live,
    /// `care` is `⊤` there: live where `f`'s child is a node.
    Model,
    /// A leaf: live where `f`'s label does not conflict with `care`'s.
    Leaf([bool; LEAF_WIDTH]),
    /// A live product with `care`'s node: a bit per `f` node.
    Bits(Vec<u64>),
}

impl SideTest {
    /// Whether `f`'s side `x` (`None` toward a marginal level) passes.
    #[inline]
    fn holds(&self, x: Ref) -> bool {
        match (self, x) {
            (SideTest::Dead, _) => false,
            (SideTest::Live, _) | (_, None) => true,
            (_, Some(x)) if x == ZERO => false,
            (SideTest::Model, _) => true,
            (SideTest::Leaf(table), Some(x)) => table[x.idx()],
            (SideTest::Bits(bits), Some(x)) => bits[x.idx() / 64] & (1 << (x.idx() % 64)) != 0,
        }
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
    /// A product of an `f` node and a `care` not yet rooted, by the node:
    /// its position plus one, `0` for none. Allocated at the first.
    by_node: Vec<u32>,
    /// Every other product's position.
    index: FxHashMap<Key, u32>,
    keys: Vec<Key>,
    live: Vec<bool>,
    marked: Vec<bool>,
}

impl LevelPairs {
    /// Record `k` if new; true iff it was. `nodes` is `f`'s nodes at the level.
    fn push(&mut self, eng: &Engine, nodes: usize, k: Key) -> Result<bool, OperationError> {
        let at = self.keys.len() as u32;
        if let (Some(a), None) = k {
            if self.by_node.is_empty() {
                eng.limits().try_resize(&mut self.by_node, nodes, 0)?;
            }
            let slot = &mut self.by_node[a.idx()];
            if *slot != 0 {
                return Ok(false);
            }
            *slot = at + 1;
        } else {
            if self.index.len() == self.index.capacity() { eng.limits().reserve_map(&mut self.index, 1)?; }
            let std::collections::hash_map::Entry::Vacant(slot) = self.index.entry(k) else {
                return Ok(false);
            };
            slot.insert(at);
        }
        eng.limits().try_push(&mut self.keys, k)?;
        eng.limits().try_push(&mut self.live, false)?;
        eng.limits().try_push(&mut self.marked, false)?;
        Ok(true)
    }

    /// The position of a recorded product.
    #[inline]
    fn at(&self, k: Key) -> usize {
        match k {
            (Some(a), None) => self.by_node[a.idx()] as usize - 1,
            k => self.index[&k] as usize,
        }
    }
}

/// The products [`Walk::descend`] has visited at one level: a bit per
/// listed product, a set for the rest.
#[derive(Default)]
struct Seen {
    bits: Vec<u64>,
    set: FxHashSet<Key>,
}

/// The marks being made, and the state of the walk that makes them.
struct Walk<'a> {
    eng: &'a Engine,
    f: &'a Tdd,
    care: &'a Tdd,
    /// The lower of the two operand roots.
    lower: VtreeIdx,
    /// Whether a vtree node lies in the lower operand root's subtree.
    inner: Vec<bool>,
    /// Whether `care`'s root lies in a vtree node's subtree: where it does
    /// not, `care` is `⊤` over the node's variables until it is rooted.
    holds_care: Vec<bool>,
    /// What [`Walk::descend`] has visited, per level.
    seen: Vec<Seen>,
    /// `care_top[v][b]`: `care` node `b` is `⊤` over `v`'s variables, so a
    /// product with it is `f`'s node under `⊤` ([`Walk::child`]).
    care_top: Vec<Vec<bool>>,
    /// `care_flat[v]`: every `care` node at `v` is `⊤`, so the level is
    /// joined on as a marginal `care` level is, never listed.
    care_flat: Vec<bool>,
    products: Vec<Products>,
    /// Per level, a `care` node's pairs keyed by their child on one side, as
    /// `(child, pair index)` ascending, for the nodes the walk joined on it.
    care_index: Vec<CareIndex>,
    remaining: u64,
    poll: PollGate<'a>,
    alive: Vec<Vec<bool>>,
    /// See [`Marking::keep`].
    keep: Vec<Vec<bool>>,
    /// Levels whose every node [`Walk::mark_listed`] kept, so
    /// [`Walk::spread_keep`] has nothing to carry down from them.
    whole_levels: Vec<bool>,
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
        let root_key = walk.top_care(r, root_key);
        match walk.run(r, root_key) {
            Ok(root_live) => {
                walk.poll.flush()?;
                walk.spread_keep()?;
                Ok(Some(Marking { alive: walk.alive, keep: walk.keep, pair_alive: walk.pair_alive, root_live }))
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
            keep: Vec::new(),
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
            if self.keep.get(v.idx()).is_some_and(|row| row[l.idx()]) {
                // Kept whole: nothing under it died.
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
        let mut products = Vec::new();
        lim.reserve_exact(&mut products, nlev)?;
        products.resize_with(nlev, || Products::Absent);
        let mut care_index = Vec::new();
        lim.reserve_exact(&mut care_index, nlev)?;
        care_index.resize_with(nlev, FxHashMap::default);
        let mut holds_care = Vec::new();
        lim.try_resize(&mut holds_care, nlev, false)?;
        let mut at = Some(care.output.vtree);
        while let Some(v) = at {
            holds_care[v.idx()] = true;
            at = vtree.node(v).parent();
        }
        let mut seen = Vec::new();
        lim.reserve_exact(&mut seen, nlev)?;
        seen.resize_with(nlev, Seen::default);
        let (care_top, care_flat) = care_tops(eng, care)?;
        Ok(Walk {
            eng,
            f,
            care,
            lower,
            inner,
            holds_care,
            seen,
            care_top,
            care_flat,
            products,
            care_index,
            remaining,
            poll: lim.gate(),
            alive: mark_rows(eng, f, false)?,
            keep: mark_rows(eng, f, false)?,
            whole_levels: Vec::new(),
            pair_alive: PairMarks::new(eng, f)?,
        })
    }

    /// `k` with a `⊤` `care` node read as `⊤` (`None`), as [`Walk::child`]
    /// reads every product it hands down.
    fn top_care(&self, v: VtreeIdx, k: Key) -> Key {
        match k {
            (fo, Some(b)) if self.care_top[v.idx()].get(b.idx()).copied().unwrap_or(false) => (fo, None),
            k => k,
        }
    }

    /// Record that `f` node `a` at `v` survives whole: a live product pairs
    /// it with `care`'s `⊤`, and every node of a diagram has a model.
    fn keep_node(&mut self, v: VtreeIdx, a: NodeIdx) {
        if !self.f.vtree.node(v).is_leaf() && !self.f.levels[v.idx()].is_marginal() {
            self.keep[v.idx()][a.idx()] = true;
        }
    }

    /// Carry [`Self::keep_node`]'s marks down to every node below a kept
    /// one, alive with all its pairs, top-down over the levels.
    fn spread_keep(&mut self) -> Result<(), OperationError> {
        let (f, vtree) = (self.f, &self.f.vtree);
        let order = vtree.internal_bottomup_slice();
        for &v in order.iter().rev() {
            if f.levels[v.idx()].is_marginal() || self.whole_levels.get(v.idx()).copied().unwrap_or(false) {
                continue;
            }
            let (lc, rc) = vtree.children(v);
            let level = &f.levels[v.idx()];
            for a in 0..self.keep[v.idx()].len() {
                if !self.keep[v.idx()][a] {
                    continue;
                }
                self.poll.poll(1)?;
                self.alive[v.idx()][a] = true;
                for p in level.pairs_iter_of_idx(a) {
                    for (cv, side) in [(lc, p.left), (rc, p.right)] {
                        if vtree.node(cv).is_leaf() || f.levels[cv.idx()].is_marginal() {
                            continue;
                        }
                        let x = ChildDecoder::structural().node(side);
                        if x != ZERO {
                            self.keep[cv.idx()][x.idx()] = true;
                        }
                    }
                }
            }
        }
        self.poll.flush()?;
        Ok(())
    }

    /// Decide the products below the lower root, then mark from the root.
    /// Returns whether the root product is live.
    fn run(&mut self, r: VtreeIdx, root_key: Key) -> Result<bool, Halt> {
        self.decide()?;
        let mut seeds = Vec::new();
        let root_live = if self.inner[r.idx()] {
            let live = self.key_live(r, root_key)?;
            if live {
                self.seed(r, root_key, &mut seeds)?;
            }
            live
        } else if let Some(live) = self.path_top(r, root_key, &mut seeds)? {
            live
        } else {
            self.top(r, root_key, &mut seeds)?
        };
        if self.functional() {
            #[cfg(test)]
            crate::test_helpers::note_restrict_path(0);
            self.mark_listed()?;
        } else {
            #[cfg(test)]
            crate::test_helpers::note_restrict_path(1);
            self.descend(seeds)?;
        }
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

    /// Take `n` probes from the allowance at once.
    #[inline]
    fn spend_many(&mut self, n: u64) -> Result<(), Halt> {
        if self.remaining < n {
            return Err(Halt::Spent);
        }
        self.remaining -= n;
        self.poll.poll(n)?;
        Ok(())
    }

    /// Resolve child references at `cv` (see [`child`]); a `care` node that
    /// is `⊤` over `cv`'s variables is read as `⊤` (`None`).
    #[inline]
    fn child(&self, cv: VtreeIdx, fo: Ref, co: Ref) -> Child {
        match child(self.f, self.care, cv, fo, co) {
            Child::Pair(k) => Child::Pair(self.top_care(cv, k)),
            c => c,
        }
    }

    /// The kind of child `cv` is, for a level below the lower root.
    fn side(&self, cv: VtreeIdx) -> Side {
        if self.f.levels[cv.idx()].is_marginal() {
            Side::FMarginal
        } else if self.care.levels[cv.idx()].is_marginal() || self.care_flat[cv.idx()] {
            Side::CareMarginal
        } else if self.f.vtree.node(cv).is_leaf() {
            Side::Leaf
        } else if matches!(self.products[cv.idx()], Products::Listed(_)) {
            Side::Listed
        } else {
            Side::Lazy
        }
    }

    /// Decide the products of every level below the lower root, children
    /// before parents. An `f` node beside `⊤` is live without a look: no
    /// node of a diagram computes `⊥`.
    fn decide(&mut self) -> Result<(), Halt> {
        let f = self.f;
        for (u, lc, rc) in f.vtree.internal_bottomup() {
            if !self.inner[u.idx()] || f.levels[u.idx()].is_marginal() || self.reads_top(u) {
                continue;
            }
            self.products[u.idx()] = self.list(u, lc, rc)?;
        }
        Ok(())
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
        let nodes = f.levels[u.idx()].nodes().len();
        if care.levels[u.idx()].nodes().len() == 1
            && care.levels[u.idx()].pair_count_at(0) <= SPINE_PAIRS
            && !sides.contains(&Side::Lazy)
        {
            // Found in `f`'s node order, so already ascending.
            #[cfg(test)]
            crate::test_helpers::note_restrict_path(2);
            let found = self.join_spine(u, lc, rc)?;
            return Ok(Products::Listed(listed(eng, nodes, &found)?));
        }
        if !sides.contains(&Side::Lazy) {
            let one = [0, 1].map(|k| sides[k] == Side::Listed && self.one_product_each(kids[k]));
            let found = if one[0] && one[1] {
                #[cfg(test)]
                crate::test_helpers::note_restrict_path(3);
                Some(self.join_paired(u, kids)?)
            } else if let Some(k) = (0..2).find(|&k| one[k]) {
                self.join_keyed(u, kids, k, sides[1 - k], cap)?
            } else {
                None
            };
            if let Some(found) = found {
                // Found in `f`'s node and pair order: ascending already where
                // `care` holds one node here.
                let found = if care.levels[u.idx()].nodes().len() == 1 { found } else { sort_found(eng, nodes, found)? };
                return Ok(Products::Listed(listed(eng, nodes, &found)?));
            }
        }
        let joinable = |k: Side| matches!(k, Side::Listed | Side::Leaf);
        let found = if joinable(sides[0]) && joinable(sides[1]) {
            self.join_both(u, kids, cap)?
        } else if let Some(s) = (0..2).find(|&s| joinable(sides[s])) {
            self.join_one(u, kids, s, sides[1 - s], cap)?
        } else {
            None
        };
        Ok(match found {
            Some(found) => {
                let found = sort_found(eng, nodes, found)?;
                Products::Listed(listed(eng, nodes, &found)?)
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
        // Each orientation costed off `f`'s pairs in place, and `f`'s pairs
        // grouped only by the outer child the cheaper one picks.
        let mut best: Option<(usize, Groups, u64)> = None;
        for o in 0..2 {
            let cg = groups(eng, care, u, o, kids[o])?;
            let cost = join_cost(eng, f, u, o, &lives[o], &lives[1 - o], &cg)?;
            if best.as_ref().is_none_or(|b| cost < b.2) {
                best = Some((o, cg, cost));
            }
        }
        let (o, cg, cost) = best.expect("two orientations were costed");
        if cost > cap {
            let [l0, l1] = lives;
            self.restore(kids[0], l0);
            self.restore(kids[1], l1);
            return Ok(None);
        }
        let fg = groups(eng, f, u, o, kids[o])?;
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

    /// Join level `u`'s pairs where `care` holds one node of few pairs, as
    /// on the path from a semi-join's care root to the variables it reads:
    /// each `f` pair is tried against each `care` pair, `f`'s nodes and
    /// pairs in storage order, so the live pairs of pairs come out ascending
    /// with nothing grouped, indexed or sorted. What a `care` pair's side
    /// asks of `f`'s child there is read once ([`Self::side_test`]): a bit
    /// per child node, where a lookup in the child's listing would be a
    /// miss a pair.
    fn join_spine(&mut self, u: VtreeIdx, lc: VtreeIdx, rc: VtreeIdx) -> Result<Found, Halt> {
        let (eng, f, care) = (self.eng, self.f, self.care);
        let mut cbuf = Vec::new();
        let mut tests: Vec<[SideTest; 2]> = Vec::new();
        for q in care.levels[u.idx()].pairs_read(0, &mut cbuf) {
            let t = [self.side_test(lc, decode(care, lc, q.left))?, self.side_test(rc, decode(care, rc, q.right))?];
            eng.limits().try_push(&mut tests, t)?;
        }
        let decoders = [f.levels[lc.idx()].child_decoder(), f.levels[rc.idx()].child_decoder()];
        let level = &f.levels[u.idx()];
        let per_pair = tests.len() as u64;
        let mut found = Vec::new();
        for a in 0..level.nodes().len() {
            let mut i = 0u32;
            for p in level.pairs_iter_of_idx(a) {
                let refs = [
                    (!decoders[0].is_marginal()).then(|| decoders[0].node(p.left)),
                    (!decoders[1].is_marginal()).then(|| decoders[1].node(p.right)),
                ];
                for (j, t) in tests.iter().enumerate() {
                    if t[0].holds(refs[0]) && t[1].holds(refs[1]) {
                        eng.limits().try_push(&mut found, (a as u32, 0, i, j as u32))?;
                    }
                }
                i += 1;
            }
            self.spend_many(u64::from(i) * per_pair)?;
        }
        Ok(found)
    }

    /// What a `care` pair's side `co` into child level `cv` asks of `f`'s
    /// child there, as [`child`] and [`Self::key_live`] decide it, read off
    /// once for every `f` pair: nothing (`f` marginal there), a label
    /// table (a leaf), a model (`care` `⊤` there: no node of a diagram
    /// computes `⊥`), or a live product with `care`'s node, one bit per
    /// `f` node from the child's listing.
    fn side_test(&self, cv: VtreeIdx, co: Ref) -> Result<SideTest, OperationError> {
        let (f, care) = (self.f, self.care);
        if co == Some(ZERO) {
            return Ok(SideTest::Dead);
        }
        if f.levels[cv.idx()].is_marginal() {
            return Ok(SideTest::Live);
        }
        let co = if care.levels[cv.idx()].is_marginal() { None } else { co };
        if f.vtree.node(cv).is_leaf() {
            let mut table = [false; LEAF_WIDTH];
            for (x, slot) in table.iter_mut().enumerate() {
                *slot = !leaf_dead(Some(NodeIdx(x as u32)), co);
            }
            return Ok(SideTest::Leaf(table));
        }
        let Some(b) = co.filter(|b| !self.care_top[cv.idx()].get(b.idx()).copied().unwrap_or(false)) else {
            return Ok(SideTest::Model);
        };
        let Products::Listed(l) = &self.products[cv.idx()] else {
            unreachable!("a spine joins only on listed children")
        };
        let mut bits = Vec::new();
        self.eng.limits().try_resize(&mut bits, (l.offsets.len() - 1).div_ceil(64), 0u64)?;
        for &(x, y) in &l.products {
            if y == b.0 {
                bits[x as usize / 64] |= 1 << (x % 64);
            }
        }
        Ok(SideTest::Bits(bits))
    }

    /// Whether every `f` node at listed child `cv` has at most one live
    /// product, as a level a semi-join's care reads node by node has.
    fn one_product_each(&self, cv: VtreeIdx) -> bool {
        match &self.products[cv.idx()] {
            Products::Listed(l) => l.offsets.windows(2).all(|w| w[1] - w[0] <= 1),
            _ => false,
        }
    }

    /// Join level `u`'s pairs where every `f` node at both listed children
    /// has at most one live product: an `f` pair's two children name the
    /// one `care` child on each side they can meet, so its partners are the
    /// `care` pairs with exactly those two children, found by a search among
    /// `care`'s pairs ordered by left child, then right. `f`'s pairs are read
    /// once in storage order, nothing of them grouped or costed: the work is
    /// a search an `f` pair and a sort of `care`'s pairs.
    fn join_paired(&mut self, u: VtreeIdx, kids: [VtreeIdx; 2]) -> Result<Found, Halt> {
        let (eng, f, care) = (self.eng, self.f, self.care);
        let lives = [self.take(kids[0]), self.take(kids[1])];
        let node = |c: EncodedChildRef| ChildDecoder::structural().node(c);
        let mut cg = groups(eng, care, u, 0, kids[0])?;
        for y in 0..cg.offsets.len() - 1 {
            let run = &mut cg.entries[cg.offsets[y] as usize..cg.offsets[y + 1] as usize];
            if run.len() > 1 {
                run.sort_unstable_by_key(|&(b, j, cin)| (node(cin).0, b, j));
            }
        }
        let level = &f.levels[u.idx()];
        let mut found = Vec::new();
        for a in 0..level.nodes().len() {
            let mut i = 0u32;
            for p in level.pairs_iter_of_idx(a) {
                let (x0, x1) = (node(p.left), node(p.right));
                let ys = (x0 != ZERO && x1 != ZERO)
                    .then(|| Some((lives[0].of(x0.0).first()?.1, lives[1].of(x1.0).first()?.1)))
                    .flatten();
                if let Some((y0, y1)) = ys {
                    let run = cg.of(y0);
                    let lo = run.partition_point(|e| node(e.2).0 < y1);
                    for &(b, j, _) in run[lo..].iter().take_while(|e| node(e.2).0 == y1) {
                        eng.limits().try_push(&mut found, (a as u32, b, i, j))?;
                    }
                }
                i += 1;
            }
            self.spend_many(u64::from(i))?;
        }
        let [l0, l1] = lives;
        self.restore(kids[0], l0);
        self.restore(kids[1], l1);
        Ok(found)
    }

    /// Join level `u`'s pairs where every `f` node at listed child `kids[k]`
    /// has at most one live product and the other child is no such listing:
    /// an `f` pair's child at `kids[k]` names the one `care` node it can meet
    /// there, and `care`'s pairs with that child are its only partners, each
    /// tested on the other child as [`Self::join_one`] tests it. `f`'s pairs
    /// are read in storage order, once to count the partners and once to
    /// test them; `None`, and nothing tested, when the partners exceed `cap`.
    fn join_keyed(&mut self, u: VtreeIdx, kids: [VtreeIdx; 2], k: usize, other_kind: Side, cap: u64) -> Result<Option<Found>, Halt> {
        let (eng, f, care) = (self.eng, self.f, self.care);
        let (ck, other) = (kids[k], kids[1 - k]);
        let live = self.take(ck);
        let cg = groups(eng, care, u, k, ck)?;
        let level = &f.levels[u.idx()];
        let partners = |p: ChildPair| {
            let x = ChildDecoder::structural().node(side_of(p, k));
            if x == ZERO { None } else { live.of(x.0).first().map(|&(_, y)| y) }
        };
        let mut est = 0u64;
        for a in 0..level.nodes().len() {
            for p in level.pairs_iter_of_idx(a) {
                if let Some(y) = partners(p) {
                    est += cg.of(y).len() as u64;
                }
            }
        }
        if est > cap {
            self.restore(ck, live);
            return Ok(None);
        }
        #[cfg(test)]
        crate::test_helpers::note_restrict_path(4);
        let mut found = Vec::new();
        for a in 0..level.nodes().len() {
            let mut probes = 0u64;
            for (i, p) in level.pairs_iter_of_idx(a).enumerate() {
                probes += 1;
                let Some(y) = partners(p) else { continue };
                let cands = cg.of(y);
                if cands.is_empty() {
                    continue;
                }
                let fo = decode(f, other, side_of(p, 1 - k));
                if other_kind == Side::CareMarginal && !self.live(other, fo, None)? {
                    continue;
                }
                for &(b, j, cside) in cands {
                    probes += 1;
                    let co = decode(care, other, cside);
                    let ok = match other_kind {
                        Side::CareMarginal => true,
                        Side::FMarginal => co != Some(ZERO),
                        _ => self.live(other, fo, co)?,
                    };
                    if ok {
                        eng.limits().try_push(&mut found, (a as u32, b, i as u32, j))?;
                    }
                }
            }
            self.spend_many(probes)?;
        }
        self.restore(ck, live);
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
            // No node of a diagram computes `⊥`.
            (Some(_), None) => Ok(true),
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
    /// that live pairs reach are left in `seeds`, visited. A product of an
    /// `f` node and a `care` that is `⊤` over its level ([`Walk::whole`]) is
    /// live and kept whole, never walked.
    fn top(&mut self, r: VtreeIdx, root_key: Key, seeds: &mut Vec<(VtreeIdx, Key)>) -> Result<bool, Halt> {
        let (eng, f, care) = (self.eng, self.f, self.care);
        let vtree = &f.vtree;
        let nodes = |v: VtreeIdx| f.levels[v.idx()].nodes().len();
        let mut levels: Vec<LevelPairs> = Vec::new();
        eng.limits().reserve_exact(&mut levels, vtree.num_nodes())?;
        levels.resize_with(vtree.num_nodes(), LevelPairs::default);
        levels[r.idx()].push(eng, nodes(r), root_key)?;
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
                            && !self.whole(cv, k)
                            && levels[cv.idx()].push(eng, nodes(cv), k)? {
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
        let mut live_pairs: Vec<u32> = Vec::new();
        while let Some((v, (fo, co))) = stack.pop() {
            let (lc, rc) = vtree.children(v);
            live_pairs.clear();
            for (k, (fl, fr)) in refs(f, v, fo).enumerate() {
                for (cl, cr) in refs(care, v, co) {
                    self.spend()?;
                    if !(self.top_live(&levels, lc, fl, cl)? && self.top_live(&levels, rc, fr, cr)?) {
                        continue;
                    }
                    if fo.is_some() {
                        eng.limits().try_push(&mut live_pairs, k as u32)?;
                    }
                    for (cv, x, y) in [(lc, fl, cl), (rc, fr, cr)] {
                        let Child::Pair(key) = self.child(cv, x, y) else { continue };
                        if self.inner[cv.idx()] {
                            self.seed(cv, key, seeds)?;
                        } else if self.whole(cv, key) {
                            self.keep_node(cv, key.0.expect("a whole product holds an f node"));
                        } else {
                            let at = levels[cv.idx()].at(key);
                            if !std::mem::replace(&mut levels[cv.idx()].marked[at], true) {
                                eng.limits().try_push(&mut stack, (cv, key))?;
                            }
                        }
                    }
                }
            }
            if let Some(a) = fo
                && !live_pairs.is_empty()
            {
                self.mark_many(v, a, live_pairs.iter().copied())?;
            }
        }
        Ok(true)
    }

    /// [`Walk::top`] where `f` is rooted at `r` and `care` lower, at `c`,
    /// as a semi-join's care sits under the target: every product above `c`
    /// is an `f` node beside a `care` not rooted yet, which is `⊤` off the
    /// path from `r` to `c`. A node on the path is live where one of its
    /// pairs leads to a live node one level down the path, its other side
    /// being an `f` node under `⊤` (live, kept whole). So liveness is one
    /// pass up the path's levels and the marks one pass down, both on
    /// arrays by node: no product is discovered or hashed. `None` where the
    /// shape does not hold (or `f` is marginal on the path), for
    /// [`Walk::top`].
    fn path_top(&mut self, r: VtreeIdx, root_key: Key, seeds: &mut Vec<(VtreeIdx, Key)>) -> Result<Option<bool>, Halt> {
        let (eng, f) = (self.eng, self.f);
        let vtree = &f.vtree;
        let c = self.care.output.vtree;
        let (Some(root), None) = root_key else { return Ok(None) };
        if f.output.vtree != r || c == r || !self.holds_care[r.idx()] {
            return Ok(None);
        }
        // The levels from `r` down to `c`'s parent.
        let mut path = Vec::new();
        let mut at = c;
        while at != r {
            at = vtree.node(at).parent().expect("r is an ancestor of c");
            eng.limits().try_push(&mut path, at)?;
        }
        path.reverse();
        if path.iter().any(|&v| f.levels[v.idx()].is_marginal()) {
            return Ok(None);
        }
        // Per path level and node: bit 0 live, bit 1 reached from the root.
        let mut flags: Vec<Vec<u8>> = Vec::new();
        eng.limits().reserve_exact(&mut flags, path.len())?;
        for &v in &path {
            let mut row = Vec::new();
            eng.limits().try_resize(&mut row, f.levels[v.idx()].nodes().len(), 0u8)?;
            flags.push(row);
        }
        // The liveness of `c`'s nodes beside `care`'s root, each decided
        // once: 0 not yet, 1 dead, 2 live.
        let mut at_c: Vec<u8> = Vec::new();
        let c_nodes = if vtree.node(c).is_leaf() { LEAF_WIDTH } else { f.levels[c.idx()].nodes().len() };
        eng.limits().try_resize(&mut at_c, c_nodes, 0u8)?;
        let last = path.len() - 1;
        for pass in [Pass::Up, Pass::Down] {
            let order: Box<dyn Iterator<Item = usize>> = match pass {
                Pass::Up => Box::new((0..path.len()).rev()),
                Pass::Down => Box::new(0..path.len()),
            };
            if pass == Pass::Down {
                if flags[0][root.idx()] & 1 == 0 {
                    return Ok(Some(false));
                }
                flags[0][root.idx()] |= 2;
            }
            let mut live_pairs: Vec<u32> = Vec::new();
            for i in order {
                let v = path[i];
                let on = path.get(i + 1).copied().unwrap_or(c);
                let (lc, rc) = vtree.children(v);
                let (on_left, off) = if lc == on { (true, rc) } else { (false, lc) };
                let level = &f.levels[v.idx()];
                for a in 0..level.nodes().len() {
                    if pass == Pass::Down && flags[i][a] != 3 {
                        continue;
                    }
                    live_pairs.clear();
                    for (k, p) in level.pairs_iter_of_idx(a).enumerate() {
                        self.spend()?;
                        let (down, side) = if on_left { (p.left, p.right) } else { (p.right, p.left) };
                        let other = decode(f, off, side);
                        if other == Some(ZERO) {
                            continue;
                        }
                        let below = decode(f, on, down);
                        let live = match below {
                            Some(x) if i < last => x != ZERO && flags[i + 1][x.idx()] & 1 != 0,
                            _ => self.live_at_care_root(c, below, &mut at_c)?,
                        };
                        if !live {
                            continue;
                        }
                        if pass == Pass::Up {
                            flags[i][a] |= 1;
                            break;
                        }
                        eng.limits().try_push(&mut live_pairs, k as u32)?;
                        if let Some(x) = other {
                            self.keep_node(off, x);
                        }
                        match below {
                            Some(x) if i < last => flags[i + 1][x.idx()] |= 2,
                            _ => {
                                if let Child::Pair(key) = self.child(c, below, None) {
                                    self.seed(c, key, seeds)?;
                                }
                            }
                        }
                    }
                    if !live_pairs.is_empty() {
                        self.mark_many(v, NodeIdx(a as u32), live_pairs.iter().copied())?;
                    }
                }
            }
        }
        Ok(Some(true))
    }

    /// Whether `f`'s reference `x` into `c`, `care`'s root level, is live
    /// beside `care`'s root, memoized in `memo` by node.
    fn live_at_care_root(&mut self, c: VtreeIdx, x: Ref, memo: &mut [u8]) -> Result<bool, Halt> {
        let Some(n) = x else { return self.live(c, None, None) };
        if n == ZERO {
            return Ok(false);
        }
        match memo[n.idx()] {
            0 => {
                let live = self.live(c, x, None)?;
                memo[n.idx()] = if live { 2 } else { 1 };
                Ok(live)
            }
            m => Ok(m == 2),
        }
    }

    /// Whether product `k` at `cv`, above the lower root, is an `f` node
    /// under a `care` that is `⊤` over `cv`'s variables: one not rooted yet
    /// whose root lies outside `cv`'s subtree, or one read as `⊤` there.
    /// Every node of a diagram has a model, so it is live and kept whole.
    #[inline]
    fn whole(&self, cv: VtreeIdx, k: Key) -> bool {
        matches!(k, (Some(_), None)) && !self.holds_care[cv.idx()]
    }

    /// Liveness of child references at `cv` for the walk above the lower root:
    /// from `levels` above it, from the decided products below.
    fn top_live(&mut self, levels: &[LevelPairs], cv: VtreeIdx, fo: Ref, co: Ref) -> Result<bool, Halt> {
        match self.child(cv, fo, co) {
            Child::Dead => Ok(false),
            Child::Live => Ok(true),
            Child::Pair(k) if self.inner[cv.idx()] => self.key_live(cv, k),
            Child::Pair(k) if self.whole(cv, k) => Ok(true),
            Child::Pair(k) => {
                let lp = &levels[cv.idx()];
                Ok(lp.live[lp.at(k)])
            }
        }
    }

    /// Hand live product `k` below the lower root to the walk down: an `f`
    /// node under `⊤` is kept whole, any other is pushed the first time.
    fn seed(&mut self, v: VtreeIdx, k: Key, seeds: &mut Vec<(VtreeIdx, Key)>) -> Result<(), OperationError> {
        if let (Some(a), None) = k {
            self.keep_node(v, a);
        } else if self.visit(v, k)? {
            self.eng.limits().try_push(seeds, (v, k))?;
        }
        Ok(())
    }

    /// Record that product `k` at `v` is visited; true iff it was new. A
    /// listed product takes a bit at its position, any other a set entry.
    fn visit(&mut self, v: VtreeIdx, k: Key) -> Result<bool, OperationError> {
        let lim = self.eng.limits();
        if let (Some(a), Some(b)) = k
            && let Products::Listed(l) = &self.products[v.idx()]
            && let Some(p) = l.position(a.0, b.0)
        {
            let seen = &mut self.seen[v.idx()];
            if seen.bits.is_empty() {
                lim.try_resize(&mut seen.bits, l.products.len().div_ceil(64), 0)?;
            }
            let (word, bit) = (p as usize / 64, 1u64 << (p % 64));
            let new = seen.bits[word] & bit == 0;
            seen.bits[word] |= bit;
            return Ok(new);
        }
        let set = &mut self.seen[v.idx()].set;
        if set.len() == set.capacity() { lim.reserve_set(set, 1)?; }
        Ok(set.insert(k))
    }

    /// Walk the live products below the lower root from `seeds`, each live
    /// and visited, and mark the `f` pairs of their live pairs.
    fn descend(&mut self, seeds: Vec<(VtreeIdx, Key)>) -> Result<(), Halt> {
        let (eng, f, care) = (self.eng, self.f, self.care);
        let vtree = &f.vtree;
        let mut stack = seeds;
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
            if !out.is_empty() {
                self.mark_many(v, a, out.iter().map(|&(i, _)| i))?;
            }
            for &(i, j) in &out {
                let p = fp[i as usize];
                let (cl, cr) = match co {
                    Some(_) => {
                        let q = cp[j as usize];
                        (decode(care, lc, q.left), decode(care, rc, q.right))
                    }
                    None => (None, None),
                };
                for (cv, x, y) in [(lc, decode(f, lc, p.left), cl), (rc, decode(f, rc, p.right), cr)] {
                    match self.child(cv, x, y) {
                        Child::Pair((Some(a), None)) => self.keep_node(cv, a),
                        Child::Pair(k) => {
                            if self.visit(cv, k)? {
                                eng.limits().try_push(&mut stack, (cv, k))?;
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
        Ok(())
    }

    /// Whether `care` is `⊤` over level `u`'s variables there: marginal, or
    /// every node `⊤`.
    fn reads_top(&self, u: VtreeIdx) -> bool {
        self.care.levels[u.idx()].is_marginal() || self.care_flat[u.idx()]
    }

    /// Whether the listings alone give the marks: every level below the
    /// lower root that `care` does not read as `⊤` is listed under a listed
    /// parent, holds no `care` node read as `⊤`, and pairs each `f` node
    /// with at most one live product, as a semi-join's care, one node a
    /// level down to the variables it reads, does. A live pair then hands
    /// each child node the one product it has, so the walk down from the
    /// root would only tell the live products it reaches from those it does
    /// not, and the nodes of those are left unreferenced once the dead pairs
    /// are dropped, for the rebuild's prune.
    fn functional(&self) -> bool {
        let f = self.f;
        for (u, _, _) in f.vtree.internal_bottomup() {
            if !self.inner[u.idx()] || f.levels[u.idx()].is_marginal() || self.reads_top(u) {
                continue;
            }
            let Products::Listed(l) = &self.products[u.idx()] else { return false };
            if self.care_top[u.idx()].iter().any(|&t| t) || l.products.windows(2).any(|w| w[0].0 == w[1].0) {
                return false;
            }
            if u != self.lower {
                let parent = f.vtree.node(u).parent().expect("a level below the lower root has a parent");
                if !matches!(self.products[parent.idx()], Products::Listed(_)) {
                    return false;
                }
            }
        }
        true
    }

    /// The marks below the lower root where [`Self::functional`] holds, read
    /// off the listings in order: each live product's live pairs, and every
    /// node of a level `care` reads as `⊤` kept whole.
    fn mark_listed(&mut self) -> Result<(), Halt> {
        let (eng, f) = (self.eng, self.f);
        eng.limits().try_resize(&mut self.whole_levels, f.vtree.num_nodes(), false)?;
        for (u, _, _) in f.vtree.internal_bottomup() {
            if !self.inner[u.idx()] || f.levels[u.idx()].is_marginal() {
                continue;
            }
            if self.reads_top(u) {
                self.keep[u.idx()].fill(true);
                self.alive[u.idx()].fill(true);
                self.whole_levels[u.idx()] = true;
                continue;
            }
            let Products::Listed(l) = &self.products[u.idx()] else {
                unreachable!("a functional walk lists every level it does not read as ⊤")
            };
            let level = &f.levels[u.idx()];
            for (p, &(a, _)) in l.products.iter().enumerate() {
                self.poll.poll(1)?;
                self.alive[u.idx()][a as usize] = true;
                let combos = &l.combos[l.starts[p] as usize..l.starts[p + 1] as usize];
                let count = level.pair_count_at(a as usize);
                self.pair_alive.mark_many(eng, u, NodeIdx(a), count, combos.iter().map(|&(i, _)| i))?;
            }
        }
        Ok(())
    }

    /// Mark `f` pairs `pairs` of node `a` at `v`, and the node, live.
    fn mark_many(&mut self, v: VtreeIdx, a: NodeIdx, pairs: impl IntoIterator<Item = u32>) -> Result<(), OperationError> {
        self.alive[v.idx()][a.idx()] = true;
        let count = self.f.levels[v.idx()].pair_count_at(a.idx());
        self.pair_alive.mark_many(self.eng, v, a, count, pairs)
    }
}

/// Per level, which `care` nodes are `⊤` over the level's variables, and
/// whether all are: a node with one pair whose children are `⊤` (a marginal
/// level, the `⊤` leaf label, or such a node), which is how a reduced
/// diagram holds `⊤`. A `⊤` written with more pairs is not seen, which only
/// costs the walk time.
fn care_tops(eng: &Engine, care: &Tdd) -> Result<(Vec<Vec<bool>>, Vec<bool>), OperationError> {
    let vtree = &care.vtree;
    let mut top = mark_rows(eng, care, false)?;
    let mut flat = Vec::new();
    eng.limits().try_resize(&mut flat, vtree.num_nodes(), false)?;
    let side_top = |top: &[Vec<bool>], cv: VtreeIdx, side: EncodedChildRef| {
        let decoder = care.levels[cv.idx()].child_decoder();
        if decoder.is_marginal() {
            return true;
        }
        let node = decoder.node(side);
        if vtree.node(cv).is_leaf() { node == ONE_LEAF_IDX } else { node != ZERO && top[cv.idx()][node.idx()] }
    };
    for (u, lc, rc) in vtree.internal_bottomup() {
        let level = &care.levels[u.idx()];
        if level.is_marginal() {
            continue;
        }
        let n = level.nodes().len();
        let mut all = n > 0;
        for b in 0..n {
            let t = level.pair_count_at(b) == 1 && {
                let p = level.pairs_iter_of_idx(b).next().expect("a node of one pair");
                side_top(&top, lc, p.left) && side_top(&top, rc, p.right)
            };
            top[u.idx()][b] = t;
            all &= t;
        }
        flat[u.idx()] = all;
    }
    Ok((top, flat))
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

/// What [`Walk::join_both`] does with side `o` of level `u` as the outer
/// child, whose live products are `outer`: per `f` child there, gather the
/// `care` pairs live with it, then look each `f` pair under it up among them
/// by its inner child. Read off `f`'s pairs in place, without grouping them.
fn join_cost(eng: &Engine, f: &Tdd, u: VtreeIdx, o: usize, outer: &Listed, inner: &Listed, cg: &Groups) -> Result<u64, OperationError> {
    let lim = eng.limits();
    let keys = outer.offsets.len() - 1;
    // Per outer child, the `care` pairs it gathers, and whether the cost
    // holds them yet.
    let mut gathered = Vec::new();
    lim.try_resize(&mut gathered, keys, 0u64)?;
    let mut counted = Vec::new();
    lim.try_resize(&mut counted, keys, false)?;
    for (x, slot) in gathered.iter_mut().enumerate() {
        *slot = outer.of(x as u32).iter().map(|&(_, y)| cg.of(y).len() as u64).sum();
    }
    let level = &f.levels[u.idx()];
    let mut cost = 0u64;
    for a in 0..level.nodes().len() {
        for p in level.pairs_iter_of_idx(a) {
            let x = ChildDecoder::structural().node(side_of(p, o));
            if x == ZERO {
                continue;
            }
            let g = gathered[x.idx()];
            if g == 0 {
                continue;
            }
            if !std::mem::replace(&mut counted[x.idx()], true) {
                cost = cost.saturating_add(g);
            }
            let xi = ChildDecoder::structural().node(side_of(p, 1 - o));
            if xi != ZERO {
                cost = cost.saturating_add((inner.of(xi.0).len() as u64).min(g));
            }
        }
    }
    Ok(cost)
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
    Listed { products, offsets, starts: Vec::new(), combos: Vec::new() }
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
    Ok(Listed { products, offsets, starts, combos })
}

/// `found` ascending: counted out by `f` node, each node's few entries
/// then sorted, where one sort of the whole would pay its logarithm.
fn sort_found(eng: &Engine, nodes: usize, found: Found) -> Result<Found, OperationError> {
    let lim = eng.limits();
    if found.len() < 2 {
        return Ok(found);
    }
    let mut offsets = Vec::new();
    lim.try_resize(&mut offsets, nodes + 1, 0u32)?;
    for e in &found {
        offsets[e.0 as usize + 1] += 1;
    }
    for a in 0..nodes {
        offsets[a + 1] += offsets[a];
    }
    let mut out = Vec::new();
    lim.try_resize(&mut out, found.len(), (0u32, 0u32, 0u32, 0u32))?;
    let mut cursor = offsets.clone();
    for e in found {
        let at = &mut cursor[e.0 as usize];
        out[*at as usize] = e;
        *at += 1;
    }
    for a in 0..nodes {
        let run = &mut out[offsets[a] as usize..offsets[a + 1] as usize];
        if run.len() > 1 {
            run.sort_unstable();
        }
    }
    Ok(out)
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
