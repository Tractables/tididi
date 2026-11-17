use crate::Engine;
use crate::diagram::{ChildPair, ChildSide, NodeIdx, StoredPairs, Tdd, TddLevel};
use crate::vtree::VtreeIdx;

use crate::limits::{Limits, OperationError};

use super::scratch::{ContractScratch, EMPTY_SLOT, TwinSlot};

/// Prefetch the twin-table slot a future iteration will probe first. The
/// probe target is a random index into a table that typically misses L2;
/// `fingerprints[]` is read sequentially, so the slot for iteration i+D is
/// known D iterations ahead. No-op on non-x86_64, and under Miri, which does
/// not implement the intrinsic.
fn prefetch_slot(p: *const TwinSlot, slot: usize) {
    crate::isa::prefetch_line(p.wrapping_add(slot));
}

/// Iterate parent pairs and yield `(parent_i, target, sibling)` to `f`,
/// where `target` is the node at level `t1` (left or right of each pair
/// depending on `t1_side`) and `sibling` is the other child.
#[inline]
fn for_each_target_sibling(
    parent_level: &TddLevel,
    t1_side: ChildSide,
    mut f: impl FnMut(u32, u32, u32),
) {
    // `target` indexes child-width-sized scratch arrays. `sibling` is passed
    // raw: it is only hashed and packed, never indexed. A walk per side, each
    // its own instance, so that no pair tests the side.
    match t1_side {
        ChildSide::Left => parent_level.for_each_node_pair(|i, pair| {
            let (target, sibling) = split_pair(&pair, ChildSide::Left);
            f(i as u32, target, sibling);
        }),
        ChildSide::Right => parent_level.for_each_node_pair(|i, pair| {
            let (target, sibling) = split_pair(&pair, ChildSide::Right);
            f(i as u32, target, sibling);
        }),
    }
}

/// The node a pair names on side `t1_side`, and the raw ref of the other
/// side. The child level on that side is structural, pair fusion handling a
/// marginal one, so the side toward it is a node index.
#[inline]
fn split_pair(pair: &ChildPair, t1_side: ChildSide) -> (u32, u32) {
    t1_side.pick((pair.left.0, pair.right.0), (pair.right.0, pair.left.0))
}

/// The splitmix64 finalizer (Steele et al., 2014) — the shared bit-diffusion
/// step behind every fingerprint in the contract module.
///
/// Constants and shift schedule are the original SplitMix64 ones, chosen for
/// their avalanche behaviour.
///
/// Callers own their own prelude (how the inputs are packed into the u64, and
/// whether a golden-ratio increment is added first) — that prelude is what
/// makes each rule's fingerprint distribution distinct, so do not fold one
/// caller's prelude in here.
#[inline(always)]
pub(crate) fn mix64(mut x: u64) -> u64 {
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
    x ^ (x >> 31)
}

/// The hash of one entry that a node's fingerprint sums: the two halves of
/// the 128-bit product of the entry, offset by a constant, with an odd
/// constant, added together by exclusive-or. One multiplication an entry,
/// where [`mix64`] takes two in a chain; the fingerprints only nominate
/// candidates, which the groups compare in full.
#[inline(always)]
fn entry_hash(entry: u64) -> u64 {
    let product = u128::from(entry ^ 0x243F_6A88_85A3_08D3) * 0x9E37_79B9_7F4A_7C15;
    (product as u64) ^ ((product >> 64) as u64)
}

/// A fingerprint multiplied by an odd constant, whose high bits place it in
/// the twin tables. The high half of [`entry_hash`]'s product grows nearly
/// linearly with an entry's parent node, so a sum of entries keeps that
/// structure in its low bits: nodes whose `k` contexts sit at parents a
/// fixed stride apart get fingerprints with equal low bits, and a table
/// indexed by them probes long runs of occupied slots. Every high bit of
/// the product depends on every bit of the fingerprint. The map is a
/// bijection, so equal fingerprints keep equal keys.
#[inline(always)]
fn spread(fp: u64) -> u64 {
    fp.wrapping_mul(0x9E37_79B9_7F4A_7C15)
}

mod groups;
mod listed;
mod refine;

use refine::Start;

pub(super) use groups::SortCounts;
pub(super) use listed::{find_listed_twin_groups, named_by, pair_mass};
pub(super) use refine::list_members;

#[cfg(test)]
mod tests;

/// The entries whose multiset per node decides which nodes are twins. The
/// three scatters of [`group_twins_by_entries`] walk it, and read the same
/// `(node, entry)` sequence each time.
pub(super) trait TwinEntries {
    /// Whether two nodes with no entry are twins.
    const EMPTY_TWINS: bool = true;

    /// Call `f(node, entry)` once per entry of every node.
    fn for_each(&self, f: impl FnMut(u32, u64));

    /// The number of entries [`Self::for_each`] yields.
    fn len(&self) -> usize;

    /// Whether none of the `width` nodes can have a twin: no two nodes share
    /// an entry, and at most one has none where two nodes with none are
    /// twins ([`Self::EMPTY_TWINS`]). `false` when that is not known.
    fn no_twin(
        &self,
        _lim: &Limits,
        _scratch: &mut ContractScratch,
        _width: usize,
    ) -> Result<bool, OperationError> {
        Ok(false)
    }
}

/// Two 32-bit values packed into one entry, `hi` in the high half.
#[inline]
fn pack(hi: u32, lo: u32) -> u64 {
    ((hi as u64) << 32) | lo as u64
}

/// Context twins: a node's entries are the `(parent node, sibling)` contexts
/// the parent level's pairs put it in, packed parent-high.
struct ContextEntries<'a> {
    parent_level: &'a TddLevel,
    t1_side: ChildSide,
    /// Whether [`no_twin`](TwinEntries::no_twin) tests the level at all.
    early_stop: bool,
}

/// A parent level with fewer entries than this, nodes and arena pairs
/// together, goes straight to the fingerprints. On the networks most
/// levels are this small, most of them repeat a sibling within a few pairs,
/// and the test's fixed cost is not repaid by the few it clears.
const EARLY_STOP_MIN_ENTRIES: usize = 128;

/// Whether a search of a whole child level under `parent_level` screens it
/// ([`TwinEntries::no_twin`]) before any fingerprint.
pub(super) fn screens(parent_level: &TddLevel) -> bool {
    parent_level.nodes().len() + parent_level.arena_len() >= EARLY_STOP_MIN_ENTRIES
}

/// The most cells a parent node's sibling table starts with. A wide node
/// most often repeats a sibling within its first few pairs, so its table
/// starts at a size that stays in cache and doubles only as siblings are
/// filed, never to the node's whole pair count up front.
const TWIN_TABLE_START_CELLS: usize = 1 << 12;

impl TwinEntries for ContextEntries<'_> {
    /// A node no pair names has no context: nothing reaches it, and it is
    /// no node's twin.
    const EMPTY_TWINS: bool = false;

    fn for_each(&self, mut f: impl FnMut(u32, u64)) {
        for_each_target_sibling(self.parent_level, self.t1_side, |pi, target, sibling| {
            f(target, pack(pi, sibling));
        });
    }

    fn len(&self) -> usize {
        self.parent_level.live_pairs()
    }

    /// Two nodes share a context only where a parent node pairs both with
    /// one sibling ([`repeats_a_sibling`]); a level whose parent nodes never
    /// repeat one, common on grids, then needs no fingerprint.
    fn no_twin(
        &self,
        lim: &Limits,
        scratch: &mut ContractScratch,
        _width: usize,
    ) -> Result<bool, OperationError> {
        if !self.early_stop {
            return Ok(false);
        }
        Ok(repeats_a_sibling(self.parent_level, self.t1_side, lim, scratch)?.is_none())
    }
}

/// The pair count of a node of `level` that pairs two nodes of the child
/// level on `side` with one sibling, if one does. Two nodes share a context
/// only there (math §2.8), so without one no node has a twin.
///
/// Each parent node is tested alone, in cache, and the test stops at the
/// first such sibling ([`node_repeats_a_sibling`]). A stored level's nodes
/// are read as slices of its arena, in a loop of their own with the side a
/// constant; an implicit level's are generated into a buffer a node at a
/// time.
pub(super) fn repeats_a_sibling(
    level: &TddLevel,
    side: ChildSide,
    lim: &Limits,
    scratch: &mut ContractScratch,
) -> Result<Option<usize>, OperationError> {
    let ContractScratch {
        twin_local,
        twin_generation,
        ..
    } = scratch;
    match (level.stored(), side) {
        (Some(stored), ChildSide::Left) => {
            stored_repeats_a_sibling::<true>(stored, lim, twin_local, twin_generation)
        }
        (Some(stored), ChildSide::Right) => {
            stored_repeats_a_sibling::<false>(stored, lim, twin_local, twin_generation)
        }
        (None, _) => {
            let (mut buf, mut cursor) = (Vec::new(), None);
            for i in 0..level.node_count() {
                let pairs = level.described_read_next(&mut cursor, i, &mut buf);
                if node_repeats_a_sibling::<false>(pairs, side, lim, twin_local, twin_generation)? {
                    return Ok(Some(pairs.len()));
                }
            }
            Ok(None)
        }
    }
}

/// A stored level of at least this many nodes, and at least twice as many
/// arena pairs, whose first 64th holds no node that repeats a sibling has
/// its widest node tested next. On a level of narrower nodes the pass that
/// finds it costs about what reading the level does.
const WIDEST_FIRST_MIN_NODES: usize = 1 << 16;

/// [`repeats_a_sibling`] of a stored level, `LEFT` saying whether the
/// child level is the left one.
///
/// Where few nodes of a level repeat a sibling, the node numbering decides
/// how much of the level the test reads before the first, and a wider node
/// is likelier to be one. On a large level the test reads a prefix in
/// order, then the widest node of the rest, then the rest in order.
fn stored_repeats_a_sibling<const LEFT: bool>(
    stored: StoredPairs<'_>,
    lim: &Limits,
    twin_local: &mut Vec<u64>,
    twin_generation: &mut u32,
) -> Result<Option<usize>, OperationError> {
    let side = if LEFT {
        ChildSide::Left
    } else {
        ChildSide::Right
    };
    let nodes = stored.nodes();
    let head = if nodes.len() >= WIDEST_FIRST_MIN_NODES && stored.arena().len() >= 2 * nodes.len() {
        nodes.len() / 64
    } else {
        nodes.len()
    };
    for node in &nodes[..head] {
        let pairs = stored.of(node);
        if node_repeats_a_sibling::<false>(pairs, side, lim, twin_local, twin_generation)? {
            return Ok(Some(pairs.len()));
        }
    }
    let rest = &nodes[head..];
    // A ranged node's words hold one pair; the last pass reads it.
    let (mut widest, mut most) = (None, refine::SMALL_PAIRS as u32);
    for node in rest {
        if node.held_count() > most {
            (widest, most) = (Some(node), node.held_count());
        }
    }
    if let Some(node) = widest {
        let pairs = stored.of(node);
        if node_repeats_a_sibling::<false>(pairs, side, lim, twin_local, twin_generation)? {
            return Ok(Some(pairs.len()));
        }
    }
    for node in rest {
        let pairs = stored.of(node);
        if node_repeats_a_sibling::<false>(pairs, side, lim, twin_local, twin_generation)? {
            return Ok(Some(pairs.len()));
        }
    }
    Ok(None)
}

/// Whether the parent node of `pairs` pairs two nodes of the child level
/// on `side` with one sibling. A node of up to four pairs compares every
/// two of them, and one of up to [`refine::SMALL_PAIRS`] sorts them on the
/// stack by sibling and compares neighbours. A wider one files its
/// siblings in a generation-stamped table, the first cells of
/// `twin_local`, a power of two at least twice the siblings filed, so a
/// probe always ends at a cell this node has not stamped; a cell holds its
/// stamp in the high half and the sibling in the low one, and finding the
/// sibling already filed is the repeat, so a wide node whose siblings
/// repeat stops at the first one, after reading a few of its pairs. The
/// table starts at most [`TWIN_TABLE_START_CELLS`] long and doubles when
/// half full, under a fresh stamp the siblings read so far are filed again
/// with, so its size follows the pairs read, not the node's width. A wide
/// node that repeats a pair, which only a marginal level allows, counts as
/// one that pairs two nodes with its sibling.
///
/// With `SETS`, a node of up to [`refine::SMALL_PAIRS`] pairs is tested for
/// the stronger condition [`screen_twins`] needs: two nodes it names with
/// the same siblings. A wider node is tested for a repeated sibling still,
/// up to half [`TWIN_TABLE_START_CELLS`] pairs; a node wider than that
/// counts as one that repeats a sibling.
#[inline(always)]
fn node_repeats_a_sibling<const SETS: bool>(
    pairs: &[ChildPair],
    side: ChildSide,
    lim: &Limits,
    twin_local: &mut Vec<u64>,
    twin_generation: &mut u32,
) -> Result<bool, OperationError> {
    // Two pairs with one sibling and two nodes.
    let repeat = |a: (u32, u32), b: (u32, u32)| a.1 == b.1 && a.0 != b.0;
    match *pairs {
        [] | [_] => return Ok(false),
        [p, q] => return Ok(repeat(split_pair(&p, side), split_pair(&q, side))),
        [p, q, r] if !SETS => {
            let (p, q, r) = (
                split_pair(&p, side),
                split_pair(&q, side),
                split_pair(&r, side),
            );
            return Ok(repeat(p, q) | repeat(p, r) | repeat(q, r));
        }
        [p, q, r, u] if !SETS => {
            let (p, q, r, u) = (
                split_pair(&p, side),
                split_pair(&q, side),
                split_pair(&r, side),
                split_pair(&u, side),
            );
            return Ok(repeat(p, q)
                | repeat(p, r)
                | repeat(p, u)
                | repeat(q, r)
                | repeat(q, u)
                | repeat(r, u));
        }
        [p, q, r] => {
            // Two pairs of one node give it two siblings, the third node one.
            let (p, q, r) = (split_pair(&p, side), split_pair(&q, side), split_pair(&r, side));
            let apart = (p.0 != q.0) & (p.0 != r.0) & (q.0 != r.0);
            return Ok(apart & ((p.1 == q.1) | (p.1 == r.1) | (q.1 == r.1)));
        }
        [p, q, r, u] => {
            let (a, b, c, d) = (split_pair(&p, side), split_pair(&q, side), split_pair(&r, side), split_pair(&u, side));
            // A node named once, by one sibling.
            let one = |x: (u32, u32), o: (u32, u32), v: (u32, u32), w: (u32, u32)| (x.0 != o.0) & (x.0 != v.0) & (x.0 != w.0);
            let (oa, ob, oc, od) = (one(a, b, c, d), one(b, a, c, d), one(c, a, b, d), one(d, a, b, c));
            let ones = (oa & ob & (a.1 == b.1))
                | (oa & oc & (a.1 == c.1))
                | (oa & od & (a.1 == d.1))
                | (ob & oc & (b.1 == c.1))
                | (ob & od & (b.1 == d.1))
                | (oc & od & (c.1 == d.1));
            // Two nodes named twice each, by the same two siblings.
            let two = |x: (u32, u32), y: (u32, u32), v: (u32, u32), w: (u32, u32)| {
                (x.0 == y.0) & (v.0 == w.0) & (x.0 != v.0) & (((x.1 == v.1) & (y.1 == w.1)) | ((x.1 == w.1) & (y.1 == v.1)))
            };
            return Ok(ones | two(a, b, c, d) | two(a, c, b, d) | two(a, d, b, c));
        }
        _ if pairs.len() <= refine::SMALL_PAIRS => {
            let n = pairs.len();
            let mut key = [0u64; refine::SMALL_PAIRS];
            for (k, pair) in pairs.iter().enumerate() {
                let (t, s) = split_pair(pair, side);
                key[k] = if SETS { pack(t, s) } else { pack(s, t) };
            }
            let key = &mut key[..n];
            key.sort_unstable();
            if !SETS {
                return Ok(key.windows(2).any(|w| w[0] >> 32 == w[1] >> 32 && w[0] != w[1]));
            }
            return Ok(runs_alike(key));
        }
        // Past what the table holds at its first size, a node is taken to
        // name two nodes alike unread: its siblings would grow the table
        // to twice its pairs, where the refinement it leaves the level to
        // reads them for less.
        _ if SETS && 2 * pairs.len() > TWIN_TABLE_START_CELLS => return Ok(true),
        _ => {}
    }
    let mut cells = (2 * pairs.len())
        .next_power_of_two()
        .min(TWIN_TABLE_START_CELLS);
    let mut stamp = next_twin_stamp(lim, twin_local, twin_generation, cells)?;
    for (read, pair) in pairs.iter().enumerate() {
        let s = split_pair(pair, side).1;
        if 2 * (read + 1) > cells {
            // Half full: double under a fresh stamp and file the
            // siblings read so far again. They are distinct, or the
            // test would have stopped.
            cells *= 2;
            stamp = next_twin_stamp(lim, twin_local, twin_generation, cells)?;
            for pair in &pairs[..read] {
                file_sibling(&mut twin_local[..cells], stamp, split_pair(pair, side).1);
            }
        }
        // The node's window of the table: a narrow node after a wide
        // one stays in cache. Cells past it keep older stamps unread.
        if file_sibling(&mut twin_local[..cells], stamp, s) {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Whether two of the runs of `key`, sorted node-high, one run a node,
/// hold the same siblings.
fn runs_alike(key: &[u64]) -> bool {
    let mut start = [0usize; refine::SMALL_PAIRS + 1];
    let mut runs = 0;
    for k in 0..key.len() {
        if k == 0 || key[k] >> 32 != key[k - 1] >> 32 {
            start[runs] = k;
            runs += 1;
        }
    }
    start[runs] = key.len();
    for i in 0..runs {
        let a = &key[start[i]..start[i + 1]];
        for j in i + 1..runs {
            let b = &key[start[j]..start[j + 1]];
            if a.len() == b.len() && a.iter().zip(b).all(|(x, y)| *x as u32 == *y as u32) {
                return true;
            }
        }
    }
    false
}

/// What [`screen_twins`] tells of a whole child level from its parent level.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Screen {
    /// No parent node pairs two nodes with one sibling: no node has a twin.
    Apart,
    /// Parent nodes pair two nodes with one sibling, but none names two
    /// nodes with the same siblings: no node has a twin.
    Named,
    /// A parent node names two nodes with the same siblings, or, past
    /// [`refine::SMALL_PAIRS`] pairs, pairs two with one or is wider than
    /// the sibling table's first size: the level may hold twins.
    Alike,
}

/// Screen the child level on `side` of `level` for twins.
///
/// Two nodes are twins only where every parent node that names one names
/// the other with the same siblings, so a level where no parent node names
/// two nodes with the same siblings has none. The parent nodes are first
/// tested for a repeated sibling ([`repeats_a_sibling`]); where one
/// repeats one, each is tested for two nodes with the same siblings
/// ([`node_repeats_a_sibling`] with `SETS`). An implicit level is tested
/// for a repeated sibling only, and is `Apart` or `Alike`.
pub(super) fn screen_twins(
    level: &TddLevel,
    side: ChildSide,
    lim: &Limits,
    scratch: &mut ContractScratch,
) -> Result<Screen, OperationError> {
    // The second test starts over rather than at the first node that
    // repeats a sibling: a first test that tells where it stopped compiles
    // to a slower loop than `repeats_a_sibling`'s, which reads every pair
    // of the levels that end `Apart`, and cost some CNF compilations 3% of
    // their instructions.
    let Some(pairs) = repeats_a_sibling(level, side, lim, scratch)? else {
        return Ok(Screen::Apart);
    };
    // Past `SMALL_PAIRS` the second test of a node is the first one.
    if pairs > refine::SMALL_PAIRS {
        return Ok(Screen::Alike);
    }
    let Some(stored) = level.stored() else {
        return Ok(Screen::Alike);
    };
    let ContractScratch { twin_local, twin_generation, .. } = scratch;
    match side {
        ChildSide::Left => stored_screen::<true>(stored, lim, twin_local, twin_generation),
        ChildSide::Right => stored_screen::<false>(stored, lim, twin_local, twin_generation),
    }
}

/// The second test of [`screen_twins`] on a stored level, `LEFT` saying
/// whether the child level is the left one.
fn stored_screen<const LEFT: bool>(
    stored: StoredPairs<'_>,
    lim: &Limits,
    twin_local: &mut Vec<u64>,
    twin_generation: &mut u32,
) -> Result<Screen, OperationError> {
    let side = if LEFT { ChildSide::Left } else { ChildSide::Right };
    for node in stored.nodes() {
        if node_repeats_a_sibling::<true>(stored.of(node), side, lim, twin_local, twin_generation)? {
            return Ok(Screen::Alike);
        }
    }
    Ok(Screen::Named)
}

/// Set a bit in `scratch.twin_named` for each node of the child level of
/// `width` nodes on `side` that a pair of `stored` names. Where the nodes'
/// pairs cover the arena, the arena is read whole, in one pass with no
/// node's bounds ([`StoredPairs::for_each_pair_whole`]); the nodes that
/// hold their pairs there then mark a node one past the level's.
fn mark_named(stored: StoredPairs<'_>, side: ChildSide, width: usize, lim: &Limits, twin_named: &mut Vec<u64>) -> Result<(), OperationError> {
    let words = width / 64 + 1;
    lim.try_resize(twin_named, words, 0u64)?;
    let named = &mut twin_named[..words];
    named.fill(0);
    let mut mark = |pair: &ChildPair| {
        let t = split_pair(pair, side).0 as usize;
        named[t / 64] |= 1 << (t % 64);
    };
    let past = NodeIdx(width as u32);
    let none = match side {
        ChildSide::Left => ChildPair::new(past, NodeIdx(0)),
        ChildSide::Right => ChildPair::new(NodeIdx(0), past),
    };
    if !stored.for_each_pair_whole(none, |pair| mark(&pair)) {
        for node in stored.nodes() {
            stored.of(node).iter().for_each(&mut mark);
        }
    }
    Ok(())
}

/// After a [`Screen::Named`] screen of a level of `width` nodes: list the
/// nodes no pair names in `scratch.unnamed`, ascending, where there are two
/// or more, and record whether every other node is named, as a refinement
/// of the whole level does at its end.
fn list_unnamed(width: usize, lim: &Limits, scratch: &mut ContractScratch) -> Result<(), OperationError> {
    let ContractScratch { twin_named, unnamed, unnamed_exact, .. } = scratch;
    let clear = |w: usize| {
        let word = !twin_named[w];
        if (w + 1) * 64 > width { word & ((1u64 << (width % 64)) - 1) } else { word }
    };
    let words = width.div_ceil(64);
    let count: usize = (0..words).map(|w| clear(w).count_ones() as usize).sum();
    *unnamed_exact = count != 1;
    if count < 2 {
        return Ok(());
    }
    lim.reserve_exact(unnamed, count)?;
    for w in 0..words {
        let mut word = clear(w);
        while word != 0 {
            unnamed.push((w * 64) as u32 + word.trailing_zeros());
            word &= word - 1;
        }
    }
    Ok(())
}

/// The next stamp of the sibling table, with `twin_local` at least `cells`
/// long. A fresh stamp per node, or per doubling, instead of clearing its
/// cells. On u32 wrap the stamps are zeroed and it restarts at 1: 0 is the
/// stamp of a cell never written, so it never names a live node.
#[inline]
fn next_twin_stamp(
    lim: &Limits,
    twin_local: &mut Vec<u64>,
    twin_generation: &mut u32,
    cells: usize,
) -> Result<u64, OperationError> {
    *twin_generation = match twin_generation.checked_add(1) {
        Some(g) => g,
        None => {
            twin_local.fill(0);
            1
        }
    };
    if twin_local.len() < cells {
        lim.try_resize(twin_local, cells, 0u64)?;
    }
    Ok((*twin_generation as u64) << 32)
}

/// File sibling `s` under `stamp` in `local`, a power of two at least eight
/// cells long with fewer than half of them stamped `stamp`, and say whether
/// `s` was filed there already.
#[inline]
fn file_sibling(local: &mut [u64], stamp: u64, s: u32) -> bool {
    let (mask, shift) = (local.len() - 1, 32 - local.len().trailing_zeros());
    let filed = stamp | s as u64;
    let mut h = (s.wrapping_mul(0x9E37_79B1) >> shift) as usize;
    loop {
        let cell = local[h];
        if cell == filed {
            return true;
        }
        if cell & !0xFFFF_FFFF != stamp {
            local[h] = filed;
            return false;
        }
        h = (h + 1) & mask;
    }
}

/// Content twins: a node's entries are its own pairs, packed left-high.
pub(super) struct ContentEntries<'a>(pub(super) &'a TddLevel);

impl TwinEntries for ContentEntries<'_> {
    fn for_each(&self, mut f: impl FnMut(u32, u64)) {
        self.0
            .for_each_node_pair(|i, pair| f(i as u32, pack(pair.left.0, pair.right.0)));
    }

    fn len(&self) -> usize {
        self.0.live_pairs()
    }

    fn no_twin(&self, _lim: &Limits, _scratch: &mut ContractScratch, _width: usize) -> Result<bool, OperationError> {
        Ok(self.0.implicit().is_some_and(crate::diagram::ImplicitLevel::nodes_distinct))
    }
}

/// The width below which a screened search of a level without a marginal
/// level goes by fingerprints. Such a search most often finds no twin; on a
/// narrow level the fingerprints and their probe table stay in the
/// first-level cache, and summing an entry costs less than the refinement's
/// sort of each parent node's pairs. On a wider one the fingerprints'
/// scattered writes miss it, where the refinement reads a bit per node.
const NARROW_SCREENED_WIDTH: usize = 1 << 12;

/// Find nodes with identical parent-context multisets at the explicit child
/// level `t1` of `parent`. Marginal children are handled by pair fusion
/// instead. See [`group_twins_by_entries`] for what is left in `scratch`.
/// The nodes no pair names are in no group; a search that read every pair
/// lists them in `scratch.unnamed` where there are two or more.
///
/// Without a marginal level no parent node repeats a pair, and the level is
/// searched by refining a partition of it parent node by parent node
/// ([`refine::refine_context_groups`]); debug builds check the groups
/// against the fingerprint search over the same contexts. Where a parent
/// node of [`refine::HAND_OFF_PAIRS`] pairs or more has them in no order,
/// the refinement gives up before its wide parent nodes, and the level goes
/// by fingerprints. With `screen`, the parent level is first tested for a
/// parent node that names two nodes with the same siblings
/// ([`screen_twins`]), without which no node has a twin: worth it where few
/// nodes are expected to have one, as under a parent level no merge has
/// changed. A level the test finds twin-free after a parent node repeats a
/// sibling lists its unnamed nodes as the refinement would: where the
/// diagram lists the level loose, or does not know
/// ([`Dirty::loose`](crate::diagram::Dirty::loose)), a pass over the parent
/// level marks the named nodes ([`mark_named`]), and otherwise every node
/// is named. Such a search
/// of a level narrower than [`NARROW_SCREENED_WIDTH`] goes by fingerprints
/// instead, and debug builds check it against the refinement.
///
/// With a marginal level the contexts are multisets, which the
/// refinement's sibling and degree splits do not count, and the level is
/// searched by fingerprints ([`group_twins_by_entries`]): there few parent
/// nodes tell two nodes apart, and the refinement, which stops once every
/// node is alone, reads nearly every pair, at more than the fingerprints'
/// cost per pair.
pub(super) fn find_twin_groups(
    eng: &Engine,
    tdd: &Tdd,
    t1: VtreeIdx,
    parent: VtreeIdx,
    t1_side: ChildSide,
    scratch: &mut ContractScratch,
    screen: bool,
) -> Result<bool, OperationError> {
    let level = &tdd.levels[t1.idx()];
    debug_assert!(
        !level.is_marginal(),
        "a marginal child is left to pair fusion"
    );
    let parent_level = &tdd.levels[parent.idx()];
    let width = level.slot_count();
    debug_assert!(
        width <= u32::MAX as usize,
        "level width {width} exceeds the u32 node-index range",
    );
    let entries = ContextEntries {
        parent_level,
        t1_side,
        early_stop: screens(parent_level),
    };
    scratch.unnamed.clear();
    scratch.unnamed_exact = false;
    scratch.marked = None;
    // Read from the scratch: a merge's change to any level forgets the
    // diagram's own answer, which takes a pass over every level to give
    // again.
    if super::sweep::diagram_marginal(tdd, scratch) {
        return group_twins_by_entries(eng, &entries, width, scratch);
    }
    let lim = eng.limits();
    // A stored child's node words, for the refinement to mark the members
    // of the groups it lists as the merge would.
    let child = level.stored().map(|stored| stored.nodes());
    let mut by_fingerprints = screen && width < NARROW_SCREENED_WIDTH;
    let refines = !by_fingerprints
        && (!screen
            || match screen_twins(parent_level, t1_side, lim, scratch)? {
                Screen::Alike => true,
                Screen::Named => {
                    // A level the diagram does not list loose has every node
                    // named. The list is scanned, which costs a search that
                    // reads few pairs more than the search itself: read it
                    // here only.
                    let loose = tdd.dirty.loose().is_none_or(|loose| loose.contains(&t1.0));
                    match (loose, parent_level.stored()) {
                        (true, Some(stored)) => {
                            mark_named(stored, t1_side, width, lim, &mut scratch.twin_named)?;
                            list_unnamed(width, lim, scratch)?;
                        }
                        _ => scratch.unnamed_exact = !loose,
                    }
                    false
                }
                Screen::Apart => false,
            });
    let found = if by_fingerprints {
        group_twins_by_entries(eng, &entries, width, scratch)?
    } else if refines {
        match refine::refine_context_groups(
            parent_level,
            t1_side,
            lim,
            scratch,
            width,
            Start::Whole { hand_off: true },
            child,
        )? {
            Some(found) => found,
            // A large parent node with its pairs in no order: the
            // fingerprints search the level for less.
            None => {
                by_fingerprints = true;
                group_twins_by_entries(eng, &entries, width, scratch)?
            }
        }
    } else {
        scratch.flat_groups.clear();
        scratch.group_starts.clear();
        false
    };
    if cfg!(debug_assertions) {
        list_members(lim, scratch)?;
        let kept = (scratch.group_starts.clone(), scratch.flat_groups.clone());
        let exact = scratch.unnamed_exact;
        let unnamed = std::mem::take(&mut scratch.unnamed);
        let other = if by_fingerprints {
            refine::refine_context_groups(
                parent_level,
                t1_side,
                lim,
                scratch,
                width,
                Start::Whole { hand_off: false },
                child,
            )?
            .expect("a refinement that may not give up does not")
        } else {
            group_twins_by_entries(eng, &entries, width, scratch)?
        };
        list_members(lim, scratch)?;
        assert_eq!(
            (found, &kept.0, &kept.1),
            (other, &scratch.group_starts, &scratch.flat_groups),
            "the refined twin groups differ from the fingerprinted ones",
        );
        // Each lists the nodes no pair names where it read every pair, and a
        // search that knows every node it did not list named lists them all.
        if (exact || !unnamed.is_empty()) && !scratch.unnamed.is_empty() {
            assert_eq!(
                unnamed, scratch.unnamed,
                "the refinement and the fingerprints list different unnamed nodes"
            );
        }
        scratch.unnamed = unnamed;
        scratch.unnamed_exact = exact;
    }
    Ok(found)
}

/// Group the `width` nodes of `level` by the multiset of their `entries`.
///
/// First compute additive fingerprints and discard nodes with unique
/// fingerprints. Only collision candidates receive full sorted signatures;
/// equal signatures form a group. Groups are stored in `scratch.flat_groups`,
/// indexed by `scratch.group_starts`, each group's members in ascending index
/// order so its first member is the lowest index. Returns whether any group
/// contains at least two nodes.
pub(super) fn group_twins_by_entries(
    eng: &Engine,
    entries: &impl TwinEntries,
    width: usize,
    scratch: &mut ContractScratch,
) -> Result<bool, OperationError> {
    let lim = eng.limits();
    scratch.flat_groups.clear();
    scratch.group_starts.clear();
    scratch.unnamed.clear();
    if width == 0 {
        return Ok(false);
    }
    // Node indices at this level are written u32-wide below (`cursors`,
    // `flat_groups`). That is the same invariant the merge side already relies
    // on (`final_remap: Vec<NodeIdx>`): every
    // ref into a level is a `NodeIdx(u32)`, so a level wider than 2^32
    // slots could not be referenced at all.
    debug_assert!(
        width <= u32::MAX as usize,
        "level width {width} exceeds the u32 node-index range",
    );

    if entries.no_twin(lim, scratch, width)? {
        return Ok(false);
    }
    // A wide level whose nodes hold one entry each groups by sorting them.
    if width >= groups::SINGLE_ENTRY_MIN_WIDTH
        && let Some(found) = groups::group_single_entries(eng, entries, width, scratch)?
    {
        return Ok(found);
    }

    // ── Pre-test: fingerprint-only scatter ────────────────────────────────────
    //
    // The common case is "no twins at this level", so only the additive
    // fingerprint is written here (no counts), touching half the cache lines
    // per entry. Accumulation is `wrapping_add`, which commutes, so the
    // result is order-independent; duplicate entries contribute 2h rather
    // than cancelling as exclusive-or would, so even-multiplicity duplicates
    // (legal at marginal boundary levels) cannot collapse a fingerprint to 0.
    // Counts are computed in a second pass only after a collision.
    lim.try_resize(&mut scratch.fingerprints, width, 0u64)?;
    scratch.fingerprints[..width].fill(0);
    entries.for_each(|node, entry| {
        let fp = &mut scratch.fingerprints[node as usize];
        *fp = fp.wrapping_add(entry_hash(entry));
    });

    // ── Fingerprint collision check + candidate marking ───────────────────────
    //
    // Every node that shares its fingerprint with an earlier one is marked a
    // twin candidate; no collision means no twins, the common case. Marking
    // scans the full width (no early exit on the first collision) so that
    // `build_twin_groups_after_collision` can skip the unique-fingerprint
    // nodes in its scatters.
    if !mark_candidates(eng, scratch, width)? {
        return Ok(false);
    }

    build_twin_groups_after_collision(eng, entries, width, scratch)
}

/// Size the open-addressing twin table for a pass that inserts at most
/// `max_occupancy` entries: the one sizing rule for `scratch.twin_hash_table`,
/// read by `probe_fingerprints`.
///
/// # Soundness
///
/// The probe loop is linear and never deletes, so a probe for a fingerprint
/// meets every equal-fingerprint entry before an empty slot whatever the size.
/// The size must exceed `max_occupancy`, or a probe for an absent fingerprint
/// wraps forever; the `div_ceil` term is at least 1, so it does.
///
/// Rounding to `4/3 · max_occupancy` caps the load factor at 3/4.
#[inline]
fn twin_table_size(max_occupancy: usize) -> usize {
    (max_occupancy + max_occupancy.div_ceil(3))
        .next_power_of_two()
        .max(4)
}

/// The fewest nodes a level has for [`mark_candidates`] to partition its
/// fingerprints: from here the probe table, 21 to 43 bytes a node, outgrows
/// a core's cache, and a probe misses it on nearly every node.
const PARTITION_MIN_WIDTH: usize = 1 << 16;

/// About how many fingerprints one partition holds: its records, 16 bytes
/// each, and its probe table, 8 to 16 bytes a fingerprint, stay in a core's
/// second-level cache.
const PARTITION_SIZE: usize = 1 << 14;

/// Mark twin candidates among `scratch.fingerprints[..width]`: every node whose
/// fingerprint is shared with ≥1 other node is flagged in
/// `scratch.is_candidate`. Returns whether any node was flagged; `false` means
/// all fingerprints are distinct, hence no twins.
///
/// A level of at least [`PARTITION_MIN_WIDTH`] nodes has its fingerprints
/// split by the high bits of their [`spread`] keys into partitions of about
/// [`PARTITION_SIZE`] first, a counting pass and a scattering pass, each
/// read in sequence; equal fingerprints share a partition, and each
/// partition is probed in a table of its own that stays in the cache. There
/// a key is kept beside its node in one word, short of its low bits, so two
/// nodes whose keys differ only there are flagged too; the groups compare
/// candidates in full.
#[inline]
fn mark_candidates(
    eng: &Engine,
    scratch: &mut ContractScratch,
    width: usize,
) -> Result<bool, OperationError> {
    let lim = eng.limits();
    if width >= PARTITION_MIN_WIDTH {
        return mark_partitioned_candidates(lim, scratch, width);
    }
    let ContractScratch {
        fingerprints,
        twin_hash_table,
        is_candidate,
        ..
    } = scratch;
    lim.try_resize(is_candidate, width, false)?;
    is_candidate[..width].fill(false);
    let mut found = false;
    probe_fingerprints(
        lim,
        twin_hash_table,
        &fingerprints[..width],
        None,
        |i, occupant| {
            if let Some(occ) = occupant {
                // Idempotent stores: re-flagging an already-flagged node is a
                // redundant write, not a miscount, so the probe needs no guards.
                is_candidate[i] = true;
                is_candidate[occ] = true;
                found = true;
            }
            true
        },
    )?;
    Ok(found)
}

/// [`mark_candidates`] on a wide level: its fingerprints counted and then
/// placed by their high bits, so that each partition is probed alone.
fn mark_partitioned_candidates(
    lim: &crate::limits::Limits,
    scratch: &mut ContractScratch,
    width: usize,
) -> Result<bool, OperationError> {
    let ContractScratch {
        fingerprints,
        is_candidate,
        part_records,
        part_starts,
        part_table,
        ..
    } = scratch;
    lim.try_resize(is_candidate, width, false)?;
    is_candidate[..width].fill(false);
    let fingerprints = &fingerprints[..width];
    let bits = (width / PARTITION_SIZE)
        .next_power_of_two()
        .trailing_zeros();
    let shift = u64::BITS - bits;
    let parts = 1usize << bits;
    // Each partition's size at the entry after its own, then their sums:
    // the start of each, and the end of the last.
    lim.try_resize(part_starts, parts + 1, 0)?;
    let starts = &mut part_starts[..parts + 1];
    starts.fill(0);
    for &fp in fingerprints {
        starts[(spread(fp) >> shift) as usize + 1] += 1;
    }
    let (mut widest, mut sum) = (0, 0);
    for start in starts.iter_mut() {
        widest = widest.max(*start);
        sum += *start;
        *start = sum;
    }
    // Placed in node order, with each start as the partition's cursor: the
    // cursors end at the next partition's start, so each start is the one
    // before it when the pass is done. A record is one word: the key
    // without its partition bits, its low bits replaced by the node, and
    // its high bits the slot it probes from. Two nodes whose keys differ
    // only in those low bits are marked too, which the groups' full
    // comparison then separates.
    let node_bits = u64::BITS - (width as u64).leading_zeros();
    let node_mask = (1u64 << node_bits) - 1;
    lim.try_resize(part_records, width, 0)?;
    for (node, &fp) in fingerprints.iter().enumerate() {
        let key = spread(fp);
        let p = (key >> shift) as usize;
        let at = starts[p] as usize;
        part_records[at] = (key << bits) & !node_mask | node as u64;
        starts[p] += 1;
    }
    // The partitions in turn, each probed linearly in a table of at least
    // twice its size that holds the records themselves: a probe ends at an
    // empty slot or at the first earlier record with its key, whose node
    // and the probing one are both marked. A node is below `node_mask`, so
    // no record is the empty slot's all-ones word.
    let size = (2 * widest as usize).next_power_of_two();
    lim.try_resize(part_table, size, u64::MAX)?;
    let mut found = false;
    let mut begin = 0usize;
    for &end in &starts[..parts] {
        let end = end as usize;
        if end - begin >= 2 {
            found |= probe_partition_records(
                &part_records[begin..end],
                part_table,
                node_bits,
                node_mask,
                is_candidate,
            );
        }
        begin = end;
    }
    Ok(found)
}

/// Probe one partition's scattered `records` in `part_table` and flag any pair
/// of nodes whose keys collide in `is_candidate`.
#[inline]
fn probe_partition_records(
    records: &[u64],
    part_table: &mut [u64],
    node_bits: u32,
    node_mask: u64,
    is_candidate: &mut [bool],
) -> bool {
    let mask = (2 * records.len()).next_power_of_two() - 1;
    let slot_shift = u64::BITS - (mask + 1).trailing_zeros();
    let table = &mut part_table[..=mask];
    table.fill(u64::MAX);
    let mut found = false;
    for &record in records {
        let key = record >> node_bits;
        let mut slot = (record >> slot_shift) as usize;
        loop {
            let held = table[slot];
            if held == u64::MAX {
                table[slot] = record;
                break;
            }
            if held >> node_bits == key {
                is_candidate[(record & node_mask) as usize] = true;
                is_candidate[(held & node_mask) as usize] = true;
                found = true;
                break;
            }
            slot = (slot + 1) & mask;
        }
    }
    found
}

/// How many fingerprints ahead the probe loop prefetches.
const PF_DIST: usize = 8;

/// Insert eligible `fingerprints` one by one into the open-addressing twin table `ht`,
/// sized and cleared here for that occupancy, probing linearly from each
/// fingerprint's home slot, the high bits of its [`spread`] key. For fingerprint `i` the probe calls
/// `visit(i, Some(j))` at every occupant `j` with the same fingerprint, in
/// probe order, and stops at the first call that returns `true`; at an empty
/// slot it inserts `i` and calls `visit(i, None)`, whose result is ignored.
/// The table stores the fingerprint beside the occupant index, so each probe
/// is one random load. Both twin-grouping passes run on this loop.
///
/// When `candidates` is supplied, unmarked nodes call `visit(i, None)` without
/// probing or occupying a slot. The table is sized for the marked nodes alone.
///
/// # Errors
///
/// `Err(OperationError::OverBudget)` if the table cannot be resized.
#[inline]
fn probe_fingerprints(
    lim: &crate::limits::Limits,
    ht: &mut Vec<TwinSlot>,
    fingerprints: &[u64],
    candidates: Option<&[bool]>,
    mut visit: impl FnMut(usize, Option<usize>) -> bool,
) -> Result<(), OperationError> {
    let width = fingerprints.len();
    debug_assert!(candidates.is_none_or(|c| c.len() == width));
    // One insert at most per eligible fingerprint.
    let occupancy = candidates.map_or(width, |c| c.iter().filter(|&&marked| marked).count());
    let table_size = twin_table_size(occupancy);
    let mask = table_size - 1;
    let shift = u64::BITS - table_size.trailing_zeros();
    let home = |fp: u64| (spread(fp) >> shift) as usize;
    lim.try_resize(ht, table_size, EMPTY_SLOT)?;
    ht[..table_size].fill(EMPTY_SLOT);
    for i in 0..width {
        if candidates.is_some_and(|c| !c[i]) {
            visit(i, None);
            continue;
        }
        // Prefetch the slot that iteration `i + PF_DIST` will first probe:
        // `fingerprints` is read sequentially, so that slot's address is known
        // ahead of time, and the probe is a random access into a table that
        // typically misses L2. The pointer is re-derived from `ht` each
        // iteration because the insert below writes through `ht`; `as_ptr`
        // is a field read and the prefetch is a hint, so nothing here is a
        // real load.
        if i + PF_DIST < width && candidates.is_none_or(|c| c[i + PF_DIST]) {
            prefetch_slot(ht.as_ptr(), home(fingerprints[i + PF_DIST]));
        }
        let fp = fingerprints[i];
        let mut slot = home(fp);
        loop {
            let s = ht[slot];
            if s.idx == u64::MAX {
                ht[slot] = TwinSlot { fp, idx: i as u64 };
                visit(i, None);
                break;
            }
            if s.fp == fp && visit(i, Some(s.idx as usize)) {
                break;
            }
            slot = (slot + 1) & mask;
        }
    }
    Ok(())
}

use groups::build_twin_groups_after_collision;
