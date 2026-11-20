//! Prune phase: remove diagram nodes not reachable from the output.
//!
//! Marks reachability top-down from the output node, then compacts the levels
//! it walked bottom-up while remapping child references. Levels that lost a
//! node go onto the contract worklists; `reduce` runs the contraction.
//!
//! [`PruneScope`] says how many levels that is: every one of them, or only the
//! ones a change at the root can have reached.
//!
//! A mark is one bit per slot, each level's marks a block of whole words. The
//! marking writes two marks per pair it reads, scattered over the slots of
//! the child levels; a bit per slot keeps those writes in a block a
//! thirty-second the size of one `u32` per slot. A level that keeps every
//! slot is left as it is. Only a level that lost a slot gets the new indices
//! of its survivors, which its parent's compaction builds from the level's
//! bits and rewrites its references through. Only a structural internal
//! level has a block: a leaf or marginal level is never compacted, so
//! nothing reads its marks. The read that settles a loose level
//! ([`settle_loose`]) counts instead, in a `u32` a slot, the pairs that name
//! each node of the children it reads, one level at a time.

use crate::diagram::{
    ChildDecoder, ChildKept, ChildPair, ChildSide, EncodedChildRef, EncodedNode, ImplicitLevel,
    NodeIdx, NodeKind, StoreRoom, Tdd, TddLevel, slots_below,
};

use crate::Engine;
use super::Unnamed;

use crate::execution::pool::{Buffers, PooledScratch, Scratch};
use crate::limits::{Limits, OperationError, PollGate, TimeTests, Transient};
use crate::vtree::{Vtree, VtreeIdx};

/// The new index of a slot that did not survive, in the remap a parent
/// rewrites its references through. Every survivor's index is below the
/// level's width, and a level of `u32::MAX` nodes cannot be allocated.
pub(super) const UNREACHED: u32 = u32::MAX;

/// The words a block of marks for `width` slots takes.
#[inline]
fn words(width: usize) -> usize {
    width.div_ceil(64)
}

/// Mark slot `s` of the block at word `base`.
#[inline(always)]
fn mark(marks: &mut [u64], base: usize, s: usize) {
    marks[base + (s >> 6)] |= 1u64 << (s & 63);
}

/// Marks into one block, holding the word the last mark went to in a register
/// and writing it back when a mark leaves that word.
///
/// A node's pairs mostly name their children's slots in ascending runs, so
/// consecutive marks into one child tend to land in one word. OR-ing each
/// into memory would make every such mark wait for the store of the one
/// before it; a run is gathered in the register and stored once.
struct Marker {
    /// The word of `marks` the pending bits belong to.
    word: usize,
    /// Marks not yet written to `word`.
    bits: u64,
}

impl Marker {
    /// A marker for the block at word `base`, with nothing pending.
    const fn new(base: usize) -> Marker {
        Marker {
            word: base,
            bits: 0,
        }
    }

    /// Mark slot `s` of the block at word `base`.
    #[inline(always)]
    fn mark(&mut self, marks: &mut [u64], base: usize, s: usize) {
        let word = base + (s >> 6);
        if word != self.word {
            marks[self.word] |= self.bits;
            self.word = word;
            self.bits = 0;
        }
        self.bits |= 1u64 << (s & 63);
    }

    /// Write the pending marks.
    #[inline(always)]
    fn finish(self, marks: &mut [u64]) {
        if self.bits != 0 {
            marks[self.word] |= self.bits;
        }
    }
}

/// The marked slots of `block`.
#[inline]
fn count(block: &[u64]) -> usize {
    block.iter().map(|w| w.count_ones() as usize).sum()
}

/// The word of a block whose first `n` slots are marked, `1 <= n <= 64`.
#[inline(always)]
fn tail_mask(n: usize) -> u64 {
    u64::MAX >> (64 - n)
}

/// Mark every one of the `width` slots of `block`, leaving the bits past
/// `width` in the last word clear.
fn mark_all(block: &mut [u64], width: usize) {
    let full = width >> 6;
    block[..full].fill(u64::MAX);
    if width & 63 != 0 {
        block[full] = tail_mask(width & 63);
    }
}

/// Whether every one of the `width` slots of `block` is marked. Bits past
/// `width` in the last word are never set.
fn all_marked(block: &[u64], width: usize) -> bool {
    let full = width >> 6;
    let tail = width & 63;
    block[..full].iter().all(|&w| w == u64::MAX) && (tail == 0 || block[full] == tail_mask(tail))
}

/// The index of every marked slot of `block`, ascending.
fn marked(block: &[u64]) -> Marked<'_> {
    Marked {
        block,
        word: 0,
        left: block.first().copied().unwrap_or(0),
    }
}

/// The marked slots of a block, ascending ([`marked`]).
struct Marked<'a> {
    block: &'a [u64],
    /// The word read, and its marks not yet given.
    word: usize,
    left: u64,
}

impl Iterator for Marked<'_> {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<usize> {
        while self.left == 0 {
            self.word += 1;
            self.left = *self.block.get(self.word)?;
        }
        let bit = self.left.trailing_zeros() as usize;
        self.left &= self.left - 1;
        Some((self.word << 6) + bit)
    }
}

/// Call `f` with the index of every unmarked one of the `width` slots of
/// `block`, ascending.
fn for_each_unmarked(block: &[u64], width: usize, mut f: impl FnMut(usize)) {
    for (w, &word) in block[..words(width)].iter().enumerate() {
        let mut x = !word;
        if (w + 1) << 6 > width {
            x &= tail_mask(width & 63);
        }
        while x != 0 {
            f((w << 6) + x.trailing_zeros() as usize);
            x &= x - 1;
        }
    }
}

/// The marked slots of `block` below slot `s`: the index slot `s` keeps once
/// the unmarked ones are dropped.
fn rank(block: &[u64], s: usize) -> u32 {
    let below: u32 = block[..s >> 6].iter().map(|w| w.count_ones()).sum();
    below + (block[s >> 6] & ((1u64 << (s & 63)) - 1)).count_ones()
}

/// The marked slots of a block of marks by their rank, read by a cursor
/// that moves forward a word at a time, and within its word a mark at a
/// time, and starts over when asked for a rank before it: rising ranks
/// cost a pass over the block and a step per mark in all.
pub(super) struct Select<'a> {
    block: &'a [u64],
    /// The word the cursor is at, and the marked slots before it.
    word: usize,
    before: usize,
    /// The word's marks the cursor has not passed, and how many it has.
    left: u64,
    passed: usize,
}

impl<'a> Select<'a> {
    pub(super) fn new(block: &'a [u64]) -> Select<'a> {
        Select {
            block,
            word: 0,
            before: 0,
            left: block.first().copied().unwrap_or(0),
            passed: 0,
        }
    }

    /// The marked slot of rank `j`, if there are more than `j`.
    pub(super) fn nth(&mut self, j: usize) -> Option<usize> {
        if j < self.before {
            *self = Select::new(self.block);
        } else if j < self.before + self.passed {
            (self.left, self.passed) = (self.block[self.word], 0);
        }
        while let Some(&w) = self.block.get(self.word) {
            let marked = w.count_ones() as usize;
            if j < self.before + marked {
                while self.before + self.passed < j {
                    self.left &= self.left - 1;
                    self.passed += 1;
                }
                return Some(self.word * 64 + self.left.trailing_zeros() as usize);
            }
            self.before += marked;
            self.word += 1;
            (self.left, self.passed) = (self.block.get(self.word).copied().unwrap_or(0), 0);
        }
        None
    }
}

/// Write into `out` the new index of each of the first `span` slots of
/// `block`: its rank among the marked ones, or [`UNREACHED`].
///
/// A word at a time: a word of marked slots numbers its 64 slots in a run, a
/// word of none leaves them unreached, and only a mixed word is read a slot
/// at a time.
pub(super) fn new_indices(block: &[u64], span: usize, out: &mut [u32]) {
    let mut next = 0u32;
    for (word, slots) in block.iter().zip(out[..span].chunks_mut(64)) {
        let full = tail_mask(slots.len());
        let word = word & full;
        if word == full {
            for (slot, index) in slots.iter_mut().zip(next..) {
                *slot = index;
            }
            next += slots.len() as u32;
        } else if word == 0 {
            slots.fill(UNREACHED);
        } else {
            for (b, slot) in slots.iter_mut().enumerate() {
                let marked = (word >> b & 1) as u32;
                *slot = if marked != 0 { next } else { UNREACHED };
                next += marked;
            }
        }
    }
}

/// The buffers one prune works in, checked out together.
#[derive(Default)]
pub(crate) struct PruneScratch {
    /// The reachability marks, one bit per slot.
    marks: Vec<u64>,
    /// The new indices a parent rewrites its references into a child level
    /// that lost a node through.
    remap: Vec<u32>,
    /// The levels the seeded walk descended into.
    visits: Vec<Visit>,
    /// Per level, whether a walk from the output would enter it, clear
    /// between prunes, and those levels in the order it would
    /// ([`slots_from_output`]).
    entered: Vec<bool>,
    from_output: Vec<VtreeIdx>,
    /// [`settle_loose`]'s flags per level and the parents it reads.
    settle_flags: Vec<[bool; 2]>,
    settle_parents: Vec<VtreeIdx>,
    /// [`settle_loose`]'s count of the pairs that name each slot of the
    /// children it reads ([`count_named`]).
    settle_counts: Vec<u32>,
    /// Marks a byte per slot, those a walk sets in a level's children
    /// before it packs them into bits ([`mark_sides`]).
    bytes: Vec<u8>,
    /// [`prune_complement`]'s levels that lose a node, top-down: the level,
    /// the slot of the node it loses, and the entry of its parent level.
    chain: Vec<[u32; 3]>,
    /// [`prune_complement`]'s stamped count of the pairs the dying node of a
    /// level gives each node of its children ([`true_children`]).
    row: Vec<u64>,
}

impl Buffers for PruneScratch {
    fn buffers(&mut self, visit: &mut dyn FnMut(&mut dyn Scratch)) {
        visit(&mut self.marks);
        visit(&mut self.remap);
        visit(&mut self.visits);
        visit(&mut self.entered);
        visit(&mut self.from_output);
        visit(&mut self.settle_flags);
        visit(&mut self.settle_parents);
        visit(&mut self.settle_counts);
        visit(&mut self.bytes);
        visit(&mut self.chain);
        visit(&mut self.row);
    }
}

impl PooledScratch for PruneScratch {
    /// Never called: every checkout of the prune's pool is
    /// `checkout_preserving`.
    fn prepare(&mut self) {}
}

/// The buffers of [`drop_clause_orphans`], pooled apart from the prune's
/// so a clause walk checks out and parks only these.
#[derive(Default)]
pub(crate) struct DropScratch {
    marks: Vec<u64>,
    remap: Vec<u32>,
    /// The levels that lost nodes and are still to be read, with the sides
    /// to read.
    lost: Vec<(VtreeIdx, [bool; 2])>,
}

impl Buffers for DropScratch {
    fn buffers(&mut self, visit: &mut dyn FnMut(&mut dyn Scratch)) {
        visit(&mut self.marks);
        visit(&mut self.remap);
        visit(&mut self.lost);
    }
}

impl PooledScratch for DropScratch {
    /// Never called: the pool is checked out with `checkout_preserving`.
    fn prepare(&mut self) {}
}

/// How much of the diagram a prune has to walk.
pub(crate) enum PruneScope {
    /// Every level, assuming nothing about how the nodes became unreachable.
    Whole,
    /// Only what a change at the root can have made unreachable.
    ///
    /// Valid when every node of every non-root level is referenced by a node of
    /// its parent level — reachable or not — and the output is the root. A
    /// level that then loses no node has a subtree that loses no node: each of
    /// its children's slots is named by one of its own surviving nodes, and so
    /// on down. The walk stops at such a level and never looks below it.
    ///
    /// A negation establishes the condition: `expand_full` leaves every
    /// structural level below the root covering the whole `lefts x rights`
    /// basis of its children, and the old output and its complement together
    /// cover the root's, so every child slot is named by some pair of the
    /// level above.
    BelowRoot,
}

/// Remove nodes not reachable from the output.
///
/// Marks reachability top-down, then compacts each level bottom-up, remapping
/// child references in the same pass. Every level that lost a node is pushed
/// onto the contract worklists (`invalidate_shrunk`), so the caller needs no
/// reseed.
///
/// `scope` says how much of the diagram has to be walked; see [`PruneScope`].
/// A `BelowRoot` scope on a diagram that is not the shape that walk starts
/// from — an output below the root, a leaf or marginal root level — walks the
/// whole diagram instead. A `Whole` scope on a structural diagram that knows
/// its loose levels ([`Dirty::loose`](crate::diagram::Dirty::loose)) walks
/// down through the levels above them, as [`settle_loose`] leaves them, and
/// the levels that lose a node, starting below the levels it knows are all
/// reached ([`Seeds`]): where the output is the root level's only node, the
/// levels with no loose level on their path from the root. A level the walk
/// leaves has lost no node and has no loose level under it, so each node
/// below it is named by a surviving node of its parent level. The prune
/// leaves no level loose.
///
/// # Errors
///
/// Returns `Err(OperationError::OverBudget)` if a reservation of the marks,
/// or of the new indices the compaction rewrites references through, is
/// refused, and the stop the settle of the loose levels polls for
/// ([`settle_loose`]). Every one comes before any mutation of `tdd`, so
/// the diagram is then untouched.
pub(crate) fn prune_unreachable(
    eng: &Engine,
    tdd: &mut Tdd,
    scope: PruneScope,
) -> Result<(), OperationError> {
    // `ZERO` sentinel: the entire diagram computes ⊥ (UNSAT). No nodes are reachable.
    if tdd.is_zero() {
        for level in &mut tdd.levels {
            level.nodes = level.take_nodes();
            level.pairs.clear();
            level.ranges.clear();
            level.dead_pairs = 0;
        }
        for t in 0..tdd.levels.len() {
            tdd.levels.mark_changed(VtreeIdx(t as u32));
        }
        tdd.dirty.set_loose(Some(Vec::new()));
        return Ok(());
    }

    let below_root = matches!(scope, PruneScope::BelowRoot) && below_root_walk_applies(tdd);
    let seeded = match tdd.dirty.loose() {
        Some(loose) if !below_root && below_root_walk_applies(tdd) && !tdd.has_marginal_level() => {
            match tdd.dirty.counted_loose() {
                Some(counted) => Some(Seeds::of(eng, tdd, counted)?),
                None if settle_reads_half(tdd, loose) => None,
                None => Some(Seeds::of(eng, tdd, &settle_loose(eng, tdd, loose)?)?),
            }
        }
        _ => None,
    };
    if below_root {
        prune_walk(eng, tdd, None, Start::Output)?;
    } else if let Some(seeds) = &seeded {
        let start = match seeds.root_clean {
            true => Start::Seeds(&seeds.seeds[..], &seeds.state[..], &seeds.top[..]),
            false => Start::Output,
        };
        prune_walk(eng, tdd, Some(&seeds.state[..]), start)?;
        #[cfg(debug_assertions)]
        debug_assert_all_reached(tdd);
    } else {
        prune_walk(eng, tdd, None, Start::Root)?;
    }
    tdd.dirty.set_loose(Some(Vec::new()));
    Ok(())
}

/// Whether settling `listed` may read as many pairs as a walk from the root
/// reads: then the walk costs less than the settle and the seeded walk
/// after it. The bound counts a parent once for each listed child.
fn settle_reads_half(tdd: &Tdd, listed: &[u32]) -> bool {
    let bound = |t: VtreeIdx| tdd.levels[t.idx()].pairs_bound();
    let read: usize = listed
        .iter()
        .filter_map(|&t| tdd.vtree.node(VtreeIdx(t)).parent())
        .map(bound)
        .sum();
    let all: usize = (0..tdd.vtree.num_nodes() as u32).map(|t| bound(VtreeIdx(t))).sum();
    read >= all
}

/// A level the walk must enter whatever it finds: a parent of a loose
/// level, as [`settle_loose`] leaves them, or a level above one.
const FORCED: u8 = 1;
/// A level every node of which the output reaches, known without a walk:
/// the root, when the output is its only node, and a level that is not
/// loose under such a level.
const CLEAN: u8 = 2;

/// Where a prune that knows its loose levels walks: the levels it must
/// enter ([`FORCED`]), and those every node of which the output reaches
/// ([`CLEAN`]), with the clean levels that have a child that is not, where
/// the walk starts.
///
/// A clean level is all reached. The root is, when the output is its only
/// node. A level `c` under a clean level `p` that is not loose has each of
/// its nodes named by a pair of some node of `p`, which is reached, so it is
/// reached too. Every level under one that is not clean is not clean, so
/// the clean levels are the top of the vtree, and the seeds are where the
/// rest hangs from it: a walk from the seeds, each with every node marked,
/// marks what one from the output would below them, and leaves alone the
/// clean levels, which lose no node and whose references into a clean
/// child need no rewrite.
pub(super) struct Seeds<'l> {
    /// Per level, [`FORCED`], and [`CLEAN`] on the clean levels that are
    /// forced or a seed's children: the walk reads it on a seed's children.
    pub(super) state: Transient<'l, Vec<u8>>,
    /// The structural internal clean levels with a structural internal
    /// child that is not clean, top-down.
    seeds: Transient<'l, Vec<VtreeIdx>>,
    /// The structural internal clean levels above a loose one, which a walk
    /// from the output enters ([`slots_from_output`]).
    top: Transient<'l, Vec<VtreeIdx>>,
    /// Whether the root is clean: otherwise nothing is, and the walk starts
    /// at the output.
    root_clean: bool,
}

impl<'l> Seeds<'l> {
    /// The seeds of `tdd`, whose loose levels are `loose`, as [`settle_loose`]
    /// leaves them. The output is the root, as the walk below the root
    /// needs ([`below_root_walk_applies`]).
    pub(super) fn of(
        eng: &'l Engine,
        tdd: &Tdd,
        loose: &[u32],
    ) -> Result<Seeds<'l>, OperationError> {
        let lim = eng.limits();
        let vtree = &tdd.vtree;
        let mut state: Transient<'_, Vec<u8>> = Transient::new(lim, Vec::new());
        lim.try_resize(&mut state, vtree.num_nodes(), 0)?;
        // The loose levels, and every level above one forced, listed.
        const LOOSE: u8 = 4;
        // Whether the level is clean is settled.
        const SETTLED: u8 = 8;
        const SEED: u8 = 16;
        let mut forced: Transient<'_, Vec<VtreeIdx>> = Transient::new(lim, Vec::new());
        for &level in loose {
            state[level as usize] |= LOOSE;
            let mut up = vtree.node(VtreeIdx(level)).parent();
            while let Some(p) = up {
                if state[p.idx()] & FORCED != 0 {
                    break;
                }
                state[p.idx()] |= FORCED;
                lim.try_push(&mut forced, p)?;
                up = vtree.node(p).parent();
            }
        }
        let root = vtree.root();
        let root_clean = tdd.levels[root.idx()].slot_count() == 1;
        let mut seeds: Transient<'_, Vec<VtreeIdx>> = Transient::new(lim, Vec::new());
        let mut top: Transient<'_, Vec<VtreeIdx>> = Transient::new(lim, Vec::new());
        if root_clean {
            // Whether each forced level is clean, settled from the root down
            // its path: the root is, and a level is when its parent is and
            // it is not loose. A level that is not forced has no loose level
            // under it, so the clean levels with a child that is not are
            // forced, and the walk reads whether a level is clean only on a
            // seed's children, below.
            let mut path: Transient<'_, Vec<VtreeIdx>> = Transient::new(lim, Vec::new());
            for &f in forced.iter() {
                path.clear();
                let mut t = f;
                let mut clean = loop {
                    if state[t.idx()] & SETTLED != 0 {
                        break state[t.idx()] & CLEAN != 0;
                    }
                    lim.try_push(&mut path, t)?;
                    match vtree.node(t).parent() {
                        Some(p) => t = p,
                        None => break true,
                    }
                };
                for &t in path.iter().rev() {
                    clean = t == root || (clean && state[t.idx()] & LOOSE == 0);
                    state[t.idx()] |= SETTLED | if clean { CLEAN } else { 0 };
                }
            }
            // A seed is a clean level with a child that is not: a loose
            // one.
            for &level in loose {
                let level = VtreeIdx(level);
                let Some(p) = vtree.node(level).parent() else {
                    continue;
                };
                if state[p.idx()] & (CLEAN | SEED) == CLEAN
                    && tdd.is_structural_internal(p)
                    && tdd.is_structural_internal(level)
                {
                    state[p.idx()] |= SEED;
                    lim.try_push(&mut seeds, p)?;
                }
            }
            seeds.sort_unstable_by_key(|&t| std::cmp::Reverse(vtree.topo_pos(t)));
            for &t in seeds.iter() {
                let (left, right) = vtree.children(t);
                for c in [left, right] {
                    if state[c.idx()] & LOOSE == 0 {
                        state[c.idx()] |= CLEAN;
                    }
                }
            }
            for &t in forced.iter() {
                if state[t.idx()] & CLEAN != 0 && tdd.is_structural_internal(t) {
                    lim.try_push(&mut top, t)?;
                }
            }
        }
        Ok(Seeds {
            state,
            seeds,
            top,
            root_clean,
        })
    }
}

/// Every slot of every structural internal level is reachable from the output.
#[cfg(debug_assertions)]
pub(crate) fn debug_assert_all_reached(tdd: &Tdd) {
    if tdd.is_zero() {
        return;
    }
    let (mut marks, mut visits) = (Vec::new(), Vec::new());
    mark_walk(
        &Limits::new(),
        tdd,
        None,
        Start::Root,
        &mut marks,
        &mut visits,
        &mut Vec::new(),
    )
    .expect("unbounded limits refuse nothing");
    for v in &visits {
        let width = tdd.levels[v.level.idx()].slot_count();
        assert!(
            all_marked(&marks[v.base..v.base + words(width)], width),
            "level {} keeps a node the output does not reach",
            v.level.0
        );
    }
}

/// Whether the seeded walk can start on `tdd` at all: it starts at the output,
/// which has to be the root, and the root level has to be one it can compact.
/// An output below the root leaves every level above it unreachable, which is
/// not a change below the root.
pub(crate) fn below_root_walk_applies(tdd: &Tdd) -> bool {
    let root = tdd.vtree.root();
    tdd.output.vtree == root
        && !tdd.vtree.node(root).is_leaf()
        && !tdd.levels[root.idx()].is_marginal()
}

// ── Settling the loose levels ────────────────────────────────────────────────

/// The levels of `listed`, loose levels of `tdd`
/// ([`Dirty::loose`](crate::diagram::Dirty::loose)), the walk from the
/// output has to enter the parents of: each that holds a node no pair of its
/// parent level names, and others only where the walk enters their parent
/// anyway, so that it enters the levels a list of exactly the first would
/// have it enter. The list keeps its order.
///
/// A listed level is read, its parent level's pairs until they have named
/// every node of it, unless its parent is the root, or a level under one of
/// the parent's children stays listed: the walk enters such a parent
/// whatever the list says. A level with no node is dropped, and one that
/// cannot be read stays listed: the root, a level that is not structural
/// internal, and one whose parent level is not. Listed siblings are read in
/// one pass, one unit of work a pair.
///
/// A listed level makes the walk enter every level above it, so one listed
/// in vain can cost a walk from the root, where reading it costs at most its
/// parent level's pairs.
///
/// # Errors
///
/// Returns `Err(OperationError::OverBudget)` if the marks a read takes are
/// refused, and the stop the work polls for.
pub(crate) fn settle_loose(
    eng: &Engine,
    tdd: &Tdd,
    listed: &[u32],
) -> Result<Vec<u32>, OperationError> {
    settle_loose_in(eng, &tdd.vtree, &|t| &tdd.levels[t.idx()], listed)
}

/// [`settle_loose`] of the diagram on `vtree` whose level at each node is
/// `level` of it.
pub(crate) fn settle_loose_in<'a>(
    eng: &Engine,
    vtree: &Vtree,
    level: &dyn Fn(VtreeIdx) -> &'a TddLevel,
    listed: &[u32],
) -> Result<Vec<u32>, OperationError> {
    if listed.is_empty() {
        return Ok(Vec::new());
    }
    let lim = eng.limits();
    // The prune's scratch, which a prune checks out after this returns: a
    // read clears the block of marks it takes.
    let mut scratch = eng.scratch.reduce.prune.checkout_preserving(eng);
    let PruneScratch {
        settle_counts: counts,
        settle_flags: flags,
        settle_parents: parents,
        ..
    } = &mut *scratch;
    // The listed levels' parents, bottom-up, so that every level under a
    // parent's children is settled before the parent is read.
    parents.clear();
    lim.reserve(parents, listed.len())?;
    parents.extend(
        listed
            .iter()
            .filter_map(|&t| vtree.node(VtreeIdx(t)).parent()),
    );
    parents.sort_unstable_by_key(|&p| vtree.topo_pos(p));
    parents.dedup();
    settle_under(lim, (vtree, level), listed, parents, flags, counts)
}

/// [`settle_loose_in`] reading only the listed levels whose parent is one of
/// `parents`, which are bottom-up and closed upward (the parent of each is
/// one of them, but for the root): the others stay listed, unread.
pub(crate) fn settle_loose_below<'a>(
    eng: &Engine,
    vtree: &Vtree,
    level: &dyn Fn(VtreeIdx) -> &'a TddLevel,
    listed: &[u32],
    parents: &[VtreeIdx],
) -> Result<Vec<u32>, OperationError> {
    if listed.is_empty() {
        return Ok(Vec::new());
    }
    let mut scratch = eng.scratch.reduce.prune.checkout_preserving(eng);
    let PruneScratch {
        settle_counts: counts,
        settle_flags: flags,
        ..
    } = &mut *scratch;
    settle_under(eng.limits(), (vtree, level), listed, parents, flags, counts)
}

/// The settle of `listed` that reads the children of each of `parents`,
/// bottom-up.
fn settle_under<'a>(
    lim: &Limits,
    (vtree, level): (&Vtree, &dyn Fn(VtreeIdx) -> &'a TddLevel),
    listed: &[u32],
    parents: &[VtreeIdx],
    flags: &mut Vec<[bool; 2]>,
    counts: &mut Vec<u32>,
) -> Result<Vec<u32>, OperationError> {
    // Per level, whether it stays listed (`AT`) and whether a level strictly
    // under it stays listed (`UNDER`): set on the levels above each one that
    // stays, up to the first set already, whose own levels above are set.
    const AT: usize = 0;
    const UNDER: usize = 1;
    flags.clear();
    lim.try_resize(flags, vtree.num_nodes(), [false; 2])?;
    for &t in listed {
        flags[t as usize][AT] = true;
    }
    let mut gate = lim.gate();
    for &p in parents {
        let (left, right) = vtree.children(p);
        if p != vtree.root() && !flags[left.idx()][UNDER] && !flags[right.idx()][UNDER] {
            let sides = [flags[left.idx()][AT], flags[right.idx()][AT]];
            let (stays, _) = unnamed_children(lim, (vtree, level), p, sides, counts, &mut gate)?;
            flags[left.idx()][AT] = stays[0];
            flags[right.idx()][AT] = stays[1];
        }
        if flags[left.idx()][AT] || flags[right.idx()][AT] {
            let mut up = Some(p);
            while let Some(t) = up.filter(|t| !flags[t.idx()][UNDER]) {
                flags[t.idx()][UNDER] = true;
                up = vtree.node(t).parent();
            }
        }
    }
    gate.flush()?;
    Ok(listed
        .iter()
        .copied()
        .filter(|&t| flags[t as usize][AT])
        .collect())
}

/// Which of the children of level `p` that `sides` asks for, left and right,
/// hold a node no pair of `p` names, reading `p`'s pairs in one pass until
/// they have named every node of each, one unit of work through `gate` a
/// pair ([`count_named`]). A level with no node holds none, and a level that
/// cannot be read counts as holding one: one that is not structural
/// internal, and each child of a `p` that is not.
///
/// Also the width of each side's block in `counts`, the left one's first, 0
/// for a side not read. A side read that holds an unnamed node was read to
/// the end, so its block's zeros below the child's node count are exactly
/// those nodes.
fn unnamed_children<'a>(
    lim: &Limits,
    (vtree, level_at): (&Vtree, &dyn Fn(VtreeIdx) -> &'a TddLevel),
    p: VtreeIdx,
    sides: [bool; 2],
    counts: &mut Vec<u32>,
    gate: &mut PollGate<'_>,
) -> Result<([bool; 2], [usize; 2]), OperationError> {
    let structural_internal = |t: VtreeIdx| !vtree.node(t).is_leaf() && !level_at(t).is_marginal();
    if sides == [false; 2] || !structural_internal(p) {
        return Ok((sides, [0; 2]));
    }
    let (left, right) = vtree.children(p);
    let children = [left, right];
    // Each side's block of counts, the left one's first: one a slot and one
    // past them ([`count_named`]), nothing for a side not read.
    let (mut width, mut unread) = ([0usize; 2], [false; 2]);
    for k in 0..2 {
        if !sides[k] {
            continue;
        }
        if !structural_internal(children[k]) {
            unread[k] = true;
            continue;
        }
        let slots = level_at(children[k]).slot_count();
        width[k] = if slots > 0 { slots + 1 } else { 0 };
    }
    let used = width[0] + width[1];
    if counts.len() < used {
        lim.try_resize(counts, used, 0)?;
    }
    counts[..used].fill(0);
    let level = level_at(p);
    let (l, r) = counts[..used].split_at_mut(width[0]);
    let named = match width.map(|n| n > 0) {
        [true, true] => count_named::<true, true>(level, [l, r], gate)?,
        [true, false] => count_named::<true, false>(level, [l, r], gate)?,
        [false, true] => count_named::<false, true>(level, [l, r], gate)?,
        [false, false] => [true; 2],
    };
    Ok(([unread[0] || !named[0], unread[1] || !named[1]], width))
}

/// [`settle_loose`] read to the end: each level of `listed` that holds a
/// node no pair of its parent level names, or cannot be read, with the
/// nodes it holds no pair names and their pairs ([`Engine::unnamed_nodes`]);
/// a level that cannot be read, and the root level, are given none. The
/// output is at the root and no level is marginal. Unlike the settle, a
/// level whose parent is the root or under a level that stays listed is read
/// too, each parent's pairs in one pass, one unit of work a pair.
pub(crate) fn count_unnamed(
    eng: &Engine,
    tdd: &Tdd,
    listed: &[u32],
) -> Result<Vec<(u32, Unnamed)>, OperationError> {
    let (vtree, lim) = (&tdd.vtree, eng.limits());
    let level_at = |t: VtreeIdx| &tdd.levels[t.idx()];
    let root = vtree.root();
    let mut held: Vec<(u32, Unnamed)> = Vec::new();
    if listed.contains(&root.0) && level_at(root).node_count() > 1 {
        lim.try_push(&mut held, (root.0, Unnamed::default()))?;
    }
    if listed.is_empty() {
        return Ok(held);
    }
    let mut scratch = eng.scratch.reduce.prune.checkout_preserving(eng);
    let PruneScratch {
        settle_counts: counts,
        settle_flags: flags,
        settle_parents: parents,
        ..
    } = &mut *scratch;
    flags.clear();
    lim.try_resize(flags, vtree.num_nodes(), [false; 2])?;
    for &t in listed {
        flags[t as usize][0] = true;
    }
    parents.clear();
    lim.reserve(parents, listed.len())?;
    parents.extend(listed.iter().filter_map(|&t| vtree.node(VtreeIdx(t)).parent()));
    parents.sort_unstable_by_key(|p| p.0);
    parents.dedup();
    lim.reserve(&mut held, listed.len())?;
    let mut gate = lim.gate();
    for &p in parents.iter() {
        let children = <[VtreeIdx; 2]>::from(vtree.children(p));
        let sides = children.map(|c| flags[c.idx()][0]);
        let (stays, width) = unnamed_children(lim, (vtree, &level_at), p, sides, counts, &mut gate)?;
        let mut base = 0;
        for k in 0..2 {
            if stays[k] {
                let mut unnamed = Unnamed::default();
                if width[k] > 0 {
                    let level = level_at(children[k]);
                    for (s, _) in counts[base..base + width[k] - 1].iter().enumerate().filter(|&(_, &n)| n == 0) {
                        unnamed.nodes += 1;
                        unnamed.pairs += level.pair_count_at(s);
                    }
                }
                held.push((children[k].0, unnamed));
            }
            base += width[k];
        }
    }
    gate.flush()?;
    for &t in listed {
        flags[t as usize][0] = false;
    }
    Ok(held)
}

/// Count in `blocks`, a count a node, the pairs of `level` that name each
/// node of its children, on the left side with `LEFT` and on the right with
/// `RIGHT`, and say whether each side has every node named; a side not read
/// has. Both children are structural internal, so a side is the index of
/// the node it names. Each block read holds a count past the child's nodes,
/// which no pair takes. One unit of work through `gate` a pair the level's
/// nodes hold, in node order, up to the first node past which every side
/// read has each of its nodes named.
///
/// A stored level whose nodes' ranges cover its arena, of fewer than
/// [`WHOLE_READ_PAIRS_A_SLOT`] pairs a node read for, is read whole, with
/// no branch a node ([`StoredPairs::for_each_pair_whole`]): the nodes' own
/// pairs, each node whose pairs are in the arena counted past the child's
/// nodes, and then the arena. Where every node read for is named, the work
/// is the level's pairs less those of the nodes after the last one to name
/// a node first: read back from the last node, each pair counted down, they
/// end at the first node that takes a count to zero. Any other level is
/// read node by node, each side keeping the first of its nodes not seen
/// counted, which only moves forward: the test before a node reads that
/// node's count, and the cursors cross each count once over the read. An
/// implicit level is read off its digits instead
/// ([`ImplicitLevel::each_slot_on`]), each count one past the first node
/// that names its slot, and charged the pairs of the nodes up to the last
/// of those, as the read node by node would be.
///
/// A count is added to in place and read only after the read, so no pair
/// waits on another: pairs that name the same node, or nodes a word apart,
/// do not wait on each other as a bit set in a shared word, or a count of
/// the nodes named so far, would.
///
/// [`StoredPairs::for_each_pair_whole`]: crate::diagram::StoredPairs::for_each_pair_whole
#[inline(always)]
fn count_named<const LEFT: bool, const RIGHT: bool>(
    level: &TddLevel,
    blocks: [&mut [u32]; 2],
    gate: &mut PollGate<'_>,
) -> Result<[bool; 2], OperationError> {
    let [left, right] = blocks;
    let past = [left.len().saturating_sub(1), right.len().saturating_sub(1)];
    // Whether every node of `block` is counted, moving `next` past those that are.
    fn all_named(block: &[u32], next: &mut usize) -> bool {
        while *next < block.len() && block[*next] != 0 {
            *next += 1;
        }
        *next == block.len()
    }
    // A level of many pairs a node read for names them all early: a read
    // node by node stops there, where the read whole reads every pair and
    // then back to the last node that names one first.
    let slots = if LEFT { past[0] } else { 0 } + if RIGHT { past[1] } else { 0 };
    let whole = |stored: &crate::diagram::StoredPairs<'_>| {
        !settle_by_node()
            && stored.nodes().len() + stored.arena().len() < WHOLE_READ_PAIRS_A_SLOT * slots
    };
    if let Some(stored) = level.stored().filter(whole) {
        let none = ChildPair::new(
            EncodedChildRef::from_raw(past[0] as u32),
            EncodedChildRef::from_raw(past[1] as u32),
        );
        if count_whole::<LEFT, RIGHT>(stored, none, left, right) {
            note_settle_read(0);
            let named = [
                !LEFT || !left[..past[0]].contains(&0),
                !RIGHT || !right[..past[1]].contains(&0),
            ];
            // Each node whose pairs are in the arena counted once past the nodes.
            let ranged = if LEFT { left[past[0]] } else { right[past[1]] } as usize;
            let pairs = stored.nodes().len() - ranged + stored.arena().len();
            let after = match named {
                [true, true] => pairs_after_last_first::<LEFT, RIGHT>(stored, left, right),
                _ => 0,
            };
            gate.poll((pairs - after) as u64)?;
            return Ok(named);
        }
        // The nodes' own pairs were counted before the arena was found
        // holding others.
        left.fill(0);
        right.fill(0);
    }
    if let Some(d) = level.implicit().filter(|_| !settle_by_node()) {
        note_settle_read(2);
        // Each slot's count is one past the first node that names it.
        let first = |block: &mut [u32], side| {
            d.each_slot_on(side, |s, node| {
                let count = &mut block[s as usize];
                *count = count.wrapping_sub(1).min(node as u32).wrapping_add(1);
            });
        };
        if LEFT {
            first(&mut left[..], ChildSide::Left);
        }
        if RIGHT {
            first(&mut right[..], ChildSide::Right);
        }
        let (left, right) = (&left[..past[0]], &right[..past[1]]);
        let named = [!LEFT || !left.contains(&0), !RIGHT || !right.contains(&0)];
        // Read node by node, the read takes every node up to the last one
        // to name a node first, or every node where one is left unnamed.
        let nodes = match named {
            [true, true] => left.iter().chain(right).copied().max().unwrap_or(0) as usize,
            _ => d.nodes(),
        };
        gate.poll((nodes * d.pairs_per_node()) as u64)?;
        return Ok(named);
    }
    note_settle_read(1);
    let (left, right) = (&mut left[..past[0]], &mut right[..past[1]]);
    let mut next = [0usize; 2];
    // `Err(None)` stops the read once every node read for is named.
    let read = level.try_for_each_node(0..level.nodes().len(), |_, pairs| {
        if (!LEFT || all_named(left, &mut next[0])) && (!RIGHT || all_named(right, &mut next[1])) {
            return Err(None);
        }
        gate.poll(pairs.len() as u64).map_err(Some)?;
        for pair in pairs {
            if LEFT {
                left[pair.left.0 as usize] = 1;
            }
            if RIGHT {
                right[pair.right.0 as usize] = 1;
            }
        }
        Ok(())
    });
    if let Err(Some(e)) = read {
        return Err(e);
    }
    Ok([
        !LEFT || all_named(left, &mut next[0]),
        !RIGHT || all_named(right, &mut next[1]),
    ])
}

/// The pairs a node read for below which [`count_named`] reads a stored
/// level whole: past them a read node by node stops early.
const WHOLE_READ_PAIRS_A_SLOT: usize = 16;

/// [`count_named`]'s read of a stored level whole
/// ([`StoredPairs::for_each_pair_whole`]), `none` naming the counts past
/// each side's nodes. Out of line, so that its loops keep the level and
/// both blocks in registers.
///
/// [`StoredPairs::for_each_pair_whole`]: crate::diagram::StoredPairs::for_each_pair_whole
#[inline(never)]
fn count_whole<const LEFT: bool, const RIGHT: bool>(
    stored: crate::diagram::StoredPairs<'_>,
    none: ChildPair,
    left: &mut [u32],
    right: &mut [u32],
) -> bool {
    stored.for_each_pair_whole(none, |pair| {
        if LEFT {
            left[pair.left.0 as usize] += 1;
        }
        if RIGHT {
            right[pair.right.0 as usize] += 1;
        }
    })
}

/// The pairs of the nodes of `stored` after the last one that names a node
/// of a side read first, `left` and `right` holding how many pairs name each
/// node: read back from the last node, each pair counted down, up to the
/// first node that takes a count to zero, which names that node first. The
/// counts are spent.
#[inline(never)]
fn pairs_after_last_first<const LEFT: bool, const RIGHT: bool>(
    stored: crate::diagram::StoredPairs<'_>,
    left: &mut [u32],
    right: &mut [u32],
) -> usize {
    let mut after = 0;
    for node in stored.nodes().iter().rev() {
        let pairs = stored.of(node);
        let mut first = false;
        for pair in pairs {
            if LEFT {
                let c = &mut left[pair.left.0 as usize];
                *c -= 1;
                first |= *c == 0;
            }
            if RIGHT {
                let c = &mut right[pair.right.0 as usize];
                *c -= 1;
                first |= *c == 0;
            }
        }
        if first {
            break;
        }
        after += pairs.len();
    }
    after
}

/// Where a child level's marks are, as the parent rewriting its references
/// reads them.
#[derive(Clone, Copy)]
struct Child {
    /// The word its block of marks starts at; unused when it kept every slot.
    base: usize,
    /// The slots its new indices are written for, the room they take: its
    /// width before the prune at least.
    span: usize,
    /// It lost a slot, so the parent rewrites its references through the new
    /// indices of the survivors; otherwise they stay as they are.
    dirty: bool,
}

impl Child {
    /// A child with its own block of marks.
    const fn at(base: usize, span: usize, dirty: bool) -> Child {
        Child { base, span, dirty }
    }

    /// A child that kept every slot.
    const KEPT: Child = Child {
        base: 0,
        span: 0,
        dirty: false,
    };

    /// What a walked level's parent reads for a child of `width` slots
    /// whose block starts at `base`: one with no block or that kept every
    /// slot needs no new indices, and one that lost a slot is walked in
    /// turn.
    fn read(marks: &[u64], base: Option<usize>, width: usize) -> Child {
        match base {
            Some(base) if !all_marked(&marks[base..base + words(width)], width) => {
                Child::at(base, width, true)
            }
            _ => Child::KEPT,
        }
    }
}

/// Compact one structural level: rewrite its references into a child that
/// lost a slot through that child's new indices, and drop its own unmarked
/// nodes. `v.base` starts at the level's block of marks, and `marks` holds the
/// children's. Returns whether the level lost a node.
///
/// A level held as the description of its pairs is kept as `plan`, decided
/// before the compaction changed any level ([`plan_levels`]): as the
/// description of what is left of it, when that is affine, or stored in the
/// room reserved for it ([`store_kept`]).
///
/// `remap` must hold the spans of the dirty children side by side.
///
/// A stored level it changes is left its live pairs for the close
/// ([`TddLevel::leave_live_pairs`]): the pairs of its marked nodes where the
/// marking read them (`v.kept`), else where it knew them before
/// ([`TddLevel::known_live_pairs`], as the level storage reads it): those
/// less the pairs of the nodes it drops, which a rewrite of references
/// leaves as they were. Where every node
/// holds the same count of pairs (a description's, or one on a stored level
/// of inline nodes alone), the level counts the nodes it keeps off the
/// marks; a stored level otherwise reads the pairs of the nodes it drops in
/// the loop that counts their arena pairs.
///
/// [`TddLevel::leave_live_pairs`]: crate::diagram::TddLevel::leave_live_pairs
/// [`TddLevel::known_live_pairs`]: crate::diagram::TddLevel::known_live_pairs
fn compact_one_level(
    tdd: &mut Tdd,
    v: Visit,
    marks: &[u64],
    remap: &mut [u32],
    plan: Option<Plan>,
) -> bool {
    let Visit {
        level: t,
        base,
        left,
        right,
        kept,
    } = v;
    let t_idx = t.idx();
    let width = tdd.levels[t_idx].slot_count();
    let own = &marks[base..base + words(width)];
    let lost = !all_marked(own, width);
    debug_assert_eq!(
        plan.is_some(),
        tdd.levels[t_idx].pairs.implicit().is_some() && (lost || left.dirty || right.dirty)
    );
    if !(lost || left.dirty || right.dirty) {
        // A level the walk read whole and the compaction leaves as it is
        // keeps no count of its own on open levels: marked changed, it is
        // left the pairs read, for the close to keep.
        let level = &tdd.levels[t_idx];
        if let Some(pairs) = kept.filter(|_| !tdd.levels.counts_hold() && level.implicit().is_none() && level.known_live_pairs(false).is_none()) {
            tdd.levels.mark_changed(t);
            tdd.levels[t_idx].leave_live_pairs(pairs);
        }
        return false;
    }
    let level = &tdd.levels[t_idx];
    let before = tdd.levels.known_live_pairs(t_idx);
    // The pairs each node holds, where they are the same for every node: a
    // description's, or one on a stored level of no arena pairs and as many
    // pairs as nodes (each node there holds at most one).
    let per_node = match level.pairs.implicit() {
        Some(d) => Some(d.pairs_per_node()),
        None if level.pairs.is_empty() && before == Some(width) => Some(1),
        None => None,
    };
    tdd.levels.mark_changed(t);

    let redescribed = match plan {
        Some(Plan::Described(d, dead)) => {
            tdd.levels[t_idx].pairs.redescribe(d);
            Some(dead)
        }
        Some(Plan::Stored(room)) => {
            let (lm, rm) = child_remaps(marks, remap, left, right);
            store_kept(tdd, t, room, own, lm, rm);
            None
        }
        None if left.dirty || right.dirty => {
            // A level that keeps most of its nodes, over children that each
            // dropped few slots, moves its references in one pass over its
            // words; otherwise its kept nodes' references are read through
            // the children's new indices.
            let kept = count(own);
            let dropped = |c: Child| {
                if c.dirty {
                    few_dropped(&marks[c.base..], c.span)
                } else {
                    Some(Dropped::NONE)
                }
            };
            match (2 * kept >= width).then(|| (dropped(left), dropped(right))) {
                Some((Some(l), Some(r))) => move_past_dropped(&mut tdd.levels[t_idx], &l, &r),
                _ => {
                    let (lm, rm) = child_remaps(marks, remap, left, right);
                    rewrite_child_refs(tdd, t, own, lm, rm);
                }
            }
            None
        }
        None => None,
    };

    let level = &mut tdd.levels[t_idx];
    if !lost {
        if let Some(pairs) = kept.or(before).filter(|_| level.implicit().is_none()) {
            level.leave_live_pairs(pairs);
        }
        return false;
    }
    // The live pairs of the stored nodes it drops, where it keeps a count.
    let mut dropped = 0usize;
    // Compact the node Vec in place, O(width) with no allocation. Dropping a
    // node abandons its pair range; the arena is swept once enough of it is
    // dead, which is legal here because no pair offset is held across the
    // call.
    let dead = match redescribed {
        Some(dead) => dead,
        None => {
            // A node drops arena pairs only where the arena holds some: on
            // a level of inline nodes alone nothing is read.
            let mut dead = 0usize;
            if kept.is_none() && before.is_some() && per_node.is_none() {
                for_each_unmarked(own, width, |i| {
                    dead += level.arena_pairs_at(i);
                    dropped += level.pair_count_at(i);
                });
            } else if !level.pairs.is_empty() {
                for_each_unmarked(own, width, |i| dead += level.arena_pairs_at(i));
            }
            keep_marked(level.nodes.stored_mut(), own);
            dead
        }
    };
    level.note_dead_pairs(dead);
    level.compact_pairs_if_stale();
    if level.implicit().is_none() {
        let pairs = match (kept, per_node, before) {
            (Some(kept), _, _) => Some(kept),
            (None, Some(k), Some(_)) => Some(count(own) * k),
            (None, None, Some(before)) => Some(before - dropped),
            (None, _, None) => None,
        };
        if let Some(pairs) = pairs {
            level.leave_live_pairs(pairs);
        }
    }
    true
}

/// The new indices of the children that lost a slot, `left` and `right`,
/// built from their marks into `remap`, side by side: `None` for a child
/// that kept every slot.
fn child_remaps<'r>(
    marks: &[u64],
    remap: &'r mut [u32],
    left: Child,
    right: Child,
) -> (Option<&'r [u32]>, Option<&'r [u32]>) {
    let (left_new, right_new) = remap.split_at_mut(if left.dirty { left.span } else { 0 });
    let left_remap = if left.dirty {
        new_indices(&marks[left.base..], left.span, left_new);
        Some(&*left_new)
    } else {
        None
    };
    let right_remap = if right.dirty {
        new_indices(&marks[right.base..], right.span, right_new);
        Some(&right_new[..right.span])
    } else {
        None
    };
    (left_remap, right_remap)
}

/// What the compaction does with a level held as the description of its
/// pairs that it changes, decided before it changes any level.
enum Plan {
    /// What is left is affine: held as this description, with the pairs of
    /// the nodes dropped.
    Described(ImplicitLevel, usize),
    /// What is left is not affine: stored, in this room.
    Stored(StoreRoom),
}

/// The [`Plan`] of the levels a compaction changes that are held as
/// descriptions, the first the compaction reaches last.
struct Plans<'l>(Transient<'l, Vec<(VtreeIdx, Plan)>>);

impl Plans<'_> {
    /// The plan of level `t`, the next level the compaction reaches, if it
    /// has one.
    fn take(&mut self, t: VtreeIdx) -> Option<Plan> {
        self.0.pop_if(|(v, _)| *v == t).map(|(_, plan)| plan)
    }
}

/// Decide, for every level in `order` (each with the word its block of
/// marks starts at and its children, in the order the compaction reaches
/// them), what the compaction does with it when it is held as the
/// description of its pairs and the compaction changes it: kept as the
/// description of what is left ([`ImplicitLevel::pruned`]), or stored in a
/// room reserved here ([`TddLevel::store_room`]). The compaction rewrites
/// levels in place and cannot stop partway, so whatever it would allocate
/// for such a level is had first; a refusal leaves the diagram as it was.
///
/// [`TddLevel::store_room`]: crate::diagram::TddLevel
///
/// # Errors
///
/// `Err(OperationError::OverBudget)` when the reading of a description or
/// a room is refused.
fn plan_levels<'l>(
    lim: &'l Limits,
    tdd: &Tdd,
    order: impl Iterator<Item = Visit>,
    marks: &[u64],
    remap: &mut [u32],
) -> Result<Plans<'l>, OperationError> {
    let mut plans = Transient::new(lim, Vec::new());
    for v in order {
        if let Some(kept) = plan_level(lim, tdd, v, marks, remap)? {
            lim.try_push(&mut plans, (v.level, kept))?;
        }
    }
    plans.reverse();
    Ok(Plans(plans))
}

/// [`plan_levels`] for `v.level`: `None` when it is stored, or the
/// compaction leaves it as it is.
fn plan_level(
    lim: &Limits,
    tdd: &Tdd,
    v: Visit,
    marks: &[u64],
    remap: &mut [u32],
) -> Result<Option<Plan>, OperationError> {
    let Visit {
        level: t,
        base,
        left,
        right,
        ..
    } = v;
    let level = &tdd.levels[t.idx()];
    if level.pairs.implicit().is_none() {
        return Ok(None);
    }
    let width = level.slot_count();
    let own = &marks[base..base + words(width)];
    if all_marked(own, width) && !left.dirty && !right.dirty {
        return Ok(None);
    }
    let (lm, rm) = child_remaps(marks, remap, left, right);
    // The nodes a child that lost one kept, as its marks count them.
    let kept_of = |c: Child| c.dirty.then(|| count(&marks[c.base..c.base + words(c.span)]));
    Ok(Some(
        match kept_described(lim, tdd, t, own, (lm, kept_of(left)), (rm, kept_of(right)))? {
            Some((d, dead)) => Plan::Described(d, dead),
            None => Plan::Stored(level.store_room(lim)?),
        },
    ))
}

/// Keep the nodes whose slots `own` marks, in their order, in place. A word
/// at a time: the nodes of a word of marked slots move as one run, and only
/// a mixed word's are read a mark at a time.
pub(super) fn keep_marked(nodes: &mut Vec<EncodedNode>, own: &[u64]) {
    let width = nodes.len();
    let mut write = 0usize;
    for (w, &word) in own[..words(width)].iter().enumerate() {
        let base = w << 6;
        let n = (width - base).min(64);
        let full = tail_mask(n);
        let mut x = word & full;
        if x == full {
            nodes.copy_within(base..base + n, write);
            write += n;
            continue;
        }
        while x != 0 {
            nodes[write] = nodes[base + x.trailing_zeros() as usize];
            write += 1;
            x &= x - 1;
        }
    }
    nodes.truncate(write);
}

/// What a prune leaves of level `t`, held as the description of its pairs,
/// as one: its marked nodes in their order, each with its pairs, whose
/// child slots move through `left.0` and `right.0`, the new indices of the
/// children that lost a node (`left.1` and `right.1` the nodes each kept),
/// when what is left is affine in a mixed radix. The new description
/// implies the nodes left, their ranges from the start of the arena, and
/// the arena keeps its length, as a written one keeps the pairs of the
/// nodes the prune drops until a sweep.
///
/// Where each side is one no drop of nodes bends, its slot moved by no node
/// digit or renumbered in the order of the pairs' positions, or the prune
/// keeps a box of the description and renumbers each child it shrinks in an
/// order the box's digits count, the new description is derived from the
/// old one, a pair of neither read ([`ImplicitLevel::kept_any`],
/// [`ImplicitLevel::kept_box`]); otherwise it is read off the moved pairs
/// and checked at every one of them
/// ([`ImplicitLevel::pruned`](crate::diagram::ImplicitLevel)).
///
/// Returns the description with the pairs of the nodes dropped, or `None`
/// when what is left is not affine, or below the floor. Writes no pair.
///
/// # Errors
///
/// `Err(OperationError::OverBudget)` when the offsets the check reads are
/// refused.
fn kept_described(
    lim: &Limits,
    tdd: &Tdd,
    t: VtreeIdx,
    own: &[u64],
    left: (Option<&[u32]>, Option<usize>),
    right: (Option<&[u32]>, Option<usize>),
) -> Result<Option<(ImplicitLevel, usize)>, OperationError> {
    let level = &tdd.levels[t.idx()];
    let d = level
        .pairs
        .implicit()
        .expect("a description is kept as one");
    let k = d.pairs_per_node();
    let nodes = count(own);
    // Past 2^31 pairs a node's range takes the side table (`ranges`); the
    // written arena's would too, and renumbered ones might not. What is left
    // below the floor is stored.
    if nodes == 0 || nodes * k < crate::diagram::floor() || level.pairs.len() >= 1 << 31 {
        return Ok(None);
    }
    // Below 2^31 pairs the description implies the nodes, node `i` of the
    // description being node `i` of the level.
    debug_assert!(level.implied_by().is_some());
    let kept = |n: Option<usize>| n.map_or(ChildKept::Whole, ChildKept::Renumbered);
    let (l, r) = (kept(left.1), kept(right.1));
    let derived = d.kept_any(nodes, l, r).or_else(|| d.kept_box(own, nodes, l, r));
    let fitted = || {
        let mut select = Select::new(own);
        d.pruned(lim, nodes, |j| select.nth(j), || marked(own), moved(left.0), moved(right.0))
    };
    #[cfg(debug_assertions)]
    if derived.is_some()
        && let Ok(fit) = fitted()
    {
        assert!(
            fit == derived,
            "a derived description differs from the fitted one"
        );
    }
    let left_of = match derived {
        Some(derived) => derived,
        None => match fitted()? {
            Some(fit) => fit,
            None => return Ok(None),
        },
    };
    // Each node the prune drops held its `k` pairs in the arena, or, at
    // one pair a node, its pair inline.
    let dead = if k >= 2 { (d.nodes() - nodes) * k } else { 0 };
    Ok(Some((left_of, dead)))
}

/// Store the marked nodes of level `t`, held as the description of its
/// pairs, in `room`, reserved for it, when what is left of it is not
/// affine: their pairs, with the child slots moved through `left` and
/// `right`, the new indices of the children that lost a node, in the arena
/// a stored level would hold ([`TddLevel::store_moved`]).
///
/// [`TddLevel::store_moved`]: crate::diagram::TddLevel
fn store_kept(
    tdd: &mut Tdd,
    t: VtreeIdx,
    room: StoreRoom,
    own: &[u64],
    left: Option<&[u32]>,
    right: Option<&[u32]>,
) {
    let marked = |i: usize| own[i >> 6] >> (i & 63) & 1 != 0;
    tdd.levels[t.idx()].store_moved(room, marked, moved(left), moved(right));
}

/// Rewrite level `t`'s references into the children that lost a slot
/// through their new indices, `left_remap` and `right_remap`, each starting
/// at its own level's first slot: `None` for a child that kept every slot,
/// whose references stay as they are. `own` is `t`'s own block of marks,
/// and only its marked nodes are rewritten.
///
/// A child that lost a slot is structural, the only kind a prune compacts,
/// so a reference into it is the child's slot, read straight through its
/// table; a side whose child kept every slot is not written.
///
/// Out of line, so that the rewrite's loop keeps its registers apart from
/// what the compaction holds across it.
#[inline(never)]
fn rewrite_child_refs(
    tdd: &mut Tdd,
    t: VtreeIdx,
    own: &[u64],
    left_remap: Option<&[u32]>,
    right_remap: Option<&[u32]>,
) {
    let (left, right) = tdd.vtree.children(t);
    debug_assert!(left_remap.is_none() || tdd.is_structural_internal(left));
    debug_assert!(right_remap.is_none() || tdd.is_structural_internal(right));
    let level = &mut tdd.levels[t.idx()];
    let kept = |s: u32| s;
    match (left_remap, right_remap) {
        (Some(l), Some(r)) => move_sides(level, own, moved_slot(l), moved_slot(r)),
        (Some(l), None) => move_sides(level, own, moved_slot(l), kept),
        (None, Some(r)) => move_sides(level, own, kept, moved_slot(r)),
        (None, None) => {}
    }
}

/// Where a slot of a structural child moves under its new indices `remap`.
#[inline(always)]
fn moved_slot(remap: &[u32]) -> impl Fn(u32) -> u32 + '_ {
    move |s| {
        let to = remap[s as usize];
        debug_assert_ne!(to, UNREACHED, "a referenced slot survives the compaction");
        to
    }
}

/// Move the left and the right side of every pair of `level`'s marked
/// nodes, by `own`, through `left` and `right`.
#[inline(always)]
fn move_sides(
    level: &mut TddLevel,
    own: &[u64],
    left: impl Fn(u32) -> u32,
    right: impl Fn(u32) -> u32,
) {
    marked(own).for_each(|i| move_node(level, i, &left, &right));
}

/// Move the left and the right side of every pair of `level`'s node `i`
/// through `left` and `right`.
#[inline(always)]
fn move_node(level: &mut TddLevel, i: usize, left: impl Fn(u32) -> u32, right: impl Fn(u32) -> u32) {
    let kind = level.nodes.stored()[i].kind();
    match kind {
        NodeKind::Inline(_) => {
            let node = &mut level.nodes.stored_mut()[i];
            node.a = left(node.a);
            node.b = right(node.b);
        }
        _ => {
            if let Some(range) = level.arena_range(kind) {
                for pair in &mut level.pairs.stored_mut()[range] {
                    pair.left = EncodedChildRef::from_raw(left(pair.left.raw()));
                    pair.right = EncodedChildRef::from_raw(right(pair.right.raw()));
                }
            }
        }
    }
}

/// Drop the nodes nothing names from the levels of the clause spine
/// (`on_spine`) that the conjunction with a clause rebuilt in `levels`: the
/// output level's nodes but the output `out` at level `out_vtree`, then,
/// top-down, a spine level's nodes its parent no longer names. Returns the
/// output's new index, the nodes dropped and the parent pairs read.
///
/// A rebuilt level names every node of its spine children but at `top`,
/// the highest level with both children on the spine, whose children are
/// read on the sides it gives. Below them a level that lost a node is read
/// on both sides.
///
/// Charges no work: the walk charged the pairs it built, and the clock
/// stays where a walk without the drop leaves it.
///
/// # Errors
///
/// Returns `Err(OperationError::OverBudget)` if the marks or the new indices
/// cannot grow; a level read before keeps its drop.
#[inline(never)]
pub(crate) fn drop_clause_orphans(
    eng: &Engine,
    vtree: &Vtree,
    levels: &mut [TddLevel],
    on_spine: &[bool],
    top: Option<(VtreeIdx, [bool; 2])>,
    (out_vtree, out): (VtreeIdx, u32),
) -> Result<(u32, [u64; 2]), OperationError> {
    let lim = eng.limits();
    let mut tally = [0u64; 2];
    let mut scratch = eng.scratch.reduce.clause_drop.checkout_preserving(eng);
    let DropScratch { marks, remap, lost } = &mut *scratch;
    lost.clear();
    let readable = |c: VtreeIdx| on_spine[c.idx()] && !vtree.node(c).is_leaf();
    let readable_sides = |t: VtreeIdx| {
        let (l, r) = vtree.children(t);
        [readable(l), readable(r)]
    };
    let width = levels[out_vtree.idx()].slot_count();
    let own = words(width);
    clear_front(lim, marks, own, 0)?;
    mark(marks, 0, out as usize);
    let out = if all_marked(marks, width) {
        out
    } else {
        let sides = readable_sides(out_vtree);
        grow_to(lim, remap, width, UNREACHED)?;
        new_indices(&marks[..own], width, remap);
        tally[0] += drop_unmarked(&mut levels[out_vtree.idx()], &marks[..own]);
        if sides != [false; 2] {
            lim.try_push(lost, (out_vtree, sides))?;
        }
        remap[out as usize]
    };
    let mut top = top;
    loop {
        let (t, mut sides) = match (lost.pop(), top) {
            (Some(lost), _) => lost,
            (None, Some(top)) => top,
            (None, None) => break,
        };
        if let Some((at, top_sides)) = top
            && at == t
        {
            sides = [sides[0] | top_sides[0], sides[1] | top_sides[1]];
            top = None;
        }
        let (l, r) = vtree.children(t);
        let width = [(l, 0), (r, 1)].map(|(c, s)| if sides[s] && readable(c) { levels[c.idx()].slot_count() } else { 0 });
        if width == [0, 0] {
            continue;
        }
        let base = words(width[0]);
        let end = base + words(width[1]);
        clear_front(lim, marks, end, 0)?;
        let parent = &levels[t.idx()];
        tally[1] += match width.map(|w| w != 0) {
            [true, true] => mark_named::<true, true>(parent, marks, base),
            [true, false] => mark_named::<true, false>(parent, marks, base),
            _ => mark_named::<false, true>(parent, marks, base),
        };
        let own = [0..base, base..end];
        let dropped = [0, 1].map(|s| width[s] != 0 && !all_marked(&marks[own[s].clone()], width[s]));
        if dropped == [false, false] {
            continue;
        }
        grow_to(lim, remap, width[0] + width[1], UNREACHED)?;
        lim.reserve(lost, 2)?;
        for (s, c) in [l, r].into_iter().enumerate() {
            if dropped[s] {
                let sides = readable_sides(c);
                let new = &mut remap[if s == 0 { 0..width[0] } else { width[0]..width[0] + width[1] }];
                new_indices(&marks[own[s].clone()], width[s], new);
                tally[0] += drop_unmarked(&mut levels[c.idx()], &marks[own[s].clone()]);
                if sides != [false; 2] {
                    lost.push((c, sides));
                }
            }
        }
        let (left_new, right_new) = remap.split_at(width[0]);
        let parent = &mut levels[t.idx()];
        let kept = |s: u32| s;
        let nodes = 0..parent.slot_count();
        match dropped {
            [true, true] => nodes.for_each(|i| move_node(parent, i, moved_slot(left_new), moved_slot(right_new))),
            [true, false] => nodes.for_each(|i| move_node(parent, i, moved_slot(left_new), kept)),
            _ => nodes.for_each(|i| move_node(parent, i, kept, moved_slot(right_new))),
        }
    }
    Ok((out, tally))
}

/// Mark in `marks` the left (`L`) and the right (`R`, block at word `base`)
/// child slots the stored level `parent` names. Returns the pairs read.
fn mark_named<const L: bool, const R: bool>(parent: &TddLevel, marks: &mut [u64], base: usize) -> u64 {
    let stored = parent.stored().expect("a rebuilt level stores its pairs");
    let mut left = Marker::new(0);
    let mut read = 0;
    for node in parent.nodes.stored() {
        let pairs = stored.of(node);
        read += pairs.len();
        for p in pairs {
            if L {
                left.mark(marks, 0, p.left.raw() as usize);
            }
            if R {
                mark(marks, base, p.right.raw() as usize);
            }
        }
    }
    left.finish(marks);
    read as u64
}

/// Set the first `n` entries of `v` to `value`, growing it to `n` if it is
/// shorter.
fn clear_front<T: Copy>(lim: &Limits, v: &mut Vec<T>, n: usize, value: T) -> Result<(), OperationError> {
    let have = v.len().min(n);
    v[..have].fill(value);
    grow_to(lim, v, n, value)
}

/// Grow `v` to `n` entries of `value` if it is shorter.
fn grow_to<T: Copy>(lim: &Limits, v: &mut Vec<T>, n: usize, value: T) -> Result<(), OperationError> {
    if v.len() < n {
        lim.try_resize(v, n, value)?;
    }
    Ok(())
}

/// Remove from the stored structural level `level` the nodes whose slots
/// `own` does not mark, and their pairs. Returns the nodes removed.
fn drop_unmarked(level: &mut TddLevel, own: &[u64]) -> u64 {
    let width = level.slot_count();
    let mut dead = 0usize;
    if !level.pairs.is_empty() {
        for_each_unmarked(own, width, |i| dead += level.arena_pairs_at(i));
    }
    keep_marked(level.nodes.stored_mut(), own);
    level.note_dead_pairs(dead);
    level.compact_pairs_if_stale();
    (width - level.slot_count()) as u64
}

/// Where a slot of a structural child, as a raw word, moves when the child
/// is renumbered by `remap`: nowhere without one.
fn moved(remap: Option<&[u32]>) -> impl Fn(i64) -> i64 + '_ {
    move |x| {
        remap.map_or(x, |m| {
            let side = EncodedChildRef::from_raw(x as u32);
            i64::from(ChildDecoder::structural().remap(side, m).raw())
        })
    }
}

/// Mark every child slot the marked nodes of level `t` name. `base` is the
/// word level `t`'s block starts at in `marks`, and `children` are its left
/// and right children's, `None` for a child whose marks nobody reads: a leaf
/// or marginal level, which the compaction never drops a node from.
///
/// Only a structural child has a block, and a side into one is its slot: no
/// side into a marginal child is read.
///
/// `bytes` is the scratch a side marked a byte a slot takes ([`mark_sides`]).
///
/// Returns, on a stored level whose marks read its marked nodes' pairs, the
/// count of those pairs, or `Err(OperationError::OverBudget)` if the
/// offsets an implicit level's nodes are read in are refused
/// ([`mark_described`]), or the bytes.
fn mark_children_of_level(
    lim: &Limits,
    tdd: &Tdd,
    t: VtreeIdx,
    base: usize,
    children: [Option<usize>; 2],
    marks: &mut [u64],
    bytes: &mut Vec<u8>,
) -> Result<Option<usize>, OperationError> {
    match children {
        [Some(l), Some(r)] => mark_sides::<true, true>(lim, tdd, t, base, [l, r], marks, bytes),
        [Some(l), None] => mark_sides::<true, false>(lim, tdd, t, base, [l, 0], marks, bytes),
        [None, Some(r)] => mark_sides::<false, true>(lim, tdd, t, base, [0, r], marks, bytes),
        [None, None] => Ok(None),
    }
}

/// [`mark_children_of_level`] marking the left child's slots at block
/// `child_base[0]` when `LEFT`, and the right child's at `child_base[1]`
/// when `RIGHT`.
///
/// A stored level whose marked nodes are many beside a side's slots marks
/// that side a byte a slot in `bytes`, packed into the side's bits after
/// the read ([`pack_bytes`]). A byte mark is a store that reads nothing,
/// where a bit is OR-ed into a shared word, which a run of marks into one
/// word holds in a register and a mark into another word writes back: on
/// scattered children every mark decides that branch and writes its word.
/// The bytes cost a clear and a pack a slot, which pay where the marks a
/// slot are about one or more ([`STAGE_SLOTS_PER_NODE`]).
///
/// A stored level counts the pairs of its marked nodes as it reads them.
fn mark_sides<const LEFT: bool, const RIGHT: bool>(
    lim: &Limits,
    tdd: &Tdd,
    t: VtreeIdx,
    base: usize,
    child_base: [usize; 2],
    marks: &mut [u64],
    bytes: &mut Vec<u8>,
) -> Result<Option<usize>, OperationError> {
    let [left_base, right_base] = child_base;
    let (left, right) = tdd.vtree.children(t);
    debug_assert!(
        (!LEFT || tdd.is_structural_internal(left)) && (!RIGHT || tdd.is_structural_internal(right)),
        "only a structural child has a block of marks"
    );
    let level = &tdd.levels[t.idx()];
    let width = level.slot_count();
    if let Some(d) = level.implicit() {
        // A side's distinct offsets a marked node, never more marks than
        // its pairs.
        mark_described::<LEFT, RIGHT>(lim, d, width, base, left_base, right_base, marks)?;
        return Ok(None);
    }
    // The sides staged as bytes: their slots, rounded up to whole words.
    let marked = count(&marks[base..base + words(width)]);
    let staged = |on: bool, c: VtreeIdx| {
        let slots = tdd.reference_slot_count(c);
        (on && slots <= marked.saturating_mul(STAGE_SLOTS_PER_NODE))
            .then_some(words(slots) * 64)
    };
    let (ls, rs) = (staged(LEFT, left), staged(RIGHT, right));
    note_sides_marked(
        u64::from(ls.is_some()) + u64::from(rs.is_some()),
        u64::from(LEFT && ls.is_none()) + u64::from(RIGHT && rs.is_none()),
    );
    if ls.is_some() || rs.is_some() {
        let (ln, rn) = (ls.unwrap_or(0), rs.unwrap_or(0));
        return match (ls, rs) {
            (Some(_), Some(_)) => mark_staged::<LEFT, RIGHT, true, true>(
                lim,
                tdd,
                t,
                (base, child_base),
                [ln, rn],
                marks,
                bytes,
            )
            .map(Some),
            (Some(_), None) => mark_staged::<LEFT, RIGHT, true, false>(
                lim,
                tdd,
                t,
                (base, child_base),
                [ln, rn],
                marks,
                bytes,
            )
            .map(Some),
            (None, Some(_)) => mark_staged::<LEFT, RIGHT, false, true>(
                lim,
                tdd,
                t,
                (base, child_base),
                [ln, rn],
                marks,
                bytes,
            )
            .map(Some),
            (None, None) => unreachable!(),
        };
    }
    let stored = level.stored().expect("a level is stored or implicit");
    let (mut left_marks, mut right_marks) = (Marker::new(left_base), Marker::new(right_base));
    let mut read = 0;
    for_each_marked_beside(width, base, marks, |marks, i| {
        let pairs = stored.of_idx(i);
        read += pairs.len();
        for &pair in pairs {
            if LEFT {
                left_marks.mark(marks, left_base, pair.left.raw() as usize);
            }
            if RIGHT {
                right_marks.mark(marks, right_base, pair.right.raw() as usize);
            }
        }
    });
    left_marks.finish(marks);
    right_marks.finish(marks);
    Ok(Some(read))
}

/// The most slots a side of a stored level has, a marked node of the level,
/// for [`mark_sides`] to mark it as bytes.
const STAGE_SLOTS_PER_NODE: usize = 2;

// A test counts the reads and has a settle read every level node by node.
#[cfg(test)]
use super::tests::{note_settle_read, note_sides_marked, settle_by_node};

/// Whether a settle reads every level node by node: never, outside a test.
#[cfg(not(test))]
#[inline(always)]
const fn settle_by_node() -> bool {
    false
}

/// How a settle read a level: whole (0), node by node (1) or off its digits (2).
#[cfg(not(test))]
#[inline(always)]
fn note_settle_read(_how: usize) {}

/// The sides of a stored level marked as bytes and as bits.
#[cfg(not(test))]
#[inline(always)]
fn note_sides_marked(_bytes: u64, _bits: u64) {}

/// [`mark_sides`] on a stored level: each side of `LEFT` and `RIGHT` that
/// `LB` and `RB` stage marked as bytes, `n` of them a side, its slots
/// rounded up to whole words, and packed into its block of `marks` at
/// `child_base`; a side not staged marked as bits. Returns the pairs of the
/// marked nodes.
#[inline(always)]
fn mark_staged<const LEFT: bool, const RIGHT: bool, const LB: bool, const RB: bool>(
    lim: &Limits,
    tdd: &Tdd,
    t: VtreeIdx,
    (base, child_base): (usize, [usize; 2]),
    n: [usize; 2],
    marks: &mut [u64],
    bytes: &mut Vec<u8>,
) -> Result<usize, OperationError> {
    let level = &tdd.levels[t.idx()];
    let stored = level.stored().expect("a staged level is stored");
    let width = level.slot_count();
    let used = n[0] + n[1];
    if bytes.len() < used {
        lim.try_resize(bytes, used, 0)?;
    }
    bytes[..used].fill(0);
    let (lb, rb) = bytes[..used].split_at_mut(n[0]);
    let (mut left_marks, mut right_marks) =
        (Marker::new(child_base[0]), Marker::new(child_base[1]));
    let mut read = 0;
    for_each_marked_beside(width, base, marks, |marks, i| {
        let pairs = stored.of_idx(i);
        read += pairs.len();
        for &pair in pairs {
            let (l, r) = (pair.left.raw() as usize, pair.right.raw() as usize);
            if LEFT {
                if LB {
                    lb[l] = 1;
                } else {
                    left_marks.mark(marks, child_base[0], l);
                }
            }
            if RIGHT {
                if RB {
                    rb[r] = 1;
                } else {
                    right_marks.mark(marks, child_base[1], r);
                }
            }
        }
    });
    left_marks.finish(marks);
    right_marks.finish(marks);
    if LB {
        pack_bytes(lb, &mut marks[child_base[0]..child_base[0] + n[0] / 64]);
    }
    if RB {
        pack_bytes(rb, &mut marks[child_base[1]..child_base[1] + n[1] / 64]);
    }
    Ok(read)
}

/// OR the byte marks of `bytes`, each 0 or 1, into `block`, bit `i` of word
/// `w` from byte `64 w + i`.
fn pack_bytes(bytes: &[u8], block: &mut [u64]) {
    for (word, chunk) in block.iter_mut().zip(bytes.chunks_exact(64)) {
        let mut bits = 0u64;
        for (k, eight) in chunk.chunks_exact(8).enumerate() {
            let x = u64::from_le_bytes(eight.try_into().expect("eight bytes"));
            // Byte `j` of `x` lands on bit `56 + j` of the product, and no
            // two of its terms meet, so nothing carries into the top byte.
            bits |= (x.wrapping_mul(0x0102_0408_1020_4080) >> 56) << (8 * k);
        }
        *word |= bits;
    }
}

/// Call `f` with the index of every marked slot of the block of `width`
/// slots at word `base` of `marks`, ascending, and `marks`: each word is read before `f` runs
/// on its slots, so that `f` may mark other blocks, a level's children's
/// beside its own.
#[inline(always)]
fn for_each_marked_beside(
    width: usize,
    base: usize,
    marks: &mut [u64],
    mut f: impl FnMut(&mut [u64], usize),
) {
    for w in 0..words(width) {
        let mut x = marks[base + w];
        while x != 0 {
            let i = (w << 6) + x.trailing_zeros() as usize;
            x &= x - 1;
            f(marks, i);
        }
    }
}

/// [`mark_sides`] on a level held as the description of its pairs, whose
/// children are structural, so that a side's word is the child's slot: the
/// slots off the digits, not off the pairs, the left child's at block
/// `left_base` when `LEFT` and the right child's at `right_base` when
/// `RIGHT`. A marked node marks its first pair's slot on a side shifted by
/// each of the offsets the place digits that move the side give, once for
/// a run of marked nodes that start at one slot there, both sides off one
/// pass over the marked nodes; with every node marked, the nodes' first
/// slots on a side are read off the node digits that move the side. What
/// is marked is what the pairs would mark, read in about the side's
/// distinct offsets a node, not its pairs.
///
/// Returns `Err(OperationError::OverBudget)` if the offsets of a side,
/// which a node of billions of pairs has billions of, are refused.
fn mark_described<const LEFT: bool, const RIGHT: bool>(
    lim: &Limits,
    d: &ImplicitLevel,
    width: usize,
    base: usize,
    left_base: usize,
    right_base: usize,
    marks: &mut [u64],
) -> Result<(), OperationError> {
    /// The marks of one side: its offsets, charged while they are read,
    /// where its block starts, and the slot the last marked node started
    /// at there.
    struct Side<'l> {
        offsets: Transient<'l, Vec<i64>>,
        base: usize,
        marker: Marker,
        last: Option<i64>,
    }
    impl<'l> Side<'l> {
        fn new(
            lim: &'l Limits,
            d: &ImplicitLevel,
            side: ChildSide,
            on: bool,
            base: usize,
        ) -> Result<Side<'l>, OperationError> {
            let mut offsets = Transient::new(lim, Vec::new());
            if on {
                d.side_offsets(side, lim, &mut offsets)?;
            }
            Ok(Side {
                offsets,
                base,
                marker: Marker::new(base),
                last: None,
            })
        }

        /// Mark the slots of a node that starts at slot `first`.
        #[inline(always)]
        fn mark(&mut self, marks: &mut [u64], first: i64) {
            for &o in self.offsets.iter() {
                self.marker.mark(marks, self.base, (first + o) as usize);
            }
        }

        /// [`mark`](Self::mark), once for a run of nodes that start at one
        /// slot.
        #[inline(always)]
        fn mark_new(&mut self, marks: &mut [u64], first: i64) {
            if self.last != Some(first) {
                self.mark(marks, first);
                self.last = Some(first);
            }
        }
    }
    let mut left = Side::new(lim, d, ChildSide::Left, LEFT, left_base)?;
    let mut right = Side::new(lim, d, ChildSide::Right, RIGHT, right_base)?;
    if all_marked(&marks[base..base + words(width)], width) {
        if LEFT {
            d.each_side_first(ChildSide::Left, |first, _| left.mark(marks, first));
        }
        if RIGHT {
            d.each_side_first(ChildSide::Right, |first, _| right.mark(marks, first));
        }
    } else {
        let mut cursor = d.cursor();
        for_each_marked_beside(width, base, marks, |marks, i| {
            let (l, r) = cursor.first_of(i);
            if LEFT {
                left.mark_new(marks, l);
            }
            if RIGHT {
                right.mark_new(marks, r);
            }
        });
    }
    left.marker.finish(marks);
    right.marker.finish(marks);
    Ok(())
}

// ── The seeded walk ──────────────────────────────────────────────────────────

/// One level a walk marked: where its marks live, and what it reads its
/// children's marks from.
#[derive(Clone, Copy)]
pub(super) struct Visit {
    level: VtreeIdx,
    /// The word this level's block of marks starts at.
    base: usize,
    left: Child,
    right: Child,
    /// The pairs of this stored level's marked nodes, where the marking
    /// read them: its live pairs once the compaction drops the rest.
    kept: Option<usize>,
}

impl Visit {
    /// A level whose block is marked but whose children are not settled yet.
    const fn new(level: VtreeIdx, base: usize) -> Visit {
        Visit {
            level,
            base,
            left: Child::KEPT,
            right: Child::KEPT,
            kept: None,
        }
    }
}

/// Where [`prune_walk`] starts.
#[derive(Clone, Copy)]
pub(super) enum Start<'a> {
    /// At the root, the output node its only slot reached: whatever else
    /// the root level holds is what a change at the root dropped.
    Output,
    /// At these levels, every node of each reached, with per level its
    /// [`Seeds`] state, and the clean levels above a loose one: the walk
    /// enters no [`CLEAN`] child.
    Seeds(&'a [VtreeIdx], &'a [u8], &'a [VtreeIdx]),
    /// At the vtree root, entering every structural level whether or not it
    /// lost a node: the prune that assumes nothing. The output may be on any
    /// level; the levels above it lose every node.
    Root,
}

/// [`prune_unreachable`]: walk down from `start`, stopping at every level
/// that loses no node, unless the start is [`Start::Root`]. With `forced`,
/// also entering every level it marks whose entry has [`FORCED`] set,
/// whether or not it lost one.
pub(super) fn prune_walk(
    eng: &Engine,
    tdd: &mut Tdd,
    forced: Option<&[u8]>,
    start: Start<'_>,
) -> Result<(), OperationError> {
    let mut scratch = eng.scratch.reduce.prune.checkout_preserving(eng);
    let PruneScratch {
        marks,
        remap,
        visits,
        entered,
        from_output,
        bytes,
        ..
    } = &mut *scratch;
    // The walk from the root takes a block and a visit for every structural
    // level: they are reserved here, where one at a time would reallocate on
    // each, in the pass that counts the slots it is charged for.
    let mut whole_slots = 0usize;
    if let Start::Root = start {
        let (mut levels, mut total) = (0usize, 0usize);
        for t in (0..tdd.vtree.num_nodes()).map(|t| VtreeIdx(t as u32)) {
            whole_slots += tdd.reference_slot_count(t);
            if tdd.is_structural_internal(t) {
                levels += 1;
                total += words(tdd.levels[t.idx()].slot_count());
            }
        }
        visits.clear();
        eng.limits().reserve_exact(marks, total.saturating_sub(marks.len()))?;
        eng.limits().reserve_exact(visits, levels)?;
    }
    let mut walk = mark_walk(eng.limits(), tdd, forced, start, marks, visits, bytes)?;
    match start {
        // A walk from the seeds is charged as one from the output would be.
        Start::Seeds(_, _, top) => {
            walk.slots = slots_from_output(eng.limits(), tdd, top, visits, entered, from_output)?;
        }
        Start::Root => walk.slots = whole_slots,
        Start::Output => {}
    }
    compact_visited(eng, tdd, marks, remap, visits, walk.remap_need)?;

    if let Some(base) = walk.output {
        tdd.output.local = NodeIdx(rank(&marks[base..], tdd.output.local.idx()));
    }

    // Both passes cross every slot the walk is charged for. The compaction
    // cannot stop partway: it rewrites levels in place, so a diagram
    // abandoned mid-pass has some levels compacted and others still naming
    // their old indices. The marking tests the time between levels; the
    // work-unit test belongs to the callers, between prunes.
    eng.limits().charge_work(2 * walk.slots as u64);
    Ok(())
}

/// What [`mark_walk`] leaves for the compaction.
struct Walked {
    /// The slots the walk is charged for.
    slots: usize,
    /// The widest pair of dirty children one level rewrites its references
    /// into, side by side.
    remap_need: usize,
    /// The word the output's level's block starts at, if it has one.
    output: Option<usize>,
}

/// The marking half of [`prune_walk`]: `visits` the levels it
/// entered, top-down, each with its block of `marks` and its children.
fn mark_walk(
    lim: &Limits,
    tdd: &Tdd,
    forced: Option<&[u8]>,
    start: Start<'_>,
    marks: &mut Vec<u64>,
    visits: &mut Vec<Visit>,
    bytes: &mut Vec<u8>,
) -> Result<Walked, OperationError> {
    let vtree = &tdd.vtree;
    let out = tdd.output;
    visits.clear();
    // Words of `marks` taken.
    let mut used = 0usize;
    let mut slots = 0usize;
    let mut remap_need = 0usize;
    let mut output = None;

    let clean = match start {
        Start::Output => {
            let width = tdd.levels[out.vtree.idx()].slot_count();
            let base = alloc_block(lim, marks, &mut used, &mut slots, width)?;
            mark(marks, base, out.local.idx());
            lim.try_push(visits, Visit::new(out.vtree, base))?;
            output = Some(base);
            None
        }
        Start::Seeds(seeds, state, _) => {
            lim.reserve(visits, seeds.len())?;
            for &t in seeds {
                let width = tdd.levels[t.idx()].slot_count();
                let base = alloc_block(lim, marks, &mut used, &mut slots, width)?;
                mark_all(&mut marks[base..base + words(width)], width);
                visits.push(Visit::new(t, base));
            }
            Some(state)
        }
        Start::Root => {
            let root = vtree.root();
            if tdd.is_structural_internal(root) {
                let width = tdd.levels[root.idx()].slot_count();
                let base = alloc_block(lim, marks, &mut used, &mut slots, width)?;
                if out.vtree == root {
                    mark(marks, base, out.local.idx());
                    output = Some(base);
                }
                lim.try_push(visits, Visit::new(root, base))?;
            }
            None
        }
    };
    let whole = matches!(start, Start::Root);
    let is_clean = |c: VtreeIdx| clean.is_some_and(|s| s[c.idx()] & CLEAN != 0);

    // A child the walk does not descend into, a leaf level, whose remap is
    // the identity, or a marginal one, which is never compacted, gets no
    // block: its marks would never be read. Such a child is charged its slots
    // when it is wider than every one met before it, and nothing otherwise.
    let mut unmarked = 0usize;
    // Time tests by pairs marked. The walk is charged at its end,
    // and changes nothing the compaction after it reads, so it can stop
    // between levels.
    let mut tests = TimeTests::new(Some(lim));

    // Marking, top-down. A level has one parent, so a level's marks are
    // complete as soon as that parent has been walked, and a queue is a valid
    // order.
    let mut i = 0;
    while i < visits.len() {
        let t = visits[i].level;
        let (left, right) = vtree.children(t);
        let (lw, rw) = (
            tdd.reference_slot_count(left),
            tdd.reference_slot_count(right),
        );
        let mut block = |c: VtreeIdx, width: usize| -> Result<Option<usize>, OperationError> {
            // A clean child, under a seed, is all reached: no block.
            if is_clean(c) {
                return Ok(None);
            }
            if !tdd.is_structural_internal(c) {
                charge_child(&mut slots, &mut unmarked, false, width);
                return Ok(None);
            }
            let base = alloc_block(lim, marks, &mut used, &mut slots, width)?;
            if whole && c == out.vtree {
                mark(marks, base, out.local.idx());
                output = Some(base);
            }
            Ok(Some(base))
        };
        let (lb, rb) = (block(left, lw)?, block(right, rw)?);

        visits[i].kept = mark_children_of_level(lim, tdd, t, visits[i].base, [lb, rb], marks, bytes)?;
        tests.after(tdd.levels[t.idx()].pairs_bound())?;

        let (lc, rc) = (Child::read(marks, lb, lw), Child::read(marks, rb, rw));
        visits[i].left = lc;
        visits[i].right = rc;
        remap_need = remap_need.max(lc.span + rc.span);
        let enter = |child: VtreeIdx, dirty: bool| {
            whole || dirty || forced.is_some_and(|f| f[child.idx()] & FORCED != 0)
        };
        if let Some(base) = lb
            && enter(left, lc.dirty)
        {
            lim.try_push(visits, Visit::new(left, base))?;
        }
        if let Some(base) = rb
            && enter(right, rc.dirty)
        {
            lim.try_push(visits, Visit::new(right, base))?;
        }
        i += 1;
    }
    Ok(Walked {
        slots,
        remap_need,
        output,
    })
}

/// Charge `slots` for a child of `width` slots: a structural child's
/// always, and an unmarked (leaf or marginal) one's when it is wider than
/// every unmarked child met before it, `unmarked` the widest.
#[inline(always)]
fn charge_child(slots: &mut usize, unmarked: &mut usize, structural: bool, width: usize) {
    if structural {
        *slots += width;
    } else if *unmarked < width {
        *slots += width;
        *unmarked = width;
    }
}

/// The slots a walk from the output would be charged for where one from
/// the seeds walked `visits`. That walk enters the levels the seeded one
/// did and, above them, the clean levels it must, which lose no node: the
/// root and `top`, the clean ones above a loose level. It takes them in its
/// queue's order, and is charged the root's block and, on each level, its
/// structural children's slots and, for another child, its slots where it
/// is wider than every one met before it. `entered` flags the levels it
/// enters, all clear again on return, and `from_output` holds them in its
/// order. Read before the walked levels are compacted.
///
/// # Errors
///
/// Returns `Err(OperationError::OverBudget)` if the room for those levels
/// is refused, before the diagram changes.
fn slots_from_output(
    lim: &Limits,
    tdd: &Tdd,
    top: &[VtreeIdx],
    visits: &[Visit],
    entered: &mut Vec<bool>,
    from_output: &mut Vec<VtreeIdx>,
) -> Result<usize, OperationError> {
    let vtree = &tdd.vtree;
    let root = vtree.root();
    if entered.len() < vtree.num_nodes() {
        lim.try_resize(entered, vtree.num_nodes(), false)?;
    }
    from_output.clear();
    lim.reserve(from_output, 1 + top.len() + visits.len())?;
    let levels = || {
        std::iter::once(root)
            .chain(top.iter().copied())
            .chain(visits.iter().map(|v| v.level))
    };
    let mut count = 0;
    for t in levels() {
        count += usize::from(!entered[t.idx()]);
        entered[t.idx()] = true;
    }
    from_output.push(root);
    let mut slots = tdd.levels[root.idx()].slot_count();
    let mut unmarked = 0usize;
    let mut i = 0;
    while i < from_output.len() {
        let (left, right) = vtree.children(from_output[i]);
        for c in [left, right] {
            let structural = tdd.is_structural_internal(c);
            charge_child(&mut slots, &mut unmarked, structural, tdd.reference_slot_count(c));
            if structural && entered[c.idx()] {
                from_output.push(c);
            }
        }
        i += 1;
    }
    for t in levels() {
        entered[t.idx()] = false;
    }
    debug_assert_eq!(
        from_output.len(),
        count,
        "a level the seeded walk entered is not under one a walk from the output enters"
    );
    Ok(slots)
}

/// Compact the levels of `visits`, each with its block of `marks`, a level
/// after its children: a level was pushed after its parent, so the visits
/// reversed put every level after its own children. `remap_need` is the
/// widest pair of dirty children a level rewrites its references into.
///
/// A marginal level is never compacted: its parents reference its value
/// store by slot, and a contraction after this prune may mint such a
/// reference to a slot nothing references now (the inline-to-slot
/// redirect). `prune_value_slots` collects the store's orphans.
fn compact_visited(
    eng: &Engine,
    tdd: &mut Tdd,
    marks: &[u64],
    remap: &mut Vec<u32>,
    visits: &[Visit],
    remap_need: usize,
) -> Result<(), OperationError> {
    eng.limits().try_resize(remap, remap_need, UNREACHED)?;
    // What the compaction does with the levels held as descriptions is
    // decided first.
    let mut plans = plan_levels(
        eng.limits(),
        tdd,
        visits.iter().rev().copied(),
        marks,
        remap,
    )?;
    // The first level compacted makes room in the worklists for every level
    // still to come, so that each list grows once however many are pushed.
    let mut room = false;
    for k in (0..visits.len()).rev() {
        let v = visits[k];
        let plan = plans.take(v.level);
        if compact_one_level(tdd, v, marks, remap, plan) {
            if !room {
                tdd.dirty.reserve_all(k + 1);
                room = true;
            }
            tdd.invalidate_shrunk(v.level);
        }
    }
    Ok(())
}

// ── The complement of a canonical diagram ────────────────────────────────────

/// Prune the negation of a canonical diagram: `tdd` is its completion with
/// the complement beside the old output at the root, as the negation leaves
/// it, and `old` the old output's slot.
///
/// Complement maps the residual classes of a level one to one, except that
/// the class whose residual was true dies, its residual now false: all the
/// prune removes is the old output and, on each level below it, the one
/// node, if any, whose residual was true. That node `u` of a child of a
/// level `t` is the one the dying node of `t` pairs with every node of the
/// sibling's completed basis: `u` is true under every completion exactly
/// when every rectangle of `u` with a sibling class lies in the true class
/// of `t`. A level has it only under a parent that has one, so the levels
/// read are a subtree of the vtree from the root down, and on each only the
/// dying node's pairs, where [`prune_walk`] reads every pair of every
/// level that loses a node.
///
/// The dying nodes are dropped as that walk's compaction drops them. A
/// level held as a description goes through the compaction itself, given
/// the marks the walk would set: every slot of the level but the dying
/// node's. A stored level drops its node in place, the nodes after it one
/// slot down, and in a stored parent the references past it move one slot
/// down in a pass over the parent's node words and arena
/// ([`renumber_past`]), where the compaction reads each reference through
/// a table. The diagram, the order of each level's nodes and how each
/// level is held come out as the walk leaves them, and the work clock is
/// charged what the walk is charged.
///
/// # Errors
///
/// As [`prune_unreachable`]: a refused reservation leaves `tdd` untouched.
pub(super) fn prune_complement(
    eng: &Engine,
    tdd: &mut Tdd,
    old: NodeIdx,
) -> Result<(), OperationError> {
    debug_assert!(below_root_walk_applies(tdd) && !tdd.has_marginal_level());
    debug_assert_ne!(old, tdd.output.local);
    let lim = eng.limits();
    let mut scratch = eng.scratch.reduce.prune.checkout_preserving(eng);
    let PruneScratch {
        marks,
        remap,
        visits,
        chain,
        row,
        ..
    } = &mut *scratch;
    let vtree = std::sync::Arc::clone(&tdd.vtree);
    let root = tdd.output.vtree;
    debug_assert_eq!(
        tdd.levels[root.idx()].slot_count(),
        2,
        "the root holds the old output and its complement"
    );
    chain.clear();
    lim.try_push(chain, [root.0, old.0, 0])?;
    // The slots the walk below the root is charged for: the root's, and on
    // each level of the chain, the levels it walks in the order it walks
    // them, the slots of its structural children and of another child wider
    // than every one met before it ([`prune_walk`]).
    let mut slots = tdd.levels[root.idx()].slot_count();
    let mut unmarked = 0usize;
    row.clear();
    let mut i = 0;
    while i < chain.len() {
        let [t, r, _] = chain[i];
        let (left, right) = vtree.children(VtreeIdx(t));
        let sides = [
            tdd.is_structural_internal(left),
            tdd.is_structural_internal(right),
        ];
        for (c, structural) in [left, right].into_iter().zip(sides) {
            charge_child(&mut slots, &mut unmarked, structural, tdd.reference_slot_count(c));
        }
        let found = true_children(lim, tdd, VtreeIdx(t), r as usize, sides, row, i as u32 + 1)?;
        for (c, u) in [left, right].into_iter().zip(found) {
            let Some(u) = u else { continue };
            lim.try_push(chain, [c.0, u as u32, i as u32])?;
        }
        i += 1;
    }
    // A level held as a description is compacted as the walk compacts it,
    // which needs the walk's marks and visits: a block per level of the
    // chain, every slot marked but the dying node's, in the walk's order,
    // which is the chain's. What happens to such a level is decided before
    // any level changes.
    let described = chain
        .iter()
        .any(|&[t, _, _]| tdd.levels[t as usize].pairs.implicit().is_some());
    let mut plans = if described {
        Some(plan_complement_described(
            eng, tdd, chain, marks, remap, visits,
        )?)
    } else {
        None
    };
    lim.charge_work(2 * slots as u64);
    // From the bottom of the chain up, as the compaction goes: a described
    // level as the compaction leaves it, a stored one in place, its node
    // dropped and the nodes after it one slot down, and in a stored parent
    // the references past it one slot down.
    tdd.dirty.reserve_all(chain.len());
    for j in (0..chain.len()).rev() {
        let [c, k, up] = chain[j];
        let c = VtreeIdx(c);
        let last = tdd.levels[c.idx()].slot_count() - 1;
        match plans.as_mut().and_then(|plans| plans.take(c)) {
            Some(plan) => {
                compact_one_level(tdd, visits[j], marks, remap, Some(plan));
            }
            None => {
                let level = &tdd.levels[c.idx()];
                let pairs = tdd
                    .levels
                    .known_live_pairs(c.idx())
                    .map(|n| n - level.pair_count_at(k as usize));
                tdd.levels.mark_changed(c);
                let level = &mut tdd.levels[c.idx()];
                retire(level, k as usize);
                if let Some(pairs) = pairs {
                    level.leave_live_pairs(pairs);
                }
            }
        }
        tdd.invalidate_shrunk(c);
        let t = VtreeIdx(chain[up as usize][0]);
        if j > 0 && k as usize != last && tdd.levels[t.idx()].pairs.implicit().is_none() {
            let on_left = vtree.children(t).0 == c;
            renumber_past(&mut tdd.levels[t.idx()], on_left, k);
        }
    }
    let local = tdd.output.local.0;
    tdd.output.local = NodeIdx(local - u32::from(local > old.0));
    Ok(())
}

/// Build the synthetic marks and visits for `chain` and plan compaction for
/// any described levels in [`prune_complement`].
fn plan_complement_described<'e>(
    eng: &'e Engine,
    tdd: &Tdd,
    chain: &[[u32; 3]],
    marks: &mut Vec<u64>,
    remap: &mut Vec<u32>,
    visits: &mut Vec<Visit>,
) -> Result<Plans<'e>, OperationError> {
    let lim = eng.limits();
    visits.clear();
    lim.reserve(visits, chain.len())?;
    let mut used = 0usize;
    for (i, &[t, k, _]) in chain.iter().enumerate() {
        let t = VtreeIdx(t);
        let width = if i == 0 {
            tdd.levels[t.idx()].slot_count()
        } else {
            tdd.reference_slot_count(t)
        };
        let base = alloc_block(lim, marks, &mut used, &mut 0, width)?;
        mark_all(&mut marks[base..base + words(width)], width);
        marks[base + (k as usize >> 6)] &= !(1u64 << (k & 63));
        visits.push(Visit::new(t, base));
    }
    let mut remap_need = 0usize;
    for (j, &[c, _, up]) in chain.iter().enumerate().skip(1) {
        let c = VtreeIdx(c);
        let child = Child::at(visits[j].base, tdd.reference_slot_count(c), true);
        let parent = &mut visits[up as usize];
        if tdd.vtree.children(parent.level).0 == c {
            parent.left = child;
        } else {
            parent.right = child;
        }
        remap_need = remap_need.max(parent.left.span + parent.right.span);
    }
    lim.try_resize(remap, remap_need, UNREACHED)?;
    plan_levels(lim, tdd, visits.iter().rev().copied(), marks, remap)
}

/// Drop node `k` of a stored `level`, the nodes after it each one slot
/// down.
fn retire(level: &mut TddLevel, k: usize) {
    debug_assert!(
        level.pairs.implicit().is_none() && level.nodes.stored().len() == level.slot_count()
    );
    let dead = level.arena_pairs_at(k);
    level.nodes.stored_mut().remove(k);
    level.note_dead_pairs(dead);
    level.compact_pairs_if_stale();
}

/// Move every reference of the stored `level` on its left side when
/// `LEFT` and on its right when `RIGHT` past the slots its child there
/// dropped, `left` or `right`, each one slot down for each dropped slot
/// below it ([`slots_below`]). The children are structural, so a side is
/// the slot it names. The node words and the arena are each read whole,
/// without a branch on a node's kind ([`EncodedNode::move_past`]); the
/// arena's dead pairs and the nodes the compaction drops move too, which
/// nothing reads.
#[inline(never)]
fn move_past<const N: usize, const LEFT: bool, const RIGHT: bool>(
    level: &mut TddLevel,
    left: &[u32; N],
    right: &[u32; N],
) {
    debug_assert!(level.pairs.implicit().is_none());
    for node in level.nodes.stored_mut() {
        node.move_past::<N, LEFT, RIGHT>(left, right);
    }
    let past = |side: EncodedChildRef, dropped: &[u32; N]| {
        EncodedChildRef::from_raw(side.0 - slots_below(side.0, dropped))
    };
    for pair in level.pairs.stored_mut() {
        if LEFT {
            pair.left = past(pair.left, left);
        }
        if RIGHT {
            pair.right = past(pair.right, right);
        }
    }
}

/// [`move_past`] on the slot `k` a child dropped, on the left side when
/// `on_left`, else the right.
fn renumber_past(level: &mut TddLevel, on_left: bool, k: u32) {
    match on_left {
        true => move_past::<1, true, false>(level, &[k], &[u32::MAX]),
        false => move_past::<1, false, true>(level, &[u32::MAX], &[k]),
    }
}

/// The most slots a child may drop for its parent to move its references
/// past them in one pass ([`move_past`]) rather than read each through the
/// child's new indices.
pub(super) const FEW_DROPPED: usize = 8;

/// The slots a child dropped, ascending and padded with `u32::MAX`, and
/// how many.
pub(super) struct Dropped {
    pub(super) slots: [u32; FEW_DROPPED],
    pub(super) len: usize,
}

impl Dropped {
    /// A child that kept every slot.
    const NONE: Dropped = Dropped {
        slots: [u32::MAX; FEW_DROPPED],
        len: 0,
    };
}

/// The unmarked slots among the first `span` of `block`, the slots a child
/// dropped, where they are [`FEW_DROPPED`] or fewer.
pub(super) fn few_dropped(block: &[u64], span: usize) -> Option<Dropped> {
    let mut dropped = Dropped::NONE;
    for (w, &word) in block[..words(span)].iter().enumerate() {
        let slots = (span - 64 * w).min(64);
        let mut lost = !word & tail_mask(slots);
        if dropped.len + lost.count_ones() as usize > FEW_DROPPED {
            return None;
        }
        while lost != 0 {
            dropped.slots[dropped.len] = (64 * w) as u32 + lost.trailing_zeros();
            dropped.len += 1;
            lost &= lost - 1;
        }
    }
    Some(dropped)
}

/// [`move_past`] on the slots each child dropped, `left` and `right`, at
/// the fewest compares a reference that cover them.
fn move_past_dropped(level: &mut TddLevel, left: &Dropped, right: &Dropped) {
    fn on<const N: usize>(level: &mut TddLevel, left: &Dropped, right: &Dropped) {
        let (l, r) = (
            left.slots[..N].try_into().expect("N slots"),
            right.slots[..N].try_into().expect("N slots"),
        );
        match (left.len > 0, right.len > 0) {
            (true, true) => move_past::<N, true, true>(level, l, r),
            (true, false) => move_past::<N, true, false>(level, l, r),
            (false, true) => move_past::<N, false, true>(level, l, r),
            (false, false) => {}
        }
    }
    match left.len.max(right.len) {
        0 => {}
        1 => on::<1>(level, left, right),
        2 => on::<2>(level, left, right),
        3 | 4 => on::<4>(level, left, right),
        _ => on::<FEW_DROPPED>(level, left, right),
    }
}

/// The nodes of the children of `t`, left and right, each where `sides`
/// asks for it (a structural internal child), that node `r` of `t` pairs
/// with every node of the other child's completed basis: every slot of an
/// internal sibling, or both literals of a leaf, which a `One` pair covers
/// at once. A child has at most one, so the read of `r`'s pairs stops once
/// it has found each one asked for.
///
/// `row` holds a count per slot, the left child's slots then the right's,
/// each stamped with the read that wrote it: a count stamped other than
/// `stamp` is zero.
fn true_children(
    lim: &Limits,
    tdd: &Tdd,
    t: VtreeIdx,
    r: usize,
    sides: [bool; 2],
    row: &mut Vec<u64>,
    stamp: u32,
) -> Result<[Option<usize>; 2], OperationError> {
    let (left, right) = tdd.vtree.children(t);
    let leaf = [
        tdd.vtree.node(left).is_leaf(),
        tdd.vtree.node(right).is_leaf(),
    ];
    let width = |c: VtreeIdx| tdd.levels[c.idx()].slot_count();
    // Side `k`'s sibling is the other child, `1 - k`.
    let full = [
        if leaf[1] { 2 } else { width(right) as u32 },
        if leaf[0] { 2 } else { width(left) as u32 },
    ];
    let offset = if sides[0] { width(left) } else { 0 };
    let need = offset + if sides[1] { width(right) } else { 0 };
    if row.len() < need {
        lim.try_resize(row, need, 0)?;
    }
    let tag = u64::from(stamp) << 32;
    let one = crate::diagram::ONE_LEAF_IDX.0;
    let mut count = |slot: usize, w: u32| -> u32 {
        let e = &mut row[slot];
        let n = if *e >> 32 == u64::from(stamp) {
            *e as u32 + w
        } else {
            w
        };
        *e = tag | u64::from(n);
        n
    };
    let (mut found, mut open) = ([None, None], sides);
    for pair in tdd.levels[t.idx()].pairs_iter_of_idx(r) {
        let (l, rr) = (pair.left.0, pair.right.0);
        if open[0] {
            let w = if leaf[1] && rr == one { 2 } else { 1 };
            if count(l as usize, w) == full[0] {
                (found[0], open[0]) = (Some(l as usize), false);
            }
        }
        if open[1] {
            let w = if leaf[0] && l == one { 2 } else { 1 };
            if count(offset + rr as usize, w) == full[1] {
                (found[1], open[1]) = (Some(rr as usize), false);
            }
        }
        if open == [false; 2] {
            break;
        }
    }
    Ok(found)
}

/// [`alloc_block`]'s arm for a block past the end of `marks`, out of line so
/// the walk's closure keeps the common arm inlined: it clears the words the
/// block reuses, and the resize zeroes the ones it adds.
#[inline(never)]
fn grow_block(lim: &Limits, marks: &mut Vec<u64>, base: usize, end: usize) -> Result<(), OperationError> {
    marks[base..].fill(0);
    lim.try_resize(marks, end, 0)
}

/// Reserve a clear block of marks for `width` fresh slots at the end of
/// `marks`, counting the slots in `slots`.
///
/// The reservation goes through the engine's limits. Nothing in the diagram has
/// been written while the walk is taking blocks, so a refusal leaves it as it
/// was.
#[inline(always)]
fn alloc_block(
    lim: &Limits,
    marks: &mut Vec<u64>,
    used: &mut usize,
    slots: &mut usize,
    width: usize,
) -> Result<usize, OperationError> {
    let base = *used;
    let end = base + words(width);
    if marks.len() < end {
        grow_block(lim, marks, base, end)?;
    } else {
        marks[base..end].fill(0);
    }
    *used = end;
    *slots += width;
    Ok(base)
}
