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
//! ([`settle_loose`]) marks a byte per slot instead, one level at a time.

use crate::diagram::{ChildDecoder, ChildKept, ChildPair, ChildSide, EncodedChildRef, EncodedNode, ImplicitLevel, NodeIdx, NodeKind, StoreRoom, Tdd, TddLevel};

use crate::Engine;

use crate::vtree::VtreeIdx;
use crate::limits::{Limits, OperationError, PollGate, Transient};
use crate::execution::pool::{Buffers, PooledScratch, Scratch};

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
        Marker { word: base, bits: 0 }
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

/// Mark every one of the `width` slots of `block`, leaving the bits past
/// `width` in the last word clear.
fn mark_all(block: &mut [u64], width: usize) {
    let full = width >> 6;
    block[..full].fill(u64::MAX);
    if width & 63 != 0 {
        block[full] = (1u64 << (width & 63)) - 1;
    }
}

/// Whether every one of the `width` slots of `block` is marked. Bits past
/// `width` in the last word are never set.
fn all_marked(block: &[u64], width: usize) -> bool {
    let full = width >> 6;
    let tail = width & 63;
    block[..full].iter().all(|&w| w == u64::MAX) && (tail == 0 || block[full] == (1u64 << tail) - 1)
}

/// Call `f` with the index of every marked slot of `block`, ascending.
#[inline(always)]
fn for_each_marked(block: &[u64], f: impl FnMut(usize)) {
    marked(block).for_each(f);
}

/// The index of every marked slot of `block`, ascending.
fn marked(block: &[u64]) -> Marked<'_> {
    Marked { block, word: 0, left: block.first().copied().unwrap_or(0) }
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
            x &= (1u64 << (width & 63)) - 1;
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
        Select { block, word: 0, before: 0, left: block.first().copied().unwrap_or(0), passed: 0 }
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
        let full = if slots.len() == 64 { u64::MAX } else { (1u64 << slots.len()) - 1 };
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
    /// The word each level's block of marks starts at, for the walk over the
    /// whole diagram.
    level_base: Vec<usize>,
    /// The reachability marks, one bit per slot.
    marks: Vec<u64>,
    /// The new indices a parent rewrites its references into a child level
    /// that lost a node through.
    remap: Vec<u32>,
    /// The ascending run a compaction remaps a child level kept whole
    /// through; it only grows, and is kept from one prune to the next.
    identity: Vec<u32>,
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
    /// The marks a [`settle_loose`] read sets, one byte per slot.
    settle_marks: Vec<u8>,
    /// [`prune_complement`]'s levels that lose a node, top-down: the level,
    /// the slot of the node it loses, and the entry of its parent level.
    chain: Vec<[u32; 3]>,
    /// [`prune_complement`]'s stamped count of the pairs the dying node of a
    /// level gives each node of its children ([`true_children`]).
    row: Vec<u64>,
}

impl Buffers for PruneScratch {
    fn buffers(&mut self, visit: &mut dyn FnMut(&mut dyn Scratch)) {
        visit(&mut self.level_base);
        visit(&mut self.marks);
        visit(&mut self.remap);
        visit(&mut self.identity);
        visit(&mut self.visits);
        visit(&mut self.entered);
        visit(&mut self.from_output);
        visit(&mut self.settle_flags);
        visit(&mut self.settle_parents);
        visit(&mut self.settle_marks);
        visit(&mut self.chain);
        visit(&mut self.row);
    }
}

impl PooledScratch for PruneScratch {
    fn prepare(&mut self) {
        self.level_base.clear();
        self.marks.clear();
        self.remap.clear();
        self.identity.clear();
        self.visits.clear();
        self.entered.clear();
        self.from_output.clear();
        self.settle_flags.clear();
        self.settle_parents.clear();
        self.settle_marks.clear();
        self.chain.clear();
        self.row.clear();
    }
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
/// onto the contract worklists (`seed_dirty_levels`), so the caller needs no
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
/// refused. Every one is taken before any mutation of `tdd`, so the diagram
/// is then untouched.
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
            let loose = settle_loose(eng, tdd, loose)?;
            Some(Seeds::of(eng, tdd, &loose)?)
        }
        _ => None,
    };
    if below_root {
        prune_below_root(eng, tdd, None, Start::Output)?;
    } else if let Some(seeds) = &seeded {
        let start = match seeds.root_clean {
            true => Start::Seeds(&seeds.seeds[..], &seeds.state[..], &seeds.top[..]),
            false => Start::Output,
        };
        prune_below_root(eng, tdd, Some(&seeds.state[..]), start)?;
        #[cfg(debug_assertions)]
        debug_assert_all_reached(tdd);
    } else {
        prune_whole(eng, tdd)?;
    }
    tdd.dirty.set_loose(Some(Vec::new()));
    Ok(())
}

/// [`prune_unreachable`] of a diagram that knows its loose levels, walked
/// from the output whether or not the root is clean: what a test charges
/// the seeded walk against.
#[cfg(test)]
pub(crate) fn prune_loose_from_output(eng: &Engine, tdd: &mut Tdd) -> Result<(), OperationError> {
    debug_assert!(below_root_walk_applies(tdd) && !tdd.has_marginal_level());
    let loose = tdd.dirty.loose().expect("the diagram knows its loose levels").to_vec();
    let loose = settle_loose(eng, tdd, &loose)?;
    let seeds = Seeds::of(eng, tdd, &loose)?;
    prune_below_root(eng, tdd, Some(&seeds.state[..]), Start::Output)?;
    tdd.dirty.set_loose(Some(Vec::new()));
    Ok(())
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
struct Seeds<'l> {
    /// Per level, [`FORCED`], and [`CLEAN`] on the clean levels that are
    /// forced or a seed's children: the walk reads it on a seed's children.
    state: Transient<'l, Vec<u8>>,
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
    fn of(eng: &'l Engine, tdd: &Tdd, loose: &[u32]) -> Result<Seeds<'l>, OperationError> {
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
            let mut path = Vec::new();
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
                let Some(p) = vtree.node(level).parent() else { continue };
                if state[p.idx()] & (CLEAN | SEED) == CLEAN && tdd.is_structural_internal(p) && tdd.is_structural_internal(level) {
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
        Ok(Seeds { state, seeds, top, root_clean })
    }
}

/// Every slot of every structural internal level is reachable from the output.
#[cfg(debug_assertions)]
pub(crate) fn debug_assert_all_reached(tdd: &Tdd) {
    if tdd.is_zero() {
        return;
    }
    let num_nodes = tdd.vtree.num_nodes();
    let mut level_base = vec![0usize; num_nodes + 1];
    for i in 0..num_nodes {
        level_base[i + 1] = level_base[i] + words(tdd.reference_slot_count(VtreeIdx(i as u32)));
    }
    let mut marks = vec![0u64; level_base[num_nodes]];
    classic_mark(&Limits::new(), tdd, &level_base, &mut marks).expect("unbounded limits refuse no offsets");
    for t in tdd.vtree.internal_bottomup().map(|(t, _, _)| t) {
        if !tdd.is_structural_internal(t) {
            continue;
        }
        let width = tdd.levels[t.idx()].slot_count();
        assert!(
            all_marked(&marks[level_base[t.idx()]..level_base[t.idx() + 1]], width),
            "level {} keeps a node the output does not reach",
            t.0
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
pub(crate) fn settle_loose(eng: &Engine, tdd: &Tdd, listed: &[u32]) -> Result<Vec<u32>, OperationError> {
    if listed.is_empty() {
        return Ok(Vec::new());
    }
    let lim = eng.limits();
    let vtree = &tdd.vtree;
    // Per level, whether it stays listed (`AT`) and whether a level strictly
    // under it stays listed (`UNDER`): set on the levels above each one that
    // stays, up to the first set already, whose own levels above are set.
    const AT: usize = 0;
    const UNDER: usize = 1;
    // The prune's scratch, which a prune checks out after this returns: a
    // read clears the block of marks it takes.
    let mut scratch = eng.scratch.reduce.prune.checkout_preserving(eng);
    let PruneScratch { settle_marks: marks, settle_flags: flags, settle_parents: parents, .. } = &mut *scratch;
    flags.clear();
    lim.try_resize(flags, vtree.num_nodes(), [false; 2])?;
    for &t in listed {
        flags[t as usize][AT] = true;
    }
    // The listed levels' parents, bottom-up, so that every level under a
    // parent's children is settled before the parent is read.
    parents.clear();
    lim.reserve(parents, listed.len())?;
    parents.extend(listed.iter().filter_map(|&t| vtree.node(VtreeIdx(t)).parent()));
    parents.sort_unstable_by_key(|&p| vtree.topo_pos(p));
    parents.dedup();
    let mut gate = lim.gate();
    for &p in parents.iter() {
        let (left, right) = vtree.children(p);
        if p != vtree.root() && !flags[left.idx()][UNDER] && !flags[right.idx()][UNDER] {
            let sides = [flags[left.idx()][AT], flags[right.idx()][AT]];
            let stays = unnamed_children(lim, tdd, p, sides, marks, &mut gate)?;
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
    Ok(listed.iter().copied().filter(|&t| flags[t as usize][AT]).collect())
}

/// Which of the children of level `p` that `sides` asks for, left and right,
/// hold a node no pair of `p` names, reading `p`'s pairs in one pass until
/// they have named every node of each, one unit of work through `gate` a
/// pair. A level with no node holds none, and a level that cannot be read
/// counts as holding one: one that is not structural internal, and each
/// child of a `p` that is not.
fn unnamed_children(
    lim: &Limits,
    tdd: &Tdd,
    p: VtreeIdx,
    sides: [bool; 2],
    marks: &mut Vec<u8>,
    gate: &mut PollGate<'_>,
) -> Result<[bool; 2], OperationError> {
    if sides == [false; 2] || !tdd.is_structural_internal(p) {
        return Ok(sides);
    }
    let (left, right) = tdd.vtree.children(p);
    let children = [left, right];
    // The nodes of each side not named yet, and the byte its block starts at.
    let (mut open, mut base, mut unread) = ([0usize; 2], [0usize; 2], [false; 2]);
    let mut used = 0;
    for k in 0..2 {
        if !sides[k] {
            continue;
        }
        if !tdd.is_structural_internal(children[k]) {
            unread[k] = true;
            continue;
        }
        let width = tdd.levels[children[k].idx()].slot_count();
        base[k] = used;
        used += width;
        if marks.len() < used {
            lim.try_resize(marks, used, 0)?;
        }
        marks[base[k]..used].fill(0);
        open[k] = width;
    }
    let level = &tdd.levels[p.idx()];
    let marks = &mut marks[..used];
    match open.map(|n| n > 0) {
        [true, true] => mark_named::<true, true>(level, &mut open, base, marks, gate)?,
        [true, false] => mark_named::<true, false>(level, &mut open, base, marks, gate)?,
        [false, true] => mark_named::<false, true>(level, &mut open, base, marks, gate)?,
        [false, false] => {}
    }
    Ok([unread[0] || open[0] > 0, unread[1] || open[1] > 0])
}

/// Mark in `marks` the nodes of `level`'s children that its pairs name, on
/// the left side with `LEFT` and on the right with `RIGHT`, each side's
/// block at its byte of `base`, and count down `open` once for each node
/// marked the first time; node by node, each node's pairs read as a slice,
/// up to the first node past which no side read has a node open. Both
/// children are structural internal, so a side is the index of the node it
/// names. One unit of work through `gate` a pair.
///
/// A mark is a byte, and marking stores a constant: the count reads the
/// byte, but no mark depends on one read before it, so pairs that name
/// the same node, or nodes a word apart, do not wait on each other's
/// stores as a bit set in a shared word would. No branch tests whether a
/// node was named already: a child of a level read to its end, as a tight
/// one mostly is, is named by many of its pairs, and the test would decide
/// every one.
#[inline(always)]
fn mark_named<const LEFT: bool, const RIGHT: bool>(
    level: &TddLevel,
    open: &mut [usize; 2],
    base: [usize; 2],
    marks: &mut [u8],
    gate: &mut PollGate<'_>,
) -> Result<(), OperationError> {
    let mut mark = |k: usize, s: u32, open: &mut [usize; 2]| {
        let m = &mut marks[base[k] + s as usize];
        open[k] -= (*m == 0) as usize;
        *m = 1;
    };
    // `Err(None)` stops the read once every node read for is named.
    let read = level.try_for_each_node(0..level.nodes().len(), |_, pairs| {
        if (!LEFT || open[0] == 0) && (!RIGHT || open[1] == 0) {
            return Err(None);
        }
        gate.poll(pairs.len() as u64).map_err(Some)?;
        for pair in pairs {
            if LEFT {
                mark(0, pair.left.0, open);
            }
            if RIGHT {
                mark(1, pair.right.0, open);
            }
        }
        Ok(())
    });
    match read {
        Ok(()) | Err(None) => Ok(()),
        Err(Some(e)) => Err(e),
    }
}

/// [`prune_unreachable`] over every level: mark from the output, then compact
/// each level bottom-up.
fn prune_whole(eng: &Engine, tdd: &mut Tdd) -> Result<(), OperationError> {
    let num_nodes = tdd.vtree.num_nodes();

    let mut scratch = eng.scratch.reduce.prune.checkout_preserving(eng);
    let PruneScratch { level_base, marks, remap, identity, .. } = &mut *scratch;

    // Flat offset table, in words: level t's marks are
    // marks[level_base[t]..level_base[t+1]]. Only a structural internal
    // level has a block; the marks of a leaf or marginal level would never be
    // read. The work charged counts every level's slots, and
    // `reference_slot_count()` gives leaf levels `LEAF_WIDTH` slots for
    // marginal nodes.
    eng.limits().try_resize(level_base, num_nodes + 1, 0usize)?;
    level_base[0] = 0;
    let mut slots = 0usize;
    for i in 0..num_nodes {
        let t = VtreeIdx(i as u32);
        let width = tdd.reference_slot_count(t);
        slots += width;
        let own = if tdd.is_structural_internal(t) { words(width) } else { 0 };
        level_base[i + 1] = level_base[i] + own;
    }
    let total = level_base[num_nodes];

    // The exact form leaves the Vec untouched on failure, so the pooled buffer
    // can be returned; through the engine's limits so the reservation is
    // charged against the byte budget.
    let need = total.saturating_sub(marks.len());
    eng.limits().reserve_exact(marks, need)?;
    // Clear the marks. Split so the grown tail is initialized once, by
    // `resize`, rather than written by `resize` and again by the fill.
    let warm = marks.len().min(total);
    marks[..warm].fill(0);
    if marks.len() < total {
        marks.resize(total, 0);
    }
    // ── Pass 1 (top-down): mark reachable nodes ──────────────────────────
    classic_mark(eng.limits(), tdd, level_base, &mut marks[..total])?;

    // `level_dirty[t]` records whether level `t` lost a node; charged for
    // this prune and handed back with it.
    let mut level_dirty: Transient<'_, Vec<bool>> = Transient::new(eng.limits(), Vec::new());
    eng.limits().try_resize(&mut level_dirty, num_nodes, false)?;
    let vtree = std::sync::Arc::clone(&tdd.vtree);
    // What the compaction reads its children's new indices from, taken
    // before it changes anything: a dirty child's indices side by side with
    // its sibling's, and the identity run for a child kept whole.
    let (mut remap_need, mut identity_need) = (0usize, 0usize);
    for &t in vtree.bottomup_slice() {
        if !tdd.is_structural_internal(t) {
            continue;
        }
        let block = &marks[level_base[t.idx()]..level_base[t.idx() + 1]];
        level_dirty[t.idx()] = !all_marked(block, tdd.reference_slot_count(t));
        let (left, right) = vtree.children(t);
        if level_dirty[left.idx()] || level_dirty[right.idx()] {
            let mut need = 0;
            for c in [left, right] {
                if level_dirty[c.idx()] {
                    need += span(level_base, c);
                } else {
                    identity_need = identity_need.max(tdd.reference_slot_count(c));
                }
            }
            remap_need = remap_need.max(need);
        }
    }
    eng.limits().try_resize(remap, remap_need, UNREACHED)?;
    identity_upto(eng, identity, identity_need)?;

    let child = |c: VtreeIdx| Child::at(level_base[c.idx()], span(level_base, c), level_dirty[c.idx()]);
    let order = vtree.bottomup_slice().iter().filter(|&&t| tdd.is_structural_internal(t)).map(|&t| {
        let (left, right) = vtree.children(t);
        (t, level_base[t.idx()], child(left), child(right))
    });
    let plans = plan_levels(eng.limits(), tdd, order, &marks[..total], remap)?;
    compact_levels(tdd, &vtree, level_base, &level_dirty, &marks[..total], remap, identity, plans);
    seed_dirty_levels(tdd, &level_dirty);

    let out = tdd.output;
    if level_dirty[out.vtree.idx()] {
        let block = &marks[level_base[out.vtree.idx()]..level_base[out.vtree.idx() + 1]];
        tdd.output.local = NodeIdx(rank(block, out.local.idx()));
    }

    // Both passes cross every reference slot, and neither can stop partway:
    // `compact_levels` rewrites levels in place, so a diagram abandoned mid-pass
    // has some levels compacted and others still naming their old indices.
    // Charge the walk so a work budget sees it; the cancellation test belongs to
    // the callers, between prunes.
    eng.limits().charge_work(2 * slots as u64);

    Ok(())
}

/// The slots level `t`'s block of marks covers, a whole number of words.
fn span(level_base: &[usize], t: VtreeIdx) -> usize {
    (level_base[t.idx() + 1] - level_base[t.idx()]) * 64
}

/// Pass 2 (bottom-up): compact every level that lost a node and rewrite the
/// references into it. `level_base` says where each level's block of marks
/// starts, and `level_dirty` whether it lost a node.
#[expect(clippy::too_many_arguments)]
fn compact_levels(
    tdd: &mut Tdd,
    vtree: &crate::vtree::Vtree,
    level_base: &[usize],
    level_dirty: &[bool],
    marks: &[u64],
    remap: &mut [u32],
    identity: &[u32],
    mut plans: Plans,
) {
    // Bottom-up topological order, so a level's children are compacted
    // before it rewrites its references into them; raw indices do not encode
    // parent/child order on a rotated vtree. A level whose children kept
    // every slot and that kept every slot itself is not touched.
    //
    // Leaf and marginal levels are never compacted. A leaf level's nodes are
    // implicit. A marginal level keeps its content in a value store that
    // parents reference by store-relative slot index, and such refs may be
    // minted after this prune (contract's inline-to-slot redirect). The walk
    // marks only the slots referenced right now, so compacting the store here
    // could drop a slot a later ref reads. Marginal stores keep their full
    // length; `prune_value_slots` collects their orphans.
    for &t in vtree.bottomup_slice() {
        if !tdd.is_structural_internal(t) {
            continue;
        }
        let (left, right) = vtree.children(t);
        let child = |c: VtreeIdx| Child::at(level_base[c.idx()], span(level_base, c), level_dirty[c.idx()]);
        let (l, r) = (child(left), child(right));
        let plan = plans.take(t);
        compact_one_level(tdd, t, &marks[level_base[t.idx()]..], marks, remap, identity, l, r, plan);
    }
}

/// Where a child level's marks are, as the parent rewriting its references
/// reads them.
#[derive(Clone, Copy)]
struct Child {
    /// The word its block of marks starts at; unused when it kept every slot.
    base: usize,
    /// The slots its new indices are written for, the room they take: its
    /// width before the prune at least. The walk below the root has the
    /// width at hand; the walk over the whole diagram, which reads it off
    /// its block's offsets once the child is compacted, rounds it up to
    /// whole words.
    span: usize,
    /// It lost a slot, so the parent rewrites its references through the new
    /// indices of the survivors; otherwise through the identity run.
    dirty: bool,
}

impl Child {
    /// A child with its own block of marks.
    const fn at(base: usize, span: usize, dirty: bool) -> Child {
        Child { base, span, dirty }
    }

    /// A child that kept every slot.
    const KEPT: Child = Child { base: 0, span: 0, dirty: false };
}

/// Compact one structural level: rewrite its references into a child that
/// lost a slot through that child's new indices, and drop its own unmarked
/// nodes. `own` starts at the level's block of marks, and `marks` holds the
/// children's. Returns whether the level lost a node.
///
/// A level held as the description of its pairs is kept as `plan`, decided
/// before the compaction changed any level ([`plan_levels`]): as the
/// description of what is left of it, when that is affine, or stored in the
/// room reserved for it ([`store_kept`]).
///
/// `remap` must hold the spans of the dirty children side by side, and
/// `identity` must be at least as long as the width of a child kept whole
/// beside a dirty one.
#[expect(clippy::too_many_arguments)]
fn compact_one_level(
    tdd: &mut Tdd,
    t: VtreeIdx,
    own: &[u64],
    marks: &[u64],
    remap: &mut [u32],
    identity: &[u32],
    left: Child,
    right: Child,
    plan: Option<Kept>,
) -> bool {
    let t_idx = t.idx();
    let width = tdd.levels[t_idx].slot_count();
    let own = &own[..words(width)];
    let lost = !all_marked(own, width);
    if lost || left.dirty || right.dirty {
        tdd.levels.mark_changed(t);
    }
    debug_assert_eq!(plan.is_some(), tdd.levels[t_idx].pairs.implicit().is_some() && (lost || left.dirty || right.dirty));

    let redescribed = match plan {
        Some(Kept::Described(d, dead)) => {
            tdd.levels[t_idx].pairs.redescribe(d);
            Some(dead)
        }
        Some(Kept::Stored(room)) => {
            let (lm, rm) = child_remaps(marks, remap, left, right);
            store_kept(tdd, t, room, own, lm, rm);
            None
        }
        None if left.dirty || right.dirty => {
            let (lm, rm) = child_remaps(marks, remap, left, right);
            rewrite_child_refs(tdd, t, own, lm.unwrap_or(identity), rm.unwrap_or(identity));
            None
        }
        None => None,
    };

    if !lost {
        return false;
    }
    // Compact the node Vec in place, O(width) with no allocation. Dropping a
    // node abandons its pair range; the arena is swept once enough of it is
    // dead, which is legal here because no pair offset is held across the
    // call.
    let level = &mut tdd.levels[t_idx];
    let dead = match redescribed {
        Some(dead) => dead,
        None => {
            // A node drops arena pairs only where the arena holds some: on
            // a level of inline nodes alone nothing is read.
            let mut dead = 0usize;
            if !level.pairs.is_empty() {
                for_each_unmarked(own, width, |i| dead += level.arena_pairs_at(i));
            }
            keep_marked(level.nodes.stored_mut(), own);
            dead
        }
    };
    level.note_dead_pairs(dead);
    level.compact_pairs_if_stale();
    true
}

/// The new indices of the children that lost a slot, `left` and `right`,
/// built from their marks into `remap`, side by side: `None` for a child
/// that kept every slot.
fn child_remaps<'r>(marks: &[u64], remap: &'r mut [u32], left: Child, right: Child) -> (Option<&'r [u32]>, Option<&'r [u32]>) {
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
enum Kept {
    /// What is left is affine: held as this description, with the pairs of
    /// the nodes dropped.
    Described(ImplicitLevel, usize),
    /// What is left is not affine: stored, in this room.
    Stored(StoreRoom),
}

/// The [`Kept`] of the levels a compaction changes that are held as
/// descriptions, in the compaction's order.
struct Plans(std::iter::Peekable<std::vec::IntoIter<(VtreeIdx, Kept)>>);

impl Plans {
    /// The plan of level `t`, the next level the compaction reaches, if it
    /// has one.
    fn take(&mut self, t: VtreeIdx) -> Option<Kept> {
        self.0.next_if(|(v, _)| *v == t).map(|(_, kept)| kept)
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
fn plan_levels(
    lim: &Limits,
    tdd: &Tdd,
    order: impl Iterator<Item = (VtreeIdx, usize, Child, Child)>,
    marks: &[u64],
    remap: &mut [u32],
) -> Result<Plans, OperationError> {
    let mut plans = Vec::new();
    for (t, base, left, right) in order {
        if let Some(kept) = plan_level(lim, tdd, t, &marks[base..], marks, remap, left, right)? {
            plans.push((t, kept));
        }
    }
    Ok(Plans(plans.into_iter().peekable()))
}

/// [`plan_levels`] for level `t`, whose block of marks starts `own`:
/// `None` when it is stored, or the compaction leaves it as it is.
#[expect(clippy::too_many_arguments)]
fn plan_level(
    lim: &Limits,
    tdd: &Tdd,
    t: VtreeIdx,
    own: &[u64],
    marks: &[u64],
    remap: &mut [u32],
    left: Child,
    right: Child,
) -> Result<Option<Kept>, OperationError> {
    let level = &tdd.levels[t.idx()];
    if level.pairs.implicit().is_none() {
        return Ok(None);
    }
    let width = level.slot_count();
    let own = &own[..words(width)];
    if all_marked(own, width) && !left.dirty && !right.dirty {
        return Ok(None);
    }
    let (lm, rm) = child_remaps(marks, remap, left, right);
    // The nodes a child that lost one kept, as its marks count them.
    let kept_of = |c: Child| c.dirty.then(|| marks[c.base..c.base + words(c.span)].iter().map(|w| w.count_ones() as usize).sum());
    Ok(Some(match kept_described(lim, tdd, t, own, (lm, kept_of(left)), (rm, kept_of(right)))? {
        Some((d, dead)) => Kept::Described(d, dead),
        None => Kept::Stored(level.store_room(lim)?),
    }))
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
        let full = if n == 64 { u64::MAX } else { (1u64 << n) - 1 };
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
    let (lc, rc) = tdd.vtree.children(t);
    let level = &tdd.levels[t.idx()];
    let d = level.pairs.implicit().expect("a description is kept as one");
    let k = d.pairs_per_node();
    let nodes: usize = own.iter().map(|w| w.count_ones() as usize).sum();
    // Past 2^31 pairs a node's range takes the side table (`ranges`); the
    // written arena's would too, and renumbered ones might not. What is left
    // below the floor is stored.
    if nodes == 0 || nodes * k < crate::diagram::floor() || level.pairs.len() >= 1 << 31 {
        return Ok(None);
    }
    // Below 2^31 pairs the description implies the nodes, node `i` of the
    // description being node `i` of the level.
    debug_assert!(level.implied_by().is_some());
    // A child's slots are its node indices unless it is marginal, whose
    // renumbering moves value slots; the derivation reads node indices.
    let kept = |c: VtreeIdx, n: Option<usize>| match n {
        None => Some(ChildKept::Whole),
        Some(n) => (!tdd.levels[c.idx()].child_decoder().is_marginal()).then_some(ChildKept::Renumbered(n)),
    };
    let derived = match (kept(lc, left.1), kept(rc, right.1)) {
        (Some(l), Some(r)) => d.kept_any(nodes, l, r).or_else(|| d.kept_box(own, nodes, l, r)),
        _ => None,
    };
    let fitted = || {
        let left = moved(tdd.levels[lc.idx()].child_decoder(), left.0);
        let right = moved(tdd.levels[rc.idx()].child_decoder(), right.0);
        let mut select = Select::new(own);
        d.pruned(lim, nodes, |j| select.nth(j), || marked(own), left, right)
    };
    #[cfg(debug_assertions)]
    if derived.is_some() && let Ok(fit) = fitted() {
        assert!(fit == derived, "a derived description differs from the fitted one");
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
fn store_kept(tdd: &mut Tdd, t: VtreeIdx, room: StoreRoom, own: &[u64], left: Option<&[u32]>, right: Option<&[u32]>) {
    let (lc, rc) = tdd.vtree.children(t);
    let left = moved(tdd.levels[lc.idx()].child_decoder(), left);
    let right = moved(tdd.levels[rc.idx()].child_decoder(), right);
    let marked = |i: usize| own[i >> 6] >> (i & 63) & 1 != 0;
    tdd.levels[t.idx()].store_moved(room, marked, left, right);
}

/// Rewrite level `t`'s child references through its child levels' remaps.
/// `own` is `t`'s own block of marks, and only its marked nodes are
/// rewritten; `left_remap` and `right_remap` are the children's new indices,
/// each starting at its own level's first slot.
///
/// Called only when a child level actually shrank: otherwise both child remaps
/// are the identity and every write would store a value back onto itself.
///
/// The three tables are parameters rather than one struct: a slice loaded out
/// of a struct loses the aliasing facts a slice parameter carries, and the
/// rewrite loop then re-reads the table pointers on every node.
fn rewrite_child_refs(
    tdd: &mut Tdd,
    t: VtreeIdx,
    own: &[u64],
    left_remap: &[u32],
    right_remap: &[u32],
) {
    let t_idx = t.idx();
    let (left, right) = tdd.vtree.children(t);
    let left_view = tdd.levels[left.idx()].child_decoder();
    let right_view = tdd.levels[right.idx()].child_decoder();
    let level = &mut tdd.levels[t_idx];
    for_each_marked(own, |i| match level.node(i).kind() {
        NodeKind::Inline(_) => {
            let node = &mut level.nodes.stored_mut()[i];
            node.a = left_view.remap(EncodedChildRef::from_raw(node.a), left_remap).0;
            node.b = right_view.remap(EncodedChildRef::from_raw(node.b), right_remap).0;
        }
        k if k.pairs_in_arena() => level.pairs_remap_indexed(i, left_remap, right_remap, left_view, right_view),
        _ => {}
    });
}

/// Where a child slot, as a raw word, moves when the child level read
/// through `view` is renumbered by `remap`: nowhere without one.
fn moved(view: ChildDecoder, remap: Option<&[u32]>) -> impl Fn(i64) -> i64 + '_ {
    move |x| remap.map_or(x, |m| i64::from(view.remap(EncodedChildRef::from_raw(x as u32), m).raw()))
}

/// Push every level prune shrank onto the contract worklists.
///
/// Removing a node simplifies the parent context of that level's children,
/// which can make two of them twins; contraction processes the parent, so the
/// shrunk level itself is what is pushed. Its own nodes cannot become twins
/// (a removal never equates two survivors), and its parents see only a
/// bijective child remap, so neither needs seeding. `level_dirty` is set on
/// non-leaf levels only, so every index pushed is a parent level.
fn seed_dirty_levels(tdd: &mut Tdd, level_dirty: &[bool]) {
    for (t_idx, dirty) in level_dirty.iter().enumerate() {
        if *dirty {
            tdd.invalidate_shrunk(VtreeIdx(t_idx as u32));
        }
    }
}

/// Pass 1: mark every slot reachable from the output node, walking the vtree
/// root first (raw indices do not follow topological order on a rotated
/// vtree). `marks` must be clear and `level_base`-indexed, in words, with a
/// block for every structural internal level.
///
/// Returns `Err(OperationError::OverBudget)` if the offsets an implicit
/// level's nodes are read in are refused ([`mark_described`]).
fn classic_mark(lim: &Limits, tdd: &Tdd, level_base: &[usize], marks: &mut [u64]) -> Result<(), OperationError> {
    let vtree = &tdd.vtree;
    // An output on a leaf or marginal level has no structural level under it.
    if !tdd.is_structural_internal(tdd.output.vtree) {
        return Ok(());
    }
    mark(marks, level_base[tdd.output.vtree.idx()], tdd.output.local.idx());
    let block = |c: VtreeIdx| tdd.is_structural_internal(c).then(|| level_base[c.idx()]);
    let topo = vtree.bottomup_slice();
    for v in topo.iter().rev() {
        let t = *v;
        // A marginal level has no pairs, and by invariant 5 every level
        // beneath it is marginal too, so there is nothing to mark below.
        if !tdd.is_structural_internal(t) {
            continue;
        }
        let (left, right) = vtree.children(t);
        mark_children_of_level(lim, tdd, t, level_base[t.idx()], block(left), block(right), marks)?;
    }
    Ok(())
}

/// Mark every child slot the marked nodes of level `t` name. `base` is the
/// word level `t`'s block starts at in `marks`, and `left` and `right` are
/// its children's, `None` for a child whose marks nobody reads: a leaf or
/// marginal level, which the compaction never drops a node from.
///
/// A side of a marginal child may be an inline count rather than a slot; such a
/// side names no child slot, so the decoder answers `None` for it and only real
/// slots are marked. A structural side is its own slot.
///
/// Returns `Err(OperationError::OverBudget)` if the offsets an implicit
/// level's nodes are read in are refused ([`mark_described`]).
fn mark_children_of_level(
    lim: &Limits,
    tdd: &Tdd,
    t: VtreeIdx,
    base: usize,
    left: Option<usize>,
    right: Option<usize>,
    marks: &mut [u64],
) -> Result<(), OperationError> {
    match (left, right) {
        (Some(l), Some(r)) => mark_sides::<true, true>(lim, tdd, t, base, l, r, marks),
        (Some(l), None) => mark_sides::<true, false>(lim, tdd, t, base, l, 0, marks),
        (None, Some(r)) => mark_sides::<false, true>(lim, tdd, t, base, 0, r, marks),
        (None, None) => Ok(()),
    }
}

/// [`mark_children_of_level`] marking the left child's slots at block
/// `left_base` when `LEFT`, and the right child's at `right_base` when
/// `RIGHT`.
fn mark_sides<const LEFT: bool, const RIGHT: bool>(
    lim: &Limits,
    tdd: &Tdd,
    t: VtreeIdx,
    base: usize,
    left_base: usize,
    right_base: usize,
    marks: &mut [u64],
) -> Result<(), OperationError> {
    let (left, right) = tdd.vtree.children(t);
    let left_view = tdd.levels[left.idx()].child_decoder();
    let right_view = tdd.levels[right.idx()].child_decoder();
    let level = &tdd.levels[t.idx()];
    let described = level.implicit();
    if let Some(d) = described
        && !left_view.is_marginal()
        && !right_view.is_marginal()
    {
        // A side's distinct offsets a marked node, never more marks than
        // its pairs.
        return mark_described::<LEFT, RIGHT>(lim, d, level.slot_count(), base, left_base, right_base, marks);
    }
    let (mut left_marks, mut right_marks) = (Marker::new(left_base), Marker::new(right_base));
    let mut mark = |marks: &mut [u64], pair: ChildPair| {
        if LEFT && let Some(s) = left_view.child(pair.left).index() {
            left_marks.mark(marks, left_base, s);
        }
        if RIGHT && let Some(s) = right_view.child(pair.right).index() {
            right_marks.mark(marks, right_base, s);
        }
    };
    // A stored level's marked nodes are read as slices of its arena, in a
    // loop apart from an implicit level's generated pairs.
    let width = level.slot_count();
    match (described, level.stored()) {
        (Some(d), _) => {
            let mut cursor = d.cursor();
            for_each_marked_beside(width, base, marks, |marks, i| {
                d.places_from(cursor.first_of(i)).for_each(|pair| mark(marks, pair));
            });
        }
        (None, Some(stored)) => for_each_marked_beside(width, base, marks, |marks, i| {
            stored.of_idx(i).iter().for_each(|&pair| mark(marks, pair));
        }),
        (None, None) => unreachable!("a level is stored or implicit"),
    }
    left_marks.finish(marks);
    right_marks.finish(marks);
    Ok(())
}

/// [`for_each_marked`] on the block of `width` slots at word `base` of
/// `marks`, with `marks` passed on to `f`: each word is read before `f` runs
/// on its slots, so that `f` may mark other blocks, a level's children's
/// beside its own.
#[inline(always)]
fn for_each_marked_beside(width: usize, base: usize, marks: &mut [u64], mut f: impl FnMut(&mut [u64], usize)) {
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
        fn new(lim: &'l Limits, d: &ImplicitLevel, side: ChildSide, on: bool, base: usize) -> Result<Side<'l>, OperationError> {
            let mut offsets = Transient::new(lim, Vec::new());
            if on {
                d.side_offsets(side, lim, &mut offsets)?;
            }
            Ok(Side { offsets, base, marker: Marker::new(base), last: None })
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

/// One level the seeded walk marked: where its marks live, and what it reads
/// its children's marks from.
#[derive(Clone, Copy)]
pub(crate) struct Visit {
    level: VtreeIdx,
    /// The word this level's block of marks starts at.
    base: usize,
    left: Child,
    right: Child,
}

impl Visit {
    /// A level whose block is marked but whose children are not settled yet.
    const fn new(level: VtreeIdx, base: usize) -> Visit {
        Visit { level, base, left: Child::KEPT, right: Child::KEPT }
    }
}

/// Where [`prune_below_root`] starts.
#[derive(Clone, Copy)]
enum Start<'a> {
    /// At the root, the output node its only slot reached: whatever else
    /// the root level holds is what a change at the root dropped.
    Output,
    /// At these levels, every node of each reached, with per level its
    /// [`Seeds`] state, and the clean levels above a loose one: the walk
    /// enters no [`CLEAN`] child.
    Seeds(&'a [VtreeIdx], &'a [u8], &'a [VtreeIdx]),
}

/// [`prune_unreachable`] under [`PruneScope::BelowRoot`]: walk down from
/// `start`, stopping at every level that loses no node. With `forced`, also
/// entering every level it marks whose entry has [`FORCED`] set, whether or
/// not it lost one.
fn prune_below_root(eng: &Engine, tdd: &mut Tdd, forced: Option<&[u8]>, start: Start<'_>) -> Result<(), OperationError> {
    let mut scratch = eng.scratch.reduce.prune.checkout_preserving(eng);
    let PruneScratch { marks, remap, identity, visits, entered, from_output, .. } = &mut *scratch;
    let vtree = std::sync::Arc::clone(&tdd.vtree);
    let root = tdd.output.vtree;
    visits.clear();
    // Words of `marks` taken, and the slots the walk is charged for.
    let mut used = 0usize;
    let mut slots = 0usize;
    // The widest pair of dirty children one level rewrites its references
    // into, side by side.
    let mut remap_need = 0usize;

    let clean = match start {
        Start::Output => {
            let root_width = tdd.levels[root.idx()].slot_count();
            let base = alloc_block(eng, marks, &mut used, &mut slots, root_width)?;
            mark(marks, base, tdd.output.local.idx());
            eng.limits().try_push(visits, Visit::new(root, base))?;
            None
        }
        Start::Seeds(seeds, state, _) => {
            eng.limits().reserve(visits, seeds.len())?;
            for &t in seeds {
                let width = tdd.levels[t.idx()].slot_count();
                let base = alloc_block(eng, marks, &mut used, &mut slots, width)?;
                mark_all(&mut marks[base..base + words(width)], width);
                visits.push(Visit::new(t, base));
            }
            Some(state)
        }
    };
    let is_clean = |c: VtreeIdx| clean.is_some_and(|s| s[c.idx()] & CLEAN != 0);

    // A child the walk does not descend into, a leaf level, whose remap is
    // the identity, or a marginal one, which is never compacted, gets no
    // block: its marks would never be read. Such a child is charged its slots
    // when it is wider than every one met before it, and nothing otherwise.
    let mut unmarked = 0usize;

    // Marking, top-down. A level has one parent, so a level's marks are
    // complete as soon as that parent has been walked, and a queue is a valid
    // order.
    let mut i = 0;
    while i < visits.len() {
        let t = visits[i].level;
        let (left, right) = vtree.children(t);
        let (lw, rw) = (tdd.reference_slot_count(left), tdd.reference_slot_count(right));
        // The identity run stands in for either child the level keeps whole.
        identity_upto(eng, identity, lw.max(rw))?;
        let mut block = |c: VtreeIdx, width: usize| -> Result<Option<usize>, OperationError> {
            // A clean child, under a seed, is all reached: no block.
            if is_clean(c) {
                return Ok(None);
            }
            if tdd.is_structural_internal(c) {
                return alloc_block(eng, marks, &mut used, &mut slots, width).map(Some);
            }
            if unmarked < width {
                slots += width;
                unmarked = width;
            }
            Ok(None)
        };
        let (lb, rb) = (block(left, lw)?, block(right, rw)?);

        mark_children_of_level(eng.limits(), tdd, t, visits[i].base, lb, rb, marks)?;

        let (lc, rc) = (settle_child(marks, lb, lw), settle_child(marks, rb, rw));
        visits[i].left = lc;
        visits[i].right = rc;
        remap_need = remap_need.max(lc.span + rc.span);
        let enter = |child: VtreeIdx, dirty: bool| dirty || forced.is_some_and(|f| f[child.idx()] & FORCED != 0);
        if let Some(base) = lb && enter(left, lc.dirty) {
            eng.limits().try_push(visits, Visit::new(left, base))?;
        }
        if let Some(base) = rb && enter(right, rc.dirty) {
            eng.limits().try_push(visits, Visit::new(right, base))?;
        }
        i += 1;
    }
    // A walk from the seeds is charged as one from the output would be.
    if let Start::Seeds(_, _, top) = start {
        slots = slots_from_output(eng.limits(), tdd, top, visits, entered, from_output)?;
    }
    compact_visited(eng, tdd, marks, remap, identity, visits, remap_need)?;

    if matches!(start, Start::Output) {
        tdd.output.local = NodeIdx(rank(&marks[visits[0].base..], tdd.output.local.idx()));
    }

    // As in `prune_whole`: both walks cross every slot they are charged for,
    // and neither can stop partway.
    eng.limits().charge_work(2 * slots as u64);
    Ok(())
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
    let levels = || std::iter::once(root).chain(top.iter().copied()).chain(visits.iter().map(|v| v.level));
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
            let width = tdd.reference_slot_count(c);
            if !tdd.is_structural_internal(c) {
                if unmarked < width {
                    slots += width;
                    unmarked = width;
                }
                continue;
            }
            slots += width;
            if entered[c.idx()] {
                from_output.push(c);
            }
        }
        i += 1;
    }
    for t in levels() {
        entered[t.idx()] = false;
    }
    debug_assert_eq!(from_output.len(), count, "a level the seeded walk entered is not under one a walk from the output enters");
    Ok(slots)
}

/// Compact the levels of `visits`, each with its block of `marks`, a level
/// after its children: a level was pushed after its parent, so the visits
/// reversed put every level after its own children. `remap_need` is the
/// widest pair of dirty children a level rewrites its references into.
fn compact_visited(
    eng: &Engine,
    tdd: &mut Tdd,
    marks: &[u64],
    remap: &mut Vec<u32>,
    identity: &[u32],
    visits: &[Visit],
    remap_need: usize,
) -> Result<(), OperationError> {
    eng.limits().try_resize(remap, remap_need, UNREACHED)?;
    // What the compaction does with the levels held as descriptions is
    // decided first.
    let order = visits.iter().rev().map(|v| (v.level, v.base, v.left, v.right));
    let mut plans = plan_levels(eng.limits(), tdd, order, marks, remap)?;
    // The first level compacted makes room in the worklists for every level
    // still to come, so that each list grows once however many are pushed.
    let mut room = false;
    for k in (0..visits.len()).rev() {
        let v = visits[k];
        let plan = plans.take(v.level);
        if compact_one_level(tdd, v.level, &marks[v.base..], marks, remap, identity, v.left, v.right, plan) {
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
/// dying node's pairs, where [`prune_below_root`] reads every pair of every
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
pub(crate) fn prune_complement(eng: &Engine, tdd: &mut Tdd, old: NodeIdx) -> Result<(), OperationError> {
    debug_assert!(below_root_walk_applies(tdd) && !tdd.has_marginal_level());
    debug_assert_ne!(old, tdd.output.local);
    let lim = eng.limits();
    let mut scratch = eng.scratch.reduce.prune.checkout_preserving(eng);
    let PruneScratch { marks, remap, identity, visits, chain, row, .. } = &mut *scratch;
    let vtree = std::sync::Arc::clone(&tdd.vtree);
    let root = tdd.output.vtree;
    debug_assert_eq!(tdd.levels[root.idx()].slot_count(), 2, "the root holds the old output and its complement");
    chain.clear();
    lim.try_push(chain, [root.0, old.0, 0])?;
    // The slots the walk below the root is charged for: the root's, and on
    // each level of the chain, the levels it walks in the order it walks
    // them, the slots of its structural children and of another child wider
    // than every one met before it ([`prune_below_root`]).
    let mut slots = tdd.levels[root.idx()].slot_count();
    let mut unmarked = 0usize;
    row.clear();
    let mut i = 0;
    while i < chain.len() {
        let [t, r, _] = chain[i];
        let (left, right) = vtree.children(VtreeIdx(t));
        let sides = [tdd.is_structural_internal(left), tdd.is_structural_internal(right)];
        for (c, structural) in [left, right].into_iter().zip(sides) {
            let width = tdd.reference_slot_count(c);
            if structural {
                slots += width;
            } else if unmarked < width {
                slots += width;
                unmarked = width;
            }
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
    let described = chain.iter().any(|&[t, _, _]| tdd.levels[t as usize].pairs.implicit().is_some());
    let mut plans = None;
    if described {
        visits.clear();
        lim.reserve(visits, chain.len())?;
        let mut used = 0usize;
        for (i, &[t, k, _]) in chain.iter().enumerate() {
            let t = VtreeIdx(t);
            let width = if i == 0 { tdd.levels[t.idx()].slot_count() } else { tdd.reference_slot_count(t) };
            let base = alloc_block(eng, marks, &mut used, &mut 0, width)?;
            mark_all(&mut marks[base..base + words(width)], width);
            marks[base + (k as usize >> 6)] &= !(1u64 << (k & 63));
            visits.push(Visit::new(t, base));
            let (left, right) = vtree.children(t);
            identity_upto(eng, identity, tdd.reference_slot_count(left).max(tdd.reference_slot_count(right)))?;
        }
        let mut remap_need = 0usize;
        for (j, &[c, _, up]) in chain.iter().enumerate().skip(1) {
            let c = VtreeIdx(c);
            let child = Child::at(visits[j].base, tdd.reference_slot_count(c), true);
            let parent = &mut visits[up as usize];
            if vtree.children(parent.level).0 == c {
                parent.left = child;
            } else {
                parent.right = child;
            }
            remap_need = remap_need.max(parent.left.span + parent.right.span);
        }
        lim.try_resize(remap, remap_need, UNREACHED)?;
        let order = visits.iter().rev().map(|v| (v.level, v.base, v.left, v.right));
        plans = Some(plan_levels(lim, tdd, order, marks, remap)?);
    }
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
                let v = visits[j];
                compact_one_level(tdd, c, &marks[v.base..], marks, remap, identity, v.left, v.right, Some(plan));
            }
            None => {
                tdd.levels.mark_changed(c);
                retire(&mut tdd.levels[c.idx()], k as usize);
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

/// Drop node `k` of a stored `level`, the nodes after it each one slot
/// down.
fn retire(level: &mut TddLevel, k: usize) {
    debug_assert!(level.pairs.implicit().is_none() && level.nodes.stored().len() == level.slot_count());
    let dead = level.arena_pairs_at(k);
    level.nodes.stored_mut().remove(k);
    level.note_dead_pairs(dead);
    level.compact_pairs_if_stale();
}

/// Move every reference of the stored `level` on its left side when
/// `on_left`, else its right, past child slot `k` one slot down: the child
/// dropped slot `k`. The child is structural, so a side is the slot it
/// names. The node words and the arena are each read whole, without a
/// branch on a node's kind ([`EncodedNode::renumber_past`]); the arena's
/// dead pairs move too, which nothing reads.
fn renumber_past(level: &mut TddLevel, on_left: bool, k: u32) {
    debug_assert!(level.pairs.implicit().is_none());
    for node in level.nodes.stored_mut() {
        node.renumber_past(on_left, k);
    }
    let past = |side: EncodedChildRef| EncodedChildRef::from_raw(side.0 - u32::from(side.0 > k));
    let arena = level.pairs.stored_mut();
    if on_left {
        arena.iter_mut().for_each(|pair| pair.left = past(pair.left));
    } else {
        arena.iter_mut().for_each(|pair| pair.right = past(pair.right));
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
    let leaf = [tdd.vtree.node(left).is_leaf(), tdd.vtree.node(right).is_leaf()];
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
        let n = if *e >> 32 == u64::from(stamp) { *e as u32 + w } else { w };
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

/// What a walked level's parent reads for a child: one with no block or that
/// kept every slot needs no new indices, and one that lost a slot is walked
/// in turn.
fn settle_child(marks: &[u64], base: Option<usize>, width: usize) -> Child {
    match base {
        Some(base) if !all_marked(&marks[base..base + words(width)], width) => Child::at(base, width, true),
        _ => Child::KEPT,
    }
}

/// Reserve a clear block of marks for `width` fresh slots at the end of
/// `marks`, counting the slots in `slots`.
///
/// The reservation goes through the engine's limits. Nothing in the diagram has
/// been written while the walk is taking blocks, so a refusal leaves it as it
/// was.
fn alloc_block(
    eng: &Engine,
    marks: &mut Vec<u64>,
    used: &mut usize,
    slots: &mut usize,
    width: usize,
) -> Result<usize, OperationError> {
    let base = *used;
    let end = base + words(width);
    if marks.len() < end {
        eng.limits().try_resize(marks, end, 0)?;
    }
    marks[base..end].fill(0);
    *used = end;
    *slots += width;
    Ok(base)
}

/// Grow the shared identity run to `n` entries, `identity[i] == i`.
///
/// Every child level a compaction keeps whole beside a dirty one remaps
/// through it, so one ascending run serves all of them, and it only ever
/// grows.
fn identity_upto(eng: &Engine, identity: &mut Vec<u32>, n: usize) -> Result<(), OperationError> {
    if identity.len() < n {
        let start = identity.len();
        eng.limits().try_resize(identity, n, 0u32)?;
        for (i, slot) in identity.iter_mut().enumerate().skip(start) {
            *slot = i as u32;
        }
    }
    Ok(())
}
