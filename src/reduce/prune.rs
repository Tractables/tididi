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
//! nothing reads its marks.

use crate::diagram::{ChildDecoder, ChildPair, ChildSide, EncodedChildRef, EncodedNode, ImplicitLevel, NodeIdx, NodeKind, Tdd};

use crate::Engine;

use crate::vtree::VtreeIdx;
use crate::limits::{Limits, OperationError, PollGate, Transient};
use crate::execution::pool::{Buffers, PooledScratch, Scratch};

/// The new index of a slot that did not survive, in the remap a parent
/// rewrites its references through. Every survivor's index is below the
/// level's width, and a level of `u32::MAX` nodes cannot be allocated.
const UNREACHED: u32 = u32::MAX;

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

/// Whether every one of the `width` slots of `block` is marked. Bits past
/// `width` in the last word are never set.
fn all_marked(block: &[u64], width: usize) -> bool {
    let full = width >> 6;
    let tail = width & 63;
    block[..full].iter().all(|&w| w == u64::MAX) && (tail == 0 || block[full] == (1u64 << tail) - 1)
}

/// Call `f` with the index of every marked slot of `block`, ascending.
#[inline(always)]
fn for_each_marked(block: &[u64], mut f: impl FnMut(usize)) {
    for (w, &word) in block.iter().enumerate() {
        let mut x = word;
        while x != 0 {
            f((w << 6) + x.trailing_zeros() as usize);
            x &= x - 1;
        }
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
fn new_indices(block: &[u64], span: usize, out: &mut [u32]) {
    let mut next = 0u32;
    for (s, slot) in out[..span].iter_mut().enumerate() {
        if block[s >> 6] >> (s & 63) & 1 != 0 {
            *slot = next;
            next += 1;
        } else {
            *slot = UNREACHED;
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
}

impl Buffers for PruneScratch {
    fn buffers(&mut self, visit: &mut dyn FnMut(&mut dyn Scratch)) {
        visit(&mut self.level_base);
        visit(&mut self.marks);
        visit(&mut self.remap);
        visit(&mut self.identity);
        visit(&mut self.visits);
    }
}

impl PooledScratch for PruneScratch {
    fn prepare(&mut self) {
        self.level_base.clear();
        self.marks.clear();
        self.remap.clear();
        self.identity.clear();
        self.visits.clear();
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
    /// `expand_full` establishes the condition for the diagram a negation
    /// complements: it leaves every structural level covering the whole
    /// `lefts x rights` basis of its children, so every child slot is named by
    /// some pair of the level above.
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
/// down from the output through the levels above them, as [`settle_loose`]
/// leaves them, and the levels that lose a node. A level the walk leaves has
/// lost no node and has no loose level under it, so each node below it is
/// named by a surviving node of its parent level. The prune leaves no level
/// loose.
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
            level.nodes.clear();
            level.pairs.clear();
            level.ranges.clear();
            level.dead_pairs = 0;
        }
        tdd.dirty.set_loose(Some(Vec::new()));
        return Ok(());
    }

    let below_root = matches!(scope, PruneScope::BelowRoot) && below_root_walk_applies(tdd);
    // The levels the walk must enter whatever it finds: the parents of the
    // loose levels, as `settle_loose` leaves them, and every level above
    // them.
    let forced = match tdd.dirty.loose() {
        Some(loose) if !below_root && below_root_walk_applies(tdd) && !tdd.has_marginal_level() => {
            let loose = settle_loose(eng, tdd, loose)?;
            let mut forced: Transient<'_, Vec<bool>> = Transient::new(eng.limits(), Vec::new());
            eng.limits().try_resize(&mut forced, tdd.vtree.num_nodes(), false)?;
            for &level in &loose {
                let mut up = tdd.vtree.node(VtreeIdx(level)).parent();
                while let Some(p) = up {
                    if forced[p.idx()] {
                        break;
                    }
                    forced[p.idx()] = true;
                    up = tdd.vtree.node(p).parent();
                }
            }
            Some(forced)
        }
        _ => None,
    };
    if below_root {
        prune_below_root(eng, tdd, None)?;
    } else if let Some(forced) = &forced {
        prune_below_root(eng, tdd, Some(forced))?;
        #[cfg(debug_assertions)]
        debug_assert_all_reached(tdd);
    } else {
        prune_whole(eng, tdd)?;
    }
    tdd.dirty.set_loose(Some(Vec::new()));
    Ok(())
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
    classic_mark(tdd, &level_base, &mut marks);
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
    let mut at = vec![false; vtree.num_nodes()];
    for &t in listed {
        at[t as usize] = true;
    }
    // The listed levels' parents, bottom-up, so that every level under a
    // parent's children is settled before the parent is read.
    let mut parents: Vec<VtreeIdx> = listed.iter().filter_map(|&t| vtree.node(VtreeIdx(t)).parent()).collect();
    parents.sort_unstable_by_key(|&p| vtree.topo_pos(p));
    parents.dedup();
    // Whether a level strictly under each level stays listed: set on the
    // levels above each one that stays, up to the first set already, whose
    // own levels above are set.
    let mut under = vec![false; vtree.num_nodes()];
    let mut gate = lim.gate();
    let mut marks: Transient<'_, Vec<u64>> = Transient::new(lim, Vec::new());
    for p in parents {
        let (left, right) = vtree.children(p);
        if p != vtree.root() && !under[left.idx()] && !under[right.idx()] {
            let sides = [at[left.idx()], at[right.idx()]];
            let stays = unnamed_children(lim, tdd, p, sides, &mut marks, &mut gate)?;
            at[left.idx()] = stays[0];
            at[right.idx()] = stays[1];
        }
        if at[left.idx()] || at[right.idx()] {
            let mut up = Some(p);
            while let Some(t) = up.filter(|t| !under[t.idx()]) {
                under[t.idx()] = true;
                up = vtree.node(t).parent();
            }
        }
    }
    gate.flush()?;
    Ok(listed.iter().copied().filter(|&t| at[t as usize]).collect())
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
    marks: &mut Transient<'_, Vec<u64>>,
    gate: &mut PollGate<'_>,
) -> Result<[bool; 2], OperationError> {
    if sides == [false; 2] || !tdd.is_structural_internal(p) {
        return Ok(sides);
    }
    let (left, right) = tdd.vtree.children(p);
    let children = [left, right];
    // The nodes of each side not named yet, and the word its block starts at.
    let (mut open, mut base, mut unread) = ([0usize; 2], [0usize; 2], [false; 2]);
    marks.clear();
    for k in 0..2 {
        if !sides[k] {
            continue;
        }
        if !tdd.is_structural_internal(children[k]) {
            unread[k] = true;
            continue;
        }
        let width = tdd.levels[children[k].idx()].slot_count();
        base[k] = marks.len();
        lim.try_resize(marks, base[k] + words(width), 0)?;
        open[k] = width;
    }
    let views = children.map(|c| tdd.levels[c.idx()].child_decoder());
    let level = &tdd.levels[p.idx()];
    let mut buf = Vec::new();
    'read: for i in 0..level.slot_count() {
        if open == [0; 2] {
            break;
        }
        let pairs = level.pairs_read(i, &mut buf);
        gate.poll(pairs.len() as u64)?;
        for pair in pairs {
            for (k, side) in [pair.left, pair.right].into_iter().enumerate() {
                if open[k] > 0 && let Some(s) = views[k].child(side).index() {
                    let (word, bit) = (base[k] + (s >> 6), 1u64 << (s & 63));
                    if marks[word] & bit == 0 {
                        marks[word] |= bit;
                        open[k] -= 1;
                    }
                }
            }
            if open == [0; 2] {
                break 'read;
            }
        }
    }
    Ok([unread[0] || open[0] > 0, unread[1] || open[1] > 0])
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
    classic_mark(tdd, level_base, &mut marks[..total]);

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

    compact_levels(tdd, &vtree, level_base, &level_dirty, &marks[..total], remap, identity);
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
fn compact_levels(
    tdd: &mut Tdd,
    vtree: &crate::vtree::Vtree,
    level_base: &[usize],
    level_dirty: &[bool],
    marks: &[u64],
    remap: &mut [u32],
    identity: &[u32],
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
        compact_one_level(tdd, t, &marks[level_base[t.idx()]..], marks, remap, identity, l, r);
    }
}

/// Where a child level's marks are, as the parent rewriting its references
/// reads them.
#[derive(Clone, Copy)]
struct Child {
    /// The word its block of marks starts at; unused when it kept every slot.
    base: usize,
    /// The slots its block covers, a whole number of words: its width
    /// rounded up, the room its new indices take.
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
/// A level held as the description of its pairs stays one when what is left
/// of it is affine ([`redescribe`]); otherwise what is left is stored
/// ([`store_kept`]).
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
) -> bool {
    let t_idx = t.idx();
    let width = tdd.levels[t_idx].slot_count();
    let own = &own[..words(width)];
    let lost = !all_marked(own, width);

    let redescribed = if left.dirty || right.dirty {
        let (left_new, right_new) = remap.split_at_mut(if left.dirty { left.span } else { 0 });
        let left_remap: &[u32] = if left.dirty {
            new_indices(&marks[left.base..], left.span, left_new);
            left_new
        } else {
            identity
        };
        let right_remap: &[u32] = if right.dirty {
            new_indices(&marks[right.base..], right.span, right_new);
            &right_new[..right.span]
        } else {
            identity
        };
        let (lm, rm) = (left.dirty.then_some(left_remap), right.dirty.then_some(right_remap));
        let kept = redescribe(tdd, t, own, lm, rm);
        if kept.is_none() {
            if tdd.levels[t_idx].pairs.implicit().is_some() {
                store_kept(tdd, t, own, lm, rm);
            } else {
                rewrite_child_refs(tdd, t, own, left_remap, right_remap);
            }
        }
        kept
    } else if lost {
        let kept = redescribe(tdd, t, own, None, None);
        if kept.is_none() && tdd.levels[t_idx].pairs.implicit().is_some() {
            store_kept(tdd, t, own, None, None);
        }
        kept
    } else {
        None
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
            let mut dead = 0usize;
            for_each_unmarked(own, width, |i| dead += level.arena_pairs_at(i));
            let mut i = 0;
            level.nodes.retain(|_| {
                let keep = own[i >> 6] >> (i & 63) & 1 != 0;
                i += 1;
                keep
            });
            dead
        }
    };
    level.note_dead_pairs(dead);
    level.compact_pairs_if_stale();
    true
}

/// Keep level `t`, held as the description of its pairs, as one through a
/// prune: its marked nodes in their order, each with its pairs, whose child
/// slots move through `left` and `right`, the new indices of the children
/// that lost a node, when what is left is affine in a mixed radix
/// ([`ImplicitLevel::pruned`](crate::diagram::ImplicitLevel)). The nodes
/// then name the ranges of the new description, from the start of the arena,
/// and the arena keeps its length, as a written one keeps the pairs of the
/// nodes the prune drops until a sweep.
///
/// Returns the pairs of the nodes dropped, or `None`, with nothing changed,
/// when the level is written or what is left is not affine. Checks every
/// pair left, and writes none.
fn redescribe(tdd: &mut Tdd, t: VtreeIdx, own: &[u64], left: Option<&[u32]>, right: Option<&[u32]>) -> Option<usize> {
    let (lc, rc) = tdd.vtree.children(t);
    let left = moved(tdd.levels[lc.idx()].child_decoder(), left);
    let right = moved(tdd.levels[rc.idx()].child_decoder(), right);
    let level = &tdd.levels[t.idx()];
    let d = level.pairs.implicit()?;
    let k = d.pairs_per_node();
    let nodes: usize = own.iter().map(|w| w.count_ones() as usize).sum();
    // Past 2^31 pairs a node's range takes the side table (`ranges`); the
    // written arena's would too, and renumbered ones might not. What is left
    // below the floor is stored.
    if nodes == 0 || nodes * k < crate::diagram::floor() || level.pairs.len() >= 1 << 31 {
        return None;
    }
    let mut select = Select::new(own);
    let kept = |j: usize| {
        let range = level.arena_range(level.nodes[select.nth(j)?].kind())?;
        (range.len() == k && range.start.is_multiple_of(k)).then_some(range.start / k)
    };
    let left_of = d.pruned(nodes, kept, left, right)?;
    let mut dead = 0usize;
    for_each_unmarked(own, level.nodes.len(), |i| dead += level.arena_pairs_at(i));
    let level = &mut tdd.levels[t.idx()];
    // No more nodes than the level held: the Vec does not grow.
    level.nodes.clear();
    level.nodes.extend((0..nodes).map(|j| EncodedNode::multi_pair((j * k) as u32, k as u32)));
    level.pairs.redescribe(left_of);
    Some(dead)
}

/// Store the marked nodes of level `t`, held as the description of its
/// pairs, when what is left of it is not affine: their pairs, with the child
/// slots moved through `left` and `right`, the new indices of the children
/// that lost a node, in the arena a stored level would hold
/// ([`TddLevel::store_moved`]).
fn store_kept(tdd: &mut Tdd, t: VtreeIdx, own: &[u64], left: Option<&[u32]>, right: Option<&[u32]>) {
    let (lc, rc) = tdd.vtree.children(t);
    let left = moved(tdd.levels[lc.idx()].child_decoder(), left);
    let right = moved(tdd.levels[rc.idx()].child_decoder(), right);
    let marked = |i: usize| own[i >> 6] >> (i & 63) & 1 != 0;
    tdd.levels[t.idx()].store_moved(marked, left, right);
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
    for_each_marked(own, |i| match level.nodes[i].kind() {
        NodeKind::Inline(_) => {
            let node = &mut level.nodes[i];
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
            tdd.invalidate(VtreeIdx(t_idx as u32));
        }
    }
}

/// Pass 1: mark every slot reachable from the output node, walking the vtree
/// root first (raw indices do not follow topological order on a rotated
/// vtree). `marks` must be clear and `level_base`-indexed, in words, with a
/// block for every structural internal level.
fn classic_mark(tdd: &Tdd, level_base: &[usize], marks: &mut [u64]) {
    let vtree = &tdd.vtree;
    // An output on a leaf or marginal level has no structural level under it.
    if !tdd.is_structural_internal(tdd.output.vtree) {
        return;
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
        mark_children_of_level(tdd, t, level_base[t.idx()], block(left), block(right), marks);
    }
}

/// Mark every child slot the marked nodes of level `t` name. `base` is the
/// word level `t`'s block starts at in `marks`, and `left` and `right` are
/// its children's, `None` for a child whose marks nobody reads: a leaf or
/// marginal level, which the compaction never drops a node from.
///
/// A side of a marginal child may be an inline count rather than a slot; such a
/// side names no child slot, so the decoder answers `None` for it and only real
/// slots are marked. A structural side is its own slot.
fn mark_children_of_level(
    tdd: &Tdd,
    t: VtreeIdx,
    base: usize,
    left: Option<usize>,
    right: Option<usize>,
    marks: &mut [u64],
) {
    match (left, right) {
        (Some(l), Some(r)) => mark_sides::<true, true>(tdd, t, base, l, r, marks),
        (Some(l), None) => mark_sides::<true, false>(tdd, t, base, l, 0, marks),
        (None, Some(r)) => mark_sides::<false, true>(tdd, t, base, 0, r, marks),
        (None, None) => {}
    }
}

/// [`mark_children_of_level`] marking the left child's slots at block
/// `left_base` when `LEFT`, and the right child's at `right_base` when
/// `RIGHT`.
fn mark_sides<const LEFT: bool, const RIGHT: bool>(
    tdd: &Tdd,
    t: VtreeIdx,
    base: usize,
    left_base: usize,
    right_base: usize,
    marks: &mut [u64],
) {
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
        if LEFT {
            mark_described(d, ChildSide::Left, level.slot_count(), base, left_base, marks);
        }
        if RIGHT {
            mark_described(d, ChildSide::Right, level.slot_count(), base, right_base, marks);
        }
        return;
    }
    let mut buf = Vec::new();
    let mut cursor = described.map(|d| (d, d.cursor()));
    let (mut left_marks, mut right_marks) = (Marker::new(left_base), Marker::new(right_base));
    let mut mark = |marks: &mut [u64], pair: ChildPair| {
        if LEFT && let Some(s) = left_view.child(pair.left).index() {
            left_marks.mark(marks, left_base, s);
        }
        if RIGHT && let Some(s) = right_view.child(pair.right).index() {
            right_marks.mark(marks, right_base, s);
        }
    };
    for w in 0..words(level.slot_count()) {
        // The level's own block is disjoint from its children's, so the word
        // read here is not one the loop below writes.
        let mut x = marks[base + w];
        while x != 0 {
            let i = (w << 6) + x.trailing_zeros() as usize;
            x &= x - 1;
            match &mut cursor {
                Some((d, c)) => d.places_from(c.first_of(i)).for_each(|pair| mark(marks, pair)),
                None => level.pairs_read(i, &mut buf).iter().for_each(|&pair| mark(marks, pair)),
            }
        }
    }
    left_marks.finish(marks);
    right_marks.finish(marks);
}

/// [`mark_sides`] on one side of a level held as the description of its
/// pairs, whose children are structural, so that a side's word is the
/// child's slot: the slots off the digits, not off the pairs. A marked
/// node marks its first pair's slot shifted by each of the offsets the
/// place digits that move the side give, once for a run of marked nodes
/// that start at one slot; with every node marked, the nodes' first slots
/// are read off the node digits that move the side. What is marked is
/// what the pairs would mark, read in about the side's distinct offsets a
/// node, not its pairs.
fn mark_described(d: &ImplicitLevel, side: ChildSide, width: usize, base: usize, side_base: usize, marks: &mut [u64]) {
    let offsets = d.side_offsets(side);
    let mut marker = Marker::new(side_base);
    let mut from = |marks: &mut [u64], first: i64| {
        for &o in &offsets {
            marker.mark(marks, side_base, (first + o) as usize);
        }
    };
    if all_marked(&marks[base..base + words(width)], width) {
        d.each_side_first(side, |first, _| from(marks, first));
    } else {
        let mut last = None;
        let mut cursor = d.cursor();
        for w in 0..words(width) {
            // The level's own block is disjoint from its children's, so the
            // word read here is not one the marks below write.
            let mut x = marks[base + w];
            while x != 0 {
                let i = (w << 6) + x.trailing_zeros() as usize;
                x &= x - 1;
                let at = cursor.first_of(i);
                let first = match side {
                    ChildSide::Left => at.0,
                    ChildSide::Right => at.1,
                };
                if last != Some(first) {
                    from(marks, first);
                    last = Some(first);
                }
            }
        }
    }
    marker.finish(marks);
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

/// [`prune_unreachable`] under [`PruneScope::BelowRoot`]: walk down from the
/// output, stopping at every level that loses no node. With `forced`, also
/// entering every level it marks, whether or not it lost one.
fn prune_below_root(eng: &Engine, tdd: &mut Tdd, forced: Option<&[bool]>) -> Result<(), OperationError> {
    let mut scratch = eng.scratch.reduce.prune.checkout_preserving(eng);
    let PruneScratch { marks, remap, identity, visits, .. } = &mut *scratch;
    let vtree = std::sync::Arc::clone(&tdd.vtree);
    let root = tdd.output.vtree;
    visits.clear();
    // Words of `marks` taken, and the slots the walk is charged for.
    let mut used = 0usize;
    let mut slots = 0usize;
    // The widest pair of dirty children one level rewrites its references
    // into, side by side.
    let mut remap_need = 0usize;

    // The root's block, with the output node the only slot reached: whatever
    // else the root level holds is what the change at the root dropped.
    let root_width = tdd.levels[root.idx()].slot_count();
    let base = alloc_block(eng, marks, &mut used, &mut slots, root_width)?;
    mark(marks, base, tdd.output.local.idx());
    eng.limits().try_push(visits, Visit::new(root, base))?;

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

        mark_children_of_level(tdd, t, visits[i].base, lb, rb, marks);

        let (lc, rc) = (settle_child(marks, lb, lw), settle_child(marks, rb, rw));
        visits[i].left = lc;
        visits[i].right = rc;
        remap_need = remap_need.max(lc.span + rc.span);
        let enter = |child: VtreeIdx, dirty: bool| dirty || forced.is_some_and(|f| f[child.idx()]);
        if let Some(base) = lb && enter(left, lc.dirty) {
            eng.limits().try_push(visits, Visit::new(left, base))?;
        }
        if let Some(base) = rb && enter(right, rc.dirty) {
            eng.limits().try_push(visits, Visit::new(right, base))?;
        }
        i += 1;
    }
    eng.limits().try_resize(remap, remap_need, UNREACHED)?;

    // Compaction, bottom-up: a level was pushed after its parent, so the walk
    // order reversed puts every level after its own children.
    let marks: &[u64] = marks;
    for k in (0..visits.len()).rev() {
        let v = visits[k];
        if compact_one_level(tdd, v.level, &marks[v.base..], marks, remap, identity, v.left, v.right) {
            tdd.invalidate(v.level);
        }
    }

    tdd.output.local = NodeIdx(rank(&marks[visits[0].base..], tdd.output.local.idx()));

    // As in `prune_whole`: both walks cross every slot they are charged for,
    // and neither can stop partway.
    eng.limits().charge_work(2 * slots as u64);
    Ok(())
}

/// What a walked level's parent reads for a child: one with no block or that
/// kept every slot needs no new indices, and one that lost a slot is walked
/// in turn.
fn settle_child(marks: &[u64], base: Option<usize>, width: usize) -> Child {
    match base {
        Some(base) if !all_marked(&marks[base..base + words(width)], width) => {
            Child::at(base, words(width) * 64, true)
        }
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
