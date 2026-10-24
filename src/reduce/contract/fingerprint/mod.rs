use crate::diagram::{ChildPair, ChildSide, Tdd, TddLevel};
use crate::Engine;
use crate::vtree::VtreeIdx;

use crate::limits::{Limits, OperationError};

use super::scratch::{ContractScratch, EMPTY_SLOT, TwinSlot};

/// Prefetch the twin-table slot a future iteration will probe first. The
/// probe target is a random index into a table that typically misses L2;
/// `fingerprints[]` is read sequentially, so the slot for iteration i+D is
/// known D iterations ahead. No-op on non-x86_64, and under Miri, which does
/// not implement the intrinsic.
fn prefetch_slot(p: *const TwinSlot, slot: usize) {
    #[cfg(all(target_arch = "x86_64", not(miri)))]
    unsafe {
        core::arch::x86_64::_mm_prefetch(
            p.add(slot) as *const i8,
            core::arch::x86_64::_MM_HINT_T0,
        );
    }
    #[cfg(not(all(target_arch = "x86_64", not(miri))))]
    let _ = (p, slot);
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
    if t1_side == ChildSide::Left { (pair.left.0, pair.right.0) } else { (pair.right.0, pair.left.0) }
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
pub(super) fn mix64(mut x: u64) -> u64 {
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
    x ^ (x >> 31)
}

mod groups;
mod listed;

pub(super) use listed::{find_listed_twin_groups, named_by, pair_mass};

#[cfg(test)]
mod tests;

/// The entries whose multiset per node decides which nodes are twins. The
/// three scatters of [`group_twins_by_entries`] walk it, and read the same
/// `(node, entry)` sequence each time.
pub(super) trait TwinEntries {
    /// Call `f(node, entry)` once per entry of every node.
    fn for_each(&self, f: impl FnMut(u32, u64));

    /// Whether none of the `width` nodes can have a twin: no two nodes share
    /// an entry, and at most one has none. `false` when that is not known.
    fn no_twin(&self, _lim: &Limits, _scratch: &mut ContractScratch, _width: usize) -> Result<bool, OperationError> {
        Ok(false)
    }

    /// Whether the entries are `(parent node, sibling)` contexts packed
    /// parent-high, each parent node's entries read together, and no node
    /// has one entry twice: then a fingerprint group is verified owner by
    /// owner ([`groups::verify_fingerprint_groups`]).
    fn owner_keyed(&self) -> bool {
        false
    }

    /// Fingerprint only the nodes that can have a twin, where the entries
    /// support it: return how many there are, with their indices ascending
    /// in `scratch.candidates` and their fingerprints in
    /// `scratch.fingerprints`, or `None` where they do not. The other nodes
    /// each hold an entry no other node holds.
    fn scatter_shared(&self, _lim: &Limits, _scratch: &mut ContractScratch, _width: usize) -> Result<Option<usize>, OperationError> {
        Ok(None)
    }

    /// Call `f(node, entry)` once per entry of the parent nodes `parents`,
    /// where entries are keyed by parent node and every entry outside them
    /// is of a node no grouping reads. Reads every entry where that is not
    /// known.
    fn for_each_of_parents(&self, _parents: &[u32], f: impl FnMut(u32, u64)) {
        self.for_each(f);
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
    /// The diagram has no marginal level, so no parent node holds a pair twice.
    distinct: bool,
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

/// The fewest nodes a child level needs before its search marks the nodes
/// that hold a private context ([`TwinEntries::scatter_shared`]): on a
/// narrower level the fingerprints and their table stay in cache, and the
/// marking costs more than the probes it saves.
const PRIVATE_MIN_WIDTH: usize = 1 << 12;

/// The most pairs a parent node holds where the private context scatter
/// finds its repeated siblings by comparing every two of them.
const COMPARE_MAX_PAIRS: usize = 8;

/// The pairs the private context scatter files in its sibling table before
/// it weighs the table: one that finds fewer than half of the pairs it filed
/// private is not used for the rest of the level.
const TABLE_TRIAL_PAIRS: u64 = 1 << 12;

impl TwinEntries for ContextEntries<'_> {
    fn for_each(&self, mut f: impl FnMut(u32, u64)) {
        for_each_target_sibling(self.parent_level, self.t1_side, |pi, target, sibling| {
            f(target, pack(pi, sibling));
        });
    }

    fn owner_keyed(&self) -> bool {
        self.distinct
    }

    /// A node holds a private context where a parent node pairs it with a
    /// sibling it pairs with no other node, and then has no twin (math
    /// §2.8): a twin would hold that context too. Each parent node is read
    /// in cache, and each pair either marks its node private or adds its
    /// context to the node's fingerprint; a node all of whose contexts are
    /// shared is fingerprinted whole, and the fingerprint of a private node
    /// is not read. Nodes no pair names are not private: two of them are
    /// twins. The parent nodes that add a context are listed in
    /// `scratch.shared_parents`: every entry of a node that is not private
    /// lies in one of them.
    ///
    /// A parent node of at most [`COMPARE_MAX_PAIRS`] pairs finds its
    /// repeated siblings by comparing every two. A wider one files its
    /// siblings in a stamped table that notes repeats, for as long as the
    /// table finds most of the pairs it files private; past that the wider
    /// nodes add every context.
    fn scatter_shared(&self, lim: &Limits, scratch: &mut ContractScratch, width: usize) -> Result<Option<usize>, OperationError> {
        let ContractScratch { shared_local, shared_generation, twin_named, fingerprints, candidates, shared_parents, .. } =
            scratch;
        let words = width.div_ceil(64);
        lim.try_resize(twin_named, words, 0u64)?;
        let private = &mut twin_named[..words];
        private.fill(0);
        lim.try_resize(fingerprints, width, 0u64)?;
        let fp = &mut fingerprints[..width];
        fp.fill(0);
        let (level, side) = (self.parent_level, self.t1_side);
        let nodes = level.node_count();
        shared_parents.clear();
        lim.reserve_exact(shared_parents, nodes)?;
        let mut mark = |t: u32| private[(t / 64) as usize] |= 1 << (t % 64);
        let mut add = |t: u32, pi: u32, s: u32| {
            let f = &mut fp[t as usize];
            *f = f.wrapping_add(mix64(pack(pi, s)));
        };
        let stored = level.stored();
        let (mut buf, mut cursor) = (Vec::new(), None);
        // Pairs the table filed, and how many of them it found private.
        let (mut filed, mut filed_private) = (0u64, 0u64);
        let mut table = true;
        for i in 0..nodes {
            let pairs = match stored {
                Some(stored) => stored.of_idx(i),
                None => level.described_read_next(&mut cursor, i, &mut buf),
            };
            let pi = i as u32;
            let n = pairs.len();
            if n <= 1 {
                if let Some(pair) = pairs.first() {
                    mark(split_pair(pair, side).0);
                }
                continue;
            }
            if n <= COMPARE_MAX_PAIRS {
                let mut sib = [0u32; COMPARE_MAX_PAIRS];
                for (k, pair) in pairs.iter().enumerate() {
                    sib[k] = split_pair(pair, side).1;
                }
                // Bit k: pair k's sibling is another pair's too.
                let mut dup = 0u32;
                for k in 1..n {
                    for j in 0..k {
                        dup |= u32::from(sib[j] == sib[k]).wrapping_neg() & ((1 << j) | (1 << k));
                    }
                }
                for (k, pair) in pairs.iter().enumerate() {
                    let t = split_pair(pair, side).0;
                    if dup >> k & 1 != 0 {
                        add(t, pi, sib[k]);
                    } else {
                        mark(t);
                    }
                }
                if dup != 0 {
                    shared_parents.push(pi);
                }
                continue;
            }
            if !table {
                for pair in pairs {
                    let (t, s) = split_pair(pair, side);
                    add(t, pi, s);
                }
                shared_parents.push(pi);
                continue;
            }
            let cells = (2 * n).next_power_of_two();
            let stamp = next_shared_stamp(lim, shared_local, shared_generation, cells)?;
            let cell_table = &mut shared_local[..cells];
            let mut repeats = false;
            for pair in pairs {
                repeats |= file_counted(cell_table, stamp, split_pair(pair, side).1);
            }
            filed += n as u64;
            if !repeats {
                // Every context here is private.
                pairs.iter().for_each(|pair| mark(split_pair(pair, side).0));
                filed_private += n as u64;
                continue;
            }
            for pair in pairs {
                let (t, s) = split_pair(pair, side);
                if repeated(cell_table, stamp, s) {
                    add(t, pi, s);
                } else {
                    mark(t);
                    filed_private += 1;
                }
            }
            shared_parents.push(pi);
            table = filed < TABLE_TRIAL_PAIRS || 2 * filed_private >= filed;
        }
        let shared = width - private.iter().map(|w| w.count_ones() as usize).sum::<usize>();
        candidates.clear();
        if shared >= 2 {
            lim.reserve_exact(candidates, shared)?;
            for (w, &word) in private.iter().enumerate() {
                let mut open = !word;
                if w + 1 == words && !width.is_multiple_of(64) {
                    open &= (1u64 << (width % 64)) - 1;
                }
                while open != 0 {
                    candidates.push(w as u32 * 64 + open.trailing_zeros());
                    open &= open - 1;
                }
            }
        }
        Ok(Some(shared))
    }

    /// Every entry of the parent nodes `parents`, ascending, in the order
    /// [`for_each`](TwinEntries::for_each) reads them.
    fn for_each_of_parents(&self, parents: &[u32], mut f: impl FnMut(u32, u64)) {
        let (level, side) = (self.parent_level, self.t1_side);
        let stored = level.stored();
        let (mut buf, mut cursor) = (Vec::new(), None);
        for &pi in parents {
            let pairs = match stored {
                Some(stored) => stored.of_idx(pi as usize),
                None => level.described_read_next(&mut cursor, pi as usize, &mut buf),
            };
            for pair in pairs {
                let (t, s) = split_pair(pair, side);
                f(t, pack(pi, s));
            }
        }
    }

    /// Two nodes share a context only where a parent node pairs both with
    /// one sibling. Each parent node is tested alone, in cache, and the test
    /// stops at the first sibling a node repeats, where twins are likely; a
    /// level whose parent nodes never repeat one, common on grids, then needs
    /// no fingerprint. The nodes no pair names are counted on the way: two
    /// of them would be twins.
    ///
    /// A node of three pairs or more files its siblings in a
    /// generation-stamped table, the first cells of `twin_local`, a power of
    /// two at least twice the siblings filed, so a probe always ends at a
    /// cell this node has not stamped. A cell holds its stamp in the high half
    /// and the sibling in the low one, and finding the sibling already filed
    /// is the repeat: a wide node whose siblings repeat stops at the first
    /// one, after reading a few of its pairs. The table starts at most
    /// [`TWIN_TABLE_START_CELLS`] long and doubles when half full, under a
    /// fresh stamp the siblings read so far are filed again with, so its size
    /// follows the pairs read, not the node's width.
    fn no_twin(&self, lim: &Limits, scratch: &mut ContractScratch, width: usize) -> Result<bool, OperationError> {
        if !self.early_stop {
            return Ok(false);
        }
        let ContractScratch { twin_local, twin_generation, twin_named, .. } = scratch;
        let words = width.div_ceil(64);
        lim.try_resize(twin_named, words, 0u64)?;
        let named = &mut twin_named[..words];
        named.fill(0);
        let mut name = |t: u32| named[(t / 64) as usize] |= 1 << (t % 64);
        let (level, side) = (self.parent_level, self.t1_side);
        // A stored level's nodes are read as slices of its arena, an
        // implicit level's generated into a buffer a node at a time, in one
        // loop.
        let stored = level.stored();
        let (mut buf, mut cursor) = (Vec::new(), None);
        for i in 0..level.node_count() {
            let pairs = match stored {
                Some(stored) => stored.of_idx(i),
                None => level.described_read_next(&mut cursor, i, &mut buf),
            };
            if pairs.len() < 3 {
                let mut last: Option<u32> = None;
                for pair in pairs {
                    let (t, s) = split_pair(pair, side);
                    if last == Some(s) {
                        return Ok(false);
                    }
                    last = Some(s);
                    name(t);
                }
                continue;
            }
            let mut cells = (2 * pairs.len()).next_power_of_two().min(TWIN_TABLE_START_CELLS);
            let mut stamp = next_twin_stamp(lim, twin_local, twin_generation, cells)?;
            for (read, pair) in pairs.iter().enumerate() {
                let (t, s) = split_pair(pair, side);
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
                    return Ok(false);
                }
                name(t);
            }
        }
        let unnamed = width - named.iter().map(|w| w.count_ones() as usize).sum::<usize>();
        Ok(unnamed <= 1)
    }
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

/// The next stamp of the counted sibling table, with `local` at least
/// `cells` long: the generation in the high 31 bits of a cell, below them
/// the repeat bit, below that the sibling. On wrap past 31 bits the cells
/// are zeroed and it restarts at 1; 0 stamps no cell.
#[inline]
fn next_shared_stamp(
    lim: &Limits,
    local: &mut Vec<u64>,
    generation: &mut u32,
    cells: usize,
) -> Result<u64, OperationError> {
    *generation += 1;
    if *generation >= 1 << 31 {
        local.fill(0);
        *generation = 1;
    }
    if local.len() < cells {
        lim.try_resize(local, cells, 0u64)?;
    }
    Ok((*generation as u64) << 33)
}

/// The repeat bit of a counted sibling cell.
const REPEATED: u64 = 1 << 32;

/// The first cell of `local`, a power of two longer than the siblings filed
/// under `stamp`, where sibling `s` is filed or would be.
#[inline]
fn counted_cell(local: &[u64], stamp: u64, s: u32) -> usize {
    let (mask, shift) = (local.len() - 1, 32 - local.len().trailing_zeros());
    let mut h = (s.wrapping_mul(0x9E37_79B1) >> shift) as usize;
    loop {
        let cell = local[h];
        if cell >> 33 != stamp >> 33 || cell as u32 == s {
            return h;
        }
        h = (h + 1) & mask;
    }
}

/// File sibling `s` under `stamp`, noting a repeat if it was filed already;
/// return whether it was.
#[inline]
fn file_counted(local: &mut [u64], stamp: u64, s: u32) -> bool {
    let h = counted_cell(local, stamp, s);
    let filed = local[h] >> 33 == stamp >> 33;
    local[h] = if filed { local[h] | REPEATED } else { stamp | s as u64 };
    filed
}

/// Whether sibling `s`, filed under `stamp`, was filed more than once.
#[inline]
fn repeated(local: &[u64], stamp: u64, s: u32) -> bool {
    local[counted_cell(local, stamp, s)] & REPEATED != 0
}

/// Content twins: a node's entries are its own pairs, packed left-high.
pub(super) struct ContentEntries<'a>(pub(super) &'a TddLevel);

impl TwinEntries for ContentEntries<'_> {
    fn for_each(&self, mut f: impl FnMut(u32, u64)) {
        self.0.for_each_node_pair(|i, pair| f(i as u32, pack(pair.left.0, pair.right.0)));
    }
}

/// Find nodes with identical parent-context multisets at the explicit child
/// level `t1` of `parent`. Marginal children are handled by pair fusion
/// instead. See [`group_twins_by_entries`] for what is left in `scratch`.
pub(super) fn find_twin_groups(
    eng: &Engine,
    tdd: &Tdd,
    t1: VtreeIdx,
    parent: VtreeIdx,
    t1_side: ChildSide,
    scratch: &mut ContractScratch,
) -> Result<bool, OperationError> {
    let level = &tdd.levels[t1.idx()];
    debug_assert!(!level.is_marginal(), "a marginal child is left to pair fusion");
    let parent_level = &tdd.levels[parent.idx()];
    let distinct = !*scratch.diagram_marginal.get_or_insert_with(|| tdd.has_marginal_level());
    let entries = ContextEntries {
        parent_level,
        t1_side,
        early_stop: screens(parent_level),
        distinct,
    };
    group_twins_by_entries(eng, &entries, level.slot_count(), scratch)
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
    if width == 0 {
        return Ok(false);
    }
    // Node indices at this level are written u32-wide below (`cursors`,
    // `flat_groups`). That is the same invariant the merge side already relies
    // on (`merge_target[i] = i as u32`, `final_remap: Vec<NodeIdx>`): every
    // ref into a level is a `NodeIdx(u32)`, so a level wider than 2^32
    // slots could not be referenced at all.
    debug_assert!(
        width <= u32::MAX as usize,
        "level width {width} exceeds the u32 node-index range",
    );

    // A wide level whose nodes hold one entry each groups by sorting them.
    if width >= groups::SINGLE_ENTRY_MIN_WIDTH
        && let Some(found) = groups::group_single_entries(eng, entries, width, scratch)?
    {
        return Ok(found);
    }
    let verify = entries.owner_keyed();
    // A level narrower than `PRIVATE_MIN_WIDTH` is not marked: its
    // fingerprints and their table stay in cache.
    if verify && width >= PRIVATE_MIN_WIDTH && let Some(shared) = entries.scatter_shared(lim, scratch, width)? {
        // Only nodes without a private context are fingerprinted and probed.
        let found = if shared < 2 || !mark_shared_candidates(eng, scratch, width)? {
            Some(false)
        } else {
            groups::verify_fingerprint_groups(eng, entries, width, true, scratch)?
        };
        let Some(found) = found else {
            // A fingerprint group that is no twin group: start over exactly.
            return exact_twin_groups(eng, entries, width, false, scratch);
        };
        if cfg!(debug_assertions) {
            check_exact(eng, entries, width, found, scratch)?;
        }
        return Ok(found);
    }
    let found = exact_twin_groups(eng, entries, width, verify, scratch)?;
    if cfg!(debug_assertions) && verify {
        check_exact(eng, entries, width, found, scratch)?;
    }
    Ok(found)
}

/// Assert that the twin groups in `scratch`, found by private contexts or
/// verified fingerprints, are the ones exact signatures give.
fn check_exact(
    eng: &Engine,
    entries: &impl TwinEntries,
    width: usize,
    found: bool,
    scratch: &mut ContractScratch,
) -> Result<(), OperationError> {
    let verified = (scratch.group_starts.clone(), scratch.flat_groups.clone());
    let exact = exact_twin_groups(eng, entries, width, false, scratch)?;
    assert_eq!(
        (exact, &scratch.group_starts, &scratch.flat_groups),
        (found, &verified.0, &verified.1),
        "the twin groups differ from the ones exact signatures give",
    );
    Ok(())
}

/// Probe the fingerprints of the nodes `scratch.candidates` lists, which
/// [`TwinEntries::scatter_shared`] wrote, and mark every node that shares
/// one, with its representative, as [`mark_candidates`] does for every node.
/// The listed fingerprints are moved to the front of `scratch.fingerprints`
/// first, so the probe reads them in order.
fn mark_shared_candidates(
    eng: &Engine,
    scratch: &mut ContractScratch,
    width: usize,
) -> Result<bool, OperationError> {
    let lim = eng.limits();
    let ContractScratch { fingerprints, twin_hash_table, is_candidate, reps, candidates, .. } = scratch;
    lim.try_resize(is_candidate, width, false)?;
    lim.try_resize(reps, width, 0u32)?;
    is_candidate[..width].fill(false);
    // A listed node's index is at least its rank, so each fingerprint is
    // read before the move writes over it.
    for (rank, &node) in candidates.iter().enumerate() {
        fingerprints[rank] = fingerprints[node as usize];
    }
    let mut found = false;
    probe_fingerprints(lim, twin_hash_table, &fingerprints[..candidates.len()], None, |k, occupant| {
        let node = candidates[k] as usize;
        if let Some(occ) = occupant {
            let occ = candidates[occ] as usize;
            is_candidate[node] = true;
            is_candidate[occ] = true;
            reps[node] = occ as u32;
            found = true;
        } else {
            reps[node] = node as u32;
        }
        true
    })?;
    Ok(found)
}

/// The twin search over every node: the screen, additive fingerprints over
/// every entry, and, for the nodes whose fingerprints collide, an owner by
/// owner verification of the fingerprint groups where `verify` asks for it
/// ([`groups::verify_fingerprint_groups`]), else or where it refuses, full
/// signatures.
fn exact_twin_groups(
    eng: &Engine,
    entries: &impl TwinEntries,
    width: usize,
    verify: bool,
    scratch: &mut ContractScratch,
) -> Result<bool, OperationError> {
    let lim = eng.limits();
    scratch.flat_groups.clear();
    scratch.group_starts.clear();
    if entries.no_twin(lim, scratch, width)? {
        return Ok(false);
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
        *fp = fp.wrapping_add(mix64(entry));
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
    if verify && let Some(found) = groups::verify_fingerprint_groups(eng, entries, width, false, scratch)? {
        return Ok(found);
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

/// Mark twin candidates among `scratch.fingerprints[..width]`: every node whose
/// fingerprint is shared with ≥1 other node is flagged in
/// `scratch.is_candidate`. Returns whether any node was flagged; `false` means
/// all fingerprints are distinct, hence no twins.
#[inline]
fn mark_candidates(
    eng: &Engine,
    scratch: &mut ContractScratch,
    width: usize,
) -> Result<bool, OperationError> {
    let lim = eng.limits();
    let ContractScratch { fingerprints, twin_hash_table, is_candidate, reps, .. } = scratch;
    lim.try_resize(is_candidate, width, false)?;
    lim.try_resize(reps, width, 0u32)?;
    is_candidate[..width].fill(false);
    let mut found = false;
    // The table keeps the first node of each fingerprint, the lowest, and a
    // later node with it stops at that occupant: each node's representative.
    probe_fingerprints(lim, twin_hash_table, &fingerprints[..width], None, |i, occupant| {
        if let Some(occ) = occupant {
            // Idempotent stores: re-flagging an already-flagged node is a
            // redundant write, not a miscount, so the probe needs no guards.
            is_candidate[i] = true;
            is_candidate[occ] = true;
            reps[i] = occ as u32;
            found = true;
        } else {
            reps[i] = i as u32;
        }
        true
    })?;
    Ok(found)
}

/// How many fingerprints ahead the probe loop prefetches.
const PF_DIST: usize = 8;

/// Insert eligible `fingerprints` one by one into the open-addressing twin table `ht`,
/// sized and cleared here for that occupancy, probing linearly from each
/// fingerprint's home slot. For fingerprint `i` the probe calls
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
            prefetch_slot(ht.as_ptr(), (fingerprints[i + PF_DIST] as usize) & mask);
        }
        let fp = fingerprints[i];
        let mut slot = (fp as usize) & mask;
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
