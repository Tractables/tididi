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
    fn for_each(&self, mut f: impl FnMut(u32, u64)) {
        for_each_target_sibling(self.parent_level, self.t1_side, |pi, target, sibling| {
            f(target, pack(pi, sibling));
        });
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
    let entries = ContextEntries {
        parent_level,
        t1_side,
        early_stop: screens(parent_level),
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
    let ContractScratch { fingerprints, twin_hash_table, is_candidate, .. } = scratch;
    lim.try_resize(is_candidate, width, false)?;
    is_candidate[..width].fill(false);
    let mut found = false;
    probe_fingerprints(lim, twin_hash_table, &fingerprints[..width], None, |i, occupant| {
        if let Some(occ) = occupant {
            // Idempotent stores: re-flagging an already-flagged node is a
            // redundant write, not a miscount, so the probe needs no guards.
            is_candidate[i] = true;
            is_candidate[occ] = true;
            found = true;
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
