use crate::diagram::{ChildDecoder, ChildPair, ChildSide, EncodedChildRef, Tdd, TddLevel};
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
/// where `target` is the child index at level `t1` (left or right of each pair
/// depending on `t1_side`) and `sibling` is the other child.
#[inline]
fn for_each_target_sibling(
    parent_level: &TddLevel,
    t1_side: ChildSide,
    target: ChildDecoder,
    mut f: impl FnMut(u32, u32, u32),
) {
    // `target` indexes child-width-sized scratch arrays, so it is the cell the
    // ref names. A side carrying an inline value names no cell (it is a count,
    // not a child node), so it never joins twin grouping; the parent rewrite
    // leaves such a ref verbatim. `sibling` is passed raw: it is only hashed
    // and packed, never indexed.
    // `pairs_of` slice iteration (compiler-vectorizable).
    for (parent_i, parent_node) in parent_level.nodes.iter().enumerate() {
        let pi = parent_i as u32;
        for pair in parent_level.pairs_of(parent_node) {
            let (t, sibling) = split_pair(pair, t1_side);
            if let Some(t) = resolve_target(target, t) {
                f(pi, t, sibling);
            }
        }
    }
}

/// A pair's side toward the child level `t1_side` names, and the raw ref of
/// the other side.
#[inline]
fn split_pair(pair: &ChildPair, t1_side: ChildSide) -> (EncodedChildRef, u32) {
    if t1_side == ChildSide::Left { (pair.left, pair.right.0) } else { (pair.right, pair.left.0) }
}

/// The cell a target-side ref names, or `None` for an inline value.
#[inline]
fn resolve_target(target: ChildDecoder, side: EncodedChildRef) -> Option<u32> {
    target.child(side).index().map(|c| c as u32)
}

/// Buckets per pair in the per-node sibling bitmap of
/// [`ContextEntries::no_twin`]: two siblings share a bucket in a node of k
/// pairs with probability about k/32, and only then are the node's siblings
/// sorted.
const LOCAL_BUCKETS_PER_PAIR: usize = 16;

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
    t1_view: ChildDecoder,
    /// Whether [`no_twin`](TwinEntries::no_twin) tests the level at all.
    early_stop: bool,
}

/// A parent level with fewer entries than this, nodes and arena pairs
/// together, goes straight to the fingerprints. On the networks most
/// levels are this small, most of them repeat a sibling within a few pairs,
/// and the test's fixed cost is not repaid by the few it clears.
const EARLY_STOP_MIN_ENTRIES: usize = 128;

impl TwinEntries for ContextEntries<'_> {
    fn for_each(&self, mut f: impl FnMut(u32, u64)) {
        for_each_target_sibling(self.parent_level, self.t1_side, self.t1_view, |pi, target, sibling| {
            f(target, pack(pi, sibling));
        });
    }

    /// Two nodes share a context only where a parent node pairs both with
    /// one sibling. Each parent node is tested alone, in cache, and the test
    /// stops at the first node that repeats a sibling, where twins are
    /// likely; a level whose parent nodes never repeat one, common on grids,
    /// then needs no fingerprint. The nodes no pair names are counted on the
    /// way: two of them would be twins.
    fn no_twin(&self, lim: &Limits, scratch: &mut ContractScratch, width: usize) -> Result<bool, OperationError> {
        if !self.early_stop {
            return Ok(false);
        }
        let ContractScratch { twin_local, twin_siblings, twin_named, .. } = scratch;
        let words = width.div_ceil(64);
        lim.try_resize(twin_named, words, 0u64)?;
        let named = &mut twin_named[..words];
        named.fill(0);
        let mut name = |t: u32| named[(t / 64) as usize] |= 1 << (t % 64);
        let (level, side, view) = (self.parent_level, self.t1_side, self.t1_view);
        for node in level.nodes.iter() {
            let pairs = level.pairs_of(node);
            if pairs.len() < 3 {
                let mut last: Option<u32> = None;
                for pair in pairs {
                    let (t, s) = split_pair(pair, side);
                    if let Some(t) = resolve_target(view, t) {
                        if last == Some(s) {
                            return Ok(false);
                        }
                        last = Some(s);
                        name(t);
                    }
                }
                continue;
            }
            // 2^k buckets of a bit, at least 64.
            let k = (pairs.len() * LOCAL_BUCKETS_PER_PAIR).next_power_of_two().trailing_zeros();
            let local_words = 1usize << (k - 6);
            if twin_local.len() < local_words {
                lim.try_resize(twin_local, local_words, 0u64)?;
            }
            let local = &mut twin_local[..local_words];
            local.fill(0);
            let bucket = |s: u32| ((s as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> (64 - k)) as usize;
            let mut shared_bucket = false;
            for pair in pairs {
                let (t, s) = split_pair(pair, side);
                if let Some(t) = resolve_target(view, t) {
                    let b = bucket(s);
                    let word = &mut local[b / 64];
                    shared_bucket |= (*word >> (b % 64)) & 1 != 0;
                    *word |= 1 << (b % 64);
                    name(t);
                }
            }
            if shared_bucket {
                twin_siblings.clear();
                for pair in pairs {
                    let (t, s) = split_pair(pair, side);
                    if resolve_target(view, t).is_some() {
                        lim.try_push(twin_siblings, s)?;
                    }
                }
                twin_siblings.sort_unstable();
                if twin_siblings.windows(2).any(|w| w[0] == w[1]) {
                    return Ok(false);
                }
            }
        }
        let unnamed = width - named.iter().map(|w| w.count_ones() as usize).sum::<usize>();
        Ok(unnamed <= 1)
    }
}

/// Content twins: a node's entries are its own pairs, packed left-high.
pub(super) struct ContentEntries<'a>(pub(super) &'a TddLevel);

impl TwinEntries for ContentEntries<'_> {
    fn for_each(&self, mut f: impl FnMut(u32, u64)) {
        for (i, node) in self.0.nodes.iter().enumerate() {
            for pair in self.0.pairs_of(node) {
                f(i as u32, pack(pair.left.0, pair.right.0));
            }
        }
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
    let parent_level = &tdd.levels[parent.idx()];
    let entries = ContextEntries {
        parent_level,
        t1_side,
        t1_view: level.child_decoder(),
        early_stop: parent_level.nodes.len() + parent_level.arena_len() >= EARLY_STOP_MIN_ENTRIES,
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
    probe_fingerprints(lim, twin_hash_table, &fingerprints[..width], |i, occupant| {
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

/// Insert `fingerprints` one by one into the open-addressing twin table `ht`,
/// sized and cleared here for that occupancy, probing linearly from each
/// fingerprint's home slot. For fingerprint `i` the probe calls
/// `visit(i, Some(j))` at every occupant `j` with the same fingerprint, in
/// probe order, and stops at the first call that returns `true`; at an empty
/// slot it inserts `i` and calls `visit(i, None)`, whose result is ignored.
/// The table stores the fingerprint beside the occupant index, so each probe
/// is one random load. Both twin-grouping passes run on this loop.
///
/// # Errors
///
/// `Err(OperationError::OverBudget)` if the table cannot be resized.
#[inline]
fn probe_fingerprints(
    lim: &crate::limits::Limits,
    ht: &mut Vec<TwinSlot>,
    fingerprints: &[u64],
    mut visit: impl FnMut(usize, Option<usize>) -> bool,
) -> Result<(), OperationError> {
    let width = fingerprints.len();
    // One insert at most per fingerprint ⇒ occupancy ≤ width.
    let table_size = twin_table_size(width);
    let mask = table_size - 1;
    lim.try_resize(ht, table_size, EMPTY_SLOT)?;
    ht[..table_size].fill(EMPTY_SLOT);
    for i in 0..width {
        // Prefetch the slot that iteration `i + PF_DIST` will first probe:
        // `fingerprints` is read sequentially, so that slot's address is known
        // ahead of time, and the probe is a random access into a table that
        // typically misses L2. The pointer is re-derived from `ht` each
        // iteration because the insert below writes through `ht`; `as_ptr`
        // is a field read and the prefetch is a hint, so nothing here is a
        // real load.
        if i + PF_DIST < width {
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
