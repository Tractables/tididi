use crate::engine::Engine;
use crate::diagram::ChildSide;
use crate::vtree::VtreeIdx;

use crate::limits::ApplyError;
use crate::diagram::*;

use super::scratch::{ContractScratch, EMPTY_SLOT, TwinSlot};

/// Prefetch the twin-table slot a future iteration will probe first. The
/// probe target is a random index into a table that typically misses L2;
/// `fingerprints[]` is read sequentially, so the slot for iteration i+D is
/// known D iterations ahead. No-op on non-x86_64.
#[inline(always)]
fn prefetch_slot(p: *const TwinSlot, slot: usize) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        core::arch::x86_64::_mm_prefetch(
            p.add(slot) as *const i8,
            core::arch::x86_64::_MM_HINT_T0,
        );
    }
    #[cfg(not(target_arch = "x86_64"))]
    let _ = (p, slot);
}

/// Iterate parent pairs and yield `(parent_i, target, sibling)` to `f`,
/// where `target` is the child index at level `t1` (left or right of each pair
/// depending on `t1_side`) and `sibling` is the other child.
///
/// Used to consolidate the three near-identical scatter passes in
/// `find_twin_groups` (fingerprint, counts, signature entries).
#[inline]
pub(super) fn for_each_target_sibling(
    parent_level: &TddLevel,
    t1_side: ChildSide,
    target: SideView,
    mut f: impl FnMut(u32, u32, u32),
) {
    // The caller uses `target` as an index into child-width-sized scratch
    // arrays, so it wants the cell the ref names. A side carrying an inline
    // value names no cell — it is a self-contained count, not a child node, so
    // it has no scratch slot and never participates in twin grouping. Skipping
    // it is what `cell()` returning `None` means here; the rewrite at the bottom
    // of `contract` leaves such a ref verbatim, so its contribution survives in
    // the parent pair-list multiset and is summed at the final count.
    //
    // The `sibling` value is passed on raw: it is only hashed and packed, never
    // indexed, and consistent tagging preserves signature equality and so twin
    // grouping.
    let resolve_target = |side: NodeIdx| target.child(side).index().map(|c| c as u32);
    // `pairs_of` slice iteration (compiler-vectorizable).
    for (parent_i, parent_node) in parent_level.nodes.iter().enumerate() {
        let pi = parent_i as u32;
        for pair in parent_level.pairs_of(parent_node) {
            if t1_side == ChildSide::Left {
                if let Some(t) = resolve_target(pair.left) {
                    f(pi, t, pair.right.0);
                }
            } else {
                if let Some(t) = resolve_target(pair.right) {
                    f(pi, t, pair.left.0);
                }
            }
        }
    }
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
pub(super) fn mix64(mut x: u64) -> u64 {
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
    x ^ (x >> 31)
}

/// Hash a (parent_idx, sibling_idx) context pair to a u64 fingerprint.
///
/// The wrapping sum of these hashes over a multiset of contexts is a cheap
/// pre-screen for twin detection (~1/2^64 false-positive probability) before
/// doing exact signature comparison. Using wrapping_add rather than exclusive-or means
/// duplicate contexts contribute 2h rather than cancelling; removal of a
/// contribution uses wrapping_sub. Order-independence holds because addition
/// commutes.
#[inline(always)]
pub(super) fn context_hash(parent_i: u32, sibling_j: u32) -> u64 {
    // Prelude: pack two 32-bit values; no increment.
    mix64((parent_i as u64) << 32 | sibling_j as u64)
}

#[cfg(test)]
mod tests;

/// Find groups of twin nodes at child level t1.
///
/// Two nodes at the same vtree level are **twins** if they appear in exactly
/// the same set of parent contexts — i.e., the same set of (parent_node_index,
/// sibling_node_index) pairs. This multiset of pairs is the node's "signature".
///
/// `t1` must be an explicit level: every parent ref into it is then a node
/// index that scatters one signature entry. A marginal child is declined by
/// `contract_child` before this is reached; pair fusion owns its redexes.
///
/// ## Algorithm
///
/// 1. **Count** how many signature entries each child node has (= number of
///    parent pairs referencing it). Early exit if all counts are unique: twins
///    must have equal-length signatures, so unique counts ⇒ no twins.
///
/// 2. **Fill** a flat signature buffer using a prefix-sum offset table. Each
///    entry packs (parent_idx, sibling_idx) into a single u64.
///
/// 3. **Group** nodes by signature:
///    - Width 2: direct slice comparison (O(n))
///    - Width 3+: open-addressing hash table keyed by the context fingerprints,
///      verify signature equality within each bucket (O(n) expected)
///
/// ## Output
///
/// Twin groups are written into `scratch.flat_groups` (concatenated group members)
/// and `scratch.group_starts` (start index of each group). Returns true if any
/// twin groups with ≥2 members were found.
pub(super) fn find_twin_groups(
    eng: &Engine,
    tdd: &Tdd,
    t: VtreeIdx,
    t1_side: ChildSide,
    child_width: usize,
    scratch: &mut ContractScratch,
) -> Result<bool, ApplyError> {
    let lim = eng.limits();
    scratch.flat_groups.clear();
    scratch.group_starts.clear();
    if child_width == 0 {
        return Ok(false);
    }
    // Node indices at this level are written u32-wide below (`cursors`,
    // `flat_groups`). That is the same invariant the merge side already relies
    // on (`merge_target[i] = i as u32`, `final_remap: Vec<NodeIdx>`): every
    // ref into a level is a `NodeIdx(u32)`, so a level wider than 2^32
    // slots could not be referenced at all.
    debug_assert!(
        child_width <= u32::MAX as usize,
        "level width {child_width} exceeds the u32 node-index range",
    );

    let parent_level = &tdd.levels[t.idx()];
    // How to read the parent refs that point at the contracted child (t1).
    let (left_c, right_c) = tdd.vtree.children(t);
    let t1_node = if t1_side == ChildSide::Left { left_c } else { right_c };
    let t1_view = tdd.levels[t1_node.idx()].side_view();

    // ── Pre-test: fingerprint-only scatter ────────────────────────────────────
    //
    // The common case in contract is "no twins at this level" — find_twin_groups
    // must return false cheaply. We write only the additive fingerprint (no counts)
    // during the pre-test scatter, halving the number of cache lines touched
    // per parent pair. A 64-bit fp alone has negligible birthday collision
    // probability (~N²/2^64), so dropping count mixing doesn't measurably
    // increase false positives.
    //
    // Accumulation is wrapping_add (commutative, so order-independent). Duplicate
    // (parent, sibling) pairs contribute 2h rather than cancelling (as exclusive-or would);
    // removal of a contribution uses wrapping_sub. This prevents even-multiplicity
    // duplicates — legal at marginal boundary levels after pair fusion folds — from
    // collapsing the fingerprint to 0 and creating false twin-candidate collisions.
    //
    // If a fp collision is detected, we compute counts[] in a second pass
    // before proceeding to entry fill.
    lim.try_resize(&mut scratch.fingerprints, child_width, 0u64)?;
    scratch.fingerprints[..child_width].fill(0);

    for_each_target_sibling(parent_level, t1_side, t1_view, |pi, target, sibling| {
        scratch.fingerprints[target as usize] =
            scratch.fingerprints[target as usize].wrapping_add(context_hash(pi, sibling));
    });

    // Tombstone slots got no scatter (fp == 0); make them non-colliding so they
    // are never marked twin candidates. No-op (one branch) on the dense path.
    let (lc, rc) = tdd.vtree.children(t);
    let child_t = if t1_side == ChildSide::Left { lc } else { rc };
    neutralize_tombstone_fingerprints(
        &tdd.levels[child_t.idx()],
        child_width,
        &mut scratch.fingerprints,
    );

    // ── Fingerprint collision check + candidate marking ───────────────────────
    //
    // Open-addressing hash table keyed by node index (compare via
    // `fingerprints[occ]`). As we probe, mark every node that shares its
    // fingerprint with an earlier one as a twin *candidate*, and count them. If
    // nothing collides, all fingerprints are distinct ⇒ no twins ⇒ return early
    // (the common case).
    //
    // Candidate marking costs nothing here, and must stay folded into this pass: the
    // same O(child_width) hash walk that detects a collision also identifies
    // *which* nodes are candidates, so `build_twin_groups_after_collision` can
    // skip the provably-twin-free unique-fingerprint majority in its O(M)
    // scatters with no separate candidate pre-pass.
    //
    // The marking deliberately does not early-exit on the first collision — it
    // must scan the full width to mark every candidate. That costs only the tail
    // of an O(child_width) pass that runs anyway, dwarfed by build's O(M)
    // scatters.
    if !mark_candidates(eng, scratch, child_width)? {
        return Ok(false);
    }

    build_twin_groups_after_collision(
        eng,
        parent_level,
        t1_side,
        t1_view,
        child_width,
        scratch,
    )
}

/// Size the open-addressing twin table for a pass that inserts at most
/// `max_occupancy` entries. The one sizing rule for `scratch.twin_hash_table`:
/// Both probe loops (`mark_candidates` and Pass 1 of
/// `build_twin_groups_after_collision`) call it.
///
/// # Soundness
///
/// Both loops probe linearly and never delete: the table is filled with
/// `EMPTY_SLOT` once, and every later write is an insert. An entry lands on the
/// first slot of its fingerprint's probe sequence that was empty at insertion
/// time, and no slot empties again, so a later probe for that fingerprint meets
/// every equal-fingerprint entry, in insertion order, before it reaches an
/// empty slot — whatever the table size. Both loops decide on those entries and
/// on hitting an empty slot, so the size does not change what they return.
///
/// The size must stay strictly above `max_occupancy`: on a full table a probe
/// for an absent fingerprint finds no empty slot and wraps forever. The
/// `div_ceil` term is at least 1, so the sum exceeds `max_occupancy` before the
/// round-up.
///
/// Rounding to `4/3 · max_occupancy` caps the load factor at 3/4, which bounds
/// the expected probe length by a constant.
#[inline]
fn twin_table_size(max_occupancy: usize) -> usize {
    (max_occupancy + max_occupancy.div_ceil(3))
        .next_power_of_two()
        .max(4)
}

/// Mark twin candidates among `scratch.fingerprints[..width]`: every node whose
/// additive context fingerprint is shared with ≥1 other node is flagged in
/// `scratch.is_candidate`. Returns whether any node was flagged — `false` means
/// all fingerprints are distinct, hence provably no twins. The fingerprint+index-keyed
/// open-addressing probe (`scratch.twin_hash_table`) detects the collision and
/// identifies *which* nodes are candidates in the same O(width) pass, so
/// `build_twin_groups_after_collision` can skip the unique-fingerprint majority
/// for free. The fingerprint is stored directly in the slot (co-located with the
/// occupant index) so each probe is one random load instead of two. Reads
/// `scratch.fingerprints`.
#[inline]
fn mark_candidates(
    eng: &Engine,
    scratch: &mut ContractScratch,
    width: usize,
) -> Result<bool, ApplyError> {
    let lim = eng.limits();
    lim.try_resize(&mut scratch.is_candidate, width, false)?;
    scratch.is_candidate[..width].fill(false);
    // One insert at most per `0..width` iteration ⇒ occupancy ≤ width.
    let table_size = twin_table_size(width);
    let mask = table_size - 1;
    let ht = &mut scratch.twin_hash_table;
    lim.try_resize(ht, table_size, EMPTY_SLOT)?;
    ht[..table_size].fill(EMPTY_SLOT);
    let mut found = false;
    const PF_DIST: usize = 8;
    let ht_ptr = ht.as_ptr();
    for i in 0..width {
        // Prefetch the twin-table slot that iteration `i + PF_DIST` will first probe.
        // `fingerprints[]` is sequential so the slot address is known ahead of time;
        // the ht probe is a random access into a table that typically misses L2.
        if i + PF_DIST < width {
            let a = i + PF_DIST;
            prefetch_slot(ht_ptr, (scratch.fingerprints[a] as usize) & mask);
        }
        let fp = scratch.fingerprints[i];
        let mut slot = (fp as usize) & mask;
        loop {
            let s = ht[slot];
            if s.idx == u64::MAX {
                ht[slot] = TwinSlot { fp, idx: i as u64 };
                break;
            }
            if s.fp == fp {
                // fingerprint match: both nodes are twin candidates. Idempotent
                // stores — re-flagging an already-flagged node is a redundant
                // write, not a miscount, so the probe loop needs no guards.
                let occ = s.idx as usize;
                scratch.is_candidate[i] = true;
                scratch.is_candidate[occ] = true;
                found = true;
                break;
            }
            slot = (slot + 1) & mask;
        }
    }
    Ok(found)
}

/// Neutralize tombstone slots before twin-candidate marking. A
/// tombstone is unreferenced, so the parent-pair scatter never touches its
/// fingerprint — it stays 0, and ≥2 tombstones then collide on 0, group by
/// their (identical, empty) signature, and get merged, which `merge_twin_data`
/// rejects (a tombstone is not internal). Give each tombstone slot a distinct
/// fingerprint so it can never share one with another node. Even if a sentinel
/// coincidentally equals a live node's fingerprint, `build_twin_groups_after_collision`'s
/// exact-signature check is the backstop — a tombstone's empty context never
/// equals a live node's. Gated on `n_tombstones > 0`, so the dense path pays
/// only a single branch per call, no per-node work.
#[inline]
pub(super) fn neutralize_tombstone_fingerprints(level: &TddLevel, width: usize, fingerprints: &mut [u64]) {
    if level.n_tombstones == 0 {
        return;
    }
    for (i, fp) in fingerprints.iter_mut().enumerate().take(width) {
        if level.nodes[i].is_tombstone() {
            *fp = context_hash(u32::MAX, i as u32);
        }
    }
}

mod groups;

use groups::build_twin_groups_after_collision;
