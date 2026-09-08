use crate::tdd::marg_slots::ChildSide;
use crate::vtree::VtreeIdx;

use crate::tdd::limits::ApplyError;
use crate::tdd::types::*;

use super::scratch::{ContractScratch, EMPTY_SLOT, TwinSlot};
use crate::tdd::limits::try_resize;

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
    target_is_marg: bool,
    mut f: impl FnMut(u32, u32, u32),
) {
    // The `target` side (t1) is the contracted child. When it is marginal its
    // refs are slot-tagged (bit 30); the caller uses `target` as an index into
    // child-width-sized scratch arrays, so mask to the bare slot. The `sibling`
    // value is only hashed/packed (never indexed), so its tag is left intact —
    // consistent tagging preserves signature equality and thus twin grouping.
    //
    // A marginal target side mixes inline refs (bit-30 clear) with overflow
    // slots (bit-30 set). An inline ref is a self-contained count, not a
    // child node — it has no scratch slot and never participates in twin grouping,
    // so skip it (the rewrite at the bottom of `contract` leaves inline refs
    // verbatim, so its contribution survives in the parent pair-list multiset and
    // is summed at final count). `resolve_target` returns the slot to scatter, or
    // `None` to skip.
    let resolve_target = |raw: u32| -> Option<u32> {
        if target_is_marg {
            match MargRef::from_raw(raw) {
                MargRef::Slot(s) => Some(s),
                MargRef::Inline(_) => None,
            }
        } else {
            Some(raw)
        }
    };
    // `pairs_of` slice iteration (compiler-vectorizable).
    for (parent_i, parent_node) in parent_level.nodes.iter().enumerate() {
        let pi = parent_i as u32;
        for pair in parent_level.pairs_of(parent_node) {
            if t1_side == ChildSide::Left {
                if let Some(t) = resolve_target(pair.left.0) {
                    f(pi, t, pair.right.0);
                }
            } else {
                if let Some(t) = resolve_target(pair.right.0) {
                    f(pi, t, pair.left.0);
                }
            }
        }
    }
}

/// The splitmix64 finalizer (Steele et al., 2014) — the shared bit-diffusion
/// step behind every fingerprint in the contract module.
///
/// Good avalanche properties: small input changes flip ~50% of output bits.
/// Constants and shift schedule are the original SplitMix64 ones.
///
/// Callers own their own PRELUDE (how the inputs are packed into the u64, and
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
/// doing exact signature comparison. Using wrapping_add rather than XOR means
/// duplicate contexts contribute 2h rather than cancelling; removal of a
/// contribution uses wrapping_sub. Order-independence holds because addition
/// commutes.
#[inline(always)]
pub(super) fn context_hash(parent_i: u32, sibling_j: u32) -> u64 {
    // Prelude: pack two 32-bit values; no increment.
    mix64((parent_i as u64) << 32 | sibling_j as u64)
}

#[cfg(test)]
#[path = "../fingerprint_mix64_tests.rs"]
mod mix64_tests;

/// Find groups of twin nodes at child level t1.
///
/// Two nodes at the same vtree level are **twins** if they appear in exactly
/// the same set of parent contexts — i.e., the same set of (parent_node_index,
/// sibling_node_index) pairs. This multiset of pairs is the node's "signature".
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
///    - Width 3+: open-addressing hash table keyed by XOR fingerprints,
///      verify signature equality within each bucket (O(n) expected)
///
/// ## Output
///
/// Twin groups are written into `scratch.flat_groups` (concatenated group members)
/// and `scratch.group_starts` (start index of each group). Returns true if any
/// twin groups with ≥2 members were found.
pub(super) fn find_twin_groups(
    tdd: &Tdd,
    t: VtreeIdx,
    t1_side: ChildSide,
    child_width: usize,
    scratch: &mut ContractScratch,
) -> Result<bool, ApplyError> {
    scratch.flat_groups.clear();
    scratch.group_starts.clear();
    if child_width == 0 {
        return Ok(false);
    }
    // Node indices at this level are written u32-wide below (`cursors`,
    // `flat_groups`). That is the same invariant the merge side already relies
    // on (`merge_target[i] = i as u32`, `final_remap: Vec<LocalNodeIdx>`): every
    // ref into a level is a `LocalNodeIdx(u32)`, so a level wider than 2^32
    // slots could not be referenced at all.
    debug_assert!(
        child_width <= u32::MAX as usize,
        "level width {child_width} exceeds the u32 node-index range",
    );

    let parent_level = &tdd.levels[t.idx()];
    // The contracted child (t1) determines whether `target` refs are slot-tagged.
    let (left_c, right_c) = tdd.vtree.children(t);
    let t1_node = if t1_side == ChildSide::Left { left_c } else { right_c };
    let t1_is_marg = tdd.levels[t1_node.idx()].is_marginal();

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
    // (parent, sibling) pairs contribute 2h rather than cancelling (as XOR would);
    // removal of a contribution uses wrapping_sub. This prevents even-multiplicity
    // duplicates — legal at marginal boundary levels after p-fusion folds — from
    // collapsing the fingerprint to 0 and creating false twin-candidate collisions.
    //
    // If a fp collision is detected, we compute counts[] in a second pass
    // before proceeding to entry fill.
    try_resize(&mut scratch.fingerprints, child_width, 0u64)?;
    scratch.fingerprints[..child_width].fill(0);

    // No-reexpand marginal levels mix inline refs (skipped by
    // `for_each_target_sibling`) with explicit slots. A slot referenced by zero
    // slot-refs gets no signature entry and sums to fingerprint 0; multiple such
    // nodes collide and would be falsely merged as twins. Track the per-node
    // slot-ref count so `mark_candidates` can exclude empty-signature nodes —
    // detect by entry-count == 0, NOT fingerprint == 0 (a real node can sum to
    // 0; it stays a candidate and is filtered by the exact-signature compare in
    // its bucket). Gated to keep every other path allocation-free and
    // byte-identical.
    let skip_empty_sig = t1_is_marg;
    if skip_empty_sig {
        try_resize(&mut scratch.sig_len, child_width, 0u32)?;
        scratch.sig_len[..child_width].fill(0);
        for_each_target_sibling(parent_level, t1_side, t1_is_marg, |pi, target, sibling| {
            scratch.fingerprints[target as usize] =
                scratch.fingerprints[target as usize].wrapping_add(context_hash(pi, sibling));
            scratch.sig_len[target as usize] += 1;
        });
    } else {
        for_each_target_sibling(parent_level, t1_side, t1_is_marg, |pi, target, sibling| {
            scratch.fingerprints[target as usize] =
                scratch.fingerprints[target as usize].wrapping_add(context_hash(pi, sibling));
        });
    }

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
    // Open-addressing hash table keyed by node INDEX (compare via
    // `fingerprints[occ]`). As we probe, mark every node that shares its
    // fingerprint with an earlier one as a twin *candidate*, and count them. If
    // nothing collides, all fingerprints are distinct ⇒ no twins ⇒ return early
    // (the common case).
    //
    // Candidate marking is FREE here, and must stay folded into this pass: the
    // same O(child_width) hash walk that detects a collision also identifies
    // *which* nodes are candidates, so `build_twin_groups_after_collision` can
    // skip the provably-twin-free unique-fingerprint majority in its O(M)
    // scatters with no separate candidate pre-pass. A standalone pre-pass was
    // measured net-negative.
    //
    // The marking deliberately does NOT early-exit on the first collision — it
    // must scan the full width to mark every candidate. That costs only the tail
    // of an O(child_width) pass that runs anyway, dwarfed by build's O(M)
    // scatters.
    if !mark_candidates(scratch, child_width, skip_empty_sig)? {
        return Ok(false);
    }

    build_twin_groups_after_collision(
        parent_level,
        t1_side,
        t1_is_marg,
        skip_empty_sig,
        child_width,
        scratch,
    )
}

/// Size the open-addressing twin table for a pass that inserts at most
/// `max_occupancy` entries. The ONE sizing rule for `scratch.twin_hash_table` —
/// both probe loops (`mark_candidates` and Pass 1 of
/// `build_twin_groups_after_collision`) call it.
///
/// ## Shrinking it is outcome-identical, not a heuristic
///
/// Both loops are LINEAR probing with NO deletions: the table is filled with
/// `EMPTY_SLOT` once before the loop and the only write afterwards is an insert.
/// An entry therefore lands on the first slot of its fingerprint's probe
/// sequence that is empty *at insertion time*, and no slot ever becomes empty
/// again — so for any later probe with that fingerprint, every slot the sequence
/// visits before that entry is occupied. A probe consequently meets ALL
/// equal-fingerprint entries, in insertion order, before it reaches the first
/// empty slot, whatever the table size is. Each loop decides only on the first
/// equal-fingerprint entry (`mark_candidates`) or the first equal-*signature*
/// one among them (Pass 1), plus hitting an empty slot, so both outputs are
/// table-size-invariant.
///
/// ## Occupancy bound, and why the inequality must be strict
///
/// Each loop runs `width` iterations and inserts at most one entry per iteration
/// — an iteration inserts into the empty slot it found, breaks on a match, or
/// skips — so occupancy is bounded by `width`, the value callers pass here.
/// The table must stay STRICTLY larger than that bound: on a full table a probe
/// for an absent fingerprint finds no empty slot and wraps forever, wedging the
/// compile. The `div_ceil` term is ≥ 1 for every `max_occupancy ≥ 1`, so the
/// sum is ≥ `max_occupancy + 1` before the power-of-two round-up can only raise
/// it further.
///
/// ## Headroom above the floor
///
/// The floor alone would admit a table one slot larger than its contents, where
/// linear probing's expected probe length (~`1/(1-α)²`) turns the O(width) pass
/// into O(width²). Rounding up to `4/3 · max_occupancy` instead caps the load
/// factor at 3/4 — bounded constant probe length — while still halving the table
/// against the former `2 · width` sizing for every width in `(2^k, 1.5 · 2^k]`.
/// Slots are 16 B, so at million-node levels that halving is tens of MiB off the
/// peak. `try_resize` is grow-only, so a smaller size lowers the high-water mark
/// the first time a wide level sizes the table rather than shrinking a pooled
/// one already grown.
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
/// `scratch.fingerprints`; when `skip_empty_sig` is set (no-reexpand marginal
/// level) also reads `scratch.sig_len` to drop empty-signature nodes from
/// candidacy.
#[inline]
fn mark_candidates(
    scratch: &mut ContractScratch,
    width: usize,
    skip_empty_sig: bool,
) -> Result<bool, ApplyError> {
    try_resize(&mut scratch.is_candidate, width, false)?;
    scratch.is_candidate[..width].fill(false);
    // One insert at most per `0..width` iteration ⇒ occupancy ≤ width.
    let table_size = twin_table_size(width);
    let mask = table_size - 1;
    let ht = &mut scratch.twin_hash_table;
    try_resize(ht, table_size, EMPTY_SLOT)?;
    ht[..table_size].fill(EMPTY_SLOT);
    let mut found = false;
    const PF_DIST: usize = 8;
    let ht_ptr = ht.as_ptr();
    for i in 0..width {
        // Prefetch the twin-table slot that iteration i+PF_DIST will first probe.
        // `fingerprints[]` is sequential so the slot address is known ahead of time;
        // the ht probe is a random access into a table that typically misses L2.
        if i + PF_DIST < width {
            let a = i + PF_DIST;
            if !(skip_empty_sig && scratch.sig_len[a] == 0) {
                prefetch_slot(ht_ptr, (scratch.fingerprints[a] as usize) & mask);
            }
        }
        // No-reexpand marginal levels only: a node with no slot-refs (its count
        // is inline in the parent pairs, or it is dead) must not participate in
        // twin grouping — skip inserting it so it neither becomes a candidate
        // nor collides another node into candidacy. `sig_len` is a disjoint
        // struct field from `ht`, so this read coexists with the `ht` borrow.
        if skip_empty_sig && scratch.sig_len[i] == 0 {
            continue;
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

/// Neutralize tombstone slots before twin-candidate marking (Tier 2). A
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
    for i in 0..width {
        if level.nodes[i].is_tombstone() {
            fingerprints[i] = context_hash(u32::MAX, i as u32);
        }
    }
}

mod groups;

use groups::build_twin_groups_after_collision;

#[cfg(test)]
#[path = "../fingerprint_twin_table_size_tests.rs"]
mod twin_table_size_tests;

#[cfg(test)]
#[path = "../fingerprint_signature_arena_tests.rs"]
mod signature_arena_tests;
