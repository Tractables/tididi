//! Materializing candidate context signatures and grouping nodes by them.

use crate::engine::Engine;
use crate::limits::ApplyError;
use crate::diagram::ChildSide;
use crate::diagram::{SideView, TddLevel};

use super::super::scratch::{ContractScratch, EMPTY_SLOT, TwinSlot};
use super::{for_each_target_sibling, prefetch_slot, twin_table_size};

/// Build twin groups for child level `t1` (parent `t`) once
/// `scratch.fingerprints[..child_width]` has a known collision and
/// `scratch.is_candidate` is marked (`mark_candidates`): counts scatter,
/// offset prefix sum, signature-entry scatter, grouping by exact signature
/// equality. Writes groups into `scratch.flat_groups` / `scratch.group_starts`
/// (which the caller must have cleared) and returns whether any ≥2-member
/// group exists.
pub(super) fn build_twin_groups_after_collision(
    eng: &Engine,
    parent_level: &TddLevel,
    t1_side: ChildSide,
    t1_view: SideView,
    child_width: usize,
    scratch: &mut ContractScratch,
) -> Result<bool, ApplyError> {
    // Only candidates get a materialized signature: equal signatures force
    // equal fingerprints, so a node with a unique fingerprint has no twin, and
    // skipping it keeps count 0, an empty signature range, and a hash slot of
    // its own. A false-positive fingerprint match is separated by the exact
    // signature compare in Pass 1.
    materialize_candidate_signatures(
        eng,
        parent_level, t1_side, t1_view, child_width, scratch,
    )?;

    // ── Group nodes by signature ──────────────────────────────────────────────
    if child_width == 2 {
        return Ok(group_width_two(scratch));
    }
    group_by_hashed_signature(eng, child_width, scratch)
}

/// Scatter each twin-candidate node's context signature into the flat entry
/// arena and canonicalize the slices that arrived out of order.
fn materialize_candidate_signatures(
    eng: &Engine,
    parent_level: &TddLevel,
    t1_side: ChildSide,
    t1_view: SideView,
    child_width: usize,
    scratch: &mut ContractScratch,
) -> Result<(), ApplyError> {
    let lim = eng.limits();
    let candidate_mass =
        count_candidate_entries(eng, parent_level, t1_side, t1_view, child_width, scratch)?;

    // ── Pass 2: fill signature entries (scatter-write) ─────────────────────────
    //
    // The arena holds exactly `candidate_mass` rows, the parent-pair fan-out of
    // the candidate nodes alone. It is the largest contract allocation, so
    // `try_resize` turns a refused allocation into `Err(OverBudget)`.
    lim.try_resize(&mut scratch.entries, candidate_mass, 0u64)?;
    lim.try_resize(&mut scratch.cursors, child_width, 0u32)?;
    lim.try_resize(&mut scratch.slice_unsorted, child_width, false)?;
    let sig_offsets = &scratch.counts;
    scratch.cursors[..child_width].copy_from_slice(&sig_offsets[..child_width]);
    scratch.slice_unsorted[..child_width].fill(false);

    for_each_target_sibling(parent_level, t1_side, t1_view, |pi, target, sibling| {
        let idx = target as usize;
        if scratch.is_candidate[idx] {
            let c = scratch.cursors[idx];
            let ci = c as usize;
            let e = ((pi as u64) << 32) | sibling as u64;
            // Fused sortedness detection (see canonicalization below): flag the
            // slice if this entry compares below its predecessor. Branchless —
            // `prev` reads index c-1 saturated to 0; that value is arbitrary
            // when `c` is the slice's first position (or 0), but the `c > lo`
            // mask discards it. Entries within a slice are written at
            // consecutive cursor positions, so adjacent-at-write = adjacent-in-
            // slice and the flag is exactly `!is_sorted(slice)` once filled.
            let prev = scratch.entries[ci.saturating_sub(1)];
            let unsorted = (c > sig_offsets[idx]) & (e < prev);
            scratch.slice_unsorted[idx] |= unsorted;
            scratch.entries[ci] = e;
            scratch.cursors[idx] = c + 1; // ≤ candidate_mass < u32::MAX, checked above
        }
    });
    canonicalize_signature_slices(child_width, scratch);
    Ok(())
}


/// Count each twin-candidate node's context entries and turn the counts into a
/// prefix-sum offset table in `scratch.counts`, returning the total entry count.
///
/// Only reached in the rare twin-present case. `counts` is sized to
/// `child_width + 1` so the grouping pass can read `sig_offsets[i + 1]` as an
/// end sentinel without a fallible push. Non-candidate nodes keep count 0.
fn count_candidate_entries(
    eng: &Engine,
    parent_level: &TddLevel,
    t1_side: ChildSide,
    t1_view: SideView,
    child_width: usize,
    scratch: &mut ContractScratch,
) -> Result<usize, ApplyError> {
    let lim = eng.limits();
    lim.try_resize(&mut scratch.counts, child_width + 1, 0u32)?;
    scratch.counts[..child_width].fill(0);
    // Candidate mass bounds every value the u32 `counts` / `cursors` arrays go
    // on to hold (each count, offset and write cursor); see the check below.
    let mut candidate_mass = 0usize;
    for_each_target_sibling(parent_level, t1_side, t1_view, |_, target, _| {
        let idx = target as usize;
        if scratch.is_candidate[idx] {
            scratch.counts[idx] += 1;
            candidate_mass += 1;
        }
    });
    // ── u32 offset boundary (checked, not assumed) ─────────────────────────────
    //
    // A level's parent-pair fan-out has no structural u32 cap
    // (`MultiPairRange::start` and `len` are u64), so refuse the level through
    // `OverBudget` rather than truncate an offset. Checked before the prefix
    // sum, so no offset is stored; the counts may have wrapped, but nothing
    // reads them after the bail (the next call re-fills from zero).
    if candidate_mass >= u32::MAX as usize {
        return Err(ApplyError::OverBudget);
    }

    // ── Build offset table (prefix sum of counts) ─────────────────────────────
    //
    // Repurpose `counts` in-place as a prefix-sum offset table: after this loop,
    // sig_offsets[i] = start of node i's signature entries in the flat buffer.
    // This avoids allocating a separate Vec for offsets.
    {
        let sig_offsets = &mut scratch.counts;
        let mut running = 0u32;
        for slot in sig_offsets.iter_mut().take(child_width) {
            let c = *slot;
            *slot = running;
            running += c; // ≤ candidate_mass < u32::MAX, checked above
        }
        sig_offsets[child_width] = running; // sentinel (pre-allocated above)
    }
    Ok(candidate_mass)
}

/// Sort the signature slices the scatter flagged as out of order.
///
/// Two nodes are twins iff their context sets are equal, and pair lists are
/// unordered, so slices are compared canonicalized; a missed sort could only
/// give a spurious mismatch (a missed twin), never a wrong merge. The scatter
/// walks parent nodes in ascending index order and entries pack the parent
/// index in the high 32 bits, so a slice arrives sorted except for inversions
/// within one parent node's pair block; only the flagged slices are sorted.
fn canonicalize_signature_slices(child_width: usize, scratch: &mut ContractScratch) {
    let sig_offsets = &scratch.counts;
    for i in 0..child_width {
        if scratch.slice_unsorted[i] {
            let lo = sig_offsets[i] as usize;
            let hi = sig_offsets[i + 1] as usize;
            scratch.entries[lo..hi].sort_unstable();
        }
    }
}

/// Width-2 fast path: the two signatures are compared directly, no hashing.
fn group_width_two(scratch: &mut ContractScratch) -> bool {
    let sig_offsets = &scratch.counts;
    // Width-2 fast path: direct comparison, no hashing.
    {
        // Reaching here means `mark_candidates` found a fingerprint collision, and
        // at width 2 the only collision possible marks both nodes — so both
        // signatures are materialized and neither slice is an unmaterialized empty
        // one that would compare equal to anything.
        debug_assert!(scratch.is_candidate[0] && scratch.is_candidate[1]);
        let sig0 = &scratch.entries[sig_offsets[0] as usize..sig_offsets[1] as usize];
        let sig1 = &scratch.entries[sig_offsets[1] as usize..sig_offsets[2] as usize];
        let found = sig0 == sig1;
        if found {
            scratch.group_starts.push(0);
            scratch.flat_groups.push(0);
            scratch.flat_groups.push(1);
        }
        found
    }


}

/// General case: bucket nodes by their additive fingerprint, verify exact
/// signature equality within a bucket, then build contiguous groups.
fn group_by_hashed_signature(
    eng: &Engine,
    child_width: usize,
    scratch: &mut ContractScratch,
) -> Result<bool, ApplyError> {
    let lim = eng.limits();
    let sig_offsets = &scratch.counts;
    // General case: open-addressing hash table keyed by pre-computed additive
    // fingerprints. O(n) expected time — no sorting needed. Within each
    // bucket we verify actual signature equality (guards against hash collisions).
    //
    // Two passes: (1) map each node to its representative via hash table,
    // (2) build contiguous groups via counting sort.
    {
        // One insert at most per `0..child_width` iteration ⇒ occupancy ≤ child_width.
        let table_size = twin_table_size(child_width);
        let mask = table_size - 1;
        let ht = &mut scratch.twin_hash_table;
        lim.try_resize(ht, table_size, EMPTY_SLOT)?;
        ht[..table_size].fill(EMPTY_SLOT);

        // Pass 1: map each node to its representative via hash table.
        // cursors[i] = representative of node i (i itself if first with this signature).
        const PF_DIST: usize = 8;
        let ht_ptr = ht.as_ptr();
        for i in 0..child_width {
            // Prefetch the twin-table slot that iteration `i + PF_DIST` will first probe.
            // `fingerprints[]` is sequential so the slot address is known ahead of time;
            // the ht probe is a random access into a table that typically misses L2.
            if i + PF_DIST < child_width {
                let a = i + PF_DIST;
                prefetch_slot(ht_ptr, (scratch.fingerprints[a] as usize) & mask);
            }
            let fp = scratch.fingerprints[i];
            let mut slot = (fp as usize) & mask;
            loop {
                let s = ht[slot];
                if s.idx == u64::MAX {
                    ht[slot] = TwinSlot { fp, idx: i as u64 };
                    scratch.cursors[i] = i as u32;
                    break;
                }
                let j = s.idx as usize;
                if s.fp == fp
                    && scratch.entries[sig_offsets[j] as usize..sig_offsets[j + 1] as usize]
                        == scratch.entries[sig_offsets[i] as usize..sig_offsets[i + 1] as usize]
                {
                    scratch.cursors[i] = j as u32;
                    break;
                }
                slot = (slot + 1) & mask;
            }
        }

        // Pass 2: build contiguous groups via counting sort. O(n).
        // Repurpose fingerprints[] as per-rep member count (additive fingerprints
        // are no longer needed after hash table construction).
        scratch.fingerprints[..child_width].fill(0);
        for i in 0..child_width {
            scratch.fingerprints[scratch.cursors[i] as usize] += 1;
        }
        // Assign group offsets for reps with ≥2 members; store write cursor.
        // `pos` counts group members, so it never exceeds `child_width` (each
        // node joins at most one group) — the `as u32` narrowings below are
        // exact by the level-width invariant asserted in `find_twin_groups`.
        let mut pos = 0usize;
        for i in 0..child_width {
            if scratch.cursors[i] as usize == i && scratch.fingerprints[i] >= 2 {
                scratch.group_starts.push(pos as u32);
                let cnt = scratch.fingerprints[i] as usize;
                scratch.fingerprints[i] = pos as u64;
                pos += cnt;
            } else {
                scratch.fingerprints[i] = u64::MAX;
            }
        }
        // Scatter-write members into flat_groups.
        lim.try_resize(&mut scratch.flat_groups, pos, 0)?;
        for i in 0..child_width {
            let rep = scratch.cursors[i] as usize;
            let cursor = scratch.fingerprints[rep];
            if cursor != u64::MAX {
                scratch.flat_groups[cursor as usize] = i as u32;
                scratch.fingerprints[rep] = cursor + 1;
            }
        }
    }

    Ok(!scratch.group_starts.is_empty())
}
