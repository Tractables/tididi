//! Materializing candidate context signatures and grouping nodes by them.

use crate::engine::Engine;
use crate::error::ApplyError;
use crate::diagram::ChildSide;
use crate::diagram::{SideView, TddLevel};

use super::super::scratch::{ContractScratch, EMPTY_SLOT, TwinSlot};
use super::{for_each_target_sibling, prefetch_slot, twin_table_size};

/// Build twin groups for child level `t1` (parent `t`) given that its context
/// fingerprints — already scattered into `scratch.fingerprints[..child_width]`
/// — have a known collision. Performs the exact work: counts scatter → offset
/// prefix sum → signature-entry scatter → grouping by exact signature equality.
/// Writes groups into `scratch.flat_groups` / `scratch.group_starts` (which the
/// caller must have cleared) and returns whether any ≥2-member group exists.
///
/// The tail of `find_twin_groups`, its only caller: it passes the fingerprint
/// state it just built (`scratch.fingerprints`, the `scratch.is_candidate`
/// marking from `mark_candidates`, and `scratch.sig_len` when `skip_empty_sig`)
/// along with the parent level and the child-side decoder, so nothing here is
/// re-derived. Both scatters below consult `is_candidate` per parent pair to
/// decide which signatures to materialize.
pub(super) fn build_twin_groups_after_collision(
    eng: &Engine,
    parent_level: &TddLevel,
    t1_side: ChildSide,
    t1_view: SideView,
    skip_empty_sig: bool,
    child_width: usize,
    scratch: &mut ContractScratch,
) -> Result<bool, ApplyError> {
    // `skip_empty_sig` (no-reexpand marginal levels): a slot referenced by zero
    // slot-refs (its count is inline in the parent pairs, or it is a
    // dead/unreferenced slot left behind because NR skips reexpand's store
    // rebuild) has an EMPTY context signature. Such a slot has no twin by
    // definition — an empty context can't match any real node's context. But its
    // accumulated fingerprint is 0, which collides with any *real* node whose
    // context hashes happen to sum to 0; both then carry empty materialized
    // signatures (the real node loses candidacy once its only fp-0 partner — the
    // dead slot — is dropped from `mark_candidates`), so the exact `[] == []`
    // compare in Pass 1 below falsely groups them. Excluding empty-signature
    // slots from grouping here (mirroring `mark_candidates`) closes that gap.
    // `sig_len` is filled by the caller immediately before this call whenever the
    // flag holds. Gated to keep every other path byte-identical.

    // ── Candidate-only signature materialization ──────────────────────────────
    //
    // A node can only be a twin of another if their full context signatures are
    // identical, which forces their additive fingerprints to be equal too. So a
    // node whose fingerprint is *unique* across the level is provably twin-free
    // and we need never materialize its signature — skipping it in the O(M)
    // counts/entries scatters below is byte-identical (it keeps count 0, an empty
    // signature range, and lands alone in an empty Pass-1 hash slot, rep = self).
    // A false-positive fingerprint match (distinct signatures, equal fingerprint)
    // still produces two candidates that the exact signature compare in Pass 1
    // separates.
    //
    // `find_twin_groups` marked `scratch.is_candidate[..]` *while detecting the
    // fingerprint collision* (the same O(child_width) hash pass), so the marking
    // is free here — there is no separate candidate pre-pass.
    //
    // The restriction is UNCONDITIONAL — no candidate-fraction gate, ever.

    materialize_candidate_signatures(
        eng,
        parent_level, t1_side, t1_view, child_width, scratch,
    )?;

    // ── Group nodes by signature ──────────────────────────────────────────────
    if child_width == 2 {
        return Ok(group_width_two(skip_empty_sig, scratch));
    }
    group_by_hashed_signature(eng, skip_empty_sig, child_width, scratch)
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
    // The arena holds exactly `candidate_mass` rows — the scattered counts sum to
    // it by construction, so it is also the prefix sum's total. That is the
    // parent-pair fan-out of the twin-candidate nodes alone, never the level's
    // whole fan-out (see the restriction above). This is still the largest
    // contract allocation and can reach GB territory on pathological CNFs, so
    // `try_resize` returns `Err(OverBudget)` if the OS allocator refuses under
    // `RLIMIT_AS`, which the caller-chain translates into v-split recovery or a
    // clean OOM exit — not a SIGABRT.
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
    // ── Compute counts[] for candidate nodes only ─────────────────────────────
    //
    // Only reached in the rare twin-present case. A second scatter pass fills
    // counts[] so we can build the signature offset prefix sum. We size to
    // `child_width + 1` to make room for the end-sentinel that the grouping
    // pass reads as `sig_offsets[i + 1]`, avoiding a fallible `.push()` later.
    // Non-candidate nodes keep count 0 (zero-length signature range).
    lim.try_resize(&mut scratch.counts, child_width + 1, 0u32)?;
    scratch.counts[..child_width].fill(0);
    // Candidate mass, accumulated as the counts are scattered. It is the exact
    // bound on every value the u32 `counts`/`cursors` arrays go on to hold (each
    // per-node count, each prefix-sum offset, each write cursor), so counting it
    // here is what makes the narrow arrays safe — see the check below.
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
    // A level's parent-pair fan-out has no structural u32 cap (`MultiPairRange::start`
    // and `len` are u64), so refuse the level rather than truncate an offset:
    // bail through the same `OverBudget` channel the `entries` allocation below
    // uses, which the caller-chain turns into v-split recovery or a clean OOM
    // exit. u32::MAX candidate rows is 32 GiB of `entries` alone, so this can
    // only fire where that allocation would fail anyway. Checked BEFORE the
    // prefix sum, hence before any offset is stored; the individual counts may
    // have wrapped on the way here, but nothing reads them after this bail (the
    // next call re-fills the array from zero).
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
/// A node's signature is the SET of (parent_idx, sibling_idx) contexts that
/// reference it, and two nodes are twins iff their sets are equal. Input-pair
/// lists are unordered, so an uncanonicalized slice comparison would be
/// sensitive to storage order and would silently miss twins whose identical
/// context sets scattered in different orders — a canonicity loss, not a count
/// error.
///
/// The scatter iterates parent nodes in ascending index order and entries pack
/// the parent index in the high 32 bits, so a slice arrives sorted except for
/// inversions within one parent node's pair block, which are rare. The scatter
/// flags exactly the unsorted slices as it writes, so only those are sorted
/// here. A flag bug could only skip a needed sort, giving a spurious mismatch
/// and a missed twin — never a wrong merge, since sorted-and-equal is
/// equivalent to multiset-equal. Only materialized slices can be flagged: the
/// skipped unique-fingerprint nodes never scatter entries.
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
fn group_width_two(skip_empty_sig: bool, scratch: &mut ContractScratch) -> bool {
    let sig_offsets = &scratch.counts;
    // Width-2 fast path: direct comparison, no hashing.
    {
        // Reaching here means `mark_candidates` found a fingerprint collision, and
        // at width 2 the only collision possible marks BOTH nodes — so both
        // signatures are materialized and neither slice is an unmaterialized empty
        // one that would compare equal to anything.
        debug_assert!(scratch.is_candidate[0] && scratch.is_candidate[1]);
        // A dead/inline slot (empty context) is never a twin (see skip_empty_sig).
        let neither_dead = !skip_empty_sig
            || (scratch.sig_len[0] != 0 && scratch.sig_len[1] != 0);
        let sig0 = &scratch.entries[sig_offsets[0] as usize..sig_offsets[1] as usize];
        let sig1 = &scratch.entries[sig_offsets[1] as usize..sig_offsets[2] as usize];
        let found = neither_dead && sig0 == sig1;
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
    skip_empty_sig: bool,
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
            // Prefetch the twin-table slot that iteration i+PF_DIST will first probe.
            // `fingerprints[]` is sequential so the slot address is known ahead of time;
            // the ht probe is a random access into a table that typically misses L2.
            if i + PF_DIST < child_width {
                let a = i + PF_DIST;
                if !(skip_empty_sig && scratch.sig_len[a] == 0) {
                    prefetch_slot(ht_ptr, (scratch.fingerprints[a] as usize) & mask);
                }
            }
            // Dead/inline slot (empty context): never a twin. Skip the hash probe
            // so it neither joins nor seeds a group. Its fingerprint is 0 (no
            // entries), which would otherwise collide in the bucket with a real
            // node whose context hashes sum to 0. See skip_empty_sig note above.
            if skip_empty_sig && scratch.sig_len[i] == 0 {
                scratch.cursors[i] = i as u32;
                continue;
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
        // `pos` counts group MEMBERS, so it never exceeds `child_width` (each
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
