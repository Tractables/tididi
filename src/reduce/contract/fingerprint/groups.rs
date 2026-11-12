//! Materializing candidate context signatures and grouping nodes by them.

use crate::Engine;
use crate::limits::{Limits, OperationError};

use super::super::scratch::ContractScratch;
use super::{TwinEntries, probe_fingerprints};

/// Build twin groups for the `child_width` nodes of one level once
/// `scratch.fingerprints[..child_width]` has a known collision and
/// `scratch.is_candidate` is marked (`mark_candidates`): counts scatter,
/// offset prefix sum, signature-entry scatter, grouping by exact signature
/// equality. Writes groups into `scratch.flat_groups` / `scratch.group_starts`
/// (which the caller must have cleared) and returns whether any ≥2-member
/// group exists.
pub(super) fn build_twin_groups_after_collision<E: TwinEntries>(
    eng: &Engine,
    entries: &E,
    child_width: usize,
    scratch: &mut ContractScratch,
) -> Result<bool, OperationError> {
    // Only candidates get a materialized signature: equal signatures force
    // equal fingerprints, so a node with a unique fingerprint has no twin, and
    // skipping it keeps count 0, an empty signature range, and its own
    // representative without a hash slot. A false-positive fingerprint match
    // is separated by the exact
    // signature compare in Pass 1.
    materialize_candidate_signatures(eng, entries, child_width, scratch)?;

    // ── Group nodes by signature ──────────────────────────────────────────────
    if child_width == 2 {
        return group_width_two(eng.limits(), scratch, E::EMPTY_TWINS);
    }
    group_by_hashed_signature(eng, child_width, scratch, E::EMPTY_TWINS)
}

/// Scatter each twin-candidate node's signature into the flat entry arena,
/// in the order the entries arrive ([`same_signature`] sorts a slice where
/// that order matters).
fn materialize_candidate_signatures(
    eng: &Engine,
    entries: &impl TwinEntries,
    child_width: usize,
    scratch: &mut ContractScratch,
) -> Result<(), OperationError> {
    let lim = eng.limits();
    let candidate_mass = count_candidate_entries(eng, entries, child_width, scratch)?;

    // ── Pass 2: fill signature entries (scatter-write) ─────────────────────────
    //
    // The arena holds exactly `candidate_mass` rows, the entries of the
    // candidate nodes alone. It is the largest contract allocation, so
    // `try_resize` turns a refused allocation into `Err(OverBudget)`.
    lim.try_resize(&mut scratch.entries, candidate_mass, 0u64)?;
    lim.try_resize(&mut scratch.cursors, child_width, 0u32)?;
    scratch.cursors[..child_width].copy_from_slice(&scratch.counts[..child_width]);

    entries.for_each(|node, e| {
        let idx = node as usize;
        if scratch.is_candidate[idx] {
            let c = scratch.cursors[idx];
            scratch.entries[c as usize] = e;
            scratch.cursors[idx] = c + 1; // ≤ candidate_mass < u32::MAX, checked above
        }
    });
    Ok(())
}

/// Replace each count in `counts` with the exclusive prefix sum of the
/// preceding counts, returning the total sum.
#[inline]
fn prefix_sum_in_place(counts: &mut [u32]) -> u32 {
    let mut sum = 0u32;
    for c in counts.iter_mut() {
        let here = *c;
        *c = sum;
        sum += here;
    }
    sum
}

/// Count each twin-candidate node's entries and turn the counts into a
/// prefix-sum offset table in `scratch.counts`, returning the total entry count.
///
/// Only reached in the rare twin-present case. `counts` is sized to
/// `child_width + 1` so the grouping pass can read `sig_offsets[i + 1]` as an
/// end sentinel without a fallible push. Non-candidate nodes keep count 0.
fn count_candidate_entries(
    eng: &Engine,
    entries: &impl TwinEntries,
    child_width: usize,
    scratch: &mut ContractScratch,
) -> Result<usize, OperationError> {
    let lim = eng.limits();
    lim.try_resize(&mut scratch.counts, child_width + 1, 0u32)?;
    scratch.counts[..child_width].fill(0);
    // Candidate mass bounds every value the u32 `counts` / `cursors` arrays go
    // on to hold (each count, offset and write cursor); see the check below.
    let mut candidate_mass = 0usize;
    entries.for_each(|node, _| {
        let idx = node as usize;
        if scratch.is_candidate[idx] {
            scratch.counts[idx] += 1;
            candidate_mass += 1;
        }
    });
    // ── u32 offset boundary (checked, not assumed) ─────────────────────────────
    //
    // A level's entry count has no structural u32 cap
    // (`PairRange::start` and `len` are u64), so refuse the level through
    // `IndexOverflow` rather than truncate an offset. Checked before the prefix
    // sum, so no offset is stored; the counts may have wrapped, but nothing
    // reads them after the bail (the next call re-fills from zero).
    if candidate_mass >= u32::MAX as usize {
        return Err(OperationError::IndexOverflow);
    }

    // ── Build offset table (prefix sum of counts) ─────────────────────────────
    //
    // Repurpose `counts` in-place as a prefix-sum offset table: after this loop,
    // sig_offsets[i] = start of node i's signature entries in the flat buffer.
    // This avoids allocating a separate Vec for offsets.
    let sig_offsets = &mut scratch.counts;
    let running = prefix_sum_in_place(&mut sig_offsets[..child_width]);
    sig_offsets[child_width] = running; // sentinel (pre-allocated above)
    Ok(candidate_mass)
}

/// Whether candidates `a` and `b` have equal signatures, as multisets.
///
/// Two nodes are twins iff their entry multisets are equal, and pair lists
/// are unordered, so slices of one length that differ as written are sorted
/// in place ([`sort_signature_slice`]) and compared again; equal as written
/// they are equal as multisets. Twins whose entries arrive in one order, and
/// the candidates no comparison reaches, are never sorted, and the scatter
/// tracks no order; a slice compared again after its sort costs a pass over
/// it. A context scatter walks parent nodes in ascending index order and
/// packs the parent index in the high 32 bits, so its slices are out of
/// order only within one parent node's pair block, and each such block is
/// sorted alone; a content slice arrives in pair-list order.
fn same_signature(
    lim: &Limits,
    entries: &mut [u64],
    sig_offsets: &[u32],
    tables: &mut SortCounts,
    a: usize,
    b: usize,
) -> Result<bool, OperationError> {
    let range = |n: usize| sig_offsets[n] as usize..sig_offsets[n + 1] as usize;
    if entries[range(a)] == entries[range(b)] {
        return Ok(true);
    }
    if range(a).len() != range(b).len() {
        return Ok(false);
    }
    for n in [a, b] {
        sort_signature_slice(lim, &mut entries[range(n)], tables)?;
    }
    Ok(entries[range(a)] == entries[range(b)])
}

/// Sort a signature slice whose entries are out of order only within runs
/// of equal high halves, as a context scatter writes them: each run that
/// holds an inversion is sorted alone ([`sort_piece`]), in one pass over the
/// slice. A slice whose high half decreases somewhere is sorted whole, and
/// so is a slice shorter than [`COUNT_MIN_LEN`], by the general sort.
/// `tables` are the counting sort's, kept for the next slice.
pub(super) fn sort_signature_slice(
    lim: &Limits,
    slice: &mut [u64],
    tables: &mut SortCounts,
) -> Result<(), OperationError> {
    if slice.len() < COUNT_MIN_LEN {
        slice.sort_unstable();
        return Ok(());
    }
    let mut k = 1;
    while k < slice.len() {
        let x = slice[k];
        if x >= slice[k - 1] {
            k += 1;
            continue;
        }
        let high = x >> 32;
        if slice[k - 1] >> 32 != high {
            return sort_piece(lim, slice, tables);
        }
        let start = slice[..k - 1].iter().rposition(|&e| e >> 32 != high).map_or(0, |p| p + 1);
        let end = slice[k + 1..].iter().position(|&e| e >> 32 != high).map_or(slice.len(), |p| k + 1 + p);
        sort_piece(lim, &mut slice[start..end], tables)?;
        k = end;
    }
    Ok(())
}

/// Entries a piece needs to be sorted by counting: a shorter one takes the
/// general sort, whose insertion sort costs less than the counting sort's
/// passes over the piece and its grid.
const COUNT_MIN_LEN: usize = 32;

/// Entries of a piece, spread over it, whose grid [`sort_by_counting`]
/// checks before its passes over the piece.
const COUNT_PROBES: usize = 8;

/// Cells per entry the grid of a piece sorted by counting may have.
const COUNT_CELLS_PER_ENTRY: u64 = 64;

/// Cells the grid of a piece sorted by counting may have in all: 4 MiB of
/// counts.
const COUNT_MAX_CELLS: u64 = 1 << 20;

/// The counting sort's tables for signature pieces, all zero between sorts.
#[derive(Default)]
pub(crate) struct SortCounts {
    /// Entries per cell of a piece's grid.
    counts: Vec<u32>,
    /// One bit per cell, set where `counts` is not zero, in bytes: entries
    /// a few cells apart set bits of different bytes, so their updates do
    /// not wait on each other.
    occupied: Vec<u8>,
}

impl crate::execution::pool::Buffers for SortCounts {
    fn buffers(&mut self, visit: &mut dyn FnMut(&mut dyn crate::execution::pool::Scratch)) {
        visit(&mut self.counts);
        visit(&mut self.occupied);
    }
}

/// Sort a piece of a signature slice: by counting where its high and low
/// halves span a grid of at most [`COUNT_CELLS_PER_ENTRY`] cells per entry
/// and [`COUNT_MAX_CELLS`] in all and it has at least [`COUNT_MIN_LEN`]
/// entries, and by the general sort otherwise.
fn sort_piece(lim: &Limits, piece: &mut [u64], tables: &mut SortCounts) -> Result<(), OperationError> {
    if piece.len() < COUNT_MIN_LEN || !sort_by_counting(lim, piece, tables)? {
        piece.sort_unstable();
    }
    Ok(())
}

/// The cells of the grid `entries` span: the range of their high halves
/// times the range of their low halves, or the range of the entries
/// themselves when they share one high half. Saturates at `u64::MAX`.
fn grid_cells(entries: impl Iterator<Item = u64>) -> u64 {
    const LOW: u64 = 0xFFFF_FFFF;
    let (mut min, mut max, mut low_min, mut low_max) = (u64::MAX, 0, LOW, 0);
    for e in entries {
        (min, max) = (min.min(e), max.max(e));
        (low_min, low_max) = (low_min.min(e & LOW), low_max.max(e & LOW));
    }
    if min > max {
        return 0;
    }
    let rows = (max >> 32) - (min >> 32) + 1;
    let cols = if rows == 1 { max - min + 1 } else { low_max - low_min + 1 };
    rows.saturating_mul(cols)
}

/// Sort `piece` by counting its entries in the grid its high and low halves
/// span, if that grid is small enough ([`sort_piece`]); whether it did.
///
/// Each entry adds one to its cell's count and sets the cell's bit; the
/// walk over the set bits writes each cell's entries in order and clears
/// its count and bit, so its time is the piece's length plus a bit per
/// cell, and the tables are zero again for the next piece.
fn sort_by_counting(lim: &Limits, piece: &mut [u64], tables: &mut SortCounts) -> Result<bool, OperationError> {
    const LOW: u64 = 0xFFFF_FFFF;
    let Some(&first) = piece.first() else {
        return Ok(true);
    };
    let most = (piece.len() as u64).saturating_mul(COUNT_CELLS_PER_ENTRY).min(COUNT_MAX_CELLS);
    // Entries spread over the piece span part of its grid: when that part
    // is already too large, the passes below would find the whole one so.
    let step = piece.len().div_ceil(COUNT_PROBES);
    let probes = piece.iter().step_by(step).chain(piece.last()).copied();
    if grid_cells(probes) > most {
        return Ok(false);
    }
    let (min, max) = piece.iter().fold((first, first), |(min, max), &e| (min.min(e), max.max(e)));
    let (high_min, rows) = (min >> 32, (max >> 32) - (min >> 32) + 1);
    if rows > most {
        return Ok(false);
    }
    // With one high half the entries order as their low halves do.
    let (low_min, low_max) = if rows == 1 {
        (min & LOW, max & LOW)
    } else {
        piece.iter().fold((LOW, 0), |(lo, hi), &e| (lo.min(e & LOW), hi.max(e & LOW)))
    };
    let cols = low_max - low_min + 1;
    let cells = match rows.checked_mul(cols) {
        Some(cells) if cells <= most => cells as usize,
        _ => return Ok(false),
    };
    let SortCounts { counts, occupied } = tables;
    lim.try_resize(counts, cells, 0)?;
    lim.try_resize(occupied, 8 * cells.div_ceil(64), 0)?;
    for &e in piece.iter() {
        let cell = (((e >> 32) - high_min) * cols + (e & LOW) - low_min) as usize;
        counts[cell] += 1;
        occupied[cell / 8] |= 1 << (cell % 8);
    }
    let mut at = 0;
    for (word, bytes) in occupied[..8 * cells.div_ceil(64)].chunks_exact_mut(8).enumerate() {
        // Byte `i`, bit `j` of the word is cell `8 * i + j` of it.
        let mut bits = u64::from_le_bytes(bytes.try_into().expect("8 bytes"));
        if bits != 0 {
            bytes.fill(0);
        }
        while bits != 0 {
            let cell = word * 64 + bits.trailing_zeros() as usize;
            bits &= bits - 1;
            let n = std::mem::take(&mut counts[cell]) as usize;
            let cell = cell as u64;
            let e = if rows == 1 { min + cell } else { ((high_min + cell / cols) << 32) | (low_min + cell % cols) };
            piece[at..at + n].fill(e);
            at += n;
        }
    }
    Ok(true)
}

/// Width-2 fast path: the two signatures are compared directly, no hashing.
/// Two empty ones are a group only with `empty_twins`.
fn group_width_two(
    lim: &Limits,
    scratch: &mut ContractScratch,
    empty_twins: bool,
) -> Result<bool, OperationError> {
    let same = same_signature(lim, &mut scratch.entries, &scratch.counts, &mut scratch.sort_counts, 0, 1)?;
    let sig_offsets = &scratch.counts;
    // Reaching here means `mark_candidates` found a fingerprint collision, and
    // at width 2 the only collision possible marks both nodes — so both
    // signatures are materialized and neither slice is an unmaterialized empty
    // one that would compare equal to anything.
    debug_assert!(scratch.is_candidate[0] && scratch.is_candidate[1]);
    let sig0 = &scratch.entries[sig_offsets[0] as usize..sig_offsets[1] as usize];
    let sig1 = &scratch.entries[sig_offsets[1] as usize..sig_offsets[2] as usize];
    let found = same && (empty_twins || !sig0.is_empty());
    if !empty_twins {
        for (node, sig) in [sig0, sig1].into_iter().enumerate() {
            if sig.is_empty() {
                lim.try_push(&mut scratch.unnamed, node as u32)?;
            }
        }
    }
    if found {
        lim.try_push(&mut scratch.group_starts, 0)?;
        lim.try_resize(&mut scratch.flat_groups, 2, 0)?;
        scratch.flat_groups[0] = 0;
        scratch.flat_groups[1] = 1;
    }
    Ok(found)
}

/// General case: bucket the candidates by their additive fingerprint, verify
/// exact signature equality within a bucket, then build contiguous groups.
///
/// A node that is no candidate has a fingerprint no other node shares, so it
/// is a group of its own and never a member: only the candidates are hashed
/// and grouped, in ascending index order, which gives the groups and their
/// order that grouping every node would. The nodes of empty signature are a
/// group only with `empty_twins`; without, the candidates among them are put
/// in `scratch.unnamed`, ascending.
fn group_by_hashed_signature(
    eng: &Engine,
    child_width: usize,
    scratch: &mut ContractScratch,
    empty_twins: bool,
) -> Result<bool, OperationError> {
    let lim = eng.limits();
    // The candidates in ascending order, and their fingerprints moved to the
    // front of `fingerprints` in that order (a candidate's index is at least
    // its rank, so the forward move reads each fingerprint before any write
    // reaches it).
    let ContractScratch {
        fingerprints,
        twin_hash_table,
        entries,
        counts,
        cursors,
        is_candidate,
        candidates,
        sort_counts,
        ..
    } = &mut *scratch;
    candidates.clear();
    let wanted = is_candidate[..child_width].iter().filter(|&&c| c).count();
    lim.reserve_exact(candidates, wanted)?;
    candidates.extend((0..child_width as u32).filter(|&i| is_candidate[i as usize]));
    let n = candidates.len();
    for (rank, &node) in candidates.iter().enumerate() {
        fingerprints[rank] = fingerprints[node as usize];
    }

    // Pass 1: map each candidate to its representative via the hash table.
    // cursors[k] = the rank of candidate k's representative (k itself if it
    // is the first with its signature).
    // A refused sort table ends each later probe at once and is returned
    // after the loop.
    let sig_offsets: &[u32] = counts;
    let mut sorted = Ok(());
    probe_fingerprints(
        lim,
        twin_hash_table,
        &fingerprints[..n],
        None,
        |k, occupant| match occupant {
            _ if sorted.is_err() => true,
            Some(j) => {
                let (a, b) = (candidates[k] as usize, candidates[j] as usize);
                match same_signature(lim, entries, sig_offsets, sort_counts, a, b) {
                    Ok(same) => {
                        if same {
                            cursors[k] = j as u32;
                        }
                        same
                    }
                    Err(e) => {
                        sorted = Err(e);
                        true
                    }
                }
            }
            None => {
                cursors[k] = k as u32;
                true
            }
        },
    )?;
    sorted?;

    // Pass 2: build contiguous groups via counting sort. O(n).
    // Repurpose fingerprints[] as per-rep member count (additive fingerprints
    // are no longer needed after hash table construction).
    let ContractScratch {
        fingerprints,
        cursors,
        candidates,
        counts,
        flat_groups,
        group_starts,
        unnamed,
        ..
    } = &mut *scratch;
    let empty = |k: usize| counts[candidates[k] as usize] == counts[candidates[k] as usize + 1];
    fingerprints[..n].fill(0);
    for &rep in &cursors[..n] {
        fingerprints[rep as usize] += 1;
    }
    // Assign group offsets for reps with ≥2 members; store write cursor.
    // `pos` counts group members, so it never exceeds `child_width` (each
    // node joins at most one group) — the `as u32` narrowings below are
    // exact by the level-width invariant asserted in `find_twin_groups`.
    let mut pos = 0usize;
    for k in 0..n {
        if cursors[k] as usize == k && fingerprints[k] >= 2 && (empty_twins || !empty(k)) {
            lim.try_push(group_starts, pos as u32)?;
            let cnt = fingerprints[k] as usize;
            fingerprints[k] = pos as u64;
            pos += cnt;
        } else {
            fingerprints[k] = u64::MAX;
        }
    }
    if !empty_twins {
        for &node in &candidates[..n] {
            if counts[node as usize] == counts[node as usize + 1] {
                lim.try_push(unnamed, node)?;
            }
        }
    }
    // Scatter-write members into flat_groups.
    lim.try_resize(flat_groups, pos, 0)?;
    for k in 0..n {
        let rep = cursors[k] as usize;
        let cursor = fingerprints[rep];
        if cursor != u64::MAX {
            flat_groups[cursor as usize] = candidates[k];
            fingerprints[rep] = cursor + 1;
        }
    }

    Ok(!group_starts.is_empty())
}

/// The fewest nodes a level needs before [`group_single_entries`] is tried:
/// below it the fingerprints are cheap, and the pass that counts the entries
/// would not repay itself.
pub(super) const SINGLE_ENTRY_MIN_WIDTH: usize = 1 << 16;

/// The bits one pass of [`group_single_entries`]' radix sort reads.
const SINGLE_ENTRY_DIGIT_BITS: u32 = 12;

/// Twin groups of a level each of whose nodes has exactly one entry, where a
/// signature is that entry: the nodes sorted by entry with a stable
/// least-significant-digit radix sort of `(entry, node)` packed into one word,
/// and each run of two or more equal entries a group. Starting from the nodes
/// in index order, a run lists its members ascending, and the groups are put
/// in the order of their lowest members, which is what the fingerprint path
/// gives. `None`, before a group is written, when some node has no entry or
/// several, or an entry and a node index do not fit one word together; the
/// caller then fingerprints.
pub(super) fn group_single_entries(
    eng: &Engine,
    entries: &impl TwinEntries,
    width: usize,
    scratch: &mut ContractScratch,
) -> Result<Option<bool>, OperationError> {
    let lim = eng.limits();
    // One entry each needs as many entries as nodes.
    let mut total = 0usize;
    entries.for_each(|_, _| total += 1);
    if total != width {
        return Ok(None);
    }
    let ContractScratch {
        fingerprints: records,
        entries: other,
        counts,
        twin_named,
        candidates,
        flat_groups,
        group_starts,
        ..
    } = &mut *scratch;
    // Each node's entry at its index. With as many entries as nodes, a node
    // named twice leaves another unnamed.
    lim.try_resize(records, width, 0u64)?;
    let words = width.div_ceil(64);
    lim.try_resize(twin_named, words, 0u64)?;
    twin_named[..words].fill(0);
    let mut repeated = false;
    entries.for_each(|node, entry| {
        let (word, bit) = ((node / 64) as usize, 1u64 << (node % 64));
        repeated |= twin_named[word] & bit != 0;
        twin_named[word] |= bit;
        records[node as usize] = entry;
    });
    if repeated {
        return Ok(None);
    }
    // The entry's halves, each cut to the bits its largest value needs, and
    // the node index below them.
    let (mut hi, mut lo) = (0u32, 0u32);
    for &entry in &records[..width] {
        hi |= (entry >> 32) as u32;
        lo |= entry as u32;
    }
    let bits = |x: u32| u32::BITS - x.leading_zeros();
    let (lo_bits, node_bits) = (bits(lo), bits(width as u32 - 1));
    let key_bits = bits(hi) + lo_bits;
    if key_bits + node_bits > u64::BITS {
        return Ok(None);
    }
    for (node, record) in records[..width].iter_mut().enumerate() {
        let key = ((*record >> 32) << lo_bits) | (*record & u64::from(u32::MAX));
        *record = (key << node_bits) | node as u64;
    }

    lim.try_resize(other, width, 0u64)?;
    lim.try_resize(counts, 1 << SINGLE_ENTRY_DIGIT_BITS, 0u32)?;
    let (mut src, mut dst): (&mut [u64], &mut [u64]) = (&mut records[..width], &mut other[..width]);
    let end = node_bits + key_bits;
    let mut shift = node_bits;
    while shift < end {
        let digit_bits = SINGLE_ENTRY_DIGIT_BITS.min(end - shift);
        let mask = (1u64 << digit_bits) - 1;
        let counts = &mut counts[..1 << digit_bits];
        counts.fill(0);
        for &record in src.iter() {
            counts[((record >> shift) & mask) as usize] += 1;
        }
        // A digit every record shares moves none of them.
        if !counts.iter().any(|&n| n as usize == width) {
            prefix_sum_in_place(counts);
            for &record in src.iter() {
                let digit = ((record >> shift) & mask) as usize;
                dst[counts[digit] as usize] = record;
                counts[digit] += 1;
            }
            std::mem::swap(&mut src, &mut dst);
        }
        shift += digit_bits;
    }
    let sorted: &[u64] = src;

    // The runs of two or more, by their lowest member.
    let node_mask = (1u64 << node_bits) - 1;
    let run_end = |start: usize| {
        let key = sorted[start] >> node_bits;
        start
            + sorted[start..]
                .iter()
                .take_while(|&&record| record >> node_bits == key)
                .count()
    };
    candidates.clear();
    let (mut start, mut members) = (0, 0);
    while start < width {
        let end = run_end(start);
        if end - start >= 2 {
            lim.try_push(candidates, start as u32)?;
            members += end - start;
        }
        start = end;
    }
    candidates.sort_unstable_by_key(|&start| sorted[start as usize] & node_mask);
    lim.reserve_exact(group_starts, candidates.len())?;
    lim.reserve_exact(flat_groups, members)?;
    for &start in candidates.iter() {
        let start = start as usize;
        group_starts.push(flat_groups.len() as u32);
        flat_groups.extend(
            sorted[start..run_end(start)]
                .iter()
                .map(|&record| (record & node_mask) as u32),
        );
    }
    Ok(Some(!group_starts.is_empty()))
}
