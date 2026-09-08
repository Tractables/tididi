//! The four-way scatter join and the chunked emit that drains it.

use super::*;

/// Output-sensitive scatter: THE scatter engine — the four-way join
/// of c1/c2 parent and child/sibling product lists. `SWAPPED = false` outer-loops
/// by right sibling s1; `SWAPPED = true` by left child a1 (every difference is a
/// pure left↔right role rename; the `if SWAPPED` branches fold at compile time).
/// A per-outer FILTERED c2 index makes the emit walk only alive `(p2, prod)`
/// entries. The emitted ParEntry *set* into `par_buckets` is order-free — sound
/// because pair lists are order-independent.
///
/// Two arms behind a shared front-end (the two reverse-index builds):
///
/// **Leaf arm** (`leaf_side_is_leaf`): iterate the non-leaf product list,
/// `CONJOIN_GRID` computes the leaf-side product. Do NOT rewrite this arm into
/// the filtered-index form: on a 3-label alphabet the grid has at most 2/9 dead
/// entries, so output-sensitivity buys nothing there, and the
/// per-product-entry loop is already selective (c2 grouped by the
/// non-leaf child). The rev_c2 keying below is identical to what the leaf
/// arm needs (normal → by right, swapped → by left; entries carry the
/// leaf-side child = leaf label), so the front-end is shared unchanged.
///
/// **General arm** (both sides non-leaf), per outer key:
///   1. Build `filtered`: for each live `(inner_live, attached)` in the outer's
///      liveness bucket, walk the opposite-keyed c2 index and bucket its parents
///      by the join's inner-c2 child, attaching the live product.
///   2. Emit: for each c1-parent sharing the outer, for each alive inner product,
///      push the precomputed alive `(p2, prod)` entries — zero dead probes.
///   3. Clear only the `filtered` buckets touched this outer.
/// The leaf arm of the scatter: one side of the join is a vtree leaf, so the
/// leaf-side product comes straight from the conjunction table and the walk
/// stays selective by iterating the non-leaf product list.
#[inline(always)]
fn scatter_leaf_arm<const SWAPPED: bool>(
    ws: &mut SparseWorkspace,
    pl_left: &[ProductEntry],
    pl_right: &[ProductEntry],
) -> Result<(), ApplyError> {
    // ── Leaf arm ──
    // Iterate the non-leaf product list; the leaf-side product comes from
    // CONJOIN_GRID. rev_entries_c2's inner child IS the leaf label here
    // (normal: a2 with left the leaf; swapped: s2 with right the leaf).
    //
    // A3: amortized cancellation/deadline poll — same rationale/soundness
    // as the general arm below; bail lands where `try_push` recovers.
    let mut ticker = crate::tdd::limits::PollTicker::new(super::super::budget::APPLY_POLL_STRIDE);
    let pl_outer = if !SWAPPED { pl_right } else { pl_left };
    for &ProductEntry { c1_idx: C1NodeIdx(outer1), c2_idx: C2NodeIdx(outer2), prod_idx: ProdNodeIdx(outer_prod) } in pl_outer {
        let off_c1 = ws.rev_offsets_c1[outer1 as usize] as usize;
        let end_c1 = ws.rev_offsets_c1[outer1 as usize + 1] as usize;
        if off_c1 == end_c1 { continue; }
        let off_c2 = ws.rev_offsets_c2[outer2 as usize] as usize;
        let end_c2 = ws.rev_offsets_c2[outer2 as usize + 1] as usize;
        for &(p2, inner2) in &ws.rev_entries_c2[off_c2..end_c2] {
            for &(p1, inner1) in &ws.rev_entries_c1[off_c1..end_c1] {
                // normal:  outer_prod = sib_idx (right pl), grid computes a_prod.
                // swapped: outer_prod = a_prod  (left pl),  grid computes sib_idx.
                let grid_prod = CONJOIN_GRID[inner1 as usize][inner2 as usize];
                if grid_prod != DEAD {
                    let (a_prod, sib_idx) = if !SWAPPED {
                        (grid_prod, outer_prod)
                    } else {
                        (outer_prod, grid_prod)
                    };
                    try_push(&mut ws.par_buckets[p1 as usize], ParEntry {
                        p2, a_prod, sib_idx,
                    })?;
                }
            }
            ticker.tick_by((end_c1 - off_c1) as u64)?;
        }
    }
    Ok(())
}

pub(crate) fn scatter_outsens<const SWAPPED: bool>(
    ws: &mut SparseWorkspace,
    c1_level: &TddLevel,
    c2_level: &TddLevel,
    k1_left: usize, k2_left: usize,
    k1_right: usize, k2_right: usize,
    pl_left: &[ProductEntry],
    pl_right: &[ProductEntry],
    // `left_is_leaf` when `!SWAPPED`; `right_is_leaf` when `SWAPPED`.
    leaf_side_is_leaf: bool,
) -> Result<(), ApplyError> {
    // c1 reverse index keyed by the outer-loop dimension:
    //   normal → by right sibling s1; swapped → by left child a1.
    if !SWAPPED {
        build_reverse_index::<true>(c1_level, k1_right, &mut ws.rev_offsets_c1, &mut ws.rev_entries_c1)?;
    } else {
        build_reverse_index::<false>(c1_level, k1_left, &mut ws.rev_offsets_c1, &mut ws.rev_entries_c1)?;
    }
    // c2 reverse index keyed by the FILTER-INNER dimension (the general arm's
    // filtering axis — and exactly the keying the leaf arm needs, which groups
    // c2 by the non-leaf outer child for selectivity; one build serves both arms):
    //   normal → by right s2 → entries (p2, a2); swapped → by left a2 → entries (p2, s2).
    if !SWAPPED {
        build_reverse_index::<true>(c2_level, k2_right, &mut ws.rev_offsets_c2, &mut ws.rev_entries_c2)?;
    } else {
        build_reverse_index::<false>(c2_level, k2_left, &mut ws.rev_offsets_c2, &mut ws.rev_entries_c2)?;
    }

    if leaf_side_is_leaf {
        return scatter_leaf_arm::<SWAPPED>(ws, pl_left, pl_right);
    }

    // ── General arm (both sides non-leaf) ──
    // Inner product table for the non-iterated product list:
    //   normal: prod_by_a1[a1] = [(a2, a_prod)] from pl_left
    //   swapped: prod_by_s1[s1] = [(s2, sib_prod)] from pl_right
    if !SWAPPED {
        ensure_buckets_cleared(&mut ws.prod_by_a1, k1_left)?;
        for &ProductEntry { c1_idx: C1NodeIdx(a1), c2_idx: C2NodeIdx(a2), prod_idx: ProdNodeIdx(a_prod) } in pl_left {
            try_push(&mut ws.prod_by_a1[a1 as usize], (a2, a_prod))?;
        }
    } else {
        ensure_buckets_cleared(&mut ws.prod_by_s1, k1_right)?;
        for &ProductEntry { c1_idx: C1NodeIdx(s1), c2_idx: C2NodeIdx(s2), prod_idx: ProdNodeIdx(sib_prod) } in pl_right {
            try_push(&mut ws.prod_by_s1[s1 as usize], (s2, sib_prod))?;
        }
    }
    // Outer-loop liveness buckets:
    //   normal: right_buckets[r1] = [(r2=s2, sib_idx)] from pl_right
    //   swapped: left_buckets[a1] = [(a2, a_prod)] from pl_left
    if !SWAPPED {
        ensure_buckets_cleared(&mut ws.right_buckets, k1_right)?;
        for &ProductEntry { c1_idx: C1NodeIdx(r1), c2_idx: C2NodeIdx(r2), prod_idx: ProdNodeIdx(sib_idx) } in pl_right {
            try_push(&mut ws.right_buckets[r1 as usize], (r2, sib_idx))?;
        }
    } else {
        ensure_buckets_cleared(&mut ws.left_buckets, k1_left)?;
        for &ProductEntry { c1_idx: C1NodeIdx(a1), c2_idx: C2NodeIdx(a2), prod_idx: ProdNodeIdx(a_prod) } in pl_left {
            try_push(&mut ws.left_buckets[a1 as usize], (a2, a_prod))?;
        }
    }

    // Per-outer filtered index: keyed by the join's inner-c2 child.
    //   normal: filtered[a2] = [(p2, sib_idx)]  (sized k2_left)
    //   swapped: filtered[s2] = [(p2, a_prod)]  (sized k2_right)
    let filtered_dim = if !SWAPPED { k2_left } else { k2_right };
    ensure_buckets_cleared(&mut ws.filtered, filtered_dim)?;
    ws.filtered_touched.clear();

    // A3: amortized cancellation/deadline poll. The sparse join had no mid-level
    // break, so a wide level could wait out a cancelled race lane / expired
    // deadline / due preempt slice. Accumulate emitted-candidate work and poll
    // every ~1M units (see `budget::PollTicker`); the bail lands at a loop level
    // already covered by `try_push`'s recovery, so the workspace stays reusable.
    let mut ticker = crate::tdd::limits::PollTicker::new(super::super::budget::APPLY_POLL_STRIDE);
    let outer_k1 = if !SWAPPED { k1_right } else { k1_left };
    for outer in 0..outer_k1 {
        let outer_empty = if !SWAPPED {
            ws.right_buckets[outer].is_empty()
        } else {
            ws.left_buckets[outer].is_empty()
        };
        if outer_empty { continue; }

        // ── Build the per-outer filtered c2 index ──
        // For each live (inner_live, attached), walk the opposite-keyed c2 index
        // and bucket each c2 parent by its inner child, carrying `attached`.
        let live_len = if !SWAPPED { ws.right_buckets[outer].len() } else { ws.left_buckets[outer].len() };
        for li in 0..live_len {
            let (c2_key, attached) = if !SWAPPED {
                ws.right_buckets[outer][li]   // (r2=s2, sib_idx)
            } else {
                ws.left_buckets[outer][li]    // (a2, a_prod)
            };
            let off = ws.rev_offsets_c2[c2_key as usize] as usize;
            let end = ws.rev_offsets_c2[c2_key as usize + 1] as usize;
            for ei in off..end {
                let (p2, inner_c2) = ws.rev_entries_c2[ei]; // normal: (p2, a2); swapped: (p2, s2)
                let bucket = &mut ws.filtered[inner_c2 as usize];
                if bucket.is_empty() {
                    ws.filtered_touched.push(inner_c2);
                }
                // re-borrow after the touched push (push borrows a different field)
                try_push(&mut ws.filtered[inner_c2 as usize], (p2, attached))?;
            }
        }

        // ── Emit: walk c1-parents sharing this outer; for each alive inner
        //    product, replay the precomputed alive filtered entries ──
        let c1_off = ws.rev_offsets_c1[outer] as usize;
        let c1_end = ws.rev_offsets_c1[outer + 1] as usize;
        for ci in c1_off..c1_end {
            let (p1, inner1) = ws.rev_entries_c1[ci];
            // Defensive bounds check.
            if !SWAPPED {
                if (inner1 as usize) >= ws.prod_by_a1.len() { return Err(ApplyError::OverBudget); }
            } else {
                if (inner1 as usize) >= ws.prod_by_s1.len() { return Err(ApplyError::OverBudget); }
            }
            let bucket = &mut ws.par_buckets[p1 as usize];
            if !SWAPPED {
                for &(a2, a_prod) in &ws.prod_by_a1[inner1 as usize] {
                    let fb = &ws.filtered[a2 as usize];
                    for &(p2, sib_idx) in fb {
                        try_push(bucket, ParEntry { p2, a_prod, sib_idx })?;
                    }
                    ticker.tick_by(fb.len() as u64)?;
                }
            } else {
                for &(s2, sib_prod) in &ws.prod_by_s1[inner1 as usize] {
                    let fb = &ws.filtered[s2 as usize];
                    for &(p2, a_prod) in fb {
                        try_push(bucket, ParEntry { p2, a_prod, sib_idx: sib_prod })?;
                    }
                    ticker.tick_by(fb.len() as u64)?;
                }
            }
        }

        // ── Clear only the filtered buckets touched this outer ──
        for ti in 0..ws.filtered_touched.len() {
            let idx = ws.filtered_touched[ti] as usize;
            ws.filtered[idx].clear();
        }
        ws.filtered_touched.clear();
    }
    Ok(())
}

/// Greedy bin-pack of c1-parent indices into chunks whose projected Phase E+F
/// transient byte cost stays under `bytes_budget`. Returns boundary indices
/// `[0, p1_a, p1_b, ..., k1]`; each chunk processes `par_buckets[boundaries[i] .. boundaries[i+1]]`.
///
/// A single p1's bucket is never split. Returns the single-chunk degenerate
/// list `[0, k1]` when `bytes_budget` is `0` or `usize::MAX`, or when the
/// whole level fits in one chunk — in which case the call site's loop runs
/// exactly once and the path is byte-for-byte equivalent to the unchunked code.
#[inline]
pub(crate) fn plan_e_f_chunks(
    par_buckets: &[Vec<ParEntry>],
    k1: usize,
    bytes_budget: usize,
) -> SmallVec<[u32; 8]> {
    let mut out: SmallVec<[u32; 8]> = SmallVec::new();
    out.push(0);
    if bytes_budget == 0 || bytes_budget == usize::MAX {
        out.push(k1 as u32);
        return out;
    }
    let entries_budget = bytes_budget / BYTES_PER_PAR_ENTRY;
    let mut acc = 0usize;
    for p1 in 0..k1 {
        let n = par_buckets[p1].len();
        if acc != 0 && acc.saturating_add(n) > entries_budget {
            out.push(p1 as u32);
            acc = 0;
        }
        acc += n;
    }
    out.push(k1 as u32);
    out
}

/// Emit output nodes for c1-parents in `[p1_start..p1_end)` (Phase E + Phase F
/// applied to one chunk). Reuses `ws.emit_pairs`, `ws.pair_counts`,
/// `ws.sorted_pairs` from scratch (cleared/resized at chunk entry) so peak
/// transient bytes stay bounded by the chunk size. After emit, drops the
/// inner allocations of `ws.par_buckets[p1_start..p1_end]` so the next
/// chunk's `emit_pairs` grows in already-released address space.
///
/// Pre: `ws.par_buckets[p1_start..p1_end]` is populated by Phase C/D.
/// `pl_output.len()` at entry == number of parents emitted by previous chunks
/// (so `prod_idx` stays sequential globally).
///
/// `emit_pairs` stores the *chunk-local* parent index (`prod_idx - chunk_parent_start`),
/// allowing `pair_counts` to be sized to `num_new_parents` rather than the
/// running total.
#[inline]
pub(crate) fn flush_chunk(
    ws: &mut SparseWorkspace,
    level: &mut TddLevel,
    pl_output: &mut Vec<ProductEntry>,
    p1_start: usize,
    p1_end: usize,
    drop_consumed: bool,
) -> Result<(), ApplyError> {
    let chunk_parent_start = pl_output.len() as u32;
    ws.emit_pairs.clear();

    flush_chunk_phase_e(ws, pl_output, chunk_parent_start, p1_start, p1_end, drop_consumed)?;
    flush_chunk_phase_f(ws, level, pl_output, chunk_parent_start)?;
    Ok(())
}

/// Phase E (chunk-local): dedup parent products via `p2_map`, emit `InputPair`s
/// into `ws.emit_pairs`, and optionally drop consumed `par_buckets` rows.
///
/// Called exclusively from `flush_chunk`.
#[inline(always)]
pub(crate) fn flush_chunk_phase_e(
    ws: &mut SparseWorkspace,
    pl_output: &mut Vec<ProductEntry>,
    chunk_parent_start: u32,
    p1_start: usize,
    p1_end: usize,
    drop_consumed: bool,
) -> Result<(), ApplyError> {
    // Defensive: guard against a prior call bailing mid-loop and leaving
    // stale touched entries (mirrors `scatter_outsens`'s own defensive
    // `ws.filtered_touched.clear()`).
    ws.p2_map_touched.clear();

    // Index-based iteration so the mutable accesses to `ws.p2_map` and
    // `ws.emit_pairs` inside the loop don't conflict with the immutable
    // borrow of `ws.par_buckets[p1]`.
    for p1 in p1_start..p1_end {
        let bucket_len = ws.par_buckets[p1].len();
        if bucket_len == 0 { continue; }

        for ei in 0..bucket_len {
            let entry = ws.par_buckets[p1][ei];
            let global_idx = {
                let slot_val = ws.p2_map[entry.p2 as usize];
                if slot_val == DEAD {
                    let idx = pl_output.len() as u32;
                    ws.p2_map[entry.p2 as usize] = idx;
                    ws.p2_map_touched.push(entry.p2);
                    try_push(pl_output, ProductEntry {
                        c1_idx: C1NodeIdx(p1 as u32),
                        c2_idx: C2NodeIdx(entry.p2),
                        prod_idx: ProdNodeIdx(idx),
                    })?;
                    idx
                } else {
                    slot_val
                }
            };
            let local = global_idx - chunk_parent_start;
            // Marginal children never reach the sparse path (guarded at
            // apply_sparse_level entry), so child refs are plain structural
            // indices — no bit-30 slot tagging here.
            let left_raw = entry.a_prod;
            let right_raw = entry.sib_idx;
            try_push(&mut ws.emit_pairs, (local, InputPair {
                left: LocalNodeIdx(left_raw),
                right: LocalNodeIdx(right_raw),
            }))?;
        }

        // Lazy-clear p2_map (only entries actually written this p1, via the
        // touched list — avoids rescanning `par_buckets[p1]` a second time).
        for ti in 0..ws.p2_map_touched.len() {
            let p2 = ws.p2_map_touched[ti];
            ws.p2_map[p2 as usize] = DEAD;
        }
        ws.p2_map_touched.clear();
    }

    // Multi-chunk mode: drop consumed par_buckets allocations (replace with
    // Vec::new()) so the backing memory is freed before the next chunk's
    // emit_pairs / sorted_pairs grow — this is the whole point of chunking.
    //
    // Single-chunk mode: leave buckets alone. The next apply's
    // ensure_buckets_cleared will `.clear()` (length=0, retain capacity),
    // preserving the cross-apply capacity reuse that the old non-chunked code
    // relied on for cheap scatter pushes.
    if drop_consumed {
        for p1 in p1_start..p1_end {
            ws.par_buckets[p1] = Vec::new();
        }
    }
    Ok(())
}

/// Phase F (chunk-local): counting-sort `ws.emit_pairs` by chunk-local parent
/// index and create output nodes in `level`.
///
/// Called exclusively from `flush_chunk`. No-ops when `ws.emit_pairs` produced
/// zero new parents for this chunk.
#[inline(always)]
pub(crate) fn flush_chunk_phase_f(
    ws: &mut SparseWorkspace,
    level: &mut TddLevel,
    pl_output: &[ProductEntry],
    chunk_parent_start: u32,
) -> Result<(), ApplyError> {
    let num_new_parents = pl_output.len() - chunk_parent_start as usize;
    if num_new_parents == 0 { return Ok(()); }

    // Copied out before `ws.sorted_pairs` is mutably borrowed below.
    let dups_legal = ws.dups_legal;

    let pc = &mut ws.pair_counts;
    try_resize(pc, num_new_parents + 1, 0)?;
    pc[..num_new_parents + 1].fill(0);
    for &(local_parent, _) in &ws.emit_pairs {
        debug_assert!((local_parent as usize) < num_new_parents);
        pc[local_parent as usize] += 1;
    }
    let mut total = 0u32;
    for i in 0..num_new_parents {
        let c = pc[i];
        pc[i] = total;
        total += c;
    }
    pc[num_new_parents] = total;

    let n = total as usize;
    let sp = &mut ws.sorted_pairs;
    try_resize(sp, n, InputPair { left: LocalNodeIdx(0), right: LocalNodeIdx(0) })?;
    for &(local_parent, pair) in &ws.emit_pairs {
        let pos = pc[local_parent as usize] as usize;
        sp[pos] = pair;
        pc[local_parent as usize] += 1;
    }
    // After the fill pass each pc[i] points one-past-end of its bucket;
    // shift right so pc[i] is back at start-of-bucket (same pass-4 restore
    // as build_reverse_index) for the node-creation loop below.
    shift_offsets_right_by_one(&mut pc[..=num_new_parents]);

    budget_reserve(&mut level.nodes, num_new_parents)?;
    for i in 0..num_new_parents {
        let start = pc[i] as usize;
        let end = pc[i + 1] as usize;
        let pair_slice = &sp[start..end];
        // No sort, no dedup. Pair lists are unordered sets and twin contraction
        // is order-independent, so the emit
        // order is free — no canonicalizing sort is required.
        //
        // The scatter cannot produce duplicate pairs when child levels are
        // canonical (no duplicate nodes ⇒ grid lookups are injective; the
        // pair-level corollary of the no-compress proof).
        // A defensive dedup here would be dead weight — one measured over the
        // whole suite removed zero pairs. In a purely Boolean diagram the pair
        // list is never a legitimate multiset, so a duplicate signals an
        // upstream canonicity violation to fix at the source. Once *any* level
        // is marginal, duplicates are legal (`ws.dups_legal` — pair lists are
        // then multisets feeding a sum) and are
        // inherited from an operand parent whose own list holds the pair twice.
        debug_assert!(
            dups_legal || {
                let mut seen = std::collections::HashSet::new();
                pair_slice.iter().all(|p| seen.insert(*p))
            },
            "sparse apply: duplicate pair emitted — canonicity violated"
        );
        // Output-pair meter, charged around the sparse builder's own emit: it
        // grows `level.pairs` with a raw `try_reserve`, so the dense walk's
        // choke point never sees these. See `ApplyLimits::pairs_in_flight`.
        let pre_pairs_cap = level.pairs.capacity();
        level.try_push_internal_node(pair_slice)
            .map_err(|_| ApplyError::OverBudget)?;
        super::super::budget::account_output_pairs(level.pairs.capacity().saturating_sub(pre_pairs_cap));
    }
    Ok(())
}
