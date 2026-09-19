//! The four-way scatter join and the chunked emit that drains it.

use crate::diagram::EncodedChildRef;

use super::*;
use crate::apply::conjoin::setup::LevelShape;
use crate::diagram::Sides;

use crate::Engine;

/// The leaf arm of the scatter: one side of the join is a vtree leaf, so the
/// leaf-side product comes straight from the conjunction table and the walk
/// stays selective by iterating the non-leaf product list.
// The level's steps are kept out of line from one another. Each runs once per
// level (or, for the chunk phases, once per chunk), so the call costs nothing
// against what it then does, and holding them apart means a change inside one
// cannot re-balance the inlining, register allocation or layout of the others:
// a measurement of one step then says what it means, instead of moving a step
// the change never touched.
#[inline(never)]
fn scatter_leaf_arm<const SWAPPED: bool>(
    eng: &Engine,
    ws: &mut SparseWorkspace,
    pl: Sides<&[ProductEntry]>,
) -> Result<(), OperationError> {
    let lim = eng.limits();
    // ── Leaf arm ──
    // Iterate the non-leaf product list; the leaf-side product comes from
    // `CONJOIN_GRID`. rev_entries_c2's inner child is the leaf label here
    // (normal: a2 with left the leaf; swapped: s2 with right the leaf).
    //
    // Amortized cancellation/deadline poll — same rationale/soundness
    // as the general arm below; bail lands where `try_push` recovers.
    let mut ticker = lim.gate_with(super::super::budget::APPLY_POLL_STRIDE);
    let pl_outer = if !SWAPPED { pl.right } else { pl.left };
    for &ProductEntry { left_idx: LeftNodeIdx(outer1), right_idx: RightNodeIdx(outer2), prod_idx: ProductNodeIdx(outer_prod) } in pl_outer {
        let off_c1 = ws.rev_offsets_c1[outer1 as usize] as usize;
        let end_c1 = ws.rev_offsets_c1[outer1 as usize + 1] as usize;
        if off_c1 == end_c1 { continue; }
        let off_c2 = ws.rev_offsets_c2[outer2 as usize] as usize;
        let end_c2 = ws.rev_offsets_c2[outer2 as usize + 1] as usize;
        for &RevEntry { parent: p2, other: inner2 } in &ws.rev_entries_c2[off_c2..end_c2] {
            for &RevEntry { parent: p1, other: inner1 } in &ws.rev_entries_c1[off_c1..end_c1] {
                // normal:  outer_prod = sib_idx (right pl), grid computes a_prod.
                // swapped: outer_prod = a_prod  (left pl),  grid computes sib_idx.
                let grid_prod = CONJOIN_GRID[inner1 as usize][inner2 as usize];
                if grid_prod != NO_PRODUCT {
                    let (a_prod, sib_idx) = if !SWAPPED {
                        (grid_prod, outer_prod)
                    } else {
                        (outer_prod, grid_prod)
                    };
                    lim.try_push(&mut ws.par_buckets[p1 as usize], ParEntry {
                        p2, a_prod, sib_idx,
                    })?;
                }
            }
            ticker.poll((end_c1 - off_c1) as u64)?;
        }
    }
    Ok(())
}

/// Output-sensitive scatter: the one scatter engine — the four-way join
/// of f/g parent and child/sibling product lists. `SWAPPED = false` outer-loops
/// by right sibling s1; `SWAPPED = true` by left child a1 (every difference is a
/// pure left↔right role rename; the `if SWAPPED` branches fold at compile time).
/// A filtered per-outer g index makes the emit walk only alive `(p2, product)`
/// entries. The emitted ParEntry *set* into `par_buckets` is order-free — sound
/// because pair lists are order-independent.
///
/// Two arms behind a shared front-end (the two reverse-index builds):
///
/// **Leaf arm** (`leaf_side_is_leaf`): iterate the non-leaf product list;
/// `CONJOIN_GRID` supplies the leaf-side product. The `rev_c2` keying is the
/// same as the general arm's (normal → by right, swapped → by left), so the
/// front-end is shared.
///
/// **General arm** (both sides non-leaf), per outer key:
///   1. Build `filtered`: for each live `(inner_live, attached)` in the outer's
///      liveness bucket, walk the opposite-keyed g index and bucket its parents
///      by the join's inner-g child, attaching the live product.
///   2. Emit: for each f-parent sharing the outer, for each alive inner product,
///      push the precomputed alive `(p2, product)` entries — zero dead probes.
///   3. Clear only the `filtered` buckets touched this outer.
pub(crate) fn scatter_outsens<const SWAPPED: bool>(
    eng: &Engine,
    ws: &mut SparseWorkspace,
    left_level: &TddLevel,
    right_level: &TddLevel,
    shape: LevelShape,
    pl: Sides<&[ProductEntry]>,
    // `left_is_leaf` when `!SWAPPED`; `right_is_leaf` when `SWAPPED`.
    leaf_side_is_leaf: bool,
) -> Result<(), OperationError> {
    build_scatter_indexes::<SWAPPED>(eng, ws, left_level, right_level, shape)?;
    if leaf_side_is_leaf {
        return scatter_leaf_arm::<SWAPPED>(eng, ws, pl);
    }
    scatter_general_arm::<SWAPPED>(eng, ws, shape, pl)
}

/// Build the two reverse indexes both arms read.
///
/// f is keyed by the outer-loop dimension (normal → right sibling `s1`,
/// swapped → left child `a1`). g is keyed by the general arm's filtering axis
/// — which is also exactly the keying the leaf arm wants, since that groups g
/// by the non-leaf outer child, so one build serves both arms: normal → by
/// right `s2`, entries `(p2, a2)`; swapped → by left `a2`, entries `(p2, s2)`.
#[inline(never)]
fn build_scatter_indexes<const SWAPPED: bool>(
    eng: &Engine,
    ws: &mut SparseWorkspace,
    left_level: &TddLevel,
    right_level: &TddLevel,
    shape: LevelShape,
) -> Result<(), OperationError> {
    // Keyed by the outer dimension: the right sibling normally, the left child
    // when swapped.
    if !SWAPPED {
        build_reverse_index::<true>(eng, left_level, shape.f.right, &mut ws.rev_offsets_c1, &mut ws.rev_entries_c1)?;
        build_reverse_index::<true>(eng, right_level, shape.g.right, &mut ws.rev_offsets_c2, &mut ws.rev_entries_c2)?;
    } else {
        build_reverse_index::<false>(eng, left_level, shape.f.left, &mut ws.rev_offsets_c1, &mut ws.rev_entries_c1)?;
        build_reverse_index::<false>(eng, right_level, shape.g.left, &mut ws.rev_offsets_c2, &mut ws.rev_entries_c2)?;
    }
    Ok(())
}

/// One direction's view of the scatter workspace.
///
/// `SWAPPED` renames left↔right throughout the join. Selecting the buffers
/// once, here, leaves the join itself written a single time: `outer` is the
/// dimension the emit loop
/// iterates, `inner` the one it joins against, and `filtered` the per-outer
/// index rebuilt from the g reverse index.
struct ScatterSides<'w> {
    /// f's reverse index, keyed by the outer dimension.
    rev_offsets_c1: &'w [u32],
    rev_entries_c1: &'w [RevEntry],
    /// g's reverse index, keyed by the filtering axis.
    rev_offsets_c2: &'w [u32],
    rev_entries_c2: &'w [RevEntry],
    /// Live products of the outer child, bucketed by their f index:
    /// `[(right_idx, prod_idx)]`.
    outer_buckets: &'w mut Vec<Vec<(u32, u32)>>,
    /// Live products of the inner child, likewise.
    inner_prods: &'w mut Vec<Vec<(u32, u32)>>,
    /// Per-outer g index, keyed by the join's inner-g child.
    filtered: TouchedBuckets<'w>,
    /// The inner-g children this outer's emit reads, and the inner f children
    /// already walked to find them.
    wanted: EpochFlags<'w>,
    inner_seen: EpochFlags<'w>,
    /// Surviving candidates, bucketed by f parent.
    par_buckets: &'w mut Vec<Vec<ParEntry>>,
    /// How many outer keys the emit loop walks.
    outer_k: usize,
}

/// A bucket array cleared per outer key by replaying the indices written into
/// it, so the clear costs the touched buckets rather than the whole array.
struct TouchedBuckets<'a> {
    buckets: &'a mut Vec<Vec<(u32, u32)>>,
    touched: &'a mut Vec<u32>,
}

impl TouchedBuckets<'_> {
    fn get(&self, key: u32) -> &[(u32, u32)] {
        &self.buckets[key as usize]
    }

    fn push(&mut self, lim: &crate::limits::Limits, key: u32, v: (u32, u32)) -> Result<(), OperationError> {
        let bucket = &mut self.buckets[key as usize];
        if bucket.is_empty() {
            self.touched.push(key);
        }
        lim.try_push(bucket, v)
    }

    fn clear_touched(&mut self) {
        for &key in self.touched.iter() {
            self.buckets[key as usize].clear();
        }
        self.touched.clear();
    }
}

/// A marking array emptied by advancing a stamp rather than by clearing it,
/// so starting a round is free however many keys the last one marked. The
/// stamp lives in the workspace and only ever moves forward, so a slot left
/// by an earlier round, level or apply reads as unmarked.
struct EpochFlags<'a> {
    stamps: &'a mut Vec<u32>,
    epoch: &'a mut u32,
    cur: u32,
}

impl EpochFlags<'_> {
    /// Empty the array: every key is unmarked again.
    #[inline]
    fn begin(&mut self) {
        self.cur = self.cur.wrapping_add(1);
        if self.cur == 0 {
            // The stamp wrapped, so a slot left by an older round could read
            // as marked. This costs one pass per 2^32 rounds.
            self.stamps.fill(0);
            self.cur = 1;
        }
        *self.epoch = self.cur;
    }

    /// Mark `key`, and report whether this call is the one that marked it.
    #[inline]
    fn mark(&mut self, key: u32) -> bool {
        let slot = &mut self.stamps[key as usize];
        if *slot == self.cur {
            return false;
        }
        *slot = self.cur;
        true
    }

    #[inline]
    fn is_set(&self, key: u32) -> bool {
        self.stamps[key as usize] == self.cur
    }
}

/// Take this direction's view of the workspace, with every bucket array this
/// arm writes cleared to `shape`'s dimensions.
#[inline(never)]
fn sides<'w, const SWAPPED: bool>(
    eng: &Engine,
    ws: &'w mut SparseWorkspace,
    shape: LevelShape,
) -> Result<ScatterSides<'w>, OperationError> {
    let LevelShape { f, g, .. } = shape;
    let (inner_k, outer_k, filtered_dim) = if !SWAPPED {
        (f.left, f.right, g.left)
    } else {
        (f.right, f.left, g.right)
    };
    let SparseWorkspace {
        rev_offsets_c1, rev_entries_c1, rev_offsets_c2, rev_entries_c2,
        prod_by_a1, prod_by_s1, right_buckets, left_buckets,
        filtered, filtered_touched, par_buckets,
        wanted, wanted_epoch, inner_seen, inner_seen_epoch, ..
    } = ws;
    let (inner_prods, outer_buckets) = if !SWAPPED {
        (prod_by_a1, right_buckets)
    } else {
        (prod_by_s1, left_buckets)
    };
    ensure_buckets_cleared(eng, inner_prods, inner_k)?;
    ensure_buckets_cleared(eng, outer_buckets, outer_k)?;
    ensure_buckets_cleared(eng, filtered, filtered_dim)?;
    filtered_touched.clear();
    eng.limits().try_resize(wanted, filtered_dim, 0u32)?;
    eng.limits().try_resize(inner_seen, inner_k, 0u32)?;
    Ok(ScatterSides {
        rev_offsets_c1, rev_entries_c1, rev_offsets_c2, rev_entries_c2,
        outer_buckets, inner_prods,
        filtered: TouchedBuckets { buckets: filtered, touched: filtered_touched },
        wanted: EpochFlags { cur: *wanted_epoch, stamps: wanted, epoch: wanted_epoch },
        inner_seen: EpochFlags { cur: *inner_seen_epoch, stamps: inner_seen, epoch: inner_seen_epoch },
        par_buckets,
        outer_k,
    })
}

impl ScatterSides<'_> {
    /// Bucket both product lists: an inner product table for the non-iterated
    /// list, and the outer loop's liveness buckets.
    ///
    ///   normal:  `prod_by_a1[a1] = [(a2, a_prod)]`,   `right_buckets[s1] = [(s2, sib_idx)]`
    ///   swapped: `prod_by_s1[s1] = [(s2, sib_prod)]`, `left_buckets[a1] = [(a2, a_prod)]`
    fn bucket_products(
        &mut self,
        lim: &crate::limits::Limits,
        pl_inner: &[ProductEntry],
        pl_outer: &[ProductEntry],
    ) -> Result<(), OperationError> {
        for &ProductEntry { left_idx: LeftNodeIdx(f), right_idx: RightNodeIdx(g), prod_idx: ProductNodeIdx(product) } in pl_inner {
            lim.try_push(&mut self.inner_prods[f as usize], (g, product))?;
        }
        for &ProductEntry { left_idx: LeftNodeIdx(f), right_idx: RightNodeIdx(g), prod_idx: ProductNodeIdx(product) } in pl_outer {
            lim.try_push(&mut self.outer_buckets[f as usize], (g, product))?;
        }
        Ok(())
    }

    /// Fill `filtered` for one outer key: for each live `(inner_live, attached)`
    /// in the outer's liveness bucket, walk the opposite-keyed g index and
    /// bucket each g parent by its inner child, carrying `attached` along.
    fn build_filtered_for_outer(
        &mut self,
        lim: &crate::limits::Limits,
        outer: usize,
    ) -> Result<(), OperationError> {
        for left_idx in 0..self.outer_buckets[outer].len() {
            let (right_key, attached) = self.outer_buckets[outer][left_idx];
            let off = self.rev_offsets_c2[right_key as usize] as usize;
            let end = self.rev_offsets_c2[right_key as usize + 1] as usize;
            for &RevEntry { parent: p2, other: inner_c2 } in &self.rev_entries_c2[off..end] {
                // The emit reads only the keys `mark_wanted_for_outer` found;
                // a g parent under any other one would be bucketed, cleared
                // and never looked at.
                if !self.wanted.is_set(inner_c2) {
                    continue;
                }
                self.filtered.push(lim, inner_c2, (p2, attached))?;
            }
        }
        Ok(())
    }

    /// Mark the inner-g children this outer's emit will read: for each
    /// distinct inner f child under the outer key, the g children its live
    /// left products name.
    ///
    /// This is the join's own semi-join, one level up. What
    /// `build_filtered_for_outer` buckets is every g parent under the outer's
    /// g keys, which is unrelated to how many of them the emit then reads:
    /// where the two sides meet in few places, most of that index is written,
    /// cleared and never looked at. Marking first bounds the build by what
    /// the emit reads, and the marking walk is itself bounded by the emit's
    /// own outer loop — it visits the same products, once per distinct inner
    /// child rather than once per f parent.
    fn mark_wanted_for_outer(&mut self, outer: usize) {
        self.wanted.begin();
        self.inner_seen.begin();
        let off = self.rev_offsets_c1[outer] as usize;
        let end = self.rev_offsets_c1[outer + 1] as usize;
        for ci in off..end {
            let inner1 = self.rev_entries_c1[ci].other;
            if !self.inner_seen.mark(inner1) {
                continue;
            }
            for &(inner_c2, _) in &self.inner_prods[inner1 as usize] {
                self.wanted.mark(inner_c2);
            }
        }
    }

    /// Emit for one outer key: walk the f parents sharing it and, for each
    /// alive inner product, replay the precomputed alive `filtered` entries —
    /// so the inner loop probes no dead cell.
    fn emit_for_outer<const SWAPPED: bool>(
        &mut self,
        lim: &crate::limits::Limits,
        outer: usize,
        ticker: &mut crate::limits::PollGate,
    ) -> Result<(), OperationError> {
        let left_off = self.rev_offsets_c1[outer] as usize;
        let left_end = self.rev_offsets_c1[outer + 1] as usize;
        for ci in left_off..left_end {
            let RevEntry { parent: p1, other: inner1 } = self.rev_entries_c1[ci];
            debug_assert!(
                (inner1 as usize) < self.inner_prods.len(),
                "reverse index names inner product {inner1} of {} built",
                self.inner_prods.len()
            );
            let bucket = &mut self.par_buckets[p1 as usize];
            for &(inner_c2, inner_prod) in &self.inner_prods[inner1 as usize] {
                let fb = self.filtered.get(inner_c2);
                for &(p2, attached) in fb {
                    // The outer side carries the sibling product normally and
                    // the left-child product when swapped.
                    let (a_prod, sib_idx) = if !SWAPPED {
                        (inner_prod, attached)
                    } else {
                        (attached, inner_prod)
                    };
                    lim.try_push(bucket, ParEntry { p2, a_prod, sib_idx })?;
                }
                ticker.poll(fb.len() as u64)?;
            }
        }
        Ok(())
    }
}

/// The general arm: both sides non-leaf. Per outer key, build the filtered g
/// index, emit against it, then clear only the buckets this outer touched.
#[inline(never)]
fn scatter_general_arm<const SWAPPED: bool>(
    eng: &Engine,
    ws: &mut SparseWorkspace,
    shape: LevelShape,
    pl: Sides<&[ProductEntry]>,
) -> Result<(), OperationError> {
    let lim = eng.limits();
    let (pl_inner, pl_outer) = if !SWAPPED { (pl.left, pl.right) } else { (pl.right, pl.left) };
    let mut s = sides::<SWAPPED>(eng, ws, shape)?;
    s.bucket_products(lim, pl_inner, pl_outer)?;

    // Amortized cancellation/deadline poll. The sparse join has no other
    // mid-level break, so a wide level could otherwise wait out an expired
    // deadline; the bail lands at a loop level `try_push`'s recovery already
    // covers, so the workspace stays reusable.
    let mut ticker = lim.gate_with(super::super::budget::APPLY_POLL_STRIDE);
    for outer in 0..s.outer_k {
        if s.outer_buckets[outer].is_empty() { continue; }
        s.mark_wanted_for_outer(outer);
        s.build_filtered_for_outer(lim, outer)?;
        s.emit_for_outer::<SWAPPED>(lim, outer, &mut ticker)?;
        s.filtered.clear_touched();
    }
    Ok(())
}

/// Greedy bin-pack of f-parent indices into chunks whose projected Phase E+F
/// transient byte cost stays under `bytes_budget`. Returns boundary indices
/// `[0, p1_a, p1_b, ..., left_width]`; each chunk processes `par_buckets[boundaries[i] .. boundaries[i+1]]`.
///
/// A single p1's bucket is never split. Returns the single-chunk degenerate
/// list `[0, left_width]` when `bytes_budget` is `0` or `usize::MAX`, or when the
/// whole level fits in one chunk — in which case the call site's loop runs
/// exactly once and the path is byte-for-byte equivalent to the unchunked code.
#[inline]
pub(crate) fn plan_e_f_chunks(
    par_buckets: &[Vec<ParEntry>],
    left_width: usize,
    bytes_budget: usize,
) -> SmallVec<[u32; 8]> {
    let mut out: SmallVec<[u32; 8]> = SmallVec::new();
    out.push(0);
    if bytes_budget == 0 || bytes_budget == usize::MAX {
        out.push(left_width as u32);
        return out;
    }
    let entries_budget = bytes_budget / BYTES_PER_PAR_ENTRY;
    let mut acc = 0usize;
    for (p1, bucket) in par_buckets.iter().enumerate().take(left_width) {
        let n = bucket.len();
        if acc != 0 && acc.saturating_add(n) > entries_budget {
            out.push(p1 as u32);
            acc = 0;
        }
        acc += n;
    }
    out.push(left_width as u32);
    out
}

/// Emit output nodes for f-parents in `[p1_start..p1_end)` (Phase E + Phase F
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
    eng: &Engine,
    ws: &mut SparseWorkspace,
    level: &mut TddLevel,
    pl_output: &mut Vec<ProductEntry>,
    p1_start: usize,
    p1_end: usize,
    drop_consumed: bool,
) -> Result<(), OperationError> {
    let chunk_parent_start = pl_output.len() as u32;
    ws.emit_pairs.clear();

    flush_chunk_phase_e(eng, ws, pl_output, chunk_parent_start, p1_start, p1_end, drop_consumed)?;
    flush_chunk_phase_f(eng, ws, level, pl_output, chunk_parent_start)?;
    Ok(())
}

/// Phase E (chunk-local): dedup parent products via `p2_map`, emit `ChildPair`s
/// into `ws.emit_pairs`, and optionally drop consumed `par_buckets` rows.
///
/// Called exclusively from `flush_chunk`.
#[inline(never)]
fn flush_chunk_phase_e(
    eng: &Engine,
    ws: &mut SparseWorkspace,
    pl_output: &mut Vec<ProductEntry>,
    chunk_parent_start: u32,
    p1_start: usize,
    p1_end: usize,
    drop_consumed: bool,
) -> Result<(), OperationError> {
    let lim = eng.limits();
    // Defensive: guard against a prior call bailing mid-loop and leaving
    // stale touched entries (mirrors `scatter_outsens`'s own defensive
    // `ws.filtered_touched.clear()`).
    ws.p2_map_touched.clear();

    // Each bucket is moved out for its walk rather than borrowed in place: the
    // emit writes `ws.p2_map` and `ws.emit_pairs`, which an outstanding borrow
    // of `ws.par_buckets` conflicts with, and indexing the bucket per entry to
    // work around that re-reads its pointer and length for every candidate.
    //
    // Multi-chunk mode wants the consumed bucket's memory freed anyway, before
    // the next chunk's emit_pairs / sorted_pairs grow — chunking bounds that
    // growth — so there it simply is not handed back. Single-chunk mode hands
    // it back, because the next apply's `ensure_buckets_cleared` only
    // `.clear()`s (length=0, capacity retained) and that capacity saves the
    // next apply's scatter pushes from growing the bucket again.
    for p1 in p1_start..p1_end {
        let bucket = std::mem::take(&mut ws.par_buckets[p1]);
        if !bucket.is_empty() {
            for &entry in bucket.iter() {
                let global_idx = {
                    let slot_val = ws.p2_map[entry.p2 as usize];
                    if slot_val == NO_PRODUCT {
                        let idx = pl_output.len() as u32;
                        ws.p2_map[entry.p2 as usize] = idx;
                        ws.p2_map_touched.push(entry.p2);
                        lim.try_push(pl_output, ProductEntry {
                            left_idx: LeftNodeIdx(p1 as u32),
                            right_idx: RightNodeIdx(entry.p2),
                            prod_idx: ProductNodeIdx(idx),
                        })?;
                        idx
                    } else {
                        slot_val
                    }
                };
                let local = global_idx - chunk_parent_start;
                // Marginal children never reach the sparse path (guarded at
                // `apply_sparse_level` entry), so child refs are plain
                // structural indices — no bit-30 slot tagging here.
                let left_raw = entry.a_prod;
                let right_raw = entry.sib_idx;
                lim.try_push(&mut ws.emit_pairs, (local, ChildPair::new(EncodedChildRef::from_raw(left_raw), EncodedChildRef::from_raw(right_raw))))?;
            }

            // Lazy-clear p2_map (only entries actually written this p1, via
            // the touched list — avoids rescanning the bucket a second time).
            for ti in 0..ws.p2_map_touched.len() {
                let p2 = ws.p2_map_touched[ti];
                ws.p2_map[p2 as usize] = NO_PRODUCT;
            }
            ws.p2_map_touched.clear();
        }

        if !drop_consumed {
            ws.par_buckets[p1] = bucket;
        }
    }
    Ok(())
}

/// Phase F (chunk-local): counting-sort `ws.emit_pairs` by chunk-local parent
/// index and create output nodes in `level`.
///
/// Called exclusively from `flush_chunk`. No-ops when `ws.emit_pairs` produced
/// zero new parents for this chunk.
#[inline(never)]
fn flush_chunk_phase_f(
    eng: &Engine,
    ws: &mut SparseWorkspace,
    level: &mut TddLevel,
    pl_output: &[ProductEntry],
    chunk_parent_start: u32,
) -> Result<(), OperationError> {
    let lim = eng.limits();
    let num_new_parents = pl_output.len() - chunk_parent_start as usize;
    if num_new_parents == 0 { return Ok(()); }

    // Copied out before `ws.sorted_pairs` is mutably borrowed below.
    let duplicates_legal = ws.duplicates_legal;

    let pc = &mut ws.pair_counts;
    lim.try_resize(pc, num_new_parents + 1, 0)?;
    pc[..num_new_parents + 1].fill(0);
    for &(local_parent, _) in &ws.emit_pairs {
        debug_assert!((local_parent as usize) < num_new_parents);
        pc[local_parent as usize] += 1;
    }
    let mut total = 0u32;
    for slot in pc.iter_mut().take(num_new_parents) {
        let c = *slot;
        *slot = total;
        total += c;
    }
    pc[num_new_parents] = total;

    let n = total as usize;
    let sp = &mut ws.sorted_pairs;
    lim.try_resize(sp, n, ChildPair::new(EncodedChildRef::from_raw(0), EncodedChildRef::from_raw(0)))?;
    for &(local_parent, pair) in &ws.emit_pairs {
        let pos = pc[local_parent as usize] as usize;
        sp[pos] = pair;
        pc[local_parent as usize] += 1;
    }
    // After the fill pass each pc[i] points one-past-end of its bucket;
    // shift right so pc[i] is back at start-of-bucket (same pass-4 restore
    // as `build_reverse_index`) for the node-creation loop below.
    shift_offsets_right_by_one(&mut pc[..=num_new_parents]);

    lim.reserve(&mut level.nodes, num_new_parents)?;
    for i in 0..num_new_parents {
        let start = pc[i] as usize;
        let end = pc[i + 1] as usize;
        let pair_slice = &sp[start..end];
        // No sort and no dedup: pair lists are order-free, and canonical child
        // levels make the grid lookups injective, so a duplicate in a purely
        // Boolean diagram is an upstream canonicity violation. Once any level
        // is marginal, duplicates are legal (`ws.duplicates_legal`).
        debug_assert!(
            duplicates_legal || {
                let mut seen = std::collections::HashSet::new();
                pair_slice.iter().all(|p| seen.insert(*p))
            },
            "sparse apply: duplicate pair emitted — canonicity violated"
        );
        // Output-pair meter, charged around the sparse builder's own emit: it
        // grows `level.pairs` with a raw `try_reserve`, so the dense walk's
        // choke point never sees these. See `Limits::pairs_in_flight`.
        let pre_pairs_cap = level.pairs.capacity();
        level.try_push_internal_node(pair_slice)
            .map_err(|_| OperationError::OverBudget)?;
        lim.charge_output_pairs(level.pairs.capacity().saturating_sub(pre_pairs_cap));
    }
    Ok(())
}
