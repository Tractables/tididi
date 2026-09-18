//! Rebuild the two diagram levels affected by a vtree rotation.
//!
//! The outer level retains its node count and ordering, so references from
//! higher levels remain valid. Other levels keep their original pairs.
//! Rebuilding expands each outer pair into triples and regroups them:
//!
//! - Left: `(A, (B, C))` becomes `((A, B), C)`. Group triples by `C` and
//!   intern the resulting sets of `(A, B)` pairs at the new inner level.
//! - Right: `((A, B), C)` becomes `(A, (B, C))`. Group by `A` and intern
//!   the sets of `(B, C)` pairs instead.
//!
//! ## Marginal context
//!
//! Without marginal levels, equal cells share an inner node and duplicate
//! outer pairs can be removed. With marginal levels, identical products may
//! represent different assignment families whose values must add. The rewrite
//! then retains one inner node per distinct inner pair and preserves both
//! cell and outer-pair multiplicities. Regrouping preserves their value sum.

use crate::diagram::{ChildDecoder, ChildPair, EncodedChildRef, NodeIdx, Tdd, TddLevel};

use rustc_hash::{FxHashMap, FxHashSet};

use crate::vtree::rotate::RotationInfo;
use crate::vtree::RotationKind;

// The `marginal_ctx` branches below expand fully instead of sharing and
// deduping; the argument is the module doc's "Marginal context" section.

pub(crate) use super::scratch::RestructureScratch;
use super::scratch::SCRATCH_RETAIN_ENTRIES;
use crate::limits::pool::release_or_clear;
use crate::limits::{Limits, OperationError};

/// Pack a search triple `(inner, src, axis)` into one `u128` whose numeric order
/// is exactly the tuple's derived lexicographic order `(inner.left,
/// inner.right, src, axis)` — all four fields are `u32`, so the packing is
/// lossless and order-preserving. The high 64 bits are the `inner` pair (its own
/// sort key); the low 64 bits are the `(src, axis)` "cell". Sorting by this key
/// therefore groups cells by `inner` and orders cells within a group by
/// `(src, axis)` — exactly what the group scan below relies on, but as a single
/// `u128` compare instead of a four-field branchy tuple compare.
#[inline]
fn pack_triple(inner: ChildPair, src: u32, axis: EncodedChildRef) -> u128 {
    ((inner.left.0 as u128) << 96)
        | ((inner.right.0 as u128) << 64)
        | ((src as u128) << 32)
        | (axis.0 as u128)
}
#[inline]
fn tri_inner_key(p: u128) -> u64 { (p >> 64) as u64 }
#[inline]
fn tri_cell(p: u128) -> u64 { p as u64 } // (src << 32) | axis — the fp/dedup key
#[inline]
fn tri_inner(p: u128) -> ChildPair {
    ChildPair::new(EncodedChildRef::from_raw((p >> 96) as u32), EncodedChildRef::from_raw((p >> 64) as u32))
}
#[inline]
fn tri_src(p: u128) -> u32 { (p >> 32) as u32 }
#[inline]
fn tri_axis(p: u128) -> EncodedChildRef { EncodedChildRef::from_raw(p as u32) }

/// Rebuild the two levels of a rotation in `dir` and return the levels the
/// rotation replaced, or `None` if the probe was abandoned.
///
/// Cells are grouped by sorting the inner pairs and scanning the runs, which
/// keeps the probe free of per-pair allocations.
///
/// The two old levels are read in place and only replaced once both new ones
/// are built, so `Ok(None)` and every error leave the diagram byte-for-byte
/// what it was on entry and the caller may probe the next candidate without
/// any undo of its own. The rebuild returns `Ok(None)` when the rotation would
/// exceed `max_pairs`, or when a bail check shows it cannot produce a
/// well-formed pair of levels.
///
/// `dir` says whether the rotation promoted `w` from v's right child (a left
/// rotation) or its left child (a right rotation), which fixes the geometry
/// of the triple expansion.
///
/// # Errors
///
/// [`OperationError::OverBudget`] when a buffer the rebuild needs is refused,
/// either by the allocator or by the armed byte budget. Every allocation the
/// expansion makes goes through the engine's limits, so a rotation too wide
/// for the host comes back as an answer rather than an abort.
pub(crate) fn restructure_inner_search(
    lim: &Limits,
    tdd: &mut Tdd,
    info: &RotationInfo,
    dir: RotationKind,
    scratch: &mut RestructureScratch,
    max_pairs: usize,
) -> Result<Option<(TddLevel, TddLevel)>, OperationError> {
    let v_idx = info.v_idx.idx();
    let w_idx = info.w_idx.idx();
    let marginal_ctx = tdd.has_marginal_level();
    // The group table indexes `triples` with `u32` offsets; `collect_triples`
    // gives up once the count reaches `max_pairs`.
    let max_pairs = max_pairs.min(u32::MAX as usize);
    // Read in place: nothing leaves the diagram until both new levels exist, so
    // an early exit has nothing to undo.
    let (old_v, old_w) = (&tdd.levels[v_idx], &tdd.levels[w_idx]);

    scratch.packed.clear();
    scratch.distinct_inner.clear();
    let Some(n_w_pairs) = collect_triples(
        lim,
        old_v,
        old_w,
        dir,
        &mut scratch.packed,
        &mut scratch.distinct_inner,
        max_pairs,
    )?
    else {
        return Ok(None);
    };

    // Phase 2: sort the packed triples. A `u128` numeric sort is order-identical
    // to sorting the `(inner, src, axis)` tuple lexicographically (see
    // `pack_triple`), but a single-key integer sort instead of a four-field
    // branchy compare. After sorting, cells for each inner pair are contiguous
    // and sorted — no per-group sort needed.
    scratch.packed.sort_unstable();

    // `group_info` addresses `triples` with u32 offsets. The u32 width of a
    // `NodeIdx` bounds node indices, not this arena-scale offset: past 2^32
    // triples the `as u32` casts below would wrap, `cells_eq` would compare
    // wrong-but-in-range cell slices, and the resulting inner-node sharing would
    // silently change the count. `write <= read <= n`, so this single check
    // covers every cast in the scan.
    let n = scratch.packed.len();
    debug_assert!(
        u32::try_from(n).is_ok(),
        "rotation restructure: {n} triples exceeds the u32 group offsets into `triples`",
    );

    // In marginal context (full expansion) the cell multiset is kept: a duplicate
    // (src,axis) cell is a legitimate separate count-contribution (two
    // marginalization-collapsed twin primes), so cell-deduping it would drop
    // count-mass. Boolean mode dedups.
    scratch.group_info.clear();
    group_by_inner_pair(lim, &mut scratch.packed, &mut scratch.group_info, marginal_ctx)?;

    scratch.inner_pair_to_idx.clear();
    let Some(inner_level) = build_inner_level(
        lim,
        &scratch.packed,
        &mut scratch.group_info,
        &mut scratch.inner_pair_to_idx,
        marginal_ctx,
        n_w_pairs,
        max_pairs,
    )?
    else {
        return Ok(None);
    };

    // Last read of `group_info` (both branches consumed it building the inner
    // level); release it before the outer level's per-v pair lists and arena.
    release_or_clear(lim, &mut scratch.group_info, SCRATCH_RETAIN_ENTRIES);

    let outer_level = match build_outer_level(
        lim,
        old_v,
        &mut scratch.packed,
        &scratch.inner_pair_to_idx,
        &mut scratch.per_v_pairs,
        dir,
        marginal_ctx,
    ) {
        Ok(level) => level,
        Err(e) => {
            // The inner level was built but never installed; its arena is about
            // to be dropped, so hand its charge back.
            lim.release_bytes(inner_level.arena_capacity_bytes());
            return Err(e);
        }
    };

    // Rotation locality: only w_idx can have fresh twins, and contraction reaches
    // a level through its parent, so the outer level is what changed here.
    tdd.invalidate(crate::vtree::VtreeIdx(v_idx as u32));
    let old_w_level = std::mem::replace(&mut tdd.levels[w_idx], inner_level);
    let old_v_level = std::mem::replace(&mut tdd.levels[v_idx], outer_level);
    Ok(Some((old_v_level, old_w_level)))
}

/// Phase 1: expand every old v-pair against the w-level into packed triples,
/// and count the distinct inner pairs (bail check 1). Returns the distinct-inner
/// count, or `Ok(None)` if the rotation would exceed `max_pairs`.
/// `distinct_inner` is released before returning — only its count survives.
///
/// # Errors
///
/// [`OperationError::OverBudget`] if the triple buffer's growth is refused.
fn collect_triples(
    lim: &Limits,
    old_v_level: &TddLevel,
    old_w_level: &TddLevel,
    dir: RotationKind,
    triples: &mut Vec<u128>,
    distinct_inner: &mut FxHashSet<ChildPair>,
    max_pairs: usize,
) -> Result<Option<usize>, OperationError> {
    for i in 0..old_v_level.nodes.len() {
        if !old_v_level.nodes[i].is_internal() { continue; }
        let src = i as u32;
        for vp in old_v_level.pairs_iter_of_idx(i) {
            let (w_local, v_axis) = match dir {
                RotationKind::Left => (ChildDecoder::structural().node(vp.right).idx(), vp.left),
                RotationKind::Right => (ChildDecoder::structural().node(vp.left).idx(), vp.right),
            };
            for wp in old_w_level.pairs_iter_of_idx(w_local) {
                let (inner, axis) = match dir {
                    RotationKind::Left => (
                        ChildPair::new(v_axis, wp.left),
                        wp.right,
                    ),
                    RotationKind::Right => (
                        ChildPair::new(wp.right, v_axis),
                        wp.left,
                    ),
                };
                distinct_inner.insert(inner);
                lim.try_push(triples, pack_triple(inner, src, axis))?;
                // Bail on the raw triple count too: one `vp` whose `w_local` fans
                // out widely can push `triples` far past `max_pairs` before the
                // end-of-vp check below runs. `triples.len() >=
                // distinct_inner.len()` always, so this bail is the tighter one.
                if triples.len() >= max_pairs {
                    return Ok(None);
                }
            }
            if distinct_inner.len() >= max_pairs {
                return Ok(None);
            }
        }
    }
    let n_w_pairs = distinct_inner.len();
    // Last read of `distinct_inner`: only its count survives (bail check 2).
    // Release it here — it is one slot per distinct inner pair and would
    // otherwise stay resident across the sort and both level builds.
    release_or_clear(lim, distinct_inner, SCRATCH_RETAIN_ENTRIES);
    Ok(Some(n_w_pairs))
}

/// One distinct inner pair: its cell list's fingerprint hash, the pair itself,
/// and the `[start, end)` bounds of its cells in the packed `triples`.
pub(super) type PairGroup = (u64, ChildPair, u32, u32);

/// Phase 2: dedup cells in-place within each inner-pair group of the sorted
/// `triples` and record the group boundaries with a rolling fingerprint hash.
/// `keep_cells` retains the cell multiset instead of deduping it.
///
/// # Errors
///
/// [`OperationError::OverBudget`] if the group table's growth is refused.
fn group_by_inner_pair(
    lim: &Limits,
    triples: &mut Vec<u128>,
    group_info: &mut Vec<PairGroup>,
    keep_cells: bool,
) -> Result<(), OperationError> {
    let mut read = 0;
    let mut write = 0;
    let n = triples.len();
    while read < n {
        let inner_key = tri_inner_key(triples[read]);
        let group_start = write as u32;
        let mut fp_hash: u64 = 0;
        let mut prev_cell = u64::MAX;
        while read < n && tri_inner_key(triples[read]) == inner_key {
            let cell = tri_cell(triples[read]); // (src << 32) | axis
            if keep_cells || cell != prev_cell {
                triples[write] = triples[read];
                write += 1;
                fp_hash = fp_hash.wrapping_mul(0x517cc1b727220a95)
                    .wrapping_add(cell);
                prev_cell = cell;
            }
            read += 1;
        }
        let inner = ChildPair::new(EncodedChildRef::from_raw((inner_key >> 32) as u32), EncodedChildRef::from_raw(inner_key as u32));
        lim.try_push(group_info, (fp_hash, inner, group_start, write as u32))?;
    }
    triples.truncate(write);
    Ok(())
}

/// Phases 3 and 4: turn the inner-pair groups into one inner level, recording
/// each pair's node index in `inner_pair_to_idx`. Returns `Ok(None)` when bail
/// check 2 says the result would exceed `max_pairs`.
///
/// Two shapes, chosen by `marginal_ctx`: full expansion, one node per distinct
/// inner pair, or cell-list clustering, one node per distinct cell list.
///
/// # Errors
///
/// [`OperationError::OverBudget`] if the level's arena is refused. The
/// partially built level is dropped, so its charge goes back first.
fn build_inner_level(
    lim: &Limits,
    triples: &[u128],
    group_info: &mut [PairGroup],
    inner_pair_to_idx: &mut FxHashMap<ChildPair, NodeIdx>,
    marginal_ctx: bool,
    n_w_pairs: usize,
    max_pairs: usize,
) -> Result<Option<TddLevel>, OperationError> {
    let mut level = TddLevel::new();
    let filled = if marginal_ctx {
        // Bail check 2 (full-expand): one inner node per distinct inner pair.
        if group_info.len() + n_w_pairs >= max_pairs {
            return Ok(None);
        }
        expand_every_pair(lim, &mut level, group_info, inner_pair_to_idx)
    } else {
        // Phase 3: sort groups by fingerprint hash, so entries that can share a
        // node land in one bucket, then count the distinct cell lists that survive.
        group_info.sort_unstable_by_key(|g| g.0);
        if count_distinct_cell_lists(triples, group_info) + n_w_pairs >= max_pairs {
            return Ok(None);
        }
        cluster_by_cell_list(lim, &mut level, triples, group_info, inner_pair_to_idx)
    };
    match filled {
        Ok(()) => Ok(Some(level)),
        Err(e) => {
            lim.release_bytes(level.arena_capacity_bytes());
            Err(e)
        }
    }
}

/// The maximal runs of equal fingerprint hash in a hash-sorted `group_info`.
/// Two groups can share an inner node only inside one such run, so the count
/// and the build walk the same partition.
fn hash_buckets(group_info: &[PairGroup]) -> impl Iterator<Item = &[PairGroup]> {
    let mut start = 0;
    std::iter::from_fn(move || {
        let hash = group_info.get(start)?.0;
        let mut end = start + 1;
        while end < group_info.len() && group_info[end].0 == hash {
            end += 1;
        }
        let bucket = &group_info[start..end];
        start = end;
        Some(bucket)
    })
}

/// Marginal full-expand: sharing is suppressed, so each distinct inner pair
/// becomes its own node and no two pairs merge under one.
///
/// The Boolean `(a∧b)∨(a'∧b')` share that miscounts a marginalized grandchild
/// never forms. With the kept cell multiset and the kept outer multiset, Σ over
/// triples = the pre-rotation count exactly.
fn expand_every_pair(
    lim: &Limits,
    inner_level: &mut TddLevel,
    group_info: &[PairGroup],
    inner_pair_to_idx: &mut FxHashMap<ChildPair, NodeIdx>,
) -> Result<(), OperationError> {
    for g in group_info {
        let idx = inner_level.push_node_within(lim, &[g.1])?;
        inner_pair_to_idx.insert(g.1, idx);
    }
    Ok(())
}

/// How many distinct cell lists the groups hold — the inner level's node count,
/// needed before any node is built so bail check 2 can decline.
///
/// The fingerprint hash only proposes a bucket; membership is decided by
/// comparing the cell lists themselves, so a hash collision costs a wasted
/// comparison and never a wrong share.
fn count_distinct_cell_lists(triples: &[u128], group_info: &[PairGroup]) -> usize {
    let mut n_fps = 0usize;
    for bucket in hash_buckets(group_info) {
        for j in 0..bucket.len() {
            let is_new = !(0..j).any(|k| {
                cells_eq(triples, bucket[j].2, bucket[j].3, bucket[k].2, bucket[k].3)
            });
            if is_new { n_fps += 1; }
        }
    }
    n_fps
}

/// Phase 4: emit one inner node per distinct cell list, with every group that
/// carries that cell list pointing at it.
fn cluster_by_cell_list(
    lim: &Limits,
    inner_level: &mut TddLevel,
    triples: &[u128],
    group_info: &[PairGroup],
    inner_pair_to_idx: &mut FxHashMap<ChildPair, NodeIdx>,
) -> Result<(), OperationError> {
    for bucket in hash_buckets(group_info) {
        if bucket.len() == 1 {
            let idx = inner_level.push_node_within(lim, &[bucket[0].1])?;
            inner_pair_to_idx.insert(bucket[0].1, idx);
            continue;
        }
        let mut processed = vec![false; bucket.len()];
        for j in 0..bucket.len() {
            if processed[j] { continue; }
            let mut pairs = vec![bucket[j].1];
            processed[j] = true;
            for k in (j + 1)..bucket.len() {
                if processed[k] { continue; }
                if cells_eq(triples, bucket[j].2, bucket[j].3,
                            bucket[k].2, bucket[k].3) {
                    pairs.push(bucket[k].1);
                    processed[k] = true;
                }
            }
            // No canonicalizing sort: this rotated level is queued for twin
            // contraction, but twin detection is order-independent
            // (`find_twin_groups` sorts each signature slice before comparing),
            // so the node's pair order is free (see `ChildPair`).
            let idx = inner_level.push_node_within(lim, &pairs)?;
            for &p in &pairs {
                inner_pair_to_idx.insert(p, idx);
            }
        }
    }
    Ok(())
}

/// Phase 5: build the outer level from the deduped (packed) triples, one node
/// per old v-node. `triples` is released once its pairs have been distributed,
/// on the error path as well as the normal one.
///
/// # Errors
///
/// [`OperationError::OverBudget`] if a pair list or the level's arena is
/// refused. The partially built level is dropped, so its charge goes back
/// first.
fn build_outer_level(
    lim: &Limits,
    old_v_level: &TddLevel,
    triples: &mut Vec<u128>,
    inner_pair_to_idx: &FxHashMap<ChildPair, NodeIdx>,
    per_v_pairs: &mut Vec<Vec<ChildPair>>,
    dir: RotationKind,
    marginal_ctx: bool,
) -> Result<TddLevel, OperationError> {
    let n_v = old_v_level.nodes.len();
    if per_v_pairs.len() < n_v {
        lim.reserve(per_v_pairs, n_v - per_v_pairs.len())?;
        per_v_pairs.resize_with(n_v, Vec::new);
    }
    for v in &mut per_v_pairs[..n_v] { v.clear(); }
    let distributed = distribute_outer_pairs(lim, triples, inner_pair_to_idx, per_v_pairs, dir);
    // Last read of `triples`: `per_v_pairs` now holds every outer pair. Release
    // the 16 B/triple buffer before the arena that copies those pairs is built.
    release_or_clear(lim, triples, SCRATCH_RETAIN_ENTRIES);
    distributed?;

    let mut outer_level = TddLevel::new();
    match fill_outer_level(lim, &mut outer_level, old_v_level, per_v_pairs, n_v, marginal_ctx) {
        Ok(()) => Ok(outer_level),
        Err(e) => {
            lim.release_bytes(outer_level.arena_capacity_bytes());
            Err(e)
        }
    }
}

/// Turn each triple into its outer pair and file it under the old v-node it
/// came from.
fn distribute_outer_pairs(
    lim: &Limits,
    triples: &[u128],
    inner_pair_to_idx: &FxHashMap<ChildPair, NodeIdx>,
    per_v_pairs: &mut [Vec<ChildPair>],
    dir: RotationKind,
) -> Result<(), OperationError> {
    for &p in triples {
        let inner = tri_inner(p);
        let src = tri_src(p);
        let axis = tri_axis(p);
        let inner_idx = inner_pair_to_idx[&inner];
        let outer_pair = match dir {
            RotationKind::Left => ChildPair::new(inner_idx, axis),
            RotationKind::Right => ChildPair::new(axis, inner_idx),
        };
        lim.try_push(&mut per_v_pairs[src as usize], outer_pair)?;
    }
    Ok(())
}

/// Emit one outer node per old v-node from the filed pair lists, keeping the
/// old level's node count and ordering so references from above stay valid.
fn fill_outer_level(
    lim: &Limits,
    outer_level: &mut TddLevel,
    old_v_level: &TddLevel,
    per_v_pairs: &mut [Vec<ChildPair>],
    n_v: usize,
    marginal_ctx: bool,
) -> Result<(), OperationError> {
    // Indexes `old_v_level.nodes` and `per_v_pairs` at the same position.
    #[expect(clippy::needless_range_loop)]
    for i in 0..n_v {
        if !old_v_level.nodes[i].is_internal() {
            lim.try_push(&mut outer_level.nodes, old_v_level.nodes[i])?;
            continue;
        }
        // Load-bearing dedup: distinct triples can produce the same outer pair,
        // so duplicates are genuinely manufactured here. The sort exists only to
        // enable the adjacent `dedup` — not to canonicalize node order.
        per_v_pairs[i].sort_unstable();
        // Marginal full-expand keeps the outer multiset: a duplicate outer pair is
        // a legitimate separate count-mass (two marginalization-collapsed twin
        // primes), so Σ over the kept multiset = the pre-rotation count exactly;
        // deduping there would drop that mass (undercount). Pure-Boolean rotations
        // dedup: under determinism a repeated outer pair is a genuinely redundant
        // path.
        if !marginal_ctx {
            per_v_pairs[i].dedup();
        }
        outer_level.push_node_within(lim, &per_v_pairs[i])?;
    }
    Ok(())
}


/// Compare two cell-list slices in the deduped (packed) triples array. Two cells
/// are equal iff their `(src, axis)` parts match — that is the low 64 bits of the
/// packed key (`tri_cell`), so the comparison reduces to a `u64` elementwise
/// equality over the two slices.
#[inline]
fn cells_eq(
    triples: &[u128],
    a_start: u32, a_end: u32,
    b_start: u32, b_end: u32,
) -> bool {
    let a = &triples[a_start as usize..a_end as usize];
    let b = &triples[b_start as usize..b_end as usize];
    a.len() == b.len() && a.iter().zip(b.iter()).all(|(&x, &y)| tri_cell(x) == tri_cell(y))
}

#[cfg(test)]
#[path = "tests/relevel/mod.rs"]
mod tests;
