//! Diagram level restructuring after a vtree rotation.
//!
//! When the vtree is rotated at node `v` (sharing the same indices `(v_idx, w_idx)`
//! before and after), only the levels at `v_idx` and `w_idx` need to be rebuilt;
//! every other level is unchanged because:
//!
//! - All other levels' pair contents reference children unaffected by the rotation.
//! - The level at `v_idx` keeps its node count and ordering, so any higher
//!   level that references `v_idx`-nodes by local index remains valid.
//!
//! Rotation Locality is the semantic argument: variable partitions outside `w`
//! are rotation-invariant, and diagram
//! canonicity then forces every level except `v_idx` (pair-list rewrite) and
//! `w_idx` (rebuild) to be bit-for-bit identical pre- and post-rotation.
//!
//! The two affected levels are derived by **expanding triples** of node ids and
//! regrouping them along a different axis. Both directions share the same
//! shape; only the "which axis is preserved" varies. See
//! `restructure_inner_search` for the unified implementation.
//!
//! ## Left rotation (v=(A, w), w=(B, C) → `v_new=(A`, B) at `w_idx`, `w_new=(v_new`, C) at `v_idx`)
//!
//! Each old v-pair `(a, w_local)` expands to triples `(a, b, c)` for each
//! `(b, c)` in `w_level[w_local]`. Triples are sorted by `(c, a, b)`. Within
//! each `c`-group the unique sorted `(a, b)`-set defines a `v_new` node (deduped
//! globally across all v-nodes via a `HashMap`); the c-group emits a single
//! `(v_new_idx, c)` pair to the corresponding `w_new` node.
//!
//! ## Right rotation (v=(w, C), w=(A, B) → v=(A, `w_new`), `w_new=(B`, C) at `w_idx`)
//!
//! Mirror image. Each old v-pair `(w_local, c)` expands to triples `(a, b, c)`
//! for each `(a, b)` in `w_level[w_local]`. Triples are sorted by `(a, b, c)`.
//! Within each `a`-group the unique sorted `(b, c)`-set defines a `w_new` node;
//! the a-group emits `(a, w_new_idx)` to the new v-pair.
//!
//! Output handling: `w_new` (or new outer) nodes are produced 1-to-1 with the
//! old v-nodes in the same order, so the output's local index at `v_idx`
//! stays valid without remapping.

use crate::diagram::Changed;
use rustc_hash::{FxHashMap, FxHashSet};

use std::sync::Arc;
use crate::vtree::Vtree;
use crate::vtree::rotate::RotationInfo;
use crate::diagram::*;


// Whole-diagram marginal context is `Tdd::has_marginal_level()`. Rotation regrouping
// must use multiset semantics EVERYWHERE in a marginalized diagram — a level
// whose immediate a/b/c aren't marginal can still carry count-bearing duplicate
// pairs that propagated up from a marginal subtree, and the Boolean dedup would
// wrongly collapse them.

// MARGINAL-CONTEXT FULL EXPANSION (the marginal_ctx branches below).
//
// A rotation regroups the products `a·b·c` of a triple into shared inner/outer
// nodes. The Boolean restructure shares an inner node across two DISTINCT inner
// pairs P1≠P2 with the same cell fingerprint — `(a1∧b1)∨(a2∧b2)` — and dedups
// duplicate outer pairs. Both are sound only under A-level determinism (primes
// mutex). A *marginalized* level breaks that: its stored count is a collapsed
// aggregate, and `dedup_fresh_store` merges distinct count-bearing subtrees that
// share a count value into one slot — so two regrouped branches can become
// content-identical "twins" whose counts must SUM, not collapse. Boolean dedup
// drops that mass (undercount); sharing-with-keep manufactures it (overcount).
// On the mc043 reproducer: dedup→25, keep+share→46, truth=32.
//
// Fix: whenever the diagram contains any marginal level (`marginal_ctx`), FULLY
// EXPAND — one inner node per distinct inner pair, keep the cell multiset, keep
// the outer multiset. Σ over the kept triples = the pre-rotation count exactly,
// BY CONSTRUCTION (the rotation only regroups the same products). The diagram is
// larger, but a later sound twin-contraction can re-share genuine Boolean twins.

/// Whether the rotation promotes `w` from v's right (left rotation) or left
/// (right rotation) child. Determines the geometry of the triple expansion.
#[derive(Clone, Copy, PartialEq, Eq)]
enum RotDir {
    Left,
    Right,
}

pub use super::scratch::RestructureScratch;
pub(crate) use super::scratch::{return_scratch, take_scratch};
use super::scratch::SCRATCH_RETAIN_ENTRIES;
use crate::engine::pool::release_or_clear;

/// Pack a search triple `(inner, src, axis)` into one `u128` whose numeric order
/// is IDENTICAL to the tuple's derived lexicographic order `(inner.left,
/// inner.right, src, axis)` — all four fields are `u32`, so the packing is
/// lossless and order-preserving. The high 64 bits are the `inner` pair (its own
/// sort key); the low 64 bits are the `(src, axis)` "cell". Sorting by this key
/// therefore groups cells by `inner` and orders cells within a group by
/// `(src, axis)` — exactly what the group scan below relies on, but as a single
/// `u128` compare instead of a four-field branchy tuple compare.
#[inline]
fn pack_triple(inner: InputPair, src: u32, axis: NodeIdx) -> u128 {
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
fn tri_inner(p: u128) -> InputPair {
    InputPair { left: NodeIdx((p >> 96) as u32), right: NodeIdx((p >> 64) as u32) }
}
#[inline]
fn tri_src(p: u128) -> u32 { (p >> 32) as u32 }
#[inline]
fn tri_axis(p: u128) -> NodeIdx { NodeIdx(p as u32) }

/// Seat `tdd` on the rotated `vtree`, which the caller has just rotated at the
/// node the paired relevel names.
///
/// Rotation is two halves: the tree changes shape, and the two levels the
/// change touches are rebuilt. The second half is
/// [`relevel_after_left_rotation`] and its right-hand twin; this is the first,
/// and between the two calls the diagram is deliberately inconsistent with its
/// own tree. That is why the two are only reachable together, through the same
/// seam, and why neither is part of what a finished diagram can do.
///
/// The seating itself is [`Tdd::reseat_vtree`], which a rotation is one caller
/// of; what this adds is the name the seam is reached under.
pub fn seat_rotated_vtree(tdd: &mut Tdd, vtree: Arc<Vtree>) {
    tdd.reseat_vtree(&vtree);
}

/// Restructure after a left rotation with early bail-out. If the number of
/// distinct inner pairs exceeds `max_inner_pairs` during triple collection,
/// the rotation is guaranteed to increase size (the rebuilt inner level's pair
/// count would exceed the threshold). Returns `None` on bail-out (levels restored to pre-rotation
/// state); `Some((old_v, old_w))` on success.
pub fn relevel_after_left_rotation(
    tdd: &mut Tdd,
    info: &RotationInfo,
    scratch: &mut RestructureScratch,
    max_inner_pairs: usize,
) -> Option<(TddLevel, TddLevel)> {
    restructure_inner_search(tdd, info, RotDir::Left, scratch, max_inner_pairs)
}

/// Restructure after a right rotation with early bail-out. See
/// `relevel_after_left_rotation`.
pub fn relevel_after_right_rotation(
    tdd: &mut Tdd,
    info: &RotationInfo,
    scratch: &mut RestructureScratch,
    max_inner_pairs: usize,
) -> Option<(TddLevel, TddLevel)> {
    restructure_inner_search(tdd, info, RotDir::Right, scratch, max_inner_pairs)
}

/// Rebuild the two levels of a rotation in `dir` and return the levels the
/// rotation replaced, or `None` if the probe was abandoned.
///
/// Cells are grouped by sorting the inner pairs and scanning the runs, which
/// keeps the probe free of per-pair allocations.
///
/// The two old levels are read in place and only replaced once both new ones
/// are built, so a `None` return leaves the diagram byte-for-byte what it was
/// on entry and the caller may probe the next candidate without any undo of
/// its own. The probe returns `None` when the rotation would exceed
/// `max_pairs`, or when a bail check shows it cannot produce a well-formed
/// pair of levels.
fn restructure_inner_search(
    tdd: &mut Tdd,
    info: &RotationInfo,
    dir: RotDir,
    scratch: &mut RestructureScratch,
    max_pairs: usize,
) -> Option<(TddLevel, TddLevel)> {
    let v_idx = info.v_idx.idx();
    let w_idx = info.w_idx.idx();
    let marginal_ctx = tdd.levels[info.a_idx.idx()].is_marginal()
        || tdd.levels[info.b_idx.idx()].is_marginal()
        || tdd.levels[info.c_idx.idx()].is_marginal()
        || tdd.has_marginal_level();
    // Read in place: nothing leaves the diagram until both new levels exist, so
    // an early exit has nothing to undo.
    let (old_v, old_w) = (&tdd.levels[v_idx], &tdd.levels[w_idx]);

    scratch.packed.clear();
    scratch.distinct_inner.clear();
    let n_w_pairs = collect_triples(
        old_v,
        old_w,
        dir,
        &mut scratch.packed,
        &mut scratch.distinct_inner,
        max_pairs,
    )?;

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
    assert!(
        u32::try_from(n).is_ok(),
        "rotation restructure: {n} triples exceeds the u32 group offsets into `triples`",
    );

    // In marginal context (full expansion) KEEP the cell multiset: a duplicate
    // (src,axis) cell is a legitimate separate count-contribution (two
    // marginalization-collapsed twin primes), so cell-deduping it would drop
    // count-mass. Boolean mode dedups.
    scratch.group_info.clear();
    group_by_inner_pair(&mut scratch.packed, &mut scratch.group_info, marginal_ctx);

    scratch.inner_pair_to_idx.clear();
    let inner_level = build_inner_level(
        &scratch.packed,
        &mut scratch.group_info,
        &mut scratch.inner_pair_to_idx,
        marginal_ctx,
        n_w_pairs,
        max_pairs,
    )?;

    // Last read of `group_info` (both branches consumed it building the inner
    // level); release it before the outer level's per-v pair lists and arena.
    release_or_clear(&mut scratch.group_info, SCRATCH_RETAIN_ENTRIES);

    let outer_level = build_outer_level(
        old_v,
        &mut scratch.packed,
        &scratch.inner_pair_to_idx,
        &mut scratch.per_v_pairs,
        dir,
        marginal_ctx,
    );

    let old_w_level = std::mem::replace(&mut tdd.levels[w_idx], inner_level);
    let old_v_level = std::mem::replace(&mut tdd.levels[v_idx], outer_level);
    // Rotation locality: only w_idx can have fresh twins, and contraction reaches
    // a level through its parent, so the outer level is what changed here.
    tdd.invalidate(crate::vtree::VtreeIdx(v_idx as u32), Changed::PAIRS);
    // Both levels were replaced wholesale and either can have come out wider,
    // so the width cache is re-derived rather than patched.
    tdd.forget_stats();
    Some((old_v_level, old_w_level))
}

/// Phase 1: expand every old v-pair against the w-level into packed triples,
/// and count the distinct inner pairs (bail check 1). Returns the distinct-inner
/// count, or `None` if the rotation would exceed `max_pairs`. `distinct_inner`
/// is released before returning — only its count survives.
fn collect_triples(
    old_v_level: &TddLevel,
    old_w_level: &TddLevel,
    dir: RotDir,
    triples: &mut Vec<u128>,
    distinct_inner: &mut FxHashSet<InputPair>,
    max_pairs: usize,
) -> Option<usize> {
    for i in 0..old_v_level.nodes.len() {
        if !old_v_level.nodes[i].is_internal() { continue; }
        let src = i as u32;
        for vp in old_v_level.pairs_iter_of_idx(i) {
            let (w_local, v_axis) = match dir {
                RotDir::Left => (vp.right.idx(), vp.left),
                RotDir::Right => (vp.left.idx(), vp.right),
            };
            for wp in old_w_level.pairs_iter_of_idx(w_local) {
                let (inner, axis) = match dir {
                    RotDir::Left => (
                        InputPair { left: v_axis, right: wp.left },
                        wp.right,
                    ),
                    RotDir::Right => (
                        InputPair { left: wp.right, right: v_axis },
                        wp.left,
                    ),
                };
                distinct_inner.insert(inner);
                triples.push(pack_triple(inner, src, axis));
                // Bail on raw triple count too, not just the deduped distinct-pair
                // count below: on a non-canonical raw segment pool (this function's
                // only caller is the joint ensemble search over un-minimized pools)
                // a single w-level node's pair list is not width-bounded, so one
                // `vp` whose `w_local` fans out heavily can push `triples` far past
                // `max_pairs` entries *before* the end-of-vp check ever runs,
                // reaching a single allocation large enough to exhaust memory
                // even under a loose `max_pairs`.
                // `triples.len() >= distinct_inner.len()` always, so
                // this is a strictly tighter, always-valid bail — checked every
                // push since the cost is one `Vec::len()` compare against the
                // hash-insert already paid on this line.
                if triples.len() >= max_pairs {
                    return None;
                }
            }
            if distinct_inner.len() >= max_pairs {
                return None;
            }
        }
    }
    let n_w_pairs = distinct_inner.len();
    // Last read of `distinct_inner`: only its count survives (bail check 2).
    // Release it here — it is one slot per distinct inner pair and would
    // otherwise stay resident across the sort and both level builds.
    release_or_clear(distinct_inner, SCRATCH_RETAIN_ENTRIES);
    Some(n_w_pairs)
}

/// Phase 2: dedup cells in-place within each inner-pair group of the sorted
/// `triples` and record the group boundaries with a rolling fingerprint hash.
/// `keep_cells` retains the cell multiset instead of deduping it.
fn group_by_inner_pair(
    triples: &mut Vec<u128>,
    group_info: &mut Vec<(u64, InputPair, u32, u32)>,
    keep_cells: bool,
) {
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
        let inner = InputPair {
            left: NodeIdx((inner_key >> 32) as u32),
            right: NodeIdx(inner_key as u32),
        };
        group_info.push((fp_hash, inner, group_start, write as u32));
    }
    triples.truncate(write);
}

/// Phases 3 and 4: cluster the groups whose cell lists agree and emit one inner
/// node per cluster, recording each pair's node index in `inner_pair_to_idx`.
/// Returns `None` when bail check 2 says the result would exceed `max_pairs`.
///
/// Marginal full-expand suppresses inner-node sharing. Each distinct inner pair
/// becomes its own node, so no two pairs merge under one node — the Boolean
/// `(a∧b)∨(a'∧b')` share that miscounts a marginalized grandchild never forms.
/// With the kept cell multiset and the kept outer multiset, Σ over triples = the
/// pre-rotation count exactly. `group_info` holds one entry per distinct inner
/// pair, so we emit one node per entry.
fn build_inner_level(
    triples: &[u128],
    group_info: &mut [(u64, InputPair, u32, u32)],
    inner_pair_to_idx: &mut FxHashMap<InputPair, NodeIdx>,
    marginal_ctx: bool,
    n_w_pairs: usize,
    max_pairs: usize,
) -> Option<TddLevel> {
    let n_groups = group_info.len();
    let mut inner_level = TddLevel::new();

    if marginal_ctx {
        // Bail check 2 (full-expand): one inner node per distinct inner pair.
        if n_groups + n_w_pairs >= max_pairs {
            return None;
        }
        for g in group_info.iter() {
            let idx = inner_level.push_internal_node(&[g.1]);
            inner_pair_to_idx.insert(g.1, idx);
        }
        return Some(inner_level);
    }

    // Phase 3: sort groups by fingerprint hash to cluster matching fingerprints.
    group_info.sort_unstable_by_key(|g| g.0);

    // Count distinct fingerprints. Within each hash bucket, compare actual
    // cell lists to handle collisions (extremely rare with 64-bit hash).
    let mut n_fps = 0usize;
    let mut gi = 0;
    while gi < n_groups {
        let bucket_hash = group_info[gi].0;
        let bucket_start = gi;
        while gi < n_groups && group_info[gi].0 == bucket_hash {
            gi += 1;
        }
        // Within this hash bucket, count distinct cell lists.
        // For each entry, check if its cell list matches any previous entry
        // in the bucket. If not, it's a new fingerprint.
        for j in bucket_start..gi {
            let mut is_new = true;
            for k in bucket_start..j {
                if cells_eq(triples, group_info[j].2, group_info[j].3,
                            group_info[k].2, group_info[k].3) {
                    is_new = false;
                    break;
                }
            }
            if is_new { n_fps += 1; }
        }
    }

    // Bail check 2.
    if n_fps + n_w_pairs >= max_pairs {
        return None;
    }

    // Phase 4: build inner level. Scan sorted group_info, grouping entries
    // with matching cell lists into the same inner node.
    gi = 0;
    while gi < n_groups {
        let bucket_hash = group_info[gi].0;
        let bucket_start = gi;
        while gi < n_groups && group_info[gi].0 == bucket_hash {
            gi += 1;
        }
        let bucket = &group_info[bucket_start..gi];
        if bucket.len() == 1 {
            let idx = inner_level.push_internal_node(&[bucket[0].1]);
            inner_pair_to_idx.insert(bucket[0].1, idx);
        } else {
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
                // contraction, but twin detection is now order-independent
                // (`find_twin_groups` sorts each signature slice before comparing),
                // so the node's pair order is free (see `InputPair`).
                let idx = inner_level.push_internal_node(&pairs);
                for &p in &pairs {
                    inner_pair_to_idx.insert(p, idx);
                }
            }
        }
    }
    Some(inner_level)
}

/// Phase 5: build the outer level from the deduped (packed) triples, one node
/// per old v-node. `triples` is released once its pairs have been distributed.
fn build_outer_level(
    old_v_level: &TddLevel,
    triples: &mut Vec<u128>,
    inner_pair_to_idx: &FxHashMap<InputPair, NodeIdx>,
    per_v_pairs: &mut Vec<Vec<InputPair>>,
    dir: RotDir,
    marginal_ctx: bool,
) -> TddLevel {
    let mut outer_level = TddLevel::new();
    let n_v = old_v_level.nodes.len();
    if per_v_pairs.len() < n_v {
        per_v_pairs.resize_with(n_v, Vec::new);
    }
    for v in &mut per_v_pairs[..n_v] { v.clear(); }
    for &p in triples.iter() {
        let inner = tri_inner(p);
        let src = tri_src(p);
        let axis = tri_axis(p);
        let inner_idx = inner_pair_to_idx[&inner];
        let outer_pair = match dir {
            RotDir::Left => InputPair { left: inner_idx, right: axis },
            RotDir::Right => InputPair { left: axis, right: inner_idx },
        };
        per_v_pairs[src as usize].push(outer_pair);
    }
    // Last read of `triples`: `per_v_pairs` now holds every outer pair. Release
    // the 16 B/triple buffer before the arena that copies those pairs is built.
    release_or_clear(triples, SCRATCH_RETAIN_ENTRIES);
    // Indexes `old_v_level.nodes` and `per_v_pairs` at the same position.
    #[allow(clippy::needless_range_loop)]
    for i in 0..n_v {
        if !old_v_level.nodes[i].is_internal() {
            outer_level.nodes.push(old_v_level.nodes[i]);
            continue;
        }
        // Load-bearing dedup: distinct triples can produce the same outer pair,
        // so duplicates are genuinely manufactured here. The sort exists only to
        // enable the adjacent `dedup` — not to canonicalize node order.
        per_v_pairs[i].sort_unstable();
        // Marginal full-expand KEEPS the outer multiset: a duplicate outer pair is
        // a legitimate separate count-mass (two marginalization-collapsed twin
        // primes), so Σ over the kept multiset = the pre-rotation count exactly;
        // deduping there would drop that mass (undercount). Pure-Boolean rotations
        // dedup: under determinism a repeated outer pair is a genuinely redundant
        // path.
        if !marginal_ctx {
            per_v_pairs[i].dedup();
        }
        outer_level.push_internal_node(&per_v_pairs[i]);
    }
    outer_level
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
#[path = "rotate_tests.rs"]
mod tests;
