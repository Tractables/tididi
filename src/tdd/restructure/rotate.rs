//! TDD level restructuring after a vtree rotation.
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
//! are rotation-invariant, and TDD
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

use std::cell::Cell;

use rustc_hash::{FxHashMap, FxHashSet};

use crate::vtree::rotate::RotationInfo;
use crate::tdd::types::*;
use crate::tdd::utils::{pool_put, pool_take};


// Whole-diagram marg context is `Tdd::has_marginal_level()`. Rotation regrouping
// must use multiset semantics EVERYWHERE in a marginalized diagram — a level
// whose immediate a/b/c aren't marginal can still carry count-bearing duplicate
// pairs that propagated up from a marginal subtree, and the Boolean dedup would
// wrongly collapse them.

// MARGINAL-CONTEXT FULL EXPANSION (the marg_ctx branches below).
//
// A rotation regroups the products `a·b·c` of a triple into shared inner/outer
// nodes. The Boolean restructure shares an inner node across two DISTINCT inner
// pairs P1≠P2 with the same cell fingerprint — `(a1∧b1)∨(a2∧b2)` — and dedups
// duplicate outer pairs. Both are sound ONLY under A-level determinism (primes
// mutex). A *marginalized* level breaks that: its stored count is a collapsed
// aggregate, and `dedup_fresh_store` merges distinct count-bearing subtrees that
// share a count value into one slot — so two regrouped branches can become
// content-identical "twins" whose counts must SUM, not collapse. Boolean dedup
// drops that mass (undercount); sharing-with-keep manufactures it (overcount).
// On the mc043 reproducer: dedup→25, keep+share→46, truth=32.
//
// Fix: whenever the diagram contains ANY marginal level (`marg_ctx`), FULLY
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

/// Reusable scratch for `restructure_after_*_rotation_bounded`. Threaded by
/// the rotation-search loops so the per-probe allocator churn is paid once per
/// search rather than once per probe. Each field is `clear()`-ed before use in
/// `restructure_inner_search`, preserving the underlying capacity; a field is
/// then released at its last read within the call rather than held across the
/// successor-level builds (see `SCRATCH_RETAIN_ENTRIES`).
#[derive(Default)]
#[doc(hidden)]
pub struct RestructureScratch {
    inner_pair_to_idx: FxHashMap<InputPair, LocalNodeIdx>,
    // Per-v-node output pair lists; outer Vec grown with `resize_with`, inner
    // Vecs `clear()`-ed per call so their capacity survives across probes.
    per_v_pairs: Vec<Vec<InputPair>>,
    distinct_inner: FxHashSet<InputPair>,
    group_info: Vec<(u64, InputPair, u32, u32)>, // (fp_hash, inner, start, end)
    // Search path triples, packed one-per-u128 (see `pack_triple`). The sort in
    // `restructure_inner_search` is the dominant cost of the joint next-merge-cost
    // probe on single-large-component pools; sorting a `Vec<u128>` by a single
    // integer key replaces the derived lexicographic compare over the
    // `(InputPair, u32, LocalNodeIdx)` tuple's four u32 fields.
    packed: Vec<u128>,
}

/// Pack a search triple `(inner, src, axis)` into one `u128` whose numeric order
/// is IDENTICAL to the tuple's derived lexicographic order `(inner.left,
/// inner.right, src, axis)` — all four fields are `u32`, so the packing is
/// lossless and order-preserving. The high 64 bits are the `inner` pair (its own
/// sort key); the low 64 bits are the `(src, axis)` "cell". Sorting by this key
/// therefore groups cells by `inner` and orders cells within a group by
/// `(src, axis)` — exactly what the group scan below relies on, but as a single
/// `u128` compare instead of a four-field branchy tuple compare.
#[inline]
fn pack_triple(inner: InputPair, src: u32, axis: LocalNodeIdx) -> u128 {
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
    InputPair { left: LocalNodeIdx((p >> 96) as u32), right: LocalNodeIdx((p >> 64) as u32) }
}
#[inline]
fn tri_src(p: u128) -> u32 { (p >> 32) as u32 }
#[inline]
fn tri_axis(p: u128) -> LocalNodeIdx { LocalNodeIdx(p as u32) }

impl RestructureScratch {
    /// Create an empty scratch buffer.
    pub fn new() -> Self { Self::default() }

    /// Empty every buffer while retaining its allocation. Called by
    /// [`take_scratch`] so a pooled scratch is indistinguishable from a fresh
    /// one except for capacity — the search's own per-use `clear()`s then
    /// become no-ops.
    fn clear(&mut self) {
        self.inner_pair_to_idx.clear();
        self.distinct_inner.clear();
        self.group_info.clear();
        self.packed.clear();
        // Keep the outer Vec's length (bounded by `PER_V_PAIRS_RETAIN` on
        // return): `restructure_inner_search` only `resize_with`s it upward and
        // clears the prefix it uses, so the inner Vecs' capacities are exactly
        // what we want to carry forward.
        for v in &mut self.per_v_pairs {
            v.clear();
        }
    }
}

// ── Thread-local scratch pool ───────────────────────────────────────────────
//
// `cluster_marginal_rotations_in_subtree` / `rotation_search` used to build a
// fresh `RestructureScratch` per call. On the canopy leaf workload that is
// ~28 scratch lifetimes per ~3 ms leaf, and each teardown freed the whole
// `per_v_pairs` fan-out: callgrind measured 1,380 `sdallocx` calls per leaf
// (0.34% of the window) under `drop_in_place<RestructureScratch>` alone, plus
// the re-growth of the same buffers (and the two hash tables) on the next
// call. Pooling follows `minimize::contract::scratch` exactly: one
// thread-local `Cell<Option<_>>`, cleared on take, capacity-capped on return.

thread_local! {
    static SCRATCH: Cell<Option<RestructureScratch>> = const { Cell::new(None) };
}

/// Maximum retained `packed` capacity (4M triples × 16 B = 64 MB). A rare wide
/// rotation search must not park its peak buffers in the pool for the rest of
/// the process; past this, the size-proportional buffers are released and the
/// next take starts from empty. Mirrors `CONTRACT_ENTRIES_CAP_LIMIT`.
const RESTRUCTURE_PACKED_CAP_LIMIT: usize = 4_000_000;

/// Maximum number of per-v-node pair lists carried across calls. The take-side
/// `clear()` walks the whole outer Vec, so an unbounded one would tax every
/// later (small) search with the widest level this thread ever saw — the pool
/// must not turn one wide rotation into a permanent per-call O(width) sweep.
/// Beyond this the tail is dropped; `restructure_inner_search` re-grows it with
/// `resize_with` exactly as it does on a cold scratch.
const PER_V_PAIRS_RETAIN: usize = 1024;

/// Take the thread's restructure scratch, cleared and ready to use. Returns a
/// fresh one when the pool is empty (first use on this thread, after a
/// capacity-capped return, or when a nested search already holds it).
pub(crate) fn take_scratch() -> RestructureScratch {
    let mut s = pool_take(&SCRATCH).unwrap_or_default();
    s.clear();
    s
}

/// Return the scratch for reuse by the next rotation search on this thread.
/// Not returning it (an unwind, an early `return`) is safe: the pool simply
/// stays empty and the next take allocates.
pub(crate) fn return_scratch(mut s: RestructureScratch) {
    s.per_v_pairs.truncate(PER_V_PAIRS_RETAIN);
    if s.packed.capacity() > RESTRUCTURE_PACKED_CAP_LIMIT {
        s.packed = Vec::new();
        s.group_info = Vec::new();
        s.per_v_pairs = Vec::new();
        s.inner_pair_to_idx = FxHashMap::default();
        s.distinct_inner = FxHashSet::default();
    }
    pool_put(&SCRATCH, Some(s));
}

/// Scratch entries kept across probes. At its last read a buffer this size or
/// smaller is only `clear()`-ed, so the next probe reuses the allocation — the
/// churn-avoidance the scratch exists for. A larger one is released outright:
/// holding a high-water buffer across the successor-level builds, which
/// allocate their own copy of the same data, is what sets this function's peak,
/// and regrowing it costs one pass next to the sort that dominates a probe big
/// enough to be over the threshold. Bail-out probes return before any release
/// point, so the search's common path keeps full capacity either way.
const SCRATCH_RETAIN_ENTRIES: usize = 1 << 16;

/// Release a scratch `Vec` at its last read (see `SCRATCH_RETAIN_ENTRIES`).
#[inline]
fn release_vec<T>(buf: &mut Vec<T>) {
    if buf.capacity() > SCRATCH_RETAIN_ENTRIES { *buf = Vec::new(); } else { buf.clear(); }
}

/// Release a scratch set at its last read (see `SCRATCH_RETAIN_ENTRIES`).
#[inline]
fn release_set(buf: &mut FxHashSet<InputPair>) {
    if buf.capacity() > SCRATCH_RETAIN_ENTRIES { *buf = FxHashSet::default(); } else { buf.clear(); }
}

/// Restructure after a left rotation with early bail-out. If the number of
/// distinct inner pairs exceeds `max_inner_pairs` during triple collection,
/// the rotation is guaranteed to increase size (since `new_w_pairs` would exceed
/// the threshold). Returns `None` on bail-out (levels restored to pre-rotation
/// state); `Some((old_v, old_w))` on success.
#[doc(hidden)]
pub fn restructure_after_left_rotation_bounded(
    tdd: &mut Tdd,
    info: &RotationInfo,
    scratch: &mut RestructureScratch,
    max_inner_pairs: usize,
) -> Option<(TddLevel, TddLevel)> {
    restructure_inner_search(tdd, info, RotDir::Left, scratch, max_inner_pairs)
}

/// Restructure after a right rotation with early bail-out. See
/// `restructure_after_left_rotation_bounded`.
#[doc(hidden)]
pub fn restructure_after_right_rotation_bounded(
    tdd: &mut Tdd,
    info: &RotationInfo,
    scratch: &mut RestructureScratch,
    max_inner_pairs: usize,
) -> Option<(TddLevel, TddLevel)> {
    restructure_inner_search(tdd, info, RotDir::Right, scratch, max_inner_pairs)
}

/// Sort-based restructure for search probes. Replaces `HashMap` cell grouping
/// with sort + linear scan, eliminating per-inner-pair Vec allocations.
/// Uses `HashSet` for bail check 1 (cheaper than `HashMap`).
fn restructure_inner_search(
    tdd: &mut Tdd,
    info: &RotationInfo,
    dir: RotDir,
    scratch: &mut RestructureScratch,
    max_pairs: usize,
) -> Option<(TddLevel, TddLevel)> {
    let v_idx = info.v_idx.idx();
    let w_idx = info.w_idx.idx();
    let marg_ctx = tdd.levels[info.a_idx.idx()].is_marginal()
        || tdd.levels[info.b_idx.idx()].is_marginal()
        || tdd.levels[info.c_idx.idx()].is_marginal()
        || tdd.has_marginal_level();
    let old_v_level = std::mem::replace(&mut tdd.levels[v_idx], TddLevel::new());
    let old_w_level = std::mem::replace(&mut tdd.levels[w_idx], TddLevel::new());

    // Phase 1: collect triples (packed one-per-u128) + HashSet for bail check 1.
    scratch.packed.clear();
    scratch.distinct_inner.clear();
    let triples = &mut scratch.packed;
    let distinct_inner = &mut scratch.distinct_inner;

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
                // `max_pairs` entries *before* the end-of-vp check ever runs —
                // observed as a >16 GiB single-allocation OOM on mc2025_track1_081
                // even with a loose `max_pairs`. `triples.len() >= distinct_inner.len()`
                // always, so this is a strictly tighter, always-valid bail — checked
                // every push since the cost is one `Vec::len()` compare against the
                // hash-insert already paid on this line.
                if triples.len() >= max_pairs {
                    tdd.levels[v_idx] = old_v_level;
                    tdd.levels[w_idx] = old_w_level;
                    return None;
                }
            }
            if distinct_inner.len() >= max_pairs {
                tdd.levels[v_idx] = old_v_level;
                tdd.levels[w_idx] = old_w_level;
                return None;
            }
        }
    }

    let n_w_pairs = distinct_inner.len();
    // Last read of `distinct_inner`: only its count survives (bail check 2).
    // Release it here — it is one slot per distinct inner pair and would
    // otherwise stay resident across the sort and both level builds.
    release_set(distinct_inner);

    // Phase 2: sort the packed triples. A `u128` numeric sort is order-identical
    // to sorting the `(inner, src, axis)` tuple lexicographically (see
    // `pack_triple`), but a single-key integer sort instead of a four-field
    // branchy compare. After sorting, cells for each inner pair are contiguous
    // and sorted — no per-group sort needed.
    triples.sort_unstable();

    // Dedup cells in-place within each inner-pair group and extract group
    // boundaries with a rolling fingerprint hash. In marginal context (full
    // expansion) KEEP the cell multiset: a duplicate (src,axis) cell is a
    // legitimate separate count-contribution (two marginalization-collapsed twin
    // primes), so cell-deduping it would drop count-mass. Boolean mode dedups.
    let keep_cells = marg_ctx;
    scratch.group_info.clear();
    let group_info = &mut scratch.group_info;
    let mut read = 0;
    let mut write = 0;
    let n = triples.len();
    // `group_info` addresses `triples` with u32 offsets. The u32 width of a
    // `LocalNodeIdx` bounds node indices, NOT this arena-scale offset: past 2^32
    // triples the `as u32` casts below would wrap, `cells_eq` would compare
    // wrong-but-in-range cell slices, and the resulting inner-node sharing would
    // silently change the count. `write <= read <= n`, so this single check
    // covers every cast in the scan. Restore the levels first so an unwind
    // leaves the diagram consistent.
    if u32::try_from(n).is_err() {
        tdd.levels[v_idx] = old_v_level;
        tdd.levels[w_idx] = old_w_level;
        panic!("rotation restructure: {n} triples exceeds the u32 group offsets into `triples`");
    }
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
            left: LocalNodeIdx((inner_key >> 32) as u32),
            right: LocalNodeIdx(inner_key as u32),
        };
        group_info.push((fp_hash, inner, group_start, write as u32));
    }
    triples.truncate(write);

    // Marginal full-expand: suppress inner-node sharing. Each distinct inner pair
    // becomes its own node, so no two pairs merge under one node — the Boolean
    // `(a∧b)∨(a'∧b')` share that miscounts a marginalized grandchild never forms.
    // With the kept cell multiset (above) and the kept outer multiset (below),
    // Σ over triples = the pre-rotation count exactly. `group_info` holds one
    // entry per distinct inner pair (Phase 2), so we emit one node per entry.
    let n_groups = group_info.len();
    let mut inner_level = TddLevel::new();
    scratch.inner_pair_to_idx.clear();
    let inner_pair_to_idx = &mut scratch.inner_pair_to_idx;

    if marg_ctx {
        // Bail check 2 (full-expand): one inner node per distinct inner pair.
        if n_groups + n_w_pairs >= max_pairs {
            tdd.levels[v_idx] = old_v_level;
            tdd.levels[w_idx] = old_w_level;
            return None;
        }
        for g in group_info.iter() {
            let idx = inner_level.push_internal_node(&[g.1]);
            inner_pair_to_idx.insert(g.1, idx);
        }
    } else {
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
            tdd.levels[v_idx] = old_v_level;
            tdd.levels[w_idx] = old_w_level;
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
                    // so the node's pair order is free (see the NOTE in `tdd/types.rs`).
                    let idx = inner_level.push_internal_node(&pairs);
                    for &p in &pairs {
                        inner_pair_to_idx.insert(p, idx);
                    }
                }
            }
        }
    }

    // Last read of `group_info` (both branches consumed it building the inner
    // level); release it before the outer level's per-v pair lists and arena.
    release_vec(group_info);

    // Phase 5: build outer level from deduped (packed) triples.
    let mut outer_level = TddLevel::new();
    let n_v = old_v_level.nodes.len();
    let per_v_pairs = &mut scratch.per_v_pairs;
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
    release_vec(triples);
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
        if !marg_ctx {
            per_v_pairs[i].dedup();
        }
        outer_level.push_internal_node(&per_v_pairs[i]);
    }

    tdd.levels[w_idx] = inner_level;
    tdd.levels[v_idx] = outer_level;
    // §9: only w_idx can have fresh twins; seed v_idx so worklist visits w_idx.
    tdd.scratch.dirty_contract.push(v_idx as u32);
    Some((old_v_level, old_w_level))
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
