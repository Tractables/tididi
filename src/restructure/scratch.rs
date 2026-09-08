//! The reusable rotation-probe scratch and its thread-local pool.

use std::cell::Cell;

use rustc_hash::{FxHashMap, FxHashSet};

use crate::diagram::*;
use crate::utils::{pool_put, pool_take};

/// Reusable scratch for `restructure_after_*_rotation_bounded`. Threaded by
/// the rotation-search loops so the per-probe allocator churn is paid once per
/// search rather than once per probe. Each field is `clear()`-ed before use in
/// `restructure_inner_search`, preserving the underlying capacity; a field is
/// then released at its last read within the call rather than held across the
/// successor-level builds (see `SCRATCH_RETAIN_ENTRIES`).
#[derive(Default)]
#[doc(hidden)]
pub struct RestructureScratch {
    pub(super) inner_pair_to_idx: FxHashMap<InputPair, LocalNodeIdx>,
    // Per-v-node output pair lists; outer Vec grown with `resize_with`, inner
    // Vecs `clear()`-ed per call so their capacity survives across probes.
    pub(super) per_v_pairs: Vec<Vec<InputPair>>,
    pub(super) distinct_inner: FxHashSet<InputPair>,
    pub(super) group_info: Vec<(u64, InputPair, u32, u32)>, // (fp_hash, inner, start, end)
    // Search path triples, packed one-per-u128 (see `pack_triple`). The sort in
    // `restructure_inner_search` is the dominant cost of the joint next-merge-cost
    // probe on single-large-component pools; sorting a `Vec<u128>` by a single
    // integer key replaces the derived lexicographic compare over the
    // `(InputPair, u32, LocalNodeIdx)` tuple's four u32 fields.
    pub(super) packed: Vec<u128>,
}

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
// fresh `RestructureScratch` per call. On a workload of many tiny diagrams that is
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
/// next take starts from empty.
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
pub(super) fn release_vec<T>(buf: &mut Vec<T>) {
    if buf.capacity() > SCRATCH_RETAIN_ENTRIES { *buf = Vec::new(); } else { buf.clear(); }
}

/// Release a scratch set at its last read (see `SCRATCH_RETAIN_ENTRIES`).
#[inline]
pub(super) fn release_set(buf: &mut FxHashSet<InputPair>) {
    if buf.capacity() > SCRATCH_RETAIN_ENTRIES { *buf = FxHashSet::default(); } else { buf.clear(); }
}
