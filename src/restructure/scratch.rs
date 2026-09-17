//! The reusable rotation-probe scratch and the engine's pool for it.

use crate::limits::pool::PooledScratch;

use rustc_hash::{FxHashMap, FxHashSet};

use crate::diagram::*;

/// Reusable scratch for `restructure_inner_search`. Threaded by
/// the rotation-search loops so the per-probe allocator churn is paid once per
/// search rather than once per probe. Each field is `clear()`-ed before use in
/// `restructure_inner_search`, preserving the underlying capacity; a field is
/// then released at its last read within the call rather than held across the
/// successor-level builds (see `SCRATCH_RETAIN_ENTRIES`).
#[derive(Default)]
pub(crate) struct RestructureScratch {
    pub(super) inner_pair_to_idx: FxHashMap<ChildPair, NodeIdx>,
    // Per-v-node output pair lists; outer Vec grown with `resize_with`, inner
    // Vecs `clear()`-ed per call so their capacity survives across probes.
    pub(super) per_v_pairs: Vec<Vec<ChildPair>>,
    pub(super) distinct_inner: FxHashSet<ChildPair>,
    pub(super) group_info: Vec<super::relevel::PairGroup>,
    // Search path triples, packed one-per-u128 (see `pack_triple`). The sort in
    // `restructure_inner_search` is the dominant cost of the joint next-merge-cost
    // probe on single-large-component pools; sorting a `Vec<u128>` by a single
    // integer key replaces the derived lexicographic compare over the
    // `(ChildPair, u32, NodeIdx)` tuple's four u32 fields.
    pub(super) packed: Vec<u128>,
}

impl PooledScratch for RestructureScratch {
    fn prepare(&mut self) {
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

    fn retain(&mut self, lim: &crate::limits::Limits) {
        self.per_v_pairs.truncate(PER_V_PAIRS_RETAIN);
        if self.packed.capacity() > RESTRUCTURE_PACKED_CAP_LIMIT {
            // Only the two size-proportional Vecs were charged through the
            // byte meter; the maps grow through their own allocator.
            let freed = self.packed.capacity() * std::mem::size_of::<u128>()
                + self.group_info.capacity() * std::mem::size_of::<super::relevel::PairGroup>();
            self.packed = Vec::new();
            self.group_info = Vec::new();
            self.per_v_pairs = Vec::new();
            self.inner_pair_to_idx = FxHashMap::default();
            self.distinct_inner = FxHashSet::default();
            lim.release_bytes(freed as u64);
        }
    }
}

/// Maximum retained `packed` capacity (4M triples × 16 B = 64 MB). A rare wide
/// rotation search must not park its peak buffers in the pool for the rest of
/// the process; past this, the size-proportional buffers are released and the
/// next take starts from empty.
const RESTRUCTURE_PACKED_CAP_LIMIT: usize = 4_000_000;

/// Maximum number of per-v-node pair lists carried across calls. The take-side
/// `clear()` walks the whole outer Vec, so an unbounded one would tax every
/// later (small) search with the widest level this engine ever saw — the pool
/// must not turn one wide rotation into a permanent per-call O(width) sweep.
/// Beyond this the tail is dropped; `restructure_inner_search` re-grows it with
/// `resize_with` exactly as it does on a cold scratch.
const PER_V_PAIRS_RETAIN: usize = 1024;

/// Scratch entries kept across probes. At its last read a buffer this size or
/// smaller is only `clear()`-ed, so the next probe reuses the allocation — the
/// churn-avoidance the scratch exists for. A larger one is released outright:
/// holding a high-water buffer across the successor-level builds, which
/// allocate their own copy of the same data, is what sets this function's peak,
/// and regrowing it costs one pass next to the sort that dominates a probe big
/// enough to be over the threshold. Bail-out probes return before any release
/// point, so the search's common path keeps full capacity either way.
pub(super) const SCRATCH_RETAIN_ENTRIES: usize = 1 << 16;

#[cfg(test)]
#[path = "tests/scratch.rs"]
mod tests;
