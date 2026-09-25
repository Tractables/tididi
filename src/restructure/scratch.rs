//! The reusable rotation-probe scratch and the engine's pool for it.

use crate::limits::Limits;
use crate::execution::pool::{Buffers, Nested, PooledScratch, Scratch};

use rustc_hash::FxHashMap;

use crate::diagram::{ChildPair, NodeIdx};

/// Reusable scratch for `rebuild_rotated_levels`. Threaded by
/// the rotation-search loops so the per-probe allocator churn is paid once per
/// search rather than once per probe. Each field is `clear()`-ed before use in
/// `rebuild_rotated_levels`, preserving the underlying capacity; a field is
/// then released at its last read within the call rather than held across the
/// successor-level builds (see `SCRATCH_RETAIN_ENTRIES`). Every field grows
/// through the engine's limits, so its growth is charged to the operation's
/// byte meter, and what the retention policy drops is given back there.
#[derive(Default)]
pub(crate) struct RestructureScratch {
    pub(super) inner_pair_to_idx: FxHashMap<ChildPair, NodeIdx>,
    // Per-v-node output pair lists; outer Vec grown with `resize_with`, inner
    // Vecs `clear()`-ed per call so their capacity survives across probes.
    pub(super) per_v_pairs: Vec<Vec<ChildPair>>,
    pub(super) group_info: Vec<super::relevel::PairGroup>,
    pub(super) bucket: BucketScratch,
    // Search path triples, packed one-per-u128 (see `pack_triple`). The sort in
    // `rebuild_rotated_levels` is the dominant cost of the joint next-merge-cost
    // probe on single-large-component pools; sorting a `Vec<u128>` by a single
    // integer key replaces the derived lexicographic compare over the
    // `(ChildPair, u32, NodeIdx)` tuple's four u32 fields.
    pub(super) packed: Vec<u128>,
}

/// The walk over one fingerprint bucket in the clustering build of the inner
/// level: which of the bucket's groups are already placed under a node, and
/// the pairs of the node being emitted.
#[derive(Default)]
pub(super) struct BucketScratch {
    pub(super) done: Vec<bool>,
    pub(super) pairs: Vec<ChildPair>,
}

impl Buffers for RestructureScratch {
    fn buffers(&mut self, visit: &mut dyn FnMut(&mut dyn Scratch)) {
        visit(&mut self.inner_pair_to_idx);
        visit(&mut Nested(&mut self.per_v_pairs));
        visit(&mut self.group_info);
        visit(&mut self.packed);
        visit(&mut self.bucket.done);
        visit(&mut self.bucket.pairs);
    }
}

impl PooledScratch for RestructureScratch {
    fn prepare(&mut self) {
        self.inner_pair_to_idx.clear();
        self.group_info.clear();
        self.packed.clear();
        self.bucket.done.clear();
        self.bucket.pairs.clear();
        // Keep the outer Vec's length (bounded by `PER_V_PAIRS_RETAIN` on
        // return): `rebuild_rotated_levels` only `resize_with`s it upward and
        // clears the prefix it uses, so the inner Vecs' capacities are exactly
        // what we want to carry forward.
        for v in &mut self.per_v_pairs {
            v.clear();
        }
    }

    /// A search whose `packed` outgrew its cap releases every buffer; any
    /// other keeps each buffer the byte cap allows.
    fn retain(&mut self, lim: &Limits) {
        if self.per_v_pairs.len() > PER_V_PAIRS_RETAIN {
            for lists in self.per_v_pairs.drain(PER_V_PAIRS_RETAIN..) {
                lim.discard(lists);
            }
        }
        if self.packed.capacity() > RESTRUCTURE_PACKED_CAP_LIMIT {
            self.release_all(lim);
        } else {
            self.release_oversized(lim);
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
/// Beyond this the tail is dropped; `rebuild_rotated_levels` re-grows it with
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

/// Retire a buffer at its last read within a probe: past
/// [`SCRATCH_RETAIN_ENTRIES`] entries of capacity the allocation goes back to
/// `lim`, otherwise the buffer is emptied and stays warm for the next probe.
///
/// A struct field read in place has to be emptied here, or the next probe
/// would see this one's contents; releasing leaves it empty as well.
pub(super) fn release_or_clear<T>(lim: &Limits, buf: &mut Vec<T>) {
    if buf.capacity() > SCRATCH_RETAIN_ENTRIES {
        lim.discard(std::mem::take(buf));
    } else {
        buf.clear();
    }
}

#[cfg(test)]
#[path = "tests/scratch.rs"]
mod tests;
