//! The `TddLevel` structure, its state predicates, and its size accessors.

mod arena;
mod marginal;
mod pairs;

use super::marg::BigSide;
use super::primitives::{ExtMulti, InputPair, TddNodeData};

/// The nodes of one vtree node's level.
///
/// A level is in one of three states, and a reader checks them in this order:
///
/// - the vtree node is a leaf: `nodes` is empty and the three nodes are
///   implicit (see [`LeafLabel`](super::LeafLabel));
/// - [`is_marginal`](Self::is_marginal): `nodes` and `pairs` are empty;
///   `marginal_counts[i]` is the model count of node `i`, with `u128::MAX`
///   meaning "exceeds `u128`, read `marginal_counts_big.get(i)`";
/// - otherwise structural: `nodes[i]` is node `i`, and its pairs are
///   [`pairs_of`](Self::pairs_of)`(&nodes[i])`. Never index `pairs` directly —
///   a single-pair node keeps its pair in the node word, not in the arena. A
///   node may be a tombstone (dead, unreferenced); [`internal_inputs_iter`]
///   skips those.
///
/// `width()` is the number of node slots in any state; `live_width()` excludes
/// tombstones.
///
/// [`internal_inputs_iter`]: Self::internal_inputs_iter
#[derive(Clone, Debug)]
pub struct TddLevel {
    /// The stored nodes, indexed by [`LocalNodeIdx`]. Empty on leaf and
    /// marginal levels.
    pub nodes: Vec<TddNodeData>,
    /// Arena holding the pairs of multi-pair nodes. Read it through
    /// [`pairs_of`](Self::pairs_of); single-pair nodes are not in it.
    pub pairs: Vec<InputPair>,
    /// Side table for multi-pair nodes whose arena start or length exceeds
    /// 2^31 (huge product grids). See `TddNodeData` for the encoding.
    pub(crate) ext: Vec<ExtMulti>,
    /// Marginal-ref state flags, packed. `marg_inlined_left` (bit 0): this
    /// level's `pairs[*].left` fields toward a marginal left child already hold
    /// INLINE MODEL COUNTS (bit-30 clear bare values), NOT fresh slot indices.
    /// Set in two places: (1) the apply pass-through path, which carries an
    /// already-inlined carrier field through verbatim; (2) the end-of-apply
    /// tagger after it emits this side. Checked by the tagger's emit arm to
    /// SKIP re-emitting (re-running `emit_or_tag` on an inline count would
    /// misread it as a slot → `counts[C]` corruption). The non-emit tag path
    /// (`| TAG`, idempotent) ignores this marker. Reset by `clear()` and
    /// `make_marginal`. `marg_inlined_right` (bit 1) mirrors for the right side.
    /// (Bit 2 is free.)
    ///
    /// Packed as a bitfield (not separate `bool`s) so the flags + `n_tombstones`
    /// fit `TddLevel`'s padding without crossing the 128 B / 2-cache-line
    /// boundary (see the size assert below). Access via the
    /// `marg_inlined_left()`/`set_marg_inlined_left(..)` style methods.
    pub(crate) marg_flags: u8,
    /// Number of tombstone slots in `nodes` — dead nodes the index-stable
    /// conjoin (Tier 2) leaves in place instead of compacting out. 0 on the
    /// dense path. `width()` still counts every slot (it is the index bound for
    /// flat-array allocation); `live_width()` subtracts this. Reset to 0 by
    /// `clear()` and after prune compaction (which physically removes them).
    pub(crate) n_tombstones: u32,
    /// TWO meanings, decided by the level:
    ///
    /// - On a **weight-marginal** level this is the level's live WIDTH — its
    ///   slot count, set by `make_marginal_weighted`. `nodes` is cleared and
    ///   `marginal_counts` is `None` there, so [`width`](Self::width) has
    ///   nowhere else to read it from. Dropping the field would break `width()`.
    /// - On every other level it is a METRIC: slots freed from this level's
    ///   marginal store by `prune_marg_slots` (deep clears plus boundary
    ///   compaction). Monotone per level, reset only by `clear()`, and it
    ///   travels with the level through `mem::swap`, so the sum over levels
    ///   ([`Tdd::retired_marg_total`]) follows the same lineage as
    ///   `total_nodes()`. A consumer offsets a size threshold by the difference
    ///   between two readings, so that slot-pruning does not deflate the
    ///   measured size; `total_nodes()` itself stays the surviving-node count.
    ///
    /// The two never overlap — slot-prune exempts weight-marginal leaves — but
    /// the overload is real, and reading the field without knowing which level
    /// it belongs to means nothing.
    pub(crate) retired_marg_width: u32,
    /// Slots in `pairs` that no live node references any more.
    ///
    /// Twin contraction mints these: a merged union is appended at the arena
    /// tail (`concat_twin_pairs`), abandoning every source range, and the parent
    /// rewrite / duplicate resolution shrink pair lists in place, abandoning
    /// their tails. `compact_pairs_if_stale`
    /// reclaims them and resets this to 0.
    ///
    /// APPROXIMATE by design — it is only the sweep TRIGGER, so an over- or
    /// under-count shifts *when* the sweep runs, never which bytes it moves
    /// (the sweep derives liveness from `nodes`/`ext`, not from this counter).
    /// Not every garbage source feeds it: prune drops a node without accounting
    /// its range (see `prune.rs`), so prune-only garbage waits for a
    /// contraction-triggered sweep instead of triggering one. Reset to 0
    /// wherever the pair arena is replaced or dropped wholesale — an
    /// enumeration here would rot; the sites are grep-able as `dead_pairs = 0`.
    pub(crate) dead_pairs: u32,
    /// `Some` on a marginal level: the model count of each node, indexed by
    /// [`LocalNodeIdx`]. `nodes` and `pairs` are then empty and
    /// `width()` is `marginal_counts.len()`. A value of `u128::MAX` means the
    /// count exceeds `u128`; the exact value is `marginal_counts_big.get(i)`.
    pub marginal_counts: Option<Vec<u128>>,
    /// Exact values of the `marginal_counts` slots that hold `u128::MAX`,
    /// keyed by the same index. `None` and an empty table both mean no slot
    /// overflowed.
    pub marginal_counts_big: Option<BigSide>,
}

/// `TddLevel` should stay compact — the hot sequential-scan stride depends on it.
/// `n_tombstones: u32` was placed among the bool fields to fit existing padding,
/// as was `dead_pairs: u32` (the pairs-arena garbage counter).
/// `retired_marg_width: u32` is the retire counter / weight-marginal width. The hot
/// per-node / per-pair minimize loops
/// iterate a level's *heap-backed* `nodes`/`pairs` arenas, not the `TddLevel`
/// structs themselves, so only the O(levels) sweeps (shrink, `total_nodes`)
/// see the stride.
const _: () = assert!(
    std::mem::size_of::<TddLevel>() <= 160,
    "TddLevel grew past 160 B"
);

impl Default for TddLevel {
    /// Returns an empty level, identical to [`TddLevel::new`].
    fn default() -> Self {
        Self::new()
    }
}

impl TddLevel {
    /// Bit positions in `marg_flags`. See the field doc.
    pub(crate) const MARG_INLINED_LEFT: u8 = 1 << 0;
    /// Right marg-child of a boundary parent is inline-encoded in the pair field
    /// (companion of [`MARG_INLINED_LEFT`](Self::MARG_INLINED_LEFT)).
    pub(crate) const MARG_INLINED_RIGHT: u8 = 1 << 1;
    // bit 2 free.
    /// Weighted/algebraic marginalization: this level has been marginalized in
    /// weighted mode. Its per-node semiring values live in the external
    /// `WeightStore` side-table (indexed by vtree level), NOT in `marginal_counts`
    /// (which stays `None`). Keeps `TddLevel` within its size budget — adding a
    /// `Vec<BigRational>` field would overflow it. Never set on the integer
    /// path, so `is_marginal()` stays byte-identical there.
    pub(crate) const MARG_WEIGHTED: u8 = 1 << 3;

    /// True if the left marg-child inline-encoding flag is set.
    #[inline(always)]
    pub(crate) fn marg_inlined_left(&self) -> bool {
        self.marg_flags & Self::MARG_INLINED_LEFT != 0
    }
    /// True if the right marg-child inline-encoding flag is set.
    #[inline(always)]
    pub(crate) fn marg_inlined_right(&self) -> bool {
        self.marg_flags & Self::MARG_INLINED_RIGHT != 0
    }
    /// Set or clear the left marg-child inline-encoding flag.
    #[inline(always)]
    pub(crate) fn set_marg_inlined_left(&mut self, v: bool) {
        if v { self.marg_flags |= Self::MARG_INLINED_LEFT }
        else { self.marg_flags &= !Self::MARG_INLINED_LEFT }
    }
    /// Set or clear the right marg-child inline-encoding flag.
    #[inline(always)]
    pub(crate) fn set_marg_inlined_right(&mut self, v: bool) {
        if v { self.marg_flags |= Self::MARG_INLINED_RIGHT }
        else { self.marg_flags &= !Self::MARG_INLINED_RIGHT }
    }

    /// An empty level: the state of every leaf level, and the starting point
    /// for building a structural level with [`push_internal_node`](Self::push_internal_node).
    pub fn new() -> Self {
        TddLevel {
            nodes: Vec::new(),
            pairs: Vec::new(),
            ext: Vec::new(),
            marg_flags: 0,
            n_tombstones: 0,
            retired_marg_width: 0,
            dead_pairs: 0,
            marginal_counts: None,
            marginal_counts_big: None,
        }
    }

    /// Reset to empty (as [`new`](Self::new)), keeping buffer capacity.
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.pairs.clear();
        self.ext.clear();
        self.marg_flags = 0;
        self.n_tombstones = 0;
        self.retired_marg_width = 0;
        self.dead_pairs = 0;
        self.marginal_counts = None;
        self.marginal_counts_big = None;
    }

    /// Number of node slots: `marginal_counts.len()` on a marginal level,
    /// else `nodes.len()` (live and tombstone). The index bound for arrays
    /// over this level; use [`live_width`](Self::live_width) to count nodes.
    /// 0 on a leaf level (its nodes are implicit).
    pub fn width(&self) -> usize {
        if let Some(counts) = &self.marginal_counts {
            counts.len()
        } else if self.is_weight_marginal() {
            // Nodes are cleared on weight-marginal levels; the slot count lives
            // in `retired_marg_width` (set by `make_marginal_weighted`).
            self.retired_marg_width as usize
        } else {
            self.nodes.len()
        }
    }

    /// `width()` minus tombstone slots — the number of nodes.
    pub fn live_width(&self) -> usize {
        self.width() - self.n_tombstones as usize
    }

    /// True if any node has more than one pair. O(width).
    #[inline]
    pub fn has_multi_pair(&self) -> bool {
        (0..self.nodes.len()).any(|i| {
            self.nodes[i].is_multi() && self.multi_len_at(i) >= 2
        })
    }

    /// True if this level has dropped its structure for per-node counts.
    /// `marginal_counts` is `Some` unless the level is weight-marginal.
    pub fn is_marginal(&self) -> bool {
        self.marginal_counts.is_some() || self.is_weight_marginal()
    }

    /// True if this level is marginal with its per-node values held in an
    /// external [`WeightStore`](crate::tdd::weight_store::WeightStore) rather
    /// than in `marginal_counts` (which stays `None`). Such a level's values
    /// cannot be read from the diagram alone.
    #[inline(always)]
    pub fn is_weight_marginal(&self) -> bool {
        self.marg_flags & Self::MARG_WEIGHTED != 0
    }


    /// Trim retained slack in `nodes`, `pairs`, and `ext` when capacity exceeds
    /// 4× length AND absolute capacity is ≥ 1 Ki slots. Called
    /// after a level is finalized in apply to release the Vec-doubling
    /// overshoot from the per-cell `try_push` emit loop, and by
    /// [`compact_pairs_if_stale`](Self::compact_pairs_if_stale) once it has
    /// truncated the pairs arena — this is the level's ONE decision about
    /// returning slack to the allocator. The ratio trades
    /// peak savings against realloc-copies on hot levels that get re-grown
    /// soon. Marginal levels (already shrunk by `make_marginal`) are skipped.
    #[inline]
    pub(crate) fn shrink_arrays(&mut self) {
        if self.marginal_counts.is_some() {
            return;
        }
        const MIN_SHRINK_CAP: usize = 1024;
        // cap > (32 / 8) * len  ⟺  8 * cap > 32 * len  ⟺  cap > 4 * len
        const SHRINK_RATIO_EIGHTHS: u64 = 32; // 4×
        let should_shrink = |cap: usize, len: usize| -> bool {
            cap >= MIN_SHRINK_CAP && (cap as u64) * 8 > SHRINK_RATIO_EIGHTHS * (len as u64)
        };
        if should_shrink(self.nodes.capacity(), self.nodes.len()) {
            self.nodes.shrink_to_fit();
        }
        if should_shrink(self.pairs.capacity(), self.pairs.len()) {
            self.pairs.shrink_to_fit();
        }
        if should_shrink(self.ext.capacity(), self.ext.len()) {
            self.ext.shrink_to_fit();
        }
    }

    /// Source-agnostic pair count. Use this for `pair_start` snapshots in
    /// the emit loop.
    #[inline]
    pub(crate) fn pair_count(&self) -> usize {
        self.pairs.len()
    }

    /// Source-agnostic pop, returns the last pair from the pairs arena.
    /// Used by `emit_product_node!`'s 1-pair inline path.
    #[inline]
    pub(crate) fn pop_pair(&mut self) -> Option<InputPair> {
        self.pairs.pop()
    }

    /// Length of the pair-arena tail starting at `start`.
    ///
    /// Pair lists are unordered sets and no operation requires a particular
    /// order (twin contraction is order-independent), so apply emit sites no
    /// longer sort the tail — they just count it via this. See the NOTE at the
    /// bottom of this file.
    #[inline]
    pub(crate) fn pair_tail_len(&self, start: usize) -> usize {
        self.pairs.len() - start
    }

}
