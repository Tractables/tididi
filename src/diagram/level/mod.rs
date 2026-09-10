//! The `TddLevel` structure, its state predicates, and its size accessors.

mod arena;
mod marginal;
mod pairs;
pub(crate) use pairs::sort_pairs;

use super::marginal_ref::{BigSide, SideView};
use super::primitives::{MultiPairRange, InputPair, NodeIdx, TddNodeData};

/// The nodes of one vtree node's level.
///
/// A level is in one of three states, and a reader checks them in this order:
///
/// - the vtree node is a leaf: `nodes` is empty and the three nodes are
///   implicit (see [`LeafLabel`](super::LeafLabel));
/// - [`is_marginal`](Self::is_marginal): it stores no nodes;
///   [`marginal_counts`](Self::marginal_counts)`[i]` is the model count of
///   node `i`, with `u128::MAX` meaning "exceeds `u128`, read
///   [`marginal_counts_big`](Self::marginal_counts_big)`.get(i)`";
/// - otherwise structural: [`nodes`](Self::nodes)`[i]` is node `i`, and its
///   pairs are [`pairs_of`](Self::pairs_of) of that slot. A node may be a
///   tombstone (dead, unreferenced); [`internal_inputs_iter`] skips those.
///
/// `width()` is the number of node slots in any state; `live_width()` excludes
/// tombstones.
///
/// [`internal_inputs_iter`]: Self::internal_inputs_iter
#[derive(Clone, Debug)]
pub struct TddLevel {
    /// The stored nodes, indexed by [`NodeIdx`]. Empty on leaf and
    /// marginal levels. Read from outside the crate through
    /// [`nodes`](Self::nodes) / [`slots_iter`](Self::slots_iter).
    pub(crate) nodes: Vec<TddNodeData>,
    /// Arena holding the pairs of multi-pair nodes. Read it through
    /// [`pairs_of`](Self::pairs_of); single-pair nodes are not in it.
    pub(crate) pairs: Vec<InputPair>,
    /// Side table for multi-pair nodes whose arena start or length exceeds
    /// 2^31 (huge product grids). See `TddNodeData` for the encoding.
    pub(crate) multi_pairs: Vec<MultiPairRange>,
    /// Which of this level's pair-side fields already hold INLINE MODEL COUNTS
    /// toward a marginal child, rather than fresh slot indices: bit 0 the left
    /// side, bit 1 the right.
    ///
    /// A boundary parent is structural, so this sits outside
    /// [`LevelState`] — a level carries it while it still has pairs. Set in two
    /// places: the apply pass-through path, which carries an already-inlined
    /// carrier field through verbatim, and the end-of-apply tagger after it
    /// emits a side. Those two readers treat it as a skip hint, never a
    /// correctness requirement: `emit_or_tag` returns an already-inline ref
    /// unchanged.
    ///
    /// The third reader is why these markers cannot simply go away: pair
    /// fusion routes a boundary whose EXPLICIT side carries an inline ref to
    /// its hashmap, because the dense scatter sizes its tables to the largest
    /// key it sees and an inline ref's tag bit puts that key past 2^30.
    ///
    /// Reset by [`clear`](Self::clear) and by marginalization, which leaves no
    /// pairs to describe.
    pub(crate) inlined_sides: u8,
    /// Number of tombstone slots in `nodes` — dead nodes the index-stable
    /// conjoin leaves in place instead of compacting out. 0 on the
    /// dense path. `width()` still counts every slot (it is the index bound for
    /// flat-array allocation); `live_width()` subtracts this. Reset to 0 by
    /// `clear()` and after prune compaction (which physically removes them).
    pub(crate) n_tombstones: u32,
    /// Slots in `pairs` that no live node references any more.
    ///
    /// Twin contraction mints these: a merged union is appended at the arena
    /// tail (`concat_twin_pairs`), abandoning every source range, and the parent
    /// rewrite / duplicate resolution shrink pair lists in place, abandoning
    /// their tails. `compact_pairs_if_stale`
    /// reclaims them and resets this to 0.
    ///
    /// Approximate by design — it is only the sweep TRIGGER, so an over- or
    /// under-count shifts *when* the sweep runs, never which bytes it moves
    /// (the sweep derives liveness from `nodes`/`multi_pairs`, not from this counter).
    /// Not every garbage source feeds it: prune drops a node without accounting
    /// its range (see `prune.rs`), so prune-only garbage waits for a
    /// contraction-triggered sweep instead of triggering one. Reset to 0
    /// wherever the pair arena is replaced or dropped wholesale — an
    /// enumeration here would rot; the sites are grep-able as `dead_pairs = 0`.
    pub(crate) dead_pairs: u32,
    /// Whether this level still denotes its functions structurally, and if not,
    /// which values it holds instead.
    pub(crate) state: LevelState,
}

/// What a level holds in place of its structure, once it has been
/// marginalized — and [`Structural`](LevelState::Structural) while it still
/// holds nodes and pairs.
///
/// The two valued arms are exclusive by construction, which is what this type
/// buys: the integer path's width carrier is its own `counts` vector, the
/// weighted path's values live in the external
/// [`WeightStore`](crate::diagram::WeightStore) and only the slot count stays
/// here. A level cannot be in both at once, and nothing has to encode "0 on
/// every other level".
#[derive(Clone, Debug)]
pub(crate) enum LevelState {
    /// Nodes and pairs; `nodes`/`pairs`/`multi_pairs` carry the level.
    Structural,
    /// Model counts, one per node slot. `u128::MAX` marks a count that exceeds
    /// `u128`, whose exact value is the entry `big` holds for that slot.
    Counts {
        counts: Vec<u128>,
        big: Option<BigSide>,
        retired: u32,
    },
    /// Semiring weights, held in the external `WeightStore` and indexed by this
    /// level's slot. Only the slot count stays here — `width()` has nowhere
    /// else to read it from, since `nodes` is cleared like the integer path.
    Weights { width: u32, retired: u32 },
}

/// What a level stores. See [`TddLevel::kind`].
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum LevelKind {
    /// Nodes and pairs: the level denotes functions structurally.
    Structural,
    /// A vtree leaf: three implicit nodes over one variable, nothing stored.
    Leaf,
    /// Marginalized: per-node values in place of structure.
    Marginal(ValueKind),
}

/// The arithmetic a [`LevelKind::Marginal`] level's values take.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum ValueKind {
    /// Model counts, in the level's own `marginal_counts`.
    Counts,
    /// Semiring weights, in the external
    /// [`WeightStore`](crate::diagram::WeightStore).
    Weights,
}

/// `TddLevel` should stay compact — the hot sequential-scan stride depends on
/// it. The per-node / per-pair minimize loops iterate a level's *heap-backed*
/// `nodes`/`pairs` arenas, not the `TddLevel` structs themselves, so only the
/// O(levels) sweeps (shrink, `node_count`) see the stride.
const _: () = assert!(
    std::mem::size_of::<TddLevel>() <= 144,
    "TddLevel grew past 144 B"
);

impl Default for TddLevel {
    /// Returns an empty level, identical to [`TddLevel::new`].
    fn default() -> Self {
        Self::new()
    }
}

impl TddLevel {
    /// Bit positions in `inlined_sides`. See the field doc.
    pub(crate) const MARGINAL_INLINED_LEFT: u8 = 1 << 0;
    /// Right marginal-child of a boundary parent is inline-encoded in the pair field
    /// (companion of [`MARGINAL_INLINED_LEFT`](Self::MARGINAL_INLINED_LEFT)).
    pub(crate) const MARGINAL_INLINED_RIGHT: u8 = 1 << 1;

    /// True if the left marginal-child inline-encoding flag is set.
    #[inline(always)]
    pub(crate) fn marginal_inlined_left(&self) -> bool {
        self.inlined_sides & Self::MARGINAL_INLINED_LEFT != 0
    }
    /// True if the right marginal-child inline-encoding flag is set.
    #[inline(always)]
    pub(crate) fn marginal_inlined_right(&self) -> bool {
        self.inlined_sides & Self::MARGINAL_INLINED_RIGHT != 0
    }
    /// Set or clear the left marginal-child inline-encoding flag.
    #[inline(always)]
    pub(crate) fn set_marginal_inlined_left(&mut self, v: bool) {
        if v { self.inlined_sides |= Self::MARGINAL_INLINED_LEFT }
        else { self.inlined_sides &= !Self::MARGINAL_INLINED_LEFT }
    }
    /// Set or clear the right marginal-child inline-encoding flag.
    #[inline(always)]
    pub(crate) fn set_marginal_inlined_right(&mut self, v: bool) {
        if v { self.inlined_sides |= Self::MARGINAL_INLINED_RIGHT }
        else { self.inlined_sides &= !Self::MARGINAL_INLINED_RIGHT }
    }
    /// True if either side carries the inline-encoding marker. A level with
    /// neither is "plain": every pair side toward a marginal child is a bare
    /// slot, so duplicate pairs cannot be count-carrying multiset entries.
    #[inline(always)]
    pub(crate) fn any_inlined_side(&self) -> bool {
        self.inlined_sides != 0
    }

    /// An empty level: the state of every leaf level, and the starting point
    /// for building a structural level with [`push_internal_node`](Self::push_internal_node).
    pub fn new() -> Self {
        TddLevel {
            nodes: Vec::new(),
            pairs: Vec::new(),
            multi_pairs: Vec::new(),
            inlined_sides: 0,
            n_tombstones: 0,
            dead_pairs: 0,
            state: LevelState::Structural,
        }
    }

    /// Reset to empty (as [`new`](Self::new)), keeping buffer capacity.
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.pairs.clear();
        self.multi_pairs.clear();
        self.inlined_sides = 0;
        self.n_tombstones = 0;
        self.dead_pairs = 0;
        self.state = LevelState::Structural;
    }

    /// Release the structural arenas and zero the counters that describe them.
    ///
    /// The shared teardown for the marginal transitions: a level whose values
    /// have moved into counts or into the weight store has no nodes, no pairs
    /// and no side table, so every counter over them (tombstones, dead arena
    /// slots, inline-emit markers) describes storage that is gone. The pages go
    /// back to the allocator rather than staying as capacity — a marginal level
    /// never grows structure again. The caller writes the new state.
    ///
    /// Not [`clear`](Self::clear): that one keeps the capacity for a level
    /// about to be rebuilt.
    fn drop_structure(&mut self) {
        self.nodes.clear();
        self.nodes.shrink_to_fit();
        self.pairs.clear();
        self.pairs.shrink_to_fit();
        self.multi_pairs.clear();
        self.multi_pairs.shrink_to_fit();
        self.n_tombstones = 0;
        self.dead_pairs = 0;
        self.inlined_sides = 0;
    }

    /// Number of node slots: `marginal_counts.len()` on a marginal level,
    /// else `nodes.len()` (live and tombstone). The index bound for arrays
    /// over this level; use [`live_width`](Self::live_width) to count nodes.
    /// 0 on a leaf level (its nodes are implicit).
    pub fn width(&self) -> usize {
        match &self.state {
            LevelState::Counts { counts, .. } => counts.len(),
            LevelState::Weights { width, .. } => *width as usize,
            LevelState::Structural => self.nodes.len(),
        }
    }

    /// `width()` minus tombstone slots — the number of nodes.
    pub fn live_width(&self) -> usize {
        self.width() - self.n_tombstones as usize
    }

    /// The node slots of a structural level, in index order — tombstones
    /// included, so slot `i` is `slots()[i]`. Empty on a leaf or marginal
    /// level, which store no nodes.
    #[inline]
    pub fn nodes(&self) -> &[TddNodeData] {
        &self.nodes
    }

    /// [`nodes`](Self::nodes) paired with each slot's index.
    ///
    /// Tombstones are yielded like any other slot; skip them with
    /// [`TddNodeData::is_tombstone`], or walk
    /// [`internal_inputs_iter`](Self::internal_inputs_iter) instead, which
    /// yields only live nodes with their pairs.
    #[inline]
    pub fn nodes_iter(&self) -> impl Iterator<Item = (NodeIdx, &TddNodeData)> {
        self.nodes.iter().enumerate().map(|(i, n)| (NodeIdx(i as u32), n))
    }

    /// Reserve room for `additional` more node slots.
    #[inline]
    pub fn reserve_nodes(&mut self, additional: usize) {
        self.nodes.reserve(additional);
    }

    /// The model count of each node of a marginal level, indexed by
    /// [`NodeIdx`]; `None` on any other level, and on a weight-marginal one
    /// (whose values live in the [`WeightStore`](crate::diagram::WeightStore)).
    ///
    /// A value of `u128::MAX` means the count exceeds `u128` and the exact one
    /// is [`marginal_counts_big`](Self::marginal_counts_big)`.get(i)`.
    #[inline]
    pub fn marginal_counts(&self) -> Option<&[u128]> {
        match &self.state {
            LevelState::Counts { counts, .. } => Some(counts),
            _ => None,
        }
    }

    /// The counts of a count-marginal level, to be written in place — slot
    /// pruning and pair fusion rewrite them without changing the level's state.
    #[inline]
    pub(crate) fn marginal_counts_mut(&mut self) -> Option<&mut Vec<u128>> {
        match &mut self.state {
            LevelState::Counts { counts, .. } => Some(counts),
            _ => None,
        }
    }

    /// Both halves of a count-marginal level's store at once: the fast column
    /// and the overflow table, which the compaction passes rewrite together.
    #[inline]
    pub(crate) fn marginal_store_mut(&mut self) -> Option<(&mut Vec<u128>, &mut Option<BigSide>)> {
        match &mut self.state {
            LevelState::Counts { counts, big, .. } => Some((counts, big)),
            _ => None,
        }
    }

    /// The exact values of the [`marginal_counts`](Self::marginal_counts)
    /// slots that hold `u128::MAX`. `None` and an empty table both mean no
    /// slot overflowed.
    #[inline]
    pub fn marginal_counts_big(&self) -> Option<&BigSide> {
        match &self.state {
            LevelState::Counts { big, .. } => big.as_ref(),
            _ => None,
        }
    }

    /// Slots this level's marginal store has retired: freed by
    /// `prune_value_slots` (deep clears plus boundary compaction). A METRIC,
    /// never a width. Monotone per level, reset only by [`clear`](Self::clear)
    /// and by a fresh marginalization, and it travels with the level through
    /// `mem::swap`, so the sum over levels (`Tdd::retired_marginal_slots`)
    /// follows the same lineage as `node_count()`. A consumer offsets a size
    /// threshold by the difference between two readings, so that slot-pruning
    /// does not deflate the measured size; `node_count()` itself stays the
    /// surviving-node count. 0 on a structural level.
    #[inline]
    pub(crate) fn retired_marginal_slots(&self) -> u32 {
        match &self.state {
            LevelState::Counts { retired, .. } | LevelState::Weights { retired, .. } => *retired,
            LevelState::Structural => 0,
        }
    }

    /// Account `n` more retired slots. A structural level retires nothing and
    /// silently ignores the call — it has no store to free from.
    #[inline]
    pub(crate) fn retire_marginal_slots(&mut self, n: u32) {
        match &mut self.state {
            LevelState::Counts { retired, .. } | LevelState::Weights { retired, .. } => {
                *retired = retired.saturating_add(n)
            }
            LevelState::Structural => {}
        }
    }

    /// The live slot count of a weight-marginal level, 0 elsewhere. `width()`
    /// reads it back; a caller that mints a slot bumps it through
    /// [`set_weight_width`](Self::set_weight_width).
    #[inline]
    pub(crate) fn weight_width(&self) -> u32 {
        match &self.state {
            LevelState::Weights { width, .. } => *width,
            _ => 0,
        }
    }

    /// Put this level into its counts state without touching the arenas, so a
    /// test can build a level whose shape `become_marginal` would have thrown
    /// away — including one an invariant check is supposed to reject.
    #[cfg(test)]
    pub(crate) fn set_counts_state(&mut self, counts: Vec<u128>, big: Option<BigSide>) {
        self.state = LevelState::Counts { counts, big, retired: 0 };
    }

    /// Heap the fast count column has reserved, 0 when the level holds no
    /// counts. A level that has finished with its store should own none —
    /// releasing the pages is the point of clearing it.
    #[inline]
    pub(crate) fn value_store_capacity(&self) -> usize {
        match &self.state {
            LevelState::Counts { counts, .. } => counts.capacity(),
            _ => 0,
        }
    }

    /// Drop the overflow table of a count-marginal level: every slot's exact
    /// value now fits the fast column. No-op elsewhere.
    #[inline]
    pub(crate) fn clear_marginal_big(&mut self) {
        if let LevelState::Counts { big, .. } = &mut self.state {
            *big = None;
        }
    }

    /// Set the live slot count of a weight-marginal level. No-op elsewhere.
    #[inline]
    pub(crate) fn set_weight_width(&mut self, w: u32) {
        if let LevelState::Weights { width, .. } = &mut self.state {
            *width = w;
        }
    }

    /// True if any node has more than one pair. O(width).
    #[inline]
    pub fn has_multi_pair(&self) -> bool {
        (0..self.nodes.len()).any(|i| {
            self.nodes[i].is_multi() && self.multi_len_at(i) >= 2
        })
    }

    /// What this level stores, as one value — the discriminator a reader
    /// wants when it needs exactly one of "does this hold values", "are refs
    /// to it [`ValueRef`](super::ValueRef)s", "which arithmetic do its values
    /// take".
    ///
    /// A level cannot tell a leaf from an empty structural level on its own —
    /// that is a fact about the vtree — so this never returns
    /// [`LevelKind::Leaf`]; [`Tdd::level_kind`](super::Tdd::level_kind) does,
    /// having the vtree at hand.
    #[inline]
    pub fn kind(&self) -> LevelKind {
        match &self.state {
            LevelState::Counts { .. } => LevelKind::Marginal(ValueKind::Counts),
            LevelState::Weights { .. } => LevelKind::Marginal(ValueKind::Weights),
            LevelState::Structural => LevelKind::Structural,
        }
    }

    /// How to read the pair sides of a parent that point at THIS level.
    ///
    /// Build it once per level visit and decode every side through it; see
    /// [`SideView`].
    #[inline]
    pub fn side_view(&self) -> SideView {
        if self.is_marginal() { SideView::marginal() } else { SideView::structural() }
    }

    /// True if this level has dropped its structure for per-node values —
    /// [`LevelKind::Marginal`] under either arithmetic.
    #[inline(always)]
    pub fn is_marginal(&self) -> bool {
        !matches!(self.state, LevelState::Structural)
    }

    /// True if this level is marginal with its per-node values held in an
    /// external [`WeightStore`](crate::diagram::WeightStore) rather
    /// than in `marginal_counts` (which stays `None`). Such a level's values
    /// cannot be read from the diagram alone.
    #[inline(always)]
    pub fn is_weight_marginal(&self) -> bool {
        matches!(self.state, LevelState::Weights { .. })
    }


    /// Trim retained slack in `nodes`, `pairs`, and `multi_pairs` when capacity exceeds
    /// 4× length AND absolute capacity is ≥ 1 Ki slots. Called
    /// after a level is finalized in apply to release the Vec-doubling
    /// overshoot from the per-cell `try_push` emit loop, and by
    /// [`compact_pairs_if_stale`](Self::compact_pairs_if_stale) once it has
    /// truncated the pairs arena — this is the level's one decision about
    /// returning slack to the allocator. The ratio trades
    /// peak savings against realloc-copies on hot levels that get re-grown
    /// soon. Marginal levels (already shrunk by `become_marginal`) are skipped.
    #[inline]
    pub(crate) fn shrink_arrays(&mut self) {
        if matches!(self.state, LevelState::Counts { .. }) {
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
        if should_shrink(self.multi_pairs.capacity(), self.multi_pairs.len()) {
            self.multi_pairs.shrink_to_fit();
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
