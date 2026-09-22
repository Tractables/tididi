//! The `TddLevel` structure, its state predicates, and its size accessors.

mod arena;
mod marginal;
mod pairs;
pub(crate) use pairs::sort_pairs;

use super::marginal_ref::{ChildDecoder, ChildSide, CountOverflow};
use super::primitives::{MultiPairRange, ChildPair, NodeIdx, EncodedNode};

/// The diagram storage associated with one vtree node.
///
/// A level is in one of three states, and a reader checks them in this order:
///
/// - the vtree node is a leaf: `nodes` is empty and the three nodes are
///   implicit (see [`LeafLabel`](super::LeafLabel));
/// - [`is_marginal`](Self::is_marginal): it stores no nodes;
///   [`marginal_counts`](Self::marginal_counts)`[i]` is the model count of
///   node `i`, with `u128::MAX` marking a value at least that large; read
///   [`marginal_counts_big`](Self::marginal_counts_big)`.get(i)` for the exact value. When
///   [`is_weight_marginal`](Self::is_weight_marginal) the counts are `None`
///   and value `i` is entry `i` of the diagram's
///   [`WeightStore::level`](crate::diagram::WeightStore::level) for this
///   level;
/// - otherwise structural: [`nodes`](Self::nodes)`[i]` is node `i`, and its
///   pairs are [`pairs_of`](Self::pairs_of) of that slot. A node may be a
///   tombstone (dead, unreferenced, [`EncodedNode::is_internal`] false);
///   [`internal_inputs_iter`] skips those.
///
/// `slot_count()` is the number of node slots in any state; `live_slot_count()` excludes
/// tombstones.
///
/// [`internal_inputs_iter`]: Self::internal_inputs_iter
#[derive(Clone, Debug)]
pub struct TddLevel {
    /// The stored nodes, indexed by [`NodeIdx`]. Empty on leaf and
    /// marginal levels. Read from outside the crate through
    /// [`nodes`](Self::nodes) / [`nodes_iter`](Self::nodes_iter).
    pub(crate) nodes: Vec<EncodedNode>,
    /// Arena holding the pairs of multi-pair nodes. Read it through
    /// [`pairs_of`](Self::pairs_of); single-pair nodes are not in it.
    pub(crate) pairs: Vec<ChildPair>,
    /// Side table for multi-pair nodes whose arena start or length exceeds
    /// 2^31 (huge product grids). See `EncodedNode` for the encoding.
    pub(crate) multi_pairs: Vec<MultiPairRange>,
    /// Which of this level's pair-side fields already hold inline model counts
    /// toward a marginal child, rather than fresh slot indices: bit 0 the left
    /// side, bit 1 the right.
    ///
    /// A boundary parent is structural, so this sits outside
    /// [`LevelState`]. Set by apply when it emits or carries through an inlined
    /// side; reset by [`clear`](Self::clear) and by marginalization.
    pub(crate) inlined_sides: u8,
    /// Number of tombstone slots in `nodes` — dead nodes the index-stable
    /// conjoin leaves in place instead of compacting out. 0 on the
    /// dense path. `slot_count()` still counts every slot (it is the index bound for
    /// flat-array allocation); `live_slot_count()` subtracts this. Reset to 0 by
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
    /// Approximate: it only triggers the sweep, which derives liveness from
    /// `nodes`/`multi_pairs`. Reset to 0 wherever the pair arena is replaced.
    pub(crate) dead_pairs: u32,
    /// Whether this level still denotes its functions structurally, and if not,
    /// which values it holds instead.
    pub(crate) state: LevelState,
}

/// What a level holds in place of its structure, once it has been
/// marginalized — and [`Structural`](LevelState::Structural) while it still
/// holds nodes and pairs.
///
/// The integer arm's width is its `counts` vector; the weighted arm's values
/// live in the external [`WeightStore`](crate::diagram::WeightStore) and only
/// the slot count stays here.
#[derive(Clone, Debug)]
pub(crate) enum LevelState {
    /// Nodes and pairs; `nodes`/`pairs`/`multi_pairs` carry the level.
    Structural,
    /// Model counts, one per node slot. `u128::MAX` marks a count at least
    /// that large, whose exact value is the entry `big` holds for that slot.
    Counts {
        counts: Vec<u128>,
        big: Option<CountOverflow>,
        retired: u32,
    },
    /// Semiring weights, held in the external `WeightStore` and indexed by this
    /// level's slot. Only the slot count stays here — `slot_count()` has nowhere
    /// else to read it from, since `nodes` is cleared like the integer path.
    Weights { width: u32, retired: u32 },
}

/// `TddLevel` stays compact: the O(levels) sweeps stride over it.
const _: () = assert!(
    std::mem::size_of::<TddLevel>() <= 144,
    "TddLevel grew past 144 B"
);

impl Default for TddLevel {
    /// Returns an empty level: no nodes, no pairs, no marginal values.
    fn default() -> Self {
        Self::new()
    }
}

impl TddLevel {
    /// `side`'s bit in `inlined_sides`. See the field doc.
    const fn inlined_bit(side: ChildSide) -> u8 {
        1 << side as u8
    }

    /// True if this level's refs toward its marginal `side` child are
    /// inline-encoded in the pair field.
    pub(crate) fn marginal_inlined(&self, side: ChildSide) -> bool {
        self.inlined_sides & Self::inlined_bit(side) != 0
    }
    /// Set or clear the marker [`marginal_inlined`](Self::marginal_inlined)
    /// reads.
    pub(crate) fn set_marginal_inlined(&mut self, side: ChildSide, v: bool) {
        if v { self.inlined_sides |= Self::inlined_bit(side) }
        else { self.inlined_sides &= !Self::inlined_bit(side) }
    }
    /// True if either side carries the inline-encoding marker. A level with
    /// neither is "plain": every pair side toward a marginal child is a bare
    /// slot, so duplicate pairs cannot be count-carrying multiset entries.
    pub(crate) fn any_inlined_side(&self) -> bool {
        self.inlined_sides != 0
    }

    /// An empty level: the state of every leaf level, and the starting point
    /// for building a structural level with [`push_internal_node`](Self::push_internal_node).
    pub(crate) fn new() -> Self {
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
    ///
    /// The marginal state and the inline markers reset with the arenas.
    pub(crate) fn clear(&mut self) {
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
    /// The teardown for the marginal transitions; the caller writes the new
    /// state. Capacity is released, since a marginal level never grows
    /// structure again ([`clear`](Self::clear) keeps it).
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

    /// Number of node slots: the number of values on a marginal level, else
    /// `nodes.len()` (live and tombstone). The index bound for arrays over
    /// this level; use [`live_slot_count`](Self::live_slot_count) to count nodes. 0 on
    /// a leaf level that is not marginal (its nodes are implicit).
    pub fn slot_count(&self) -> usize {
        match &self.state {
            LevelState::Counts { counts, .. } => counts.len(),
            LevelState::Weights { width, .. } => *width as usize,
            LevelState::Structural => self.nodes.len(),
        }
    }

    /// `slot_count()` minus tombstone slots — the number of nodes.
    pub fn live_slot_count(&self) -> usize {
        self.slot_count() - self.n_tombstones as usize
    }

    /// The node slots of a structural level, in index order — tombstones
    /// included, so slot `i` is `nodes()[i]`. Empty on a leaf or marginal
    /// level, which store no nodes.
    #[inline]
    pub fn nodes(&self) -> &[EncodedNode] {
        &self.nodes
    }

    /// [`nodes`](Self::nodes) paired with each slot's index.
    ///
    /// Tombstones are yielded like any other slot; skip them with
    /// [`EncodedNode::is_tombstone`], or walk
    /// [`internal_inputs_iter`](Self::internal_inputs_iter) instead, which
    /// yields only live nodes with their pairs.
    #[inline]
    pub(crate) fn nodes_iter(&self) -> impl Iterator<Item = (NodeIdx, &EncodedNode)> {
        self.nodes.iter().enumerate().map(|(i, n)| (NodeIdx(i as u32), n))
    }

    /// The model count of each node of a marginal level, indexed by
    /// [`NodeIdx`]; `None` on any other level, and on a weight-marginal one
    /// (whose values live in the [`WeightStore`](crate::diagram::WeightStore)).
    ///
    /// A value of `u128::MAX` means the count is at least that large. Read
    /// [`marginal_counts_big`](Self::marginal_counts_big)`.get(i)` for its exact
    /// value, including when the count equals `u128::MAX`.
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
    pub(crate) fn marginal_store_mut(&mut self) -> Option<(&mut Vec<u128>, &mut Option<CountOverflow>)> {
        match &mut self.state {
            LevelState::Counts { counts, big, .. } => Some((counts, big)),
            _ => None,
        }
    }

    /// The exact values of the [`marginal_counts`](Self::marginal_counts)
    /// slots that hold `u128::MAX`. `None` and an empty table both mean no
    /// slot needs an entry in the table.
    #[inline]
    pub fn marginal_counts_big(&self) -> Option<&CountOverflow> {
        match &self.state {
            LevelState::Counts { big, .. } => big.as_ref(),
            _ => None,
        }
    }

    /// Slots this level's marginal store has retired (freed by
    /// `prune_value_slots`); a metric, never a width. Monotone per level,
    /// reset only by [`clear`](Self::clear) and by a fresh marginalization;
    /// 0 on a structural level.
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

    /// The live slot count of a weight-marginal level, 0 elsewhere. `slot_count()`
    /// reads it back; a caller that mints a slot bumps it through
    /// [`set_weight_width`](Self::set_weight_width).
    #[inline]
    pub(crate) fn weight_width(&self) -> u32 {
        match &self.state {
            LevelState::Weights { width, .. } => *width,
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
    pub(crate) fn has_multi_pair(&self) -> bool {
        (0..self.nodes.len()).any(|i| {
            self.nodes[i].kind().pairs_in_arena() && self.multi_len_at(i) >= 2
        })
    }

    /// How to read the pair sides of a parent that point at this level.
    ///
    /// Build it once per level visit and decode every side through it; see
    /// [`ChildDecoder`].
    #[inline]
    pub fn child_decoder(&self) -> ChildDecoder {
        if self.is_marginal() { ChildDecoder::marginal() } else { ChildDecoder::structural() }
    }

    /// True if this level has dropped its structure for per-node values,
    /// under either arithmetic.
    pub fn is_marginal(&self) -> bool {
        !matches!(self.state, LevelState::Structural)
    }

    /// True if this level is marginal with its per-node values held in an
    /// external [`WeightStore`](crate::diagram::WeightStore) rather
    /// than in `marginal_counts` (which stays `None`). Such a level's values
    /// cannot be read from the diagram alone.
    pub fn is_weight_marginal(&self) -> bool {
        matches!(self.state, LevelState::Weights { .. })
    }


    /// Trim retained slack in `nodes`, `pairs`, and `multi_pairs` when capacity exceeds
    /// 4× length and absolute capacity is ≥ 1 Ki slots. The ratio trades peak
    /// savings against realloc-copies on levels that are re-grown soon.
    /// Count-marginal levels (already shrunk by `become_marginal`) are skipped.
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

    /// Clone the level, reserving every arena through `lim` so an allocation
    /// the host cannot serve comes back as [`OperationError::OverBudget`](crate::limits::OperationError::OverBudget) instead
    /// of aborting the process.
    pub(crate) fn try_clone_on(&self, lim: &crate::limits::Limits) -> Result<TddLevel, crate::limits::OperationError> {
        fn copy<T: Copy>(
            lim: &crate::limits::Limits,
            src: &[T],
        ) -> Result<Vec<T>, crate::limits::OperationError> {
            let mut out = Vec::new();
            lim.reserve_exact(&mut out, src.len())?;
            out.extend_from_slice(src);
            Ok(out)
        }
        let state = match &self.state {
            LevelState::Counts { counts, big, retired } => LevelState::Counts {
                counts: copy(lim, counts)?,
                big: big.clone(),
                retired: *retired,
            },
            other => other.clone(),
        };
        Ok(TddLevel {
            nodes: copy(lim, &self.nodes)?,
            pairs: copy(lim, &self.pairs)?,
            multi_pairs: copy(lim, &self.multi_pairs)?,
            inlined_sides: self.inlined_sides,
            n_tombstones: self.n_tombstones,
            dead_pairs: self.dead_pairs,
            state,
        })
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
    pub(crate) fn pop_pair(&mut self) -> Option<ChildPair> {
        self.pairs.pop()
    }

    /// Length of the pair-arena tail starting at `start`.
    #[inline]
    pub(crate) fn pair_tail_len(&self, start: usize) -> usize {
        self.pairs.len() - start
    }

}

#[cfg(test)]
mod tests;
