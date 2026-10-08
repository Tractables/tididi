//! The `TddLevel` structure, its state predicates, and its size accessors.

mod arena;
mod count_overflow;
mod implicit;
mod marginal;
mod nodes;
mod pairs;
pub use count_overflow::CountOverflow;
pub use implicit::{described, redescribed, stored_moved, Digit, ImplicitLevel, FLOOR};
pub(crate) use implicit::{floor, stored_levels_forced, PairArena, Places};
pub use nodes::{Nodes, NodesIter};
pub(crate) use nodes::NodeArena;
pub use pairs::{Pairs, StoredPairs};
pub(crate) use pairs::sort_pairs;
pub(crate) use marginal::{assert_can_make_marginal, non_marginal_child};

use super::marginal_ref::{ChildDecoder, ChildSide};
use super::primitives::{PairRange, ChildPair};
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

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
/// - otherwise structural: [`node`](Self::node)`(i)` is node `i`, and its
///   pairs are [`pairs_iter_of_idx`](Self::pairs_iter_of_idx) of that slot.
///
/// `slot_count()` is the number of node or value slots in any state.
#[derive(Clone, Debug)]
pub struct TddLevel {
    /// The stored nodes, indexed by [`NodeIdx`](super::NodeIdx). Empty on
    /// leaf and marginal levels, and on an implicit level, whose description
    /// implies them. Read through [`node`](Self::node) and
    /// [`nodes`](Self::nodes).
    pub(crate) nodes: NodeArena,
    /// Arena holding the pairs of multi-pair nodes. Read it through
    /// [`pairs_iter_of`](Self::pairs_iter_of); single-pair nodes are not in it.
    pub(crate) pairs: PairArena,
    /// Side table for multi-pair nodes whose arena start or length exceeds
    /// 2^31 (huge product grids). See `EncodedNode` for the encoding.
    pub(crate) ranges: Vec<PairRange>,
    /// Which of this level's pair sides hold value references into a marginal
    /// child that have been through `inline_small_marginal_refs`: bit 0 the
    /// left side, bit 1 the right. A side toward a structural child, or one
    /// whose child became marginal since, has its bit clear.
    ///
    /// A boundary parent is structural, so this sits outside
    /// [`LevelState`]. Set by apply when it emits or carries through such a
    /// side and by the inlining pass; reset by [`clear`](Self::clear) and by
    /// marginalization.
    pub(crate) value_ref_sides: u8,
    /// Slots in `pairs` that no live node references any more.
    ///
    /// Twin contraction mints these: a merged union is appended at the arena
    /// tail (`concat_twin_pairs`), abandoning every source range, and the parent
    /// rewrite / duplicate resolution shrink pair lists in place, abandoning
    /// their tails. Pruning and restriction do the same when they drop pairs.
    /// `compact_pairs_if_stale` reclaims them and resets this to 0.
    ///
    /// Approximate: it only triggers the sweep, which derives liveness from
    /// `nodes`/`ranges`. Reset to 0 wherever the pair arena is replaced.
    pub(crate) dead_pairs: u32,
    /// A node that held a different number of pairs from node 0 when
    /// [`close`](Self::close) last read a fit of this level, or 0 for none
    /// or one past 16 bits: the next fit reads that node's count first, and
    /// fails at once while it still differs. A hint, read again before it is
    /// trusted, so any value is safe and nothing that changes the level has
    /// to update it. Sixteen bits keep the level in its size.
    pub(crate) uneven: u16,
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
    /// Nodes and pairs; `nodes`/`pairs`/`ranges` carry the level. With the
    /// live pairs a closed diagram read off it ([`HeldPairs`]).
    Structural(HeldPairs),
    /// Model counts, one per node slot ([`CountBox`]).
    Counts(CountBox),
    /// Semiring weights, held in the external `WeightStore` and indexed by this
    /// level's slot. Only the slot count stays here — `slot_count()` has nowhere
    /// else to read it from, since `nodes` is cleared like the integer path.
    Weights { width: u32, retired: u32 },
}

impl LevelState {
    /// A structural level's state, its live pairs not read.
    pub(crate) fn structural() -> Self {
        LevelState::Structural(HeldPairs::unknown())
    }
}

/// The live pairs of a structural level, kept once a closed diagram read
/// them ([`Tdd::pair_count`](crate::Tdd::pair_count)), so that the count
/// reads each level a diagram's last operation left as it was once. Not a
/// hint: trusted while the diagram is closed. A level starts with it
/// unknown, and the closes that end an operation forget it on every level
/// the operation changed ([`forget_held_pairs`](TddLevel::forget_held_pairs)),
/// those being the levels whose pairs a closed diagram may hold changed. A
/// debug build checks it against the level's pairs at every close. Atomic,
/// so that a level is shared as a plain one is.
#[derive(Debug)]
pub(crate) struct HeldPairs(AtomicU64);

/// [`HeldPairs`] not read.
const UNKNOWN_PAIRS: u64 = u64::MAX;

impl HeldPairs {
    pub(crate) const fn unknown() -> Self {
        HeldPairs(AtomicU64::new(UNKNOWN_PAIRS))
    }

    /// The pairs kept, if read.
    #[inline]
    pub(crate) fn get(&self) -> Option<u64> {
        let n = self.0.load(Relaxed);
        (n != UNKNOWN_PAIRS).then_some(n)
    }

    /// Keep `n` pairs; a count of `u64::MAX` reads as unknown.
    #[inline]
    pub(crate) fn set(&self, n: u64) {
        self.0.store(n, Relaxed);
    }
}

impl Clone for HeldPairs {
    fn clone(&self) -> Self {
        HeldPairs(AtomicU64::new(self.0.load(Relaxed)))
    }
}

/// A count-marginal level's values, boxed, and dropped out of line: a level's
/// drop, of a structural level nearly always, is then a test of its state,
/// which the drop of every level and every reset of a state inline.
#[derive(Debug)]
pub(crate) struct CountBox(std::mem::ManuallyDrop<Box<CountState>>);

impl CountBox {
    pub(crate) fn new(state: CountState) -> Self {
        CountBox(std::mem::ManuallyDrop::new(Box::new(state)))
    }
}

impl Drop for CountBox {
    #[cold]
    #[inline(never)]
    fn drop(&mut self) {
        // Safety: the box is dropped here once, and never read after.
        unsafe { std::mem::ManuallyDrop::drop(&mut self.0) }
    }
}

impl Clone for CountBox {
    fn clone(&self) -> Self {
        CountBox::new(CountState::clone(self))
    }
}

impl std::ops::Deref for CountBox {
    type Target = CountState;
    fn deref(&self) -> &CountState {
        &self.0
    }
}

impl std::ops::DerefMut for CountBox {
    fn deref_mut(&mut self) -> &mut CountState {
        &mut self.0
    }
}

/// A count-marginal level's values.
#[derive(Clone, Debug)]
pub(crate) struct CountState {
    /// Model counts, one per node slot. `u128::MAX` marks a count at least
    /// that large, whose exact value is the entry `big` holds for that slot.
    pub(crate) counts: Vec<u128>,
    pub(crate) big: Option<CountOverflow>,
    pub(crate) retired: u32,
}

/// `TddLevel` stays compact: the O(levels) sweeps stride over it.
const _: () = assert!(
    std::mem::size_of::<TddLevel>() <= 104,
    "TddLevel grew past 104 B"
);

impl Default for TddLevel {
    /// Returns an empty level: no nodes, no pairs, no marginal values.
    fn default() -> Self {
        Self::new()
    }
}

impl TddLevel {
    /// `side`'s bit in `value_ref_sides`. See the field doc.
    const fn side_bit(side: ChildSide) -> u8 {
        1 << side as u8
    }

    /// True if this level's references on `side` are value references into
    /// a marginal child that the inlining pass has been over; see
    /// [`value_ref_sides`](Self::value_ref_sides).
    pub(crate) fn has_value_refs(&self, side: ChildSide) -> bool {
        self.value_ref_sides & Self::side_bit(side) != 0
    }
    /// Set or clear the marker [`has_value_refs`](Self::has_value_refs)
    /// reads.
    pub(crate) fn set_has_value_refs(&mut self, side: ChildSide, v: bool) {
        if v { self.value_ref_sides |= Self::side_bit(side) }
        else { self.value_ref_sides &= !Self::side_bit(side) }
    }
    /// True if either side holds value references. A level with neither is
    /// "plain": every pair side toward a marginal child is a bare slot, so
    /// duplicate pairs cannot be count-carrying multiset entries.
    pub(crate) fn any_value_ref_side(&self) -> bool {
        self.value_ref_sides != 0
    }

    /// An empty level: the state of every leaf level, and the starting point
    /// for building a structural level with [`push_internal_node`](Self::push_internal_node).
    pub(crate) fn new() -> Self {
        TddLevel {
            nodes: NodeArena::default(),
            pairs: PairArena::default(),
            ranges: Vec::new(),
            value_ref_sides: 0,
            dead_pairs: 0,
            uneven: 0,
            state: LevelState::structural(),
        }
    }

    /// Reset to empty (as [`new`](Self::new)), keeping buffer capacity: on
    /// a level whose nodes are implied, the capacity their arena would have,
    /// as an implicit arena becomes a stored one of its capacity.
    ///
    /// The marginal state and the inline markers reset with the arenas.
    pub(crate) fn clear(&mut self) {
        if self.implied_by().is_some() {
            self.nodes = NodeArena::from(Vec::with_capacity(self.pairs.node_capacity()));
        }
        self.nodes.clear();
        self.pairs.clear();
        self.ranges.clear();
        self.value_ref_sides = 0;
        self.dead_pairs = 0;
        self.uneven = 0;
        self.state = LevelState::structural();
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
        self.ranges.clear();
        self.ranges.shrink_to_fit();
        self.dead_pairs = 0;
        self.value_ref_sides = 0;
    }

    /// Number of slots: the number of values on a marginal level, else
    /// `nodes.len()`. The index bound for arrays over this level. 0 on a leaf
    /// level that is not marginal (its nodes are implicit). Use
    /// [`Tdd::reference_slot_count`](crate::Tdd::reference_slot_count) to size
    /// arrays indexed by child references.
    pub fn slot_count(&self) -> usize {
        match &self.state {
            LevelState::Counts(c) => c.counts.len(),
            LevelState::Weights { width, .. } => *width as usize,
            LevelState::Structural(_) => self.node_count(),
        }
    }

    /// The model count of each node of a marginal level, indexed by
    /// [`NodeIdx`](super::NodeIdx); `None` on any other level, and on a weight-marginal one
    /// (whose values live in the [`WeightStore`](crate::diagram::WeightStore)).
    ///
    /// A value of `u128::MAX` means the count is at least that large. Read
    /// [`marginal_counts_big`](Self::marginal_counts_big)`.get(i)` for its exact
    /// value, including when the count equals `u128::MAX`.
    #[inline]
    pub fn marginal_counts(&self) -> Option<&[u128]> {
        match &self.state {
            LevelState::Counts(c) => Some(&c.counts),
            _ => None,
        }
    }

    /// A count-marginal level's column as one view, its fast values and
    /// overflow table together; `None` on any other level. The view carries
    /// no `all_u64` certificate.
    #[inline]
    pub(crate) fn count_column(&self) -> Option<crate::value::CountRef<'_>> {
        match &self.state {
            LevelState::Counts(c) => Some(crate::value::CountRef::new(&c.counts, c.big.as_ref())),
            _ => None,
        }
    }

    /// Both halves of a count-marginal level's store at once: the fast column
    /// and the overflow table, which the compaction passes rewrite together.
    #[inline]
    pub(crate) fn marginal_store_mut(&mut self) -> Option<(&mut Vec<u128>, &mut Option<CountOverflow>)> {
        match &mut self.state {
            LevelState::Counts(c) => {
                let c: &mut CountState = c;
                Some((&mut c.counts, &mut c.big))
            }
            _ => None,
        }
    }

    /// The exact values of the [`marginal_counts`](Self::marginal_counts)
    /// slots that hold `u128::MAX`. `None` and an empty table both mean no
    /// slot needs an entry in the table.
    #[inline]
    pub fn marginal_counts_big(&self) -> Option<&CountOverflow> {
        match &self.state {
            LevelState::Counts(c) => c.big.as_ref(),
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
            LevelState::Counts(c) => c.retired,
            LevelState::Weights { retired, .. } => *retired,
            LevelState::Structural(_) => 0,
        }
    }

    /// Account `n` more retired slots. A structural level retires nothing and
    /// silently ignores the call — it has no store to free from.
    #[inline]
    pub(crate) fn retire_marginal_slots(&mut self, n: u32) {
        match &mut self.state {
            LevelState::Counts(c) => c.retired = c.retired.saturating_add(n),
            LevelState::Weights { retired, .. } => *retired = retired.saturating_add(n),
            LevelState::Structural(_) => {}
        }
    }

    /// Set the live slot count of a weight-marginal level. No-op elsewhere.
    #[inline]
    pub(crate) fn set_weight_width(&mut self, w: u32) {
        if let LevelState::Weights { width, .. } = &mut self.state {
            *width = w;
        }
    }

    /// True if any node has more than one pair. O(width), and O(1) on a
    /// level whose arena is empty, whose nodes then hold one pair or none.
    #[inline]
    pub(crate) fn has_multi_pair(&self) -> bool {
        !self.pairs.is_empty()
            && self.nodes().iter().any(|n| n.kind().pairs_in_arena() && self.multi_range(&n).len() >= 2)
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
        !matches!(self.state, LevelState::Structural(_))
    }

    /// True if this level is marginal with its per-node values held in an
    /// external [`WeightStore`](crate::diagram::WeightStore) rather
    /// than in `marginal_counts` (which stays `None`). Such a level's values
    /// cannot be read from the diagram alone.
    pub fn is_weight_marginal(&self) -> bool {
        matches!(self.state, LevelState::Weights { .. })
    }


    /// Trim retained slack in `nodes`, `pairs`, and `ranges` when capacity exceeds
    /// 4× length and absolute capacity is ≥ 1 Ki slots. The ratio trades peak
    /// savings against realloc-copies on levels that are re-grown soon.
    /// Count-marginal levels (already shrunk by `become_marginal`) are skipped.
    #[inline]
    pub(crate) fn shrink_arrays(&mut self) {
        if matches!(self.state, LevelState::Counts(_)) {
            return;
        }
        const MIN_SHRINK_CAP: usize = 1024;
        // cap > (32 / 8) * len  ⟺  8 * cap > 32 * len  ⟺  cap > 4 * len
        const SHRINK_RATIO_EIGHTHS: u64 = 32; // 4×
        let should_shrink = |cap: usize, len: usize| -> bool {
            cap >= MIN_SHRINK_CAP && (cap as u64) * 8 > SHRINK_RATIO_EIGHTHS * (len as u64)
        };
        if should_shrink(self.node_capacity(), self.node_count()) {
            self.shrink_nodes();
        }
        if should_shrink(self.pairs.capacity(), self.pairs.len()) {
            self.pairs.shrink_to_fit();
        }
        if should_shrink(self.ranges.capacity(), self.ranges.len()) {
            self.ranges.shrink_to_fit();
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
            LevelState::Counts(c) => LevelState::Counts(CountBox::new(CountState {
                counts: copy(lim, &c.counts)?,
                big: c.big.clone(),
                retired: c.retired,
            })),
            other => other.clone(),
        };
        // Implied nodes are charged as stored ones copied at their count.
        if let Some(d) = self.implied_by() {
            lim.charge_bytes((d.nodes() as u64).saturating_mul(std::mem::size_of::<super::primitives::EncodedNode>() as u64))?;
        }
        Ok(TddLevel {
            nodes: NodeArena::from(copy(lim, self.nodes.stored())?),
            pairs: self.pairs.try_clone_on(lim)?,
            ranges: copy(lim, &self.ranges)?,
            value_ref_sides: self.value_ref_sides,
            dead_pairs: self.dead_pairs,
            uneven: self.uneven,
            state,
        })
    }

    /// Length of the pair arena, dead entries included: the start offset the
    /// next multi-pair node's range gets. Not the level's pair count; see
    /// [`live_pairs`](Self::live_pairs) for that.
    #[inline]
    pub(crate) fn arena_len(&self) -> usize {
        self.pairs.len()
    }

    /// Pop the last pair off the arena.
    #[inline]
    pub(crate) fn pop_pair(&mut self) -> Option<ChildPair> {
        self.pairs.stored_mut().pop()
    }

    /// Length of the pair-arena tail starting at `start`, on a level being
    /// built, whose pairs are stored.
    #[inline]
    pub(crate) fn pair_tail_len(&self, start: usize) -> usize {
        self.pairs.stored_len() - start
    }

}

#[cfg(test)]
mod tests;
