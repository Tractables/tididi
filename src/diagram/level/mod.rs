//! The `TddLevel` structure, its state predicates, and its size accessors.

mod arena;
mod ranges;
pub(crate) use ranges::RangeTable;
mod count_overflow;
mod implicit;
mod marginal;
mod nodes;
mod pairs;
pub use count_overflow::CountOverflow;
pub use implicit::{described, redescribed, stored_moved, Digit, ImplicitLevel, FLOOR};
pub(crate) use implicit::{floor, kept_capacity, stored_levels_forced, ChildKept, NodeChunks, NodeCursor, PairArena, Places, StoreRoom};
pub use nodes::{Nodes, NodesIter};
pub(crate) use nodes::NodeArena;
pub use pairs::{Pairs, StoredPairs};
pub(crate) use pairs::{decoded, inline_pair, sort_pairs};
pub(crate) use marginal::{assert_can_make_marginal, non_marginal_child};

use super::marginal_ref::{ChildDecoder, ChildSide};
use super::primitives::{ChildPair, EncodedNode};
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
    pub(crate) ranges: RangeTable,
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
    /// Twin contraction's parent rewrite and duplicate resolution mint these:
    /// they shrink pair lists in place, abandoning their tails. Pruning and restriction do the same when they drop pairs.
    /// `compact_pairs_if_stale` reclaims them and resets this to 0.
    ///
    /// Approximate: it only triggers the sweep, which derives liveness from
    /// `nodes`/`ranges`. Reset to 0 wherever the pair arena is replaced.
    pub(crate) dead_pairs: u32,
    /// A node that held a different number of pairs from node 0 when a
    /// [`close`](Self::close) last looked for one, or 0 for none or one
    /// past 16 bits: the next fit reads that node's count first, and
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
/// them ([`Tdd::pair_count`](crate::Tdd::pair_count)) or left by the
/// operation that last changed the level for the close that ends it
/// ([`leave_live_pairs`](TddLevel::leave_live_pairs)): the count reads a
/// level at most once, and not at all where its operation counted it. Not
/// a hint: trusted while the diagram is closed. A level starts with it
/// unknown. An edit that marks the level changed forgets it
/// ([`forget_held_pairs`](TddLevel::forget_held_pairs)); a close keeps a
/// count left on a level it closes and forgets any other
/// ([`close_held_pairs`](TddLevel::close_held_pairs)), the levels it closes
/// being those the operation may have changed. A debug build counts the
/// pairs again where a count is left, and checks every kept count at every
/// close. Atomic, so that a level is shared as a plain one is.
#[derive(Debug)]
pub(crate) struct HeldPairs(AtomicU64);

/// [`HeldPairs`] not read.
const UNKNOWN_PAIRS: u64 = u64::MAX;

/// The bit of a count left for the close ([`HeldPairs::leave`]): no read
/// takes it until a close keeps it.
const LEFT_PAIRS: u64 = 1 << 63;

impl HeldPairs {
    pub(crate) const fn unknown() -> Self {
        HeldPairs(AtomicU64::new(UNKNOWN_PAIRS))
    }

    /// The pairs kept, if read or kept by a close.
    #[inline]
    pub(crate) fn get(&self) -> Option<u64> {
        let n = self.0.load(Relaxed);
        (n & LEFT_PAIRS == 0).then_some(n)
    }

    /// Keep `n` pairs; a count of `u64::MAX` reads as unknown.
    #[inline]
    pub(crate) fn set(&self, n: u64) {
        self.0.store(n, Relaxed);
    }

    /// Leave `n` pairs for the next close of the level to keep.
    #[inline]
    fn leave(&mut self, n: u64) {
        debug_assert!(n < LEFT_PAIRS, "a level holds fewer than 2^63 pairs");
        *self.0.get_mut() = n | LEFT_PAIRS;
    }

    /// What a close of the level does: keep the count left for it, forget
    /// any other.
    #[inline]
    fn close(&mut self) {
        let n = self.0.get_mut();
        *n = if *n != UNKNOWN_PAIRS && *n & LEFT_PAIRS != 0 { *n & !LEFT_PAIRS } else { UNKNOWN_PAIRS };
    }

    /// What a close of a level no unmarked edit changed does: keep the
    /// count left for it, and any count kept before.
    #[inline]
    fn settle(&mut self) {
        let n = self.0.get_mut();
        if *n != UNKNOWN_PAIRS {
            *n &= !LEFT_PAIRS;
        }
    }

    /// Whether a count is left for a close to keep.
    #[inline]
    pub(crate) fn is_left(&self) -> bool {
        self.left().is_some()
    }

    /// The count left for a close to keep, if any.
    #[inline]
    fn left(&self) -> Option<u64> {
        let n = self.0.load(Relaxed);
        (n != UNKNOWN_PAIRS && n & LEFT_PAIRS != 0).then_some(n & !LEFT_PAIRS)
    }

    /// Forget a count left for a close, keeping a count kept already.
    #[inline]
    fn drop_left(&mut self) {
        if self.is_left() {
            *self.0.get_mut() = UNKNOWN_PAIRS;
        }
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
    std::mem::size_of::<TddLevel>() <= 88,
    "TddLevel grew past 88 B"
);

/// The node and pair arenas of a level's copy, and the live pairs its
/// level keeps ([`TddLevel::try_clone_to`]).
type CopiedArenas<'a> = (Vec<EncodedNode>, Vec<ChildPair>, &'a HeldPairs);

/// A copy of `src` reserved through `lim` with room for `room` more.
#[inline(always)]
fn copy_reserved<T: Copy>(
    lim: &crate::limits::Limits,
    src: &[T],
    room: usize,
) -> Result<Vec<T>, crate::limits::OperationError> {
    let mut out = Vec::new();
    lim.reserve_exact(&mut out, src.len().saturating_add(room))?;
    out.extend_from_slice(src);
    Ok(out)
}

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
            ranges: RangeTable::default(),
            value_ref_sides: 0,
            dead_pairs: 0,
            uneven: 0,
            state: LevelState::structural(),
        }
    }

    /// Reset to empty (as [`new`](Self::new)), keeping buffer capacity: on
    /// a level whose nodes are implied, the capacity their arena would have,
    /// as an implicit arena becomes a stored one of its capacity, where the
    /// allocator grants them.
    ///
    /// The marginal state and the inline markers reset with the arenas.
    pub(crate) fn clear(&mut self) {
        self.clear_keeping(usize::MAX);
    }

    /// [`clear`](Self::clear), where a node or pair arena that a description
    /// stood for is given its capacity only up to `max_bytes`
    /// ([`PairArena::clear_keeping`]); stored arenas keep theirs, which the
    /// pool caps ([`crate::diagram::pool`]).
    pub(crate) fn clear_keeping(&mut self, max_bytes: usize) {
        self.clear_nodes(max_bytes / std::mem::size_of::<EncodedNode>());
        self.pairs.clear_keeping(max_bytes / std::mem::size_of::<ChildPair>());
        self.ranges.clear();
        self.value_ref_sides = 0;
        self.dead_pairs = 0;
        self.uneven = 0;
        self.state = LevelState::structural();
    }

    /// The node arena emptied, for a level built in this one's place, of
    /// the capacity it has ([`clear_nodes`](Self::clear_nodes)).
    pub(crate) fn take_nodes(&mut self) -> NodeArena {
        self.clear_nodes(usize::MAX);
        std::mem::take(&mut self.nodes)
    }

    /// Empty the node arena in place, keeping its capacity: on a level whose
    /// nodes are implied, an arena of the capacity it would have, up to
    /// `max` nodes and where the allocator grants it ([`kept_capacity`]),
    /// the description still implying the level's nodes.
    #[inline]
    fn clear_nodes(&mut self, max: usize) {
        match self.implied_by() {
            Some(_) => self.nodes = NodeArena::from(kept_capacity(self.pairs.node_capacity(), max)),
            None => self.nodes.stored_mut().clear(),
        }
    }

    /// Release the structural arenas and zero the counters that describe them.
    ///
    /// The teardown for the marginal transitions; the caller writes the new
    /// state. Capacity is released, since a marginal level never grows
    /// structure again ([`clear`](Self::clear) keeps it).
    fn drop_structure(&mut self) {
        self.nodes.clear();
        self.nodes.shrink_to_fit();
        self.pairs.clear_keeping(0);
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

    /// True if any node has more than one pair. O(width), and O(1) on an
    /// implicit level, whose nodes have the same pairs each, and on a level
    /// whose arena is empty, whose nodes then hold one pair or none.
    #[inline]
    pub(crate) fn has_multi_pair(&self) -> bool {
        match self.pairs.implicit() {
            Some(d) => d.pairs_per_node() >= 2,
            None => {
                !self.pairs.is_empty()
                    && self.nodes.stored().iter().any(|n| n.kind().pairs_in_arena() && self.multi_range(n).len() >= 2)
            }
        }
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
    ///
    /// A reduction calls this on every level, most of them small: a length
    /// is read only for an arena whose capacity is at the floor or over it.
    #[inline]
    pub(crate) fn shrink_arrays(&mut self) {
        if matches!(self.state, LevelState::Counts(_)) {
            return;
        }
        const MIN_SHRINK_CAP: usize = 1024;
        // cap > (32 / 8) * len  ⟺  8 * cap > 32 * len  ⟺  cap > 4 * len
        const SHRINK_RATIO_EIGHTHS: u64 = 32; // 4×
        #[inline(always)]
        fn should_shrink(cap: usize, len: impl FnOnce() -> usize) -> bool {
            cap >= MIN_SHRINK_CAP && (cap as u64) * 8 > SHRINK_RATIO_EIGHTHS * (len() as u64)
        }
        if should_shrink(self.node_capacity(), || self.node_count()) {
            self.shrink_nodes();
        }
        if should_shrink(self.pairs.capacity(), || self.pairs.len()) {
            self.pairs.shrink_to_fit();
        }
        if should_shrink(self.ranges.capacity(), || self.ranges.len()) {
            self.ranges.shrink_to_fit();
        }
    }

    /// Clone the level into `out`, reserving every arena through `lim` so
    /// an allocation the host cannot serve comes back as
    /// [`OperationError::OverBudget`](crate::limits::OperationError::OverBudget)
    /// instead of aborting the process. A diagram's copy writes its levels
    /// to their places: a copy returned by value is built beside its place
    /// and moved there, and on a diagram of many small levels that move
    /// costs about what the copy does, reading wide what the copy has just
    /// written narrow.
    #[inline]
    pub(crate) fn try_clone_to<'a>(
        &self,
        lim: &crate::limits::Limits,
        out: &'a mut std::mem::MaybeUninit<TddLevel>,
    ) -> Result<&'a mut TddLevel, crate::limits::OperationError> {
        Ok(match self.try_copy_arenas(lim, 0, 0)? {
            Some(arenas) => out.write(self.with_arenas(arenas)),
            None => out.write(self.try_clone_held(lim, 0, 0)?),
        })
    }

    /// [`try_clone_to`](Self::try_clone_to) written over `out`, whose
    /// storage is dropped, with room for `nodes` more nodes and `pairs`
    /// more arena pairs on stored arenas, so a copy that is about to grow is
    /// written once. An implicit pair arena keeps its description and takes
    /// no room. Refused, `out` is as it was.
    #[inline]
    pub(crate) fn try_clone_over(
        &self,
        lim: &crate::limits::Limits,
        nodes: usize,
        pairs: usize,
        out: &mut TddLevel,
    ) -> Result<(), crate::limits::OperationError> {
        let copy = match self.try_copy_arenas(lim, nodes, pairs)? {
            Some(arenas) => self.with_arenas(arenas),
            None => self.try_clone_held(lim, nodes, pairs)?,
        };
        drop(std::mem::replace(out, copy));
        Ok(())
    }

    /// The arenas of a copy of a structural level that stores its nodes and
    /// pairs and keeps no range table, with room for `nodes` more nodes and
    /// `pairs` more pairs, and the live pairs it keeps; `None` for any other
    /// level, which [`try_clone_held`](Self::try_clone_held) copies.
    #[inline(always)]
    fn try_copy_arenas(
        &self,
        lim: &crate::limits::Limits,
        nodes: usize,
        pairs: usize,
    ) -> Result<Option<CopiedArenas<'_>>, crate::limits::OperationError> {
        let (LevelState::Structural(held), Some(stored)) = (&self.state, self.pairs.stored()) else {
            return Ok(None);
        };
        if !self.ranges.is_empty() {
            return Ok(None);
        }
        let n = copy_reserved(lim, self.nodes.stored(), nodes)?;
        let p = copy_reserved(lim, stored, pairs)?;
        Ok(Some((n, p, held)))
    }

    /// The copy of this level whose arenas
    /// [`try_copy_arenas`](Self::try_copy_arenas) made.
    #[inline(always)]
    fn with_arenas(&self, (nodes, pairs, held): CopiedArenas<'_>) -> TddLevel {
        TddLevel {
            nodes: NodeArena::from(nodes),
            pairs: PairArena::from(pairs),
            ranges: RangeTable::default(),
            value_ref_sides: self.value_ref_sides,
            dead_pairs: self.dead_pairs,
            uneven: self.uneven,
            state: LevelState::Structural(held.clone()),
        }
    }

    /// The copy, with room as [`try_clone_over`](Self::try_clone_over)
    /// gives it, of a level that [`try_copy_arenas`](Self::try_copy_arenas)
    /// does not copy: a marginal or implicit one, or one with a range table.
    #[inline(never)]
    fn try_clone_held(
        &self,
        lim: &crate::limits::Limits,
        nodes: usize,
        pairs: usize,
    ) -> Result<TddLevel, crate::limits::OperationError> {
        let state = match &self.state {
            LevelState::Counts(c) => LevelState::Counts(CountBox::new(CountState {
                counts: copy_reserved(lim, &c.counts, 0)?,
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
            nodes: NodeArena::from(copy_reserved(lim, self.nodes.stored(), nodes)?),
            pairs: self.pairs.try_clone_on(lim, pairs)?,
            ranges: self.ranges.try_clone_on(lim)?,
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
