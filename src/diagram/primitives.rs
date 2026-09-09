//! Primitive node types: `NodeIdx`, `TddNodeId`, `LeafLabel`, `InputPair`,
//! `TddNodeData`, `MultiPairRange`, and related constants.

use crate::vtree::VtreeIdx;

/// Index of a node within one level.
///
/// On a leaf level the three implicit nodes are `0..LEAF_WIDTH`
/// ([`ONE_LEAF_IDX`], [`POS_LEAF_IDX`], [`NEG_LEAF_IDX`]); on a structural
/// level it indexes the level's slots; on a marginal level it indexes its
/// counts. In a pair whose child level is marginal the raw `u32` is a tagged
/// reference rather than a plain index — decode it with
/// [`SideView`](super::SideView).
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug, Ord, PartialOrd)]
#[repr(transparent)]  // guaranteed same layout as bare u32 (no padding/tag)
pub struct NodeIdx(pub u32);

impl NodeIdx {
    /// The index as a `usize`.
    #[inline(always)]
    pub fn idx(self) -> usize { self.0 as usize }
}

/// Sentinel index for the constant-false function.
///
/// Appears only in [`Tdd::output`](super::Tdd::output) (the diagram is
/// unsatisfiable; [`Tdd::is_zero`](super::Tdd::is_zero)), never in a stored
/// pair: no stored node computes false.
pub const ZERO: NodeIdx = NodeIdx(u32::MAX);

/// A node of the diagram: its level (a vtree node) and its index in that level.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct TddNodeId {
    /// The vtree node whose level holds the node.
    pub vtree: VtreeIdx,
    /// Index of the node within that level.
    pub local: NodeIdx,
}

/// The function denoted by a node of a leaf level, over that leaf's variable.
///
/// The discriminant is the node's local index: a leaf level stores nothing,
/// and a pair pointing at index `i` of a leaf level denotes
/// `LeafLabel::from_idx(i)`. `One` is index 0 so that the constant-true node
/// sits at local index 0 on every level, leaf or internal.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
#[repr(u32)]
pub enum LeafLabel {
    /// Constant true.
    One = 0,
    /// The variable itself, `x`.
    Pos = 1,
    /// Its negation, `¬x`.
    Neg = 2,
    /// Constant false. A sentinel: never the target of a pair.
    Zero = 3,
}

/// Number of implicit nodes on a leaf level (`One`, `Pos`, `Neg`), the value
/// [`Tdd::effective_width`](super::Tdd::effective_width) reports there.
pub const LEAF_WIDTH: usize = 3;

/// Local index of the constant-true node on a leaf level.
pub const ONE_LEAF_IDX: NodeIdx = NodeIdx(LeafLabel::One as u32);
/// Local index of the positive-literal node on a leaf level.
pub const POS_LEAF_IDX: NodeIdx = NodeIdx(LeafLabel::Pos as u32);
/// Local index of the negative-literal node on a leaf level.
pub const NEG_LEAF_IDX: NodeIdx = NodeIdx(LeafLabel::Neg as u32);

impl LeafLabel {
    /// The label at local index `i` of a leaf level (`0..LEAF_WIDTH`).
    ///
    /// # Panics
    ///
    /// Panics if `i >= LEAF_WIDTH`.
    #[inline(always)]
    pub fn from_idx(i: usize) -> LeafLabel {
        match i {
            0 => LeafLabel::One,
            1 => LeafLabel::Pos,
            2 => LeafLabel::Neg,
            _ => unreachable!("invalid leaf index: {i}"),
        }
    }
}

/// One input of a structural node: a node of the left child level and a node
/// of the right child level, denoting the conjunction of the two.
///
/// Each side is a local index into that child level, in range for its
/// `effective_width`, and is never [`ZERO`]. When the child level is marginal
/// the side is a tagged reference instead of a plain index and must be read
/// through [`SideView::child`](super::SideView::child). A node's pairs are
/// unordered and pairwise disjoint as functions.
///
/// `#[repr(C)]`: a single-pair node stores its pair in the two `u32` words of
/// [`TddNodeData`] and reads it back by pointer cast.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug, Ord, PartialOrd)]
#[repr(C)]
pub struct InputPair {
    /// Node in the left child level.
    pub left: NodeIdx,
    /// Node in the right child level.
    pub right: NodeIdx,
}

/// Bytes one input pair occupies in a diagram — the per-pair storage size, and
/// the unit `Tdd::size()` counts in.
///
/// It is the same 8 bytes under both encodings: a multi-pair node's pairs live
/// in the level's `pairs: Vec<InputPair>` arena, and a single-pair node stores
/// its pair INLINE in the two `u32` fields of `TddNodeData` (which is why
/// `InputPair` is `#[repr(C)]` at all — see the type doc above). So a pair count
/// converts to a byte figure without having to know which encoding holds it.
///
/// Named here, beside the layout it describes, so a caller converting a pair
/// count to bytes (the level-arena reclaim in `pool.rs`) does not restate the
/// layout.
pub(crate) const INPUT_PAIR_BYTES: usize = size_of::<InputPair>();

impl InputPair {
    /// Whether this pair can be stored inline in a `TddNodeData` node without
    /// aliasing the leaf or `multi_ranged` encoding.
    #[inline]
    pub(crate) fn can_inline(&self) -> bool {
        self.right.0 & LEAF_BIT == 0 && self.left.0 & MULTI_BIT == 0
    }
}

/// Bit 31 of a node's `b` word: the node stores a leaf label, not pairs.
/// See the encoding table on [`TddNodeData`].
pub(super) const LEAF_BIT: u32 = 1 << 31;
/// Bit 31 of a node's `a` word: the node's pairs live in the level's arena.
/// See the encoding table on [`TddNodeData`].
pub(super) const MULTI_BIT: u32 = 1 << 31;
/// Sentinel `b` for a **tombstone**: a dead node slot that survives in `nodes`
/// instead of being compacted out (the index-stable conjoin). Chosen as
/// `LEAF_BIT | 1` so it cannot collide with any live encoding:
/// - real leaves have `b == LEAF_BIT` exactly (low bits clear);
/// - internals (inline/multi) have bit 31 clear (`b < 2^31`).
///
/// Bit 31 set means `is_internal()` returns `false`, so every `is_internal()`-
/// gated reader (`size`, `internal_inputs_iter`, all model-count paths, prune
/// reachability) skips a tombstone for free. A tombstone is always
/// *unreferenced* (no live node points at it), so it is unreachable and prune's
/// `retain` reclaims it — `leaf_label()` is never reached through child
/// traversal. The only readers that must explicitly discount tombstones are the
/// count-only `node_count`/`max_width` (via `TddLevel::live_width`).
pub(super) const TOMBSTONE_B: u32 = LEAF_BIT | 1;
/// Sentinel value for `b` that marks an extended multi-pair node (side-table form).
/// Chosen as 1 because `pair_len` == 1 is forbidden for multi (caller uses inline),
/// so 1 cannot appear as a legitimate normal-multi `pair_len`.
pub(super) const RANGE_SENTINEL: u32 = 1;

/// Side-table entry for extended multi-pair nodes (`pair_start` or `pair_len` ≥ 2^31).
/// The node data holds `(a = multi_pairs_idx | MULTI_BIT, b = RANGE_SENTINEL)`, and this struct
/// holds the actual start/len. Only allocated when the 31-bit encoding would overflow.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub(crate) struct MultiPairRange {
    pub start: u64,
    pub len: u64,
}

/// A stored node: 8 bytes encoding where its pairs live.
///
/// A reader never decodes the word itself: [`TddLevel::pairs_of`] and
/// [`TddLevel::pairs_iter_of`] resolve a node to its pairs, and
/// [`is_internal`](Self::is_internal) / [`is_tombstone`](Self::is_tombstone)
/// classify it. Every node of a valid diagram is internal or a tombstone (a
/// dead slot, unreferenced, that `minimize` removes).
///
/// The two `u32` words carry a four-way encoding:
///
/// ```text
/// ┌──────────────────────────────┬──────────────────────────────┐
/// │         a (u32)              │         b (u32)              │
/// ├──────────────────────────────┼──────────────────────────────┤
/// │ LeafLabel as u32             │ LEAF_BIT (1<<31)             │  ← leaf
/// │ left child index             │ right child index            │  ← inline pair
/// │ pair_start | MULTI_BIT       │ pair_len (∈ {0, 2, 3, …})    │  ← normal multi-pair
/// │ multi_pairs_idx     | MULTI_BIT      │ RANGE_SENTINEL (= 1)           │  ← extended multi-pair
/// └──────────────────────────────┴──────────────────────────────┘
/// ```
///
/// Decoding stays cheap, and the hot test is first: `b & LEAF_BIT != 0` means
/// leaf; else `a & MULTI_BIT == 0` means inline pair; else `b == 1` means
/// extended multi-pair (its size lives in the level's `multi_pairs` table); else
/// normal multi-pair. `LEAF_BIT` and `MULTI_BIT` are both `1 << 31`;
/// `pair_len == 1` is forbidden for multi-pair (the caller converts it to
/// inline), which is what leaves `b == 1` free as the extended sentinel.
/// `pair_len == 0` is legal.
///
/// **Inline pairs** are most of the nodes, and store their single
/// [`InputPair`] directly in `(a, b)`. `TddNodeData` and `InputPair` are both
/// `#[repr(C)]` over the same two `u32`s, so `pairs_of` hands back
/// `&[InputPair; 1]` by pointer cast rather than copying.
///
/// **Multi-pair** nodes name a contiguous range of the level's `pairs` arena.
/// The normal form packs `(pair_start, pair_len)` into the two words when both
/// fit in 31 bits; past that (a huge product grid on a pathological formula)
/// the node goes to the extended form, whose `u64` start and length live in
/// the level's side table — so the 2× cost falls only on the nodes that need
/// it, never on every node.
///
/// [`TddLevel::pairs_of`]: super::TddLevel::pairs_of
/// [`TddLevel::pairs_iter_of`]: super::TddLevel::pairs_iter_of
#[derive(Copy, Clone, Eq, PartialEq)]
#[repr(C)]
pub struct TddNodeData {
    pub(crate) a: u32,  // leaf: label; inline: left child; multi: pair_start | MULTI_BIT
    pub(crate) b: u32,  // leaf: LEAF_BIT; inline: right child; multi: pair_len
}

impl TddNodeData {
    /// Create a leaf node for `label`.
    #[inline(always)]
    #[cfg(test)]
    pub(crate) fn leaf(label: LeafLabel) -> Self {
        TddNodeData { a: label as u32, b: LEAF_BIT }
    }

    /// Create an inline single-pair node. `a` and `b` store the pair's left/right indices.
    /// Caller must verify `pair.can_inline()` — violating this aliases the leaf or
    /// `multi_ranged` encoding and causes silent data corruption.
    #[inline(always)]
    pub(crate) fn inline(pair: InputPair) -> Self {
        debug_assert!(pair.can_inline(), "pair cannot be inlined: would alias leaf/multi_ranged encoding");
        TddNodeData { a: pair.left.0, b: pair.right.0 }
    }

    /// Create a normal multi-pair node referencing the pairs arena at
    /// `[pair_start, pair_start+pair_len)`. Both must fit in 31 bits; use
    /// `TddLevel::encode_multi` for arbitrary sizes (it promotes to extended
    /// form when needed). `pair_len` may be 0 (used by full.rs for empty placeholders).
    #[inline(always)]
    pub(crate) fn multi_pair(pair_start: u32, pair_len: u32) -> Self {
        debug_assert!(pair_start & MULTI_BIT == 0, "pair_start too large; use encode_multi");
        debug_assert!(pair_len & LEAF_BIT == 0, "pair_len overflow; use encode_multi");
        TddNodeData { a: pair_start | MULTI_BIT, b: pair_len }
    }

    /// Create an extended multi-pair node whose `(start, len)` live in the level's
    /// `multi_pairs` side table at `multi_pairs_idx`. `b = RANGE_SENTINEL` (= 1) distinguishes this
    /// from normal multi (which has `pair_len` ∈ {0, 2, 3, …}).
    #[inline(always)]
    pub(crate) fn multi_ranged(multi_pairs_idx: u32) -> Self {
        debug_assert!(multi_pairs_idx & MULTI_BIT == 0, "multi_pairs_idx too large");
        TddNodeData { a: multi_pairs_idx | MULTI_BIT, b: RANGE_SENTINEL }
    }

    /// True when the node holds no pairs: a tombstone, or a leaf-label node
    /// (which no valid diagram stores).
    #[inline(always)]
    pub fn is_leaf(&self) -> bool { self.b & LEAF_BIT != 0 }

    /// True for a node with pairs (inline or multi-pair).
    #[inline(always)]
    pub fn is_internal(&self) -> bool { self.b & LEAF_BIT == 0 }

    /// A dead node slot kept in place (not compacted) by the index-stable
    /// conjoin. `a = u32::MAX` is a tripwire: it is not a valid leaf label, so
    /// `leaf_label()` panics in debug if a tombstone is ever mistaken for a real
    /// leaf. See `TOMBSTONE_B`.
    #[inline(always)]
    #[cfg(test)]
    pub(crate) fn tombstone() -> Self { TddNodeData { a: u32::MAX, b: TOMBSTONE_B } }

    /// True for a dead slot left in place by an index-stable rewrite. It is
    /// referenced by no pair; `width()` still counts it, `live_width()` does
    /// not, and `minimize` removes it.
    #[inline(always)]
    pub fn is_tombstone(&self) -> bool { self.b == TOMBSTONE_B }

    /// True for a node with exactly one pair, stored in the node word.
    #[inline(always)]
    pub fn is_inline(&self) -> bool { self.b & LEAF_BIT == 0 && self.a & MULTI_BIT == 0 }

    /// True for a node whose pairs live in the level's `pairs` arena.
    #[inline(always)]
    pub fn is_multi(&self) -> bool { self.b & LEAF_BIT == 0 && self.a & MULTI_BIT != 0 }

    /// True for extended multi-pair nodes (start/len live in `level.multi_pairs`).
    /// Disambiguated by `b == 1` — impossible for normal multi since `pair_len` == 1
    /// is forbidden (caller uses inline).
    #[inline(always)]
    pub(crate) fn is_multi_ranged(&self) -> bool {
        self.a & MULTI_BIT != 0 && self.b == RANGE_SENTINEL
    }

    /// True for normal (non-extended) multi-pair nodes.
    #[inline(always)]
    pub(crate) fn is_multi_normal(&self) -> bool {
        self.b & LEAF_BIT == 0 && self.a & MULTI_BIT != 0 && self.b != RANGE_SENTINEL
    }

    /// Ext table index for an extended multi node. Only valid when `is_multi_ranged()`.
    #[inline(always)]
    pub(crate) fn multi_pairs_idx(&self) -> u32 {
        debug_assert!(self.is_multi_ranged());
        self.a & !MULTI_BIT
    }

    /// Decode the leaf label. Only valid for leaf nodes.
    #[inline(always)]
    pub(crate) fn leaf_label(&self) -> LeafLabel {
        debug_assert!(self.is_leaf());
        match self.a {
            0 => LeafLabel::One,
            1 => LeafLabel::Pos,
            2 => LeafLabel::Neg,
            3 => LeafLabel::Zero,
            _ => unreachable!(),
        }
    }

    /// The pair of an [`is_inline`](Self::is_inline) node.
    #[inline(always)]
    pub fn inline_pair(&self) -> InputPair {
        debug_assert!(self.is_inline());
        InputPair { left: NodeIdx(self.a), right: NodeIdx(self.b) }
    }

    /// Shrink `pair_len` for a **normal** multi-pair node (used during dedup remapping).
    /// Caller must ensure `new_len` >= 2; use `TddNodeData::inline` to convert to inline.
    /// For extended nodes, use `TddLevel::set_pair_len` which updates the side table.
    #[inline(always)]
    pub(crate) fn set_pair_len(&mut self, new_len: u32) {
        debug_assert!(self.is_multi_normal(), "use TddLevel::set_pair_len for extended");
        debug_assert!(new_len >= 2, "use TddNodeData::inline for single-pair conversion");
        debug_assert!(new_len & LEAF_BIT == 0);
        self.b = new_len;
    }
}

impl std::fmt::Debug for TddNodeData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_tombstone() {
            write!(f, "Tombstone")
        } else if self.is_leaf() {
            write!(f, "Leaf({:?})", self.leaf_label())
        } else if self.is_inline() {
            write!(f, "Inline {{ left: {}, right: {} }}", self.a, self.b)
        } else if self.is_multi_ranged() {
            write!(f, "MultiExt {{ multi_pairs_idx: {} }}", self.a & !MULTI_BIT)
        } else {
            write!(f, "Multi {{ pair_start: {}, pair_len: {} }}", self.a & !MULTI_BIT, self.b)
        }
    }
}
