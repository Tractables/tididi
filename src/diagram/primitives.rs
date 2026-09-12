//! Primitive node types: `NodeIdx`, `TddNodeId`, `LeafLabel`, `ChildPair`,
//! `EncodedNode`, `MultiPairRange`, and related constants.

use crate::vtree::VtreeIdx;

/// Index of a node or value slot within one level, or an implicit leaf label.
///
/// Pair sides use [`EncodedChildRef`]; decode them through
/// [`ChildDecoder`](super::ChildDecoder) before indexing storage.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug, Ord, PartialOrd)]
#[repr(transparent)]  // guaranteed same layout as bare u32 (no padding/tag)
pub struct NodeIdx(pub u32);

impl NodeIdx {
    /// The index as a `usize`.
    #[inline(always)]
    pub fn idx(self) -> usize { self.0 as usize }

    /// True when the word is a reserved sentinel rather than a reference into
    /// a level.
    ///
    /// Bit 31 is the one test, whatever the side's child level turns out to
    /// be: a structural index is bounded by the level's width and a marginal
    /// side leaves bit 31 clear by construction (see
    /// [`MarginalSide`](super::MarginalSide)), so only [`ZERO`] and the
    /// scratch words that carry it set it.
    #[inline(always)]
    pub(crate) fn is_reserved(self) -> bool {
        self.0 & RESERVED_BIT != 0
    }
}

/// Bit 31 of a stored side: the word is a reserved sentinel, not an index.
pub(crate) const RESERVED_BIT: u32 = 1 << 31;

/// Sentinel index for the constant-false function.
///
/// Appears only in [`Tdd::output`](super::Tdd::output) (the diagram is
/// unsatisfiable; [`Tdd::is_zero`](super::Tdd::is_zero)), never in a stored
/// pair: no stored node computes false. It is the one value with bit 31 set
/// that a side can hold, which is the test `NodeIdx::is_reserved` makes.
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
/// and a pair pointing at index `i` of a leaf level denotes the label whose
/// discriminant is `i`. `One` is index 0 so that the constant-true node
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
/// [`Tdd::reference_slot_count`](super::Tdd::reference_slot_count) reports there.
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
    pub(crate) fn from_idx(i: usize) -> LeafLabel {
        match i {
            0 => LeafLabel::One,
            1 => LeafLabel::Pos,
            2 => LeafLabel::Neg,
            _ => panic!("LeafLabel::from_idx: {i} is not below the leaf width {}", LEAF_WIDTH),
        }
    }
}

/// An encoded child reference whose interpretation is determined by its child level.
///
/// A word may encode a node index, a marginal value slot, or an inline count.
/// [`ChildDecoder`](super::ChildDecoder) distinguishes these cases; the raw word
/// is not a storage index. The transparent layout preserves the stored pair format.
///
/// ```compile_fail
/// use tididi::diagram::EncodedChildRef;
/// let encoded = EncodedChildRef::from_raw(7);
/// let index = encoded.idx(); // Decode the child reference before indexing.
/// ```
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug, Ord, PartialOrd)]
#[repr(transparent)]
pub struct EncodedChildRef(pub(crate) u32);

impl EncodedChildRef {
    /// Preserve a raw word; its validity is checked in the context of a child level.
    #[inline(always)]
    pub const fn from_raw(raw: u32) -> Self { Self(raw) }

    /// The encoded word, for persistence and representation inspection.
    #[inline(always)]
    pub const fn raw(self) -> u32 { self.0 }

    /// Whether this word is a reserved sentinel rather than a stored child reference.
    #[inline(always)]
    pub(crate) fn is_reserved(self) -> bool { self.0 & RESERVED_BIT != 0 }
}

impl From<NodeIdx> for EncodedChildRef {
    /// Encode an untagged node index, value slot, or implicit leaf label.
    #[inline(always)]
    fn from(index: NodeIdx) -> Self { Self(index.0) }
}

/// One input of a structural node: references to its left and right children.
///
/// Each side is an encoded reference and never the [`ZERO`] sentinel.
/// Node indices and value slots are bounded by the child level's reference
/// slots; inline values do not index storage. Read each side through
/// [`ChildDecoder::child`](super::ChildDecoder::child). A node's pairs are
/// unordered and pairwise disjoint as functions.
///
/// `#[repr(C)]`: a single-pair node stores its pair in the two `u32` words of
/// [`EncodedNode`] and reads it back by pointer cast.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug, Ord, PartialOrd)]
#[repr(C)]
pub struct ChildPair {
    /// Encoded reference into the left child level.
    pub left: EncodedChildRef,
    /// Encoded reference into the right child level.
    pub right: EncodedChildRef,
}

/// Bytes one input pair occupies in a diagram, the unit `Tdd::pair_count()` counts
/// in. The same under both encodings: a single-pair node stores its pair in
/// the two `u32` fields of `EncodedNode`.
pub(crate) const CHILD_PAIR_BYTES: usize = size_of::<ChildPair>();

impl ChildPair {
    /// Form an ordered pair of child references, encoding plain indices when supplied.
    #[inline]
    pub fn new(left: impl Into<EncodedChildRef>, right: impl Into<EncodedChildRef>) -> Self {
        Self { left: left.into(), right: right.into() }
    }

    /// Whether this pair can be stored inline in an `EncodedNode` without
    /// aliasing the leaf or `multi_ranged` encoding.
    #[inline]
    pub(crate) fn can_inline(&self) -> bool {
        self.right.0 & LEAF_BIT == 0 && self.left.0 & MULTI_BIT == 0
    }
}

/// Bit 31 of a node's `b` word: the node stores a leaf label, not pairs.
/// See the encoding table on [`EncodedNode`].
pub(super) const LEAF_BIT: u32 = 1 << 31;
/// Bit 31 of a node's `a` word: the node's pairs live in the level's arena.
/// See the encoding table on [`EncodedNode`].
pub(super) const MULTI_BIT: u32 = 1 << 31;
/// Sentinel `b` for a tombstone: a dead, unreferenced node slot left in
/// `nodes` by an index-stable rewrite. `LEAF_BIT | 1` collides with no live
/// encoding (a real leaf has `b == LEAF_BIT`; an internal node has bit 31
/// clear), and bit 31 set makes every `is_internal()`-gated reader skip it.
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
    pub(crate) start: u64,
    pub(crate) len: u64,
}

/// A stored node: 8 bytes encoding where its pairs live.
///
/// A reader never decodes the word itself: [`TddLevel::pairs_of`] and
/// [`TddLevel::pairs_iter_of`] resolve a node to its pairs, and
/// [`is_internal`](Self::is_internal) classifies it. Every node of a valid
/// diagram is internal or a tombstone (a dead slot, unreferenced, that
/// `minimize` removes).
///
/// The two `u32` words carry a four-way encoding:
///
/// ```text
/// ┌───────────────────────────────┬───────────────────────────────┐
/// │         a (u32)               │         b (u32)               │
/// ├───────────────────────────────┼───────────────────────────────┤
/// │ LeafLabel as u32              │ `LEAF_BIT` (1<<31)            │  ← leaf
/// │ left child index              │ right child index             │  ← inline pair
/// │ pair_start | `MULTI_BIT`      │ pair_len (∈ {0, 2, 3, …})     │  ← normal multi-pair
/// │ multi_pairs_idx | `MULTI_BIT` │ `RANGE_SENTINEL` (= 1)        │  ← extended multi-pair
/// └───────────────────────────────┴───────────────────────────────┘
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
/// [`ChildPair`] directly in `(a, b)`. `EncodedNode` and `ChildPair` are both
/// `#[repr(C)]` over the same two `u32`s, so `pairs_of` hands back
/// `&[ChildPair; 1]` by pointer cast rather than copying.
///
/// **Multi-pair** nodes name a contiguous range of the level's `pairs` arena.
/// The normal form packs `(pair_start, pair_len)` into the two words when both
/// fit in 31 bits; past that the node goes to the extended form, whose `u64`
/// start and length live in the level's side table.
///
/// [`TddLevel::pairs_of`]: super::TddLevel::pairs_of
/// [`TddLevel::pairs_iter_of`]: super::TddLevel::pairs_iter_of
#[derive(Copy, Clone, Eq, PartialEq)]
#[repr(C)]
pub struct EncodedNode {
    pub(crate) a: u32,  // leaf: label; inline: left child; multi: pair_start | MULTI_BIT
    pub(crate) b: u32,  // leaf: LEAF_BIT; inline: right child; multi: pair_len
}

impl EncodedNode {
    /// Create an inline single-pair node. `a` and `b` store the pair's left/right indices.
    /// Caller must verify `pair.can_inline()` — violating this aliases the leaf or
    /// `multi_ranged` encoding and causes silent data corruption.
    #[inline(always)]
    pub(crate) fn inline(pair: ChildPair) -> Self {
        debug_assert!(pair.can_inline(), "pair cannot be inlined: would alias leaf/multi_ranged encoding");
        EncodedNode { a: pair.left.0, b: pair.right.0 }
    }

    /// Create a normal multi-pair node referencing the pairs arena at
    /// `[pair_start, pair_start+pair_len)`. Both must fit in 31 bits; use
    /// `TddLevel::encode_multi` for arbitrary sizes (it promotes to extended
    /// form when needed). `pair_len` may be 0, for an empty placeholder node.
    #[inline(always)]
    pub(crate) fn multi_pair(pair_start: u32, pair_len: u32) -> Self {
        debug_assert!(pair_start & MULTI_BIT == 0, "pair_start too large; use encode_multi");
        debug_assert!(pair_len & LEAF_BIT == 0, "pair_len overflow; use encode_multi");
        EncodedNode { a: pair_start | MULTI_BIT, b: pair_len }
    }

    /// Create an extended multi-pair node whose `(start, len)` live in the level's
    /// `multi_pairs` side table at `multi_pairs_idx`. `b = RANGE_SENTINEL` (= 1) distinguishes this
    /// from normal multi (which has `pair_len` ∈ {0, 2, 3, …}).
    #[inline(always)]
    pub(crate) fn multi_ranged(multi_pairs_idx: u32) -> Self {
        debug_assert!(multi_pairs_idx & MULTI_BIT == 0, "multi_pairs_idx too large");
        EncodedNode { a: multi_pairs_idx | MULTI_BIT, b: RANGE_SENTINEL }
    }

    /// True when the node holds no pairs: a tombstone, or a leaf-label node
    /// (which no valid diagram stores).
    #[inline(always)]
    pub(crate) fn is_leaf(&self) -> bool { self.b & LEAF_BIT != 0 }

    /// True for a node with pairs (inline or multi-pair).
    #[inline(always)]
    pub fn is_internal(&self) -> bool { self.b & LEAF_BIT == 0 }

    /// True for a dead slot left in place by an index-stable rewrite. It is
    /// referenced by no pair; `slot_count()` still counts it, `live_slot_count()` does
    /// not, and `minimize` removes it.
    #[inline(always)]
    pub(crate) fn is_tombstone(&self) -> bool { self.b == TOMBSTONE_B }

    /// True for a node with exactly one pair, stored in the node word.
    #[inline(always)]
    pub(crate) fn is_inline(&self) -> bool { self.b & LEAF_BIT == 0 && self.a & MULTI_BIT == 0 }

    /// True for a node whose pairs live in the level's `pairs` arena.
    #[inline(always)]
    pub(crate) fn is_multi(&self) -> bool { self.b & LEAF_BIT == 0 && self.a & MULTI_BIT != 0 }

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
        if self.a == LeafLabel::Zero as u32 {
            LeafLabel::Zero
        } else {
            LeafLabel::from_idx(self.a as usize)
        }
    }

    /// The pair of an [`is_inline`](Self::is_inline) node.
    #[inline(always)]
    pub(crate) fn inline_pair(&self) -> ChildPair {
        debug_assert!(self.is_inline());
        ChildPair::new(EncodedChildRef::from_raw(self.a), EncodedChildRef::from_raw(self.b))
    }

    /// Shrink `pair_len` for a **normal** multi-pair node (used during dedup remapping).
    /// Caller must ensure `new_len` >= 2; use `EncodedNode::inline` to convert to inline.
    /// For extended nodes, use `TddLevel::set_pair_len` which updates the side table.
    #[inline(always)]
    pub(crate) fn set_pair_len(&mut self, new_len: u32) {
        debug_assert!(self.is_multi_normal(), "use TddLevel::set_pair_len for extended");
        debug_assert!(new_len >= 2, "use EncodedNode::inline for single-pair conversion");
        debug_assert!(new_len & LEAF_BIT == 0);
        self.b = new_len;
    }
}

impl std::fmt::Debug for EncodedNode {
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
