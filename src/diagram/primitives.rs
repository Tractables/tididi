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
    pub fn idx(self) -> usize { self.0 as usize }

    /// Node slots one level can address. A stored pair side reserves bit 31
    /// (`RESERVED_BIT`), so an index carries 31 bits and a level that has
    /// filled them has nowhere to put another node.
    pub(crate) const MAX_LIVE: usize = RESERVED_BIT as usize;

    /// True when the word is a reserved sentinel rather than a reference into
    /// a level.
    ///
    /// Bit 31 is the one test, whatever the side's child level turns out to
    /// be: a structural index is bounded by the level's width and a marginal
    /// side leaves bit 31 clear by construction (see
    /// [`ValueRef`](super::ValueRef)), so only [`ZERO`] and the
    /// scratch words that carry it set it.
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
    pub const fn from_raw(raw: u32) -> Self { Self(raw) }

    /// The encoded word, for persistence and representation inspection.
    pub const fn raw(self) -> u32 { self.0 }

    /// Whether this word is a reserved sentinel rather than a stored child reference.
    pub(crate) fn is_reserved(self) -> bool { self.0 & RESERVED_BIT != 0 }
}

impl From<NodeIdx> for EncodedChildRef {
    /// Encode an untagged node index, value slot, or implicit leaf label.
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

impl ChildPair {
    /// Form an ordered pair of child references, encoding plain indices when supplied.
    #[inline]
    pub fn new(left: impl Into<EncodedChildRef>, right: impl Into<EncodedChildRef>) -> Self {
        Self { left: left.into(), right: right.into() }
    }

    /// The pair as one word, left side high: the key the intern table hashes
    /// and the one a pair list is sorted by, left side first.
    #[inline]
    pub(crate) fn key(self) -> u64 {
        ((self.left.0 as u64) << 32) | self.right.0 as u64
    }
}

/// Bit 31 of a node's `a` word: the node's pairs live in the level's arena.
/// A stored pair side never has bit 31 set, which is what keeps an inline
/// pair's left side apart from this. See the encoding table on [`EncodedNode`].
pub(super) const MULTI_BIT: u32 = 1 << 31;
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
/// A reader never decodes the words itself: [`TddLevel::pairs_of`] and
/// [`TddLevel::pairs_iter_of`] resolve a node to its pairs.
///
/// The two `u32` words carry a three-way encoding:
///
/// ```text
/// ┌───────────────────────────────┬───────────────────────────────┐
/// │         a (u32)               │         b (u32)               │
/// ├───────────────────────────────┼───────────────────────────────┤
/// │ left child index              │ right child index             │  ← inline pair
/// │ pair_start | `MULTI_BIT`      │ pair_len (∈ {0, 2, 3, …})     │  ← normal multi-pair
/// │ multi_pairs_idx | `MULTI_BIT` │ `RANGE_SENTINEL` (= 1)        │  ← extended multi-pair
/// └───────────────────────────────┴───────────────────────────────┘
/// ```
///
/// The cases are tested in that order, hottest first: `a & MULTI_BIT == 0`
/// means inline pair (a stored side never has bit 31 set); else `b == 1`
/// means extended multi-pair (its size lives in the level's `multi_pairs`
/// table); else normal multi-pair. A single pair is always stored inline,
/// so `pair_len == 1` never occurs for multi-pair, which is what leaves
/// `b == 1` free as the extended sentinel. `pair_len == 0` is legal.
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
    pub(crate) a: u32,  // inline: left child; multi: pair_start | MULTI_BIT
    pub(crate) b: u32,  // inline: right child; multi: pair_len
}

/// What an [`EncodedNode`]'s two words encode, as returned by
/// [`EncodedNode::kind`].
///
/// The cases are exactly the rows of the encoding table on [`EncodedNode`].
/// Each carries the payload that case has, so a reader that matches never
/// asks a second question to find out whether the payload it wants is there.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub(crate) enum NodeKind {
    /// One pair, stored in the node's own two words.
    Inline(ChildPair),
    /// Pairs at `[start, start + len)` of the level's `pairs` arena.
    Multi { start: u32, len: u32 },
    /// Pairs whose 64-bit `(start, len)` live in the level's `multi_pairs`
    /// side table at this index.
    MultiRanged(u32),
}

impl NodeKind {
    /// Whether the node's pairs live in the level's `pairs` arena rather than
    /// in the node's own two words.
    pub(crate) fn pairs_in_arena(self) -> bool {
        matches!(self, NodeKind::Multi { .. } | NodeKind::MultiRanged(_))
    }
}

impl EncodedNode {
    /// Create an inline single-pair node. `a` and `b` store the pair's left/right indices.
    /// A side with bit 31 set would alias the multi-pair encoding; no stored
    /// pair has one.
    pub(crate) fn inline(pair: ChildPair) -> Self {
        debug_assert!(
            !pair.left.is_reserved() && !pair.right.is_reserved(),
            "a stored pair side never has bit 31 set"
        );
        EncodedNode { a: pair.left.0, b: pair.right.0 }
    }

    /// Create a normal multi-pair node referencing the pairs arena at
    /// `[pair_start, pair_start+pair_len)`. Both must fit in 31 bits; use
    /// `TddLevel::encode_multi` for arbitrary sizes (it promotes to extended
    /// form when needed). `pair_len` may be 0, for an empty placeholder node.
    pub(crate) fn multi_pair(pair_start: u32, pair_len: u32) -> Self {
        debug_assert!(pair_start & MULTI_BIT == 0, "pair_start too large; use encode_multi");
        debug_assert!(pair_len & MULTI_BIT == 0, "pair_len too large; use encode_multi");
        EncodedNode { a: pair_start | MULTI_BIT, b: pair_len }
    }

    /// Create an extended multi-pair node whose `(start, len)` live in the level's
    /// `multi_pairs` side table at `multi_pairs_idx`. `b = RANGE_SENTINEL` (= 1) distinguishes this
    /// from normal multi (which has `pair_len` ∈ {0, 2, 3, …}).
    pub(crate) fn multi_ranged(multi_pairs_idx: u32) -> Self {
        debug_assert!(multi_pairs_idx & MULTI_BIT == 0, "multi_pairs_idx too large");
        EncodedNode { a: multi_pairs_idx | MULTI_BIT, b: RANGE_SENTINEL }
    }

    /// Decode the two words into the case they encode.
    ///
    /// This is the only reader of the bit layout; everything else matches on
    /// what comes back, so a new case has to be handled at every site.
    pub(crate) fn kind(&self) -> NodeKind {
        if self.a & MULTI_BIT == 0 {
            NodeKind::Inline(ChildPair::new(
                EncodedChildRef::from_raw(self.a),
                EncodedChildRef::from_raw(self.b),
            ))
        } else if self.b == RANGE_SENTINEL {
            NodeKind::MultiRanged(self.a & !MULTI_BIT)
        } else {
            NodeKind::Multi { start: self.a & !MULTI_BIT, len: self.b }
        }
    }

    /// True for a node with pairs, which every stored node is.
    pub fn is_internal(&self) -> bool { true }
}

impl std::fmt::Debug for EncodedNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.kind() {
            NodeKind::Inline(_) => write!(f, "Inline {{ left: {}, right: {} }}", self.a, self.b),
            NodeKind::MultiRanged(idx) => write!(f, "MultiExt {{ multi_pairs_idx: {idx} }}"),
            NodeKind::Multi { start, len } => {
                write!(f, "Multi {{ pair_start: {start}, pair_len: {len} }}")
            }
        }
    }
}

/// The input pairs of one node, as yielded by [`TddLevel::pairs_iter_of`]
/// and [`TddLevel::internal_inputs_iter`].
///
/// Yields owned [`ChildPair`]s in storage order, which carries no meaning
/// (a node is the set of its pairs). Implements [`ExactSizeIterator`], so
/// `len()` is the node's pair count.
///
/// [`TddLevel::pairs_iter_of`]: super::TddLevel::pairs_iter_of
/// [`TddLevel::internal_inputs_iter`]: super::TddLevel::internal_inputs_iter
#[derive(Clone)]
pub struct PairsIter<'a>(PairStorage<'a>);

#[derive(Clone)]
enum PairStorage<'a> {
    Inline(Option<ChildPair>),
    Slice(std::slice::Iter<'a, ChildPair>),
}

impl<'a> PairsIter<'a> {
    #[inline]
    pub(super) fn inline(pair: ChildPair) -> Self {
        PairsIter(PairStorage::Inline(Some(pair)))
    }

    #[inline]
    pub(super) fn slice(pairs: &'a [ChildPair]) -> Self {
        PairsIter(PairStorage::Slice(pairs.iter()))
    }
}

impl std::fmt::Debug for PairsIter<'_> {
    /// How many pairs are still to come, which is all an iterator's state is.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let remaining = match &self.0 {
            PairStorage::Inline(opt) => usize::from(opt.is_some()),
            PairStorage::Slice(iter) => iter.len(),
        };
        f.debug_struct("PairsIter").field("remaining", &remaining).finish()
    }
}

impl<'a> Iterator for PairsIter<'a> {
    type Item = ChildPair;
    #[inline]
    fn next(&mut self) -> Option<ChildPair> {
        match &mut self.0 {
            PairStorage::Inline(opt) => opt.take(),
            PairStorage::Slice(iter) => iter.next().copied(),
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = match &self.0 {
            PairStorage::Inline(Some(_)) => 1,
            PairStorage::Inline(None) => 0,
            PairStorage::Slice(iter) => iter.len(),
        };
        (n, Some(n))
    }
}

impl<'a> ExactSizeIterator for PairsIter<'a> {}
