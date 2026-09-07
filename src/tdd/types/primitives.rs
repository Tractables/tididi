//! Primitive node types: `LocalNodeIdx`, `TddNodeId`, `LeafLabel`, `InputPair`,
//! `TddNodeData`, `ExtMulti`, and related constants.

use crate::vtree::VtreeIdx;

/// Index of a node within one level.
///
/// On a leaf level the three implicit nodes are `0..LEAF_WIDTH`
/// ([`ONE_LEAF_IDX`], [`POS_LEAF_IDX`], [`NEG_LEAF_IDX`]); on a structural
/// level it indexes `nodes`; on a marginal level it indexes `marginal_counts`.
/// In a pair whose child level is marginal the raw `u32` is a tagged reference,
/// not a plain index — see [`resolve_marg_ref`](super::resolve_marg_ref).
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug, Ord, PartialOrd)]
#[repr(transparent)]  // guaranteed same layout as bare u32 (no padding/tag)
pub struct LocalNodeIdx(pub u32);

impl LocalNodeIdx {
    /// The index as a `usize`.
    #[inline(always)]
    pub fn idx(self) -> usize { self.0 as usize }
}

/// Sentinel index for the constant-false function.
///
/// Appears only in [`Tdd::output`](super::Tdd::output) (the diagram is
/// unsatisfiable; [`Tdd::is_zero`](super::Tdd::is_zero)), never in a stored
/// pair: no stored node computes false.
pub const ZERO: LocalNodeIdx = LocalNodeIdx(u32::MAX);

/// A node of the diagram: its level (a vtree node) and its index in that level.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct TddNodeId {
    /// The vtree node whose level holds the node.
    pub vtree: VtreeIdx,
    /// Index of the node within that level.
    pub local: LocalNodeIdx,
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
pub const ONE_LEAF_IDX: LocalNodeIdx = LocalNodeIdx(LeafLabel::One as u32);
/// Local index of the positive-literal node on a leaf level.
pub const POS_LEAF_IDX: LocalNodeIdx = LocalNodeIdx(LeafLabel::Pos as u32);
/// Local index of the negative-literal node on a leaf level.
pub const NEG_LEAF_IDX: LocalNodeIdx = LocalNodeIdx(LeafLabel::Neg as u32);

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
/// through [`resolve_marg_ref`](super::resolve_marg_ref). A node's pairs are
/// unordered and pairwise disjoint as functions.
///
/// `#[repr(C)]`: a single-pair node stores its pair in the two `u32` words of
/// [`TddNodeData`] and reads it back by pointer cast.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug, Ord, PartialOrd)]
#[repr(C)]
pub struct InputPair {
    /// Node in the left child level.
    pub left: LocalNodeIdx,
    /// Node in the right child level.
    pub right: LocalNodeIdx,
}

/// Bytes ONE input pair occupies in a diagram — the per-pair storage size, and
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
pub const INPUT_PAIR_BYTES: usize = size_of::<InputPair>();

impl InputPair {
    /// Whether this pair can be stored inline in a `TddNodeData` node without
    /// aliasing the leaf or `multi_extended` encoding.
    #[inline]
    pub fn can_inline(&self) -> bool {
        self.right.0 & LEAF_BIT == 0 && self.left.0 & MULTI_BIT == 0
    }
}

/// Packed TDD node data: leaf label, inline single pair, or multi-pair arena reference.
///
/// Each node is exactly 8 bytes (two u32 fields `a` and `b`) with a four-way encoding:
///
/// ```text
/// ┌──────────────────────────────┬──────────────────────────────┐
/// │         a (u32)              │         b (u32)              │
/// ├──────────────────────────────┼──────────────────────────────┤
/// │ LeafLabel as u32             │ LEAF_BIT (1<<31)             │  ← leaf
/// │ left child index             │ right child index            │  ← inline pair
/// │ pair_start | MULTI_BIT       │ pair_len (∈ {0, 2, 3, …})    │  ← normal multi-pair
/// │ ext_idx     | MULTI_BIT      │ EXT_SENTINEL (= 1)           │  ← extended multi-pair
/// └──────────────────────────────┴──────────────────────────────┘
/// ```
///
/// Decoding stays cheap: `is_leaf = b & LEAF_BIT != 0` (one check, hot path).
///   - `b & LEAF_BIT != 0` → leaf
///   - else `a & MULTI_BIT == 0` → inline pair
///   - else `b == 1` → extended multi-pair (size lives in `level.ext[ext_idx]`)
///   - else → normal multi-pair
///
/// `LEAF_BIT == MULTI_BIT == 1 << 31`. `pair_len == 1` is forbidden for multi-pair
/// (caller converts to inline), so `b == 1` is a free sentinel. `pair_len == 0` is
/// legal (used by full.rs for empty placeholders).
///
/// **Inline pairs** (60–95% of nodes) store a single `InputPair` directly in `(a, b)`.
/// Since `TddNodeData` and `InputPair` are both `#[repr(C)]` with identical `{u32, u32}`
/// layout, `pairs_of_idx` returns `&[InputPair; 1]` via a raw pointer cast (zero overhead).
///
/// **Multi-pair** nodes reference a contiguous slice in the level's `pairs` arena. The
/// normal form packs `(pair_start, pair_len)` into `(a & !MULTI_BIT, b)` when both fit
/// in 31 bits. When either exceeds 2^31 (huge product grids on pathological CNFs), the
/// node is stored in extended form: `a` indexes a side table `level.ext` that holds
/// u64 start/len, avoiding an unconditional 2× memory cost on every node.
pub(super) const LEAF_BIT: u32 = 1 << 31;
pub(super) const MULTI_BIT: u32 = 1 << 31;
/// Sentinel `b` for a **tombstone**: a dead node slot that survives in `nodes`
/// instead of being compacted out (Tier 2 index-stable conjoin). Chosen as
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
/// count-only `total_nodes`/`max_width` (via `TddLevel::live_width`).
pub(super) const TOMBSTONE_B: u32 = LEAF_BIT | 1;
/// Sentinel value for `b` that marks an extended multi-pair node (side-table form).
/// Chosen as 1 because `pair_len` == 1 is forbidden for multi (caller uses inline),
/// so 1 cannot appear as a legitimate normal-multi `pair_len`.
pub(super) const EXT_SENTINEL: u32 = 1;

/// Side-table entry for extended multi-pair nodes (`pair_start` or `pair_len` ≥ 2^31).
/// The node data holds `(a = ext_idx | MULTI_BIT, b = EXT_SENTINEL)`, and this struct
/// holds the actual start/len. Only allocated when the 31-bit encoding would overflow.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub struct ExtMulti {
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
/// Encoding (`LEAF_BIT`/`MULTI_BIT` table above): a single-pair node holds its
/// pair in the two words ([`is_inline`](Self::is_inline)); a multi-pair node
/// references a range of the level's `pairs` arena ([`is_multi`](Self::is_multi)).
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
    pub fn leaf(label: LeafLabel) -> Self {
        TddNodeData { a: label as u32, b: LEAF_BIT }
    }

    /// Create an inline single-pair node. `a` and `b` store the pair's left/right indices.
    /// Caller MUST verify `pair.can_inline()` — violating this aliases the leaf or
    /// `multi_extended` encoding and causes silent data corruption.
    #[inline(always)]
    pub fn inline(pair: InputPair) -> Self {
        debug_assert!(pair.can_inline(), "pair cannot be inlined: would alias leaf/multi_extended encoding");
        TddNodeData { a: pair.left.0, b: pair.right.0 }
    }

    /// Create a normal multi-pair node referencing the pairs arena at
    /// `[pair_start, pair_start+pair_len)`. Both must fit in 31 bits; use
    /// `TddLevel::encode_multi` for arbitrary sizes (it promotes to extended
    /// form when needed). `pair_len` may be 0 (used by full.rs for empty placeholders).
    #[inline(always)]
    pub fn multi_pair(pair_start: u32, pair_len: u32) -> Self {
        debug_assert!(pair_start & MULTI_BIT == 0, "pair_start too large; use encode_multi");
        debug_assert!(pair_len & LEAF_BIT == 0, "pair_len overflow; use encode_multi");
        TddNodeData { a: pair_start | MULTI_BIT, b: pair_len }
    }

    /// Create an extended multi-pair node whose `(start, len)` live in the level's
    /// `ext` side table at `ext_idx`. `b = EXT_SENTINEL` (= 1) distinguishes this
    /// from normal multi (which has `pair_len` ∈ {0, 2, 3, …}).
    #[inline(always)]
    pub fn multi_extended(ext_idx: u32) -> Self {
        debug_assert!(ext_idx & MULTI_BIT == 0, "ext_idx too large");
        TddNodeData { a: ext_idx | MULTI_BIT, b: EXT_SENTINEL }
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
    pub fn tombstone() -> Self { TddNodeData { a: u32::MAX, b: TOMBSTONE_B } }

    /// True for a dead slot left in place by an index-stable rewrite. It is
    /// referenced by no pair; `width()` still counts it, `live_width()` does
    /// not, and `minimize` removes it.
    #[inline(always)]
    pub fn is_tombstone(&self) -> bool { self.b == TOMBSTONE_B }

    /// True for a node with exactly one pair, stored in the node word.
    #[inline(always)]
    pub fn is_inline(&self) -> bool { self.b & LEAF_BIT == 0 && self.a & MULTI_BIT == 0 }

    /// Test-support: overwrite an INLINE node's left-child index, deliberately
    /// corrupting the node. Used by downstream corruption-detection tests
    /// (`tests/tdd_invariants_compile.rs`) that need to flip a child ref
    /// without exposing the raw `a`/`b` encoding fields. Panics if the node is
    /// not inline. (public-release P3a)
    #[doc(hidden)]
    pub fn corrupt_inline_left_for_test(&mut self, left: u32) {
        assert!(self.is_inline(), "corrupt_inline_left_for_test: node is not inline");
        self.a = left;
    }

    /// True for a node whose pairs live in the level's `pairs` arena.
    #[inline(always)]
    pub fn is_multi(&self) -> bool { self.b & LEAF_BIT == 0 && self.a & MULTI_BIT != 0 }

    /// True for extended multi-pair nodes (start/len live in `level.ext`).
    /// Disambiguated by `b == 1` — impossible for normal multi since `pair_len` == 1
    /// is forbidden (caller uses inline).
    #[inline(always)]
    pub fn is_multi_extended(&self) -> bool {
        self.a & MULTI_BIT != 0 && self.b == EXT_SENTINEL
    }

    /// True for normal (non-extended) multi-pair nodes.
    #[inline(always)]
    pub fn is_multi_normal(&self) -> bool {
        self.b & LEAF_BIT == 0 && self.a & MULTI_BIT != 0 && self.b != EXT_SENTINEL
    }

    /// Ext table index for an extended multi node. Only valid when `is_multi_extended()`.
    #[inline(always)]
    pub fn ext_idx(&self) -> u32 {
        debug_assert!(self.is_multi_extended());
        self.a & !MULTI_BIT
    }

    /// Decode the leaf label. Only valid for leaf nodes.
    #[inline(always)]
    pub fn leaf_label(&self) -> LeafLabel {
        debug_assert!(self.is_leaf());
        match self.a {
            0 => LeafLabel::One,
            1 => LeafLabel::Pos,
            2 => LeafLabel::Neg,
            3 => LeafLabel::Zero,
            _ => unreachable!(),
        }
    }

    /// Start offset in the pairs arena. Only valid for **normal** multi-pair nodes;
    /// extended nodes must be queried via `TddLevel::multi_start_at`.
    #[inline(always)]
    pub fn multi_start(&self) -> u32 {
        debug_assert!(self.is_multi_normal(), "use TddLevel::multi_start_at for extended");
        self.a & !MULTI_BIT
    }

    /// Number of pairs in the arena slice. Only valid for **normal** multi-pair nodes;
    /// extended nodes must be queried via `TddLevel::multi_len_at`.
    #[inline(always)]
    pub fn multi_len(&self) -> u32 {
        debug_assert!(self.is_multi_normal(), "use TddLevel::multi_len_at for extended");
        self.b
    }

    /// Number of pairs for any internal node (1 for inline, `b` for normal multi).
    /// Only valid for inline and normal multi; extended nodes must be queried via
    /// `TddLevel::pair_count_at`.
    #[inline(always)]
    pub fn pair_count(&self) -> u32 {
        debug_assert!(self.is_internal());
        debug_assert!(!self.is_multi_extended(), "use TddLevel::pair_count_at for extended");
        if self.is_inline() { 1 } else { self.b }
    }

    /// The pair of an [`is_inline`](Self::is_inline) node.
    #[inline(always)]
    pub fn inline_pair(&self) -> InputPair {
        debug_assert!(self.is_inline());
        InputPair { left: LocalNodeIdx(self.a), right: LocalNodeIdx(self.b) }
    }

    /// Returns the range into the level's pairs arena. Only valid for **normal**
    /// multi-pair nodes; extended nodes must be queried via `TddLevel::pair_range_at`.
    #[inline(always)]
    pub fn pair_range(&self) -> std::ops::Range<usize> {
        debug_assert!(self.is_multi_normal(), "use TddLevel::pair_range_at for extended");
        let start = (self.a & !MULTI_BIT) as usize;
        start..start + self.b as usize
    }

    /// Shrink `pair_len` for a **normal** multi-pair node (used during dedup remapping).
    /// Caller must ensure `new_len` >= 2; use `TddNodeData::inline` to convert to inline.
    /// For extended nodes, use `TddLevel::set_pair_len` which updates the side table.
    #[inline(always)]
    pub fn set_pair_len(&mut self, new_len: u32) {
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
        } else if self.is_multi_extended() {
            write!(f, "MultiExt {{ ext_idx: {} }}", self.a & !MULTI_BIT)
        } else {
            write!(f, "Multi {{ pair_start: {}, pair_len: {} }}", self.a & !MULTI_BIT, self.b)
        }
    }
}
