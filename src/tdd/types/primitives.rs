//! Primitive node types: `LocalNodeIdx`, `TddNodeId`, `LeafLabel`, `InputPair`,
//! `TddNodeData`, `ExtMulti`, and related constants.

use crate::vtree::VtreeIdx;

/// Index of a node within a vtree level's node list.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug, Ord, PartialOrd)]
#[repr(transparent)]  // guaranteed same layout as bare u32 (no padding/tag)
#[doc(hidden)]
pub struct LocalNodeIdx(pub u32);

impl LocalNodeIdx {
    /// The index as a `usize`.
    #[inline(always)]
    pub fn idx(self) -> usize { self.0 as usize }
}

/// Sentinel index representing the constant-false (⊥) function (UNSAT).
///
/// This is a **virtual node** — never stored in any level's `nodes` Vec.
/// When `Tdd::output.local == ZERO`, the TDD computes the constant-false
/// function (the formula is unsatisfiable).
///
/// **Key invariant:** no real node computes false. `Leaf(Zero)` and
/// `Internal { pair_len: 0 }` never appear in any level's node list.
/// The constant-false function is represented *only* via this sentinel
/// in the output field. Enforced by `check_no_false_nodes()` in tests.
#[doc(hidden)]
pub const ZERO: LocalNodeIdx = LocalNodeIdx(u32::MAX);

/// Global node identifier: vtree position + local index.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
#[doc(hidden)]
pub struct TddNodeId {
    /// Vtree level (node) containing the referenced node.
    pub vtree: VtreeIdx,
    /// Index of the node within that level.
    pub local: LocalNodeIdx,
}

/// Label for a leaf-level TDD node.
///
/// `One = 0` so that the constant-true child is at local index 0 on every
/// level — leaf or internal. This eliminates the leaf-vs-internal encoding
/// asymmetry that previously made mid-compile rotation search unsound: a
/// rotation that flips a child between leaf and internal would have changed
/// the "constant ONE child" index from 2 (`ONE_LEAF_IDX`) to 0 (or vice
/// versa), invalidating sibling TDDs' pair encodings.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
#[repr(u32)]
#[doc(hidden)]
pub enum LeafLabel {
    /// Constant true: satisfied by any assignment.
    One = 0,
    /// Positive literal (x): satisfied when x=1.
    Pos = 1,
    /// Negative literal (¬x): satisfied when x=0.
    Neg = 2,
    /// Constant false: never satisfied. Sentinel only — never stored.
    Zero = 3,
}

/// Number of implicit leaf labels (One, Pos, Neg). Every leaf level has exactly
/// this many virtual nodes, indexed `0..LEAF_WIDTH`. Zero is excluded (sentinel only).
#[doc(hidden)]
pub const LEAF_WIDTH: usize = 3;

/// Implicit index of the One (constant-true) leaf label.
/// Equal to `LocalNodeIdx(0)` — same as the constant-true representative on
/// internal levels, so ONE-chain pair encodings are uniform across the vtree.
#[doc(hidden)]
pub const ONE_LEAF_IDX: LocalNodeIdx = LocalNodeIdx(LeafLabel::One as u32);
/// Implicit index of the Pos (positive literal) leaf label.
#[doc(hidden)]
pub const POS_LEAF_IDX: LocalNodeIdx = LocalNodeIdx(LeafLabel::Pos as u32);
/// Implicit index of the Neg (negative literal) leaf label.
#[doc(hidden)]
pub const NEG_LEAF_IDX: LocalNodeIdx = LocalNodeIdx(LeafLabel::Neg as u32);

impl LeafLabel {
    /// Convert a leaf node index (0..3) to its label. Implicit leaf levels use
    /// the index as the label: 0=One, 1=Pos, 2=Neg.
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

/// A pair of child node indices that form an input to an internal TDD node.
/// Left child is from the vtree's left subtree, right from the right subtree.
///
/// `#[repr(C)]` is required for the inline-pair encoding in `TddNodeData`: when a
/// node stores exactly one pair, the two u32 fields of `TddNodeData` hold the pair's
/// left and right indices directly (same memory layout, cast via raw pointer).
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug, Ord, PartialOrd)]
#[repr(C)]
#[doc(hidden)]
pub struct InputPair {
    /// Left child index (from the vtree's left subtree).
    pub left: LocalNodeIdx,
    /// Right child index (from the vtree's right subtree).
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

/// Packed 8-byte TDD node: leaf label, inline single pair, or multi-pair arena
/// reference. See the encoding table on the `LEAF_BIT`/`MULTI_BIT` constants
/// above for the four-way `(a, b)` layout.
#[derive(Copy, Clone, Eq, PartialEq)]
#[repr(C)]
#[doc(hidden)]
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

    /// True when the node is a leaf. Also true for tombstones (bit 31 of `b` set);
    /// use `is_tombstone` to distinguish.
    #[inline(always)]
    pub fn is_leaf(&self) -> bool { self.b & LEAF_BIT != 0 }

    /// True for internal (inline or multi-pair) nodes.
    #[inline(always)]
    pub fn is_internal(&self) -> bool { self.b & LEAF_BIT == 0 }

    /// A dead node slot kept in place (not compacted) by the index-stable
    /// conjoin. `a = u32::MAX` is a tripwire: it is not a valid leaf label, so
    /// `leaf_label()` panics in debug if a tombstone is ever mistaken for a real
    /// leaf. See `TOMBSTONE_B`.
    #[inline(always)]
    pub fn tombstone() -> Self { TddNodeData { a: u32::MAX, b: TOMBSTONE_B } }

    /// True for tombstone slots. Note `is_leaf()` is also true for a tombstone
    /// (bit 31 of `b` is set) and `is_internal()` is false — readers that must
    /// distinguish a tombstone from a real leaf check this first.
    #[inline(always)]
    pub fn is_tombstone(&self) -> bool { self.b == TOMBSTONE_B }

    /// True for inline single-pair nodes (the common case after `apply_and`).
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

    /// True for multi-pair nodes (both normal and extended) that reference the pairs arena.
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

    /// Return the inline pair. Only valid for inline nodes.
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
