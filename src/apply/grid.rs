//! Per-level grid state for `apply_and`'s product construction.
//!
//! The product grid is a single flat `Vec<u32>` shared across vtree levels
//! (see `conjoin::SCRATCH_NODE_IDX`). This module names the four possible
//! states of each level's slice of that arena — base offset, allocated-or-not,
//! and the ordering guarantee the producer gives — and folds them into one
//! descriptor instead of parallel arrays with sentinel values.
//!
//! `DenseStrict` is the only variant whose row-major live-cell values are
//! strictly monotone. Nothing consumes that guarantee at present (no emit site
//! needs a particular pair order); it is recorded as cheap producer metadata.
//! `DenseWeak` and `Leaf` look identical at runtime (both have an allocated
//! `base`) but carry different invariants, so keeping them separate stops a
//! future caller from treating a scattered grid as ordered.

/// Per-level descriptor for the product grid's slice of the flat arena.
///
/// Four mutually exclusive states, one per producer in `apply_and`:
///
/// | Variant        | Producer                                 | Monotonicity |
/// | -------------- | ---------------------------------------- | ------------ |
/// | `Sparse`       | Sparse scatter pipeline (no grid at all) | N/A          |
/// | `Leaf`         | Leaf fill from `CONJOIN_GRID`            | Non-monotone |
/// | `DenseWeak`    | `ensure_grid` scatter from product list  | Weak at best |
/// | `DenseStrict`  | Sequential emit: dense internal, identity shortcuts | **Strict**   |
///
/// An order-sensitive consumer would be admissible **only if both child levels
/// are `DenseStrict`**; none exists today.
#[derive(Clone, Copy, Debug)]
pub(super) enum LevelGrid {
    /// No grid allocated; consumers must go through `product_lists[t]`.
    Sparse,
    /// Grid populated by leaf `CONJOIN_GRID` lookup. Read-only for consumers.
    Leaf { base: usize },
    /// Grid materialised by `ensure_grid` scatter from a product list.
    /// Row-major live-cell order comes from the scatter, not a sequential
    /// emit — not strictly monotone.
    DenseWeak { base: usize },
    /// Grid filled by sequential `level.nodes.len()` assignments in row-major
    /// `(i, j)` order over live cells. The resulting live-cell values are
    /// strictly monotonically increasing — the strongest ordering guarantee any
    /// producer offers.
    DenseStrict { base: usize },
}

impl LevelGrid {
    /// Base offset into the flat `node_idx` arena, or `None` if unallocated.
    #[inline]
    pub fn base(&self) -> Option<usize> {
        match self {
            LevelGrid::Sparse => None,
            LevelGrid::Leaf { base }
            | LevelGrid::DenseWeak { base }
            | LevelGrid::DenseStrict { base } => Some(*base),
        }
    }

    /// Base offset, asserting the grid is allocated. Use at call sites that
    /// have just proven allocation (e.g. after `ensure_grid`).
    #[inline]
    pub(crate) fn base_unchecked(&self) -> usize {
        self.base().expect("expected allocated grid, found Sparse")
    }

    #[inline]
    pub(crate) fn is_sparse(&self) -> bool { matches!(self, LevelGrid::Sparse) }
}
