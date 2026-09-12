//! Per-level grid state for `apply_and`'s product construction.
//!
//! The product grid is a single flat `Vec<u32>` shared across vtree levels
//! (see `eng.apply().node_idx`). This module names each level's slice of that
//! arena — base offset and allocated-or-not — and folds the pair into one
//! descriptor instead of parallel arrays with sentinel values.

/// Per-level descriptor for the product grid's slice of the flat arena.
#[derive(Clone, Copy, Debug)]
pub(crate) enum LevelGrid {
    /// No grid allocated; consumers must go through `product_lists[t]`.
    Sparse,
    /// Grid allocated at `base` and filled — by the leaf `CONJOIN_GRID` lookup
    /// or by one of the internal-level producers (the scatter from a product
    /// list, the sequential dense emit, an identity shortcut).
    Materialized { base: usize },
}

impl LevelGrid {
    /// Base offset into the flat `node_idx` arena, or `None` if unallocated.
    #[inline]
    pub(crate) fn base(&self) -> Option<usize> {
        match self {
            LevelGrid::Sparse => None,
            LevelGrid::Materialized { base } => Some(*base),
        }
    }

    /// Base offset, asserting the grid is allocated. Use at call sites that
    /// have just proven allocation (e.g. After `ensure_grid`).
    #[inline]
    pub(crate) fn base_unchecked(&self) -> usize {
        self.base().expect("expected allocated grid, found Sparse")
    }

    #[inline]
    pub(crate) fn is_sparse(&self) -> bool { matches!(self, LevelGrid::Sparse) }
}
