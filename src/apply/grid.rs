//! Per-level grid state for `apply_and`'s product construction.
//!
//! The product grid is a single flat `Vec<u32>` shared across vtree levels
//! (see `eng.apply().node_idx`). This module names the three possible states of
//! each level's slice of that arena — base offset and allocated-or-not — and
//! folds them into one descriptor instead of parallel arrays with sentinel
//! values.

/// Per-level descriptor for the product grid's slice of the flat arena.
#[derive(Clone, Copy, Debug)]
pub(crate) enum LevelGrid {
    /// No grid allocated; consumers must go through `product_lists[t]`.
    Sparse,
    /// Grid populated by leaf `CONJOIN_GRID` lookup. Read-only for consumers.
    Leaf { base: usize },
    /// Grid materialized by any of the internal-level producers: the scatter
    /// from a product list, the sequential dense emit, or an identity shortcut.
    Dense { base: usize },
}

impl LevelGrid {
    /// Base offset into the flat `node_idx` arena, or `None` if unallocated.
    #[inline]
    pub fn base(&self) -> Option<usize> {
        match self {
            LevelGrid::Sparse => None,
            LevelGrid::Leaf { base } | LevelGrid::Dense { base } => Some(*base),
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
