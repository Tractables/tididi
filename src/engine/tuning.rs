//! Thresholds the apply routes decide by.

/// The thresholds an [`Engine`](crate::Engine)'s operations decide by.
///
/// Production runs on [`Tuning::default`]; a caller that needs another route
/// taken installs its own values on its own engine, so the choice is a field on
/// the session rather than ambient state a whole thread shares.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Tuning {
    /// Grid cells above which a level takes the sparse route.
    pub(crate) sparse_min_grid: usize,
    /// How much sparser than its grid a level must be to take the sparse route.
    pub(crate) sparse_sparsity_factor: u128,
    /// Soft byte budget for the sparse path's transient emission buffers.
    ///
    /// A level whose whole projected transient fits inside the budget is
    /// emitted in one chunk, which preserves the cross-apply bucket capacity
    /// reuse; the budget exists for the wide levels that do not fit, which
    /// split into several chunks and release each consumed range before the
    /// next one grows. `usize::MAX` never splits.
    pub(crate) sparse_chunk_bytes: usize,
}

impl Default for Tuning {
    fn default() -> Tuning {
        Tuning {
            sparse_min_grid: 4096,
            sparse_sparsity_factor: 64,
            sparse_chunk_bytes: 256 * 1024 * 1024,
        }
    }
}
