use super::*;

mod fallible;
mod support;
pub(crate) use support::fuse_pairs;
mod weighted;
mod window;
mod bitmap;

/// The widths at which pair fusion changes how it works on one node: the
/// fewest pairs whose grouping sorts rather than hashes them, and the fewest
/// plans at which the rewrite tests membership on a bitmap. For a caller's
/// test that means to reach both.
pub(crate) fn fusion_widths() -> (usize, usize) {
    (plan::SORT_MIN, rewrite::BITMAP_MIN_PLANS)
}
