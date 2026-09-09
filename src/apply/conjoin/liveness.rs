//! dead-pair pre-filter primitives.
//!
//! The both-multi-pair conjunction path (multi-pair on both sides) scans all (p1, p2) input
//! pairs of the two operand nodes. Most pairs resolve to NO_PRODUCT child conjunctions,
//! so we precompute per-level liveness masks for O(1) skip decisions.
//!
//! Columns are mapped to u128 mask bits through a power-of-two bucket: bit
//! index = `col >> shift`, with `shift` chosen by [`bucket_shift`] so at most
//! 128 buckets cover the side's width. For `side_width ≤ 128` the shift is 0 and
//! the masks are bit-exact; wider grids get an approximate filter (a set
//! bucket bit means "some column in this bucket is alive") at the same
//! single-register test cost. False positives only — a clear intersection
//! always proves every covered (row, col) cell is NO_PRODUCT, so skips stay sound;
//! the exact per-cell NO_PRODUCT check in the scatter loops catches the rest.

use crate::engine::Engine;
use super::{ApplyError, NO_PRODUCT, TddLevel, InputPair};
use crate::diagram::MAX_LEVEL_ARENA_BYTES;
use super::marginal_plan::Sides;

/// One child side's two dead-pair pre-filter masks.
///
/// They are rebuilt from scratch at every `both_multi_pair` level ([`build_live_cols_bitmask`]
/// and [`build_reach_masks`] both `clear()` then resize-with-`0`, so no pooled
/// content can survive into a later level).
#[derive(Debug, Default)]
pub(crate) struct PrefilterSideMasks {
    /// Per-f-row live-column bitmasks for this child.
    pub(super) live_cols: Vec<u128>,
    /// Per-g-node reach bitmasks for g's references to this child.
    pub(super) reach: Vec<u128>,
}

impl PrefilterSideMasks {
    fn release_oversized(&mut self) {
        crate::engine::pool::release_if_oversized(&mut self.live_cols, MAX_LEVEL_ARENA_BYTES);
        crate::engine::pool::release_if_oversized(&mut self.reach, MAX_LEVEL_ARENA_BYTES);
    }
}

/// Both sides' dead-pair pre-filter masks as one pooled scratch bundle.
///
/// Bundled so that one pool slot (`eng.apply().prefilter_masks`) and one
/// retention rule cover all four buffers, instead of each apply re-growing
/// them from empty.
pub(crate) type PrefilterMaskScratch = Sides<PrefilterSideMasks>;

impl PrefilterMaskScratch {
    /// The module's scratch-retention rule, applied buffer by buffer (a
    /// struct-held pool can't round-trip each one through `Pool::put_bounded`).
    pub(super) fn release_oversized(&mut self) {
        self.left.release_oversized();
        self.right.release_oversized();
    }
}

/// Smallest shift such that `ceil(side_width / 2^shift) ≤ 128`, i.e. the
/// column-to-bucket shift for a u128 liveness mask over `side_width` columns.
#[inline]
pub(super) fn bucket_shift(side_width: usize) -> u32 {
    if side_width <= 128 {
        0
    } else {
        // ceil_log2(side_width) - 7: halve until ≤ 128 buckets remain.
        (side_width - 1).ilog2() + 1 - 7
    }
}

/// Per-row live-column-bucket mask. `live_cols[a]` has bit `b >> shift` set
/// iff `node_idx[base + a*side_width + b] != NO_PRODUCT` for some `b` in that bucket.
pub(super) fn build_live_cols_bitmask(
    eng: &Engine,
    k1_side: usize,
    side_width: usize,
    base: usize,
    node_idx: &[u32],
    live_cols: &mut Vec<u128>,
    shift: u32,
) -> Result<(), ApplyError> {
    let lim = eng.limits();
    debug_assert!(side_width == 0 || (side_width - 1) >> shift < 128);
    live_cols.clear();
    lim.try_resize(live_cols, k1_side, 0u128)?;
    let bucket = 1usize << shift;
    // Indexes `live_cols` and, through a computed row base, `node_idx`.
    #[allow(clippy::needless_range_loop)]
    for a in 0..k1_side {
        let row_base = base + a * side_width;
        let mut mask = 0u128;
        // Per bucket: stop at the first alive column (one set bit per bucket).
        let mut b0 = 0;
        let mut bit = 1u128;
        while b0 < side_width {
            let b1 = (b0 + bucket).min(side_width);
            for b in b0..b1 {
                if node_idx[row_base + b] != NO_PRODUCT {
                    mask |= bit;
                    break;
                }
            }
            b0 = b1;
            bit <<= 1;
        }
        live_cols[a] = mask;
    }
    Ok(())
}

/// Per-g-node child-reach bucket mask. For each g node `j`, `reach[j]` is
/// the union of (1 << (idx >> shift)) for every child index referenced by
/// `j`'s input pairs on the specified side (left or right, via `pair_side`).
///
/// Combined with `live_cols[a]`, gives the O(1) skip test:
/// `(live_cols[a] & reach[j]) == 0` ⇒ no alive (i, j) conjunction.
pub(super) fn build_reach_masks(
    eng: &Engine,
    level: &TddLevel,
    k_level: usize,
    reach: &mut Vec<u128>,
    pair_side: impl Fn(&InputPair) -> usize,
    shift: u32,
) -> Result<(), ApplyError> {
    let lim = eng.limits();
    reach.clear();
    lim.try_resize(reach, k_level, 0u128)?;
    // Indexes `reach` and `level.nodes` at the same position.
    #[allow(clippy::needless_range_loop)]
    for j in 0..k_level {
        let node = &level.nodes[j];
        if !node.is_internal() { continue; }
        for p in level.pairs_of(node) {
            reach[j] |= 1u128 << (pair_side(p) >> shift);
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "liveness_tests.rs"]
mod tests;
