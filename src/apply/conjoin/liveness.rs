//! dead-pair pre-filter primitives.
//!
//! The both-multi-pair conjunction path (multi-pair on both sides) scans all (p1, p2) input
//! pairs of the two operand nodes. Most pairs resolve to `NO_PRODUCT` child conjunctions,
//! so we precompute per-level liveness masks for O(1) skip decisions.
//!
//! Columns are mapped to u128 mask bits through a power-of-two bucket: bit
//! index = `col >> shift`, with `shift` chosen by [`bucket_shift`] so at most
//! 128 buckets cover the side's width. For `side_width ≤ 128` the shift is 0 and
//! the masks are bit-exact; wider grids get an approximate filter (a set
//! bucket bit means "some column in this bucket is alive") at the same
//! single-register test cost. False positives only — a clear intersection
//! always proves every covered (row, col) cell is `NO_PRODUCT`, so skips stay sound;
//! the exact per-cell `NO_PRODUCT` check in the scatter loops catches the rest.

use crate::Engine;
use super::{OperationError, NO_PRODUCT, TddLevel, ChildPair};
use crate::diagram::Sides;

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
    fn buffers(&mut self, visit: &mut dyn FnMut(&mut dyn crate::execution::pool::Scratch)) {
        visit(&mut self.live_cols);
        visit(&mut self.reach);
    }
}

/// Both sides' dead-pair pre-filter masks as one pooled scratch bundle.
///
/// Bundled so that one workspace field and one
/// retention rule cover all four buffers, instead of each apply re-growing
/// them from empty.
pub(crate) type PrefilterMaskScratch = Sides<PrefilterSideMasks>;

impl PrefilterMaskScratch {
    pub(super) fn buffers(&mut self, visit: &mut dyn FnMut(&mut dyn crate::execution::pool::Scratch)) {
        self.left.buffers(visit);
        self.right.buffers(visit);
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
) -> Result<(), OperationError> {
    let lim = eng.limits();
    debug_assert!(side_width == 0 || (side_width - 1) >> shift < 128);
    live_cols.clear();
    lim.try_resize(live_cols, k1_side, 0u128)?;
    if side_width == 0 {
        return Ok(());
    }
    // The rows in order, and one dispatch for the whole grid rather than one
    // per row: `shift` is a property of the side's width.
    let grid = &node_idx[base..base + k1_side * side_width];
    let rows = grid.chunks_exact(side_width);
    if shift == 0 {
        for (slot, row) in live_cols[..k1_side].iter_mut().zip(rows) {
            *slot = row_mask_exact(row);
        }
    } else {
        for (slot, row) in live_cols[..k1_side].iter_mut().zip(rows) {
            *slot = row_mask_bucketed(row, shift);
        }
    }
    Ok(())
}

/// The bit-exact mask of a row: bit `b` set iff column `b` is alive.
///
/// One column is one compare and one 64-bit shift-or, in the two halves the
/// mask's 128 bits divide into, so a row pays no 128-bit shift and nothing
/// that scales with the mask rather than the row. This is the width every
/// level at or under the mask's 128 columns takes.
#[inline]
fn row_mask_exact(row: &[u32]) -> u128 {
    debug_assert!(row.len() <= 128);
    fn half(part: &[u32]) -> u64 {
        let mut bits = 0u64;
        for (b, &v) in part.iter().enumerate() {
            bits |= u64::from(v != NO_PRODUCT) << b;
        }
        bits
    }
    // Most sides are narrower than the low half, and there are as many rows as
    // the child has nodes, so the half that is always empty on those is worth
    // not assembling.
    if row.len() <= 64 {
        return u128::from(half(row));
    }
    let (low, high) = row.split_at(64);
    u128::from(half(low)) | (u128::from(half(high)) << 64)
}

/// The bucketed mask of a row, for a side wider than the 128 mask bits: bit
/// `b >> shift` set iff some column of that bucket is alive. Each bucket stops
/// at its first alive column.
#[inline]
fn row_mask_bucketed(row: &[u32], shift: u32) -> u128 {
    let mut mask = 0u128;
    // The bucket's bit walks up with the buckets: a 128-bit shift by a
    // variable is a branchy sequence, where doubling is two instructions.
    let mut bit = 1u128;
    for bucket in row.chunks(1usize << shift) {
        if bucket.iter().any(|&v| v != NO_PRODUCT) {
            mask |= bit;
        }
        bit <<= 1;
    }
    mask
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
    pair_side: impl Fn(&ChildPair) -> usize,
    shift: u32,
) -> Result<(), OperationError> {
    let lim = eng.limits();
    reach.clear();
    lim.try_resize(reach, k_level, 0u128)?;
    // Indexes `reach` and `level.nodes` at the same position.
    #[expect(clippy::needless_range_loop)]
    for j in 0..k_level {
        let node = &level.nodes[j];
        for p in level.pairs_of(node) {
            reach[j] |= 1u128 << (pair_side(p) >> shift);
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "tests/liveness/mod.rs"]
mod tests;
