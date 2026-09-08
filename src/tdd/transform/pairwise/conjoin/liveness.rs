//! NxM dead-pair pre-filter primitives.
//!
//! The NxM conjunction path (multi-pair on both sides) scans all (p1, p2) input
//! pairs of the two operand nodes. Most pairs resolve to DEAD child conjunctions,
//! so we precompute per-level liveness masks for O(1) skip decisions.
//!
//! Columns are mapped to u128 mask bits through a power-of-two bucket: bit
//! index = `col >> shift`, with `shift` chosen by [`bucket_shift`] so at most
//! 128 buckets cover the side's width. For `k2_side ≤ 128` the shift is 0 and
//! the masks are bit-exact; wider grids get an approximate filter (a set
//! bucket bit means "some column in this bucket is alive") at the same
//! single-register test cost. False positives only — a clear intersection
//! always proves every covered (row, col) cell is DEAD, so skips stay sound;
//! the exact per-cell DEAD check in the scatter loops catches the rest.

use super::{ApplyError, DEAD, TddLevel, InputPair, try_resize};
use crate::tdd::utils::release_if_oversized;
use crate::tdd::types::MAX_LEVEL_ARENA_BYTES;

/// The four NxM pre-filter masks as ONE pooled scratch bundle.
///
/// They are rebuilt from scratch at every `nxm` level ([`build_live_cols_bitmask`]
/// and [`build_reach_masks`] both `clear()` then resize-with-`0`, so no pooled
/// content can survive into a later level), and used to be four fresh `Vec`s per
/// apply — every apply re-grew all four from empty. Bundled so ONE thread-local
/// pool slot (`super::SCRATCH_NXM_MASKS`) and ONE retention rule cover all four.
#[derive(Debug)]
pub(super) struct NxmMaskScratch {
    /// Per-c1-row live-column bitmasks for the left child.
    pub(super) live_left_cols: Vec<u128>,
    /// Per-c2-node reach bitmasks for c2's left-child references.
    pub(super) reach_c2_left: Vec<u128>,
    /// Per-c1-row live-column bitmasks for the right child.
    pub(super) live_right_cols: Vec<u128>,
    /// Per-c2-node reach bitmasks for c2's right-child references.
    pub(super) reach_c2_right: Vec<u128>,
}

impl NxmMaskScratch {
    pub(super) const fn new() -> Self {
        Self {
            live_left_cols: Vec::new(),
            reach_c2_left: Vec::new(),
            live_right_cols: Vec::new(),
            reach_c2_right: Vec::new(),
        }
    }

    /// The module's scratch-retention rule, applied field by field (a
    /// struct-held pool can't round-trip each buffer through `pool_put_bounded`).
    pub(super) fn release_oversized(&mut self) {
        release_if_oversized(&mut self.live_left_cols, MAX_LEVEL_ARENA_BYTES);
        release_if_oversized(&mut self.reach_c2_left, MAX_LEVEL_ARENA_BYTES);
        release_if_oversized(&mut self.live_right_cols, MAX_LEVEL_ARENA_BYTES);
        release_if_oversized(&mut self.reach_c2_right, MAX_LEVEL_ARENA_BYTES);
    }
}

impl Default for NxmMaskScratch {
    fn default() -> Self { Self::new() }
}

/// Smallest shift such that `ceil(k2_side / 2^shift) ≤ 128`, i.e. the
/// column-to-bucket shift for a u128 liveness mask over `k2_side` columns.
#[inline]
pub(super) fn bucket_shift(k2_side: usize) -> u32 {
    if k2_side <= 128 {
        0
    } else {
        // ceil_log2(k2_side) - 7: halve until ≤ 128 buckets remain.
        (k2_side - 1).ilog2() + 1 - 7
    }
}

/// Per-row live-column-bucket mask. `live_cols[a]` has bit `b >> shift` set
/// iff `node_idx[base + a*k2_side + b] != DEAD` for some `b` in that bucket.
pub(super) fn build_live_cols_bitmask(
    k1_side: usize,
    k2_side: usize,
    base: usize,
    node_idx: &[u32],
    live_cols: &mut Vec<u128>,
    shift: u32,
) -> Result<(), ApplyError> {
    debug_assert!(k2_side == 0 || (k2_side - 1) >> shift < 128);
    live_cols.clear();
    try_resize(live_cols, k1_side, 0u128)?;
    let bucket = 1usize << shift;
    for a in 0..k1_side {
        let row_base = base + a * k2_side;
        let mut mask = 0u128;
        // Per bucket: stop at the first alive column (one set bit per bucket).
        let mut b0 = 0;
        let mut bit = 1u128;
        while b0 < k2_side {
            let b1 = (b0 + bucket).min(k2_side);
            for b in b0..b1 {
                if node_idx[row_base + b] != DEAD {
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

/// Per-c2-node child-reach bucket mask. For each c2 node `j`, `reach[j]` is
/// the union of (1 << (idx >> shift)) for every child index referenced by
/// `j`'s input pairs on the specified side (left or right, via `pair_side`).
///
/// Combined with `live_cols[a]`, gives the O(1) skip test:
/// `(live_cols[a] & reach[j]) == 0` ⇒ no alive (i, j) conjunction.
pub(super) fn build_reach_masks(
    level: &TddLevel,
    k_level: usize,
    reach: &mut Vec<u128>,
    pair_side: impl Fn(&InputPair) -> usize,
    shift: u32,
) -> Result<(), ApplyError> {
    reach.clear();
    try_resize(reach, k_level, 0u128)?;
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
