//! dead-pair pre-filter primitives.
//!
//! The both-multi-pair conjunction path (multi-pair on both sides) scans all (p1, p2) input
//! pairs of the two operand nodes. Most pairs resolve to `NO_PRODUCT` child conjunctions,
//! so we precompute per-level liveness masks for O(1) skip decisions. A level with one
//! multi-pair operand builds them too where its grid is large
//! ([`one_sided_masks_pay`]): there a cell has one pair on a side, and the masks cull the
//! cells whose every candidate is dead on one side.
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

/// The fewest cells a product grid of two multi-pair levels has for its
/// level to build the masks. Under it they cost more than they save: on such
/// a level, typically two nodes a side, the masks scan both child grids to
/// cull under a tenth of its cells, where on a grid of 8 to 63 cells they
/// cull about half.
pub(super) const MASK_MIN_CELLS: usize = 8;

/// The fewest cells a product grid of a level with one multi-pair operand
/// has for its level to build the masks. A culled cell there skips one pair
/// on a side, against several on a level of two multi-pair operands, and a
/// cell that survives pays both column tests. On grids of 64 cells or more
/// such levels typically have about two cells in three with every candidate
/// dead on one side; on grids of 16 to 63 cells the culls do not pay for the
/// tests.
pub(super) const ONE_SIDED_MASK_MIN_CELLS: usize = 64;

/// Whether a level with one multi-pair operand builds the masks: a grid of
/// `cells` cells, at least [`ONE_SIDED_MASK_MIN_CELLS`], over children whose
/// g widths `child_g_widths` take bit-exact masks ([`bucket_shift`] 0). A
/// wider child's bucketed mask culls a cell only where its whole bucket is
/// dead, which nothing has measured on these levels.
pub(super) fn one_sided_masks_pay(cells: usize, child_g_widths: Sides<usize>) -> bool {
    cells >= ONE_SIDED_MASK_MIN_CELLS
        && bucket_shift(child_g_widths.left) == 0
        && bucket_shift(child_g_widths.right) == 0
}

/// One child side's two dead-pair pre-filter masks.
///
/// They are rebuilt from scratch at every masked level ([`build_live_cols_bitmask`]
/// and [`build_reach_masks`] both `clear()` and then write every entry, so no
/// pooled content can survive into a later level).
#[derive(Debug, Default)]
pub(crate) struct PrefilterSideMasks {
    /// Per-f-row live-column bitmasks for this child.
    pub(super) live_cols: Vec<u128>,
    /// Per-g-node reach bitmasks for g's references to this child.
    pub(super) reach: Vec<u128>,
    /// The child grid's cells as bits, which [`build_live_cols_bitmask`]
    /// cuts the rows' masks out of ([`alive_bits`]).
    bits: Vec<u64>,
}

impl PrefilterSideMasks {
    fn buffers(&mut self, visit: &mut dyn FnMut(&mut dyn crate::execution::pool::Scratch)) {
        visit(&mut self.live_cols);
        visit(&mut self.reach);
        visit(&mut self.bits);
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
///
/// A bit-exact side (`shift` 0) reads its grid once as one bit a cell
/// ([`alive_bits`]) and cuts each row's mask out of those bits, a few
/// instructions a row whatever its width. Read a cell at a time, with a
/// compare and a shift each, the scan cost about twelve instructions a cell
/// and as much as the culls saved on the levels of one multi-pair operand,
/// whose child grids typically have rows of 2 to 40 cells.
pub(super) fn build_live_cols_bitmask(
    eng: &Engine,
    k1_side: usize,
    side_width: usize,
    base: usize,
    node_idx: &[u32],
    out: &mut PrefilterSideMasks,
    shift: u32,
) -> Result<(), OperationError> {
    let lim = eng.limits();
    debug_assert!(side_width == 0 || (side_width - 1) >> shift < 128);
    let PrefilterSideMasks { live_cols, bits, .. } = out;
    live_cols.clear();
    lim.reserve_exact(live_cols, k1_side)?;
    if side_width == 0 {
        live_cols.extend(std::iter::repeat_n(0, k1_side));
        return Ok(());
    }
    let grid = &node_idx[base..base + k1_side * side_width];
    if shift != 0 {
        live_cols.extend(grid.chunks_exact(side_width).map(|row| row_mask_bucketed(row, shift)));
        return Ok(());
    }
    alive_bits(lim, grid, bits)?;
    let bits = &bits[..];
    // Row `a`'s cells are bits `a * side_width ..` of the grid's: one window
    // of 64 bits on a side of 64 columns or fewer, two on a wider one.
    let low = |n: usize| if n >= 64 { u64::MAX } else { (1u64 << n) - 1 };
    if side_width <= 64 {
        let keep = low(side_width);
        live_cols.extend((0..k1_side).map(|a| u128::from(bits_from(bits, a * side_width) & keep)));
    } else {
        let keep = low(side_width - 64);
        live_cols.extend((0..k1_side).map(|a| {
            let at = a * side_width;
            u128::from(bits_from(bits, at)) | u128::from(bits_from(bits, at + 64) & keep) << 64
        }));
    }
    Ok(())
}

/// One bit a cell of `grid` into `bits`, set where the cell holds a
/// product: 64 cells a word, the first in the lowest bit, and a zero word
/// after the last, so that [`bits_from`] reads past no end.
fn alive_bits(lim: &crate::limits::Limits, grid: &[u32], bits: &mut Vec<u64>) -> Result<(), OperationError> {
    bits.clear();
    lim.reserve_exact(bits, grid.len() / 64 + 2)?;
    let words = grid.chunks_exact(64);
    let tail = words.remainder();
    bits.extend(words.map(|cells| alive_word(cells.try_into().expect("a chunk of 64 cells"))));
    if !tail.is_empty() {
        bits.push(tail.iter().rev().fold(0, |word, &v| word << 1 | u64::from(v != NO_PRODUCT)));
    }
    bits.push(0);
    Ok(())
}

/// The 64 cells of one word of [`alive_bits`], in two halves of 32. Of a
/// fixed length, each half unrolls into vector compares without a branch,
/// four cells to a baseline x86-64 vector and each cell's bit taken from a
/// constant; one fold into a 64-bit word took twice the instructions.
#[inline(always)]
fn alive_word(cells: &[u32; 64]) -> u64 {
    let mut word = 0u64;
    for (k, half) in cells.chunks_exact(32).enumerate() {
        let bits = half.iter().enumerate().fold(0u32, |bits, (b, &v)| bits | u32::from(v != NO_PRODUCT) << b);
        word |= u64::from(bits) << (32 * k);
    }
    word
}

/// The 64 bits of `bits` from bit `at` on.
#[inline(always)]
fn bits_from(bits: &[u64], at: usize) -> u64 {
    let (word, offset) = (at >> 6, at & 63);
    ((u128::from(bits[word + 1]) << 64 | u128::from(bits[word])) >> offset) as u64
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
    reach.clear();
    eng.limits().reserve_exact(reach, k_level)?;
    // Node `j`'s pairs set `reach[j]`: a stored level's read as slices of its
    // arena, an implicit one's generated a chunk of nodes at a time, in
    // order. Through an iterator of `(node, pairs)` the walk was a call a
    // node, about fifty instructions each.
    level.try_for_each_node::<std::convert::Infallible>(0..k_level, |_, pairs| {
        reach.push(pairs.iter().fold(0u128, |m, p| m | 1u128 << (pair_side(p) >> shift)));
        Ok(())
    }).unwrap_or_else(|never| match never {});
    Ok(())
}

#[cfg(test)]
#[path = "tests/liveness/mod.rs"]
mod tests;
