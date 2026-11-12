//! Sets of a level's nodes as bits in `u64` words: node `t` is bit `t % 64`
//! of word `t / 64`.

use crate::limits::{Limits, OperationError};

/// The word of node `t` and its bit in that word.
#[inline(always)]
pub(super) fn at(t: u32) -> (usize, u64) {
    ((t / 64) as usize, 1 << (t % 64))
}

/// Whether `bits` holds node `t`.
#[inline(always)]
pub(super) fn has(bits: &[u64], t: u32) -> bool {
    let (w, b) = at(t);
    bits[w] & b != 0
}

/// Add node `t` to `bits`.
#[inline(always)]
pub(super) fn insert(bits: &mut [u64], t: u32) {
    let (w, b) = at(t);
    bits[w] |= b;
}

/// The bits of word `w` that stand for one of `width` nodes.
#[inline(always)]
pub(super) fn word_mask(width: usize, w: usize) -> u64 {
    if (w + 1) * 64 > width { (1 << (width % 64)) - 1 } else { u64::MAX }
}

/// `bits` resized to an empty set of `width` nodes.
pub(super) fn cleared<'a>(lim: &Limits, bits: &'a mut Vec<u64>, width: usize) -> Result<&'a mut [u64], OperationError> {
    let words = width.div_ceil(64);
    lim.try_resize(bits, words, 0u64)?;
    let bits = &mut bits[..words];
    bits.fill(0);
    Ok(bits)
}
