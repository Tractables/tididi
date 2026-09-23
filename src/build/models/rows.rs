//! Normalize packed rows into sorted, distinct assignments in leaf order.

use crate::limits::{Limits, OperationError};
use super::layout::Layout;

/// Words one packed row occupies. The floor of one keeps a row of no
/// variables a readable slice rather than an empty one.
pub(super) fn words_per_row(num_vars: usize) -> usize {
    num_vars.div_ceil(64).max(1)
}

/// Re-encode the rows into leaf order, sort them and drop the repeats.
///
/// The order is by bit position, the highest first, which is the order
/// `compare_value` reads a node's value in. A node whose bits reach the top
/// of the row therefore finds the rows already grouped — see
/// `order_by_value`.
pub(super) fn distinct_rows(
    lim: &Limits,
    num_vars: usize,
    layout: &Layout,
    rows: &[u64],
    w: usize,
) -> Result<Vec<u64>, OperationError> {
    let n = rows.len() / w;
    let mut packed: Vec<u64> = Vec::new();
    lim.try_resize(&mut packed, rows.len(), 0u64)?;
    let mut gate = lim.gate();
    if layout.is_identity() {
        // `vars` already runs in the vtree's leaf order, so re-encoding a row
        // is a copy and only the bits past the last variable have to go.
        let tail = num_vars - (w - 1) * 64;
        let tail_mask = if tail == 64 { !0u64 } else { (1u64 << tail) - 1 };
        for k in 0..n {
            gate.poll(w as u64)?;
            let to = &mut packed[k * w..(k + 1) * w];
            to.copy_from_slice(&rows[k * w..(k + 1) * w]);
            to[w - 1] &= tail_mask;
        }
    } else {
        for k in 0..n {
            gate.poll(w as u64)?;
            for (j, &word) in rows[k * w..(k + 1) * w].iter().enumerate() {
                // A row's bits past the last variable carry no assignment.
                let used = num_vars - j * 64;
                let mut live = if used >= 64 { word } else { word & ((1u64 << used) - 1) };
                while live != 0 {
                    let bit = live.trailing_zeros() as usize;
                    live &= live - 1;
                    let to = layout.position[j * 64 + bit] as usize;
                    packed[k * w + to / 64] |= 1u64 << (to % 64);
                }
            }
        }
    }

    if w == 1 {
        // One word is the whole row, so the words sort and deduplicate where
        // they are and the detour through a permutation buys nothing.
        gate.flush()?;
        sort_words(lim, &mut packed, num_vars)?;
        packed.dedup();
        return Ok(packed);
    }

    let mut order: Vec<u32> = Vec::new();
    lim.reserve_exact(&mut order, n)?;
    order.extend(0..n as u32);
    order.sort_unstable_by(|&a, &b| compare_row(&packed, w, a, b));

    let mut out: Vec<u64> = Vec::new();
    lim.reserve_exact(&mut out, packed.len())?;
    for &k in &order {
        gate.poll(w as u64)?;
        let row = row_at(&packed, w, k);
        if out.len() < w || &out[out.len() - w..] != row {
            out.extend_from_slice(row);
        }
    }
    gate.flush()?;
    lim.discard(packed);
    lim.discard(order);
    Ok(out)
}

/// Bits of a row one radix pass places, which puts the counts in the
/// first-level cache and covers a row of any width in at most six passes.
const RADIX_BITS: usize = 11;

/// Rows below which the comparison sort wins: a radix pass is linear but reads
/// and writes the whole buffer whatever the rows look like.
pub(super) const RADIX_MIN_ROWS: usize = 1 << 14;

/// Sort one-word rows ascending.
///
/// The words are bit-packed values over `num_vars` bits, so a radix sort
/// places them in a fixed number of linear passes where a comparison sort
/// takes a logarithmic number over the whole buffer. Both leave the same
/// order: on integers there is only one.
pub(super) fn sort_words(lim: &Limits, packed: &mut Vec<u64>, num_vars: usize) -> Result<(), OperationError> {
    let m = packed.len();
    if m < RADIX_MIN_ROWS {
        packed.sort_unstable();
        return Ok(());
    }
    let mask = (1u64 << RADIX_BITS) - 1;
    let mut other: Vec<u64> = Vec::new();
    lim.try_resize(&mut other, m, 0u64)?;
    let mut counts: Vec<u32> = Vec::new();
    lim.try_resize(&mut counts, 1 << RADIX_BITS, 0u32)?;

    let mut gate = lim.gate();
    for pass in 0..num_vars.div_ceil(RADIX_BITS) {
        gate.poll(m as u64)?;
        let shift = pass * RADIX_BITS;
        counts.fill(0);
        for &word in packed.iter() {
            counts[((word >> shift) & mask) as usize] += 1;
        }
        if counts[((packed[0] >> shift) & mask) as usize] as usize == m {
            // Every row holds the same digit, so this pass would copy the
            // buffer onto itself. A bit-packed table often has such a pass.
            continue;
        }
        let mut at = 0u32;
        for count in counts.iter_mut() {
            let here = *count;
            *count = at;
            at += here;
        }
        for &word in packed.iter() {
            let digit = ((word >> shift) & mask) as usize;
            other[counts[digit] as usize] = word;
            counts[digit] += 1;
        }
        std::mem::swap(packed, &mut other);
    }
    gate.flush()?;
    lim.discard(other);
    lim.discard(counts);
    Ok(())
}

/// Compare two whole rows, high word first, so that the order agrees with
/// `compare_value` on any value that reaches the top of the row.
fn compare_row(packed: &[u64], w: usize, a: u32, b: u32) -> std::cmp::Ordering {
    let (x, y) = (row_at(packed, w, a), row_at(packed, w, b));
    for i in (0..w).rev() {
        let ord = x[i].cmp(&y[i]);
        if ord != std::cmp::Ordering::Equal {
            return ord;
        }
    }
    std::cmp::Ordering::Equal
}

/// Row `k` of a buffer of `w`-word rows.
#[inline]
pub(super) fn row_at(packed: &[u64], w: usize, k: u32) -> &[u64] {
    let start = k as usize * w;
    &packed[start..start + w]
}
