//! Normalize packed rows into sorted, distinct assignments in leaf order.

use std::borrow::Cow;

use super::layout::Layout;
use crate::limits::{Limits, OperationError};
use crate::sort::Radix;

/// Words one packed row occupies. The floor of one keeps a row of no
/// variables a readable slice rather than an empty one.
pub(super) fn words_per_row(num_vars: usize) -> usize {
    num_vars.div_ceil(64).max(1)
}

/// Bits needed to write every index below `count`.
pub(super) fn index_bits(count: usize) -> u32 {
    usize::BITS - count.saturating_sub(1).leading_zeros()
}

/// Copy the `width` bits of `src` from bit `start` on into `dst`, low word
/// first, clearing the bits of the last word past `width`.
pub(super) fn extract(src: &[u64], start: usize, width: usize, dst: &mut [u64]) {
    let (skip, shift) = (start / 64, start % 64);
    for (k, out) in dst.iter_mut().enumerate() {
        let low = src.get(skip + k).copied().unwrap_or(0) >> shift;
        let high = match shift {
            0 => 0,
            _ => src.get(skip + k + 1).copied().unwrap_or(0) << (64 - shift),
        };
        *out = low | high;
    }
    let tail = width % 64;
    if let (true, Some(last)) = (tail != 0, dst.last_mut()) {
        *last &= (1u64 << tail) - 1;
    }
    if width == 0 {
        dst.fill(0);
    }
}

/// Re-encode the rows into leaf order, sort them and drop the repeats.
///
/// The order is numeric, the highest bit first, which is the order the split
/// reads a node's values in: the rows are the values of the lowest node over
/// every constrained variable, grouped by its left child's part. One-word
/// rows already in leaf order, ascending strictly and clear past the last
/// variable are those values as they stand, and are borrowed, not copied.
pub(super) fn distinct_rows<'a>(
    lim: &Limits,
    radix: &mut Radix,
    num_vars: usize,
    layout: &Layout,
    rows: &'a [u64],
    w: usize,
) -> Result<Cow<'a, [u64]>, OperationError> {
    if w == 1 && layout.is_identity() {
        lim.gate().poll(rows.len() as u64)?;
        if ascending_within(rows, num_vars) {
            return Ok(Cow::Borrowed(rows));
        }
    }
    owned_rows(lim, radix, num_vars, layout, rows, w).map(Cow::Owned)
}

/// Rows read per step of [`ascending_within`], between which it may stop
/// early; within a step the test has no branch.
const ASCENDING_STEP: usize = 4096;

/// Whether the one-word rows ascend strictly and hold no bit at or past
/// `num_vars`.
fn ascending_within(rows: &[u64], num_vars: usize) -> bool {
    let past = if num_vars >= 64 { 0 } else { !0u64 << num_vars };
    let mut last: Option<u64> = None;
    for step in rows.chunks(ASCENDING_STEP) {
        let mut ok = step.iter().fold(0u64, |acc, &word| acc | word) & past == 0;
        ok &= step
            .windows(2)
            .fold(true, |acc, pair| acc & (pair[0] < pair[1]));
        ok &= last.is_none_or(|last| last < step[0]);
        if !ok {
            return false;
        }
        last = step.last().copied();
    }
    true
}

/// [`distinct_rows`] into a buffer of its own.
fn owned_rows(
    lim: &Limits,
    radix: &mut Radix,
    num_vars: usize,
    layout: &Layout,
    rows: &[u64],
    w: usize,
) -> Result<Vec<u64>, OperationError> {
    let n = rows.len() / w;
    let mut packed: Vec<u64> = Vec::new();
    lim.reserve_exact(&mut packed, rows.len())?;
    let mut gate = lim.gate();
    if layout.is_identity() {
        // `vars` already runs from the rightmost leaf to the leftmost, the
        // layout's bit order, so re-encoding a row is a copy and only the bits
        // past the last variable have to go.
        let tail = num_vars - (w - 1) * 64;
        let tail_mask = if tail == 64 {
            !0u64
        } else {
            (1u64 << tail) - 1
        };
        gate.poll(rows.len() as u64)?;
        packed.extend_from_slice(rows);
        for row in packed.chunks_exact_mut(w) {
            row[w - 1] &= tail_mask;
        }
    } else {
        packed.resize(rows.len(), 0);
        for k in 0..n {
            gate.poll(w as u64)?;
            for (j, &word) in rows[k * w..(k + 1) * w].iter().enumerate() {
                // A row's bits past the last variable carry no assignment.
                let used = num_vars - j * 64;
                let mut live = if used >= 64 {
                    word
                } else {
                    word & ((1u64 << used) - 1)
                };
                while live != 0 {
                    let bit = live.trailing_zeros() as usize;
                    live &= live - 1;
                    let to = layout.position[j * 64 + bit] as usize;
                    packed[k * w + to / 64] |= 1u64 << (to % 64);
                }
            }
        }
    }

    gate.flush()?;
    sorted_distinct(lim, radix, num_vars, packed, w)
}

/// Rows already in leaf order, `w` words each, sorted with their repeats
/// dropped.
pub(super) fn sorted_distinct(
    lim: &Limits,
    radix: &mut Radix,
    num_vars: usize,
    mut packed: Vec<u64>,
    w: usize,
) -> Result<Vec<u64>, OperationError> {
    let n = packed.len() / w;
    let mut gate = lim.gate();
    if w == 1 {
        // One word is the whole row, so the words sort and deduplicate where
        // they are and the detour through a permutation buys nothing.
        // Rows handed over in order, as a sorted table often is, need no sort.
        if !packed.is_sorted() {
            // The words are bit-packed values, which a radix sort places in a
            // fixed number of linear passes.
            radix.sort(lim, &mut packed, 0, num_vars)?;
        }
        packed.dedup();
        return Ok(packed);
    }

    // Rows handed over in strictly ascending order are sorted and distinct.
    gate.poll(packed.len() as u64)?;
    if (1..n as u32).all(|k| compare_row(&packed, w, k - 1, k).is_lt()) {
        gate.flush()?;
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
        if out.len() < w || !same_row(&out[out.len() - w..], row) {
            out.extend_from_slice(row);
        }
    }
    gate.flush()?;
    lim.discard(packed);
    lim.discard(order);
    Ok(out)
}

/// Compare two whole rows as numbers, high word first.
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

/// Whether two rows of one width are equal. A row is a few words, which
/// this compares in less time than a call to `memcmp` takes.
#[inline]
pub(super) fn same_row(a: &[u64], b: &[u64]) -> bool {
    a.iter().zip(b).fold(0, |acc, (x, y)| acc | (x ^ y)) == 0
}

/// Row `k` of a buffer of `w`-word rows.
#[inline]
pub(super) fn row_at(packed: &[u64], w: usize, k: u32) -> &[u64] {
    let start = k as usize * w;
    &packed[start..start + w]
}
