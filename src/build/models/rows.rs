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
/// The order is numeric, the highest bit first, which is the order the split
/// reads a node's values in: the rows are the values of the lowest node over
/// every constrained variable, grouped by its right child's part.
pub(super) fn distinct_rows(
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
        let tail_mask = if tail == 64 { !0u64 } else { (1u64 << tail) - 1 };
        gate.poll(rows.len() as u64)?;
        packed.extend_from_slice(rows);
        for row in packed.chunks_exact_mut(w) {
            row[w - 1] &= tail_mask;
        }
    } else if layout.is_reversed() {
        // `vars` runs left to right in leaf order and the leftmost leaf holds
        // the highest bit, so re-encoding reverses a row's bits; the bits past
        // the last variable land below the row and shift out.
        gate.poll(rows.len() as u64)?;
        let shift = w * 64 - num_vars;
        if w == 1 {
            packed.extend(rows.iter().map(|&word| word.reverse_bits() >> shift));
        } else {
            packed.resize(rows.len(), 0);
            for (row, out) in rows.chunks_exact(w).zip(packed.chunks_exact_mut(w)) {
                for (to, &word) in out.iter_mut().zip(row.iter().rev()) {
                    *to = word.reverse_bits();
                }
                if shift > 0 {
                    for j in 0..w {
                        let above = out.get(j + 1).map_or(0, |&next| next << (64 - shift));
                        out[j] = out[j] >> shift | above;
                    }
                }
            }
        }
    } else if w == 1 && n >= BYTE_TABLE_MIN_ROWS {
        // One table per byte of the input word holds where that byte's bits
        // land, so a row is re-encoded by eight lookups instead of one step
        // per set bit.
        let mut table: Vec<u64> = Vec::new();
        lim.try_resize(&mut table, 8 * 256, 0u64)?;
        for (bit, &to) in layout.position.iter().enumerate() {
            let (byte, within) = (bit / 8, bit % 8);
            for (value, entry) in table[byte * 256..][..256].iter_mut().enumerate() {
                if (value >> within) & 1 == 1 {
                    *entry |= 1u64 << to;
                }
            }
        }
        gate.poll(n as u64)?;
        packed.extend(rows.iter().map(|&word| {
            (0..8).fold(0, |acc, byte| acc | table[byte * 256 + ((word >> (8 * byte)) & 0xff) as usize])
        }));
        lim.discard(table);
    } else {
        packed.resize(rows.len(), 0);
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
        // Rows handed over in order, as a sorted table often is, need no sort.
        if !packed.is_sorted() {
            // The words are bit-packed values, which a radix sort places in a
            // fixed number of linear passes.
            let lo = ordered_low_bits(layout, &packed);
            radix.sort(lim, &mut packed, lo, num_vars - lo)?;
        }
        packed.dedup();
        return Ok(packed);
    }

    // Rows handed over in strictly ascending order are sorted and distinct.
    gate.poll(rows.len() as u64)?;
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
        if out.len() < w || &out[out.len() - w..] != row {
            out.extend_from_slice(row);
        }
    }
    gate.flush()?;
    lim.discard(packed);
    lim.discard(order);
    Ok(out)
}

/// The widest trailing subtree whose bits already ascend down the rows.
///
/// A trailing subtree's variables, the vtree's rightmost, are the low bits
/// of a re-encoded row. A table sorted on the columns placed there, as
/// tables usually are on their first column, lists its rows in ascending
/// order of those bits, and the radix sort need not place them. Zero when no
/// trailing subtree is ordered.
fn ordered_low_bits(layout: &Layout, packed: &[u64]) -> usize {
    let mut below = 64;
    loop {
        // The widest leading subtree narrower than the last one tried.
        let widest = (0..layout.lo.len())
            .filter(|&t| layout.lo[t] == 0 && layout.count[t] > 0 && layout.count[t] < below)
            .map(|t| layout.count[t])
            .max();
        let Some(width) = widest else { return 0 };
        let mask = (1u64 << width) - 1;
        if packed.windows(2).all(|pair| pair[0] & mask <= pair[1] & mask) {
            return width as usize;
        }
        below = width;
    }
}

/// Rows from which re-encoding a one-word row goes through byte tables,
/// whose construction costs about as much as re-encoding this many rows bit
/// by bit.
const BYTE_TABLE_MIN_ROWS: usize = 64;

/// Bits of a key one radix pass places at most, which puts the counts in the
/// first-level cache, while the keys are fewer than a pass of
/// [`RADIX_LARGE_BITS`] would count.
const RADIX_BITS: usize = 11;

/// Rows below which the comparison sort wins: a radix pass is linear but reads
/// and writes the whole buffer whatever the rows look like, and clears its
/// counts. From a few thousand rows up, three passes or fewer take well under
/// the comparison sort's time.
pub(super) const RADIX_MIN_ROWS: usize = 1 << 11;

/// Passes past which a comparison sort wins at any size. Each pass moves
/// every key through the cache, and at four a comparison sort, which works
/// on ever smaller pieces that fit it, is the faster of the two — after one
/// counting pass has cut the keys into such pieces ([`Radix::sort_wide`]).
const RADIX_MAX_PASSES: usize = 3;

/// Bits one pass places at most when [`RADIX_BITS`] would take more than
/// [`RADIX_MAX_PASSES`] passes. Three passes over 4096 counts each still beat
/// the comparison sort that a key of 34 to 36 bits would fall back to, such
/// as the parent and high atoms of a large split's triples.
const RADIX_WIDE_BITS: usize = 12;

/// Bits one pass places at most once the keys are at least as many as its
/// counts. A pass costs about the same per key whatever its digit up to
/// this width, so the fewest passes are the fastest: 25 bits sort in two
/// rather than three, a 13-bit key in one.
pub(super) const RADIX_LARGE_BITS: usize = 14;

/// The buffers a radix sort moves keys through, kept for the next sort.
#[derive(Default)]
pub(super) struct Radix {
    other: Vec<u64>,
    counts: Vec<u32>,
}

impl Radix {
    /// Drop the buffers and hand their charge back.
    pub(super) fn discard(self, lim: &Limits) {
        lim.discard(self.other);
        lim.discard(self.counts);
    }

    /// The buffer a sort moves keys through, which holds nothing between
    /// sorts and may be lent out, as long as it comes back.
    pub(super) fn spare(&mut self) -> &mut Vec<u64> {
        &mut self.other
    }

    /// Sort keys ascending in their low `lo + bits` bits, then in the bits
    /// above, given that the input already lists them in ascending order of
    /// their low `lo` bits, and keys equal in their low `lo + bits` bits in
    /// ascending order of the bits above.
    ///
    /// Under that precondition a stable sort on bits `lo..lo + bits` alone
    /// leaves the ascending order, so a radix sort places only those bits.
    /// A key that carries its position below or above a value, as a
    /// tie-break, sorts this way in as many passes as the value is wide.
    pub(super) fn sort(&mut self, lim: &Limits, keys: &mut Vec<u64>, lo: usize, bits: usize) -> Result<(), OperationError> {
        let m = keys.len();
        if bits == 0 {
            // Nothing lies above the ordered low bits.
            return Ok(());
        }
        let most = if m >> RADIX_LARGE_BITS > 0 { RADIX_LARGE_BITS } else { RADIX_BITS };
        let passes = match bits.div_ceil(most) {
            passes if passes > RADIX_MAX_PASSES && bits <= RADIX_MAX_PASSES * RADIX_WIDE_BITS => RADIX_MAX_PASSES,
            passes => passes,
        };
        if m < RADIX_MIN_ROWS {
            // Rotated, the sorted bits lead and the bits above break ties.
            let sorted = (lo + bits) as u32;
            let mut gate = lim.gate();
            gate.poll((m * passes) as u64)?;
            keys.sort_unstable_by_key(|&key| key.rotate_right(sorted));
            return gate.flush();
        }
        if passes > RADIX_MAX_PASSES {
            return self.sort_wide(lim, keys, lo, bits);
        }
        let digit = bits.div_ceil(passes);
        let mask = (1u64 << digit) - 1;
        let buckets = 1usize << digit;
        // The last digit ends where the sorted bits do, overlapping the one
        // before it rather than reading bits the sort must not look at. A
        // stable pass on an overlapping digit still leaves the order the
        // digits below set among keys it ties.
        let shift = |pass: usize| lo + (pass * digit).min(bits - digit);
        // Every slot of the buffer is written before it is read, so stale
        // contents from an earlier sort can stay.
        if self.other.len() < m {
            lim.try_resize(&mut self.other, m, 0u64)?;
        }
        self.other.truncate(m);
        lim.try_resize(&mut self.counts, passes * buckets, 0u32)?;
        let counts = &mut self.counts[..passes * buckets];

        // A pass permutes the keys without changing them, so one read counts
        // every pass's digits.
        let mut gate = lim.gate();
        gate.poll(m as u64)?;
        counts.fill(0);
        for &word in keys.iter() {
            for pass in 0..passes {
                counts[pass * buckets + ((word >> shift(pass)) & mask) as usize] += 1;
            }
        }
        for pass in 0..passes {
            gate.poll(m as u64)?;
            let shift = shift(pass);
            let counts = &mut counts[pass * buckets..][..buckets];
            if counts[((keys[0] >> shift) & mask) as usize] as usize == m {
                // Every key holds the same digit, so this pass would copy the
                // buffer onto itself. A bit-packed table often has such a pass.
                continue;
            }
            let mut at = 0u32;
            for count in counts.iter_mut() {
                let here = *count;
                *count = at;
                at += here;
            }
            for &word in keys.iter() {
                let digit = ((word >> shift) & mask) as usize;
                self.other[counts[digit] as usize] = word;
                counts[digit] += 1;
            }
            std::mem::swap(keys, &mut self.other);
        }
        gate.flush()
    }

    /// [`sort`](Self::sort) for sorted bits too wide for
    /// [`RADIX_MAX_PASSES`] passes: one counting pass on their top digit,
    /// then a comparison sort of each digit's run — pieces of a few keys
    /// that stay in the cache, where one comparison sort of every key would
    /// not. The counting pass places nothing when the keys already ascend
    /// in that digit, as the rows of a table sorted on its leading column
    /// but not within it do; the runs are then sorted where they lie.
    fn sort_wide(&mut self, lim: &Limits, keys: &mut Vec<u64>, lo: usize, bits: usize) -> Result<(), OperationError> {
        let m = keys.len();
        let sorted = (lo + bits) as u32;
        // About eight keys a run, in at most 2^16 counts.
        let digit = (m.ilog2() as usize).saturating_sub(3).clamp(RADIX_BITS, 16).min(bits);
        let shift = lo + bits - digit;
        let mask = (1u64 << digit) - 1;
        let top = |key: u64| ((key >> shift) & mask) as usize;
        let mut gate = lim.gate();
        gate.poll(m as u64)?;
        if !keys.windows(2).all(|pair| top(pair[0]) <= top(pair[1])) {
            gate.poll(2 * m as u64)?;
            if self.other.len() < m {
                lim.try_resize(&mut self.other, m, 0u64)?;
            }
            self.other.truncate(m);
            lim.try_resize(&mut self.counts, 1 << digit, 0u32)?;
            let counts = &mut self.counts[..1 << digit];
            counts.fill(0);
            for &key in keys.iter() {
                counts[top(key)] += 1;
            }
            let mut at = 0u32;
            for count in counts.iter_mut() {
                let here = *count;
                *count = at;
                at += here;
            }
            for &key in keys.iter() {
                let d = top(key);
                self.other[counts[d] as usize] = key;
                counts[d] += 1;
            }
            std::mem::swap(keys, &mut self.other);
        }
        gate.poll((m * (bits - digit).div_ceil(RADIX_BITS)) as u64)?;
        for run in keys.chunk_by_mut(|a, b| top(*a) == top(*b)) {
            run.sort_unstable_by_key(|&key| key.rotate_right(sorted));
        }
        gate.flush()
    }
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

/// Row `k` of a buffer of `w`-word rows.
#[inline]
pub(super) fn row_at(packed: &[u64], w: usize, k: u32) -> &[u64] {
    let start = k as usize * w;
    &packed[start..start + w]
}
