//! Radix sorting of bit-packed keys, alone or carried by records.

use crate::limits::{Limits, OperationError};

/// Bits of a key one radix pass places at most, which puts the counts in the
/// first-level cache, while the keys are fewer than a pass of
/// [`RADIX_LARGE_BITS`] would count.
const RADIX_BITS: usize = 11;

/// Rows below which the comparison sort wins: a radix pass is linear but reads
/// and writes the whole buffer whatever the rows look like, and clears its
/// counts. From a few thousand rows up, three passes or fewer take well under
/// the comparison sort's time.
pub(crate) const RADIX_MIN_ROWS: usize = 1 << 11;

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
pub(crate) const RADIX_LARGE_BITS: usize = 14;

/// The buffers a radix sort moves keys or records through, kept for the
/// next sort.
pub(crate) struct Radix<T = u64> {
    other: Vec<T>,
    counts: Vec<u32>,
}

impl<T> Default for Radix<T> {
    fn default() -> Self {
        Radix { other: Vec::new(), counts: Vec::new() }
    }
}

impl<T> crate::execution::pool::Buffers for Radix<T> {
    fn buffers(&mut self, visit: &mut dyn FnMut(&mut dyn crate::execution::pool::Scratch)) {
        visit(&mut self.other);
        visit(&mut self.counts);
    }
}

impl<T> Radix<T> {
    /// Drop the buffers and hand their charge back.
    pub(crate) fn discard(self, lim: &Limits) {
        lim.discard(self.other);
        lim.discard(self.counts);
    }
}

impl Radix {
    /// Sort keys ascending in their low `lo + bits` bits, then in the bits
    /// above, given that the input already lists them in ascending order of
    /// their low `lo` bits, and keys equal in their low `lo + bits` bits in
    /// ascending order of the bits above.
    ///
    /// Under that precondition a stable sort on bits `lo..lo + bits` alone
    /// leaves the ascending order, so a radix sort places only those bits.
    /// A key that carries its position below or above a value, as a
    /// tie-break, sorts this way in as many passes as the value is wide.
    pub(crate) fn sort(&mut self, lim: &Limits, keys: &mut Vec<u64>, lo: usize, bits: usize) -> Result<(), OperationError> {
        let mut gate = lim.gate();
        self.sort_polling(lim, keys, lo, bits, &mut |work| gate.poll(work))?;
        gate.flush()
    }

    /// [`sort`](Self::sort), reporting its work to `poll` rather than to the
    /// work clock, for a caller that charges the work in its own measure.
    pub(crate) fn sort_polling(
        &mut self,
        lim: &Limits,
        keys: &mut Vec<u64>,
        lo: usize,
        bits: usize,
        poll: &mut dyn FnMut(u64) -> Result<(), OperationError>,
    ) -> Result<(), OperationError> {
        self.sort_by_key_polling(lim, keys, |&key| key, lo, bits, poll)
    }
}

impl<T: Copy> Radix<T> {
    /// [`sort_polling`](Radix::sort_polling) of records by the key `key`
    /// gives each, moving the records with their keys.
    pub(crate) fn sort_by_key_polling(
        &mut self,
        lim: &Limits,
        keys: &mut Vec<T>,
        key: impl Fn(&T) -> u64,
        lo: usize,
        bits: usize,
        poll: &mut dyn FnMut(u64) -> Result<(), OperationError>,
    ) -> Result<(), OperationError> {
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
            poll((m * passes) as u64)?;
            keys.sort_unstable_by_key(|k| key(k).rotate_right(sorted));
            return Ok(());
        }
        if passes > RADIX_MAX_PASSES {
            return self.sort_wide(lim, keys, &key, lo, bits, poll);
        }
        self.sort_passes(lim, keys, &key, (lo, bits), passes, poll)
    }

    /// The passes of [`sort`](Self::sort), least significant digit first:
    /// `passes` digits of equal width over the sorted bits.
    fn sort_passes(
        &mut self,
        lim: &Limits,
        keys: &mut Vec<T>,
        key: &impl Fn(&T) -> u64,
        (lo, bits): (usize, usize),
        passes: usize,
        poll: &mut dyn FnMut(u64) -> Result<(), OperationError>,
    ) -> Result<(), OperationError> {
        let m = keys.len();
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
        resize_other(lim, &mut self.other, m, keys[0])?;
        lim.try_resize(&mut self.counts, passes * buckets, 0u32)?;
        let counts = &mut self.counts[..passes * buckets];

        // A pass permutes the keys without changing them, so one read counts
        // every pass's digits.
        poll(m as u64)?;
        count_digits(keys, key, counts, passes, shift, digit);
        for pass in 0..passes {
            poll(m as u64)?;
            let shift = shift(pass);
            let counts = &mut counts[pass * buckets..][..buckets];
            if counts[((key(&keys[0]) >> shift) & mask) as usize] as usize == m {
                // Every key holds the same digit, so this pass would copy the
                // buffer onto itself. A bit-packed table often has such a pass.
                continue;
            }
            scatter_pass(keys, &mut self.other, key, counts, shift, digit);
            std::mem::swap(keys, &mut self.other);
        }
        Ok(())
    }

    /// [`sort`](Self::sort) for sorted bits too wide for
    /// [`RADIX_MAX_PASSES`] passes: one counting pass on their top digit,
    /// then a comparison sort of each digit's run — pieces of a few keys
    /// that stay in the cache, where one comparison sort of every key would
    /// not. The counting pass places nothing when the keys already ascend
    /// in that digit, as the rows of a table sorted on its leading column
    /// but not within it do; the runs are then sorted where they lie.
    fn sort_wide(
        &mut self,
        lim: &Limits,
        keys: &mut Vec<T>,
        key: &impl Fn(&T) -> u64,
        lo: usize,
        bits: usize,
        poll: &mut dyn FnMut(u64) -> Result<(), OperationError>,
    ) -> Result<(), OperationError> {
        let m = keys.len();
        let sorted = (lo + bits) as u32;
        // About eight keys a run, in at most 2^16 counts.
        let digit = (m.ilog2() as usize).saturating_sub(3).clamp(RADIX_BITS, 16).min(bits);
        let shift = lo + bits - digit;
        let mask = (1u64 << digit) - 1;
        let top = |k: &T| ((key(k) >> shift) & mask) as usize;
        poll(m as u64)?;
        if !keys.windows(2).all(|pair| top(&pair[0]) <= top(&pair[1])) {
            poll(2 * m as u64)?;
            resize_other(lim, &mut self.other, m, keys[0])?;
            lim.try_resize(&mut self.counts, 1 << digit, 0u32)?;
            let counts = &mut self.counts[..1 << digit];
            counts.fill(0);
            for k in keys.iter() {
                counts[top(k)] += 1;
            }
            scatter_pass(keys, &mut self.other, key, counts, shift, digit);
            std::mem::swap(keys, &mut self.other);
        }
        poll((m * (bits - digit).div_ceil(RADIX_BITS)) as u64)?;
        for run in keys.chunk_by_mut(|a, b| top(a) == top(b)) {
            run.sort_unstable_by_key(|k| key(k).rotate_right(sorted));
        }
        Ok(())
    }
}

/// Resize `other` to exactly `m` slots, filling new ones with `fill`.
#[inline]
fn resize_other<T: Clone>(lim: &Limits, other: &mut Vec<T>, m: usize, fill: T) -> Result<(), OperationError> {
    lim.try_resize(other, m, fill)?;
    other.truncate(m);
    Ok(())
}

/// Prefix-sum `counts` in place and scatter `from` into `to` by the
/// `digit` bits at `shift`.
#[inline]
fn scatter_pass<T: Copy>(from: &[T], to: &mut [T], key: &impl Fn(&T) -> u64, counts: &mut [u32], shift: usize, digit: usize) {
    // Counts of exactly `1 << digit` let the compiler drop the bounds check
    // of a masked digit.
    let (counts, mask) = (&mut counts[..1 << digit], (1u64 << digit) - 1);
    let mut at = 0u32;
    for count in counts.iter_mut() {
        let here = *count;
        *count = at;
        at += here;
    }
    for k in from {
        let d = ((key(k) >> shift) & mask) as usize;
        to[counts[d] as usize] = *k;
        counts[d] += 1;
    }
}

/// Count the digits of every pass in one read of `keys`: pass `p`'s
/// `digit` bits at `shift(p)` into `counts[p << digit..]`, cleared first.
fn count_digits<T>(keys: &[T], key: &impl Fn(&T) -> u64, counts: &mut [u32], passes: usize, shift: impl Fn(usize) -> usize, digit: usize) {
    match passes {
        1 => count_passes::<T, 1>(keys, key, counts, shift, digit),
        2 => count_passes::<T, 2>(keys, key, counts, shift, digit),
        _ => count_passes::<T, RADIX_MAX_PASSES>(keys, key, counts, shift, digit),
    }
}

/// [`count_digits`] for `P` passes. With the pass count and the count
/// tables' lengths known to the compiler, a key's passes unroll without
/// bounds checks; a loop over a run-time count recomputed every shift and
/// checked every index, at twice the instructions.
fn count_passes<T, const P: usize>(keys: &[T], key: &impl Fn(&T) -> u64, counts: &mut [u32], shift: impl Fn(usize) -> usize, digit: usize) {
    let (buckets, mask) = (1usize << digit, (1u64 << digit) - 1);
    let shifts: [usize; P] = std::array::from_fn(shift);
    let counts = &mut counts[..P * buckets];
    counts.fill(0);
    let mut tables = counts.chunks_exact_mut(buckets);
    let tables: [&mut [u32]; P] = std::array::from_fn(|_| tables.next().unwrap_or_default());
    for k in keys {
        let k = key(k);
        for pass in 0..P {
            tables[pass][((k >> shifts[pass]) & mask) as usize] += 1;
        }
    }
}
