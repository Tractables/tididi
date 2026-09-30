//! Radix sorting of bit-packed keys.

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

/// The buffers a radix sort moves keys through, kept for the next sort.
#[derive(Default)]
pub(crate) struct Radix {
    other: Vec<u64>,
    counts: Vec<u32>,
}

impl Radix {
    /// Drop the buffers and hand their charge back.
    pub(crate) fn discard(self, lim: &Limits) {
        lim.discard(self.other);
        lim.discard(self.counts);
    }

    /// The buffer a sort moves keys through, which holds nothing between
    /// sorts and may be lent out, as long as it comes back.
    pub(crate) fn spare(&mut self) -> &mut Vec<u64> {
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
            keys.sort_unstable_by_key(|&key| key.rotate_right(sorted));
            return Ok(());
        }
        if passes > RADIX_MAX_PASSES {
            return self.sort_wide(lim, keys, lo, bits, poll);
        }
        if passes > 1 && m >= RADIX_TOP_MIN_ROWS {
            return self.sort_by_top(lim, keys, lo, bits, poll);
        }
        self.sort_passes(lim, keys, lo, bits, passes, poll)
    }

    /// The passes of [`sort`](Self::sort), least significant digit first:
    /// `passes` digits of equal width over the sorted bits.
    fn sort_passes(
        &mut self,
        lim: &Limits,
        keys: &mut Vec<u64>,
        lo: usize,
        bits: usize,
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
        if self.other.len() < m {
            lim.try_resize(&mut self.other, m, 0u64)?;
        }
        self.other.truncate(m);
        lim.try_resize(&mut self.counts, passes * buckets, 0u32)?;
        let counts = &mut self.counts[..passes * buckets];

        // A pass permutes the keys without changing them, so one read counts
        // every pass's digits.
        poll(m as u64)?;
        counts.fill(0);
        for &word in keys.iter() {
            for pass in 0..passes {
                counts[pass * buckets + ((word >> shift(pass)) & mask) as usize] += 1;
            }
        }
        for pass in 0..passes {
            poll(m as u64)?;
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
        Ok(())
    }

    /// [`sort`](Self::sort) for many keys and more than one pass: the top
    /// digit placed first, in one pass over every key, and then each
    /// digit's run sorted on the bits below it by passes of its own. A pass
    /// over every key writes to as many places at once as its digit has
    /// values, each in a page and a cache line of its own, and that is what
    /// the passes over tens of millions of keys wait on; a run's passes
    /// read and write a few thousand keys that stay in the cache. Both are
    /// a stable sort on the sorted bits, so the order is the passes' own.
    fn sort_by_top(
        &mut self,
        lim: &Limits,
        keys: &mut Vec<u64>,
        lo: usize,
        bits: usize,
        poll: &mut dyn FnMut(u64) -> Result<(), OperationError>,
    ) -> Result<(), OperationError> {
        let m = keys.len();
        // About `2^RUN_BITS` keys a run; the passes over every key left
        // `bits` at least `RADIX_BITS + 1` wide, so a bit is left below.
        let mut digit = (m.ilog2() as usize).saturating_sub(RUN_BITS).clamp(8, RADIX_LARGE_BITS).min(bits - 1);
        // Each pass over a run moves its keys once, and an odd number of
        // passes once more to copy them back. A run's digits take up to
        // `RADIX_WIDE_BITS`, since the run is in the cache and its counts
        // are cleared once for every pass, and the top digit widens, as far
        // as a pass over every key places, to leave the runs two of them.
        if bits - digit > 2 * RADIX_WIDE_BITS && bits - 2 * RADIX_WIDE_BITS <= RADIX_LARGE_BITS {
            digit = bits - 2 * RADIX_WIDE_BITS;
        }
        let rest = bits - digit;
        let (shift, mask, buckets) = (lo + rest, (1u64 << digit) - 1, 1usize << digit);
        let run_digit = rest.div_ceil(rest.div_ceil(RADIX_WIDE_BITS));
        let run_passes = rest.div_ceil(run_digit);
        let Radix { other, counts } = self;
        if other.len() < m {
            lim.try_resize(other, m, 0u64)?;
        }
        other.truncate(m);
        lim.try_resize(counts, buckets + 1 + (run_passes << run_digit), 0u32)?;
        let (top, run_counts) = counts.split_at_mut(buckets + 1);
        poll(m as u64)?;
        top.fill(0);
        for &key in keys.iter() {
            top[((key >> shift) & mask) as usize + 1] += 1;
        }
        let first = ((keys[0] >> shift) & mask) as usize;
        if top[first + 1] as usize == m {
            // One run: the keys stay where they are.
            for (d, end) in top.iter_mut().enumerate() {
                *end = if d < first { 0 } else { m as u32 };
            }
        } else {
            poll(m as u64)?;
            for d in 1..=buckets {
                top[d] += top[d - 1];
            }
            for &key in keys.iter() {
                let d = ((key >> shift) & mask) as usize;
                other[top[d] as usize] = key;
                top[d] += 1;
            }
            std::mem::swap(keys, other);
        }
        // Each digit's run now ends where its count does, and its share of
        // the other buffer is free.
        poll((m * run_passes) as u64)?;
        let sorted = (lo + bits) as u32;
        let mut start = 0;
        for &end in &top[..buckets] {
            let end = end as usize;
            let run = &mut keys[start..end];
            if run.len() < RUN_RADIX_MIN {
                run.sort_unstable_by_key(|&key| key.rotate_right(sorted));
            } else {
                sort_run(run, &mut other[start..end], run_counts, (lo, rest), (run_digit, run_passes));
            }
            start = end;
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
        keys: &mut Vec<u64>,
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
        let top = |key: u64| ((key >> shift) & mask) as usize;
        poll(m as u64)?;
        if !keys.windows(2).all(|pair| top(pair[0]) <= top(pair[1])) {
            poll(2 * m as u64)?;
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
        poll((m * (bits - digit).div_ceil(RADIX_BITS)) as u64)?;
        for run in keys.chunk_by_mut(|a, b| top(*a) == top(*b)) {
            run.sort_unstable_by_key(|&key| key.rotate_right(sorted));
        }
        Ok(())
    }
}

/// Keys from which [`Radix::sort`] places the top digit first when it
/// takes more than one pass: past the second-level cache.
const RADIX_TOP_MIN_ROWS: usize = if cfg!(test) { 1 << 13 } else { 1 << 18 };

/// About how many keys, as a power of two, one run of
/// [`Radix::sort_by_top`] holds.
const RUN_BITS: usize = 12;

/// Keys of a run of [`Radix::sort_by_top`] below which a comparison sort
/// places them: its passes clear their counts per run.
const RUN_RADIX_MIN: usize = 256;

/// A stable sort of `run` on its bits `lo..lo + bits`, in `passes` digits
/// of `digit` bits, the last overlapping the one before it as in
/// [`Radix::sort`], through `buf` as long as `run` and `counts`, which
/// holds `passes << digit` counts at least.
fn sort_run(run: &mut [u64], buf: &mut [u64], counts: &mut [u32], (lo, bits): (usize, usize), (digit, passes): (usize, usize)) {
    let n = run.len();
    let (mask, buckets) = ((1u64 << digit) - 1, 1usize << digit);
    let shift = |pass: usize| lo + (pass * digit).min(bits - digit);
    let counts = &mut counts[..passes * buckets];
    counts.fill(0);
    for &key in run.iter() {
        for pass in 0..passes {
            counts[pass * buckets + ((key >> shift(pass)) & mask) as usize] += 1;
        }
    }
    let mut in_buf = false;
    for pass in 0..passes {
        let shift = shift(pass);
        let counts = &mut counts[pass * buckets..][..buckets];
        let (from, to): (&[u64], &mut [u64]) = if in_buf { (&*buf, &mut *run) } else { (&*run, &mut *buf) };
        if counts[((from[0] >> shift) & mask) as usize] as usize == n {
            continue;
        }
        let mut at = 0u32;
        for count in counts.iter_mut() {
            let here = *count;
            *count = at;
            at += here;
        }
        for &key in from {
            let d = ((key >> shift) & mask) as usize;
            to[counts[d] as usize] = key;
            counts[d] += 1;
        }
        in_buf = !in_buf;
    }
    if in_buf {
        run.copy_from_slice(buf);
    }
}
