//! The atoms of every constrained vtree node, computed from the root down.
//!
//! A node's atoms follow from its parent's. Write a parent value as `(v, s)`,
//! `v` the node's part and `s` its sibling's. The completions of `v` are the
//! pairs of an `s` and a completion of `(v, s)`, and a parent atom is exactly
//! a class of parent values with equal completion sets. So `v` and `v'` share
//! an atom when the sets `{(s, atom of (v, s))}` over the parent values that
//! occur are equal. Each node therefore reads its parent's distinct values,
//! not the rows, and the distinct values thin out towards the leaves.
//!
//! Only the leaves and the nodes constrained on both sides are split. A node
//! whose other side is free takes its constrained child's values unchanged.
//!
//! A node's value holds its right child's variables in its low bits and its
//! left child's above them (see `Layout`). The split works in those terms:
//! a value's *low* part is its right child's value and its *high* part its
//! left child's, and the parent values ascend by high part first.

use std::borrow::Cow;

use crate::diagram::{NEG_LEAF_IDX, NodeIdx, ONE_LEAF_IDX, POS_LEAF_IDX};
use crate::limits::{Charged, Limits, OperationError};
use crate::vtree::Vtree;

use super::layout::Layout;
use super::plan::{Decomposition, Emitted, Plan, sorted_triples};
use super::rows::{index_bits, same_row, words_per_row};
use crate::sort::Radix;

/// The distinct values one constrained vtree node takes over the rows, and
/// the atom each belongs to.
struct Values<'a> {
    /// Words per value, low word first.
    words: usize,
    /// The values in ascending order, `words` apiece. Bits past the node's
    /// width are clear. The root's may be the caller's rows, borrowed where
    /// they came sorted and distinct; a split that overwrites its parent's
    /// values copies them first.
    data: Cow<'a, [u64]>,
    /// The atom of each value. Atoms are numbered by their smallest value.
    atom: Vec<u32>,
    /// How many atoms there are.
    atoms: u32,
}

impl Values<'_> {
    fn len(&self) -> usize {
        self.atom.len()
    }
}

impl Values<'static> {
    fn owned(words: usize, data: Vec<u64>, (atom, atoms): (Vec<u32>, u32)) -> Self {
        Values {
            words,
            data: data.into(),
            atom,
            atoms,
        }
    }
}

impl Charged for Values<'_> {
    fn charged_bytes(&self) -> u64 {
        let data = match &self.data {
            Cow::Owned(data) => data.charged_bytes(),
            Cow::Borrowed(_) => 0,
        };
        data + self.atom.charged_bytes()
    }
}

/// The buffers one split reuses from the last.
#[derive(Default)]
struct Scratch {
    /// Each parent value's low part, `low_words` apiece.
    low: Vec<u64>,
    /// Each parent value's high part, `high_words` apiece.
    high: Vec<u64>,
    /// The parent values ordered by their low part; in `split_hashed`, the
    /// occupied low slots.
    order: Vec<u32>,
    /// Sort keys: a low part beside the value's index, below it on the
    /// one-word path (see `KeyLayout`) and above it otherwise.
    sort_keys: Vec<u64>,
    /// The low child's value index of each parent value.
    low_of: Vec<u32>,
    /// The high child's value index of each parent value; in
    /// `hashed_split`, each low slot's atom.
    high_of: Vec<u32>,
    /// Where each low value's run starts in the low order. In
    /// `confirm_lows`, where each low slot's list starts, and in
    /// `by_parent_atom`, where each parent atom's pairs start.
    low_starts: Vec<u32>,
    /// Where each high value's run starts among the parent values.
    high_starts: Vec<u32>,
    /// Per high value, the next of its run's values the low order meets,
    /// or of its slots among the node's pairs.
    cursor: Vec<u32>,
    /// Per parent value, its low value index above its atom (in
    /// `runs_of_words`, the low value itself): the keys of the high child's
    /// runs. Once they are numbered, the packed triples. In `split_hashed`,
    /// each occupied low slot's run hashed.
    high_keys: Vec<u64>,
    /// Per parent value in low order, its high value index above its
    /// atom: the keys of the low child's runs. In `confirm_lows`, the
    /// lists of the low slots whose hashes are shared.
    low_keys: Vec<u64>,
    /// The table that numbers runs of keys.
    runs: RunTable,
    /// The run each low atom first appears in, by atom.
    low_first: Vec<u32>,
    /// The run each high atom first appears in, by atom.
    high_first: Vec<u32>,
    /// Packed triples too wide for one word.
    wide_triples: Vec<u128>,
    /// Each high value's run hashed.
    high_hash: Vec<u64>,
    /// The hashes of the low runs of `group_single`, in run order (see
    /// `low_runs`).
    low_hash: Vec<u64>,
    /// The open-addressed table that numbers hashed runs.
    slots: Vec<u64>,
    /// Each low slot's run hashed beside its value count, in
    /// `split_hashed`, where one pass updates both: side by side, a value
    /// reaches both in one cache line. Once the lists are laid out, the
    /// count is where the slot's next pair goes.
    low_tally: Vec<(u64, u32)>,
    /// Each low slot's rank among the low values.
    rank: Vec<u32>,
    /// The radix sort's buffers.
    radix: Radix,
}

impl Scratch {
    /// Drop the buffers and hand their charge back.
    fn discard(self, lim: &Limits) {
        let Scratch {
            low,
            high,
            order,
            sort_keys,
            low_of,
            high_of,
            low_starts,
            high_starts,
            cursor,
            high_keys,
            low_keys,
            runs,
            low_first,
            high_first,
            wide_triples,
            high_hash,
            low_hash,
            slots,
            low_tally,
            rank,
            radix,
        } = self;
        for buf in [
            low, high, sort_keys, high_keys, low_keys, high_hash, low_hash, slots,
        ] {
            lim.discard(buf);
        }
        for buf in [
            order,
            low_of,
            high_of,
            low_starts,
            high_starts,
            cursor,
            low_first,
            high_first,
            rank,
        ] {
            lim.discard(buf);
        }
        lim.discard(low_tally);
        lim.discard(wide_triples);
        let RunTable {
            hashes,
            slots: run_slots,
            direct,
        } = runs;
        lim.discard(hashes);
        lim.discard(run_slots);
        lim.discard(direct);
        radix.discard(lim);
    }
}

/// Numbers runs of keys by their content.
#[derive(Default)]
struct RunTable {
    /// Each run's hash.
    hashes: Vec<u64>,
    /// The open-addressed table of [`number_hashed`].
    slots: Vec<u64>,
    /// The number of each lone key, addressed by the key's two halves, when
    /// their ranges are small enough to address.
    direct: Vec<u32>,
}

/// Compute the atoms of every constrained node from the sorted distinct rows,
/// which hold `w` words each. Returns a plan for every constrained leaf and
/// every node constrained on both sides, indexed by vtree node.
pub(super) fn plan(
    lim: &Limits,
    vtree: &Vtree,
    layout: &Layout,
    sorted: Cow<'_, [u64]>,
    w: usize,
    radix: Radix,
) -> Result<Vec<Option<Plan>>, OperationError> {
    let mut atom = Vec::new();
    lim.try_resize(&mut atom, sorted.len() / w, 0u32)?;
    // Every constrained variable lies under the lowest node whose subtree
    // holds them all, so every row there has the same empty completion.
    let whole = Values {
        words: w,
        data: sorted,
        atom,
        atoms: 1,
    };
    let mut scratch = Scratch {
        radix,
        ..Scratch::default()
    };
    let plans = super::plan::plan(
        lim,
        vtree,
        layout,
        whole,
        |values| leaf_nodes(lim, values),
        |values, widths| split(lim, &mut scratch, values, widths),
    );
    scratch.discard(lim);
    plans
}

/// A leaf has one literal atom, two literal atoms, or one free atom.
fn leaf_nodes(lim: &Limits, values: &Values) -> Result<Vec<NodeIdx>, OperationError> {
    let mut locals = Vec::new();
    lim.reserve_exact(&mut locals, 2)?;
    match (values.len(), values.atoms) {
        (2, 1) => locals.push(ONE_LEAF_IDX),
        (2, _) => locals.extend([NEG_LEAF_IDX, POS_LEAF_IDX]),
        _ => locals.push(if values.data[0] == 0 {
            NEG_LEAF_IDX
        } else {
            POS_LEAF_IDX
        }),
    }
    Ok(locals)
}

/// Split a node's values into its children's, whose widths are `(low,
/// high)` bits: the low part of a value is its right child's value. The
/// parent's values may be overwritten.
fn split(
    lim: &Limits,
    s: &mut Scratch,
    parent: &mut Values,
    widths: (usize, usize),
) -> Result<(Decomposition, Values<'static>, Values<'static>), OperationError> {
    let n = parent.len();
    // A value that is its own atom leaves every child value its own atom:
    // a completion names the parent atom, which only that value has.
    let distinct = parent.atoms as usize == n;
    // Values that share atoms split without a sort when their low part
    // addresses a table, and with one when a single atom holds them all.
    if !distinct {
        if let Some(out) = split_hashed(lim, s, parent, widths.0)? {
            return Ok(out);
        }
        if parent.atoms == 1
            && parent.words == 1
            && let Some(out) = split_single(lim, s, parent, widths)?
        {
            return Ok(out);
        }
    }
    let parent = &*parent;
    let key = match parent.words == 1 {
        true => word_key(lim, parent, widths, distinct)?,
        false => None,
    };
    split_parts(lim, s, parent, widths, distinct, key)
}

/// The sort key of one-word values, if one fits a word. Unless every value
/// is its own atom, a high run names its low partners by value, above their
/// atoms, which takes a low part of at most 32 bits.
fn word_key(
    lim: &Limits,
    parent: &Values,
    (low_width, high_width): (usize, usize),
    distinct: bool,
) -> Result<Option<KeyLayout>, OperationError> {
    if !distinct && low_width > 32 {
        return Ok(None);
    }
    // The high values number at most the parent values and the high part's
    // range; only when that bound leaves the index out are they counted.
    let n = parent.len();
    let bound = n.min(1usize << high_width.min(usize::BITS as usize - 1));
    let key = KeyLayout::new(n, low_width, bound, parent.atoms, distinct);
    if key.is_some_and(|key| key.indexed) {
        return Ok(key);
    }
    let high_runs = count_high_runs(lim, parent, low_width)?;
    Ok(KeyLayout::new(n, low_width, high_runs, parent.atoms, distinct))
}

/// How many distinct high parts one-word values have, the low part being
/// their low `low_width` bits.
fn count_high_runs(
    lim: &Limits,
    parent: &Values,
    low_width: usize,
) -> Result<usize, OperationError> {
    lim.gate().poll(parent.len() as u64)?;
    let runs = parent
        .data
        .windows(2)
        .filter(|pair| pair[0] >> low_width != pair[1] >> low_width)
        .count();
    Ok(runs + usize::from(!parent.data.is_empty()))
}

/// The fields of a one-word sort key, low to high: the value's low part,
/// its index, and unless every value is its own atom, its atom and its high
/// value's index. Unless every value is its own atom, the index is left out
/// when the key does not fit a word with it.
#[derive(Clone, Copy)]
struct KeyLayout {
    /// Bits of the low part.
    low_width: usize,
    /// Whether the key carries the value's index.
    indexed: bool,
    /// Bits of the index.
    index_bits: usize,
    /// Bits of the atom.
    atom_bits: usize,
    /// Bits of the high value's index.
    run_bits: usize,
}

impl KeyLayout {
    /// The layout for `n` values, if one fits a word.
    fn new(
        n: usize,
        low_width: usize,
        high_runs: usize,
        atoms: u32,
        distinct: bool,
    ) -> Option<KeyLayout> {
        let (atom_bits, run_bits) = match distinct {
            true => (0, 0),
            false => (
                index_bits(atoms as usize) as usize,
                index_bits(high_runs) as usize,
            ),
        };
        let indexed = KeyLayout {
            low_width,
            indexed: true,
            index_bits: index_bits(n) as usize,
            atom_bits,
            run_bits,
        };
        let bare = KeyLayout {
            indexed: false,
            index_bits: 0,
            ..indexed
        };
        match (indexed.fits(), distinct) {
            (true, _) => Some(indexed),
            (false, false) if bare.fits() => Some(bare),
            _ => None,
        }
    }

    /// Whether every field fits, with every shift below a word.
    fn fits(self) -> bool {
        self.low_width + self.index_bits + self.atom_bits + self.run_bits < 64
    }

    fn key(self, low: u64, index: usize, atom: u32, run: usize) -> u64 {
        let index = if self.indexed { index as u64 } else { 0 };
        let above = ((run as u64) << self.atom_bits | atom as u64) << self.index_bits | index;
        above << self.low_width | low
    }

    fn low(self, key: u64) -> u64 {
        key & ((1u64 << self.low_width) - 1)
    }

    /// The value's index, when the key carries it.
    fn index(self, key: u64) -> usize {
        (key >> self.low_width & ((1u64 << self.index_bits) - 1)) as usize
    }

    fn atom(self, key: u64) -> u32 {
        (key >> (self.low_width + self.index_bits) & ((1u64 << self.atom_bits) - 1)) as u32
    }

    fn run(self, key: u64) -> usize {
        (key >> (self.low_width + self.index_bits + self.atom_bits)) as usize
    }
}

/// Split one-word values that all lie in one parent atom, as the values of
/// the node over every constrained variable do.
///
/// A completion then names the one atom only, so a high value's atom is the
/// set of its low parts and a low value's the set of its high values, and
/// the node's pairs are the distinct pairs of child atoms. Each side's runs
/// are hashed as they are found, so numbering them reads a hash per run
/// rather than the keys, and the keys themselves are only read to confirm a
/// match. The values are overwritten by their sort keys, the low part
/// below the high value's index, which one pass writes, the radix sort
/// orders by the low part, and `group_single` reads. `None` when a key does
/// not fit a word.
fn split_single(
    lim: &Limits,
    s: &mut Scratch,
    parent: &mut Values,
    (low_width, high_width): (usize, usize),
) -> Result<Option<(Decomposition, Values<'static>, Values<'static>)>, OperationError> {
    let n = parent.len();
    let bound = n.min(1usize << high_width.min(usize::BITS as usize - 1));
    if low_width + index_bits(bound) as usize >= 64
        && low_width + index_bits(count_high_runs(lim, parent, low_width)?) as usize >= 64
    {
        return Ok(None);
    }
    let mask = (1u64 << low_width) - 1;
    let mut gate = lim.gate();

    // The values ascend by high part, so each high value is one run of
    // them, and a run lists its low parts in ascending order.
    gate.poll(n as u64)?;
    s.high_starts.clear();
    s.high_hash.clear();
    s.high.clear();
    if let Cow::Borrowed(rows) = parent.data {
        let mut owned = Vec::new();
        lim.reserve_exact(&mut owned, n)?;
        owned.extend_from_slice(rows);
        parent.data = Cow::Owned(owned);
    }
    let data = parent.data.to_mut();
    let mut last = data[0] >> low_width;
    lim.try_push(&mut s.high_starts, 0)?;
    lim.try_push(&mut s.high, last)?;
    let (mut hash, mut index) = (RUN_SEED, 0u64);
    for (e, slot) in data.iter_mut().enumerate() {
        let (high, low) = (*slot >> low_width, *slot & mask);
        if high != last {
            lim.try_push(&mut s.high_hash, hash)?;
            lim.try_push(&mut s.high_starts, e as u32)?;
            lim.try_push(&mut s.high, high)?;
            (hash, index, last) = (RUN_SEED, index + 1, high);
        }
        hash = mix(hash, low);
        *slot = index << low_width | low;
    }
    lim.try_push(&mut s.high_hash, hash)?;
    gate.flush()?;
    let (high_starts, keys) = (&s.high_starts, &parent.data);
    let same_lows = |a: u32, b: u32| {
        let (x, y) = (run(high_starts, a, n), run(high_starts, b, n));
        x.len() == y.len()
            && keys[x]
                .iter()
                .zip(&keys[y])
                .all(|(&p, &q)| (p ^ q) & mask == 0)
    };
    let (high_atom, high_atoms) =
        number_hashed(lim, &mut s.slots, &s.high_hash, &mut s.high_first, same_lows)?;

    // Keys of one low part ascend by high value, as the values did.
    s.radix.sort(lim, parent.data.to_mut(), 0, low_width)?;
    lim.check_stop()?;
    let (split, low_atom, low_atoms) = group_single(lim, s, &parent.data, low_width, &high_atom)?;
    let low_data = low_parts(lim, &parent.data, &s.low_starts, mask)?;
    // The high runs' parts are the high child's values, so their buffer is
    // handed over rather than copied.
    let low = Values::owned(1, low_data, (low_atom, low_atoms));
    let high = Values::owned(1, std::mem::take(&mut s.high), (high_atom, high_atoms));
    Ok(Some((split, low, high)))
}

/// The low part of each run of `keys` that starts at `starts`, in order:
/// the low child's values.
fn low_parts(
    lim: &Limits,
    keys: &[u64],
    starts: &[u32],
    mask: u64,
) -> Result<Vec<u64>, OperationError> {
    lim.gate().poll(starts.len() as u64)?;
    let mut out = Vec::new();
    lim.reserve_exact(&mut out, starts.len())?;
    out.extend(starts.iter().map(|&start| keys[start as usize] & mask));
    Ok(out)
}

/// The low side of `split_single` and the node's pairs, from the keys
/// sorted by low part. Each low value's run and hash go into `s`. Returns
/// the pairs, each low value's atom, and how many low atoms there are.
///
/// Runs of one atom list the same keys, so the first run of each high atom
/// lists all the node's pairs. The keys come by low value, and each one is
/// written at its high value's next slot among those runs' pairs.
fn group_single(
    lim: &Limits,
    s: &mut Scratch,
    keys: &[u64],
    low_width: usize,
    high_atom: &[u32],
) -> Result<(Decomposition, Vec<u32>, u32), OperationError> {
    let n = keys.len();
    let mask = (1u64 << low_width) - 1;
    low_runs(lim, s, keys, low_width)?;
    let low_count = s.low_starts.len();
    let low_starts = &s.low_starts;
    let same_highs = |a: u32, b: u32| {
        let (x, y) = (run(low_starts, a, n), run(low_starts, b, n));
        x.len() == y.len()
            && keys[x]
                .iter()
                .zip(&keys[y])
                .all(|(&p, &q)| p >> low_width == q >> low_width)
    };
    let (low_atom, low_atoms) =
        number_hashed(lim, &mut s.slots, &s.low_hash, &mut s.low_first, same_highs)?;

    // Where each high value's low values go: the first run of each high
    // atom lists the atom's, in atom order, and the other runs none.
    let merged = low_atoms as usize != low_count;
    let high_count = s.high_starts.len();
    s.cursor.clear();
    lim.reserve_exact(&mut s.cursor, high_count)?;
    let mut held = 0u32;
    for (high, range) in runs(&s.high_starts, n).enumerate() {
        if s.high_first[high_atom[high] as usize] as usize == high {
            s.cursor.push(held);
            held += range.len() as u32;
        } else {
            s.cursor.push(u32::MAX);
        }
    }
    let mut lows = Vec::new();
    lim.try_resize(&mut lows, held as usize, 0u32)?;
    let mut gate = lim.gate();
    gate.poll(n as u64)?;
    let (mut last, mut index) = (keys[0] & mask, 0u32);
    for &key in keys {
        index += u32::from(key & mask != last);
        last = key & mask;
        let at = &mut s.cursor[(key >> low_width) as usize];
        if *at != u32::MAX {
            lows[*at as usize] = if merged { low_atom[index as usize] } else { index };
            *at += 1;
        }
    }
    // Each group's low values ascend as the low order meets them, and so do
    // their atoms unless low values merge.
    let sizes = s
        .high_first
        .iter()
        .map(|&k| run(&s.high_starts, k, n).len());
    let ends = close_groups(lim, &mut lows, sizes, merged)?;
    gate.flush()?;
    Ok((Decomposition::Grouped { ends, lows }, low_atom, low_atoms))
}

/// Total length of the runs in `starts` indexed by `firsts`.
#[inline(always)]
fn held_keys(starts: &[u32], firsts: &[u32], n: usize) -> usize {
    firsts.iter().map(|&k| run(starts, k, n).len()).sum()
}

/// Copy unique adjacent elements from `v[start..end]` to `v[kept..]`,
/// returning the new `kept` cursor.
#[inline(always)]
fn dedup_range<T: Copy + PartialEq>(
    v: &mut [T],
    start: usize,
    end: usize,
    mut kept: usize,
) -> usize {
    for at in start..end {
        if at == start || v[at] != v[kept - 1] {
            v[kept] = v[at];
            kept += 1;
        }
    }
    kept
}

/// Drop adjacent duplicates from `v[start..]`.
#[inline(always)]
pub(super) fn dedup_tail<T: Copy + PartialEq>(v: &mut Vec<T>, start: usize) {
    let len = v.len();
    let kept = dedup_range(v, start, len, start);
    v.truncate(kept);
}

/// The passes of `group_single` over the keys sorted by low part: where
/// each low value's run starts, into `s.low_starts`, and then the runs'
/// hashes into `s.low_hash`.
fn low_runs(
    lim: &Limits,
    s: &mut Scratch,
    keys: &[u64],
    low_width: usize,
) -> Result<(), OperationError> {
    let n = keys.len();
    let mask = (1u64 << low_width) - 1;
    let mut gate = lim.gate();
    gate.poll(n as u64)?;
    s.low_starts.clear();
    // A run starts at most at every key, and at most at every low part
    // from the first key's to the last's, since the keys ascend by low
    // part, so the starts never grow the buffer past this.
    let most = runs_between(keys[0] & mask, keys[n - 1] & mask, n);
    lim.reserve_exact(&mut s.low_starts, most)?;
    s.low_starts.push(0);
    let mut last = keys[0] & mask;
    for (i, &key) in keys.iter().enumerate() {
        let low = key & mask;
        if low != last {
            s.low_starts.push(i as u32);
            last = low;
        }
    }
    let count = s.low_starts.len();
    debug_assert!(count <= most, "the keys ascend by low part");
    gate.poll(n as u64)?;
    s.low_hash.clear();
    lim.reserve_exact(&mut s.low_hash, count)?;
    for range in runs(&s.low_starts, n) {
        s.low_hash.push(
            keys[range]
                .iter()
                .fold(RUN_SEED, |hash, &key| mix(hash, key >> low_width)),
        );
    }
    gate.flush()?;
    Ok(())
}

/// Where each group of `lows` ends, the groups listed one after another,
/// `sizes` long. When low values merged, a group can list an atom twice
/// and out of order, so each such group is sorted and its repeats dropped,
/// which moves the later groups down; one that ascends strictly, as a group
/// holding no two merged values does, is only moved.
fn close_groups(
    lim: &Limits,
    lows: &mut Vec<u32>,
    sizes: impl ExactSizeIterator<Item = usize>,
    merged: bool,
) -> Result<Vec<u32>, OperationError> {
    let mut ends = Vec::new();
    lim.reserve_exact(&mut ends, sizes.len())?;
    let (mut start, mut kept) = (0, 0);
    for size in sizes {
        let end = start + size;
        if merged {
            if !lows[start..end].windows(2).all(|pair| pair[0] < pair[1]) {
                lows[start..end].sort_unstable();
            }
            kept = dedup_range(lows, start, end, kept);
        } else {
            kept = end;
        }
        ends.push(kept as u32);
        start = end;
    }
    lows.truncate(kept);
    Ok(ends)
}

/// Split one-word values whose low part is narrow enough to address, into
/// children whose atoms are found by hashing rather than sorting.
///
/// The parent's order lists each high value's pairs of a low part and an
/// atom in ascending order and, since it ascends by high value, meets each
/// low value's pairs of a high value and an atom in ascending order too.
/// One pass therefore hashes both sides' completion sets: a high value's
/// as its run ends, a low value's in the slot its low part addresses, which
/// every value of it updates. Reading the slots in order ranks the low
/// values without a sort. A high value's hash that agrees with another is
/// confirmed against the runs themselves. The low values are numbered by
/// hash alone, which merges every two that share an atom. Where it merges
/// any, the pairs it gives are checked by counting the values their
/// products hold as they are built: only a collision, which makes the count
/// exceed the parent's values, lists the low values' sets, high value by
/// high value, to confirm them, and builds the pairs again.
///
/// The first run of each high atom realizes all the triples there are.
/// With one parent atom they come out as its pairs, high atom by high
/// atom, sorted only where low values merge. Otherwise a counting sort by
/// parent atom groups them, and each group is sorted and deduplicated.
/// `None` when the low part is too wide to address.
fn split_hashed(
    lim: &Limits,
    s: &mut Scratch,
    parent: &Values,
    low_width: usize,
) -> Result<Option<(Decomposition, Values<'static>, Values<'static>)>, OperationError> {
    let n = parent.len();
    // The table's size is compared in 64 bits: on a 32-bit target
    // `1usize << 32` wraps to 1, which let a 32-bit low part through to a
    // table of one slot.
    if parent.words != 1 || low_width > 32 || 1u64 << low_width > (2 * n).max(DIRECT_LOWS) as u64 {
        return Ok(None);
    }
    let single = parent.atoms == 1;
    let mask = (1u64 << low_width) - 1;
    let (data, atoms) = (&parent.data, &parent.atom);
    let mut gate = lim.gate();

    gate.poll(n as u64)?;
    s.high_starts.clear();
    s.high_hash.clear();
    s.high.clear();
    s.low_tally.clear();
    lim.try_resize(&mut s.low_tally, 1 << low_width, (RUN_SEED, 0u32))?;
    lim.try_push(&mut s.high_starts, 0)?;
    lim.try_push(&mut s.high, data[0] >> low_width)?;
    if single {
        hash_sides::<true>(lim, s, parent, low_width)?;
    } else {
        hash_sides::<false>(lim, s, parent, low_width)?;
    }
    let high_count = s.high_starts.len();
    let high_starts = &s.high_starts;
    let same_pairs = |a: u32, b: u32| {
        let (x, y) = (run(high_starts, a, n), run(high_starts, b, n));
        x.len() == y.len()
            && data[x.clone()]
                .iter()
                .zip(&data[y.clone()])
                .all(|(&p, &q)| (p ^ q) & mask == 0)
            && (single || atoms[x] == atoms[y])
    };
    let (high_atom, high_atoms) =
        number_hashed(lim, &mut s.slots, &s.high_hash, &mut s.high_first, same_pairs)?;

    // The low values in ascending order, each slot's rank among them, and
    // the hashes in that order.
    let lows = s.low_tally.iter().filter(|&&(_, count)| count > 0).count();
    s.order.clear();
    lim.reserve_exact(&mut s.order, lows)?;
    s.rank.clear();
    lim.try_resize(&mut s.rank, 1 << low_width, 0u32)?;
    s.high_keys.clear();
    lim.reserve_exact(&mut s.high_keys, lows)?;
    let mut low_data = Vec::new();
    lim.reserve_exact(&mut low_data, lows)?;
    gate.poll(1 << low_width)?;
    for (slot, &(hash, count)) in s.low_tally.iter().enumerate() {
        if count > 0 {
            s.rank[slot] = s.order.len() as u32;
            s.order.push(slot as u32);
            s.high_keys.push(hash);
            low_data.push(slot as u64);
        }
    }
    // Equal sets hash alike, so the numbering by hash merges every pair of
    // low values that shares an atom, and a collision merges more. Every
    // parent value lies in the product of its two child atoms, and that
    // product is among its parent atom's pairs, so the products cover the
    // values; they hold no other value and none twice exactly when their
    // sizes sum to the parent's values.
    let (mut low_atom, mut low_atoms) =
        number_hashed(lim, &mut s.slots, &s.high_keys, &mut s.low_first, |_, _| {
            true
        })?;
    let mut split = None;
    if low_atoms as usize != lows {
        let sizes = (
            atom_sizes(lim, &high_atom, high_atoms)?,
            atom_sizes(lim, &low_atom, low_atoms)?,
        );
        let low = (&low_atom[..], low_atoms);
        let (built, models) = hashed_split(lim, s, parent, low_width, &high_atom, low, Some(&sizes))?;
        lim.discard(sizes.0);
        lim.discard(sizes.1);
        if models == n as u128 {
            split = Some(built);
        } else {
            built.discard(lim);
            let by_hash = low_atom;
            (low_atom, low_atoms) = confirm_lows(lim, s, parent, low_width, &by_hash, low_atoms)?;
            lim.discard(by_hash);
        }
    }
    let split = match split {
        Some(split) => split,
        None => hashed_split(lim, s, parent, low_width, &high_atom, (&low_atom, low_atoms), None)?.0,
    };
    gate.flush()?;

    let mut high_data = Vec::new();
    lim.reserve_exact(&mut high_data, high_count)?;
    high_data.extend_from_slice(&s.high);
    let low = Values::owned(1, low_data, (low_atom, low_atoms));
    let high = Values::owned(1, high_data, (high_atom, high_atoms));
    Ok(Some((split, low, high)))
}

/// The node's pairs in `split_hashed` once the low values are numbered by
/// `low_atom` (by rank): the first run of each high atom realizes all the
/// triples there are. With one parent atom they come out as its pairs, high
/// atom by high atom, sorted only where low values merge. Otherwise a
/// counting sort by parent atom groups them, and each group is sorted and
/// deduplicated. Leaves each low slot's atom in `s.high_of`. With `sizes`,
/// each high and low atom's value count, also returns how many values the
/// pairs' products hold, counted while each group is in the cache.
fn hashed_split(
    lim: &Limits,
    s: &mut Scratch,
    parent: &Values,
    low_width: usize,
    high_atom: &[u32],
    (low_atom, low_atoms): (&[u32], u32),
    sizes: Option<&(Vec<u64>, Vec<u64>)>,
) -> Result<(Decomposition, u128), OperationError> {
    let n = parent.len();
    let mut models = 0u128;
    let merged = low_atoms as usize != low_atom.len();
    let mask = (1u64 << low_width) - 1;
    let data = &parent.data;
    let atom_of = &mut s.high_of;
    atom_of.clear();
    lim.reserve_exact(atom_of, 1 << low_width)?;
    atom_of.extend(s.rank.iter().map(|&rank| {
        if merged {
            low_atom[rank as usize]
        } else {
            rank
        }
    }));
    let held = held_keys(&s.high_starts, &s.high_first, n);
    let mut gate = lim.gate();
    gate.poll(held as u64)?;
    let split = if parent.atoms == 1 {
        // Each group of the one parent atom lists its low parts in ascending
        // order, and so their atoms unless low values merge. A group is
        // closed as it is written, while it is in the cache: sorted and its
        // repeats dropped where merged values leave it out of order.
        let mut lows = Vec::new();
        lim.reserve_exact(&mut lows, held)?;
        let mut ends = Vec::new();
        lim.reserve_exact(&mut ends, s.high_first.len())?;
        for (h, &k) in s.high_first.iter().enumerate() {
            let start = lows.len();
            lows.extend(
                data[run(&s.high_starts, k, n)]
                    .iter()
                    .map(|&value| atom_of[(value & mask) as usize]),
            );
            if merged && !lows[start..].windows(2).all(|pair| pair[0] < pair[1]) {
                lows[start..].sort_unstable();
                dedup_tail(&mut lows, start);
            }
            if let Some((high_size, low_size)) = sizes {
                let width: u64 = lows[start..].iter().map(|&l| low_size[l as usize]).sum();
                models += u128::from(high_size[h]) * u128::from(width);
            }
            ends.push(lows.len() as u32);
        }
        Decomposition::Grouped { ends, lows }
    } else {
        let (ends, pairs) = by_parent_atom(lim, s, parent, low_width, high_atom, held)?;
        if let Some((high_size, low_size)) = sizes {
            models = pairs
                .iter()
                .map(|&pair| {
                    u128::from(high_size[(pair >> 32) as usize])
                        * u128::from(low_size[pair as u32 as usize])
                })
                .sum();
        }
        Decomposition::ByAtom { ends, pairs }
    };
    gate.flush()?;
    Ok((split, models))
}

/// How many values each of `atoms` atoms holds, `atom` naming each value's.
fn atom_sizes(lim: &Limits, atom: &[u32], atoms: u32) -> Result<Vec<u64>, OperationError> {
    let mut size = Vec::new();
    lim.try_resize(&mut size, atoms as usize, 0u64)?;
    for &a in atom {
        size[a as usize] += 1;
    }
    Ok(size)
}

impl Decomposition {
    /// Hand back the buffers of a split that is rebuilt.
    fn discard(self, lim: &Limits) {
        match self {
            Decomposition::Grouped { ends, lows } => {
                lim.discard(ends);
                lim.discard(lows);
            }
            Decomposition::ByAtom { ends, pairs } => {
                lim.discard(ends);
                lim.discard(pairs);
            }
            Decomposition::Triples { triples, .. } => lim.discard(triples),
        }
    }
}

/// The pass of `split_hashed` over the parent's values: each high value's
/// run, its start and its hash, and each low slot's hash and value count.
/// `SINGLE` when one atom holds every value, which the pairs then leave out.
fn hash_sides<const SINGLE: bool>(
    lim: &Limits,
    s: &mut Scratch,
    parent: &Values,
    low_width: usize,
) -> Result<(), OperationError> {
    let mask = (1u64 << low_width) - 1;
    // A high value's run starts at most at every value, and at most at
    // every high part from the first value's to the last's, since the
    // values ascend by high part; reserved at once, the pass never grows
    // the buffers.
    let n = parent.len();
    let most = runs_between(
        parent.data[0] >> low_width,
        parent.data[n - 1] >> low_width,
        n,
    );
    lim.reserve_exact(&mut s.high_hash, most)?;
    lim.reserve_exact(&mut s.high_starts, most)?;
    lim.reserve_exact(&mut s.high, most)?;
    let mut last = parent.data[0] >> low_width;
    let (mut hash, mut index) = (RUN_SEED, 0u64);
    for (e, &value) in parent.data.iter().enumerate() {
        let (high, low) = (value >> low_width, value & mask);
        if high != last {
            lim.try_push(&mut s.high_hash, hash)?;
            lim.try_push(&mut s.high_starts, e as u32)?;
            lim.try_push(&mut s.high, high)?;
            (hash, index, last) = (RUN_SEED, index + 1, high);
        }
        let atom = if SINGLE { 0 } else { parent.atom[e] as u64 };
        hash = mix(hash, atom << 32 | low);
        let tally = &mut s.low_tally[low as usize];
        *tally = (mix(tally.0, index << 32 | atom), tally.1 + 1);
    }
    debug_assert!(
        s.high_starts.len() <= most,
        "the values ascend by high part"
    );
    lim.try_push(&mut s.high_hash, hash)
}

/// Number the low values of `split_hashed` by their completion sets, listed
/// low value by low value in rank order, each as ascending pairs of a high
/// value's index and an atom.
///
/// Only the values that share their number in `by_hash` — the numbering by
/// hash alone, `hashed` numbers — are listed. Equal sets hash alike, and
/// values that hash alike probe to one number, so a value alone under its
/// number has a set no other value has: its list is left empty, and it is
/// equal to none. Every value is still read, but only one whose set is
/// listed is written, the others passing by their slot's mark.
fn confirm_lows(
    lim: &Limits,
    s: &mut Scratch,
    parent: &Values,
    low_width: usize,
    by_hash: &[u32],
    hashed: u32,
) -> Result<(Vec<u32>, u32), OperationError> {
    let n = parent.len();
    let mask = (1u64 << low_width) - 1;
    let lows = s.order.len();
    let mut sharing = Vec::new();
    lim.try_resize(&mut sharing, hashed as usize, 0u8)?;
    for &id in by_hash {
        sharing[id as usize] = sharing[id as usize].saturating_add(1);
    }
    s.low_starts.clear();
    lim.reserve_exact(&mut s.low_starts, lows + 1)?;
    let mut at = 0u32;
    for (&slot, &id) in s.order.iter().zip(by_hash) {
        s.low_starts.push(at);
        at += if sharing[id as usize] > 1 {
            s.low_tally[slot as usize].1
        } else {
            0
        };
    }
    s.low_starts.push(at);
    let spare = at;
    // Each slot's count becomes where its next pair goes, so a value finds
    // its place through its own slot, not through its rank.
    for (&slot, w) in s.order.iter().zip(s.low_starts.windows(2)) {
        s.low_tally[slot as usize].1 = if w[0] == w[1] { spare } else { w[0] };
    }
    lim.discard(sharing);
    s.low_keys.clear();
    lim.try_resize(&mut s.low_keys, spare as usize + 1, 0u64)?;
    lim.gate().poll(n as u64)?;
    for (high, range) in runs(&s.high_starts, n).enumerate() {
        for e in range {
            let at = &mut s.low_tally[(parent.data[e] & mask) as usize].1;
            if *at != spare {
                s.low_keys[*at as usize] = (high as u64) << 32 | parent.atom[e] as u64;
                *at += 1;
            }
        }
    }
    let (starts, lists) = (&s.low_starts, &s.low_keys);
    let list = |k: u32| &lists[starts[k as usize] as usize..starts[k as usize + 1] as usize];
    number_hashed(lim, &mut s.slots, &s.high_keys, &mut s.low_first, |a, b| {
        let (x, y) = (list(a), list(b));
        !x.is_empty() && x == y
    })
}

/// The pairs of `split_hashed` with more than one parent atom: those of
/// the first run of each high atom, grouped by parent atom through a
/// counting sort, then sorted and deduplicated group by group. Returns
/// where each parent atom's pairs end, and the pairs.
fn by_parent_atom(
    lim: &Limits,
    s: &mut Scratch,
    parent: &Values,
    low_width: usize,
    high_atom: &[u32],
    held: usize,
) -> Result<(Vec<u32>, Vec<u64>), OperationError> {
    let n = parent.len();
    let mask = (1u64 << low_width) - 1;
    let atoms = parent.atoms as usize;
    let first = |high: usize| s.high_first[high_atom[high] as usize] as usize == high;
    s.cursor.clear();
    lim.try_resize(&mut s.cursor, atoms + 1, 0u32)?;
    for (high, range) in runs(&s.high_starts, n).enumerate() {
        if first(high) {
            for &atom in &parent.atom[range] {
                s.cursor[atom as usize + 1] += 1;
            }
        }
    }
    for atom in 0..atoms {
        s.cursor[atom + 1] += s.cursor[atom];
    }
    s.low_starts.clear();
    lim.reserve_exact(&mut s.low_starts, atoms + 1)?;
    s.low_starts.extend_from_slice(&s.cursor);
    let mut pairs = Vec::new();
    lim.try_resize(&mut pairs, held, 0u64)?;
    for (high, range) in runs(&s.high_starts, n).enumerate() {
        if first(high) {
            let pair = (high_atom[high] as u64) << 32;
            for e in range {
                let at = &mut s.cursor[parent.atom[e] as usize];
                pairs[*at as usize] = pair | s.high_of[(parent.data[e] & mask) as usize] as u64;
                *at += 1;
            }
        }
    }
    let mut ends = Vec::new();
    lim.reserve_exact(&mut ends, atoms)?;
    let mut kept = 0;
    for atom in 0..atoms {
        let (start, end) = (s.low_starts[atom] as usize, s.low_starts[atom + 1] as usize);
        if end - start > 1 {
            pairs[start..end].sort_unstable();
        }
        kept = dedup_range(&mut pairs, start, end, kept);
        ends.push(kept as u32);
    }
    pairs.truncate(kept);
    Ok((ends, pairs))
}

/// Low parts no wider than this many values' worth of bits take a slot
/// apiece, whatever the value count.
const DIRECT_LOWS: usize = 1 << 12;

/// The hash of a run before its first key. Not zero, which `mix` keeps
/// under a zero key: a run that leads with one would hash as the rest of it
/// does, as a high value's run leading with low value 0 would, or a low
/// value's leading with the first high value.
const RUN_SEED: u64 = 0x243f_6a88_85a3_08d3;

/// Fold `key` into a run's hash, the order of the keys counting.
#[inline(always)]
fn mix(hash: u64, key: u64) -> u64 {
    #[cfg(test)]
    if tests::WEAK_HASH.with(std::cell::Cell::get) {
        return hash + (key & 1);
    }
    (hash.rotate_left(5) ^ key).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95)
}

/// Spread a run's hash over every bit, since `number_hashed` addresses its
/// table with the high bits and tags an entry with the low ones.
#[inline(always)]
fn finish(mut hash: u64) -> u64 {
    hash ^= hash >> 33;
    hash = hash.wrapping_mul(0xff51_afd7_ed55_8ccd);
    hash ^= hash >> 33;
    hash = hash.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    hash ^ hash >> 33
}

/// Number runs by content in order of first appearance, given each run's
/// hash. Runs of equal content hash alike, and `same` tells whether two runs
/// of equal hash hold equal content. Returns each run's number and how many
/// there are, and leaves in `firsts` the run where each number first
/// appears.
///
/// The table is open-addressed with linear probing, at most half full. An
/// entry holds a number and the low half of its hash, so a probe reads the
/// runs only when the halves agree, and they agree for another content once
/// in 2^32.
fn number_hashed(
    lim: &Limits,
    slots: &mut Vec<u64>,
    hashes: &[u64],
    firsts: &mut Vec<u32>,
    same: impl Fn(u32, u32) -> bool,
) -> Result<(Vec<u32>, u32), OperationError> {
    let runs = hashes.len();
    let bits = index_bits(2 * runs).max(4);
    let size = 1usize << bits;
    slots.clear();
    lim.try_resize(slots, size, 0u64)?;
    let mut ids = Vec::new();
    lim.reserve_exact(&mut ids, runs)?;
    firsts.clear();
    let mut gate = lim.gate();
    gate.poll(runs as u64)?;
    for (k, &hash) in hashes.iter().enumerate() {
        ids.push(probe(lim, slots, bits, hash, k as u32, firsts, &same)?);
    }
    gate.flush()?;
    Ok((ids, firsts.len() as u32))
}

/// The number of run `k`, of `hash`, in the table `slots` of
/// [`number_hashed`], addressed by `bits` bits: a run of equal content's,
/// or the next number, which `firsts` then records.
#[inline(always)]
fn probe(
    lim: &Limits,
    slots: &mut [u64],
    bits: u32,
    hash: u64,
    k: u32,
    firsts: &mut Vec<u32>,
    same: &impl Fn(u32, u32) -> bool,
) -> Result<u32, OperationError> {
    let hash = finish(hash);
    let tag = hash << 32;
    let mut at = (hash >> (64 - bits)) as usize;
    loop {
        let entry = slots[at];
        if entry == 0 {
            let id = firsts.len() as u32;
            slots[at] = tag | (id as u64 + 1);
            lim.try_push(firsts, k)?;
            return Ok(id);
        }
        if entry >> 32 == tag >> 32 {
            let id = entry as u32 - 1;
            if same(firsts[id as usize], k) {
                return Ok(id);
            }
        }
        at = (at + 1) & (slots.len() - 1);
    }
}

/// The indices `0..count`: the atoms of values that are each their own.
fn identity(lim: &Limits, count: usize) -> Result<Vec<u32>, OperationError> {
    let mut atoms = Vec::new();
    lim.reserve_exact(&mut atoms, count)?;
    atoms.extend(0..count as u32);
    Ok(atoms)
}

/// The positions of run `k` of those starting at `starts`, the last running
/// to `n`.
fn run(starts: &[u32], k: u32, n: usize) -> std::ops::Range<usize> {
    let start = starts[k as usize] as usize;
    start..starts.get(k as usize + 1).map_or(n, |&end| end as usize)
}

/// Every run of those starting at `starts`, the last running to `n`.
fn runs(starts: &[u32], n: usize) -> impl Iterator<Item = std::ops::Range<usize>> + '_ {
    (0..starts.len() as u32).map(move |k| run(starts, k, n))
}

/// The most runs `n` keys sorted on a part make, that part being `first`
/// in the first key and `last` in the last: one a key, and one a part
/// between the two.
fn runs_between(first: u64, last: u64, n: usize) -> usize {
    usize::try_from(last.saturating_sub(first)).map_or(n, |span| n.min(span.saturating_add(1)))
}

/// Runs of one atom list the same keys, so they realize the same triples.
/// Reading the first run of every atom on one side finds them all; whether
/// the low side's first runs hold no more values than the high's.
fn from_low(n: usize, s: &Scratch) -> bool {
    held_keys(&s.low_starts, &s.low_first, n) <= held_keys(&s.high_starts, &s.high_first, n)
}

/// Split values through their runs: by high part in parent order, and by
/// low part in the order a sort gives. One-word values whose sort keys `key`
/// describes find both around one sort of those keys (`runs_of_words`),
/// other values through their materialized parts (`runs_of_parts`).
fn split_parts(
    lim: &Limits,
    s: &mut Scratch,
    parent: &Values,
    (low_width, high_width): (usize, usize),
    distinct: bool,
    key: Option<KeyLayout>,
) -> Result<(Decomposition, Values<'static>, Values<'static>), OperationError> {
    let n = parent.len();
    let (lw, rw) = (words_per_row(low_width), words_per_row(high_width));
    let written = match key {
        Some(key) => runs_of_words(lim, s, parent, low_width, key)?,
        None => {
            runs_of_parts(lim, s, parent, low_width, high_width, distinct)?;
            None
        }
    };
    let (low_count, high_count) = (s.low_starts.len(), s.high_starts.len());

    let (low_atom, low_atoms, high_atom, high_atoms) = if distinct {
        (
            identity(lim, low_count)?,
            low_count as u32,
            identity(lim, high_count)?,
            high_count as u32,
        )
    } else {
        // A high value's run lists its low partners in ascending order, and
        // a low value's run its high partners; each beside the parent atom.
        // `runs_of_words` names a low partner by value, which compares as
        // its index would.
        let low_values = match key {
            Some(_) => usize::try_from(1u64 << low_width).unwrap_or(usize::MAX),
            None => {
                s.high_keys.clear();
                lim.reserve_exact(&mut s.high_keys, n)?;
                s.high_keys.extend(
                    s.low_of[..n]
                        .iter()
                        .zip(&parent.atom)
                        .map(|(&l, &a)| (l as u64) << 32 | a as u64),
                );
                low_count
            }
        };
        let (high_atom, high_atoms) = number_runs(
            lim,
            &mut s.runs,
            &s.high_keys,
            &s.high_starts,
            (low_values, parent.atoms),
            &mut s.high_first,
        )?;
        let (low_atom, low_atoms) = number_runs(
            lim,
            &mut s.runs,
            &s.low_keys,
            &s.low_starts,
            (high_count, parent.atoms),
            &mut s.low_first,
        )?;
        if let Some(key) = key
            && !from_low(n, s)
        {
            words_low_of(lim, s, n, key)?;
        }
        (low_atom, low_atoms, high_atom, high_atoms)
    };
    let triples = match written {
        Some(triples) => triples,
        None => triples(
            lim,
            s,
            parent,
            (&low_atom, low_atoms),
            (&high_atom, high_atoms),
        )?,
    };

    let mut low_data = Vec::new();
    lim.reserve_exact(&mut low_data, low_count * lw)?;
    let mut high_data = Vec::new();
    lim.reserve_exact(&mut high_data, high_count * rw)?;
    if let Some(key) = key {
        let lows = s.low_starts.iter();
        low_data.extend(lows.map(|&i| key.low(s.sort_keys[i as usize])));
        let highs = s.high_starts.iter();
        high_data.extend(highs.map(|&e| parent.data[e as usize] >> low_width));
    } else {
        // Word by word: a copy of a few words through `memcpy` costs more
        // than the copy.
        for &i in &s.low_starts {
            let e = s.order[i as usize] as usize;
            low_data.extend(s.low[e * lw..][..lw].iter().copied());
        }
        for &e in &s.high_starts {
            high_data.extend(s.high[e as usize * rw..][..rw].iter().copied());
        }
    }
    let split = Decomposition::Triples {
        atoms: parent.atoms as usize,
        triples,
    };
    let low = Values::owned(lw, low_data, (low_atom, low_atoms));
    let high = Values::owned(rw, high_data, (high_atom, high_atoms));
    Ok((split, low, high))
}

/// The runs of `runs_of_parts` for one-word values, whose sort keys `key`
/// describes, read off the values and their sorted keys: one pass over the
/// values finds the high runs and writes the keys, the radix sort orders
/// them by low part, and a second pass finds the low runs. Each key carries
/// its value's atom and high value, which the low runs read in low order,
/// so neither pass gathers. When every value is its own atom, the two
/// passes write the triples themselves, which are returned; otherwise they
/// leave in `high_keys` each value's low part above its atom, and neither
/// `low_of` nor `high_of`.
fn runs_of_words(
    lim: &Limits,
    s: &mut Scratch,
    parent: &Values,
    low_width: usize,
    key: KeyLayout,
) -> Result<Option<Vec<[u32; 3]>>, OperationError> {
    let n = parent.len();
    let distinct = parent.atoms as usize == n;
    let low_mask = (1u64 << low_width) - 1;
    let mut gate = lim.gate();
    s.sort_keys.clear();
    lim.reserve_exact(&mut s.sort_keys, n)?;
    s.high_starts.clear();
    s.high_keys.clear();
    let mut triples = Vec::new();
    if distinct {
        lim.reserve_exact(&mut triples, n)?;
    } else {
        lim.reserve_exact(&mut s.high_keys, n)?;
    }
    gate.poll(n as u64)?;
    let mut last = 0u64;
    for (e, (&value, &atom)) in parent.data.iter().zip(&parent.atom).enumerate() {
        let high = value >> low_width;
        if e == 0 || high != last {
            lim.try_push(&mut s.high_starts, e as u32)?;
            last = high;
        }
        let run = s.high_starts.len() - 1;
        let low = value & low_mask;
        if distinct {
            // Every value is its own atom, numbered in value order.
            triples.push([atom, run as u32, 0]);
            s.sort_keys.push(key.key(low, e, 0, 0));
        } else {
            s.high_keys.push(low << 32 | atom as u64);
            s.sort_keys.push(key.key(low, e, atom, run));
        }
    }

    // Keys of one low part differ above it, where they ascend in parent
    // order: a value's high part does, and so does its index. The sort
    // keeps that order, so each low value's run lists its keys canonically.
    s.radix.sort(lim, &mut s.sort_keys, 0, low_width)?;
    lim.check_stop()?;
    s.low_starts.clear();
    s.low_keys.clear();
    if !distinct {
        lim.reserve_exact(&mut s.low_keys, n)?;
    }
    gate.poll(n as u64)?;
    for (i, &sorted) in s.sort_keys.iter().enumerate() {
        let low = key.low(sorted);
        if i == 0 || low != last {
            lim.try_push(&mut s.low_starts, i as u32)?;
            last = low;
        }
        if distinct {
            triples[key.index(sorted)][2] = s.low_starts.len() as u32 - 1;
        } else {
            s.low_keys
                .push((key.run(sorted) as u64) << 32 | key.atom(sorted) as u64);
        }
    }
    gate.flush()?;
    Ok(distinct.then_some(triples))
}

/// Each value's low run in `low_of`, read off the sorted keys of
/// `runs_of_words`.
fn words_low_of(
    lim: &Limits,
    s: &mut Scratch,
    n: usize,
    key: KeyLayout,
) -> Result<(), OperationError> {
    if s.low_of.len() < n {
        lim.try_resize(&mut s.low_of, n, 0u32)?;
    }
    if key.indexed {
        for (l, range) in runs(&s.low_starts, n).enumerate() {
            for &sorted in &s.sort_keys[range] {
                s.low_of[key.index(sorted)] = l as u32;
            }
        }
        return Ok(());
    }
    // A high value's run ascends by low part, so the low order meets its
    // values in parent order, and a cursor per run tells which.
    s.cursor.clear();
    lim.reserve_exact(&mut s.cursor, s.high_starts.len())?;
    s.cursor.extend_from_slice(&s.high_starts);
    for (l, range) in runs(&s.low_starts, n).enumerate() {
        for &sorted in &s.sort_keys[range] {
            let at = &mut s.cursor[key.run(sorted)];
            s.low_of[*at as usize] = l as u32;
            *at += 1;
        }
    }
    Ok(())
}

/// Find the runs of the parent values by their high part, in parent order,
/// and by their low part, in the order `s.order` lists them in: each value's
/// run index in `high_of` and `low_of`, where each run starts in
/// `high_starts` and `low_starts`, and unless every value is its own atom,
/// in `low_keys` each value's high value index above its atom, in low
/// order.
fn runs_of_parts(
    lim: &Limits,
    s: &mut Scratch,
    parent: &Values,
    low_width: usize,
    high_width: usize,
    distinct: bool,
) -> Result<(), OperationError> {
    let n = parent.len();
    let (lw, rw) = (words_per_row(low_width), words_per_row(high_width));
    parts(lim, s, parent, low_width, high_width)?;

    // The parent's order sorts by the high part first, so each high value
    // is one run of parent values.
    let mut gate = lim.gate();
    s.high_of.clear();
    lim.reserve_exact(&mut s.high_of, n)?;
    s.high_starts.clear();
    gate.poll(n as u64)?;
    for e in 0..n {
        let new_run = match rw {
            1 => e == 0 || s.high[e] != s.high[e - 1],
            _ => e == 0 || !same_row(&s.high[e * rw..][..rw], &s.high[(e - 1) * rw..][..rw]),
        };
        if new_run {
            lim.try_push(&mut s.high_starts, e as u32)?;
        }
        s.high_of.push(s.high_starts.len() as u32 - 1);
    }

    // The low parts need sorting. A stable order keeps each low value's run
    // in ascending high part, so the run's keys are listed canonically.
    let packed = order_by_low(lim, s, n, low_width, lw)?;
    if s.low_of.len() < n {
        lim.try_resize(&mut s.low_of, n, 0u32)?;
    }
    s.low_starts.clear();
    gate.poll(n as u64)?;
    if let Some(shift) = packed {
        // The sorted keys carry the low values themselves, in order.
        let mut last = 0u64;
        for (i, &key) in s.sort_keys.iter().enumerate() {
            let value = key >> shift;
            if i == 0 || value != last {
                lim.try_push(&mut s.low_starts, i as u32)?;
                last = value;
            }
            s.low_of[s.order[i] as usize] = s.low_starts.len() as u32 - 1;
        }
    } else {
        for i in 0..n {
            let e = s.order[i] as usize;
            let previous = if i == 0 { 0 } else { s.order[i - 1] as usize };
            let new_run = match lw {
                1 => i == 0 || s.low[e] != s.low[previous],
                _ => i == 0 || !same_row(&s.low[e * lw..][..lw], &s.low[previous * lw..][..lw]),
            };
            if new_run {
                lim.try_push(&mut s.low_starts, i as u32)?;
            }
            s.low_of[e] = s.low_starts.len() as u32 - 1;
        }
    }
    s.low_keys.clear();
    if !distinct {
        lim.reserve_exact(&mut s.low_keys, n)?;
        let (high_of, atom) = (&s.high_of, &parent.atom);
        s.low_keys.extend(
            s.order
                .iter()
                .map(|&e| (high_of[e as usize] as u64) << 32 | atom[e as usize] as u64),
        );
    }
    gate.flush()
}

/// Cut every parent value into its low part, the low `low_width` bits, and
/// its high part, the `high_width` bits above them.
fn parts(
    lim: &Limits,
    s: &mut Scratch,
    parent: &Values,
    low_width: usize,
    high_width: usize,
) -> Result<(), OperationError> {
    let n = parent.len();
    let (lw, rw) = (words_per_row(low_width), words_per_row(high_width));
    s.low.clear();
    s.high.clear();
    let mut gate = lim.gate();
    if parent.words == 1 {
        // Both parts are narrower than the parent's one word.
        lim.reserve_exact(&mut s.low, n * lw)?;
        lim.reserve_exact(&mut s.high, n * rw)?;
        let mask = (1u64 << low_width) - 1;
        gate.poll(n as u64)?;
        s.low.extend(parent.data.iter().map(|&v| v & mask));
        s.high.extend(parent.data.iter().map(|&v| v >> low_width));
    } else if parent.words == 2 && lw == 1 && rw == 1 {
        // Both parts fit a word: cut each value as one 128-bit number.
        lim.reserve_exact(&mut s.low, n)?;
        lim.reserve_exact(&mut s.high, n)?;
        let mask = tail_mask(low_width);
        gate.poll(2 * n as u64)?;
        let value = |v: &[u64]| u128::from(v[1]) << 64 | u128::from(v[0]);
        s.low
            .extend(parent.data.chunks_exact(2).map(|v| value(v) as u64 & mask));
        s.high
            .extend(parent.data.chunks_exact(2).map(|v| (value(v) >> low_width) as u64));
    } else {
        lim.try_resize(&mut s.low, n * lw, 0)?;
        lim.try_resize(&mut s.high, n * rw, 0)?;
        gate.poll((n * parent.words) as u64)?;
        let (skip, shift) = (low_width / 64, low_width % 64);
        let (low_tail, high_tail) = (tail_mask(low_width), tail_mask(high_width));
        let values = parent.data.chunks_exact(parent.words);
        let parts = s.low.chunks_exact_mut(lw).zip(s.high.chunks_exact_mut(rw));
        for (value, (low, high)) in values.zip(parts) {
            low.copy_from_slice(&value[..lw]);
            low[lw - 1] &= low_tail;
            for (k, out) in high.iter_mut().enumerate() {
                let word = |at: usize| u128::from(value.get(at).copied().unwrap_or(0));
                *out = ((word(skip + k + 1) << 64 | word(skip + k)) >> shift) as u64;
            }
            high[rw - 1] &= high_tail;
        }
    }
    gate.flush()
}

/// The mask of the bits a part `width` bits wide uses in its last word.
fn tail_mask(width: usize) -> u64 {
    match width % 64 {
        0 if width > 0 => !0,
        tail => (1u64 << tail) - 1,
    }
}

/// Order the parent values by their low part, ties in parent order, into
/// `s.order`. Returns the shift that reads a low value off `s.sort_keys`
/// when those are the sorted keys, one per position of `s.order`.
fn order_by_low(
    lim: &Limits,
    s: &mut Scratch,
    n: usize,
    low_width: usize,
    lw: usize,
) -> Result<Option<usize>, OperationError> {
    s.order.clear();
    lim.reserve_exact(&mut s.order, n)?;
    let shift = index_bits(n) as usize;
    if lw == 1 {
        // The index below the low part makes every key distinct, and in
        // index order before sorting, so any sort of the keys is stable.
        // Bits of the low part that do not fit above the index, its lowest,
        // order each run the other bits leave tied.
        let top = (low_width + shift).saturating_sub(64);
        let mask = (1u64 << shift) - 1;
        s.sort_keys.clear();
        lim.reserve_exact(&mut s.sort_keys, n)?;
        s.sort_keys.extend(
            s.low
                .iter()
                .enumerate()
                .map(|(e, &v)| v >> top << shift | e as u64),
        );
        s.radix.sort(lim, &mut s.sort_keys, shift, low_width - top)?;
        if top > 0 {
            let low = &s.low;
            for run in s.sort_keys.chunk_by_mut(|a, b| a >> shift == b >> shift) {
                if run.len() > 1 {
                    for key in run.iter_mut() {
                        *key = low[(*key & mask) as usize] << (64 - top) >> (64 - top - shift) | *key & mask;
                    }
                    run.sort_unstable();
                }
            }
        }
        s.order
            .extend(s.sort_keys.iter().map(|&key| (key & mask) as u32));
        lim.check_stop()?;
        return Ok((top == 0).then_some(shift));
    }
    s.order.extend(0..n as u32);
    let low = &s.low;
    s.order.sort_unstable_by(|&a, &b| {
        let (x, y) = (&low[a as usize * lw..][..lw], &low[b as usize * lw..][..lw]);
        x.iter().rev().cmp(y.iter().rev()).then(a.cmp(&b))
    });
    lim.check_stop()?;
    Ok(None)
}

/// Number the runs of `keys` that start at `starts`, the last running to the
/// end, by content, in order of first appearance. Every key is an index below
/// `bounds.0` above an index below `bounds.1`. Returns each run's number and
/// how many distinct runs there are, and leaves in `firsts` the run where each
/// number first appears.
///
/// Lone keys whose range is small enough are numbered through a slot per
/// key. Otherwise each run is hashed and the hashes are numbered by
/// [`number_hashed`], confirming a match against the runs themselves.
fn number_runs(
    lim: &Limits,
    table: &mut RunTable,
    keys: &[u64],
    starts: &[u32],
    bounds: (usize, u32),
    firsts: &mut Vec<u32>,
) -> Result<(Vec<u32>, u32), OperationError> {
    let n = keys.len();
    let mut gate = lim.gate();
    gate.poll(n as u64)?;
    if starts.len() == n {
        // Every run is one key, which is its own content.
        let span = bounds.0.saturating_mul(bounds.1 as usize);
        if span <= (4 * n).max(DIRECT_MIN_SPAN) {
            let mut ids = Vec::new();
            lim.reserve_exact(&mut ids, n)?;
            firsts.clear();
            table.direct.clear();
            lim.try_resize(&mut table.direct, span, u32::MAX)?;
            for (k, &key) in keys.iter().enumerate() {
                let slot = &mut table.direct
                    [(key >> 32) as usize * bounds.1 as usize + (key as u32) as usize];
                if *slot == u32::MAX {
                    *slot = firsts.len() as u32;
                    lim.try_push(firsts, k as u32)?;
                }
                ids.push(*slot);
            }
            gate.flush()?;
            return Ok((ids, firsts.len() as u32));
        }
        gate.flush()?;
        return number_hashed(lim, &mut table.slots, keys, firsts, |a, b| {
            keys[a as usize] == keys[b as usize]
        });
    }
    table.hashes.clear();
    lim.reserve_exact(&mut table.hashes, starts.len())?;
    table.hashes.extend(runs(starts, n).map(|range| {
        keys[range]
            .iter()
            .fold(RUN_SEED, |hash, &key| mix(hash, key))
    }));
    gate.flush()?;
    let same = |a: u32, b: u32| keys[run(starts, a, n)] == keys[run(starts, b, n)];
    number_hashed(lim, &mut table.slots, &table.hashes, firsts, same)
}

/// Address ranges this small number lone keys directly whatever the key
/// count, which costs less than hashing them.
const DIRECT_MIN_SPAN: usize = 1 << 12;

/// The distinct `[atom, high atom, low atom]` triples of the parent values,
/// in ascending order. Unless every value is its own atom, reads the low
/// runs' values from `s.low_keys`, each value's high run above its atom in
/// low order, or else each value's low run from `s.low_of`.
fn triples(
    lim: &Limits,
    s: &mut Scratch,
    parent: &Values,
    (low_atom, low_atoms): (&[u32], u32),
    (high_atom, high_atoms): (&[u32], u32),
) -> Result<Vec<[u32; 3]>, OperationError> {
    let n = parent.len();
    if parent.atoms as usize == n {
        // Every value is its own atom, numbered in value order: one triple
        // apiece, already in order.
        let mut out = Vec::new();
        lim.reserve_exact(&mut out, n)?;
        out.extend((0..n).map(|e| {
            [
                parent.atom[e],
                high_atom[s.high_of[e] as usize],
                low_atom[s.low_of[e] as usize],
            ]
        }));
        return Ok(out);
    }
    let from_low = from_low(n, s);
    let (low_keys, low_of) = (&s.low_keys, &s.low_of);
    let (low_starts, high_starts) = (&s.low_starts, &s.high_starts);
    let (low_first, high_first) = (&s.low_first, &s.high_first);
    let each = |emit: &mut dyn FnMut([u32; 3])| {
        if from_low {
            for (l, &k) in low_first.iter().enumerate() {
                for &key in &low_keys[run(low_starts, k, n)] {
                    emit([key as u32, high_atom[(key >> 32) as usize], l as u32]);
                }
            }
        } else {
            for (r, &k) in high_first.iter().enumerate() {
                for e in run(high_starts, k, n) {
                    emit([parent.atom[e], r as u32, low_atom[low_of[e] as usize]]);
                }
            }
        }
    };
    let atoms = (parent.atoms, high_atoms, low_atoms);
    sorted_triples(
        lim,
        &mut s.radix,
        &mut s.high_keys,
        &mut s.wide_triples,
        atoms,
        if from_low { Emitted::ByLast } else { Emitted::ByMid },
        each,
    )
}

#[cfg(test)]
#[path = "tests/split.rs"]
mod tests;
