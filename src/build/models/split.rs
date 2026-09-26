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

use std::hash::Hasher;

use rustc_hash::{FxHashMap, FxHasher};

use crate::diagram::{NodeIdx, NEG_LEAF_IDX, ONE_LEAF_IDX, POS_LEAF_IDX};
use crate::limits::{Charged, Limits, OperationError};
use crate::vtree::{Vtree, VtreeIdx};

use super::layout::Layout;
use super::rows::Radix;

/// The distinct values one constrained vtree node takes over the rows, and
/// the atom each belongs to.
struct Values {
    /// Words per value, low word first.
    words: usize,
    /// The values in ascending order, `words` apiece. Bits past the node's
    /// width are clear.
    data: Vec<u64>,
    /// The atom of each value. Atoms are numbered by their smallest value.
    atom: Vec<u32>,
    /// How many atoms there are.
    atoms: u32,
}

impl Values {
    fn len(&self) -> usize {
        self.atom.len()
    }
}

impl Charged for Values {
    fn charged_bytes(&self) -> u64 {
        self.data.charged_bytes() + self.atom.charged_bytes()
    }
}

/// What the build needs at one vtree node that is split.
pub(super) enum Plan {
    /// A constrained leaf's atoms, as the leaf nodes that stand for them.
    Leaf(Vec<NodeIdx>),
    /// A node with constrained variables under both children.
    Branch(Decomposition),
}

/// The atoms of a node constrained on both sides, as child atoms.
pub(super) enum Decomposition {
    /// Atoms listed through their triples.
    Triples {
        /// How many atoms the node has.
        atoms: usize,
        /// Every realized `[atom, high atom, low atom]`, once each, in
        /// ascending order: the child atoms in the vtree's order, left then
        /// right. The two child atoms determine the parent's, so the
        /// triples of one atom are the pairs of its node.
        triples: Vec<[u32; 3]>,
    },
    /// One atom, whose pairs are listed by high atom: high atom `a` pairs
    /// with the low atoms `lows[ends[a - 1]..ends[a]]`, ascending, taking
    /// `ends[-1]` as zero. The node over every constrained variable has one
    /// atom and as many pairs as there are rows, and this list takes a
    /// third of the triples' space.
    Grouped {
        /// Where each high atom's low atoms end.
        ends: Vec<u32>,
        /// The low atoms of every pair.
        lows: Vec<u32>,
    },
}

impl Decomposition {
    /// How many atoms the node has.
    pub(super) fn atoms(&self) -> usize {
        match self {
            Decomposition::Triples { atoms, .. } => *atoms,
            Decomposition::Grouped { .. } => 1,
        }
    }
}

/// The buffers one split reuses from the last.
#[derive(Default)]
struct Scratch {
    /// Each parent value's low part, `low_words` apiece.
    low: Vec<u64>,
    /// Each parent value's high part, `high_words` apiece.
    high: Vec<u64>,
    /// The parent values ordered by their low part.
    order: Vec<u32>,
    /// Sort keys: a low part beside the value's index, below it on the
    /// one-word path (see `KeyLayout`) and above it otherwise. The keys
    /// `group_single` places by block.
    sort_keys: Vec<u64>,
    /// Sort keys too wide for one word.
    wide_keys: Vec<u128>,
    /// The low child's value index of each parent value.
    low_of: Vec<u32>,
    /// The high child's value index of each parent value.
    high_of: Vec<u32>,
    /// Where each low value's run starts in `order`.
    low_starts: Vec<u32>,
    /// Where each high value's run starts among the parent values.
    high_starts: Vec<u32>,
    /// Per high value, the next of its run's values the low order meets,
    /// or of its slots among the node's pairs.
    cursor: Vec<u32>,
    /// Per parent value, its low value index above its atom: the keys of
    /// the high child's runs. Once they are numbered, the packed triples.
    high_keys: Vec<u64>,
    /// Per parent value in low order, its high value index above its
    /// atom: the keys of the low child's runs.
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
    /// Each low value's run hashed, by the low part where it addresses a
    /// slot.
    low_hash: Vec<u64>,
    /// The open-addressed table that numbers hashed runs.
    slots: Vec<u64>,
    /// How many values each low slot has. Each block's next free place in
    /// `group_single`.
    low_count: Vec<u32>,
    /// Each low slot's rank among the low values. Per high value, the atom
    /// of the low value whose one key it is, in `group_single`.
    rank: Vec<u32>,
    /// The radix sort's buffers.
    radix: Radix,
}

impl Scratch {
    /// Drop the buffers and hand their charge back.
    fn discard(self, lim: &Limits) {
        let Scratch { low, high, order, sort_keys, wide_keys, low_of, high_of, low_starts, high_starts,
            cursor, high_keys, low_keys, runs, low_first, high_first, wide_triples, high_hash, low_hash,
            slots, low_count, rank, radix } = self;
        for buf in [low, high, sort_keys, high_keys, low_keys, high_hash, low_hash, slots] {
            lim.discard(buf);
        }
        for buf in [order, low_of, high_of, low_starts, high_starts, cursor, low_first, high_first, low_count, rank] {
            lim.discard(buf);
        }
        lim.discard(wide_keys);
        lim.discard(wide_triples);
        let RunTable { head, next, first, direct } = runs;
        lim.discard(head);
        lim.discard(next);
        lim.discard(first);
        lim.discard(direct);
        radix.discard(lim);
    }
}

/// Numbers runs of keys by their content.
#[derive(Default)]
struct RunTable {
    /// The first number under each hash of a run, or under a lone key itself.
    head: FxHashMap<u64, u32>,
    /// The next number sharing a number's hash, or `u32::MAX`.
    next: Vec<u32>,
    /// Each number's first run, as `(start, length)` in the keys.
    first: Vec<(u32, u32)>,
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
    sorted: Vec<u64>,
    w: usize,
    radix: Radix,
) -> Result<Vec<Option<Plan>>, OperationError> {
    let mut plans: Vec<Option<Plan>> = Vec::new();
    lim.reserve_exact(&mut plans, vtree.num_nodes())?;
    plans.resize_with(vtree.num_nodes(), || None);
    let mut atom = Vec::new();
    lim.try_resize(&mut atom, sorted.len() / w, 0u32)?;
    // Every constrained variable lies under the lowest node whose subtree
    // holds them all, so every row there has the same empty completion.
    let whole = Values { words: w, data: sorted, atom, atoms: 1 };
    let mut pending = Vec::new();
    lim.try_push(&mut pending, (split_node(vtree, layout, vtree.root()), whole))?;
    let mut scratch = Scratch { radix, ..Scratch::default() };
    let mut planned = 0u64;
    while let Some((t, mut values)) = pending.pop() {
        lim.check_stop()?;
        if vtree.node(t).is_leaf() {
            plans[t.idx()] = Some(Plan::Leaf(leaf_nodes(lim, &values)?));
            lim.discard(values);
            continue;
        }
        let (high, low) = vtree.children(t);
        let widths = (layout.count[low.idx()] as usize, layout.count[high.idx()] as usize);
        let (split, low_values, high_values) = split(lim, &mut scratch, &mut values, widths)?;
        // Each atom is a node of the result, so the output cap can refuse
        // before the build begins.
        planned += split.atoms() as u64;
        lim.check_output_cap(planned)?;
        lim.discard(values);
        plans[t.idx()] = Some(Plan::Branch(split));
        lim.try_push(&mut pending, (split_node(vtree, layout, low), low_values))?;
        lim.try_push(&mut pending, (split_node(vtree, layout, high), high_values))?;
    }
    scratch.discard(lim);
    Ok(plans)
}

/// The node at or below the constrained node `t` whose atoms are `t`'s: `t`
/// itself if it is a leaf or constrained on both sides, else the same for its
/// constrained child.
fn split_node(vtree: &Vtree, layout: &Layout, mut t: VtreeIdx) -> VtreeIdx {
    while !vtree.node(t).is_leaf() {
        let (left, right) = vtree.children(t);
        match (layout.count[left.idx()], layout.count[right.idx()]) {
            (0, _) => t = right,
            (_, 0) => t = left,
            _ => break,
        }
    }
    t
}

/// A leaf has one literal atom, two literal atoms, or one free atom.
fn leaf_nodes(lim: &Limits, values: &Values) -> Result<Vec<NodeIdx>, OperationError> {
    let mut locals = Vec::new();
    lim.reserve_exact(&mut locals, 2)?;
    match (values.len(), values.atoms) {
        (2, 1) => locals.push(ONE_LEAF_IDX),
        (2, _) => locals.extend([NEG_LEAF_IDX, POS_LEAF_IDX]),
        _ => locals.push(if values.data[0] == 0 { NEG_LEAF_IDX } else { POS_LEAF_IDX }),
    }
    Ok(locals)
}

/// Words a value of `width` bits occupies.
fn words_for(width: usize) -> usize {
    width.div_ceil(64).max(1)
}

/// Bits needed to write every index below `count`.
fn index_bits(count: usize) -> u32 {
    usize::BITS - count.saturating_sub(1).leading_zeros()
}

/// Split a node's values into its children's, whose widths are `(low,
/// high)` bits: the low part of a value is its right child's value. The
/// parent's values may be overwritten.
fn split(
    lim: &Limits,
    s: &mut Scratch,
    parent: &mut Values,
    widths: (usize, usize),
) -> Result<(Decomposition, Values, Values), OperationError> {
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
    // Otherwise a high run names its low partners by value, above their
    // atoms, which takes a low part of at most 32 bits.
    if parent.words == 1 && (distinct || widths.0 <= 32) {
        // The high values number at most the parent values and the high
        // part's range; only when that bound leaves the index out are they
        // counted.
        let bound = n.min(1usize << widths.1.min(usize::BITS as usize - 1));
        let mut key = KeyLayout::new(n, widths.0, bound, parent.atoms, distinct);
        if !key.is_some_and(|key| key.indexed) {
            let high_runs = count_high_runs(lim, parent, widths.0)?;
            key = KeyLayout::new(n, widths.0, high_runs, parent.atoms, distinct);
        }
        if let Some(key) = key {
            return split_words(lim, s, parent, widths.0, key);
        }
    }
    split_parts(lim, s, parent, widths, distinct)
}

/// How many distinct high parts one-word values have, the low part being
/// their low `low_width` bits.
fn count_high_runs(lim: &Limits, parent: &Values, low_width: usize) -> Result<usize, OperationError> {
    lim.gate().poll(parent.len() as u64)?;
    let runs = parent.data.windows(2).filter(|pair| pair[0] >> low_width != pair[1] >> low_width).count();
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
    fn new(n: usize, low_width: usize, high_runs: usize, atoms: u32, distinct: bool) -> Option<KeyLayout> {
        let (atom_bits, run_bits) = match distinct {
            true => (0, 0),
            false => (index_bits(atoms as usize) as usize, index_bits(high_runs) as usize),
        };
        let indexed = KeyLayout { low_width, indexed: true, index_bits: index_bits(n) as usize, atom_bits, run_bits };
        let bare = KeyLayout { indexed: false, index_bits: 0, ..indexed };
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

/// Split one-word values whose sort keys `key` describes.
///
/// One pass over the parent values finds the high runs and writes the sort
/// keys, the radix sort orders them by low part, and a second pass finds the
/// low runs. Each key carries its value's atom and high value, which the
/// low runs and the triples read in low order, so neither pass gathers.
/// When every value is its own atom, the two passes write the triples
/// themselves, and no run needs numbering.
fn split_words(
    lim: &Limits,
    s: &mut Scratch,
    parent: &Values,
    low_width: usize,
    key: KeyLayout,
) -> Result<(Decomposition, Values, Values), OperationError> {
    let n = parent.len();
    let distinct = parent.atoms as usize == n;
    let low_mask = (1u64 << low_width) - 1;
    let mut gate = lim.gate();

    // The parent's order sorts by the high part first, so each high value
    // is one run of parent values.
    let mut triples = Vec::new();
    s.sort_keys.clear();
    lim.reserve_exact(&mut s.sort_keys, n)?;
    s.high_starts.clear();
    s.high_keys.clear();
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
            // A high run names its low partners by value, which compares
            // as their indices would.
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
            s.low_keys.push((key.run(sorted) as u64) << 32 | key.atom(sorted) as u64);
        }
    }
    gate.flush()?;

    let mut low_data = Vec::new();
    lim.reserve_exact(&mut low_data, s.low_starts.len())?;
    low_data.extend(s.low_starts.iter().map(|&i| key.low(s.sort_keys[i as usize])));
    let mut high_data = Vec::new();
    lim.reserve_exact(&mut high_data, s.high_starts.len())?;
    high_data.extend(s.high_starts.iter().map(|&e| parent.data[e as usize] >> low_width));
    let (low_count, high_count) = (s.low_starts.len(), s.high_starts.len());

    let (low_atom, low_atoms, high_atom, high_atoms) = if distinct {
        (identity(lim, low_count)?, low_count as u32, identity(lim, high_count)?, high_count as u32)
    } else {
        let values = usize::try_from(1u64 << low_width).unwrap_or(usize::MAX);
        let (high_atom, high_atoms) =
            number_runs(lim, &mut s.runs, &s.high_keys, &s.high_starts, (values, parent.atoms), &mut s.high_first)?;
        let (low_atom, low_atoms) =
            number_runs(lim, &mut s.runs, &s.low_keys, &s.low_starts, (high_count, parent.atoms), &mut s.low_first)?;
        let from_low = from_low(n, s);
        if !from_low {
            // Emitting by high value names each value's low value.
            if s.low_of.len() < n {
                lim.try_resize(&mut s.low_of, n, 0u32)?;
            }
            if key.indexed {
                for (l, range) in runs(&s.low_starts, n).enumerate() {
                    for &sorted in &s.sort_keys[range] {
                        s.low_of[key.index(sorted)] = l as u32;
                    }
                }
            } else {
                // A high value's run ascends by low part, so the low order
                // meets its values in parent order, and a cursor per run
                // tells which.
                s.cursor.clear();
                lim.reserve_exact(&mut s.cursor, high_count)?;
                s.cursor.extend_from_slice(&s.high_starts);
                for (l, range) in runs(&s.low_starts, n).enumerate() {
                    for &sorted in &s.sort_keys[range] {
                        let at = &mut s.cursor[key.run(sorted)];
                        s.low_of[*at as usize] = l as u32;
                        *at += 1;
                    }
                }
            }
        }
        let (sort_keys, low_of) = (&s.sort_keys, &s.low_of);
        let (low_starts, high_starts) = (&s.low_starts, &s.high_starts);
        let (low_first, high_first) = (&s.low_first, &s.high_first);
        let each = |emit: &mut dyn FnMut([u32; 3])| {
            if from_low {
                for (l, &k) in low_first.iter().enumerate() {
                    for &sorted in &sort_keys[run(low_starts, k, n)] {
                        emit([key.atom(sorted), high_atom[key.run(sorted)], l as u32]);
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
        triples = sorted_triples(lim, &mut s.radix, &mut s.high_keys, &mut s.wide_triples, atoms, !from_low, each)?;
        (low_atom, low_atoms, high_atom, high_atoms)
    };
    let split = Decomposition::Triples { atoms: parent.atoms as usize, triples };
    let low = Values { words: 1, data: low_data, atom: low_atom, atoms: low_atoms };
    let high = Values { words: 1, data: high_data, atom: high_atom, atoms: high_atoms };
    Ok((split, low, high))
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
) -> Result<Option<(Decomposition, Values, Values)>, OperationError> {
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
    let data = &mut parent.data;
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
    // The radix sort's buffer holds nothing until the sort below, and the
    // rows' sort has mapped it wherever the rows were many enough to need
    // it, so the table that numbers the high runs borrows it rather than
    // mapping fresh memory.
    std::mem::swap(&mut s.slots, s.radix.spare());
    let (high_starts, keys) = (&s.high_starts, &parent.data);
    let same_lows = |a: u32, b: u32| {
        let (x, y) = (run(high_starts, a, n), run(high_starts, b, n));
        x.len() == y.len() && keys[x].iter().zip(&keys[y]).all(|(&p, &q)| (p ^ q) & mask == 0)
    };
    let numbered = number_hashed(lim, &mut s.slots, &s.high_hash, &mut s.high_first, same_lows);
    std::mem::swap(&mut s.slots, s.radix.spare());
    let (high_atom, high_atoms) = numbered?;

    // Keys of one low part ascend by high value, as the values did.
    s.radix.sort(lim, &mut parent.data, 0, low_width)?;
    lim.check_stop()?;
    let (split, low_atom, low_atoms) = group_single(lim, s, &parent.data, low_width, &high_atom)?;
    // The runs' parts are the children's values, so their buffers are
    // handed over rather than copied.
    let low = Values { words: 1, data: std::mem::take(&mut s.low), atom: low_atom, atoms: low_atoms };
    let high = Values { words: 1, data: std::mem::take(&mut s.high), atom: high_atom, atoms: high_atoms };
    Ok(Some((split, low, high)))
}

/// The low side of `split_single` and the node's pairs, from the keys
/// sorted by low part. Each low value's run, part and hash go into `s`.
/// Returns the pairs, each low value's atom, and how many low atoms there
/// are.
///
/// Runs of one high atom hold the same low parts, so the first run of each
/// lists the atom's pairs, but the keys come by low value, and each key's
/// place among its high value's is a jump. Taken one key at a time over
/// every high value, those jumps miss the cache. The keys are laid out by
/// blocks of high values instead, each block where its values lie in the
/// parent's order, which writes to few places at once, and each block then
/// places its keys among its own values' pairs, which are close by.
fn group_single(
    lim: &Limits,
    s: &mut Scratch,
    keys: &[u64],
    low_width: usize,
    high_atom: &[u32],
) -> Result<(Decomposition, Vec<u32>, u32), OperationError> {
    let n = keys.len();
    let high_count = s.high_starts.len();
    let bits = ((index_bits(high_count) as usize).saturating_sub(BLOCK_BITS), index_bits(n) as usize);
    // The radix sort is done with its buffer, which is mapped wherever the
    // sort needed it, so the keys are placed there.
    std::mem::swap(&mut s.sort_keys, s.radix.spare());
    let lone = place_by_block(lim, s, keys, low_width, bits)?;
    let low_count = s.low_starts.len();
    let low_starts = &s.low_starts;
    let same_highs = |a: u32, b: u32| {
        let (x, y) = (run(low_starts, a, n), run(low_starts, b, n));
        x.len() == y.len() && keys[x].iter().zip(&keys[y]).all(|(&p, &q)| p >> low_width == q >> low_width)
    };
    // A low value of one key is completed by that key's high value alone,
    // so where such values are at least half of them, as on a wide low
    // part, they are numbered through a slot per high value, unhashed.
    let (low_atom, low_atoms) = if 2 * lone >= low_count {
        let high = |k: u32| (keys[k as usize] >> low_width) as usize;
        let runs = (low_starts.as_slice(), n, low_count - lone);
        number_lone(lim, &mut s.slots, &s.low_hash, &mut s.low_first, runs, (&mut s.rank, high_count, high), same_highs)?
    } else {
        number_hashed(lim, &mut s.slots, &s.low_hash, &mut s.low_first, same_highs)?
    };

    // Where each high value's low values go: the first run of each high
    // atom lists the atom's, in atom order, and the other runs none.
    let merged = low_atoms as usize != low_count;
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
    let lookup = |low: u32| if merged { low_atom[low as usize] } else { low };
    gather_by_block(&s.sort_keys[..n], &s.high_starts, &mut s.cursor, bits, &mut lows, lookup);
    std::mem::swap(&mut s.sort_keys, s.radix.spare());
    // Each group's low values ascend as the low order meets them, and so do
    // their atoms unless low values merge.
    let sizes = s.high_first.iter().map(|&k| run(&s.high_starts, k, n).len());
    let ends = close_groups(lim, &mut lows, sizes, merged)?;
    gate.flush()?;
    Ok((Decomposition::Grouped { ends, lows }, low_atom, low_atoms))
}

/// The pass of `group_single` over the keys sorted by low part: each low
/// value's run, its start, part and hash, into `s`, and each key placed in
/// its block of `s.sort_keys` as its high value's place in the block,
/// `bits.0` bits of it, above its low value's index, `bits.1` bits. Block
/// `b` holds high values `b << bits.0` on and lies where their runs do, so
/// it has room for exactly their keys. Returns how many low values have
/// one key.
fn place_by_block(
    lim: &Limits,
    s: &mut Scratch,
    keys: &[u64],
    low_width: usize,
    (shift, low_bits): (usize, usize),
) -> Result<usize, OperationError> {
    let n = keys.len();
    let mask = (1u64 << low_width) - 1;
    let blocks = ((s.high_starts.len() - 1) >> shift) + 1;
    s.low_count.clear();
    lim.reserve_exact(&mut s.low_count, blocks)?;
    s.low_count.extend((0..blocks).map(|block| s.high_starts[block << shift]));
    let (next, in_block) = (&mut s.low_count, (1u64 << shift) - 1);
    let placed = &mut s.sort_keys;
    lim.try_resize(placed, n, 0u64)?;
    lim.gate().poll(n as u64)?;
    s.low_starts.clear();
    s.low_hash.clear();
    s.low.clear();
    let mut last = keys[0] & mask;
    lim.try_push(&mut s.low_starts, 0)?;
    lim.try_push(&mut s.low, last)?;
    let (mut hash, mut index, mut start, mut lone) = (RUN_SEED, 0u64, 0usize, 0usize);
    for (i, &key) in keys.iter().enumerate() {
        let low = key & mask;
        if low != last {
            lim.try_push(&mut s.low_hash, hash)?;
            lim.try_push(&mut s.low_starts, i as u32)?;
            lim.try_push(&mut s.low, low)?;
            lone += usize::from(i - start == 1);
            (hash, index, last, start) = (RUN_SEED, index + 1, low, i);
        }
        let high = key >> low_width;
        hash = mix(hash, high);
        let at = &mut next[(high >> shift) as usize];
        placed[*at as usize] = (high & in_block) << low_bits | index;
        *at += 1;
    }
    lim.try_push(&mut s.low_hash, hash)?;
    Ok(lone + usize::from(n - start == 1))
}

/// Write each key of `placed`, laid out by `place_by_block`, to its high
/// value's next slot of `lows`, as `lookup` of its low value's index. A
/// high value's slots start at its `cursor`, and `u32::MAX` there leaves
/// its keys out.
fn gather_by_block(
    placed: &[u64],
    high_starts: &[u32],
    cursor: &mut [u32],
    (shift, low_bits): (usize, usize),
    lows: &mut [u32],
    lookup: impl Fn(u32) -> u32,
) {
    let low_mask = (1u64 << low_bits) - 1;
    for first in (0..high_starts.len()).step_by(1 << shift) {
        let end = high_starts.get(first + (1 << shift)).map_or(placed.len(), |&end| end as usize);
        let cursor = &mut cursor[first..];
        for &pair in &placed[high_starts[first] as usize..end] {
            let at = &mut cursor[(pair >> low_bits) as usize];
            if *at != u32::MAX {
                lows[*at as usize] = lookup((pair & low_mask) as u32);
                *at += 1;
            }
        }
    }
}

/// High values one block of `group_single` spans at most, as a power of
/// two: few enough blocks to write to at once, and a block's pairs few
/// enough to stay in the cache.
const BLOCK_BITS: usize = 9;

/// Where each group of `lows` ends, the groups listed one after another,
/// `sizes` long. When low values merged, a group can list an atom twice
/// and out of order, so each is sorted and its repeats dropped, which moves
/// the later groups down.
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
            lows[start..end].sort_unstable();
            for at in start..end {
                if at == start || lows[at] != lows[kept - 1] {
                    lows[kept] = lows[at];
                    kept += 1;
                }
            }
        } else {
            kept = end;
        }
        ends.push(kept as u32);
        start = end;
    }
    lows.truncate(kept);
    Ok(ends)
}

/// [`number_hashed`] for runs of which many hold one key. The runs start
/// at `runs.0`, the last running to `runs.1`, and `runs.2` of them hold
/// more than one. A run of one key is numbered through `lone.0`, a slot
/// for each of `lone.1` keys, addressed by `lone.2` of the key's position:
/// the key is the run's content. Only the longer runs are hashed.
fn number_lone(
    lim: &Limits,
    slots: &mut Vec<u64>,
    hashes: &[u64],
    firsts: &mut Vec<u32>,
    (starts, n, long): (&[u32], usize, usize),
    (lone, keys, key): (&mut Vec<u32>, usize, impl Fn(u32) -> usize),
    same: impl Fn(u32, u32) -> bool,
) -> Result<(Vec<u32>, u32), OperationError> {
    let runs = hashes.len();
    let bits = index_bits(2 * long).max(4);
    slots.clear();
    lim.try_resize(slots, 1 << bits, 0u64)?;
    lone.clear();
    lim.try_resize(lone, keys, u32::MAX)?;
    let mut ids = Vec::new();
    lim.reserve_exact(&mut ids, runs)?;
    firsts.clear();
    let mut gate = lim.gate();
    gate.poll(runs as u64)?;
    for (k, &hash) in hashes.iter().enumerate() {
        let range = run(starts, k as u32, n);
        let id = if range.len() == 1 {
            let slot = &mut lone[key(range.start as u32)];
            if *slot == u32::MAX {
                *slot = firsts.len() as u32;
                lim.try_push(firsts, k as u32)?;
            }
            *slot
        } else {
            probe(lim, slots, bits, hash, k as u32, firsts, &same)?
        };
        ids.push(id);
    }
    gate.flush()?;
    Ok((ids, firsts.len() as u32))
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
/// values without a sort. Hashes that agree are confirmed against the sets
/// themselves; a high value's is its run, and the low values' are listed,
/// high value by high value, only when two of their hashes agree.
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
) -> Result<Option<(Decomposition, Values, Values)>, OperationError> {
    let n = parent.len();
    if parent.words != 1 || low_width > 32 || 1usize << low_width > (2 * n).max(DIRECT_LOWS) {
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
    s.low_hash.clear();
    lim.try_resize(&mut s.low_hash, 1 << low_width, RUN_SEED)?;
    s.low_count.clear();
    lim.try_resize(&mut s.low_count, 1 << low_width, 0u32)?;
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
            && data[x.clone()].iter().zip(&data[y.clone()]).all(|(&p, &q)| (p ^ q) & mask == 0)
            && (single || atoms[x] == atoms[y])
    };
    let (high_atom, high_atoms) = number_hashed(lim, &mut s.slots, &s.high_hash, &mut s.high_first, same_pairs)?;

    // The low values in ascending order, each slot's rank among them, and
    // the hashes in that order.
    let lows = s.low_count.iter().filter(|&&count| count > 0).count();
    s.order.clear();
    lim.reserve_exact(&mut s.order, lows)?;
    s.rank.clear();
    lim.try_resize(&mut s.rank, 1 << low_width, 0u32)?;
    s.high_keys.clear();
    lim.reserve_exact(&mut s.high_keys, lows)?;
    let mut low_data = Vec::new();
    lim.reserve_exact(&mut low_data, lows)?;
    gate.poll(1 << low_width)?;
    for (slot, &count) in s.low_count.iter().enumerate() {
        if count > 0 {
            s.rank[slot] = s.order.len() as u32;
            s.order.push(slot as u32);
            s.high_keys.push(s.low_hash[slot]);
            low_data.push(slot as u64);
        }
    }
    // Numbering by hash alone finds every low value distinct only when no
    // two hashes agree, and then no two low values share an atom.
    let (mut low_atom, mut low_atoms) = number_hashed(lim, &mut s.slots, &s.high_keys, &mut s.low_first, |_, _| true)?;
    let merged = low_atoms as usize != lows;
    if merged {
        lim.discard(low_atom);
        (low_atom, low_atoms) = confirm_lows(lim, s, parent, low_width)?;
    }

    // Runs of one high atom hold the same pairs, so they realize the same
    // triples, and the first run of each realizes them all.
    let atom_of = &mut s.high_of;
    atom_of.clear();
    lim.reserve_exact(atom_of, 1 << low_width)?;
    atom_of.extend(s.rank.iter().map(|&rank| if merged { low_atom[rank as usize] } else { rank }));
    let held = s.high_first.iter().map(|&k| run(&s.high_starts, k, n).len()).sum::<usize>();
    gate.poll(held as u64)?;
    let split = if single {
        // Each group of the one parent atom lists its low parts in ascending
        // order, and so their atoms unless low values merge.
        let mut lows = Vec::new();
        lim.reserve_exact(&mut lows, held)?;
        for &k in &s.high_first {
            lows.extend(data[run(&s.high_starts, k, n)].iter().map(|&value| atom_of[(value & mask) as usize]));
        }
        let sizes = s.high_first.iter().map(|&k| run(&s.high_starts, k, n).len());
        let ends = close_groups(lim, &mut lows, sizes, merged)?;
        Decomposition::Grouped { ends, lows }
    } else {
        let triples = by_parent_atom(lim, s, parent, low_width, &high_atom, held)?;
        Decomposition::Triples { atoms: parent.atoms as usize, triples }
    };
    gate.flush()?;

    let mut high_data = Vec::new();
    lim.reserve_exact(&mut high_data, high_count)?;
    high_data.extend_from_slice(&s.high);
    let low = Values { words: 1, data: low_data, atom: low_atom, atoms: low_atoms };
    let high = Values { words: 1, data: high_data, atom: high_atom, atoms: high_atoms };
    Ok(Some((split, low, high)))
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
        let slot = low as usize;
        s.low_hash[slot] = mix(s.low_hash[slot], index << 32 | atom);
        s.low_count[slot] += 1;
    }
    lim.try_push(&mut s.high_hash, hash)
}

/// Number the low values of `split_hashed` by their completion sets, listed
/// low value by low value in rank order, each as ascending pairs of a high
/// value's index and an atom.
fn confirm_lows(
    lim: &Limits,
    s: &mut Scratch,
    parent: &Values,
    low_width: usize,
) -> Result<(Vec<u32>, u32), OperationError> {
    let n = parent.len();
    let mask = (1u64 << low_width) - 1;
    let lows = s.order.len();
    s.low_starts.clear();
    lim.reserve_exact(&mut s.low_starts, lows + 1)?;
    let mut at = 0u32;
    for &slot in &s.order {
        s.low_starts.push(at);
        at += s.low_count[slot as usize];
    }
    s.low_starts.push(at);
    s.cursor.clear();
    lim.reserve_exact(&mut s.cursor, lows)?;
    s.cursor.extend_from_slice(&s.low_starts[..lows]);
    s.low_keys.clear();
    lim.try_resize(&mut s.low_keys, n, 0u64)?;
    lim.gate().poll(n as u64)?;
    for (high, range) in runs(&s.high_starts, n).enumerate() {
        for e in range {
            let at = &mut s.cursor[s.rank[(parent.data[e] & mask) as usize] as usize];
            s.low_keys[*at as usize] = (high as u64) << 32 | parent.atom[e] as u64;
            *at += 1;
        }
    }
    let (starts, lists) = (&s.low_starts, &s.low_keys);
    let list = |k: u32| &lists[starts[k as usize] as usize..starts[k as usize + 1] as usize];
    number_hashed(lim, &mut s.slots, &s.high_keys, &mut s.low_first, |a, b| list(a) == list(b))
}

/// The triples of `split_hashed` with more than one parent atom: those of
/// the first run of each high atom, grouped by parent atom through a
/// counting sort, then sorted and deduplicated group by group.
fn by_parent_atom(
    lim: &Limits,
    s: &mut Scratch,
    parent: &Values,
    low_width: usize,
    high_atom: &[u32],
    held: usize,
) -> Result<Vec<[u32; 3]>, OperationError> {
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
    s.sort_keys.clear();
    lim.try_resize(&mut s.sort_keys, held, 0u64)?;
    for (high, range) in runs(&s.high_starts, n).enumerate() {
        if first(high) {
            let pair = (high_atom[high] as u64) << 32;
            for e in range {
                let at = &mut s.cursor[parent.atom[e] as usize];
                s.sort_keys[*at as usize] = pair | s.high_of[(parent.data[e] & mask) as usize] as u64;
                *at += 1;
            }
        }
    }
    let mut triples = Vec::new();
    lim.reserve_exact(&mut triples, held)?;
    for atom in 0..atoms {
        let group = &mut s.sort_keys[s.low_starts[atom] as usize..s.low_starts[atom + 1] as usize];
        if group.len() > 1 {
            sort_small_by(group, !0);
        }
        let mut last = u64::MAX;
        for &pair in group.iter() {
            if pair != last {
                triples.push([atom as u32, (pair >> 32) as u32, pair as u32]);
                last = pair;
            }
        }
    }
    Ok(triples)
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

/// Runs of one atom list the same keys, so they realize the same triples.
/// Reading the first run of every atom on one side finds them all; whether
/// the low side's first runs hold no more values than the high's.
fn from_low(n: usize, s: &Scratch) -> bool {
    let held = |starts: &[u32], firsts: &[u32]| firsts.iter().map(|&k| run(starts, k, n).len()).sum::<usize>();
    held(&s.low_starts, &s.low_first) <= held(&s.high_starts, &s.high_first)
}

/// Split values of any width through their materialized parts.
fn split_parts(
    lim: &Limits,
    s: &mut Scratch,
    parent: &Values,
    (low_width, high_width): (usize, usize),
    distinct: bool,
) -> Result<(Decomposition, Values, Values), OperationError> {
    let n = parent.len();
    let (lw, rw) = (words_for(low_width), words_for(high_width));
    runs_of_parts(lim, s, parent, low_width, high_width, distinct)?;
    let (low_count, high_count) = (s.low_starts.len(), s.high_starts.len());

    let (low_atom, low_atoms, high_atom, high_atoms) = if distinct {
        (identity(lim, low_count)?, low_count as u32, identity(lim, high_count)?, high_count as u32)
    } else {
        // A high value's run lists its low partners in ascending order, and
        // a low value's run its high partners; each beside the parent atom.
        s.high_keys.clear();
        lim.reserve_exact(&mut s.high_keys, n)?;
        s.high_keys.extend(s.low_of[..n].iter().zip(&parent.atom).map(|(&l, &a)| (l as u64) << 32 | a as u64));
        let (high_atom, high_atoms) =
            number_runs(lim, &mut s.runs, &s.high_keys, &s.high_starts, (low_count, parent.atoms), &mut s.high_first)?;
        let (low_atom, low_atoms) =
            number_runs(lim, &mut s.runs, &s.low_keys, &s.low_starts, (high_count, parent.atoms), &mut s.low_first)?;
        (low_atom, low_atoms, high_atom, high_atoms)
    };
    let triples = triples(lim, s, parent, (&low_atom, low_atoms), (&high_atom, high_atoms))?;

    let mut low_data = Vec::new();
    lim.reserve_exact(&mut low_data, low_count * lw)?;
    for &i in &s.low_starts {
        let e = s.order[i as usize] as usize;
        low_data.extend_from_slice(&s.low[e * lw..][..lw]);
    }
    let mut high_data = Vec::new();
    lim.reserve_exact(&mut high_data, high_count * rw)?;
    for &e in &s.high_starts {
        high_data.extend_from_slice(&s.high[e as usize * rw..][..rw]);
    }
    let split = Decomposition::Triples { atoms: parent.atoms as usize, triples };
    let low = Values { words: lw, data: low_data, atom: low_atom, atoms: low_atoms };
    let high = Values { words: rw, data: high_data, atom: high_atom, atoms: high_atoms };
    Ok((split, low, high))
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
    let (lw, rw) = (words_for(low_width), words_for(high_width));
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
            _ => e == 0 || s.high[e * rw..][..rw] != s.high[(e - 1) * rw..][..rw],
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
            if i == 0 || s.low[e * lw..][..lw] != s.low[previous * lw..][..lw] {
                lim.try_push(&mut s.low_starts, i as u32)?;
            }
            s.low_of[e] = s.low_starts.len() as u32 - 1;
        }
    }
    s.low_keys.clear();
    if !distinct {
        lim.reserve_exact(&mut s.low_keys, n)?;
        let (high_of, atom) = (&s.high_of, &parent.atom);
        s.low_keys.extend(s.order.iter().map(|&e| (high_of[e as usize] as u64) << 32 | atom[e as usize] as u64));
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
    let (lw, rw) = (words_for(low_width), words_for(high_width));
    s.low.clear();
    lim.reserve_exact(&mut s.low, n * lw)?;
    s.high.clear();
    lim.reserve_exact(&mut s.high, n * rw)?;
    let mut gate = lim.gate();
    if parent.words == 1 {
        // Both parts are narrower than the parent's one word.
        let mask = (1u64 << low_width) - 1;
        gate.poll(n as u64)?;
        s.low.extend(parent.data.iter().map(|&v| v & mask));
        s.high.extend(parent.data.iter().map(|&v| v >> low_width));
    } else {
        s.low.resize(n * lw, 0);
        s.high.resize(n * rw, 0);
        for e in 0..n {
            gate.poll(parent.words as u64)?;
            let value = &parent.data[e * parent.words..][..parent.words];
            extract(value, 0, low_width, &mut s.low[e * lw..][..lw]);
            extract(value, low_width, high_width, &mut s.high[e * rw..][..rw]);
        }
    }
    gate.flush()
}

/// Copy the `width` bits of `src` from bit `start` on into `dst`, low word
/// first, clearing the bits of the last word past `width`.
fn extract(src: &[u64], start: usize, width: usize, dst: &mut [u64]) {
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
    if lw == 1 && low_width + shift <= 64 {
        // The index below the low part makes every key distinct, and in
        // index order before sorting, so any sort of the keys is stable.
        s.sort_keys.clear();
        lim.reserve_exact(&mut s.sort_keys, n)?;
        s.sort_keys.extend(s.low.iter().enumerate().map(|(e, &v)| v << shift | e as u64));
        s.radix.sort(lim, &mut s.sort_keys, shift, low_width)?;
        let mask = (1u64 << shift) - 1;
        s.order.extend(s.sort_keys.iter().map(|&key| (key & mask) as u32));
        lim.check_stop()?;
        return Ok(Some(shift));
    }
    if lw == 1 {
        s.wide_keys.clear();
        lim.reserve_exact(&mut s.wide_keys, n)?;
        s.wide_keys.extend(s.low.iter().enumerate().map(|(e, &v)| (v as u128) << 32 | e as u128));
        s.wide_keys.sort_unstable();
        s.order.extend(s.wide_keys.iter().map(|&key| key as u32));
    } else {
        s.order.extend(0..n as u32);
        let low = &s.low;
        s.order.sort_unstable_by(|&a, &b| {
            let (x, y) = (&low[a as usize * lw..][..lw], &low[b as usize * lw..][..lw]);
            x.iter().rev().cmp(y.iter().rev()).then(a.cmp(&b))
        });
    }
    lim.check_stop()?;
    Ok(None)
}

/// Number the runs of `keys` that start at `starts`, the last running to the
/// end, by content, in order of first appearance. Every key is an index below
/// `bounds.0` above an index below `bounds.1`. Returns each run's number and
/// how many distinct runs there are, and leaves in `firsts` the run where each
/// number first appears.
fn number_runs(
    lim: &Limits,
    table: &mut RunTable,
    keys: &[u64],
    starts: &[u32],
    bounds: (usize, u32),
    firsts: &mut Vec<u32>,
) -> Result<(Vec<u32>, u32), OperationError> {
    let n = keys.len();
    let mut ids = Vec::new();
    lim.reserve_exact(&mut ids, starts.len())?;
    firsts.clear();
    let mut gate = lim.gate();
    gate.poll(n as u64)?;
    if starts.len() == n {
        // Every run is one key, which is its own content.
        let span = bounds.0.saturating_mul(bounds.1 as usize);
        if span <= (4 * n).max(DIRECT_MIN_SPAN) {
            table.direct.clear();
            lim.try_resize(&mut table.direct, span, u32::MAX)?;
            for (k, &key) in keys.iter().enumerate() {
                let slot = &mut table.direct[(key >> 32) as usize * bounds.1 as usize + (key as u32) as usize];
                if *slot == u32::MAX {
                    *slot = firsts.len() as u32;
                    lim.try_push(firsts, k as u32)?;
                }
                ids.push(*slot);
            }
        } else {
            reset(lim, &mut table.head, n);
            lim.reserve_map(&mut table.head, n)?;
            for (k, &key) in keys.iter().enumerate() {
                let next = firsts.len() as u32;
                let id = *table.head.entry(key).or_insert(next);
                if id == next {
                    lim.try_push(firsts, k as u32)?;
                }
                ids.push(id);
            }
        }
        gate.flush()?;
        return Ok((ids, firsts.len() as u32));
    }
    reset(lim, &mut table.head, starts.len());
    // Every run may be new, and reserving for all of them at once spares the
    // table its growth steps.
    lim.reserve_map(&mut table.head, starts.len())?;
    table.next.clear();
    table.first.clear();
    for (k, &start) in starts.iter().enumerate() {
        let end = starts.get(k + 1).map_or(n, |&e| e as usize);
        let run = &keys[start as usize..end];
        let mut hasher = FxHasher::default();
        hasher.write_usize(run.len());
        for &key in run {
            hasher.write_u64(key);
        }
        let hash = hasher.finish();
        let chain = table.head.get(&hash).copied().unwrap_or(u32::MAX);
        let mut candidate = chain;
        while candidate != u32::MAX {
            let (at, len) = table.first[candidate as usize];
            if keys[at as usize..][..len as usize] == *run {
                break;
            }
            candidate = table.next[candidate as usize];
        }
        if candidate == u32::MAX {
            candidate = table.first.len() as u32;
            lim.try_push(&mut table.first, (start, run.len() as u32))?;
            lim.try_push(&mut table.next, chain)?;
            lim.try_push(firsts, k as u32)?;
            table.head.insert(hash, candidate);
        }
        ids.push(candidate);
    }
    gate.flush()?;
    Ok((ids, table.first.len() as u32))
}

/// Address ranges this small number lone keys directly whatever the key
/// count, which costs less than hashing them.
const DIRECT_MIN_SPAN: usize = 1 << 12;

/// Empty `map` for about `expected` entries. Clearing costs time in the
/// capacity, so a table grown by a much larger split is dropped instead.
fn reset(lim: &Limits, map: &mut FxHashMap<u64, u32>, expected: usize) {
    if map.capacity() > 4 * expected + 1024 {
        lim.discard(std::mem::take(map));
    } else {
        map.clear();
    }
}

/// The distinct `[atom, high atom, low atom]` triples of the parent values,
/// in ascending order.
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
            [parent.atom[e], high_atom[s.high_of[e] as usize], low_atom[s.low_of[e] as usize]]
        }));
        return Ok(out);
    }
    let from_low = from_low(n, s);
    let (order, low_of, high_of) = (&s.order, &s.low_of, &s.high_of);
    let (low_starts, high_starts) = (&s.low_starts, &s.high_starts);
    let (low_first, high_first) = (&s.low_first, &s.high_first);
    let each = |emit: &mut dyn FnMut([u32; 3])| {
        if from_low {
            for (l, &k) in low_first.iter().enumerate() {
                for &e in &order[run(low_starts, k, n)] {
                    emit([parent.atom[e as usize], high_atom[high_of[e as usize] as usize], l as u32]);
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
    sorted_triples(lim, &mut s.radix, &mut s.high_keys, &mut s.wide_triples, atoms, !from_low, each)
}

/// The triples `each` emits, sorted and each once: values of one atom can
/// decompose into the same pair of child atoms. `each` emits them grouped by
/// their middle atom, ascending, when `by_mid`, and else by their last.
/// `packed` and `wide` are buffers for the triples packed into one word or
/// two.
fn sorted_triples(
    lim: &Limits,
    radix: &mut Radix,
    packed: &mut Vec<u64>,
    wide: &mut Vec<u128>,
    atoms: (u32, u32, u32),
    by_mid: bool,
    each: impl Fn(&mut dyn FnMut([u32; 3])),
) -> Result<Vec<[u32; 3]>, OperationError> {
    let mut out = Vec::new();
    let bits = (index_bits(atoms.0 as usize), index_bits(atoms.1 as usize), index_bits(atoms.2 as usize));
    if bits.0 + bits.1 + bits.2 < 64 {
        let (low, mid) = (bits.2, bits.1 + bits.2);
        let mut keys = std::mem::take(packed);
        keys.clear();
        let mut refused = Ok(());
        each(&mut |[a, l, r]| {
            if refused.is_ok() {
                refused = lim.try_push(&mut keys, (a as u64) << mid | (l as u64) << low | r as u64);
            }
        });
        // The keys come out grouped by one child atom, ascending. Ordering
        // them by the child atoms is then a sort of each middle atom's small
        // group by its last atoms, or nothing at all, and only the parent
        // atom's bits are left to the radix sort.
        let ordered = if by_mid {
            let (low_mask, group_mask) = ((1u64 << low) - 1, (1u64 << (mid - low)) - 1);
            let mut start = 0;
            while start < keys.len() {
                let group = (keys[start] >> low) & group_mask;
                let mut end = start + 1;
                while end < keys.len() && (keys[end] >> low) & group_mask == group {
                    end += 1;
                }
                sort_small_by(&mut keys[start..end], low_mask);
                start = end;
            }
            mid
        } else {
            low
        };
        let sorted =
            refused.and_then(|()| radix.sort(lim, &mut keys, ordered as usize, (bits.0 + mid - ordered) as usize));
        keys.dedup();
        *packed = keys;
        sorted?;
        lim.reserve_exact(&mut out, packed.len())?;
        let (low_mask, mid_mask) = ((1u64 << low) - 1, (1u64 << (mid - low)) - 1);
        out.extend(packed.iter().map(|&key| {
            [(key >> mid) as u32, ((key >> low) & mid_mask) as u32, (key & low_mask) as u32]
        }));
    } else {
        let mut keys = std::mem::take(wide);
        keys.clear();
        let mut refused = Ok(());
        each(&mut |[a, l, r]| {
            if refused.is_ok() {
                refused = lim.try_push(&mut keys, (a as u128) << 64 | (l as u128) << 32 | r as u128);
            }
        });
        keys.sort_unstable();
        keys.dedup();
        *wide = keys;
        refused?;
        lim.reserve_exact(&mut out, wide.len())?;
        out.extend(wide.iter().map(|&key| [(key >> 64) as u32, (key >> 32) as u32, key as u32]));
    }
    lim.check_stop()?;
    Ok(out)
}

/// Sort `keys` by their bits under `mask`. A group of one child atom's
/// triples is short, and the other child's atoms often ascend already,
/// which an insertion sort passes over in one comparison apiece.
fn sort_small_by(keys: &mut [u64], mask: u64) {
    if keys.len() > 16 {
        keys.sort_unstable_by_key(|&key| key & mask);
        return;
    }
    for i in 1..keys.len() {
        let key = keys[i];
        let mut j = i;
        while j > 0 && keys[j - 1] & mask > key & mask {
            keys[j] = keys[j - 1];
            j -= 1;
        }
        keys[j] = key;
    }
}

#[cfg(test)]
#[path = "tests/split.rs"]
mod tests;
