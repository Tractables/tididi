//! The probe join: a sparse level built one `f` node at a time, each of its
//! pairs finding the `g` pairs it meets by lookups.
//!
//! The scatter ([`scatter_join`]) joins a level by its outer child: per live
//! outer product it builds a filtered index of the `g` pairs under it and
//! walks the `f` pairs against it, dropping each candidate into its `f`
//! parent's bucket, which the emit then drains. Where the level has millions
//! of outer products with a candidate or two each, the per-outer setup and the
//! bucket each candidate is scattered into are most of the work. Two operands
//! that share a block of variables have that shape where each holds, at the
//! block's root, one node per value of the block (or per set of values), so
//! that every node of one meets one node of the other, and where above the
//! block one operand is a single node whose pairs are those values.
//!
//! This route walks `f`'s nodes in order instead. A pair `(a, b)` of `f`
//! meets a pair `(c, d)` of `g` where both products `a ∧ c` and `b ∧ d` live;
//! three ways of finding those pairs are priced, and the cheapest runs when it
//! is linear in the level's pairs:
//!
//! - **by left** ([`Probe::Left`]): each left product `a ∧ c` of `a`, from the
//!   left child's product list, then each `g` pair with left child `c`, from
//!   a reverse index of `g` by left child, its right product read directly —
//!   from a grid, by arithmetic on a complete child, or as the identity where
//!   an operand is constant-true over the child. Costs `Σ_{a ∧ c live}
//!   |g pairs under c|` over `f`'s pairs: the shape above a shared block,
//!   where the other child holds variables of `g` alone, over which `f` is
//!   constant-true.
//! - **by right** ([`Probe::Right`]): the same with the children's roles
//!   exchanged.
//! - **by pair** ([`Probe::Pairs`]): each left product `a ∧ c` and each right
//!   product `b ∧ d`, and the `g` pair `(c, d)` from a table of `g`'s pairs.
//!   Costs `|left products of a| · |right products of b|` per pair of `f`:
//!   the shared block's root, where neither half of the block alone picks
//!   the value but both together do.
//!
//! Each `f` node's candidates go to the emit the scatter uses ([`emit_parent`])
//! as soon as the node is walked, so no candidate is bucketed. The candidates
//! are the scatter's own, so the level is too, up to the order of its nodes
//! within one `f` parent and of the pairs within a node.

use super::*;
use crate::apply::conjoin::setup::LevelShape;
use crate::diagram::Sides;
use crate::execution::pool::{Buffers, Scratch};
use crate::limits::Limits;

/// How a child level reads one product cell `(a, b)` without its list.
#[derive(Clone, Copy)]
pub(crate) enum CellLookup<'a> {
    /// `g` is constant-true over the child: its one node is index 0, and the
    /// product of `f`'s node `a` with it is `a`.
    GIdentity,
    /// `f` is constant-true over the child: the product of `g`'s node `b`
    /// with `f`'s one node is `b`.
    FIdentity,
    /// Every cell produced a node, in cell order: `(a, b)` is node
    /// `a · g_width + b`.
    Complete { g_width: usize },
    /// The child's grid, row-major over `g`'s width.
    Grid { cells: &'a [u32], g_width: usize },
    /// Only the child's product list holds its products.
    List,
}

impl CellLookup<'_> {
    /// Whether a cell is read directly, without the product list.
    fn direct(&self) -> bool {
        !matches!(self, CellLookup::List)
    }

    /// The product of cell `(a, b)`, `NO_PRODUCT` where it is false. Only a
    /// direct lookup is read.
    #[inline(always)]
    fn get(&self, a: u32, b: u32) -> u32 {
        match *self {
            CellLookup::GIdentity => {
                debug_assert_eq!(b, 0, "a constant-true operand has one node");
                a
            }
            CellLookup::FIdentity => {
                debug_assert_eq!(a, 0, "a constant-true operand has one node");
                b
            }
            CellLookup::Complete { g_width } => (a as usize * g_width + b as usize) as u32,
            CellLookup::Grid { cells, g_width } => cells[a as usize * g_width + b as usize],
            CellLookup::List => unreachable!("a list child is read through its list"),
        }
    }
}

/// The way a level's `f` pairs find the `g` pairs they meet.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Probe {
    /// Through the left products and `g`'s pairs by left child; the right
    /// product read directly.
    Left,
    /// Through the right products and `g`'s pairs by right child; the left
    /// product read directly.
    Right,
    /// Through both products, and a table of `g`'s pairs.
    Pairs,
}

/// Levels with fewer pairs than this, `f`'s and `g`'s together, are left to
/// the scatter: pricing the probe costs a pass over both, which a small level
/// does not repay.
const PROBE_MIN_PAIRS: u64 = 1 << 12;

/// The probe runs where its priced lookups are at most this many per pair of
/// the level, `f`'s and `g`'s together: a pass linear in the operands, which
/// is what the scatter's own reverse indices cost before it joins anything.
const PROBE_PER_PAIR: u64 = 2;

/// An empty slot of the pair table: no child index is `u32::MAX`.
const EMPTY_KEY: u64 = u64::MAX;

/// One slot of the pair table: a `g` pair, packed as `left << 32 | right`,
/// and the `g` node holding it.
#[derive(Clone, Copy)]
pub(super) struct PairSlot {
    key: u64,
    parent: u32,
}

impl Default for PairSlot {
    fn default() -> Self {
        PairSlot { key: EMPTY_KEY, parent: 0 }
    }
}

/// The probe join's buffers, kept in the sparse workspace between levels.
#[derive(Default)]
pub(crate) struct ProbeBuffers {
    /// Bucket bounds of the left and right children's product lists by `f`
    /// index.
    left_offsets: Vec<u32>,
    right_offsets: Vec<u32>,
    /// `g`'s pairs counted by left child and by right child.
    g_by_left: Vec<u32>,
    g_by_right: Vec<u32>,
    /// `g`'s pairs grouped by the child the walk goes through.
    index: Grouped<RevEntry>,
    /// `g`'s pairs by both children, open addressing, a power of two.
    table: Vec<PairSlot>,
    /// `g`'s pairs by both children, dense, left-major; `NO_PRODUCT` where
    /// no pair is, whenever `dense_clean`.
    dense: Vec<u32>,
    dense_clean: bool,
    /// One `f` node's candidates.
    candidates: Vec<ParEntry>,
}

impl Buffers for ProbeBuffers {
    fn buffers(&mut self, visit: &mut dyn FnMut(&mut dyn Scratch)) {
        visit(&mut self.left_offsets);
        visit(&mut self.right_offsets);
        visit(&mut self.g_by_left);
        visit(&mut self.g_by_right);
        self.index.buffers(visit);
        visit(&mut self.table);
        visit(&mut self.dense);
        visit(&mut self.candidates);
    }
}

/// The table slot to start probing `key` at, of `mask + 1` slots.
#[inline(always)]
fn slot_of(key: u64, mask: usize) -> usize {
    (key.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 32) as usize & mask
}

#[inline(always)]
fn pair_key(left: u32, right: u32) -> u64 {
    (u64::from(left) << 32) | u64::from(right)
}

/// Count `level`'s pairs into `by_left` and `by_right`, by child index, and
/// return the total.
fn count_pairs(
    lim: &Limits,
    level: &TddLevel,
    widths: Sides<usize>,
    by_left: &mut Vec<u32>,
    by_right: &mut Vec<u32>,
) -> Result<u64, OperationError> {
    by_left.clear();
    by_right.clear();
    lim.try_resize(by_left, widths.left, 0)?;
    lim.try_resize(by_right, widths.right, 0)?;
    let mut total = 0u64;
    for (_, pairs) in level.internal_inputs_iter() {
        total += pairs.len() as u64;
        for pair in pairs {
            by_left[pair.left.raw() as usize] += 1;
            by_right[pair.right.raw() as usize] += 1;
        }
    }
    if total > u64::from(u32::MAX) {
        return Err(OperationError::IndexOverflow);
    }
    Ok(total)
}

/// The pairs of `f`'s level whose cost [`plan`] reads exactly; past it, it
/// reads every `k`-th node for some `k` that leaves about this many.
const PLAN_SAMPLE_PAIRS: u64 = 1 << 14;

/// Price the three probes on a level and return the one to run, or `None`
/// where none is linear in the level's pairs and the scatter should join it.
///
/// A probe's cost is the lookups it makes per pair of `f`, read off the
/// children's product lists and `g`'s pair counts by child: exactly on a
/// level of up to [`PLAN_SAMPLE_PAIRS`] pairs, and on a wider one from the
/// pairs of evenly spaced nodes, scaled up. A sample only decides a route;
/// any route builds the same level.
///
/// `pairs_ok` admits [`Probe::Pairs`], whose table holds each `g` pair once:
/// it is withheld where pair lists may be multisets.
#[expect(clippy::too_many_arguments)]
fn plan(
    eng: &Engine,
    pb: &mut ProbeBuffers,
    f_level: &TddLevel,
    g_level: &TddLevel,
    shape: LevelShape,
    pl: Sides<&[ProductEntry]>,
    cells: Sides<CellLookup<'_>>,
    pairs_ok: bool,
) -> Result<Option<(Probe, u64)>, OperationError> {
    let lim = eng.limits();
    let by_left = cells.right.direct();
    let by_right = cells.left.direct();
    if !(by_left || by_right || pairs_ok) {
        return Ok(None);
    }
    let f_pairs: u64 = (0..f_level.nodes().len()).map(|i| f_level.pair_count_at(i) as u64).sum();
    let g_pairs = count_pairs(lim, g_level, Sides { left: shape.g.left, right: shape.g.right },
        &mut pb.g_by_left, &mut pb.g_by_right)?;
    if f_pairs + g_pairs < PROBE_MIN_PAIRS && !probe_forced() {
        return Ok(None);
    }
    bucket_offsets(lim, pl.left, shape.f.left, &mut pb.left_offsets)?;
    bucket_offsets(lim, pl.right, shape.f.right, &mut pb.right_offsets)?;
    let (lo, ro) = (&pb.left_offsets, &pb.right_offsets);
    let walk = |list: &[ProductEntry], offsets: &[u32], by_g: &[u32], a: usize| -> u64 {
        list[offsets[a] as usize..offsets[a + 1] as usize].iter().map(|e| u64::from(by_g[e.g_idx.idx()]) + 1).sum()
    };
    let stride = (f_pairs / PLAN_SAMPLE_PAIRS).max(1) as usize;
    let (mut sampled, mut left, mut right, mut both) = (0u64, 0u64, 0u64, 0u64);
    for i in (0..f_level.nodes().len()).step_by(stride) {
        for pair in f_level.pairs_iter_of_idx(i) {
            let (a, b) = (pair.left.raw() as usize, pair.right.raw() as usize);
            sampled += 1;
            if by_left {
                left += walk(pl.left, lo, &pb.g_by_left, a);
            }
            if by_right {
                right += walk(pl.right, ro, &pb.g_by_right, b);
            }
            let n_left = u64::from(lo[a + 1] - lo[a]);
            let n_right = u64::from(ro[b + 1] - ro[b]);
            both = both.saturating_add(n_left * n_right + 1);
        }
    }
    let scale = |cost: u64| cost.saturating_mul(f_pairs) / sampled.max(1);
    let options = [
        (by_left, scale(left), Probe::Left),
        (by_right, scale(right), Probe::Right),
        (pairs_ok, scale(both), Probe::Pairs),
    ];
    let best = options.iter().filter(|o| o.0).min_by_key(|o| o.1).map(|o| (o.1, o.2));
    let level_pairs = f_pairs + g_pairs;
    Ok(best.and_then(|(cost, probe)| {
        (probe_forced() || cost <= PROBE_PER_PAIR.saturating_mul(level_pairs)).then_some((probe, level_pairs))
    }))
}

/// Build the level at `shape.t` by the probe join when it prices as linear,
/// and say whether it did; otherwise leave everything for the scatter.
///
/// The children must be internal and joined (no pass-through side), and the
/// level must have more than one `f` or `g` node: one of each is the
/// scatter's direct case.
///
/// # Errors
///
/// [`OperationError::OverBudget`] when a buffer, the output's arenas or the
/// product list cannot grow, [`OperationError::IndexOverflow`] past `u32`
/// indices, and [`OperationError::Stopped`] when the stop fires on the poll.
#[expect(clippy::too_many_arguments)]
pub(super) fn probe_level(
    eng: &Engine,
    ws: &mut SparseWorkspace,
    f: &Tdd,
    g: &Tdd,
    shape: LevelShape,
    level: &mut TddLevel,
    pl: Sides<&[ProductEntry]>,
    pl_output: &mut Vec<ProductEntry>,
    cells: Sides<CellLookup<'_>>,
    pairs_ok: bool,
    duplicates_legal: bool,
) -> Result<bool, OperationError> {
    if probe_forced_off() {
        return Ok(false);
    }
    let t = shape.t.idx();
    let (f_level, g_level) = (&f.levels[t], &g.levels[t]);
    // The buffers leave the workspace for the level, so the emit can borrow
    // it, and go back on every outcome, so what they hold stays charged to
    // the pool that will release it.
    let mut pb = std::mem::take(&mut ws.probe);
    let result = (|| {
        let Some((probe, level_pairs)) = plan(eng, &mut pb, f_level, g_level, shape, pl, cells, pairs_ok)? else {
            return Ok(false);
        };
        note_probe(probe);
        // The emit looks each candidate's `g` parent up here; it leaves the
        // map as it found it, all `NO_PRODUCT`.
        eng.limits().try_resize(&mut ws.p2_map, shape.g.here, NO_PRODUCT)?;
        match probe {
            Probe::Left => walk_one_side::<false>(eng, ws, &mut pb, f_level, g_level, shape, level, pl, pl_output, cells.right, duplicates_legal)?,
            Probe::Right => walk_one_side::<true>(eng, ws, &mut pb, f_level, g_level, shape, level, pl, pl_output, cells.left, duplicates_legal)?,
            Probe::Pairs => walk_pairs(eng, ws, &mut pb, f_level, g_level, shape, level, pl, pl_output, level_pairs, duplicates_legal)?,
        }
        Ok(true)
    })();
    ws.probe = pb;
    result
}

/// The walk through one child: through the left child unswapped, the right
/// one `SWAPPED`, the other child's product read by `other`.
#[expect(clippy::too_many_arguments)]
fn walk_one_side<const SWAPPED: bool>(
    eng: &Engine,
    ws: &mut SparseWorkspace,
    pb: &mut ProbeBuffers,
    f_level: &TddLevel,
    g_level: &TddLevel,
    shape: LevelShape,
    level: &mut TddLevel,
    pl: Sides<&[ProductEntry]>,
    pl_output: &mut Vec<ProductEntry>,
    other: CellLookup<'_>,
    duplicates_legal: bool,
) -> Result<(), OperationError> {
    let lim = eng.limits();
    let (list, offsets, counts, key_width) = if !SWAPPED {
        (pl.left, &pb.left_offsets, &pb.g_by_left, shape.g.left)
    } else {
        (pl.right, &pb.right_offsets, &pb.g_by_right, shape.g.right)
    };
    build_reverse_index::<SWAPPED>(eng, g_level, key_width, Some(&counts[..key_width]), &mut pb.index)?;
    let index = pb.index.view();
    let candidates = &mut pb.candidates;
    let mut gate = lim.gate_with(super::super::budget::APPLY_POLL_STRIDE);
    for i in 0..shape.f.here {
        candidates.clear();
        let mut work = 0u64;
        for pair in f_level.pairs_iter_of_idx(i) {
            let (a, b) = if !SWAPPED { (pair.left.raw(), pair.right.raw()) } else { (pair.right.raw(), pair.left.raw()) };
            let bucket = &list[offsets[a as usize] as usize..offsets[a as usize + 1] as usize];
            for e in bucket {
                let through = e.prod_idx.0;
                let under = index.bucket(e.g_idx.idx());
                work += under.len() as u64 + 1;
                for r in under {
                    let across = other.get(b, r.other);
                    if across == NO_PRODUCT {
                        continue;
                    }
                    let (left_prod, right_prod) = if !SWAPPED { (through, across) } else { (across, through) };
                    lim.try_push(candidates, ParEntry { g_parent: r.parent, left_prod, right_prod })?;
                }
            }
        }
        gate.poll(work + 1)?;
        if !candidates.is_empty() {
            emit_parent(eng, ws, level, pl_output, i, candidates, duplicates_legal)?;
        }
    }
    gate.flush()
}

/// Cells of `g`'s pair map, left child by right child, up to which the
/// map is a dense array rather than a hash table, when the level has at
/// least a quarter as many pairs: the array is filled once and cleared by
/// the pairs that wrote it, and a lookup is one read.
const DENSE_PAIR_CELLS: usize = 1 << 24;

/// The walk through both children, each candidate's `g` pair found in a
/// map of `g`'s pairs: a dense array over the two children's widths where
/// it is small, else an open-addressing table.
#[expect(clippy::too_many_arguments)]
fn walk_pairs(
    eng: &Engine,
    ws: &mut SparseWorkspace,
    pb: &mut ProbeBuffers,
    f_level: &TddLevel,
    g_level: &TddLevel,
    shape: LevelShape,
    level: &mut TddLevel,
    pl: Sides<&[ProductEntry]>,
    pl_output: &mut Vec<ProductEntry>,
    level_pairs: u64,
    duplicates_legal: bool,
) -> Result<(), OperationError> {
    let lim = eng.limits();
    let width = shape.g.right;
    let cells = shape.g.left.saturating_mul(width);
    if cells <= DENSE_PAIR_CELLS && cells as u64 <= 4 * level_pairs && !pairs_hashed_forced() {
        if !pb.dense_clean {
            pb.dense.fill(NO_PRODUCT);
        }
        lim.try_resize(&mut pb.dense, cells, NO_PRODUCT)?;
        pb.dense_clean = false;
        for (parent, pairs) in g_level.internal_inputs_iter() {
            for pair in pairs {
                let at = pair.left.raw() as usize * width + pair.right.raw() as usize;
                debug_assert_eq!(pb.dense[at], NO_PRODUCT, "a g pair in two nodes, or twice in one");
                pb.dense[at] = parent as u32;
            }
        }
        let dense = &pb.dense;
        walk_both(eng, ws, f_level, level, pl, pl_output, (&pb.left_offsets, &pb.right_offsets), &mut pb.candidates, duplicates_legal,
            #[inline(always)]
            |c, d| dense[c as usize * width + d as usize])?;
        for (_, pairs) in g_level.internal_inputs_iter() {
            for pair in pairs {
                pb.dense[pair.left.raw() as usize * width + pair.right.raw() as usize] = NO_PRODUCT;
            }
        }
        pb.dense_clean = true;
        return Ok(());
    }
    let g_pairs: usize = pb.g_by_left[..shape.g.left].iter().map(|&n| n as usize).sum();
    let slots = (2 * g_pairs).next_power_of_two().max(16);
    let mask = slots - 1;
    pb.table.clear();
    lim.try_resize(&mut pb.table, slots, PairSlot::default())?;
    let table = &mut pb.table;
    for (parent, pairs) in g_level.internal_inputs_iter() {
        for pair in pairs {
            let key = pair_key(pair.left.raw(), pair.right.raw());
            let mut s = slot_of(key, mask);
            while table[s].key != EMPTY_KEY {
                debug_assert!(table[s].key != key, "a g pair in two nodes, or twice in one");
                s = (s + 1) & mask;
            }
            table[s] = PairSlot { key, parent: parent as u32 };
        }
    }
    let table = &pb.table;
    walk_both(eng, ws, f_level, level, pl, pl_output, (&pb.left_offsets, &pb.right_offsets), &mut pb.candidates, duplicates_legal,
        #[inline(always)]
        |c, d| {
            let key = pair_key(c, d);
            let mut s = slot_of(key, mask);
            loop {
                let slot = table[s];
                if slot.key == key {
                    return slot.parent;
                }
                if slot.key == EMPTY_KEY {
                    return NO_PRODUCT;
                }
                s = (s + 1) & mask;
            }
        })
}

/// The walk of [`walk_pairs`] with `find` naming the `g` node of a pair of
/// `g`'s children, `NO_PRODUCT` for none.
#[expect(clippy::too_many_arguments)]
#[inline(always)]
fn walk_both(
    eng: &Engine,
    ws: &mut SparseWorkspace,
    f_level: &TddLevel,
    level: &mut TddLevel,
    pl: Sides<&[ProductEntry]>,
    pl_output: &mut Vec<ProductEntry>,
    (lo, ro): (&[u32], &[u32]),
    candidates: &mut Vec<ParEntry>,
    duplicates_legal: bool,
    find: impl Fn(u32, u32) -> u32,
) -> Result<(), OperationError> {
    let lim = eng.limits();
    let mut gate = lim.gate_with(super::super::budget::APPLY_POLL_STRIDE);
    for i in 0..f_level.nodes().len() {
        candidates.clear();
        let mut work = 0u64;
        for pair in f_level.pairs_iter_of_idx(i) {
            let (a, b) = (pair.left.raw() as usize, pair.right.raw() as usize);
            let lefts = &pl.left[lo[a] as usize..lo[a + 1] as usize];
            let rights = &pl.right[ro[b] as usize..ro[b + 1] as usize];
            work += (lefts.len() * rights.len()) as u64 + 1;
            for l in lefts {
                for r in rights {
                    let parent = find(l.g_idx.0, r.g_idx.0);
                    if parent != NO_PRODUCT {
                        lim.try_push(candidates, ParEntry { g_parent: parent, left_prod: l.prod_idx.0, right_prod: r.prod_idx.0 })?;
                    }
                }
            }
        }
        gate.poll(work)?;
        if !candidates.is_empty() {
            emit_parent(eng, ws, level, pl_output, i, candidates, duplicates_legal)?;
        }
    }
    gate.flush()
}

// A test sends every level past the probe, as the oracle it is checked
// against, or takes it wherever it is admissible whatever it prices at, and
// counts the levels each probe built.
#[cfg(test)]
use crate::apply::conjoin::tests::{note_probe, pairs_hashed_forced, probe_forced, probe_forced_off};

#[cfg(not(test))]
#[inline(always)]
fn probe_forced_off() -> bool {
    false
}

#[cfg(not(test))]
#[inline(always)]
fn probe_forced() -> bool {
    false
}

#[cfg(not(test))]
#[inline(always)]
fn note_probe(_probe: Probe) {}

#[cfg(not(test))]
#[inline(always)]
fn pairs_hashed_forced() -> bool {
    false
}
