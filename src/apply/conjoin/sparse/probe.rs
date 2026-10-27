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
//! Each `f` node's candidates are written into the level's pair arena and
//! emitted as the node ends, as the scatter's routed walk does
//! ([`walk_nodes`]), so no candidate is bucketed. The candidates
//! are the scatter's own, so the level is too, up to the order of its nodes
//! within one `f` parent and of the pairs within a node. A walk through one
//! child that would write `f`'s level as `f` stores it, each pair its own
//! candidate and each node's pairs under one `g` parent, takes `f`'s level
//! instead ([`settle_f`]).

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
    level.for_each_node_pair(|_, pair| {
        total += 1;
        by_left[pair.left.raw() as usize] += 1;
        by_right[pair.right.raw() as usize] += 1;
    });
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
/// [`Probe::Pairs`], whose table holds each `g` pair once, is withheld where
/// pair lists may be multisets, which `multisets` tells, read only on a
/// level past the size gate (it may scan the diagram). A child `listed` does
/// not name has no product list (a complete child read by arithmetic): no
/// probe walks through it, and [`Probe::Pairs`], which walks through both,
/// is withheld.
#[expect(clippy::too_many_arguments)]
fn plan(
    eng: &Engine,
    pb: &mut ProbeBuffers,
    f_level: &TddLevel,
    g_level: &TddLevel,
    shape: LevelShape,
    pl: Sides<&[ProductEntry]>,
    listed: Sides<bool>,
    cells: Sides<CellLookup<'_>>,
    multisets: impl FnOnce() -> bool,
) -> Result<Option<(Probe, u64)>, OperationError> {
    let lim = eng.limits();
    let by_left = listed.left && cells.right.direct();
    let by_right = listed.right && cells.left.direct();
    let both_listed = listed.left && listed.right;
    if !(by_left || by_right || both_listed) {
        return Ok(None);
    }
    let f_pairs = f_level.live_pairs() as u64;
    let g_pairs = count_pairs(lim, g_level, Sides { left: shape.g.left, right: shape.g.right },
        &mut pb.g_by_left, &mut pb.g_by_right)?;
    if f_pairs + g_pairs < PROBE_MIN_PAIRS && !probe_forced() {
        return Ok(None);
    }
    let pairs_ok = both_listed && !multisets();
    if !(by_left || by_right || pairs_ok) {
        return Ok(None);
    }
    // An unlisted side's offsets are left empty: no option reading them is open.
    pb.left_offsets.clear();
    pb.right_offsets.clear();
    if listed.left {
        bucket_offsets(lim, pl.left, shape.f.left, &mut pb.left_offsets)?;
    }
    if listed.right {
        bucket_offsets(lim, pl.right, shape.f.right, &mut pb.right_offsets)?;
    }
    let (lo, ro) = (&pb.left_offsets, &pb.right_offsets);
    let walk = |list: &[ProductEntry], offsets: &[u32], by_g: &[u32], a: usize| -> u64 {
        list[offsets[a] as usize..offsets[a + 1] as usize].iter().map(|e| u64::from(by_g[e.g_idx.idx()]) + 1).sum()
    };
    let stride = (f_pairs / PLAN_SAMPLE_PAIRS).max(1) as usize;
    let (mut sampled, mut left, mut right, mut both) = (0u64, 0u64, 0u64, 0u64);
    // An implicit level's sampled nodes are read in order, each first pair
    // stepped on from the last's.
    let mut cursor = None;
    for i in (0..f_level.nodes().len()).step_by(stride) {
        for pair in f_level.pairs_iter_next(&mut cursor, i) {
            let (a, b) = (pair.left.raw() as usize, pair.right.raw() as usize);
            sampled += 1;
            if by_left {
                left += walk(pl.left, lo, &pb.g_by_left, a);
            }
            if by_right {
                right += walk(pl.right, ro, &pb.g_by_right, b);
            }
            if pairs_ok {
                let n_left = u64::from(lo[a + 1] - lo[a]);
                let n_right = u64::from(ro[b + 1] - ro[b]);
                both = both.saturating_add(n_left * n_right + 1);
            }
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

/// Build the level at `shape.t` into `emit` by the probe join when it prices
/// as linear, and say what it made of the level ([`Made::F`] where the level
/// is `f`'s, which the caller moves or copies in); `None` leaves everything
/// for the scatter.
///
/// The children must be internal and joined (no pass-through side), and the
/// level must have more than one `f` or `g` node: one of each is the
/// scatter's direct case. A child `listed` does not name has no product
/// list and is read only directly.
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
    pl: Sides<&[ProductEntry]>,
    listed: Sides<bool>,
    cells: Sides<CellLookup<'_>>,
    multisets: impl FnOnce() -> bool,
    emit: Emit<'_>,
) -> Result<Option<Made>, OperationError> {
    if probe_forced_off() {
        return Ok(None);
    }
    let t = shape.t.idx();
    let (f_level, g_level) = (&f.levels[t], &g.levels[t]);
    // The buffers leave the workspace for the level, so the emit can borrow
    // it, and go back on every outcome, so what they hold stays charged to
    // the pool that will release it.
    let mut pb = std::mem::take(&mut ws.probe);
    let result = (|| {
        let Some((probe, level_pairs)) = plan(eng, &mut pb, f_level, g_level, shape, pl, listed, cells, multisets)? else {
            return Ok(None);
        };
        note_probe(probe);
        // The emit looks each candidate's `g` parent up here; it leaves the
        // map as it found it, all `NO_PRODUCT`.
        eng.limits().try_resize(&mut ws.emit.p2_map, shape.g.here, NO_PRODUCT)?;
        let made = match probe {
            Probe::Left => walk_one_side::<false>(eng, ws, &mut pb, f_level, g_level, shape, pl, cells.right, emit)?,
            Probe::Right => walk_one_side::<true>(eng, ws, &mut pb, f_level, g_level, shape, pl, cells.left, emit)?,
            Probe::Pairs => {
                walk_pairs(eng, ws, &mut pb, f_level, g_level, shape, pl, level_pairs, emit)?;
                Made::Written
            }
        };
        Ok(Some(made))
    })();
    ws.probe = pb;
    result
}

/// Walk `f`'s nodes with `walk`, each node's candidates written into the
/// level's pair arena and emitted as the node ends ([`walk_nodes`]).
fn walk_level<const SWAPPED: bool, W: NodeWalk>(
    eng: &Engine,
    ws: &mut SparseWorkspace,
    walk: &W,
    f_level: &TddLevel,
    emit: Emit<'_>,
) -> Result<(), OperationError> {
    let SparseWorkspace { node_owners, emit: scratch, .. } = ws;
    let nodes = Walked::All(f_level.nodes().len());
    walk_nodes::<SWAPPED, W>(eng, walk, f_level, nodes, Collect::Nodes(emit), node_owners, scratch)
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
    pl: Sides<&[ProductEntry]>,
    other: CellLookup<'_>,
    mut emit: Emit<'_>,
) -> Result<Made, OperationError> {
    let (list, offsets, counts, key_width) = if !SWAPPED {
        (pl.left, &pb.left_offsets, &pb.g_by_left, shape.g.left)
    } else {
        (pl.right, &pb.right_offsets, &pb.g_by_right, shape.g.right)
    };
    build_reverse_index::<SWAPPED>(eng, g_level, key_width, Some(&counts[..key_width]), &mut pb.index)?;
    let f_width = if !SWAPPED { shape.f.left } else { shape.f.right };
    let direct = one_each(list, f_width) && counts[..key_width].iter().all(|&n| n == 1);
    let walk = OneSide { list, offsets, index: pb.index.view(), other, direct };
    // Where each pair's candidate is the pair itself, the level is f's
    // where each f node's pairs meet one g parent.
    let itself = direct
        && matches!(other, CellLookup::GIdentity)
        && list.iter().enumerate().all(|(i, e)| e.prod_idx.0 as usize == i);
    if itself && f_level.stored().is_some() {
        // Each walked child's g parent: its one product's one g pair's.
        let parents = &mut ws.node_owners;
        parents.clear();
        eng.limits().reserve(parents, list.len())?;
        parents.extend(list.iter().map(|e| walk.index.entries[e.g_idx.idx()].parent));
        if settle_f::<SWAPPED>(eng, parents, f_level, &mut emit)? {
            return Ok(Made::F);
        }
    }
    walk_level::<SWAPPED, _>(eng, ws, &walk, f_level, emit)?;
    Ok(Made::Written)
}

/// Whether the level a [`OneSide`] walk of `f_level` would make is f's
/// level, node for node and pair for pair, as f stores it; where it is,
/// list each node's product and charge the clock as the walk would, write
/// nothing, and leave the level's live pairs ([`Made::F`]). The walk is
/// direct, and each pair's candidate is the pair itself: the walked
/// child's products are numbered as its nodes, and the other side's
/// product is f's child; `parents` holds each walked child's g parent.
/// The level is then f's where the pairs of each f node meet one g parent,
/// f's node words place its pairs in its arena in order, and the level's
/// arenas, empty, hold f's without growing, as the walk's pushes would.
/// Otherwise nothing is listed or charged: the test reads f's pairs and
/// charges nothing.
fn settle_f<const SWAPPED: bool>(
    eng: &Engine,
    parents: &[u32],
    f_level: &TddLevel,
    emit: &mut Emit<'_>,
) -> Result<bool, OperationError> {
    use crate::diagram::NodeKind;
    let Some(stored) = f_level.stored() else {
        return Ok(false);
    };
    let Emit { level, out, live, .. } = emit;
    let (f_arena, f_nodes) = (stored.arena(), stored.nodes());
    // The walk pushes a node's first pair before it takes a one-pair
    // node's back, so the arena needs one place past f's.
    let room = out.is_empty()
        && level.pairs.stored_len() == 0
        && level.nodes.stored().is_empty()
        && f_arena.len() < level.pairs.stored_mut().capacity().min(1 << 31)
        && f_nodes.len() <= level.nodes.stored_mut().capacity();
    if !room || crate::apply::conjoin::lend_forced_off() {
        return Ok(false);
    }
    let parent = |pair: &ChildPair| parents[if !SWAPPED { pair.left.raw() } else { pair.right.raw() } as usize];
    // f's arena pairs the node words have placed so far.
    let mut placed = 0usize;
    for node in f_nodes {
        let len = match node.kind() {
            NodeKind::Inline(_) => continue,
            NodeKind::Multi { start, len } if start as usize == placed && len != 0 => len,
            _ => return Ok(false),
        };
        let node_pairs = &f_arena[placed..placed + len as usize];
        placed += len as usize;
        let first = parent(&node_pairs[0]);
        if !node_pairs[1..].iter().fold(true, |same, pair| same & (parent(pair) == first)) {
            return Ok(false);
        }
    }
    if placed != f_arena.len() {
        return Ok(false);
    }
    // The products, each its node's, and the walk's charges, a node at a
    // time and each before the node's product as the walk's: two units a
    // pair and one a node.
    let lim = eng.limits();
    let mut ticker = lim.gate_with(super::super::budget::APPLY_POLL_STRIDE);
    let mut pairs = 0;
    for (i, node) in f_nodes.iter().enumerate() {
        let node_pairs = stored.of(node);
        pairs += node_pairs.len();
        ticker.poll(2 * node_pairs.len() as u64 + 1)?;
        let product = ProductEntry {
            f_idx: FNodeIdx(i as u32),
            g_idx: GNodeIdx(parent(&node_pairs[0])),
            prod_idx: ProductNodeIdx(i as u32),
        };
        if out.len() < out.capacity() {
            out.push(product);
        } else {
            lim.try_push(out, product)?;
        }
    }
    crate::apply::conjoin::note_lent();
    note_settled();
    **live = Some(pairs);
    Ok(true)
}

/// What [`walk_one_side`] reads for each `f` pair: the walked child's
/// product list and its bucket bounds by `f` index, `g`'s pairs by the
/// walked child, and the other child's products.
///
/// Where the walked child lists one product per `f` index, in order
/// ([`one_each`]), and `g` has one pair under each of its walked child's
/// nodes, the walk is `direct`: a pair's product is read at its child and
/// the `g` pair under it at the product's `g` index, without bucket bounds.
struct OneSide<'a> {
    list: &'a [ProductEntry],
    offsets: &'a [u32],
    index: GroupedView<'a, RevEntry>,
    other: CellLookup<'a>,
    direct: bool,
}

/// The pairs a [`OneSide`] walk reads the products and `g` pairs of before
/// it reads any: a group's misses on the product list and the index then
/// overlap.
const SIDE_GROUP: usize = 16;

impl OneSide<'_> {
    /// The direct walk of a node's pairs, the other child's product read by
    /// `other` (the walk's [`CellLookup`], one case of it): a group of
    /// pairs' products is read, the cache asked for the `g` pair under each,
    /// and then each is read.
    #[inline(always)]
    fn direct_node<const SWAPPED: bool>(
        &self,
        pairs: &[ChildPair],
        other: impl Fn(u32, u32) -> u32,
        push: &mut impl FnMut(ParEntry) -> Result<(), OperationError>,
    ) -> Result<(), OperationError> {
        let mut group = [(0u32, 0u32, 0u32); SIDE_GROUP];
        for chunk in pairs.chunks(SIDE_GROUP) {
            for (looked, pair) in group.iter_mut().zip(chunk) {
                let (a, b) = if !SWAPPED { (pair.left.raw(), pair.right.raw()) } else { (pair.right.raw(), pair.left.raw()) };
                let e = &self.list[a as usize];
                prefetch_at(self.index.entries, e.g_idx.idx());
                *looked = (e.g_idx.0, e.prod_idx.0, b);
            }
            for &(g, through, b) in &group[..chunk.len()] {
                let r = &self.index.entries[g as usize];
                let across = other(b, r.other);
                if across != NO_PRODUCT {
                    let (left_prod, right_prod) = if !SWAPPED { (through, across) } else { (across, through) };
                    push(ParEntry { g_parent: r.parent, left_prod, right_prod })?;
                }
            }
        }
        Ok(())
    }
}

impl NodeWalk for OneSide<'_> {
    #[inline(always)]
    fn node<const SWAPPED: bool>(
        &self,
        pairs: &[ChildPair],
        ticker: &mut crate::limits::PollGate,
        mut push: impl FnMut(ParEntry) -> Result<(), OperationError>,
    ) -> Result<(), OperationError> {
        if self.direct {
            // A bucket of one product over one `g` pair: two units a pair,
            // as below.
            match self.other {
                CellLookup::GIdentity => self.direct_node::<SWAPPED>(pairs, |a, _| a, &mut push)?,
                CellLookup::FIdentity => self.direct_node::<SWAPPED>(pairs, |_, b| b, &mut push)?,
                other => self.direct_node::<SWAPPED>(pairs, |a, b| other.get(a, b), &mut push)?,
            }
            return ticker.poll(2 * pairs.len() as u64 + 1);
        }
        let mut work = 0u64;
        for pair in pairs {
            let (a, b) = if !SWAPPED { (pair.left.raw(), pair.right.raw()) } else { (pair.right.raw(), pair.left.raw()) };
            let bucket = &self.list[self.offsets[a as usize] as usize..self.offsets[a as usize + 1] as usize];
            for e in bucket {
                let through = e.prod_idx.0;
                let under = self.index.bucket(e.g_idx.idx());
                work += under.len() as u64 + 1;
                for r in under {
                    let across = self.other.get(b, r.other);
                    if across == NO_PRODUCT {
                        continue;
                    }
                    let (left_prod, right_prod) = if !SWAPPED { (through, across) } else { (across, through) };
                    push(ParEntry { g_parent: r.parent, left_prod, right_prod })?;
                }
            }
        }
        ticker.poll(work + 1)
    }
}

/// Cells of `g`'s pair map, left child by right child, up to which the
/// map is a dense array rather than a hash table, when the level has at
/// least a quarter as many pairs: the array is filled once and cleared by
/// the pairs that wrote it, and a lookup is one read.
const DENSE_PAIR_CELLS: usize = 1 << 24;

/// The walk through both children, each candidate's `g` pair found in a
/// map of `g`'s pairs: a dense array over the two children's widths where
/// it is small ([`DENSE_PAIR_CELLS`]) or no larger than the hash table,
/// else an open-addressing table.
#[expect(clippy::too_many_arguments)]
fn walk_pairs(
    eng: &Engine,
    ws: &mut SparseWorkspace,
    pb: &mut ProbeBuffers,
    f_level: &TddLevel,
    g_level: &TddLevel,
    shape: LevelShape,
    pl: Sides<&[ProductEntry]>,
    level_pairs: u64,
    emit: Emit<'_>,
) -> Result<(), OperationError> {
    let lim = eng.limits();
    let width = shape.g.right;
    let cells = shape.g.left.saturating_mul(width);
    let offsets = (&pb.left_offsets[..], &pb.right_offsets[..]);
    let one_each = one_each(pl.left, shape.f.left) && one_each(pl.right, shape.f.right);
    let g_pairs: usize = pb.g_by_left[..shape.g.left].iter().map(|&n| n as usize).sum();
    let slots = (2 * g_pairs).next_power_of_two().max(16);
    let small = cells <= DENSE_PAIR_CELLS && cells as u64 <= 4 * level_pairs;
    let no_larger = cells.saturating_mul(std::mem::size_of::<u32>()) <= slots * std::mem::size_of::<PairSlot>();
    if (small || no_larger) && !pairs_hashed_forced() {
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
        let pipelined = std::mem::size_of_val(&dense[..]) >= emit.thresholds.pipelined_bytes;
        let walk = BothSides::new(pl, offsets, one_each, pipelined,
            #[inline(always)]
            |c: u32, d: u32| dense[c as usize * width + d as usize],
            #[inline(always)]
            |c: u32, d: u32| prefetch_at(dense, c as usize * width + d as usize));
        walk_level::<false, _>(eng, ws, &walk, f_level, emit)?;
        for (_, pairs) in g_level.internal_inputs_iter() {
            for pair in pairs {
                pb.dense[pair.left.raw() as usize * width + pair.right.raw() as usize] = NO_PRODUCT;
            }
        }
        pb.dense_clean = true;
        return Ok(());
    }
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
    let pipelined = std::mem::size_of_val(&table[..]) >= emit.thresholds.pipelined_bytes;
    let walk = BothSides::new(pl, offsets, one_each, pipelined,
        #[inline(always)]
        |c: u32, d: u32| {
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
        },
        #[inline(always)]
        |c: u32, d: u32| prefetch_at(table, slot_of(pair_key(c, d), mask)));
    walk_level::<false, _>(eng, ws, &walk, f_level, emit)
}

/// What [`walk_pairs`] reads for each `f` pair: both children's product
/// lists and their bucket bounds by `f` index, and `find`, naming the `g`
/// node of a pair of `g`'s children, `NO_PRODUCT` for none. Where both
/// lists hold one product per `f` index, in order ([`one_each`]), a pair's
/// products are read at its children, without bucket bounds, and on a map
/// of at least [`SparseThresholds::pipelined_bytes`] (`pipelined`) the
/// walk asks the cache for a group of pairs' map places (`touch`) before
/// it reads any.
struct BothSides<'a, F, T> {
    pl: Sides<&'a [ProductEntry]>,
    offsets: (&'a [u32], &'a [u32]),
    one_each: bool,
    pipelined: bool,
    find: F,
    touch: T,
}

/// The pairs a pipelined [`BothSides`] walk asks the cache for at once.
const PAIR_GROUP: usize = 16;

impl<'a, F: Fn(u32, u32) -> u32, T: Fn(u32, u32)> BothSides<'a, F, T> {
    fn new(
        pl: Sides<&'a [ProductEntry]>,
        offsets: (&'a [u32], &'a [u32]),
        one_each: bool,
        pipelined: bool,
        find: F,
        touch: T,
    ) -> Self {
        BothSides { pl, offsets, one_each, pipelined, find, touch }
    }

    /// The pipelined walk of a node's pairs, one product a side.
    #[inline(always)]
    fn node_pipelined(
        &self,
        pairs: &[ChildPair],
        push: &mut impl FnMut(ParEntry) -> Result<(), OperationError>,
    ) -> Result<(), OperationError> {
        let mut group = [(0u32, 0u32, 0u32, 0u32); PAIR_GROUP];
        for chunk in pairs.chunks(PAIR_GROUP) {
            for (looked, pair) in group.iter_mut().zip(chunk) {
                let l = &self.pl.left[pair.left.raw() as usize];
                let r = &self.pl.right[pair.right.raw() as usize];
                (self.touch)(l.g_idx.0, r.g_idx.0);
                *looked = (l.g_idx.0, r.g_idx.0, l.prod_idx.0, r.prod_idx.0);
            }
            for &(c, d, left_prod, right_prod) in &group[..chunk.len()] {
                let parent = (self.find)(c, d);
                if parent != NO_PRODUCT {
                    push(ParEntry { g_parent: parent, left_prod, right_prod })?;
                }
            }
        }
        Ok(())
    }
}

impl<F: Fn(u32, u32) -> u32, T: Fn(u32, u32)> NodeWalk for BothSides<'_, F, T> {
    #[inline(always)]
    fn node<const SWAPPED: bool>(
        &self,
        pairs: &[ChildPair],
        ticker: &mut crate::limits::PollGate,
        mut push: impl FnMut(ParEntry) -> Result<(), OperationError>,
    ) -> Result<(), OperationError> {
        if self.one_each && self.pipelined {
            self.node_pipelined(pairs, &mut push)?;
            return ticker.poll(2 * pairs.len() as u64);
        }
        if self.one_each {
            // One product a side: two units a pair, as below.
            for pair in pairs {
                let l = &self.pl.left[pair.left.raw() as usize];
                let r = &self.pl.right[pair.right.raw() as usize];
                let parent = (self.find)(l.g_idx.0, r.g_idx.0);
                if parent != NO_PRODUCT {
                    push(ParEntry { g_parent: parent, left_prod: l.prod_idx.0, right_prod: r.prod_idx.0 })?;
                }
            }
            return ticker.poll(2 * pairs.len() as u64);
        }
        let (lo, ro) = self.offsets;
        let mut work = 0u64;
        for pair in pairs {
            let (a, b) = (pair.left.raw() as usize, pair.right.raw() as usize);
            let lefts = &self.pl.left[lo[a] as usize..lo[a + 1] as usize];
            let rights = &self.pl.right[ro[b] as usize..ro[b + 1] as usize];
            work += (lefts.len() * rights.len()) as u64 + 1;
            for l in lefts {
                for r in rights {
                    let parent = (self.find)(l.g_idx.0, r.g_idx.0);
                    if parent != NO_PRODUCT {
                        push(ParEntry { g_parent: parent, left_prod: l.prod_idx.0, right_prod: r.prod_idx.0 })?;
                    }
                }
            }
        }
        ticker.poll(work)
    }
}

// A test sends every level past the probe, as the oracle it is checked
// against, or takes it wherever it is admissible whatever it prices at, and
// counts the levels each probe built and those it took from f.
#[cfg(test)]
use crate::apply::conjoin::tests::{note_probe, note_settled, pairs_hashed_forced, probe_forced, probe_forced_off};

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
fn note_settled() {}

#[cfg(not(test))]
#[inline(always)]
fn pairs_hashed_forced() -> bool {
    false
}
