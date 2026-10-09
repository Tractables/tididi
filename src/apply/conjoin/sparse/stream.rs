//! Counting a one-product root without building one of its children.
//!
//! [`CandidateFold`] counts the root of a conjunction whose operands have one
//! node each there without storing it, but it reads both of the root's
//! children as built levels. A child can hold far more products than either
//! operand has pairs: under a 4-cycle's root one child has a product for each
//! pair of vertices two steps apart. Here the root is counted from that
//! child's own children, and the child is never built.
//!
//! Write `c` for the child that is not built, `o` for the other one and `cl`,
//! `cr` for `c`'s children; `o`, `cl` and `cr` are built. Choose one operand as
//! the pivot `P` and call the other `Q`. Each node's pairs are disjoint, so the
//! count unfolds over the root's pairs and `c`'s pairs into
//!
//! ```text
//! Σ_p Σ_q L(p, q) · V(p, q)
//! L(p, q) = Σ_{(p_cl, p_cr) ∈ p, (q_cl, q_cr) ∈ q} C_cl(p_cl, q_cl) · C_cr(p_cr, q_cr)
//! V(p, q) = Σ_{(p, p_o) ∈ P's root, (q, q_o) ∈ Q's root} C_o(p_o, q_o)
//! ```
//!
//! over the nodes `p` of `P` and `q` of `Q` at `c`, with `C_x` the count of a
//! product at a built level `x` (0 where it is dead). `L(p, q)` is the count
//! of the product `c` would hold, and `V(p, q)` what the root weighs it by.
//! One `p` at a time, `V(p, ·)` is summed top-down into a stamped column over
//! `Q`'s nodes at `c`, and `L(p, ·)` is walked bottom-up through the live
//! products of `cl` and `cr` and folded against that column as it is found.
//! No product of `c` is stored, so the memory is the operands, the built
//! levels' products and counts, and one column.
//!
//! The work is the walk that finds `c`'s candidates, one per pair of pairs
//! whose child products both live, plus the top-down walk that sums `V`. The
//! second can dominate: when many of `P`'s nodes share a root pair's `o`-side
//! child, that child's products are walked once for each of them, where a
//! built `c` would let the root read them once. [`price`] counts both walks
//! exactly from the index sizes, for either pivot, before anything is built.

use num_bigint::BigUint;

use super::*;
use crate::limits::Limits;

/// One of the two operands.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Operand {
    F,
    G,
}

/// One operand's widths at the levels a streamed count reads.
#[derive(Clone, Copy, Debug)]
pub(crate) struct StreamWidths {
    pub(crate) c: usize,
    pub(crate) o: usize,
    pub(crate) cl: usize,
    pub(crate) cr: usize,
}

/// A built level as a streamed count reads it: its live products and, by
/// product index, their counts, every one of which fits `u64`.
#[derive(Clone, Copy)]
pub(crate) struct Built<'a> {
    pub(crate) products: &'a [ProductEntry],
    pub(crate) counts: &'a [u128],
}

/// Everything a streamed root count reads.
#[derive(Clone, Copy)]
pub(crate) struct StreamInput<'a> {
    /// The operands' levels at the root, one node each.
    pub(crate) f_root: &'a TddLevel,
    pub(crate) g_root: &'a TddLevel,
    /// The operands' levels at `c`.
    pub(crate) f_c: &'a TddLevel,
    pub(crate) g_c: &'a TddLevel,
    /// Whether `c` is the root's left child.
    pub(crate) c_left: bool,
    pub(crate) o: Built<'a>,
    pub(crate) cl: Built<'a>,
    pub(crate) cr: Built<'a>,
    pub(crate) f_widths: StreamWidths,
    pub(crate) g_widths: StreamWidths,
}

/// What a streamed count costs with one pivot, in index entries walked.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct StreamCost {
    pub(crate) pivot: Operand,
    /// The top-down walk that sums `V`.
    pub(crate) weigh: u128,
    /// The bottom-up walk through `c`'s candidates, each pair of `P` at `c`
    /// taking the cheaper of its two directions.
    pub(crate) walk: u128,
}

impl StreamCost {
    /// Both walks.
    pub(crate) fn total(self) -> u128 {
        self.weigh + self.walk
    }
}

/// The input seen from one pivot: `P`'s and `Q`'s levels and widths.
#[derive(Clone, Copy)]
struct Oriented<'a> {
    pivot: Operand,
    p_root: &'a TddLevel,
    q_root: &'a TddLevel,
    p_c: &'a TddLevel,
    q_c: &'a TddLevel,
    p: StreamWidths,
    q: StreamWidths,
    c_left: bool,
}

impl<'a> Oriented<'a> {
    fn new(input: &StreamInput<'a>, pivot: Operand) -> Self {
        let StreamInput { f_root, g_root, f_c, g_c, c_left, f_widths, g_widths, .. } = *input;
        match pivot {
            Operand::F => Oriented { pivot, p_root: f_root, q_root: g_root, p_c: f_c, q_c: g_c, p: f_widths, q: g_widths, c_left },
            Operand::G => Oriented { pivot, p_root: g_root, q_root: f_root, p_c: g_c, q_c: f_c, p: g_widths, q: f_widths, c_left },
        }
    }

    /// A product's `(P, Q)` node indices.
    #[inline(always)]
    fn pq(&self, e: &ProductEntry) -> (u32, u32) {
        match self.pivot {
            Operand::F => (e.f_idx.0, e.g_idx.0),
            Operand::G => (e.g_idx.0, e.f_idx.0),
        }
    }

    /// A root pair's `(c, o)` children.
    #[inline(always)]
    fn at_root(&self, pair: &ChildPair) -> (u32, u32) {
        if self.c_left { (pair.left.0, pair.right.0) } else { (pair.right.0, pair.left.0) }
    }
}

/// A histogram of `keys` over `n` keys, counted as `u32`.
fn histogram(lim: &Limits, n: usize, keys: impl Iterator<Item = u32>) -> Result<Vec<u32>, OperationError> {
    let mut counts = Vec::new();
    lim.try_resize(&mut counts, n, 0u32)?;
    keys.for_each(|k| counts[k as usize] += 1);
    Ok(counts)
}

/// For each `P` node at a built level, its live products and what walking
/// them into `Q`'s pairs reaches: `(products, Σ deg_q[q])` over its products.
fn row_weights(
    lim: &Limits,
    view: &Oriented<'_>,
    products: &[ProductEntry],
    p_width: usize,
    deg_q: &[u32],
) -> Result<(Vec<u32>, Vec<u64>), OperationError> {
    let mut len = Vec::new();
    let mut reach = Vec::new();
    lim.try_resize(&mut len, p_width, 0u32)?;
    lim.try_resize(&mut reach, p_width, 0u64)?;
    for e in products {
        let (p, q) = view.pq(e);
        len[p as usize] += 1;
        reach[p as usize] += u64::from(deg_q[q as usize]);
    }
    Ok((len, reach))
}

/// How `c`'s candidates are found from one pair `(p_cl, p_cr)` of `P`: walk
/// the live products of one child and probe the other child's.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Direction {
    /// Walk `cl`'s products of `p_cl` into `Q`'s pairs by their `cl` child,
    /// probing `cr`.
    ByLeft,
    /// Walk `cr`'s products of `p_cr` into `Q`'s pairs by their `cr` child,
    /// probing `cl`.
    ByRight,
}

/// The per-pair pricing both [`price`] and the count read.
struct PairPricing {
    cl_len: Vec<u32>,
    cl_reach: Vec<u64>,
    cr_len: Vec<u32>,
    cr_reach: Vec<u64>,
    /// Whether a probe of `cl` (`cr`) is one lookup, one operand having one
    /// node there, so no row is marked for it.
    cl_direct: bool,
    cr_direct: bool,
}

impl PairPricing {
    fn new(lim: &Limits, view: &Oriented<'_>, input: &StreamInput<'_>) -> Result<Self, OperationError> {
        let q_by_cl = histogram(lim, view.q.cl, view.q_c.pairs_with_parent().map(|(_, pair)| pair.left.0))?;
        let q_by_cr = histogram(lim, view.q.cr, view.q_c.pairs_with_parent().map(|(_, pair)| pair.right.0))?;
        let (cl_len, cl_reach) = row_weights(lim, view, input.cl.products, view.p.cl, &q_by_cl)?;
        let (cr_len, cr_reach) = row_weights(lim, view, input.cr.products, view.p.cr, &q_by_cr)?;
        Ok(PairPricing {
            cl_len, cl_reach, cr_len, cr_reach,
            cl_direct: view.p.cl == 1 || view.q.cl == 1,
            cr_direct: view.p.cr == 1 || view.q.cr == 1,
        })
    }

    /// The cheaper direction for the pair `(p_cl, p_cr)`, and its cost.
    #[inline]
    fn choose(&self, p_cl: u32, p_cr: u32) -> (Direction, u128) {
        let (l, r) = (p_cl as usize, p_cr as usize);
        let mark = |direct: bool, len: u32| if direct { 0 } else { u128::from(len) };
        let by_left = 1 + mark(self.cr_direct, self.cr_len[r]) + u128::from(self.cl_reach[l]);
        let by_right = 1 + mark(self.cl_direct, self.cl_len[l]) + u128::from(self.cr_reach[r]);
        if by_left <= by_right { (Direction::ByLeft, by_left) } else { (Direction::ByRight, by_right) }
    }
}

/// The exact cost of a streamed count with `pivot`: see [`StreamCost`].
///
/// # Errors
///
/// [`OperationError::OverBudget`] when a counter array is refused.
pub(crate) fn price(lim: &Limits, input: &StreamInput<'_>, pivot: Operand) -> Result<StreamCost, OperationError> {
    let view = Oriented::new(input, pivot);
    // The weighing: each root pair of `P` walks the live products of its
    // `o`-side child into `Q`'s root pairs by their `o`-side child.
    let q_by_o = histogram(lim, view.q.o, view.q_root.pairs_with_parent().map(|(_, pair)| view.at_root(&pair).1))?;
    let (_, o_reach) = row_weights(lim, &view, input.o.products, view.p.o, &q_by_o)?;
    let weigh = view.p_root.pairs_with_parent()
        .map(|(_, pair)| 1 + u128::from(o_reach[view.at_root(&pair).1 as usize]))
        .sum();
    let pricing = PairPricing::new(lim, &view, input)?;
    let walk = view.p_c.pairs_with_parent()
        .map(|(_, pair)| pricing.choose(pair.left.0, pair.right.0).1)
        .sum();
    Ok(StreamCost { pivot, weigh, walk })
}

/// How much longer than the walk through `c`'s candidates the weighing may
/// be for a streamed count to be taken over building `c`.
///
/// Building `c` walks its candidates too, stores each as a pair, and folds
/// them again for the root, so a stream that weighs no more than this factor
/// times its walk does no more work than the build and holds none of it.
const WEIGH_FACTOR: u128 = 2;

/// The pivot to stream the root with, or `None` when building `c` is the
/// better choice: the cheaper pivot by [`price`], taken when its weighing is
/// within [`WEIGH_FACTOR`] of its walk.
///
/// # Errors
///
/// [`OperationError::OverBudget`] when a counter array is refused.
pub(crate) fn choose_pivot(lim: &Limits, input: &StreamInput<'_>) -> Result<Option<Operand>, OperationError> {
    let by_f = price(lim, input, Operand::F)?;
    let by_g = price(lim, input, Operand::G)?;
    let best = if by_g.total() < by_f.total() { by_g } else { by_f };
    Ok(match forced() {
        Some(forced) => forced,
        None => (best.weigh <= WEIGH_FACTOR * best.walk).then_some(best.pivot),
    })
}

// Tests pin the choice; production always prices it.
#[cfg(test)]
use super::tests::forced_stream as forced;

#[cfg(not(test))]
fn forced() -> Option<Option<Operand>> {
    None
}

/// A bound on the candidates building a level would find: the fewest steps
/// any of the scatter's walks takes, each pair of one operand at the level
/// stepping through the live products of one child that share its node
/// there. Only compares levels; see [`estimate_scatter_direction`] for the
/// walks.
///
/// # Errors
///
/// [`OperationError::OverBudget`] when a counter array is refused.
pub(crate) fn candidate_bound(
    lim: &Limits,
    f: &TddLevel,
    g: &TddLevel,
    left: &[ProductEntry],
    right: &[ProductEntry],
    f_widths: StreamWidths,
    g_widths: StreamWidths,
) -> Result<u128, OperationError> {
    // The pairs one operand has at `c` for each of its nodes at one child,
    // summed over that child's live products.
    let walk = |operand: Operand, on_left: bool| {
        let (level, widths) = match operand { Operand::F => (f, f_widths), Operand::G => (g, g_widths) };
        let (width, products) = if on_left { (widths.cl, left) } else { (widths.cr, right) };
        let side = level.pairs_with_parent().map(|(_, pair)| if on_left { pair.left.0 } else { pair.right.0 });
        let counts = histogram(lim, width, side)?;
        let node = |e: &ProductEntry| match operand { Operand::F => e.f_idx.0, Operand::G => e.g_idx.0 };
        Ok::<u128, OperationError>(products.iter().map(|e| u128::from(counts[node(e) as usize])).sum())
    };
    Ok(walk(Operand::F, true)?
        .min(walk(Operand::F, false)?)
        .min(walk(Operand::G, true)?)
        .min(walk(Operand::G, false)?))
}

/// A column over one operand's nodes whose entries belong to the current
/// round only: starting a round empties it without touching it. Each entry
/// keeps its stamp beside its value, so a read or a write is one access.
struct Stamped<T> {
    slots: Vec<(u32, T)>,
    round: u32,
}

impl<T: Copy + Default> Stamped<T> {
    fn new(lim: &Limits, n: usize) -> Result<Self, OperationError> {
        let mut slots = Vec::new();
        lim.try_resize(&mut slots, n, (0u32, T::default()))?;
        Ok(Stamped { slots, round: 0 })
    }

    /// Empty the column.
    #[inline]
    fn begin(&mut self) {
        self.round = self.round.wrapping_add(1);
        if self.round == 0 {
            // The stamp wrapped, so an entry of an older round could read as
            // this one's. One pass per 2^32 rounds.
            self.slots.fill((0, T::default()));
            self.round = 1;
        }
    }

    #[inline(always)]
    fn get(&self, k: u32) -> Option<T> {
        let (stamp, value) = self.slots[k as usize];
        (stamp == self.round).then_some(value)
    }

    #[inline(always)]
    fn set(&mut self, k: u32, v: T) {
        self.slots[k as usize] = (self.round, v);
    }
}

/// How a pair's walk looks up the other child's product and its count: in a
/// table over one side's nodes when the other operand has one node there,
/// and otherwise in the row of the pair's own node, marked afresh for each
/// pair with what each of its products carries (`M`: the product's index
/// for the indirect walk, its count for the dense walk).
///
/// A table holds the count itself, 0 where no product lives, so the walk
/// reads one entry of a table as wide as a level rather than a product index
/// and then that product's count, in a column as long as the level's
/// products and read at random. A product whose count is 0 adds nothing
/// either way.
enum Probe<M> {
    /// `Q` has one node: the count of `p`'s product is `table[p]`.
    ByP(Vec<u64>),
    /// `P` has one node: the count of `q`'s product is `table[q]`.
    ByQ(Vec<u64>),
    /// The current row, marked by `Q` node with its products.
    Marked(Stamped<M>),
}

impl<M: Copy + Default> Probe<M> {
    fn new(lim: &Limits, view: &Oriented<'_>, built: Built<'_>, p_width: usize, q_width: usize) -> Result<Self, OperationError> {
        let table = |n: usize, key: &dyn Fn(u32, u32) -> u32| -> Result<Vec<u64>, OperationError> {
            let mut table = Vec::new();
            lim.try_resize(&mut table, n, 0u64)?;
            for e in built.products {
                let (p, q) = view.pq(e);
                // Every count the stream reads fits `u64` (see `Built`).
                let count = built.counts[e.prod_idx.0 as usize];
                debug_assert!(u64::try_from(count).is_ok());
                table[key(p, q) as usize] = count as u64;
            }
            Ok(table)
        };
        if q_width == 1 {
            Ok(Probe::ByP(table(p_width, &|p, _| p)?))
        } else if p_width == 1 {
            Ok(Probe::ByQ(table(q_width, &|_, q| q)?))
        } else {
            Ok(Probe::Marked(Stamped::new(lim, q_width)?))
        }
    }

    /// Make `row`, the products of the pair's own node, the one probed.
    #[inline]
    fn open(&mut self, row: &[(u32, M)]) {
        if let Probe::Marked(marks) = self {
            marks.begin();
            for &(q, mark) in row {
                marks.set(q, mark);
            }
        }
    }

    /// The count of `(p, q)`'s product in the opened row, `read` taking a
    /// mark to its count, if the product lives (a table's may read as
    /// absent where its count is 0).
    #[inline(always)]
    fn count(&self, p: u32, q: u32, read: impl Fn(M) -> u128) -> Option<u128> {
        let count = match self {
            Probe::ByP(table) => table[p as usize],
            Probe::ByQ(table) => table[q as usize],
            Probe::Marked(marks) => return marks.get(q).map(read),
        };
        (count != 0).then_some(u128::from(count))
    }
}

/// A running count: `u128` until it overflows, then exact.
#[derive(Default)]
struct Total {
    fast: u128,
    big: BigUint,
}

impl Total {
    /// Add `a · b · v`, `a` and `b` below `2^64`.
    #[inline(always)]
    fn add3(&mut self, a: u128, b: u128, v: u128) {
        match (a * b).checked_mul(v) {
            Some(x) => match self.fast.checked_add(x) {
                Some(sum) => self.fast = sum,
                None => {
                    self.big += self.fast;
                    self.fast = x;
                }
            },
            None => self.big += BigUint::from(a) * b * v,
        }
    }

    fn finish(self) -> BigUint {
        self.big + self.fast
    }
}

/// `items` grouped by the key `item` gives each, over `n` keys.
fn grouped<S, T: Copy + Default, I: Iterator<Item = S> + Clone>(
    lim: &Limits,
    n: usize,
    items: I,
    item: impl Fn(S) -> (usize, T),
) -> Result<Grouped<T>, OperationError> {
    let mut out = Grouped::default();
    counting_sort(lim, n, items, item, None, T::default(), &mut out)?;
    Ok(out)
}

/// A built level's live products by their `P` node, each holding its `Q`
/// node and its product index.
fn rows_by_p(
    lim: &Limits,
    view: &Oriented<'_>,
    products: &[ProductEntry],
    p_width: usize,
) -> Result<Grouped<(u32, u32)>, OperationError> {
    grouped(lim, p_width, products.iter(), |e| {
        let (p, q) = view.pq(e);
        (p as usize, (q, e.prod_idx.0))
    })
}

/// Count the root with `pivot`, never building `c`: see the module docs.
///
/// Every count in `input`'s columns must fit `u64`.
///
/// # Errors
///
/// [`OperationError::OverBudget`] when an index or column is refused, and the
/// stop the engine's limits install, polled as the walks go.
pub(crate) fn count(eng: &Engine, input: &StreamInput<'_>, pivot: Operand) -> Result<BigUint, OperationError> {
    let lim = eng.limits();
    let view = Oriented::new(input, pivot);
    let pricing = PairPricing::new(lim, &view, input)?;
    match choose_walk(input) {
        Walk::Indirect => count_indirect(eng, input, &view, &pricing),
        Walk::Dense { wide: false, fold } => count_dense::<u32>(eng, input, &view, &pricing, fold),
        Walk::Dense { wide: true, fold } => count_dense::<u64>(eng, input, &view, &pricing, fold),
    }
}

/// [`count`] with `V(p, ·)` in a stamped `u128` column and every candidate
/// found through its product's row and the `Q` pairs that own it.
fn count_indirect(
    eng: &Engine,
    input: &StreamInput<'_>,
    view: &Oriented<'_>,
    pricing: &PairPricing,
) -> Result<BigUint, OperationError> {
    let lim = eng.limits();
    let view = *view;
    let (p, q) = (view.p, view.q);

    // The top-down side: `P`'s root pairs by their `c` child, the live
    // products of `o` by their `P` node, and `Q`'s root pairs by their `o`
    // child.
    let p_root = grouped(lim, p.c, view.p_root.pairs_with_parent(), |(_, pair)| {
        let (at_c, at_o) = view.at_root(&pair);
        (at_c as usize, at_o)
    })?;
    let q_root = grouped(lim, q.o, view.q_root.pairs_with_parent(), |(_, pair)| {
        let (at_c, at_o) = view.at_root(&pair);
        (at_o as usize, at_c)
    })?;
    let o_rows = rows_by_p(lim, &view, input.o.products, p.o)?;
    // The bottom-up side: both children's live products by their `P` node,
    // and `Q`'s pairs at `c` by each child, holding the node and the other
    // child.
    let cl_rows = rows_by_p(lim, &view, input.cl.products, p.cl)?;
    let cr_rows = rows_by_p(lim, &view, input.cr.products, p.cr)?;
    let q_by_cl = grouped(lim, q.cl, view.q_c.pairs_with_parent(), |(node, pair)| (pair.left.0 as usize, (node, pair.right.0)))?;
    let q_by_cr = grouped(lim, q.cr, view.q_c.pairs_with_parent(), |(node, pair)| (pair.right.0 as usize, (node, pair.left.0)))?;
    let mut probe_cl: Probe<u32> = Probe::new(lim, &view, input.cl, p.cl, q.cl)?;
    let mut probe_cr: Probe<u32> = Probe::new(lim, &view, input.cr, p.cr, q.cr)?;

    let (col_o, col_cl, col_cr) = (input.o.counts, input.cl.counts, input.cr.counts);
    let mut weights: Stamped<u128> = Stamped::new(lim, q.c)?;
    let mut total = Total::default();
    let mut ticker = lim.gate_with(super::super::budget::APPLY_POLL_STRIDE);
    for (p_node, node) in view.p_c.nodes().iter().enumerate() {
        // `V(p, ·)`.
        weights.begin();
        let (mut weighed, mut any) = (0u64, false);
        for &p_o in p_root.view().bucket(p_node) {
            for &(q_o, prod) in o_rows.view().bucket(p_o as usize) {
                let c_o = col_o[prod as usize];
                let owners = q_root.view().bucket(q_o as usize);
                for &q_node in owners {
                    let v = weights.get(q_node).unwrap_or(0);
                    weights.set(q_node, v + c_o);
                }
                weighed += 1 + owners.len() as u64;
                any |= !owners.is_empty();
            }
        }
        ticker.poll(weighed)?;
        if !any {
            continue;
        }
        // `L(p, ·)`, folded against `V(p, ·)` candidate by candidate.
        for pair in view.p_c.pairs_iter_of(&node) {
            let (p_cl, p_cr) = (pair.left.0, pair.right.0);
            let walked = match pricing.choose(p_cl, p_cr).0 {
                Direction::ByLeft => {
                    let cr_row = cr_rows.view().bucket(p_cr as usize);
                    probe_cr.open(cr_row);
                    let mut walked = cr_row.len() as u64;
                    for &(q_cl, prod_l) in cl_rows.view().bucket(p_cl as usize) {
                        let c_l = col_cl[prod_l as usize];
                        let owners = q_by_cl.view().bucket(q_cl as usize);
                        walked += 1 + owners.len() as u64;
                        for &(q_node, q_cr) in owners {
                            let Some(v) = weights.get(q_node) else { continue };
                            let Some(c_r) = probe_cr.count(p_cr, q_cr, |prod| col_cr[prod as usize]) else { continue };
                            total.add3(c_l, c_r, v);
                        }
                    }
                    walked
                }
                Direction::ByRight => {
                    let cl_row = cl_rows.view().bucket(p_cl as usize);
                    probe_cl.open(cl_row);
                    let mut walked = cl_row.len() as u64;
                    for &(q_cr, prod_r) in cr_rows.view().bucket(p_cr as usize) {
                        let c_r = col_cr[prod_r as usize];
                        let owners = q_by_cr.view().bucket(q_cr as usize);
                        walked += 1 + owners.len() as u64;
                        for &(q_node, q_cl) in owners {
                            let Some(v) = weights.get(q_node) else { continue };
                            let Some(c_l) = probe_cl.count(p_cl, q_cl, |prod| col_cl[prod as usize]) else { continue };
                            total.add3(c_l, c_r, v);
                        }
                    }
                    walked
                }
            };
            ticker.poll(walked)?;
        }
    }
    ticker.flush()?;
    Ok(total.finish())
}

/// How a streamed count holds `V(p, ·)` and reads the probed child's count.
///
/// Every walk finds the same candidates in the same order: per pair of `P`
/// at `c`, the walked child's products of its node, and per product the `Q`
/// pairs that own it. The indirect walk keeps a stamped `u128` slot of `V`
/// per `Q` node and reads each product's count from its level's column,
/// where the product's index points. The dense walk keeps `V` in a plain
/// integer, several times smaller, so far more of it stays in cache, and
/// carries each product's count in its row, so no column is read at random.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Walk {
    /// `V` in a stamped `u128` column.
    Indirect,
    /// `V` in a column of `u64` (`wide`) or `u32`, emptied by walking back
    /// over what the round wrote. With `fold`, a probe that is a table is
    /// read into the walked side's owners before the walk (see [`Owners`]).
    Dense { wide: bool, fold: bool },
}

/// The walk a streamed count takes: dense when `V` fits a plain integer.
/// `V(p, q)` sums `C_o` over pairs `(p_o, q_o)` that are distinct for one
/// `(p, q)` (a node's pairs are distinct, and `p` and `q` fix the `c` side of
/// each), so the sum of every count at `o` bounds every entry.
fn choose_walk(input: &StreamInput<'_>) -> Walk {
    let o_total = input.o.products.iter()
        .fold(0u128, |sum, e| sum.saturating_add(input.o.counts[e.prod_idx.0 as usize]));
    let wide = if o_total <= u128::from(u32::MAX) {
        false
    } else if o_total <= u128::from(u64::MAX) {
        true
    } else {
        return Walk::Indirect;
    };
    match forced_walk() {
        Some(Walk::Indirect) => Walk::Indirect,
        Some(Walk::Dense { wide: forced_wide, fold }) => Walk::Dense { wide: wide || forced_wide, fold },
        None => Walk::Dense { wide, fold: true },
    }
}

// Tests pin the walk; production always chooses it.
#[cfg(test)]
use super::tests::forced_walk;

#[cfg(not(test))]
fn forced_walk() -> Option<Walk> {
    None
}

/// An entry of the dense walk's `V` column.
trait Weight: Copy + Default {
    /// `c`, which the caller has bounded by the type's maximum.
    fn from_count(c: u128) -> Self;
    /// `self + c`, which the same bound keeps in range.
    fn plus(self, c: Self) -> Self;
    fn wide(self) -> u128;
}

impl Weight for u32 {
    #[inline(always)]
    fn from_count(c: u128) -> Self {
        c as u32
    }
    #[inline(always)]
    fn plus(self, c: Self) -> Self {
        self + c
    }
    #[inline(always)]
    fn wide(self) -> u128 {
        u128::from(self)
    }
}

impl Weight for u64 {
    #[inline(always)]
    fn from_count(c: u128) -> Self {
        c as u64
    }
    #[inline(always)]
    fn plus(self, c: Self) -> Self {
        self + c
    }
    #[inline(always)]
    fn wide(self) -> u128 {
        u128::from(self)
    }
}

/// How far ahead of its read the dense walk asks the cache for an entry of
/// `V`: the walk's reads are independent, so the only limit on how many
/// misses are in flight is how early each is asked for.
const AHEAD: usize = 16;

/// Ask the cache for `column[k]`. A hint only; no-op off `x86_64` and under
/// Miri, which lacks the intrinsic.
#[inline(always)]
fn prefetch<T>(column: &[T], k: usize) {
    #[cfg(all(target_arch = "x86_64", not(miri)))]
    {
        let at = column.as_ptr().wrapping_add(k).cast::<i8>();
        // Sound whatever the address: a prefetch reads nothing the program
        // sees and never faults.
        unsafe { core::arch::x86_64::_mm_prefetch(at, core::arch::x86_64::_MM_HINT_T0) };
    }
    #[cfg(not(all(target_arch = "x86_64", not(miri))))]
    let _ = (column, k);
}

/// `Q`'s pairs at `c` by the child a walk starts from, as that walk reads
/// them, each with the pair's node.
///
/// A candidate's probed count `C(p_o, q_o)`, `p_o` and `q_o` the other
/// child's nodes in the pairs of `P` and `Q`, is a table's entry when one
/// operand has one node there, and then it is a factor of one pair alone:
/// `table[p_o]` when `Q` has one node, a factor of the pair of `P`, and
/// `table[q_o]` when `P` has one, a factor of the pair of `Q`. Folded, each
/// pair of `Q` carries its factor, 1 in the first case, and is left out
/// where it is 0; the walk then reads no table per candidate, and sums
/// `V · factor` over a product's owners before multiplying by the product's
/// count and the pair of `P`'s factor. That is the same sum over the same
/// candidates, less those whose term is 0, grouped by product.
enum Owners {
    /// No walk starts from this child.
    Unread,
    /// Each with the pair's other child, for the probe to look up.
    Probed(Grouped<(u32, u32)>),
    /// Each with the pair's factor of the probed count, below `2^32`.
    Folded(Grouped<(u32, u32)>),
}

impl Owners {
    /// The owners a walk from the left (`left`) or right child reads, the
    /// other child's `probe` folded in where `fold` allows it: a table
    /// whose `Q` factors all fit `u32`.
    fn new(
        lim: &Limits,
        view: &Oriented<'_>,
        left: bool,
        q_width: usize,
        probe: &Probe<u64>,
        fold: bool,
    ) -> Result<Self, OperationError> {
        let pairs = view.q_c.pairs_with_parent().map(move |(node, pair)| {
            let (this, other) = if left { (pair.left.0, pair.right.0) } else { (pair.right.0, pair.left.0) };
            (this, node, other)
        });
        let by_q = match probe {
            Probe::ByQ(table) if fold && table.iter().all(|&c| c <= u64::from(u32::MAX)) => Some(table.as_slice()),
            _ => None,
        };
        #[cfg(test)]
        super::tests::note_owners(match (by_q.is_some(), fold, probe) {
            (true, _, _) => 1,
            (false, true, Probe::ByP(_)) => 0,
            (false, _, Probe::ByP(_)) => 2,
            (false, _, Probe::ByQ(_)) => 3,
            (false, _, Probe::Marked(_)) => 4,
        });
        if let Some(table) = by_q {
            let live = pairs.filter(|&(_, _, other)| table[other as usize] != 0);
            let owners = grouped(lim, q_width, live, |(this, node, other)| {
                (this as usize, (node, table[other as usize] as u32))
            })?;
            Ok(Owners::Folded(owners))
        } else if fold && matches!(probe, Probe::ByP(_)) {
            Ok(Owners::Folded(grouped(lim, q_width, pairs, |(this, node, _)| (this as usize, (node, 1)))?))
        } else {
            Ok(Owners::Probed(grouped(lim, q_width, pairs, |(this, node, other)| (this as usize, (node, other)))?))
        }
    }
}

/// One child's side of the dense walk: whatever its pairs' walks and the
/// other side's probes read.
struct DenseSide {
    /// The products by their `P` node, with their `Q` node and their count,
    /// those whose count is 0 left out: what a walk steps through and a
    /// marked probe opens.
    rows: Grouped<(u32, u64)>,
    owners: Owners,
}

impl DenseSide {
    /// `rows` says whether anything reads the product rows.
    fn new(
        lim: &Limits,
        view: &Oriented<'_>,
        built: Built<'_>,
        p_width: usize,
        rows: bool,
        owners: Owners,
    ) -> Result<Self, OperationError> {
        let rows = if rows {
            let live = built.products.iter().filter(|e| built.counts[e.prod_idx.0 as usize] != 0);
            grouped(lim, p_width, live, |e| {
                let (p, q) = view.pq(e);
                // Every count the stream reads fits `u64` (see `Built`).
                (p as usize, (q, built.counts[e.prod_idx.0 as usize] as u64))
            })?
        } else {
            Grouped::default()
        };
        Ok(DenseSide { rows, owners })
    }

    /// The products of `P` node `p`, empty when no walk or probe reads them
    /// (a probe that reads them always has them).
    #[inline]
    fn row(&self, p: u32) -> &[(u32, u64)] {
        if self.rows.offsets.is_empty() { &[] } else { self.rows.view().bucket(p as usize) }
    }
}

/// Add a pair's candidates to `total` with the probe folded into `owners`:
/// `row` holds the walked child's products of the pair's `P` node, `factor`
/// is the pair's `P` factor of the probed count.
#[inline(always)]
fn walk_folded<W: Weight>(
    row: &[(u32, u64)],
    owners: GroupedView<'_, (u32, u32)>,
    factor: u64,
    weights: &[W],
    total: &mut Total,
) -> u64 {
    let mut walked = 0u64;
    for (j, &(q_this, c_this)) in row.iter().enumerate() {
        if let Some(&(next, _)) = row.get(j + 1) {
            owners.prefetch_bucket(next as usize);
        }
        let bucket = owners.bucket(q_this as usize);
        walked += 1 + bucket.len() as u64;
        // Each term is below `2^96` and a bucket holds fewer than `2^32`, so
        // the sum stays below `2^128`.
        let mut sum = 0u128;
        for (i, &(q_node, k)) in bucket.iter().enumerate() {
            if let Some(&(ahead, _)) = bucket.get(i + AHEAD) {
                prefetch(weights, ahead as usize);
            }
            sum += weights[q_node as usize].wide() * u128::from(k);
        }
        total.add3(u128::from(c_this), u128::from(factor), sum);
    }
    walked
}

/// Add a pair's candidates to `total` through the open `probe`: `row` holds
/// the walked child's products of the pair's `P` node, `other` is the pair's
/// probed child.
#[inline(always)]
fn walk_probed<W: Weight>(
    row: &[(u32, u64)],
    owners: GroupedView<'_, (u32, u32)>,
    other: u32,
    probe: &Probe<u64>,
    weights: &[W],
    total: &mut Total,
) -> u64 {
    let mut walked = 0u64;
    for &(q_this, c_this) in row {
        let owners = owners.bucket(q_this as usize);
        walked += 1 + owners.len() as u64;
        for &(q_node, q_other) in owners {
            let v = weights[q_node as usize].wide();
            if v == 0 {
                continue;
            }
            let Some(c_other) = probe.count(other, q_other, u128::from) else { continue };
            total.add3(u128::from(c_this), c_other, v);
        }
    }
    walked
}

/// Add `L(p, ·)`'s candidates from the pair with `this` at the walked child
/// and `other` at the probed one to `total`, against `V(p, ·)` in `weights`.
#[inline(always)]
fn walk_pair<W: Weight>(
    walked: &DenseSide,
    probed: &DenseSide,
    this: u32,
    other: u32,
    probe: &mut Probe<u64>,
    weights: &[W],
    total: &mut Total,
) -> u64 {
    match &walked.owners {
        Owners::Folded(owners) => {
            let factor = match probe {
                Probe::ByP(table) => table[other as usize],
                _ => 1,
            };
            if factor == 0 {
                return 1;
            }
            walk_folded(walked.row(this), owners.view(), factor, weights, total)
        }
        Owners::Probed(owners) => {
            let row = probed.row(other);
            probe.open(row);
            row.len() as u64 + walk_probed(walked.row(this), owners.view(), other, probe, weights, total)
        }
        Owners::Unread => unreachable!("a child a pair walks from has its owners"),
    }
}

/// Ask the cache for what the weighing reads a few of `tops` from now: the
/// `o` rows of `tops[j + 2]`, and the `Q` root pairs that own the first
/// product of `tops[j + 1]`, whose row the last call asked for. Each is at a
/// random place, and a run's start is a miss the walk would otherwise wait
/// on.
#[inline(always)]
fn prefetch_rows<W: Copy>(tops: &[u32], j: usize, o_rows: GroupedView<'_, (u32, W)>, q_root: GroupedView<'_, u32>) {
    if let Some(&ahead) = tops.get(j + 2) {
        o_rows.prefetch_bucket(ahead as usize);
    }
    if let Some(&(q_o, _)) = tops.get(j + 1).and_then(|&next| o_rows.bucket(next as usize).first()) {
        q_root.prefetch_bucket(q_o as usize);
    }
}

/// [`count`] with `V(p, ·)` in a dense column of `W`; see [`Walk`]. The sum
/// is the indirect walk's over the same candidates: only where each term's
/// factors are read from differs, and, folded, how the terms are grouped
/// (see [`Owners`]).
fn count_dense<W: Weight>(
    eng: &Engine,
    input: &StreamInput<'_>,
    view: &Oriented<'_>,
    pricing: &PairPricing,
    fold: bool,
) -> Result<BigUint, OperationError> {
    let lim = eng.limits();
    let view = *view;
    let (p, q) = (view.p, view.q);

    // The top-down side, as the indirect walk has it, with each `o`
    // product's count beside its `Q` node.
    let p_root = grouped(lim, p.c, view.p_root.pairs_with_parent(), |(_, pair)| {
        let (at_c, at_o) = view.at_root(&pair);
        (at_c as usize, at_o)
    })?;
    let q_root = grouped(lim, q.o, view.q_root.pairs_with_parent(), |(_, pair)| {
        let (at_c, at_o) = view.at_root(&pair);
        (at_o as usize, at_c)
    })?;
    let col_o = input.o.counts;
    let o_rows = grouped(
        lim, p.o,
        input.o.products.iter().filter(|e| col_o[e.prod_idx.0 as usize] != 0),
        |e| {
            let (pn, qn) = view.pq(e);
            (pn as usize, (qn, W::from_count(col_o[e.prod_idx.0 as usize])))
        },
    )?;

    // Which walks and probes run decides what each side builds.
    let (mut by_left, mut by_right) = (false, false);
    view.p_c.pairs_with_parent().for_each(|(_, pair)| match pricing.choose(pair.left.0, pair.right.0).0 {
        Direction::ByLeft => by_left = true,
        Direction::ByRight => by_right = true,
    });
    let mut probe_cl: Probe<u64> = Probe::new(lim, &view, input.cl, p.cl, q.cl)?;
    let mut probe_cr: Probe<u64> = Probe::new(lim, &view, input.cr, p.cr, q.cr)?;
    let owners = |walked: bool, left: bool, q_width: usize, probe: &Probe<u64>| {
        if walked { Owners::new(lim, &view, left, q_width, probe, fold) } else { Ok(Owners::Unread) }
    };
    let (left_owners, right_owners) = (owners(by_left, true, q.cl, &probe_cr)?, owners(by_right, false, q.cr, &probe_cl)?);
    // A side's rows are read by its walks, and opened by the other side's
    // walks when its probe is marked and not folded.
    let opens = |owners: &Owners| matches!(owners, Owners::Probed(_));
    let (open_cl, open_cr) = (opens(&right_owners) && matches!(probe_cl, Probe::Marked(_)), opens(&left_owners) && matches!(probe_cr, Probe::Marked(_)));
    let left = DenseSide::new(lim, &view, input.cl, p.cl, by_left || open_cl, left_owners)?;
    let right = DenseSide::new(lim, &view, input.cr, p.cr, by_right || open_cr, right_owners)?;

    let mut weights: Vec<W> = Vec::new();
    lim.try_resize(&mut weights, q.c, W::default())?;
    let mut total = Total::default();
    let mut ticker = lim.gate_with(super::super::budget::APPLY_POLL_STRIDE);
    for (p_node, node) in view.p_c.nodes().iter().enumerate() {
        // `V(p, ·)`.
        let (mut weighed, mut any) = (0u64, false);
        let tops = p_root.view().bucket(p_node);
        for (j, &p_o) in tops.iter().enumerate() {
            prefetch_rows(tops, j, o_rows.view(), q_root.view());
            for &(q_o, c_o) in o_rows.view().bucket(p_o as usize) {
                let owners = q_root.view().bucket(q_o as usize);
                for (i, &q_node) in owners.iter().enumerate() {
                    if let Some(&ahead) = owners.get(i + AHEAD) {
                        prefetch(&weights, ahead as usize);
                    }
                    let slot = &mut weights[q_node as usize];
                    *slot = slot.plus(c_o);
                }
                weighed += 1 + owners.len() as u64;
                any |= !owners.is_empty();
            }
        }
        ticker.poll(weighed)?;
        if !any {
            continue;
        }
        // `L(p, ·)`, folded against `V(p, ·)` candidate by candidate.
        for pair in view.p_c.pairs_iter_of(&node) {
            let (p_cl, p_cr) = (pair.left.0, pair.right.0);
            let walked = match pricing.choose(p_cl, p_cr).0 {
                Direction::ByLeft => walk_pair(&left, &right, p_cl, p_cr, &mut probe_cr, &weights, &mut total),
                Direction::ByRight => walk_pair(&right, &left, p_cr, p_cl, &mut probe_cl, &weights, &mut total),
            };
            ticker.poll(walked)?;
        }
        // Empty `V(p, ·)` by walking back over what the round wrote.
        for (j, &p_o) in tops.iter().enumerate() {
            prefetch_rows(tops, j, o_rows.view(), q_root.view());
            for &(q_o, _) in o_rows.view().bucket(p_o as usize) {
                for &q_node in q_root.view().bucket(q_o as usize) {
                    weights[q_node as usize] = W::default();
                }
            }
        }
    }
    ticker.flush()?;
    Ok(total.finish())
}
