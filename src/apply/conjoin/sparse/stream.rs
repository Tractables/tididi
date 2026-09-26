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

/// Every pair of a level with the index of the node holding it.
fn pairs_with_parent(level: &TddLevel) -> impl Iterator<Item = (u32, ChildPair)> + Clone + '_ {
    level.nodes.iter().enumerate()
        .flat_map(|(parent, node)| level.pairs_of(node).iter().map(move |&pair| (parent as u32, pair)))
}

/// A histogram of `keys` over `n` keys, counted as `u32`.
fn histogram(lim: &Limits, n: usize, keys: impl Iterator<Item = u32>) -> Result<Vec<u32>, OperationError> {
    let mut counts = Vec::new();
    lim.try_resize(&mut counts, n, 0u32)?;
    for k in keys {
        counts[k as usize] += 1;
    }
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
        let q_by_cl = histogram(lim, view.q.cl, pairs_with_parent(view.q_c).map(|(_, pair)| pair.left.0))?;
        let q_by_cr = histogram(lim, view.q.cr, pairs_with_parent(view.q_c).map(|(_, pair)| pair.right.0))?;
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
    let q_by_o = histogram(lim, view.q.o, pairs_with_parent(view.q_root).map(|(_, pair)| view.at_root(&pair).1))?;
    let (_, o_reach) = row_weights(lim, &view, input.o.products, view.p.o, &q_by_o)?;
    let weigh = pairs_with_parent(view.p_root)
        .map(|(_, pair)| 1 + u128::from(o_reach[view.at_root(&pair).1 as usize]))
        .sum();
    let pricing = PairPricing::new(lim, &view, input)?;
    let walk = pairs_with_parent(view.p_c)
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
        let side = pairs_with_parent(level).map(|(_, pair)| if on_left { pair.left.0 } else { pair.right.0 });
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

/// How a pair's walk looks up the other child's product: in a table over
/// one side's nodes when the other operand has one node there, and otherwise
/// in the row of the pair's own node, marked afresh for each pair.
enum Probe {
    /// `Q` has one node: the product of `p` is `table[p]`.
    ByP(Vec<u32>),
    /// `P` has one node: the product of `q` is `table[q]`.
    ByQ(Vec<u32>),
    /// The current row, marked by `Q` node.
    Marked(Stamped<u32>),
}

impl Probe {
    fn new(lim: &Limits, view: &Oriented<'_>, products: &[ProductEntry], p_width: usize, q_width: usize) -> Result<Self, OperationError> {
        let table = |n: usize, key: &dyn Fn(u32, u32) -> u32| -> Result<Vec<u32>, OperationError> {
            let mut table = Vec::new();
            lim.try_resize(&mut table, n, NO_PRODUCT)?;
            for e in products {
                let (p, q) = view.pq(e);
                table[key(p, q) as usize] = e.prod_idx.0;
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
    fn open(&mut self, row: &[(u32, u32)]) {
        if let Probe::Marked(marks) = self {
            marks.begin();
            for &(q, prod) in row {
                marks.set(q, prod);
            }
        }
    }

    /// The product of `(p, q)` in the opened row, if it lives.
    #[inline(always)]
    fn get(&self, p: u32, q: u32) -> Option<u32> {
        let prod = match self {
            Probe::ByP(table) => table[p as usize],
            Probe::ByQ(table) => table[q as usize],
            Probe::Marked(marks) => return marks.get(q),
        };
        (prod != NO_PRODUCT).then_some(prod)
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
    let (p, q) = (view.p, view.q);

    // The top-down side: `P`'s root pairs by their `c` child, the live
    // products of `o` by their `P` node, and `Q`'s root pairs by their `o`
    // child.
    let p_root = grouped(lim, p.c, pairs_with_parent(view.p_root), |(_, pair)| {
        let (at_c, at_o) = view.at_root(&pair);
        (at_c as usize, at_o)
    })?;
    let q_root = grouped(lim, q.o, pairs_with_parent(view.q_root), |(_, pair)| {
        let (at_c, at_o) = view.at_root(&pair);
        (at_o as usize, at_c)
    })?;
    let o_rows = rows_by_p(lim, &view, input.o.products, p.o)?;
    // The bottom-up side: both children's live products by their `P` node,
    // and `Q`'s pairs at `c` by each child, holding the node and the other
    // child.
    let cl_rows = rows_by_p(lim, &view, input.cl.products, p.cl)?;
    let cr_rows = rows_by_p(lim, &view, input.cr.products, p.cr)?;
    let q_by_cl = grouped(lim, q.cl, pairs_with_parent(view.q_c), |(node, pair)| (pair.left.0 as usize, (node, pair.right.0)))?;
    let q_by_cr = grouped(lim, q.cr, pairs_with_parent(view.q_c), |(node, pair)| (pair.right.0 as usize, (node, pair.left.0)))?;
    let mut probe_cl = Probe::new(lim, &view, input.cl.products, p.cl, q.cl)?;
    let mut probe_cr = Probe::new(lim, &view, input.cr.products, p.cr, q.cr)?;

    let (col_o, col_cl, col_cr) = (input.o.counts, input.cl.counts, input.cr.counts);
    let mut weights: Stamped<u128> = Stamped::new(lim, q.c)?;
    let mut total = Total::default();
    let mut ticker = lim.gate_with(super::super::budget::APPLY_POLL_STRIDE);
    for (p_node, node) in view.p_c.nodes.iter().enumerate() {
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
        for pair in view.p_c.pairs_of(node) {
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
                            let Some(prod_r) = probe_cr.get(p_cr, q_cr) else { continue };
                            total.add3(c_l, col_cr[prod_r as usize], v);
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
                            let Some(prod_l) = probe_cl.get(p_cl, q_cl) else { continue };
                            total.add3(col_cl[prod_l as usize], c_r, v);
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
