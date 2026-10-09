//! Pricing a conjunction level before any of its storage is claimed.
//!
//! A level's output holds one pair for every pair `(a₁, s₁)` of `f` and pair
//! `(a₂, s₂)` of `g` at the level whose left children make a live product
//! `(a₁, a₂)` and whose right children make one, `(s₁, s₂)`: each such
//! combination is one pair of the output node `(p₁, p₂)` of the two nodes
//! holding them, and no two combinations write the same pair. Once the
//! children are built, that count is read off the operands' pairs and the
//! children's live products without joining them. Grouping the combinations
//! by `f`'s right child `s₁` and `g`'s left child `a₂`, the two filters
//! separate:
//!
//! ```text
//! pairs = Σ_{s₁, a₂} X(s₁, a₂) · Y(s₁, a₂)
//! X(s₁, a₂) = #{ f pair (a₁, s₁) : (a₁, a₂) live }
//! Y(s₁, a₂) = #{ g pair (a₂, s₂) : (s₁, s₂) live }
//! ```
//!
//! `X` is one walk over the left child's live products, each stepping
//! through `f`'s pairs over its `f` node, and `Y` one walk over the right
//! child's; grouping by `f`'s left child and `g`'s right child instead is the
//! mirror image. The table is as wide as `f` at one child times `g` at the
//! other, which is small exactly where a level is dangerous: where `f`'s
//! nodes differ only on one side and `g`'s only on the other, every product
//! of a node of `f` and a node of `g` is live and the level holds the product
//! of the two widths. Where both tables are wide, or the table would take
//! more than half the memory left, the count falls back to a floor: with `A` the combinations live on the left, `B` those live on the
//! right and `U` all of them, at least `|A| + |B| − |U|` are live on both.
//! A side every product of which is live (a complete child, or a side the
//! level carries) kills no combination, and the count is then the other
//! side's `|A|`.
//!
//! The conjunction prices a level only where memory is bounded and the
//! level's combinations could outgrow what is left; a level whose output
//! needs more than that is refused before it is built
//! ([`LevelRefusal`](crate::limits::LevelRefusal)). Debug builds count every
//! level they can and check the count against the level built.

use crate::limits::{Charged, Limits, OperationError};
use crate::diagram::{ChildPair, ChildSide, EncodedNode, Tdd, TddLevel};
use crate::vtree::VtreeIdx;
use super::{Complete, Passthrough, NO_PRODUCT};
use super::products::{ProductEntry, Products};
use super::route::Route;
use super::setup::{ApplyRun, LevelShape};

/// The fewest bytes one output pair holds: a slot of the pairs arena, or,
/// as a node's one pair, the node's own word.
const BYTES_PER_PAIR: u64 = {
    let (pair, node) = (std::mem::size_of::<ChildPair>(), std::mem::size_of::<EncodedNode>());
    (if pair < node { pair } else { node }) as u64
};

/// The bytes one cell of a level's product grid holds.
const BYTES_PER_CELL: u64 = std::mem::size_of::<u32>() as u64;

/// Steps the exact count may take beyond its inputs before it gives way to
/// the floor.
const SPARE_STEPS: u128 = 1 << 20;

/// One child side's live products, as a level's count reads them.
#[derive(Clone, Copy)]
pub(super) enum Live<'a> {
    /// Every product of the two operands' nodes at the child is live.
    All,
    /// The live products, listed.
    List(&'a [ProductEntry]),
    /// The child's product grid: `f`'s node `i` and `g`'s node `j` meet in
    /// cell `i * g_width + j`, `NO_PRODUCT` where their product is false.
    Grid { cells: &'a [u32], f_width: usize, g_width: usize },
}

impl Live<'_> {
    /// Visit each live product as its `f` node and its `g` node.
    #[inline]
    fn each(&self, mut visit: impl FnMut(u32, u32)) {
        match *self {
            Live::All => unreachable!("a side whose every product is live is not walked"),
            Live::List(list) => list.iter().for_each(|e| visit(e.f_idx.0, e.g_idx.0)),
            Live::Grid { cells, f_width, g_width } => {
                for i in 0..f_width {
                    let row = &cells[i * g_width..(i + 1) * g_width];
                    for (j, &cell) in row.iter().enumerate() {
                        if cell != NO_PRODUCT {
                            visit(i as u32, j as u32);
                        }
                    }
                }
            }
        }
    }
}

/// A child side's live products and both operands' widths there.
#[derive(Clone, Copy)]
pub(super) struct Side<'a> {
    pub(super) live: Live<'a>,
    pub(super) f_width: usize,
    pub(super) g_width: usize,
}

/// A child side's live products as the conjunction holds them, where it
/// holds them in a form a count reads: every one live when the child is
/// complete or an operand is constant-true over it, so that each node of the
/// other is its own product; else the child's grid, else its list. `None`
/// for a child with neither. Under a filter, which may have dropped products
/// either rule names, only the grid and the list are read.
pub(super) fn side_of<'a>(run: &'a ApplyRun, child: VtreeIdx, f_width: usize, g_width: usize, filtered: bool) -> Option<Side<'a>> {
    let (c, products) = (child.idx(), &*run.products);
    let every = products.is_complete(c) || run.f_identity[c] || run.g_identity[c];
    let live = if !filtered && every {
        Live::All
    } else if let Some(base) = products.arena.materialized(c) {
        Live::Grid { cells: &products.arena.slab()[base.idx()..], f_width, g_width }
    } else if products.has_list(c) {
        Live::List(products.list(c))
    } else {
        return None;
    };
    Some(Side { live, f_width, g_width })
}

/// A level's output pairs, as its count found them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Pairs {
    /// The pairs, or a floor on them.
    pub(super) count: u128,
    /// Whether `count` is the pairs themselves.
    pub(super) exact: bool,
}

impl Pairs {
    /// Whether a level built with `built` pairs agrees with the count.
    pub(super) fn admits(self, built: u128) -> bool {
        if self.exact { built == self.count } else { built >= self.count }
    }
}

/// The pairs a built level holds.
pub(super) fn built_pairs(level: &TddLevel) -> u128 {
    (0..level.nodes().len()).map(|i| level.pair_count_at(i) as u128).sum()
}

/// A bound on the combinations of `f`'s and `g`'s pairs at a level, read off
/// the arenas' lengths and node counts without a pass: a node of one pair
/// holds it inline, outside the arena.
pub(super) fn combinations_bound(f: &TddLevel, g: &TddLevel) -> u128 {
    let pairs = |l: &TddLevel| l.pairs.len() as u128 + l.node_count() as u128;
    pairs(f).saturating_mul(pairs(g))
}

/// The bytes left for a level to claim, where the engine bounds memory.
pub(super) fn headroom(lim: &Limits) -> Option<u64> {
    (!lim.memory_unbounded()).then(|| lim.headroom())
}

/// Refuse the level at `t` when `needed` bytes are more than `headroom`.
pub(super) fn refuse_over(lim: &Limits, t: VtreeIdx, needed: u64, headroom: u64) -> Result<(), OperationError> {
    match needed > headroom {
        true => Err(lim.refuse_level(t, needed, headroom)),
        false => Ok(()),
    }
}

/// The bytes a level's output holds at least, at `pairs` pairs.
pub(super) fn output_bytes(pairs: u128) -> u64 {
    u64::try_from(pairs.saturating_mul(u128::from(BYTES_PER_PAIR))).unwrap_or(u64::MAX)
}

/// The bytes a dense level's grids add at least: its own `cells`, and each
/// ungridded child's that it materializes, less the slab already claimed,
/// every cell of which a grid may reuse.
pub(super) fn grid_bytes(products: &Products, cells: u128) -> u64 {
    if !products.arena.is_bump() {
        return 0;
    }
    let reusable = products.arena.slab().len() as u128;
    u64::try_from(cells.saturating_sub(reusable).saturating_mul(u128::from(BYTES_PER_CELL))).unwrap_or(u64::MAX)
}

/// A scratch column charged to `lim` while it lives, or to nothing.
struct Column<'a, T> {
    lim: Option<&'a Limits>,
    v: Vec<T>,
}

impl<'a, T: Clone> Column<'a, T> {
    fn new(lim: Option<&'a Limits>, len: usize, value: T) -> Result<Self, OperationError> {
        let mut v = Vec::new();
        match lim {
            Some(lim) => lim.try_resize(&mut v, len, value)?,
            None => v.resize(len, value),
        }
        Ok(Column { lim, v })
    }
}

impl<T> Drop for Column<'_, T> {
    fn drop(&mut self) {
        if let Some(lim) = self.lim {
            lim.release_bytes(self.v.charged_bytes());
        }
    }
}

impl<T> std::ops::Deref for Column<'_, T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        &self.v
    }
}

impl<T> std::ops::DerefMut for Column<'_, T> {
    fn deref_mut(&mut self) -> &mut [T] {
        &mut self.v
    }
}

/// Which child of a pair a pass reads.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Child {
    Left,
    Right,
}

impl Child {
    #[inline(always)]
    fn of(self, pair: ChildPair) -> u32 {
        match self {
            Child::Left => pair.left.0,
            Child::Right => pair.right.0,
        }
    }

    #[inline(always)]
    fn other(self) -> Child {
        match self {
            Child::Left => Child::Right,
            Child::Right => Child::Left,
        }
    }
}

/// How many of `level`'s pairs reference each of the `width` nodes at one
/// child; `None` when a pair references past `width`, which no count reads.
fn histogram<'a>(lim: Option<&'a Limits>, level: &TddLevel, child: Child, width: usize) -> Result<Option<Column<'a, u32>>, OperationError> {
    let mut counts = Column::new(lim, width, 0u32)?;
    let fits = level.pairs_with_parent().try_for_each(|(_, pair)| {
        let slot = counts.get_mut(child.of(pair) as usize).ok_or(())?;
        *slot += 1;
        Ok::<(), ()>(())
    });
    Ok(fits.is_ok().then_some(counts))
}

/// `level`'s pairs grouped by their `key` child: each group, in key order,
/// the other child of each of its pairs. `counts` is the histogram of the
/// key child.
fn grouped<'a>(lim: Option<&'a Limits>, level: &TddLevel, key: Child, counts: &[u32]) -> Result<(Column<'a, u64>, Column<'a, u32>), OperationError> {
    let mut starts = Column::new(lim, counts.len() + 1, 0u64)?;
    for (k, &c) in counts.iter().enumerate() {
        starts[k + 1] = starts[k] + u64::from(c);
    }
    let total = starts[counts.len()] as usize;
    let mut others = Column::new(lim, total, 0u32)?;
    let mut fill = Column::new(lim, counts.len(), 0u64)?;
    fill.copy_from_slice(&starts[..counts.len()]);
    level.pairs_with_parent().for_each(|(_, pair)| {
        let k = key.of(pair) as usize;
        others[fill[k] as usize] = key.other().of(pair);
        fill[k] += 1;
    });
    Ok((starts, others))
}

/// What one walk over a side's live products reads: the combinations live
/// there, and the steps each orientation's exact count would take over it.
struct SideSums {
    /// `Σ deg_f · deg_g` over the live products: the combinations live here.
    live: u128,
    /// `Σ deg_f` over them.
    f_steps: u128,
    /// `Σ deg_g` over them.
    g_steps: u128,
}

/// Walk a side's live products against both operands' histograms there;
/// `None` when a product names a node past a histogram.
fn side_sums(side: &Side<'_>, hf: &[u32], hg: &[u32]) -> Option<SideSums> {
    let mut sums = SideSums { live: 0, f_steps: 0, g_steps: 0 };
    let mut fits = true;
    side.live.each(|i, j| match (hf.get(i as usize), hg.get(j as usize)) {
        (Some(&df), Some(&dg)) => {
            sums.live += u128::from(df) * u128::from(dg);
            sums.f_steps += u128::from(df);
            sums.g_steps += u128::from(dg);
        }
        _ => fits = false,
    });
    fits.then_some(sums)
}

/// The output pairs of the level `f` and `g` hold at `f_level` and
/// `g_level`, over children whose live products `left` and `right` give,
/// exactly where the walks fit their inputs and half the memory left, and as
/// a floor otherwise; `None` where a side's products are not readable. Scratch
/// is charged to `lim` when it is given.
///
/// # Errors
///
/// [`OperationError::OverBudget`] when a charged scratch column is refused.
fn level_pairs(
    lim: Option<&Limits>,
    f_level: &TddLevel,
    g_level: &TddLevel,
    left: Side<'_>,
    right: Side<'_>,
) -> Result<Option<Pairs>, OperationError> {
    let exact = |count| Ok(Some(Pairs { count, exact: true }));
    let count_pairs = |l: &TddLevel| l.pair_counts().map(|c| c as u128).sum::<u128>();
    let (pf, pg) = (count_pairs(f_level), count_pairs(g_level));
    let all = pf * pg;
    // A side whose every product is live kills no combination.
    let one_side = |side: &Side<'_>, child: Child| -> Result<Option<u128>, OperationError> {
        let Some(hf) = histogram(lim, f_level, child, side.f_width)? else { return Ok(None) };
        let Some(hg) = histogram(lim, g_level, child, side.g_width)? else { return Ok(None) };
        Ok(side_sums(side, &hf, &hg).map(|s| s.live))
    };
    match (left.live, right.live) {
        (Live::All, Live::All) => return exact(all),
        (Live::All, _) => return one_side(&right, Child::Right)?.map_or(Ok(None), exact),
        (_, Live::All) => return one_side(&left, Child::Left)?.map_or(Ok(None), exact),
        _ => {}
    }
    let (Some(hf_l), Some(hg_l), Some(hf_r), Some(hg_r)) = (
        histogram(lim, f_level, Child::Left, left.f_width)?,
        histogram(lim, g_level, Child::Left, left.g_width)?,
        histogram(lim, f_level, Child::Right, right.f_width)?,
        histogram(lim, g_level, Child::Right, right.g_width)?,
    ) else {
        return Ok(None);
    };
    let (Some(a), Some(b)) = (side_sums(&left, &hf_l, &hg_l), side_sums(&right, &hf_r, &hg_r)) else {
        return Ok(None);
    };
    // Grouped by f's right child and g's left one, `X` walks the left
    // side's products through f's pairs and `Y` the right side's through
    // g's; the mirror groups by f's left child and g's right one.
    let by_right = (right.f_width as u128 * left.g_width as u128, a.f_steps + b.g_steps);
    let by_left = (left.f_width as u128 * right.g_width as u128, b.f_steps + a.g_steps);
    let inputs = pf + pg + a.f_steps.max(a.g_steps) + b.f_steps.max(b.g_steps);
    let cost = |(cells, steps): (u128, u128)| cells + steps;
    let (mirror, chosen) = if cost(by_left) < cost(by_right) { (true, by_left) } else { (false, by_right) };
    // `X` and `Y`, one `u64` a cell each.
    let scratch = chosen.0.saturating_mul(16);
    let spare = lim.and_then(headroom).is_some_and(|h| scratch > u128::from(h / 2));
    if spare || cost(chosen) > 8 * inputs + SPARE_STEPS {
        let floor = (a.live + b.live).saturating_sub(all);
        return Ok(Some(Pairs { count: floor, exact: false }));
    }
    let count = match mirror {
        false => cross(lim, f_level, g_level, (&left, &hf_l), (&right, &hg_r), Child::Left)?,
        true => cross(lim, f_level, g_level, (&right, &hf_r), (&left, &hg_l), Child::Right)?,
    };
    exact(count)
}

/// The exact count, grouped by f's child opposite `first` and g's child on
/// `first`'s side: `X` walks `first`'s live products through f's pairs over
/// each product's `f` node, `Y` walks `second`'s through g's pairs over each
/// product's `g` node, and the count is `Σ X · Y` over the table.
fn cross(
    lim: Option<&Limits>,
    f_level: &TddLevel,
    g_level: &TddLevel,
    (first, hf): (&Side<'_>, &[u32]),
    (second, hg): (&Side<'_>, &[u32]),
    first_child: Child,
) -> Result<u128, OperationError> {
    // Rows are f's nodes at the second child, columns g's at the first.
    let columns = first.g_width;
    let cells = second.f_width * columns;
    let (f_starts, f_others) = grouped(lim, f_level, first_child, hf)?;
    let (g_starts, g_others) = grouped(lim, g_level, first_child.other(), hg)?;
    let mut x = Column::new(lim, cells, 0u64)?;
    let mut y = Column::new(lim, cells, 0u64)?;
    first.live.each(|i, j| {
        let (from, to) = (f_starts[i as usize] as usize, f_starts[i as usize + 1] as usize);
        for &row in &f_others[from..to] {
            x[row as usize * columns + j as usize] += 1;
        }
    });
    second.live.each(|i, j| {
        let (from, to) = (g_starts[j as usize] as usize, g_starts[j as usize + 1] as usize);
        for &column in &g_others[from..to] {
            y[i as usize * columns + column as usize] += 1;
        }
    });
    Ok(x.iter().zip(y.iter()).map(|(&a, &b)| u128::from(a) * u128::from(b)).sum())
}

/// Price a level a dense route builds, before it claims its grid: the grid
/// cells it and its ungridded children claim, and, on a route whose children
/// hold no marginal level, the pairs its output holds. Refuses a level that
/// needs more than the memory left, and returns the count where it took one:
/// where the level could outgrow what is left, and, in a debug build, on
/// every level it can read, for the check against the level built.
///
/// # Errors
///
/// [`OperationError::OverBudget`] for a refused level, recorded in the
/// meters ([`Limits::refuse_level`]), or a refused scratch column.
/// A count whose own scratch the memory left refused is no count: the level
/// is then priced without it, and a refusal is always the level's, named.
fn counted(pairs: Result<Option<Pairs>, OperationError>) -> Result<Option<Pairs>, OperationError> {
    match pairs {
        Err(OperationError::OverBudget) => Ok(None),
        other => other,
    }
}

pub(super) fn price_dense_level(
    lim: &Limits,
    run: &ApplyRun,
    f: &Tdd,
    g: &Tdd,
    shape: LevelShape,
    route: Route,
    filtered: bool,
) -> Result<Option<Pairs>, OperationError> {
    let LevelShape { t, left, right, f: fw, g: gw } = shape;
    let products = &*run.products;
    let cells = |f_width: usize, g_width: usize| f_width as u128 * g_width as u128;
    let ungridded = |c: VtreeIdx, f_width, g_width| match products.arena.is_bump() && products.arena.is_sparse(c.idx()) {
        true => cells(f_width, g_width),
        false => 0,
    };
    let grid = cells(fw.here, gw.here) + ungridded(left, fw.left, gw.left) + ungridded(right, fw.right, gw.right);
    let grid = grid_bytes(products, grid);
    let room = headroom(lim);
    let (f_level, g_level) = (f.level(t), g.level(t));
    let due = room.is_some_and(|h| grid.saturating_add(output_bytes(combinations_bound(f_level, g_level))) > h);
    let structural = matches!(route, Route::PlainDense | Route::Dense);
    let pairs = match structural && (due || cfg!(debug_assertions)) {
        true => match (side_of(run, left, fw.left, gw.left, filtered), side_of(run, right, fw.right, gw.right, filtered)) {
            (Some(l), Some(r)) => counted(level_pairs(due.then_some(lim), f_level, g_level, l, r))?,
            _ => None,
        },
        false => None,
    };
    if let Some(h) = room {
        let output = match due {
            true => pairs.map_or(0, |p| output_bytes(p.count)),
            false => 0,
        };
        refuse_over(lim, t, grid.saturating_add(output), h)?;
    }
    Ok(pairs)
}

/// Price a level the sparse scatter builds, its joined children's product
/// lists in hand: the pairs its output holds, a carried or complete side
/// killing none. Refuses and counts as [`price_dense_level`] does.
///
/// # Errors
///
/// As [`price_dense_level`].
#[expect(clippy::too_many_arguments)]
pub(super) fn price_sparse_level(
    lim: &Limits,
    run: &ApplyRun,
    f: &Tdd,
    g: &Tdd,
    shape: LevelShape,
    passthrough: Option<Passthrough>,
    complete: Option<Complete>,
    filtered: bool,
) -> Result<Option<Pairs>, OperationError> {
    let LevelShape { t, left, right, f: fw, g: gw } = shape;
    let (f_level, g_level) = (f.level(t), g.level(t));
    let room = headroom(lim);
    let due = room.is_some_and(|h| output_bytes(combinations_bound(f_level, g_level)) > h);
    if !due && !cfg!(debug_assertions) {
        return Ok(None);
    }
    let every = |side| passthrough.is_some_and(|p| p.side == side) || complete.is_some_and(|c| c.side == side);
    let side = |child: VtreeIdx, which, f_width, g_width| match every(which) {
        true => Some(Side { live: Live::All, f_width, g_width }),
        false => side_of(run, child, f_width, g_width, filtered),
    };
    let pairs = match (side(left, ChildSide::Left, fw.left, gw.left), side(right, ChildSide::Right, fw.right, gw.right)) {
        (Some(l), Some(r)) => counted(level_pairs(due.then_some(lim), f_level, g_level, l, r))?,
        _ => None,
    };
    if let (true, Some(h), Some(p)) = (due, room, pairs) {
        refuse_over(lim, t, output_bytes(p.count), h)?;
    }
    Ok(pairs)
}

#[cfg(test)]
#[path = "tests/price.rs"]
mod tests;
