//! The implicit route: a level whose pairs are a complete product, written as
//! the description of its pairs.
//!
//! A level of the plain dense routes whose two child sides are
//! [complete](crate::apply::conjoin::products::Products::is_complete) reads
//! its child products by arithmetic: product `(a, c)` of the left child is
//! slot `a · W + c`, `W` the width of `g`'s left child level, and likewise on
//! the right. When the two operand levels are themselves affine in the digits
//! of a mixed radix (an [`ImplicitLevel`] or a level that fits one), every
//! cell is live and the level's pairs are the product description
//! [`ImplicitLevel::product`] gives, in the row loop's order. This route
//! charges the work clock and the meters what the row loop would, and keeps
//! the level as the description of its pairs, which implies its nodes.
//!
//! A summed level whose operand levels are affine and whose child grids are
//! complete is summed off the same description ([`plan_sums`]).

use crate::apply::conjoin::cell::GROUPED_MIN_PAIRS;
use crate::apply::conjoin::*;
use crate::diagram::{floor, ChildPair, EncodedNode, ImplicitLevel};

/// The description of level `t`'s output, when the implicit route can build
/// it: the engine's memory is unbounded (so the route's skipping the row
/// loop's scratch cannot change a later decision), both operand levels hold
/// every node of their width with the same number of pairs, both are affine,
/// a grouped level's runs are digits, and the product holds at least the
/// floor's pairs (`FLOOR`). `grouped` says the level's N×M cells may take
/// the grouped walk.
pub(super) fn plan(
    lim: &crate::limits::Limits,
    f: &TddLevel,
    g: &TddLevel,
    shape: LevelShape,
    grouped: bool,
) -> Option<ImplicitLevel> {
    let (df, dg) = fit_operands(lim, f, g, shape)?;
    compose_described(&df, &dg, (shape.g.left, shape.g.right), grouped)
}

/// The descriptions of `f` and `g` where the implicit route can build
/// `shape`'s level from them.
fn fit_operands(
    lim: &crate::limits::Limits,
    f: &TddLevel,
    g: &TddLevel,
    shape: LevelShape,
) -> Option<(ImplicitLevel, ImplicitLevel)> {
    // A level that fits a description holds its arena's length in pairs or,
    // at one pair a node, its nodes' count, so the product of the two bounds
    // bounds the product's pairs: under the floor, the row loop writes the
    // level, as it writes any product of fewer, and no operand is read for a
    // fit.
    let bound = |l: &TddLevel| l.pairs.len().max(l.nodes().len());
    if !lim.memory_unbounded()
        || f.nodes().len() != shape.f.here
        || g.nodes().len() != shape.g.here
        || bound(f).saturating_mul(bound(g)) < floor()
    {
        return None;
    }
    // A stored operand that closing reads a fit of fits none, by the
    // canonical form: only one of nodes of one pair, or of fewer pairs than
    // the floor, is read for a fit.
    let described = |l: &TddLevel| match l.implicit() {
        Some(d) => Some(d.clone()),
        None if l.closes_by_fit() => None,
        None => ImplicitLevel::fit(l),
    };
    Some((described(f)?, described(g)?))
}

/// The product of `df` and `dg` at `stride` where its pairs reach the floor.
fn compose_described(
    df: &ImplicitLevel,
    dg: &ImplicitLevel,
    stride: (usize, usize),
    grouped: bool,
) -> Option<ImplicitLevel> {
    // The row loop groups a cell when its f row and g column both have
    // `GROUPED_MIN_PAIRS` pairs and the column table recorded the column's
    // runs, which it does for every long column while their pairs fit `u32`.
    let grouped = grouped
        && df.pairs_per_node() >= GROUPED_MIN_PAIRS
        && dg.pairs_per_node() >= GROUPED_MIN_PAIRS
        && dg.pairs() <= u32::MAX as usize;
    // A product of fewer pairs than the floor is stored: the row loop writes
    // it.
    ImplicitLevel::product(df, dg, stride, grouped)
        .filter(|p| p.pairs() >= floor())
}

/// The description of level `t`'s output over complete or carried child
/// sides, when `f` and `g` are affine there (`plan_marginal`, `plan_sums`).
#[expect(clippy::too_many_arguments)]
pub(super) fn plan_marginal(
    lim: &crate::limits::Limits,
    f: &TddLevel,
    g: &TddLevel,
    shape: LevelShape,
    sides: Sides<super::super::marginal_plan::SidePlan>,
    complete: Sides<bool>,
    grouped: bool,
    slab: &[u32],
    bases: Sides<Option<GridBase>>,
) -> Option<ImplicitLevel> {
    use super::super::marginal_plan::Carrier;
    let (df, dg) = fit_operands(lim, f, g, shape)?;
    let (fl, fh) = df.slot_range();
    let (gl, gh) = dg.slot_range();
    let grid_complete = |comp: bool, base: Option<GridBase>, cells: usize| {
        comp || base.is_some_and(|b| slab[b.idx()..][..cells].iter().enumerate().all(|(c, &s)| s as usize == c))
    };
    let side_stride = |carrier: Option<Carrier>, f_zero: bool, g_zero: bool, comp: bool, base: Option<GridBase>, fw: usize, gw: usize| match carrier {
        Some(Carrier::F) => g_zero.then_some(1),
        Some(Carrier::G) => f_zero.then_some(1),
        None => grid_complete(comp, base, fw * gw).then_some(gw),
    };
    let (fw, gw) = (shape.f, shape.g);
    let sl = side_stride(sides.left.carrier, fl.0 == 0 && fh.0 == 0, gl.0 == 0 && gh.0 == 0, complete.left, bases.left, fw.left, gw.left)?;
    let sr = side_stride(sides.right.carrier, fl.1 == 0 && fh.1 == 0, gl.1 == 0 && gh.1 == 0, complete.right, bases.right, fw.right, gw.right)?;
    compose_described(&df, &dg, (sl, sr), grouped)
}

/// The description of a summed level's pairs and the work the row loop
/// charges for them, where the level can be summed off the description
/// (`run_level_described_stream`): the implicit route's description
/// ([`plan_marginal`]), a level no stop can fire within, and each child side
/// either carried from one operand, the other naming slot 0 there, or a
/// complete child grid, each cell holding the slot its index names. A summed
/// level's grid is complete where every cell of it is live, its cells' slots
/// taken in cell order. No pair dies on such a level, so the row loop
/// would charge every cell all its pairs ([`work`]).
#[expect(clippy::too_many_arguments)]
pub(super) fn plan_sums(
    lim: &crate::limits::Limits,
    f: &TddLevel,
    g: &TddLevel,
    shape: LevelShape,
    sides: Sides<super::super::marginal_plan::SidePlan>,
    complete: Sides<bool>,
    grouped: bool,
    slab: &[u32],
    bases: Sides<GridBase>,
) -> Option<(ImplicitLevel, u64)> {
    let bases = Sides { left: Some(bases.left), right: Some(bases.right) };
    let product = plan_marginal(lim, f, g, shape, sides, complete, grouped, slab, bases)?;
    let work = work((shape.f.here, shape.g.here), &product);
    lim.cannot_stop_within(work, 0).then_some((product, work))
}

/// The work the row loop charges a level of `product`'s shape on which no
/// candidate dies: one unit per pair of every cell and one per cell for
/// its row.
fn work(nodes: (usize, usize), product: &ImplicitLevel) -> u64 {
    (nodes.0 as u64) * (nodes.1 as u64) * (product.pairs_per_node() as u64 + 1)
}

/// What the row loop charges a level of `product`'s shape: the work clock
/// ([`work`]), and the output-pair meter, the doublings of the reserved
/// capacity `charged` that the level's pairs pass, with their count.
pub(super) fn charges(nodes: (usize, usize), product: &ImplicitLevel, charged: usize) -> (u64, usize, u32) {
    let work = work(nodes, product);
    let exact = product.arena_len();
    let (mut c, mut doublings) = (charged, 0);
    while exact > c {
        c = super::super::budget::doubled_pairs_capacity(c);
        doublings += 1;
    }
    (work, c - charged, doublings)
}

/// Write the level `product` describes into `level`, whose arenas are open
/// as the row loop's and empty, and its cells into the output slab `cells`:
/// node `c` at cell `c`. `size` is the nodes and arena pairs the row loop
/// would reserve the arenas for ([`complete_level_size`](super::level)):
/// the description is given the capacities that reservation would give,
/// charged as it would charge them, and holds no allocation of them. The
/// work clock and the output-pair meter are charged as the row loop charges
/// them; the caller has made sure no stop can fire on the way
/// ([`Limits::cannot_stop_within`]).
///
/// [`Limits::cannot_stop_within`]: crate::limits::Limits::cannot_stop_within
pub(super) fn write(
    eng: &Engine,
    product: ImplicitLevel,
    level: &mut TddLevel,
    size: (usize, usize),
    cells: &mut [u32],
    work: u64,
    meter: (usize, u32),
) -> Result<(), OperationError> {
    let lim = eng.limits();
    debug_assert_eq!(cells.len(), product.nodes());
    debug_assert!(level.nodes().is_empty() && level.pairs.is_empty());
    for (c, cell) in cells.iter_mut().enumerate() {
        *cell = c as u32;
    }
    let (nodes, pairs) = size;
    // The description holds the level, and implies its nodes where their
    // words fit: the node arena the level opened, one node a cell, is
    // dropped, its capacity kept as the one the reservation would give it.
    // Past that the nodes' words are stored, in an arena reserved for them.
    let node_capacity = if product.implies_nodes() {
        let held = level.nodes.imply();
        lim.charge_as_reserved::<EncodedNode>(held, nodes)?.max(product.nodes())
    } else {
        lim.reserve_exact(level.nodes.stored_mut(), nodes)?;
        product.write_nodes(level)?;
        0
    };
    lim.charge_output_pairs(meter.0);
    for _ in 0..meter.1 {
        super::super::note_scheduled_charge();
    }
    let capacity = lim.charge_as_reserved::<ChildPair>(level.pairs.capacity(), pairs)?;
    level.pairs.describe(product, capacity, node_capacity);
    lim.charge_work(work);
    lim.check_stop()
}
