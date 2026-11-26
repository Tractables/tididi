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
//! A level of the marginal-child route takes the same description where each
//! child side is complete or carried from one operand ([`plan_marginal`]),
//! and a summed level whose operand levels are affine over such sides is
//! summed off it ([`plan_sums`]).

use crate::apply::conjoin::cell::GROUPED_MIN_PAIRS;
use crate::apply::conjoin::*;
use crate::diagram::{ChildPair, ChildSide, EncodedNode, ImplicitLevel, floor};

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
    Some((described(f)?, described(g)?))
}

/// The passed child of a pass level (`take_pass_level`), whose product of
/// the carrier's slot `s` there with passing node `i` is a cell of its
/// grid.
pub(in crate::apply::conjoin) enum PassedChild<F> {
    /// Every cell live, `(a, c)` for slots `a` of `f` and `c` of `g` at
    /// `a · width + c`.
    Complete { width: usize },
    /// Cell `(s, i)` read, the product or `NO_PRODUCT`.
    Cells(F),
}

/// The passing nodes of a pass level that keep their cells: `(count,
/// first, step)`.
pub(in crate::apply::conjoin) type Kept = (usize, usize, usize);

/// The description of a pass level's output (`take_pass_level`) where
/// every cell of the passed child the carrier's pairs read is an affine
/// function of the two slots it is read at, and the carrier's level is
/// affine (described, or fitting one), with the passing nodes that keep
/// their cells: `(count, first, step)`.
///
/// The passing nodes that keep a product with every slot the carrier's
/// pairs name on the passed side, the others none, must step evenly, and
/// the products be `at + a · s + b · i` over them
/// ([`read_passed`]). The level is then the carrier's, its passed side so
/// moved, taken with one pair a kept passing node, the cells of the others
/// dead. The other conditions are [`plan`]'s, for the carrier's level of
/// `wo` nodes.
pub(in crate::apply::conjoin) fn plan_pass(
    lim: &crate::limits::Limits,
    carrier: &TddLevel,
    (wp, wo): (usize, usize),
    (carrier_f, passed): (bool, ChildSide),
    child: PassedChild<impl Fn(i64, i64) -> u32>,
) -> Option<(ImplicitLevel, Kept)> {
    let bound = carrier.pairs.len().max(carrier.nodes().len());
    if !lim.memory_unbounded() || carrier.nodes().len() != wo || bound.saturating_mul(wp) < floor() {
        return None;
    }
    let dc = described(carrier)?;
    let ((n, i0, step), (at, a, b)) = match child {
        PassedChild::Complete { width } => {
            let w = width as i64;
            ((wp, 0, 1), if carrier_f { (0, w, 1) } else { (0, 1, w) })
        }
        PassedChild::Cells(cell) => read_passed(&slots_on(&dc, passed), wp, cell)?,
    };
    let first = at + b * i0 as i64;
    let product = if n == 1 {
        Some(dc.mapped_on(passed, (a, first))).filter(|p| p.pairs() >= floor())
    } else {
        let slot = b * step as i64;
        let dp = match passed {
            ChildSide::Left => ImplicitLevel::one_step(n, (first, 0), (slot, 0))?,
            ChildSide::Right => ImplicitLevel::one_step(n, (0, first), (0, slot))?,
        };
        let dc = dc.mapped_on(passed, (a, 0));
        let (df, dg) = if carrier_f { (&dc, &dp) } else { (&dp, &dc) };
        compose_described(df, dg, (1, 1), false)
    };
    product.map(|p| (p, (n, i0, step)))
}

/// The passing nodes of `0..wp` that keep a product with the carrier's
/// slots `slots`, as `(count, first, step)`, and `(at, a, b)` with
/// `cell(s, i) = at + a · s + b · i` over them, where each passing node
/// keeps a product with every slot or with none and those that keep them
/// step evenly. Reads each cell of the slots and the passing nodes once at
/// most, not each pair.
fn read_passed(
    slots: &[i64],
    wp: usize,
    cell: impl Fn(i64, i64) -> u32,
) -> Option<(Kept, (i64, i64, i64))> {
    let &s0 = slots.first()?;
    let mut kept = (0..wp as i64).filter(|&i| cell(s0, i) != NO_PRODUCT);
    let i0 = kept.next()?;
    let step = kept.next().map_or(1, |i1| i1 - i0);
    let n = 1 + (0..wp as i64).filter(|&i| i > i0 && cell(s0, i) != NO_PRODUCT).count();
    let live = |i: i64| i >= i0 && (i - i0) % step == 0 && (i - i0) / step < n as i64;
    if (0..wp as i64).any(|i| live(i) != (cell(s0, i) != NO_PRODUCT)) {
        return None;
    }
    let kept: Vec<i64> = (0..n as i64).map(|m| i0 + m * step).collect();
    let affine = affine_cells(slots, &kept, &cell)?;
    let dead = slots[1..].iter().all(|&s| (0..wp as i64).filter(|&i| !live(i)).all(|i| cell(s, i) == NO_PRODUCT));
    dead.then_some(((n, i0 as usize, step as usize), affine))
}

/// The distinct child slots on `side` that the pairs `d` describes name, in
/// order.
fn slots_on(d: &ImplicitLevel, side: ChildSide) -> Vec<i64> {
    let mut slots = Vec::new();
    d.each_slot_on(side, |s, _| slots.push(s));
    slots.sort_unstable();
    slots.dedup();
    slots
}

/// The offset and slopes `(at, a, b)` with `cell(s, c) = at + a · s + b · c`
/// at every slot `s` of `ss` and `c` of `cs`, each in order and distinct,
/// every such cell live: read off the first cells, then checked at each.
/// A slope is never 0 where its side has two slots: two pairs would then
/// have one product.
fn affine_cells(ss: &[i64], cs: &[i64], cell: impl Fn(i64, i64) -> u32) -> Option<(i64, i64, i64)> {
    let (&s0, &c0) = (ss.first()?, cs.first()?);
    let value = |s: i64, c: i64| Some(cell(s, c)).filter(|&p| p != NO_PRODUCT).map(i64::from);
    let slope = |rise: i64, run: i64| (rise != 0 && rise % run == 0).then(|| rise / run);
    let v0 = value(s0, c0)?;
    let a = match ss.get(1) {
        Some(&s1) => slope(value(s1, c0)? - v0, s1 - s0)?,
        None => 0,
    };
    let b = match cs.get(1) {
        Some(&c1) => slope(value(s0, c1)? - v0, c1 - c0)?,
        None => 0,
    };
    let at = v0 - a * s0 - b * c0;
    ss.iter()
        .all(|&s| cs.iter().all(|&c| value(s, c) == Some(at + a * s + b * c)))
        .then_some((at, a, b))
}

/// The description of an operand level the implicit route reads: its own,
/// or the one it fits. A stored level that closing reads a fit of fits
/// none, by the canonical form: only one of nodes of one pair, or of fewer
/// pairs than the floor, is read for a fit.
fn described(l: &TddLevel) -> Option<ImplicitLevel> {
    match l.implicit() {
        Some(d) => Some(d.clone()),
        None if l.closes_by_fit() => None,
        None => ImplicitLevel::fit(l),
    }
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
    ImplicitLevel::product(df, dg, stride, grouped).filter(|p| p.pairs() >= floor())
}

/// The description of level `t`'s output over child sides that are each
/// complete or carried, when `f` and `g` are affine there ([`plan`]'s
/// conditions on the operands). A carried side takes the carrying operand's
/// slots as they are, at stride 1, where the other operand names slot 0 on
/// that side at every pair; any other side is read through its child grid,
/// which must be complete: every cell live, its slot the cell's index,
/// which `complete` records or the grid at `bases` shows.
#[expect(clippy::too_many_arguments)]
pub(super) fn plan_marginal(
    lim: &crate::limits::Limits,
    f: &TddLevel,
    g: &TddLevel,
    shape: LevelShape,
    sides: Sides<SidePlan>,
    complete: Sides<bool>,
    grouped: bool,
    slab: &[u32],
    bases: Sides<Option<GridBase>>,
) -> Option<ImplicitLevel> {
    use crate::apply::conjoin::marginal_plan::Carrier;
    let (df, dg) = fit_operands(lim, f, g, shape)?;
    let (fl, fh) = df.slot_range();
    let (gl, gh) = dg.slot_range();
    let in_cell_order = |complete: bool, base: Option<GridBase>, cells: usize| {
        complete
            || base.is_some_and(|b| {
                slab[b.idx()..][..cells]
                    .iter()
                    .enumerate()
                    .all(|(c, &s)| s as usize == c)
            })
    };
    // The stride of a side's slots in the product: 1 where one operand's
    // slots are carried and the other's are all 0, the other operand's
    // width at the child where the child grid numbers its cells in order.
    let stride = |carrier: Option<Carrier>,
                  zero: (bool, bool),
                  complete: bool,
                  base: Option<GridBase>,
                  widths: (usize, usize)| match carrier {
        Some(Carrier::F) => zero.1.then_some(1),
        Some(Carrier::G) => zero.0.then_some(1),
        None => in_cell_order(complete, base, widths.0 * widths.1).then_some(widths.1),
    };
    let (fw, gw) = (shape.f, shape.g);
    let left = stride(
        sides.left.carrier,
        (fl.0 == 0 && fh.0 == 0, gl.0 == 0 && gh.0 == 0),
        complete.left,
        bases.left,
        (fw.left, gw.left),
    )?;
    let right = stride(
        sides.right.carrier,
        (fl.1 == 0 && fh.1 == 0, gl.1 == 0 && gh.1 == 0),
        complete.right,
        bases.right,
        (fw.right, gw.right),
    )?;
    compose_described(&df, &dg, (left, right), grouped)
}

/// The description of a summed level's pairs and the work the row loop
/// charges for them, where the level can be summed off the description
/// (`run_level_described_stream`): the description [`plan_marginal`] gives
/// over its child sides, each carried or a complete child grid, and a level
/// no stop can fire within. A summed level's grid is complete where every
/// cell of it is live, its cells' slots taken in cell order. No pair dies
/// on such a level, so the row loop would charge every cell all its pairs
/// ([`work`]).
#[expect(clippy::too_many_arguments)]
pub(super) fn plan_sums(
    lim: &crate::limits::Limits,
    f: &TddLevel,
    g: &TddLevel,
    shape: LevelShape,
    sides: Sides<SidePlan>,
    complete: Sides<bool>,
    grouped: bool,
    slab: &[u32],
    bases: Sides<GridBase>,
) -> Option<(ImplicitLevel, u64)> {
    let bases = Sides {
        left: Some(bases.left),
        right: Some(bases.right),
    };
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
pub(in crate::apply::conjoin) fn charges(
    nodes: (usize, usize),
    product: &ImplicitLevel,
    charged: usize,
) -> (u64, usize, u32) {
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
pub(in crate::apply::conjoin) fn write(
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
    // Each node the description holds, implied or written, has an index.
    if product.nodes() > lim.level_width_cap() {
        return Err(OperationError::IndexOverflow);
    }
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
        lim.charge_as_reserved::<EncodedNode>(held, nodes)?
            .max(product.nodes())
    } else {
        lim.reserve_exact(level.nodes.stored_mut(), nodes)?;
        product.write_nodes(level, lim.level_width_cap())?;
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
