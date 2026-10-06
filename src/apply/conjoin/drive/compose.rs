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
//! writes the nodes the row loop would, charges the work clock and the meters
//! what the row loop would, and keeps the pairs as their description.

use crate::apply::conjoin::cell::GROUPED_MIN_PAIRS;
use crate::apply::conjoin::*;
use crate::diagram::ImplicitLevel;

/// The description of level `t`'s output, when the implicit route can build
/// it: the engine's memory is unbounded (so the route's skipping the row
/// loop's scratch cannot change a later decision), both operand levels hold
/// every node of their width with the same number of pairs, both are affine,
/// and a grouped level's runs are digits. `grouped` says the level's N×M
/// cells may take the grouped walk.
pub(super) fn plan(
    lim: &crate::limits::Limits,
    f: &TddLevel,
    g: &TddLevel,
    shape: LevelShape,
    grouped: bool,
) -> Option<ImplicitLevel> {
    if !lim.memory_unbounded() || f.nodes.len() != shape.f.here || g.nodes.len() != shape.g.here {
        return None;
    }
    let described = |l: &TddLevel| l.implicit().cloned().or_else(|| ImplicitLevel::fit(l));
    let df = described(f)?;
    let dg = described(g)?;
    // The row loop groups a cell when its f row and g column both have
    // `GROUPED_MIN_PAIRS` pairs and the column table recorded the column's
    // runs, which it does for every long column while their pairs fit `u32`.
    let grouped = grouped
        && df.pairs_per_node() >= GROUPED_MIN_PAIRS
        && dg.pairs_per_node() >= GROUPED_MIN_PAIRS
        && dg.pairs() <= u32::MAX as usize;
    ImplicitLevel::product(&df, &dg, (shape.g.left, shape.g.right), grouped)
}

/// What the row loop charges a level of `product`'s shape: the work clock,
/// one unit per pair of every cell and one per cell for its row, and the
/// output-pair meter, the doublings of the reserved capacity `charged`
/// that the level's pairs pass, with their count.
pub(super) fn charges(nodes: (usize, usize), product: &ImplicitLevel, charged: usize) -> (u64, usize, u32) {
    let cells = (nodes.0 as u64) * (nodes.1 as u64);
    let work = cells * (product.pairs_per_node() as u64 + 1);
    let exact = product.arena_len();
    let (mut c, mut doublings) = (charged, 0);
    while exact > c {
        c = super::super::budget::doubled_pairs_capacity(c);
        doublings += 1;
    }
    (work, c - charged, doublings)
}

/// Write the level `product` describes into `level`, whose arenas are open
/// and reserved as the row loop's, and its cells into the output slab
/// `cells`: node `c` at cell `c`. The work clock and the output-pair meter
/// are charged as the row loop charges them; the caller has made sure no
/// stop can fire on the way ([`Limits::cannot_stop_within`]).
pub(super) fn write(
    eng: &Engine,
    product: ImplicitLevel,
    level: &mut TddLevel,
    cells: &mut [u32],
    work: u64,
    meter: (usize, u32),
) -> Result<(), OperationError> {
    let lim = eng.limits();
    debug_assert_eq!(cells.len(), product.nodes());
    for (c, cell) in cells.iter_mut().enumerate() {
        *cell = c as u32;
    }
    product.write_nodes(lim, level)?;
    lim.charge_output_pairs(meter.0);
    for _ in 0..meter.1 {
        super::super::note_scheduled_charge();
    }
    if product.pairs_per_node() >= 2 {
        let capacity = level.pairs.capacity();
        level.pairs.describe(product, capacity);
    }
    lim.charge_work(work);
    lim.check_stop()
}
