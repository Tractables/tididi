//! The relabelling route: a level where one operand is one node with one pair.
//!
//! Above the variables an operand depends on, each level of it holds the one
//! node its output reaches, with one pair: the path down to its support. Call
//! that operand the narrow one and the other the carrier. The product of the
//! carrier's node `i` with the narrow node `(l, r)` is node `i`'s own pairs
//! `(a, b)`, each side carried to its product `(a ∧ l, b ∧ r)` at the child
//! level, less the pairs one of whose products is false. No two pairs meet,
//! so the level is one pass over the carrier's pairs with two array reads
//! each, where the general routes join the two operands' pairs on their
//! children. Distinct carrier pairs stay distinct, since the products of the
//! distinct nodes of one child level are distinct nodes, so the level needs
//! no deduplication either.
//!
//! Where neither side renumbers anything — each child's products are the
//! carrier's nodes in order, as when the child is complete and the narrow
//! operand has one node there — no pair changes and none dies: the carrier's
//! level is the output as it stands and moves into it, as the identity fast
//! path moves one. A conjunction with an operand whose support sits under one
//! subtree then costs the levels of that subtree; every level above it is
//! moved, whatever its size.

use crate::apply::conjoin::held::Held;
use crate::diagram::EncodedChildRef;
use super::identity::publish_identity_level;
use super::products::{FNodeIdx, GNodeIdx, Grid, ProductEntry, ProductNodeIdx};
use super::*;

/// How one child side of a relabelled level maps the carrier's child
/// references.
enum SideMap<'a> {
    /// Each reference is its own product.
    Identity,
    /// A leaf child: the product of a carrier label with the narrow
    /// operand's label, from the leaf conjunction table.
    Leaf(u32),
    /// The product of each carrier node, `NO_PRODUCT` where false.
    Nodes(&'a [u32]),
}

impl<'a> SideMap<'a> {
    /// The map of a child side whose narrow-operand reference is `label`:
    /// the leaf table on a leaf, else `buf` when the product store filled it,
    /// else the identity.
    fn of(leaf: bool, label: u32, filled: bool, buf: &'a [u32]) -> SideMap<'a> {
        if leaf {
            SideMap::Leaf(label)
        } else if filled {
            SideMap::Nodes(buf)
        } else {
            SideMap::Identity
        }
    }

    #[inline(always)]
    fn get(&self, child: u32) -> u32 {
        match self {
            SideMap::Identity => child,
            SideMap::Leaf(label) => CONJOIN_GRID[child as usize][*label as usize],
            // Safety: a structural reference indexes its child level, whose
            // width the map was built over.
            SideMap::Nodes(map) => unsafe { *map.get_unchecked(child as usize) },
        }
    }

    fn is_identity(&self) -> bool {
        match self {
            SideMap::Identity => true,
            SideMap::Leaf(label) => *label == LeafLabel::One as u32,
            SideMap::Nodes(_) => false,
        }
    }
}

/// The one pair of `tdd`'s level `t` when the level has one node with one
/// pair, and nothing otherwise: nothing on a level of other than one slot
/// (`width`), on a marginal level, whose one slot is a value and no node,
/// or on a level that holds no node.
pub(super) fn single_pair(tdd: &Held<'_>, t: usize, width: usize) -> Option<ChildPair> {
    if width != 1 {
        return None;
    }
    let level = &tdd.level(t);
    if level.is_marginal() || level.nodes().len() != 1 || level.pair_count_at(0) != 1 {
        return None;
    }
    level.pairs_iter_of_idx(0).next()
}

/// Build the level at `shape.t` by relabelling when one operand is one node
/// with one pair there, and say whether it did.
///
/// Declines, leaving everything as it was, on any level the general routes
/// must see: a marginal level on either operand, a marginal child other
/// than one the output carries from the carrier, a target, a weighted
/// sweep, and a root the count mode counts instead of building. So in a sweep that
/// sums levels out (`and_marginalizing_where`) it takes the levels beside and
/// above the targets: each is structure whose children's products are one
/// node per live cell, as in a plain conjunction. And on an operand an
/// earlier sum left marginal levels in, it reads through each one the other
/// operand is constant-true over, whose counts the output keeps as they
/// are.
///
/// # Errors
///
/// [`OperationError::OverBudget`] when the level's arenas or the product
/// store's growth is refused, and [`OperationError::Stopped`] when the stop
/// fires on the poll inside the pass.
pub(super) fn take_relabel_level(
    eng: &Engine,
    run: &mut ApplyRun,
    f: &mut Held<'_>,
    g: &mut Held<'_>,
    shape: LevelShape,
    sweep: &Sweep<'_>,
) -> Result<bool, OperationError> {
    let (t, left, right) = (shape.t, shape.left, shape.right);
    let ti = t.idx();
    // Neither operand one node here: no level is cheaper to turn away.
    if shape.f.here != 1 && shape.g.here != 1 {
        return Ok(false);
    }
    if relabel_forced_off()
        || sweep.ws.is_some()
        || sweep.targets.contains(ti)
        || (sweep.count_root && t == sweep.vtree.root())
        || f.level(ti).is_marginal()
        || g.level(ti).is_marginal()
    {
        return Ok(false);
    }
    // The one-pair test before the marginal state, which reads three levels.
    let (carrier_f, one) = match (
        single_pair(g, ti, shape.g.here),
        single_pair(f, ti, shape.f.here),
    ) {
        (Some(pair), _) => (true, pair),
        (None, Some(pair)) => (false, pair),
        (None, None) => return Ok(false),
    };
    let marginal = run.level_marginal(Operands { f: f.here(shape), g: g.here(shape) }, shape, sweep.targets);
    if marginal.is_target {
        return Ok(false);
    }
    // A marginal child is read through only where the output's level there
    // is the carrier's own, which an identity fast path moved because the
    // narrow operand is constant-true over it: each carrier reference to
    // it, a slot or an inline count, is then its own product, as the
    // general routes pass that side through.
    let carried = |c: VtreeIdx| run.lists.carried_from(c.idx(), carrier_f);
    if (marginal.left_any && !carried(left)) || (marginal.right_any && !carried(right)) {
        return Ok(false);
    }
    if marginal.left_any || marginal.right_any {
        note_read_through();
    }

    let vtree = sweep.vtree;
    let pools = &eng.scratch.apply;
    let (mut left_buf, mut right_buf) = (
        pools.relabel_left.checkout(eng),
        pools.relabel_right.checkout(eng),
    );
    let side_map = |c: VtreeIdx, label: u32, buf: &mut Vec<u32>| -> Result<bool, OperationError> {
        if vtree.node(c).is_leaf() {
            return Ok(false);
        }
        let ci = c.idx();
        let identity = Operands {
            f: run.f_identity[ci],
            g: run.g_identity[ci],
        };
        run.products.column(
            eng,
            ci,
            (run.f_widths[ci], run.g_widths[ci]),
            (carrier_f, label),
            identity,
            buf,
        )
    };
    let left_filled = side_map(left, one.left.raw(), &mut left_buf)?;
    let right_filled = side_map(right, one.right.raw(), &mut right_buf)?;
    let lm = SideMap::of(
        vtree.node(left).is_leaf(),
        one.left.raw(),
        left_filled,
        &left_buf,
    );
    let rm = SideMap::of(
        vtree.node(right).is_leaf(),
        one.right.raw(),
        right_filled,
        &right_buf,
    );
    let width = if carrier_f {
        shape.f.here
    } else {
        shape.g.here
    };

    if lm.is_identity() && rm.is_identity() {
        // Every pair is its own product: the carrier's level is the output.
        match carrier_f {
            true => f.carry(eng.limits(), ti, &mut run.levels[ti])?,
            false => g.carry(eng.limits(), ti, &mut run.levels[ti])?,
        }
        run.lists.carry(ti, carrier_f, true);
        publish_identity_level(run.products, ti, width);
        run.products.note_complete(ti);
        note_relabelled(true);
        return Ok(true);
    }

    let mut map = pools.relabel_map.checkout(eng);
    let carrier = if carrier_f {
        &f.level(ti)
    } else {
        &g.level(ti)
    };
    let pairs = relabel_level(eng, carrier, &mut run.levels[ti], width, &lm, &rm, &mut map)?;
    // A plain conjunction's seat closes the level as it stands.
    if run.plain {
        run.levels[ti].leave_live_pairs(pairs);
    }
    let nodes = run.levels[ti].nodes().len();
    run.products
        .publish_relabelled(eng, ti, carrier_f, &map, nodes)?;
    note_relabelled(false);
    Ok(true)
}

/// Write the carrier level's nodes into `level`, each pair carried through
/// the two side maps and dropped where either side's product is false, and
/// a node only where a pair survives; `map` receives each carrier node's
/// output node, `NO_PRODUCT` for one that died. Returns the level's live
/// pairs.
fn relabel_level(
    eng: &Engine,
    carrier: &TddLevel,
    level: &mut TddLevel,
    width: usize,
    lm: &SideMap<'_>,
    rm: &SideMap<'_>,
    map: &mut Vec<u32>,
) -> Result<usize, OperationError> {
    let lim = eng.limits();
    map.clear();
    lim.reserve(map, width)?;
    // The arena takes no more than the carrier's, plus the one pair of a
    // one-pair node, which waits there until the node is known to keep it
    // alone and stores it inline.
    let pre_pairs_cap = level.pairs.capacity();
    level.reserve_on(lim, width, carrier.pairs.len() + 1)?;
    lim.charge_output_pairs(level.pairs.capacity().saturating_sub(pre_pairs_cap));
    let mut gate = lim.gate_with(APPLY_POLL_STRIDE);
    let mut inline = 0usize;
    // An implicit carrier's pairs are generated a run of nodes at a time.
    carrier.try_for_each_node::<OperationError>(0..width, |_, pairs| {
        gate.poll(pairs.len() as u64)?;
        let out = level.pairs.stored_mut();
        let start = out.len();
        for pair in pairs {
            let l = lm.get(pair.left.raw());
            if l == NO_PRODUCT {
                continue;
            }
            let r = rm.get(pair.right.raw());
            if r == NO_PRODUCT {
                continue;
            }
            out.push(ChildPair::new(NodeIdx(l), NodeIdx(r)));
        }
        let survivors = out.len() - start;
        if survivors == 0 {
            map.push(NO_PRODUCT);
            return Ok(());
        }
        map.push(level.nodes.stored().len() as u32);
        if survivors == 1 {
            let pair = out.pop().expect("one survivor");
            level.nodes.stored_mut().push(EncodedNode::inline(pair));
            inline += 1;
        } else {
            level
                .try_push_multi_by_range(lim.level_width_cap(), start, survivors)?;
        }
        Ok(())
    })?;
    gate.flush()?;
    level.shrink_arrays();
    lim.level_settled(level.pairs.len() as u64);
    Ok(level.pairs.len() + inline)
}

/// Build the level at `shape.t`, where an operand passes through
/// ([`Held::passes`]): its node `i` there is the identity over its node `i`
/// at the child it passes to, and it has no variable under the other child.
/// Where the other operand has a level here, the product of the passing
/// node `i` with its node `j` is node `j`'s pairs, the passed side carried
/// to its product with `i` and the other side as it stands; where it also
/// passes, to the other child, it is the one pair of `i` and `j`. Cells are
/// emitted in order, `f`'s node major.
///
/// # Errors
///
/// [`OperationError::OverBudget`] when the level's arenas or the product
/// store's growth is refused, and [`OperationError::Stopped`] when the stop
/// fires on the poll inside the pass.
#[inline(never)]
pub(super) fn take_pass_level(
    eng: &Engine,
    run: &mut ApplyRun,
    f: &mut Held<'_>,
    g: &mut Held<'_>,
    shape: LevelShape,
    vtree: &crate::vtree::Vtree,
) -> Result<(), OperationError> {
    let lim = eng.limits();
    let (t, left, right) = (shape.t, shape.left, shape.right);
    let ti = t.idx();
    let (wf, wg) = (shape.f.here, shape.g.here);
    let pass_f = f.passes(ti);
    let passing = if pass_f { &*f } else { &*g };
    let passed_left = !passing.is_free(left.idx());
    let (cp, co) = if passed_left { (left, right) } else { (right, left) };
    let pinned = passing.pinned(ti, cp.idx());
    let literals = passing.literals(ti);
    if pass_f && g.passes(ti) {
        let pins = Operands { f: pinned, g: g.pinned(ti, co.idx()) };
        let rows = Operands { f: literals, g: g.literals(ti) };
        return cross_level(eng, run, ti, (wf, wg), passed_left, pins, rows);
    }
    let (wp, wo) = if pass_f { (wf, wg) } else { (wg, wf) };
    let identity_p = if pass_f { &run.f_identity } else { &run.g_identity };
    if wp == 1 && identity_p[cp.idx()] && identity_p[co.idx()] {
        // The passing operand is constant true here.
        return match pass_f {
            true => super::identity::apply_identity_fast_path::<false>(eng, shape, g, run),
            false => super::identity::apply_identity_fast_path::<true>(eng, shape, f, run),
        };
    }
    let carrier_f = !pass_f;
    let fixed = |i: usize| pinned.unwrap_or(i as u32);
    let pools = &eng.scratch.apply;
    let cp_leaf = vtree.node(cp).is_leaf();
    let c = cp.idx();
    let widths = (run.f_widths[c], run.g_widths[c]);
    let grid = match cp_leaf {
        true => None,
        false => Some(run.products.grid(c, widths, Operands { f: run.f_identity[c], g: run.g_identity[c] })),
    };
    let narrow = if carrier_f { widths.1 } else { widths.0 };
    let unchanged = match grid {
        None => wp == 1 && fixed(0) == LeafLabel::One as u32,
        Some(grid) => matches!(grid, Grid::Cells) && narrow == 1,
    };
    if unchanged {
        // Every pair is its own product: the carrier's level is the output.
        match carrier_f {
            true => f.carry(lim, ti, &mut run.levels[ti])?,
            false => g.carry(lim, ti, &mut run.levels[ti])?,
        }
        run.lists.carry(ti, carrier_f, true);
        publish_identity_level(run.products, ti, wo);
        run.products.note_complete(ti);
        return Ok(());
    }
    // The passed child's products with passing node `i` are its cells
    // `(k, fixed(i))` over the carrier's nodes `k` where the carrier is `f`,
    // else `(fixed(i), k)`, read off its grid in place; a sparse level's are
    // scattered into one row of the carrier's width per passing node.
    let (si, sk) = if carrier_f { (1, widths.1) } else { (widths.1, 1) };
    let stride = if carrier_f { widths.0 } else { widths.1 };
    // A sparse child's products are read by the carrier's node there
    // instead: offsets in `columns`, then each one's passing nodes and
    // products in `inverse`.
    let mut columns = pools.relabel_left.checkout(eng);
    let mut inverse = pools.relabel_right.checkout(eng);
    let mut sparse = false;
    if let Some(Grid::Lists) = grid {
        sparse = !pass_listed_forced_off()
            && run.products.invert_columns(eng, c, carrier_f, (pinned, wp), stride, (&mut columns, &mut inverse))?;
        if !sparse {
            run.products.scatter_columns(eng, c, carrier_f, (pinned, wp), stride, &mut columns)?;
        }
    }
    let mut map = pools.relabel_map.checkout(eng);
    map.clear();
    let mut list = Vec::new();
    if !sparse {
        lim.reserve(&mut map, wf * wg)?;
    }
    let carrier = if carrier_f { f.level(ti) } else { g.level(ti) };
    let level = &mut run.levels[ti];
    let pre_pairs_cap = level.pairs.capacity();
    // The sparse cells write at most one pair per listed product of a
    // carrier pair's passed child, and a node only where one is written: the
    // dense bounds, every cell and every carrier pair per passing node, run to
    // gigabytes on wide levels whose lists are short.
    let (nodes, pairs) = match sparse {
        true => {
            let mut listed = 0;
            carrier.for_each_node_pair(|_, pair| {
                let y = if passed_left { pair.left.raw() } else { pair.right.raw() } as usize;
                listed += (columns[y + 1] - columns[y]) as usize;
            });
            (listed.min(wf * wg), listed + 1)
        }
        false => (wf * wg, wp * carrier.pairs.len() + 1),
    };
    level.reserve_on(lim, nodes, pairs)?;
    lim.charge_output_pairs(level.pairs.capacity().saturating_sub(pre_pairs_cap));
    let mut rows = PassRows {
        carrier,
        width: wo,
        passing: wp,
        f_major: pass_f,
        literals,
        gate: lim.gate_with(APPLY_POLL_STRIDE),
        width_cap: lim.level_width_cap(),
        inline: 0,
    };
    // Each read is built once per passing node and moves its offsets in: one
    // that borrowed them reloaded them after every pair the cell wrote.
    let base = move |i: usize| fixed(i) as usize * si;
    let cols: &[u32] = &columns;
    // Safety, for the reads below: a structural reference indexes its child
    // level, whose width the grid and the columns were built over.
    match (passed_left, grid) {
        (true, Some(Grid::Lists)) if sparse => rows.emit_listed::<true>(lim, level, &mut list, &columns, &inverse)?,
        (false, Some(Grid::Lists)) if sparse => rows.emit_listed::<false>(lim, level, &mut list, &columns, &inverse)?,
        (true, None) => rows.emit::<true, true, _>(level, &mut map, |i| { let x = fixed(i) as usize; move |k| CONJOIN_GRID[k as usize][x] })?,
        (false, None) => rows.emit::<false, true, _>(level, &mut map, |i| { let x = fixed(i) as usize; move |k| CONJOIN_GRID[k as usize][x] })?,
        (true, Some(Grid::Cells)) => rows.emit::<true, false, _>(level, &mut map, |i| { let b = base(i); move |k| (b + k as usize * sk) as u32 })?,
        (false, Some(Grid::Cells)) => rows.emit::<false, false, _>(level, &mut map, |i| { let b = base(i); move |k| (b + k as usize * sk) as u32 })?,
        (true, Some(Grid::Slab(slab))) => rows.emit::<true, true, _>(level, &mut map, |i| { let b = base(i); move |k| unsafe { *slab.get_unchecked(b + k as usize * sk) } })?,
        (false, Some(Grid::Slab(slab))) => rows.emit::<false, true, _>(level, &mut map, |i| { let b = base(i); move |k| unsafe { *slab.get_unchecked(b + k as usize * sk) } })?,
        (true, Some(Grid::Lists)) => rows.emit::<true, true, _>(level, &mut map, |i| { let row = &cols[i * stride..]; move |k| unsafe { *row.get_unchecked(k as usize) } })?,
        (false, Some(Grid::Lists)) => rows.emit::<false, true, _>(level, &mut map, |i| { let row = &cols[i * stride..]; move |k| unsafe { *row.get_unchecked(k as usize) } })?,
    }
    let (mut gate, inline) = (rows.gate, rows.inline);
    gate.flush()?;
    level.shrink_arrays();
    lim.level_settled(level.pairs.len() as u64);
    let nodes = level.nodes().len();
    if run.plain {
        let pairs = level.pairs.len() + inline;
        level.leave_live_pairs(pairs);
    }
    match sparse {
        true => {
            note_pass_listed(carrier_f, list.len(), wf * wg);
            run.products.publish_listed(ti, (wf, wg), list, nodes)
        }
        false => run.products.publish_cells(eng, ti, wg, &map, nodes),
    }
}

/// The cells of [`take_pass_level`] on a level where one operand passes and
/// the other, the carrier, has `width` nodes: the passing operand's node `i`
/// with the carrier's node `j` is `j`'s pairs, the passed side carried to its
/// product with `i`.
struct PassRows<'a> {
    carrier: &'a TddLevel,
    width: usize,
    passing: usize,
    /// Whether the passing operand is `f`, whose node is the major index of
    /// the cells.
    f_major: bool,
    /// Where the passed child is a leaf, the labels that reach a node
    /// ([`Held::literals`]).
    literals: Option<u8>,
    gate: crate::limits::PollGate<'a>,
    width_cap: usize,
    inline: usize,
}

impl PassRows<'_> {
    /// Emit every cell into `level` in cell order, `f`'s node major, and
    /// the output node of each cell, or `NO_PRODUCT`, into `map`; `passed(i)`
    /// maps the passed child's node `c` to its product with passing node `i`,
    /// which is `NO_PRODUCT` only where `DEAD`.
    #[inline(always)]
    fn emit<const LEFT: bool, const DEAD: bool, P: Fn(u32) -> u32>(
        &mut self,
        level: &mut TddLevel,
        map: &mut Vec<u32>,
        passed: impl Fn(usize) -> P,
    ) -> Result<(), OperationError> {
        let carrier = self.carrier;
        match self.f_major {
            true => {
                for i in 0..self.passing {
                    let row = passed(i);
                    carrier.try_for_each_node::<OperationError>(0..self.width, |_, pairs| {
                        self.gate.poll(pairs.len() as u64)?;
                        self.cell::<LEFT, DEAD>(level, map, pairs, i, &row)
                    })?;
                }
                Ok(())
            }
            false => carrier.try_for_each_node::<OperationError>(0..self.width, |_, pairs| {
                self.gate.poll((pairs.len() * self.passing) as u64)?;
                let ([pair], None) = (pairs, self.literals) else {
                    return (0..self.passing).try_for_each(|i| self.cell::<LEFT, DEAD>(level, map, pairs, i, passed(i)));
                };
                let c = if LEFT { pair.left.raw() } else { pair.right.raw() };
                let nodes = level.nodes.stored_mut();
                let before = nodes.len();
                map.extend((0..self.passing).map(|i| {
                    let product = passed(i)(c);
                    if DEAD && product == NO_PRODUCT {
                        return NO_PRODUCT;
                    }
                    nodes.push(EncodedNode::inline(match LEFT {
                        true => ChildPair::new(NodeIdx(product), pair.right),
                        false => ChildPair::new(pair.left, NodeIdx(product)),
                    }));
                    (nodes.len() - 1) as u32
                }));
                self.inline += nodes.len() - before;
                Ok(())
            }),
        }
    }

    /// [`PassRows::emit`] where the passed child's products are listed by
    /// the carrier's node there: those of carrier node `y` are
    /// `inverse[2 * k]`, the passing node, and `inverse[2 * k + 1]`, the
    /// product, for `k` from `offsets[y]` to `offsets[y + 1]`. Only the
    /// products of the carrier's pairs are visited, so a dead cell costs
    /// nothing. The live cells go to `list` in cell order. A listed child is
    /// internal: on a chain over a leaf it lists no product of a label the
    /// chain leaves out ([`Held::literals`]), so the cells of that label die
    /// here.
    fn emit_listed<const LEFT: bool>(
        &mut self,
        lim: &crate::limits::Limits,
        level: &mut TddLevel,
        list: &mut Vec<ProductEntry>,
        offsets: &[u32],
        inverse: &[u32],
    ) -> Result<(), OperationError> {
        let products = |y: u32| &inverse[2 * offsets[y as usize] as usize..2 * offsets[y as usize + 1] as usize];
        match self.f_major {
            true => self.emit_listed_by_passing::<LEFT>(lim, level, list, &products),
            false => self.emit_listed_by_carrier::<LEFT>(lim, level, list, &products),
        }
    }

    /// [`PassRows::emit_listed`] where the carrier is `f`, whose node is
    /// the major index: a carrier node's pairs are carried to their products
    /// with every passing node at once and grouped by passing node.
    fn emit_listed_by_carrier<'p, const LEFT: bool>(
        &mut self,
        lim: &crate::limits::Limits,
        level: &mut TddLevel,
        list: &mut Vec<ProductEntry>,
        products: &impl Fn(u32) -> &'p [u32],
    ) -> Result<(), OperationError> {
        let carrier = self.carrier;
        // A product's pair keyed by its passing node, then by its place in
        // the carrier node, so that a sort groups the cells and keeps the
        // carrier's pair order in each.
        let mut cell: crate::limits::growth::Transient<'_, Vec<(u64, ChildPair)>> =
            crate::limits::growth::Transient::new(lim, Vec::new());
        carrier.try_for_each_node::<OperationError>(0..self.width, |x, pairs| {
            self.gate.poll(pairs.len() as u64)?;
            // A one-pair node's products sorted by passing node are its cells
            // in order. A child's list is not always in cell order.
            if let [pair] = pairs
                && products(passed_side::<LEFT>(pair).0).chunks_exact(2).is_sorted_by_key(|found| found[0])
            {
                let (y, other) = passed_side::<LEFT>(pair);
                let found = products(y);
                lim.reserve(list, found.len() / 2)?;
                for found in found.chunks_exact(2) {
                    list.push(ProductEntry {
                        f_idx: FNodeIdx(x as u32),
                        g_idx: GNodeIdx(found[0]),
                        prod_idx: ProductNodeIdx(level.nodes.stored().len() as u32),
                    });
                    level.nodes.stored_mut().push(EncodedNode::inline(carried::<LEFT>(other, found[1])));
                }
                self.inline += found.len() / 2;
                return Ok(());
            }
            let total: usize = pairs.iter().map(|pair| products(passed_side::<LEFT>(pair).0).len() / 2).sum();
            if total == 0 {
                return Ok(());
            }
            cell.clear();
            lim.reserve(&mut *cell, total)?;
            for pair in pairs {
                let (y, other) = passed_side::<LEFT>(pair);
                for found in products(y).chunks_exact(2) {
                    let seq = cell.len() as u64;
                    cell.push(((found[0] as u64) << 32 | seq, carried::<LEFT>(other, found[1])));
                }
            }
            cell.sort_unstable_by_key(|&(key, _)| key);
            lim.reserve(list, cell.len())?;
            let mut k = 0;
            while k < cell.len() {
                let i = cell[k].0 >> 32;
                let end = k + cell[k..].iter().take_while(|&&(key, _)| key >> 32 == i).count();
                self.push_cell(level, list, (x as u32, i as u32), &cell[k..end])?;
                k = end;
            }
            Ok(())
        })
    }

    /// [`PassRows::emit_listed`] where the passing operand is `f`, whose
    /// node is the major index: the products of every carrier pair are
    /// gathered in carrier order into one bucket per passing node, a stable
    /// counting sort, so a bucket holds its cells in carrier node order and
    /// each cell's pairs in the carrier's order.
    fn emit_listed_by_passing<'p, const LEFT: bool>(
        &mut self,
        lim: &crate::limits::Limits,
        level: &mut TddLevel,
        list: &mut Vec<ProductEntry>,
        products: &impl Fn(u32) -> &'p [u32],
    ) -> Result<(), OperationError> {
        use crate::limits::growth::Transient;
        let carrier = self.carrier;
        // Bucket `i` starts at `ends[i]`, which then runs to its end as the
        // bucket is written.
        let mut ends: Transient<'_, Vec<usize>> = Transient::new(lim, Vec::new());
        lim.try_resize(&mut *ends, self.passing + 1, 0)?;
        carrier.for_each_node_pair(|_, pair| {
            for found in products(passed_side::<LEFT>(&pair).0).chunks_exact(2) {
                ends[found[0] as usize + 1] += 1;
            }
        });
        for i in 0..self.passing {
            ends[i + 1] += ends[i];
        }
        let total = ends[self.passing];
        if total == 0 {
            return Ok(());
        }
        let mut bucket: Transient<'_, Vec<(u32, ChildPair)>> = Transient::new(lim, Vec::new());
        lim.try_resize(&mut *bucket, total, (0, ChildPair::new(NodeIdx(0), NodeIdx(0))))?;
        carrier.try_for_each_node::<OperationError>(0..self.width, |j, pairs| {
            self.gate.poll(pairs.len() as u64)?;
            for pair in pairs {
                let (y, other) = passed_side::<LEFT>(pair);
                for found in products(y).chunks_exact(2) {
                    let at = &mut ends[found[0] as usize];
                    bucket[*at] = (j as u32, carried::<LEFT>(other, found[1]));
                    *at += 1;
                }
            }
            Ok(())
        })?;
        lim.reserve(list, total)?;
        let mut start = 0;
        for i in 0..self.passing {
            let end = ends[i];
            self.gate.poll((end - start) as u64)?;
            let mut k = start;
            while k < end {
                let j = bucket[k].0;
                let run = k + bucket[k..end].iter().take_while(|&&(x, _)| x == j).count();
                self.push_cell(level, list, (i as u32, j), &bucket[k..run])?;
                k = run;
            }
            start = end;
        }
        Ok(())
    }

    /// Write the cell `(f, g)`, whose pairs are those of `run` in order, as
    /// the next node of `level`, and its entry in `list`, which has room.
    fn push_cell<K>(
        &mut self,
        level: &mut TddLevel,
        list: &mut Vec<ProductEntry>,
        (f, g): (u32, u32),
        run: &[(K, ChildPair)],
    ) -> Result<(), OperationError> {
        list.push(ProductEntry {
            f_idx: FNodeIdx(f),
            g_idx: GNodeIdx(g),
            prod_idx: ProductNodeIdx(level.nodes.stored().len() as u32),
        });
        if let [(_, pair)] = run {
            level.nodes.stored_mut().push(EncodedNode::inline(*pair));
            self.inline += 1;
            return Ok(());
        }
        let out = level.pairs.stored_mut();
        let start = out.len();
        out.extend(run.iter().map(|(_, pair)| *pair));
        level.try_push_multi_by_range(self.width_cap, start, run.len())
    }

    /// The cell of passing node `i` with the carrier node of `pairs`.
    #[inline(always)]
    fn cell<const LEFT: bool, const DEAD: bool>(
        &mut self,
        level: &mut TddLevel,
        map: &mut Vec<u32>,
        pairs: &[ChildPair],
        i: usize,
        passed: impl Fn(u32) -> u32,
    ) -> Result<(), OperationError> {
        if self.literals.is_some_and(|rows| rows >> i & 1 == 0) {
            map.push(NO_PRODUCT);
            return Ok(());
        }
        if let [pair] = pairs {
            let product = passed(if LEFT { pair.left.raw() } else { pair.right.raw() });
            if DEAD && product == NO_PRODUCT {
                map.push(NO_PRODUCT);
            } else {
                map.push(level.nodes.stored().len() as u32);
                level.nodes.stored_mut().push(EncodedNode::inline(match LEFT {
                    true => ChildPair::new(NodeIdx(product), pair.right),
                    false => ChildPair::new(pair.left, NodeIdx(product)),
                }));
                self.inline += 1;
            }
            return Ok(());
        }
        let out = level.pairs.stored_mut();
        let start = out.len();
        if !DEAD {
            out.extend(pairs.iter().map(|pair| match LEFT {
                true => ChildPair::new(NodeIdx(passed(pair.left.raw())), pair.right),
                false => ChildPair::new(pair.left, NodeIdx(passed(pair.right.raw()))),
            }));
            map.push(level.nodes.stored().len() as u32);
            return level.try_push_multi_by_range(self.width_cap, start, pairs.len());
        }
        // The level's pairs were reserved for every cell, so the cell's fit;
        // they are written through a local count rather than `Vec::push`,
        // whose length and pointer the loop would reload at each pair.
        let room = &mut out.spare_capacity_mut()[..pairs.len()];
        let mut survivors = 0;
        for pair in pairs {
            let (c, other) = if LEFT { (pair.left.raw(), pair.right) } else { (pair.right.raw(), pair.left) };
            let product = passed(c);
            if product == NO_PRODUCT {
                continue;
            }
            // Safety: at most one pair is written per pair read.
            unsafe { room.get_unchecked_mut(survivors) }.write(match LEFT {
                true => ChildPair::new(NodeIdx(product), other),
                false => ChildPair::new(other, NodeIdx(product)),
            });
            survivors += 1;
        }
        // Safety: the first `survivors` pairs past `start` were written.
        unsafe { out.set_len(start + survivors) };
        if survivors == 0 {
            map.push(NO_PRODUCT);
            return Ok(());
        }
        map.push(level.nodes.stored().len() as u32);
        if survivors == 1 {
            let pair = out.pop().expect("one survivor");
            level.nodes.stored_mut().push(EncodedNode::inline(pair));
            self.inline += 1;
            Ok(())
        } else {
            level.try_push_multi_by_range(self.width_cap, start, survivors)
        }
    }
}

/// A carrier pair's child on the passed side, and its other side.
#[inline(always)]
fn passed_side<const LEFT: bool>(pair: &ChildPair) -> (u32, EncodedChildRef) {
    if LEFT { (pair.left.raw(), pair.right) } else { (pair.right.raw(), pair.left) }
}

/// The carrier pair whose passed side became `product` and whose other side
/// is `other`.
#[inline(always)]
fn carried<const LEFT: bool>(other: EncodedChildRef, product: u32) -> ChildPair {
    match LEFT {
        true => ChildPair::new(NodeIdx(product), other),
        false => ChildPair::new(other, NodeIdx(product)),
    }
}

/// [`take_pass_level`] where both operands pass, `f` to the left child
/// where `f_left`: every cell is a node, the one pair of the two nodes,
/// an operand's output where `pins` names it, but those of a leaf's label
/// that `rows` leaves out ([`Held::literals`]).
fn cross_level(
    eng: &Engine,
    run: &mut ApplyRun,
    ti: usize,
    (wf, wg): (usize, usize),
    f_left: bool,
    pins: Operands<Option<u32>>,
    rows: Operands<Option<u8>>,
) -> Result<(), OperationError> {
    let lim = eng.limits();
    let cells = wf * wg;
    let read = |rows: Option<u8>, i: u32| rows.is_none_or(|rows| rows >> i & 1 == 1);
    let level = &mut run.levels[ti];
    level.reserve_on(lim, cells, 0)?;
    lim.gate().poll_each(cells as u64)?;
    if rows.f.is_none() && rows.g.is_none() {
        let nodes = level.nodes.stored_mut();
        for i in 0..wf as u32 {
            for j in 0..wg as u32 {
                let (a, b) = (pins.f.unwrap_or(i), pins.g.unwrap_or(j));
                let (l, r) = if f_left { (a, b) } else { (b, a) };
                nodes.push(EncodedNode::inline(ChildPair::new(NodeIdx(l), NodeIdx(r))));
            }
        }
    }
    let mut map = eng.scratch.apply.relabel_map.checkout(eng);
    map.clear();
    if rows.f.is_some() || rows.g.is_some() {
        lim.reserve(&mut map, cells)?;
        let nodes = level.nodes.stored_mut();
        for i in 0..wf as u32 {
            for j in 0..wg as u32 {
                if !read(rows.f, i) || !read(rows.g, j) {
                    map.push(NO_PRODUCT);
                    continue;
                }
                map.push(nodes.len() as u32);
                let (a, b) = (pins.f.unwrap_or(i), pins.g.unwrap_or(j));
                let (l, r) = if f_left { (a, b) } else { (b, a) };
                nodes.push(EncodedNode::inline(ChildPair::new(NodeIdx(l), NodeIdx(r))));
            }
        }
    }
    level.shrink_arrays();
    lim.level_settled(0);
    let nodes = level.nodes().len();
    if run.plain {
        level.leave_live_pairs(nodes);
    }
    if map.is_empty() {
        if let Some(base) = run.products.arena.materialized(ti) {
            for (k, cell) in run.products.arena.slab_mut()[base.idx()..base.idx() + cells].iter_mut().enumerate() {
                *cell = k as u32;
            }
        }
        run.products.record_live(ti, cells);
        run.products.note_complete(ti);
        return Ok(());
    }
    run.products.publish_cells(eng, ti, wg, &map, nodes)
}
