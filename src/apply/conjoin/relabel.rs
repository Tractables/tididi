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
use super::identity::publish_identity_level;
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
/// sums levels out (`and_marginalizing`) it takes the levels beside and
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
    // A label of a leaf passed over that its reader does not name.
    let unread = |i: usize| literals.is_some_and(|rows| rows >> i & 1 == 0);
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
    // The passed child's products with each passing node, one column of the
    // carrier's width there each: none on a leaf, which reads the table, or
    // where the one column is the identity.
    let mut columns = pools.relabel_left.checkout(eng);
    let mut column = pools.relabel_right.checkout(eng);
    columns.clear();
    let cp_leaf = vtree.node(cp).is_leaf();
    let mut stride = 0;
    if !cp_leaf {
        let c = cp.idx();
        let widths = (run.f_widths[c], run.g_widths[c]);
        let identity = Operands { f: run.f_identity[c], g: run.g_identity[c] };
        stride = if carrier_f { widths.0 } else { widths.1 };
        for i in 0..wp {
            let filled = run.products.column(eng, c, widths, (carrier_f, fixed(i)), identity, &mut column)?;
            if !filled {
                debug_assert_eq!(wp, 1);
                stride = 0;
                break;
            }
            lim.reserve(&mut columns, stride)?;
            columns.extend_from_slice(&column);
        }
    }
    let passed_map = |i: usize, child: u32| -> u32 {
        if cp_leaf {
            CONJOIN_GRID[child as usize][fixed(i) as usize]
        } else if stride == 0 {
            child
        } else {
            // Safety: a structural reference indexes its child level, whose
            // width the columns were built over.
            unsafe { *columns.get_unchecked(i * stride + child as usize) }
        }
    };
    let unchanged = match cp_leaf {
        false => stride == 0,
        true => wp == 1 && fixed(0) == LeafLabel::One as u32,
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
    let mut map = pools.relabel_map.checkout(eng);
    map.clear();
    lim.reserve(&mut map, wf * wg)?;
    let carrier = if carrier_f { f.level(ti) } else { g.level(ti) };
    let level = &mut run.levels[ti];
    let pre_pairs_cap = level.pairs.capacity();
    level.reserve_on(lim, wf * wg, wp * carrier.pairs.len() + 1)?;
    lim.charge_output_pairs(level.pairs.capacity().saturating_sub(pre_pairs_cap));
    let mut gate = lim.gate_with(APPLY_POLL_STRIDE);
    let mut inline = 0usize;
    let mut emit = |pairs: &[ChildPair], i: usize, level: &mut TddLevel, map: &mut Vec<u32>| -> Result<(), OperationError> {
        gate.poll(pairs.len() as u64)?;
        if unread(i) {
            map.push(NO_PRODUCT);
            return Ok(());
        }
        let out = level.pairs.stored_mut();
        let start = out.len();
        for pair in pairs {
            let (l, r) = match passed_left {
                true => (passed_map(i, pair.left.raw()), pair.right.raw()),
                false => (pair.left.raw(), passed_map(i, pair.right.raw())),
            };
            if l == NO_PRODUCT || r == NO_PRODUCT {
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
            level.try_push_multi_by_range(lim.level_width_cap(), start, survivors)?;
        }
        Ok(())
    };
    match pass_f {
        true => {
            for i in 0..wp {
                carrier.try_for_each_node::<OperationError>(0..wo, |_, pairs| emit(pairs, i, level, &mut map))?;
            }
        }
        false => carrier.try_for_each_node::<OperationError>(0..wo, |_, pairs| {
            (0..wp).try_for_each(|i| emit(pairs, i, level, &mut map))
        })?,
    }
    gate.flush()?;
    level.shrink_arrays();
    lim.level_settled(level.pairs.len() as u64);
    let nodes = level.nodes().len();
    if run.plain {
        let pairs = level.pairs.len() + inline;
        level.leave_live_pairs(pairs);
    }
    run.products.publish_cells(eng, ti, wg, &map, nodes)
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
