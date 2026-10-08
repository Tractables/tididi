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

use super::*;
use super::identity::publish_identity_level;

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
/// pair, and nothing otherwise.
pub(super) fn single_pair(tdd: &Tdd, t: usize, width: usize) -> Option<ChildPair> {
    if width != 1 {
        return None;
    }
    let level = &tdd.levels[t];
    if level.pair_count_at(0) != 1 {
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
/// sweep, a one-product level with a target child ([`fuses_target_child`]),
/// and a root the count mode counts instead of building. So in a sweep that
/// sums levels out (`and_marginalizing`) it takes the levels beside and
/// above the targets: each is structure whose children's products are one
/// node per live cell, as in a plain conjunction. And on an operand an
/// earlier sum left marginal levels in, it reads through each one the other
/// operand is constant-true over, whose counts the output keeps as they
/// are. The caller has already excluded a filtered or quantified sweep.
///
/// # Errors
///
/// [`OperationError::OverBudget`] when the level's arenas or the product
/// store's growth is refused, and [`OperationError::Stopped`] when the stop
/// fires on the poll inside the pass.
pub(super) fn take_relabel_level(
    eng: &Engine,
    run: &mut ApplyRun,
    f: &mut Tdd,
    g: &mut Tdd,
    shape: LevelShape,
    sweep: &Sweep<'_, '_>,
) -> Result<bool, OperationError> {
    let (t, left, right) = (shape.t, shape.left, shape.right);
    let ti = t.idx();
    if relabel_forced_off()
        || sweep.ws.is_some()
        || sweep.targets.contains(ti)
        || fuses_target_child(sweep, t, shape.f.here, shape.g.here)
        || (sweep.count_root && t == sweep.vtree.root())
        || f.levels[ti].is_marginal()
        || g.levels[ti].is_marginal()
    {
        return Ok(false);
    }
    let marginal = run.level_marginal(f, g, shape, sweep.targets);
    if marginal.is_target {
        return Ok(false);
    }
    let (carrier_f, one) = match (single_pair(g, ti, shape.g.here), single_pair(f, ti, shape.f.here)) {
        (Some(pair), _) => (true, pair),
        (None, Some(pair)) => (false, pair),
        (None, None) => return Ok(false),
    };
    // A marginal child is read through only where the output's level there
    // is the carrier's own, which an identity fast path moved because the
    // narrow operand is constant-true over it: each carrier reference to
    // it, a slot or an inline count, is then its own product, as the
    // general routes pass that side through.
    let carried = |c: VtreeIdx| run.carried.iter().any(|&(x, from_f)| x == c.idx() && from_f == carrier_f);
    if (marginal.left_any && !carried(left)) || (marginal.right_any && !carried(right)) {
        return Ok(false);
    }
    if marginal.left_any || marginal.right_any {
        note_read_through();
    }

    let vtree = sweep.vtree;
    let pools = &eng.scratch.apply;
    let (mut left_buf, mut right_buf) = (pools.relabel_left.checkout(eng), pools.relabel_right.checkout(eng));
    let side_map = |c: VtreeIdx, label: u32, buf: &mut Vec<u32>| -> Result<bool, OperationError> {
        if vtree.node(c).is_leaf() {
            return Ok(false);
        }
        let ci = c.idx();
        let identity = Operands { f: run.f_identity[ci], g: run.g_identity[ci] };
        run.products.column(eng, ci, run.f_widths[ci], run.g_widths[ci], carrier_f, label, identity, buf)
    };
    let left_filled = side_map(left, one.left.raw(), &mut left_buf)?;
    let right_filled = side_map(right, one.right.raw(), &mut right_buf)?;
    let lm = SideMap::of(vtree.node(left).is_leaf(), one.left.raw(), left_filled, &left_buf);
    let rm = SideMap::of(vtree.node(right).is_leaf(), one.right.raw(), right_filled, &right_buf);
    let width = if carrier_f { shape.f.here } else { shape.g.here };

    if lm.is_identity() && rm.is_identity() {
        // Every pair is its own product: the carrier's level is the output.
        let carrier = if carrier_f { &mut f.levels } else { &mut g.levels };
        std::mem::swap(&mut run.levels[ti], &mut carrier[ti]);
        run.carried.push((ti, carrier_f));
        run.relabel_moved.push(ti);
        publish_identity_level(run.products, ti, width);
        run.products.note_complete(ti);
        note_relabelled(true);
        return Ok(true);
    }

    let mut map = pools.relabel_map.checkout(eng);
    let carrier = if carrier_f { &f.levels[ti] } else { &g.levels[ti] };
    relabel_level(eng, carrier, &mut run.levels[ti], width, &lm, &rm, &mut map)?;
    let nodes = run.levels[ti].nodes().len();
    run.products.publish_relabelled(eng, ti, carrier_f, &map, nodes)?;
    note_relabelled(false);
    Ok(true)
}

/// Whether `t` is a level of one node in each operand (`f_here`, `g_here`)
/// with a target child: the one product a [`ConjoinMode::Sum`] sweep may sum
/// that child out of as it finds the pairs (`sums_root`), and whose pairs
/// the two-step path fuses once the child is marginal. The relabelling route
/// leaves it to the general routes in every mode, so the two paths build it
/// alike, pair for pair; one node in each operand, it has nothing to save.
pub(super) fn fuses_target_child(sweep: &Sweep<'_, '_>, t: VtreeIdx, f_here: usize, g_here: usize) -> bool {
    if f_here != 1 || g_here != 1 || sweep.targets.is_empty() {
        return false;
    }
    let (left, right) = sweep.vtree.children(t);
    sweep.targets.contains(left.idx()) || sweep.targets.contains(right.idx())
}

/// Write the carrier level's nodes into `level`, each pair carried through
/// the two side maps and dropped where either side's product is false, and
/// a node only where a pair survives; `map` receives each carrier node's
/// output node, `NO_PRODUCT` for one that died.
fn relabel_level(
    eng: &Engine,
    carrier: &TddLevel,
    level: &mut TddLevel,
    width: usize,
    lm: &SideMap<'_>,
    rm: &SideMap<'_>,
    map: &mut Vec<u32>,
) -> Result<(), OperationError> {
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
    // An implicit carrier's pairs are generated into `buf` a node at a time.
    let mut buf = Vec::new();
    for i in 0..width {
        let pairs = carrier.pairs_read(i, &mut buf);
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
            continue;
        }
        map.push(level.nodes.stored().len() as u32);
        if survivors == 1 {
            let pair = out.pop().expect("one survivor");
            level.nodes.stored_mut().push(EncodedNode::inline(pair));
        } else {
            level.try_push_multi_by_range(start, survivors).map_err(|()| OperationError::OverBudget)?;
        }
    }
    gate.flush()?;
    level.shrink_arrays();
    lim.level_settled(level.pairs.len() as u64);
    Ok(())
}
