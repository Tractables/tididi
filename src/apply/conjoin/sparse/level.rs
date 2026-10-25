//! The sparse route's whole-level entry point.

use super::*;
use crate::apply::conjoin::marginal_plan::{Carrier, SidePlan};
use crate::apply::conjoin::setup::LevelShape;
use crate::diagram::{ChildSide, Sides};

/// The child side of a sparse level that is a marginal pass-through, and the
/// operand that carries it.
///
/// The other operand is the identity at that child, so the side kills no pair
/// and each output pair takes the carrier's field there verbatim (see
/// `plan_marginal_level`). The level is then a join on its other child alone,
/// which the scatter's leaf arm walks: the pass-through side takes the place
/// of the leaf side, with the carried field for the conjunction table.
#[derive(Clone, Copy)]
pub(crate) struct Passthrough {
    /// The child the carried side is.
    pub(crate) side: ChildSide,
    /// The operand whose field is carried.
    pub(crate) carrier: Carrier,
}

impl Passthrough {
    /// The pass-through side of a level with exactly one, from its marginal plan.
    pub(crate) fn of(sides: Sides<SidePlan>) -> Option<Passthrough> {
        match (sides.left.carrier, sides.right.carrier) {
            (Some(carrier), None) => Some(Passthrough { side: ChildSide::Left, carrier }),
            (None, Some(carrier)) => Some(Passthrough { side: ChildSide::Right, carrier }),
            _ => None,
        }
    }
}

/// The child side of a sparse level read by arithmetic, without its product
/// list: a complete child, whose cell `(i, j)` is node `i * g_width + j`.
/// Its list would name every cell of its grid, where the level joins only
/// the cells its parents' pairs meet; the scatter's leaf arm reads it as it
/// reads a leaf, and the probe through the other child.
#[derive(Clone, Copy)]
pub(crate) struct Complete {
    /// The child read by arithmetic.
    pub(crate) side: ChildSide,
    /// `g`'s width at that child.
    pub(crate) g_width: usize,
}

/// Join one level into `collect`: choose the arm and its direction, then
/// join.
///
/// A leaf child is put on the inner side ([`leaf_direction`]) and the leaf
/// arm joins the level; so it does with a pass-through side, or a
/// `complete` side, which it reads where it reads a leaf. With both
/// children non-leaf and listed, the general arm joins the level in the
/// direction `estimate_scatter_direction` picks. Into the level's nodes
/// ([`Collect::Nodes`]) both arms walk f's nodes and emit each f parent as
/// its walk ends, the general arm through every outer key's filtered index
/// ([`scatter_routed`]). On a level with one product the candidates go
/// where `collect` says instead, from the general arm's walk by outer key
/// ([`scatter_join`]): the level's pair arena ([`finish_direct`]), a count
/// ([`count_sparse_level`]) or the sums of a child summed out
/// ([`sum_sparse_level`]).
#[expect(clippy::too_many_arguments)]
fn scatter_level(
    eng: &Engine,
    ws: &mut SparseWorkspace,
    f: &Tdd,
    g: &Tdd,
    shape: LevelShape,
    pl: Sides<&[ProductEntry]>,
    collect: Collect<'_, '_>,
    passthrough: Option<Passthrough>,
    complete: Option<Complete>,
) -> Result<(), OperationError> {
    let lim = eng.limits();
    let t_idx = shape.t.idx();
    let (f_level, g_level) = (&f.levels[t_idx], &g.levels[t_idx]);
    let leaves = Sides {
        left: f.vtree.node(shape.left).is_leaf(),
        right: f.vtree.node(shape.right).is_leaf(),
    };
    let inner = match (passthrough, complete) {
        (Some(p), _) => Some((p.side, Inner::Carried(p.carrier))),
        (None, Some(c)) => Some((c.side, Inner::Complete { g_width: c.g_width })),
        (None, None) => None,
    };
    debug_assert!(complete.is_none() || passthrough.is_none(), "a level reads one side without its list");
    lim.try_resize(&mut ws.emit.p2_map, shape.g.here, NO_PRODUCT)?;

    // The leaf arm reads the pass-through side, or the complete one, where
    // it reads a leaf, on the inner side: the left child unswapped, the
    // right one swapped.
    let leaf_side = match inner {
        Some((side, _)) => Some(side == ChildSide::Right),
        None => leaf_direction(leaves),
    };
    if let Some(swapped) = leaf_side {
        let inner = inner.map(|(_, inner)| inner);
        return if !swapped {
            scatter_leaf_arm::<false>(eng, ws, f_level, g_level, shape, pl, collect, inner)
        } else {
            scatter_leaf_arm::<true>(eng, ws, f_level, g_level, shape, pl, collect, inner)
        };
    }

    // The estimator sums what each direction walks around the emit. Do not
    // substitute a plain grid-size proxy — it ignores selectivity and
    // mispicks on wide×wide segment conjoins.
    let choice = estimate_scatter_direction(eng, &mut ws.est_counts, f_level, g_level, pl.left, pl.right, shape)?;
    match (collect, choice.swapped) {
        (Collect::Nodes(emit), false) => scatter_routed::<false>(eng, ws, f_level, g_level, shape, pl, emit),
        (Collect::Nodes(emit), true) => scatter_routed::<true>(eng, ws, f_level, g_level, shape, pl, emit),
        (collect, false) => scatter_join::<false>(eng, ws, f_level, g_level, shape, pl, collect),
        (collect, true) => scatter_join::<true>(eng, ws, f_level, g_level, shape, pl, collect),
    }
}

/// The join direction on a level with a leaf child, `None` when neither child
/// is a leaf and the direction estimate decides.
///
/// The leaf goes on the inner side, the left child unswapped and the right
/// one swapped, where the leaf arm reads its products from `CONJOIN_GRID`
/// ([`scatter_leaf_arm`]). With two leaf children the left one is inner.
pub(super) fn leaf_direction(leaves: Sides<bool>) -> Option<bool> {
    (leaves.left || leaves.right).then_some(!leaves.left)
}

/// The sparse workspace, checked out for one level.
///
/// `p2_map` is lazily cleared — the emit restores only the entries it wrote,
/// the g parents of the products it just made — so a bail mid-level (an
/// `OverBudget` out of a `try_push` deep in the scatter) would leave stale
/// product indices behind, and the next level would read them as live and
/// undercount. The guard makes that impossible without any state surviving
/// the call: the repair runs in `Drop`, on the bail path only, because
/// [`WsGuard::scatter_clean`] disarms it once the level's own cleanup has
/// finished.
///
/// Nothing else needs repair. Every other buffer is resized, filled or
/// cleared over its live range when the next level enters (`sides`,
/// `Rows::reset`, `build_reverse_index`, the emit), the marking
/// arrays are emptied by advancing their epoch, and `filtered`'s touched
/// list is cleared here: a stale one would index a narrower level's buckets
/// out of bounds.
struct WsGuard<'a> {
    ws: crate::execution::pool::PoolGuard<'a, SparseWorkspace>,
    repair: bool,
}

impl<'a> WsGuard<'a> {
    fn new(eng: &'a Engine) -> Self {
        WsGuard { ws: eng.scratch.sparse.checkout(eng), repair: true }
    }

    /// The lookup tables are all `NO_PRODUCT` again; nothing to repair.
    fn scatter_clean(&mut self) {
        self.repair = false;
    }
}

impl Drop for WsGuard<'_> {
    fn drop(&mut self) {
        if self.repair {
            self.ws.emit.p2_map.fill(NO_PRODUCT);
            self.ws.filtered_touched.clear();
        }
    }
}

impl std::ops::Deref for WsGuard<'_> {
    type Target = SparseWorkspace;
    fn deref(&self) -> &SparseWorkspace {
        &self.ws
    }
}

impl std::ops::DerefMut for WsGuard<'_> {
    fn deref_mut(&mut self) -> &mut SparseWorkspace {
        &mut self.ws
    }
}

/// Build one internal level by the sparse scatter-filter-dedup pipeline:
/// reverse indices over the parent pairs, a scatter from the live child
/// products upward, then dedup and emit. Only alive products are touched.
///
/// With `passthrough`, that side is carried rather than joined: its product
/// list is not read, and the output pairs hold the carrier's field there.
/// With `complete`, that side is joined by arithmetic, and its product list,
/// which the caller did not build, is not read either.
///
/// Steps, per f parent in order:
///   join: the parent's candidates, by the leaf arm's or the general arm's
///         walk of f's nodes
///   emit: dedup the candidates' g parents into products via
///         `p2_map[g_parent]` and write each product's node into the level
///
/// Each f parent is emitted as soon as its walk ends, so the level holds
/// one parent's candidates at a time, written to the level's pair arena as
/// the walk finds them.
///
/// A level where f and g have one node each has no emit step: its one
/// product's pairs are the candidates, and the join writes them into the
/// level.
///
/// # Errors
///
/// [`OperationError::OverBudget`] when a workspace or output reservation is refused.
#[expect(clippy::too_many_arguments)]
pub(crate) fn apply_sparse_level(
    eng: &Engine,
    shape: LevelShape,
    f: &Tdd,
    g: &Tdd,
    levels: &mut [TddLevel],
    lists: ProductLists<'_>,
    cells: Sides<CellLookup<'_>>,
    thresholds: SparseThresholds,
    passthrough: Option<Passthrough>,
    complete: Option<Complete>,
) -> Result<(), OperationError> {
    let t_idx = shape.t.idx();
    let ProductLists { left, right, out: pl_output } = lists;
    let pl = Sides { left, right };

    assert_no_marginal_children(t_idx, shape.left, shape.right, f, g, levels, passthrough);

    seed_output(eng, &f.levels[t_idx], &g.levels[t_idx], &mut levels[t_idx], pl_output)?;
    let mut guard = WsGuard::new(eng);
    let ws = &mut *guard;

    // Duplicate pairs in one node's list are legal once any level of the
    // diagram is marginal — pair lists are then multisets feeding a sum.
    // A duplicate here is *inherited*: an operand parent whose own list
    // holds the same pair twice produces the same product pair twice, which
    // is exactly the multiplicity the count recurrence needs. Only the
    // pure-Boolean case still guarantees set-ness, so that is where the
    // emit's check stays armed. `cfg!` is a compile-time constant, so the
    // level scan is dead code in release.
    let duplicates_legal = cfg!(debug_assertions)
        && (f.levels.iter().any(|l| l.is_marginal())
            || g.levels.iter().any(|l| l.is_marginal())
            || levels.iter().any(|l| l.is_marginal()));

    // ── Fused scatter-filter ──────────────────────────────────────
    //
    // Four-way join: parent(p1,p2) <- f(p1,a1,s1) /\ g(p2,a2,s2)
    //                                /\ left_alive(a1,a2) /\ right_alive(s1,s2)
    //
    // When the inner child is a leaf, the reverse index for the
    // opposite operand is keyed by the non-leaf child for selectivity,
    // and `CONJOIN_GRID` supplies the leaf product directly; a
    // pass-through inner side supplies the carried field the same way.

    // A level where f and g have one node each has one product at most,
    // `(0, 0)`, and every candidate is one of its pairs: the scatter writes
    // them straight into the output level, and nothing is left to group.
    if shape.f.here == 1 && shape.g.here == 1 {
        let level = &mut levels[t_idx];
        let base = level.pairs.len();
        scatter_level(eng, ws, f, g, shape, pl, Collect::Direct(&mut *level), passthrough, complete)?;
        finish_direct(eng, level, base, pl_output, duplicates_legal)?;
        #[cfg(debug_assertions)]
        debug_check_flushed_level(pl_output, &levels[t_idx]);
        guard.scatter_clean();
        return Ok(());
    }

    // A level of many products, each found by a lookup or two, is walked by
    // `f` node instead of scattered (`probe_level`). It joins both children,
    // so a pass-through side, and a leaf, which the leaf arm already reads by
    // lookup, stay with the scatter; and its pair table holds a `g` pair once,
    // so it is withheld where pair lists may be multisets. A complete side
    // has no list to walk through: the probe reads it only directly.
    let joined = passthrough.is_none()
        && !f.vtree.node(shape.left).is_leaf()
        && !f.vtree.node(shape.right).is_leaf();
    if joined {
        // Whether any level of the operands or of the output is marginal,
        // a scan of every level: read only where the probe would run, past
        // its size gate, so that a diagram of many levels does not pay a
        // scan on each of its small ones. The level being built is split
        // out of the scan, and read on its own.
        let (below, rest) = levels.split_at_mut(t_idx);
        let (level, above) = rest.split_first_mut().expect("the level is one of the levels");
        let here = level.is_marginal();
        let multisets = || {
            here || f.levels.iter().any(|l| l.is_marginal())
                || g.levels.iter().any(|l| l.is_marginal())
                || below.iter().chain(above.iter()).any(|l| l.is_marginal())
        };
        let listed = Sides {
            left: complete.is_none_or(|c| c.side != ChildSide::Left),
            right: complete.is_none_or(|c| c.side != ChildSide::Right),
        };
        if probe_level(eng, ws, f, g, shape, level, pl, listed, pl_output, cells, multisets, duplicates_legal)? {
            #[cfg(debug_assertions)]
            debug_check_flushed_level(pl_output, &levels[t_idx]);
            guard.scatter_clean();
            return Ok(());
        }
    }

    let emit = Emit { level: &mut levels[t_idx], out: pl_output, duplicates_legal, thresholds };
    scatter_level(eng, ws, f, g, shape, pl, Collect::Nodes(emit), passthrough, complete)?;

    #[cfg(debug_assertions)]
    debug_check_flushed_level(pl_output, &levels[t_idx]);

    guard.scatter_clean();
    Ok(())
}


/// Reserve the level's pair and node arenas and its product list at the
/// sizes of the larger operand's level there, where the budget leaves three
/// times that, so that the emit writes into them without the copies of
/// growth by doubling. A level that holds more grows from there. Seeding
/// at the two operands' sizes together was no faster and left its slack
/// with the result, a third more peak memory on some programs. The pair
/// arena's seed is charged to the output-pair meter, as
/// `open_level_arenas` charges the dense route's.
fn seed_output(
    eng: &Engine,
    f: &TddLevel,
    g: &TddLevel,
    level: &mut TddLevel,
    products: &mut Vec<ProductEntry>,
) -> Result<(), OperationError> {
    let lim = eng.limits();
    let pairs = f.pairs.len().max(g.pairs.len());
    let nodes = f.node_count().max(g.node_count());
    let bytes = (pairs * std::mem::size_of::<ChildPair>()
        + nodes * (std::mem::size_of::<crate::diagram::EncodedNode>() + std::mem::size_of::<ProductEntry>())) as u64;
    if bytes.saturating_mul(3) >= lim.headroom() {
        return Ok(());
    }
    let arena = level.pairs.stored_mut();
    let before = arena.capacity();
    lim.reserve_exact(arena, pairs.saturating_sub(arena.len()))?;
    lim.charge_output_pairs(arena.capacity() - before);
    let stored = level.nodes.stored_mut();
    lim.reserve_exact(stored, nodes.saturating_sub(stored.len()))?;
    lim.reserve_exact(products, nodes.saturating_sub(products.len()))
}

/// Count a level where f and g have one node each, whose one product's pairs
/// are the candidates, by folding each candidate into `fold` instead of
/// storing it: [`apply_sparse_level`]'s direct case, with the level left
/// empty and no product recorded.
///
/// # Errors
///
/// [`OperationError::OverBudget`] when a workspace reservation is refused.
#[expect(clippy::too_many_arguments)]
pub(crate) fn count_sparse_level(
    eng: &Engine,
    shape: LevelShape,
    f: &Tdd,
    g: &Tdd,
    levels: &[TddLevel],
    pl: Sides<&[ProductEntry]>,
    passthrough: Option<Passthrough>,
    fold: &mut CandidateFold<'_>,
) -> Result<(), OperationError> {
    debug_assert!(shape.f.here == 1 && shape.g.here == 1, "a counted level has one product");
    assert_no_marginal_children(shape.t.idx(), shape.left, shape.right, f, g, levels, passthrough);
    let mut guard = WsGuard::new(eng);
    scatter_level(eng, &mut guard, f, g, shape, pl, Collect::Fold(fold), passthrough, None)?;
    guard.scatter_clean();
    Ok(())
}

/// Sum one child of a one-product root out as the scatter emits its
/// candidates: each candidate adds its summed-side node's count to its kept
/// product in `sum`, and no pair is written ([`ChildSum`]). The joined
/// children are still structural here — the summed child is made marginal
/// only once its counts are read — so the marginal-child refusal holds. A
/// `complete` side is read by arithmetic, as [`apply_sparse_level`] reads
/// it, so the candidates come in the order the two-step path's level has.
#[expect(clippy::too_many_arguments)]
pub(crate) fn sum_sparse_level(
    eng: &Engine,
    shape: LevelShape,
    f: &Tdd,
    g: &Tdd,
    levels: &[TddLevel],
    pl: Sides<&[ProductEntry]>,
    complete: Option<Complete>,
    sum: &mut ChildSum,
) -> Result<(), OperationError> {
    debug_assert!(shape.f.here == 1 && shape.g.here == 1, "a summed level has one product");
    assert_no_marginal_children(shape.t.idx(), shape.left, shape.right, f, g, levels, None);
    let mut guard = WsGuard::new(eng);
    scatter_level(eng, &mut guard, f, g, shape, pl, Collect::Sum(sum), None, complete)?;
    guard.scatter_clean();
    Ok(())
}

/// Refuse a sparse apply whose joined children hold marginal levels.
///
/// Every other marginal-child level goes to the dedicated marginal-child dispatch.
/// This matters because the reverse index buckets parents by the decoded child
/// coordinate: under inline encoding a marginal ref decodes to the count, not a
/// per-node index, collapsing equal-count children into one bucket and dropping
/// multiplicity. A pass-through side is exempt: it is carried, never used as a
/// key. The operand-child checks are load-bearing — an inline marginal
/// ref can only exist on a marginal child level. Armed in every build, release
/// included, so a routing regression aborts loudly instead of miscounting.
fn assert_no_marginal_children(
    t_idx: usize,
    left: VtreeIdx,
    right: VtreeIdx,
    f: &Tdd,
    g: &Tdd,
    levels: &[TddLevel],
    passthrough: Option<Passthrough>,
) {
    let carried = |side| passthrough.is_some_and(|p| p.side == side);
    let marginal = |c: VtreeIdx| {
        f.levels[c.idx()].is_marginal() || g.levels[c.idx()].is_marginal() || levels[c.idx()].is_marginal()
    };
    cheap_assert!(
        (carried(ChildSide::Left) || !marginal(left)) && (carried(ChildSide::Right) || !marginal(right)),
        "apply_sparse_level reached with a marginal joined child (t={t_idx} l={} r={}): \
         the dedicated marginal-child dispatch was bypassed",
        left.idx(), right.idx()
    );
}

/// Check the invariants a flushed level leaves behind: `pl_output` grew
/// monotonically with `prod_idx[i] == i`, and it has one entry per node.
#[cfg(debug_assertions)]
fn debug_check_flushed_level(pl_output: &[ProductEntry], level: &TddLevel) {
    for (i, e) in pl_output.iter().enumerate() {
        debug_assert_eq!(e.prod_idx.0 as usize, i,
            "pl_output[{}].prod_idx = {} but expected {}", i, e.prod_idx.0, i);
    }
    debug_assert!(level.nodes().len() == pl_output.len(),
        "level.nodes().len() {} != pl_output.len() {}", level.nodes().len(), pl_output.len());
}
