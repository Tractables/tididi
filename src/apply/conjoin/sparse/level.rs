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

/// Run the scatter for one level: choose which side to iterate and how the
/// candidates are collected, then join. Returns whether the candidates were
/// collected flat, which is where the emit reads them from.
///
/// With both children non-leaf the direction comes from
/// `estimate_scatter_direction`, and its emit-step count decides between a
/// bucket per f parent and the flat list (`flat_candidates_win`). A leaf
/// child is put on the inner side ([`leaf_direction`]), so the leaf arm
/// joins the level, into buckets. A pass-through side is put on the inner
/// side too, and the leaf arm joins the level on its other child. With
/// `direct`, the candidates go to that level's pair arena instead: see
/// [`finish_direct`].
#[expect(clippy::too_many_arguments)]
fn scatter_level(
    eng: &Engine,
    ws: &mut SparseWorkspace,
    f: &Tdd,
    g: &Tdd,
    shape: LevelShape,
    pl: Sides<&[ProductEntry]>,
    thresholds: SparseThresholds,
    direct: Option<&mut TddLevel>,
    passthrough: Option<Passthrough>,
) -> Result<bool, OperationError> {
    let lim = eng.limits();
    let t_idx = shape.t.idx();
    let leaves = Sides {
        left: f.vtree.node(shape.left).is_leaf(),
        right: f.vtree.node(shape.right).is_leaf(),
    };
    let (swap_direction, flat) = match (passthrough, leaf_direction(leaves)) {
        // The leaf arm reads the pass-through side where it reads a leaf,
        // on the inner side: the left child unswapped, the right one swapped.
        (Some(p), _) => (p.side == ChildSide::Right, false),
        (None, Some(swapped)) => (swapped, false),
        (None, None) => {
            // The estimator sums what each direction walks around the emit.
            // Do not substitute a plain grid-size proxy — it ignores
            // selectivity and mispicks on wide×wide segment conjoins.
            let choice = estimate_scatter_direction(
                eng,
                &mut ws.est_counts,
                &f.levels[t_idx], &g.levels[t_idx], pl.left, pl.right,
                shape,
            )?;
            (choice.swapped, flat_candidates_win(thresholds, shape.f.here, choice.emit_steps))
        }
    };

    let flat = flat && direct.is_none();
    if flat {
        ws.par_flat.clear();
    } else if direct.is_none() {
        ensure_buckets_cleared(eng, &mut ws.par_buckets, shape.f.here)?;
    }
    lim.try_resize(&mut ws.p2_map, shape.g.here, NO_PRODUCT)?;

    // The general arm carries no dead-probe inner loop; the leaf arm keeps
    // the leaf fast-path shape.
    let collect = match direct {
        Some(level) => Collect::Direct(level),
        None if flat => Collect::Flat,
        None => Collect::Buckets,
    };
    let carrier = passthrough.map(|p| p.carrier);
    if !swap_direction {
        scatter_join::<false>(eng, ws, &f.levels[t_idx], &g.levels[t_idx], shape, pl, leaves, collect, carrier)?;
    } else {
        scatter_join::<true>(eng, ws, &f.levels[t_idx], &g.levels[t_idx], shape, pl, leaves, collect, carrier)?;
    }
    if flat {
        sort_candidates(eng, ws, shape.f.here)?;
        // Sorted, the flat list is dead; its capacity stays for the next
        // level and the retention cap decides its fate at checkout's end.
        ws.par_flat.clear();
    }
    Ok(flat)
}

/// The join direction on a level with a leaf child, `None` when neither child
/// is a leaf and the direction estimate decides.
///
/// The leaf goes on the inner side, the left child unswapped and the right
/// one swapped, where the leaf arm reads its products from `CONJOIN_GRID`
/// (see [`runs_leaf_arm`]). With two leaf children the left one is inner.
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
/// `ensure_buckets_cleared`, `build_reverse_index`, the emit), the marking
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
            self.ws.p2_map.fill(NO_PRODUCT);
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
///
/// Steps:
///   scatter: fused scatter-filter by the outer child
///   emit:    per f parent, dedup its candidates' g parents into products via
///            `p2_map[g_parent]` and write each product's node into the level
///
/// A level where f and g have one node each has no emit step: its one
/// product's pairs are the candidates, and the scatter writes them into the
/// level.
///
/// The emit is chunked by f-parent index range when the projected transient
/// exceeds `thresholds.chunk_bytes`; each chunk's `par_buckets` rows are
/// then dropped as they are emitted, so the output grows in their space. A
/// level whose candidates were collected flat holds them in one sorted list
/// the chunks read in place, so chunking bounds only what the output grows by.
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
    thresholds: SparseThresholds,
    passthrough: Option<Passthrough>,
) -> Result<(), OperationError> {
    let t_idx = shape.t.idx();
    let ProductLists { left, right, out: pl_output } = lists;
    let pl = Sides { left, right };

    assert_no_marginal_children(t_idx, shape.left, shape.right, f, g, levels, passthrough);

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
        scatter_level(eng, ws, f, g, shape, pl, thresholds, Some(&mut *level), passthrough)?;
        finish_direct(eng, level, base, pl_output, duplicates_legal)?;
        #[cfg(debug_assertions)]
        debug_check_flushed_level(pl_output, &levels[t_idx]);
        guard.scatter_clean();
        return Ok(());
    }

    let flat = scatter_level(eng, ws, f, g, shape, pl, thresholds, None, passthrough)?;

    // `plan_chunks` greedy-packs f-parent indices into emit chunks under the
    // sparse chunk budget (`usize::MAX` disables). A level that fits in one
    // chunk is emitted once with `drop_consumed=false`, preserving
    // cross-apply par_buckets capacity reuse. Wider levels split into
    // several chunks with `drop_consumed=true`, releasing each
    // `par_buckets[p1]` once it is emitted, before the output grows further.
    // `pl_output` grows across chunks, so `prod_idx` stays sequential over
    // the level.
    let level = &mut levels[t_idx];
    let boundaries = if flat {
        plan_chunks(ws.par_sorted.offsets.windows(2).map(|w| (w[1] - w[0]) as usize), shape.f.here, thresholds.chunk_bytes)
    } else {
        plan_chunks(ws.par_buckets.iter().map(Vec::len), shape.f.here, thresholds.chunk_bytes)
    };
    let is_chunked = boundaries.len() > 2;
    for window in boundaries.windows(2) {
        let (p1_start, p1_end) = (window[0] as usize, window[1] as usize);
        emit_chunk(eng, ws, level, pl_output, p1_start..p1_end, flat, is_chunked, duplicates_legal)?;
    }

    #[cfg(debug_assertions)]
    debug_check_flushed_level(pl_output, &levels[t_idx]);

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
///
/// `par_buckets` needs no check — single-chunk mode iterated it by reference and
/// multi-chunk mode replaced each consumed bucket, and either way the next
/// apply's `ensure_buckets_cleared` resets the lengths.
#[cfg(debug_assertions)]
fn debug_check_flushed_level(pl_output: &[ProductEntry], level: &TddLevel) {
    for (i, e) in pl_output.iter().enumerate() {
        debug_assert_eq!(e.prod_idx.0 as usize, i,
            "pl_output[{}].prod_idx = {} but expected {}", i, e.prod_idx.0, i);
    }
    debug_assert!(level.nodes.len() == pl_output.len(),
        "level.nodes.len() {} != pl_output.len() {}", level.nodes.len(), pl_output.len());
}
