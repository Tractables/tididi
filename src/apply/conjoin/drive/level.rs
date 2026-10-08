//! One level of the bottom-up conjunction: routing a level to the sparse,
//! dense or streaming build and running it.
//!
//! The driver in [`super`] decides the order levels are visited in and what
//! happens at the boundaries; everything here is what a single level costs.

use crate::apply::conjoin::*;
use crate::Engine;
use super::Sweep;
use crate::diagram::ChildSide;
use crate::apply::conjoin::cell::GROUPED_MIN_PAIRS;
use crate::apply::conjoin::sparse::stream::{
    candidate_bound, choose_pivot, count as stream_count, Built, StreamInput, StreamWidths,
};

/// One level's build decision: its shape, the route chosen for it, and the
/// marginal plan the route was chosen on.
#[derive(Clone, Copy)]
pub(super) struct LevelBuild {
    pub(super) shape: LevelShape,
    pub(super) route: Route,
    pub(super) plan: MarginalPlan,
}

/// Run the sparse scatter pipeline for a level [`Route::Sparse`] was chosen
/// for: build the joined children's product lists, scatter-filter-dedup over
/// the live products, and record the result.
///
/// A pass-through side in `plan` is carried rather than joined, so it needs
/// no product list, and the level's fields on that side are marked inline
/// for the end-of-apply tagger, as [`Route::SparseMarg`] marks them.
pub(super) fn run_sparse_level(
    eng: &Engine,
    run: &mut ApplyRun,
    f: &mut Tdd,
    g: &mut Tdd,
    shape: LevelShape,
    plan: &MarginalPlan,
) -> Result<(), OperationError> {
    let LevelShape { t, left, right, f: fw, g: gw } = shape;
    let (ti, li, ri) = (t.idx(), left.idx(), right.idx());
    let passthrough = Passthrough::of(plan.sides);
    let carried = |side| passthrough.is_some_and(|p| p.side == side);
    // Ensure the joined children have product lists for the scatter pipeline.
    if !carried(ChildSide::Left) {
        run.ensure_product_list_for_child(eng, li, fw.left, gw.left)?;
    }
    if !carried(ChildSide::Right) {
        run.ensure_product_list_for_child(eng, ri, fw.right, gw.right)?;
    }

    apply_sparse_level(
        eng,
        shape, f, g,
        run.levels,
        run.products.lists(li, ri, ti),
        run.thresholds,
        passthrough,
    )?;
    run.products.finish_sparse(&mut run.levels[ti], ti);
    mark_passthrough_inlined(
        &mut run.levels[ti],
        Sides { left: carried(ChildSide::Left), right: carried(ChildSide::Right) },
    );
    Ok(())
}

/// Whether [`count_sparse_root`] may take a level the sparse route was chosen
/// for: the sweep wants only the count, the level is the root both operands
/// output at, f and g have one node there, neither child is carried, and no
/// weight, filter or target changes what the one product's count is.
pub(super) fn counts_root(
    sweep: &Sweep<'_, '_>,
    f: &Tdd,
    g: &Tdd,
    shape: LevelShape,
    plan: &MarginalPlan,
) -> bool {
    let t = shape.t;
    sweep.count_root
        && t == sweep.vtree.root()
        && f.output.vtree == t
        && g.output.vtree == t
        && shape.f.here == 1
        && shape.g.here == 1
        && Passthrough::of(plan.sides).is_none()
        && sweep.ws.is_none()
        && sweep.filter.is_none()
        && !sweep.targets.contains(t.idx())
}

/// Count the root level instead of building it: the model count of the
/// conjunction, which is the fold of the one root product's pairs against
/// the count columns of the two children.
///
/// The children's columns are folded first, over the levels built so far,
/// as a model count of the finished diagram would fold them; the scatter
/// then folds each candidate in where [`run_sparse_level`] would store it.
pub(super) fn count_sparse_root(
    eng: &Engine,
    run: &mut ApplyRun,
    f: &Tdd,
    g: &Tdd,
    shape: LevelShape,
    vtree: &crate::vtree::Vtree,
) -> Result<num_bigint::BigUint, OperationError> {
    use crate::value::{CountVec, FoldInput, IntFold, Retention, ValueDomain};
    let LevelShape { t, left, right, f: fw, g: gw } = shape;
    let (ti, li, ri) = (t.idx(), left.idx(), right.idx());
    run.ensure_product_list_for_child(eng, li, fw.left, gw.left)?;
    run.ensure_product_list_for_child(eng, ri, fw.right, gw.right)?;

    let lim = eng.limits();
    let mut computed: Vec<Option<CountVec>> = Vec::new();
    lim.reserve_exact(&mut computed, vtree.num_nodes())?;
    computed.resize_with(vtree.num_nodes(), || None);
    let levels: &[TddLevel] = run.levels;
    let marginal = |i: usize| levels[i].is_marginal();
    let input = FoldInput { vtree, levels, store: &() };
    let mut gate = lim.gate();
    for child in [left, right] {
        IntFold::ensure(eng, child, input, &mut computed, &marginal, Retention::Frontier, |w| gate.poll(w))?;
    }
    gate.flush()?;

    let mut fold = CandidateFold::new(
        lim,
        IntFold::child_view(li, vtree, &levels[li], &computed, &()),
        IntFold::child_view(ri, vtree, &levels[ri], &computed, &()),
    )?;
    let lists = run.products.lists(li, ri, ti);
    count_sparse_level(
        eng, shape, f, g, levels,
        Sides { left: lists.left, right: lists.right },
        run.thresholds, None, &mut fold,
    )?;
    Ok(fold.finish())
}

/// Whether [`sum_sparse_root`] may take a level the sparse route was chosen
/// for, and if so the summed child's side of its pairs: the sweep sums out
/// one child `c` of the level ([`Sweep::sum_child`]), f and g have one node
/// there, so the level is one product (the root of the conjunction, or a
/// level under it that every pair above reaches through that one node, as
/// a join's operands reach the subtree of the variables they bind), neither
/// child is carried, no weight, filter or quantified subtree is in play and
/// neither the level nor its other child is a target, `c` is internal, and
/// no level of `c`'s subtree is marginal in the output built so far or was
/// marginal in an operand at entry: `c` was built structural, as the
/// two-step path finds it once the sweep is over.
pub(super) fn sums_root(
    sweep: &Sweep<'_, '_>,
    run: &ApplyRun,
    shape: LevelShape,
    plan: &MarginalPlan,
) -> Option<ChildSide> {
    let c = sweep.sum_child?;
    let LevelShape { t, left, right, .. } = shape;
    let vtree = sweep.vtree;
    let (side, kept) = if c == right {
        (ChildSide::Right, left)
    } else if c == left {
        (ChildSide::Left, right)
    } else {
        return None;
    };
    let takes = shape.f.here == 1
        && shape.g.here == 1
        && Passthrough::of(plan.sides).is_none()
        && sweep.ws.is_none()
        && sweep.filter.is_none()
        && sweep.quantified.is_empty()
        && !sweep.targets.contains(t.idx())
        && !sweep.targets.contains(kept.idx())
        && !vtree.node(c).is_leaf()
        && vtree.subtree(c).all(|x| !run.entry_marginality.either(x.idx()) && !run.levels[x.idx()].is_marginal());
    takes.then_some(side)
}

/// Build the root, the one-product level [`sums_root`] took, with its child
/// `c`, on `side` of the root's pairs, summed out as the scatter finds the
/// pairs ([`ChildSum`]), instead of building every pair and fusing them once
/// `c` is marginalized.
///
/// `c`'s column is folded over the levels built so far, as
/// [`Engine::marginalize_levels`](crate::Engine::marginalize_levels) folds
/// it on the finished diagram, whose levels under the root these are (a
/// level above the root is built from the root's one node, whose number the
/// sum does not change); the
/// scatter adds each candidate's count to its kept product's sum; `c` and
/// every level under it are made marginal as that pass makes them
/// ([`install_summed`]); and the root's one node is written from the sums
/// ([`ChildSum::write_root`]). Returns the levels made marginal, children
/// before parents, for the caller to mark for re-contraction as the pass
/// marks them; `None` where the root has no pair, which leaves every level
/// as the sparse route leaves it.
pub(super) fn sum_sparse_root(
    eng: &Engine,
    run: &mut ApplyRun,
    f: &Tdd,
    g: &Tdd,
    shape: LevelShape,
    vtree: &crate::vtree::Vtree,
    side: ChildSide,
) -> Result<Option<Vec<VtreeIdx>>, OperationError> {
    use crate::value::{CountVec, FoldInput, IntFold, Retention, ValueDomain};
    let LevelShape { t, left, right, f: fw, g: gw } = shape;
    let (ti, li, ri) = (t.idx(), left.idx(), right.idx());
    let (c, kept) = match side {
        ChildSide::Right => (right, left),
        ChildSide::Left => (left, right),
    };
    run.ensure_product_list_for_child(eng, li, fw.left, gw.left)?;
    run.ensure_product_list_for_child(eng, ri, fw.right, gw.right)?;

    let lim = eng.limits();
    let mut computed: Vec<Option<CountVec>> = Vec::new();
    lim.reserve_exact(&mut computed, vtree.num_nodes())?;
    computed.resize_with(vtree.num_nodes(), || None);
    {
        let levels: &[TddLevel] = run.levels;
        let marginal = |i: usize| levels[i].is_marginal();
        let input = FoldInput { vtree, levels, store: &() };
        let mut gate = lim.gate();
        // `Retention::All`: the install takes every walked level's column,
        // as the two-step pass does.
        IntFold::ensure(eng, c, input, &mut computed, &marginal, Retention::All, |w| gate.poll(w))?;
        gate.flush()?;
    }
    let col = computed[c.idx()].take().expect("the fold leaves the summed child's column");
    let kept_width = match vtree.node(kept).is_leaf() {
        true => crate::diagram::LEAF_WIDTH,
        false => run.levels[kept.idx()].slot_count(),
    };
    let mut sum = ChildSum::new(lim, side, col, kept_width)?;
    let lists = run.products.lists(li, ri, ti);
    sum_sparse_level(
        eng, shape, f, g, run.levels,
        Sides { left: lists.left, right: lists.right },
        run.thresholds, &mut sum,
    )?;
    if sum.is_empty() {
        run.products.finish_sparse(&mut run.levels[ti], ti);
        return Ok(None);
    }
    computed[c.idx()] = Some(sum.take_column());
    let installed = install_summed(eng, run.levels, vtree, c, &mut computed)?;
    let [root, summed] = run.levels.get_disjoint_mut([ti, c.idx()]).expect("a level and its child are distinct");
    let base = root.pairs.len();
    let lists = run.products.lists(li, ri, ti);
    sum.write_root(eng, root, summed, base, lists.out)?;
    run.products.finish_sparse(&mut run.levels[ti], ti);
    Ok(Some(installed))
}

/// Make `c` and every level under it marginal from the columns in
/// `computed`, children before parents, as
/// [`Engine::marginalize_levels`](crate::Engine::marginalize_levels) does on
/// a finished diagram, and return the levels made marginal in that order.
///
/// `c`'s column is deduplicated as that pass installs it, so its store holds
/// one slot per value (invariant 10); its parent, the root, is not built
/// yet, so no reference is remapped. A level under `c` is freed as its
/// parent is installed, which is all the pass leaves of it either way, so it
/// is installed as the sweep's streaming route installs one, undeduplicated.
fn install_summed(
    eng: &Engine,
    levels: &mut [TddLevel],
    vtree: &crate::vtree::Vtree,
    c: VtreeIdx,
    computed: &mut [Option<crate::value::CountVec>],
) -> Result<Vec<VtreeIdx>, OperationError> {
    use crate::marginal::transition::{cascade, free_subsumed_marginal_children, install_streamed, InstallTarget, InternalLevel, MarginalDomain};

    /// The output levels, installing `c` as a finished diagram's pass
    /// installs it and noting each level installed.
    struct Summing<'l> {
        levels: &'l mut [TddLevel],
        c: VtreeIdx,
        installed: Vec<VtreeIdx>,
    }
    impl InstallTarget for Summing<'_> {
        fn levels(&self) -> &[TddLevel] {
            self.levels
        }

        fn install<K: MarginalDomain>(&mut self, vtree: &crate::vtree::Vtree, t: InternalLevel, col: K::Col, store: &mut K::Store) {
            let v = t.vtree_idx();
            debug_assert!(self.installed.len() < self.installed.capacity(), "reserved for every internal level of the subtree");
            self.installed.push(v);
            if v != self.c {
                install_streamed::<K>(self.levels, vtree, t, col, store);
                return;
            }
            crate::diagram::assert_can_make_marginal(self.levels, vtree, v);
            let remap = K::install(&mut self.levels[v.idx()], t, col, store);
            debug_assert!(remap.is_some(), "the integer domain dedups the column it installs");
            free_subsumed_marginal_children(self.levels, vtree, v, K::weight_store(store));
        }
    }

    let internal = vtree.subtree(c).filter(|&x| !vtree.node(x).is_leaf()).count();
    let mut installed = Vec::new();
    eng.limits().reserve_exact(&mut installed, internal)?;
    let mut target = Summing { levels, c, installed };
    cascade::<crate::value::IntFold, Summing<'_>>(&mut target, vtree, c, computed, &mut ());
    debug_assert!(target.levels[c.idx()].is_marginal(), "the cascade installs the summed child");
    Ok(target.installed)
}

/// Whether the sweep holds back the level at `t` unbuilt, for the root count
/// to stream: the sweep wants only the count of a root both operands output
/// at with one node, `t` is a child of that root, and neither operand is
/// constant-true over it; `t`, its children and its sibling are internal,
/// no weight, filter or quantified subtree touches any of them, and none of
/// the levels the stream reads built (`t`'s children and its sibling) is a
/// target: a target at `t` only decides how `t` is built if the stream
/// declines. Neither operand may be marginal at any of them, now or at
/// entry (an identity fast path moves a marginal level out of its operand),
/// nor carry counts in its references from the root or `t`: the stream reads
/// those references as node indices. Whether the root then streams `t` is
/// priced there, once the levels it reads are built
/// ([`count_streamed_root`]).
pub(super) fn holds_back(
    sweep: &Sweep<'_, '_>,
    run: &ApplyRun,
    f: &Tdd,
    g: &Tdd,
    t: VtreeIdx,
    left: VtreeIdx,
    right: VtreeIdx,
) -> bool {
    let vtree = sweep.vtree;
    let root = vtree.root();
    if !sweep.count_root
        || sweep.ws.is_some()
        || sweep.filter.is_some()
        || !sweep.quantified.is_empty()
        || vtree.node(t).parent() != Some(root)
        || f.output.vtree != root
        || g.output.vtree != root
        || run.f_widths[root.idx()] != 1
        || run.g_widths[root.idx()] != 1
    {
        return false;
    }
    let (root_left, root_right) = vtree.children(root);
    let sibling = if root_left == t { root_right } else { root_left };
    let internal = [t, left, right, sibling].iter().all(|&x| !vtree.node(x).is_leaf());
    let read_built = [left, right, sibling].iter().all(|&x| !sweep.targets.contains(x.idx()));
    let untouched = [root, t, left, right, sibling].iter().all(|&x| {
        !run.entry_marginality.either(x.idx())
            && !f.levels[x.idx()].is_marginal()
            && !g.levels[x.idx()].is_marginal()
    });
    let valued = |x: VtreeIdx| {
        [f, g].iter().any(|d| {
            let level = &d.levels[x.idx()];
            level.has_value_refs(ChildSide::Left) || level.has_value_refs(ChildSide::Right)
        })
    };
    let identity = |widths: &[usize], id: &[bool]| widths[t.idx()] == 1 && id[left.idx()] && id[right.idx()];
    internal
        && read_built
        && untouched
        && !valued(root)
        && !valued(t)
        && !identity(run.f_widths, run.f_identity)
        && !identity(run.g_widths, run.g_identity)
}

/// The widths a streamed count reads, one operand's.
fn stream_widths(widths: &[usize], c: VtreeIdx, o: VtreeIdx, cl: VtreeIdx, cr: VtreeIdx) -> StreamWidths {
    StreamWidths { c: widths[c.idx()], o: widths[o.idx()], cl: widths[cl.idx()], cr: widths[cr.idx()] }
}

/// Which of the held root children the root count would stream: the one
/// whose build would find the more candidates, by [`candidate_bound`].
pub(super) fn pick_streamed(
    eng: &Engine,
    run: &mut ApplyRun,
    f: &Tdd,
    g: &Tdd,
    held: &[(VtreeIdx, VtreeIdx, VtreeIdx)],
) -> Result<usize, OperationError> {
    if held.len() < 2 {
        return Ok(0);
    }
    let mut best = (0, 0u128);
    for (i, &(t, left, right)) in held.iter().enumerate() {
        for child in [left, right] {
            run.ensure_product_list_for_child(eng, child.idx(), run.f_widths[child.idx()], run.g_widths[child.idx()])?;
        }
        // `o` is not read by the bound; `t` stands in for it.
        let bound = candidate_bound(
            eng.limits(), f.level(t), g.level(t),
            run.products.list(left.idx()), run.products.list(right.idx()),
            stream_widths(run.f_widths, t, t, left, right),
            stream_widths(run.g_widths, t, t, left, right),
        )?;
        if i == 0 || bound > best.1 {
            best = (i, bound);
        }
    }
    Ok(best.0)
}

/// Count the root without building its held child `c`, when pricing says
/// the stream does not outweigh the build; `None` leaves `c` for the sweep
/// to build.
///
/// Every level the stream reads is built by now: `c`'s children and the
/// root's other child. Their counts are folded as a model count of the
/// finished diagram would fold them, and must all fit `u64`, which is also
/// declined otherwise, as is a read level the sweep left marginal.
pub(super) fn count_streamed_root(
    eng: &Engine,
    run: &mut ApplyRun,
    f: &Tdd,
    g: &Tdd,
    sweep: &Sweep<'_, '_>,
    root: (VtreeIdx, VtreeIdx, VtreeIdx),
    c: (VtreeIdx, VtreeIdx, VtreeIdx),
) -> Result<Option<num_bigint::BigUint>, OperationError> {
    use crate::value::{CountVec, FoldInput, IntFold, Retention, ValueDomain};
    let vtree = sweep.vtree;
    let (t, root_left, root_right) = root;
    let (c_t, cl, cr) = c;
    let c_left = c_t == root_left;
    let o = if c_left { root_right } else { root_left };
    if [o, cl, cr].iter().any(|x| run.levels[x.idx()].is_marginal()) {
        return Ok(None);
    }
    for x in [o, cl, cr] {
        run.ensure_product_list_for_child(eng, x.idx(), run.f_widths[x.idx()], run.g_widths[x.idx()])?;
    }
    let lim = eng.limits();
    // Priced on the index sizes alone, before any count is folded: a root
    // that builds `c` after all folds its own columns.
    let read = StreamLevels { root: (t, c_left), c, o };
    let Some(pivot) = choose_pivot(lim, &read.input(run, f, g, [&[]; 3]))? else {
        return Ok(None);
    };

    let mut computed: Vec<Option<CountVec>> = Vec::new();
    lim.reserve_exact(&mut computed, vtree.num_nodes())?;
    computed.resize_with(vtree.num_nodes(), || None);
    let levels: &[TddLevel] = run.levels;
    let marginal = |i: usize| levels[i].is_marginal();
    let input = FoldInput { vtree, levels, store: &() };
    let mut gate = lim.gate();
    for x in [o, cl, cr] {
        IntFold::ensure(eng, x, input, &mut computed, &marginal, Retention::Frontier, |w| gate.poll(w))?;
    }
    gate.flush()?;
    let column = |x: VtreeIdx| {
        let side = IntFold::child_view(x.idx(), vtree, &levels[x.idx()], &computed, &());
        (!side.view.is_marginal() && side.col.all_u64()).then(|| side.col.fast_slice())
    };
    let (Some(col_o), Some(col_cl), Some(col_cr)) = (column(o), column(cl), column(cr)) else {
        return Ok(None);
    };
    stream_count(eng, &read.input(run, f, g, [col_o, col_cl, col_cr]), pivot).map(Some)
}

/// The levels a streamed root count reads: the root with whether `c` is its
/// left child, `c` with its children, and the root's other child.
struct StreamLevels {
    root: (VtreeIdx, bool),
    c: (VtreeIdx, VtreeIdx, VtreeIdx),
    o: VtreeIdx,
}

impl StreamLevels {
    /// The stream's input, with `counts` for `o`, `cl` and `cr` (empty for
    /// pricing, which reads none).
    fn input<'a>(&self, run: &'a ApplyRun, f: &'a Tdd, g: &'a Tdd, counts: [&'a [u128]; 3]) -> StreamInput<'a> {
        let ((t, c_left), (c_t, cl, cr), o) = (self.root, self.c, self.o);
        let built = |x: VtreeIdx, counts| Built { products: run.products.list(x.idx()), counts };
        StreamInput {
            f_root: f.level(t),
            g_root: g.level(t),
            f_c: f.level(c_t),
            g_c: g.level(c_t),
            c_left,
            o: built(o, counts[0]),
            cl: built(cl, counts[1]),
            cr: built(cr, counts[2]),
            f_widths: stream_widths(run.f_widths, c_t, o, cl, cr),
            g_widths: stream_widths(run.g_widths, c_t, o, cl, cr),
        }
    }
}

/// Give both children a dense grid and bump-allocate this level's own,
/// returning the base the cell build writes into.
///
/// The sparse-marginal route allocates only a reused `right_width`-row scratch instead of
/// the dense `left_width * right_width` slab: the row driver processes one structural row at a
/// time into it and records the surviving cells in the output product list, so
/// the slab is never materialized and the grandparent densifies the level
/// lazily. That route's base is the scratch, not a slab base, which is why it
/// is returned rather than read back off the grid descriptor.
fn materialize_children_and_grid(
    eng: &Engine,
    run: &mut ApplyRun,
    shape: LevelShape,
    use_sparse_marginal: bool,
) -> Result<GridBase, OperationError> {
    let LevelShape { t, left, right, f: fw, g: gw } = shape;
    let (ti, li, ri) = (t.idx(), left.idx(), right.idx());
    // ── Dense path: ensure children have grids ───────────────────
    //
    // Only when the arena bumps: a child processed by the sparse pipeline has
    // no grid, so materialize one. On this path the child is known ungridded —
    // there is nothing to scan — so the identity fast path is the only way to
    // build its product list.
    if run.products.arena.is_bump() {
        if run.products.arena.is_sparse(li) {
            run.materialize_dense_child(eng, li, fw.left, gw.left)?;
        }
        if run.products.arena.is_sparse(ri) {
            run.materialize_dense_child(eng, ri, fw.right, gw.right)?;
        }
    }

    // Dense output owns an f-by-g slab. Sparse marginal output reuses one
    // g-width row, records live products separately, and remains ungridded
    // until its parent needs a dense view. Only a bump arena uses that route;
    // a preplanned arena already owns the full slab.
    let cells = if use_sparse_marginal { gw.here } else { fw.here * gw.here };
    let base = run.products.arena.alloc(eng, ti, cells)?;
    if use_sparse_marginal {
        run.products.arena.set_sparse(ti);
    } else {
        run.products.arena.set_dense(ti, base);
    }
    Ok(base)
}


/// Pre-size the level's two arenas and arm its emit-growth policy.
///
/// Both reserves are sized from an exact upper bound and then capped at
/// [`LEVEL_RESERVE_CAP_BYTES`]: most levels are low-survival, so an uncapped
/// reserve would routinely grab orders of magnitude more than the level ends up
/// using and charge every byte of it. A level that outgrows the cap keeps
/// growing through the ordinary fallible push path, and `finalize_level`'s
/// `shrink_arrays` hands the unused tail back. Both are fallible: under a tight
/// budget even the capped reservation may not fit.
///
/// A streaming route folds counts and writes neither arena, so it reserves
/// nothing and arms no growth mode. Called exactly once per level on every
/// route, so a previous level's near-cap decision cannot leak into this one.
/// Inlined at both of [`build_level_dense`]'s sites, each a level's one call.
#[inline(always)]
pub(super) fn open_level_arenas(
    lim: &crate::limits::Limits,
    f: &Tdd,
    g: &Tdd,
    shape: LevelShape,
    level: &mut TddLevel,
    route: Route,
) -> Result<(), OperationError> {
    let emits_pairs = matches!(
        route,
        Route::MarginalChild | Route::PlainDense | Route::Dense | Route::SparseMarg
    );
    if !emits_pairs {
        debug_assert!(matches!(route, Route::Stream { .. }), "the sparse route opens no arena here");
        lim.begin_level(None);
        return Ok(());
    }
    let (t, left_width, right_width) = (shape.t, shape.f.here, shape.g.here);
    // One node per live cell, and compaction only removes dead ones, so
    // `left_width * right_width` is an exact bound.
    let nodes_reserve = left_width
        .saturating_mul(right_width)
        .min(LEVEL_RESERVE_NODES_CAP)
        .max(left_width.max(right_width));
    lim.reserve(level.nodes.stored_mut(), nodes_reserve)?;

    // The one per-level emit-pair bound. Every product pair emits at most
    // once, and a product node of one pair is held inline, not in the
    // arena: the cell of two one-pair nodes writes none there. An operand
    // level holds `a` pairs in its arena, its multi-pair nodes', and the
    // pairs of at most `n` one-pair nodes inline, so this level's arena
    // takes at most `(af + nf)(ag + ng) - nf·ng = af·(ag + ng) + nf·ag`
    // pairs. A bound of the arenas alone, `af·ag`, was zero against a level
    // of one-pair nodes, which seeded nothing and left the arena to double
    // from empty. Used twice — once to pick the growth mode, once to size
    // the pairs arena — computed once so the two can never disagree.
    let pairs = |l: &TddLevel| (l.pairs.len() as u128, l.node_count() as u128);
    let ((af, nf), (ag, ng)) = (pairs(f.level(t)), pairs(g.level(t)));
    let emit_pair_bound = af.saturating_mul(ag.saturating_add(ng)).saturating_add(nf.saturating_mul(ag));
    lim.begin_level(Some(emit_pair_bound));
    // Seed `level.pairs` at that bound instead of letting it double from
    // empty on every level; the emit's own `try_push_pair_into` choke
    // point, under the growth mode just armed, carries a level that
    // outgrows the cap.
    let pairs_reserve =
        emit_pair_bound.min(LEVEL_RESERVE_PAIRS_CAP as u128) as usize;
    let pre_pairs_cap = level.pairs.capacity();
    lim.reserve(level.pairs.stored_mut(), pairs_reserve)?;
    // Output-pair meter: this bulk seed is real arena capacity the
    // emit walk will not charge again.
    lim.charge_output_pairs(level.pairs.capacity().saturating_sub(pre_pairs_cap));
    Ok(())
}

/// Build every cell of this level on the row loop its route names.
///
/// A streaming marginalization target folds each cell to a value instead of
/// materializing product nodes; [`Route::MarginalChild`] runs the shared cell
/// kernel with `MarginalLookup` sides; [`Route::Dense`] and
/// [`Route::PlainDense`] assume no marginal child and read both child grids
/// positionally.
#[expect(clippy::too_many_arguments)]
fn run_row_loop(
    eng: &Engine,
    route: Route,
    lookups: PlainLookups,
    rows: RowLoop<'_>,
    scratch: RowScratch<'_>,
    level: &mut TddLevel,
    env: StreamEnv<'_>,
    stream_state: &mut Option<StreamLevelState>,
) -> Result<(), OperationError> {
    let cell_ctx = rows.ctx;
    if lookups != PlainLookups::Grid {
        debug_assert!(matches!(route, Route::PlainDense | Route::Dense), "only the plain routes read a side by arithmetic");
        return run_level_rows_complete(eng, rows, scratch, level, lookups);
    }

    macro_rules! stream_rows {
        ($l:expr, $r:expr) => {
            run_level_rows_stream_count(
                eng,
                rows, scratch,
                $l, $r,
                stream_state.as_mut().expect("Route::Stream implies an open stream column"),
                env,
            )?
        };
    }
    macro_rules! plain_rows {
        ($dense:literal) => {
            run_level_rows_plain::<$dense>(eng, rows, scratch, level)?
        };
    }

    match route {
        // A streaming marginalization target collapses to Σ left × right per cell:
        // nothing downstream survives, so build each alive cell's scalar from
        // the surviving refs and never materialize a product node. Which side
        // carries counts is what picks the lookups: a marginal side is read
        // through `MarginalLookup`, which decodes a count payload or degrades
        // to a dense grid read; structural sides are read positionally.
        Route::Stream { marginal_children: true } => {
            let left = child_lookup::MarginalLookup::new(&cell_ctx.sides.left);
            let right = child_lookup::MarginalLookup::new(&cell_ctx.sides.right);
            stream_rows!(&left, &right)
        }
        Route::Stream { marginal_children: false } => {
            let left = child_lookup::DenseLookup { base: cell_ctx.sides.left.base, stride: cell_ctx.sides.left.stride };
            let right = child_lookup::DenseLookup { base: cell_ctx.sides.right.base, stride: cell_ctx.sides.right.stride };
            stream_rows!(&left, &right)
        }
        // A materializing level with at least one marginal child or
        // pass-through side. It runs the same per-cell kernel as the
        // structural routes, with `MarginalLookup` sides; isolating it here is
        // what lets those routes assume no child is marginal and no side is
        // carried through. Refs come out in the encoding the structural path
        // produces, so this level still relies on the end-of-apply tagger.
        // Both-marginal levels route here too — pure Σ left × right with no
        // structural product — and carrying them here keeps them off the
        // inline tagger that would corrupt them.
        Route::MarginalChild => run_level_rows_marginal(eng, rows, scratch, level)?,
        // The four level-invariant guards all hold, so the simplified emit
        // runs: no streaming, no dead-pair masks, no pass-through side.
        Route::PlainDense => plain_rows!(true),
        // Dead-pair masks only: the positional lookups carry nothing through.
        Route::Dense => {
            debug_assert!(
                !cell_ctx.sides.left.plan.is_passthrough() && !cell_ctx.sides.right.plan.is_passthrough(),
                "a pass-through side takes the marginal-child build"
            );
            plain_rows!(false)
        }
        Route::Sparse | Route::SparseMarg => {
            unreachable!("the sparse routes build their level before the row loop")
        }
    }
    Ok(())
}

/// Build this level's dead-pair liveness masks, which need the children's
/// grids materialized and so cannot be computed with the rest of the marginal plan.
///
/// # Errors
///
/// Propagates a refused reservation for the mask buffers.
fn build_level_prefilter_masks(
    eng: &Engine,
    run: &mut ApplyRun,
    g: &Tdd,
    shape: LevelShape,
    plan: &MarginalPlan,
    bases: Sides<GridBase>,
) -> Result<(), OperationError> {
    let LevelShape { t, f: fw, g: gw, .. } = shape;
    let right_level = g.level(t);
    build_side_masks::<false>(
        eng, right_level, gw.here,
        ChildGrid { plan: plan.sides.left, f_width: fw.left, g_width: gw.left, base: bases.left.idx() },
        run.products.arena.slab(), &mut run.prefilter_masks.left,
    )?;
    build_side_masks::<true>(
        eng, right_level, gw.here,
        ChildGrid { plan: plan.sides.right, f_width: fw.right, g_width: gw.right, base: bases.right.idx() },
        run.products.arena.slab(), &mut run.prefilter_masks.right,
    )
}

/// The run buffers [`finish_sparse_marginal_level`] writes, borrowed field by
/// field: the output level is already split out of the same `ApplyRun`.
struct SparseMargScratch<'a> {
    f_pairs: &'a mut Vec<ChildPair>,
    g_pairs: &'a mut Vec<ChildPair>,
    products: &'a mut super::super::products::Products,
}

/// Build a [`Route::SparseMarg`] level and close it out.
///
/// Runs the shared emit kernel, but writes into the reused `right_width`-row scratch —
/// `cell_ctx.output_grid_base` serves directly as the row base and the action's
/// `grid_row` is 0, so a grid position is just `row_base + j` — and records each surviving cell in
/// the output product list instead of a dense slab. The scratch goes back
/// immediately: the level is tagged sparse, its product list is the
/// authoritative representation, and the grandparent densifies it lazily.
///
/// This route returns before `finalize_level`, so it does that tail's
/// inline-emit marking itself. Exactly one side is the marginal pass-through.
///
/// # Errors
///
/// Propagates a refused reservation from the row driver.
fn finish_sparse_marginal_level(
    eng: &Engine,
    shape: LevelShape,
    rows: RowLoop<'_>,
    output_grid_base: GridBase,
    level: &mut TddLevel,
    scratch: SparseMargScratch<'_>,
    passthrough: Sides<bool>,
) -> Result<(), OperationError> {
    let LevelShape { t, g: gw, .. } = shape;
    let SparseMargScratch {
        f_pairs, g_pairs, products,
    } = scratch;
    let (node_idx, product_list) = products.row_buffers(t.idx());
    run_level_rows_marginal_sparse(
        eng,
        rows,
        RowScratch { f_pairs, g_pairs, node_idx },
        level,
        product_list,
    )?;
    products.arena.free(output_grid_base, gw.here);
    products.finish_sparse(level, t.idx());
    // The rows were built in order, so the product list numbers the live
    // cells in cell order, as a dense emit would.
    products.note_built(t.idx(), shape.f.here, gw.here, level.nodes().len());
    mark_passthrough_inlined(level, passthrough);
    Ok(())
}


/// Gather one level's per-cell context: grid geometry, pass-through carriers,
/// the dead-pair liveness masks, and the resolved g column table.
///
/// `right_cols` resolves every g column once for the level, so `process_cell`
/// indexes the table instead of re-deriving column j's slice on every row. On
/// identity-mask levels the descriptors are zero-copy borrows of g's own
/// storage; on marginal-mask levels they point into a decode arena the table owns
/// and budget-charges. `None` — marginal-encoded g, the budget refusing the
/// arena, or a level whose rows read its columns too few times again
/// ([`column_table_pays`]) — falls back to the per-cell derivation.
fn build_cell_ctx<'a>(
    shape: LevelShape,
    plan: &MarginalPlan,
    output_grid_base: usize,
    bases: Sides<GridBase>,
    masks: Option<&'a crate::apply::conjoin::liveness::PrefilterMaskScratch>,
    right_cols: Option<&'a RightColumns>,
) -> CellCtx<'a> {
    // A level that built no masks hands the kernel empty ones, so a read of
    // a mask it did not build is out of bounds rather than stale.
    let side = |plan: SidePlan, base: usize, stride: usize, masks: Option<&'a liveness::PrefilterSideMasks>| ChildPlan {
        plan, base, stride: stride as u32,
        live_cols: masks.map_or(&[], |m| &m.live_cols),
        reach: masks.map_or(&[], |m| &m.reach),
    };
    CellCtx {
        output_grid_base,
        right_width: shape.g.here,
        masked: masks.is_some(),
        sides: Sides {
            left: side(plan.sides.left, bases.left.idx(), shape.g.left, masks.map(|m| &m.left)),
            right: side(plan.sides.right, bases.right.idx(), shape.g.right, masks.map(|m| &m.right)),
        },
        right_cols,
    }
}

/// The column reads past each column's first that a level's cells must make
/// before the column table ([`RightColumns`]) costs less than resolving a
/// column at every cell that reads it.
const COLUMN_TABLE_MIN_REREADS: usize = 32;

/// Whether a level builds its column table: where its rows read the columns
/// again often enough to pay for it, and wherever a cell could take the
/// grouped walk, which reads the table's runs. That walk needs a row and a
/// column of [`GROUPED_MIN_PAIRS`] pairs, which only an operand level of
/// that many pairs can hold, so a level without the table writes its pairs
/// in the order it would with one.
#[inline(always)]
fn column_table_pays(f_level: &TddLevel, g_level: &TddLevel, f_width: usize, g_width: usize, grouped: bool) -> bool {
    let groups = grouped && f_level.pairs.len() >= GROUPED_MIN_PAIRS && g_level.pairs.len() >= GROUPED_MIN_PAIRS;
    groups || f_width.saturating_sub(1).saturating_mul(g_width) >= COLUMN_TABLE_MIN_REREADS
}

/// How a level on `route` reads its child sides: by arithmetic on each
/// [complete](crate::apply::conjoin::products::Products::is_complete) side
/// of a plain route, from the grid otherwise. The pair capacity of a level
/// with both sides complete is filled in once its arenas are open
/// ([`reserve_complete_level`]).
fn plain_lookups(route: Route, complete: Sides<bool>) -> PlainLookups {
    if grid_lookups_forced() || !matches!(route, Route::PlainDense | Route::Dense) {
        return PlainLookups::Grid;
    }
    match (complete.left, complete.right) {
        (true, true) => PlainLookups::Complete { charged: usize::MAX },
        (true, false) => PlainLookups::CompleteLeft,
        (false, true) => PlainLookups::CompleteRight,
        (false, false) => PlainLookups::Grid,
    }
}

/// Reserve the arenas of a level whose two child sides are complete to
/// exactly what its row loop writes, and return the pair capacity the
/// output-pair meter has been charged for (see
/// [`ReservedEmitSink`](crate::apply::conjoin::cell::ReservedEmitSink)).
///
/// No candidate dies on such a level, so every cell of a node of `f` with
/// pairs and a node of `g` with pairs is one node, holding the product of
/// their pair counts; a cell of two one-pair nodes stores its pair inline
/// and every other cell stores its pairs in the arena. Both counts take one
/// pass over the two operand levels' nodes.
///
/// Under the bounded-growth mode the arena grows as it would on the grid
/// route, and `usize::MAX` says so. The reservations are charged to the
/// budget like any other and refuse with [`OperationError::OverBudget`].
/// Inlined at both of [`build_level_dense`]'s sites, as its count of each
/// operand level is.
#[inline(always)]
fn reserve_complete_level(
    lim: &crate::limits::Limits,
    f_level: &TddLevel,
    g_level: &TddLevel,
    level: &mut TddLevel,
) -> Result<usize, OperationError> {
    if lim.bounded_growth() {
        return Ok(usize::MAX);
    }
    let charged = level.pairs.capacity();
    /// Pairs in all, nodes with pairs, and nodes with one pair: an implicit
    /// level's off its description, every node of which has its `k` pairs,
    /// and a stored one's off its nodes ([`TddLevel::pair_census`]).
    #[inline(always)]
    fn census(level: &TddLevel) -> (u128, u128, u128) {
        if let Some(d) = level.implicit() {
            let (nodes, k) = (d.nodes() as u128, d.pairs_per_node() as u128);
            return (nodes * k, if k > 0 { nodes } else { 0 }, if k == 1 { nodes } else { 0 });
        }
        let (pairs, live, single) = level.pair_census();
        (u128::from(pairs), u128::from(live), u128::from(single))
    }
    let (f_pairs, f_live, f_single) = census(f_level);
    let (g_pairs, g_live, g_single) = census(g_level);
    let fit = |n: u128| usize::try_from(n).map_err(|_| OperationError::OverBudget);
    let nodes = fit(f_live * g_live)?;
    let pairs = fit(f_pairs * g_pairs - f_single * g_single)?;
    lim.reserve_exact(level.nodes.stored_mut(), nodes)?;
    lim.reserve_exact(level.pairs.stored_mut(), pairs)?;
    Ok(charged)
}

/// Build one level on the dense product grid: route plan, child grids, cell
/// context, emit-growth mode, the row loop, and the per-level tail.
pub(super) fn build_level_dense(
    eng: &Engine,
    run: &mut ApplyRun,
    f: &mut Tdd,
    g: &mut Tdd,
    level: LevelBuild,
    sweep: &mut Sweep<'_, '_>,
) -> Result<(), OperationError> {
    let lim = eng.limits();
    let LevelBuild { shape, route, plan } = level;
    let LevelShape { t, left, right, f: fw, g: gw } = shape;
    let (ti, li, ri) = (t.idx(), left.idx(), right.idx());
    let MarginalPlan { sides, both_multi_pair } = plan;
    let passthrough = Sides { left: sides.left.is_passthrough(), right: sides.right.is_passthrough() };
    let use_sparse_marginal = route == Route::SparseMarg;
    // Materialize any sparse child grid and bump-allocate this level's own.
    // The route is already known, which is what lets this skip `ensure_grid`
    // for a sparse child on a level that will take the plain-dense emit.
    let output_grid_base = materialize_children_and_grid(eng, run, shape, use_sparse_marginal)?;

    // Child grid geometry: product `(a, b)` sits at `base + a * g_child_width + b`.
    let bases = Sides {
        left: run.products.arena.materialized(li).expect("the left child's grid is materialized"),
        right: run.products.arena.materialized(ri).expect("the right child's grid is materialized"),
    };

    let mut lookups = plain_lookups(
        route,
        Sides { left: run.products.is_complete(li), right: run.products.is_complete(ri) },
    );
    // Only the grid-reading dead-pair liveness masks are deferred this far: they need
    // the materialized child grids, and `both_multi_pair` implies a route that has them.
    // With both sides complete no mask can clear, and the row loop reads none; on a
    // grid of fewer than `MASK_MIN_CELLS` cells they cost more than they cull.
    let masked = both_multi_pair
        && !matches!(lookups, PlainLookups::Complete { .. })
        && fw.here.saturating_mul(gw.here) >= liveness::MASK_MIN_CELLS;
    if masked {
        build_level_prefilter_masks(eng, run, g, shape, &plan, bases)?;
    }

    let mut stream_state: Option<StreamLevelState> =
        build_stream_state(eng, shape, run.levels, run.stream_cache, sweep)?;

    let grouped = both_multi_pair && !passthrough.left && !passthrough.right;
    // A level of two complete sides whose operand levels are affine takes
    // the implicit route; any other reads its operands' pairs, an implicit
    // operand's generated from its description.
    let composed = (matches!(lookups, PlainLookups::Complete { .. })
        && !use_sparse_marginal
        && stream_state.is_none()
        && sweep.filter.is_none()
        && !crate::diagram::stored_levels_forced())
    .then(|| super::compose::plan(lim, f.level(t), g.level(t), shape, grouped))
    .flatten();
    // The arenas the implicit route opened before it found it would have to
    // stop inside the level, or grow the arena in bounded steps, which the
    // row loop does on the written pairs: the capacity the meter was charged
    // for. The planned route runs only with nothing bounding memory, so the
    // column table built after the arenas changes nothing either reads.
    // Both routes end in the one per-level tail below the block.
    let mut opened = None;
    'routes: {
        if let Some(product) = composed {
            let level = &mut run.levels[ti];
            open_level_arenas(lim, f, g, shape, level, route)?;
            let charged = reserve_complete_level(lim, f.level(t), g.level(t), level)?;
            note_lookups(PlainLookups::Complete { charged });
            let (work, meter, doublings) = super::compose::charges((fw.here, gw.here), &product, charged);
            if charged != usize::MAX && lim.cannot_stop_within(work, meter as u64) {
                let cells = fw.here * gw.here;
                let slab = &mut run.products.arena.slab_mut()[output_grid_base.idx()..output_grid_base.idx() + cells];
                super::compose::write(eng, product, &mut run.levels[ti], slab, work, (meter, doublings))?;
                break 'routes;
            }
            opened = Some(charged);
        }

        // `t` and its two vtree children are three distinct tree nodes, so these
        // are three disjoint level slots: the streaming row loops read the child
        // columns in place while the output level is exclusively borrowed. The
        // split's borrow ends with the block, before the per-level tail retakes
        // `levels`.
        let [level, left_level, right_level] = run.levels
            .get_disjoint_mut([ti, li, ri])
            .expect("a vtree node and its two children are distinct level indices");
        let (left_level, right_level) = (&*left_level, &*right_level);

        let right_cols = column_table_pays(f.level(t), g.level(t), fw.here, gw.here, grouped)
            .then(|| RightColumns::build(eng, g.level(t), gw.here, sides.left.view, sides.right.view, grouped))
            .flatten();
        let masks = masked.then_some(&*run.prefilter_masks);
        let cell_ctx = build_cell_ctx(shape, &plan, output_grid_base.idx(), bases, masks, right_cols.as_ref());

        match opened {
            Some(charged) => lookups = PlainLookups::Complete { charged },
            None => {
                open_level_arenas(lim, f, g, shape, level, route)?;
                if let PlainLookups::Complete { charged } = &mut lookups {
                    *charged = reserve_complete_level(lim, f.level(t), g.level(t), level)?;
                }
                note_lookups(lookups);
            }
        }

        if use_sparse_marginal {
            return finish_sparse_marginal_level(
                eng, shape,
                RowLoop {
                    f_level: f.level(t), g_level: g.level(t),
                    children: Sides { left: left_level, right: right_level },
                    ctx: &cell_ctx, f_width: fw.here,
                },
                output_grid_base, level,
                SparseMargScratch {
                    f_pairs: run.f_pairs_scratch,
                    g_pairs: run.g_pairs_scratch,
                    products: run.products,
                },
                passthrough,
            );
        }

        run_row_loop(
            eng, route, lookups,
            RowLoop {
                f_level: f.level(t), g_level: g.level(t),
                children: Sides { left: left_level, right: right_level },
                ctx: &cell_ctx, f_width: fw.here,
            },
            RowScratch {
                f_pairs: run.f_pairs_scratch,
                g_pairs: run.g_pairs_scratch,
                node_idx: run.products.arena.slab_mut(),
            },
            level,
            StreamEnv {
                left_idx: li,
                right_idx: ri,
                vtree: sweep.vtree,
                cache: run.stream_cache,
                ws: sweep.ws.as_deref(),
            },
            &mut stream_state,
        )?;
    }

    // Per-level tail: stream commit, live_counts, grid tag, shrink,
    // pass-through flags. See `finalize_level`.
    finalize_level(eng, stream_state, shape, output_grid_base, passthrough, run, sweep);
    Ok(())
}

#[cfg(test)]
#[path = "tests/level.rs"]
mod tests;
