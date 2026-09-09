//! The bottom-up fold that freezes scheduled levels, in either value domain.

use crate::diagram::Changed;
use crate::value_fold::ColumnRetention;
use crate::diagram::{assert_can_make_marginal, Tdd};
use crate::engine::Engine;
use crate::engine::PollGate;
use crate::error::ApplyError;
use crate::vtree::{Vtree, VtreeIdx};
use crate::diagram::WeightStore;

use super::kind::{Column, InternalLevel, IntValues, ValueKind, WeightValues};
use super::store::{free_subsumed_marginal_children, remap_parent_refs_pretag};

/// Freeze `targets` into per-node model counts.
///
/// `targets` must be sorted bottom-up so that each level's children are
/// already frozen (or are leaves) by the time it is reached.
///
/// # Errors
///
/// Returns `Err(ApplyError::Deadline)` if the caller's wall passed while the
/// pass was running and the post-apply poll is armed. See [`freeze_targets`]
/// for what the diagram looks like after a cut.
pub(crate) fn marginalize_batch(
    eng: &Engine,
    tdd: &mut Tdd,
    targets: &[VtreeIdx],
    vtree: &Vtree,
) -> Result<(), ApplyError> {
    // The integer readers read the level's own `marginal_counts`, which a
    // weight-marginal level does not have. Every call site must route a
    // weighted diagram to the weighted pass instead; asserted here so a future
    // bypass fails at the entry point rather than deep inside a reader.
    debug_assert!(
        tdd.weights.is_none(),
        "the integer pass cannot run on a diagram carrying a weight store"
    );
    freeze_targets::<IntValues>(eng, tdd, targets, vtree, &mut ())
}

/// Freeze `targets` into per-node exact semiring values in `ws`.
///
/// # Errors
///
/// As [`marginalize_batch`].
pub(crate) fn marginalize_batch_weighted(
    eng: &Engine,
    tdd: &mut Tdd,
    targets: &[VtreeIdx],
    vtree: &Vtree,
    ws: &mut WeightStore,
) -> Result<(), ApplyError> {
    freeze_targets::<WeightValues>(eng, tdd, targets, vtree, ws)
}

/// Freeze every target in order, then sum out the leaf targets.
///
/// The pass's ONE preemption point sits between targets, amortized. This walk
/// is where a leaf compile forgets its variables, and on a near-root step it is
/// minutes of folding with no return to the caller, so without it the grant is
/// observed only at the step seam past it. It is metered in nodes of the target
/// level — the unit the fold, the dedup and the parent remap all scale with —
/// and with no stop axis installed it is an add and three cell loads per target.
///
/// Cutting BETWEEN targets and never inside one is what makes the cut clean:
/// each iteration freezes exactly one level and settles the diagram around it,
/// so the prefix already done is a complete pass of its own once the domain's
/// end sweep has run over it — which is why the cut still runs that sweep
/// before returning the error.
///
/// The leaf targets are summed out LAST. The integer domain requires it: its
/// end sweep keys off the pass-entry snapshot, so a leaf flipped frozen earlier
/// would have its side re-resolved as bare slots, misreading the inline
/// references. The weighted domain has no such hazard, but the order is still
/// the right one — a leaf frozen before its internal parent in the same pass
/// would have its column installed and immediately freed again by the parent's
/// subsumed-child reclaim.
fn freeze_targets<K: ValueKind>(
    eng: &Engine,
    tdd: &mut Tdd,
    targets: &[VtreeIdx],
    vtree: &Vtree,
    store: &mut K::Store,
) -> Result<(), ApplyError> {
    if targets.is_empty() {
        return Ok(());
    }
    let lim = eng.limits();
    let was_frozen: Vec<bool> = tdd.levels.iter().map(|l| l.is_marginal()).collect();
    // One column per vtree level, built on demand. (`CountVec` is deliberately
    // not `Clone`, so the None-filled buffer cannot use `vec![None; n]`.)
    let mut computed: Vec<Option<Column<K>>> = (0..vtree.num_nodes()).map(|_| None).collect();
    let mut poll = PollGate::new(lim.reduce_poll_stride());

    let mut cut = None;
    for &d in targets {
        if let Err(e) = lim.poll(&mut poll, tdd.levels[d.idx()].width() as u64 + 1) {
            cut = Some(e);
            break;
        }
        freeze_level::<K>(eng, tdd, d, vtree, store, &mut computed);
    }

    K::end_sweep(tdd, &was_frozen);
    if let Some(e) = cut {
        return Err(e);
    }
    for &d in targets {
        if vtree.node(d).is_leaf() {
            K::sum_out_leaf(eng, tdd, d, vtree, store);
        }
    }
    Ok(())
}

/// Freeze one internal level: fold its per-node values, freeze the levels
/// beneath it, and install the result. A no-op on a leaf, an empty level, or
/// one that is already frozen.
fn freeze_level<K: ValueKind>(
    eng: &Engine,
    tdd: &mut Tdd,
    d: VtreeIdx,
    vtree: &Vtree,
    store: &mut K::Store,
    computed: &mut [Option<Column<K>>],
) {
    let di = d.idx();
    let Some(level) = InternalLevel::new(vtree, d) else {
        return; // a leaf target is summed out at the end of the pass instead
    };
    if tdd.levels[di].is_marginal() || tdd.levels[di].width() == 0 {
        return;
    }

    let (left, right) = vtree.children(d);
    K::ensure(eng, tdd, left, vtree, store, computed, ColumnRetention::All);
    K::ensure(eng, tdd, right, vtree, store, computed, ColumnRetention::All);

    let width = tdd.levels[di].width();
    let mut col = K::alloc_column(eng, width, store);
    for (i, _pairs) in tdd.levels[di].internal_inputs_iter() {
        let v = K::fold_node(tdd, &tdd.levels[di], i, left.idx(), right.idx(), store, computed);
        K::set_slot(eng, &mut col, i, v);
    }

    // Park the column where the cascade below can reach it (uncompacted,
    // indexed by node index), then take it back for the install. Moved, not
    // copied: the column has a single owner across the whole window, so a
    // width-sized duplicate would be pure peak memory.
    computed[di] = Some(col);

    // Freeze the children FIRST (bottom-up), so that by the time `d` is frozen
    // both of them are frozen or are leaves — the `assert_can_make_marginal`
    // precondition.
    cascade::<K>(tdd, vtree, left, store, computed);
    cascade::<K>(tdd, vtree, right, store, computed);

    let col = computed[di].take().expect("the column was just computed for this level");
    freeze::<K>(tdd, vtree, level, col, store);
    // Nothing re-fills `computed[di]`: once `d` is frozen every reader takes
    // the frozen branch and reads the installed store. The buffer accumulates
    // across all of the pass's targets, so holding the uncompacted column past
    // this point would raise the pass's cumulative peak, not just a transient.
}

/// Walk down from a level whose parent is being frozen, freezing every
/// still-explicit internal descendant from the columns the ensure walk cached.
fn cascade<K: ValueKind>(
    tdd: &mut Tdd,
    vtree: &Vtree,
    t: VtreeIdx,
    store: &mut K::Store,
    computed: &mut [Option<Column<K>>],
) {
    let Some(level) = InternalLevel::new(vtree, t) else {
        return;
    };
    if tdd.levels[t.idx()].is_marginal() {
        return;
    }
    // Children first, so they are frozen (or leaves) by the time `t` is.
    let (l_child, r_child) = vtree.children(t);
    cascade::<K>(tdd, vtree, l_child, store, computed);
    cascade::<K>(tdd, vtree, r_child, store, computed);

    let Some(col) = computed[t.idx()].take() else {
        // No cached column: the ancestor's fold never visited here, so this
        // level's cells are structurally unreachable from the target's pair
        // lists and will never be queried.
        return;
    };
    freeze::<K>(tdd, vtree, level, col, store);
}

/// Install `col` as `t`'s frozen store and settle the diagram around it:
/// rewrite the parent's references if the domain minted new slots, mark the
/// parent for re-contraction, and free the children `t` now subsumes.
fn freeze<K: ValueKind>(
    tdd: &mut Tdd,
    vtree: &Vtree,
    level: InternalLevel,
    col: Column<K>,
    store: &mut K::Store,
) {
    let t = level.vtree_idx();
    assert_can_make_marginal(&tdd.levels, vtree, t);

    let parent = vtree.node(t).parent();
    if let Some(parent_vi) = parent {
        // The load-bearing seed is the boundary parent that stays explicit;
        // within a freezing subtree the parent usually freezes too, and
        // contraction then skips it harmlessly.
        tdd.invalidate(parent_vi, Changed::PAIRS);
    }

    let remap = K::install(tdd, level, col, store);

    // Only meaningful while the parent is still explicit — a frozen parent has
    // no pair lists to redirect.
    if let (Some(remap), Some(parent_vi)) = (remap, parent)
        && !tdd.levels[parent_vi.idx()].is_marginal() {
            let (pl, _) = vtree.children(parent_vi);
            remap_parent_refs_pretag(tdd, t, parent_vi, pl == t, &remap);
        }

    // `t` now subsumes its children — free their dead stores (O(1)).
    free_subsumed_marginal_children(tdd, vtree, t, K::weight_store(store));
}
