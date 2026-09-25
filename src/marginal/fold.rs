//! The bottom-up fold that marginalizes scheduled levels, in either value domain.

use crate::diagram::Tdd;
use crate::Engine;
use crate::limits::OperationError;
use crate::vtree::{Vtree, VtreeIdx};

use crate::value::{FoldInput, IntFold, Retention};
use super::transition::{InternalLevel, MarginalDomain, install_finished};

/// Marginalize `targets` into per-node model counts.
///
/// Internal targets sum out their descendants before storing their own values.
///
/// # Errors
///
/// Returns `Err(OperationError::Stopped)` if the caller's wall passed while the
/// pass was running and the post-apply poll is armed. See [`marginalize_targets`]
/// for what the diagram looks like after a cut.
pub(crate) fn marginalize_batch(
    eng: &Engine,
    tdd: &mut Tdd,
    targets: &[VtreeIdx],
    vtree: &Vtree,
) -> Result<(), OperationError> {
    // The integer readers read the level's own `marginal_counts`, which a
    // weight-marginal level does not have.
    debug_assert!(
        tdd.weights.is_none(),
        "the integer pass cannot run on a diagram carrying a weight store"
    );
    marginalize_targets::<IntFold>(eng, tdd, targets, vtree, &mut ())
}

/// Marginalize every target in order, then sum out the leaf targets.
///
/// The deadline is polled between targets, metered in nodes of the target
/// level, which the fold, the dedup and the parent remap all scale with. A cut
/// falls between targets, never inside one, and the domain's end sweep still
/// runs over the prefix before the error is returned, so the diagram left
/// behind is the one a pass over that prefix would have produced.
///
/// Leaf targets come last: the integer end sweep keys off the pass-entry
/// snapshot, so a leaf flipped marginal earlier would have its side
/// re-resolved as bare slots and its inline references misread.
pub(super) fn marginalize_targets<K: MarginalDomain>(
    eng: &Engine,
    tdd: &mut Tdd,
    targets: &[VtreeIdx],
    vtree: &Vtree,
    store: &mut K::Store,
) -> Result<(), OperationError> {
    if targets.is_empty() {
        return Ok(());
    }
    let lim = eng.limits();
    let was_marginal: Vec<bool> = tdd.levels.iter().map(|l| l.is_marginal()).collect();
    // One column per vtree level, built on demand. (`CountVec` is not `Clone`,
    // so the buffer cannot use `vec![None; n]`.)
    let mut computed: Vec<Option<K::Col>> = (0..vtree.num_nodes()).map(|_| None).collect();
    let mut poll = lim.gate();

    let mut cut = None;
    for &d in targets {
        if let Err(e) = poll.poll(tdd.levels[d.idx()].slot_count() as u64 + 1) {
            cut = Some(e);
            break;
        }
        if let Err(e) = marginalize_level::<K>(eng, tdd, d, vtree, store, &mut computed) {
            cut = Some(e);
            break;
        }
    }

    K::end_sweep(tdd, &was_marginal);
    if let Some(e) = cut {
        return Err(e);
    }
    for &d in targets {
        if vtree.node(d).is_leaf() {
            K::sum_out_leaf(tdd, d, vtree, store);
        }
    }
    Ok(())
}

/// Marginalize one internal level: fold its per-node values, marginalize the levels
/// beneath it, and install the result. A no-op on a leaf, an empty level, or
/// one that is already marginal.
fn marginalize_level<K: MarginalDomain>(
    eng: &Engine,
    tdd: &mut Tdd,
    d: VtreeIdx,
    vtree: &Vtree,
    store: &mut K::Store,
    computed: &mut [Option<K::Col>],
) -> Result<(), OperationError> {
    let di = d.idx();
    let Some(level) = InternalLevel::new(vtree, d) else {
        return Ok(()); // a leaf target is summed out at the end of the pass instead
    };
    if tdd.levels[di].is_marginal() || tdd.levels[di].slot_count() == 0 {
        return Ok(());
    }

    let (left, right) = vtree.children(d);
    // `Retention::All` is not a choice here: the cascade takes every
    // walked level's column to install it as that level's store.
    ensure_below::<K>(eng, tdd, left, vtree, store, computed)?;
    ensure_below::<K>(eng, tdd, right, vtree, store, computed)?;

    let zero = K::zero(store);
    let col = K::fold_column(
        eng, d, FoldInput { vtree, levels: &tdd.levels, store },
        computed, &zero, |_| Ok(()),
    )?;

    // Marginalize the children before `d` (bottom-up), so that by the time `d` is marginal
    // both of them are marginal or are leaves — the `assert_can_make_marginal`
    // precondition.
    cascade::<K>(tdd, vtree, left, store, computed);
    cascade::<K>(tdd, vtree, right, store, computed);

    install_finished::<K>(tdd, vtree, level, col, store);
    Ok(())
}

/// Walk down from a level whose parent is being marginal, marginalizing every
/// still-explicit internal descendant from the columns the ensure walk cached.
fn cascade<K: MarginalDomain>(
    tdd: &mut Tdd,
    vtree: &Vtree,
    t: VtreeIdx,
    store: &mut K::Store,
    computed: &mut [Option<K::Col>],
) {
    let Some(level) = InternalLevel::new(vtree, t) else {
        return;
    };
    if tdd.levels[t.idx()].is_marginal() {
        return;
    }
    // Children first, so they are marginal (or leaves) by the time `t` is.
    let (l_child, r_child) = vtree.children(t);
    cascade::<K>(tdd, vtree, l_child, store, computed);
    cascade::<K>(tdd, vtree, r_child, store, computed);

    let Some(col) = computed[t.idx()].take() else {
        // No cached column: the ancestor's fold never visited here, so this
        // level's cells are structurally unreachable from the target's pair
        // lists and will never be queried.
        return;
    };
    install_finished::<K>(tdd, vtree, level, col, store);
}

/// Populate the column of `t` and everything below it that a fold at `t` will
/// read.
fn ensure_below<K: MarginalDomain>(
    eng: &Engine,
    tdd: &Tdd,
    t: VtreeIdx,
    vtree: &Vtree,
    store: &K::Store,
    computed: &mut [Option<K::Col>],
) -> Result<(), OperationError> {
    let marginal = |i: usize| tdd.levels[i].is_marginal();
    K::ensure(
        eng,
        t,
        FoldInput { vtree, levels: &tdd.levels, store },
        computed,
        &marginal,
        Retention::All,
        |_| Ok(()),
    )
}
