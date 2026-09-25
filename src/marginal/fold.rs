//! The bottom-up fold that marginalizes scheduled levels, in either value domain.

use crate::diagram::Tdd;
use crate::Engine;
use crate::limits::OperationError;
use crate::vtree::{Vtree, VtreeIdx};

use crate::value::{FoldInput, IntFold, Retention};
use super::transition::{MarginalDomain, cascade};

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

/// Marginalize one internal level: fold its per-node values and those of the
/// structural levels beneath it, then install them bottom-up. A no-op on a
/// leaf, an empty level, or one that is already marginal.
fn marginalize_level<K: MarginalDomain>(
    eng: &Engine,
    tdd: &mut Tdd,
    d: VtreeIdx,
    vtree: &Vtree,
    store: &mut K::Store,
    computed: &mut [Option<K::Col>],
) -> Result<(), OperationError> {
    let level = &tdd.levels[d.idx()];
    // A leaf target is summed out at the end of the pass instead.
    if vtree.node(d).is_leaf() || level.is_marginal() || level.slot_count() == 0 {
        return Ok(());
    }
    // `Retention::All` is not a choice here: the cascade takes every
    // walked level's column to install it as that level's store.
    let marginal = |i: usize| tdd.levels[i].is_marginal();
    K::ensure(
        eng,
        d,
        FoldInput { vtree, levels: &tdd.levels, store },
        computed,
        &marginal,
        Retention::All,
        |_| Ok(()),
    )?;
    cascade::<K, Tdd>(tdd, vtree, d, computed, store);
    Ok(())
}
