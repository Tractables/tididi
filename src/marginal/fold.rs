//! The bottom-up fold that marginalizes scheduled levels, in either value domain.

use crate::diagram::Tdd;
use crate::Engine;
use crate::limits::OperationError;
use crate::limits::PollGate;
use crate::vtree::{Vtree, VtreeIdx};

use crate::value::{FoldInput, IntFold, Retention};
use super::transition::{Finished, MarginalDomain, cascade};

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
    marginalize_targets::<IntFold>(eng, tdd, targets, vtree, &mut (), |_, _, _| Ok(()))
}

/// Marginalize every target in order, then sum out the leaf targets.
///
/// The deadline is polled between targets, metered in nodes of the target
/// level, which the fold, the dedup and the parent remap all scale with, and
/// inside a target's fold, metered in the nodes and pairs it reads below the
/// target. A cut inside a fold falls before anything is installed, so either
/// way the target is done whole or not at all, and the domain's end sweep
/// still runs over the prefix before the error is returned, so the diagram
/// left behind is the one a pass over that prefix would have produced. A
/// refused install ([`cascade`]) cuts the same way, past the levels the
/// cascade installed below it, which a pass targeting them would have
/// installed alike. A refused end sweep leaves the sides it had not reached
/// with bare slot references, which still read the same values, and a
/// refused leaf sum-out leaves that leaf and the leaf targets after it as
/// they were.
///
/// Leaf targets come last, after the end sweep: a leaf is never installed,
/// and summing it out inlines its parent's references to it.
///
/// `scale(i, column, store)` runs on the column of each internal target
/// `targets[i]` before it is installed.
pub(super) fn marginalize_targets<K: MarginalDomain>(
    eng: &Engine,
    tdd: &mut Tdd,
    targets: &[VtreeIdx],
    vtree: &Vtree,
    store: &mut K::Store,
    mut scale: impl FnMut(usize, &mut K::Col, &K::Store) -> Result<(), OperationError>,
) -> Result<(), OperationError> {
    if targets.is_empty() {
        return Ok(());
    }
    let lim = eng.limits();
    let mut made_marginal = Vec::new();
    // One column per vtree level, built on demand. (`CountVec` is not `Clone`,
    // so the buffer cannot use `vec![None; n]`.)
    let mut computed: Vec<Option<K::Col>> = (0..vtree.num_nodes()).map(|_| None).collect();
    let mut poll = lim.gate();

    let mut cut = None;
    for (i, &d) in targets.iter().enumerate() {
        if let Err(e) = poll.poll(tdd.levels[d.idx()].slot_count() as u64 + 1) {
            cut = Some(e);
            break;
        }
        let scale = |col: &mut K::Col, store: &K::Store| scale(i, col, store);
        let target = Finished { tdd: &mut *tdd, made_marginal: &mut made_marginal };
        if let Err(e) = marginalize_level::<K>(eng, target, d, vtree, (store, scale), &mut computed, &mut poll) {
            cut = Some(e);
            break;
        }
    }

    let swept = K::end_sweep(lim, tdd, &made_marginal);
    if let Some(e) = cut {
        return Err(e);
    }
    swept?;
    for &d in targets {
        if vtree.node(d).is_leaf() {
            K::sum_out_leaf(lim, tdd, d, vtree, store)?;
        }
    }
    Ok(())
}

/// Marginalize one internal level: fold its per-node values and those of the
/// structural levels beneath it, then install them bottom-up. A no-op on a
/// leaf, an empty level, or one that is already marginal.
fn marginalize_level<K: MarginalDomain>(
    eng: &Engine,
    mut target: Finished<'_>,
    d: VtreeIdx,
    vtree: &Vtree,
    (store, scale): (&mut K::Store, impl FnOnce(&mut K::Col, &K::Store) -> Result<(), OperationError>),
    computed: &mut [Option<K::Col>],
    poll: &mut PollGate<'_>,
) -> Result<(), OperationError> {
    let tdd = &*target.tdd;
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
        |units| poll.poll(units),
    )?;
    scale(computed[d.idx()].as_mut().expect("the target's column"), store)?;
    cascade::<K, Finished<'_>>(eng.limits(), &mut target, vtree, d, computed, store)
}
