//! Summing levels out and the epilogue restoring the marginal invariants.
//!
//! A marginalized level stops carrying pair structure and carries one value per
//! node instead: the number of assignments to its vtree subtree that reach that
//! node, or — with a [`Weights`](crate::diagram::Weights) attached — that node's weighted value. The
//! encoding a parent reads those values through is [`crate::diagram`]; the
//! reduction passes the epilogue calls are [`crate::reduce`]; counting over a
//! partly marginalized diagram is [`crate::query`].
//!
//! [`Tdd::marginalize`] sums out the subtrees over the variables it names
//! and restores invariants 7, 8 and 10 before it returns. Folding a weighted
//! diagram down to its value is [`Tdd::weighted_value`].

pub(crate) mod transition;
mod fold;
mod leaf;

use crate::Engine;
pub(crate) use fold::marginalize_batch;
pub(crate) use leaf::canonicalize_weighted_leaf_refs;
pub(crate) use leaf::{marginalize_leaf_inline, marginalize_leaf_weighted};

use crate::value::WeightFold;
use crate::limits::OperationError;
use crate::diagram::Tdd;
use crate::vtree::{VarId, Vtree, VtreeIdx, VtreeNode};
use crate::reduce::{restore_marginal_invariants, trim};

/// Marginalize every structural level whose two children are both marginal,
/// visiting affected parents until no level qualifies; returns the number of levels marginalized.
///
/// `rebuild_levels` never collapses a node to counts, so a rotation
/// that brings two marginal children together leaves a structural parent over
/// two marginal children, which is not a canonical marginal form. Each round
/// collects the bottom layer of such levels and marginalizes it; a freshly marginal level can complete a cluster one level
/// up, hence the loop. A diagram already in canonical form makes this a no-op.
///
/// # Errors
///
/// Passes through the fold's `Err(OperationError::Stopped)`. The
/// clusters closed before the cut stay closed; the rest are still structural
/// levels over two marginal children, which is the state this pass exists to
/// finish and a caller that resumes will find waiting for it.
pub(crate) fn marginalize_closure(eng: &Engine, tdd: &mut Tdd) -> Result<usize, OperationError> {
    let vtree = std::sync::Arc::clone(&tdd.vtree);
    let eligible = |tdd: &Tdd, t: VtreeIdx| {
        if vtree.node(t).is_leaf() || tdd.levels[t.idx()].is_marginal() || tdd.levels[t.idx()].slot_count() == 0 {
            return false;
        }
        let (l, r) = vtree.children(t);
        tdd.levels[l.idx()].is_marginal() && tdd.levels[r.idx()].is_marginal()
    };
    let mut targets: Vec<_> = vtree.bottomup().filter(|&t| eligible(tdd, t)).collect();
    let mut next = Vec::new();
    let mut total = 0usize;
    while !targets.is_empty() {
        targets.sort_unstable();
        targets.dedup();
        total += targets.len();
        evaluate_levels(eng, tdd, &targets, &vtree)?;
        for &t in &targets {
            if let Some(parent) = vtree.node(t).parent()
                && eligible(tdd, parent) {
                next.push(parent);
            }
        }
        targets.clear();
        std::mem::swap(&mut targets, &mut next);
    }
    Ok(total)
}

impl Engine {
    /// Sum out the variables `summable` names, under this engine's resource
    /// limits.
    ///
    /// See [`Tdd::marginalize`] for what is summed and the contract of
    /// `summable`.
    pub fn marginalize(&self, f: &mut Tdd, summable: impl Fn(VarId) -> bool) -> Result<(), OperationError> {
        marginalize(self, f, &summable)
    }
}

/// The one integer-vs-weighted dispatch of the pass: a weighted diagram's
/// targets are weight-marginal and carry no integer counts, so the integer
/// batch may not run on them.
fn evaluate_levels(eng: &Engine, f: &mut Tdd, levels: &[VtreeIdx], vtree: &Vtree) -> Result<(), OperationError> {
    match vtree.weights() {
        Some(ws) => fold::marginalize_targets::<WeightFold>(eng, f, levels, vtree, ws),
        None => marginalize_batch(eng, f, levels, vtree),
    }
}

/// The internal nodes of `vtree` whose childless nodes are all finished: a
/// region, a leaf for which `summed` holds, or a variable leaf `summable`
/// names. Returns them as a mask over `vtree`'s nodes and, ascending, the
/// finished nodes that have a parent that is not finished, internal or a
/// variable leaf `summable` names that `summed` does not, and the root if it
/// is a finished internal node.
pub(crate) fn finished(
    eng: &Engine,
    vtree: &Vtree,
    summed: impl Fn(VtreeIdx) -> bool,
    summable: &dyn Fn(VarId) -> bool,
) -> Result<(Vec<bool>, Vec<VtreeIdx>), OperationError> {
    let lim = eng.limits();
    let mut fin = Vec::new();
    lim.try_resize(&mut fin, vtree.num_nodes(), false)?;
    let mut leaves = Vec::new();
    for t in vtree.bottomup() {
        fin[t.idx()] = match *vtree.node(t) {
            VtreeNode::Internal { left, right, .. } => fin[left.idx()] && fin[right.idx()],
            _ if vtree.node(t).is_region() || summed(t) => true,
            VtreeNode::Leaf { var, .. } => {
                let done = summable(var);
                // A root leaf stays: its output names a label, not a value.
                if done && vtree.node(t).parent().is_some() {
                    lim.try_push(&mut leaves, t)?;
                }
                done
            }
        };
    }
    let mut maximal = Vec::new();
    for t in vtree.internal_bottomup_slice().iter().copied().chain(leaves) {
        if fin[t.idx()] && vtree.node(t).parent().is_none_or(|p| !fin[p.idx()]) {
            lim.try_push(&mut maximal, t)?;
        }
    }
    maximal.sort_unstable();
    for t in vtree.bottomup().filter(|&t| vtree.node(t).is_leaf()) {
        fin[t.idx()] = false;
    }
    Ok((fin, maximal))
}

/// Fold each of `maximal`, nodes of `f`'s vtree none under another, that is
/// not summed out yet, restore the marginal invariants at all of them and, on
/// a projection, make every one that is summed out a region.
pub(crate) fn sum_out(eng: &Engine, f: &mut Tdd, maximal: &[VtreeIdx]) -> Result<(), OperationError> {
    let vtree = std::sync::Arc::clone(&f.vtree);
    let mut levels = Vec::new();
    eng.limits().reserve_exact(&mut levels, maximal.len())?;
    levels.extend(maximal.iter().copied().filter(|&t| !f.levels[t.idx()].is_marginal()));
    f.levels.hold_counts(f.levels.is_closed());
    f.levels.forget_pair_total();
    let folded = match levels.is_empty() {
        true => Ok(()),
        false => evaluate_levels(eng, f, &levels, &vtree),
    };
    let restored = restore_marginal_invariants(eng, f, maximal, &vtree);
    let mut trimmed = Ok(());
    if let Some(home) = vtree.home() {
        levels.clear();
        let summed_leaf = maximal.iter().any(|&t| vtree.node(t).is_leaf() && f.levels[t.idx()].is_marginal());
        levels.extend(maximal.iter().copied().filter(|&t| !vtree.node(t).is_leaf() && f.levels[t.idx()].is_marginal()));
        if summed_leaf || !levels.is_empty() {
            trimmed = trim(eng, f, &home, None, &levels).map(drop);
        }
    }
    let done = trimmed.and(folded).and(restored);
    if done.is_err() {
        f.levels.forget_left_pairs();
    }
    done
}

/// [`Engine::marginalize`]: the nodes [`finished`] names, summed out by
/// [`sum_out`]; on a home the diagram is trimmed first, and an internal
/// summed-out level that is not a region is refused.
pub(crate) fn marginalize(eng: &Engine, f: &mut Tdd, summable: &dyn Fn(VarId) -> bool) -> Result<(), OperationError> {
    let _op = eng.limits().enter()?;
    if let Some(home) = f.vtree.home() {
        if !f.has_marginal_level() {
            trim(eng, f, &home, None, &[])?;
        } else if let Some(t) = f.vtree.bottomup().find(|&t| !f.vtree.node(t).is_leaf() && f.levels[t.idx()].is_marginal()) {
            return Err(OperationError::MarginalLevel(t));
        }
    }
    if f.vtree.num_nodes() == 0 || f.is_zero() {
        return Ok(());
    }
    let (_, maximal) = finished(eng, &f.vtree, |t| f.levels[t.idx()].is_marginal(), summable)?;
    sum_out(eng, f, &maximal)
}

#[cfg(test)]
pub(crate) mod tests;
