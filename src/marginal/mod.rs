//! Summing levels out and the epilogue restoring the marginal invariants.
//!
//! A marginalized level stops carrying pair structure and carries one value per
//! node instead: the number of assignments to its vtree subtree that reach that
//! node, or — with a [`WeightStore`](crate::diagram::WeightStore) attached — that node's weighted value. The
//! encoding a parent reads those values through is [`crate::diagram`]; the
//! reduction passes the epilogue calls are [`crate::reduce`]; counting over a
//! partly marginalized diagram is [`crate::query`].
//!
//! [`Tdd::marginalize_levels`] sums out a bottom-up group of levels and
//! restores invariants 7, 8 and 10 before it returns. Folding a weighted
//! diagram down to its value is [`Tdd::weighted_value`].

pub(crate) mod transition;
mod fold;
mod leaf;

use crate::Engine;
pub(crate) use fold::marginalize_batch;
pub(crate) use leaf::canonicalize_weighted_leaf_refs;
pub(crate) use leaf::{marginalize_leaf_inline, marginalize_leaf_weighted};

use num_bigint::BigUint;

use crate::value::{Count, CountRead, CountVec, IntFold, WeightFold};
use crate::limits::OperationError;
use crate::diagram::{LeafLabel, Tdd, WeightStore, WeightValue};
use crate::vtree::{VarId, Vtree, VtreeIdx};
use crate::reduce::{restore_marginal_invariants, trim};

/// Marginalize every structural level whose two children are both marginal,
/// visiting affected parents until no level qualifies; returns the number of levels marginalized.
///
/// `rebuild_levels` never collapses a node to counts, so a rotation
/// that brings two marginal children together leaves a structural parent over
/// two marginal children, which is not a canonical marginal form. Each round
/// collects the bottom layer of such levels and marginalizes it through
/// [`marginalize_levels`]; a freshly marginal level can complete a cluster one level
/// up, hence the loop. A diagram already in canonical form makes this a no-op.
///
/// # Errors
///
/// Passes through [`marginalize_levels`]'s `Err(OperationError::Stopped)`. The
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
    /// Sum out selected levels under this engine's resource limits.
    ///
    /// See [`Tdd::marginalize_levels`] for target, arithmetic and preservation
    /// semantics. Invalid targets are rejected before mutation. Allocation or
    /// stop refusals may leave a completed prefix; retrying finishes the remaining
    /// levels and cleanup while preserving the diagram's value.
    pub fn marginalize_levels(&self, f: &mut Tdd, levels: &[VtreeIdx]) -> Result<(), OperationError> {
        marginalize_levels(self, f, levels)
    }

    /// Sum out home nodes under this engine's resource limits.
    ///
    /// See [`Tdd::marginalize`] for target, arithmetic and form semantics.
    pub fn marginalize(&self, f: &mut Tdd, targets: &[VtreeIdx]) -> Result<(), OperationError> {
        marginalize(self, f, targets)
    }
}

/// Sum out validated target subtrees and restore the marginal invariants.
pub(crate) fn marginalize_levels(eng: &Engine, f: &mut Tdd, levels: &[VtreeIdx]) -> Result<(), OperationError> {
    let _op = eng.limits().enter()?;
    f.check_level_indices(levels)?;
    let vtree = std::sync::Arc::clone(&f.vtree);
    evaluate_levels(eng, f, levels, &vtree)?;
    restore_marginal_invariants(eng, f, levels, &vtree)
}

/// The one integer-vs-weighted dispatch of the pass: a weighted diagram's
/// targets are weight-marginal and carry no integer counts, so the integer
/// batch may not run on them.
fn evaluate_levels(eng: &Engine, f: &mut Tdd, levels: &[VtreeIdx], vtree: &Vtree) -> Result<(), OperationError> {
    evaluate_scaled(eng, f, levels, vtree, &[])
}

/// [`evaluate_levels`], the values of internal target `levels[i]` multiplied
/// by the factor of the variables `free[i]`, if `free` is not empty.
fn evaluate_scaled(eng: &Engine, f: &mut Tdd, levels: &[VtreeIdx], vtree: &Vtree, free: &[Vec<VarId>]) -> Result<(), OperationError> {
    if let Some(mut ws) = f.weights.take() {
        let scale = |i: usize, col: &mut Vec<WeightValue>, ws: &WeightStore| {
            let factor = free.get(i).and_then(|vars| vars.iter().map(|&var| ws.leaf_val(var, LeafLabel::One)).reduce(|a, b| a.mul(&b)));
            if let Some(factor) = factor {
                col.iter_mut().for_each(|v| *v = v.mul(&factor));
            }
            Ok(())
        };
        let r = fold::marginalize_targets::<WeightFold>(eng, f, levels, vtree, &mut ws, scale);
        f.weights = Some(ws);
        r
    } else if free.is_empty() {
        marginalize_batch(eng, f, levels, vtree)
    } else {
        let scale = |i: usize, col: &mut CountVec, _: &()| {
            let k = free[i].len() as u32;
            for j in (0..col.len()).filter(|_| k > 0) {
                let scaled = match col.get(j) {
                    CountRead::Fast(v) if v.leading_zeros() >= k => Count::from_u128(v << k),
                    CountRead::Fast(v) => Count::Big(BigUint::from(v) << k),
                    CountRead::Big(v) => Count::Big(v << k),
                };
                col.set(eng, j, scaled)?;
            }
            Ok(())
        };
        fold::marginalize_targets::<IntFold>(eng, f, levels, vtree, &mut (), scale)
    }
}

/// Sum out the home variables under each node of `targets`, nodes of the
/// home of `f`'s vtree; on a root vtree, [`marginalize_levels`].
///
/// Per target not under another, `t` is the own node at it
/// ([`Vtree::own_at`]): none, or a region, leaves `f` as it is, the
/// variables already free or summed; a leaf is summed out where it is
/// (the home's other variables under the target stay free); an internal
/// node is folded, its values multiplied by the factor of the target's
/// free variables, and becomes a region holding the target's leaves.
pub(crate) fn marginalize(eng: &Engine, f: &mut Tdd, targets: &[VtreeIdx]) -> Result<(), OperationError> {
    let Some(home) = f.vtree.home().cloned() else {
        return marginalize_levels(eng, f, targets);
    };
    let _op = eng.limits().enter()?;
    if let Some(&h) = targets.iter().find(|h| h.idx() >= home.vtree().num_nodes()) {
        return Err(OperationError::LevelNotInVtree(h));
    }
    if !f.has_marginal_level() {
        trim(eng, f, &home, &[])?;
    } else if let Some(t) = f.vtree.bottomup().find(|&t| !f.vtree.node(t).is_leaf() && f.levels[t.idx()].is_marginal()) {
        return Err(OperationError::MarginalLevel(t));
    }
    let mut ranges: Vec<_> = targets.iter().map(|&h| (home.rank_range(h), h)).collect();
    ranges.sort_unstable_by_key(|(r, _)| (r.start, std::cmp::Reverse(r.end)));
    let mut levels = Vec::new();
    let mut free = Vec::new();
    let mut end = None;
    for (range, h) in ranges {
        if end.is_some_and(|end| range.end <= end) {
            continue;
        }
        end = Some(range.end);
        let Some(t) = f.vtree.own_at(h) else { continue };
        let node = f.vtree.node(t);
        if node.is_region() || f.levels[t.idx()].is_marginal() {
            continue;
        }
        levels.push((t, h));
        free.push(if node.is_leaf() { Vec::new() } else { f.vtree.free_vars_under(h) });
    }
    if levels.is_empty() {
        return Ok(());
    }
    // The fold multiplies the factor of a target's free variables into its
    // values, and only the closing trim takes them out of the free set: a
    // refusal between the two would count them twice, so it puts the diagram
    // back as it came.
    let collapses = levels.iter().any(|&(t, _)| !f.vtree.node(t).is_leaf());
    let backup = if collapses { Some(f.try_clone_on(eng)?) } else { None };
    let summed = sum_out(eng, f, &home, levels, &free);
    if summed.is_err()
        && let Some(backup) = backup
    {
        let refused = std::mem::replace(f, backup);
        crate::diagram::return_levels(eng, refused.levels.into_vec());
    }
    summed
}

/// The body of [`marginalize`] after its targets are chosen: fold each
/// `levels[i]` scaled by the factor of `free[i]`, then collapse the internal
/// ones into regions.
fn sum_out(eng: &Engine, f: &mut Tdd, home: &crate::vtree::Home, levels: Vec<(VtreeIdx, VtreeIdx)>, free: &[Vec<VarId>]) -> Result<(), OperationError> {
    let vtree = std::sync::Arc::clone(&f.vtree);
    let own: Vec<VtreeIdx> = levels.iter().map(|&(t, _)| t).collect();
    let folded = evaluate_scaled(eng, f, &own, &vtree, free);
    let restored = restore_marginal_invariants(eng, f, &own, &vtree);
    let mut collapse: Vec<_> =
        levels.into_iter().filter(|&(t, _)| !vtree.node(t).is_leaf() && f.levels[t.idx()].is_marginal()).collect();
    collapse.sort_unstable();
    if !collapse.is_empty() {
        trim(eng, f, home, &collapse)?;
    }
    folded.and(restored)
}

#[cfg(test)]
pub(crate) mod tests;
