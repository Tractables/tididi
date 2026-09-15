//! Summing levels out and the epilogue restoring the marginal invariants.
//!
//! A marginalized level stops carrying pair structure and carries one value per
//! node instead: the number of assignments to its vtree subtree that reach that
//! node, or — with a [`WeightStore`] attached — that node's weighted value. The
//! encoding a parent reads those values through is [`crate::diagram`]; the
//! reduction passes the epilogue calls are [`crate::reduce`]; counting over a
//! partly marginalized diagram is [`crate::query`].
//!
//! [`Tdd::marginalize_levels`] sums out a bottom-up group of levels and
//! restores invariants 7, 8 and 10 before it returns. Folding a weighted
//! diagram down to its value is [`Tdd::weighted_value`].

mod column;
pub(crate) use column::{install_int_column, install_weight_column};
pub(crate) mod transition;
mod fold;
mod leaf;
mod store;
pub(crate) use store::free_subsumed_marginal_children;

use crate::engine::Engine;
pub(crate) use fold::marginalize_batch;
pub(crate) use leaf::canonicalize_apply_leaf_refs;
pub(crate) use leaf::{marginalize_leaf_inline, marginalize_leaf_weighted};
pub(crate) use store::dedup_fresh_store;

use crate::value::WeightFold;
use crate::limits::OperationError;
use crate::diagram::Tdd;
use crate::diagram::WeightStore;
use crate::vtree::{Vtree, VtreeIdx};
use crate::reduce::contract::pair_fusion::fuse_pairs_at_parents;
use crate::reduce::slot_prune::prune_value_slots;

/// Marginalize every structural level whose two children are both marginal,
/// visiting affected parents until no level qualifies; returns the number of levels marginalized.
///
/// `restructure_inner_search` never collapses a node to counts, so a rotation
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
}

/// Sum out validated target subtrees and restore the marginal invariants.
pub(crate) fn marginalize_levels(eng: &Engine, f: &mut Tdd, levels: &[VtreeIdx]) -> Result<(), OperationError> {
    f.check_level_indices(levels)?;
    let _op = eng.limits().begin_operation();
    let vtree = std::sync::Arc::clone(&f.vtree);
    evaluate_levels(eng, f, levels, &vtree)?;
    restore_marginal_invariants(eng, f, levels, &vtree)
}

/// The one integer-vs-weighted dispatch of the pass: a weighted diagram's
/// targets are weight-marginal and carry no integer counts, so the integer
/// batch may not run on them.
fn evaluate_levels(eng: &Engine, f: &mut Tdd, levels: &[VtreeIdx], vtree: &Vtree) -> Result<(), OperationError> {
    if let Some(mut ws) = f.weights.take() {
        let r = fold::marginalize_targets::<WeightFold>(eng, f, levels, vtree, &mut ws);
        f.weights = Some(ws);
        r
    } else {
        marginalize_batch(eng, f, levels, vtree)
    }
}

/// The epilogue of [`marginalize_levels`]: fuse the redexes marginalizing just minted, then
/// collect the slots it orphaned.
///
/// Fusion is what makes a parent P-saturated — at most one pair per (left
/// child, marginal side) — and it is skipped in the log domain, where two
/// slots that fusion would fold carry values whose sum is not representable
/// without loss. The prune runs either way: marginalization inlines small counts
/// and so orphans their slots whatever the arithmetic.
fn restore_marginal_invariants(
    eng: &Engine,
    f: &mut Tdd,
    levels: &[VtreeIdx],
    vtree: &Vtree,
) -> Result<(), OperationError> {
    let log_domain = f.weights().is_some_and(WeightStore::is_log);
    if !log_domain {
        let mut parents: Vec<VtreeIdx> =
            levels.iter().filter_map(|&l| vtree.node(l).parent()).collect();
        parents.sort_unstable();
        parents.dedup();
        fuse_pairs_at_parents(eng, f, &parents)?;
        #[cfg(debug_assertions)]
        crate::test_helpers::check::marginal::debug_assert_pair_fusion_saturated(f, Some(&parents), "marginalize_levels");
    }
    prune_value_slots(eng, f);
    Ok(())
}

#[cfg(test)]
mod tests;
