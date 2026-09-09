//! Marginalization primitives: freezing vtree levels into per-node counts —
//! or, when a [`WeightStore`] is attached to the diagram, per-node semiring
//! values — and the schedule deciding when each level may be frozen.

mod fold;
mod kind;
mod leaf;
mod schedule;
mod store;

use crate::engine::Engine;
pub use schedule::{intra_batch_completions, marginalize_schedule};
pub(crate) use fold::{marginalize_batch, marginalize_batch_weighted};
pub(crate) use leaf::{
    canonicalize_apply_leaf_refs, debug_check_leaf_columns_pinned, find_leaf_slot_by_value,
    leaf_column_vals,
};
#[cfg(test)]
pub(crate) use leaf::{marginalize_leaf_inline, marginalize_leaf_weighted};
#[cfg(test)]
pub(crate) use store::dedup_fresh_store;

use crate::counts::ColumnRetention;
use store::ensure_weights;
use crate::error::ApplyError;
use crate::diagram::{LeafLabel, Tdd};
use crate::diagram::WeightVal;
use crate::weight_store::WeightStore;
use crate::vtree::{Vtree, VtreeIdx, VtreeNode};

/// Global marginal-closure pass: marginalize **every** structural level whose
/// two children are both marginal, to fixpoint.
///
/// A rotation that brings two marginal children together leaves the new parent
/// level *structural* — `restructure_after_*_rotation_bounded` only shuffles
/// node-index references, it never collapses a node to counts. But a node whose **both**
/// children are fully summed out (marginal) is itself fully summed out and MUST
/// be in marginal form for the diagram to stay canonical and count correctly
/// (skipping this "cascade up" is exactly the bug behind the original
/// count-unsafe parent-of-marginal rotation).
/// This pass closes every such cluster across the whole diagram at once.
///
/// After a re-search sweep performs many rotations, marginal clusters can appear
/// anywhere (a relaxed parent-of-marginal rotation leaves a structural parent over
/// two marginal children — count-unsafe until closed). Rather than hook each
/// committed rotation, run this once after the sweep: it scans all levels, collects
/// the bottom layer of structural-over-two-marginal levels, marginalizes them via
/// [`marginalize`], and repeats until no level qualifies (a freshly-marginal
/// level can complete a cluster one level up).
///
/// It is a **no-op** when the diagram is already in canonical marginal form (the
/// normal bottom-up marginalize leaves no unclosed clusters), so it is safe to run
/// unconditionally — it only does work when rotations created clusters. Returns the
/// number of levels marginalized.
///
/// # Errors
///
/// Passes through [`marginalize`]'s `Err(ApplyError::Deadline)`. The
/// clusters closed before the cut stay closed; the rest are still structural
/// levels over two marginal children, which is the state this pass exists to
/// finish and a caller that resumes will find waiting for it.
pub(crate) fn marginalize_closure(eng: &Engine, tdd: &mut Tdd, vtree: &Vtree) -> Result<usize, ApplyError> {
    let n = vtree.num_nodes();
    let mut total = 0usize;
    loop {
        let mut targets: Vec<VtreeIdx> = Vec::new();
        for i in 0..n {
            if vtree.node(VtreeIdx(i as u32)).is_leaf() || tdd.levels[i].is_marginal() {
                continue;
            }
            let t = VtreeIdx(i as u32);
            let (l, r) = vtree.children(t);
            if tdd.levels[l.idx()].is_marginal() && tdd.levels[r.idx()].is_marginal() {
                targets.push(t);
            }
        }
        if targets.is_empty() {
            break;
        }
        // bottom-up topo order = ascending index after reindex_bottomup.
        targets.sort_by_key(|t| t.idx());
        total += targets.len();
        // The one integer-vs-weighted dispatch: a weighted diagram's targets
        // are weight-marginal and carry no integer counts, so the integer batch
        // may not run on them.
        let r = if let Some(mut ws) = tdd.weights.take() {
            let r = marginalize_batch_weighted(eng, tdd, &targets, vtree, &mut ws);
            tdd.weights = Some(ws);
            r
        } else {
            marginalize_batch(eng, tdd, &targets, vtree)
        };
        r?;
    }
    Ok(total)
}

/// The diagram's value under its attached [`WeightStore`], or `None` in
/// integer mode.
///
/// A weighted marginalization usually leaves the output level explicit and
/// freezes only levels below it, so this folds the explicit levels above the
/// frozen ones on demand from the store's values and leaf weights; when the
/// output level is itself frozen it reads the stored value directly.
///
/// # Panics
///
/// Panics if the output level is frozen but its value is absent from the
/// store.
pub fn weighted_value(tdd: &Tdd) -> Option<WeightVal> {
    let eng = Engine::new();
    let ws = tdd.weights.as_ref()?;
    let vtree = std::sync::Arc::clone(&tdd.vtree);
    Some(weighted_output_value(&eng, tdd, &vtree, ws))
}

pub(crate) fn weighted_output_value(eng: &Engine, tdd: &Tdd, vtree: &Vtree, ws: &WeightStore) -> WeightVal {
    // UNSAT / constant-false output: the ZERO sentinel carries no level slot
    // (`output.local` is the ZERO idx, out of range for any real level), so the
    // weighted value is exactly zero — mirrors `model_count`'s `is_zero()` guard.
    if tdd.is_zero() {
        return ws.wzero();
    }
    let out_t = tdd.output.vtree.idx();
    let out_i = tdd.output.local.idx();
    if tdd.levels[out_t].is_weight_marginal() {
        return ws.level(out_t).expect("output level weight-marginalized")[out_i].clone();
    }
    // All-backbone / single-residual-var output: when preprocessing forces every
    // variable, the driver promotes one var to live and the compile collapses the
    // output to a LEAF level. `ensure_weights` early-returns on leaf levels (their
    // bases come from the semiring on demand), so `computed[out_t]` would stay
    // `None` and the unwrap below would panic. Fold the leaf base directly —
    // mirrors `read_marginal_weight`'s leaf branch and the model counter's
    // leaf-seeding on the integer path. (One = w_pos+w_neg, Pos = w_pos, Neg = w_neg.)
    if let VtreeNode::Leaf { var, .. } = *vtree.node(VtreeIdx(out_t as u32)) {
        return ws.leaf_val(var, LeafLabel::from_idx(out_i));
    }
    let mut computed: Vec<Option<Vec<WeightVal>>> = vec![None; vtree.num_nodes()];
    // Root-only read: the single value below is the ONLY thing taken from
    // `computed`, so the walk releases each child column as its parent's
    // completes ([`ColumnRetention::Frontier`]) — peak is the walk frontier,
    // not one `Vec<WeightVal>` per level of the whole diagram. `out_t` is the
    // walk root, so its column is the one the walk never frees.
    ensure_weights(eng, tdd, tdd.output.vtree, vtree, ws, &mut computed, ColumnRetention::Frontier);
    computed[out_t]
        .as_ref()
        .expect("output level weights ensured")[out_i]
        .clone()
}

/// Sum out `levels`, freezing each one into per-node values.
///
/// A frozen level stops carrying pair structure and carries one value per node
/// instead: the number of assignments to its whole vtree subtree that reach
/// that node, or — when the diagram has a [`WeightStore`] attached
/// ([`Tdd::attach_weights`]) — that node's semiring value. Counting then folds
/// `Σ count(left) × count(right)` over a node's pairs and stops at a frozen
/// level; leaves count by label (`One` → 2, `Pos`/`Neg` → 1, `Zero` → 0), so an
/// unconstrained variable contributes its factor of two through the fold.
/// Summing out a leaf writes its fixed count inline into the parent's
/// references, which is what makes a parent's `Pos` and `Neg` branches twins.
/// With weights attached, a leaf's three column entries are `w⁺+w⁻`, `w⁺`,
/// `w⁻` instead.
///
/// `levels` must be sorted bottom-up ([`marginalize_schedule`] returns each
/// group that way): a level is frozen only once its children are frozen or are
/// leaves.
///
/// # Errors
///
/// Returns `ApplyError::Deadline` if the caller's wall passed while the pass
/// was running and the post-apply poll is armed. The levels frozen before the
/// cut keep their values and the end-sweep tagger has run over them, so the
/// diagram left behind is exactly the one a pass over that prefix would have
/// produced — well-formed, readable, and count-preserving.
pub fn marginalize(eng: &Engine, f: &mut Tdd, levels: &[VtreeIdx]) -> Result<(), ApplyError> {
    let vtree = std::sync::Arc::clone(&f.vtree);
    if let Some(mut ws) = f.weights.take() {
        let r = marginalize_batch_weighted(eng, f, levels, &vtree, &mut ws);
        f.weights = Some(ws);
        return r;
    }
    marginalize_batch(eng, f, levels, &vtree)
}

#[cfg(test)]
#[path = "marginal_alloc_guard_tests.rs"]
mod marginal_alloc_guard_tests;

#[cfg(test)]
#[path = "deadline_tests.rs"]
mod marginalize_deadline_tests;

#[cfg(test)]
#[path = "mod_tests.rs"]
mod marginalize_tests;
