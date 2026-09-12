//! Summing levels out and the epilogue restoring the marginal invariants.
//!
//! A marginalized level stops carrying pair structure and carries one value per
//! node instead: the number of assignments to its vtree subtree that reach that
//! node, or — with a [`WeightStore`] attached — that node's weighted value. The
//! encoding a parent reads those values through is [`crate::diagram`]; the
//! reduction passes the epilogue calls are [`crate::reduce`]; counting over a
//! partly marginalized diagram is [`crate::query`].
//!
//! Entry points: [`marginalize`] sums out a bottom-up group of levels and
//! restores invariants 7, 8 and 10 before it returns. Folding a weighted
//! diagram down to its value is [`crate::query::weighted_value`].

mod column;
pub(crate) use column::{column_of, install_int_column, install_weight_column, LevelColumns};
mod fold;
mod leaf;
mod store;
pub(crate) use store::{free_subsumed_marginal_children, read_count, read_weight};

use crate::engine::Engine;
pub(crate) use fold::marginalize_batch;
pub(crate) use leaf::canonicalize_apply_leaf_refs;
pub(crate) use leaf::{marginalize_leaf_inline, marginalize_leaf_weighted};
pub(crate) use store::dedup_fresh_store;

use crate::value::ColumnRetention;
use crate::limits::RecoveryPanic;
use crate::value::{unwrap_infallible, ValueDomain, WeightFold};
use crate::limits::ApplyError;
use crate::diagram::{LeafLabel, Tdd};
use crate::diagram::WeightVal;
use crate::diagram::WeightStore;
use crate::vtree::{Vtree, VtreeIdx, VtreeNode};
use crate::reduce::contract::pair_fusion::fuse_pairs_at_parents;
use crate::reduce::slot_prune::prune_value_slots;

/// Marginalize every structural level whose two children are both marginal,
/// repeating until no level qualifies; returns the number of levels marginalized.
///
/// `restructure_inner_search` never collapses a node to counts, so a rotation
/// that brings two marginal children together leaves a structural parent over
/// two marginal children, which is not a canonical marginal form. Each round
/// collects the bottom layer of such levels and marginalizes it through
/// [`marginalize`]; a freshly marginal level can complete a cluster one level
/// up, hence the loop. A diagram already in canonical form makes this a no-op.
///
/// # Errors
///
/// Passes through [`marginalize`]'s `Err(ApplyError::Deadline)`. The
/// clusters closed before the cut stay closed; the rest are still structural
/// levels over two marginal children, which is the state this pass exists to
/// finish and a caller that resumes will find waiting for it.
pub(crate) fn marginalize_closure(eng: &Engine, tdd: &mut Tdd) -> Result<usize, ApplyError> {
    let vtree = std::sync::Arc::clone(&tdd.vtree);
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
        // bottom-up topo order = ascending index after the bottom-up reindex.
        targets.sort_by_key(|t| t.idx());
        total += targets.len();
        marginalize_levels(eng, tdd, &targets, &vtree)?;
    }
    Ok(total)
}

/// The weighted value of `tdd`'s output node under `ws`; `tdd` must have been
/// weighted with `ws`.
pub(crate) fn weighted_output_value(eng: &Engine, tdd: &Tdd, ws: &WeightStore) -> WeightVal {
    let vtree = &tdd.vtree;
    // UNSAT / constant-false output: the `ZERO` sentinel carries no level slot
    // (`output.local` is the `ZERO` idx, out of range for any real level), so the
    // weighted value is exactly zero — mirrors `model_count`'s `is_zero()` guard.
    if tdd.is_zero() {
        return ws.wzero();
    }
    let out_t = tdd.output.vtree.idx();
    let out_i = tdd.output.local.idx();
    if tdd.levels[out_t].is_weight_marginal() {
        return ws.level(out_t).expect("output level weight-marginalized")[out_i].clone();
    }
    // Leaf output level: the fold below stores nothing for leaves (their values
    // come from the semiring on demand), so read the leaf value directly.
    if let VtreeNode::Leaf { var, .. } = *vtree.node(VtreeIdx(out_t as u32)) {
        return ws.leaf_val(var, LeafLabel::from_idx(out_i));
    }
    let mut computed: Vec<Option<Vec<WeightVal>>> = vec![None; vtree.num_nodes()];
    // Only the root value is read, so child columns are released as their
    // parent completes (`ColumnRetention::Frontier`). The "already stored" test
    // is this diagram's own marginality rather than `WeightStore::is_set`: the
    // store is shared, so a column at this index may belong to another live
    // `Tdd` while this diagram's level is still structural.
    let marginal = |i: usize| tdd.levels[i].is_marginal();
    unwrap_infallible(WeightFold::ensure::<RecoveryPanic>(
        eng,
        out_t,
        vtree,
        &tdd.levels,
        &mut computed,
        ws,
        &marginal,
        ColumnRetention::Frontier,
    ));
    computed[out_t]
        .as_ref()
        .expect("output level weights ensured")[out_i]
        .clone()
}

/// Sum out `levels`, marginalizing each one into per-node values.
///
/// This sums out vtree *levels* and is permanent and count-preserving; summing
/// a *variable* out is existential quantification, which is
/// [`project_vars`](crate::apply::project_vars).
///
/// A marginal level stops carrying pair structure and carries one value per node
/// instead: the number of assignments to its whole vtree subtree that reach
/// that node, or — when the diagram has a [`WeightStore`] attached
/// ([`Tdd::set_weights`]) — that node's semiring value. Counting then folds
/// `Σ count(left) × count(right)` over a node's pairs and stops at a marginal
/// level; leaves count by label (`One` → 2, `Pos`/`Neg` → 1, `Zero` → 0), so an
/// unconstrained variable contributes its factor of two through the fold.
/// Summing out a leaf writes its fixed count inline into the parent's
/// references, which is what makes a parent's `Pos` and `Neg` branches twins.
/// With weights attached, a leaf's three column entries are `w⁺+w⁻`, `w⁺`,
/// `w⁻` instead.
///
/// `levels` must be sorted bottom-up: a level is marginal only once its
/// children are marginal or are leaves. A leaf may be named; a level already
/// marginal is passed over, and ⊥ stays ⊥. Marginality is
/// permanent, and no later conjunction may constrain a summed-out level —
/// [`Engine::and_clause`](crate::Engine::and_clause) panics on a clause whose
/// spine reaches one, and [`Engine::and`](crate::Engine::and) accepts a level
/// marginal in both operands only where one is constant-true there — so sum
/// a level out only once every clause over its variables is in. The model
/// count is preserved; the count-marginal form keeps the diagram readable by
/// [`Tdd::model_count`], the weighted form by
/// [`weighted_value`](crate::query::weighted_value).
///
/// # Panics
///
/// Panics if a named internal level has a child that is neither a leaf nor
/// marginal once its turn comes — `levels` out of bottom-up order, or a
/// child left out.
///
/// # Post-conditions
///
/// On `Ok`: every level in `levels` is marginal, their parents' sides are in
/// decoded form, and no marginal slot is orphaned or duplicated. Freezing a
/// level mints slots at its parents that a later pass must not read twice, so
/// the two passes that restore those invariants — the fusion sweep at the
/// parents of `levels` and the slot prune — run here rather than being left to
/// the caller. In the bounded-precision log domain the fusion sweep is skipped:
/// it would find redexes whose values it must not add together, so only the
/// prune runs.
///
/// # Errors
///
/// Returns `ApplyError::Deadline` if the caller's wall passed while the pass
/// was running and the post-apply poll is armed. The levels marginal before the
/// cut keep their values and the end-sweep tagger has run over them, so the
/// diagram left behind is exactly the one a pass over that prefix would have
/// produced — well-formed, readable, and count-preserving.
///
/// Returns `ApplyError::OverBudget` if the fusion sweep's rewrite is refused.
///
/// ```
/// # use std::sync::Arc;
/// # use tididi::{Engine, Tdd};
/// # use tididi::marginal::marginalize;
/// # use tididi::vtree::Vtree;
/// # let vtree = Arc::new(Vtree::balanced(4));
/// # let engine = Engine::new();
/// # // The root's left child: an internal level whose own children are leaves.
/// # let (left, _right) = vtree.children(vtree.root());
/// let mut f = Tdd::clause(&vtree, [1, -2]) & Tdd::clause(&vtree, [2, 3]);
/// let before = f.model_count();
///
/// marginalize(&engine, &mut f, &[left]).unwrap();
/// assert!(f.has_marginal_level());
/// assert_eq!(f.model_count(), before);   // summing a level out preserves the count
///
/// // A byte budget of zero refuses the pass's first reservation.
/// use tididi::limits::LimitSet;
/// let mut g = Tdd::clause(&vtree, [1, -2]) & Tdd::clause(&vtree, [2, 3]);
/// let _armed = engine.limits().scope(LimitSet::none().budget(Some(0)));
/// match marginalize(&engine, &mut g, &[left]) {
///     Ok(()) => unreachable!("no reservation can be granted"),
///     Err(e) => assert_eq!(e, tididi::ApplyError::OverBudget),
/// }
/// ```
pub fn marginalize(eng: &Engine, f: &mut Tdd, levels: &[VtreeIdx]) -> Result<(), ApplyError> {
    let vtree = std::sync::Arc::clone(&f.vtree);
    marginalize_levels(eng, f, levels, &vtree)?;
    restore_marginal_invariants(eng, f, levels, &vtree)
}

/// The one integer-vs-weighted dispatch of the pass: a weighted diagram's
/// targets are weight-marginal and carry no integer counts, so the integer
/// batch may not run on them.
fn marginalize_levels(eng: &Engine, f: &mut Tdd, levels: &[VtreeIdx], vtree: &Vtree) -> Result<(), ApplyError> {
    if let Some(mut ws) = f.weights.take() {
        let r = fold::marginalize_targets::<WeightFold>(eng, f, levels, vtree, &mut ws);
        f.weights = Some(ws);
        r
    } else {
        marginalize_batch(eng, f, levels, vtree)
    }
}

/// The epilogue of [`marginalize`]: fuse the redexes marginalizing just minted, then
/// collect the slots it orphaned.
///
/// Fusion is what makes a parent P-saturated — at most one pair per (left
/// child, marginal side) — and it is skipped in the log domain, where two
/// slots that fusion would fold carry values whose sum is not representable
/// without loss. The prune runs either way: marginalize inlines small counts
/// and so orphans their slots whatever the arithmetic.
fn restore_marginal_invariants(
    eng: &Engine,
    f: &mut Tdd,
    levels: &[VtreeIdx],
    vtree: &Vtree,
) -> Result<(), ApplyError> {
    let log_domain = f.weights().is_some_and(WeightStore::is_log);
    if !log_domain {
        let mut parents: Vec<VtreeIdx> =
            levels.iter().filter_map(|&l| vtree.node(l).parent()).collect();
        parents.sort_unstable();
        parents.dedup();
        fuse_pairs_at_parents(eng, f, &parents)?;
        #[cfg(debug_assertions)]
        crate::test_helpers::check::marginal::debug_assert_pair_fusion_saturated(f, Some(&parents), "marginalize");
    }
    prune_value_slots(eng, f);
    Ok(())
}

#[cfg(test)]
mod tests;
