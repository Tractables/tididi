//! Canonical form: pruning, twin contraction, pair fusion, slot pruning.
//!
//! A conjunction leaves a diagram that denotes the right function but is not
//! the smallest representation of it; the passes here bring it back to the
//! canonical one. Producing the diagram is [`crate::apply`]; summing levels out
//! is [`crate::marginal`], whose epilogue calls the last two passes here.
//!
//! Entry points: [`minimize`] is the infallible form, [`try_minimize`] the one
//! that hands a refused reservation back, and [`ReductionPlan`] selects which
//! passes run.
//!
//! The passes, in the order a full reduction runs them:
//!
//! 1. **Prune** (`prune.rs`): remove nodes not reachable from the output, by a
//!    top-down reachability mark and a bottom-up compaction with a monotone
//!    remap.
//! 2. **Twin contraction** (`contract/`): merge nodes with identical parent
//!    context — the same set of (parent node, sibling) pairs. Twins compute
//!    functions whose disjunction replaces them both without changing the
//!    output.
//! 3. **Pair fusion and slot pruning** (`contract/pair_fusion/`,
//!    `slot_prune/`): what a freshly marginalized level needs — fusing pairs
//!    that share a structural-side child, and dropping value slots nothing
//!    references.
//!
//! **Prune to contract.** The phases share only the `Tdd` dirty-contract
//! worklists: prune and the content-twin merge push the levels they changed
//! through `Tdd::invalidate`, and the contract passes drain those lists.

mod prune;
pub(crate) mod scratch;
pub(crate) mod contract;
pub(crate) mod slot_prune; // post-tagger marginal-slot compaction
mod content_twins;


// ── Minimize options ─────────────────────────────────────────────────────────

/// The reduction passes to run, with a content-twin policy only for a full pass.
#[derive(Debug)]
#[non_exhaustive]
pub enum ReductionPlan<'a> {
    /// Prune unreachable nodes and orphaned marginal slots.
    Prune,
    /// Contract inner-node twins.
    Contract,
    /// Prune, contract inner and leaf twins, then apply the content-twin policy.
    Full(ContentTwins<'a>),
}

impl Default for ReductionPlan<'_> {
    fn default() -> Self { Self::Full(ContentTwins::Always) }
}

/// Whether eligible content-twin scans run and retain their adaptive schedule.
#[derive(Debug, Default)]
#[non_exhaustive]
pub enum ContentTwins<'a> {
    /// Omit the content-twin scan.
    Skip,
    /// Run eligible scans without retaining a schedule between calls.
    #[default]
    Always,
    /// Carry the scan schedule across successive diagrams.
    Adaptive(&'a mut ContentTwinProbe),
}

/// Scheduling state for the content-twin canonicalization scan above its
/// size cap: below the cap every minimize scans, above it the first call scans
/// (`next_scan_at_nodes` starts at 0) and the next probe is scheduled at 4x the pre-scan
/// size — unless the scan landed back under the cap, which resets `next_scan_at_nodes` to
/// 0 so the next above-cap call scans again.
///
/// A caller that minimizes a *fresh* diagram each step (a bottom-up compile
/// accumulator, say) must keep one of these across the steps and hand it to
/// [`ContentTwins::Adaptive`]; state carried on the diagram itself
/// would reset to "always scan" every step. Passing none is equivalent to
/// passing a fresh probe: the scan runs and the updated schedule is discarded.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct ContentTwinProbe {
    /// Node count at which a skipped (above-cap) scan is re-attempted.
    /// 0 = scan on the next above-cap call.
    pub(crate) next_scan_at_nodes: u64,
}

use crate::engine::Engine;
use self::contract::contract_leaf::contract_leaf_twins;
use self::contract::contract_all_twins;
use self::prune::prune_unreachable;
use crate::limits::ApplyError;
use crate::diagram::Tdd;

/// Snapshot per-level `is_marginal` flags so a later
/// `assert_no_demarginalization` can detect a violation of invariant 5 (marginality is
/// permanent — see `test_helpers::check::marginal`) and name the offending pass.
#[cfg(debug_assertions)]
fn snapshot_marginal_flags(tdd: &Tdd) -> Vec<bool> {
    tdd.levels.iter().map(|l| l.is_marginal()).collect()
}

/// Assert no level present in `before` (as marginal) has become structural.
/// Panics naming the level and the `pass` that violated invariant 5.
#[cfg(debug_assertions)]
fn assert_no_demarginalization(tdd: &Tdd, before: &[bool], pass: &str) {
    for (i, &was_marginal) in before.iter().enumerate() {
        if was_marginal && !tdd.levels[i].is_marginal() {
            panic!(
                "invariant 5 violated: vtree level {i} was marginal before `{pass}` \
                 but is structural after — minimize must never un-marginalize a node \
                 A marginal node's mass may only roll UP into a \
                 marginalized parent, never be discarded."
            );
        }
    }
}

// ── Public minimize variants ─────────────────────────────────────────────

/// Minimize a diagram to its canonical form: afterwards every node is
/// reachable from the output and no two nodes at one level compute the same
/// function. The function and the model count are unchanged; marginal levels
/// stay marginal and their invariants are restored with the rest. ⊥ is left
/// as it is.
///
/// # Panics
///
/// Panics when an allocation is refused. A caller that must survive a refusal
/// calls [`try_minimize`] and handles [`ApplyError::OverBudget`]. Runs on a
/// fresh engine with no limits armed, so no deadline can fire inside it.
///
/// ```
/// use std::sync::Arc;
/// use tididi::Tdd;
/// use tididi::reduce::minimize;
/// use tididi::vtree::Vtree;
///
/// let vtree = Arc::new(Vtree::balanced(4));
/// let mut f = Tdd::clause(&vtree, [1, -2]) & Tdd::clause(&vtree, [2, 3]);
/// let before = (f.size(), f.model_count());
///
/// minimize(&mut f);
/// assert_eq!(f.model_count(), before.1);   // the function is unchanged
/// assert!(f.size() <= before.0);           // the representation is canonical
/// ```
pub fn minimize(f: &mut Tdd) {
    let eng = Engine::new();
    try_minimize(&eng, f, ReductionPlan::default())
        .expect("minimize: an allocation was refused; use try_minimize to handle it");
}

/// Fallible version of [`minimize`]: the passes `opts` selects, with every
/// allocation charged to the engine's limits. Only [`ReductionPlan::Full`]
/// establishes the canonical form.
///
/// # Errors
///
/// Returns `Err(ApplyError::OverBudget)` if a budget-gated reservation is
/// refused, or `Err(ApplyError::Deadline)` if an armed stop poll fires. Every
/// pass reserves its growth before it mutates anything, so on either error
/// the diagram is as it was at the last pass boundary: well-formed, and the
/// caller may keep and count it.
///
/// ```
/// use std::sync::Arc;
/// use tididi::{ApplyError, Engine, Tdd};
/// use tididi::limits::LimitSet;
/// use tididi::reduce::{try_minimize, ReductionPlan};
/// use tididi::vtree::Vtree;
///
/// let engine = Engine::new();
/// let vtree = Arc::new(Vtree::balanced(20_000));
/// let mut f = Tdd::clause(&vtree, [1, -2]) & Tdd::clause(&vtree, [2, 3]);
/// let before = f.model_count();
///
/// // A byte budget of zero refuses the first budget-gated pass.
/// let _armed = engine.limits().scope(LimitSet::none().budget(Some(0)));
/// match try_minimize(&engine, &mut f, ReductionPlan::default()) {
///     Ok(()) => {}
///     Err(e) => assert_eq!(e, ApplyError::OverBudget),
/// }
/// // Either way the diagram is well-formed and still counts the same.
/// assert_eq!(f.model_count(), before);
/// ```
pub fn try_minimize(eng: &Engine, f: &mut Tdd, opts: ReductionPlan<'_>) -> Result<(), ApplyError> {
    let _op = eng.limits().begin_operation();
    let content_twins = match opts {
        ReductionPlan::Contract => return contract_all_twins(eng, f),
        ReductionPlan::Prune => {
            // Prune removes nodes, which can create twins in a shrunk level's
            // children; `prune_unreachable` seeds the contract worklists with
            // those levels so a later contraction pass covers them.
            prune_unreachable(eng, f)?;
            // Pairs the prune removed may have orphaned marginal count slots.
            crate::reduce::slot_prune::prune_value_slots(eng, f);
            return Ok(());
        }
        ReductionPlan::Full(policy) => policy,
    };

    // Invariant 5 guard: snapshot the marginal flags before the structural
    // passes so `assert_no_demarginalization` can name the offending pass.
    #[cfg(debug_assertions)]
    let i1_snap = snapshot_marginal_flags(f);

    prune_unreachable(eng, f)?;
    #[cfg(debug_assertions)]
    assert_no_demarginalization(f, &i1_snap, "prune");

    contract_twins_and_leaves(eng, f)?;
    #[cfg(debug_assertions)]
    assert_no_demarginalization(f, &i1_snap, "contract+leaf");

    // Eligibility and the probe schedule are documented on `right_gated`, which
    // also runs the slot-prune sweep the structural passes above leave due.
    match content_twins {
        ContentTwins::Skip => {},
        ContentTwins::Always => content_twins::right_gated(eng, f, None)?,
        ContentTwins::Adaptive(probe) => content_twins::right_gated(eng, f, Some(probe))?,
    }

    // Release the doubling overshoot a rebuilt pair arena leaves behind.
    // `shrink_arrays` only acts when capacity exceeds 4x the length, so levels
    // without slack pay nothing.
    for level in &mut f.levels {
        level.shrink_arrays();
    }

    Ok(())
}

/// Twin contraction, then leaf-twin contraction, then twin contraction again
/// if the leaf pass fired. Both passes drain a worklist, so a clean diagram
/// costs one empty check each.
pub(super) fn contract_twins_and_leaves(eng: &Engine, tdd: &mut Tdd) -> Result<(), ApplyError> {
    contract_all_twins(eng, tdd)?;
    // Leaf labels are implicit indices, not stored nodes, so inner-node twin
    // contraction cannot reach them; the leaf rewrite can mint inner twins.
    if contract_leaf_twins(eng, tdd)? {
        contract_all_twins(eng, tdd)?;
    }
    Ok(())
}


#[cfg(test)]
mod tests;
