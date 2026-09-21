//! The content-twin canonicalization fixpoint and its size gate.

use crate::diagram::Pass;
use crate::Engine;
use crate::limits::OperationError;
use crate::diagram::Tdd;

use super::{contract_twins_and_leaves, prune_unreachable, ContentTwinSchedule};
use super::prune::PruneScope;

// Node count below which the content-twin scan runs on every minimize; above
// it the scan runs only when the galloping probe in `scan_if_due` says so.
// The scan is O(nodes) with a hashing constant, so on a small diagram it is
// cheap next to the passes around it and on a large one it is not.
pub(super) const C2_SCAN_MAX_NODES: u64 = 131_072;

/// Run the content-twin canonicalization when the diagram is eligible and the
/// probe schedule says so.
///
/// The scan runs only on a diagram with a marginal level, since content twins
/// need one. Below `C2_SCAN_MAX_NODES` every call scans; above it the probe
/// schedule on [`ContentTwinSchedule`] decides, and `None` behaves like a fresh
/// probe. Weighted mode always scans: weighted marginal-side refs are per-node
/// slots, so twins holding equal values stay distinct until the content-twin
/// merge collapses them.
pub(super) fn scan_if_due(
    eng: &Engine,
    tdd: &mut Tdd,
    probe: Option<&mut ContentTwinSchedule>,
) -> Result<(), OperationError> {
    let mut scratch = ContentTwinSchedule::default();
    let probe = probe.unwrap_or(&mut scratch);
    if tdd.has_marginal_level() {
        let node_count: u64 = tdd.levels.iter().map(|l| l.nodes.len() as u64).sum();
        let cap = C2_SCAN_MAX_NODES;
        let run = tdd.weights.is_some()
            || node_count <= cap
            || node_count >= probe.next_scan_at_nodes;
        if run {
            canonicalize_content_twins(eng, tdd)?;
            // Schedule the next above-cap probe at 4x the pre-scan size; a scan
            // that lands back under the cap resets the schedule.
            let total_nodes_after: u64 =
                tdd.levels.iter().map(|l| l.nodes.len() as u64).sum();
            probe.next_scan_at_nodes = if total_nodes_after > cap {
                node_count.saturating_mul(4)
            } else {
                0
            };
        }
    }
    Ok(())
}

/// Run the content-twin canonicalization to a fixpoint on `tdd`: a slot
/// prune, then rounds of content-twin merge, node prune, twin contraction and
/// slot prune until a round merges nothing. Precondition: post-tagger form,
/// as `prune_value_slots` requires.
pub(crate) fn canonicalize_content_twins(eng: &Engine, tdd: &mut Tdd) -> Result<(), OperationError> {
    let pre_stats = crate::reduce::slot_prune::prune_value_slots(eng, tdd);

    // The rescan worklist collects the levels each round touches; it is
    // drained into `next_filter` and restricts the next scan. Clear it at
    // entry so nothing left by work outside this call reaches the first
    // worklist, then seed it with the slot prune's value merges.
    tdd.dirty.clear(Pass::ContentTwin);
    tdd.dirty.requeue(Pass::ContentTwin, pre_stats.value_merged_levels.iter().copied());

    // `None` is the full scan of the first round; `Some(set)` a worklist scan.
    // Each round either merges a content twin, which the prune then removes,
    // strictly decreasing the node count, or merges none and breaks.
    let mut next_filter: Option<rustc_hash::FxHashSet<u32>> = None;

    loop {
        // Termination is argued from the node count below, not bounded by a
        // count, so this is where an installed deadline or stop callback cuts
        // in. A round is a scan of at least one level, so one test per round
        // costs nothing next to it.
        eng.limits().check_stop()?;

        // An empty worklist means no level was touched last round, so no new
        // content twin can exist.
        if let Some(ref set) = next_filter
            && set.is_empty() {
                break;
            }
        tdd.dirty.clear(Pass::ContentTwin);

        // Each duplicate has its parent refs (and the output ref) rewritten
        // onto the canonical node and is left unreferenced for the prune. The
        // rewrite may mint a duplicate pair at the parent; the scan dirties
        // the parent so pair fusion folds it where the level is marginal.
        let filter_ref = next_filter.as_ref();
        let merged =
            crate::reduce::contract::content_twin::merge_content_equal_nodes(
                eng, tdd, filter_ref,
            )?;
        if merged == 0 {
            break;
        }
        // The prune removes the unreferenced duplicates and seeds the contract
        // worklists with the levels it shrank; the contraction then merges the
        // context twins the ref rewrite created.
        prune_unreachable(eng, tdd, PruneScope::Whole)?;
        contract_twins_and_leaves(eng, tdd)?;

        let slot_stats = crate::reduce::slot_prune::prune_value_slots(eng, tdd);
        // A value merge at marginal level v can mint content twins at v's parent.
        tdd.dirty.requeue(Pass::ContentTwin, slot_stats.value_merged_levels.iter().copied());

        let raw = tdd.dirty.take(Pass::ContentTwin);
        let mut set: rustc_hash::FxHashSet<u32> = rustc_hash::FxHashSet::default();
        set.extend(raw);
        next_filter = Some(set);
    }
    // Leave the worklist empty outside this call.
    tdd.dirty.clear(Pass::ContentTwin);

    Ok(())
}
