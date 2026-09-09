//! The content-twin canonicalization fixpoint and its size gate.

use crate::engine::Engine;
use crate::error::ApplyError;
use crate::diagram::Tdd;

use super::{contract_leaf_twins, contract_only, instrumented_prune, ContentTwinProbe};

// ── Content-twin scan mode ─────────────────────────────────────────────────────────────
//
// Content-twin scan policy (fixed): the scan runs on every
// minimize that has marginal levels — leaf marginalization (always on) mints
// `(X, Inline(_))` content twins that only the content merge
// collapses — subject to the galloping-probe node cap below. Marg-free TDDs skip the
// scan (the merge stands down without a marginal level). Not runtime-configurable,
// and not triggered by memory pressure.

// Node count below which the content-twin scan runs on every minimize. Above
// it the scan still runs, but only when the galloping probe below says so.
//
// A policy value, not a derived one: the scan is O(nodes) with a hashing
// constant, so on a small diagram it is free relative to the passes around it
// and on a large one it is not. 2^17 is where that stops being true in
// practice. Changing it trades reduction quality for pass cost; it is not
// runtime-configurable, so one number describes every run.
pub(super) const C2_SCAN_MAX_NODES: u64 = 131_072;

// The content-twin worklist is always on — no escape hatch. Every scan after
// the first rescans only the levels the previous one touched; an empty worklist
// breaks the fixpoint early.

/// The content-twin canonicalization pass with its own
/// galloping-probe size gate, split out of `try_minimize` for readability.
/// Sole home for its gating rationale; `try_minimize` just calls it.
///
/// The scan only fires when a marginal level exists (`is_marginal()`); below the
/// node cap every call scans (cheap insurance), above it the galloping probe
/// (`probe.next_scan_at_nodes`) throttles to 4×-growth intervals. So this is cheap
/// when not needed and bounded when it is. A caller that wants the schedule to
/// survive across calls passes its own [`ContentTwinProbe`]; `None` behaves
/// like a fresh probe (scan now, updated schedule discarded).
///
/// ELIGIBILITY. Boundary content-twins require a marginal level, so marg-free
/// TDDs skip the scan entirely (empty boundary set → guaranteed no-op, but it
/// would still cost an O(levels) walk + allocations per minimize call). When
/// marginal levels are present the scan runs unconditionally (subject to the
/// probe cap below): leaf marginalization (always on) inlines a leaf's 0/1/2
/// count at its boundary parent, minting `(X, Inline(1))`-style content twins
/// that ONLY the content-twin boundary-parent merge collapses — so every
/// marginal compile needs it.
///
/// WEIGHTED mode FORCES the scan on (bypasses the cap). Integer
/// counting parks free-var multiplicity in the count, so equal-count twins
/// merge through ordinary canonicalization; weighted marg-side refs are
/// per-node slots, so equal-VALUE twins stay distinct unless the content-twin
/// merge (with `duplicate_pair_resolve`'s weighted value-scaling) collapses them. Without
/// it a weighted compile grows about as 2^free. Keyed on the attached weight
/// store, so the integer solve record
/// (set with the normal-path scan disabled) is untouched.
///
/// GALLOPING-PROBE POLICY: below the cap every minimize scans (cheap
/// insurance); above it the first call always scans (`next_scan_at_nodes` starts 0), then
/// the next probe is scheduled at 4× the pre-scan size — unless the scan
/// dropped the TDD back under the cap, which resets to
/// scan-on-next-above-cap-call. No productivity branch: node shrink is NOT a
/// payoff signal (the wasteful monster scans shrink the most), so rewarding
/// shrink with a sooner probe
/// just re-fires them. Cap fixed at `C2_SCAN_MAX_NODES` (2^17).
pub(super) fn c2_gated(
    eng: &Engine,
    tdd: &mut Tdd,
    probe: Option<&mut ContentTwinProbe>,
) -> Result<(), ApplyError> {
    let mut scratch = ContentTwinProbe::default();
    let probe = probe.unwrap_or(&mut scratch);
    if tdd.has_marginal_level() {
        let node_count: u64 = tdd.levels.iter().map(|l| l.nodes.len() as u64).sum();
        let cap = C2_SCAN_MAX_NODES;
        let run = tdd.weights.is_some() // weighted: scan every minimize (bypass cap)
            || node_count <= cap
            || node_count >= probe.next_scan_at_nodes;
        if run {
            canonicalize_content_twins(eng, tdd)?;
            // Update galloping-probe state: schedule the next above-cap probe
            // at 4x the pre-scan size; a scan that lands back under the cap
            // resets the schedule (next above-cap call fires immediately).
            //
            // The 4x multiple is deliberately yield-BLIND. A yield-aware
            // back-off — probe less often after a probe that merged little — was
            // measured and lost badly: even a zero-yield probe is load-bearing
            // SIZE CONTROL, because deferring it lets the working diagram bloat
            // and every pass in between (twin scans, p-fusion plan scans, the
            // eventual content-twin scan itself) then runs on the bigger
            // diagram. 4 is a policy value, like the cap it schedules against.
            let total_nodes_after: u64 =
                tdd.levels.iter().map(|l| l.nodes.len() as u64).sum();
            probe.next_scan_at_nodes = if total_nodes_after > cap {
                node_count.saturating_mul(4)
            } else {
                0
            };
        } // end if run (galloping-probe gate)
    }
    Ok(())
}

/// Run the content-twin canonicalization fixpoint on `tdd`.
///
/// This is the inner body of the galloping-probe gate in `try_minimize`: it
/// slot-prunes, then iterates the content-twin scan
/// (`merge_content_equal_nodes`) / node-prune /
/// `contract_only` / `contract_leaf_twins` loop to a fixpoint.
///
/// The calling context in `try_minimize` owns the gating logic (enabled check,
/// marginal-level check, probe cap / run decision) and the probe-state update
/// (`ContentTwinProbe::next_scan_at_nodes`) — this function performs only the canonicalization
/// work itself.
///
/// `pub(crate)` so that tests can call it directly to exercise twin canonicality
/// and slot-count uniqueness independent of the production scan policy.
///
/// The loop always runs to fixpoint (no wall-time budget).
pub(crate) fn canonicalize_content_twins(eng: &Engine, tdd: &mut Tdd) -> Result<(), ApplyError> {
    // Pre-loop slot-prune.
    let pre_stats = crate::reduce::slot_prune::prune_value_slots(eng, tdd);

    // Worklist-driven fixpoint setup.
    // c2_rescan accumulates dirtied vtree indices during each iteration; at the
    // END of one it is drained into `next_filter` and restricts the next scan.
    // The first iteration always scans every explicit level. Clear c2_rescan at
    // loop entry so entries left by work outside this call cannot contaminate
    // the first worklist.
    tdd.clear_c2_worklist();
    // Seed the worklist from the pre-loop slot-prune value-merged levels, so the
    // first iteration's outgoing filter is non-empty when slot-prune already
    // changed something. (The first scan is full either way.)
    tdd.extend_c2_worklist(pre_stats.value_merged_levels.iter().copied());

    // `next_filter`: None = full scan (the first iteration), Some(set) =
    // worklist scan. Drained from c2_rescan at the end of each iteration.

    // TERMINATION. Each iteration either merges at least one content twin — which
    // strictly decreases the node count, and prune then removes the merged nodes
    // — or merges none and breaks. The node count is a non-negative integer, so
    // the loop cannot run forever; the warning below is for a bug that violates
    // that (a scan and a contract undoing one another), not for a slow diagram.
    let mut next_filter: Option<rustc_hash::FxHashSet<u32>> = None;

    let mut fixpoint_iters = 0u32;
    loop {
        // Worklist early-break: if the filter is non-None and empty, no level
        // was touched last iteration, so no new content twins can exist. Never
        // fires on the first iteration, which uses filter=None.
        if let Some(ref set) = next_filter
            && set.is_empty() {
                break;
            }
        fixpoint_iters += 1;
        if fixpoint_iters.is_multiple_of(64) {
            eprintln!(
                "WARN content-twin fixpoint slow: {} iterations (suspect scan/contract cycle)",
                fixpoint_iters
            );
        }

        // Clear c2_rescan first, so it collects only THIS iteration's mutations.
        tdd.clear_c2_worklist();

        // Step 1: content-twin scan over every explicit level (children before
        // parents, so one pass chases the merge cascade upward), optionally
        // filtered to the worklist.  EVERY dup gets its parent refs (and the
        // output ref) rewritten onto the canonical node and is left unreferenced
        // for prune — including co-referenced twins, whose rewrite mints a
        // duplicate pair at the parent.  Those duplicates are legal multiset
        // entries (see the ruling in `merge_content_equal_nodes`'s doc); the scan
        // dirties the parent so p-fusion folds them wherever that level is
        // marg-flagged.
        let filter_ref = next_filter.as_ref();
        let merged =
            crate::reduce::contract::content_twin::merge_content_equal_nodes(
                eng, tdd, filter_ref,
            )?;
        if merged == 0 {
            // Clean: no content-twin remains anywhere the filter reached.
            break;
        }
        // Step 2: node-prune GCs the now-unreferenced dup nodes through the
        // established reachability machinery (the merge pass must not
        // tombstone them itself — streaming applies assert tombstone-free
        // levels). Also reseeds contract worklists for shrunk levels
        // (`invalidate` → the rescan list).  `merged > 0` here (the loop broke
        // otherwise), so the prune always has work.
        instrumented_prune(eng, tdd)?;
        // Step 3: context-based contract — merges any fresh twins created by
        // the grandparent ref rewrite in step 1 (concat merge; duplicate pairs
        // are legal multiset entries at the marg-flagged boundary level).
        // contract_all_twins_topdown pushes fired parents to c2_rescan.
        contract_only(eng, tdd)?;
        if contract_leaf_twins(eng, tdd) {
            contract_only(eng, tdd)?;
        }

        let slot_stats = crate::reduce::slot_prune::prune_value_slots(eng, tdd);
        // Feed slot-prune value-merged levels into the worklist: a value merge
        // at marginal level v can mint new content-twins at v's parent.
        tdd.extend_c2_worklist(slot_stats.value_merged_levels.iter().copied());

        // Drain c2_rescan into the next filter set (dedup via the hash set).
        // Round 1 (next_filter is None) transitions to Some after the first
        // round; subsequent rounds replace the set in place.
        let raw = tdd.take_c2_worklist();
        let mut set: rustc_hash::FxHashSet<u32> = rustc_hash::FxHashSet::default();
        set.extend(raw);
        next_filter = Some(set);
    }
    // Clear c2_rescan on exit so the field is empty outside this call.
    tdd.clear_c2_worklist();

    Ok(())
}
