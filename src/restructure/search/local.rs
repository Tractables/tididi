//! Objective-generic greedy vtree-rotation search over a single compiled diagram.
//!
//! This is the small, principled, public form of vtree search: given a compiled
//! [`Tdd`], repeatedly rotate its vtree to descend some caller-chosen objective
//! until no single rotation improves it.
//!
//! The entry point minimizes a marginal-free input once up front, so a
//! non-canonical diagram is accepted directly; see `rotation_search_on`.
//!
//! # Probe / accept / revert mechanics
//!
//! Each sweep visits every internal vtree node bottom-up and probes both a left
//! and a right rotation at it through the shared `probe`: rotate, guard,
//! rebuild the two affected levels under a bound, then score the move with
//! [`RotationObjective::delta`] over the old vs. new two-level contents and
//! **accept iff the delta is strictly negative**. A rotation keeps a canonical
//! diagram canonical, so no reduction pass follows it. A declined probe
//! leaves the diagram bit-for-bit as it was. The search terminates when a full
//! sweep accepts nothing, or when `max_sweeps` is reached.
//!
//! # Locality
//!
//! The restructure touches only the levels at `v_idx` and `w_idx`
//! (rotation locality, argued in the `restructure::relevel` module doc), so an
//! objective scores a move from those two levels' before/after contents and a
//! revert restores only them. A rotation regroups the same products, so the
//! search preserves the model count for any objective, marginal diagrams
//! included.

use crate::limits::OperationError;
use crate::engine::Engine;
use crate::vtree::RotationKind;
use crate::vtree::rotate::RotationInfo;
use crate::diagram::{Tdd, TddLevel};
use crate::restructure::relevel::{return_scratch, take_scratch};

use super::probe::*;

/// Scores a candidate rotation for [`Engine::rotation_search`].
///
/// By Rotation Locality a rotation changes exactly
/// the two affected levels, so the objective is handed precisely their old and
/// new contents and nothing else.
pub trait RotationObjective {
    /// Score a probed rotation. `before` is the `(v, w)` levels prior to the
    /// rotation; `after` is the same two levels after the restructure.
    /// A **negative** result means the move improves the objective — the search
    /// accepts a rotation iff `delta < 0`.
    fn delta(
        &mut self,
        before: (&TddLevel, &TddLevel),
        after: (&TddLevel, &TddLevel),
    ) -> i64;
}

/// The default objective: minimize total diagram size, measured as the summed
/// input-pair count of the two affected levels (`live_pairs`, the
/// per-level component of [`Tdd::pair_count`]). Marginal levels contribute zero pairs,
/// so the metric falls back sensibly on marginal diagrams. By locality a negative
/// two-level delta is exactly a strict decrease in whole-diagram size.
pub(crate) struct SizeDelta;

impl RotationObjective for SizeDelta {
    fn delta(
        &mut self,
        before: (&TddLevel, &TddLevel),
        after: (&TddLevel, &TddLevel),
    ) -> i64 {
        let old = before.0.live_pairs() + before.1.live_pairs();
        let new = after.0.live_pairs() + after.1.live_pairs();
        new as i64 - old as i64
    }
}

/// Tunables for [`Engine::rotation_search`].
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct RotationSearchConfig {
    /// Bail bound forwarded to the bounded restructure probes: a rotation whose
    /// rebuilt level would exceed this many input pairs is abandoned (and the
    /// diagram restored) rather than materialized. `usize::MAX` never bails.
    pub max_inner_pairs: usize,
    /// Cap on the number of full sweeps. `None` runs to a local-minimum fixpoint
    /// (a sweep that accepts no rotation).
    pub max_sweeps: Option<usize>,
}

impl Default for RotationSearchConfig {
    fn default() -> Self {
        RotationSearchConfig { max_inner_pairs: usize::MAX, max_sweeps: None }
    }
}

/// Outcome counters for an [`Engine::rotation_search`] run.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct RotationSearchStats {
    /// Number of rotations scored by the objective (applicable, non-blocked probes).
    pub probes: usize,
    /// Number of rotations accepted (strict improvements that were kept).
    pub accepts: usize,
    /// Number of full bottom-up sweeps performed.
    pub sweeps: usize,
}

/// The search behind [`Engine::rotation_search`]: sweep every internal vtree
/// node, probing a left and a right rotation at each, and accept a move
/// whenever the objective strictly improves ([`RotationObjective::delta`]
/// `< 0`). Sweeps repeat until one accepts nothing (a local minimum) or
/// `config.max_sweeps` is hit. The engine's stop is polled once per pivot.
///
/// # Errors
///
/// [`OperationError::Stopped`] when the armed stop fires between pivots. The
/// diagram is left at whatever point the search had reached — canonical,
/// count-correct, and safe to keep or to search again.
pub(crate) fn rotation_search_on<O: RotationObjective>(
    eng: &Engine,
    tdd: &mut Tdd,
    objective: &mut O,
    config: &RotationSearchConfig,
) -> Result<RotationSearchStats, OperationError> {
    let _op = eng.limits().begin_operation();
    let mut stats = RotationSearchStats { probes: 0, accepts: 0, sweeps: 0 };
    let mut rule = Counted { objective, probes: 0, accepts: 0 };
    // Pooled across searches on this thread (cleared on take, so behavior is
    // capacity-only) — see `restructure::scratch::take_scratch`.
    let mut scratch = take_scratch(eng);

    // Rotation-locality precondition: the locality assertion and the v/w-only
    // probe revert need a canonical input, so establish it once here.
    // Marginal-free diagrams only: in marginal context the restructure keeps
    // the child multiset without Boolean dedup, which is what preserves the
    // count, and a minimize here would collapse it.
    if !tdd.levels.iter().any(|l| l.is_marginal()) {
        crate::reduce::try_minimize(eng, tdd, crate::reduce::ReductionPlan::default())?;
    }

    loop {
        if let Some(cap) = config.max_sweeps
            && stats.sweeps >= cap {
                break;
            }
        stats.sweeps += 1;

        // Unevaluated snapshot of the internal nodes each sweep: an accept relabels
        // parent/child relations, so re-collecting keeps the walk honest (a stale
        // index can only mis-skip a probe, never break count-soundness — the next
        // sweep re-picks it up).
        let internals: Vec<crate::vtree::VtreeIdx> =
            tdd.vtree.internal_bottomup().map(|(v, _, _)| v).collect();

        let mut accepted_this_sweep = 0usize;
        for v in internals {
            // Once per pivot, not once per probe: the two probes below share
            // the pivot's setup, and a stop between them would leave the
            // sweep's accept count describing half a pivot.
            if eng.limits().should_stop() {
                return_scratch(eng, scratch);
                return Err(OperationError::Stopped);
            }
            for &kind in &[RotationKind::Left, RotationKind::Right] {
                let kept = probe(
                    eng, tdd, v, kind, &mut rule, &mut scratch, config.max_inner_pairs,
                )?;
                if kept {
                    accepted_this_sweep += 1;
                }
            }
        }
        if accepted_this_sweep == 0 {
            break;
        }
    }
    return_scratch(eng, scratch);
    stats.probes = rule.probes;
    stats.accepts = rule.accepts;
    Ok(stats)
}

/// The search's own [`ProbeRule`]: the caller's objective plus the tallies
/// [`RotationSearchStats`] reports. `delta` runs once per scored probe and
/// `on_accept` once per kept rotation, so the counts are the definitions.
struct Counted<'a, O> {
    objective: &'a mut O,
    probes: usize,
    accepts: usize,
}

impl<O: RotationObjective> RotationObjective for Counted<'_, O> {
    fn delta(
        &mut self,
        before: (&TddLevel, &TddLevel),
        after: (&TddLevel, &TddLevel),
    ) -> i64 {
        self.probes += 1;
        self.objective.delta(before, after)
    }
}

impl<O: RotationObjective> ProbeRule for Counted<'_, O> {
    fn on_accept(
        &mut self,
        _eng: &Engine,
        _tdd: &mut Tdd,
        _info: &RotationInfo,
    ) -> Result<(), OperationError> {
        self.accepts += 1;
        Ok(())
    }
}

#[cfg(test)]
mod tests;
