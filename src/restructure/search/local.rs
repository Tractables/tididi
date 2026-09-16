//! Greedy vtree-rotation search under a caller-supplied objective.
//!
//! Each bottom-up sweep probes left and right rotations, retaining strict
//! improvements. Only the two affected levels are rebuilt and scored; a
//! rejected probe restores them. Sweeps stop at a local minimum or the
//! configured limit. Marginal-free inputs are minimized before the first sweep.

use crate::limits::OperationError;
use crate::engine::Engine;
use crate::vtree::RotationKind;
use crate::vtree::rotate::RotationInfo;
use crate::diagram::{Tdd, TddLevel};

use super::probe::*;

/// Scores a candidate rotation for [`Tdd::rotation_search`].
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

/// Minimize the summed pair count of the two affected levels.
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

/// Limits on the work performed by [`Tdd::rotation_search`].
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

/// Work performed by [`Tdd::rotation_search`].
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
    let mut scratch = eng.restructure().checkout();

    // Rotation-locality precondition: the locality assertion and the v/w-only
    // probe revert need a canonical input, so establish it once here.
    // Marginal-free diagrams only: in marginal context the restructure keeps
    // the child multiset without Boolean dedup, which is what preserves the
    // count, and a minimize here would collapse it.
    if !tdd.levels.iter().any(|l| l.is_marginal()) {
        eng.reduce(tdd, crate::reduce::ReductionPlan::default())?;
    }

    let mut search = SearchTree::new(tdd);
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
            search.tdd.vtree.internal_bottomup().map(|(v, _, _)| v).collect();

        let mut accepted_this_sweep = 0usize;
        for v in internals {
            // Once per pivot, not once per probe: the two probes below share
            // the pivot's setup, and a stop between them would leave the
            // sweep's accept count describing half a pivot.
            if eng.limits().should_stop() {
                return Err(OperationError::Stopped);
            }
            for &kind in &[RotationKind::Left, RotationKind::Right] {
                let kept = probe(
                    eng, search.tdd, v, kind, &mut rule, &mut scratch, config.max_inner_pairs,
                )?;
                if kept {
                    search.original = None;
                    accepted_this_sweep += 1;
                }
            }
        }
        if accepted_this_sweep == 0 {
            break;
        }
    }
    stats.probes = rule.probes;
    stats.accepts = rule.accepts;
    Ok(stats)
}

/// Keep one private vtree across probes, restoring shared identity if none was accepted.
struct SearchTree<'a> {
    tdd: &'a mut Tdd,
    original: Option<std::sync::Arc<crate::Vtree>>,
}

impl<'a> SearchTree<'a> {
    /// Detach a shared tree once before probing any pivots.
    fn new(tdd: &'a mut Tdd) -> Self {
        use std::sync::Arc;
        let original = (Arc::strong_count(&tdd.vtree) > 1 || Arc::weak_count(&tdd.vtree) > 0)
            .then(|| Arc::clone(&tdd.vtree));
        Arc::make_mut(&mut tdd.vtree);
        Self { tdd, original }
    }
}

impl Drop for SearchTree<'_> {
    fn drop(&mut self) {
        if let Some(original) = self.original.take() { self.tdd.vtree = original; }
    }
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
#[path = "tests/local/mod.rs"]
mod tests;
