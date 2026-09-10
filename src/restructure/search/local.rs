//! Objective-generic greedy vtree-rotation search over a single compiled diagram.
//!
//! This is the small, principled, public form of vtree search: given a compiled
//! [`Tdd`], repeatedly rotate its vtree to descend some caller-chosen objective
//! until no single rotation improves it.
//!
//! The entry point canonicalizes a marginal-free input once up front (via the shared
//! `minimize`) so the per-probe locality machinery — single-level twin
//! contraction, the narrow v/w-only revert, the v/w-only size delta — has the
//! canonical input rotation locality requires. A correct-count but
//! non-canonical diagram (e.g. clause-by-clause `apply_and_clause`, which never
//! runs a global contraction) is therefore accepted directly; see
//! `rotation_search`.
//!
//! # Probe / accept / revert mechanics
//!
//! Each sweep visits every internal vtree node bottom-up and probes both a left
//! and a right rotation at it through the shared `core::probe`: rotate, guard,
//! rebuild the two affected levels under a bound, re-minimize, then score the
//! move with [`RotationObjective::delta`] over the old vs. new two-level
//! contents and **accept iff the delta is strictly negative**. A declined probe
//! leaves the diagram bit-for-bit as it was. The search terminates when a full
//! sweep accepts nothing, or when `max_sweeps` is reached.
//!
//! # Locality argument
//!
//! A vtree rotation is O(1) pointer surgery on `(v_idx, w_idx)`; the restructure
//! and re-minimize touch **only** those two levels (Rotation Locality). Every
//! other level is bit-for-bit unchanged.
//! Hence an objective only needs the before/after contents of those two levels
//! to score the move, and a revert only needs to restore them — no whole-diagram
//! snapshot or resize is required per probe. Because rotations are pure variable
//! reorders, the search is model-count-preserving for any objective (including
//! for marginal diagrams: the bounded restructure uses full multiset expansion in
//! marginal context, so `#F` survives — see `fuzz_search_preserves_marginal_count`
//! in `restructure/relevel.rs`).

use crate::error::ApplyError;
use crate::engine::Engine;
use crate::vtree::RotationKind;
use crate::vtree::rotate::RotationInfo;
use crate::diagram::{Tdd, TddLevel};
use crate::restructure::relevel::{return_scratch, take_scratch};

use super::core::*;

/// Scores a candidate rotation for [`rotation_search`].
///
/// By Rotation Locality a rotation changes exactly
/// the two affected levels, so the objective is handed precisely their old and
/// new contents and nothing else.
pub trait RotationObjective {
    /// Score a probed rotation. `before` is the `(v, w)` levels prior to the
    /// rotation; `after` is the same two levels after restructure + minimize.
    /// A **negative** result means the move improves the objective — the search
    /// accepts a rotation iff `delta < 0`.
    fn delta(
        &mut self,
        before: (&TddLevel, &TddLevel),
        after: (&TddLevel, &TddLevel),
    ) -> i64;
}

/// The default objective: minimize total diagram size, measured as the summed
/// input-pair count of the two affected levels (`level_pair_count`, the
/// per-level component of [`Tdd::size`]). Marginal levels contribute zero pairs,
/// so the metric falls back sensibly on marginal diagrams. By locality a negative
/// two-level delta is exactly a strict decrease in whole-diagram size.
pub(crate) struct SizeDelta;

impl RotationObjective for SizeDelta {
    fn delta(
        &mut self,
        before: (&TddLevel, &TddLevel),
        after: (&TddLevel, &TddLevel),
    ) -> i64 {
        let old = level_pair_count(before.0) + level_pair_count(before.1);
        let new = level_pair_count(after.0) + level_pair_count(after.1);
        new as i64 - old as i64
    }
}

/// Tunables for [`rotation_search`].
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

/// Outcome counters for a [`rotation_search`] run.
#[derive(Debug, Clone)]
pub struct RotationSearchStats {
    /// Number of rotations scored by the objective (applicable, non-blocked probes).
    pub probes: usize,
    /// Number of rotations accepted (strict improvements that were kept).
    pub accepts: usize,
    /// Number of full bottom-up sweeps performed.
    pub sweeps: usize,
}

/// Greedy local vtree-rotation search under a caller-chosen `objective`.
///
/// Sweeps every internal vtree node, probing a left and a right rotation at each,
/// accepting a move whenever the objective strictly improves ([`RotationObjective::delta`]
/// `< 0`), and re-minimizing after each accept. Sweeps repeat until one accepts
/// nothing (a local minimum) or `config.max_sweeps` is hit.
///
/// Model-count-preserving for any objective — rotations are pure variable
/// reorders, sound even for marginal diagrams (the bounded restructure uses full
/// multiset expansion in marginal context).
///
/// Returns a [`RotationSearchStats`] recording the probe, accept, and sweep tallies.
///
/// The search is objective-generic: implement [`RotationObjective`] to descend a
/// metric other than size. Here a custom objective accepts a rotation only when it
/// shrinks the *wider* of the two affected levels:
///
/// ```
/// use std::sync::Arc;
/// use tididi::Tdd;
/// use tididi::restructure::search::{rotation_search, RotationObjective, RotationSearchConfig};
/// use tididi::diagram::TddLevel;
/// use tididi::vtree::Vtree;
///
/// struct MinPeak;
/// impl RotationObjective for MinPeak {
///     fn delta(&mut self, b: (&TddLevel, &TddLevel), a: (&TddLevel, &TddLevel)) -> i64 {
///         a.0.width().max(a.1.width()) as i64 - b.0.width().max(b.1.width()) as i64
///     }
/// }
///
/// let vtree = Arc::new(Vtree::balanced(4));
/// let mut f = Tdd::clause(&vtree, [1, 2]) & Tdd::clause(&vtree, [3, 4]);
/// let before = f.model_count();
/// let stats = rotation_search(&mut f, &mut MinPeak, &RotationSearchConfig::default());
/// assert_eq!(f.model_count(), before); // count-preserving under any objective
/// assert!(stats.sweeps >= 1);
/// ```
pub fn rotation_search<O: RotationObjective>(
    tdd: &mut Tdd,
    objective: &mut O,
    config: &RotationSearchConfig,
) -> RotationSearchStats {
    rotation_search_on(&Engine::new(), tdd, objective, config)
        .expect("rotation_search: nothing armed on a transient engine, so no stop can fire")
}

/// [`rotation_search`] on a caller's engine, polling its stop once per pivot.
///
/// The engine method [`Engine::rotation_search`] is this function; the free
/// entry above is it on an unarmed transient engine plus an `expect`.
///
/// # Errors
///
/// [`ApplyError::Deadline`] when the armed stop fires between pivots. The
/// diagram is left at whatever point the search had reached — canonical,
/// count-correct, and safe to keep or to search again.
pub(crate) fn rotation_search_on<O: RotationObjective>(
    eng: &Engine,
    tdd: &mut Tdd,
    objective: &mut O,
    config: &RotationSearchConfig,
) -> Result<RotationSearchStats, ApplyError> {
    let mut stats = RotationSearchStats { probes: 0, accepts: 0, sweeps: 0 };
    let mut rule = Counted { objective, probes: 0, accepts: 0 };
    // Pooled across searches on this thread (cleared on take, so behavior is
    // capacity-only) — see `restructure::scratch::take_scratch`.
    let mut scratch = take_scratch(eng);

    // Rotation-locality precondition. The single-level locality tightening
    // this search relies on at every probe — the debug-asserted "only w_idx gets
    // fresh twins" (`minimize_after_rotation` → `contract_all_twins_with_locality`),
    // the narrow v/w-only probe revert in `core::probe`, and the v/w-only size
    // delta — all hold only for a canonical (fully twin-contracted) input. A
    // public caller may legitimately hand us a correct-count but non-canonical
    // diagram: e.g. the api-guide's clause-by-clause `Tdd::one` +
    // `apply_and_clause` pattern, which rebuilds only each clause's spine and
    // never runs a global twin contraction, so residual twins (and stale
    // `dirty_contract` entries) survive. On such an input the first probe's
    // contract pass resolves those pre-existing twins at a level *above* w_idx,
    // tripping the locality assertion (twin equivalence is semantic: a
    // non-canonical diagram can re-surface an unresolved twin at a different level
    // after a rotation). Establish the precondition once, up front, via the
    // shared `minimize` (the single canonicalization source of truth — no
    // duplicated contract loop). On an already-canonical diagram (the internal
    // caller's common case) this is a provable O(1) no-op: both dirty worklists
    // are empty, so prune and contract early-return without touching a level.
    //
    // Gated to marginal-free diagrams, mirroring the assertion's own `!has_marginal`
    // gate: in marginal context the bounded restructure deliberately keeps the
    // child multiset *without* Boolean dedup (count-safety comes from the
    // preserved multiset, not canonicalization — see `minimize_after_rotation`'s
    // marginal exception and `rotate.rs`'s full-expand path), the locality
    // assertion is off, and running a full minimize here would collapse the
    // count-bearing twin multiset that path must preserve.
    if !tdd.levels.iter().any(|l| l.is_marginal()) {
        crate::reduce::minimize(tdd);
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
                return Err(ApplyError::Deadline);
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
    ) -> Result<(), ApplyError> {
        self.accepts += 1;
        Ok(())
    }
}

#[cfg(test)]
#[path = "local_tests.rs"]
mod tests;
