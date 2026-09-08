//! Objective-generic greedy vtree-rotation search over a single compiled TDD.
//!
//! This is the small, principled, public form of vtree search: given a compiled
//! [`Tdd`], repeatedly rotate its vtree to descend some caller-chosen objective
//! until no single rotation improves it.
//!
//! The entry point canonicalizes a marg-free input once up front (via the shared
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
//! and a right rotation at it. A probe:
//!
//! 1. clones the vtree, rotates the clone, and reads the resulting
//!    [`RotationInfo`] (the two affected node indices `v_idx`/`w_idx`);
//! 2. installs the rotated vtree, then calls the bounded restructure to rebuild
//!    the two affected levels — which returns the *old* `(v, w)` levels on
//!    success (kept for a cheap revert) or `None` if it bailed (having already
//!    restored the diagram);
//! 3. re-minimizes locally via [`minimize_after_rotation`];
//! 4. scores the move with [`RotationObjective::delta`] over the old vs. new
//!    two-level contents, and **accepts iff the delta is strictly negative**.
//!
//! On accept the topo order is fixed up in place and the move stands. On reject
//! the diagram is restored exactly: the saved vtree `Arc` is reinstated and the
//! two levels are overwritten with the returned old levels (plus the saved
//! `output`). The search terminates when a full sweep accepts nothing, or when
//! `max_sweeps` is reached.
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
//! in `tdd/restructure/relevel.rs`).

use std::sync::Arc;

use crate::vtree::RotationKind;
use crate::vtree::rotate::{rotate_left, rotate_right, RotationInfo};
use crate::diagram::{Tdd, TddLevel};
use crate::reduce::minimize_after_rotation;
use crate::restructure::relevel::{RestructureScratch, return_scratch, take_scratch};

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
pub struct SizeDelta;

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
    let mut stats = RotationSearchStats { probes: 0, accepts: 0, sweeps: 0 };
    // Pooled across searches on this thread (cleared on take, so behavior is
    // capacity-only) — see `rotate::take_scratch`.
    let mut scratch = take_scratch();

    // Rotation-locality precondition. The single-level locality tightening
    // this search relies on at every probe — the debug-asserted "only w_idx gets
    // fresh twins" (`minimize_after_rotation` → `contract_all_twins_with_locality`),
    // the narrow v/w-only probe revert in `try_rotate`, and the v/w-only size
    // delta — all hold ONLY for a CANONICAL (fully twin-contracted) input. A
    // public caller may legitimately hand us a correct-count but NON-canonical
    // diagram: e.g. the api-guide's clause-by-clause `Tdd::one` +
    // `apply_and_clause` pattern, which rebuilds only each clause's spine and
    // never runs a global twin contraction, so residual twins (and stale
    // `dirty_contract` entries) survive. On such an input the first probe's
    // contract pass resolves those pre-existing twins at a level *above* w_idx,
    // tripping the locality assertion (twin equivalence is semantic: a
    // non-canonical TDD can re-surface an unresolved twin at a different level
    // after a rotation). Establish the precondition once, up front, via the
    // shared `minimize` (the single canonicalization source of truth — no
    // duplicated contract loop). On an already-canonical diagram (the internal
    // caller's common case) this is a provable O(1) no-op: both dirty worklists
    // are empty, so prune and contract early-return without touching a level.
    //
    // Gated to marg-free diagrams, mirroring the assertion's own `!has_marginal`
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
        if let Some(cap) = config.max_sweeps {
            if stats.sweeps >= cap {
                break;
            }
        }
        stats.sweeps += 1;

        // Fresh snapshot of the internal nodes each sweep: an accept relabels
        // parent/child relations, so re-collecting keeps the walk honest (a stale
        // index can only mis-skip a probe, never break count-soundness — the next
        // sweep re-picks it up).
        let internals: Vec<crate::vtree::VtreeIdx> =
            tdd.vtree.internal_bottomup().map(|(v, _, _)| v).collect();

        let mut accepted_this_sweep = 0usize;
        for v in internals {
            for &kind in &[RotationKind::Left, RotationKind::Right] {
                if try_rotate(tdd, v, kind, objective, config, &mut scratch, &mut stats) {
                    accepted_this_sweep += 1;
                }
            }
        }
        if accepted_this_sweep == 0 {
            break;
        }
    }
    return_scratch(scratch);
    stats
}

/// Probe one `(pivot, kind)` rotation and keep it iff the objective improves.
/// Returns whether the move was accepted. On reject the diagram is restored
/// bit-for-bit. Bumps `stats.probes`/`stats.accepts`.
fn try_rotate<O: RotationObjective>(
    tdd: &mut Tdd,
    v: crate::vtree::VtreeIdx,
    kind: RotationKind,
    objective: &mut O,
    config: &RotationSearchConfig,
    scratch: &mut RestructureScratch,
    stats: &mut RotationSearchStats,
) -> bool {
    // Clone-and-rotate on a throwaway vtree; `None` = inapplicable at this pivot
    // (e.g. a rotation child is a leaf) — nothing installed, nothing to undo.
    let mut vt = (*tdd.vtree).clone();
    let info: Option<RotationInfo> = match kind {
        RotationKind::Left => rotate_left(&mut vt, v),
        RotationKind::Right => rotate_right(&mut vt, v),
    };
    let Some(info) = info else { return false };

    // v/w-marginal rotations are genuinely unhandled (their pairs would have to
    // be decomposed into children that no longer exist); grandchild-marginal is
    // count-safe and proceeds. The pointer rotation left the levels untouched, so
    // reading them here (still on the original vtree) is valid.
    if any_rotation_level_marginal(tdd, &info) {
        return false;
    }

    let v_idx = info.v_idx.idx();
    let w_idx = info.w_idx.idx();
    let saved_output = tdd.output;
    let saved_vtree = std::mem::replace(&mut tdd.vtree, Arc::new(vt));

    // Restructure the two affected levels; `None` = bailed past the bound, with
    // the diagram already restored — just reinstate the vtree.
    let Some((old_v, old_w)) =
        restructure_kind_bounded(tdd, &info, kind, scratch, config.max_inner_pairs)
    else {
        tdd.vtree = saved_vtree;
        return false;
    };
    minimize_after_rotation(tdd, info.w_idx);

    stats.probes += 1;
    let delta = objective.delta(
        (&old_v, &old_w),
        (&tdd.levels[v_idx], &tdd.levels[w_idx]),
    );

    if delta < 0 {
        // Full topo fixup (pointers + refresh_filtered_topo) so later sweeps and
        // queries see a consistent vtree.
        Arc::make_mut(&mut tdd.vtree)
            .fixup_topo_after_rotate(&info, kind);
        stats.accepts += 1;
        true
    } else {
        // Exact revert: restore the saved vtree Arc and the two old levels.
        tdd.vtree = saved_vtree;
        tdd.levels[v_idx] = old_v;
        tdd.levels[w_idx] = old_w;
        tdd.output = saved_output;
        false
    }
}

/// Convenience wrapper: [`rotation_search`] with [`SizeDelta`] and the default
/// [`RotationSearchConfig`], descending the compiled TDD to a size local minimum.
///
/// Accepts a correct-count but non-canonical input (e.g. a clause-by-clause
/// `apply_and_clause` accumulator), canonicalizing it once up front. Rotations are
/// pure variable reorders, so the model count is preserved:
///
/// ```
/// use std::sync::Arc;
/// use tididi::apply::apply_and_clause;
/// use tididi::restructure::search::search_to_local_min;
/// use tididi::vtree::Vtree;
/// use tididi::Tdd;
///
/// let vtree = Arc::new(Vtree::balanced(4));
/// let mut acc = Tdd::one(&vtree);
/// for clause in &[[1, -2], [2, 3], [-1, 4]] {
///     let lits: Vec<_> = clause.iter().map(|&n| n.into()).collect();
///     acc = apply_and_clause(&mut acc, &lits);
/// }
/// let before = acc.model_count();
/// let stats = search_to_local_min(&mut acc);
/// assert_eq!(acc.model_count(), before); // size search preserves the count
/// // RotationSearchStats { probes, accepts, sweeps } is Debug-printable:
/// assert!(format!("{stats:?}").contains("probes"));
/// ```
pub fn search_to_local_min(f: &mut Tdd) -> RotationSearchStats {
    rotation_search(f, &mut SizeDelta, &RotationSearchConfig::default())
}

#[cfg(test)]
#[path = "local_tests.rs"]
mod tests;
