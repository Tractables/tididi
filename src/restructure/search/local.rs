//! Vtree-rotation search under a caller-supplied objective and acceptance policy.
//!
//! Each bottom-up sweep probes rotations at every internal node and hands each
//! probe's score to the policy. Only the levels a probe rebuilds are scored; a
//! rejected probe restores them. Sweeps stop when the policy says so or at the
//! configured limit. Marginal-free inputs are minimized before the first sweep.

use smallvec::SmallVec;

use crate::limits::OperationError;
use crate::Engine;
use crate::restructure::scratch::RestructureScratch;
use crate::vtree::{RotationKind, Vtree, VtreeIdx};
use crate::vtree::rotate::RotationInfo;
use crate::diagram::Tdd;

use super::policy::AcceptancePolicy;
use super::probe::*;

/// The longest sequence a neighborhood probes, which is what the trial's
/// inline storage is sized for.
const MAX_SEQUENCE: usize = 3;

/// Scores a probed rotation sequence for [`Tdd::rotation_search`].
///
/// A rotation changes exactly the two levels at its pivot, so the probe holds
/// precisely the levels the sequence rebuilt, before and after, and nothing
/// else needs reading.
pub trait RotationObjective {
    /// Score a probed sequence. A **negative** result means it improves the
    /// objective, which is what [`Greedy`](super::Greedy) keeps.
    fn delta(&mut self, probe: &RotationProbe<'_>) -> i64;
}

/// Accept rotations that reduce the total number of live pairs.
///
/// Use with [`Tdd::rotation_search`]; implement [`RotationObjective`] for
/// another cost.
#[derive(Debug, Default, Clone, Copy)]
pub struct MinimizePairs;

impl RotationObjective for MinimizePairs {
    #[inline]
    fn delta(&mut self, probe: &RotationProbe<'_>) -> i64 {
        probe.live_pairs_delta()
    }
}

/// How many rotations a sweep tries at each pivot.
///
/// A wider neighborhood crosses a hill a narrower one stops at: a single
/// rotation that grows the diagram can be the first half of a pair that
/// shrinks it. The wider settings are tried at a pivot only where the
/// narrower ones kept nothing, so a diagram with improvements left costs no
/// more to search than it does under [`Single`](Self::Single).
#[derive(Debug, Default, Clone, Copy, Eq, PartialEq)]
#[non_exhaustive]
pub enum Neighborhood {
    /// One rotation at the pivot.
    #[default]
    Single,
    /// Also two rotations, the second at the pivot, its parent or one of its
    /// children.
    Pair,
    /// Also three rotations, each connected to the one before it.
    Triple,
}

/// Limits on the work performed by [`Tdd::rotation_search`].
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct RotationSearchConfig {
    /// Bail bound forwarded to the bounded restructure probes: a rotation whose
    /// rebuilt level would exceed this many input pairs is abandoned (and the
    /// diagram restored) rather than materialized. `usize::MAX` never bails,
    /// and the probe may then ask for more memory than the host has; a policy
    /// that keeps worsening moves reaches that case often enough that
    /// [`Engine::rotation_search_with`] refuses the combination.
    pub max_inner_pairs: usize,
    /// Cap on the number of full sweeps. `None` lets the acceptance policy
    /// decide when to stop.
    pub max_sweeps: Option<usize>,
    /// How many rotations a sweep tries at each pivot.
    pub neighborhood: Neighborhood,
}

impl Default for RotationSearchConfig {
    fn default() -> Self {
        RotationSearchConfig {
            max_inner_pairs: usize::MAX,
            max_sweeps: None,
            neighborhood: Neighborhood::Single,
        }
    }
}

/// Work performed by [`Tdd::rotation_search`].
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct RotationSearchStats {
    /// Sequences the objective scored: those that applied, rebuilt their
    /// levels within the bound, and reached the acceptance policy.
    pub probes: usize,
    /// Sequences the policy kept.
    pub accepts: usize,
    /// Number of full bottom-up sweeps performed.
    pub sweeps: usize,
}

/// The search behind [`Engine::rotation_search_with`]: sweep every internal
/// vtree node, probe the configured neighborhood at each, and keep a sequence
/// whenever `policy` accepts its score. Sweeps repeat until the policy stops
/// or `config.max_sweeps` is hit. The engine's stop is polled once per pivot.
///
/// A policy that keeps worsening moves does not end on its best diagram, so
/// the search logs what it kept and rewinds to the best prefix with inverse
/// rotations before returning.
///
/// # Errors
///
/// [`OperationError::UnboundedSearch`] when `policy` may keep a worsening move
/// and `config.max_inner_pairs` has no bound, before anything is probed.
/// [`OperationError::OverBudget`] from the opening reduction or a rebuild that
/// does not fit. [`OperationError::Stopped`] when the armed stop fires between
/// pivots. The diagram is left at whatever point the search had reached —
/// canonical, count-correct, and safe to keep or to search again.
pub(crate) fn rotation_search_on<O: RotationObjective, A: AcceptancePolicy>(
    eng: &Engine,
    tdd: &mut Tdd,
    objective: &mut O,
    policy: &mut A,
    config: &RotationSearchConfig,
) -> Result<RotationSearchStats, OperationError> {
    if policy.may_worsen() && config.max_inner_pairs == usize::MAX {
        return Err(OperationError::UnboundedSearch {
            option: "RotationSearchConfig::max_inner_pairs",
            needed_by: "a policy that keeps worsening moves",
        });
    }
    let _op = eng.limits().begin_operation();
    let mut rule = Policed { objective, policy, stats: RotationSearchStats::default(), log: Vec::new() };
    let mut scratch = eng.restructure().checkout(eng.limits());

    // Rotation-locality precondition: the locality assertion and the level
    // revert need a canonical input, so establish it once here.
    // Marginal-free diagrams only: in marginal context the restructure keeps
    // the child multiset without Boolean dedup, which is what preserves the
    // count, and a minimize here would collapse it.
    if !tdd.levels.iter().any(|l| l.is_marginal()) {
        eng.reduce(tdd, crate::reduce::ReductionPlan::default())?;
    }

    let mut search = super::SearchTree::new(tdd);
    loop {
        if let Some(cap) = config.max_sweeps
            && rule.stats.sweeps >= cap {
                break;
            }
        rule.stats.sweeps += 1;

        // Unevaluated snapshot of the internal nodes each sweep: an accept relabels
        // parent/child relations, so re-collecting keeps the walk honest (a stale
        // index can only mis-skip a probe, never break count-soundness — the next
        // sweep re-picks it up).
        let internals: Vec<VtreeIdx> =
            search.tdd.vtree.internal_bottomup().map(|(v, _, _)| v).collect();

        let mut accepted_this_sweep = 0usize;
        for v in internals {
            // Once per pivot, not once per probe: the probes below share the
            // pivot's setup, and a stop between them would leave the sweep's
            // accept count describing half a pivot.
            eng.limits().check_stop()?;
            let kept = sweep_pivot(eng, &mut search, v, &mut rule, &mut scratch, config)?;
            accepted_this_sweep += kept;
        }
        if !rule.policy.keep_sweeping(&rule.stats, accepted_this_sweep) {
            break;
        }
    }
    let Policed { stats, log, .. } = rule;
    rewind_to_best(eng, search.tdd, &log, &mut scratch)?;
    Ok(stats)
}

/// Probe the configured neighborhood at one pivot and return how many
/// sequences were kept.
///
/// The wider neighborhoods are tried only where the narrower ones kept
/// nothing: a pivot that still has a single improving rotation does not need
/// the pairs, and the pairs are where the cost is.
fn sweep_pivot<O: RotationObjective, A: AcceptancePolicy>(
    eng: &Engine,
    search: &mut super::SearchTree<'_>,
    v: VtreeIdx,
    rule: &mut Policed<'_, O, A>,
    scratch: &mut RestructureScratch,
    config: &RotationSearchConfig,
) -> Result<usize, OperationError> {
    let mut kept = 0usize;
    for &kind in &KINDS {
        if try_sequence(eng, search, &[RotationMove { pivot: v, kind }], rule, scratch, config)? {
            kept += 1;
        }
    }
    if kept > 0 || config.neighborhood == Neighborhood::Single {
        return Ok(kept);
    }
    // A move followed by its inverse puts the tree back where it was: two
    // rebuilds for a sequence that scores zero, so it is not tried. The tree
    // is restored between probes, which is why the connected pivots can be
    // read once per pivot rather than once per direction.
    let second = connected(&search.tdd.vtree, v);
    for &k1 in &KINDS {
        let first = RotationMove { pivot: v, kind: k1 };
        for &p2 in &second {
            let third = (config.neighborhood == Neighborhood::Triple)
                .then(|| connected(&search.tdd.vtree, p2));
            for &k2 in &KINDS {
                let next = RotationMove { pivot: p2, kind: k2 };
                if next == first.inverse() {
                    continue;
                }
                let pair = [first, next];
                if try_sequence(eng, search, &pair, rule, scratch, config)? {
                    return Ok(1);
                }
                let Some(third) = &third else { continue };
                for &p3 in third {
                    for &k3 in &KINDS {
                        let last = RotationMove { pivot: p3, kind: k3 };
                        if last == next.inverse() {
                            continue;
                        }
                        let triple = [first, next, last];
                        if try_sequence(eng, search, &triple, rule, scratch, config)? {
                            return Ok(1);
                        }
                    }
                }
            }
        }
    }
    Ok(0)
}

/// Probe one sequence under the search's rule and bound.
fn try_sequence<O: RotationObjective, A: AcceptancePolicy>(
    eng: &Engine,
    search: &mut super::SearchTree<'_>,
    moves: &[RotationMove],
    rule: &mut Policed<'_, O, A>,
    scratch: &mut RestructureScratch,
    config: &RotationSearchConfig,
) -> Result<bool, OperationError> {
    search.probe_moves(eng, moves, rule, scratch, config.max_inner_pairs)
}

/// The pivots a sequence may continue at after turning the internal node
/// `v`: `v` itself, its parent, and whichever of its children are internal.
/// A leaf is never a pivot and no rotation makes one internal, so a leaf
/// child is left out here rather than found out after the first move's
/// rebuild. A pivot that is no longer internal when the sequence reaches it
/// simply abandons that sequence.
fn connected(vtree: &Vtree, v: VtreeIdx) -> SmallVec<[VtreeIdx; 4]> {
    let mut out: SmallVec<[VtreeIdx; 4]> = SmallVec::new();
    out.push(v);
    if let Some(parent) = vtree.node(v).parent() {
        out.push(parent);
    }
    let (left, right) = vtree.children(v);
    for child in [left, right] {
        if !vtree.node(child).is_leaf() {
            out.push(child);
        }
    }
    out
}

/// Undo the tail of `log` that the search should not have kept, so the
/// diagram is the best one the search passed through.
///
/// Each entry's inverse moves are applied in reverse order, which returns the
/// vtree to the shape it had at the chosen prefix. The diagram is rebuilt
/// rather than restored, so it is the same function on the same vtree and not
/// necessarily node-for-node the earlier one.
fn rewind_to_best(
    eng: &Engine,
    tdd: &mut Tdd,
    log: &[(SmallVec<[RotationMove; MAX_SEQUENCE]>, i64)],
    scratch: &mut RestructureScratch,
) -> Result<(), OperationError> {
    let mut cost = 0i64;
    let mut best = 0i64;
    let mut prefix = 0usize;
    for (i, (_, delta)) in log.iter().enumerate() {
        cost += *delta;
        if cost < best {
            best = cost;
            prefix = i + 1;
        }
    }
    for (moves, _) in log[prefix..].iter().rev() {
        let inverse: SmallVec<[RotationMove; MAX_SEQUENCE]> =
            moves.iter().rev().map(|mv| mv.inverse()).collect();
        probe_moves(eng, tdd, &inverse, &mut Forced, scratch, usize::MAX)?;
    }
    Ok(())
}

/// The two directions a sweep tries at every pivot.
const KINDS: [RotationKind; 2] = [RotationKind::Left, RotationKind::Right];

/// The search's own [`ProbeRule`]: the caller's objective and policy, and the
/// tallies [`RotationSearchStats`] reports.
struct Policed<'a, O, A> {
    objective: &'a mut O,
    policy: &'a mut A,
    /// The probes and accepts so far, and the sweeps the search loop counts.
    stats: RotationSearchStats,
    /// Every kept sequence and its score, for the rewind. Empty unless the
    /// policy may keep a worsening sequence.
    log: Vec<(SmallVec<[RotationMove; MAX_SEQUENCE]>, i64)>,
}

impl<O: RotationObjective, A: AcceptancePolicy> ProbeRule for Policed<'_, O, A> {
    #[inline]
    fn keeps(&mut self, probe: &RotationProbe<'_>, _info: &RotationInfo) -> bool {
        self.stats.probes += 1;
        let delta = self.objective.delta(probe);
        let kept = self.policy.accept(probe, delta);
        self.policy.observe(probe, delta, kept);
        if kept {
            // A kept sequence commits without fail, so this is the accept count.
            self.stats.accepts += 1;
            if self.policy.may_worsen() {
                self.log.push((SmallVec::from_slice(probe.moves()), delta));
            }
        }
        kept
    }
}

#[cfg(test)]
#[path = "tests/local/mod.rs"]
mod tests;
