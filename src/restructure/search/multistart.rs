//! Search the same function from several starting shapes and keep the best.
//!
//! A descent stops at the first shape it cannot improve, and which shape that
//! is depends on where it started. Restarting from a randomly perturbed copy
//! and keeping whichever result is smaller costs one search per restart and
//! two diagrams of memory.

use crate::diagram::Tdd;
use crate::limits::OperationError;
use crate::vtree::rng::Lcg;
use crate::vtree::{RotationKind, VtreeIdx};


use super::{RotationMove, RotationObjective, RotationSearchConfig, RotationSearchStats};

/// How [`Engine::rotation_multistart`](crate::Engine::rotation_multistart)
/// spends its restarts.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct MultistartConfig {
    /// Searches to run from a perturbed copy, after the one from the diagram
    /// as it arrives. Zero makes the call a plain
    /// [`rotation_search`](crate::Engine::rotation_search) and copies nothing.
    pub restarts: usize,
    /// Rotations applied to a copy, at random pivots, before searching it.
    pub kick: usize,
    /// The seed the kicks are drawn from. The same seed gives the same
    /// restarts.
    pub seed: u64,
    /// The configuration each search runs under. Its `max_inner_pairs` bound
    /// also applies to the random rotations before each search, and it must
    /// be set when `restarts` and `kick` are both nonzero.
    pub search: RotationSearchConfig,
}

impl Default for MultistartConfig {
    /// Four restarts of eight rotations each, seed 0, and the default search
    /// configuration. That search configuration has no `max_inner_pairs`
    /// bound, and the restarts need one, so set it before searching.
    fn default() -> MultistartConfig {
        MultistartConfig {
            restarts: 4,
            kick: 8,
            seed: 0,
            search: RotationSearchConfig::default(),
        }
    }
}

/// What [`Engine::rotation_multistart`](crate::Engine::rotation_multistart) did.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct MultistartStats {
    /// Searches run, counting the one from the diagram as it arrived.
    pub rounds: usize,
    /// Which round the returned diagram came from. Round 0 is the diagram as
    /// it arrived.
    pub best_round: usize,
    /// The work every round performed, added up.
    pub search: RotationSearchStats,
}

impl crate::Engine {
    /// Search from the diagram as it is, then from `config.restarts` perturbed
    /// copies, and leave `tdd` holding whichever result has the fewest nodes.
    ///
    /// Each restart copies the diagram, applies `config.kick` rotations at
    /// random pivots without scoring them, and runs
    /// [`rotation_search`](Self::rotation_search) — the greedy policy — on the
    /// copy. Two diagrams are live at once for the duration, so peak memory is
    /// about twice the largest one.
    ///
    /// The result is the same function on a vtree over the same variables. It
    /// is not necessarily the diagram that arrived: with `restarts` at zero
    /// this is exactly one `rotation_search`, and the objective decides
    /// whether that shrinks the diagram.
    ///
    /// # Errors
    ///
    /// [`OperationError::UnboundedSearch`] when `config.restarts` and
    /// `config.kick` are both nonzero and `config.search.max_inner_pairs`
    /// has no bound, before anything is searched: a kick keeps its rotation
    /// whatever the rebuild costs, and only the bound keeps it from asking
    /// for more memory than the host has. Otherwise whatever
    /// [`rotation_search`](Self::rotation_search) returns, from the round
    /// that hit it. The diagram is then the best result of the rounds that
    /// finished, which is still the same function.
    pub fn rotation_multistart<O: RotationObjective>(
        &self,
        tdd: &mut Tdd,
        objective: &mut O,
        config: &MultistartConfig,
    ) -> Result<MultistartStats, OperationError> {
        let _op = self.limits().enter()?;
        if config.restarts > 0 && config.kick > 0 && config.search.max_inner_pairs == usize::MAX {
            return Err(OperationError::UnboundedSearch {
                option: "MultistartConfig::search.max_inner_pairs",
                needed_by: "restarts with kicks",
            });
        }
        let mut stats = MultistartStats {
            rounds: 1,
            best_round: 0,
            search: RotationSearchStats::default(),
        };
        let first = self.rotation_search(tdd, objective, &config.search)?;
        stats.search += &first;
        let mut rng = Lcg::new(config.seed);
        for round in 1..=config.restarts {
            let mut candidate = tdd.clone();
            kick(self, &mut candidate, config.kick, config.search.max_inner_pairs, &mut rng)?;
            let run = self.rotation_search(&mut candidate, objective, &config.search)?;
            stats.search += &run;
            stats.rounds += 1;
            if candidate.node_count() < tdd.node_count() {
                *tdd = candidate;
                stats.best_round = round;
            }
        }
        Ok(stats)
    }
}

/// Apply up to `count` rotations at random pivots, keeping each one whatever
/// it does to the diagram. A draw that names a pivot no rotation applies at is
/// spent, not redrawn, which keeps the number of draws a function of the seed
/// alone.
fn kick(
    eng: &crate::Engine,
    tdd: &mut Tdd,
    count: usize,
    bound: usize,
    rng: &mut Lcg,
) -> Result<(), OperationError> {
    let _op = eng.limits().begin_operation();
    let mut scratch = eng.scratch.restructure.checkout(eng.limits());
    let mut search = super::SearchTree::new(tdd);
    for _ in 0..count {
        let nodes = search.tdd.vtree().num_nodes() as u64;
        let pivot = VtreeIdx(rng.below(nodes) as u32);
        let kind = if rng.below(2) == 0 { RotationKind::Left } else { RotationKind::Right };
        eng.limits().check_stop()?;
        let turn = RotationMove { pivot, kind };
        search.probe_moves(eng, &[turn], &mut super::probe::Forced, &mut scratch, bound)?;
    }
    Ok(())
}

/// Add one round's work to the running total.
#[cfg(test)]
#[path = "tests/multistart.rs"]
mod tests;
