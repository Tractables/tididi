//! What a rotation search does with a scored move: keep it, or keep looking.
//!
//! A search separates the cost it is minimizing — the
//! [`RotationObjective`](super::RotationObjective) — from the rule that turns a
//! score into a decision. [`Greedy`] keeps strict improvements and stops at the
//! first local minimum, which is what [`Engine::rotation_search`] does.
//! [`Tabu`] and [`Annealing`] keep a worsening move under conditions that let
//! the search leave a minimum a descent cannot.
//!
//! The regime these pay off in is compressing one diagram that will be reused
//! many times, where the search runs once and the result is kept. For a
//! diagram built, queried and dropped, the extra sweeps usually cost more than
//! the storage they save.

use std::collections::HashMap;

use rustc_hash::FxBuildHasher;

use crate::vtree::rng::Lcg;

use super::{RotationMove, RotationProbe, RotationSearchStats};

/// Whether a probed sequence of rotations is kept, given its cost.
///
/// A policy is consulted once per scored sequence and may hold state across a
/// search: [`observe`](Self::observe) reports every decision and
/// [`keep_sweeping`](Self::keep_sweeping) ends the search.
pub trait AcceptancePolicy {
    /// Keep this sequence? `delta` is the objective's score for `probe`, and
    /// a negative one improves it.
    fn accept(&mut self, probe: &RotationProbe<'_>, delta: i64) -> bool;

    /// Record the outcome, kept or not, for a policy with memory.
    fn observe(&mut self, _probe: &RotationProbe<'_>, _delta: i64, _kept: bool) {}

    /// Continue after a finished sweep that kept `accepted` sequences?
    ///
    /// The default is a descent: keep sweeping while the last sweep kept
    /// something, and stop at the local minimum where it does not.
    fn keep_sweeping(&mut self, _stats: &RotationSearchStats, accepted: usize) -> bool {
        accepted > 0
    }

    /// Does this policy ever keep a sequence that worsens the objective?
    ///
    /// A search reads it twice: an unbounded rebuild is refused for a policy
    /// that answers `true` (see
    /// [`RotationSearchConfig::max_inner_pairs`](super::RotationSearchConfig::max_inner_pairs)),
    /// and such a search logs the moves it keeps so it can return the best
    /// diagram it passed through rather than the last one.
    fn may_worsen(&self) -> bool {
        false
    }
}

/// Keep strict improvements, stop at the first local minimum.
///
/// The policy [`Engine::rotation_search`](crate::Engine::rotation_search) uses.
#[derive(Debug, Default, Clone, Copy)]
pub struct Greedy;

impl AcceptancePolicy for Greedy {
    #[inline]
    fn accept(&mut self, _probe: &RotationProbe<'_>, delta: i64) -> bool {
        delta < 0
    }
}

/// Take one uphill step out of a local minimum, then refuse to walk back into
/// it for `tenure` moves.
///
/// Improving sequences are kept as [`Greedy`] keeps them. When a whole sweep
/// keeps nothing, the next sweep keeps its first worsening sequence, and the
/// inverse of every move kept stays forbidden for the next `tenure` accepted
/// moves — long enough that the search has to leave the minimum rather than
/// step back into it. A sequence that would beat the best cost seen is kept
/// whether or not it is forbidden.
///
/// The search stops after `patience` sweeps that do not improve on the best
/// cost, and rewinds to the diagram that had it.
#[derive(Debug)]
pub struct Tabu {
    /// How many later accepted moves a kept move's inverse stays forbidden
    /// for. The default is 8.
    pub tenure: usize,
    /// How many sweeps without a new best cost the search runs before it
    /// stops. The default is 2.
    pub patience: usize,
    /// Accepted moves so far: the clock the tenures are measured on.
    step: usize,
    /// Forbidden move to the step it is forbidden until.
    forbidden: HashMap<RotationMove, usize, FxBuildHasher>,
    /// Objective cost relative to the diagram the search started from.
    cost: i64,
    /// The lowest `cost` reached.
    best: i64,
    /// Whether `best` moved during the sweep now finishing.
    improved: bool,
    /// Sweeps since `best` last moved.
    since_best: usize,
    /// Whether the last sweep kept nothing, which is what licenses one
    /// worsening move.
    stuck: bool,
}

impl Tabu {
    /// A tabu policy with the given tenure and patience.
    pub fn new(tenure: usize, patience: usize) -> Tabu {
        Tabu {
            tenure,
            patience,
            step: 0,
            forbidden: HashMap::default(),
            cost: 0,
            best: 0,
            improved: false,
            since_best: 0,
            stuck: false,
        }
    }
}

impl Default for Tabu {
    /// Tenure 8, patience 2.
    fn default() -> Tabu {
        Tabu::new(8, 2)
    }
}

impl AcceptancePolicy for Tabu {
    fn accept(&mut self, probe: &RotationProbe<'_>, delta: i64) -> bool {
        // Aspiration: a sequence that reaches a new best is kept whether or not
        // one of its moves is forbidden, since the reason to forbid a move is
        // that it leads back somewhere already seen.
        if self.cost + delta < self.best {
            return true;
        }
        let forbidden = probe
            .moves()
            .iter()
            .any(|mv| self.forbidden.get(mv).is_some_and(|&until| until > self.step));
        if forbidden {
            return false;
        }
        if delta < 0 {
            return true;
        }
        // One uphill step per stalled sweep. Taking every worsening sequence on
        // offer would walk away from the minimum rather than out of it.
        std::mem::take(&mut self.stuck)
    }

    fn observe(&mut self, probe: &RotationProbe<'_>, delta: i64, kept: bool) {
        if !kept {
            return;
        }
        self.step += 1;
        self.cost += delta;
        let until = self.step + self.tenure;
        for mv in probe.moves() {
            self.forbidden.insert(mv.inverse(), until);
        }
        if self.cost < self.best {
            self.best = self.cost;
            self.improved = true;
        }
    }

    fn keep_sweeping(&mut self, _stats: &RotationSearchStats, accepted: usize) -> bool {
        if std::mem::take(&mut self.improved) {
            self.since_best = 0;
        } else {
            self.since_best += 1;
        }
        self.stuck = accepted == 0;
        self.since_best <= self.patience
    }

    fn may_worsen(&self) -> bool {
        true
    }
}

/// Keep a worsening sequence with a probability that falls as the search runs.
///
/// An improving sequence is always kept. A sequence that costs `delta` more is
/// kept with probability `exp(-delta / t)`, where `t` starts at `start` and is
/// multiplied by `cooling` after every sweep. The draws come from the crate's
/// own generator, so a given `seed` gives the same search every time.
///
/// The search stops once `t` is cold enough that no uphill step is realistic
/// and a sweep improves nothing, and rewinds to the best diagram it passed
/// through. A sequence that costs nothing is always kept, so a sweep that
/// keeps something is not by itself a reason to go on: a diagram with free
/// variables has such sequences at every temperature.
#[derive(Debug)]
pub struct Annealing {
    /// The temperature the search starts at. The default is 4.
    pub start: f64,
    /// What the temperature is multiplied by after each sweep. The default is
    /// 0.5.
    pub cooling: f64,
    temperature: f64,
    /// Whether the sweep now running kept a strictly improving sequence.
    improved: bool,
    rng: Lcg,
}

/// Below this temperature a unit-cost uphill step is taken about once in a
/// hundred, which is close enough to a descent to leave the rest to one.
const COLD: f64 = 0.22;

impl Annealing {
    /// An annealing policy with the given seed, start temperature and cooling
    /// factor.
    pub fn new(seed: u64, start: f64, cooling: f64) -> Annealing {
        Annealing { start, cooling, temperature: start, improved: false, rng: Lcg::new(seed) }
    }
}

impl Default for Annealing {
    /// Seed 0, start temperature 4, cooling 0.5.
    fn default() -> Annealing {
        Annealing::new(0, 4.0, 0.5)
    }
}

impl AcceptancePolicy for Annealing {
    fn accept(&mut self, _probe: &RotationProbe<'_>, delta: i64) -> bool {
        if delta < 0 {
            return true;
        }
        if self.temperature <= 0.0 {
            return false;
        }
        let probability = (-(delta as f64) / self.temperature).exp();
        // `next_u64` is 31 bits, so this is a draw in `0.0 .. 1.0`.
        let draw = self.rng.next_u64() as f64 / (1u64 << 31) as f64;
        draw < probability
    }

    fn observe(&mut self, _probe: &RotationProbe<'_>, delta: i64, kept: bool) {
        if kept && delta < 0 {
            self.improved = true;
        }
    }

    fn keep_sweeping(&mut self, _stats: &RotationSearchStats, _accepted: usize) -> bool {
        self.temperature *= self.cooling;
        let improved = std::mem::take(&mut self.improved);
        self.temperature >= COLD || improved
    }

    fn may_worsen(&self) -> bool {
        true
    }
}
