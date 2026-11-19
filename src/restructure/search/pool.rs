//! One vtree move applied to several diagrams at once, and a descent of such
//! moves.
//!
//! Operands of a binary operation must share one vtree allocation, so a host
//! that holds many diagrams on one vtree — the live formulas of a compiler —
//! can change the vtree only for all of them together. [`Engine::rotate_pool_if`]
//! applies one move to every member and keeps it for all or for none;
//! [`Engine::pool_search`] runs a greedy descent of such moves on the members'
//! total live pairs, and can restart it from random moves ([`PoolRestarts`]).
//!
//! A move is a rotation, optionally *crossed*: the promoted child's two
//! children are swapped first. A swap alone changes no diagram's size, since a
//! level is symmetric in its two sides, but rotations alone keep the
//! left-to-right order of the leaves. Crossing is what lets the search change
//! that order: at a node with children `A` and `(B, C)` the plain left
//! rotation groups `A` with `B`, the crossed one `A` with `C`.

use std::collections::HashMap;
use std::sync::Arc;

use crate::Engine;
use crate::diagram::{Tdd, TddLevel};
use crate::limits::{Limits, OperationError, Transient};
#[cfg(debug_assertions)]
use crate::restructure::relevel::{Prepared, prepare_levels};
use crate::restructure::relevel::{
    RotationStep, rebuild_levels, score_levels, score_levels_crossed_too,
};
use crate::restructure::scratch::{ProbeTables, RestructureScratch};
use crate::restructure::transport::seat_group;
use crate::vtree::rng::Lcg;
use crate::vtree::rotate::{PendingTopo, RotationInfo, rotate_pointers};
use crate::vtree::{Home, RotationKind, VarId, Vtree, VtreeIdx, VtreeNode};

use super::RotationMove;
use super::probe::{Preimage, Preimages, pivot_pairs};

/// A rotation, with the promoted child's children swapped first if
/// `crossed`.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct PoolMove {
    /// The rotation.
    pub rotation: RotationMove,
    /// Swap the children of the node the rotation promotes before rotating.
    pub crossed: bool,
}

/// What a probed move did to each member, shown to the decision that keeps
/// or reverts it.
#[derive(Debug)]
pub struct PoolProbe<'a> {
    before: &'a [usize],
    after: &'a [usize],
}

impl PoolProbe<'_> {
    /// Live pairs of each member's two rebuilt levels before the move.
    pub fn before(&self) -> &[usize] {
        self.before
    }

    /// Live pairs of each member's two rebuilt levels after the move.
    pub fn after(&self) -> &[usize] {
        self.after
    }

    /// Live pairs over all members after the move, minus before. Negative
    /// means the move shrank the pool; no other level moved.
    pub fn live_pairs_delta(&self) -> i64 {
        self.after.iter().map(|&p| p as i64).sum::<i64>()
            - self.before.iter().map(|&p| p as i64).sum::<i64>()
    }
}

/// Bounds of a [`pool_search`](Engine::pool_search). Start from
/// [`PoolSearchConfig::default`] and change what the search needs with the
/// `with_*` setters.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct PoolSearchConfig {
    /// Sweeps over the internal nodes in each descent, the first and each
    /// restart's; a sweep that keeps nothing ends that descent earlier.
    pub max_sweeps: usize,
    /// A move whose rebuild of one member's levels would pass this many
    /// pairs is abandoned. `usize::MAX` is no bound.
    pub max_inner_pairs: usize,
    /// Also try the crossed moves.
    pub crossed: bool,
    /// The search ends, keeping what it kept, once it has charged this many
    /// units to the work clock, its opening reduction included. `u64::MAX`
    /// is no bound.
    pub max_work_units: u64,
    /// Restarts from random moves once the descent stops at a local minimum.
    pub restarts: PoolRestarts,
}

impl Default for PoolSearchConfig {
    fn default() -> Self {
        PoolSearchConfig {
            max_sweeps: 4,
            max_inner_pairs: usize::MAX,
            crossed: true,
            max_work_units: u64::MAX,
            restarts: PoolRestarts::default(),
        }
    }
}

impl PoolSearchConfig {
    /// Set [`max_sweeps`](Self::max_sweeps).
    #[must_use]
    pub fn with_max_sweeps(mut self, sweeps: usize) -> PoolSearchConfig {
        self.max_sweeps = sweeps;
        self
    }

    /// Set [`max_inner_pairs`](Self::max_inner_pairs).
    #[must_use]
    pub fn with_max_inner_pairs(mut self, pairs: usize) -> PoolSearchConfig {
        self.max_inner_pairs = pairs;
        self
    }

    /// Set [`crossed`](Self::crossed).
    #[must_use]
    pub fn with_crossed(mut self, crossed: bool) -> PoolSearchConfig {
        self.crossed = crossed;
        self
    }

    /// Set [`max_work_units`](Self::max_work_units).
    #[must_use]
    pub fn with_max_work_units(mut self, units: u64) -> PoolSearchConfig {
        self.max_work_units = units;
        self
    }

    /// Set [`restarts`](Self::restarts).
    #[must_use]
    pub fn with_restarts(mut self, restarts: PoolRestarts) -> PoolSearchConfig {
        self.restarts = restarts;
        self
    }
}

/// The restarts of a [`pool_search`](Engine::pool_search): once its descent
/// keeps nothing in a sweep, each restart makes random moves that grow the
/// pool by little, descends again, and keeps the result only if the pool
/// ends smaller than before the restart, by `min_gain_permille` at least;
/// otherwise every member and the tree are put back as they were. The
/// default runs none. The same seed, members and configuration give the
/// same restarts, moves and charges.
///
/// A restart's descent probes only the moves the random ones could have
/// changed: every move the earlier descents refused on levels left as they
/// were stays refused. So a restart costs far less than a search from
/// scratch, and ends where that search would. The search's
/// [`max_work_units`](PoolSearchConfig::max_work_units) bounds the restarts
/// too: one under way when it is reached ends there and is judged as it
/// stands. A search stopped by `max_sweeps` before a sweep kept nothing, or
/// over a member that has discarded the structure at a level, runs none.
///
/// ```
/// use std::sync::Arc;
/// use tididi::restructure::search::{PoolRestarts, PoolSearchConfig};
/// use tididi::{Engine, Vtree};
///
/// // Two formulas sharing one right-linear vtree over x1..x6.
/// let engine = Engine::new();
/// let vtree = Arc::new(Vtree::linear(6));
/// let mut f = engine.or(engine.cube(&vtree, [1, 4])?, engine.cube(&vtree, [2, 5])?)?;
/// let mut g = engine.or(engine.cube(&vtree, [3, 6])?, engine.cube(&vtree, [-1, -4])?)?;
/// let (f_models, g_models) = (engine.model_count(&f)?, engine.model_count(&g)?);
/// let config = PoolSearchConfig::default()
///     .with_max_sweeps(usize::MAX)
///     .with_restarts(PoolRestarts::default().with_count(2).with_min_gain_permille(20).with_seed(7));
/// let stats = engine.pool_search(&mut [&mut f, &mut g], &config)?;
/// // The descent ran until a sweep kept nothing, so both restarts ran.
/// assert_eq!(stats.restarts, 2);
/// assert!(stats.pairs_after <= stats.pairs_before);
/// assert!(Arc::ptr_eq(f.vtree(), g.vtree()));
/// assert_eq!((engine.model_count(&f)?, engine.model_count(&g)?), (f_models, g_models));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct PoolRestarts {
    /// Restarts to run. Zero, the default, ends the search at the descent's
    /// local minimum.
    pub count: usize,
    /// Random moves each restart keeps before it descends: a pivot, a
    /// direction and, if the search crosses, a crossing drawn at random, the
    /// move kept when it adds at most `slack_permille` thousandths of the
    /// pool's live pairs. A restart draws at most 50 moves for each one it is
    /// to keep.
    pub kick: usize,
    /// How much a random move may grow the pool, in thousandths of its live
    /// pairs.
    pub slack_permille: u32,
    /// The seed the random moves are drawn from: the same seed and pool give
    /// the same restarts.
    pub seed: u64,
    /// The least share of the pool's live pairs a restart must remove to be
    /// kept, in thousandths of the pairs it began with. Zero keeps any
    /// restart that ends smaller; 1000 or more keeps none.
    pub min_gain_permille: u32,
}

impl Default for PoolRestarts {
    fn default() -> Self {
        PoolRestarts {
            count: 0,
            kick: 20,
            slack_permille: 50,
            seed: 0,
            min_gain_permille: 0,
        }
    }
}

impl PoolRestarts {
    /// Set [`count`](Self::count).
    #[must_use]
    pub fn with_count(mut self, count: usize) -> PoolRestarts {
        self.count = count;
        self
    }

    /// Set [`kick`](Self::kick).
    #[must_use]
    pub fn with_kick(mut self, kick: usize) -> PoolRestarts {
        self.kick = kick;
        self
    }

    /// Set [`slack_permille`](Self::slack_permille).
    #[must_use]
    pub fn with_slack_permille(mut self, slack: u32) -> PoolRestarts {
        self.slack_permille = slack;
        self
    }

    /// Set [`seed`](Self::seed).
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> PoolRestarts {
        self.seed = seed;
        self
    }

    /// Set [`min_gain_permille`](Self::min_gain_permille).
    #[must_use]
    pub fn with_min_gain_permille(mut self, gain: u32) -> PoolRestarts {
        self.min_gain_permille = gain;
        self
    }
}

/// What a [`pool_search`](Engine::pool_search) did.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct PoolSearchStats {
    /// Sweeps started.
    pub sweeps: usize,
    /// Moves probed, abandoned ones included, and moves refused again from
    /// an earlier probe's refusal without being probed.
    pub probes: usize,
    /// Moves kept, not counting those of restarts put back.
    pub accepts: usize,
    /// Live pairs over all members before the search, after the opening
    /// reduction.
    pub pairs_before: usize,
    /// Live pairs over all members after the search and its closing
    /// reduction.
    pub pairs_after: usize,
    /// Work charged by the search, its two reductions included.
    pub work_units: u64,
    /// Restarts begun.
    pub restarts: usize,
    /// Restarts whose result was kept.
    pub restarts_kept: usize,
    /// Work charged by the restarts, part of `work_units`.
    pub restart_work_units: u64,
}

/// Work units a member's rebuild charges beyond the pairs of its two old and
/// two new levels: building two levels, and keeping them or putting the old
/// ones back, costs a few allocations and the member's saved state however
/// small the levels are. In units of the pairs the rebuild reads.
const REBUILD_UNITS: u64 = 64;

/// Work units a probe charges beyond what its members charge: the move made
/// on the tree and undone, and the probe's tables.
const PROBE_UNITS: u64 = 16;

impl Engine {
    /// Apply `mv` to every member, show the change to `accept`, and keep it
    /// for all members or for none.
    ///
    /// The members share one vtree allocation or are on projections of one
    /// home ([`Vtree::projection`], the home's root among them or not), and
    /// `mv`'s pivot is a node of the home (of the shared vtree when that is
    /// not a projection). The move is made on the projection of the home on the members'
    /// variables, the *union*, at the node standing for the pivot; a pivot
    /// with no node there, or one that is a leaf, does not apply. A member
    /// with leaves under all three subtrees the move turns has two levels
    /// rebuilt; any other keeps its levels, and shows `accept` none
    /// before or after.
    ///
    /// A kept move on a vtree that is not a projection leaves the members
    /// sharing a new allocation, the rotated tree. Otherwise it ends on a new
    /// home: the old one with the union's new shape, every member on the
    /// projection of the new home on its own variables, the home's root for
    /// members that were on the old root, so counts stay over the home's
    /// variables. A declined or abandoned move leaves every member and every
    /// allocation as they were. A move that does not apply, a member with
    /// summed-out levels, or a member whose rebuild would pass `bound` pairs
    /// abandons the move and returns `Ok(false)` without calling `accept`. An
    /// empty pool keeps nothing. Each member's two levels are rebuilt before
    /// `accept` is asked, and put back if it declines. The copy of the trees
    /// charges their nodes to the work clock, and each member rebuilt the
    /// pairs of its two old and two new levels and the fixed cost of a
    /// rebuild.
    ///
    /// The members should be canonical: a rebuilt level is then
    /// interchangeable with the one it replaced. A kept move can leave twins
    /// in a rebuilt level, which a later minimize contracts.
    ///
    /// # Errors
    ///
    /// [`OperationError::VtreeMismatch`] when the members neither share one
    /// vtree allocation nor one home, [`OperationError::LevelNotInVtree`] for
    /// an invalid pivot, [`OperationError::OverBudget`] for refused storage, or
    /// [`OperationError::Stopped`] for cancellation. An error leaves every
    /// member as it was; storage refused once `accept` has kept the move
    /// does too.
    pub fn rotate_pool_if(
        &self,
        members: &mut [&mut Tdd],
        mv: PoolMove,
        bound: usize,
        accept: impl FnOnce(&PoolProbe<'_>) -> bool,
    ) -> Result<bool, OperationError> {
        let _op = self.limits().enter()?;
        if members.is_empty() {
            return Ok(false);
        }
        let (mut tree, mut groups) = pool_tree(members)?;
        let pivot = mv.rotation.pivot;
        if pivot.idx() >= tree.link_home().map_or(tree.num_nodes(), |home| home.vtree().num_nodes()) {
            return Err(OperationError::LevelNotInVtree(pivot));
        }
        if members.iter().any(|m| m.has_marginal_level()) {
            return Ok(false);
        }
        let Some(pivot) = tree.node_at_home(pivot) else {
            return Ok(false);
        };
        let mv = PoolMove {
            rotation: RotationMove { pivot, ..mv.rotation },
            ..mv
        };
        let Some(pending) = move_in_place(&mut tree, mv) else {
            return Ok(false);
        };
        let info = pending.commit_in_probe(&mut tree);
        let copy = PROBE_UNITS.saturating_add(tree.num_nodes() as u64);
        let copy = copy.saturating_add(groups.nodes() as u64);
        let mut scratch = self.scratch.restructure.checkout(self);
        let mut touched = Vec::new();
        let listed = groups.touch(&info, mv, &mut touched);
        let kept = probe(self, members, Touched::new(listed, mv, bound), copy, &mut scratch, accept)?;
        if kept {
            groups.kept(&tree, &info, mv, members, None);
            end_on_home(self, members, tree, groups);
        }
        Ok(kept)
    }

    /// A greedy descent over the members' shared vtree: at every internal
    /// node, the rotations both ways, crossed and not if `config.crossed`,
    /// each kept when it lowers the members' total live pairs. Sweeps repeat
    /// until one keeps nothing, `config.max_sweeps` is reached, or the
    /// search has charged `config.max_work_units`. A descent that stops
    /// because a sweep kept nothing is followed by `config.restarts`
    /// ([`PoolRestarts`]), which keep a result only when it is smaller by
    /// their minimum gain. A restart refused a buffer is put back and ends
    /// the restarts, the search going on to its closing reduction.
    /// Every member is reduced before the search and after it. Over a member
    /// with summed-out levels the search keeps no move: each move is counted
    /// as probed and refused at no charge.
    ///
    /// The members are as [`rotate_pool_if`](Self::rotate_pool_if) takes
    /// them, and the descent runs on their union: a probe reads and rebuilds
    /// only the members with leaves under all three subtrees its move turns.
    /// The members keep their functions. A search that kept no move leaves
    /// every member's vtree allocation as it was. One that kept a move ends as
    /// a kept [`rotate_pool_if`](Self::rotate_pool_if) does, once: the
    /// members share a new allocation, or they are on projections of one new
    /// home, which [`Vtree::home`] of any member's vtree returns, the
    /// union's shape lifted into the old home. Stops are polled once per
    /// pivot and within each probe; an error keeps the moves kept so far and
    /// returns without the closing reduction. A probe scores the move
    /// member by member and stops once the scores refuse it, and charges
    /// every member the pairs of its two levels all the same; a move the
    /// scores keep then charges what [`rotate_pool_if`](Self::rotate_pool_if)
    /// does, less the copy of the tree, and the nodes whose place in the
    /// bottom-up order it changes: the descent moves one copy in place.
    ///
    /// # Errors
    ///
    /// As [`rotate_pool_if`](Self::rotate_pool_if), and the errors of the
    /// reductions.
    pub fn pool_search(
        &self,
        members: &mut [&mut Tdd],
        config: &PoolSearchConfig,
    ) -> Result<PoolSearchStats, OperationError> {
        let _op = self.limits().enter()?;
        let mut stats = PoolSearchStats::default();
        if members.is_empty() {
            return Ok(stats);
        }
        refuse_regions(members)?;
        // A reduction may narrow a member's tree, so the trees are read after.
        stats.pairs_before = self.reduce_members(members)?;
        // The union and the groups' tables, moved in place as moves are kept.
        // The members keep their allocations until the descent ends, which
        // seats them on these if it kept a move: their levels follow the
        // moves, and nothing in a probe reads a member's tree.
        let (mut tree, mut groups) = pool_tree(members)?;
        let start = self.limits().work_units();
        let mut changed = false;
        let pool = (&mut tree, &mut groups, &mut changed);
        let result = self.pool_search_from(members, config, start, &mut stats, pool);
        if changed {
            end_on_home(self, members, tree, groups);
        }
        let result = result.and_then(|()| {
            stats.pairs_after = self.reduce_members(members)?;
            Ok(())
        });
        stats.work_units = self.limits().work_units() - start;
        result.map(|()| stats)
    }

    /// The descent of [`pool_search`](Self::pool_search) on the pool's union
    /// `tree` and `groups`, its work counted from the clock's `start`;
    /// `changed` set once a move is kept for good.
    fn pool_search_from(
        &self,
        members: &mut [&mut Tdd],
        config: &PoolSearchConfig,
        start: u64,
        stats: &mut PoolSearchStats,
        (tree, groups, changed): (&mut Vtree, &mut Groups, &mut bool),
    ) -> Result<(), OperationError> {
        let mut scratch = self.scratch.restructure.checkout(self);
        PoolDescent::new(self, tree, groups, config, start, &mut scratch, stats, members)
            .and_then(|mut descent| descent.run(members, changed))
    }

    /// Reduce every member; their live pairs after.
    fn reduce_members(&self, members: &mut [&mut Tdd]) -> Result<usize, OperationError> {
        for m in members.iter_mut() {
            self.reduce(m, crate::reduce::ReductionPlan::default())?;
        }
        Ok(live_pairs(members))
    }
}

/// State shared across the descent of [`Engine::pool_search`] and its restarts.
struct PoolDescent<'a> {
    eng: &'a Engine,
    /// The union, moved in place.
    tree: &'a mut Vtree,
    groups: &'a mut Groups,
    config: &'a PoolSearchConfig,
    until: u64,
    scratch: &'a mut RestructureScratch,
    stats: &'a mut PoolSearchStats,
    known: Known,
    crossed_scores: Transient<'a, Vec<Option<Option<usize>>>>,
    /// The probe under way's [`Listed`] members.
    touched: Vec<(usize, RotationInfo)>,
    marginal: bool,
}

impl<'l> PoolDescent<'l> {
    #[expect(clippy::too_many_arguments, reason = "the descent's state, borrowed from the search")]
    fn new(
        eng: &'l Engine,
        tree: &'l mut Vtree,
        groups: &'l mut Groups,
        config: &'l PoolSearchConfig,
        start: u64,
        scratch: &'l mut RestructureScratch,
        stats: &'l mut PoolSearchStats,
        members: &[&mut Tdd],
    ) -> Result<Self, OperationError> {
        let lim = eng.limits();
        let known = Known::new(tree.num_nodes());
        // Asked once: a kept move rebuilds every member's two levels and so
        // forgets what each member knows of its levels, and asking at every
        // probe read every level of every member again, most of a search over
        // many small members. A rotation rebuilds structural levels as
        // structural levels, so the answer holds for the whole search.
        let marginal = members.iter().any(|m| m.has_marginal_level());
        // The scores of the crossed move at the pivot and rotation under way,
        // one per member, taken with the plain move's ([`Twin`]).
        let mut crossed_scores = Transient::new(lim, Vec::new());
        lim.reserve_exact(&mut crossed_scores, members.len())?;
        let until = start.saturating_add(config.max_work_units);
        Ok(PoolDescent {
            eng,
            tree,
            groups,
            config,
            until,
            scratch,
            stats,
            known,
            crossed_scores,
            touched: Vec::with_capacity(members.len()),
            marginal,
        })
    }

    /// The descent of [`Engine::pool_search`] and its restarts, between its two
    /// reductions, on `self.tree`, the members' tree with the moves kept so far.
    /// `changed` is set once a move is kept for good.
    fn run(&mut self, members: &mut [&mut Tdd], changed: &mut bool) -> Result<(), OperationError> {
        let lim = self.eng.limits();
        let accepts = self.stats.accepts;
        let converged = self.descend(members, None);
        *changed |= self.stats.accepts > accepts;
        if !converged? || self.marginal || self.config.restarts.count == 0 {
            return Ok(());
        }
        // The restarts. Rotations keep every internal node internal.
        let internals: Vec<VtreeIdx> = self.tree.internal_bottomup().map(|(v, _, _)| v).collect();
        if internals.is_empty() {
            return Ok(());
        }
        let restarts = &self.config.restarts;
        let begun = lim.work_units();
        let mut rng = Lcg::new(restarts.seed);
        let mut result = Ok(());
        while self.stats.restarts < restarts.count && lim.work_units() < self.until {
            self.stats.restarts += 1;
            let mut undo = match Undo::begin(lim, members, self.tree, &self.known, self.groups) {
                Ok(undo) => undo,
                // A restart is an improvement the search may go without.
                Err(OperationError::OverBudget) => break,
                Err(e) => {
                    result = Err(e);
                    break;
                }
            };
            let accepts = self.stats.accepts;
            let outcome = self.restart(members, &internals, &mut rng, &mut undo);
            if outcome.is_ok()
                && live_pairs(members) < kept_below(undo.pairs, restarts.min_gain_permille)
            {
                self.stats.restarts_kept += 1;
                *changed |= self.stats.accepts > accepts;
                drop(undo);
            } else {
                undo.revert(members, self.tree, &mut self.known, self.groups);
                self.stats.accepts = accepts;
            }
            match outcome {
                Ok(()) => {}
                // A restart refused a buffer, the pool back where the descent
                // left it: the search ends there, as it would have without
                // restarts.
                Err(OperationError::OverBudget) => break,
                Err(e) => {
                    result = Err(e);
                    break;
                }
            }
        }
        self.stats.restart_work_units = lim.work_units() - begun;
        result
    }

    /// One restart of [`PoolDescent::run`]: `restarts.kick` random moves kept, then a
    /// descent, every kept move's replaced levels filed with `undo`. The caller
    /// keeps the result or reverts it.
    fn restart(
        &mut self,
        members: &mut [&mut Tdd],
        internals: &[VtreeIdx],
        rng: &mut Lcg,
        undo: &mut Undo<'l>,
    ) -> Result<(), OperationError> {
        let lim = self.eng.limits();
        let restarts = &self.config.restarts;
        let (mut kept, mut draws) = (0usize, 0usize);
        while kept < restarts.kick && draws < restarts.kick.saturating_mul(KICK_DRAWS) {
            draws += 1;
            lim.check_stop()?;
            if lim.work_units() >= self.until {
                return Ok(());
            }
            let v = internals[rng.below(internals.len() as u64) as usize];
            let kind = if rng.below(2) == 0 {
                RotationKind::Left
            } else {
                RotationKind::Right
            };
            let crossed = self.config.crossed && rng.below(2) == 0;
            let slack = (live_pairs(members) as u64)
                .saturating_mul(u64::from(restarts.slack_permille))
                / 1000;
            let (left, right) = self.tree.children(v);
            let parent = self.tree.node(v).parent();
            let mv = PoolMove {
                rotation: RotationMove { pivot: v, kind },
                crossed,
            };
            self.stats.probes += 1;
            let slack = usize::try_from(slack)
                .unwrap_or(usize::MAX)
                .saturating_add(1);
            if self.probe_in_place(members, mv, TwinMode::Alone, slack, Some(undo))? {
                kept += 1;
                self.stats.accepts += 1;
                #[cfg(test)]
                tests::note_accept(mv);
                self.known.kept(self.tree, mv, left, right, parent);
            }
        }
        self.descend(members, Some(undo)).map(|_| ())
    }
}

/// The live pairs a restart that began at `pairs` must end under to be kept.
fn kept_below(pairs: usize, min_gain_permille: u32) -> usize {
    let gain = (pairs as u64).saturating_mul(u64::from(min_gain_permille)) / 1000;
    pairs.saturating_sub(usize::try_from(gain).unwrap_or(usize::MAX))
}

/// The draws a restart makes for each random move it is to keep.
const KICK_DRAWS: usize = 50;

/// The members' total live pairs.
fn live_pairs(members: &[&mut Tdd]) -> usize {
    members.iter().map(|m| m.pair_count()).sum()
}

/// What a descent knows of the moves it refused, which holds while the
/// levels those moves read are unchanged.
#[derive(Clone)]
struct Known {
    /// The pivots whose every move was probed and refused since the last
    /// kept move that could change their outcome: a kept move clears its
    /// pivot, the pivot's two children and its parent. A refused pivot is
    /// passed over.
    refused: Vec<bool>,
    /// The charge of each move's last refusal, kept while the two levels the
    /// move reads are unchanged. A move reads its pivot's level and its
    /// promoted child's; a kept move rewrites its own pivot's and promoted
    /// child's. So it changes every move at those two nodes and, at the
    /// pivot's parent, the moves that promote the pivot, and no other. A
    /// move refused before is refused again at the same charge, so it is
    /// charged and not probed: at the pivots the descent visits, it decides
    /// and charges what probing every move would. A pivot in `refused` is
    /// passed over uncharged.
    refusals: Vec<[Option<u64>; 4]>,
}

impl Known {
    fn new(nodes: usize) -> Known {
        Known {
            refused: vec![false; nodes],
            refusals: vec![[None; 4]; nodes],
        }
    }

    /// Forget what the kept move `mv` may have changed, `tree` the tree after
    /// it, `left`, `right` and `parent` the pivot's children and parent before
    /// it.
    fn kept(
        &mut self,
        tree: &Vtree,
        mv: PoolMove,
        left: VtreeIdx,
        right: VtreeIdx,
        parent: Option<VtreeIdx>,
    ) {
        let v = mv.rotation.pivot;
        let promoted = match mv.rotation.kind {
            RotationKind::Left => right,
            RotationKind::Right => left,
        };
        self.refusals[v.idx()] = [None; 4];
        self.refusals[promoted.idx()] = [None; 4];
        if let Some(parent) = parent {
            let at_parent = if tree.children(parent).1 == v {
                RotationKind::Left
            } else {
                RotationKind::Right
            };
            self.refusals[parent.idx()][move_slot(at_parent, false)] = None;
            self.refusals[parent.idx()][move_slot(at_parent, true)] = None;
        }
        for t in [Some(v), Some(left), Some(right), parent]
            .into_iter()
            .flatten()
        {
            self.refused[t.idx()] = false;
        }
    }
}

/// What a restart needs to put the pool back as it began: the union and the
/// groups, what the descent knew, each member's header, and the first level
/// each kept move replaced at each node.
///
/// The members of a group are rebuilt and mirrored together, so what is
/// known of their levels is kept by *slot*, an own node of a group.
struct Undo<'a> {
    tree: Vtree,
    known: Known,
    groups: Groups,
    /// Each member's header and the levels as the restart found them, room
    /// for one per member and node of its tree reserved up front.
    pre: Preimages<'a>,
    /// By slot: whether the levels of the group's members there are in `pre`.
    filed: Vec<bool>,
    /// By slot: whether the levels there were mirrored an odd number of
    /// times before they were filed.
    flipped: Vec<bool>,
    /// The members' live pairs when the restart began.
    pairs: usize,
}

impl<'a> Undo<'a> {
    fn begin(
        lim: &'a Limits,
        members: &mut [&mut Tdd],
        tree: &Vtree,
        known: &Known,
        groups: &Groups,
    ) -> Result<Undo<'a>, OperationError> {
        let levels = members.iter().map(|m| m.vtree.num_nodes()).fold(0usize, usize::saturating_add);
        let mut pre = Preimages::new(lim, members.len(), levels)?;
        for (i, m) in members.iter_mut().enumerate() {
            pre.file_header(i, m, true);
        }
        Ok(Undo {
            tree: tree.clone(),
            known: known.clone(),
            groups: groups.clone(),
            pre,
            filed: vec![false; groups.slots],
            flipped: vec![false; groups.slots],
            pairs: live_pairs(members),
        })
    }

    /// The slot of member `i`'s node `t`.
    fn slot(&self, i: usize, t: VtreeIdx) -> usize {
        self.groups.groups[self.groups.of[i]].base + t.idx()
    }

    /// File the levels a kept move replaced; only the first replaced at a
    /// slot is kept.
    fn file(&mut self, mut replaced: Transient<'a, Vec<Preimage<'a>>>) {
        let start = self.pre.levels.len();
        for p in replaced.drain(..) {
            if !self.filed[self.slot(p.diagram, p.node)] {
                self.pre.levels.push(p);
            }
        }
        for j in start..self.pre.levels.len() {
            let slot = self.slot(self.pre.levels[j].diagram, self.pre.levels[j].node);
            self.filed[slot] = true;
        }
    }

    /// Record that the levels at `slot` were mirrored.
    fn flip(&mut self, slot: usize) {
        if !self.filed[slot] {
            self.flipped[slot] ^= true;
        }
    }

    /// Put every member, the union, the groups and what the descent knew
    /// back as the restart found them.
    fn revert(self, members: &mut [&mut Tdd], tree: &mut Vtree, known: &mut Known, groups: &mut Groups) {
        let Undo { tree: t, known: k, groups: g, pre, flipped, .. } = self;
        pre.put_back_then(members, |members| {
            for group in &g.groups {
                let nodes = group.kids.as_ref().map_or(t.num_nodes(), Vec::len);
                for node in (0..nodes).filter(|&n| flipped[group.base + n]) {
                    for &i in &group.members {
                        mirror_level(&mut *members[i], VtreeIdx(node as u32));
                    }
                }
            }
        });
        *tree = t;
        *known = k;
        *groups = g;
    }
}

/// How [`PoolDescent::probe_in_place`] uses `crossed_scores`.
#[derive(Clone, Copy)]
enum TwinMode {
    Alone,
    Fill,
    Given,
}

impl<'l> PoolDescent<'l> {
    /// The sweeps of a descent, until one keeps nothing, `self.config.max_sweeps`
    /// sweeps, or the work clock reaches `self.until`: `Ok(true)` when a sweep kept
    /// nothing. Every kept move's replaced levels go to `undo` when given.
    fn descend(
        &mut self,
        members: &mut [&mut Tdd],
        mut undo: Option<&mut Undo<'l>>,
    ) -> Result<bool, OperationError> {
        let crossings: &[bool] = if self.config.crossed {
            &[false, true]
        } else {
            &[false]
        };
        let until = self.until;
        let spent = |eng: &Engine| eng.limits().work_units() >= until;
        let mut sweeps = 0usize;
        while sweeps < self.config.max_sweeps && !spent(self.eng) {
            sweeps += 1;
            self.stats.sweeps += 1;
            let internals: Vec<VtreeIdx> =
                self.tree.internal_bottomup().map(|(v, _, _)| v).collect();
            let mut kept = 0usize;
            for v in internals {
                self.eng.limits().check_stop()?;
                if spent(self.eng) {
                    return Ok(false);
                }
                if self.known.refused[v.idx()] {
                    continue;
                }
                match self.probe_pivot(members, v, crossings, undo.as_deref_mut())? {
                    Some(true) => kept += 1,
                    Some(false) => self.known.refused[v.idx()] = true,
                    None => return Ok(false),
                }
            }
            if kept == 0 {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Probe the candidate moves at pivot `v`: `Ok(Some(true))` when a move was
    /// accepted, `Ok(Some(false))` when all moves at `v` were refused, and
    /// `Ok(None)` when the work budget `self.until` was reached mid-pivot.
    fn probe_pivot(
        &mut self,
        members: &mut [&mut Tdd],
        v: VtreeIdx,
        crossings: &[bool],
        mut undo: Option<&mut Undo<'l>>,
    ) -> Result<Option<bool>, OperationError> {
        let (left, right) = self.tree.children(v);
        let parent = self.tree.node(v).parent();
        for kind in [RotationKind::Left, RotationKind::Right] {
            let mut twin_scored = false;
            for &crossed in crossings {
                if self.eng.limits().work_units() >= self.until {
                    return Ok(None);
                }
                let slot = move_slot(kind, crossed);
                self.stats.probes += 1;
                if let Some(units) = self.known.refusals[v.idx()][slot]
                    && replay_refusals()
                {
                    self.eng.limits().charge_work(units);
                    continue;
                }
                let mv = PoolMove {
                    rotation: RotationMove { pivot: v, kind },
                    crossed,
                };
                // A plain move whose crossing is to be probed scores the
                // crossing too; refused, it leaves the levels as they
                // were, and the crossing reads its scores.
                let twin = if crossed {
                    if twin_scored {
                        TwinMode::Given
                    } else {
                        TwinMode::Alone
                    }
                } else if self.config.crossed
                    && self.known.refusals[v.idx()][move_slot(kind, true)].is_none()
                {
                    self.crossed_scores.clear();
                    twin_scored = true;
                    TwinMode::Fill
                } else {
                    TwinMode::Alone
                };
                let before = self.eng.limits().work_units();
                if self.marginal
                    || !self.probe_in_place(members, mv, twin, 0, undo.as_deref_mut())?
                {
                    self.known.refusals[v.idx()][slot] =
                        Some(self.eng.limits().work_units() - before);
                    continue;
                }
                debug_assert_eq!(
                    members.iter().any(|m| m.has_marginal_level()),
                    self.marginal
                );
                self.stats.accepts += 1;
                #[cfg(test)]
                tests::note_accept(mv);
                self.known.kept(self.tree, mv, left, right, parent);
                return Ok(Some(true));
            }
        }
        Ok(Some(false))
    }

    /// One probe of the descent or of a restart's random moves: `mv` made on
    /// `self.tree` in place and kept when the members' live pairs end below their
    /// total plus `slack`, else undone. A kept move's replaced levels go to
    /// `undo` when given.
    fn probe_in_place(
        &mut self,
        members: &mut [&mut Tdd],
        mv: PoolMove,
        twin: TwinMode,
        slack: usize,
        mut undo: Option<&mut Undo<'l>>,
    ) -> Result<bool, OperationError> {
        let PoolDescent { eng, tree, groups, config, scratch, crossed_scores, touched, .. } = self;
        let eng: &'l Engine = eng;
        let Some(pending) = move_in_place(tree, mv) else {
            return Ok(false);
        };
        let info = pending.info();
        let listed = Touched::new(groups.touch(&info, mv, touched), mv, config.max_inner_pairs);
        let twin = twin.with(crossed_scores);
        let outcome = probe_scored(eng, members, listed, scratch, twin, slack, undo.as_deref_mut());
        let kept = matches!(outcome, Ok(true));
        end_move(eng, tree, pending, mv, kept);
        if kept {
            groups.kept(tree, &info, mv, members, undo);
        }
        outcome
    }
}

impl TwinMode {
    fn with(self, crossed_scores: &mut Vec<Option<Option<usize>>>) -> Twin<'_> {
        match self {
            TwinMode::Alone => Twin::Alone,
            TwinMode::Fill => Twin::Fill(crossed_scores),
            TwinMode::Given => Twin::Given(crossed_scores),
        }
    }
}

/// Commit `pending`, the move `mv` made on `tree` by [`move_in_place`], if
/// `kept`, else take it back.
fn end_move(eng: &Engine, tree: &mut Vtree, pending: PendingTopo, mv: PoolMove, kept: bool) {
    let info = pending.info();
    if kept {
        // The bottom-up order's repair moves the promoted node past the
        // subtree it now needs below it, and every node between.
        let misplaced = match mv.rotation.kind {
            RotationKind::Left => info.a_idx,
            RotationKind::Right => info.c_idx,
        };
        let moved = tree
            .topo_pos(misplaced)
            .saturating_sub(tree.topo_pos(info.w_idx));
        eng.limits().charge_work(u64::from(moved));
        pending.commit_in_probe(tree);
    } else {
        pending.revert(tree);
        if mv.crossed {
            tree.swap_children_in_probe(info.w_idx);
        }
    }
}

/// Whether [`PoolDescent::probe_pivot`] charges a cached refusal rather than probing again.
#[inline]
fn replay_refusals() -> bool {
    #[cfg(test)]
    if !tests::replays_refusal() {
        return false;
    }
    true
}

/// The slot of a move among its pivot's four in [`Known::refusals`].
fn move_slot(kind: RotationKind, crossed: bool) -> usize {
    2 * usize::from(kind == RotationKind::Right) + usize::from(crossed)
}

/// Make `mv` on `tree`, its children swapped first if crossed, with the
/// bottom-up order's repair owed; `None`, the tree untouched, where the move
/// does not apply. A crossed move that is reverted owes the swap back.
fn move_in_place(tree: &mut Vtree, mv: PoolMove) -> Option<PendingTopo> {
    let RotationMove { pivot, kind } = mv.rotation;
    let VtreeNode::Internal { left, right, .. } = *tree.node(pivot) else {
        return None;
    };
    let promoted = match kind {
        RotationKind::Left => right,
        RotationKind::Right => left,
    };
    if tree.node(promoted).is_leaf() {
        return None;
    }
    if mv.crossed {
        tree.swap_children_in_probe(promoted);
    }
    rotate_pointers(tree, pivot, kind)
}

/// [`OperationError::MarginalLevel`] for a member with a region.
fn refuse_regions(members: &[&mut Tdd]) -> Result<(), OperationError> {
    match members.iter().find_map(|m| m.vtree.regions().next()) {
        Some(region) => Err(OperationError::MarginalLevel(region)),
        None => Ok(()),
    }
}

/// The tree a search over `members` moves, a private copy: their shared
/// tree, or the union of their projections, with their groups.
fn pool_tree(members: &[&mut Tdd]) -> Result<(Vtree, Groups), OperationError> {
    refuse_regions(members)?;
    let first = &members[0].vtree;
    let shared = members.iter().all(|m| Arc::ptr_eq(&m.vtree, first));
    let home_of = |v: &Arc<Vtree>| v.link_home().map_or(Arc::as_ptr(v), |home| Arc::as_ptr(home.vtree()));
    if members.iter().any(|m| home_of(&m.vtree) != home_of(first)) {
        return Err(OperationError::VtreeMismatch);
    }
    let group = |members: Vec<usize>, kids| Group { members, kids, union_of: Vec::new(), base: 0 };
    let mut groups: Vec<Group> = Vec::new();
    let mut of = Vec::with_capacity(members.len());
    let mut root = None;
    if shared {
        groups.push(group((0..members.len()).collect(), None));
        of.resize(members.len(), 0);
        root = Some(0);
    } else {
        let mut index = HashMap::new();
        for (i, m) in members.iter().enumerate() {
            let union = m.vtree.link_home().is_none();
            let g = *index.entry(Arc::as_ptr(&m.vtree)).or_insert_with(|| {
                groups.push(group(Vec::new(), (!union).then(|| children_table(&m.vtree))));
                groups.len() - 1
            });
            if union {
                root = Some(i);
            }
            groups[g].members.push(i);
            of.push(g);
        }
    }
    let tree_of = |group: &Group| &members[group.members[0]].vtree;
    let union = match root {
        Some(i) => Vtree::clone(&members[i].vtree),
        None => {
            let home = first.link_home().expect("a projection has a home");
            let vars: Vec<VarId> =
                groups.iter().flat_map(|g| tree_of(g).leaf_bottomup().map(|(_, var)| var)).collect();
            match vars.is_empty() {
                true => Vtree::clone(&Vtree::empty_projection(home)),
                false => Arc::unwrap_or_clone(Vtree::projection(home, vars).expect("a home has its projections' leaves")),
            }
        }
    };
    let listed = if groups.iter().any(|g| g.kids.is_some()) { union.num_nodes() } else { 0 };
    let mut lists = Lists { head: vec![NONE; listed], at: Vec::new() };
    let mut tail = vec![NONE; listed];
    let mut slots = 0;
    for (g, group) in groups.iter_mut().enumerate() {
        group.base = slots;
        if group.kids.is_none() {
            slots += union.num_nodes();
            continue;
        }
        let tree = &members[group.members[0]].vtree;
        slots += tree.num_nodes();
        group.union_of = (0..tree.num_nodes() as u32)
            .map(|t| union.node_at_home(tree.home_node(VtreeIdx(t))).expect("the union has every own node"))
            .collect();
        for &t in tree.internal_bottomup_slice() {
            let u = group.union_of[t.idx()];
            lists.append(&mut tail[u.idx()], u, g, t);
        }
    }
    let seen = vec![0; groups.len()];
    let on_union = groups.iter().position(|g| g.kids.is_none());
    Ok((union, Groups { groups, of, on_union, lists, seen, slots }))
}

/// No entry: the end of a list in [`Lists`].
const NONE: u32 = u32::MAX;

/// By node of the union, the groups on projections with an own node there and
/// that node, as lists linked through one buffer: a kept move relinks entries
/// between its two nodes and allocates nothing.
#[derive(Clone)]
struct Lists {
    /// By node of the union: its first entry in `at`; empty when no group
    /// is on a projection.
    head: Vec<u32>,
    at: Vec<At>,
}

#[derive(Clone, Copy)]
struct At {
    group: u32,
    own: VtreeIdx,
    next: u32,
}

impl Lists {
    /// The entries at union node `u`.
    fn of(&self, u: VtreeIdx) -> impl Iterator<Item = At> + '_ {
        let first = self.head.get(u.idx()).and_then(|&e| self.at.get(e as usize));
        std::iter::successors(first.copied(), |e| {
            self.at.get(e.next as usize).copied()
        })
    }

    /// Add group `g`'s own node `own` last at `u`, whose last entry is `tail`.
    fn append(&mut self, tail: &mut u32, u: VtreeIdx, g: usize, own: VtreeIdx) {
        let e = self.at.len() as u32;
        self.at.push(At { group: g as u32, own, next: NONE });
        self.link(tail, u, e);
    }

    /// Put entry `e` last at `u`, whose last entry is `tail`.
    fn link(&mut self, tail: &mut u32, u: VtreeIdx, e: u32) {
        self.at[e as usize].next = NONE;
        match *tail {
            NONE => self.head[u.idx()] = e,
            last => self.at[last as usize].next = e,
        }
        *tail = e;
    }
}

/// The members of a pool by vtree, the members on each in a group; members
/// that share one tree are one group, whose tree is the union.
///
/// A group's own nodes are those of its members' vtree, which the search
/// leaves as it is; the group's table holds the projection of the union on
/// its leaves. Each own node stands at the node of the union over the same
/// leaves: a move at the union's `v` promoting `w` is the same move on the
/// tables of the groups with own nodes at both, and leaves every other table
/// as it is, its own node at `v` or `w` standing afterwards where the union
/// has its leaves under both children.
#[derive(Clone)]
struct Groups {
    groups: Vec<Group>,
    /// By member: its group.
    of: Vec<usize>,
    /// The group on the union's tree, which every move touches and
    /// [`Lists`] leaves out.
    on_union: Option<usize>,
    lists: Lists,
    /// By group: where a move being made has its own nodes, 1 for `v` and 2
    /// for `w`; 0 between moves.
    seen: Vec<u8>,
    /// The slots of [`Undo`], one per own node of each group.
    slots: usize,
}

#[derive(Clone)]
struct Group {
    members: Vec<usize>,
    /// By own node: its two children, left first; `None` for the group on
    /// the union's tree. A leaf's entry is never read.
    kids: Option<Vec<(VtreeIdx, VtreeIdx)>>,
    /// By node of the tree: the node of the union over the same leaves.
    union_of: Vec<VtreeIdx>,
    /// The first of the group's slots.
    base: usize,
}

impl Groups {
    /// The nodes of the groups' tables.
    fn nodes(&self) -> usize {
        self.groups.iter().filter_map(|g| g.kids.as_ref()).map(Vec::len).sum()
    }

    /// The members `mv`, made on the union as `info`, touches: those on the
    /// union, and listed in `touched` the members of each group with own
    /// nodes at both nodes of `mv`, with the rotation `mv` is on them.
    fn touch<'t>(
        &'t mut self,
        info: &'t RotationInfo,
        mv: PoolMove,
        touched: &'t mut Vec<(usize, RotationInfo)>,
    ) -> Listed<'t> {
        touched.clear();
        let Groups { groups, on_union, lists, seen, .. } = self;
        for e in lists.of(info.w_idx) {
            seen[e.group as usize] = 2;
        }
        for At { group: g, own: t, .. } in lists.of(info.v_idx) {
            if seen[g as usize] != 2 {
                continue;
            }
            let group = &groups[g as usize];
            let kids = group.kids.as_ref().expect("the group on the union is not listed");
            let moved = rotated(kids, t, mv).0;
            touched.extend(group.members.iter().map(|&i| (i, moved)));
        }
        for e in lists.of(info.w_idx) {
            seen[e.group as usize] = 0;
        }
        let on_union = on_union.map_or(&[][..], |g| &groups[g].members[..]);
        Listed { on_union, info, rest: touched }
    }

    /// After `mv`, made on `union` as `info`, was kept: the table of each
    /// group with own nodes at both of its nodes moves with the union; each
    /// group with an own node at one of them stands there where the union
    /// now has that node's leaves under both children. The node of a group
    /// at `w` alone is mirrored when the move crossed `w`'s children, with
    /// each member's level there, and the mirror recorded in `undo`.
    fn kept(
        &mut self,
        union: &Vtree,
        info: &RotationInfo,
        mv: PoolMove,
        members: &mut [&mut Tdd],
        mut undo: Option<&mut Undo<'_>>,
    ) {
        let Groups { groups, lists, seen, .. } = self;
        if lists.at.is_empty() {
            return;
        }
        let (v, w) = (info.v_idx, info.w_idx);
        for (u, bit) in [(v, 1), (w, 2)] {
            for e in lists.of(u) {
                seen[e.group as usize] |= bit;
            }
        }
        let firsts = [lists.head[v.idx()], lists.head[w.idx()]];
        lists.head[v.idx()] = NONE;
        lists.head[w.idx()] = NONE;
        let mut tails = [NONE; 2];
        for (from, mut e) in [v, w].into_iter().zip(firsts) {
            while e != NONE {
                let At { group: g, own: t, next } = lists.at[e as usize];
                let group = &mut groups[g as usize];
                let kids = group.kids.as_mut().expect("the group on the union is not listed");
                let to = 'to: {
                    if seen[g as usize] == 3 {
                        if from == v {
                            let (moved, v_kids, w_kids) = rotated(kids, t, mv);
                            kids[t.idx()] = v_kids;
                            kids[moved.w_idx.idx()] = w_kids;
                        }
                        break 'to from;
                    }
                    let (l, r) = kids[t.idx()];
                    let under_w = |u: VtreeIdx| child_toward(union, u, v) == w;
                    let to = if under_w(group.union_of[l.idx()]) && under_w(group.union_of[r.idx()]) { w } else { v };
                    group.union_of[t.idx()] = to;
                    if mv.crossed && from == w {
                        kids[t.idx()] = (r, l);
                        for &i in &group.members {
                            mirror_level(&mut *members[i], t);
                        }
                        if let Some(undo) = undo.as_deref_mut() {
                            undo.flip(group.base + t.idx());
                        }
                    }
                    debug_assert_eq!(
                        child_toward(union, group.union_of[kids[t.idx()].0.idx()], to),
                        union.children(to).0
                    );
                    to
                };
                lists.link(&mut tails[usize::from(to == w)], to, e);
                e = next;
            }
        }
        for u in [v, w] {
            for e in lists.of(u) {
                seen[e.group as usize] = 0;
            }
        }
    }
}

/// The child of `top` that `u`, a node below it, is under.
fn child_toward(tree: &Vtree, mut u: VtreeIdx, top: VtreeIdx) -> VtreeIdx {
    loop {
        let parent = tree.node(u).parent().expect("a node below `top`");
        if parent == top {
            return u;
        }
        u = parent;
    }
}

/// Exchange the two sides of `m`'s level at `t`, a node of its tree whose
/// children were exchanged.
fn mirror_level(m: &mut Tdd, t: VtreeIdx) {
    let (canonical, closed) = (m.levels.is_canonical(m.output), m.levels.is_closed());
    m.levels[t.idx()].swap_sides();
    m.levels.mark_changed(t);
    m.close_marked_levels(closed);
    if canonical {
        m.levels.certify(m.output);
    }
}

/// By node of `tree`: its two children, left first; a leaf is its own
/// children.
fn children_table(tree: &Vtree) -> Vec<(VtreeIdx, VtreeIdx)> {
    (0..tree.num_nodes() as u32)
        .map(VtreeIdx)
        .map(|t| match *tree.node(t) {
            VtreeNode::Internal { left, right, .. } => (left, right),
            VtreeNode::Leaf { .. } => (t, t),
        })
        .collect()
}

/// `mv` at own node `v` of a table of children `kids`: the rotation as
/// [`rotate_pointers`] names it after the swap a crossed move makes first,
/// and the new children of `v` and of the promoted node.
fn rotated(
    kids: &[(VtreeIdx, VtreeIdx)],
    v: VtreeIdx,
    mv: PoolMove,
) -> (RotationInfo, (VtreeIdx, VtreeIdx), (VtreeIdx, VtreeIdx)) {
    let (l, r) = kids[v.idx()];
    let children = |w: VtreeIdx| if mv.crossed { (kids[w.idx()].1, kids[w.idx()].0) } else { kids[w.idx()] };
    match mv.rotation.kind {
        RotationKind::Left => {
            let (w, x) = (r, l);
            let (z, y) = children(w);
            (RotationInfo { v_idx: v, w_idx: w, a_idx: x, b_idx: z, c_idx: y }, (w, y), (x, z))
        }
        RotationKind::Right => {
            let (w, x) = (l, r);
            let (y, z) = children(w);
            (RotationInfo { v_idx: v, w_idx: w, a_idx: y, b_idx: z, c_idx: x }, (y, w), (z, x))
        }
    }
}

/// The end of a search on `union` and `groups` that kept a move: the members
/// on one allocation of the union, or seated on projections of the home the
/// union's moves imply.
fn end_on_home(eng: &Engine, members: &mut [&mut Tdd], union: Vtree, groups: Groups) {
    let mut bottom_up = vec![Vec::new(); groups.groups.len()];
    for &u in union.internal_bottomup_slice() {
        for e in groups.lists.of(u) {
            bottom_up[e.group as usize].push(e.own);
        }
    }
    let on_union = groups.on_union.map_or(&[][..], |g| &groups.groups[g].members);
    let home = match union.lifted_home() {
        Some(home) => {
            if !on_union.is_empty() {
                seat_group(eng, members, on_union, &children_table(&union), union.internal_bottomup_slice(), &home);
            }
            home
        }
        None => {
            let alone = groups.groups.len() == usize::from(groups.on_union.is_some());
            if alone && !on_union.first().is_some_and(|&i| members[i].vtree.is_home()) {
                let root = Arc::new(union);
                for &i in on_union {
                    members[i].vtree = Arc::clone(&root);
                }
                return;
            }
            let home = Home::new(union);
            let root = Arc::clone(home.vtree());
            for &i in on_union {
                members[i].vtree = Arc::clone(&root);
            }
            if alone {
                return;
            }
            home
        }
    };
    for (group, bottom_up) in groups.groups.iter().zip(&bottom_up) {
        if let Some(kids) = &group.kids {
            seat_group(eng, members, &group.members, kids, bottom_up, &home);
        }
    }
}

/// The members a move touches, by their index among those touched: each
/// one's index among the members and the rotation made on its tree.
trait TouchedAt<'t>: Copy {
    fn len(self) -> usize;
    fn member(self, k: usize) -> usize;
    fn info(self, k: usize) -> &'t RotationInfo;
}

/// The members a move touches, each with the rotation on its own nodes: those
/// on the union, which see the move as `info`, then the listed rest.
#[derive(Clone, Copy)]
struct Listed<'t> {
    on_union: &'t [usize],
    info: &'t RotationInfo,
    rest: &'t [(usize, RotationInfo)],
}

impl<'t> TouchedAt<'t> for Listed<'t> {
    fn len(self) -> usize {
        self.on_union.len() + self.rest.len()
    }

    fn member(self, k: usize) -> usize {
        match self.on_union.get(k) {
            Some(&i) => i,
            None => self.rest[k - self.on_union.len()].0,
        }
    }

    fn info(self, k: usize) -> &'t RotationInfo {
        match k < self.on_union.len() {
            true => self.info,
            false => &self.rest[k - self.on_union.len()].1,
        }
    }
}

/// A move as each member it touches sees it.
#[derive(Clone, Copy)]
struct Touched<A> {
    at: A,
    dir: RotationKind,
    crossed: bool,
    max_pairs: usize,
}

impl<A> Touched<A> {
    fn new(at: A, mv: PoolMove, max_pairs: usize) -> Touched<A> {
        Touched {
            at,
            dir: mv.rotation.kind,
            crossed: mv.crossed,
            max_pairs,
        }
    }
}

impl<'t, A: TouchedAt<'t>> Touched<A> {
    fn len(&self) -> usize {
        self.at.len()
    }

    fn member(&self, k: usize) -> usize {
        self.at.member(k)
    }

    fn step(&self, k: usize) -> RotationStep<'t> {
        RotationStep {
            info: self.at.info(k),
            dir: self.dir,
            crossed: self.crossed,
            max_pairs: self.max_pairs,
        }
    }
}

/// Member `i`'s two levels at `step` rebuilt within `step.max_pairs`, and
/// what it needs to be put back filed in `pre`: the new levels' pairs, or
/// `None` where the rebuild would pass `step.max_pairs`.
fn rebuild_member(
    lim: &Limits,
    i: usize,
    m: &mut Tdd,
    step: RotationStep<'_>,
    pre: &mut Preimages<'_>,
    scratch: &mut RestructureScratch,
) -> Result<Option<usize>, OperationError> {
    pre.file_header(i, m, false);
    m.levels.forget();
    // No member has a marginal level here.
    let Some((outer, inner)) = rebuild_levels(lim, m, step, false, scratch)? else {
        return Ok(None);
    };
    pre.file(i, step.info.v_idx, outer);
    pre.file(i, step.info.w_idx, inner);
    Ok(Some(pivot_pairs(m, step.info)))
}

/// Rebuild every member `moved` touches, by its index in `moved`, stopping
/// on the first refusal or error, what each needs to be put back filed in
/// `pre`: the outcome for [`settle`].
fn rebuild_members<'t>(
    lim: &Limits,
    members: &mut [&mut Tdd],
    moved: Touched<impl TouchedAt<'t>>,
    scratch: &mut RestructureScratch,
    pre: &mut Preimages<'_>,
    mut on_rebuilt: impl FnMut(usize, usize) -> Result<(), OperationError>,
) -> Result<bool, OperationError> {
    for k in 0..moved.len() {
        let i = moved.member(k);
        let Some(pairs) = rebuild_member(lim, i, &mut *members[i], moved.step(k), pre, scratch)? else {
            return Ok(false);
        };
        on_rebuilt(k, pairs)?;
    }
    Ok(true)
}

/// The two levels of each member `moved` touches rebuilt, the change shown
/// to `accept`, and the move kept for all members or put back for all;
/// `extra` units charged for the move itself. The members' trees are the
/// caller's: a kept move leaves their levels on the shapes the rotations
/// were made on and their pointers on the old trees.
///
/// Each member is charged the pairs of its two old and two new levels and
/// [`REBUILD_UNITS`].
///
/// The descent's probes decide from scores instead ([`probe_scored`]).
fn probe<'t>(
    eng: &Engine,
    members: &mut [&mut Tdd],
    moved: Touched<impl TouchedAt<'t>>,
    extra: u64,
    scratch: &mut RestructureScratch,
    accept: impl FnOnce(&PoolProbe<'_>) -> bool,
) -> Result<bool, OperationError> {
    let lim = eng.limits();
    let mut gate = lim.gate();
    gate.poll(extra)?;
    let mut before = Transient::new(lim, Vec::new());
    lim.reserve_exact(&mut before, members.len())?;
    let mut after = Transient::new(lim, Vec::new());
    lim.reserve_exact(&mut after, members.len())?;
    before.resize(members.len(), 0);
    after.resize(members.len(), 0);
    for k in 0..moved.len() {
        let (i, info) = (moved.member(k), moved.step(k).info);
        let m = &members[i];
        before[i] = m.levels[info.v_idx.idx()].live_pairs() + m.levels[info.w_idx.idx()].live_pairs();
    }
    let mut pre = Preimages::new(lim, moved.len(), 2 * moved.len())?;
    let rebuilt = rebuild_members(lim, members, moved, scratch, &mut pre, |k, pairs| {
        let i = moved.member(k);
        after[i] = pairs;
        gate.poll((before[i] + pairs) as u64 + REBUILD_UNITS)
    });
    // The gate's residue charged and the stop tested before `accept`
    // decides.
    let outcome = rebuilt.and_then(|rebuilt| {
        gate.flush()?;
        lim.check_stop()?;
        Ok(rebuilt
            && accept(&PoolProbe {
                before: &before,
                after: &after,
            }))
    });
    settle(members, pre, outcome, None)
}

/// [`probe`] for a move kept exactly when it leaves the members' total below
/// what it was plus `slack`: the descent's, which must lower it, with no
/// slack, and a restart's random ones with some.
///
/// Each member is scored ([`score_levels`]), which is never above the pairs
/// its rebuild holds and almost always equal to them. A score reads the
/// member's two levels and builds nothing, so the probe charges each
/// member their pairs and [`REBUILD_UNITS`]. The members are scored in
/// turn, each under a cap: the old total less the scores before it and the
/// least the members after it can hold, a pair for each node of their outer
/// level. A score that reaches its cap refuses the move, and the members
/// after it are not read, though charged: the charge depends on the levels
/// alone. A move whose scores do not lower the total is refused, the
/// members only read. Otherwise every member is rebuilt in turn, charged
/// what preparing it costs, and the move kept if their pairs lower the total
/// after all; if not, or if a rebuild passes `step.max_pairs` or fails, every
/// member is put back. So the scores decide, and no member's rebuild waits
/// for the others'.
fn probe_scored<'l, 't>(
    eng: &'l Engine,
    members: &mut [&mut Tdd],
    moved: Touched<impl TouchedAt<'t>>,
    scratch: &mut RestructureScratch,
    twin: Twin<'_>,
    slack: usize,
    undo: Option<&mut Undo<'l>>,
) -> Result<bool, OperationError> {
    // The tables leave the scratch, whose rest the scores and the rebuilds
    // read, and go back however the probe ends: a search makes many probes
    // over few members, which fresh tables would each allocate.
    let mut tables = std::mem::take(&mut scratch.probe);
    tables.clear();
    let kept = probe_scored_with(eng, members, moved, scratch, &mut tables, twin, slack, undo);
    scratch.probe = tables;
    kept
}

/// [`probe_scored`] in `tables`, which, like the scores of `twin`, are by
/// index in `moved`.
#[expect(clippy::too_many_arguments, reason = "probe_scored's arguments and its tables")]
fn probe_scored_with<'l, 't>(
    eng: &'l Engine,
    members: &mut [&mut Tdd],
    moved: Touched<impl TouchedAt<'t>>,
    scratch: &mut RestructureScratch,
    tables: &mut ProbeTables,
    mut twin: Twin<'_>,
    slack: usize,
    undo: Option<&mut Undo<'l>>,
) -> Result<bool, OperationError> {
    debug_assert!(match twin {
        Twin::Alone => true,
        Twin::Fill(_) => !moved.crossed,
        Twin::Given(_) => moved.crossed,
    });
    let lim = eng.limits();
    let mut gate = lim.gate();
    gate.poll(PROBE_UNITS)?;
    let n = moved.len();
    let ProbeTables { before, after, floors, order } = tables;
    lim.reserve_exact(before, n)?;
    // A member ends with at least a pair for each of its outer nodes.
    lim.reserve_exact(floors, n)?;
    for k in 0..n {
        let (i, info) = (moved.member(k), moved.step(k).info);
        let m = &members[i];
        let (outer, owners) = pairs_and_owners(&m.levels[info.v_idx.idx()]);
        before.push(outer + m.levels[info.w_idx.idx()].live_pairs());
        floors.push(owners);
    }
    let total = before.iter().sum::<usize>();
    // The total the move must stay under.
    let limit = total.saturating_add(slack);
    lim.reserve_exact(after, n)?;
    after.extend_from_slice(before);
    // The largest members first: a move they refuse is refused with less
    // of the pool read.
    lim.reserve_exact(order, n)?;
    order.extend(0..n);
    order.sort_by_key(|&k| std::cmp::Reverse(before[k]));
    if let Twin::Fill(slots) = &mut twin {
        slots.resize(n, None);
    }
    // The floors of the members still to score, and the pairs of those
    // scored, of the move and of its crossing where it is scored too.
    let mut rest = floors.iter().sum::<usize>();
    let (mut sum, mut sum_crossed) = (0usize, Some(0usize));
    let cap = |sum: usize, rest: usize| limit.saturating_sub(sum + rest);
    for (at, &i) in order.iter().enumerate() {
        let (m, step) = (&members[moved.member(i)], moved.step(i));
        rest -= floors[i];
        #[cfg(test)]
        tests::note_score();
        let score = &mut scratch.score;
        let new = match &mut twin {
            Twin::Alone => score_levels(lim, m, step, score, cap(sum, rest))?,
            Twin::Fill(slots) => {
                // A crossing refused by an earlier member's score is capped
                // at nothing.
                let crossed_cap = sum_crossed.map_or(0, |s| cap(s, rest));
                let [plain, crossed] = score_levels_crossed_too(
                    lim,
                    m,
                    step.info,
                    step.dir,
                    score,
                    step.max_pairs,
                    [cap(sum, rest), crossed_cap],
                )?;
                slots[i] = Some(crossed);
                sum_crossed = sum_crossed.zip(crossed).map(|(s, n)| s + n);
                plain
            }
            Twin::Given(slots) => match slots.get(i) {
                Some(&Some(given)) => given,
                _ => score_levels(lim, m, step, score, cap(sum, rest))?,
            },
        };
        let Some(new) = new else {
            // Refused, by the cap or the bound: this member and the ones
            // after it charged as if scored.
            for &j in &order[at..] {
                gate.poll(before[j] as u64 + REBUILD_UNITS)?;
            }
            #[cfg(debug_assertions)]
            debug_check_refusal(lim, members, moved, limit, scratch)?;
            gate.flush()?;
            lim.check_stop()?;
            return Ok(false);
        };
        gate.poll(before[i] as u64 + REBUILD_UNITS)?;
        after[i] = new;
        sum += new;
    }
    // A score is at most the pairs: a total not under the limit is a move
    // the search refuses.
    if sum >= limit {
        #[cfg(debug_assertions)]
        debug_check_scores(lim, members, moved, after, scratch)?;
        gate.flush()?;
        lim.check_stop()?;
        return Ok(false);
    }
    let mut pre = Preimages::new(lim, n, 2 * n)?;
    let outcome = rebuild_members(
        lim,
        members,
        moved,
        scratch,
        &mut pre,
        |i, pairs| {
            debug_assert!(
                pairs >= after[i],
                "a score is at most the pairs: {} > {pairs}",
                after[i]
            );
            after[i] = pairs;
            gate.poll((before[i] + pairs) as u64 + REBUILD_UNITS)
        },
    );
    // The gate's residue charged and the stop tested before the move is
    // kept.
    let outcome = outcome.and_then(|kept| {
        gate.flush()?;
        lim.check_stop()?;
        Ok(kept && after.iter().sum::<usize>() < limit)
    });
    settle(members, pre, outcome, undo)
}

/// A level's live pairs and the nodes that hold them.
fn pairs_and_owners(level: &TddLevel) -> (usize, usize) {
    match level.implicit() {
        Some(d) => (
            d.pairs(),
            if d.pairs_per_node() == 0 {
                0
            } else {
                d.nodes()
            },
        ),
        None => {
            let (pairs, live, _) = level.pair_census();
            (pairs as usize, live as usize)
        }
    }
}

/// What a scored probe ([`probe_scored`]) knows of the crossing of its move:
/// the same rotation at the same pivot with the promoted node's children
/// swapped, which reads the same two levels.
enum Twin<'s> {
    /// Nothing: the move scored alone.
    Alone,
    /// The move is plain and its crossing is still to be probed: each
    /// member's score of the crossing taken with the move's, filed at the
    /// member's index among those touched, `None` for a member not reached.
    Fill(&'s mut Vec<Option<Option<usize>>>),
    /// The move is the crossing of a plain move refused just before, on the
    /// same levels, which filed these scores.
    Given(&'s [Option<Option<usize>>]),
}

/// The end of a probe's rebuilds, `pre` holding what each member rebuilt
/// needs to be put back: where `outcome` keeps the move, every member's two
/// rebuilt levels kept ([`Preimages::keep_rebuilt`]) and the levels they
/// replaced filed with `undo` when given; otherwise every member rebuilt put
/// back as it was. Returns `outcome`.
fn settle<'a>(
    members: &mut [&mut Tdd],
    pre: Preimages<'a>,
    outcome: Result<bool, OperationError>,
    undo: Option<&mut Undo<'a>>,
) -> Result<bool, OperationError> {
    if !matches!(outcome, Ok(true)) {
        pre.put_back(members);
        return outcome;
    }
    let replaced = pre.keep_rebuilt(members);
    if let Some(undo) = undo {
        undo.file(replaced);
    }
    Ok(true)
}

/// A debug build's check of a move its scores refused: every touched
/// member's preparation gives at least its score.
#[cfg(debug_assertions)]
fn debug_check_scores<'t>(
    lim: &Limits,
    members: &[&mut Tdd],
    moved: Touched<impl TouchedAt<'t>>,
    scores: &[usize],
    scratch: &mut RestructureScratch,
) -> Result<(), OperationError> {
    for (k, &score) in scores.iter().enumerate() {
        let (m, step) = (&members[moved.member(k)], moved.step(k));
        let mut prepared = Prepared::default();
        let new = prepare_levels(lim, m, step, false, scratch, &mut prepared)?;
        lim.discard(prepared);
        if let Some(new) = new {
            debug_assert!(
                new >= score,
                "a score is at most the pairs: {score} > {new}"
            );
        }
    }
    Ok(())
}

/// The check, in debug builds, that a move whose scores were refused short
/// of the last member is one whose rebuilt members would hold at least
/// `total` pairs.
#[cfg(debug_assertions)]
fn debug_check_refusal<'t>(
    lim: &Limits,
    members: &[&mut Tdd],
    moved: Touched<impl TouchedAt<'t>>,
    total: usize,
    scratch: &mut RestructureScratch,
) -> Result<(), OperationError> {
    let mut pairs = 0usize;
    for k in 0..moved.len() {
        let (m, step) = (&members[moved.member(k)], moved.step(k));
        let mut prepared = Prepared::default();
        let new = prepare_levels(lim, m, step, false, scratch, &mut prepared)?;
        lim.discard(prepared);
        let Some(new) = new else { return Ok(()) };
        pairs += new;
    }
    debug_assert!(
        pairs >= total,
        "a capped score refused a move to {pairs} pairs from {total}"
    );
    Ok(())
}

#[cfg(test)]
#[path = "tests/pool.rs"]
mod tests;
