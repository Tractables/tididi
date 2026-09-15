//! The `Tdd` struct.
//!
//! A node's pair list is unordered: the node's identity is its set of pairs,
//! and a multiset once any level is marginal (a repeated pair then carries
//! multiplicity). Twin contraction sorts each signature before comparing, so
//! an operation may push pairs in any order.

mod reach;
mod operations;
mod worklists;

#[cfg(test)]
mod tests;

use std::sync::Arc;

use crate::vtree::{Vtree, VtreeIdx};
use crate::diagram::WeightStore;

use super::build_error::TddBuildError;
use super::level::TddLevel;
use super::primitives::{LEAF_WIDTH, TddNodeId, ZERO};

/// The reduction passes' worklists on a diagram: which levels changed since the
/// last contraction, and which the content-twin scan still has to revisit. Not
/// serialized, and never part of the function the diagram denotes.
///
/// The fields are private to this module. Everything that changes a diagram
/// states what it changed through [`Tdd::invalidate`], which is the one place
/// that decides which worklist owes what; everything that consumes a worklist
/// goes through the `take_*` accessors below.
#[derive(Clone, Debug, Default)]
pub(crate) struct Dirty {
    /// Internal vtree node indices whose pair lists changed since the last
    /// `contract_all_twins` pass; it consumes the list to seed its worklist
    /// (children of dirty parents) instead of scanning every level. May hold
    /// duplicates and stale entries (filtered at consume time). A level absent
    /// from the list is at its contraction fixpoint.
    contract: Vec<u32>,
    /// The same, for the leaf-side twin contraction (`contract_leaf_twins`).
    leaf_contract: Vec<u32>,
    /// Worklist for the content-twin fixpoint: vtree indices whose
    /// boundary-parent levels may have gained new content twins since the last
    /// scan round. Only meaningful inside `canonicalize_content_twins`; empty
    /// outside it.
    right_rescan: Vec<u32>,
}

/// What a rewrite did to one level, as the reduction passes see it.
///
/// A rewrite states this and nothing else; [`Tdd::invalidate`] turns it into
/// worklist entries. The three are independent and combine with `|`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Changed(u8);

impl Changed {
    /// This level's pair lists were rewritten in place — refs, lengths, or
    /// order. Its children's contexts moved, so they are twin candidates, and
    /// its own leaf-side verdict is stale.
    pub(crate) const PAIRS: Changed = Changed(1 << 0);
    /// Nodes of this level were merged or dropped, so the parent's references
    /// into it changed identity: the parent may now hold twins.
    pub(crate) const NODES: Changed = Changed(1 << 1);
    /// Marginal values behind references from this level were merged or
    /// renumbered. Structurally the same as `PAIRS` for the worklists — the
    /// refs this level holds mean something different than they did.
    pub(crate) const VALUES: Changed = Changed(1 << 2);

    /// Does `self` include any of `other`'s kinds?
    #[inline]
    fn intersects(self, other: Changed) -> bool {
        self.0 & other.0 != 0
    }
}

impl std::ops::BitOr for Changed {
    type Output = Changed;
    #[inline]
    fn bitor(self, rhs: Changed) -> Changed {
        Changed(self.0 | rhs.0)
    }
}

/// A Tree Decision Diagram: a Boolean function decomposed along a vtree.
///
/// Build and combine diagrams without keeping an engine:
///
/// ```
/// use std::sync::Arc;
/// use tididi::{and, Tdd, Vtree};
///
/// let tree = Arc::new(Vtree::balanced(2));
/// let x = Tdd::literal(&tree, 1);
/// let y = Tdd::literal(&tree, 2);
/// let f = and(x, y.negate()?)?;
/// assert_eq!(f.model_count(), 1u32.into());
/// # let mut f = f;
/// # tididi::reduce::minimize(&mut f);
/// # tididi::test_helpers::assert_canonical(&f);
/// # Ok::<(), tididi::OperationError>(())
/// ```
///
/// The shared vtree retains reusable execution scratch. Composition functions
/// such as [`and`](crate::and) return errors and consume their operands.
/// Queries borrow diagrams; [`try_model_count`](Self::try_model_count) returns
/// errors while [`model_count`](Self::model_count) panics on failure. [`Context::with_limits`](crate::Context::with_limits) lends a
/// batch engine for explicit execution limits. Binary operations require
/// operands to share the same `Arc<Vtree>` allocation, independently of which
/// batch produced them.
///
/// The diagram owns one [`TddLevel`] per vtree node and shares the vtree by
/// `Arc`. The function it denotes is the node `output`; every other stored
/// node is a subfunction over its vtree node's variables. See the
/// [module docs](super) for how to walk it. A minimized diagram is canonical
/// for its vtree; one built level by level
/// ([`TddBuilder`](crate::diagram::TddBuilder)) is not until
/// [`minimize`](crate::reduce::minimize) runs.
///
/// Cloning copies the level storage and shares the vtree; it is not a cheap
/// node-handle copy. Use borrowed references for read-only queries, and let
/// Rust drop diagrams when they are no longer needed. Canonicality is up to node order within
/// each level, so comparing output identifiers from different diagrams does
/// not establish functional equality.
///
/// A structural diagram retains its Boolean choices and supports all Boolean
/// operations. [`marginalize_levels`](crate::marginal::marginalize_levels) can
/// replace selected subtrees with counts or fixed weighted values; a diagram
/// with those marginal levels supports only operations that can use the retained
/// information. Each operation states its requirements.
///
/// Clone an operand when two transformations need to start from it:
///
/// ```
/// use std::sync::Arc;
/// use tididi::{Tdd, Vtree};
/// use tididi::vtree::VarId;
///
/// let tree = Arc::new(Vtree::balanced(3));
/// let f = Tdd::try_clause(&tree, [1, 2])?;
/// let with_first = f.clone().condition_var(VarId(0), true)?;
/// let without_first = f.clone().condition_var(VarId(0), false)?;
/// assert_eq!(with_first.model_count(), 8u32.into());
/// assert_eq!(without_first.model_count(), 4u32.into());
/// assert_eq!(f.model_count(), 6u32.into()); // the retained original
/// // Conditioning substitutes x1 but keeps it free in the counting universe.
/// # for diagram in [&f, &with_first, &without_first] { tididi::test_helpers::assert_canonical(diagram); }
/// # Ok::<(), tididi::OperationError>(())
/// ```
#[derive(Clone, Debug)]
pub struct Tdd {
    /// The vtree the diagram is decomposed along. Operands of a binary
    /// operation must share it (`Arc::ptr_eq`). Read it with
    /// [`vtree`](Self::vtree); the only way to change it is
    /// [`reseat_vtree`](Self::reseat_vtree).
    pub(crate) vtree: Arc<Vtree>,
    /// One level per vtree node: `levels[t.idx()]` is the level of `t`
    /// ([`level`](Self::level)).
    pub(crate) levels: Vec<TddLevel>,
    /// The node denoting the function: a node of the root level, or
    /// `local == ZERO` for the constant-false function ([`is_zero`](Self::is_zero)).
    pub(crate) output: TddNodeId,
    /// Which levels the reduction passes still have to revisit (not part of
    /// the function denoted).
    pub(crate) dirty: Dirty,
    /// Per-node semiring values for the diagram's weight-marginal levels, when
    /// the caller put the diagram in weighted mode ([`set_weights`]).
    /// `None` is integer mode: marginal levels carry model counts instead.
    ///
    /// [`set_weights`]: Self::set_weights
    pub(crate) weights: Option<WeightStore>,
}

impl Tdd {
    /// Reject target indices outside this diagram before a pass reads or changes levels.
    pub(crate) fn check_level_indices(&self, targets: &[VtreeIdx]) -> Result<(), crate::OperationError> {
        for &target in targets {
            if target.idx() >= self.levels.len() {
                return Err(crate::OperationError::LevelNotInVtree(target));
            }
        }
        Ok(())
    }

    /// Require pair structure or an implicit leaf label at a valid level index.
    pub(crate) fn require_structure_at(&self, level: VtreeIdx) -> Result<(), crate::OperationError> {
        if self.levels[level.idx()].is_marginal() {
            return Err(crate::OperationError::MarginalLevel(level));
        }
        Ok(())
    }

    /// Require every level's structure before an operation complements the diagram.
    pub(crate) fn require_structure(&self) -> Result<(), crate::OperationError> {
        for (i, _) in self.levels.iter().enumerate() {
            self.require_structure_at(VtreeIdx(i as u32))?;
        }
        Ok(())
    }

    /// The vtree the diagram is decomposed along.
    ///
    /// Operands of a binary operation must share it (`Arc::ptr_eq`).
    ///
    /// Build another operand on the existing diagram's shared vtree:
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{Tdd, Vtree};
    ///
    /// let tree = Arc::new(Vtree::balanced(3));
    /// let f = Tdd::clause(&tree, [1, 2]);
    /// let g = Tdd::clause(f.vtree(), [-2, 3]);
    /// assert!(Arc::ptr_eq(f.vtree(), g.vtree()));
    /// # tididi::test_helpers::assert_canonical(&f);
    /// # tididi::test_helpers::assert_canonical(&g);
    /// ```
    #[inline]
    pub fn vtree(&self) -> &Arc<Vtree> {
        &self.vtree
    }

    /// The node denoting the function: a node of the root level, or
    /// `local == ZERO` for the constant-false function
    /// ([`is_zero`](Self::is_zero)).
    #[inline]
    pub fn output(&self) -> TddNodeId {
        self.output
    }

    /// Every level, in vtree-node order — `levels()[t.idx()]` is the level of
    /// `t`. For one level, [`level`](Self::level) says it more directly.
    #[inline]
    pub fn levels(&self) -> &[TddLevel] {
        &self.levels
    }

    /// Clone the diagram with every level arena reserved through `eng`, so a
    /// copy the host cannot serve comes back as
    /// [`OperationError::OverBudget`](crate::OperationError::OverBudget) rather than
    /// aborting the process. `Clone` is the same copy without that guard.
    ///
    /// The weight store, when there is one, is copied by its own `Clone`.
    pub(crate) fn try_clone_on(
        &self,
        eng: &crate::engine::Engine,
    ) -> Result<Tdd, crate::limits::OperationError> {
        let lim = eng.limits();
        let mut levels = Vec::new();
        lim.reserve_exact(&mut levels, self.levels.len())?;
        for level in &self.levels {
            levels.push(level.try_clone_on(lim)?);
        }
        Ok(Tdd {
            vtree: Arc::clone(&self.vtree),
            levels,
            output: self.output,
            dirty: self.dirty.clone(),
            weights: self.weights.clone(),
        })
    }

    /// Seat the diagram on `vtree`, a numbering of the same node set the
    /// diagram's levels are indexed by, so that diagrams over a rebuilt or
    /// rotated tree share one `Arc` again (operands of an operation must be
    /// `Arc::ptr_eq`).
    ///
    /// The two trees need not have the same shape ([`Vtree::same_tree`]); a
    /// rotation leaves the nodes each in-flight diagram describes untouched.
    /// Only the node count is checked. The caller owes the rest.
    pub(crate) fn reseat_vtree(&mut self, vtree: &Arc<Vtree>) {
        debug_assert_eq!(
            self.vtree.num_nodes(), vtree.num_nodes(),
            "reseat_vtree onto a tree of a different size leaves levels unaddressable",
        );
        self.vtree = Arc::clone(vtree);
    }

    /// Assemble a diagram from levels built by hand, unchecked.
    ///
    /// The caller guarantees the invariants
    /// [`check_levels`](crate::diagram::builder::check_levels) checks; nothing
    /// here verifies them, and a violation surfaces later as a wrong answer or
    /// a panic. The result need not be canonical:
    /// [`minimize`](crate::reduce::minimize) makes it so. Every
    /// internal level is marked for twin contraction, so the first minimize
    /// visits all of them.
    ///
    /// Outside the crate, [`TddBuilder`](crate::diagram::TddBuilder) is the way
    /// in: it establishes what this trusts.
    pub(crate) fn from_levels_unchecked(vtree: Arc<Vtree>, levels: Vec<TddLevel>, output: TddNodeId) -> Self {
        Self::assemble(Arc::clone(&vtree), levels, output, Dirty::default(),
            vtree.internal_bottomup().map(|(t, _, _)| t), |list, n| { list.reserve(n); Ok(()) }, || Ok(()))
            .expect("infallible worklist reservation")
    }

    /// Assemble trusted levels, charging initial reduction worklists to the engine.
    pub(crate) fn try_from_levels_on(eng: &crate::Engine, vtree: Arc<Vtree>, levels: Vec<TddLevel>, output: TddNodeId) -> Result<Self, crate::OperationError> {
        let lim = eng.limits();
        let mut gate = crate::limits::PollGate::new(lim.reduce_poll_stride());
        let result = Self::assemble(Arc::clone(&vtree), levels, output, Dirty::default(),
            vtree.internal_bottomup().map(|(t, _, _)| t),
            |list, n| lim.reserve(list, n), || lim.poll(&mut gate, 1))?;
        lim.flush_poll(&mut gate)?;
        Ok(result)
    }

    /// Construct a diagram from raw levels with contract worklists supplied by
    /// the caller, instead of
    /// [`from_levels_unchecked`](Self::from_levels_unchecked)' every-internal-level seed.
    ///
    /// A level absent from a worklist is taken to be at its contraction
    /// fixpoint ([`Dirty`]), so the caller owes two things:
    ///
    /// 1. Every level whose pair list this operation changed is in `rebuilt`;
    /// 2. `carried` is the input diagram's own [`Dirty`], so nothing the input
    ///    had outstanding is dropped.
    ///
    /// Seeding only the rewritten levels makes the following contraction cost
    /// proportional to them rather than to the vtree.
    pub(crate) fn try_with_levels_dirty(
        eng: &crate::Engine, vtree: Arc<Vtree>, levels: Vec<TddLevel>, output: TddNodeId,
        carried: Dirty, rebuilt: &[VtreeIdx],
    ) -> Result<Self, crate::OperationError> {
        let lim = eng.limits();
        let mut gate = crate::limits::PollGate::new(lim.reduce_poll_stride());
        let result = Self::assemble(vtree, levels, output, carried, rebuilt.iter().copied(),
            |list, n| lim.reserve(list, n), || lim.poll(&mut gate, 1))?;
        lim.flush_poll(&mut gate)?;
        Ok(result)
    }

    /// Seed and compact reduction worklists using the caller's allocation policy.
    fn assemble(
        vtree: Arc<Vtree>, levels: Vec<TddLevel>, output: TddNodeId, mut dirty: Dirty,
        rebuilt: impl Iterator<Item = VtreeIdx>,
        mut reserve: impl FnMut(&mut Vec<u32>, usize) -> Result<(), crate::OperationError>,
        mut poll: impl FnMut() -> Result<(), crate::OperationError>,
    ) -> Result<Self, crate::OperationError> {
        let minimum = rebuilt.size_hint().0;
        for list in [&mut dirty.contract, &mut dirty.leaf_contract] {
            if minimum > list.capacity() - list.len() { reserve(list, minimum)?; }
        }
        for t in rebuilt {
            poll()?;
            for list in [&mut dirty.contract, &mut dirty.leaf_contract] {
                if list.len() == list.capacity() { reserve(list, 1)?; }
                list.push(t.0);
            }
        }
        // Bound the carried lists: entries are level indices, so a list longer
        // than `n` holds duplicates, and a chain of applies that never drains a
        // list would otherwise grow it without bound. Dedup keeps the set the
        // list denotes, and fires at most once per `n` pushes.
        let n = vtree.num_nodes();
        for list in [&mut dirty.contract, &mut dirty.leaf_contract] {
            if list.len() > n {
                list.sort_unstable();
                list.dedup();
            }
        }
        Ok(Self { vtree, levels, output, dirty, weights: None })
    }

    /// Put the diagram in weighted mode: its weight-marginal levels keep their
    /// per-node semiring values in `ws` instead of model counts.
    ///
    /// Attach the store before the first operation that marginalizes a level:
    /// a level summed out without one holds counts, and nothing converts them.
    /// A structural diagram may replace its table; after marginalization, stored
    /// values and their weight interpretation must remain consistent.
    /// Conjunction (`&`, [`Engine::and`](crate::Engine::and),
    /// [`Engine::and_clause`](crate::Engine::and_clause)), projection,
    /// conditioning, negation, disjunction and care restriction preserve weights.
    /// [`Tdd::graft_over`] accepts a destination store for renamed parts.
    /// A weight-marginal level exists only in a diagram carrying a store.
    ///
    /// # Errors
    ///
    /// Refuses count-marginal levels, missing or inconsistent columns, and a
    /// different weight configuration after marginalization; leaves the diagram unchanged.
    ///
    /// Keep a weighted value while releasing structure:
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{Engine, Vtree};
    /// use tididi::diagram::{Arithmetic, RationalWeights, WeightStore};
    /// use tididi::marginal::marginalize_levels;
    ///
    /// let engine = Engine::new();
    /// let tree = Arc::new(Vtree::balanced(2));
    /// let mut f = engine.clause(&tree, [1, 2])?;
    /// f.set_weights(WeightStore::new(RationalWeights::unit(2), Arithmetic::ExactRational))?;
    /// let before = engine.weighted_value(&f)?.unwrap().into_rational();
    /// marginalize_levels(&engine, &mut f, &[tree.root()])?;
    /// assert!(f.has_marginal_level());
    /// assert_eq!(engine.weighted_value(&f)?.unwrap().into_rational(), before);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// [`TddBuildError`] for missing variable weights, incompatible stored
    /// marginal values, or a changed weight interpretation after marginalization.
    /// Validation completes before the store is replaced; an error leaves the
    /// diagram and its previous store unchanged.
    pub fn set_weights(&mut self, ws: WeightStore) -> Result<(), TddBuildError> {
        ws.check_levels(&self.vtree, &self.levels)?;
        if self.levels.iter().any(TddLevel::is_weight_marginal)
            && self.weights.as_ref().is_some_and(|old| !old.compatible(&ws))
        {
            return Err(TddBuildError::IncompatibleWeights);
        }
        self.weights = Some(ws);
        Ok(())
    }

    /// The attached weight store, or `None` in integer mode.
    pub fn weights(&self) -> Option<&WeightStore> {
        self.weights.as_ref()
    }

    /// Detach the weight store, leaving the diagram in integer mode.
    ///
    /// # Errors
    ///
    /// [`TddBuildError::WeightedLevelWithoutStore`] naming the first
    /// weight-marginal level, whose per-node values live in the store. The
    /// diagram is untouched and still carries its store.
    pub fn take_weights(&mut self) -> Result<Option<WeightStore>, TddBuildError> {
        if let Some(level) = self.levels.iter().position(TddLevel::is_weight_marginal) {
            return Err(TddBuildError::WeightedLevelWithoutStore {
                level: VtreeIdx(level as u32),
            });
        }
        Ok(self.detach_weights())
    }

    /// [`take_weights`](Self::take_weights) without the level scan, for a
    /// restructuring that moves the store onto the diagram replacing this one.
    pub(crate) fn detach_weights(&mut self) -> Option<WeightStore> {
        self.weights.take()
    }

    /// The store of a weighted diagram.
    ///
    /// Panics if the diagram has none; a caller has already established that it
    /// is weighted, and a weight-marginal level exists only with a store.
    #[inline]
    pub(crate) fn weight_store(&self) -> &WeightStore {
        self.weights
            .as_ref()
            .expect("a weighted operation on a diagram with no weight store")
    }

    /// [`Tdd::weight_store`] for a caller that writes.
    #[inline]
    pub(crate) fn weight_store_mut(&mut self) -> &mut WeightStore {
        self.weights
            .as_mut()
            .expect("a weighted operation on a diagram with no weight store")
    }

    /// Whether the output is the [`ZERO`] sentinel for the constant-false function.
    ///
    /// For a structural diagram this decides unsatisfiability without minimization.
    /// A zero marginal value need not have been collapsed to the sentinel; use
    /// [`Engine::is_sat`](crate::Engine::is_sat) to require structural input.
    pub fn is_zero(&self) -> bool {
        self.output.local == ZERO
    }

    /// True if any level of the diagram is marginal, in which case a pair list
    /// anywhere in the diagram is a multiset. O(levels).
    pub fn has_marginal_level(&self) -> bool {
        self.levels.iter().any(|l| l.is_marginal())
    }

    /// The level of vtree node `idx`.
    ///
    /// # Panics
    ///
    /// Panics if `idx` is not a node of the diagram's vtree.
    pub fn level(&self, idx: VtreeIdx) -> &TddLevel {
        &self.levels[idx.idx()]
    }

    /// [`TddLevel::slot_count`] at `idx`: 0 on a non-marginal leaf level. Use
    /// [`reference_slot_count`](Self::reference_slot_count) to size arrays indexed by
    /// child references.
    pub fn slot_count_at(&self, idx: VtreeIdx) -> usize {
        self.levels[idx.idx()].slot_count()
    }

    /// The number of reference slots at `idx`, excluding encoded inline values:
    /// [`LEAF_WIDTH`] on a leaf level, else [`TddLevel::slot_count`].
    pub fn reference_slot_count(&self, idx: VtreeIdx) -> usize {
        if self.vtree.node(idx).is_leaf() {
            LEAF_WIDTH
        } else {
            self.levels[idx.idx()].slot_count()
        }
    }

    /// The largest [`TddLevel::live_slot_count`] over all levels; 0 for ⊥.
    ///
    /// Marginal value slots contribute to width; implicit ordinary leaf nodes do not.
    pub fn max_width(&self) -> usize {
        self.levels.iter().map(TddLevel::live_slot_count).max().unwrap_or(0)
    }

    /// Number of live structural nodes and marginal value slots over all levels.
    ///
    /// Implicit ordinary leaf nodes are excluded.
    pub fn node_count(&self) -> usize {
        self.levels.iter().map(|l| l.live_slot_count()).sum()
    }

    /// Running total of marginal-count slots the slot prune has collected,
    /// summed over all levels. Non-decreasing over a diagram's life; it resets
    /// only where a level is cleared or reset.
    ///
    /// It is the correction term for a size comparison across a prune. Record
    /// it alongside [`Tdd::node_count`] at the baseline instant; at comparison
    /// time add `retired_marginal_slots() - baseline` back to the node count,
    /// or pruning silently deflates the metric.
    pub fn retired_marginal_slots(&self) -> usize {
        self.levels.iter().map(|l| l.retired_marginal_slots() as usize).sum()
    }

    /// Total number of pairs over all stored nodes — the size of the diagram.
    /// A marginal level holds no pairs and contributes nothing.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::Tdd;
    /// use tididi::vtree::Vtree;
    ///
    /// let vtree = Arc::new(Vtree::balanced(4));
    /// let mut f = Tdd::clause(&vtree, [1, -2]) & Tdd::clause(&vtree, [2, 3]);
    /// tididi::reduce::minimize(&mut f);
    /// assert!(f.pair_count() > 0);
    /// assert!(f.pair_count_at_most(f.pair_count()));
    /// assert!(!f.pair_count_at_most(f.pair_count() - 1));
    ///
    /// // Conditioning cannot grow the diagram.
    /// let g = tididi::apply::condition_var(&f, tididi::vtree::VarId(0), true);
    /// assert!(g.pair_count() <= f.pair_count());
    /// ```
    pub fn pair_count(&self) -> usize {
        self.levels.iter().map(TddLevel::live_pairs).sum()
    }

    /// Whether the diagram has at most `cap` input pairs.
    ///
    /// The cost is bounded by `cap` rather than by the diagram: the scan stops
    /// at the first node that carries the total past `cap`.
    pub fn pair_count_at_most(&self, cap: usize) -> bool {
        let mut total = 0usize;
        for n in self.levels.iter().flat_map(TddLevel::pair_counts) {
            total += n;
            if total > cap {
                return false;
            }
        }
        true
    }
}
