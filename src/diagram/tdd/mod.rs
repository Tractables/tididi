// scenario: docs/scenarios.md#ownership

//! The `Tdd` struct.
//!
//! A node's pair list is unordered: the node's identity is its set of pairs,
//! and a multiset once any level is marginal (a repeated pair then carries
//! multiplicity). Twin contraction sorts each signature before comparing, so
//! an operation may push pairs in any order.

mod edit;
mod reach;
mod operations;
mod renumber;
mod state;

pub(crate) use state::{Dirty, Known, LevelCounts, Pass};

#[cfg(test)]
mod tests;

use std::sync::Arc;

use crate::vtree::{SameLayout, Vtree, VtreeIdx};
use crate::diagram::Weights;

use super::level::TddLevel;
use super::primitives::{LEAF_WIDTH, ONE_LEAF_IDX, TddNodeId, TRUE, ZERO};

/// A Boolean function represented by a Tree Decision Diagram.
///
/// Build smaller functions and combine them with [`and`](crate::and) and
/// [`or`](crate::or):
///
/// ```
/// use std::sync::Arc;
/// use tididi::{and, literal, Vtree};
///
/// let vtree = Arc::new(Vtree::balanced(2));
/// let x = literal(&vtree, 1)?;
/// let y = literal(&vtree, 2)?;
/// let f = and(x, y.negate()?)?;
/// assert_eq!(f.model_count()?, 1u32.into());
/// # let mut f = f;
/// # f.minimize()?;
/// # tididi::test_helpers::assert_canonical(&f);
/// # Ok::<(), tididi::OperationError>(())
/// ```
///
/// Each diagram owns its nodes and shares a vtree through `Arc`. Operands must
/// share the same `Arc<Vtree>` allocation, even if different batches built them,
/// or be on projections of one home ([`Vtree::projection`](crate::Vtree::projection)),
/// which an operation places on the projection holding all their variables.
/// Operations reuse scratch from the vtree's context automatically; use
/// [`Context::with_limits`](crate::Context::with_limits) for a bounded batch.
///
/// # Keep a circuit for later queries
///
/// Queries borrow diagrams. Transformations taking `Tdd` consume it; clone a
/// diagram when several transformations need the original. Cloning copies its
/// storage, so prefer borrowing for queries, and pass a conjunction
/// ([`and`](crate::and)) a borrowed operand, which it reads in place.
///
/// ```
/// use std::sync::Arc;
/// use tididi::{Tdd, Vtree};
/// use tididi::vtree::VarId;
///
/// let vtree = Arc::new(Vtree::balanced(3));
/// let f = Tdd::clause(&vtree, [1, 2])?;
/// let with_first = f.clone().condition_var(VarId(1), true)?;
/// let without_first = f.clone().condition_var(VarId(1), false)?;
/// assert_eq!(with_first.model_count()?, 8u32.into());
/// assert_eq!(without_first.model_count()?, 4u32.into());
/// assert_eq!(f.model_count()?, 6u32.into()); // the retained original
/// // Conditioning substitutes x1 but keeps it free in the counting universe.
/// # for diagram in [&f, &with_first, &without_first] { tididi::test_helpers::assert_canonical(diagram); }
/// # Ok::<(), tididi::OperationError>(())
/// ```
///
/// # Storage and minimization
///
/// A diagram has one [`TddLevel`] per vtree node. Its output denotes the whole
/// function; other nodes represent subfunctions. The [module reference](super)
/// explains traversal and child references.
///
/// [`minimize`](Self::minimize) makes a structural diagram canonical for its
/// vtree, up to node numbering and pair order. Compare functions with
/// [`equivalent`](Self::equivalent), rather than comparing output identifiers.
/// Diagrams assembled with [`TddBuilder`](crate::diagram::TddBuilder) need an
/// explicit minimization step to become canonical.
///
/// [`marginalize_levels`](Self::marginalize_levels) replaces selected subtrees
/// with counts or fixed weighted values. This discards Boolean choices; each
/// operation states whether it accepts such marginal levels.
#[derive(Clone, Debug)]
pub struct Tdd {
    /// The vtree the diagram is decomposed along. Operands of a binary
    /// operation must share it (`Arc::ptr_eq`). Read it with
    /// [`vtree`](Self::vtree). It changes only through
    /// [`reseat_vtree_unchecked`](Self::reseat_vtree_unchecked) and the vtree
    /// search, [`rotation_search`](Self::rotation_search), which edits it in place.
    pub(crate) vtree: Arc<Vtree>,
    /// One level per vtree node: `levels[t.idx()]` is the level of `t`
    /// ([`level`](Self::level)).
    pub(crate) levels: state::LevelStorage,
    /// The node denoting the function: a node of the root level, or
    /// `local == ZERO` for the constant-false function ([`is_zero`](Self::is_zero)).
    pub(crate) output: TddNodeId,
    /// Which levels the reduction passes still have to revisit (not part of
    /// the function denoted).
    pub(crate) dirty: Dirty,
}

/// An owned operand of a conjunction that takes each operand owned or
/// borrowed ([`and`](crate::and)), which consumes it.
impl From<Tdd> for std::borrow::Cow<'_, Tdd> {
    fn from(tdd: Tdd) -> Self {
        std::borrow::Cow::Owned(tdd)
    }
}

/// A borrowed operand of a conjunction that takes each operand owned or
/// borrowed ([`and`](crate::and)), which reads it in place.
impl<'a> From<&'a Tdd> for std::borrow::Cow<'a, Tdd> {
    fn from(tdd: &'a Tdd) -> Self {
        std::borrow::Cow::Borrowed(tdd)
    }
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
        match self.levels.first_marginal() {
            Some(level) => Err(crate::OperationError::MarginalLevel(level)),
            None => Ok(()),
        }
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
    /// let vtree = Arc::new(Vtree::balanced(3));
    /// let f = Tdd::clause(&vtree, [1, 2])?;
    /// let g = Tdd::clause(f.vtree(), [-2, 3])?;
    /// assert!(Arc::ptr_eq(f.vtree(), g.vtree()));
    /// # tididi::test_helpers::assert_canonical(&f);
    /// # tididi::test_helpers::assert_canonical(&g);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    pub fn vtree(&self) -> &Arc<Vtree> {
        &self.vtree
    }

    /// The home this diagram is on: its projection's home, or its vtree when
    /// that is a home's root; `None` on a vtree that is neither.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::Tdd;
    /// use tididi::vtree::{Home, VarId, Vtree};
    ///
    /// let home = Home::new(Vtree::balanced(4));
    /// let f = Tdd::clause(&Vtree::projection(&home, [VarId(1), VarId(3)])?, [1, -3])?;
    /// assert!(Arc::ptr_eq(f.home().unwrap().vtree(), home.vtree()));
    /// let all = Tdd::clause(home.vtree(), [2, 4])?;
    /// assert!(Arc::ptr_eq(all.home().unwrap().vtree(), home.vtree()));
    /// assert!(Tdd::clause(&Arc::new(Vtree::balanced(4)), [1])?.home().is_none());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn home(&self) -> Option<crate::vtree::Home> {
        self.vtree.home()
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

    /// Clone the diagram with every level arena and weight column reserved
    /// through `eng`, so a copy the host cannot serve comes back as
    /// [`OperationError::OverBudget`](crate::OperationError::OverBudget) rather than
    /// aborting the process. `Clone` is the same copy without that guard.
    pub(crate) fn try_clone_on(
        &self,
        eng: &crate::Engine,
    ) -> Result<Tdd, crate::limits::OperationError> {
        let lim = eng.limits();
        let mut levels = Vec::new();
        lim.reserve_exact(&mut levels, self.levels.len())?;
        for level in &self.levels {
            // Each level is written in its place (`TddLevel::try_clone_to`).
            level.try_clone_to(lim, &mut levels.spare_capacity_mut()[0])?;
            // Safety: the slot past the length was just written.
            unsafe { levels.set_len(levels.len() + 1) };
        }
        let mut levels = state::LevelStorage::from(levels);
        if self.levels.is_canonical(self.output) { levels.certify(self.output); }
        levels.copy_closed_from(&self.levels);
        if let Some(counts) = self.levels.counts() { levels.keep_counts(counts.clone()); }
        Ok(Tdd {
            vtree: Arc::clone(&self.vtree),
            levels,
            output: self.output,
            dirty: self.dirty.clone(),
        })
    }
}

impl crate::Engine {
    /// A copy of `f`, as `Clone` makes it, with every level's storage and
    /// weight column reserved through this engine's limits: a copy that the
    /// byte budget does not cover, or that the allocator cannot serve, comes back as
    /// [`OperationError::OverBudget`](crate::OperationError::OverBudget)
    /// instead of aborting the process. Operations consume their operands,
    /// so a caller that keeps a diagram and hands copies of it to them
    /// takes the copies here where memory may run short. A copy is not
    /// tested against the stop rules and charges no work.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{Engine, OperationError, Vtree};
    /// use tididi::limits::LimitConfig;
    ///
    /// let engine = Engine::new();
    /// let vtree = Arc::new(Vtree::balanced(3));
    /// let f = engine.clause(&vtree, [1, 2])?;
    /// let g = engine.try_clone(&f)?;
    /// assert!(engine.equivalent(&f, &g)?);
    /// let _refused = engine.limits().scope(LimitConfig::none().with_memory_budget_bytes(Some(0)));
    /// assert_eq!(engine.try_clone(&f).err(), Some(OperationError::OverBudget));
    /// # Ok::<(), OperationError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// [`OperationError::OverBudget`](crate::OperationError::OverBudget)
    /// where the copy is refused.
    pub fn try_clone(&self, f: &Tdd) -> Result<Tdd, crate::limits::OperationError> {
        let _op = self.limits().begin_operation();
        f.try_clone_on(self)
    }
}

#[cfg(debug_assertions)]
#[cold]
fn debug_check_reseat_on_home(from: &Vtree, onto: &Vtree) {
    assert!(onto.on_home(), "reseat_vtree_unchecked moves a diagram on a home off every home");
    if let Err(e) = onto.validate_home() {
        panic!("reseat_vtree_unchecked onto a vtree whose home link is invalid: {e}");
    }
    for i in 0..from.num_nodes() {
        let t = VtreeIdx(i as u32);
        let (a, b) = (from.node(t), onto.node(t));
        let same = match (a, b) {
            (crate::vtree::VtreeNode::Leaf { var: x, .. }, crate::vtree::VtreeNode::Leaf { var: y, .. }) => x == y,
            (crate::vtree::VtreeNode::Internal { .. }, crate::vtree::VtreeNode::Internal { .. }) => true,
            _ => false,
        };
        assert!(same, "reseat_vtree_unchecked: node {i} is {a:?} before and {b:?} after");
    }
}

impl Tdd {
    /// Replace the vtree allocation without rebuilding any diagram level.
    ///
    /// This can reunite diagrams whose independent subtrees share a vtree that
    /// another operation rotated. It does not rewrite stored nodes or remap child references.
    ///
    /// # Safety
    ///
    /// The replacement must have the same node count and root index. Every
    /// stored level must remain valid at its current index: its references must
    /// name live nodes or marginal values in the replacement's child levels,
    /// and weighted columns must retain their interpretation. Leaf variables
    /// must retain their meaning wherever the diagram depends on them. The
    /// resulting diagram must satisfy all [`TddBuilder`](crate::diagram::TddBuilder)
    /// invariants, including determinism. A topology change is permitted only
    /// where the diagram's stored structure remains valid under it.
    /// Invalid references can cause out-of-bounds reads in later operations.
    ///
    /// For a diagram on a home ([`Vtree::home`]), the replacement must be on
    /// a home too, with a link to it that is valid for the replacement's own
    /// shape, and must carry the same variable at every leaf index. No level
    /// is written, so the result then names a leaf true, or sits on leaves it
    /// does not use, exactly when the diagram did.
    ///
    /// # Panics
    ///
    /// Debug builds check that the node count is unchanged and, for a diagram
    /// on a home, the two conditions above.
    pub unsafe fn reseat_vtree_unchecked(&mut self, vtree: &Arc<Vtree>) {
        debug_assert_eq!(
            self.vtree.num_nodes(), vtree.num_nodes(),
            "reseat_vtree_unchecked onto a tree of a different size leaves levels unaddressable",
        );
        #[cfg(debug_assertions)]
        if self.vtree.on_home() && self.vtree.num_nodes() > 0 {
            debug_check_reseat_on_home(&self.vtree, vtree);
        }
        self.levels.forget();
        self.vtree = Arc::clone(vtree);
    }

    /// This diagram on the vtree `same` pairs with its own.
    ///
    /// # Panics
    ///
    /// Panics if `same` does not name this diagram's vtree.
    pub(crate) fn reseat(&mut self, same: &SameLayout) {
        let onto = Arc::clone(same.other(&self.vtree));
        // Safety: the two vtrees of `same` have the same nodes at the same
        // indices with the same variables.
        unsafe { self.reseat_vtree_unchecked(&onto) };
    }

    /// Seat hand-built test fixture levels, unchecked.
    ///
    /// Nothing verifies the levels, and `output` may name a node below the
    /// root, which [`TddBuilder::finish_unchecked`](super::TddBuilder::finish_unchecked)
    /// would refuse in a debug build. Every internal level is marked for twin
    /// contraction, so the first minimize visits all of them. Operations
    /// assemble their results through `Assembly`, which returns the levels
    /// to the engine's pool when seating is refused.
    #[cfg(any(test, debug_assertions, feature = "testing"))]
    pub(crate) fn from_levels_unchecked(vtree: Arc<Vtree>, levels: Vec<TddLevel>, output: TddNodeId) -> Self {
        let builder = super::TddBuilder::from_levels(vtree, levels);
        let dirty = builder.seed_worklists(Dirty::default(), None, None).expect("untracked worklists cannot be refused");
        builder.seat(output, dirty, None)
    }

    /// The weights the diagram is computed under: those of its vtree
    /// ([`Home::weighted`](crate::vtree::Home::weighted),
    /// [`Vtree::with_weights`]); `None` in integer mode.
    pub fn weights(&self) -> Option<&Weights> {
        self.weight_table()
    }

    /// The one read of the weights a diagram is computed under. A sum-out
    /// stores weight values exactly when this is `Some`.
    #[inline]
    pub(crate) fn weight_table(&self) -> Option<&Weights> {
        self.vtree.weights()
    }

    /// Whether the diagram holds weight values: a summed-out level on a
    /// weighted vtree.
    #[inline]
    pub(crate) fn holds_weights(&self) -> bool {
        self.vtree.weights().is_some() && self.has_marginal_level()
    }

    /// The weights of a weighted diagram.
    ///
    /// Panics if the diagram has none; a caller has already established that it
    /// is weighted, and a weight-marginal level exists only under weights.
    #[inline]
    pub(crate) fn weight_store(&self) -> &Weights {
        self.weight_table()
            .expect("a weighted operation on a diagram with no weights")
    }

    /// Whether the output is the [`ZERO`] sentinel for the constant-false function.
    ///
    /// This decides unsatisfiability without minimization, reading summed-out
    /// variables as [`Tdd::is_sat`] does. On a diagram with summed-out
    /// regions the answer is exact for model counts and for positive weights;
    /// with zero or negative weights a sum can cancel to zero without the
    /// output being [`ZERO`].
    pub fn is_zero(&self) -> bool {
        self.output.local == ZERO
    }

    /// Whether the output is the [`TRUE`] sentinel: the
    /// constant-true function on a vtree with no node
    /// ([`Vtree::empty_projection`]). A diagram with levels may denote true
    /// too; [`trim`](Self::trim) turns one into this constant.
    pub fn is_one(&self) -> bool {
        self.output.local == TRUE
    }

    /// Whether the diagram is certified canonical: its output reaches every
    /// node it stores and no two nodes of a level denote the same function,
    /// so [`Engine::minimize`](crate::Engine::minimize) would leave it as it
    /// is. A diagram on a vtree with no node is certified. An operation's
    /// documentation says whether its result keeps the certificate of its
    /// operands; a result without it may still be canonical, unknown to the
    /// library. The certificate does not say that no pair names a leaf as
    /// true: a diagram an embedding padded keeps its padding.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{and, Engine, Tdd, Vtree};
    /// let vtree = Arc::new(Vtree::balanced(2));
    /// let f = Tdd::clause(&vtree, [-1, 2])?;
    /// let mut y = and(f, Tdd::clause(&vtree, [1, 2])?)?;
    /// assert!(!y.is_canonical());
    /// Engine::new().minimize(&mut y)?;
    /// assert!(y.is_canonical());
    /// # Ok::<(), tididi::OperationError>(())
    /// ```
    pub fn is_canonical(&self) -> bool {
        self.vtree.num_nodes() == 0 || self.levels.is_canonical(self.output)
    }

    /// Certify the diagram canonical, for a producer that built it so: every
    /// node reached and no twin.
    pub(crate) fn certify_built(&mut self) {
        self.levels.certify(self.output);
        self.dirty.set_loose(Some(Vec::new()));
        #[cfg(debug_assertions)]
        {
            crate::reduce::prune::debug_assert_all_reached(self);
            if !self.has_marginal_level() {
                crate::reduce::contract::contract_leaf::assert_no_full_twin_side(self);
            }
        }
    }

    /// True on a vtree that is one leaf, the output naming that leaf `One`.
    pub(crate) fn is_one_leaf_one(&self) -> bool {
        self.output.local == ONE_LEAF_IDX && self.vtree.num_nodes() == 1
    }

    /// True if any level of the diagram is marginal, in which case a pair list
    /// anywhere in the diagram is a multiset. The first read after an edit is
    /// O(levels); later reads are constant-time.
    pub fn has_marginal_level(&self) -> bool {
        self.levels.first_marginal().is_some()
    }

    /// The level of vtree node `idx`.
    ///
    /// # Panics
    ///
    /// Panics if `idx` is not a node of the diagram's vtree.
    pub fn level(&self, idx: VtreeIdx) -> &TddLevel {
        &self.levels[idx.idx()]
    }

    /// The number of reference slots at `idx`, excluding encoded inline values:
    /// [`LEAF_WIDTH`] on a variable's leaf, else [`TddLevel::slot_count`].
    pub fn reference_slot_count(&self, idx: VtreeIdx) -> usize {
        if self.vtree.node(idx).is_var_leaf() {
            LEAF_WIDTH
        } else {
            self.levels[idx.idx()].slot_count()
        }
    }

    /// Whether `idx` is an internal level that still holds structure: the
    /// only kind of level whose nodes a walk descends into.
    pub(crate) fn is_structural_internal(&self, idx: VtreeIdx) -> bool {
        !self.vtree.node(idx).is_leaf() && !self.levels[idx.idx()].is_marginal()
    }

    /// The largest [`TddLevel::slot_count`] over all levels; 0 for ⊥.
    ///
    /// Marginal value slots contribute to width; implicit ordinary leaf nodes do not.
    pub fn max_width(&self) -> usize {
        self.levels.max_width()
    }

    /// Number of structural nodes and marginal value slots over all levels.
    ///
    /// Implicit ordinary leaf nodes are excluded.
    pub fn node_count(&self) -> usize {
        self.levels.node_count()
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

    /// Total number of logical pairs across structural nodes, including pairs
    /// generated by implicit levels. Marginal levels contribute nothing.
    /// This counts the represented pairs, not the memory used to store them.
    ///
    /// Repeated calls cache the total until the levels change. The first call
    /// sums the levels; later calls are constant-time.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::Tdd;
    /// use tididi::vtree::Vtree;
    ///
    /// let vtree = Arc::new(Vtree::balanced(4));
    /// let mut f = Tdd::clause(&vtree, [1, -2])? & Tdd::clause(&vtree, [2, 3])?;
    /// f.minimize()?;
    /// assert!(f.pair_count() > 0);
    /// assert!(f.pair_count_at_most(f.pair_count()));
    /// assert!(!f.pair_count_at_most(f.pair_count() - 1));
    ///
    /// // Conditioning cannot grow the diagram.
    /// let g = (f).clone().condition_var(tididi::vtree::VarId(1), true)?;
    /// assert!(g.pair_count() <= f.pair_count());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn pair_count(&self) -> usize {
        self.levels.pair_count()
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
