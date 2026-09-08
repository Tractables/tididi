//! The session object every operation runs on: the limits it is held to and the
//! scratch it reuses.
//!
//! An [`Engine`] is what a caller keeps between operations. Holding the scratch
//! makes the reuse explicit — two engines never share a buffer, and dropping one
//! frees everything it warmed up — and holding the limits makes the arming
//! explicit: what a conjunction is allowed to spend is a field a caller sets,
//! not ambient state it inherits.
//!
//! The free functions and operator sugar elsewhere in this crate are thin
//! wrappers that build a transient engine, run the operation on it, and panic on
//! failure. There is one implementation underneath.

mod limits;
mod ops;
mod memory;
mod meters;
mod poll;
mod stop;

pub use limits::{LimitSet, Limits};
pub use memory::MemPressure;
pub use meters::{ApplyMeters, MergePosition};
pub use stop::{Scheduled, Stop, StopAt};

pub(crate) use limits::{PollGate, PAIR_ELEM_BYTES};

#[cfg(test)]
pub(crate) use limits::DENSE_GROWTH_DECISION_THRESHOLD;

#[cfg(test)]
pub(crate) use memory::{vas_headroom_with_margin, SOFT_HEADROOM_MARGIN_BYTES};

#[cfg(test)]
#[path = "headroom_tests.rs"]
mod headroom_tests;

#[cfg(test)]
#[path = "limits_tests.rs"]
mod limits_tests;

/// The limits and scratch one caller's operations run on.
///
/// Build one per compile and thread it through: every conjunction, reduction,
/// marginalization and restructuring takes `&mut Engine`, reuses the buffers it
/// holds, and is cut by the limits armed on it.
#[derive(Default)]
pub struct Engine {
    limits: Limits,
    apply: crate::apply::conjoin::ApplyScratch,
    build: crate::build::BuildScratch,
    restrict: crate::apply::conjoin::RestrictScratch,
    clause: crate::apply::conjoin_clause::ClauseScratch,
    reduce: crate::reduce::scratch::ReduceScratch,
    restructure: crate::restructure::scratch::RestructurePool,
    sparse: std::cell::RefCell<crate::apply::conjoin::SparseWorkspace>,
    levels: crate::diagram::LevelPool,
    /// See [`Engine::set_leaf_marginalize_inlines`].
    leaf_marginalize_inlines: std::cell::Cell<bool>,
}

impl Engine {
    /// A fresh engine: nothing armed, no scratch warmed up.
    #[must_use]
    pub fn new() -> Engine {
        Engine {
            limits: Limits::new(),
            apply: crate::apply::conjoin::ApplyScratch::new(),
            build: crate::build::BuildScratch::default(),
            restrict: crate::apply::conjoin::RestrictScratch::default(),
            clause: crate::apply::conjoin_clause::ClauseScratch::default(),
            reduce: crate::reduce::scratch::ReduceScratch::default(),
            restructure: crate::restructure::scratch::RestructurePool::default(),
            sparse: std::cell::RefCell::new(crate::apply::conjoin::SparseWorkspace::default()),
            levels: crate::diagram::LevelPool::default(),
            leaf_marginalize_inlines: std::cell::Cell::new(true),
        }
    }

    /// Whether summing out a vtree leaf may inline the leaf's fixed count into
    /// its parent's references, dropping the leaf's Boolean structure. Returns
    /// the previous setting, for a caller that restores it.
    ///
    /// On by default: it is the size win that makes a parent's `(·,x)` and
    /// `(·,¬x)` branches twins for contraction. A caller that still needs to
    /// read the leaf's labels afterwards must turn it off — projection is the
    /// case in the field, since ∃-forget cofactors leaves by their Pos/Neg
    /// labels and an inlined leaf no longer carries them.
    pub fn set_leaf_marginalize_inlines(&self, inlines: bool) -> bool {
        self.leaf_marginalize_inlines.replace(inlines)
    }

    /// Whether [`Engine::set_leaf_marginalize_inlines`] is on.
    #[must_use]
    pub fn leaf_marginalize_inlines(&self) -> bool {
        self.leaf_marginalize_inlines.get()
    }

    /// A fresh engine with `set` armed.
    #[must_use]
    pub fn with_limits(set: LimitSet) -> Engine {
        let engine = Engine::new();
        engine.limits.install(set);
        engine
    }

    /// The limits armed on this engine.
    #[must_use]
    pub fn limit_set(&self) -> LimitSet {
        self.limits.armed()
    }

    /// Arm `set`, returning what was armed before — which is what a caller
    /// restores when its scope ends.
    pub fn set_limits(&mut self, set: LimitSet) -> LimitSet {
        self.limits.install(set)
    }

    /// The buffers the conjunctions on this engine reuse.
    #[must_use]
    #[inline]
    pub(crate) fn apply(&self) -> &crate::apply::conjoin::ApplyScratch {
        &self.apply
    }

    /// The clause-build pools.
    #[must_use]
    #[inline]
    pub(crate) fn build(&self) -> &crate::build::BuildScratch {
        &self.build
    }

    /// The restricted-apply pools.
    #[must_use]
    #[inline]
    pub(crate) fn restrict_pool(&self) -> &crate::apply::conjoin::RestrictScratch {
        &self.restrict
    }

    /// The clause-conjunction pools.
    #[must_use]
    #[inline]
    pub(crate) fn clause_pool(&self) -> &crate::apply::conjoin_clause::ClauseScratch {
        &self.clause
    }

    /// The reduction pools.
    #[must_use]
    #[inline]
    pub(crate) fn reduce(&self) -> &crate::reduce::scratch::ReduceScratch {
        &self.reduce
    }

    /// The rotation-search pool.
    #[must_use]
    #[inline]
    pub(crate) fn restructure(&self) -> &crate::restructure::scratch::RestructurePool {
        &self.restructure
    }

    /// The sparse-level workspace.
    #[must_use]
    #[inline]
    pub(crate) fn sparse(&self) -> &std::cell::RefCell<crate::apply::conjoin::SparseWorkspace> {
        &self.sparse
    }

    /// The recycled level arrays.
    #[must_use]
    #[inline]
    pub(crate) fn levels(&self) -> &crate::diagram::LevelPool {
        &self.levels
    }

    /// The limits themselves, for reading the meters and for the operations
    /// that charge against them.
    #[must_use]
    pub fn limits(&self) -> &Limits {
        &self.limits
    }

    /// A fresh engine whose schedule stops the first operation that asks it —
    /// the preemption the deadline tests assert, without a wall clock.
    #[cfg(test)]
    #[must_use]
    pub(crate) fn with_stop_now() -> Engine {
        Engine::with_limits(LimitSet::none().schedule(Some(|_, _| Scheduled::Stop)))
    }

    /// The number of satisfying assignments of `tdd`, under this engine's
    /// limits.
    ///
    /// [`query::model_count`](crate::query::model_count) is the same count with
    /// nothing armed to interrupt it.
    ///
    /// # Errors
    ///
    /// Propagates the armed stop, polled at every level of the bottom-up pass.
    ///
    /// # Panics
    ///
    /// Panics if `tdd` is poisoned (a mid-rewrite `OverBudget` left it in an
    /// inconsistent state); the caller must drop and recover instead of
    /// counting it.
    pub fn try_model_count(&self, tdd: &crate::Tdd) -> Result<num_bigint::BigUint, crate::error::ApplyError> {
        crate::query::count::try_model_count(self, tdd)
    }

    /// Release everything this engine retains — every scratch allocation and
    /// every pooled buffer — leaving the armed limits alone.
    ///
    /// Called between a failed operation and whatever a caller does to recover
    /// from it, so the recovery starts on a clean allocator slate rather than
    /// inheriting the peak the failure left behind. An operation cut by an
    /// unwind leaves its buffers parked at full capacity; this is the reclaim.
    ///
    /// Sound only between operations: an apply in flight holds the sparse
    /// workspace borrowed, and resetting under it panics.
    pub fn reset(&self) {
        self.apply.drain();
        self.build.drain();
        self.restrict.drain();
        self.clause.drain();
        self.reduce.drain();
        self.restructure.drain();
        crate::apply::conjoin::reset_sparse_ws(self);
        crate::diagram::drop_pools(self);
    }
}
