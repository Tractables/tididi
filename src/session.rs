//! The [`Engine`] struct itself. Declared last in `lib.rs`: it is the one
//! struct that names every module's scratch, so it sits below all of them
//! rather than inside `engine`, whose own files are leaves.

use crate::engine::Limits;
#[cfg(test)]
use crate::engine::{LimitSet, Scheduled};

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
            apply: crate::apply::conjoin::ApplyScratch::default(),
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
        let engine = Engine::new();
        engine.limits.install(LimitSet::none().schedule(Some(|_, _| Scheduled::Stop)));
        engine
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