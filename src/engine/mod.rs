//! The hub: the one struct every operation is a method on, holding the scratch
//! each of them reuses and the limits they all run under.
//!
//! An engine holds what an operation runs under and what it reuses, never what
//! it produces: the diagram's contents belong to [`crate::diagram`], the limits
//! themselves to [`crate::limits`], and the operations to [`crate::apply`],
//! [`crate::marginal`], [`crate::reduce`] and [`crate::restructure`].
//!
//! Entry points: [`Engine::new`] opens a session; [`Engine::limits`] reaches the
//! armed [`Limits`], which [`LimitConfig`](crate::limits::LimitConfig) describes and
//! [`Limits::install`]/[`Limits::scope`] arm; [`Limits::meters`] reads what the
//! last operation spent, and [`MemoryHooks`](crate::limits::MemoryHooks)
//! installs the host's memory probes.
//!
//! Two engines never share a buffer, and dropping one frees its scratch.
//!
//! Every operation has one real form — an `Engine` method, or a function that
//! takes the engine — and at most one sugar, which is the spelling a doc
//! example writes. A sugar is a forward that builds a transient engine and
//! panics on failure.
//!
//! | Operation | Real form | Sugar |
//! |---|---|---|
//! | build | [`Engine::clause`], [`Engine::cube`], [`Engine::one`], [`Engine::zero`] | [`Tdd::clause`](crate::Tdd::clause), [`Tdd::one`](crate::Tdd::one), [`Tdd::zero`](crate::Tdd::zero) |
//! | conjunction, disjunction | [`Engine::and`], [`Engine::and_clause`], [`Engine::or`] | `&`, `\|` |
//! | negation | [`negate`](crate::apply::negate()), which reduces on a transient engine | `!` |
//! | conditioning | [`Engine::condition_var`], [`Engine::condition_vars`] | [`condition_var`](crate::apply::condition_var), [`condition_vars`](crate::apply::condition_vars) |
//! | projection | [`Engine::exists_var`], [`Engine::exists_vars`] | [`exists_var`](crate::apply::exists_var), [`exists_vars`](crate::apply::exists_vars) |
//! | restriction | [`Engine::restrict_to_care`] | [`restrict_to_care`](crate::apply::restrict_to_care()) |
//! | marginalization | [`marginalize_levels`](crate::marginal::marginalize_levels) | none |
//! | reduction | [`try_reduce`](crate::reduce::try_reduce) | [`minimize`](crate::reduce::minimize) |
//! | rotation search | [`Engine::rotation_search`] | none |
//! | model count | [`Engine::model_count`] | [`Tdd::model_count`](crate::Tdd::model_count) |
//!
//! [`ModelCounter`](crate::query::ModelCounter) also runs its
//! passes on an engine; the other reads in [`crate::query`] take none and
//! are never cut.

use crate::limits::Limits;

/// The limits and scratch one caller's operations run on.
///
/// Build one per compile and thread it through: every conjunction, reduction,
/// marginalization and restructuring takes `&Engine`, reuses the buffers it
/// holds, and is cut by the limits armed on it. Not `Sync`: one engine serves
/// one thread.
pub struct Engine {
    limits: Limits,
    apply: crate::apply::conjoin::ApplyScratch,
    clause: crate::apply::conjoin_clause::ClauseScratch,
    reduce: crate::reduce::scratch::ReduceScratch,
    restructure: crate::restructure::scratch::RestructurePool,
    sparse: std::cell::RefCell<crate::apply::conjoin::SparseWorkspace>,
    levels: crate::diagram::LevelPool,
    /// See [`Engine::set_leaf_marginalize_inlines`].
    leaf_marginalize_inlines: std::cell::Cell<bool>,
}

impl std::fmt::Debug for Engine {
    /// What is armed on the engine and how much recycled level capacity it is
    /// sitting on — the scratch buffers themselves are working memory.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Engine")
            .field("armed", &self.limits.armed())
            .field("pooled_levels", &self.levels.occupancy())
            .field("leaf_marginalize_inlines", &self.leaf_marginalize_inlines.get())
            .finish()
    }
}

impl Default for Engine {
    fn default() -> Engine {
        Engine::new()
    }
}

impl Engine {
    /// A fresh engine: nothing armed, no scratch warmed up.
    #[must_use]
    pub fn new() -> Engine {
        Engine {
            limits: Limits::new(),
            apply: crate::apply::conjoin::ApplyScratch::default(),
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
    /// `(·,¬x)` branches twins for contraction. Off, the parent sums the leaf
    /// through its labels instead and the leaf keeps them. A caller that will
    /// still read a leaf's labels after summing it out — projection cofactors
    /// by them, and no operation turns the setting off on its own — turns it
    /// off before the marginalization. [`Engine::reset`] leaves it as it is.
    pub fn set_leaf_marginalize_inlines(&self, inlines: bool) -> bool {
        self.leaf_marginalize_inlines.replace(inlines)
    }

    /// Whether [`Engine::set_leaf_marginalize_inlines`] is on.
    #[must_use]
    pub(crate) fn leaf_marginalize_inlines(&self) -> bool {
        self.leaf_marginalize_inlines.get()
    }

    /// The buffers the conjunctions on this engine reuse.
    #[must_use]
    #[inline]
    pub(crate) fn apply(&self) -> &crate::apply::conjoin::ApplyScratch {
        &self.apply
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


    /// Release everything this engine retains — every scratch allocation and
    /// every pooled buffer. The armed limits, the meters and the
    /// leaf-inlining setting stay as they are.
    ///
    /// Called between a failed operation and whatever a caller does to recover
    /// from it, so the recovery starts on a clean allocator slate rather than
    /// inheriting the peak the failure left behind. An operation cut by an
    /// unwind leaves its buffers parked at full capacity; this is the reclaim.
    ///
    /// # Panics
    ///
    /// If a conjunction on this engine is in flight — reachable only from a
    /// schedule hook or a memory probe — since it holds the sparse workspace
    /// borrowed.
    pub fn reset(&self) {
        self.apply.drain();
        self.clause.drain();
        self.reduce.drain();
        self.restructure.drain();
        crate::apply::conjoin::reset_sparse_ws(self);
        crate::diagram::drop_pools(self);
    }
}
