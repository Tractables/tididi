//! Per-batch scratch storage and execution limits.

use crate::limits::Limits;
use super::Context;

/// An execution workspace for checked operations and explicit resource limits.
///
/// [`Context::run`] and [`Context::with_limits`] lend an engine across a sequence
/// of operations. An engine can also be created directly with [`new`](Self::new).
/// It retains scratch buffers between calls; diagrams own their results and can
/// outlive it. Retained scratch shares one capacity ceiling across operations;
/// [`clear_scratch`](Self::clear_scratch) releases it explicitly.
/// Operands may come from different engines; binary operations require a shared
/// vtree allocation, as described on [`Tdd`](crate::Tdd).
/// Ordinary [`and`](crate::and), [`or`](crate::or) and diagram queries use the
/// vtree's [`Context`] automatically.
/// An engine is `Send` but not `Sync`: it can move between threads, and one
/// thread uses it at a time.
///
/// # Bound a batch
///
/// Use the supplied engine for every operation that must obey the batch limits.
/// Ordinary free functions and diagram methods use separate unlimited checkouts.
///
/// ```
/// use std::sync::Arc;
/// use tididi::{Tdd, Vtree};
/// use tididi::limits::LimitConfig;
/// let vtree = Arc::new(Vtree::balanced(3));
/// let f = vtree.context().with_limits(
///     LimitConfig::none().with_memory_budget_bytes(Some(1_000_000)),
///     |engine| {
///         let either = engine.clause(&vtree, [1, 2])?;
///         let not_third = engine.cube(&vtree, [-3])?;
///         let mut f = engine.and(either, not_third)?;
///         engine.minimize(&mut f)?;
///         assert_eq!(engine.model_count(&f)?, 3u32.into());
///         Ok::<Tdd, tididi::OperationError>(f)
///     },
/// )?;
/// assert_eq!(f.model_count()?, 3u32.into());
/// # Ok::<(), tididi::OperationError>(())
/// ```
///
/// A directly constructed engine starts without limits; [`Limits::scope`] applies
/// a temporary configuration. Each method states what remains after a refusal.
/// Transformations taking `Tdd` consume it on success and error; queries borrowing
/// it leave it unchanged. The infallible [`one`](Self::one) and [`zero`](Self::zero)
/// methods do not check limits; use an empty [`cube`](Self::cube) or
/// [`clause`](Self::clause) for checked constant construction.
pub struct Engine {
    pub(super) context: std::sync::Weak<Context>,
    pub(super) limits: Limits,
    apply: crate::apply::conjoin::ApplyScratch,
    clause: crate::apply::conjoin_clause::ClauseScratch,
    reduce: crate::reduce::ReduceScratch,
    negate: crate::apply::negate::NegateScratch,
    restructure: crate::execution::pool::Pool<crate::restructure::scratch::RestructureScratch>,
    sparse: crate::execution::pool::Pool<crate::apply::conjoin::SparseWorkspace>,
    levels: crate::diagram::LevelPool,
    model_layout: crate::execution::pool::Pool<crate::build::models::layout::Layout>,
}

impl std::fmt::Debug for Engine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Engine")
            .field("armed", &self.limits.armed())
            .field("pooled_levels", &self.levels.occupancy())
            .finish()
    }
}

impl Default for Engine {
    fn default() -> Engine {
        Engine::new()
    }
}

impl Engine {
    /// Create an engine with no installed limits and no retained working buffers.
    ///
    /// A returned diagram owns its storage and can outlive the engine:
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{Engine, Vtree};
    ///
    /// let vtree = Arc::new(Vtree::balanced(2));
    /// let f = {
    ///     let engine = Engine::new();
    ///     engine.clause(&vtree, [1, 2])?
    /// };
    /// assert_eq!(f.model_count()?, 3u32.into());
    /// # Ok::<(), tididi::OperationError>(())
    /// ```
    #[must_use]
    pub fn new() -> Engine {
        Engine {
            context: std::sync::Weak::new(),
            limits: Limits::new(),
            apply: crate::apply::conjoin::ApplyScratch::default(),
            clause: crate::apply::conjoin_clause::ClauseScratch::default(),
            reduce: crate::reduce::ReduceScratch::default(),
            negate: crate::apply::negate::NegateScratch::default(),
            restructure: crate::execution::pool::Pool::default(),
            sparse: crate::execution::pool::Pool::default(),
            levels: crate::diagram::LevelPool::default(),
            model_layout: crate::execution::pool::Pool::default(),
        }
    }

    /// Share a vtree with this execution batch's context.
    ///
    /// An engine checked out by [`Context::run`] or [`Context::with_limits`]
    /// associates the vtree with that context. A standalone engine gives the
    /// vtree a fresh context. Sharing a context does not make separate vtree
    /// allocations compatible operands.
    #[must_use]
    pub fn bind_vtree(&self, vtree: crate::Vtree) -> std::sync::Arc<crate::Vtree> {
        match self.context.upgrade() {
            Some(context) => context.bind(vtree),
            None => std::sync::Arc::new(Context::new()).bind(vtree),
        }
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
    pub(crate) fn reduce_scratch(&self) -> &crate::reduce::ReduceScratch {
        &self.reduce
    }

    /// The negation pools.
    #[must_use]
    #[inline]
    pub(crate) fn negate_scratch(&self) -> &crate::apply::negate::NegateScratch {
        &self.negate
    }

    /// The rotation-search pool.
    #[must_use]
    #[inline]
    pub(crate) fn restructure(&self) -> &crate::execution::pool::Pool<crate::restructure::scratch::RestructureScratch> {
        &self.restructure
    }

    /// The sparse-level workspace.
    #[must_use]
    #[inline]
    pub(crate) fn sparse(&self) -> &crate::execution::pool::Pool<crate::apply::conjoin::SparseWorkspace> {
        &self.sparse
    }

    /// The model-table layout pool.
    #[must_use]
    #[inline]
    pub(crate) fn model_layout(&self) -> &crate::execution::pool::Pool<crate::build::models::layout::Layout> {
        &self.model_layout
    }

    /// The recycled level arrays.
    #[must_use]
    #[inline]
    pub(crate) fn levels(&self) -> &crate::diagram::LevelPool {
        &self.levels
    }

    /// Access the engine's limit configuration and work measurements.
    ///
    /// [`Limits::scope`] installs limits temporarily; [`Limits::edit`] changes
    /// selected settings while preserving the others.
    #[must_use]
    pub fn limits(&self) -> &Limits {
        &self.limits
    }


    /// Release retained scratch buffers and pooled storage, preserving limits and meters.
    ///
    /// Call between operations to release capacity retained by completed or
    /// refused work. Diagram results remain intact.
    /// [`Context::clear_scratch`] releases the idle workspace in a shared context.
    /// Active operations keep their checked-out buffers until they finish.
    pub fn clear_scratch(&self) {
        self.apply.drain(&self.limits);
        self.clause.drain(&self.limits);
        self.reduce.drain(&self.limits);
        self.negate.drain(&self.limits);
        self.restructure.drain(&self.limits);
        self.sparse.drain(&self.limits);
        self.levels.drain(&self.limits);
        self.model_layout.drain(&self.limits);
    }
}
