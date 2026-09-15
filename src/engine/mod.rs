//! Shared execution contexts and the engines lent to batches.
//!
//! [`Engine::new`] creates an engine; [`Engine::limits`] configures its limits.
//! Diagrams own their results independently of the engine and can outlive it.
//! [`Context`] lends engines to batches and retains their scratch between calls.
//!
//! The [task guide](crate::guide::api) links construction, transformation and
//! query examples; [`Limits`] documents configuration and work measurements.

use crate::limits::Limits;

mod context;
pub use context::Context;

/// An execution workspace for checked operations and explicit resource limits.
///
/// [`Context::run`] and [`Context::with_limits`] lend an engine across a sequence
/// of operations. An engine can also be created directly with [`new`](Self::new).
/// It retains scratch buffers between calls; diagrams own their results and can
/// outlive it.
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
/// let tree = Arc::new(Vtree::balanced(3));
/// let f = tree.context().with_limits(
///     LimitConfig::none().with_memory_budget_bytes(Some(1_000_000)),
///     |engine| {
///         let either = engine.clause(&tree, [1, 2])?;
///         let not_third = engine.cube(&tree, [-3])?;
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
    context: std::sync::Weak<Context>,
    limits: Limits,
    apply: crate::apply::conjoin::ApplyScratch,
    clause: crate::apply::conjoin_clause::ClauseScratch,
    reduce: crate::reduce::scratch::ReduceScratch,
    restructure: crate::limits::pool::Pool<crate::restructure::scratch::RestructureScratch>,
    sparse: std::cell::RefCell<crate::apply::conjoin::SparseWorkspace>,
    levels: crate::diagram::LevelPool,
}

impl std::fmt::Debug for Engine {
    /// What is armed on the engine and how much recycled level capacity it is
    /// sitting on — the scratch buffers themselves are working memory.
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
    /// let tree = Arc::new(Vtree::balanced(2));
    /// let f = {
    ///     let engine = Engine::new();
    ///     engine.clause(&tree, [1, 2])?
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
            reduce: crate::reduce::scratch::ReduceScratch::default(),
            restructure: crate::limits::pool::Pool::default(),
            sparse: std::cell::RefCell::new(crate::apply::conjoin::SparseWorkspace::default()),
            levels: crate::diagram::LevelPool::default(),
        }
    }

    /// Share a vtree with this execution batch's context.
    ///
    /// An engine checked out by [`Context::run`] or [`Context::with_limits`]
    /// associates the tree with that context. A standalone engine gives the
    /// tree a fresh context. Sharing a context does not make separate vtree
    /// allocations compatible operands.
    #[must_use]
    pub fn bind_vtree(&self, tree: crate::Vtree) -> std::sync::Arc<crate::Vtree> {
        match self.context.upgrade() {
            Some(context) => context.bind(tree),
            None => std::sync::Arc::new(Context::new()).bind(tree),
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
    pub(crate) fn reduce_scratch(&self) -> &crate::reduce::scratch::ReduceScratch {
        &self.reduce
    }

    /// The rotation-search pool.
    #[must_use]
    #[inline]
    pub(crate) fn restructure(&self) -> &crate::limits::pool::Pool<crate::restructure::scratch::RestructureScratch> {
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
    ///
    /// # Panics
    ///
    /// If a conjunction on this engine is in flight — reachable only from a
    /// schedule hook or a memory probe — since it holds the sparse workspace
    /// borrowed.
    pub fn clear_scratch(&self) {
        self.apply.drain();
        self.clause.drain();
        self.reduce.drain();
        self.restructure.drain();
        crate::apply::conjoin::reset_sparse_ws(self);
        crate::diagram::drop_pools(self);
    }
}
