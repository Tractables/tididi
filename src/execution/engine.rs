//! Per-batch scratch storage and execution limits.

use crate::limits::Limits;
use super::Context;
use super::pool::{Drain, Pool, Pools, ScratchLedger};

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
    pub(crate) scratch: EngineScratch,
}

/// Every pool an engine parks scratch in between operations, one field per
/// operation that reuses it, and the total they park.
#[derive(Default)]
pub(crate) struct EngineScratch {
    pub(super) ledger: ScratchLedger,
    pub(crate) apply: crate::apply::conjoin::ApplyScratch,
    pub(crate) clause: crate::apply::conjoin_clause::ClauseScratch,
    pub(crate) reduce: crate::reduce::ReduceScratch,
    pub(crate) negate: crate::apply::negate::NegateScratch,
    pub(crate) restructure: Pool<crate::restructure::scratch::RestructureScratch>,
    pub(crate) sparse: Pool<crate::apply::conjoin::SparseWorkspace>,
    pub(crate) levels: crate::diagram::LevelPool,
    pub(crate) model_layout: Pool<crate::build::models::layout::Layout>,
}

impl Pools for EngineScratch {
    fn pools(&self, visit: &mut dyn FnMut(&dyn Drain)) {
        self.apply.pools(visit);
        self.clause.pools(visit);
        self.reduce.pools(visit);
        self.negate.pools(visit);
        visit(&self.restructure);
        visit(&self.sparse);
        self.levels.pools(visit);
        visit(&self.model_layout);
    }
}

impl std::fmt::Debug for Engine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Engine")
            .field("armed", &self.limits.armed())
            .field("pooled_levels", &self.scratch.levels.occupancy())
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
            scratch: EngineScratch::default(),
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
        self.scratch.drain(self);
        debug_assert_eq!(self.scratch.ledger.bytes(), 0, "a pool is missing from the list in `EngineScratch`");
    }
}
