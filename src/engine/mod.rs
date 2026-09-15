//! Run checked operations with reusable scratch and caller-controlled limits.
//!
//! [`Engine::new`] creates an engine; [`Engine::limits`] configures its limits.
//! Diagrams own their results independently of the engine and can outlive it.
//! Each engine keeps its own scratch buffers and frees them when dropped.
//!
//! The [task guide](crate::guide::api) links construction, transformation and
//! query examples; [`Limits`] documents configuration and work measurements.

use crate::limits::Limits;

/// Checked operations, reusable working memory, and resource limits for diagrams.
///
/// Reuse an engine across a sequence of operations; it retains scratch buffers
/// for the next call. Diagrams own their results and can outlive the engine.
/// An engine is `Send` but not `Sync`: it can move between threads, and one
/// thread uses it at a time.
///
/// # Build, then query
///
/// Choose a shared vtree for all operands. Here a clause expresses “at least one
/// of the first two variables,” and a cube fixes the third to false:
///
/// ```
/// use std::sync::Arc;
/// use tididi::{Engine, Vtree};
///
/// let engine = Engine::new();
/// let tree = Arc::new(Vtree::balanced(3));
/// let either = engine.clause(&tree, [1, 2])?;
/// let not_third = engine.cube(&tree, [-3])?;
/// let f = engine.and(either, not_third)?;
/// assert_eq!(engine.model_count(&f)?, 3u32.into());
/// # Ok::<(), tididi::OperationError>(())
/// ```
///
/// Transformations taking `Tdd` consume their operands, including on error;
/// queries taking `&Tdd` leave them available. See [`Tdd`](crate::Tdd) for copying
/// an operand when several transformations need it.
///
/// # Choose the operation you need
///
/// The [task guide](crate::guide::api) groups the methods by user task.
/// [`and`](Self::and) can leave a nonminimal representation; use
/// [`try_minimize`](crate::reduce::try_minimize) when canonical form is needed.
/// [`model_count`](Self::model_count), [`is_sat`](Self::is_sat), and
/// [`satisfying_assignment`](Self::satisfying_assignment) accept nonminimal
/// structural diagrams, so a first build-and-query program needs no minimization
/// step for correctness. Minimize to remove redundant storage or before an
/// operation whose contract requires canonical form.
///
/// # Bound a computation
///
/// A new engine has no limits installed. Use [`Limits::scope`] to apply a
/// [`LimitConfig`](crate::limits::LimitConfig) to a block and restore the prior
/// configuration afterward. Checked methods return [`OperationError`](crate::OperationError)
/// on refusal; the method's contract specifies what remains after an error.
///
/// The `Tdd` constructors and Boolean operators are convenience forms with a
/// temporary engine and panic-on-error behavior. The infallible
/// [`one`](Self::one) and [`zero`](Self::zero) methods also do not check resource
/// limits; the empty [`cube`](Self::cube) or [`clause`](Self::clause) provides a
/// checked constant when needed.
pub struct Engine {
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
    /// assert_eq!(f.model_count(), 3u32.into());
    /// # Ok::<(), tididi::OperationError>(())
    /// ```
    #[must_use]
    pub fn new() -> Engine {
        Engine {
            limits: Limits::new(),
            apply: crate::apply::conjoin::ApplyScratch::default(),
            clause: crate::apply::conjoin_clause::ClauseScratch::default(),
            reduce: crate::reduce::scratch::ReduceScratch::default(),
            restructure: crate::limits::pool::Pool::default(),
            sparse: std::cell::RefCell::new(crate::apply::conjoin::SparseWorkspace::default()),
            levels: crate::diagram::LevelPool::default(),
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
    pub(crate) fn reduce(&self) -> &crate::reduce::scratch::ReduceScratch {
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


    /// Release everything this engine retains — every scratch allocation and
    /// every pooled buffer. The armed limits and meters stay as they are.
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
