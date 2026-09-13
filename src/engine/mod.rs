//! Run checked operations with reusable scratch and caller-controlled limits.
//!
//! [`Engine::new`] creates an engine; [`Engine::limits`] configures its limits.
//! Diagrams own their results independently of the engine and can outlive it.
//! Each engine keeps its own scratch buffers and frees them when dropped.
//!
//! The [task guide](crate::guide::api) links construction, transformation and
//! query examples; [`Limits`] documents configuration and work measurements.

use crate::limits::Limits;

/// The limits and scratch one caller's operations run on.
///
/// Build one per compile and thread it through: every conjunction, reduction,
/// marginalization and restructuring takes `&Engine`, reuses the buffers it
/// holds, and is cut by the limits armed on it. An engine is `Send` and may move
/// between threads; it is not `Sync` and serves one thread at a time.
///
/// Propagate errors with `?` through a sequence of operations:
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
/// assert_eq!(engine.model_count(&f)?, 3u32.into()); // (x1 ∨ x2) ∧ ¬x3
/// # let mut f = f;
/// # tididi::reduce::try_minimize(&engine, &mut f)?;
/// # tididi::test_helpers::assert_canonical(&f);
/// # Ok::<(), tididi::OperationError>(())
/// ```
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
    ///     engine.clause(&tree, [1, 2]).unwrap()
    /// };
    /// assert_eq!(f.model_count(), 3u32.into());
    /// # tididi::test_helpers::assert_canonical(&f);
    /// ```
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
