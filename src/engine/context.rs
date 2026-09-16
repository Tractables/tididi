//! Shared scratch checkout for diagram operations and explicitly bounded batches.

use std::sync::{Arc, Mutex};

use super::Engine;
use crate::limits::{LimitConfig, Limits};
use crate::Vtree;

/// Reuse execution scratch across diagrams without storing their nodes.
///
/// Every vtree starts with a context. Ordinary diagram operations use that
/// context automatically. Bind several trees to one context when their
/// operations should reuse the same scratch capacity.
///
/// [`run`](Self::run) lends an engine for a batch of checked operations;
/// [`with_limits`](Self::with_limits) also installs limits for that batch.
/// Nested and concurrent calls use separate engines while a checkout is active.
/// No lock is held while an operation or user callback runs.
///
/// The context retains at most one idle engine. Dropping the last context
/// reference frees its parked scratch; diagrams continue to own their results.
#[derive(Default)]
pub struct Context {
    /// Checkout moves only the pointer to the retained engine.
    idle: Mutex<Option<Box<Engine>>>,
}

impl std::fmt::Debug for Context {
    /// Describe the context without inspecting an active or parked engine.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Context").finish_non_exhaustive()
    }
}

impl Context {
    /// Create a context with no retained scratch.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Run a batch with reusable scratch and no limits initially installed.
    ///
    /// Calls made through the supplied engine share its scratch and limit
    /// configuration. Nested context calls and ordinary diagram methods check
    /// out their own engines; they do not inherit this batch's limits.
    ///
    /// Configuration, callbacks and work measurements are discarded when the
    /// batch returns. An unwinding batch also discards its scratch.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::engine::Context;
    /// use tididi::Vtree;
    /// let context = Arc::new(Context::new());
    /// let vtree = context.bind(Vtree::balanced(3));
    /// let f = context.run(|operations| {
    ///     let either = operations.clause(&vtree, [1, 2])?;
    ///     operations.and(either, operations.literal(&vtree, 3)?)
    /// })?;
    /// assert_eq!(f.model_count()?, 3u32.into());
    /// # Ok::<(), tididi::OperationError>(())
    /// ```
    pub fn run<R>(self: &Arc<Self>, run: impl FnOnce(&Engine) -> R) -> R {
        let parked = self.idle.lock().unwrap_or_else(|error| error.into_inner()).take();
        let mut engine = parked.unwrap_or_default();
        engine.context = Arc::downgrade(self);
        let checkout = Checkout { context: self, engine: Some(engine) };
        run(checkout.engine.as_ref().unwrap())
    }

    /// Run a batch with `config` installed on its checked operations.
    ///
    /// Use the supplied engine throughout the bounded work. Independent and
    /// nested context calls have separate limits and meters. Each top-level
    /// engine operation resets its meters; component operations within it share
    /// their charges, following [`Limits`].
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::engine::Context;
    /// use tididi::limits::LimitConfig;
    /// use tididi::{OperationError, Vtree};
    /// let context = Arc::new(Context::new());
    /// let vtree = context.bind(Vtree::balanced(3));
    /// let limit = LimitConfig::none().with_memory_budget_bytes(Some(0));
    /// let result = context.with_limits(limit, |operations| operations.clause(&vtree, [1, 2]));
    /// assert_eq!(result.unwrap_err(), OperationError::OverBudget);
    /// assert!(context.run(|operations| operations.clause(&vtree, [1, 2])).is_ok());
    /// ```
    pub fn with_limits<R>(self: &Arc<Self>, config: LimitConfig, run: impl FnOnce(&Engine) -> R) -> R {
        self.run(|engine| {
            let _scope = engine.limits().scope(config);
            run(engine)
        })
    }

    /// Associate `tree` with this context and share it for diagram construction.
    ///
    /// The tree's shape and variable ids are unchanged. Two separately bound
    /// trees still have distinct allocations and cannot be binary operands.
    #[must_use]
    pub fn bind(self: &Arc<Self>, tree: Vtree) -> Arc<Vtree> {
        Arc::new(tree.with_context(Arc::clone(self)))
    }

    /// Release parked scratch without affecting operations already running.
    ///
    /// An active batch may return its scratch afterward; call between batches
    /// when all retained scratch should be released.
    pub fn clear_scratch(&self) {
        let parked = self.idle.lock().unwrap_or_else(|error| error.into_inner()).take();
        drop(parked);
    }
}

/// Return a finished engine after dropping configuration outside the pool lock.
struct Checkout<'a> {
    context: &'a Context,
    engine: Option<Box<Engine>>,
}

impl Drop for Checkout<'_> {
    fn drop(&mut self) {
        let Some(mut engine) = self.engine.take() else { return; };
        if std::thread::panicking() { return; }
        engine.limits = Limits::new();
        engine.context = std::sync::Weak::new();
        let previous = self.context.idle.lock().unwrap_or_else(|error| error.into_inner())
            .replace(engine);
        drop(previous);
    }
}

#[cfg(test)]
mod tests;
