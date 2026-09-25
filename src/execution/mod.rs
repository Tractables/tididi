//! Shared scratch storage and bounded batches.
//!
//! Ordinary operations reuse the vtree's [`Context`]. Use
//! [`Context::with_limits`] to lend an [`Engine`] to a bounded batch.

mod context;
mod engine;
pub(crate) mod pool;

pub use context::Context;
pub use engine::Engine;
