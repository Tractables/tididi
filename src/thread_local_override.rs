//! Scoped overrides of thread-local knobs: install a value for a lexical
//! scope, restore the prior one on drop (panic-safe).

use std::cell::{Cell, RefCell};
use std::thread::LocalKey;

/// A thread-local cell whose value can be swapped out and back.
pub trait Swappable {
    /// The stored value.
    type Value;
    /// Store `v`, returning what was stored before.
    fn swap(&self, v: Self::Value) -> Self::Value;
}

impl<T: Copy> Swappable for Cell<T> {
    type Value = T;
    fn swap(&self, v: T) -> T {
        self.replace(v)
    }
}

impl<T> Swappable for RefCell<T> {
    type Value = T;
    fn swap(&self, v: T) -> T {
        self.replace(v)
    }
}

/// A value installed in a thread-local slot for the lifetime of this guard;
/// dropping it restores the value the slot held before.
#[must_use = "the override lasts only as long as the guard; bind it to a name"]
pub struct Scoped<S: Swappable + 'static> {
    slot: &'static LocalKey<S>,
    prior: Option<S::Value>,
}

impl<S: Swappable + 'static> Scoped<S> {
    /// Install `value` in `slot` until the returned guard drops.
    pub(crate) fn install(slot: &'static LocalKey<S>, value: S::Value) -> Self {
        let prior = slot.with(|c| c.swap(value));
        Self { slot, prior: Some(prior) }
    }

    /// Run `body` with `value` installed in `slot`.
    #[cfg(test)]
    pub(crate) fn run<R>(slot: &'static LocalKey<S>, value: S::Value, body: impl FnOnce() -> R) -> R {
        let _guard = Self::install(slot, value);
        body()
    }
}

impl<S: Swappable + 'static> Drop for Scoped<S> {
    fn drop(&mut self) {
        if let Some(prior) = self.prior.take() {
            self.slot.with(|c| {
                c.swap(prior);
            });
        }
    }
}
