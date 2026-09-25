//! Reusable scratch values owned by an engine.
//!
//! A checkout takes ownership, allowing nested operations to obtain fresh
//! scratch from the same pool. Completed checkouts return their buffers;
//! unwinding discards them.

use std::cell::Cell;

use crate::Engine;
use crate::limits::{Charged, Limits};

/// Maximum retained scratch capacity between operations. Larger buffers are
/// released so an unusually large operation does not permanently retain them.
/// Sized so a repeated wide conjunction keeps the buffers it fills — a
/// smaller cap re-grows them, and pays their page faults, on every call.
/// Level arenas have a separate limit, `diagram::MAX_LEVEL_ARENA_BYTES`.
pub(crate) const SCRATCH_RETAIN_BYTES: usize = 128 * 1024 * 1024;

/// Combined capacity estimate parked in one engine, including recycled diagram levels.
/// This counts backing storage, not allocator metadata or live operation results.
pub(crate) const ENGINE_RETAIN_BYTES: usize = 256 * 1024 * 1024;

/// A scratch value parked between operations, absent while checked out.
///
/// A nested checkout finds an empty pool and creates a fresh working set.
pub(crate) struct Pool<T> {
    value: Cell<Option<T>>,
    bytes: Cell<usize>,
}

impl<T> Default for Pool<T> {
    fn default() -> Self {
        Pool { value: Cell::new(None), bytes: Cell::new(0) }
    }
}

impl<T> Pool<T> {
    /// Whether this slot contains a parked value.
    pub(crate) fn occupied(&self) -> bool { self.bytes.get() != 0 }

    /// Remove the parked value, if any, and its claim on the engine's allowance.
    #[inline]
    fn unpark(&self, eng: &Engine) -> Option<T> {
        eng.scratch.ledger.release(self.bytes.replace(0));
        self.value.take()
    }
}

impl<T: Default> Pool<T> {
    /// Take the parked value, creating an empty one if the pool is vacant.
    #[inline]
    pub(crate) fn take(&self, eng: &Engine) -> T {
        self.unpark(eng).unwrap_or_default()
    }
}

impl<T: PooledScratch> Pool<T> {
    /// Retire a working set and admit it against the engine's shared ceiling.
    #[inline]
    pub(crate) fn put(&self, eng: &Engine, mut value: T) {
        self.drain(eng);
        value.retain(eng.limits());
        let bytes = value.retained_bytes();
        if eng.scratch.ledger.admit(bytes) {
            self.bytes.set(bytes);
            self.value.set(Some(value));
        }
    }
}

/// A pool, whatever it holds, as [`Pools`] lists it.
pub(crate) trait Drain {
    /// Drop whatever this pool retains.
    fn drain(&self, eng: &Engine);
}

impl<T> Drain for Pool<T> {
    #[inline]
    fn drain(&self, eng: &Engine) {
        self.unpark(eng);
    }
}

/// A group of pools that can list them.
///
/// Whatever has to reach every pool an engine owns, such as
/// [`Engine::clear_scratch`], walks this one list.
pub(crate) trait Pools {
    /// Hand every pool in this group to `visit`, once each.
    fn pools(&self, visit: &mut dyn FnMut(&dyn Drain));

    /// Drop whatever every pool in this group retains.
    fn drain(&self, eng: &Engine) {
        self.pools(&mut |pool| pool.drain(eng));
    }
}

/// The capacity an engine's pools hold between operations, kept within
/// [`ENGINE_RETAIN_BYTES`].
///
/// It belongs with the pools rather than with the batch's [`Limits`]: a batch
/// ends by replacing the limits, and the pools and this total carry over.
#[derive(Default)]
pub(crate) struct ScratchLedger(Cell<usize>);

impl ScratchLedger {
    /// The bytes parked now.
    pub(crate) fn bytes(&self) -> usize { self.0.get() }

    /// Add `bytes` if the total stays within the ceiling, and say whether it did.
    #[inline]
    fn admit(&self, bytes: usize) -> bool {
        let total = self.0.get().saturating_add(bytes);
        let fits = total <= ENGINE_RETAIN_BYTES;
        if fits { self.0.set(total); }
        fits
    }

    /// Remove `bytes` that a pool no longer parks.
    #[inline]
    fn release(&self, bytes: usize) {
        self.0.set(self.0.get() - bytes);
    }
}

/// A value that owns scratch buffers and can list them.
///
/// The retained byte count and the retention rules all walk this one list, so
/// a buffer is trimmed exactly when it is counted.
pub(crate) trait Buffers {
    /// Hand every buffer this value owns to `visit`, once each.
    fn buffers(&mut self, visit: &mut dyn FnMut(&mut dyn Scratch));

    /// Backing capacity of every buffer, including owned inner allocations.
    fn retained_bytes(&mut self) -> usize {
        let mut bytes = 0u64;
        self.buffers(&mut |buf| bytes = bytes.saturating_add(buf.charged_bytes()));
        usize::try_from(bytes).unwrap_or(usize::MAX)
    }

    /// Apply [`release_if_oversized`] to each buffer on its own.
    fn release_oversized(&mut self, lim: &Limits) {
        self.buffers(&mut |buf| release_if_oversized(lim, buf));
    }

    /// Release every buffer, giving the freed bytes back to `lim`.
    fn release_all(&mut self, lim: &Limits) {
        self.buffers(&mut |buf| {
            lim.release_bytes(buf.charged_bytes());
            buf.release();
        });
    }
}

/// Checkout preparation and capacity retention for an engine-owned working set.
pub(crate) trait PooledScratch: Default + Buffers {
    /// Invalidate previous results before the working set is used again.
    fn prepare(&mut self);
    /// Release allocations that exceed this working set's retention policy,
    /// giving the freed bytes back to `lim`. Each buffer is judged alone unless
    /// the working set needs a rule of its own.
    fn retain(&mut self, lim: &Limits) {
        self.release_oversized(lim);
    }
}

impl<T> Buffers for Vec<T> {
    #[inline]
    fn buffers(&mut self, visit: &mut dyn FnMut(&mut dyn Scratch)) { visit(self); }
}

impl<T> PooledScratch for Vec<T> {
    #[inline]
    fn prepare(&mut self) { self.clear(); }
}

impl<T: PooledScratch> Pool<T> {
    /// Check out scratch, returning it automatically on an ordinary exit.
    ///
    /// `eng` is held for the return: the retention policy runs when the guard
    /// drops, whatever it frees is given back to the engine's byte meter, and
    /// what is kept is admitted against the engine's ceiling.
    #[inline]
    pub(crate) fn checkout<'a>(&'a self, eng: &'a Engine) -> PoolGuard<'a, T> {
        let mut guard = self.checkout_preserving(eng);
        guard.prepare();
        guard
    }

    /// Keep initialized entries for algorithms that overwrite their live range.
    /// The caller must invalidate stale results before reading them.
    #[inline]
    pub(crate) fn checkout_preserving<'a>(&'a self, eng: &'a Engine) -> PoolGuard<'a, T> {
        PoolGuard { pool: self, eng, value: self.take(eng) }
    }
}

/// Own checked-out scratch without borrowing the pool's contents.
///
/// Unwinding discards partially updated scratch; ordinary exits apply its
/// retention policy and park it for the next checkout, including nested uses.
pub(crate) struct PoolGuard<'a, T: PooledScratch> {
    pool: &'a Pool<T>,
    eng: &'a Engine,
    value: T,
}

impl<T: PooledScratch> std::ops::Deref for PoolGuard<'_, T> {
    type Target = T;
    #[inline]
    fn deref(&self) -> &T { &self.value }
}

impl<T: PooledScratch> std::ops::DerefMut for PoolGuard<'_, T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut T { &mut self.value }
}

impl<T: PooledScratch> Drop for PoolGuard<'_, T> {
    fn drop(&mut self) {
        if !std::thread::panicking() {
            self.pool.put(self.eng, std::mem::take(&mut self.value));
        }
    }
}

/// A charged scratch buffer that can give its allocation back.
pub(crate) trait Scratch: Charged {
    /// Give the allocation back, leaving an empty buffer.
    fn release(&mut self);
}

impl<T> Scratch for Vec<T> {
    #[inline]
    fn release(&mut self) {
        *self = Vec::new();
    }
}

impl<T, S: Default> Scratch for std::collections::HashSet<T, S> {
    #[inline]
    fn release(&mut self) {
        *self = Self::default();
    }
}

impl<K, V, S: Default> Scratch for std::collections::HashMap<K, V, S> {
    #[inline]
    fn release(&mut self) {
        *self = Self::default();
    }
}

/// Only a spilled small vector holds an allocation.
impl<A: smallvec::Array> Charged for smallvec::SmallVec<A> {
    #[inline]
    fn charged_bytes(&self) -> u64 {
        if !self.spilled() { return 0; }
        (self.capacity() as u64).saturating_mul(std::mem::size_of::<A::Item>() as u64)
    }
}

impl<A: smallvec::Array> Scratch for smallvec::SmallVec<A> {
    #[inline]
    fn release(&mut self) {
        *self = smallvec::SmallVec::new();
    }
}

/// A vector of buffers counted and released as one: its spine and every inner
/// buffer. Visit a `Vec<Vec<_>>` through this, or only the spine is counted.
pub(crate) struct Nested<'a, B>(pub(crate) &'a mut Vec<B>);

impl<B: Charged> Charged for Nested<'_, B> {
    fn charged_bytes(&self) -> u64 {
        self.0.iter().fold(self.0.charged_bytes(), |bytes, inner| bytes.saturating_add(inner.charged_bytes()))
    }
}

impl<B: Charged> Scratch for Nested<'_, B> {
    fn release(&mut self) {
        *self.0 = Vec::new();
    }
}

/// Drop `buf`'s allocation, leaving it empty, if what it retains exceeds
/// [`SCRATCH_RETAIN_BYTES`], and give the freed bytes back to `lim`.
///
/// The rule a parked working set's buffers are kept by: [`Buffers::release_oversized`]
/// applies it to each buffer a working set lists.
#[inline]
pub(crate) fn release_if_oversized<B: Scratch + ?Sized>(lim: &Limits, buf: &mut B) {
    let bytes = buf.charged_bytes();
    if bytes > SCRATCH_RETAIN_BYTES as u64 {
        buf.release();
        lim.release_bytes(bytes);
    }
}

#[cfg(test)]
#[path = "tests/pool.rs"]
mod tests;
