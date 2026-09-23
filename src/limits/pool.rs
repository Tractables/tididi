//! Reusable scratch values owned by an engine.
//!
//! A checkout takes ownership, allowing nested operations to obtain fresh
//! scratch from the same pool. Completed checkouts return their buffers;
//! unwinding discards them.

use std::cell::Cell;

use super::Limits;

/// Maximum retained scratch capacity between operations. Larger buffers are
/// released so an unusually large operation does not permanently retain them.
/// Sized so a repeated wide conjunction keeps the buffers it fills — a
/// smaller cap re-grows them, and pays their page faults, on every call.
/// Level arenas have a separate limit, `diagram::MAX_LEVEL_ARENA_BYTES`.
pub(crate) const SCRATCH_RETAIN_BYTES: usize = 128 * 1024 * 1024;

/// A scratch value parked between operations, absent while checked out.
///
/// A nested checkout finds an empty pool and creates a fresh working set.
pub(crate) struct Pool<T>(Cell<Option<T>>);

impl<T> Default for Pool<T> {
    fn default() -> Self {
        Pool(Cell::new(None))
    }
}

impl<T: Default> Pool<T> {
    /// Take the parked value, creating an empty one if the pool is vacant.
    #[inline]
    pub(crate) fn take(&self) -> T {
        self.0.take().unwrap_or_default()
    }
}

impl<T> Pool<T> {
    /// Drop whatever this pool retains.
    #[inline]
    pub(crate) fn drain(&self) {
        self.0.take();
    }

    /// Park `value`, replacing whatever is there.
    #[inline]
    pub(crate) fn put(&self, value: T) {
        self.0.set(Some(value));
    }
}

impl<T> Pool<Vec<T>> {
    /// Park `v`, dropping its allocation first if it is oversized; see
    /// [`release_if_oversized`].
    #[inline]
    pub(crate) fn put_bounded(&self, lim: &Limits, mut v: Vec<T>) {
        release_if_oversized(lim, &mut v);
        self.put(v);
    }
}

/// Checkout preparation and capacity retention for an engine-owned working set.
pub(crate) trait PooledScratch: Default {
    /// Invalidate previous results before the working set is used again.
    fn prepare(&mut self);
    /// Release allocations that exceed this working set's retention policy,
    /// giving the freed bytes back to `lim`.
    fn retain(&mut self, lim: &Limits);
}

impl<T> PooledScratch for Vec<T> {
    #[inline]
    fn prepare(&mut self) { self.clear(); }
    #[inline]
    fn retain(&mut self, lim: &Limits) { release_if_oversized(lim, self); }
}

impl<T: PooledScratch> Pool<T> {
    /// Check out scratch, returning it automatically on an ordinary exit.
    ///
    /// `lim` is held for the return: the retention policy runs when the guard
    /// drops, and whatever it frees is given back to the byte meter there.
    #[inline]
    pub(crate) fn checkout<'a>(&'a self, lim: &'a Limits) -> PoolGuard<'a, T> {
        let mut guard = self.checkout_preserving(lim);
        guard.prepare();
        guard
    }

    /// Keep initialized entries for algorithms that overwrite their live range.
    /// The caller must invalidate stale results before reading them.
    #[inline]
    pub(crate) fn checkout_preserving<'a>(&'a self, lim: &'a Limits) -> PoolGuard<'a, T> {
        PoolGuard { pool: self, lim, value: self.take() }
    }
}

/// Own checked-out scratch without borrowing the pool's contents.
///
/// Unwinding discards partially updated scratch; ordinary exits apply its
/// retention policy and park it for the next checkout, including nested uses.
pub(crate) struct PoolGuard<'a, T: PooledScratch> {
    pool: &'a Pool<T>,
    lim: &'a Limits,
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
            self.value.retain(self.lim);
            self.pool.put(std::mem::take(&mut self.value));
        }
    }
}

/// A scratch buffer that can report what it retains and give it back.
pub(crate) trait Scratch {
    /// Entries of capacity this buffer holds.
    fn entries(&self) -> usize;
    /// Bytes one entry occupies.
    fn entry_bytes(&self) -> usize;
    /// Give the allocation back, leaving an empty buffer.
    fn release(&mut self);
    /// Empty the buffer, keeping the allocation.
    fn clear(&mut self);

    /// Bytes of capacity this buffer holds.
    #[inline]
    fn retained_bytes(&self) -> usize {
        self.entries().saturating_mul(self.entry_bytes())
    }
}

impl<T> Scratch for Vec<T> {
    #[inline]
    fn entries(&self) -> usize {
        self.capacity()
    }
    #[inline]
    fn entry_bytes(&self) -> usize {
        std::mem::size_of::<T>()
    }
    #[inline]
    fn release(&mut self) {
        *self = Vec::new();
    }
    #[inline]
    fn clear(&mut self) {
        Vec::clear(self);
    }
}

impl<T, S: Default> Scratch for std::collections::HashSet<T, S> {
    #[inline]
    fn entries(&self) -> usize {
        self.capacity()
    }
    #[inline]
    fn entry_bytes(&self) -> usize {
        std::mem::size_of::<T>()
    }
    #[inline]
    fn release(&mut self) {
        *self = Self::default();
    }
    #[inline]
    fn clear(&mut self) {
        Self::clear(self);
    }
}

impl<K, V, S: Default> Scratch for std::collections::HashMap<K, V, S> {
    #[inline]
    fn entries(&self) -> usize {
        self.capacity()
    }
    #[inline]
    fn entry_bytes(&self) -> usize {
        std::mem::size_of::<(K, V)>()
    }
    #[inline]
    fn release(&mut self) {
        *self = Self::default();
    }
    #[inline]
    fn clear(&mut self) {
        Self::clear(self);
    }
}

/// Drop `buf`'s allocation, leaving it empty, if what it retains exceeds
/// [`SCRATCH_RETAIN_BYTES`].
///
/// The single implementation of the scratch-retention rule.
/// [`Pool::put_bounded`] applies it to a pooled buffer; scratch held as struct
/// fields, which cannot round-trip through a pool per buffer, applies it field
/// by field.
#[inline]
pub(crate) fn release_if_oversized<B: Scratch + ?Sized>(lim: &Limits, buf: &mut B) {
    let bytes = buf.retained_bytes();
    if bytes > SCRATCH_RETAIN_BYTES {
        buf.release();
        lim.release_bytes(bytes as u64);
    }
}

/// Release a scratch buffer once its contents have been read for the last time:
/// over `max_entries` of capacity the allocation goes back, otherwise the buffer
/// is emptied and stays warm for the next call.
///
/// The difference from [`release_if_oversized`] is who empties the buffer. A
/// pooled buffer is handed back already spent, so parking it is enough; a
/// buffer held as a struct field is read in place and has to be emptied here,
/// or the next call sees the last one's contents. Clearing a buffer that was
/// just released is a no-op, which is why both cases are one call.
#[inline]
pub(crate) fn release_or_clear<B: Scratch + ?Sized>(lim: &Limits, buf: &mut B, max_entries: usize) {
    if buf.entries() > max_entries {
        let bytes = buf.retained_bytes();
        buf.release();
        lim.release_bytes(bytes as u64);
    } else {
        buf.clear();
    }
}
