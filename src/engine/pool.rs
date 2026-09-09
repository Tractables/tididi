//! One parked scratch buffer, and the rule for how much of it to keep.
//!
//! Every operation in the crate reuses buffers across calls, and they all park
//! them the same way: a [`Cell`] the operation empties on entry and refills on
//! exit. Taking rather than borrowing is what lets a pooled buffer be held
//! across a recursive call into the same pool's owner — the borrow checker
//! would refuse the second borrow, and a `RefCell` would panic on it.
//!
//! The pools themselves hang off the [`Engine`](crate::engine::Engine), so two
//! engines never share a buffer and dropping one frees everything it warmed up.

use std::cell::Cell;

/// A parked scratch value.
///
/// `take` empties the pool and hands the value over; `put` parks it again.
/// A pool that has been taken from and not yet returned to holds
/// `T::default()`, which for the `Vec`s and maps this is used with is an empty
/// buffer — so a re-entrant taker gets a fresh one rather than a stale view of
/// the buffer above it on the stack.
pub(crate) struct Pool<T>(Cell<T>);

impl<T: Default> Default for Pool<T> {
    fn default() -> Self {
        Pool(Cell::new(T::default()))
    }
}

impl<T: Default> Pool<T> {
    /// Take the parked value, leaving an empty one behind.
    #[inline]
    pub(crate) fn take(&self) -> T {
        self.0.take()
    }

    /// Drop whatever this pool retains.
    #[inline]
    pub(crate) fn drain(&self) {
        self.0.take();
    }
}

impl<T> Pool<T> {
    /// Park `value`, replacing whatever is there.
    #[inline]
    pub(crate) fn put(&self, value: T) {
        self.0.set(value);
    }
}

impl<T> Pool<Vec<T>> {
    /// Park `v`, dropping its allocation first if it is oversized — see
    /// [`release_if_oversized`].
    #[inline]
    pub(crate) fn put_bounded(&self, mut v: Vec<T>, max_bytes: usize) {
        crate::engine::pool::release_if_oversized(&mut v, max_bytes);
        self.0.set(v);
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

/// Drop `buf`'s allocation (leaving it empty) if what it retains exceeds
/// `max_bytes`. Keeps small buffers warm while preventing rare large calls from
/// leaking GiB-scale scratch into RSS.
///
/// The single implementation of the scratch-retention rule.
/// [`Pool::put_bounded`] applies it to a pooled buffer; scratch held as struct
/// fields, which cannot round-trip through a pool per buffer, applies it field
/// by field.
///
/// Companion to the per-arena cap in `reset_level`
/// (`diagram::MAX_LEVEL_ARENA_BYTES`): scratch pools are not part of any
/// diagram's retained-capacity accounting, so they cannot trip the caller's
/// step budget, but they DO inflate real RSS — invisible to the soft apply
/// budget's predictive checks.
#[inline]
pub(crate) fn release_if_oversized<B: Scratch + ?Sized>(buf: &mut B, max_bytes: usize) {
    if buf.retained_bytes() > max_bytes {
        buf.release();
    }
}

/// Release a scratch buffer at its LAST read: over `max_entries` of capacity
/// the allocation goes back, otherwise the buffer is emptied and stays warm for
/// the next call.
///
/// The difference from [`release_if_oversized`] is who empties the buffer. A
/// pooled buffer is handed back already spent, so parking it is enough; a
/// buffer held as a struct field is read in place and has to be emptied here,
/// or the next call sees the last one's contents. Clearing a buffer that was
/// just released is a no-op, which is why both cases are one call.
#[inline]
pub(crate) fn release_or_clear<B: Scratch + ?Sized>(buf: &mut B, max_entries: usize) {
    if buf.entries() > max_entries {
        buf.release();
    } else {
        buf.clear();
    }
}
