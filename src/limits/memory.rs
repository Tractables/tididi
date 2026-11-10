//! Host memory hooks and the address-space headroom derived from them.


/// Generous finite headroom returned by [`Limits::headroom`](super::Limits::headroom) when
/// `RLIMIT_AS` is unlimited. With no address-space ceiling there is
/// nothing for the emit's `Vec`-doubling transient to trip, so plain doubling
/// is unconditionally safe; 1 TiB dwarfs any real reservation while staying
/// finite so the u128 `3 × bound × pair_bytes < h` comparison never overflows.
pub(crate) const VAS_UNLIMITED_HEADROOM: u64 = 1 << 40; // 1 TiB

/// Address space held back below `RLIMIT_AS` by the headroom
/// [`Limits::headroom`](super::Limits::headroom) derives when no soft budget is
/// armed, so the fallible reserves refuse before an infallible allocation
/// elsewhere in the process lands on a full address space and aborts.
pub(crate) const SOFT_HEADROOM_MARGIN_BYTES: u64 = 1536 * 1024 * 1024; // 1.5 GiB

/// `limit − SOFT_HEADROOM_MARGIN_BYTES − mapped`, saturating to zero.
#[inline]
pub(crate) fn vas_headroom_with_margin(limit: u64, mapped: u64) -> u64 {
    limit
        .saturating_sub(SOFT_HEADROOM_MARGIN_BYTES)
        .saturating_sub(mapped)
}

/// The share of a reading's room, as a shift, that the bytes charged since
/// may reach before [`Limits::headroom_short_of`](super::Limits::headroom_short_of)
/// reads the host again: an eighth.
const ROOM_READING_SPAN_SHIFT: u32 = 3;

/// An address-space headroom read from the host, with the in-flight meter
/// and the bytes released when it was read.
#[derive(Clone, Copy, Debug)]
pub(super) struct RoomReading {
    headroom: u64,
    in_flight: u64,
    released: u64,
}

impl super::Limits {
    /// Available bytes under the soft budget, or the host's address-space
    /// ceiling minus its mapped bytes and a safety margin. An unlimited host
    /// ceiling yields [`VAS_UNLIMITED_HEADROOM`].
    #[inline]
    pub(crate) fn headroom(&self) -> u64 {
        if let Some(h) = self.budget_headroom() {
            return h;
        }
        match self.address_space_limit() {
            Some(limit) => {
                let mem = self.memory_hooks.borrow().clone();
                let headroom = vas_headroom_with_margin(limit, mem.mapped_bytes());
                self.room_reading.set(Some(RoomReading {
                    headroom,
                    in_flight: self.in_flight_bytes.get(),
                    released: self.released_bytes.get(),
                }));
                headroom
            },
            None => VAS_UNLIMITED_HEADROOM,
        }
    }

    /// `None` when `bytes` fit in the [`headroom`](Self::headroom), and
    /// otherwise the headroom, read afresh.
    ///
    /// Under the address-space ceiling, the room the operation's last read
    /// of the host's mapped bytes left, less every byte charged since,
    /// answers that `bytes` fit when it covers them and those charged bytes
    /// are at most an eighth of that room; otherwise the host is read. A
    /// charged byte maps at most one byte, and one freed and charged again
    /// counts twice, so the room it answers from is never more than a read
    /// would find, but for what was mapped since without a charge. A
    /// headroom returned is always a read, so whatever a caller refuses or
    /// bounds on it is decided as it was on a read before every call.
    #[inline]
    pub(crate) fn headroom_short_of(&self, bytes: u64) -> Option<u64> {
        if !self.budget_armed.get()
            && let Some(seen) = self.room_reading.get()
        {
            let released = self.released_bytes.get().wrapping_sub(seen.released);
            let charged = self.in_flight_bytes.get().saturating_add(released).saturating_sub(seen.in_flight);
            if charged <= seen.headroom >> ROOM_READING_SPAN_SHIFT && bytes <= seen.headroom - charged {
                return None;
            }
        }
        let headroom = self.headroom();
        (bytes > headroom).then_some(headroom)
    }

    /// The installed address-space ceiling, answered once per install.
    fn address_space_limit(&self) -> Option<u64> {
        match self.vas_limit.get() {
            Some(v) => v,
            None => {
                let mem = self.memory_hooks.borrow().clone();
                let v = mem.address_space_limit();
                self.vas_limit.set(Some(v));
                v
            }
        }
    }
}

/// Owned host memory hooks installed through [`LimitConfig::with_memory_hooks`](crate::limits::LimitConfig::with_memory_hooks).
#[derive(Clone, Default)]
pub struct MemoryHooks(Option<std::sync::Arc<dyn MemoryObserver>>);

impl std::fmt::Debug for MemoryHooks {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemoryHooks").field("installed", &self.0.is_some()).finish()
    }
}

impl MemoryHooks {
    /// No memory ceiling or reclamation callbacks.
    pub const NONE: Self = Self(None);

    /// Own allocation, mapped-byte, address-space and reclamation callbacks, in that order.
    ///
    /// The address-space ceiling is cached until the next limits installation;
    /// the other callbacks run when growth or a new conjunction asks for them.
    /// Within one operation, a conjunction level that fits in the room the
    /// last mapped-bytes reading left, less the bytes the operation charged
    /// since, is priced without another reading while those bytes stay under
    /// an eighth of that room: memory mapped meanwhile outside the
    /// operation's charges is seen at the next reading.
    /// Captured state must support transfer between threads.
    pub fn new(
        preflight: impl Fn(u64) + Send + Sync + 'static,
        mapped: impl Fn() -> u64 + Send + Sync + 'static,
        ceiling: impl Fn() -> Option<u64> + Send + Sync + 'static,
        reclaim: impl Fn() + Send + Sync + 'static,
    ) -> Self {
        Self(Some(std::sync::Arc::new(Callbacks { preflight, mapped, ceiling, reclaim })))
    }

    /// Notify the host before an allocation.
    pub(crate) fn preflight_alloc(&self, bytes: u64) {
        if let Some(hooks) = &self.0 { hooks.preflight_alloc(bytes); }
    }

    /// Whether no hooks are installed.
    pub(crate) fn is_none(&self) -> bool {
        self.0.is_none()
    }

    /// The host's mapped and retained bytes, or zero with no observer.
    pub(crate) fn mapped_bytes(&self) -> u64 {
        self.0.as_ref().map_or(0, |hooks| hooks.mapped_bytes())
    }

    /// The host's address-space ceiling, if bounded.
    pub(crate) fn address_space_limit(&self) -> Option<u64> {
        self.0.as_ref().and_then(|hooks| hooks.address_space_limit())
    }

    /// Ask the host to reclaim at the start of a conjunction.
    pub(crate) fn eager_reclaim(&self) {
        if let Some(hooks) = &self.0 { hooks.eager_reclaim(); }
    }
}

/// The four host memory observations made through one owned context.
trait MemoryObserver: Send + Sync {
    /// Notify the host before an allocation.
    fn preflight_alloc(&self, bytes: u64);
    /// Read mapped and retained bytes.
    fn mapped_bytes(&self) -> u64;
    /// Read the address-space ceiling.
    fn address_space_limit(&self) -> Option<u64>;
    /// Reclaim before a conjunction.
    fn eager_reclaim(&self);
}

/// Captured callables sharing one allocation and lifetime.
struct Callbacks<A, B, C, D> {
    preflight: A,
    mapped: B,
    ceiling: C,
    reclaim: D,
}

impl<A: Fn(u64) + Send + Sync, B: Fn() -> u64 + Send + Sync, C: Fn() -> Option<u64> + Send + Sync, D: Fn() + Send + Sync> MemoryObserver for Callbacks<A, B, C, D> {
    fn preflight_alloc(&self, bytes: u64) { (self.preflight)(bytes); }
    fn mapped_bytes(&self) -> u64 { (self.mapped)() }
    fn address_space_limit(&self) -> Option<u64> { (self.ceiling)() }
    fn eager_reclaim(&self) { (self.reclaim)(); }
}
