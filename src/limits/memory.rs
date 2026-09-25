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
