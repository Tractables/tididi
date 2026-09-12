//! Host memory probes and the address-space headroom derived from them.


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

/// What the apply engine needs from its host to stay inside a memory ceiling:
/// the mapped high-water bytes the ceiling is charged against, the
/// address-space ceiling itself, a release notice before a large allocation,
/// and a once-per-apply eager-reclaim nudge. Plain `fn` pointers — the
/// growth path pays a load and an indirect call, nothing more. The default is
/// every probe a no-op: no ceiling, no pressure, plain doubling growth.
/// Installed through [`LimitSet::mem_pressure`](crate::limits::LimitSet::mem_pressure).
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct MemPressure {
    /// Called with the byte size of a growth allocation about to be made, so
    /// the host can release reclaimable memory before the kernel charges it.
    pub preflight_alloc: fn(u64),
    /// Mapped (plus retained) high-water bytes, the figure an address-space
    /// limit charges against.
    pub mapped_bytes: fn() -> u64,
    /// The address-space ceiling in bytes, or `None` when unlimited. Consulted
    /// once per install and cached.
    pub address_space_limit: fn() -> Option<u64>,
    /// Called once per top-level apply; the host may switch to eager
    /// reclamation when mapped bytes near the ceiling.
    pub eager_reclaim: fn(),
}

impl MemPressure {
    /// Every probe a no-op.
    pub const NONE: MemPressure = MemPressure {
        preflight_alloc: |_| {},
        mapped_bytes: || 0,
        address_space_limit: || None,
        eager_reclaim: || {},
    };
}

impl Default for MemPressure {
    fn default() -> Self {
        Self::NONE
    }
}
