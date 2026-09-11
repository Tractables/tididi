//! Host memory probes and the address-space headroom derived from them.


/// Generous finite headroom returned by [`Limits::headroom`] when
/// `RLIMIT_AS` is unlimited. With no address-space ceiling there is
/// nothing for the emit's `Vec`-doubling transient to trip, so plain doubling
/// is unconditionally safe; 1 TiB dwarfs any real reservation while staying
/// finite so the u128 `3 × bound × pair_bytes < h` comparison never overflows.
pub(crate) const VAS_UNLIMITED_HEADROOM: u64 = 1 << 40; // 1 TiB

/// Safety margin of address space the *guarded* apply path refuses to consume,
/// subtracted from the `RLIMIT_AS − mapped` headroom [`Limits::headroom`]
/// derives when no soft budget is armed.
///
/// **Abort class it protects against.** Rust's infallible allocations abort the
/// process on failure (`memory allocation of N bytes failed`, then an abort signal) via a
/// `#[rustc_nounwind]` handler — the panic cannot unwind, so the handled-OOM →
/// Shannon-recovery cascade never runs. Our fallible reserves
/// (`Limits::reserve*`) and the dense-precount gates surface `OverBudget`
/// cleanly, but if they let the process consume address space right up to
/// `RLIMIT_AS`, any moderate *unguarded* transient — a count-walk level vec, a
/// projection row buffer, a recovery child's raw alloc — lands on a full
/// address space and aborts uncatchably — which is the common way a compile
/// under a tight ceiling dies. Holding this much room
/// back below the ceiling keeps the guarded path from ever reaching the wall,
/// so those transients have somewhere to land and the *handled* failure fires
/// first (recovery gets its chance).
///
/// The margin covers several of the small unguarded transients that abort this
/// way. It is not sized to cover a large guarded apply transient; the budget
/// and precount gates already handle those.
pub(crate) const SOFT_HEADROOM_MARGIN_BYTES: u64 = 1536 * 1024 * 1024; // 1.5 GiB

/// The no-soft-budget arithmetic behind [`Limits::headroom`] (factored out for
/// unit tests): `limit − SOFT_HEADROOM_MARGIN_BYTES − mapped`, saturating.
///
/// Holds [`SOFT_HEADROOM_MARGIN_BYTES`] back below the `RLIMIT_AS` ceiling so the
/// guarded path refuses the last margin of address space — see that constant's
/// doc for the uncatchable-abort class this prevents. Saturating: a small or
/// already-exhausted address space just reports zero headroom (⇒ most bounded
/// growth), never wraps.
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
