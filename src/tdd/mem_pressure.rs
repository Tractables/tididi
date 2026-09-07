//! Installed memory-pressure interface: decouples the TDD apply hot path from
//! the jemalloc-backed `crate::mem` implementation.
//!
//! `tdd` reads memory pressure through a small table of plain `fn` pointers
//! ([`MemPressureHooks`]) installed once at binary startup. With no probe
//! installed (pure library use) the accessors return the "no budget" defaults:
//! no VAS pressure (`mapped_bytes` = 0), unlimited headroom
//! (`rlimit_as_bytes` = `None`), and no jemalloc purge (the decay hooks are
//! no-ops). That default is the intended library contract — plain doubling
//! growth, no OS-memory machinery. The compiler installs the jemalloc-backed
//! hooks in `driver::run`, restoring byte-identical behavior for the binary.
//!
//! `fn` pointers, NOT `dyn Trait`: the accessors are `#[inline(always)]` and the
//! `OnceLock::get()` is a single relaxed atomic load, so the hot-path shape is a
//! load + indirect call when installed, or the inlined default constant when not.

use std::sync::OnceLock;

/// Installable table of memory-pressure probes. All fields are plain `fn`
/// pointers into the downstream memory-pressure implementation (jemalloc-backed) — never
/// `dyn`, so the apply hot path pays only a load + indirect call.
#[derive(Clone, Copy)]
pub struct MemPressureHooks {
    /// Preflight a pending allocation of `request_bytes`: flip jemalloc from
    /// lazy to eager decay and purge dirty pages before the kernel charges the
    /// request against `RLIMIT_AS`. Default: no-op (no purge).
    pub preflight_alloc_decay: fn(u64),
    /// Current mapped+retained high-water bytes (what `RLIMIT_AS` charges
    /// against). Default: `0` (= no VAS pressure).
    pub mapped_bytes: fn() -> u64,
    /// The `RLIMIT_AS` ceiling in bytes, or `None` if unlimited. Default: `None`
    /// (= unlimited headroom).
    pub rlimit_as_bytes: fn() -> Option<u64>,
    /// Once-per-top-level-apply hook that engages eager jemalloc decay when
    /// mapped+retained nears the ceiling. Default: no-op.
    pub maybe_engage_eager_decay: fn(),
}

static PROBE: OnceLock<MemPressureHooks> = OnceLock::new();

/// Install the memory-pressure probes. Call once, at binary startup, before any
/// compilation. Panics on a second install: the probe table is process-global
/// and must not change mid-run.
///
/// # Panics
///
/// Panics if called more than once (the probe table is process-global and is
/// installed exactly once at startup).
pub fn install(probe: MemPressureHooks) {
    if PROBE.set(probe).is_err() {
        panic!(
            "tdd::mem_pressure::install called twice; the MemPressureHooks table is \
             process-global and installed exactly once at startup"
        );
    }
}

/// Preflight-decay before a growth allocation. No-op when no probe is installed.
#[inline(always)]
pub(crate) fn preflight_alloc_decay(request_bytes: u64) {
    if let Some(p) = PROBE.get() {
        (p.preflight_alloc_decay)(request_bytes);
    }
}

/// Mapped+retained high-water bytes. `0` (no VAS pressure) when uninstalled.
#[inline(always)]
pub(crate) fn mapped_bytes() -> u64 {
    match PROBE.get() {
        Some(p) => (p.mapped_bytes)(),
        None => 0,
    }
}

/// `RLIMIT_AS` ceiling. `None` (unlimited headroom) when uninstalled.
#[inline(always)]
pub(crate) fn rlimit_as_bytes() -> Option<u64> {
    match PROBE.get() {
        Some(p) => (p.rlimit_as_bytes)(),
        None => None,
    }
}

/// Once-per-apply eager-decay hook. No-op when uninstalled.
#[inline(always)]
pub(crate) fn maybe_engage_eager_decay() {
    if let Some(p) = PROBE.get() {
        (p.maybe_engage_eager_decay)();
    }
}
