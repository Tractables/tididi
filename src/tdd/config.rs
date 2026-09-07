//! Installed runtime tuning knobs: decouples the TDD library from the process
//! environment (public-release P2c).
//!
//! Historically each of these knobs was a `TIDIDI_*` environment variable read
//! (and memoized) at its point of use inside `tdd`. For the `tididi`
//! extraction the library must read **no** environment variables — configuration
//! arrives as data. This module holds a single [`TddConfig`] table of resolved
//! values installed once, at binary startup, by the downstream compile driver (which is
//! the ONE place that reads the `TIDIDI_*` names). Pure library use (tests,
//! embedders) that never installs sees [`TddConfig::production_default`] — the
//! exact behavior you got with every variable unset, so defaults are unchanged.
//!
//! The read path is a single relaxed atomic load (`OnceLock::get`) plus a field
//! read — the same shape as the per-site `OnceLock<bool>` memos this replaced.
//!
//! Knobs that were pure write-only diagnostics (`TIDIDI_PROBE_LVL`,
//! `TIDIDI_PROJECT_SIZE_TRACE`, `DEAD_MARK_APPLY_TIMING`,
//! `TIDIDI_MARG_CLUSTER_DEBUG`) were deleted with their env triggers rather than
//! carried here ("instrumentation dies with its trigger"). The public-release
//! P4b cut removed every other research/diagnostic seam this table used to
//! carry (sparse-kernel tuning, scatter direction/chunking, reduce order,
//! contract strategy, weighted-inline representation, marginalization
//! shrink/Sethi-Ullman ordering, count normalization, informed recovery hints,
//! (P) fusion kill-switch) — each was hardcoded to its production default or
//! deleted outright, leaving only the correctness checkers below.

use std::sync::OnceLock;

/// Resolved runtime tuning knobs. Every field's [`production_default`] value is
/// the behavior obtained with the corresponding `TIDIDI_*` variable unset.
///
/// [`production_default`]: TddConfig::production_default
#[derive(Clone, Copy, Debug)]
pub struct TddConfig {
    // ── correctness checkers (kept as debug capabilities) ──
    /// `TIDIDI_MARG_CANON_CHECK`: joint-fixpoint canonical-invariant assertion.
    pub marg_canon_check: bool,
    /// `TIDIDI_MARG_GAUGE_AUDIT`: projective (ray) gauge-redundancy audit report.
    pub marg_gauge_audit: bool,
    /// `TIDIDI_MARG_MC_CHECK`: per-rewrite model-count-preservation snapshots.
    pub marg_mc_check: bool,
    /// `TIDIDI_MARG_SLOT_CHECK`: latched first-violation marg-slot reporter.
    pub marg_slot_check: bool,
}

impl TddConfig {
    /// The all-knobs-unset production defaults — byte-identical to the behavior
    /// before P2c moved the env reads driver-side.
    pub const fn production_default() -> Self {
        TddConfig {
            marg_canon_check: false,
            marg_gauge_audit: false,
            marg_mc_check: false,
            marg_slot_check: false,
        }
    }
}

impl Default for TddConfig {
    fn default() -> Self {
        Self::production_default()
    }
}

static DEFAULT: TddConfig = TddConfig::production_default();
static INSTALLED: OnceLock<TddConfig> = OnceLock::new();

/// Install the resolved tuning table. Call once, at binary startup, before any
/// compile. Panics on a second install: the table is process-global and must not
/// change mid-run. Pure library use that never installs sees
/// [`TddConfig::production_default`].
///
/// # Panics
///
/// Panics if called more than once (the tuning table is process-global and is
/// installed exactly once at startup).
pub fn install(tuning: TddConfig) {
    if INSTALLED.set(tuning).is_err() {
        panic!(
            "tdd::config::install called twice; the tuning table is process-global \
             and installed exactly once at startup"
        );
    }
}

/// The active tuning table: the installed one, or the production defaults when
/// nothing was installed. A single relaxed atomic load.
#[inline]
pub(crate) fn get() -> &'static TddConfig {
    INSTALLED.get().unwrap_or(&DEFAULT)
}
