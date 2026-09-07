//! External side-table of per-node semiring (weighted) marginal values for
//! `--weighted` algebraic model counting (MCC Track 4 PWMC/WMC, and the Track 5B
//! complex primitives later).
//!
//! Why a side-table and not a `TddLevel` field: `TddLevel` is at its 136 B size
//! budget (a static assert guards it), so a `Vec<WeightVal>` field would
//! overflow it and perturb the hot sequential-scan stride of the integer `--mc`
//! path. Instead a level is flagged `TddLevel::MARG_WEIGHTED` and its values
//! live here, indexed by vtree level. The integer marginalization store
//! (`marginal_counts` / `marginal_counts_big`) is untouched and stays `None` in
//! weighted mode — the two are mutually exclusive within one compile.
//!
//! The weighted cascade reuses the SAME structural marginalization machinery as
//! the integer path (scheduling, cascade order, parent-ref remap, dedup); only
//! the per-node payload differs — a [`WeightVal`] (exact `BigRational`, or in
//! log mode a bounded-precision `SignedLog`) instead of a `u128`/`BigUint`
//! model count. Leaf base values and the fold arithmetic come from
//! [`RationalSemiring`] (always parsed exactly), converted once per leaf read
//! to the active mode by [`WeightStore::leaf_val`].

use std::sync::atomic::{AtomicI8, Ordering};

use rustc_hash::FxHashMap;

use crate::tdd::query::semiring::{weight_key, RationalSemiring, Semiring, SignedLog, WeightKey, WeightVal};
use crate::tdd::types::{LeafLabel, MARG_INLINE_MAX};
use crate::vtree::VarId;

/// Driver-set per-instance default for log mode, resolved by problem track and
/// read by every [`WeightStore::new`] (6 deep call sites with no `config`
/// access). `-1` = unset (fall back to exact), `0` = exact, `1` = log.
/// Single-threaded (`TiDiDi` targets one CPU per CNF), so a process global is the
/// same shape as the existing `WEIGHT_CTX` thread-local.
static LOG_MODE_DECISION: AtomicI8 = AtomicI8::new(-1);

/// Driver hook: set the per-track default precision. WMC (Track 2, 1%
/// tolerance) → `true` (bounded log, solves the arithmetic-precision-bound
/// timeouts); PWMC (Track 4, exact required) → `false`.
///
/// Called from exactly ONE place — `driver::arm_weighted_precision_domain`, run
/// before every route that can answer a weighted instance (the PWMC dispatch,
/// the DPLL-canopy stage, the weighted scalar-DVE early exit, the marginalizing
/// cascade). It used to be called from inside two of those routes, which left
/// the ones that answer earlier reading [`resolve_log_mode`]'s "nobody decided"
/// fallback instead of the track's decision. Keep it a single caller: a second
/// one is a second answer to the same question.
#[doc(hidden)]
pub fn set_weighted_log_default(log: bool) {
    LOG_MODE_DECISION.store(if log { 1 } else { 0 }, Ordering::Relaxed);
    // Announce the domain from the ONE place that decides it. Which domain a
    // weighted run is in changes which value-plumbing optimizations are even
    // reachable (every `weight_key`-equality rewrite is exact-domain only), so
    // a probe log that does not say the domain cannot be attributed to a
    // change — a lesson this cost real box-hours to learn. Unconditional and
    // driver-side: it fires once per weighted compile, never on integer runs.
    eprintln!(
        "c weighted precision domain: {}",
        if log { "log (bounded)" } else { "exact rational" }
    );
}

/// Resolve the active log-mode flag: the driver-set per-track default, or
/// exact (the safe, byte-identical default for the full-diagram oracle path
/// that never calls the setter).
#[doc(hidden)]
pub fn resolve_log_mode() -> bool {
    LOG_MODE_DECISION.load(Ordering::Relaxed) == 1
}

/// Per-level weighted marginal values. `per_level[vtree_idx]` is `Some(vals)`
/// once that level is weight-marginalized; `vals[slot]` is the semiring value of
/// the node occupying that marginal slot (post-dedup slot index, the same index
/// the level's marg-side pair refs point at).
pub struct WeightStore {
    per_level: Vec<Option<Vec<WeightVal>>>,
    /// Global interned-value table for the weighted INLINE marg-ref optimization
    /// (`TIDIDI_WEIGHTED_INLINE`). A marg-side `MargRef::Inline(gidx)` in weighted
    /// mode indexes `interned[gidx]` (NOT an integer count). Equal values map to one
    /// gidx, so equal-value marg children produce equal inline pair fields → they
    /// collapse at emit and let their parents merge — the integer-inline cascade,
    /// which the per-node `Slot` form blocks. `intern_map` is the value→gidx dedup.
    interned: Vec<WeightVal>,
    intern_map: FxHashMap<WeightKey, u32>,
    /// Leaf base weights + the exact parsed-weight source of truth. Folds in the
    /// weighted path go through [`WeightVal`]; leaf reads convert via [`WeightStore::leaf_val`].
    pub semiring: RationalSemiring,
    /// Bounded-precision log mode (see [`resolve_log_mode`]). Read once at
    /// construction. When false the store is byte-identical to the exact path.
    #[doc(hidden)]
    pub log_mode: bool,
}

impl WeightStore {
    /// `num_levels` = `vtree.num_nodes()`; every level starts unmarginalized.
    pub fn new(num_levels: usize, semiring: RationalSemiring) -> Self {
        let log_mode = resolve_log_mode();
        Self {
            per_level: vec![None; num_levels],
            interned: Vec::new(),
            intern_map: FxHashMap::default(),
            semiring,
            log_mode,
        }
    }

    /// The additive identity in the active mode.
    #[inline]
    #[doc(hidden)]
    pub fn wzero(&self) -> WeightVal {
        if self.log_mode {
            WeightVal::Log(SignedLog::zero())
        } else {
            // The canonical exact zero: 0 always fits the small representation.
            WeightVal::ExactSmall(0)
        }
    }

    /// Leaf base value for `(var, label)` in the active mode. The exact parsed
    /// weight from [`RationalSemiring`] is authoritative; in log mode it is
    /// converted to `SignedLog` exactly once here (per leaf read).
    #[inline]
    pub fn leaf_val(&self, var: VarId, label: LeafLabel) -> WeightVal {
        let r = self.semiring.leaf(var, label);
        if self.log_mode {
            WeightVal::Log(SignedLog::from_rational(&r))
        } else {
            WeightVal::exact(r)
        }
    }

    /// Intern a weighted value into the global table, returning its `gidx` for a
    /// `MargRef::Inline(gidx)` ref. Returns `None` if the table is full (gidx would
    /// exceed `MARG_INLINE_MAX` = 2^30-1, the 30-bit pair-field budget) — caller
    /// then falls back to a per-level `Slot`. Equal values dedup to one gidx.
    #[doc(hidden)]
    pub fn intern(&mut self, val: WeightVal) -> Option<u32> {
        let key = weight_key(&val);
        if let Some(&g) = self.intern_map.get(&key) {
            return Some(g);
        }
        let g = self.interned.len();
        if g > MARG_INLINE_MAX as usize {
            return None;
        }
        let g = g as u32;
        self.interned.push(val);
        self.intern_map.insert(key, g);
        Some(g)
    }

    /// Resolve a `gidx` from a weighted `MargRef::Inline(gidx)` to its value.
    #[inline]
    #[doc(hidden)]
    pub fn interned_value(&self, gidx: u32) -> &WeightVal {
        &self.interned[gidx as usize]
    }

    /// Store the computed weighted values for a level (called at the point the
    /// integer path would `make_marginal`). Auto-grows the table: a marginalizing
    /// compile can restructure the vtree (v-split adds nodes), so `level` may
    /// exceed the `num_levels` seen at construction.
    #[doc(hidden)]
    pub fn set_level(&mut self, level: usize, vals: Vec<WeightVal>) {
        if level >= self.per_level.len() {
            self.per_level.resize(level + 1, None);
        }
        self.per_level[level] = Some(vals);
    }

    /// Read a level's weighted values, if it has been weight-marginalized.
    #[inline]
    pub fn level(&self, level: usize) -> Option<&[WeightVal]> {
        self.per_level.get(level).and_then(|o| o.as_deref())
    }

    /// Scoped `&mut` into ONE level's value vec, for the slot-prune boundary
    /// COMPACTION (`WeightFold::compact_store`) and nothing else.
    ///
    /// That pass is the only writer that rewrites a level's values IN PLACE
    /// (survivors swapped down into the prefix, then truncated). It cannot use
    /// [`WeightStore::level`] (read-only) and using [`WeightStore::set_level`]
    /// costs exactly what the in-place form exists to avoid: a second
    /// full-length vec of `WeightVal`s live beside the old one at peak, each
    /// value no smaller than a `u128` and usually a multi-limb `BigRational`.
    ///
    /// Deliberately NOT a general mutation hook — every other writer goes
    /// through `set_level` (replace a level wholesale) or `push_value` (append
    /// one slot, get its index back). Those two disciplines are what the
    /// marg-side ref walkers assume; an arbitrary in-place edit that moved or
    /// dropped slots WITHOUT rewriting the parent refs in the same pass would
    /// silently invalidate them.
    #[inline]
    pub(crate) fn level_vals_mut(&mut self, level: usize) -> Option<&mut Vec<WeightVal>> {
        self.per_level.get_mut(level).and_then(|o| o.as_mut())
    }

    /// Append `val` as a fresh slot to a weight-marginalized level, returning the
    /// new slot index. Mirrors the integer `push_count_slot` mint path used by the
    /// C2 twin-fold (`dup_resolve`): no value interning here — slot-prune merges
    /// equal-valued slots on the next prune pass. Panics if the level was not yet
    /// `set_level`'d (a scaled ref into a non-marginalized level is a bug).
    ///
    /// # Panics
    ///
    /// Panics if `level` has no weighted store allocated (it was never
    /// `set_level`'d).
    #[doc(hidden)]
    pub fn push_value(&mut self, level: usize, val: WeightVal) -> usize {
        let vec = self.per_level[level]
            .as_mut()
            .expect("push_value: level has no weighted store");
        let idx = vec.len();
        vec.push(val);
        idx
    }

    /// True once `set_level` has been called for `level`.
    #[inline]
    #[doc(hidden)]
    pub fn is_set(&self, level: usize) -> bool {
        matches!(self.per_level.get(level), Some(Some(_)))
    }
}
