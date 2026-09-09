//! The u128-sentinel + `BigUint`-side-table integer count discipline, as types.
//!
//! Every integer model-count fold in the codebase (in-apply streaming,
//! post-compile marginalization) re-derives the same shape: a fast `u128`
//! accumulator that overflows into an exact `BigUint`, encoded as a sentinel
//! value in the fast slot with the real value in a lazily-built, slot-keyed
//! side table ([`BigSide`], defined next to the marg-ref encoding it shares a
//! slot space with). This module gives that shape names — [`Count`] (one fold
//! result), [`CountRead`] (a borrowed read of a stored slot), and [`CountVec`]
//! (the column) — so the sentinel/promotion rules are defined once.
//! This module is stage 1, the "cold quadrant": [`CountVec`] used by
//! `compile_marginalize.rs`'s post-compile fold; `conjoin::stream`'s in-apply
//! streaming counts stay on their own hand-rolled scratch until stage 2.
//!
//! This module covers only the *scratch buffers*, not `TddLevel`/`marginal_counts`
//! or the `WeightStore`.

use crate::engine::Engine;
use std::marker::PhantomData;

use num_bigint::BigUint;

use crate::diagram::BigSide;

/// Canonical home of the "u128 fold overflowed" sentinel. A fold result equal
/// to this exact value is ambiguous between "the true count is u128::MAX" and
/// "the count overflowed and the real value lives in the side table" — see
/// [`Count::from_u128`] for how that ambiguity is resolved. Re-exported by
/// `conjoin::stream` (which historically defined this constant locally)
/// so existing users keep compiling unchanged.
pub(crate) const STREAM_OVERFLOW: u128 = u128::MAX;

/// The scalar result of one integer count fold: either it fit in a `u128`, or
/// it overflowed into an exact arbitrary-precision count.
///
/// Replaces the ad-hoc `Result<u128, BigUint>` (`compute_cell_count`) and
/// `(u128, Option<BigUint>)` (`compute_marginal_node_int`) fold-result
/// spellings used before this module existed.
pub(crate) enum Count {
    Fast(u128),
    Big(BigUint),
}

impl Count {
    /// Build a `Count` from a raw `u128` fold total, applying the ONE
    /// exact-max promotion rule: a total that lands exactly on
    /// [`STREAM_OVERFLOW`] (`u128::MAX`) is indistinguishable from the
    /// overflow sentinel itself, so it is promoted to `Big` even though it
    /// numerically fits in a `u128`. No fold site outside this constructor
    /// may re-derive the `v == STREAM_OVERFLOW` check.
    pub(crate) fn from_u128(v: u128) -> Count {
        if v == STREAM_OVERFLOW {
            Count::Big(BigUint::from(u128::MAX))
        } else {
            Count::Fast(v)
        }
    }
}

/// A borrowed read of one stored [`CountVec`] slot — the decoded counterpart
/// of [`Count`] (which owns its `Big` value; this borrows it).
pub(crate) enum CountRead<'a> {
    Fast(u128),
    Big(&'a BigUint),
}

/// Fallible-allocation strategy for a [`CountVec`]'s backing `Vec`s — the one
/// axis that intentionally differs between the in-apply and finished-`Tdd`
/// contexts. `reserve`/`reserve_exact`
/// mirror `Vec::try_reserve`/`Vec::try_reserve_exact`'s growth strategies
/// (amortized-doubling vs. exact), mapped to the policy's own error type.
pub(crate) trait ReservePolicy {
    type Err;
    fn reserve<T>(eng: &Engine, v: &mut Vec<T>, additional: usize) -> Result<(), Self::Err>;
    fn reserve_exact<T>(eng: &Engine, v: &mut Vec<T>, additional: usize) -> Result<(), Self::Err>;
}

/// [`ReservePolicy`] for the in-apply streaming counts path
/// (`conjoin::stream`). Delegates to the soft apply-budget tracker in
/// `conjoin::budget` (armed by `set_apply_budget`) — the one place it is read — so a
/// resize that would exceed the remaining envelope returns a cooperative
/// `Err(ApplyError::OverBudget)` instead of allocating. No accounting is
/// re-implemented here; both methods are pure delegation.
pub(crate) struct ApplyBudget;

impl ReservePolicy for ApplyBudget {
    type Err = crate::error::ApplyError;

    #[inline(always)]
    fn reserve<T>(eng: &Engine, v: &mut Vec<T>, additional: usize) -> Result<(), Self::Err> {
        let lim = eng.limits();
        lim.reserve(v, additional)
    }

    #[inline(always)]
    fn reserve_exact<T>(eng: &Engine, v: &mut Vec<T>, additional: usize) -> Result<(), Self::Err> {
        let lim = eng.limits();
        lim.reserve_exact(v, additional)
    }
}

/// [`ReservePolicy`] for post-compile marginalization scratch
/// (`compile_marginalize.rs`'s `marginalize_batch`/`ensure_counts` and
/// friends).
///
/// On pathological levels a marginal-count buffer can require a single
/// 10–17 GiB allocation. The infallible `vec![0u128; width]` (or a plain
/// `.clone()`) invokes Rust's alloc-error handler on failure, which
/// **aborts** (SIGABRT, rc=-6) when the heap is at the `RLIMIT_AS` ceiling:
/// the unwind machinery's own allocation also fails, double-faulting past the
/// recovery cascade's `catch_unwind`.
///
/// `try_reserve`/`try_reserve_exact` instead return `Err` *without*
/// committing the allocation or touching the abort handler, leaving the heap
/// at its pre-attempt level. We then raise a controlled panic from normal
/// code, which unwinds cleanly into the `catch_unwind` of the caller's
/// memory-budget recovery path, triggering a Shannon-split retry instead of
/// killing the process. These
/// panics MUST remain ordinary unwinding panics — no abort, no panic hooks —
/// since recovery depends on catching them. Mirrors the already-fallible
/// apply-stream counts path (`ApplyBudget`, above), which instead maps the
/// same failure to a cooperative `Err`.
pub(crate) struct RecoveryPanic;

impl ReservePolicy for RecoveryPanic {
    type Err = std::convert::Infallible;

    fn reserve<T>(_eng: &Engine, v: &mut Vec<T>, additional: usize) -> Result<(), Self::Err> {
        if v.try_reserve(additional).is_err() {
            panic_over_budget::<T>(additional);
        }
        Ok(())
    }

    fn reserve_exact<T>(_eng: &Engine, v: &mut Vec<T>, additional: usize) -> Result<(), Self::Err> {
        if v.try_reserve_exact(additional).is_err() {
            panic_over_budget::<T>(additional);
        }
        Ok(())
    }
}

/// Raise the controlled recovery-split panic for `RecoveryPanic` (see its doc
/// comment). `#[cold]`/`#[inline(never)]` since this only ever runs on the
/// already-doomed over-budget branch.
#[cold]
#[inline(never)]
fn panic_over_budget<T>(additional: usize) -> ! {
    panic!(
        "CountVec buffer of {additional} entries ({:.1} GiB) over budget — triggering recovery split",
        (additional as f64 * std::mem::size_of::<T>() as f64) / (1u64 << 30) as f64,
    );
}

/// The count column: a `width`-indexed sequence of [`Count`] values, stored as
/// a dense `u128` fast array plus a sparse, slot-keyed [`BigSide`] holding the
/// exact value of the slots that overflowed.
///
/// Invariants:
/// - `fast[i] == STREAM_OVERFLOW` ⇔ `big` holds an entry for slot `i`.
/// - The side table is SPARSE: it carries one entry per overflowing slot, never
///   one per slot, so a `Fast` write costs nothing there and a column with no
///   overflow owns no side-table heap at all. (The dense predecessor sized
///   itself to the full column width on the first `Big` write, and needed a
///   trailing-lazy convention plus a pad at the storage handoff to keep the
///   `Fast` push allocation-free; both are gone.) See [`BigSide`].
/// - `all_u64` is true iff every stored value fits in `u64`. It is
///   incrementally maintained and monotonic: a `Big` value or a `Fast` value
///   `> u64::MAX` clears it *permanently* — it never returns to `true`, even
///   if the offending slot is later overwritten with a small value. This
///   makes the "sentinel slot must defeat the certificate" property
///   structurally true rather than a re-derived check at each read site.
///
/// `R: ReservePolicy` monomorphizes the fallible-allocation discipline; see
/// [`ApplyBudget`]/[`RecoveryPanic`].
pub(crate) struct CountVec<R: ReservePolicy> {
    fast: Vec<u128>,
    big: Option<BigSide>,
    all_u64: bool,
    _res: PhantomData<R>,
}

impl<R: ReservePolicy> CountVec<R> {
    /// A fresh `width`-element column, all zeroed (0 fits `u64`, so
    /// `all_u64` starts `true`). Reserves exactly `width` before filling.
    pub(crate) fn try_with_width(eng: &Engine, width: usize) -> Result<Self, R::Err> {
        let mut fast: Vec<u128> = Vec::new();
        R::reserve_exact(eng, &mut fast, width)?;
        fast.resize(width, 0u128);
        Ok(CountVec {
            fast,
            big: None,
            all_u64: true,
            _res: PhantomData,
        })
    }

    /// An empty column with `cap` slots reserved exactly up front (the
    /// streaming output column pre-reserves `k1.max(k2)` and then grows
    /// fallibly via [`Self::push`]).
    pub(crate) fn try_with_capacity(eng: &Engine, cap: usize) -> Result<Self, R::Err> {
        let mut fast: Vec<u128> = Vec::new();
        R::reserve_exact(eng, &mut fast, cap)?;
        Ok(CountVec {
            fast,
            big: None,
            all_u64: true,
            _res: PhantomData,
        })
    }

    /// Borrow this column as a [`CountRef`], carrying the certificate rather
    /// than re-deriving it (a re-scan could disagree with the incrementally
    /// maintained flag on a column whose overflow slot was later overwritten).
    #[inline(always)]
    pub(crate) fn as_count_ref(&self) -> CountRef<'_> {
        CountRef {
            fast: &self.fast,
            big: self.big.as_ref(),
            all_u64: self.all_u64,
        }
    }

    /// Overwrite slot `i` (pre-sized fill; see [`Self::try_with_width`]).
    #[inline(always)]
    pub(crate) fn set(&mut self, eng: &Engine, i: usize, c: Count) -> Result<(), R::Err> {
        match c {
            Count::Fast(v) => {
                debug_assert!(
                    v != STREAM_OVERFLOW,
                    "Count::Fast carrying the overflow sentinel — Count::from_u128 should have promoted this to Big"
                );
                // A slot that stops overflowing (the pinned counter recomputes
                // a level when pins change) must lose its exact value, or the
                // sentinel ⇔ entry invariant breaks in the stale direction.
                // Gated on the OLD cell so an ordinary fast write costs nothing:
                // only a genuine Big→Fast transition touches the side table.
                if std::mem::replace(&mut self.fast[i], v) == STREAM_OVERFLOW
                    && let Some(big) = self.big.as_mut() {
                        big.take(i);
                    }
                if v > u64::MAX as u128 {
                    self.all_u64 = false;
                }
            }
            Count::Big(b) => {
                self.fast[i] = STREAM_OVERFLOW;
                self.big
                    .get_or_insert_with(BigSide::default)
                    .try_insert::<R>(eng, i, b)?;
                self.all_u64 = false;
            }
        }
        Ok(())
    }

    /// Append one value, growing by amortized doubling. A `Big` append records
    /// one side-table entry under the new slot's index; a `Fast` append leaves
    /// the side table untouched — with sparse storage that is simply what an
    /// absent entry already means, so no backfill is needed to keep the hot
    /// path allocation-free (the dense predecessor had to pad with `None`).
    #[inline(always)]
    pub(crate) fn push(&mut self, eng: &Engine, c: Count) -> Result<(), R::Err> {
        R::reserve(eng, &mut self.fast, 1)?;
        match c {
            Count::Fast(v) => {
                debug_assert!(
                    v != STREAM_OVERFLOW,
                    "Count::Fast carrying the overflow sentinel — Count::from_u128 should have promoted this to Big"
                );
                self.fast.push(v);
                if v > u64::MAX as u128 {
                    self.all_u64 = false;
                }
            }
            Count::Big(b) => {
                self.fast.push(STREAM_OVERFLOW);
                let idx = self.fast.len() - 1;
                // Strictly ascending key ⇒ an O(1) amortized push inside `BigSide`.
                self.big
                    .get_or_insert_with(BigSide::default)
                    .try_insert::<R>(eng, idx, b)?;
                self.all_u64 = false;
            }
        }
        Ok(())
    }

    /// Decode slot `i`: a sentinel fast value reads through to the big table.
    #[inline(always)]
    pub(crate) fn get(&self, i: usize) -> CountRead<'_> {
        let raw = self.fast[i];
        if raw == STREAM_OVERFLOW {
            let b = self
                .big_val(i)
                .expect("CountVec: sentinel fast slot without a big value — invariant violated");
            CountRead::Big(b)
        } else {
            CountRead::Fast(raw)
        }
    }

    /// Raw fast-slot read (sentinel included, no big-table decode).
    #[inline(always)]
    pub(crate) fn fast_val(&self, i: usize) -> u128 {
        self.fast[i]
    }

    /// Raw big-table read; `None` when slot `i` has no overflow value (no big
    /// table yet, or a plain `Fast` slot).
    #[inline(always)]
    pub(crate) fn big_val(&self, i: usize) -> Option<&BigUint> {
        self.big.as_ref().and_then(|v| v.get(i))
    }

    #[inline(always)]
    pub(crate) fn len(&self) -> usize {
        self.fast.len()
    }

    /// Test-only inspector (production reads the certificate through the
    /// borrowed view, [`CountRef::all_u64`]).
    #[cfg(test)]
    #[inline(always)]
    pub(crate) fn all_u64(&self) -> bool {
        self.all_u64
    }

    /// Test-only inspector (production readers go through `get`/`big_val`).
    #[cfg(test)]
    #[inline(always)]
    pub(crate) fn has_big(&self) -> bool {
        self.big.is_some()
    }

    /// Fallible clone: reserves both backing arrays exactly before copying,
    /// so an over-budget duplicate raises the policy's error instead of an
    /// infallible allocator abort. Test-only since the borrowed-view rewrite
    /// removed production column duplication; kept (with [`Self::clone_guarded`])
    /// as the round-trip coverage of the fast/big split.
    #[cfg(test)]
    pub(crate) fn try_clone(&self, eng: &Engine) -> Result<Self, R::Err> {
        let mut fast: Vec<u128> = Vec::new();
        R::reserve_exact(eng, &mut fast, self.fast.len())?;
        fast.extend_from_slice(&self.fast);
        let big = match &self.big {
            Some(b) => Some(b.try_clone::<R>(eng)?),
            None => None,
        };
        Ok(CountVec {
            fast,
            big,
            all_u64: self.all_u64,
            _res: PhantomData,
        })
    }

    /// Storage-handoff escape hatch: unwrap into the raw `(fast, big)` pair at
    /// the `dedup_fresh_store`/`make_marginal` boundary. Both halves are exactly
    /// what `TddLevel` stores, so the handoff is a move — no re-shaping.
    pub(crate) fn into_parts(self) -> (Vec<u128>, Option<BigSide>) {
        (self.fast, self.big)
    }
}

/// The `all_u64` certificate of a raw fast column: every stored value fits in
/// `u64`. A sentinel (`STREAM_OVERFLOW`) or any value `> u64::MAX` fails it, so
/// `all_u64 ⇒ no overflow slot present`. The ONE derivation —
/// [`CountRef::from_parts_scanned`] routes every adopted-raw-array view through
/// it, so two views of the same raw arrays can never certify differently.
#[inline]
fn certify_all_u64(fast: &[u128]) -> bool {
    fast.iter().all(|&c| c <= u64::MAX as u128)
}

/// Borrowed twin of [`CountVec`]: a read-only view over raw `(fast, big)`
/// arrays plus their certificate, with the same decode rules and no ownership.
///
/// Exists so a fold can read a column that lives somewhere else — a
/// `TddLevel`'s `marginal_counts` storage, or another `CountVec` — without
/// copying it. That matters at exactly one place: the apply-side streaming
/// child columns, where the sources are wide marginal stores (hundreds of
/// millions of slots) and a copy would double them at the moment streaming
/// exists to relieve.
#[derive(Clone, Copy)]
pub(crate) struct CountRef<'a> {
    fast: &'a [u128],
    big: Option<&'a BigSide>,
    all_u64: bool,
}

impl<'a> CountRef<'a> {
    /// View raw arrays that are NOT a `CountVec` (a level's marginal storage,
    /// or the fixed leaf-label slots). The certificate is scanned
    /// ([`certify_all_u64`]).
    #[inline]
    pub(crate) fn from_parts_scanned(fast: &'a [u128], big: Option<&'a BigSide>) -> Self {
        CountRef {
            fast,
            big,
            all_u64: certify_all_u64(fast),
        }
    }

    /// Raw view of the fast column (sentinels included) — for the
    /// monomorphized unchecked-read fold fast path (`stream::read_fast`).
    #[inline(always)]
    pub(crate) fn fast_slice(&self) -> &'a [u128] {
        self.fast
    }

    /// Raw fast-slot read (sentinel included, no big-table decode).
    #[inline(always)]
    pub(crate) fn fast_val(&self, i: usize) -> u128 {
        self.fast[i]
    }

    /// Raw big-table read; `None` when slot `i` has no overflow value.
    #[inline(always)]
    pub(crate) fn big_val(&self, i: usize) -> Option<&'a BigUint> {
        self.big.and_then(|v| v.get(i))
    }

    #[inline(always)]
    pub(crate) fn len(&self) -> usize {
        self.fast.len()
    }

    #[inline(always)]
    pub(crate) fn all_u64(&self) -> bool {
        self.all_u64
    }
}

mod fold;

pub use fold::*;

impl CountVec<RecoveryPanic> {
    /// Infallible convenience wrapper (`RecoveryPanic::Err = Infallible`, so
    /// the fallible form can never actually return `Err` — it panics first).
    pub(crate) fn with_width(eng: &Engine, width: usize) -> Self {
        unwrap_infallible(Self::try_with_width(eng, width))
    }

    pub(crate) fn set_i(&mut self, eng: &Engine, i: usize, c: Count) {
        unwrap_infallible(self.set(eng, i, c))
    }

    /// Test-only fixture builder (production fills go through `push`/`set_i`).
    #[cfg(test)]
    pub(crate) fn push_i(&mut self, eng: &Engine, c: Count) {
        unwrap_infallible(self.push(eng, c))
    }

    /// Test-only infallible `try_clone` (production duplicates of a marginal
    /// store go through the fallible form, which propagates `OverBudget`).
    #[cfg(test)]
    pub(crate) fn clone_guarded(&self, eng: &Engine) -> Self {
        unwrap_infallible(self.try_clone(eng))
    }
}

#[cfg(test)]
mod tests;
