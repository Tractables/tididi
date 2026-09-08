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
//! `Storage` stays out of scope here. This module
//! only replaces the *scratch buffers*, not `TddLevel`/`marginal_counts` or
//! the `WeightStore`.

use std::marker::PhantomData;

use num_bigint::BigUint;

use crate::tdd::query::semiring::WeightVal;
use crate::tdd::types::{BigSide, InputPair, TddLevel};
use crate::vtree::{Vtree, VtreeIdx};

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
    fn reserve<T>(v: &mut Vec<T>, additional: usize) -> Result<(), Self::Err>;
    fn reserve_exact<T>(v: &mut Vec<T>, additional: usize) -> Result<(), Self::Err>;
}

/// [`ReservePolicy`] for the in-apply streaming counts path
/// (`conjoin::stream`). Delegates to the soft apply-budget tracker in
/// `conjoin::budget` (armed by `set_apply_budget`) — the one place it is read — so a
/// resize that would exceed the remaining envelope returns a cooperative
/// `Err(ApplyError::OverBudget)` instead of allocating. No accounting is
/// re-implemented here; both methods are pure delegation.
pub(crate) struct ApplyBudget;

impl ReservePolicy for ApplyBudget {
    type Err = crate::tdd::limits::ApplyError;

    #[inline(always)]
    fn reserve<T>(v: &mut Vec<T>, additional: usize) -> Result<(), Self::Err> {
        crate::tdd::limits::budget_reserve(v, additional)
    }

    #[inline(always)]
    fn reserve_exact<T>(v: &mut Vec<T>, additional: usize) -> Result<(), Self::Err> {
        crate::tdd::limits::budget_reserve_exact(v, additional)
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
/// code, which unwinds cleanly into
/// [`crate::recovery::compile_mc_with_recovery`]'s `catch_unwind` and
/// triggers a Shannon-split retry instead of killing the process. These
/// panics MUST remain ordinary unwinding panics — no abort, no panic hooks —
/// since recovery depends on catching them. Mirrors the already-fallible
/// apply-stream counts path (`ApplyBudget`, above), which instead maps the
/// same failure to a cooperative `Err`.
pub(crate) struct RecoveryPanic;

impl ReservePolicy for RecoveryPanic {
    type Err = std::convert::Infallible;

    fn reserve<T>(v: &mut Vec<T>, additional: usize) -> Result<(), Self::Err> {
        if v.try_reserve(additional).is_err() {
            panic_over_budget::<T>(additional);
        }
        Ok(())
    }

    fn reserve_exact<T>(v: &mut Vec<T>, additional: usize) -> Result<(), Self::Err> {
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
    pub(crate) fn try_with_width(width: usize) -> Result<Self, R::Err> {
        let mut fast: Vec<u128> = Vec::new();
        R::reserve_exact(&mut fast, width)?;
        fast.resize(width, 0u128);
        Ok(CountVec { fast, big: None, all_u64: true, _res: PhantomData })
    }

    /// An empty column with `cap` slots reserved exactly up front (the
    /// streaming output column pre-reserves `k1.max(k2)` and then grows
    /// fallibly via [`Self::push`]).
    pub(crate) fn try_with_capacity(cap: usize) -> Result<Self, R::Err> {
        let mut fast: Vec<u128> = Vec::new();
        R::reserve_exact(&mut fast, cap)?;
        Ok(CountVec { fast, big: None, all_u64: true, _res: PhantomData })
    }

    /// Borrow this column as a [`CountRef`], carrying the certificate rather
    /// than re-deriving it (a re-scan could disagree with the incrementally
    /// maintained flag on a column whose overflow slot was later overwritten).
    #[inline(always)]
    pub(crate) fn as_count_ref(&self) -> CountRef<'_> {
        CountRef { fast: &self.fast, big: self.big.as_ref(), all_u64: self.all_u64 }
    }

    /// Overwrite slot `i` (pre-sized fill; see [`Self::try_with_width`]).
    #[inline(always)]
    pub(crate) fn set(&mut self, i: usize, c: Count) -> Result<(), R::Err> {
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
                if std::mem::replace(&mut self.fast[i], v) == STREAM_OVERFLOW {
                    if let Some(big) = self.big.as_mut() {
                        big.take(i);
                    }
                }
                if v > u64::MAX as u128 {
                    self.all_u64 = false;
                }
            }
            Count::Big(b) => {
                self.fast[i] = STREAM_OVERFLOW;
                self.big
                    .get_or_insert_with(BigSide::default)
                    .try_insert::<R>(i, b)?;
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
    pub(crate) fn push(&mut self, c: Count) -> Result<(), R::Err> {
        R::reserve(&mut self.fast, 1)?;
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
                    .try_insert::<R>(idx, b)?;
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

    /// Raw fast-slot read (sentinel included, no big-table decode). For shim
    /// code that mirrors the pre-`CountVec` `computed_counts`/`marginal_counts`
    /// read pattern.
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
    pub(crate) fn try_clone(&self) -> Result<Self, R::Err> {
        let mut fast: Vec<u128> = Vec::new();
        R::reserve_exact(&mut fast, self.fast.len())?;
        fast.extend_from_slice(&self.fast);
        let big = match &self.big {
            Some(b) => Some(b.try_clone::<R>()?),
            None => None,
        };
        Ok(CountVec { fast, big, all_u64: self.all_u64, _res: PhantomData })
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
        CountRef { fast, big, all_u64: certify_all_u64(fast) }
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

// ── MargFold: the value-kind axis of the marginalization fold ────────────────
//
// Stage 3 of the count-discipline unification: the four
// mirror families of "walk children, fold Σ left×right per node" collapse to
//   - ONE recursive ensure walk ([`ensure_fold_walk`]), generic over the value
//     kind (`F: MargFold`) AND the reservation policy (`R: ReservePolicy`) —
//     both contexts (in-apply `&[TddLevel]` snapshot, finished `Tdd`) walk the
//     same `&[TddLevel]` + `Vtree` shape, so one walk serves all quadrants;
//   - ONE two-pass integer fold discipline ([`IntFold::fold`]) and ONE clean
//     weighted fold ([`WeightFold::fold`]).
// The child READERS (how a pair's u32 ref resolves to a value: bit-30 tagged
// refs + snapshot columns in-apply; marg slots / bit-31 ZERO sentinel /
// interned weights on a finished Tdd) stay context-owned adapter closures
// handed to the fold — they are storage, not fold (design §1).
//
// Deviation from the design sketch: `fold` is an INHERENT method on each
// value-kind ZST rather than a trait method, because the two reader shapes
// genuinely differ (integer: one lazy `CountRead` reader per side; weighted:
// one `Cow<WeightVal>` reader per side plus an explicit zero). The trait
// carries only the COLUMN CONTRACT — both how the ensure walk builds a column
// (pre-size + `set_col`) and how the apply-side streaming driver builds one
// (`try_with_capacity` + `push_col`, one push per alive cell). The apply
// driver's remaining per-value-kind pieces (child snapshot, per-cell fold,
// level commit) hang off the `StreamPayload` sub-trait in
// `transform::pairwise::conjoin::stream`, which needs apply-local types this
// module has no business knowing.

/// The value-kind axis of the marginalization fold: what scalar a per-node
/// fold produces and what scratch column stores it. See module comment above.
pub(crate) trait MargFold {
    /// One per-node fold result (`Count` | `WeightVal`).
    type Scalar;
    /// The scratch column, parameterized by the fallibility policy.
    type Col<R: ReservePolicy>;
    /// A fresh `width`-element column of `zero`s, reserved through `R` — the
    /// single fallible-allocation point of the ensure walk (this is what
    /// delivers A1: the weighted column allocates through the SAME fallible
    /// path as the integer one, per policy).
    fn alloc_col<R: ReservePolicy>(
        width: usize,
        zero: &Self::Scalar,
    ) -> Result<Self::Col<R>, R::Err>;
    /// Store one fold result at slot `i` of a pre-sized column.
    fn set_col<R: ReservePolicy>(
        col: &mut Self::Col<R>,
        i: usize,
        v: Self::Scalar,
    ) -> Result<(), R::Err>;
    /// An EMPTY column that will be filled by [`Self::push_col`], with room for
    /// `cap` appends pre-reserved where the value kind reserves at all. The
    /// append-built counterpart of [`Self::alloc_col`] (which pre-sizes and is
    /// filled by [`Self::set_col`]) — the apply-side streaming output column is
    /// built this way, one push per alive cell.
    fn try_with_capacity<R: ReservePolicy>(cap: usize) -> Result<Self::Col<R>, R::Err>;
    /// Append one fold result to an append-built column.
    fn push_col<R: ReservePolicy>(
        col: &mut Self::Col<R>,
        v: Self::Scalar,
    ) -> Result<(), R::Err>;
    /// Number of values currently stored in a column.
    fn col_len<R: ReservePolicy>(col: &Self::Col<R>) -> usize;
}

/// Integer model counts: u128 fast path overflowing into exact `BigUint`.
pub(crate) struct IntFold;

/// Exact weighted (`--weighted`) semiring values: no overflow machinery.
pub(crate) struct WeightFold;

impl MargFold for IntFold {
    type Scalar = Count;
    type Col<R: ReservePolicy> = CountVec<R>;

    fn alloc_col<R: ReservePolicy>(width: usize, _zero: &Count) -> Result<CountVec<R>, R::Err> {
        CountVec::try_with_width(width)
    }

    #[inline(always)]
    fn set_col<R: ReservePolicy>(col: &mut CountVec<R>, i: usize, v: Count) -> Result<(), R::Err> {
        col.set(i, v)
    }

    fn try_with_capacity<R: ReservePolicy>(cap: usize) -> Result<CountVec<R>, R::Err> {
        CountVec::try_with_capacity(cap)
    }

    #[inline(always)]
    fn push_col<R: ReservePolicy>(col: &mut CountVec<R>, v: Count) -> Result<(), R::Err> {
        col.push(v)
    }

    #[inline(always)]
    fn col_len<R: ReservePolicy>(col: &CountVec<R>) -> usize {
        col.len()
    }
}

impl MargFold for WeightFold {
    type Scalar = WeightVal;
    type Col<R: ReservePolicy> = Vec<WeightVal>;

    fn alloc_col<R: ReservePolicy>(width: usize, zero: &WeightVal) -> Result<Vec<WeightVal>, R::Err> {
        let mut v: Vec<WeightVal> = Vec::new();
        R::reserve_exact(&mut v, width)?;
        v.resize(width, zero.clone());
        Ok(v)
    }

    #[inline(always)]
    fn set_col<R: ReservePolicy>(col: &mut Vec<WeightVal>, i: usize, v: WeightVal) -> Result<(), R::Err> {
        col[i] = v;
        Ok(())
    }

    /// `cap` is deliberately IGNORED: the weighted streaming column is an
    /// ordinary `Vec<WeightVal>` grown by plain `push`, with no upfront
    /// reservation and no budget charge (rationals live outside the `CountVec`
    /// reserve policy; the per-pair transient is charged by the apply's
    /// collect sink instead). Pre-reserving here would newly charge the
    /// weighted path against the soft budget — a behavior change, not a
    /// simplification.
    fn try_with_capacity<R: ReservePolicy>(_cap: usize) -> Result<Vec<WeightVal>, R::Err> {
        Ok(Vec::new())
    }

    #[inline(always)]
    fn push_col<R: ReservePolicy>(col: &mut Vec<WeightVal>, v: WeightVal) -> Result<(), R::Err> {
        col.push(v);
        Ok(())
    }

    #[inline(always)]
    fn col_len<R: ReservePolicy>(col: &Vec<WeightVal>) -> usize {
        col.len()
    }
}

impl IntFold {
    /// The ONE two-pass integer fold: `Σ over pairs (left × right)`.
    ///
    /// Pass 1 accumulates in `u128` with `checked_mul`/`checked_add`, breaking
    /// to pass 2 on the first overflow or the first `Big` child read. Pass 2
    /// re-reads every pair (hence `P: Clone`) into an exact `BigUint` total
    /// with mixed-magnitude branching — the u128×u128 sub-case skips `BigUint`
    /// multiplication entirely, the mixed cases use scalar multiply (one alloc
    /// for the product), and only the both-`Big` case takes the full bigint
    /// multiply. Callgrind on a huge-count instance showed mul3+alloc/free
    /// dominating the naive both-sides-`BigUint::from` fold this replaces
    /// (which the ensure walks used until stage 3); results are identical.
    ///
    /// Exact-max promotion (a pass-1 total that lands exactly on the overflow
    /// sentinel) is [`Count::from_u128`]'s job — never re-derived here.
    pub(crate) fn fold<'a, P, L, R>(pairs: P, l: L, r: R) -> Count
    where
        P: Iterator<Item = InputPair> + Clone,
        L: Fn(usize) -> CountRead<'a>,
        R: Fn(usize) -> CountRead<'a>,
    {
        let mut total: u128 = 0;
        let mut overflowed = false;
        for pair in pairs.clone() {
            let (CountRead::Fast(lc), CountRead::Fast(rc)) =
                (l(pair.left.idx()), r(pair.right.idx()))
            else {
                overflowed = true;
                break;
            };
            match lc.checked_mul(rc).and_then(|p| total.checked_add(p)) {
                Some(t) => total = t,
                None => {
                    overflowed = true;
                    break;
                }
            }
        }
        if !overflowed {
            return Count::from_u128(total);
        }
        let mut bt = BigUint::ZERO;
        for pair in pairs {
            match (l(pair.left.idx()), r(pair.right.idx())) {
                (CountRead::Fast(a), CountRead::Fast(b)) => {
                    if let Some(p) = a.checked_mul(b) {
                        bt += p;
                    } else {
                        let mut t = BigUint::from(a);
                        t *= b;
                        bt += t;
                    }
                }
                (CountRead::Big(a), CountRead::Fast(b)) => bt += a * b,
                (CountRead::Fast(a), CountRead::Big(b)) => bt += b * a,
                (CountRead::Big(a), CountRead::Big(b)) => bt += a * b,
            }
        }
        Count::Big(bt)
    }
}

impl WeightFold {
    /// The ONE weighted fold: `Σ over pairs (left × right)` in the exact
    /// semiring. Rationals don't overflow, so a single clean pass; readers
    /// hand back `Cow` so slot/snapshot reads stay borrow-only and only
    /// interned/leaf/store reads pay a clone.
    pub(crate) fn fold<'a, P, L, R>(pairs: P, l: L, r: R, zero: WeightVal) -> WeightVal
    where
        P: Iterator<Item = InputPair>,
        L: Fn(usize) -> std::borrow::Cow<'a, WeightVal>,
        R: Fn(usize) -> std::borrow::Cow<'a, WeightVal>,
    {
        let mut total = zero;
        for pair in pairs {
            let lc = l(pair.left.idx());
            let rc = r(pair.right.idx());
            total.add_assign(&lc.mul(&rc));
        }
        total
    }
}

/// Lifetime policy for the per-level value columns a bottom-up fold pass
/// builds — the one knob shared by `ensure_fold_walk` and the finished-`Tdd`
/// hybrid counter (`query::count::IncrementalPinnedCounter`), so "when does a
/// column die" is decided in exactly one place.
///
/// The vtree is a TREE: every level has exactly ONE parent, hence exactly one
/// in-pass consumer of its column. So a child's column is provably dead the
/// moment its parent's column is complete, and a pass that reads only the ROOT
/// value can hold the frontier (max antichain) instead of the whole diagram.
/// That is [`Self::Frontier`]. Any consumer that re-reads a NON-root column
/// after the pass needs [`Self::All`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[doc(hidden)]
pub enum ColumnRetention {
    /// Keep every level's column for the caller. Required by the marginalize
    /// cascades (each level's column is `take`n and installed as that level's
    /// marginal store), by the Gray-code re-pin counter (cached columns are
    /// reused across pin flips), and by any caller that keeps the whole array.
    All,
    /// Free each child column as soon as its parent's column is complete.
    /// ROOT-ONLY callers must opt in explicitly — this is never a default.
    Frontier,
}

/// The ONE recursive ensure walk (design §3): populate `computed[li]` with a
/// per-node fold column, recursing into children first, skipping levels that
/// are already computed, already marginal (per the context's `already_done`
/// predicate — `is_marginal()` in three quadrants, `WeightStore::is_set` on
/// the finished-Tdd weighted one), or vtree leaves (their values resolve on
/// demand inside the context's readers).
///
/// `fold_node(li, i, left_i, right_i, computed)` is the per-quadrant adapter:
/// it wires the context's child readers into [`IntFold::fold`] /
/// [`WeightFold::fold`]. It receives `computed` as an argument (not a capture)
/// so the walk can keep the unique `&mut` between fold calls. A fold of level
/// `t` reads `computed[l_i]`/`computed[r_i]` and NOTHING else — that is what
/// makes [`ColumnRetention::Frontier`] sound.
///
/// `retain` is the column-lifetime policy. Under
/// [`ColumnRetention::Frontier`] the walk releases each child column right
/// after the parent's column is stored, so on return ONLY `computed[li]` (the
/// walk root, which has no parent inside the walk) is populated — the caller
/// must read that column and nothing else. Under [`ColumnRetention::All`]
/// every visited level keeps its column, which is what the marginalize
/// cascades consume. `Frontier` also gives up the walk's memoization for the
/// freed subtrees, so it is for ONE root-only walk per `computed` buffer; a
/// second walk over an overlapping subtree would recompute it.
pub(crate) fn ensure_fold_walk<F, R, G, N>(
    li: usize,
    vtree: &Vtree,
    levels: &[TddLevel],
    computed: &mut [Option<F::Col<R>>],
    zero: &F::Scalar,
    already_done: &G,
    fold_node: &N,
    retain: ColumnRetention,
) -> Result<(), R::Err>
where
    F: MargFold,
    R: ReservePolicy,
    G: Fn(usize) -> bool,
    N: Fn(usize, usize, usize, usize, &[Option<F::Col<R>>]) -> F::Scalar,
{
    if computed[li].is_some() || already_done(li) || vtree.node(VtreeIdx(li as u32)).is_leaf() {
        return Ok(());
    }
    let (left, right) = vtree.children(VtreeIdx(li as u32));
    let (l_i, r_i) = (left.idx(), right.idx());
    ensure_fold_walk::<F, R, G, N>(l_i, vtree, levels, computed, zero, already_done, fold_node, retain)?;
    ensure_fold_walk::<F, R, G, N>(r_i, vtree, levels, computed, zero, already_done, fold_node, retain)?;

    // Fallible alloc — `width` can reach ~1B on pathological levels, where an
    // infallible `vec![zero; width]` would abort past the recovery cascade.
    // The policy routes this through the soft budget (`ApplyBudget`) or the
    // controlled recovery panic (`RecoveryPanic`).
    let width = levels[li].width();
    let mut col = F::alloc_col::<R>(width, zero)?;
    for (i, _pairs) in levels[li].internal_inputs_iter() {
        F::set_col(&mut col, i, fold_node(li, i, l_i, r_i, computed))?;
    }
    computed[li] = Some(col);
    if retain == ColumnRetention::Frontier {
        // Single-parent argument (see [`ColumnRetention`]): `l_i`/`r_i` are
        // strict descendants of the walk root, `li` is their ONLY parent, and
        // `li`'s column is now complete — so nothing in this walk, and nothing
        // a root-only caller does after it, can read them again. Freeing here
        // (not at the end) is what turns the live set into the frontier.
        computed[l_i] = None;
        computed[r_i] = None;
    }
    Ok(())
}

/// Unwrap a `Result` whose error type is uninhabited. A tiny helper so the
/// `RecoveryPanic` infallible wrappers below stay one-liners.
#[inline]
pub(crate) fn unwrap_infallible<T>(r: Result<T, std::convert::Infallible>) -> T {
    match r {
        Ok(v) => v,
        Err(e) => match e {},
    }
}

impl CountVec<RecoveryPanic> {
    /// Infallible convenience wrapper (`RecoveryPanic::Err = Infallible`, so
    /// the fallible form can never actually return `Err` — it panics first).
    pub(crate) fn with_width(width: usize) -> Self {
        unwrap_infallible(Self::try_with_width(width))
    }

    pub(crate) fn set_i(&mut self, i: usize, c: Count) {
        unwrap_infallible(self.set(i, c))
    }

    /// Test-only fixture builder (production fills go through `push`/`set_i`).
    #[cfg(test)]
    pub(crate) fn push_i(&mut self, c: Count) {
        unwrap_infallible(self.push(c))
    }

    /// Test-only infallible `try_clone` (production duplicates of a marginal
    /// store go through the fallible form, which propagates `OverBudget`).
    #[cfg(test)]
    pub(crate) fn clone_guarded(&self) -> Self {
        unwrap_infallible(self.try_clone())
    }
}

#[cfg(test)]
#[path = "counts_tests.rs"]
mod tests;
