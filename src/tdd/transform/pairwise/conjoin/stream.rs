//! Streaming-marginal emit support for the scheduled `apply_and_fallible` path.
//!
//! When `marginalize_targets[t_idx]` is true during the bottom-up scatter, the
//! dense path emits the output level as `marginal_counts` directly — for each
//! alive cell, build the pairs into `level.pairs` (so process_cell! is reused
//! verbatim), compute `Σ counts_left[lc] * counts_right[rc]`, then truncate
//! the pairs/node away before the next cell. Peak transient is bounded by the
//! largest single cell, not Σ pairs. The post-apply `marginalize_batch` sees
//! the level as already-marginal and skips it.
//!
//! Mirrors `compile.rs::ensure_counts` / `get_marginal_count` but operates on
//! a `&[TddLevel]` slice — apply_and's output is still being built, so we
//! can't hand a finished `Tdd` to the existing helpers.
//!
//! # One driver, two value kinds
//!
//! Integer (`--mc`) and weighted (`--weighted`) streaming share ONE driver.
//! The value-kind axis is [`crate::tdd::counts::MargFold`] (scalar + column
//! contract, `counts.rs`) extended here by [`StreamPayload`], which adds the
//! four things the apply-side driver needs and that genuinely differ between
//! the two kinds:
//!
//! | hook | why it must stay per-kind |
//! |---|---|
//! | `fold_node` | the child READERS differ (lazy `CountRead` vs `Cow<WeightVal>`); `MargRef::Inline` is integer-side only |
//! | `child_view` | the borrow shape and the marginal-LEAF semantics differ (integer: always a `CountRef` into the child's storage, fixed `[2,1,1]` slots for an empty inline store; weighted: `Cow`, since the `WeightStore` column and the semiring leaf bases can only be produced owned) |
//! | `fold_cell` | integer carries the u128-fast-path/`BigUint`-overflow discipline; rationals cannot overflow, so the weighted fold is a single clean pass |
//! | `store_level` | integer commits raw `(fast, big)` arrays into the level (no reshaping — both sides hold the same sparse side table); weighted commits slot count + `WeightStore` payload |
//!
//! Everything else — the ensure walk, the descendant cascade, the state
//! build, the per-cell push/remap, the commit precondition — is written once,
//! generic over `F: StreamPayload`, and monomorphized at the ONE runtime
//! branch in [`build_stream_state`] (and its mirror in
//! `cell::run_level_rows_stream_count`, the row-loop dispatch point).

use crate::vtree::VtreeIdx;
use crate::tdd::types;
use crate::tdd::types::{decode_marg_coord, MargRef, MARG_VALUE_MASK, MARG_OVERFLOW_TAG};
use crate::tdd::query::WeightVal;
use crate::tdd::weight_store::WeightStore;
use super::{ApplyError, budget_reserve_exact, TddLevel, InputPair, LeafLabel};
use super::cell::bothmarg_collapse_enabled;

pub(crate) use crate::tdd::counts::STREAM_OVERFLOW;
use crate::tdd::counts::{
    ensure_fold_walk, ApplyBudget, ColumnRetention, Count, CountRead, CountRef, CountVec, IntFold,
    MargFold, WeightFold,
};

/// The fixed conceptual slots of a LEAF child's column, ordered
/// `{One=2, Pos=1, Neg=1}` per `LeafLabel::from_idx` (LEAF_WIDTH = 3). A
/// `static` so a leaf view borrows it rather than minting a `vec![2, 1, 1]`
/// per level.
static LEAF_COUNTS: [u128; 3] = [2, 1, 1];

/// Per-child READ VIEW of the child level's fold column, taken before the dense
/// scatter loop.
///
/// Borrowed, not copied: the column lives in the child level's marginal storage
/// (or the per-apply computed scratch) and is read there in place. `t` and its
/// two vtree children are distinct nodes, so the caller splits the three level
/// slots apart once (`slice::get_disjoint_mut` in the driver loop) and the
/// child views coexist with the `&mut level` borrow taken for output. Copying
/// them instead doubled a wide marginal child's storage at exactly the moment
/// streaming exists to relieve — a 536M-slot column is an 8 GiB single alloc.
///
/// Integer instantiation (`StreamChild<IntFold>`, aliased
/// [`StreamChildCounts`]): the `all_u64` certificate rides on the [`CountRef`]
/// (see `counts.rs`) — when both children certify, [`compute_cell_count`]
/// takes the widening-multiply fast path (`u64×u64→u128` is a single `mul`
/// that can never overflow the u128 product, max (2^64-1)^2 < 2^128), so the
/// heavy u128 `checked_mul` is skipped and only the running-total `checked_add`
/// guards overflow. Inline bit-30-tagged refs always decode to ≤
/// MARG_INLINE_MAX (u64), so the certificate over the column alone covers
/// every read in the cell loop.
pub(super) struct StreamChild<'a, F: StreamPayload> {
    pub(super) col: F::ChildCol<'a>,
    /// True iff this view is of a marginal child level (its refs are
    /// marg-side slot refs, possibly bit-30 tagged). When set, the per-cell
    /// fold decodes the ref before indexing the column. For a
    /// non-marginal/leaf child the ref is a plain node index (bit 30 may be a
    /// real high bit) — do NOT decode.
    pub(super) is_marg: bool,
}

/// The integer instantiation, spelled out because it is the one the hot
/// `--mc` path and the overflow validation tests name directly.
pub(super) type StreamChildCounts<'a> = StreamChild<'a, IntFold>;

/// Per-level streaming state for ONE value kind, live only for the row loop:
/// both child views plus a mutable borrow of the level's output column.
///
/// The output column itself is owned by the driver loop's [`StreamLevelState`]
/// so it outlives the child borrows — the level tail retakes `&mut levels` to
/// commit it, which it could not do while a view into `levels` was alive.
pub(super) struct StreamState<'a, F: StreamPayload> {
    pub(super) left: StreamChild<'a, F>,
    pub(super) right: StreamChild<'a, F>,
    pub(super) counts: &'a mut F::Col<ApplyBudget>,
    /// The diagram's weight store while `F = WeightFold`; `None` in integer mode.
    pub(super) ws: Option<&'a WeightStore>,
}

/// The level's in-flight output column, indexed by alive-cell position and
/// handed to [`StreamPayload::store_level`] on commit. The runtime value-kind
/// choice is made once per level in [`build_stream_state`]; everything
/// downstream of the two `match` arms (here and in
/// `cell::run_level_rows_stream_count`) is statically monomorphized — there is
/// no dynamic dispatch and no inert second payload.
///
/// Deliberately BORROW-FREE: it is carried across the whole cell-build route
/// dispatch to the commit, so it must not pin a borrow of `levels`.
pub(super) enum StreamLevelState {
    Int(CountVec<ApplyBudget>),
    /// `--weighted`: exact `BigRational` semiring values carried into the
    /// external `WeightStore`.
    Weighted(Vec<WeightVal>),
}

/// The apply-side payload axis: what the ONE streaming driver needs from a
/// value kind on top of [`MargFold`]'s column contract. Five hooks (`zero`,
/// `fold_node`, `child_view`, `fold_cell`, `store_level`), each one a place
/// where the two kinds genuinely differ — see the table in the module
/// comment.
pub(super) trait StreamPayload: MargFold + Sized {
    /// How this kind reads ONE child's column for the duration of the row loop.
    /// Borrowed wherever the storage can lend a reference ([`CountRef`] over a
    /// level's raw marginal arrays or another column; `Cow::Borrowed` over the
    /// weighted computed scratch); the weighted `WeightStore` column is copied,
    /// so that one case stays owned.
    type ChildCol<'a>;

    /// The additive identity of this value kind, resolved once per ensure
    /// walk (the weighted one reads the diagram's `WeightStore`).
    fn zero(ws: Option<&WeightStore>) -> Self::Scalar;

    /// Fold one node of `levels[lvl]`: `Σ over its pairs (left × right)`, with
    /// this kind's child readers resolving each `u32` ref against the
    /// in-flight `levels` slice and the per-batch `computed` scratch.
    ///
    /// Impls carry `#[inline]`: the pre-unification form was a closure the
    /// ensure walk monomorphized and inlined, and the walk's per-node loop must
    /// not gain a call.
    fn fold_node(
        lvl: usize,
        i: usize,
        l_i: usize,
        r_i: usize,
        vtree: &crate::vtree::Vtree,
        levels: &[TddLevel],
        computed: &[Option<Self::Col<ApplyBudget>>],
        zero: &Self::Scalar,
        ws: Option<&WeightStore>,
    ) -> Self::Scalar;

    /// Open a read view of child level `li`'s column. `level` is `levels[li]`,
    /// handed in already split off from the output level's `&mut` borrow.
    ///
    /// Fallible only for the one case that must still materialize (the
    /// weighted `WeightStore` column): that copy can be multi-GiB on extreme
    /// widths, so it routes through [`try_clone_counts`] and propagates
    /// `OverBudget` (→ v-split / conditioning-deepen recovery) instead of
    /// aborting. Every borrowing case allocates nothing and cannot fail.
    fn child_view<'a>(
        li: usize,
        vtree: &crate::vtree::Vtree,
        level: &'a TddLevel,
        computed: &'a [Option<Self::Col<ApplyBudget>>],
        ws: Option<&WeightStore>,
    ) -> Result<StreamChild<'a, Self>, ApplyError>;

    /// Collapse one alive cell's collected pairs to a single scalar:
    /// `Σ left[p.left] × right[p.right]`.
    fn fold_cell(
        pairs: &[InputPair],
        left: &StreamChild<'_, Self>,
        right: &StreamChild<'_, Self>,
        ws: Option<&WeightStore>,
    ) -> Self::Scalar;

    /// Commit a finished column into `levels[li]`, turning the level marginal.
    /// The caller has already checked the marginalization precondition
    /// ([`types::assert_can_make_marginal`]).
    fn store_level(
        levels: &mut [TddLevel],
        li: usize,
        col: Self::Col<ApplyBudget>,
        ws: Option<&mut WeightStore>,
    );
}

/// Ensure `computed[li]` is populated (or `levels[li]` is already marginal).
/// Mirrors `compile_marginalize.rs::ensure_counts` — both are the shared
/// [`ensure_fold_walk`], here with this context's per-kind readers wired in
/// through [`StreamPayload::fold_node`]. ("counts" is the historical name; the
/// walk carries weighted semiring values under `F = WeightFold` just the same.)
///
/// [`ColumnRetention::All`] is mandatory here and takes no caller knob: every
/// caller pairs this with [`cascade_marginalize_in_apply`], which `take`s the
/// column of EVERY level in the walked subtree to install it as that level's
/// marginal store. Frontier release would free exactly those columns.
pub(super) fn ensure_level_counts<F: StreamPayload>(
    li: usize,
    vtree: &crate::vtree::Vtree,
    levels: &[TddLevel],
    computed: &mut [Option<F::Col<ApplyBudget>>],
    ws: Option<&WeightStore>,
) -> Result<(), ApplyError> {
    let zero = F::zero(ws);
    ensure_fold_walk::<F, ApplyBudget, _, _>(
        li,
        vtree,
        levels,
        computed,
        &zero,
        &|i| levels[i].is_marginal(),
        &|lvl, i, l_i, r_i, computed| {
            F::fold_node(lvl, i, l_i, r_i, vtree, levels, computed, &zero, ws)
        },
        ColumnRetention::All,
    )
}

/// Cascade marginalization through every explicit non-leaf descendant of
/// `li`, bottom-up. Each level's column must already be populated in
/// `computed` (call [`ensure_level_counts`] first). Mirrors
/// `compile.rs::cascade_marginalize` but operates on the in-flight `levels`
/// slice during apply rather than a finished TDD.
///
/// Soundness: bottom-up order satisfies `assert_can_make_marginal` at each
/// call site (by the time we marginalize `li`, both children of `li` are
/// already marginal or leaves). The caller is responsible for ensuring `li`
/// itself is a sound streaming target (no future references) — within the
/// apply gate this is guaranteed because `marginalize_targets[li] == true` flags
/// any descendant we'd touch, and the schedule monotonicity (a parent's
/// streaming step is at least as late as any descendant's) covers descendants
/// that aren't in this apply call's target set but were targets of an earlier
/// sub-batch and only stayed explicit because of the width gate.
pub(super) fn cascade_marginalize_in_apply<F: StreamPayload>(
    li: usize,
    vtree: &crate::vtree::Vtree,
    levels: &mut [TddLevel],
    computed: &mut [Option<F::Col<ApplyBudget>>],
    mut ws: Option<&mut WeightStore>,
) {
    if vtree.node(VtreeIdx(li as u32)).is_leaf() || levels[li].is_marginal() {
        return;
    }
    let (left, right) = vtree.children(VtreeIdx(li as u32));
    cascade_marginalize_in_apply::<F>(left.idx(), vtree, levels, computed, ws.as_deref_mut());
    cascade_marginalize_in_apply::<F>(right.idx(), vtree, levels, computed, ws.as_deref_mut());
    let Some(col) = computed[li].take() else {
        // No cached column: `ensure_level_counts` did not visit this branch
        // (cells structurally unreachable from the target's pair lists). Bail
        // rather than fabricate values.
        return;
    };
    types::assert_can_make_marginal(levels, vtree, VtreeIdx(li as u32));
    F::store_level(levels, li, col, ws);
}

/// Budget-tracked clone of a value slice. Mirrors `budget_reserve_exact`:
/// reserves the exact destination capacity through the fallible path (so an
/// `RLIMIT_AS` failure or a tripped soft budget surfaces as
/// `Err(OverBudget)` rather than a `handle_alloc_error` process abort), then
/// fills it without a reallocation. Same abort class as the
/// `try_with_capacity` on the output column at the dense
/// [`build_stream_state`] caller (`mc2025_track1_057`, 2026-05-21): a
/// weight-marginal level can carry hundreds of millions of slots.
///
/// ONE caller left — [`WeightFold::child_view`]'s `WeightStore` case, whose
/// column must be copied out from under the output level's `&mut`. Every other
/// child column is read in place through [`CountRef`] / `Cow::Borrowed`; do not
/// reintroduce a copy there, it is the whole point of the borrowed view.
#[inline]
fn try_clone_counts<T: Clone>(src: &[T]) -> Result<Vec<T>, ApplyError> {
    let mut dst = Vec::new();
    budget_reserve_exact(&mut dst, src.len())?;
    dst.extend(src.iter().cloned());
    Ok(dst)
}

// ── Integer payload (`--mc` model counts) ────────────────────────────────────

/// Resolve one child ref to a count read, on the in-flight `levels` slice.
/// One lazy reader for both widths — a `Big` read hands back the borrowed
/// `BigUint` directly, with no separate overflow fetch.
///
/// Self-describing decode under the bit-30-clear==slot polarity: bit 30 alone
/// disambiguates, no parent-side inline provenance needed.
///   bit-30 SET   → inline count value (strip the tag; ≤ MARG_INLINE_MAX, so
///                  never an overflow sentinel).
///   bit-30 CLEAR → bare slot index into the store (a mid-apply pre-tag read
///                  of an in-flight pair's `.idx()` is a bare node index,
///                  which IS its slot index — unambiguously a slot, never
///                  misread as a count). This is what fixes the #63
///                  over/undercount.
#[inline]
fn read_level_count<'a>(
    li: usize,
    ki: usize,
    vtree: &crate::vtree::Vtree,
    levels: &'a [TddLevel],
    computed: &'a [Option<CountVec<ApplyBudget>>],
) -> CountRead<'a> {
    if let Some(ic) = &levels[li].marginal_counts {
        let raw = ki as u32;
        if raw & MARG_OVERFLOW_TAG != 0 {
            return CountRead::Fast((raw & MARG_VALUE_MASK) as u128);
        }
        // Bare slot — mask is a no-op (bit 30 clear), kept for parity with the
        // tagged-read discipline. See task #43.
        let idx = decode_marg_coord(raw, MARG_VALUE_MASK) as usize;
        let v = ic[idx];
        if v != STREAM_OVERFLOW {
            return CountRead::Fast(v);
        }
        if let Some(bv) = levels[li]
            .marginal_counts_big
            .as_ref()
            .and_then(|ib| ib.get(idx))
        {
            return CountRead::Big(bv);
        }
        // Belt-and-braces fallback: the computed-value side may hold the big.
        if let Some(bv) = computed[li].as_ref().and_then(|cv| cv.big_val(ki)) {
            return CountRead::Big(bv);
        }
        unreachable!("big count not available for level {} node {}", li, ki);
    }
    if let Some(c) = &computed[li] {
        return c.get(ki);
    }
    if vtree.node(VtreeIdx(li as u32)).is_leaf() {
        return CountRead::Fast(match LeafLabel::from_idx(ki) {
            LeafLabel::Zero => 0,
            LeafLabel::One => 2,
            LeafLabel::Pos | LeafLabel::Neg => 1,
        });
    }
    unreachable!("counts not available for level {}", li);
}

/// Sum `Σ counts_left[p.left] * counts_right[p.right]` over `pairs`. Returns
/// `Count::Fast(total)` for the common u128 case, `Count::Big(big_total)` when
/// u128 overflowed (or one of the inputs is already at `STREAM_OVERFLOW`).
/// Monomorphized fast-path read of one child count for the all-u64 fold. `MARG`
/// is the child level's `is_marg` flag, lifted to a const so the per-pair branch
/// folds away at compile time:
/// - `MARG=false` (non-marginal): the ref is a bare index — `decode_marg_coord`
///   with `u32::MAX` is the identity, so the read is a single load, no tag test.
/// - `MARG=true` (marginal): a bit-30-SET ref is an inline count (`≤
///   MARG_INLINE_MAX`); otherwise the bare slot indexes `counts`.
///
/// # Safety
/// `get_unchecked` carries the SAME in-bounds guarantee the prior checked
/// `counts[idx]` relied on: the fold visits only live cells whose child refs
/// decode to valid `counts` indices (dead / bit-31 sentinel refs are pruned
/// before the fold). A `debug_assert` re-checks the bound in test/debug builds.
#[inline(always)]
unsafe fn read_fast<const MARG: bool>(raw: u32, c: &StreamChildCounts<'_>) -> u128 {
    if MARG {
        if raw & MARG_OVERFLOW_TAG != 0 {
            (raw & MARG_VALUE_MASK) as u128
        } else {
            let idx = decode_marg_coord(raw, MARG_VALUE_MASK) as usize;
            debug_assert!(idx < c.col.len(), "read_fast marg slot OOB");
            unsafe { *c.col.fast_slice().get_unchecked(idx) }
        }
    } else {
        // mask == u32::MAX ⇒ decode is identity ⇒ idx == raw.
        let idx = raw as usize;
        debug_assert!(idx < c.col.len(), "read_fast non-marg OOB");
        unsafe { *c.col.fast_slice().get_unchecked(idx) }
    }
}

/// The all-u64 two-accumulator fold, monomorphized on each side's `is_marg`
/// flag (`LM`/`RM`). Returns `None` on u128 overflow in either lane or the final
/// combine — the caller then re-folds the same pairs through the exact BigUint
/// path, so the result is identical, just a rare slow fallback. Branchless inner
/// loop: the mask/tag test is gone (compile-time via `read_fast`), the bounds
/// check is gone (`get_unchecked`), and even/odd products retire into two
/// independent `adc` chains (the carry-chain break).
#[inline(always)]
fn fold_fast<const LM: bool, const RM: bool>(
    pairs: &[InputPair],
    left: &StreamChildCounts<'_>,
    right: &StreamChildCounts<'_>,
) -> Option<u128> {
    let mut t0: u128 = 0;
    let mut t1: u128 = 0;
    let mut it = pairs.chunks_exact(2);
    for c in it.by_ref() {
        // SAFETY: see `read_fast` — fold pairs are live cells with in-bounds refs.
        let (l0, r0, l1, r1) = unsafe {
            (
                read_fast::<LM>(c[0].left.0, left),
                read_fast::<RM>(c[0].right.0, right),
                read_fast::<LM>(c[1].left.0, left),
                read_fast::<RM>(c[1].right.0, right),
            )
        };
        let p0 = (l0 as u64 as u128) * (r0 as u64 as u128);
        let p1 = (l1 as u64 as u128) * (r1 as u64 as u128);
        match (t0.checked_add(p0), t1.checked_add(p1)) {
            (Some(a), Some(b)) => {
                t0 = a;
                t1 = b;
            }
            _ => return None,
        }
    }
    for pair in it.remainder() {
        // SAFETY: see `read_fast`.
        let (lc, rc) = unsafe {
            (
                read_fast::<LM>(pair.left.0, left),
                read_fast::<RM>(pair.right.0, right),
            )
        };
        let prod = (lc as u64 as u128) * (rc as u64 as u128);
        t0 = t0.checked_add(prod)?;
    }
    t0.checked_add(t1)
}

pub(super) fn compute_cell_count(
    pairs: &[InputPair],
    left: &StreamChildCounts<'_>,
    right: &StreamChildCounts<'_>,
) -> Count {
    // Tag-at-creation: a marginal child's refs may carry the bit-30 slot tag —
    // strip it before indexing. Non-marginal/leaf children use a plain index
    // (no mask). `decode_marg_coord` is a no-op on a bare ref.
    let left_mask = if left.is_marg { MARG_VALUE_MASK } else { u32::MAX };
    let right_mask = if right.is_marg { MARG_VALUE_MASK } else { u32::MAX };
    // Returns (count_value_or_sentinel, slot_index). For an inline bit-30-SET
    // ref the value IS the count and the slot index is unused (such a count is
    // ≤ MARG_INLINE_MAX, never STREAM_OVERFLOW, so the big path never
    // dereferences the sentinel index).
    #[inline(always)]
    fn read_marg_count(raw: u32, c: &StreamChildCounts<'_>, mask: u32) -> (u128, usize) {
        // Self-describing under the bit-30-clear==slot polarity: for a marginal
        // child (mask == MARG_VALUE_MASK) a bit-30-SET ref is an inline count; a
        // bit-30-CLEAR ref is a slot index (a fresh mid-apply grid index is a bare
        // node index = its slot, decoded correctly here — the #63 fix).
        if mask != u32::MAX && raw & MARG_OVERFLOW_TAG != 0 {
            ((raw & MARG_VALUE_MASK) as u128, usize::MAX)
        } else {
            let idx = decode_marg_coord(raw, mask) as usize;
            (c.col.fast_val(idx), idx)
        }
    }
    let mut total: u128 = 0;
    let mut overflowed = false;
    if left.col.all_u64() && right.col.all_u64() {
        // Compute-bound fast path. Every read is ≤ u64::MAX (slots certified by
        // all_u64; inline-tagged refs are ≤ MARG_INLINE_MAX), so the product is
        // a `u64×u64→u128` widening multiply — a single `mul` that LLVM emits
        // from the zero-extended operands and that can never overflow the u128
        // product. No per-pair STREAM_OVERFLOW check (the sentinel can't appear)
        // and no u128 `checked_mul` (the dependency-chain-heavy cross-product
        // sequence is gone).
        //
        // The loop is monomorphized on each side's `is_marg` flag (`fold_fast`
        // dispatch): the loop-invariant mask/tag branch and the `counts[idx]`
        // bounds check both fold away, leaving load → widening `mul` →
        // two-accumulator `adc`. The two independent running totals break the
        // serial `adc` carry chain (even/odd products retire into separate
        // chains). Overflow in either lane or the final combine returns `None`
        // and falls to the shared BigUint re-loop below (re-reads all pairs from
        // scratch) — identical result, just a rare slow fallback.
        let res = match (left.is_marg, right.is_marg) {
            (false, false) => fold_fast::<false, false>(pairs, left, right),
            (false, true) => fold_fast::<false, true>(pairs, left, right),
            (true, false) => fold_fast::<true, false>(pairs, left, right),
            (true, true) => fold_fast::<true, true>(pairs, left, right),
        };
        match res {
            Some(t) => total = t,
            None => overflowed = true,
        }
    } else {
        for pair in pairs {
            let (lc, _) = read_marg_count(pair.left.0, left, left_mask);
            let (rc, _) = read_marg_count(pair.right.0, right, right_mask);
            if lc == STREAM_OVERFLOW || rc == STREAM_OVERFLOW {
                overflowed = true;
                break;
            }
            match lc.checked_mul(rc).and_then(|p| total.checked_add(p)) {
                Some(t) => total = t,
                None => { overflowed = true; break; }
            }
        }
    }
    if !overflowed {
        // `Count::from_u128` owns the exact-max promotion (total == u128::MAX
        // == STREAM_OVERFLOW must not be stored as a fast value).
        return Count::from_u128(total);
    }
    let big_total = {
        let mut bt = num_bigint::BigUint::ZERO;
        // Branch the inner product by which inputs are still u128 vs already
        // overflowed. The u128/u128 fast path skips BigUint multiply entirely
        // (just an AddAssign<u128>); the mixed paths use scalar BigUint
        // multiply (one alloc for the product, no `BigUint::from(u128)`
        // intermediate); only the both-BigUint case takes the full bigint
        // multiply. Callgrind on mc2021_track1_128 showed mul3+alloc/free
        // dominating runtime — this trims the alloc count per pair from
        // 1–3 BigUints down to 0–1.
        for pair in pairs {
            let (lc_val, li) = read_marg_count(pair.left.0, left, left_mask);
            let (rc_val, ri) = read_marg_count(pair.right.0, right, right_mask);
            let lc_is_big = lc_val == STREAM_OVERFLOW;
            let rc_is_big = rc_val == STREAM_OVERFLOW;
            match (lc_is_big, rc_is_big) {
                (false, false) => {
                    // Both inputs u128; the running BigUint sum is needed
                    // only because aggregate `bt` already overflowed.
                    if let Some(prod) = lc_val.checked_mul(rc_val) {
                        bt += prod;
                    } else {
                        // u128 × u128 overflows: 128-bit BigUint then scalar.
                        let mut tmp = num_bigint::BigUint::from(lc_val);
                        tmp *= rc_val;
                        bt += tmp;
                    }
                }
                (true, false) => {
                    let lc_ref = left.col.big_val(li)
                        .expect("missing BigUint for overflowed left count");
                    bt += lc_ref * rc_val;
                }
                (false, true) => {
                    let rc_ref = right.col.big_val(ri)
                        .expect("missing BigUint for overflowed right count");
                    bt += rc_ref * lc_val;
                }
                (true, true) => {
                    let lc_ref = left.col.big_val(li)
                        .expect("missing BigUint for overflowed left count");
                    let rc_ref = right.col.big_val(ri)
                        .expect("missing BigUint for overflowed right count");
                    bt += lc_ref * rc_ref;
                }
            }
        }
        bt
    };
    Count::Big(big_total)
}

impl StreamPayload for IntFold {
    /// Always borrowed — the integer child column is either a level's raw
    /// marginal arrays, a `computed` scratch column, or the static leaf slots.
    type ChildCol<'a> = CountRef<'a>;

    #[inline]
    fn zero(_ws: Option<&WeightStore>) -> Count {
        Count::Fast(0)
    }

    #[inline]
    fn fold_node(
        lvl: usize,
        i: usize,
        l_i: usize,
        r_i: usize,
        vtree: &crate::vtree::Vtree,
        levels: &[TddLevel],
        computed: &[Option<CountVec<ApplyBudget>>],
        _zero: &Count,
        _ws: Option<&WeightStore>,
    ) -> Count {
        // Iterate the `&[InputPair]` slice directly (compiler-vectorizable;
        // in-flight levels are never packed).
        IntFold::fold(
            levels[lvl].pairs_of_idx(i).iter().copied(),
            |k| read_level_count(l_i, k, vtree, levels, computed),
            |k| read_level_count(r_i, k, vtree, levels, computed),
        )
    }

    fn child_view<'a>(
        li: usize,
        vtree: &crate::vtree::Vtree,
        level: &'a TddLevel,
        computed: &'a [Option<CountVec<ApplyBudget>>],
        _ws: Option<&WeightStore>,
    ) -> Result<StreamChildCounts<'a>, ApplyError> {
        let is_marg = level.marginal_counts.is_some();
        // Raw-storage sources (`marginal_counts`/`marginal_counts_big` on the level)
        // are viewed through `CountRef::from_parts_scanned` (u64-fit certificate
        // scanned over the stored slots; STREAM_OVERFLOW = u128::MAX fails the scan,
        // so all_u64 ⇒ no overflow sentinel present). A `computed` source is already
        // a `CountVec` and lends its own incrementally maintained certificate. The
        // sources are mutually exclusive: `ensure_level_counts` only fills `computed`
        // for non-marginal levels, and `cascade_marginalize_in_apply` moves an entry
        // into level storage when the level becomes marginal.
        //
        // No arm allocates: every one borrows storage that already exists.
        let col = if vtree.node(VtreeIdx(li as u32)).is_leaf()
            && level.marginal_counts.as_ref().is_some_and(|c| c.is_empty())
        {
            // Marginal LEAF (leaf marginalization): empty store, all counts inline at
            // the parent. Its conceptual slots are the fixed leaf labels — return
            // them so any stray bare-label ref (slot 0/1/2) still decodes correctly;
            // inline refs bypass this column entirely. This integer-side fixed-slot
            // rule has NO weighted counterpart (see `WeightFold::child_view`,
            // which resolves the semiring leaf bases instead).
            CountRef::from_parts_scanned(&LEAF_COUNTS, level.marginal_counts_big.as_ref())
        } else if let Some(ic) = &level.marginal_counts {
            CountRef::from_parts_scanned(ic, level.marginal_counts_big.as_ref())
        } else if let Some(c) = &computed[li] {
            c.as_count_ref()
        } else if vtree.node(VtreeIdx(li as u32)).is_leaf() {
            CountRef::from_parts_scanned(&LEAF_COUNTS, None)
        } else {
            unreachable!("IntFold::child_view: no counts for level {}", li);
        };

        Ok(StreamChild { col, is_marg })
    }

    #[inline(always)]
    fn fold_cell(
        pairs: &[InputPair],
        left: &StreamChildCounts<'_>,
        right: &StreamChildCounts<'_>,
        _ws: Option<&WeightStore>,
    ) -> Count {
        compute_cell_count(pairs, left, right)
    }

    #[inline]
    fn store_level(
        levels: &mut [TddLevel],
        li: usize,
        col: CountVec<ApplyBudget>,
        _ws: Option<&mut WeightStore>,
    ) {
        // No side-table reshaping at the handoff: `CountVec` and `TddLevel`
        // hold the SAME sparse slot-keyed overflow table, so this is a move.
        // (The dense predecessor had to pad an append-built column's
        // trailing-lazy `big` out to `fast.len()` here.)
        let (fast, big) = col.into_parts();
        // EMIT-SITE DEDUP IS FORBIDDEN HERE.
        //
        // Eager value-dedup of apply-emit-born stores seeds a feedback loop on
        // large instances: birth-shared slot refs → boundary twin merges concat
        // pair lists → duplicate (X,c) pairs → p-fusion sums them, minting new
        // count slots → wider marginal stores → larger apply grids → 17× explicit-
        // node explosion (58M vs 3.3M peak on mc2020_track1_192, R204).
        //
        // C3 for emit-born stores is established instead at post-tagger slot-prune
        // (`prune_marg_slots`, tididi/src/tdd/minimize/slot_prune.rs), where small
        // counts are already inline refs and only genuinely large counts remain as
        // slots — making birth-shared refs impossible.
        levels[li].make_marginal(fast, big);
    }
}

// ── Weighted payload (`--weighted` algebraic model counting) ─────────────────
//
// The weighted hooks swap the `u128`/`BigUint` model-count payload for an exact
// `BigRational` semiring value carried in the external `WeightStore` (installed
// thread-local for the duration of the compile). BigRational doesn't overflow,
// so there is NO big/overflow second pass — a single clean fold. The per-child
// value lookup is the weighted analogue of `read_level_count`: read from the
// `WeightStore` for a weight-marginal child, the semiring leaf base for a leaf,
// else the per-batch `computed` scratch.

/// Weighted analogue of [`read_level_count`] /
/// `compile_marginalize::read_marginal_weight`, operating on the in-flight
/// `levels` slice. Resolves a child node's exact semiring value. Returns
/// `Cow`: the per-batch `computed` read borrows (no clone); store slots and
/// leaf bases clone.
#[inline]
fn read_level_weight<'a>(
    child: usize,
    node_ref: usize,
    vtree: &crate::vtree::Vtree,
    levels: &[TddLevel],
    computed_weights: &'a [Option<Vec<WeightVal>>],
    ws: &WeightStore,
) -> std::borrow::Cow<'a, WeightVal> {
    if levels[child].is_weight_marginal() {
        let slot = match MargRef::from_raw(node_ref as u32) {
            MargRef::Inline(_) => unreachable!("weighted marg-side refs are bare slots"),
            MargRef::Slot(s) => s as usize,
        };
        return std::borrow::Cow::Owned(
            ws.level(child).expect("weight-marginal level set")[slot].clone(),
        );
    }
    if let crate::vtree::VtreeNode::Leaf { var, .. } = *vtree.node(VtreeIdx(child as u32)) {
        return std::borrow::Cow::Owned(ws.leaf_val(var, LeafLabel::from_idx(node_ref)));
    }
    if let Some(w) = &computed_weights[child] {
        return std::borrow::Cow::Borrowed(&w[node_ref]);
    }
    unreachable!("weighted value not available for level {}", child);
}

/// Weighted analogue of [`compute_cell_count`]. `Σ left[idx(p.left)] * right[idx(p.right)]`.
/// No overflow handling.
pub(super) fn compute_cell_weight(
    pairs: &[InputPair],
    left: &[WeightVal],
    right: &[WeightVal],
    left_is_marg: bool,
    right_is_marg: bool,
    ws: &WeightStore,
) -> WeightVal {
    // Resolve a marg/non-marg ref to its value; both index the snapshot by
    // reference.
    #[inline(always)]
    fn resolve<'a>(raw: u32, is_marg: bool, snap: &'a [WeightVal]) -> std::borrow::Cow<'a, WeightVal> {
        if is_marg {
            match MargRef::from_raw(raw) {
                MargRef::Inline(_) => unreachable!("weighted marg-side refs are bare slots"),
                MargRef::Slot(s) => std::borrow::Cow::Borrowed(&snap[s as usize]),
            }
        } else {
            std::borrow::Cow::Borrowed(&snap[raw as usize])
        }
    }
    WeightFold::fold(
        pairs.iter().copied(),
        |k| resolve(k as u32, left_is_marg, left),
        |k| resolve(k as u32, right_is_marg, right),
        ws.wzero(),
    )
}

impl StreamPayload for WeightFold {
    /// `Cow`, not a plain borrow: the `computed_weights` scratch and nothing
    /// else can lend a reference. The `WeightStore` column must be copied out
    /// from under the output level's `&mut`, and the semiring leaf bases are
    /// computed on the spot, so those two arms own.
    type ChildCol<'a> = std::borrow::Cow<'a, [WeightVal]>;

    fn zero(ws: Option<&WeightStore>) -> WeightVal {
        ws.expect("weighted apply without a weight store").wzero()
    }

    #[inline]
    fn fold_node(
        lvl: usize,
        i: usize,
        l_i: usize,
        r_i: usize,
        vtree: &crate::vtree::Vtree,
        levels: &[TddLevel],
        computed: &[Option<Vec<WeightVal>>],
        zero: &WeightVal,
        ws: Option<&WeightStore>,
    ) -> WeightVal {
        let ws = ws.expect("weighted apply without a weight store");
        WeightFold::fold(
            levels[lvl].pairs_of_idx(i).iter().copied(),
            |k| read_level_weight(l_i, k, vtree, levels, computed, ws),
            |k| read_level_weight(r_i, k, vtree, levels, computed, ws),
            zero.clone(),
        )
    }

    fn child_view<'a>(
        li: usize,
        vtree: &crate::vtree::Vtree,
        level: &'a TddLevel,
        computed_weights: &'a [Option<Vec<WeightVal>>],
        ws: Option<&WeightStore>,
    ) -> Result<StreamChild<'a, WeightFold>, ApplyError> {
        let ws = ws.expect("weighted apply without a weight store");
        if level.is_weight_marginal() {
            // Keyed on THIS level's own marginality flag, not on whether the
            // `WeightStore` happens to hold a column for this vtree
            // index — so a level that is structural HERE never decodes against
            // another `Tdd`'s values. For a weight-marginal LEAF the two agree by
            // construction: the pin invariant
            // (`marginalize::marginalize_leaf_weighted`) keeps its column equal,
            // slot for slot, to the label-ordered `leaf_val` triple the structural
            // branch below builds.
            //
            // The ONE column that must still be copied: the store is held apart
            // from the level slice for the whole apply, so its column cannot be
            // lent alongside the output level's `&mut`. Fallible for the same
            // reason the integer path used to be.
            let col = try_clone_counts(ws.level(li).expect("weight-marginal level set"))?;
            return Ok(StreamChild { col: std::borrow::Cow::Owned(col), is_marg: true });
        }
        if let crate::vtree::VtreeNode::Leaf { var, .. } = *vtree.node(VtreeIdx(li as u32)) {
            // LEAF_WIDTH = 3, ordered {One, Pos, Neg} per LeafLabel::from_idx —
            // weighted analogue of `IntFold::child_view`'s `LEAF_COUNTS`, but
            // resolving the semiring leaf bases rather than fixed counts. Built by
            // `marginalize::leaf_column_vals`, the ONE definition of that triple
            // (the same one `marginalize_leaf_weighted` pins into the store), so
            // the structural and marginal branches cannot drift apart. Fixed
            // 3-element alloc, so no budget reservation (the bases are not
            // `const`, hence no static to borrow as the integer twin does).
            let col: Vec<WeightVal> =
                crate::tdd::transform::unary::marginalize::leaf_column_vals(ws, var);
            return Ok(StreamChild { col: std::borrow::Cow::Owned(col), is_marg: false });
        }
        let col = computed_weights[li]
            .as_ref()
            .expect("WeightFold::child_view: no values for level");
        Ok(StreamChild { col: std::borrow::Cow::Borrowed(col), is_marg: false })
    }

    #[inline(always)]
    fn fold_cell(
        pairs: &[InputPair],
        left: &StreamChild<'_, WeightFold>,
        right: &StreamChild<'_, WeightFold>,
        ws: Option<&WeightStore>,
    ) -> WeightVal {
        compute_cell_weight(
            pairs,
            &left.col,
            &right.col,
            left.is_marg,
            right.is_marg,
            ws.expect("weighted apply without a weight store"),
        )
    }

    #[inline]
    fn store_level(
        levels: &mut [TddLevel],
        li: usize,
        col: Vec<WeightVal>,
        ws: Option<&mut WeightStore>,
    ) {
        // Structurally the integer commit's mirror (raw
        // `make_marginal_weighted_with_slots`, no parent contract-dirty marking —
        // the shared level-state machine establishes C3 at slot-prune, which runs
        // in weighted mode too via `prune_marg_slots_generic::<WeightFold>`; only
        // the integer count-preservation localizer around it is gated off) but the
        // payload goes to the `WeightStore`.
        let slots = col.len() as u32;
        levels[li].make_marginal_weighted_with_slots(slots);
        ws.expect("weighted apply without a weight store").set_level(li, col);
    }
}

#[cfg(test)]
#[path = "stream_overflow_validation_tests.rs"]
mod overflow_validation_tests;

/// Single source of truth for the streaming-eligibility gate: a level streams
/// its marginal iff it is a marginalize target AND the streaming gate is on
/// (`TIDIDI_BOTHMARG_NOCOLLAPSE` unset; off ⇒ don't stream, materialize +
/// post-apply `marginalize_batch`). Consulted per level by the emit-growth mode
/// decision (the `stream_marginal` local in the driver loop) and by
/// [`build_stream_state`]'s setup; the commit then keys off `stream_state` being
/// `Some` rather than re-reading the predicate. Do NOT re-inline the predicate
/// at a call site — it is cheap, and the cold per-level path can afford the
/// call.
#[inline]
pub(super) fn stream_marginal_eligible(marginalize_targets: Option<&[bool]>, t_idx: usize) -> bool {
    marginalize_targets.is_some_and(|t| t[t_idx])
        && bothmarg_collapse_enabled()
}

/// Phase: streaming-marginal setup (inside the `for (t, left, right) in vtree.internal_bottomup()` loop).
///
/// Prepares the children and opens the [`StreamLevelState`] output column if
/// this level is a streaming target. Called after the NxM dead-pair pre-filter
/// block, before the dedicated marginal-child dispatch. This is the ONE place
/// the value kind is chosen at runtime; both arms run the same generic
/// [`open_stream_output`].
///
/// Runs while the whole `levels` slice is still mutably available — the
/// cascade re-marginalizes arbitrary descendants, not just the two children —
/// and returns nothing that borrows it. The child columns are attached later,
/// per row loop, by [`attach_children`].
///
/// In `--weighted` mode streaming is ON (it carries BigRational values into the
/// external `WeightStore`) UNLESS the batch toggle (`TIDIDI_WEIGHTED_NOSTREAM`)
/// forces the BATCH-only path. Not weighted → integer streaming, unchanged.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(super) fn build_stream_state(
    t_idx: usize,
    left_idx: usize,
    right_idx: usize,
    k1: usize,
    k2: usize,
    marginalize_targets: Option<&[bool]>,
    vtree: &crate::vtree::Vtree,
    levels: &mut Vec<TddLevel>,
    stream_computed: &mut Vec<Option<CountVec<ApplyBudget>>>,
    stream_computed_weights: &mut Vec<Option<Vec<WeightVal>>>,
    ws: Option<&mut WeightStore>,
) -> Result<Option<StreamLevelState>, ApplyError> {
    if !stream_marginal_eligible(marginalize_targets, t_idx) {
        return Ok(None);
    }
    if ws.is_some() {
        Ok(Some(StreamLevelState::Weighted(open_stream_output::<WeightFold>(
            left_idx, right_idx, k1, k2, vtree, levels, stream_computed_weights, ws,
        )?)))
    } else {
        Ok(Some(StreamLevelState::Int(open_stream_output::<IntFold>(
            left_idx, right_idx, k1, k2, vtree, levels, stream_computed, None,
        )?)))
    }
}

/// The value-kind-generic streaming setup: compute both children's columns,
/// cascade-marginalize any still-explicit non-leaf descendant, and open the
/// output column.
///
/// Step 2 (the cascade) restores the structural contract that streaming this
/// level requires its descendants to be marginal-or-leaf. The width gate at
/// lower levels may have left descendants explicit; those descendants would
/// have been targets of an earlier sub-batch (or this one), so marginalizing
/// them now is sound — no future clause references them.
///
/// The output column's initial capacity is bounded by alive cells (≤ k1*k2) but
/// typically far fewer — ask for `k1.max(k2)` and let it grow. That reservation
/// must be FALLIBLE: `k1.max(k2)` can reach ~1B on extreme widths, where an
/// infallible `Vec::with_capacity` triggered the 16 GiB single-alloc abort on
/// mc2025_track1_057 (cap-trigger probe, 2026-05-21). `?` propagates
/// `OverBudget` so v-split recovery engages instead of process::abort.
fn open_stream_output<F: StreamPayload>(
    left_idx: usize,
    right_idx: usize,
    k1: usize,
    k2: usize,
    vtree: &crate::vtree::Vtree,
    levels: &mut [TddLevel],
    computed: &mut Vec<Option<F::Col<ApplyBudget>>>,
    mut ws: Option<&mut WeightStore>,
) -> Result<F::Col<ApplyBudget>, ApplyError> {
    // 1. Compute the fold column for every non-leaf non-marginal descendant.
    ensure_level_counts::<F>(left_idx, vtree, levels, computed, ws.as_deref())?;
    ensure_level_counts::<F>(right_idx, vtree, levels, computed, ws.as_deref())?;
    // 2. Cascade-marginalize any still-explicit non-leaf descendant.
    cascade_marginalize_in_apply::<F>(left_idx, vtree, levels, computed, ws.as_deref_mut());
    cascade_marginalize_in_apply::<F>(right_idx, vtree, levels, computed, ws.as_deref_mut());
    F::try_with_capacity::<ApplyBudget>(k1.max(k2))
}

/// Phase: streaming row loop (per value kind, per route).
///
/// Binds the two child column views to this level's in-flight output column for
/// the duration of one row loop. The views borrow `left_level`/`right_level`
/// directly, which is why the driver loop splits `levels[t_idx]` and the two
/// child slots apart up front: `t` and its two vtree children are three
/// distinct nodes of a tree, so the split is total and the output level stays
/// exclusively borrowed while the children are read.
///
/// The returned state must not outlive the row loop — the level tail retakes
/// `&mut levels` to commit [`StreamLevelState`], which owns the column this
/// only borrows.
pub(super) fn attach_children<'a, F: StreamPayload>(
    left_idx: usize,
    right_idx: usize,
    vtree: &crate::vtree::Vtree,
    left_level: &'a TddLevel,
    right_level: &'a TddLevel,
    computed: &'a [Option<F::Col<ApplyBudget>>],
    counts: &'a mut F::Col<ApplyBudget>,
    ws: Option<&'a WeightStore>,
) -> Result<StreamState<'a, F>, ApplyError> {
    Ok(StreamState {
        left: F::child_view(left_idx, vtree, left_level, computed, ws)?,
        right: F::child_view(right_idx, vtree, right_level, computed, ws)?,
        counts,
        ws,
    })
}

/// Phase: streaming commit (inside the `for (t, left, right) in vtree.internal_bottomup()` loop).
///
/// Converts the completed [`StreamLevelState`] into a marginal level. The
/// caller keeps the `if let Some(st) = stream_state.take()` guard; this
/// function receives the unwrapped state. C3 is established later by
/// `prune_marg_slots` — emit-site dedup is forbidden, see
/// [`IntFold::store_level`].
///
/// Marginalization precondition (checked once, before the value-kind branch):
/// both children of `t` must already be marginal (or leaves). For
/// streaming-marginal during apply, the marginalize_schedule guarantees
/// descendants of `t` in the schedule are processed first (apply runs
/// bottom-up).
#[inline(always)]
pub(super) fn commit_stream_state(
    st: StreamLevelState,
    t: VtreeIdx,
    t_idx: usize,
    vtree: &crate::vtree::Vtree,
    levels: &mut Vec<TddLevel>,
    ws: Option<&mut WeightStore>,
) {
    types::assert_can_make_marginal(levels, vtree, t);
    match st {
        StreamLevelState::Int(counts) => IntFold::store_level(levels, t_idx, counts, None),
        StreamLevelState::Weighted(counts) => WeightFold::store_level(levels, t_idx, counts, ws),
    }
}
