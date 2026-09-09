//! The integer arm of the streaming fold.

use super::*;

/// Budget-tracked clone of a value slice. Mirrors `budget_reserve_exact`:
/// reserves the exact destination capacity through the fallible path (so an
/// `RLIMIT_AS` failure or a tripped soft budget surfaces as
/// `Err(OverBudget)` rather than a `handle_alloc_error` process abort), then
/// fills it without a reallocation. Same abort class as the
/// `try_with_capacity` on the output column at the dense
/// [`build_stream_state`] caller: a weight-marginal level can carry hundreds of
/// millions of slots.
///
/// ONE caller left — [`WeightFold::child_view`]'s `WeightStore` case, whose
/// column must be copied out from under the output level's `&mut`. Every other
/// child column is read in place through [`CountRef`] / `Cow::Borrowed`; do not
/// reintroduce a copy there, it is the whole point of the borrowed view.
#[inline]
pub(crate) fn try_clone_counts<T: Clone>(eng: &Engine, src: &[T]) -> Result<Vec<T>, ApplyError> {
    let lim = eng.limits();
    let mut dst = Vec::new();
    lim.reserve_exact(&mut dst, src.len())?;
    dst.extend(src.iter().cloned());
    Ok(dst)
}

// ── Integer payload (model counts) ──────────────────────────────────────────


/// Sum `Σ counts_left[p.left] * counts_right[p.right]` over `pairs`. Returns
/// `Count::Fast(total)` for the common u128 case, `Count::Big(big_total)` when
/// u128 overflowed (or one of the inputs is already at `STREAM_OVERFLOW`).
/// Monomorphized fast-path read of one child count for the all-u64 fold. `MARG`
/// is the child level's `is_marg` flag, lifted to a const so the per-pair branch
/// folds away at compile time:
/// - `MARG=false` (non-marginal): the ref is a bare index, so the read is a
///   single load with no tag test.
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
        if let Some(c) = ValueRef::inline_count(raw) {
            c as u128
        } else {
            let idx = SideView::valued().coord(NodeIdx(raw)).idx();
            debug_assert!(idx < c.col.len(), "read_fast marg slot OOB");
            unsafe { *c.col.fast_slice().get_unchecked(idx) }
        }
    } else {
        // A structural side needs no decode: the ref is the index.
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
pub(crate) fn fold_fast<const LM: bool, const RM: bool>(
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

/// Read one pair side as `(count_value_or_sentinel, slot_index)`.
///
/// For an inline bit-30-SET ref the value IS the count and the slot index is
/// unused: such a count is at most `MARG_INLINE_MAX`, never `STREAM_OVERFLOW`,
/// so the big path never dereferences the sentinel index.
///
/// The polarity is self-describing: for a marginal child a bit-30-SET ref is an
/// inline count and a bit-30-CLEAR ref is a slot index (a fresh mid-apply grid
/// index is a bare node index, which is its slot, and decodes correctly here).
#[inline(always)]
fn read_marg_count(raw: u32, c: &StreamChildCounts<'_>, view: SideView) -> (u128, usize) {
    if view.is_valued()
        && let Some(c) = ValueRef::inline_count(raw)
    {
        (c as u128, usize::MAX)
    } else {
        let idx = view.coord(NodeIdx(raw)).idx();
        (c.col.fast_val(idx), idx)
    }
}


/// Re-sum every pair in arbitrary precision, once a u128 accumulation overflowed.
///
/// The inner product branches on which inputs are still u128 and which already
/// overflowed. The u128/u128 path skips BigUint multiplication entirely (just an
/// `AddAssign<u128>`); the mixed paths use a scalar BigUint multiply, one
/// allocation for the product and no `BigUint::from(u128)` intermediate; only
/// the both-big case takes a full bigint multiply. Profiling showed multiply
/// plus allocation dominating this loop, and the branch trims it from one to
/// three allocations per pair down to zero or one.
fn sum_pairs_big(
    pairs: &[InputPair],
    left: &StreamChildCounts<'_>,
    right: &StreamChildCounts<'_>,
    left_view: SideView,
    right_view: SideView,
) -> num_bigint::BigUint {
    let mut bt = num_bigint::BigUint::ZERO;
    for pair in pairs {
        let (lc_val, li) = read_marg_count(pair.left.0, left, left_view);
        let (rc_val, ri) = read_marg_count(pair.right.0, right, right_view);
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
}

pub(crate) fn compute_cell_count(
    pairs: &[InputPair],
    left: &StreamChildCounts<'_>,
    right: &StreamChildCounts<'_>,
) -> Count {
    // Tag-at-creation: a marginal child's refs may carry the bit-30 slot tag —
    // strip it before indexing. Non-marginal and leaf children index verbatim.
    let left_view = if left.is_marg { SideView::valued() } else { SideView::structural() };
    let right_view = if right.is_marg { SideView::valued() } else { SideView::structural() };
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
            let (lc, _) = read_marg_count(pair.left.0, left, left_view);
            let (rc, _) = read_marg_count(pair.right.0, right, right_view);
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
    Count::Big(sum_pairs_big(pairs, left, right, left_view, right_view))
}

impl ValueDomain for IntFold {
    /// Counts live inside the level itself, so the domain carries no state
    /// beside the diagram.
    type Store = ();

    /// Always borrowed — the integer child column is either a level's raw
    /// marginal arrays, a `computed` scratch column, or the static leaf slots.
    type ChildCol<'a> = CountRef<'a>;

    #[inline]
    fn zero(_store: &()) -> Count {
        Count::Fast(0)
    }

    fn weight_store(_store: &mut ()) -> Option<&mut WeightStore> {
        None
    }

    #[inline]
    fn fold_node<R: ReservePolicy>(
        lvl: usize,
        i: usize,
        l_i: usize,
        r_i: usize,
        vtree: &crate::vtree::Vtree,
        levels: &[TddLevel],
        computed: &[Option<CountVec<R>>],
        _zero: &Count,
        _store: &(),
    ) -> Count {
        IntFold::fold(
            levels[lvl].pairs_iter_of_idx(i),
            |k| crate::marginal::read_count(l_i, k, vtree, levels, computed),
            |k| crate::marginal::read_count(r_i, k, vtree, levels, computed),
        )
    }

    fn child_view<'a, R: ReservePolicy>(
        _eng: &Engine,
        li: usize,
        vtree: &crate::vtree::Vtree,
        level: &'a TddLevel,
        computed: &'a [Option<CountVec<R>>],
        _store: &(),
    ) -> Result<StreamChild<'a, IntFold>, ApplyError> {
        let is_marg = level.marginal_counts().is_some();
        // Raw-storage sources (`marginal_counts`/`marginal_counts_big` on the level)
        // are viewed through `CountRef::from_parts_scanned` (u64-fit certificate
        // scanned over the stored slots; STREAM_OVERFLOW = u128::MAX fails the scan,
        // so all_u64 ⇒ no overflow sentinel present). A `computed` source is already
        // a `CountVec` and lends its own incrementally maintained certificate. The
        // sources are mutually exclusive: the ensure walk only fills `computed`
        // for non-marginal levels, and the in-apply cascade moves an entry
        // into level storage when the level becomes marginal.
        //
        // No arm allocates: every one borrows storage that already exists.
        let col = if vtree.node(VtreeIdx(li as u32)).is_leaf()
            && level.marginal_counts().is_some_and(|c| c.is_empty())
        {
            // Marginal LEAF (leaf marginalization): empty store, all counts inline at
            // the parent. Its conceptual slots are the fixed leaf labels — return
            // them so any stray bare-label ref (slot 0/1/2) still decodes correctly;
            // inline refs bypass this column entirely. This integer-side fixed-slot
            // rule has NO weighted counterpart (see `WeightFold::child_view`,
            // which resolves the semiring leaf bases instead).
            CountRef::from_parts_scanned(&LEAF_COUNTS, level.marginal_counts_big())
        } else if let Some(ic) = level.marginal_counts() {
            CountRef::from_parts_scanned(ic, level.marginal_counts_big())
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
        _store: &(),
    ) -> Count {
        compute_cell_count(pairs, left, right)
    }

    #[inline]
    fn commit_in_flight<R: ReservePolicy>(
        levels: &mut [TddLevel],
        li: usize,
        col: CountVec<R>,
        _store: &mut (),
    ) {
        crate::marginal::install_int_column(levels, li, col);
    }

    /// Counts are deduped before they are installed, so the level is C3 — no
    /// two slots share a value — from birth rather than by a later canon pass.
    /// That is what mints new slot numbers, and why this domain returns a
    /// remap.
    fn install(
        tdd: &mut Tdd,
        t: InternalLevel,
        col: CountVec<RecoveryPanic>,
        _store: &mut (),
    ) -> Option<Vec<u32>> {
        let (fast, big) = col.into_parts();
        let (counts, big, remap) = crate::marginal::dedup_fresh_store(fast, big);
        tdd.levels[t.vtree_idx().idx()].make_marginal(counts, big);
        Some(remap)
    }

    fn sum_out_leaf(
        eng: &Engine,
        tdd: &mut Tdd,
        leaf: VtreeIdx,
        vtree: &crate::vtree::Vtree,
        _store: &mut (),
    ) {
        crate::marginal::marginalize_leaf_inline(eng, tdd, leaf, vtree);
    }

    /// Make every marg-side slot reference this pass persisted self-describing,
    /// once, at the pass's chokepoint.
    ///
    /// This path does not go through `apply_and_fallible`, so the end-of-apply
    /// tagger never runs on it, and canon's no-duplicate early return leaves
    /// untouched boundary references raw. The snapshot is what keeps the sweep
    /// off children that a PRIOR pass marginalized: those already carry inline
    /// counts, and re-resolving them as bare slots would misread them.
    fn end_sweep(tdd: &mut Tdd, was_frozen: &[bool]) {
        crate::diagram::tag_all_marg_side_slots(tdd, Some(was_frozen));
    }
}
