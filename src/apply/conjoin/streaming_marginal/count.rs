//! The integer arm of the streaming fold.

use super::*;
use crate::diagram::LEAF_COUNTS;

// ── Integer payload (model counts) ──────────────────────────────────────────


/// Read one child count for the all-`u64` fold. `MARGINAL`
/// is the child level's `is_marginal` flag, lifted to a const so the per-pair branch
/// folds away at compile time:
/// - `MARGINAL=false` (non-marginal): the ref is a bare index, so the read is a
///   single load with no tag test.
/// - `MARGINAL=true` (marginal): a ref with bit 30 set is an inline count
///   (at most `MARGINAL_INLINE_MAX`); otherwise the bare slot indexes `counts`.
///
/// # Safety
/// `get_unchecked` carries the same in-bounds guarantee the prior checked
/// `counts[idx]` relied on: the fold visits only live cells whose child refs
/// decode to valid `counts` indices (dead / bit-31 sentinel refs are pruned
/// before the fold). A `debug_assert` re-checks the bound in test/debug builds.
#[inline(always)]
unsafe fn read_fast<const MARGINAL: bool>(raw: u32, c: &StreamChildCounts<'_>) -> u128 {
    if MARGINAL {
        if let Some(c) = ValueRef::inline_count(raw) {
            c as u128
        } else {
            let idx = SideView::marginal().coord(NodeIdx(raw)).idx();
            debug_assert!(idx < c.col.len(), "marginal slot index is past the column end");
            unsafe { *c.col.fast_slice().get_unchecked(idx) }
        }
    } else {
        // A structural side needs no decode: the ref is the index.
        let idx = raw as usize;
        debug_assert!(idx < c.col.len(), "structural index is past the column end");
        unsafe { *c.col.fast_slice().get_unchecked(idx) }
    }
}

/// The all-u64 two-accumulator fold, monomorphized on each side's `is_marginal`
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
        // Safety: see `read_fast` — fold pairs are live cells with in-bounds refs.
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
        // Safety: see `read_fast`.
        let (lc, rc) = unsafe {
            (
                read_fast::<LM>(pair.left.0, left),
                read_fast::<RM>(pair.right.0, right),
            )
        };
        let product = (lc as u64 as u128) * (rc as u64 as u128);
        t0 = t0.checked_add(product)?;
    }
    t0.checked_add(t1)
}

/// Read one pair side as `(count_value_or_sentinel, slot_index)`.
///
/// For an inline ref with bit 30 set the value is itself the count and the slot index is
/// unused: such a count is at most `MARGINAL_INLINE_MAX`, never `COUNT_OVERFLOW`,
/// so the big path never dereferences the sentinel index.
///
/// The polarity is self-describing: for a marginal child a ref with bit 30 set is
/// an inline count and one with bit 30 clear is a slot index (a fresh mid-apply grid
/// index is a bare node index, which is its slot, and decodes correctly here).
#[inline(always)]
fn read_marginal_count(raw: u32, c: &StreamChildCounts<'_>, view: SideView) -> (u128, usize) {
    if view.is_marginal()
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
/// the both-big case takes a full bigint multiply. The branch trims the
/// allocations per pair from as many as three down to zero or one.
fn sum_pairs_big(
    pairs: &[InputPair],
    left: &StreamChildCounts<'_>,
    right: &StreamChildCounts<'_>,
    left_view: SideView,
    right_view: SideView,
) -> num_bigint::BigUint {
    let mut bt = num_bigint::BigUint::ZERO;
    for pair in pairs {
        let (lc_val, left_idx) = read_marginal_count(pair.left.0, left, left_view);
        let (rc_val, right_idx) = read_marginal_count(pair.right.0, right, right_view);
        let lc_is_big = lc_val == COUNT_OVERFLOW;
        let rc_is_big = rc_val == COUNT_OVERFLOW;
        match (lc_is_big, rc_is_big) {
            (false, false) => {
                // Both inputs u128; the running BigUint sum is needed
                // only because aggregate `bt` already overflowed.
                if let Some(product) = lc_val.checked_mul(rc_val) {
                    bt += product;
                } else {
                    // u128 × u128 overflows: 128-bit BigUint then scalar.
                    let mut tmp = num_bigint::BigUint::from(lc_val);
                    tmp *= rc_val;
                    bt += tmp;
                }
            }
            (true, false) => {
                let lc_ref = left.col.big_val(left_idx)
                    .expect("missing BigUint for overflowed left count");
                bt += lc_ref * rc_val;
            }
            (false, true) => {
                let rc_ref = right.col.big_val(right_idx)
                    .expect("missing BigUint for overflowed right count");
                bt += rc_ref * lc_val;
            }
            (true, true) => {
                let lc_ref = left.col.big_val(left_idx)
                    .expect("missing BigUint for overflowed left count");
                let rc_ref = right.col.big_val(right_idx)
                    .expect("missing BigUint for overflowed right count");
                bt += lc_ref * rc_ref;
            }
        }
    }
    bt
}

/// Sum `Σ counts_left[p.left] * counts_right[p.right]` over `pairs`: `Count::Fast`
/// when the total fits `u128`, `Count::Big` when it overflowed or an input is
/// already at `COUNT_OVERFLOW`.
pub(crate) fn compute_cell_count(
    pairs: &[InputPair],
    left: &StreamChildCounts<'_>,
    right: &StreamChildCounts<'_>,
) -> Count {
    // Tag-at-creation: a marginal child's refs may carry the bit-30 slot tag —
    // strip it before indexing. Non-marginal and leaf children index verbatim.
    let left_view = if left.is_marginal { SideView::marginal() } else { SideView::structural() };
    let right_view = if right.is_marginal { SideView::marginal() } else { SideView::structural() };
    let mut total: u128 = 0;
    let mut overflowed = false;
    if left.col.all_u64() && right.col.all_u64() {
        // Every read is at most `u64::MAX` (slots certified by `all_u64`,
        // inline refs at most `MARGINAL_INLINE_MAX`), so each product is a
        // widening `u64×u64→u128` multiply that cannot overflow and the
        // `COUNT_OVERFLOW` sentinel cannot appear. `fold_fast` is monomorphized
        // on each side's marginality; a `None` (sum overflow) falls to the
        // `BigUint` re-loop below, which re-reads the pairs.
        let res = match (left.is_marginal, right.is_marginal) {
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
            let (lc, _) = read_marginal_count(pair.left.0, left, left_view);
            let (rc, _) = read_marginal_count(pair.right.0, right, right_view);
            if lc == COUNT_OVERFLOW || rc == COUNT_OVERFLOW {
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
        // `Count::from_u128` owns the exact-max promotion (a total equal to
        // `u128::MAX` == `COUNT_OVERFLOW` must not be stored as a fast value).
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
    fn stream_columns(cache: &StreamCache) -> &[Option<CountVec<ApplyBudget>>] {
        cache.int()
    }

    #[inline]
    fn store_of(_ws: Option<&WeightStore>) -> &() {
        &()
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
        left_idx: usize,
        vtree: &crate::vtree::Vtree,
        level: &'a TddLevel,
        computed: &'a [Option<CountVec<R>>],
        _store: &(),
    ) -> Result<StreamChild<'a, IntFold>, ApplyError> {
        let is_marginal = level.marginal_counts().is_some();
        // Raw-storage sources (`marginal_counts`/`marginal_counts_big` on the level)
        // are viewed through `CountRef::from_parts_scanned` (u64-fit certificate
        // scanned over the stored slots; `COUNT_OVERFLOW` = `u128::MAX` fails the scan,
        // so all_u64 ⇒ no overflow sentinel present). A `computed` source is already
        // a `CountVec` and lends its own incrementally maintained certificate. The
        // sources are mutually exclusive: the ensure walk only fills `computed`
        // for non-marginal levels, and the in-apply cascade moves an entry
        // into level storage when the level becomes marginal.
        //
        // No arm allocates: every one borrows storage that already exists.
        let col = if vtree.node(VtreeIdx(left_idx as u32)).is_leaf()
            && level.marginal_counts().is_some_and(|c| c.is_empty())
        {
            // Marginal leaf (leaf marginalization): empty store, all counts inline at
            // the parent. Its conceptual slots are the fixed leaf labels — return
            // them so any stray bare-label ref (slot 0/1/2) still decodes correctly;
            // inline refs bypass this column entirely. This integer-side fixed-slot
            // rule has no weighted counterpart (see `WeightFold::child_view`,
            // which resolves the semiring leaf bases instead).
            CountRef::from_parts_scanned(&LEAF_COUNTS, level.marginal_counts_big())
        } else if let Some(ic) = level.marginal_counts() {
            CountRef::from_parts_scanned(ic, level.marginal_counts_big())
        } else if let Some(c) = &computed[left_idx] {
            c.as_count_ref()
        } else if vtree.node(VtreeIdx(left_idx as u32)).is_leaf() {
            CountRef::from_parts_scanned(&LEAF_COUNTS, None)
        } else {
            unreachable!("IntFold::child_view: no counts for level {}", left_idx);
        };

        Ok(StreamChild { col, is_marginal })
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
        left_idx: usize,
        col: CountVec<R>,
        _store: &mut (),
    ) {
        crate::marginal::install_int_column(levels, left_idx, col);
    }

    /// Counts are deduped before they are installed, so the level satisfies invariant 10 — no
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
        tdd.levels[t.vtree_idx().idx()].become_marginal(counts, big);
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

    /// Make every marginal-side slot reference this pass persisted self-describing,
    /// once, at the pass's chokepoint.
    ///
    /// This path does not go through `apply_and_fallible`, so the end-of-apply
    /// tagger never runs on it, and canon's no-duplicate early return leaves
    /// untouched boundary references raw. The snapshot is what keeps the sweep
    /// off children that a prior pass marginalized: those already carry inline
    /// counts, and re-resolving them as bare slots would misread them.
    fn end_sweep(tdd: &mut Tdd, was_marginal: &[bool]) {
        crate::diagram::tag_all_marginal_side_slots(tdd, Some(was_marginal));
    }
}
