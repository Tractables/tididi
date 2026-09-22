//! The integer arm of the streaming fold.

use super::*;
use crate::diagram::{EncodedChildRef, ChildPair, ChildDecoder, ChildRef, MarginalSide, ValueRef, TddLevel, WeightStore};
use crate::value::{Count, CountRead, CountRef, CountVec, IntFold, COUNT_OVERFLOW};
use crate::diagram::LEAF_COUNTS;

pub(crate) type StreamChildCounts<'a> = StreamChild<'a, IntFold>;

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
unsafe fn read_fast<const MARGINAL: bool>(raw: u32, c: &StreamChildCounts<'_>) -> u128 {
    if MARGINAL {
        if let ValueRef::Inline(c) = ValueRef::from_raw(MarginalSide(raw)) {
            c as u128
        } else {
            let idx = ChildDecoder::marginal().coord(EncodedChildRef::from_raw(raw)) as usize;
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
pub(crate) fn fold_fast<const LM: bool, const RM: bool>(
    pairs: &[ChildPair],
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

/// Read the fast count or overflow sentinel without consulting the exact side table.
fn read_marginal_count(raw: u32, c: &StreamChildCounts<'_>, view: ChildDecoder) -> u128 {
    if view.is_marginal()
        && let ValueRef::Inline(c) = ValueRef::from_raw(MarginalSide(raw))
    {
        c as u128
    } else {
        let idx = view.coord(EncodedChildRef::from_raw(raw)) as usize;
        c.col.fast_val(idx)
    }
}


/// Resolve a pair side without exposing the count column's overflow encoding.
fn read_count<'a>(side: EncodedChildRef, child: &StreamChildCounts<'a>, view: ChildDecoder) -> CountRead<'a> {
    match view.child(side) {
        ChildRef::Value(ValueRef::Inline(value)) => CountRead::Fast(value as u128),
        ChildRef::Node(crate::diagram::NodeIdx(index))
        | ChildRef::Value(ValueRef::Slot(index)) => child.col.get(index as usize),
    }
}

/// Sum `Σ counts_left[p.left] * counts_right[p.right]` over `pairs`: `Count::Fast`
/// when the total fits `u128`, `Count::Big` when it overflowed or an input is
/// already at `COUNT_OVERFLOW`.
pub(crate) fn compute_cell_count(
    pairs: &[ChildPair],
    left: &StreamChildCounts<'_>,
    right: &StreamChildCounts<'_>,
) -> Count {
    // Tag-at-creation: a marginal child's refs may carry the bit-30 slot tag —
    // strip it before indexing. Non-marginal and leaf children index verbatim.
    let left_view = if left.is_marginal { ChildDecoder::marginal() } else { ChildDecoder::structural() };
    let right_view = if right.is_marginal { ChildDecoder::marginal() } else { ChildDecoder::structural() };
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
            let lc = read_marginal_count(pair.left.0, left, left_view);
            let rc = read_marginal_count(pair.right.0, right, right_view);
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
    Count::Big(IntFold::sum_exact(
        pairs.iter().copied(),
        |side| read_count(side, left, left_view),
        |side| read_count(side, right, right_view),
    ))
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

    #[inline]
    fn stream_columns(cache: &StreamCache) -> &[Option<CountVec>] {
        cache.int()
    }

    #[inline]
    fn store_of(_ws: Option<&WeightStore>) -> &() {
        &()
    }

    #[inline]
    fn fold_node(at: &FoldScope<'_, IntFold>, i: usize) -> Count {
        let FoldInput { vtree, levels, .. } = at.input;
        IntFold::fold(
            levels[at.lvl].pairs_iter_of_idx(i),
            |k| crate::value::read::read_count(at.left, k, vtree, levels, at.computed),
            |k| crate::value::read::read_count(at.right, k, vtree, levels, at.computed),
        )
    }

    fn child_view<'a>(
        left_idx: usize,
        vtree: &crate::vtree::Vtree,
        level: &'a TddLevel,
        computed: &'a [Option<CountVec>],
        _store: &'a (),
    ) -> StreamChild<'a, IntFold> {
        let is_marginal = level.marginal_counts().is_some();
        // Raw-storage sources (`marginal_counts`/`marginal_counts_big` on the level)
        // are viewed through `CountRef::from_parts_scanned` (u64-fit certificate
        // scanned over the stored slots; `COUNT_OVERFLOW` = `u128::MAX` fails the scan,
        // so `all_u64` ⇒ no overflow sentinel present). A `computed` source is already
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

        StreamChild { col, is_marginal }
    }

    fn fold_cell(
        pairs: &[ChildPair],
        left: &StreamChildCounts<'_>,
        right: &StreamChildCounts<'_>,
        _store: &(),
    ) -> Count {
        compute_cell_count(pairs, left, right)
    }

}

