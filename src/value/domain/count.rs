//! The integer arm of the streaming fold.

use super::*;
use crate::diagram::{EncodedChildRef, ChildPair, ChildDecoder, ValueRef, TddLevel, WeightStore};
use crate::value::{Count, CountRead, CountRef, CountVec, IntFold};
use crate::diagram::{LEAF_COUNTS, LeafLabel, leaf_count};

pub(crate) type StreamChildCounts<'a> = StreamChild<'a, IntFold>;

// ── Integer payload (model counts) ──────────────────────────────────────────


/// Read one child count for the all-`u64` fold. `MARGINAL`
/// is whether the child's view decodes marginal-side references, lifted to a const so the per-pair branch
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
        match ChildDecoder::marginal().value(EncodedChildRef::from_raw(raw)) {
            ValueRef::Inline(value) => value as u128,
            ValueRef::Slot(slot) => {
                let idx = slot as usize;
                debug_assert!(idx < c.col.len(), "marginal slot index is past the column end");
                unsafe { *c.col.fast_slice().get_unchecked(idx) }
            }
        }
    } else {
        // A structural side needs no decode: the ref is the index.
        let idx = raw as usize;
        debug_assert!(idx < c.col.len(), "structural index is past the column end");
        unsafe { *c.col.fast_slice().get_unchecked(idx) }
    }
}

/// The all-u64 two-accumulator fold, monomorphized on whether each side's view
/// decodes marginal-side references (`LM`/`RM`). Returns `None` on u128 overflow in either lane or the final
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

/// Resolve one child ref of level `level_idx` to a count read, against the
/// diagram's levels and the per-batch `computed` scratch; a `Big` read
/// borrows the `BigUint`.
///
/// Marginal leaves can retain bare label references with an empty count
/// column; their counts still come from the fixed leaf labels.
#[inline]
fn read_level_count<'a>(
    level_idx: usize,
    side: EncodedChildRef,
    vtree: &Vtree,
    levels: &'a [TddLevel],
    computed: &'a [Option<CountVec>],
) -> CountRead<'a> {
    if let Some(ic) = levels[level_idx].marginal_counts() {
        if side.is_reserved() {
            return CountRead::Fast(0); // ZERO sentinel — never decode (mirrors emit_or_tag)
        }
        return match ChildDecoder::marginal().value(side) {
            ValueRef::Inline(v) => CountRead::Fast(v as u128),
            // A marginal leaf keeps an empty store under the inline path (all
            // counts live inline at the parent), so a bare slot ref here is a
            // leaf-label index with a fixed count — decode it directly rather
            // than indexing the (empty) store. Reached by paths that leave a
            // leaf-side ref bare (e.g. projection) instead of inlining it.
            ValueRef::Slot(s) if vtree.node(VtreeIdx(level_idx as u32)).is_leaf() => {
                CountRead::Fast(leaf_count(LeafLabel::from_idx(s as usize)))
            }
            ValueRef::Slot(s) => CountRead::from_slot(ic, levels[level_idx].marginal_counts_big(), s as usize),
        };
    }
    // Check pre-computed buffer (non-marginal level: plain index).
    if let Some(counts) = &computed[level_idx] {
        return counts.get(ChildDecoder::structural().node(side).idx());
    }
    // Leaf level: fixed counts.
    if vtree.node(VtreeIdx(level_idx as u32)).is_leaf() {
        return CountRead::Fast(leaf_count(LeafLabel::from_idx(ChildDecoder::structural().node(side).idx())));
    }
    unreachable!("counts not available for level {}", level_idx);
}

/// Sum `Σ counts_left[p.left] * counts_right[p.right]` over `pairs`: `Count::Fast`
/// when the total fits `u128`, `Count::Big` when it overflowed or an input is
/// already an exact big count.
pub(crate) fn compute_cell_count(
    pairs: &[ChildPair],
    left: &StreamChildCounts<'_>,
    right: &StreamChildCounts<'_>,
) -> Count {
    let read_left = |k: EncodedChildRef| left.col.read(left.view, k);
    let read_right = |k: EncodedChildRef| right.col.read(right.view, k);
    if !(left.col.all_u64() && right.col.all_u64()) {
        return IntFold::fold(pairs.iter().copied(), read_left, read_right);
    }
    // Every read is at most `u64::MAX` (slots certified by `all_u64`, inline
    // refs at most `MARGINAL_INLINE_MAX`), so each product is a widening
    // `u64×u64→u128` multiply that cannot overflow and no read is an
    // overflowed slot. `fold_fast` is monomorphized on each side's
    // marginality; a `None` (sum overflow) falls to the exact sum, which
    // re-reads the pairs.
    let res = match (left.view.is_marginal(), right.view.is_marginal()) {
        (false, false) => fold_fast::<false, false>(pairs, left, right),
        (false, true) => fold_fast::<false, true>(pairs, left, right),
        (true, false) => fold_fast::<true, false>(pairs, left, right),
        (true, true) => fold_fast::<true, true>(pairs, left, right),
    };
    match res {
        // `Count::from_u128` owns the exact-max promotion (a total equal to
        // `COUNT_OVERFLOW` must not be stored as a fast value).
        Some(total) => Count::from_u128(total),
        None => Count::Big(IntFold::sum_exact(pairs.iter().copied(), read_left, read_right)),
    }
}

impl ValueDomain for IntFold {
    /// Counts live inside the level itself, so the domain carries no state
    /// beside the diagram.
    type Store = ();

    /// Always borrowed — the integer child column is either a level's raw
    /// marginal arrays, a `computed` scratch column, or the static leaf slots.
    type ChildCol<'a> = CountRef<'a>;

    type Scalar = Count;
    type Col = CountVec;

    fn alloc_col(
        eng: &Engine,
        width: usize,
        _store: &(),
    ) -> Result<CountVec, OperationError> {
        CountVec::try_with_width(eng, width)
    }

    fn set_col(
        eng: &Engine,
        col: &mut CountVec,
        i: usize,
        v: Count,
    ) -> Result<(), OperationError> {
        col.set(eng, i, v)
    }

    fn try_with_capacity(
        eng: &Engine,
        cap: usize,
    ) -> Result<CountVec, OperationError> {
        CountVec::try_with_capacity(eng, cap)
    }

    fn push_col(
        eng: &Engine,
        col: &mut CountVec,
        v: Count,
    ) -> Result<(), OperationError> {
        col.push(eng, v)
    }

    fn col_len(col: &CountVec) -> usize {
        col.len()
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
            |k| read_level_count(at.left, k, vtree, levels, at.computed),
            |k| read_level_count(at.right, k, vtree, levels, at.computed),
        )
    }

    fn child_view<'a>(
        level_idx: usize,
        vtree: &crate::vtree::Vtree,
        level: &'a TddLevel,
        computed: &'a [Option<CountVec>],
        _store: &'a (),
    ) -> StreamChild<'a, IntFold> {
        let view = if level.marginal_counts().is_some() { ChildDecoder::marginal() } else { ChildDecoder::structural() };
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
        let col = if vtree.node(VtreeIdx(level_idx as u32)).is_leaf()
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
        } else if let Some(c) = &computed[level_idx] {
            c.as_count_ref()
        } else if vtree.node(VtreeIdx(level_idx as u32)).is_leaf() {
            CountRef::from_parts_scanned(&LEAF_COUNTS, None)
        } else {
            unreachable!("IntFold::child_view: no counts for level {}", level_idx);
        };

        StreamChild { col, view }
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


#[cfg(test)]
#[path = "tests/count.rs"]
mod tests;
