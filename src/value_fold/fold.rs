//! The value-kind axis of the marginalization fold, and the walk that drives it.

use crate::engine::Engine;

use num_bigint::BigUint;

use crate::diagram::{InputPair, TddLevel};
use crate::diagram::WeightVal;
use crate::vtree::{Vtree, VtreeIdx};

use super::{Count, CountRead, CountVec};
use crate::engine::ReservePolicy;

// ── MargFold: the value-kind axis of the marginalization fold ────────────────
//
// The four mirror families of "walk children, fold Σ left×right per node"
// collapse to
//   - ONE recursive ensure walk ([`ensure_fold_walk`]), generic over the value
//     kind (`F: MargFold`) AND the reservation policy (`R: ReservePolicy`) —
//     both contexts (in-apply `&[TddLevel]` snapshot, finished `Tdd`) walk the
//     same `&[TddLevel]` + `Vtree` shape, so one walk serves all quadrants;
//   - ONE two-pass integer fold discipline ([`IntFold::fold`]) and ONE clean
//     weighted fold ([`WeightFold::fold`]).
// The child READERS (how a pair's u32 ref resolves to a value: bit-30 tagged
// refs + snapshot columns in-apply; marg slots / bit-31 ZERO sentinel /
// interned weights on a finished Tdd) stay context-owned adapter closures
// handed to the fold — they are storage, not fold.
//
// `fold` is an INHERENT method on each
// value-kind ZST rather than a trait method, because the two reader shapes
// genuinely differ (integer: one lazy `CountRead` reader per side; weighted:
// one `Cow<WeightVal>` reader per side plus an explicit zero). The trait
// carries only the COLUMN CONTRACT — both how the ensure walk builds a column
// (pre-size + `set_col`) and how the apply-side streaming driver builds one
// (`try_with_capacity` + `push_col`, one push per alive cell). The apply
// driver's remaining per-value-kind pieces (child snapshot, per-cell fold,
// level commit) hang off the `StreamPayload` sub-trait in
// `apply::conjoin::stream`, which needs apply-local types this
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
        eng: &Engine,
        width: usize,
        zero: &Self::Scalar,
    ) -> Result<Self::Col<R>, R::Err>;
    /// Store one fold result at slot `i` of a pre-sized column.
    fn set_col<R: ReservePolicy>(
        eng: &Engine,
        col: &mut Self::Col<R>,
        i: usize,
        v: Self::Scalar,
    ) -> Result<(), R::Err>;
    /// An EMPTY column that will be filled by [`Self::push_col`], with room for
    /// `cap` appends pre-reserved where the value kind reserves at all. The
    /// append-built counterpart of [`Self::alloc_col`] (which pre-sizes and is
    /// filled by [`Self::set_col`]) — the apply-side streaming output column is
    /// built this way, one push per alive cell.
    fn try_with_capacity<R: ReservePolicy>(
        eng: &Engine,
        cap: usize,
    ) -> Result<Self::Col<R>, R::Err>;
    /// Append one fold result to an append-built column.
    fn push_col<R: ReservePolicy>(
        eng: &Engine,
        col: &mut Self::Col<R>,
        v: Self::Scalar,
    ) -> Result<(), R::Err>;
    /// Number of values currently stored in a column.
    fn col_len<R: ReservePolicy>(col: &Self::Col<R>) -> usize;
}

/// Integer model counts: u128 fast path overflowing into exact `BigUint`.
pub(crate) struct IntFold;

/// Exact weighted semiring values: no overflow machinery.
pub(crate) struct WeightFold;

impl MargFold for IntFold {
    type Scalar = Count;
    type Col<R: ReservePolicy> = CountVec<R>;

    fn alloc_col<R: ReservePolicy>(
        eng: &Engine,
        width: usize,
        _zero: &Count,
    ) -> Result<CountVec<R>, R::Err> {
        CountVec::try_with_width(eng, width)
    }

    #[inline(always)]
    fn set_col<R: ReservePolicy>(
        eng: &Engine,
        col: &mut CountVec<R>,
        i: usize,
        v: Count,
    ) -> Result<(), R::Err> {
        col.set(eng, i, v)
    }

    fn try_with_capacity<R: ReservePolicy>(
        eng: &Engine,
        cap: usize,
    ) -> Result<CountVec<R>, R::Err> {
        CountVec::try_with_capacity(eng, cap)
    }

    #[inline(always)]
    fn push_col<R: ReservePolicy>(
        eng: &Engine,
        col: &mut CountVec<R>,
        v: Count,
    ) -> Result<(), R::Err> {
        col.push(eng, v)
    }

    #[inline(always)]
    fn col_len<R: ReservePolicy>(col: &CountVec<R>) -> usize {
        col.len()
    }
}

impl MargFold for WeightFold {
    type Scalar = WeightVal;
    type Col<R: ReservePolicy> = Vec<WeightVal>;

    fn alloc_col<R: ReservePolicy>(
        eng: &Engine,
        width: usize,
        zero: &WeightVal,
    ) -> Result<Vec<WeightVal>, R::Err> {
        let mut v: Vec<WeightVal> = Vec::new();
        R::reserve_exact(eng, &mut v, width)?;
        v.resize(width, zero.clone());
        Ok(v)
    }

    #[inline(always)]
    fn set_col<R: ReservePolicy>(
        _eng: &Engine,
        col: &mut Vec<WeightVal>,
        i: usize,
        v: WeightVal,
    ) -> Result<(), R::Err> {
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
    fn try_with_capacity<R: ReservePolicy>(
        _eng: &Engine,
        _cap: usize,
    ) -> Result<Vec<WeightVal>, R::Err> {
        Ok(Vec::new())
    }

    #[inline(always)]
    fn push_col<R: ReservePolicy>(
        _eng: &Engine,
        col: &mut Vec<WeightVal>,
        v: WeightVal,
    ) -> Result<(), R::Err> {
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
/// builds: the one knob shared by the marginalization folds and
/// [`PinnedCounter`](crate::query::PinnedCounter).
///
/// Every vtree level has exactly one parent, hence exactly one in-pass
/// consumer of its column, so a child's column is dead once its parent's is
/// complete. A pass that reads only the root value can therefore hold the
/// frontier instead of the whole diagram ([`Self::Frontier`]); any consumer
/// that re-reads a non-root column after the pass needs [`Self::All`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
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

/// The ONE recursive ensure walk: populate `computed[li]` with a
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
// The fold's per-level buffers are passed separately so they can be borrowed
// independently of the diagram they index into.
#[allow(clippy::too_many_arguments)]
pub(crate) fn ensure_fold_walk<F, R, G, N>(
    eng: &Engine,
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
    ensure_fold_walk::<F, R, G, N>(
        eng,
        l_i,
        vtree,
        levels,
        computed,
        zero,
        already_done,
        fold_node,
        retain,
    )?;
    ensure_fold_walk::<F, R, G, N>(
        eng,
        r_i,
        vtree,
        levels,
        computed,
        zero,
        already_done,
        fold_node,
        retain,
    )?;

    // Fallible alloc — `width` can reach ~1B on pathological levels, where an
    // infallible `vec![zero; width]` would abort past the recovery cascade.
    // The policy routes this through the soft budget (`ApplyBudget`) or the
    // controlled recovery panic (`RecoveryPanic`).
    let width = levels[li].width();
    let mut col = F::alloc_col::<R>(eng, width, zero)?;
    for (i, _pairs) in levels[li].internal_inputs_iter() {
        F::set_col(eng, &mut col, i, fold_node(li, i, l_i, r_i, computed))?;
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
