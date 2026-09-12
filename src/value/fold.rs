//! The value-kind axis of the marginalization fold, and the walk that drives it.

use crate::engine::Engine;

use num_bigint::BigUint;

use crate::diagram::InputPair;
use crate::diagram::WeightVal;
use crate::vtree::{Vtree, VtreeIdx};

use super::{Count, CountRead, CountVec};
use crate::limits::ReservePolicy;
pub(crate) use crate::limits::unwrap_infallible;

/// The value-kind axis of the marginalization fold: what scalar a per-node
/// fold produces and what scratch column stores it.
///
/// The trait carries only the column contract: the ensure walk builds a
/// column by [`Self::alloc_col`] and [`Self::set_col`], the apply-side
/// streaming driver by [`Self::try_with_capacity`] and [`Self::push_col`], one
/// push per alive cell. Each kind's `fold` is an inherent method
/// ([`IntFold::fold`], [`WeightFold::fold`]) because the two reader shapes
/// differ: one lazy `CountRead` per side against one `Cow<WeightVal>` per
/// side plus an explicit zero. The readers are closures the context hands in,
/// since how a ref resolves to a value is the storage's business. The apply
/// driver's remaining per-kind pieces (child view, per-cell fold, in-flight
/// commit) are on `ValueDomain`.
pub(crate) trait MarginalFold {
    /// One per-node fold result (`Count` | `WeightVal`).
    type Scalar;
    /// The scratch column, parameterized by the fallibility policy.
    type Col<R: ReservePolicy>;
    /// A fresh `width`-element column of `zero`s, reserved through `R` — the
    /// single fallible-allocation point of the ensure walk; the weighted column
    /// allocates through the same fallible path as the integer one.
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
    /// An empty column that will be filled by [`Self::push_col`], with room for
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

impl MarginalFold for IntFold {
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

impl MarginalFold for WeightFold {
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

    /// `cap` is ignored: the weighted streaming column is grown by plain
    /// `push` with no budget charge; the per-pair transient is charged by the
    /// apply's collect sink instead.
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
    /// The two-pass integer fold: `Σ over pairs (left × right)`.
    ///
    /// Pass 1 accumulates in `u128` with `checked_mul`/`checked_add`, breaking
    /// to pass 2 on the first overflow or the first `Big` child read. Pass 2
    /// re-reads every pair (hence `P: Clone`) into an exact `BigUint` total,
    /// branching on magnitude so that only a pair whose product leaves `u128`
    /// touches the bigint allocator. A pass-1 total that lands on the overflow
    /// sentinel is promoted by [`Count::from_u128`].
    ///
    /// Both passes skip a pair with a zero operand without reading the other
    /// side.
    pub(crate) fn fold<'a, P, L, R>(pairs: P, l: L, r: R) -> Count
    where
        P: Iterator<Item = InputPair> + Clone,
        L: Fn(usize) -> CountRead<'a>,
        R: Fn(usize) -> CountRead<'a>,
    {
        let mut total: u128 = 0;
        let mut overflowed = false;
        for pair in pairs.clone() {
            // A zero operand contributes nothing, so the other side is never
            // read; under a pinned cofactor evaluation most pairs have one.
            let CountRead::Fast(lc) = l(pair.left.idx()) else {
                overflowed = true;
                break;
            };
            if lc == 0 {
                continue;
            }
            let CountRead::Fast(rc) = r(pair.right.idx()) else {
                overflowed = true;
                break;
            };
            if rc == 0 {
                continue;
            }
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
            // Same skip, and here it also buys the allocation a zero operand
            // would otherwise pay for on the mixed-magnitude branches.
            let (l, r) = (l(pair.left.idx()), r(pair.right.idx()));
            if matches!(l, CountRead::Fast(0)) || matches!(r, CountRead::Fast(0)) {
                continue;
            }
            match (l, r) {
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
    /// The one weighted fold: `Σ over pairs (left × right)` in the exact
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
/// [`IncrementalCounter`](crate::query::IncrementalCounter).
///
/// Every vtree level has exactly one parent, hence exactly one in-pass
/// consumer of its column, so a child's column is dead once its parent's is
/// complete. A pass that reads only the root value can therefore hold the
/// frontier instead of the whole diagram ([`Self::Frontier`]); any consumer
/// that re-reads a non-root column after the pass needs [`Self::All`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[non_exhaustive]
pub enum ColumnRetention {
    /// Keep every level's column for the caller.
    All,
    /// Free each child column as soon as its parent's column is complete.
    Frontier,
}

/// Visit the levels under `root` whose column is not yet in hand, children
/// before parents, and compute each one's column.
///
/// `held(cols, i)` says level `i`'s column is already in hand, which stops
/// the walk there: neither that level nor anything below it is visited.
/// `compute(cols, t)` fills level `t`'s column; when it runs, both children's
/// columns are complete or held. `release(cols, i)` frees level `i`'s column.
///
/// Under [`ColumnRetention::Frontier`] a level's two children are released
/// as soon as its column is complete — the vtree is a tree, so that level was
/// their only consumer — and the live set is the walk frontier rather than
/// one column per level. `keep` is exempt: the one level whose column the
/// caller reads afterwards, which for a whole-diagram query can be a leaf and
/// so a child of some level. The root is never released, having no parent
/// inside the walk. `Frontier` also gives up memoization for the freed
/// subtrees, so it is for one root-only walk per column buffer; a second
/// walk over an overlapping subtree would recompute it.
///
/// The visit order is left-to-right postorder: a level is computed as soon as
/// its subtree is, so under `Frontier` the live set is at most one column per
/// ancestor of the level being computed.
#[allow(clippy::too_many_arguments)]
pub(crate) fn walk_bottom_up<C, E>(
    vtree: &Vtree,
    root: VtreeIdx,
    cols: &mut [C],
    held: impl Fn(&[C], usize) -> bool,
    mut compute: impl FnMut(&mut [C], VtreeIdx) -> Result<(), E>,
    mut release: impl FnMut(&mut [C], usize),
    retain: ColumnRetention,
    keep: VtreeIdx,
) -> Result<(), E> {
    // Answered before the stack exists: most walks an apply asks for find
    // their root already held.
    if held(cols, root.idx()) {
        return Ok(());
    }
    // Each level is popped twice: once on the way down, once after its
    // subtree is done.
    let mut stack = vec![(root, false)];
    while let Some((t, subtree_done)) = stack.pop() {
        if subtree_done {
            compute(cols, t)?;
            if retain == ColumnRetention::Frontier && !vtree.node(t).is_leaf() {
                let (l, r) = vtree.children(t);
                for c in [l, r] {
                    if c != keep {
                        release(cols, c.idx());
                    }
                }
            }
        } else if !held(cols, t.idx()) {
            stack.push((t, true));
            if !vtree.node(t).is_leaf() {
                let (l, r) = vtree.children(t);
                stack.push((r, false));
                stack.push((l, false));
            }
        }
    }
    Ok(())
}

