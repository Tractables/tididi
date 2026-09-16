//! The value-kind axis of the marginalization fold, and the walk that drives it.

use crate::limits::OperationError;
use crate::diagram::EncodedChildRef;

use crate::Engine;

use num_bigint::BigUint;

use crate::diagram::ChildPair;
use crate::diagram::WeightValue;
use crate::vtree::{Vtree, VtreeIdx};

use super::{Count, CountRead, CountVec};

/// The value-kind axis of the marginalization fold: what scalar a per-node
/// fold produces and what scratch column stores it.
///
/// The trait carries only the column contract: the ensure walk builds a
/// column by [`Self::alloc_col`] and [`Self::set_col`], the apply-side
/// streaming driver by [`Self::try_with_capacity`] and [`Self::push_col`], one
/// push per alive cell. Each kind's `fold` is an inherent method
/// ([`IntFold::fold`], [`WeightFold::fold`]) because the two reader shapes
/// differ: one lazy `CountRead` per side against one `Cow<WeightValue>` per
/// side plus an explicit zero. The readers are closures the context hands in,
/// since how a ref resolves to a value is the storage's business. The apply
/// driver's remaining per-kind pieces (child view, per-cell fold, in-flight
/// commit) are on `ValueDomain`.
pub(crate) trait MarginalFold {
    /// One per-node fold result (`Count` | `WeightValue`).
    type Scalar;
    /// The scratch column for this domain.
    type Col;
    /// A fresh `width`-element column of `zero`s, reserved through the engine.
    fn alloc_col(
        eng: &Engine,
        width: usize,
        zero: &Self::Scalar,
    ) -> Result<Self::Col, OperationError>;
    /// Store one fold result at slot `i` of a pre-sized column.
    fn set_col(
        eng: &Engine,
        col: &mut Self::Col,
        i: usize,
        v: Self::Scalar,
    ) -> Result<(), OperationError>;
    /// An empty column that will be filled by [`Self::push_col`], with room for
    /// `cap` appends pre-reserved. The append-built counterpart of [`Self::alloc_col`] (which pre-sizes and is
    /// filled by [`Self::set_col`]) — the apply-side streaming output column is
    /// built this way, one push per alive cell.
    fn try_with_capacity(
        eng: &Engine,
        cap: usize,
    ) -> Result<Self::Col, OperationError>;
    /// Append one fold result to an append-built column.
    fn push_col(
        eng: &Engine,
        col: &mut Self::Col,
        v: Self::Scalar,
    ) -> Result<(), OperationError>;
    /// Number of values currently stored in a column.
    fn col_len(col: &Self::Col) -> usize;
}

/// Integer model counts: u128 fast path overflowing into exact `BigUint`.
pub(crate) struct IntFold;

/// Exact weighted semiring values: no overflow machinery.
pub(crate) struct WeightFold;

impl MarginalFold for IntFold {
    type Scalar = Count;
    type Col = CountVec;

    fn alloc_col(
        eng: &Engine,
        width: usize,
        _zero: &Count,
    ) -> Result<CountVec, OperationError> {
        CountVec::try_with_width(eng, width)
    }

    #[inline(always)]
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

    #[inline(always)]
    fn push_col(
        eng: &Engine,
        col: &mut CountVec,
        v: Count,
    ) -> Result<(), OperationError> {
        col.push(eng, v)
    }

    #[inline(always)]
    fn col_len(col: &CountVec) -> usize {
        col.len()
    }
}

impl MarginalFold for WeightFold {
    type Scalar = WeightValue;
    type Col = Vec<WeightValue>;

    fn alloc_col(
        eng: &Engine,
        width: usize,
        zero: &WeightValue,
    ) -> Result<Vec<WeightValue>, OperationError> {
        let mut v: Vec<WeightValue> = Vec::new();
        eng.limits().reserve_exact(&mut v, width)?;
        v.resize(width, zero.clone());
        Ok(v)
    }

    #[inline(always)]
    fn set_col(
        _eng: &Engine,
        col: &mut Vec<WeightValue>,
        i: usize,
        v: WeightValue,
    ) -> Result<(), OperationError> {
        col[i] = v;
        Ok(())
    }

    fn try_with_capacity(
        eng: &Engine,
        cap: usize,
    ) -> Result<Vec<WeightValue>, OperationError> {
        let mut col = Vec::new();
        eng.limits().reserve_exact(&mut col, cap)?;
        Ok(col)
    }

    #[inline(always)]
    fn push_col(
        eng: &Engine,
        col: &mut Vec<WeightValue>,
        v: WeightValue,
    ) -> Result<(), OperationError> {
        eng.limits().reserve(col, 1)?;
        col.push(v);
        Ok(())
    }

    #[inline(always)]
    fn col_len(col: &Vec<WeightValue>) -> usize {
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
        P: Iterator<Item = ChildPair> + Clone,
        L: Fn(EncodedChildRef) -> CountRead<'a>,
        R: Fn(EncodedChildRef) -> CountRead<'a>,
    {
        let mut total: u128 = 0;
        let mut overflowed = false;
        for pair in pairs.clone() {
            // A zero operand contributes nothing, so the other side is never
            // read; under a pinned cofactor evaluation most pairs have one.
            let CountRead::Fast(lc) = l(pair.left) else {
                overflowed = true;
                break;
            };
            if lc == 0 {
                continue;
            }
            let CountRead::Fast(rc) = r(pair.right) else {
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
            let (l, r) = (l(pair.left), r(pair.right));
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
    pub(crate) fn fold<'a, P, L, R>(pairs: P, l: L, r: R, zero: WeightValue) -> WeightValue
    where
        P: Iterator<Item = ChildPair>,
        L: Fn(EncodedChildRef) -> std::borrow::Cow<'a, WeightValue>,
        R: Fn(EncodedChildRef) -> std::borrow::Cow<'a, WeightValue>,
    {
        let mut total = zero;
        for pair in pairs {
            let lc = l(pair.left);
            let rc = r(pair.right);
            total.add_assign(&lc.mul(&rc));
        }
        total
    }
}

/// Lifetime policy for the per-level value columns a bottom-up fold pass
/// builds: the one knob shared by the marginalization folds and
/// [`ModelCounter`](crate::query::ModelCounter).
///
/// Every vtree level has exactly one parent, hence exactly one in-pass
/// consumer of its column, so a child's column is dead once its parent's is
/// complete. A pass that reads only the root value can therefore hold the
/// frontier instead of the whole diagram ([`Self::Frontier`]); any consumer
/// that re-reads a non-root column after the pass needs [`Self::All`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[non_exhaustive]
pub(crate) enum ColumnRetention {
    /// Keep every level's column for the caller.
    All,
    /// Free each child column as soon as its parent's column is complete.
    Frontier,
}

impl ColumnRetention {
    /// The frontier [`walk_bottom_up`] releases: `Some(keep)` under
    /// [`Frontier`](Self::Frontier), with `keep` the one level exempt from
    /// release; `None` under [`All`](Self::All).
    pub(crate) fn frontier(self, keep: VtreeIdx) -> Option<VtreeIdx> {
        match self {
            ColumnRetention::All => None,
            ColumnRetention::Frontier => Some(keep),
        }
    }
}

/// Visit the levels under `root` whose column is not yet in hand, children
/// before parents, and compute each one's column.
///
/// `held(cols, i)` says level `i`'s column is already in hand, which stops
/// the walk there: neither that level nor anything below it is visited.
/// `compute(cols, t)` fills level `t`'s column; when it runs, both children's
/// columns are complete or held. `release(cols, i)` frees level `i`'s column.
///
/// With `frontier` set (see [`ColumnRetention::frontier`]) a level's two
/// children are released as soon as its column is complete — the vtree is a
/// tree, so that level was their only consumer — and the live set is the walk
/// frontier rather than one column per level. The level named in `frontier`
/// is exempt: the one whose column the caller reads afterwards, which for a
/// whole-diagram query can be a leaf and so a child of some level. The root
/// is never released, having no parent inside the walk. Releasing also gives
/// up memoization for the freed subtrees, so it is for one root-only walk per
/// column buffer; a second walk over an overlapping subtree would recompute
/// it.
///
/// The visit order is left-to-right postorder: a level is computed as soon as
/// its subtree is, so with a frontier the live set is at most one column per
/// ancestor of the level being computed. Parent links carry the return path,
/// so the traversal itself allocates no stack.
pub(crate) fn walk_bottom_up<C, E>(
    vtree: &Vtree,
    root: VtreeIdx,
    cols: &mut [C],
    held: impl Fn(&[C], usize) -> bool,
    mut compute: impl FnMut(&mut [C], VtreeIdx) -> Result<(), E>,
    mut release: impl FnMut(&mut [C], usize),
    frontier: Option<VtreeIdx>,
) -> Result<(), E> {
    let mut t = root;
    let mut subtree_done = false;
    loop {
        if subtree_done || !held(cols, t.idx()) {
            if !subtree_done && !vtree.node(t).is_leaf() {
                t = vtree.children(t).0;
                continue;
            }
            compute(cols, t)?;
            if let Some(keep) = frontier
                && !vtree.node(t).is_leaf()
            {
                let (l, r) = vtree.children(t);
                for c in [l, r] {
                    if c != keep { release(cols, c.idx()); }
                }
            }
        }
        if t == root { return Ok(()); }
        let parent = vtree.node(t).parent().expect("a non-root walk node has a parent");
        if t == vtree.children(parent).0 {
            t = vtree.children(parent).1;
            subtree_done = false;
        } else {
            t = parent;
            subtree_done = true;
        }
    }
}
