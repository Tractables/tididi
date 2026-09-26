//! The two value domains' sum-of-products folds, and the walk that drives them.

use crate::diagram::EncodedChildRef;

use num_bigint::BigUint;

use crate::diagram::ChildPair;
use crate::diagram::WeightValue;
use crate::vtree::{Vtree, VtreeIdx};

use super::{Count, CountRead};

/// Integer model counts: u128 fast path overflowing into exact `BigUint`.
pub(crate) struct IntFold;

/// Exact weighted semiring values: no overflow machinery.
pub(crate) struct WeightFold;

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
    /// Zero products are skipped; the fast pass can skip the right read when
    /// the left count is zero.
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
        Count::Big(Self::sum_exact(pairs, l, r))
    }

    /// `Σ over pairs (left × right)` for a node whose children are both
    /// structural, with each child's counts a raw column every value of which
    /// fits `u64` (the `all_u64` certificate): a reference is then the index
    /// of its count, and every product is one widening multiply that cannot
    /// overflow. `None` when the total leaves `u128`; the caller then takes
    /// the exact [`Self::fold`].
    ///
    /// Two accumulators, so consecutive pairs do not wait on one carry chain,
    /// and no branch on a zero count: under the certificate a zero product
    /// costs what any other does.
    #[inline]
    pub(crate) fn fold_structural_u64(pairs: &[ChildPair], left: &[u128], right: &[u128]) -> Option<u128> {
        let read = |col: &[u128], side: EncodedChildRef| col[side.0 as usize] as u64 as u128;
        let (mut t0, mut t1) = (0u128, 0u128);
        let mut two = pairs.chunks_exact(2);
        for p in two.by_ref() {
            t0 = t0.checked_add(read(left, p[0].left) * read(right, p[0].right))?;
            t1 = t1.checked_add(read(left, p[1].left) * read(right, p[1].right))?;
        }
        for p in two.remainder() {
            t0 = t0.checked_add(read(left, p.left) * read(right, p.right))?;
        }
        t0.checked_add(t1)
    }

    /// Sum products in arbitrary precision after a fast fold refuses the total.
    /// Readers supply decoded values; this fold does not know their storage layout.
    ///
    /// Kept out of line: it runs only when a total leaves `u128`, and inlining
    /// it would put the bigint loop into every caller's fast path.
    #[cold]
    #[inline(never)]
    pub(crate) fn sum_exact<'a>(
        pairs: impl Iterator<Item = ChildPair>,
        l: impl Fn(EncodedChildRef) -> CountRead<'a>,
        r: impl Fn(EncodedChildRef) -> CountRead<'a>,
    ) -> BigUint {
        let mut bt = BigUint::ZERO;
        for pair in pairs {
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
        bt
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
///
/// For a [`ModelCounter`](crate::query::ModelCounter), [`Self::All`] keeps
/// the counts so a pin change refreshes only the levels above it, at a column
/// per level of the diagram; [`Self::Frontier`] holds only the columns the
/// walk still needs and repeats the whole fold after a change.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[non_exhaustive]
pub enum Retention {
    /// Keep every level's column for the caller.
    All,
    /// Free each child column as soon as its parent's column is complete.
    Frontier,
}

impl Retention {
    /// The frontier [`walk_bottom_up`] releases: `Some(keep)` under
    /// [`Frontier`](Self::Frontier), with `keep` the one level exempt from
    /// release; `None` under [`All`](Self::All).
    pub(crate) fn frontier(self, keep: VtreeIdx) -> Option<VtreeIdx> {
        match self {
            Retention::All => None,
            Retention::Frontier => Some(keep),
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
/// With `frontier` set (see [`Retention::frontier`]) a level's two
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
