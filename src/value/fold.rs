//! The two value domains' sum-of-products folds, and the walk that drives them.

use crate::diagram::EncodedChildRef;

use num_bigint::BigUint;

use crate::diagram::ChildPair;
use crate::diagram::WeightValue;
use crate::vtree::{Vtree, VtreeIdx};

use super::{Count, CountRead, CountVec};
use crate::diagram::{NodeChunks, Pairs, TddLevel};

/// How many nodes ahead of the one it sums [`IntFold::fill_structural_u64`]
/// asks the cache for the child counts a node reads.
const NODES_AHEAD: usize = 8;

/// Ask the cache for the line holding `column[k]`, which need not be in
/// bounds: a prefetch reads nothing the program sees and never faults. A
/// no-op off `x86_64` and under Miri, which lacks the intrinsic.
#[inline(always)]
fn prefetch_at<T>(column: &[T], k: usize) {
    #[cfg(all(target_arch = "x86_64", not(miri)))]
    {
        let at = column.as_ptr().wrapping_add(k).cast::<i8>();
        // Sound whatever the address, as above.
        unsafe { core::arch::x86_64::_mm_prefetch(at, core::arch::x86_64::_MM_HINT_T0) };
    }
    #[cfg(not(all(target_arch = "x86_64", not(miri))))]
    let _ = (column, k);
}

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
        // Many diagrams hold most of their nodes with one pair.
        if let [p] = pairs {
            return Some((left[p.left.0 as usize] as u64 as u128) * (right[p.right.0 as usize] as u64 as u128));
        }
        Self::fold_structural_by(pairs.iter().copied(),
            |side| left[side.0 as usize] as u64 as u128,
            |side| right[side.0 as usize] as u64 as u128)
    }

    /// The same fold with each reader specialized to its column storage.
    /// Readers return values at most `u64::MAX`, so their product fits u128.
    #[inline]
    pub(crate) fn fold_structural_by(
        mut pairs: impl Iterator<Item = ChildPair>,
        left: impl Fn(EncodedChildRef) -> u128,
        right: impl Fn(EncodedChildRef) -> u128,
    ) -> Option<u128> {
        let (mut t0, mut t1) = (0u128, 0u128);
        while let Some(p) = pairs.next() {
            t0 = t0.checked_add(left(p.left) * right(p.right))?;
            if let Some(p) = pairs.next() {
                t1 = t1.checked_add(left(p.left) * right(p.right))?;
            }
        }
        t0.checked_add(t1)
    }

    /// Fold the nodes `range` of a structural level both of whose children
    /// are structural with every count fitting `u64` (`left` and `right`
    /// their raw columns, as [`Self::fold_structural_u64`] reads them) into
    /// `col`, while each total fits the fast lane; returns the first node it
    /// did not fill, `range.end` when it filled them all. An implicit
    /// level's nodes are summed off their pairs generated a run of nodes at
    /// a time ([`Self::fill_described_u64`]).
    ///
    /// The child counts the node [`NODES_AHEAD`] on reads are requested from
    /// the cache while the current one is summed: the reads of a level are
    /// independent, and a wide child's column is read at random.
    pub(crate) fn fill_structural_u64(
        level: &TddLevel,
        left: &[u128],
        right: &[u128],
        col: &mut CountVec,
        range: std::ops::Range<usize>,
    ) -> usize {
        let Some(stored) = level.stored() else {
            return Self::fill_described_u64(level, left, right, col, range);
        };
        col.fill_fast(range, |i| {
            if let Some(ahead) = stored.nodes().get(i + NODES_AHEAD) {
                for p in stored.of(ahead).iter().take(2) {
                    prefetch_at(left, p.left.0 as usize);
                    prefetch_at(right, p.right.0 as usize);
                }
            }
            Self::fold_structural_u64(stored.of_idx(i), left, right)
        })
    }

    /// [`Self::fill_structural_u64`] into a column of `u64` counts whose
    /// children's columns are `u64` too: fills the nodes `range` while each
    /// total fits `u64`, and returns the first node it did not fill. An
    /// implicit level's nodes are summed off their generated pairs
    /// ([`Self::fill_described_narrow`]).
    pub(crate) fn fill_structural_narrow(
        level: &TddLevel,
        left: &[u64],
        right: &[u64],
        col: &mut [u64],
        range: std::ops::Range<usize>,
    ) -> usize {
        let Some(stored) = level.stored() else {
            return Self::fill_described_narrow(level, left, right, col, range);
        };
        for i in range.clone() {
            if let Some(ahead) = stored.nodes().get(i + NODES_AHEAD) {
                for p in stored.of(ahead).iter().take(2) {
                    prefetch_at(left, p.left.0 as usize);
                    prefetch_at(right, p.right.0 as usize);
                }
            }
            let total = match stored.of_idx(i) {
                [p] => Some(left[p.left.0 as usize] as u128 * right[p.right.0 as usize] as u128),
                pairs => Self::fold_structural_by(pairs.iter().copied(),
                    |k| left[k.0 as usize] as u128, |k| right[k.0 as usize] as u128),
            };
            match total {
                Some(total) if total <= u64::MAX as u128 => col[i] = total as u64,
                _ => return i,
            }
        }
        range.end
    }

    /// [`Self::fill_structural_narrow`] on an implicit level, its pairs
    /// generated a run of nodes at a time as [`Self::fill_described_u64`]
    /// generates them.
    #[inline(never)]
    fn fill_described_narrow(
        level: &TddLevel,
        left: &[u64],
        right: &[u64],
        col: &mut [u64],
        range: std::ops::Range<usize>,
    ) -> usize {
        let Pairs::Implicit(d) = level.pair_view() else {
            return range.start;
        };
        let k = d.pairs_per_node();
        if k == 0 {
            return range.start;
        }
        let mut chunks = NodeChunks::new(d, range.clone());
        let mut buf = Vec::new();
        while let Some(start) = chunks.fill(&mut buf) {
            for i in start..start + buf.len() / k {
                let pairs = buf[(i - start) * k..][..k].iter().copied();
                match Self::fold_structural_by(pairs, |x| left[x.0 as usize] as u128, |x| right[x.0 as usize] as u128) {
                    Some(total) if total <= u64::MAX as u128 => col[i] = total as u64,
                    _ => return i,
                }
            }
        }
        range.end
    }

    /// [`Self::fill_structural_u64`] on an implicit level: the pairs of the
    /// nodes `range` generated off the level's description a run of nodes
    /// at a time ([`NodeChunks`]), `k` a node, and each node summed as a
    /// stored node's pairs are. Out of line, so that the stored levels'
    /// fill stays small where it is called.
    #[inline(never)]
    fn fill_described_u64(
        level: &TddLevel,
        left: &[u128],
        right: &[u128],
        col: &mut CountVec,
        range: std::ops::Range<usize>,
    ) -> usize {
        let Pairs::Implicit(d) = level.pair_view() else {
            return range.start;
        };
        let k = d.pairs_per_node();
        if k == 0 {
            return range.start;
        }
        let mut chunks = NodeChunks::new(d, range.clone());
        let mut buf = Vec::new();
        while let Some(start) = chunks.fill(&mut buf) {
            let end = start + buf.len() / k;
            let filled = col.fill_fast(start..end, |i| Self::fold_structural_u64(&buf[(i - start) * k..][..k], left, right));
            if filled < end {
                return filled;
            }
        }
        range.end
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
