//! Arithmetic and column access shared by standalone and streaming folds.
//!
//! Two domains exist — integer model counts, and exact semiring weights held
//! in an external [`WeightStore`] — and two folds consume them: the
//! marginalization cascade over a finished diagram, and the streaming column
//! built inside an apply. Both folds are written once, against
//! [`ValueDomain`]; where they differ is the reservation policy of the scratch
//! column, which is a method type parameter.

use crate::diagram::{ChildPair, Tdd, TddLevel, WeightStore};
use crate::engine::Engine;
use crate::limits::{ApplyBudget, RecoveryPanic, ReservePolicy};
use crate::limits::OperationError;
use crate::vtree::{Vtree, VtreeIdx};

use super::{walk_bottom_up, ColumnRetention, MarginalFold, StreamCache};

/// The scratch column of one level of the marginalization cascade, which
/// reserves through [`RecoveryPanic`].
pub(crate) type Column<D> = <D as MarginalFold>::Col<RecoveryPanic>;

/// The diagram a fold reads: its vtree, its levels, and the domain's store.
pub(crate) struct FoldInput<'a, D: ValueDomain> {
    pub(crate) vtree: &'a Vtree,
    pub(crate) levels: &'a [TddLevel],
    pub(crate) store: &'a D::Store,
}

impl<D: ValueDomain> Clone for FoldInput<'_, D> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<D: ValueDomain> Copy for FoldInput<'_, D> {}

/// One level of the ensure walk, as [`ValueDomain::fold_node`] sees it: the
/// level and its two children, the diagram, the columns computed so far, and
/// the domain's zero.
pub(crate) struct FoldScope<'a, D: ValueDomain, R: ReservePolicy> {
    pub(crate) lvl: usize,
    pub(crate) left: usize,
    pub(crate) right: usize,
    pub(crate) input: FoldInput<'a, D>,
    pub(crate) computed: &'a [Option<D::Col<R>>],
    pub(crate) zero: &'a D::Scalar,
}

/// Per-child read view of the child level's fold column, taken before the dense
/// scatter loop.
///
/// The column is borrowed from the child level's marginal storage or the
/// per-apply scratch, not copied, so a wide marginal child's storage is not
/// doubled; the caller splits the output level and its two children apart once
/// so the views coexist with the `&mut` output borrow. In the integer domain
/// the column's `all_u64` certificate lets the cell fold use a widening
/// `u64×u64→u128` multiply, which cannot overflow, in place of `checked_mul`;
/// inline refs decode to at most `MARGINAL_INLINE_MAX`, so the certificate
/// covers every read in the cell loop.
pub(crate) struct StreamChild<'a, D: ValueDomain> {
    pub(crate) col: D::ChildCol<'a>,
    /// True iff this view is of a marginal child level (its refs are
    /// marginal-side slot refs, possibly bit-30 tagged). When set, the per-cell
    /// fold decodes the ref before indexing the column. For a
    /// non-marginal/leaf child the ref is a plain node index (bit 30 may be a
    /// real high bit) — do not decode.
    pub(crate) is_marginal: bool,
}

/// One value domain: the arithmetic, its column, and what the two folds need
/// from it.
///
/// Each method is a place where the two domains differ; what they share (the
/// bottom-up ensure walk, the `Σ pairs (left × right)` fold, the column
/// contract of [`MarginalFold`]) is written once and takes no hook here.
pub(crate) trait ValueDomain: MarginalFold + Sized {
    /// State the domain carries beside the diagram: the weight store, or
    /// nothing at all.
    type Store;

    /// How this domain reads one child's column for the duration of a row
    /// loop. Borrowed wherever the storage can lend a reference; the weighted
    /// `WeightStore` column is copied, so that one case stays owned.
    type ChildCol<'a>;

    /// The additive identity, which the weighted domain must read from its
    /// store. Resolved once per walk, not once per node: a weighted zero is a
    /// `BigRational` clone.
    fn zero(store: &Self::Store) -> Self::Scalar;

    /// This domain's already-computed child columns inside the per-apply cache.
    ///
    /// The cache is one enum because an apply runs a single value kind
    /// throughout; this hook is where that kind is read back out.
    fn stream_columns(cache: &StreamCache) -> &[Option<Self::Col<ApplyBudget>>];

    /// This domain's store, given the apply's weight store.
    ///
    /// The integer domain carries no state and ignores the argument; the
    /// weighted domain requires one, and a weighted column is only ever opened
    /// where a store is attached.
    fn store_of(ws: Option<&WeightStore>) -> &Self::Store;

    /// Fold node `i` of the scope's level: `Σ over its pairs (left × right)`,
    /// with this domain's child readers resolving each `u32` ref against the
    /// levels and the columns computed so far.
    ///
    /// Impls carry `#[inline]`: the ensure walk's per-node loop calls this once
    /// per node, and must not gain a call there.
    fn fold_node<R: ReservePolicy>(at: &FoldScope<'_, Self, R>, i: usize) -> Self::Scalar;

    /// Open a read view of child level `left_idx`'s column. `level` is `levels[left_idx]`,
    /// handed in already split off from the output level's `&mut` borrow.
    ///
    /// Fallible only where the view must be materialized (the weighted
    /// `WeightStore` column), whose copy is reserved through `R` and so can
    /// return `OverBudget`; every borrowing case allocates nothing.
    fn child_view<'a, R: ReservePolicy>(
        eng: &Engine,
        left_idx: usize,
        vtree: &Vtree,
        level: &'a TddLevel,
        computed: &'a [Option<Self::Col<R>>],
        store: &Self::Store,
    ) -> Result<StreamChild<'a, Self>, OperationError>;

    /// Collapse one alive cell's collected pairs to a single scalar:
    /// `Σ left[p.left] × right[p.right]`.
    fn fold_cell(
        pairs: &[ChildPair],
        left: &StreamChild<'_, Self>,
        right: &StreamChild<'_, Self>,
        store: &Self::Store,
    ) -> Self::Scalar;

    /// Populate `computed[root]` and every column below it that a fold at
    /// `root` will read.
    ///
    /// One walk for both folds. `already_marginal` is the caller's answer to
    /// "this level's values are already stored, don't recompute them" — the
    /// cascade and the apply key that on different state, which is why it is
    /// asked of the caller rather than of the domain.
    fn ensure<R: ReservePolicy>(
        eng: &Engine,
        root: VtreeIdx,
        input: FoldInput<'_, Self>,
        computed: &mut [Option<Self::Col<R>>],
        already_marginal: &dyn Fn(usize) -> bool,
        retain: ColumnRetention,
    ) -> Result<(), R::Err> {
        let FoldInput { vtree, levels, store } = input;
        let zero = Self::zero(store);
        walk_bottom_up(
            vtree,
            root,
            computed,
            // A leaf's values resolve on demand inside the readers, so it
            // never gets a column.
            |computed, i| {
                computed[i].is_some()
                    || already_marginal(i)
                    || vtree.node(VtreeIdx(i as u32)).is_leaf()
            },
            |computed, t| {
                let lvl = t.idx();
                let (l, r) = vtree.children(t);
                // Reserved through `R`: an infallible `vec![zero; width]`
                // would abort past the recovery cascade on a wide level.
                let mut col = Self::alloc_col::<R>(eng, levels[lvl].slot_count(), &zero)?;
                let at = FoldScope { lvl, left: l.idx(), right: r.idx(), input, computed, zero: &zero };
                for (i, _pairs) in levels[lvl].internal_inputs_iter() {
                    let v = Self::fold_node(&at, i);
                    Self::set_col(eng, &mut col, i, v)?;
                }
                computed[lvl] = Some(col);
                Ok(())
            },
            |computed, i| computed[i] = None,
            // The root is never released, so nothing is exempt.
            retain.frontier(root),
        )
    }
}

/// Where a marginal level's per-slot values live, for the slot prune
/// (`reduce::slot_prune`). Implemented on the same two domains as
/// [`ValueDomain`].
pub(crate) trait SlotStore {
    /// Slot count of level `v`'s store: the domain of the remap that
    /// `compact_store` fills, and the pre-compaction width.
    fn store_len(tdd: &Tdd, v: VtreeIdx) -> usize;

    /// Compact level `v`'s store to `referenced` with value-dedup, write
    /// the composed `old_slot → new_slot` map into `remap`, and commit the
    /// compacted store. Returns `(new_len, values_merged)`; `values_merged`
    /// counts referenced slots that landed on an earlier equal-valued slot.
    fn compact_store(tdd: &mut Tdd, v: VtreeIdx, referenced: &[u32], remap: &mut [u32]) -> (usize, usize);

    /// Fold a completed compaction of level `v` into the retirement tally.
    /// **The two impls are inverted and must stay that way** (increment vs
    /// assign) — see each impl's comment.
    fn update_width(tdd: &mut Tdd, v: VtreeIdx, freed: usize, new_len: usize);
}

mod count;
mod weight;
