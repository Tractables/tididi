//! What a value domain is: one arithmetic, one column, and the handful of
//! answers the two folds need from it.
//!
//! Two domains exist — integer model counts, and exact semiring weights held
//! in an external [`WeightStore`] — and two folds consume them: the
//! marginalization cascade over a finished diagram, and the streaming column
//! built inside an apply. Both folds are written once, against
//! [`ValueDomain`], so a domain answers each question exactly once no matter
//! which fold is asking. Where the two folds genuinely differ, they differ in
//! the reservation policy of the scratch column, which rides along as a method
//! type parameter rather than splitting the contract in two.

use crate::diagram::{InputPair, Tdd, TddLevel, WeightStore};
use crate::engine::{Engine, RecoveryPanic, ReservePolicy};
use crate::error::ApplyError;
use crate::vtree::{Vtree, VtreeIdx};

use super::{ensure_fold_walk, ColumnRetention, MargFold};

/// A vtree index that is known to be an internal node.
///
/// Only an internal level has a column to marginalize: a leaf's values are the three
/// constants of its variable, which every reader resolves by label. Minting
/// this token is the one place that distinction is checked, so no generic
/// marginalize path can reach a leaf's column — the weighted leaf pin (a shared,
/// label-ordered 3-slot cache) depends on nothing ever installing, deduping or
/// compacting it.
#[derive(Copy, Clone, Debug)]
pub(crate) struct InternalLevel(VtreeIdx);

impl InternalLevel {
    /// `None` at a vtree leaf.
    #[inline]
    pub(crate) fn new(vtree: &Vtree, t: VtreeIdx) -> Option<Self> {
        (!vtree.node(t).is_leaf()).then_some(InternalLevel(t))
    }

    #[inline]
    pub(crate) fn vtree_idx(self) -> VtreeIdx {
        self.0
    }
}

/// The scratch column of one level of the marginalization cascade, which
/// reserves through [`RecoveryPanic`].
pub(crate) type Column<D> = <D as MargFold>::Col<RecoveryPanic>;

/// Per-child READ VIEW of the child level's fold column, taken before the dense
/// scatter loop.
///
/// Borrowed, not copied: the column lives in the child level's marginal storage
/// (or the per-apply computed scratch) and is read there in place. `t` and its
/// two vtree children are distinct nodes, so the caller splits the three level
/// slots apart once (`slice::get_disjoint_mut` in the driver loop) and the
/// child views coexist with the `&mut level` borrow taken for output. Copying
/// them instead doubled a wide marginal child's storage at exactly the moment
/// streaming exists to relieve — a 536M-slot column is an 8 GiB single alloc.
///
/// Integer instantiation (`StreamChild<IntFold>`, aliased
/// `StreamChildCounts`): the `all_u64` certificate rides on the `CountRef`
/// — when both children certify, the cell fold takes the widening-multiply
/// fast path (`u64×u64→u128` is a single `mul` that can never overflow the
/// u128 product, max (2^64-1)^2 < 2^128), so the heavy u128 `checked_mul` is
/// skipped and only the running-total `checked_add` guards overflow. Inline
/// bit-30-tagged refs always decode to ≤ MARG_INLINE_MAX (u64), so the
/// certificate over the column alone covers every read in the cell loop.
pub(crate) struct StreamChild<'a, D: ValueDomain> {
    pub(crate) col: D::ChildCol<'a>,
    /// True iff this view is of a marginal child level (its refs are
    /// marg-side slot refs, possibly bit-30 tagged). When set, the per-cell
    /// fold decodes the ref before indexing the column. For a
    /// non-marginal/leaf child the ref is a plain node index (bit 30 may be a
    /// real high bit) — do NOT decode.
    pub(crate) is_marg: bool,
}

/// One value domain: the arithmetic, its column, and what the two folds need
/// from it.
///
/// Every method is a place where the two domains genuinely differ. What they
/// share — the bottom-up ensure walk, the `Σ pairs (left × right)` discipline,
/// the column contract of [`MargFold`] — is written once elsewhere and takes
/// no hook here.
pub(crate) trait ValueDomain: MargFold + Sized {
    /// State the domain carries beside the diagram: the weight store, or
    /// nothing at all.
    type Store;

    /// How this domain reads ONE child's column for the duration of a row
    /// loop. Borrowed wherever the storage can lend a reference; the weighted
    /// `WeightStore` column is copied, so that one case stays owned.
    type ChildCol<'a>;

    /// The additive identity, which the weighted domain must read from its
    /// store. Resolved once per walk, not once per node: a weighted zero is a
    /// `BigRational` clone.
    fn zero(store: &Self::Store) -> Self::Scalar;

    /// The store the marginal columns of this domain go into, when there is an
    /// external one. Only the subsumed-child reclaim needs it generically.
    fn weight_store(store: &mut Self::Store) -> Option<&mut WeightStore>;

    /// Fold node `i` of `levels[lvl]`: `Σ over its pairs (left × right)`, with
    /// this domain's child readers resolving each `u32` ref against `levels`
    /// and the per-batch `computed` scratch.
    ///
    /// Impls carry `#[inline]`: the pre-unification form was a closure the
    /// ensure walk monomorphized and inlined, and the walk's per-node loop must
    /// not gain a call.
    // The per-level scratch buffers are passed as separate parameters so the
    // borrow checker can split them; bundling them in a struct would force one
    // shared borrow across the level loop.
    #[allow(clippy::too_many_arguments)]
    fn fold_node<R: ReservePolicy>(
        lvl: usize,
        i: usize,
        l_i: usize,
        r_i: usize,
        vtree: &Vtree,
        levels: &[TddLevel],
        computed: &[Option<Self::Col<R>>],
        zero: &Self::Scalar,
        store: &Self::Store,
    ) -> Self::Scalar;

    /// Open a read view of child level `left_idx`'s column. `level` is `levels[left_idx]`,
    /// handed in already split off from the output level's `&mut` borrow.
    ///
    /// Fallible only for the one case that must still materialize (the
    /// weighted `WeightStore` column): that copy can be multi-GiB on extreme
    /// widths, so it propagates `OverBudget` (→ v-split / conditioning-deepen
    /// recovery) instead of aborting. Every borrowing case allocates nothing
    /// and cannot fail.
    fn child_view<'a, R: ReservePolicy>(
        eng: &Engine,
        left_idx: usize,
        vtree: &Vtree,
        level: &'a TddLevel,
        computed: &'a [Option<Self::Col<R>>],
        store: &Self::Store,
    ) -> Result<StreamChild<'a, Self>, ApplyError>;

    /// Collapse one alive cell's collected pairs to a single scalar:
    /// `Σ left[p.left] × right[p.right]`.
    fn fold_cell(
        pairs: &[InputPair],
        left: &StreamChild<'_, Self>,
        right: &StreamChild<'_, Self>,
        store: &Self::Store,
    ) -> Self::Scalar;

    /// Commit a finished column into `levels[left_idx]` mid-apply, turning the level
    /// marginal. The caller has already checked the marginalization
    /// precondition (`diagram::assert_can_make_marginal`).
    ///
    /// The in-flight twin of [`Self::install`]: same column, but the diagram
    /// around it is still being built, so nothing is deduped and no parent
    /// reference is rewritten.
    fn commit_in_flight<R: ReservePolicy>(
        levels: &mut [TddLevel],
        left_idx: usize,
        col: Self::Col<R>,
        store: &mut Self::Store,
    );

    /// Install `col` as `t`'s marginal store on a finished diagram, and say how
    /// the slot numbering changed.
    ///
    /// `Some(remap)` means the domain minted canonical slots — `remap[old]` is
    /// where a node's value ended up, and the parent's references into `t` must
    /// be rewritten through it. `None` means every node kept its own slot and a
    /// bare reference is already correct.
    fn install(
        tdd: &mut Tdd,
        t: InternalLevel,
        col: Column<Self>,
        store: &mut Self::Store,
    ) -> Option<Vec<u32>>;

    /// Sum out a single-variable vtree LEAF target into its parent's
    /// references. The lookup-only leaf path: it reads the variable's three
    /// constants and writes references, and never mints a column slot.
    fn sum_out_leaf(
        eng: &Engine,
        tdd: &mut Tdd,
        leaf: VtreeIdx,
        vtree: &Vtree,
        store: &mut Self::Store,
    );

    /// Whatever the domain owes the whole diagram once a cascade pass is over.
    /// `was_marginal` is the pass-entry marginality snapshot.
    fn end_sweep(tdd: &mut Tdd, was_marginal: &[bool]);

    /// Populate `computed[left_idx]` and every column below it that a fold at `left_idx`
    /// will read.
    ///
    /// One walk for both folds. `already_marginal` is the caller's answer to
    /// "this level's values are already stored, don't recompute them" — the
    /// cascade and the apply key that on different state, which is why it is
    /// asked of the caller rather than of the domain.
    #[allow(clippy::too_many_arguments)]
    fn ensure<R: ReservePolicy>(
        eng: &Engine,
        left_idx: usize,
        vtree: &Vtree,
        levels: &[TddLevel],
        computed: &mut [Option<Self::Col<R>>],
        store: &Self::Store,
        already_marginal: &dyn Fn(usize) -> bool,
        retain: ColumnRetention,
    ) -> Result<(), R::Err> {
        let zero = Self::zero(store);
        ensure_fold_walk::<Self, R, _, _>(
            eng,
            left_idx,
            vtree,
            levels,
            computed,
            &zero,
            &already_marginal,
            &|lvl, i, l_i, r_i, computed| {
                Self::fold_node(lvl, i, l_i, r_i, vtree, levels, computed, &zero, store)
            },
            retain,
        )
    }
}

/// Where a marginal level's per-slot VALUES live, for the one prune skeleton
/// (`reduce::slot_prune`). Implemented on the same two domains as
/// [`ValueDomain`], so where a domain's values live sits next to how they
/// fold. Four hooks, each a place where the two domains genuinely differ.
pub(crate) trait SlotStore {
    /// Slot count of level `v`'s store: the domain of the remap that
    /// `compact_store` fills, and the pre-compaction width.
    fn store_len(tdd: &Tdd, v: VtreeIdx) -> usize;

    /// Free level `v`'s DEAD DEEP store (its marginal parent already consumed
    /// these values), returning the slot count freed. Returns 0 — touching
    /// nothing — when the store is already empty. The level stays in marginal
    /// mode; only the payload goes.
    fn clear_dead_store(tdd: &mut Tdd, v: VtreeIdx) -> usize;

    /// Compact level `v`'s store to `referenced` with value-dedup, write
    /// the composed `old_slot → new_slot` map into `remap`, and commit the
    /// compacted store. Returns `(new_len, values_merged)`; `values_merged`
    /// counts referenced slots that landed on an earlier equal-valued slot.
    fn compact_store(tdd: &mut Tdd, v: VtreeIdx, referenced: &[u32], remap: &mut [u32]) -> (usize, usize);

    /// Fold a completed compaction of level `v` into the retirement tally.
    /// **The two impls are INVERTED and must stay that way** (increment vs
    /// assign) — see each impl's comment.
    fn update_width(tdd: &mut Tdd, v: VtreeIdx, freed: usize, new_len: usize);
}
