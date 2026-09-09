//! The value domain a marginalization freezes a level into.
//!
//! Two domains exist: integer model counts, and exact semiring weights held in
//! an external [`WeightStore`]. They differ in three ways and no more — what a
//! node folds to, where the frozen column lives, and how a parent references a
//! frozen node — so the driver in [`super::fold`] is written once against this
//! trait and each domain supplies only those three answers.

use crate::counts::{
    unwrap_infallible, ColumnRetention, Count, CountVec, IntFold, MargFold, RecoveryPanic,
    WeightFold,
};
use crate::diagram::{Tdd, TddLevel};
use crate::engine::Engine;
use crate::diagram::WeightVal;
use crate::vtree::{Vtree, VtreeIdx};
use crate::weight_store::WeightStore;

use super::leaf::{marginalize_leaf_inline, marginalize_leaf_weighted};
use super::store::{
    compute_marginal_node_int, compute_marginal_node_weight, dedup_fresh_store, ensure_counts,
    ensure_weights,
};

/// A vtree index that is known to be an internal node.
///
/// Only an internal level has a column to freeze: a leaf's values are the three
/// constants of its variable, which every reader resolves by label. Minting
/// this token is the one place that distinction is checked, so no generic
/// freeze path can reach a leaf's column — the weighted leaf pin (a shared,
/// label-ordered 3-slot cache) depends on nothing ever installing, deduping or
/// compacting it.
#[derive(Copy, Clone, Debug)]
pub(super) struct InternalLevel(VtreeIdx);

impl InternalLevel {
    /// `None` at a vtree leaf.
    #[inline]
    pub(super) fn new(vtree: &Vtree, t: VtreeIdx) -> Option<Self> {
        (!vtree.node(t).is_leaf()).then_some(InternalLevel(t))
    }

    #[inline]
    pub(super) fn vtree_idx(self) -> VtreeIdx {
        self.0
    }
}

/// The scratch column of one level, in the domain's value type.
pub(super) type Column<K> = <<K as ValueKind>::Fold as MargFold>::Col<RecoveryPanic>;

/// One value domain of marginalization.
pub(super) trait ValueKind {
    /// The fold discipline over this domain's scalar — [`IntFold`]'s two-pass
    /// u128/`BigUint` count, or [`WeightFold`]'s single exact pass.
    type Fold: MargFold;
    /// State the domain carries beside the diagram: the weight store, or
    /// nothing at all.
    type Store;

    /// The store the frozen columns of this domain go into, when there is an
    /// external one. Only the subsumed-child reclaim needs it generically.
    fn weight_store(store: &mut Self::Store) -> Option<&mut WeightStore>;

    /// A fresh `width`-slot column of zeros.
    fn alloc_column(eng: &Engine, width: usize, store: &Self::Store) -> Column<Self>;

    /// Store one node's folded value at slot `i`.
    fn set_slot(eng: &Engine, col: &mut Column<Self>, i: usize, v: <Self::Fold as MargFold>::Scalar);

    /// Populate `computed[t]` and every column below it that the fold at `t`
    /// will read — the one shared bottom-up ensure walk, with this domain's
    /// readers wired in.
    fn ensure(
        eng: &Engine,
        tdd: &Tdd,
        t: VtreeIdx,
        vtree: &Vtree,
        store: &Self::Store,
        computed: &mut [Option<Column<Self>>],
        retain: ColumnRetention,
    );

    /// Fold node `i` of level `d`: `Σ over pairs (left × right)`.
    fn fold_node(
        tdd: &Tdd,
        level: &TddLevel,
        i: usize,
        li: usize,
        ri: usize,
        store: &Self::Store,
        computed: &[Option<Column<Self>>],
    ) -> <Self::Fold as MargFold>::Scalar;

    /// Install `col` as `t`'s frozen store, and say how the slot numbering
    /// changed.
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
    fn sum_out_leaf(eng: &Engine, tdd: &mut Tdd, leaf: VtreeIdx, vtree: &Vtree, store: &mut Self::Store);

    /// Whatever the domain owes the whole diagram once the pass is over.
    /// `was_frozen` is the pass-entry marginality snapshot.
    fn end_sweep(tdd: &mut Tdd, was_frozen: &[bool]);
}

/// Integer model counts, stored inside the level itself.
pub(super) struct IntValues;

/// Exact semiring weights, stored in the diagram's external [`WeightStore`].
pub(super) struct WeightValues;

impl ValueKind for IntValues {
    type Fold = IntFold;
    type Store = ();

    fn weight_store(_store: &mut ()) -> Option<&mut WeightStore> {
        None
    }

    fn alloc_column(eng: &Engine, width: usize, _store: &()) -> CountVec<RecoveryPanic> {
        CountVec::with_width(eng, width)
    }

    fn set_slot(eng: &Engine, col: &mut CountVec<RecoveryPanic>, i: usize, v: Count) {
        col.set_i(eng, i, v);
    }

    fn ensure(
        eng: &Engine,
        tdd: &Tdd,
        t: VtreeIdx,
        vtree: &Vtree,
        _store: &(),
        computed: &mut [Option<CountVec<RecoveryPanic>>],
        _retain: ColumnRetention,
    ) {
        // `ColumnRetention::All` is not a choice here: the cascade takes every
        // walked level's column to install it as that level's store.
        ensure_counts(eng, tdd, t, vtree, computed);
    }

    fn fold_node(
        tdd: &Tdd,
        level: &TddLevel,
        i: usize,
        li: usize,
        ri: usize,
        _store: &(),
        computed: &[Option<CountVec<RecoveryPanic>>],
    ) -> Count {
        compute_marginal_node_int(tdd, level, i, li, ri, computed)
    }

    /// Counts are deduped before they are installed, so the level is C3 — no
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
        let (counts, big, remap) = dedup_fresh_store(fast, big);
        tdd.levels[t.vtree_idx().idx()].make_marginal(counts, big);
        Some(remap)
    }

    fn sum_out_leaf(eng: &Engine, tdd: &mut Tdd, leaf: VtreeIdx, vtree: &Vtree, _store: &mut ()) {
        marginalize_leaf_inline(eng, tdd, leaf, vtree);
    }

    /// Make every marg-side slot reference this pass persisted self-describing,
    /// once, at the pass's chokepoint.
    ///
    /// This path does not go through `apply_and_fallible`, so the end-of-apply
    /// tagger never runs on it, and canon's no-duplicate early return leaves
    /// untouched boundary references raw. The snapshot is what keeps the sweep
    /// off children that a PRIOR pass marginalized: those already carry inline
    /// counts, and re-resolving them as bare slots would misread them.
    fn end_sweep(tdd: &mut Tdd, was_frozen: &[bool]) {
        crate::diagram::tag_all_marg_side_slots(tdd, Some(was_frozen));
    }
}

impl ValueKind for WeightValues {
    type Fold = WeightFold;
    type Store = WeightStore;

    fn weight_store(store: &mut WeightStore) -> Option<&mut WeightStore> {
        Some(store)
    }

    fn alloc_column(eng: &Engine, width: usize, store: &WeightStore) -> Vec<WeightVal> {
        unwrap_infallible(WeightFold::alloc_col::<RecoveryPanic>(eng, width, &store.wzero()))
    }

    fn set_slot(eng: &Engine, col: &mut Vec<WeightVal>, i: usize, v: WeightVal) {
        unwrap_infallible(WeightFold::set_col::<RecoveryPanic>(eng, col, i, v));
    }

    fn ensure(
        eng: &Engine,
        tdd: &Tdd,
        t: VtreeIdx,
        vtree: &Vtree,
        store: &WeightStore,
        computed: &mut [Option<Vec<WeightVal>>],
        retain: ColumnRetention,
    ) {
        ensure_weights(eng, tdd, t, vtree, store, computed, retain);
    }

    fn fold_node(
        tdd: &Tdd,
        level: &TddLevel,
        i: usize,
        li: usize,
        ri: usize,
        store: &WeightStore,
        computed: &[Option<Vec<WeightVal>>],
    ) -> WeightVal {
        compute_marginal_node_weight(tdd, level, i, li, ri, store, computed)
    }

    /// The weighted store is full width and its references stay bare slots
    /// (slot index == node index), so nothing is minted and the parent's
    /// references need no rewrite.
    fn install(
        tdd: &mut Tdd,
        t: InternalLevel,
        col: Vec<WeightVal>,
        store: &mut WeightStore,
    ) -> Option<Vec<u32>> {
        let ti = t.vtree_idx().idx();
        tdd.levels[ti].make_marginal_weighted();
        store.set_level(ti, col);
        None
    }

    fn sum_out_leaf(eng: &Engine, tdd: &mut Tdd, leaf: VtreeIdx, vtree: &Vtree, store: &mut WeightStore) {
        marginalize_leaf_weighted(eng, tdd, leaf, vtree, store);
    }

    /// Nothing: weighted marg-side references are bare slots end to end, so
    /// there is no tag to apply and no snapshot to key it off.
    fn end_sweep(_tdd: &mut Tdd, _was_frozen: &[bool]) {}
}
