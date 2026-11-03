//! Installing marginal columns and settling the levels that refer to them.

use crate::diagram::{Tdd, TddLevel, WeightStore, WeightValue, assert_can_make_marginal};
use crate::limits::{Limits, OperationError};
use crate::value::{CountVec, FloatCol, FloatFold, IntFold, WeightFold, ValueDomain};
use crate::vtree::{Vtree, VtreeIdx};

/// A vtree index that is known to be an internal node.
///
/// Only an internal level has a column to marginalize: a leaf's values are the three
/// constants of its variable, which every reader resolves by label. Both
/// install paths, [`install_finished`] and [`install_streamed`], take this
/// token, and minting it is the one place that distinction is checked, so no
/// generic marginalization path can reach a leaf's column — the weighted leaf
/// pin (a shared, label-ordered 3-slot cache) depends on nothing ever
/// installing, deduping or compacting it.
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

/// Storage transitions required by a marginalization pass.
pub(crate) trait MarginalDomain: ValueDomain {
    /// The store the marginal columns of this domain go into, when there is an
    /// external one. Only the subsumed-child reclaim needs it generically.
    fn weight_store(store: &mut Self::Store) -> Option<&mut WeightStore>;

    /// Commit a finished column into level `t` mid-apply, turning the level
    /// marginal. The caller has already checked the marginalization
    /// precondition (`diagram::assert_can_make_marginal`).
    ///
    /// The in-flight twin of [`Self::install`]: same column, but the diagram
    /// around it is still being built, so nothing is deduped and no parent
    /// reference is rewritten.
    ///
    /// Returns the column when the domain leaves it with the apply rather
    /// than in the level's store: the apply then keeps it for `t`'s parent
    /// to read, drops it once the parent is marginal, and stores what is
    /// left when its sweep ends.
    fn commit_in_flight(
        levels: &mut [TddLevel],
        t: InternalLevel,
        col: Self::Col,
        store: &mut Self::Store,
    ) -> Option<Self::Col>;

    /// Ready a finished column for [`Self::install`] on a finished diagram,
    /// and say how the slot numbering changed.
    ///
    /// `Some(remap)` means the domain minted canonical slots — `remap[old]` is
    /// where a node's value ended up, and the parent's references into the
    /// level must be rewritten through it. `None` means every node kept its
    /// own slot and a bare reference is already correct.
    fn dedup(col: &mut Self::Col) -> Option<Vec<u32>>;

    /// Install `col`, as [`Self::dedup`] left it, as `t`'s marginal store on a
    /// finished diagram.
    fn install(
        level: &mut TddLevel,
        t: InternalLevel,
        col: Self::Col,
        store: &mut Self::Store,
    );

    /// Sum out a single-variable vtree leaf target into its parent's
    /// references. The lookup-only leaf path: it reads the variable's three
    /// constants and writes references, and never mints a column slot.
    ///
    /// # Errors
    ///
    /// `Err(OperationError::OverBudget)` when `lim` refuses the move of an
    /// implicit parent's references; the leaf and its parent are then as
    /// they were.
    fn sum_out_leaf(
        lim: &Limits,
        tdd: &mut Tdd,
        leaf: VtreeIdx,
        vtree: &Vtree,
        store: &mut Self::Store,
    ) -> Result<(), OperationError>;

    /// Whatever the domain owes the whole diagram once a cascade pass is over.
    /// `was_marginal` is the pass-entry marginality snapshot.
    ///
    /// # Errors
    ///
    /// `Err(OperationError::OverBudget)` when `lim` refuses the move of an
    /// implicit level's references.
    fn end_sweep(lim: &Limits, tdd: &mut Tdd, was_marginal: &[bool]) -> Result<(), OperationError>;

}

impl MarginalDomain for IntFold {
    fn weight_store(_store: &mut ()) -> Option<&mut WeightStore> {
        None
    }

    #[inline]
    fn commit_in_flight(
        levels: &mut [TddLevel],
        t: InternalLevel,
        col: CountVec,
        _store: &mut (),
    ) -> Option<CountVec> {
        let (counts, big) = col.into_parts();
        let i = t.vtree_idx().idx();
        crate::diagram::MarginalStorage::new(&mut levels[i], None, i).install_counts(counts, big);
        None
    }

    /// Counts are deduped before they are installed, so the level satisfies invariant 10 — no
    /// two slots share a value — from birth rather than by a later canon pass.
    /// That is what mints new slot numbers, and why this domain returns a
    /// remap.
    fn dedup(col: &mut CountVec) -> Option<Vec<u32>> {
        Some(col.dedup())
    }

    fn install(
        level: &mut TddLevel,
        t: InternalLevel,
        col: CountVec,
        _store: &mut (),
    ) {
        let (counts, big) = col.into_parts();
        crate::diagram::MarginalStorage::new(level, None, t.vtree_idx().idx()).install_counts(counts, big);
    }

    fn sum_out_leaf(
        lim: &Limits,
        tdd: &mut Tdd,
        leaf: VtreeIdx,
        vtree: &crate::vtree::Vtree,
        _store: &mut (),
    ) -> Result<(), OperationError> {
        crate::marginal::marginalize_leaf_inline(lim, tdd, leaf, vtree)
    }

    /// Make every marginal-side slot reference this pass persisted self-describing,
    /// once, at the pass's chokepoint.
    ///
    /// This path does not run the conjunction sweep, so the end-of-apply
    /// tagger never runs on it, and canon's no-duplicate early return leaves
    /// untouched boundary references raw. The snapshot is what keeps the sweep
    /// off children that a prior pass marginalized: those already carry inline
    /// counts, and re-resolving them as bare slots would misread them.
    fn end_sweep(lim: &Limits, tdd: &mut Tdd, was_marginal: &[bool]) -> Result<(), OperationError> {
        crate::diagram::inline_small_marginal_refs(lim, tdd, Some(was_marginal))
    }
}
impl MarginalDomain for WeightFold {
    fn weight_store(store: &mut WeightStore) -> Option<&mut WeightStore> {
        Some(store)
    }

    #[inline]
    fn commit_in_flight(
        levels: &mut [TddLevel],
        t: InternalLevel,
        col: Vec<WeightValue>,
        store: &mut WeightStore,
    ) -> Option<Vec<WeightValue>> {
        // No parent contract-dirty marking: the shared level-state machine
        // establishes invariant 10 at slot-prune, which runs in weighted mode too via
        // `prune_value_slots`; only the integer
        // count-preservation localizer around it is gated off.
        let i = t.vtree_idx().idx();
        crate::diagram::MarginalStorage::new(&mut levels[i], Some(store), i).install_weights(col);
        None
    }

    /// The weighted store is full width and its references stay bare slots
    /// (slot index == node index), so nothing is minted and the parent's
    /// references need no rewrite.
    fn dedup(_col: &mut Vec<WeightValue>) -> Option<Vec<u32>> {
        None
    }

    fn install(
        level: &mut TddLevel,
        t: InternalLevel,
        col: Vec<WeightValue>,
        store: &mut WeightStore,
    ) {
        // The column is full width, one slot per node, which is the slot count
        // the level records.
        crate::diagram::MarginalStorage::new(level, Some(store), t.vtree_idx().idx()).install_weights(col);
    }

    fn sum_out_leaf(
        lim: &Limits,
        tdd: &mut Tdd,
        leaf: VtreeIdx,
        vtree: &crate::vtree::Vtree,
        store: &mut WeightStore,
    ) -> Result<(), OperationError> {
        crate::marginal::marginalize_leaf_weighted(lim, tdd, leaf, vtree, store)
    }

    /// Nothing: weighted marginal-side references are bare slots end to end, so
    /// there is no tag to apply and no snapshot to key it off.
    fn end_sweep(_lim: &Limits, _tdd: &mut Tdd, _was_marginal: &[bool]) -> Result<(), OperationError> {
        Ok(())
    }
}
/// The streamed domain of the scaled-float arithmetic: its in-flight columns
/// stay with the apply, and a finished diagram's go to the store as
/// [`WeightValue::Scaled`] values, as the weighted domain's do.
impl MarginalDomain for FloatFold {
    fn weight_store(store: &mut WeightStore) -> Option<&mut WeightStore> {
        Some(store)
    }

    /// The level becomes marginal at the column's width with no store
    /// column, and the column goes back to the apply: a parent summed out
    /// in the same sweep reads it there, so it never takes a
    /// [`WeightValue`] per slot.
    #[inline]
    fn commit_in_flight(
        levels: &mut [TddLevel],
        t: InternalLevel,
        col: FloatCol,
        store: &mut WeightStore,
    ) -> Option<FloatCol> {
        let i = t.vtree_idx().idx();
        crate::diagram::MarginalStorage::new(&mut levels[i], Some(store), i).hold_weights(col.len() as u32);
        Some(col)
    }

    fn dedup(_col: &mut FloatCol) -> Option<Vec<u32>> {
        None
    }

    fn install(
        level: &mut TddLevel,
        t: InternalLevel,
        col: FloatCol,
        store: &mut WeightStore,
    ) {
        let values = col.iter().map(WeightValue::scaled).collect();
        WeightFold::install(level, t, values, store);
    }

    fn sum_out_leaf(
        lim: &Limits,
        tdd: &mut Tdd,
        leaf: VtreeIdx,
        vtree: &crate::vtree::Vtree,
        store: &mut WeightStore,
    ) -> Result<(), OperationError> {
        WeightFold::sum_out_leaf(lim, tdd, leaf, vtree, store)
    }

    fn end_sweep(lim: &Limits, tdd: &mut Tdd, was_marginal: &[bool]) -> Result<(), OperationError> {
        WeightFold::end_sweep(lim, tdd, was_marginal)
    }
}

/// Install `col` as `t`'s marginal store and settle the diagram around it:
/// rewrite the parent's references if the domain minted new slots, mark the
/// parent for re-contraction, and free the children `t` now subsumes.
///
/// # Errors
///
/// `Err(OperationError::OverBudget)` when `lim` refuses the move of an
/// implicit parent's references. It is planned before anything is installed,
/// so `t` is then still structural and its parent as it was.
pub(crate) fn install_finished<K: MarginalDomain>(
    lim: &Limits,
    tdd: &mut Tdd,
    vtree: &Vtree,
    level: InternalLevel,
    mut col: K::Col,
    store: &mut K::Store,
) -> Result<(), OperationError> {
    let t = level.vtree_idx();
    assert_can_make_marginal(&tdd.levels, vtree, t);

    let remap = K::dedup(&mut col);
    tdd.install_marginal_level(lim, t, remap.as_deref(), |storage| K::install(storage, level, col, store))?;

    // `t` now subsumes its children — free their dead stores (O(1)).
    free_subsumed_marginal_children(&mut tdd.levels, vtree, t, K::weight_store(store));
    Ok(())
}

/// Install a streamed column without renumbering slots, then release its
/// children, with any column of theirs the apply kept in `computed`.
#[inline]
pub(crate) fn install_streamed<K: MarginalDomain>(
    levels: &mut [TddLevel],
    vtree: &Vtree,
    level: InternalLevel,
    col: K::Col,
    computed: &mut [Option<K::Col>],
    store: &mut K::Store,
) {
    let t = level.vtree_idx();
    assert_can_make_marginal(levels, vtree, t);
    computed[t.idx()] = K::commit_in_flight(levels, level, col, store);
    free_subsumed_marginal_children(levels, vtree, t, K::weight_store(store));
    let (l, r) = vtree.children(t);
    computed[l.idx()] = None;
    computed[r.idx()] = None;
}

/// Where a [`cascade`] installs the columns it takes: a finished diagram, or
/// an apply's output levels while they are still being built.
pub(crate) trait InstallTarget {
    /// The levels, read for the marginality test on the way down.
    fn levels(&self) -> &[TddLevel];

    /// Install `col` as level `t`'s marginal store. `computed` is the
    /// cascade's column table, for a target that keeps a column there.
    ///
    /// # Errors
    ///
    /// `Err(OperationError::OverBudget)` when `lim` refuses what a finished
    /// diagram's install reserves; `t` is then still structural.
    fn install<K: MarginalDomain>(
        &mut self,
        lim: &Limits,
        vtree: &Vtree,
        t: InternalLevel,
        col: K::Col,
        computed: &mut [Option<K::Col>],
        store: &mut K::Store,
    ) -> Result<(), OperationError>;
}

/// Installs through [`install_finished`], which dedups the column and
/// rewrites the parent's references.
impl InstallTarget for Tdd {
    fn levels(&self) -> &[TddLevel] {
        &self.levels
    }

    fn install<K: MarginalDomain>(
        &mut self,
        lim: &Limits,
        vtree: &Vtree,
        t: InternalLevel,
        col: K::Col,
        _computed: &mut [Option<K::Col>],
        store: &mut K::Store,
    ) -> Result<(), OperationError> {
        install_finished::<K>(lim, self, vtree, t, col, store)
    }
}

/// Installs through [`install_streamed`]: the parent is still being built, so
/// no reference is rewritten and nothing is refused.
impl InstallTarget for [TddLevel] {
    fn levels(&self) -> &[TddLevel] {
        self
    }

    fn install<K: MarginalDomain>(
        &mut self,
        _lim: &Limits,
        vtree: &Vtree,
        t: InternalLevel,
        col: K::Col,
        computed: &mut [Option<K::Col>],
        store: &mut K::Store,
    ) -> Result<(), OperationError> {
        install_streamed::<K>(self, vtree, t, col, computed, store);
        Ok(())
    }
}

/// Marginalize `t` and every still-structural internal level below it from
/// the columns an ensure walk left in `computed`, children before parents, so
/// both children of a level are marginal or leaves by the time it is
/// installed.
///
/// A level with no column in `computed` is left structural: the walk that
/// filled `computed` did not visit it.
///
/// # Errors
///
/// The first refusal of an install ([`InstallTarget::install`]). The levels
/// installed before it stay installed, children before parents, as a cascade
/// from them would have left them; the level refused and those above it are
/// still structural.
pub(crate) fn cascade<K: MarginalDomain, S: InstallTarget + ?Sized>(
    lim: &Limits,
    target: &mut S,
    vtree: &Vtree,
    t: VtreeIdx,
    computed: &mut [Option<K::Col>],
    store: &mut K::Store,
) -> Result<(), OperationError> {
    let Some(level) = InternalLevel::new(vtree, t) else {
        return Ok(());
    };
    if target.levels()[t.idx()].is_marginal() {
        return Ok(());
    }
    let (left, right) = vtree.children(t);
    cascade::<K, S>(lim, target, vtree, left, computed, store)?;
    cascade::<K, S>(lim, target, vtree, right, computed, store)?;
    if let Some(col) = computed[t.idx()].take() {
        target.install::<K>(lim, vtree, level, col, computed, store)?;
    }
    Ok(())
}

/// Free the per-node store of `parent`'s marginal children; call it at the
/// moment `parent` becomes marginal.
///
/// A marginal parent carries no pair lists, so nothing reads its children's
/// stores again. The integer store is emptied and the weighted column cleared
/// through `ws`; either way the child's `slot_count()` then reports 0 while it stays
/// marginal. A weight-marginal leaf's column is left alone (invariant 11).
pub(crate) fn free_subsumed_marginal_children(
    levels: &mut [TddLevel],
    vtree: &Vtree,
    parent: VtreeIdx,
    mut ws: Option<&mut WeightStore>,
) {
    if vtree.node(parent).is_leaf() {
        return;
    }
    let (l, r) = vtree.children(parent);
    for c in [l.idx(), r.idx()] {
        crate::diagram::MarginalStorage::new(&mut levels[c], ws.as_deref_mut(), c)
            .clear(vtree.node(VtreeIdx(c as u32)).is_leaf());
    }
}

