//! Installing marginal columns and settling the levels that refer to them.

use crate::diagram::{Tdd, TddLevel, WeightStore, WeightValue, assert_can_make_marginal};
use crate::value::{Column, CountVec, IntFold, WeightFold, ValueDomain};
use crate::vtree::{Vtree, VtreeIdx};
use crate::value::slots::{compact_count_slots, truncate_with_slack};
use crate::diagram::CountOverflow;

/// A vtree index that is known to be an internal node.
///
/// Only an internal level has a column to marginalize: a leaf's values are the three
/// constants of its variable, which every reader resolves by label. Minting
/// this token is the one place that distinction is checked, so no generic
/// marginalization path can reach a leaf's column — the weighted leaf pin (a shared,
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

/// Storage transitions required by a marginalization pass.
pub(crate) trait MarginalDomain: ValueDomain {
    /// The store the marginal columns of this domain go into, when there is an
    /// external one. Only the subsumed-child reclaim needs it generically.
    fn weight_store(store: &mut Self::Store) -> Option<&mut WeightStore>;

    /// Commit a finished column into `levels[left_idx]` mid-apply, turning the level
    /// marginal. The caller has already checked the marginalization
    /// precondition (`diagram::assert_can_make_marginal`).
    ///
    /// The in-flight twin of [`Self::install`]: same column, but the diagram
    /// around it is still being built, so nothing is deduped and no parent
    /// reference is rewritten.
    fn commit_in_flight(
        levels: &mut [TddLevel],
        left_idx: usize,
        col: Self::Col,
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
        level: &mut TddLevel,
        t: InternalLevel,
        col: Column<Self>,
        store: &mut Self::Store,
    ) -> Option<Vec<u32>>;

    /// Sum out a single-variable vtree leaf target into its parent's
    /// references. The lookup-only leaf path: it reads the variable's three
    /// constants and writes references, and never mints a column slot.
    fn sum_out_leaf(
        tdd: &mut Tdd,
        leaf: VtreeIdx,
        vtree: &Vtree,
        store: &mut Self::Store,
    );

    /// Whatever the domain owes the whole diagram once a cascade pass is over.
    /// `was_marginal` is the pass-entry marginality snapshot.
    fn end_sweep(tdd: &mut Tdd, was_marginal: &[bool]);

}

impl MarginalDomain for IntFold {
    fn weight_store(_store: &mut ()) -> Option<&mut WeightStore> {
        None
    }

    #[inline]
    fn commit_in_flight(
        levels: &mut [TddLevel],
        left_idx: usize,
        col: CountVec,
        _store: &mut (),
    ) {
        let (counts, big) = col.into_parts();
        crate::diagram::MarginalStorage::new(&mut levels[left_idx], None, left_idx).install_counts(counts, big);
    }

    /// Counts are deduped before they are installed, so the level satisfies invariant 10 — no
    /// two slots share a value — from birth rather than by a later canon pass.
    /// That is what mints new slot numbers, and why this domain returns a
    /// remap.
    fn install(
        level: &mut TddLevel,
        t: InternalLevel,
        col: CountVec,
        _store: &mut (),
    ) -> Option<Vec<u32>> {
        let (fast, big) = col.into_parts();
        let (counts, big, remap) = dedup_fresh_store(fast, big);
        crate::diagram::MarginalStorage::new(level, None, t.vtree_idx().idx()).install_counts(counts, big);
        Some(remap)
    }

    fn sum_out_leaf(
        tdd: &mut Tdd,
        leaf: VtreeIdx,
        vtree: &crate::vtree::Vtree,
        _store: &mut (),
    ) {
        crate::marginal::marginalize_leaf_inline(tdd, leaf, vtree);
    }

    /// Make every marginal-side slot reference this pass persisted self-describing,
    /// once, at the pass's chokepoint.
    ///
    /// This path does not go through `apply_and_fallible`, so the end-of-apply
    /// tagger never runs on it, and canon's no-duplicate early return leaves
    /// untouched boundary references raw. The snapshot is what keeps the sweep
    /// off children that a prior pass marginalized: those already carry inline
    /// counts, and re-resolving them as bare slots would misread them.
    fn end_sweep(tdd: &mut Tdd, was_marginal: &[bool]) {
        crate::diagram::inline_small_marginal_refs(tdd, Some(was_marginal));
    }
}
impl MarginalDomain for WeightFold {
    fn weight_store(store: &mut WeightStore) -> Option<&mut WeightStore> {
        Some(store)
    }

    #[inline]
    fn commit_in_flight(
        levels: &mut [TddLevel],
        left_idx: usize,
        col: Vec<WeightValue>,
        store: &mut WeightStore,
    ) {
        // No parent contract-dirty marking: the shared level-state machine
        // establishes invariant 10 at slot-prune, which runs in weighted mode too via
        // `prune_value_slots`; only the integer
        // count-preservation localizer around it is gated off.
        crate::diagram::MarginalStorage::new(&mut levels[left_idx], Some(store), left_idx).install_weights(col);
    }

    /// The weighted store is full width and its references stay bare slots
    /// (slot index == node index), so nothing is minted and the parent's
    /// references need no rewrite.
    fn install(
        level: &mut TddLevel,
        t: InternalLevel,
        col: Vec<WeightValue>,
        store: &mut WeightStore,
    ) -> Option<Vec<u32>> {
        // The column is full width — one slot per node, tombstones included —
        // which is the slot count the level records.
        crate::diagram::MarginalStorage::new(level, Some(store), t.vtree_idx().idx()).install_weights(col);
        None
    }

    fn sum_out_leaf(
        tdd: &mut Tdd,
        leaf: VtreeIdx,
        vtree: &crate::vtree::Vtree,
        store: &mut WeightStore,
    ) {
        crate::marginal::marginalize_leaf_weighted(tdd, leaf, vtree, store);
    }

    /// Nothing: weighted marginal-side references are bare slots end to end, so
    /// there is no tag to apply and no snapshot to key it off.
    fn end_sweep(_tdd: &mut Tdd, _was_marginal: &[bool]) {}
}
/// Install `col` as `t`'s marginal store and settle the diagram around it:
/// rewrite the parent's references if the domain minted new slots, mark the
/// parent for re-contraction, and free the children `t` now subsumes.
pub(crate) fn install_finished<K: MarginalDomain>(
    tdd: &mut Tdd,
    vtree: &Vtree,
    level: InternalLevel,
    col: Column<K>,
    store: &mut K::Store,
) {
    let t = level.vtree_idx();
    assert_can_make_marginal(&tdd.levels, vtree, t);

    tdd.install_marginal_level(t, |storage| K::install(storage, level, col, store));

    // `t` now subsumes its children — free their dead stores (O(1)).
    free_subsumed_marginal_children(&mut tdd.levels, vtree, t, K::weight_store(store));
}

/// Install a streamed column without renumbering slots, then release its children.
#[inline]
pub(crate) fn install_streamed<K: MarginalDomain>(
    levels: &mut [TddLevel],
    vtree: &Vtree,
    t: VtreeIdx,
    col: K::Col,
    store: &mut K::Store,
) {
    assert_can_make_marginal(levels, vtree, t);
    K::commit_in_flight(levels, t.idx(), col, store);
    free_subsumed_marginal_children(levels, vtree, t, K::weight_store(store));
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


/// Compact a freshly built marginal store so that each count value occupies at
/// most one slot (invariant 10), returning the deduped store and a slot remap,
/// `remap[old] = new`.
///
/// Duplicates merge onto the first slot holding their value, so the returned
/// column may be shorter than the input; with no duplicates both come back
/// unchanged. Refs are not remapped here: the caller must redirect every
/// parent-side ref into the old store through `remap` (the marginalization pass
/// does so with `remap_refs_into`). An emit-born store does not pass through
/// here; its invariant 10 is established by `prune_value_slots`.
///
/// The count column is compacted in place, so no second full-length column is
/// resident at the peak; the overflow table is rekeyed into a fresh
/// [`CountOverflow`], which costs only the surviving overflow entries.
pub(crate) fn dedup_fresh_store(
    mut counts: Vec<u128>,
    mut big: Option<CountOverflow>,
) -> (Vec<u128>, Option<CountOverflow>, Vec<u32>) {
    let n = counts.len();
    // Written for every `i`, so the remap is final as it is written and the
    // overflow table can be rekeyed in one drain once it is complete.
    let mut remap: Vec<u32> = vec![0; n];
    let (new_len, _) = compact_count_slots(&mut counts, &mut big, 0..n, &mut remap);

    if new_len == n {
        // No duplicates: `remap` is the identity, and so would be the rekey.
        return (counts, big, remap);
    }

    truncate_with_slack(&mut counts, new_len);
    (counts, big, remap)
}
