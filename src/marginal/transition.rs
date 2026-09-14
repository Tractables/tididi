//! Installing marginal columns and settling the levels that refer to them.

use crate::diagram::{Changed, Tdd, TddLevel, WeightStore, WeightValue, assert_can_make_marginal, remap_refs_into};
use crate::value::{Column, CountVec, IntFold, WeightFold, ValueDomain};
use crate::vtree::{Vtree, VtreeIdx};
use super::free_subsumed_marginal_children;

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
        tdd: &mut Tdd,
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
        crate::marginal::install_int_column(levels, left_idx, col);
    }

    /// Counts are deduped before they are installed, so the level satisfies invariant 10 — no
    /// two slots share a value — from birth rather than by a later canon pass.
    /// That is what mints new slot numbers, and why this domain returns a
    /// remap.
    fn install(
        tdd: &mut Tdd,
        t: InternalLevel,
        col: CountVec,
        _store: &mut (),
    ) -> Option<Vec<u32>> {
        let (fast, big) = col.into_parts();
        let (counts, big, remap) = crate::marginal::dedup_fresh_store(fast, big);
        tdd.levels[t.vtree_idx().idx()].become_marginal(counts, big);
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
        crate::diagram::tag_all_marginal_side_slots(tdd, Some(was_marginal));
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
        // `prune_marginal_slots_generic::<WeightFold>`; only the integer
        // count-preservation localizer around it is gated off.
        crate::marginal::install_weight_column(levels, left_idx, col, store);
    }

    /// The weighted store is full width and its references stay bare slots
    /// (slot index == node index), so nothing is minted and the parent's
    /// references need no rewrite.
    fn install(
        tdd: &mut Tdd,
        t: InternalLevel,
        col: Vec<WeightValue>,
        store: &mut WeightStore,
    ) -> Option<Vec<u32>> {
        // The column is full width — one slot per node, tombstones included —
        // which is the slot count the level records.
        crate::marginal::install_weight_column(&mut tdd.levels, t.vtree_idx().idx(), col, store);
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

    let parent = vtree.node(t).parent();
    if let Some(parent_vi) = parent {
        // The load-bearing seed is the boundary parent that stays explicit;
        // within a marginalizing subtree the parent usually marginalizes too, and
        // contraction then skips it harmlessly.
        tdd.invalidate(parent_vi, Changed::PAIRS);
    }

    let remap = K::install(tdd, level, col, store);

    // Only meaningful while the parent is still explicit — a marginal parent has
    // no pair lists to redirect. Every parent ref into `t` is still a bare slot
    // index here (the tagger has not run), and `remap[old_slot] = new_slot`
    // came from `dedup_fresh_store`, so the store is born satisfying
    // invariant 10 rather than waiting for a later pass.
    if let (Some(remap), Some(parent_vi)) = (remap, parent)
        && !tdd.levels[parent_vi.idx()].is_marginal() {
            remap_refs_into(tdd, t, &remap);
        }

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
