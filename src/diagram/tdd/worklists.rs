//! The worklists the reduction passes drain.

use crate::vtree::VtreeIdx;

use super::{Changed, Dirty, Tdd};

impl Tdd {
    /// Take everything this diagram still owes the reduction passes, leaving it
    /// owing nothing. For an operation that rebuilds a diagram from this one
    /// and must carry the obligation into the result.
    #[inline]
    pub(crate) fn take_worklists(&mut self) -> Dirty {
        std::mem::take(&mut self.dirty)
    }

    /// The one place that maps "what changed at `level`" to the worklists.
    ///
    /// Every in-place rewrite calls this for each level it touched, the way an
    /// apply seeds the levels it rebuilt. A level absent from every worklist is
    /// asserted to be at its contraction fixpoint, so a rewrite that stays
    /// silent about a level it changed leaves the diagram non-canonical.
    ///
    /// Re-pushing a level already on a worklist is fine: `contract_all_twins`
    /// dedups through `needs_check`, and leaf contraction re-checks anyway.
    #[inline]
    pub(crate) fn invalidate(&mut self, level: VtreeIdx, what: Changed) {
        self.invalidate_with(level, what, |list, t| { list.push(t); Ok(()) })
            .expect("infallible worklist push");
    }

    /// Record a change under the engine's limits; the caller discards the diagram if a worklist push fails.
    pub(crate) fn try_invalidate(&mut self, eng: &crate::Engine, level: VtreeIdx, what: Changed) -> Result<(), crate::OperationError> {
        self.invalidate_with(level, what, |list, t| eng.limits().try_push(list, t))
    }

    /// Map a change to its reduction obligations using the caller's push policy.
    fn invalidate_with(
        &mut self, level: VtreeIdx, what: Changed,
        mut push: impl FnMut(&mut Vec<u32>, u32) -> Result<(), crate::OperationError>,
    ) -> Result<(), crate::OperationError> {
        let raw = level.0;
        if what.intersects(Changed::PAIRS | Changed::VALUES) {
            push(&mut self.dirty.contract, raw)?;
            push(&mut self.dirty.leaf_contract, raw)?;
            push(&mut self.dirty.right_rescan, raw)?;
        }
        if what.intersects(Changed::NODES)
            && let Some(parent) = self.vtree.node(level).parent()
        {
            push(&mut self.dirty.contract, parent.0)?;
            push(&mut self.dirty.leaf_contract, parent.0)?;
            push(&mut self.dirty.right_rescan, parent.0)?;
        }
        Ok(())
    }

    /// Take the twin-contraction worklist, leaving it empty. The sweep owns the
    /// list it took; a sweep cut short puts what it did not reach back with
    /// [`Tdd::requeue_contract`] or [`Tdd::restore_contract_worklist`].
    #[inline]
    pub(crate) fn take_contract_worklist(&mut self) -> Vec<u32> {
        std::mem::take(&mut self.dirty.contract)
    }

    /// Take the leaf-contraction worklist, leaving it empty.
    #[inline]
    pub(crate) fn take_leaf_worklist(&mut self) -> Vec<u32> {
        std::mem::take(&mut self.dirty.leaf_contract)
    }

    /// Take the content-twin rescan worklist, leaving it empty.
    #[inline]
    pub(crate) fn take_c2_worklist(&mut self) -> Vec<u32> {
        std::mem::take(&mut self.dirty.right_rescan)
    }

    /// Put a whole taken worklist back, for a sweep that failed before it
    /// consumed any of it.
    #[inline]
    pub(crate) fn restore_contract_worklist(&mut self, list: Vec<u32>) {
        self.dirty.contract = list;
    }

    /// Put one level back on the twin-contraction worklist, for a sweep unwound
    /// mid-flight. Not an invalidation: the level was already owed a check, and
    /// this hands the obligation back rather than creating one.
    #[inline]
    pub(crate) fn requeue_contract(&mut self, level: u32) {
        self.dirty.contract.push(level);
    }

    /// Put levels back on the leaf-contraction worklist, for a sweep unwound
    /// mid-flight; the counterpart of [`Tdd::requeue_contract`].
    #[inline]
    pub(crate) fn requeue_leaf_contract(&mut self, levels: impl IntoIterator<Item = u32>) {
        self.dirty.leaf_contract.extend(levels);
    }

    /// Empty the content-twin rescan worklist. The content-twin fixpoint drives
    /// its own rounds through that list, so it starts each round from a known
    /// set rather than from whatever ran before it.
    #[inline]
    pub(crate) fn clear_c2_worklist(&mut self) {
        self.dirty.right_rescan.clear();
    }

    /// Add `levels` to the content-twin rescan worklist, for the fixpoint's own
    /// seeding — a pass it just ran reported the levels it changed.
    #[inline]
    pub(crate) fn extend_c2_worklist(&mut self, levels: impl IntoIterator<Item = u32>) {
        self.dirty.right_rescan.extend(levels);
    }

    /// True when no reduction pass has work left: nothing was edited since the
    /// last [`minimize`](Tdd::minimize), or every edit since has been reduced.
    #[inline]
    pub(crate) fn worklists_empty(&self) -> bool {
        self.dirty.contract.is_empty() && self.dirty.leaf_contract.is_empty() && self.dirty.right_rescan.is_empty()
    }

    /// Put `carried` — the worklists a probe took before it ran — back
    /// underneath whatever the probe itself recorded, keeping both.
    ///
    /// The accept path of a rotation trial needs this rather than a clear. The
    /// worklists are maintained incrementally and never rebuilt, so a level
    /// dropped here keeps its stale contexts until something re-dirties it,
    /// and the invariant that a level absent from every worklist is at its
    /// contraction fixpoint stops holding.
    #[inline]
    pub(crate) fn merge_carried_dirty(&mut self, mut carried: Dirty) {
        carried.contract.append(&mut self.dirty.contract);
        carried.leaf_contract.append(&mut self.dirty.leaf_contract);
        carried.right_rescan.append(&mut self.dirty.right_rescan);
        self.dirty = carried;
    }

}
