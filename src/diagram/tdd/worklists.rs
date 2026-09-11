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
        // The rewrite that reports here is also the one that could have made
        // this level the widest, so this is where the width cache hears about
        // it. Folding a width in can only raise a bound, so the report may
        // arrive either side of the rewrite it describes.
        self.observe_level(level);
        let raw = level.0;
        if what.intersects(Changed::PAIRS | Changed::VALUES) {
            self.dirty.contract.push(raw);
            self.dirty.leaf_contract.push(raw);
            self.dirty.right_rescan.push(raw);
        }
        if what.intersects(Changed::NODES)
            && let Some(parent) = self.vtree.node(level).parent()
        {
            self.dirty.contract.push(parent.0);
            self.dirty.leaf_contract.push(parent.0);
            self.dirty.right_rescan.push(parent.0);
        }
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

    /// Empty the twin-contraction worklists. For a pass that has just proved
    /// every level canonical by other means.
    #[inline]
    pub(crate) fn clear_worklists(&mut self) {
        self.dirty.contract.clear();
        self.dirty.leaf_contract.clear();
        self.dirty.right_rescan.clear();
    }
}

// Test support.
impl Tdd {
    /// The twin-contraction worklist, for a test that asserts on what a rewrite
    /// seeded.
    #[cfg(test)]
    pub(crate) fn contract_worklist(&self) -> &[u32] {
        &self.dirty.contract
    }

    /// Drive the twin-contraction worklist directly, for a test that wants a
    /// sweep to start from exactly `levels`.
    #[cfg(test)]
    pub(crate) fn seed_contract_worklist(&mut self, levels: impl IntoIterator<Item = u32>) {
        self.dirty.contract.clear();
        self.dirty.contract.extend(levels);
    }

    /// The same for the leaf-contraction worklist.
    #[cfg(test)]
    pub(crate) fn seed_leaf_worklist(&mut self, levels: impl IntoIterator<Item = u32>) {
        self.dirty.leaf_contract.clear();
        self.dirty.leaf_contract.extend(levels);
    }
}
