//! What a diagram still owes the reduction passes.
//!
//! Three passes each drain their own worklist: twin contraction, leaf-twin
//! contraction, and the content-twin scan. They run at different times, so
//! each needs its own cursor, but they are told about the same events — a
//! level whose references changed is a candidate for all three.
//!
//! A level absent from a pass's worklist is asserted to be at that pass's
//! fixpoint. Everything that changes a diagram therefore has to say so through
//! [`Tdd::invalidate`], and a pass that is cut short has to hand back what it
//! did not reach.

use crate::vtree::VtreeIdx;

use super::Tdd;

/// The reduction pass a worklist belongs to.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Pass {
    /// Inner-node twin contraction (`contract_all_twins`).
    Contract,
    /// Leaf-side twin contraction (`contract_leaf_twins`).
    LeafContract,
    /// The content-twin fixpoint (`canonicalize_content_twins`). Read only
    /// inside that pass, which clears it on entry so a round starts from a
    /// known set rather than from whatever ran before.
    ContentTwin,
}

impl Pass {
    /// Every pass, for the writes that tell all three about one event.
    const ALL: [Pass; 3] = [Pass::Contract, Pass::LeafContract, Pass::ContentTwin];
}

/// One worklist per reduction pass: the vtree levels that pass still has to
/// revisit. Not serialized, and never part of the function the diagram
/// denotes.
///
/// A list may hold duplicates and stale entries; every consumer filters at
/// drain time. What it may not do is *omit* a level whose references changed,
/// which is the invariant [`Tdd::invalidate`] exists to keep.
#[derive(Clone, Debug, Default)]
pub(crate) struct Dirty {
    /// Indexed by `Pass`; see [`Dirty::list`].
    lists: [Vec<u32>; 3],
}

impl Dirty {
    /// One pass's list.
    #[inline]
    fn list(&mut self, pass: Pass) -> &mut Vec<u32> {
        &mut self.lists[pass as usize]
    }

    /// Tell every pass that `level` needs revisiting, through `push` so the
    /// caller decides whether a refused allocation is possible.
    #[inline]
    fn push_all(
        &mut self,
        level: u32,
        push: &mut impl FnMut(&mut Vec<u32>, u32) -> Result<(), crate::OperationError>,
    ) -> Result<(), crate::OperationError> {
        for pass in Pass::ALL {
            push(self.list(pass), level)?;
        }
        Ok(())
    }

    /// Take `pass`'s list, leaving it empty. The caller owns what it took: a
    /// sweep cut short hands back what it did not reach with
    /// [`restore`](Self::restore) or [`requeue`](Self::requeue).
    #[inline]
    pub(crate) fn take(&mut self, pass: Pass) -> Vec<u32> {
        std::mem::take(self.list(pass))
    }

    /// Put a whole taken list back, for a sweep that failed before it consumed
    /// any of it.
    #[inline]
    pub(crate) fn restore(&mut self, pass: Pass, list: Vec<u32>) {
        *self.list(pass) = list;
    }

    /// Add levels to `pass`'s list.
    ///
    /// Not an invalidation: for a sweep unwound mid-flight these levels were
    /// already owed a check and this hands the obligation back, and for the
    /// content-twin fixpoint's own seeding a pass it just ran reported what it
    /// changed. Either way no new obligation is created, which is why this
    /// does not go through [`Tdd::invalidate`].
    #[inline]
    pub(crate) fn requeue(&mut self, pass: Pass, levels: impl IntoIterator<Item = u32>) {
        self.list(pass).extend(levels);
    }

    /// Empty `pass`'s list. The content-twin fixpoint drives its own rounds
    /// through its list, so it starts each round from a known set rather than
    /// from whatever ran before it.
    #[inline]
    pub(crate) fn clear(&mut self, pass: Pass) {
        self.list(pass).clear();
    }

    /// True when no pass has work left: nothing was edited since the last
    /// [`minimize`](Tdd::minimize), or every edit since has been reduced.
    #[inline]
    pub(crate) fn is_empty(&self) -> bool {
        self.lists.iter().all(Vec::is_empty)
    }

    /// Put `carried` — the lists a probe took before it ran — back underneath
    /// whatever the probe itself recorded, keeping both.
    ///
    /// The accept path of a rotation trial needs this rather than a clear. The
    /// lists are maintained incrementally and never rebuilt, so a level
    /// dropped here keeps its stale contexts until something re-dirties it,
    /// and the invariant that an absent level is at its fixpoint stops
    /// holding.
    #[inline]
    pub(crate) fn merge_under(&mut self, mut carried: Dirty) {
        for pass in Pass::ALL {
            carried.list(pass).append(self.list(pass));
        }
        *self = carried;
    }

    /// Bound every list: entries are level indices, so a list longer than `n`
    /// holds duplicates, and a chain of applies that never drains one would
    /// otherwise grow it without bound. Dedup keeps the set the list denotes,
    /// and fires at most once per `n` pushes.
    pub(crate) fn dedup_above(&mut self, n: usize) {
        for list in &mut self.lists {
            if list.len() > n {
                list.sort_unstable();
                list.dedup();
            }
        }
    }

    /// Seed the contraction passes with the levels an assembly rebuilt, through
    /// the caller's allocation policy.
    pub(crate) fn seed_rebuilt(
        &mut self,
        rebuilt: impl Iterator<Item = VtreeIdx>,
        mut reserve: impl FnMut(&mut Vec<u32>, usize) -> Result<(), crate::OperationError>,
        mut poll: impl FnMut() -> Result<(), crate::OperationError>,
    ) -> Result<(), crate::OperationError> {
        let minimum = rebuilt.size_hint().0;
        for pass in [Pass::Contract, Pass::LeafContract] {
            let list = self.list(pass);
            if minimum > list.capacity() - list.len() {
                reserve(list, minimum)?;
            }
        }
        for t in rebuilt {
            poll()?;
            for pass in [Pass::Contract, Pass::LeafContract] {
                let list = self.list(pass);
                if list.len() == list.capacity() {
                    reserve(list, 1)?;
                }
                list.push(t.0);
            }
        }
        Ok(())
    }
}

impl Tdd {
    /// Take everything this diagram still owes the reduction passes, leaving
    /// it owing nothing. For an operation that rebuilds a diagram from this
    /// one and must carry the obligation into the result.
    #[inline]
    pub(crate) fn take_worklists(&mut self) -> Dirty {
        std::mem::take(&mut self.dirty)
    }

    /// Record that `level`'s pair lists changed — its references, their
    /// lengths or their order, or the marginal values behind them.
    ///
    /// Every in-place rewrite calls this for each level it touched, the way an
    /// apply seeds the levels it rebuilt. A rewrite that stays silent about a
    /// level it changed leaves the diagram non-canonical.
    ///
    /// Re-recording a level already on a worklist is fine: `contract_all_twins`
    /// dedups through `needs_check`, and leaf contraction re-checks anyway.
    #[inline]
    pub(crate) fn invalidate(&mut self, level: VtreeIdx) {
        self.invalidate_with(level, false, |list, t| { list.push(t); Ok(()) })
            .expect("infallible worklist push");
    }

    /// [`invalidate`](Self::invalidate), and additionally that nodes of
    /// `level` were merged or dropped — so its parent's references into it
    /// changed identity, and the parent may now hold twins of its own.
    #[inline]
    pub(crate) fn invalidate_with_parent(&mut self, level: VtreeIdx) {
        self.invalidate_with(level, true, |list, t| { list.push(t); Ok(()) })
            .expect("infallible worklist push");
    }

    /// [`invalidate`](Self::invalidate) under the engine's limits; the caller
    /// discards the diagram if a worklist push fails.
    pub(crate) fn try_invalidate(&mut self, eng: &crate::Engine, level: VtreeIdx) -> Result<(), crate::OperationError> {
        self.invalidate_with(level, false, |list, t| eng.limits().try_push(list, t))
    }

    /// Map a change to its reduction obligations using the caller's push policy.
    fn invalidate_with(
        &mut self, level: VtreeIdx, nodes_moved: bool,
        mut push: impl FnMut(&mut Vec<u32>, u32) -> Result<(), crate::OperationError>,
    ) -> Result<(), crate::OperationError> {
        self.dirty.push_all(level.0, &mut push)?;
        if nodes_moved
            && let Some(parent) = self.vtree.node(level).parent()
        {
            self.dirty.push_all(parent.0, &mut push)?;
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "tests/worklists.rs"]
mod tests;
