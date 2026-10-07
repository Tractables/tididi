//! Owned level storage, established canonical form and pending reduction work.
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

use std::ops::{Deref, DerefMut};
use super::{TddLevel, TddNodeId};

/// Any mutable access forgets the guarantee, including raw level indexing.
/// The output is recorded separately because changing it need not touch levels.
///
/// The same holds for the closed form of the levels: `closed` says every
/// level is closed ([`TddLevel::close`]), as the end of an operation leaves
/// them, and any mutable access clears it. `changed` lists the levels the
/// edits since then say they changed ([`mark_changed`](Self::mark_changed)),
/// so that a reduction that began on closed levels closes only those.
#[derive(Clone, Debug, Default)]
pub(crate) struct LevelStorage {
    levels: Vec<TddLevel>,
    canonical_output: Option<TddNodeId>,
    closed: bool,
    changed: Vec<VtreeIdx>,
}

impl LevelStorage {
    pub(crate) fn is_canonical(&self, output: TddNodeId) -> bool {
        self.canonical_output == Some(output)
    }

    pub(crate) fn certify(&mut self, output: TddNodeId) {
        self.canonical_output = Some(output);
    }

    pub(crate) fn forget(&mut self) { self.canonical_output = None; }

    /// A mutable access: neither the certification nor the closed form is
    /// known to hold any more.
    fn touch(&mut self) {
        self.forget();
        self.closed = false;
    }

    /// Whether every level is closed, and nothing has had mutable access to
    /// the levels since.
    pub(crate) fn is_closed(&self) -> bool { self.closed }

    /// Record that an edit changed level `t`'s nodes or pairs.
    pub(crate) fn mark_changed(&mut self, t: VtreeIdx) { self.changed.push(t); }

    /// Record that every level is closed again: the levels are back as they
    /// were when [`is_closed`](Self::is_closed) last held, as a rollback puts
    /// them back or a renumbering moves them.
    pub(crate) fn reinstate_closed(&mut self) {
        self.closed = true;
        self.changed.clear();
    }

    /// Take what `from` knows of its levels' closed form, for a copy of them.
    pub(crate) fn copy_closed_from(&mut self, from: &LevelStorage) {
        self.closed = from.closed;
        self.changed.clone_from(&from.changed);
    }

    /// Close every level ([`TddLevel::close`]). A close changes how a level
    /// holds its pairs, not the diagram, so a canonical diagram stays
    /// certified.
    pub(crate) fn close(&mut self) {
        for level in &mut self.levels {
            level.close();
        }
        self.closed = true;
        self.changed.clear();
    }

    /// [`close`](Self::close) the levels in `changed` only, every other level
    /// being closed already.
    pub(crate) fn close_changed(&mut self, changed: &[VtreeIdx]) {
        for &t in changed {
            self.levels[t.idx()].close();
        }
        self.closed = true;
        self.changed.clear();
    }

    /// [`close_changed`](Self::close_changed) the levels the edits marked
    /// changed, every other level being closed already; every level when
    /// the edits marked as many as there are levels.
    pub(crate) fn close_marked(&mut self) {
        if self.changed.len() >= self.levels.len() {
            return self.close();
        }
        let Self { levels, changed, .. } = self;
        for &t in changed.iter() {
            levels[t.idx()].close();
        }
        self.closed = true;
        self.changed.clear();
    }

    pub(crate) fn into_vec(self) -> Vec<TddLevel> { self.levels }
}

impl From<Vec<TddLevel>> for LevelStorage {
    fn from(levels: Vec<TddLevel>) -> Self { Self { levels, ..Self::default() } }
}

impl Deref for LevelStorage {
    type Target = Vec<TddLevel>;
    fn deref(&self) -> &Self::Target { &self.levels }
}

impl DerefMut for LevelStorage {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.touch();
        &mut self.levels
    }
}

impl<'a> IntoIterator for &'a LevelStorage {
    type Item = &'a TddLevel;
    type IntoIter = std::slice::Iter<'a, TddLevel>;
    fn into_iter(self) -> Self::IntoIter { self.levels.iter() }
}

impl<'a> IntoIterator for &'a mut LevelStorage {
    type Item = &'a mut TddLevel;
    type IntoIter = std::slice::IterMut<'a, TddLevel>;
    fn into_iter(self) -> Self::IntoIter {
        self.touch();
        self.levels.iter_mut()
    }
}


/// The reduction pass a worklist belongs to.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Pass {
    /// Inner-node twin contraction (`contract_all_twins`).
    Contract,
    /// Leaf-side twin contraction (`contract_leaf_twins`).
    LeafContract,
    /// The content-twin fixpoint (`Reduction::content_twins`). Read only
    /// inside that pass, which clears it on entry so a round starts from a
    /// known set rather than from whatever ran before.
    ContentTwin,
}

impl Pass {
    /// Every pass, for the writes that tell all three about one event.
    const ALL: [Pass; 3] = [Pass::Contract, Pass::LeafContract, Pass::ContentTwin];
}

/// The passes an assembly seeds: the content-twin fixpoint drives its own
/// rounds from inside itself and takes nothing from an assembly.
const SEEDED: [Pass; 2] = [Pass::Contract, Pass::LeafContract];

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
    /// The levels that may hold a node no node of their parent level names,
    /// reachable or not; every other level's nodes are each named by some
    /// node of its parent level. `None` when that is not known, which is
    /// what every diagram starts as and what any edit makes it. A prune
    /// leaves none, and it walks only as much as these levels need.
    loose: Option<Vec<u32>>,
}

impl Dirty {
    /// One pass's list.
    #[inline]
    fn list(&mut self, pass: Pass) -> &mut Vec<u32> {
        &mut self.lists[pass as usize]
    }

    /// Tell every pass that `level` needs revisiting, charged to `eng` when
    /// there is one.
    #[inline]
    fn push_all(
        &mut self,
        level: u32,
        eng: Option<&crate::Engine>,
    ) -> Result<(), crate::OperationError> {
        self.loose = None;
        for pass in Pass::ALL {
            match eng {
                Some(eng) => eng.limits().try_push(self.list(pass), level)?,
                None => self.list(pass).push(level),
            }
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

    /// True when no pass has work left: a whole-diagram
    /// [`minimize`](Tdd::minimize) ended, and nothing was edited since.
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
        carried.loose = match (carried.loose.take(), self.loose.take()) {
            (Some(mut under), Some(over)) => {
                under.extend(over);
                Some(under)
            }
            _ => None,
        };
        *self = carried;
    }

    /// Copy pending work under the operation's allocation budget.
    pub(crate) fn clone_on(&self, eng: &crate::Engine) -> Result<Self, crate::OperationError> {
        let mut out = Self::default();
        for pass in Pass::ALL {
            let from = &self.lists[pass as usize];
            let to = out.list(pass);
            eng.limits().reserve_exact(to, from.len())?;
            to.extend_from_slice(from);
        }
        if let Some(from) = &self.loose {
            let mut to = Vec::new();
            eng.limits().reserve_exact(&mut to, from.len())?;
            to.extend_from_slice(from);
            out.loose = Some(to);
        }
        Ok(out)
    }

    /// Rename every entry through `map`, old level index to new, for levels
    /// moved to other indices of another vtree.
    pub(crate) fn remap(&mut self, map: &[VtreeIdx]) {
        for list in self.lists.iter_mut().chain(self.loose.as_mut()) {
            for entry in list.iter_mut() {
                *entry = map[*entry as usize].0;
            }
        }
    }

    /// Bound every list: entries are level indices, so a list longer than `n`
    /// holds duplicates, and a chain of applies or in-place edits that never
    /// drains one would otherwise grow it without bound. Dedup keeps the set
    /// the list denotes. It fires above `2n` and leaves at most `n`, so at
    /// most once per `n` pushes; fired above `n`, a list that already named
    /// nearly every level was sorted again on nearly every push, as the
    /// levels a run of clause conjunctions rebuilds between two reductions.
    pub(crate) fn dedup_above(&mut self, n: usize) {
        for list in self.lists.iter_mut().chain(self.loose.as_mut()) {
            if list.len() > 2 * n {
                list.sort_unstable();
                list.dedup();
            }
        }
    }

    /// The levels that may hold a node their parent level does not name, if
    /// that is known.
    #[inline]
    pub(crate) fn loose(&self) -> Option<&[u32]> {
        self.loose.as_deref()
    }

    /// Record which levels may hold a node their parent level does not name:
    /// `None` for not known, and no level once nothing is unreachable.
    #[inline]
    pub(crate) fn set_loose(&mut self, levels: Option<Vec<u32>>) {
        self.loose = levels;
    }

    /// The worklists an assembled diagram starts with: these lists, carried
    /// over from the diagram it was rebuilt from, plus the levels the
    /// assembly `rebuilt` for the contraction passes, deduplicated once they
    /// outgrow the vtree.
    ///
    /// `eng` charges the seeding to a budget and polls for cancellation as it
    /// goes; `None` is the untracked assembly, which grows through `Vec` and
    /// cannot fail.
    pub(crate) fn seeded(
        mut self,
        vtree: &crate::vtree::Vtree,
        rebuilt: impl Iterator<Item = VtreeIdx>,
        eng: Option<&crate::Engine>,
    ) -> Result<Self, crate::OperationError> {
        // Which levels an assembly left loose is its own to say; see
        // `set_loose`.
        self.loose = None;
        self.seed_rebuilt(rebuilt, eng)?;
        self.dedup_above(vtree.num_nodes());
        Ok(self)
    }

    /// Push each of `rebuilt` onto the contraction passes' lists.
    fn seed_rebuilt(
        &mut self,
        rebuilt: impl Iterator<Item = VtreeIdx>,
        eng: Option<&crate::Engine>,
    ) -> Result<(), crate::OperationError> {
        let Some(eng) = eng else {
            for t in rebuilt {
                for pass in SEEDED {
                    self.list(pass).push(t.0);
                }
            }
            return Ok(());
        };
        let lim = eng.limits();
        let mut gate = lim.gate();
        let minimum = rebuilt.size_hint().0;
        for pass in SEEDED {
            let list = self.list(pass);
            if minimum > list.capacity() - list.len() {
                lim.reserve(list, minimum)?;
            }
        }
        for t in rebuilt {
            gate.poll(1)?;
            for pass in SEEDED {
                let list = self.list(pass);
                if list.len() == list.capacity() {
                    lim.reserve(list, 1)?;
                }
                list.push(t.0);
            }
        }
        gate.flush()
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
        self.invalidate_with(level, false, None).expect("an untracked push cannot be refused");
    }

    /// [`invalidate`](Self::invalidate), and additionally that nodes of
    /// `level` were merged or dropped — so its parent's references into it
    /// changed identity, and the parent may now hold twins of its own.
    #[inline]
    pub(crate) fn invalidate_with_parent(&mut self, level: VtreeIdx) {
        self.invalidate_with(level, true, None).expect("an untracked push cannot be refused");
    }

    /// [`invalidate`](Self::invalidate) under the engine's limits; the caller
    /// discards the diagram if a worklist push fails.
    pub(crate) fn try_invalidate(&mut self, eng: &crate::Engine, level: VtreeIdx) -> Result<(), crate::OperationError> {
        self.invalidate_with(level, false, Some(eng))
    }

    /// Map a change to its reduction obligations, charged to `eng` when there
    /// is one.
    ///
    /// The lists are deduplicated once they hold more entries than the vtree
    /// has levels, so a run of edits with no reduction in between keeps them
    /// bounded.
    #[inline]
    fn invalidate_with(
        &mut self, level: VtreeIdx, nodes_moved: bool, eng: Option<&crate::Engine>,
    ) -> Result<(), crate::OperationError> {
        self.dirty.push_all(level.0, eng)?;
        if nodes_moved
            && let Some(parent) = self.vtree.node(level).parent()
        {
            self.dirty.push_all(parent.0, eng)?;
        }
        self.dirty.dedup_above(self.vtree.num_nodes());
        Ok(())
    }
}

#[cfg(test)]
#[path = "tests/worklists.rs"]
mod tests;
