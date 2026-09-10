//! The `Tdd` struct.

use std::sync::Arc;

use crate::vtree::{Vtree, VtreeIdx};
use crate::diagram::WeightStore;

use super::level::{LevelKind, TddLevel};
use super::primitives::{LEAF_WIDTH, TddNodeId, ZERO};
use super::stats::LevelStats;

/// The reduction passes' worklists on a diagram: which levels changed since the
/// last contraction, and which the content-twin scan still has to revisit. Not
/// serialized, and never part of the function the diagram denotes.
///
/// The fields are private to this module. Everything that changes a diagram
/// states what it changed through [`Tdd::invalidate`], which is the one place
/// that decides which worklist owes what; everything that consumes a worklist
/// goes through the `take_*` accessors below.
#[derive(Clone, Debug, Default)]
pub(crate) struct Dirty {
    /// Internal vtree node indices whose pair lists changed since the last
    /// `contract_all_twins` pass; it consumes the list to seed its worklist
    /// (children of dirty parents) instead of scanning every level. May hold
    /// duplicates and stale entries (filtered at consume time). A level absent
    /// from the list is at its contraction fixpoint.
    contract: Vec<u32>,
    /// The same, for the leaf-side twin contraction (`contract_leaf_twins`).
    leaf_contract: Vec<u32>,
    /// Worklist for the content-twin fixpoint: vtree indices whose
    /// boundary-parent levels may have gained new content twins since the last
    /// scan round. Only meaningful inside `canonicalize_content_twins`; empty
    /// outside it.
    right_rescan: Vec<u32>,
}

/// What a rewrite did to one level, as the reduction passes see it.
///
/// A rewrite states this and nothing else; [`Tdd::invalidate`] turns it into
/// worklist entries. The three are independent and combine with `|`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Changed(u8);

impl Changed {
    /// This level's pair lists were rewritten in place — refs, lengths, or
    /// order. Its children's contexts moved, so they are twin candidates, and
    /// its own leaf-side verdict is stale.
    pub(crate) const PAIRS: Changed = Changed(1 << 0);
    /// Nodes of this level were merged or dropped, so the parent's references
    /// into it changed identity: the parent may now hold twins.
    pub(crate) const NODES: Changed = Changed(1 << 1);
    /// Marginal values behind references from this level were merged or
    /// renumbered. Structurally the same as `PAIRS` for the worklists — the
    /// refs this level holds mean something different than they did.
    pub(crate) const VALUES: Changed = Changed(1 << 2);

    /// Does `self` include any of `other`'s kinds?
    #[inline]
    fn intersects(self, other: Changed) -> bool {
        self.0 & other.0 != 0
    }
}

impl std::ops::BitOr for Changed {
    type Output = Changed;
    #[inline]
    fn bitor(self, rhs: Changed) -> Changed {
        Changed(self.0 | rhs.0)
    }
}

/// A Tree Decision Diagram: a Boolean function decomposed along a vtree.
///
/// The diagram owns one [`TddLevel`] per vtree node and shares the vtree by
/// `Arc`. The function it denotes is the node `output`; every other stored
/// node is a subfunction over its vtree node's variables. See the
/// [module docs](super) for how to walk it. A minimized diagram is canonical
/// for its vtree; one built level by level
/// ([`TddBuilder`](crate::diagram::TddBuilder)) is not until
/// [`minimize`](crate::reduce::minimize) runs.
#[derive(Clone, Debug)]
pub struct Tdd {
    /// The vtree the diagram is decomposed along. Operands of a binary
    /// operation must share it (`Arc::ptr_eq`). Read it with
    /// [`vtree`](Self::vtree); the only way to change it is
    /// [`reseat_vtree`](Self::reseat_vtree).
    pub(crate) vtree: Arc<Vtree>,
    /// One level per vtree node: `levels[t.idx()]` is the level of `t`
    /// ([`level`](Self::level)).
    pub(crate) levels: Vec<TddLevel>,
    /// The node denoting the function: a node of the root level, or
    /// `local == ZERO` for the constant-false function ([`is_zero`](Self::is_zero)).
    pub(crate) output: TddNodeId,
    /// Which levels the reduction passes still have to revisit (not part of
    /// the function denoted).
    pub(crate) dirty: Dirty,
    /// Per-node semiring values for the diagram's weight-marginal levels, when
    /// the caller put the diagram in weighted mode ([`set_weights`]).
    /// `None` is integer mode: marginal levels carry model counts instead.
    ///
    /// [`set_weights`]: Self::set_weights
    pub(crate) weights: Option<WeightStore>,
    /// Whole-level-array quantities carried with the diagram rather than
    /// re-derived per read (not part of the function denoted). See
    /// [`stats`](super::stats).
    pub(crate) stats: LevelStats,
}

impl Tdd {
    /// The vtree the diagram is decomposed along.
    ///
    /// Operands of a binary operation must share it (`Arc::ptr_eq`).
    #[inline]
    pub fn vtree(&self) -> &Arc<Vtree> {
        &self.vtree
    }

    /// The node denoting the function: a node of the root level, or
    /// `local == ZERO` for the constant-false function
    /// ([`is_zero`](Self::is_zero)).
    #[inline]
    pub fn output(&self) -> TddNodeId {
        self.output
    }

    /// Every level, in vtree-node order — `levels()[t.idx()]` is the level of
    /// `t`. For one level, [`level`](Self::level) says it more directly.
    #[inline]
    pub fn levels(&self) -> &[TddLevel] {
        &self.levels
    }

    /// Clone the diagram with every level arena reserved through `eng`, so a
    /// copy the host cannot serve comes back as
    /// [`ApplyError::OverBudget`](crate::ApplyError::OverBudget) rather than
    /// aborting the process. `Clone` is the same copy without that guard.
    ///
    /// The weight store, when there is one, is copied by its own `Clone`.
    pub(crate) fn try_clone_on(
        &self,
        eng: &crate::engine::Engine,
    ) -> Result<Tdd, crate::error::ApplyError> {
        let lim = eng.limits();
        let mut levels = Vec::new();
        lim.reserve_exact(&mut levels, self.levels.len())?;
        for level in &self.levels {
            levels.push(level.try_clone_on(lim)?);
        }
        Ok(Tdd {
            vtree: Arc::clone(&self.vtree),
            levels,
            output: self.output,
            dirty: self.dirty.clone(),
            weights: self.weights.clone(),
            stats: self.stats.clone(),
        })
    }

    /// Seat the diagram on `vtree`, a numbering of the same node set the
    /// diagram's levels are indexed by.
    ///
    /// A caller that rebuilds a vtree while several diagrams over it are in
    /// flight ends up holding `Arc`s that are not the same allocation, which
    /// the `Arc::ptr_eq` that operands of one operation must satisfy fails.
    /// This makes them one `Arc` again.
    ///
    /// It does not require the two trees to have the same shape
    /// ([`Vtree::same_tree`]): a rotation changes the shape while leaving the
    /// nodes each in-flight diagram actually describes untouched, and reseating
    /// those diagrams on the rotated tree is exactly how a mid-compile rotation
    /// is propagated. What must hold is that the levels stay addressable, so
    /// that is what is checked. The caller owes the rest.
    pub(crate) fn reseat_vtree(&mut self, vtree: &Arc<Vtree>) {
        debug_assert_eq!(
            self.vtree.num_nodes(), vtree.num_nodes(),
            "reseat_vtree onto a tree of a different size leaves levels unaddressable",
        );
        self.vtree = Arc::clone(vtree);
        // The parent set and the level a maximum was read at are both stated
        // against the tree, and a reseat is how a rotation reaches an
        // in-flight diagram — so what the cache claims no longer holds.
        self.forget_stats();
    }

    /// Assemble a diagram from levels built by hand, unchecked.
    ///
    /// The caller guarantees the invariants
    /// [`check_levels`](crate::diagram::builder::check_levels) checks; nothing
    /// here verifies them, and a violation surfaces later as a wrong answer or
    /// a panic. The result need not be canonical:
    /// [`minimize`](crate::reduce::minimize) makes it so. Every
    /// internal level is marked for twin contraction, so the first minimize
    /// visits all of them.
    ///
    /// Outside the crate, [`TddBuilder`](crate::diagram::TddBuilder) is the way
    /// in: it establishes what this trusts.
    pub(crate) fn from_levels_unchecked(vtree: Arc<Vtree>, levels: Vec<TddLevel>, output: TddNodeId) -> Self {
        let n = vtree.num_nodes();
        let rebuilt: Vec<VtreeIdx> = (0..n)
            .map(|i| VtreeIdx(i as u32))
            .filter(|&t| !vtree.node(t).is_leaf())
            .collect();
        Self::with_levels_dirty(vtree, levels, output, Dirty::default(), &rebuilt)
    }

    /// Construct a diagram from raw levels with contract worklists supplied by
    /// the caller, instead of
    /// [`from_levels_unchecked`](Self::from_levels_unchecked)' every-internal-level seed.
    ///
    /// The seeding contract both worklists carry throughout the crate is
    /// "a level absent from the list is at its contraction fixpoint" — every
    /// pair-mutating site marks its own changed levels ([`Tdd::invalidate`]).
    /// `from_levels_unchecked` satisfies it
    /// the blunt way, by naming every internal level; an operation that knows
    /// which levels it rewrote can satisfy it exactly, and the resulting sweep
    /// is identical because the levels it drops were provably going to no-op.
    ///
    /// The caller owes two things, and both must hold for its result to match
    /// `from_levels_unchecked`:
    ///
    /// 1. Every level whose pair list this operation changed is in `rebuilt`;
    /// 2. Every level the input diagram had outstanding is carried over — the
    ///    input's own [`Dirty`], which an operation that rebuilds a diagram
    ///    would otherwise silently drop.
    ///
    /// The sole production caller is the clause-specialized apply
    /// (`apply::conjoin_clause::conjoin_clause_into`), which
    /// rewrites exactly the clause's spine and hands its accumulator's
    /// outstanding work straight through. On a vtree with hundreds of thousands
    /// of levels, seeding a ~10-level spine instead of every internal level is
    /// the difference between an O(vtree) and an O(spine) contraction per
    /// clause.
    pub(crate) fn with_levels_dirty(
        vtree: Arc<Vtree>,
        levels: Vec<TddLevel>,
        output: TddNodeId,
        carried: Dirty,
        rebuilt: &[VtreeIdx],
    ) -> Self {
        let mut dirty = carried;
        dirty.contract.reserve(rebuilt.len());
        dirty.leaf_contract.reserve(rebuilt.len());
        for t in rebuilt {
            dirty.contract.push(t.0);
            dirty.leaf_contract.push(t.0);
        }
        // Bound the carried lists. Both consumers dedup (a repeat entry is
        // re-checked and no-ops), so a list longer than the vtree has nodes is
        // carrying nothing but duplicates — a chain of applies whose minimize
        // never drains a list (a contract-only minimize leaves the
        // `leaf_contract` list alone; only `contract_leaf_twins` drains it)
        // would otherwise grow it by one spine per clause forever. Entries are
        // level indices into this vtree, so a deduplicated list is at most `n`
        // long and the compaction can fire at most once per `n` pushes:
        // amortized O(1), and the set the list denotes is unchanged, so it is
        // invisible to both consumers.
        let n = vtree.num_nodes();
        for list in [&mut dirty.contract, &mut dirty.leaf_contract] {
            if list.len() > n {
                list.sort_unstable();
                list.dedup();
            }
        }
        Self { vtree, levels, output, dirty, weights: None, stats: LevelStats::unknown() }
    }

    /// Take everything this diagram still owes the reduction passes, leaving it
    /// owing nothing. For an operation that rebuilds a diagram from this one
    /// and must carry the obligation into the result.
    #[inline]
    pub(crate) fn take_worklists(&mut self) -> Dirty {
        std::mem::take(&mut self.dirty)
    }

    /// Put the diagram in weighted mode: its weight-marginal levels keep their
    /// per-node semiring values in `ws` instead of model counts.
    ///
    /// Attach the store before the first operation that marginalizes a level. A
    /// conjunction and a projection both move the store to their result, so
    /// only the accumulator of a weighted build needs one.
    ///
    /// This is the second half of the store-presence invariant every weighted
    /// operation relies on: *a weight-marginal level exists only in a diagram
    /// carrying a store*. A diagram assembled level by level states the store
    /// up front instead, with
    /// [`TddBuilder::with_weights`](crate::diagram::TddBuilder::with_weights);
    /// its `finish` refuses a weight-marginal level without one.
    pub fn set_weights(&mut self, ws: WeightStore) {
        self.weights = Some(ws);
    }

    /// The attached weight store, or `None` in integer mode.
    pub fn weights(&self) -> Option<&WeightStore> {
        self.weights.as_ref()
    }

    /// Detach the weight store, leaving the diagram in integer mode. The values
    /// of any already-marginal level go with it.
    pub fn take_weights(&mut self) -> Option<WeightStore> {
        self.weights.take()
    }

    /// The store of a weighted diagram.
    ///
    /// For the operations that only run on a weighted diagram and have already
    /// established that — a weighted fold, the weighted half of pair fusion,
    /// the slot pruner under a weight store. Panics if the diagram has none,
    /// which is a violated build invariant rather than a caller error: a
    /// weight-marginal level exists only in a diagram carrying a store, and
    /// both constructors refuse the alternative.
    #[inline]
    pub(crate) fn weight_store(&self) -> &WeightStore {
        self.weights
            .as_ref()
            .expect("a weighted operation on a diagram with no weight store")
    }

    /// [`Tdd::weight_store`] for a caller that writes.
    #[inline]
    pub(crate) fn weight_store_mut(&mut self) -> &mut WeightStore {
        self.weights
            .as_mut()
            .expect("a weighted operation on a diagram with no weight store")
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

    /// True if this diagram denotes the constant-false function: `output.local`
    /// is the [`ZERO`] sentinel, the only representation of ⊥ (no stored node
    /// computes false).
    pub fn is_zero(&self) -> bool {
        self.output.local == ZERO
    }

    /// True if any level of the diagram is marginal — the whole-diagram
    /// "marginal context" predicate.
    ///
    /// Single source of truth for a question several subsystems ask: once
    /// marginalization has collapsed any level, a node's pair list is a legal
    /// multiset feeding `Σ_pairs c(left)·c(right)` rather than a set, and that
    /// holds everywhere in the diagram — count-bearing duplicate pairs
    /// propagate up from a marginal subtree into levels whose own children
    /// are all explicit (see `reduce::contract::content_twin`).
    /// Readers: the content-twin merge's scope gate, its `right_gated` caller,
    /// rotation's multiset-semantics switch, and contract's debug duplicate
    /// check. O(levels) — a bookkeeping-level sweep, not a hot-path one.
    pub fn has_marginal_level(&self) -> bool {
        self.levels.iter().any(|l| l.is_marginal())
    }

    /// The level of vtree node `idx`.
    pub fn level(&self, idx: VtreeIdx) -> &TddLevel {
        &self.levels[idx.idx()]
    }

    /// What the level of `idx` stores, [`LevelKind::Leaf`] included — the
    /// vtree is what tells a leaf level from an empty structural one, so this
    /// is the complete answer [`TddLevel::kind`] cannot give on its own.
    pub fn level_kind(&self, idx: VtreeIdx) -> LevelKind {
        if self.vtree.node(idx).is_leaf() {
            LevelKind::Leaf
        } else {
            self.levels[idx.idx()].kind()
        }
    }

    /// [`TddLevel::width`] of the level of `idx`: 0 on a leaf level. Use
    /// [`effective_width`](Self::effective_width) to size arrays indexed by
    /// child references.
    pub fn width_at(&self, idx: VtreeIdx) -> usize {
        self.levels[idx.idx()].width()
    }

    /// The index bound for references into the level of `idx`:
    /// [`LEAF_WIDTH`] on a leaf level, else [`TddLevel::width`].
    pub fn effective_width(&self, idx: VtreeIdx) -> usize {
        if self.vtree.node(idx).is_leaf() {
            LEAF_WIDTH
        } else {
            self.levels[idx.idx()].width()
        }
    }

    /// The largest [`TddLevel::live_width`] over all levels; 0 for ⊥.
    ///
    /// Served from the diagram's own cache when that cache can vouch for the
    /// value, and swept otherwise.
    pub fn max_width(&self) -> usize {
        if let Some(w) = self.cached_max_width() {
            return w;
        }
        self.levels
            .iter()
            .map(|l| l.live_width())
            .max()
            .unwrap_or(0)
    }

    /// Number of stored nodes over all levels (implicit leaf nodes excluded).
    pub fn node_count(&self) -> usize {
        self.levels.iter().map(|l| l.live_width()).sum()
    }

    /// Running total of marginal-count slots the slot prune has collected,
    /// summed over all levels. Non-decreasing over a diagram's life; it resets
    /// only where a level is cleared or reset.
    ///
    /// It is the correction term for a size comparison across a prune. Record
    /// it alongside [`Tdd::node_count`] at the baseline instant; at comparison
    /// time add `retired_marginal_slots() - baseline` back to the node count,
    /// or pruning silently deflates the metric.
    pub fn retired_marginal_slots(&self) -> usize {
        self.levels.iter().map(|l| l.retired_marginal_slots() as usize).sum()
    }

    /// Allocate an all-false `[vtree_idx][local_idx]` reachability matrix sized to
    /// each level's effective width.
    fn empty_reach_matrix(&self) -> Vec<Vec<bool>> {
        (0..self.vtree.num_nodes())
            .map(|i| vec![false; self.effective_width(VtreeIdx(i as u32))])
            .collect()
    }

    /// Top-down reachability propagation over a pre-seeded root set. Every root
    /// node must already be marked `true` in `reachable`; on return every node
    /// reachable from those roots is marked. Single source of truth for the
    /// traversal shared by [`reachable_nodes`] (output-seeded) and
    /// [`reachable_from_root_level`] (root-level-seeded).
    fn propagate_reachability(&self, reachable: &mut [Vec<bool>]) {
        for (t, left_vtree, right_vtree) in self.vtree.internal_bottomup().rev() {
            // Marg-side refs are bit-30-tagged slot indices (or, post-Phase-B,
            // inline counts). Decode before indexing the child reachability
            // vector: a slot ref masks to its bare index; an inline-count ref
            // has no child node, so it marks nothing.
            let left_view = self.levels[left_vtree.idx()].side_view();
            let right_view = self.levels[right_vtree.idx()].side_view();
            let level = self.level(t);
            for (i, node) in level.nodes.iter().enumerate() {
                if !reachable[t.idx()][i] {
                    continue;
                }
                for pair in level.pairs_of(node) {
                    if pair.left != ZERO
                        && let Some(s) = left_view.child(pair.left).index() {
                            reachable[left_vtree.idx()][s] = true;
                        }
                    if pair.right != ZERO
                        && let Some(s) = right_view.child(pair.right).index() {
                            reachable[right_vtree.idx()][s] = true;
                        }
                }
            }
        }
    }

    /// Which nodes `output` reaches, as `[vtree index][local index]` over
    /// `effective_width`; all false for ⊥. A minimized diagram reaches every
    /// stored node.
    pub fn reachable_nodes(&self) -> Vec<Vec<bool>> {
        let mut reachable = self.empty_reach_matrix();
        if self.is_zero() {
            return reachable;
        }
        reachable[self.output.vtree.idx()][self.output.local.idx()] = true;
        self.propagate_reachability(&mut reachable);
        reachable
    }

    /// Reachability seeded from every node at the vtree root level, not just the
    /// single `output`. The ray classification runs mid-compile, where the root
    /// level can hold several live candidate nodes that are not yet joined into
    /// one output; seeding only from `output` would then mis-classify those as
    /// dead. Shares `propagate_reachability` with [`reachable_nodes`](Self::reachable_nodes). For a
    /// `ZERO` (UNSAT) diagram the root level is empty, so the result is all-false.
    #[cfg(test)]
    pub(crate) fn reachable_from_root_level(&self) -> Vec<Vec<bool>> {
        let mut reachable = self.empty_reach_matrix();
        for slot in reachable[self.vtree.root().idx()].iter_mut() {
            *slot = true;
        }
        self.propagate_reachability(&mut reachable);
        reachable
    }

    /// Total number of pairs over all stored nodes — the size of the diagram.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::Tdd;
    /// use tididi::vtree::Vtree;
    ///
    /// let vtree = Arc::new(Vtree::balanced(4));
    /// let mut f = Tdd::clause(&vtree, [1, -2]) & Tdd::clause(&vtree, [2, 3]);
    /// tididi::reduce::minimize(&mut f);
    /// assert!(f.size() > 0);
    /// assert!(f.size_at_most(f.size()));
    /// assert!(!f.size_at_most(f.size() - 1));
    ///
    /// // Conditioning cannot grow the diagram.
    /// let g = tididi::apply::condition_var(&f, tididi::vtree::VarId(0), true);
    /// assert!(g.size() <= f.size());
    /// ```
    pub fn size(&self) -> usize {
        let mut total = 0usize;
        for level in &self.levels {
            for i in 0..level.nodes.len() {
                if level.nodes[i].is_internal() {
                    total += level.pair_count_at(i);
                }
            }
        }
        total
    }

    /// Whether the diagram has at most `cap` input pairs.
    ///
    /// The cost is bounded by `cap` rather than by the diagram, which is what a
    /// caller asking a threshold question about a large accumulator once per
    /// compile step needs: sizing a multi-million-pair diagram at every step is
    /// `O(steps x size)`, while the threshold is answered after a few nodes.
    pub fn size_at_most(&self, cap: usize) -> bool {
        let mut total = 0usize;
        for level in &self.levels {
            for i in 0..level.nodes.len() {
                if level.nodes[i].is_internal() {
                    total += level.pair_count_at(i);
                    if total > cap {
                        return false;
                    }
                }
            }
        }
        true
    }
}

// Note: diagram node pair lists are *unordered* — there is no sorted invariant,
// globally maintained or otherwise. A node's identity is its (multi)set of pairs.
// In a purely Boolean diagram the list is a set: uniqueness comes from apply's
// injective product construction + determinism, not from sorting (see
// the no-compress proof). Once any level is marginal the
// list is a genuine multiset — pairs feed a sum, so a repeated pair carries real
// multiplicity. The conjoin hot path does not sort.
//
// No operation requires a consistent pair order. Twin contraction's exact
// signature comparison (`reduce::contract::find_twin_groups`) canonicalizes
// each node's signature before the `==`, so it is a set comparison regardless
// of the order parents stored their pairs in, and pushing pairs in arbitrary
// order is safe. (`compile_models` sorts, but only to support its own
// adjacent-`dedup`.)
