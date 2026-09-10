//! The `Tdd` struct.

use crate::diagram::{ChildRef, ValueRef};
use std::sync::Arc;

use crate::vtree::{Vtree, VtreeIdx};
use crate::diagram::WeightStore;

use super::build_error::TddBuildError;
use super::level::{LevelKind, TddLevel};
use super::primitives::{LEAF_WIDTH, NodeIdx, TddNodeId, ZERO};

/// The reduction passes' worklists on a diagram: which levels changed since the
/// last contraction, and which the content-twin scan still has to revisit. Not
/// serialized, and never part of the function the diagram denotes.
///
/// The fields are private to this module. Everything that changes a diagram
/// says WHAT it changed through [`Tdd::invalidate`], which is the one place
/// that decides which worklist owes what; everything that consumes a worklist
/// goes through the `take_*` accessors below. Before that mapping had a name,
/// twenty sites pushed into these three vectors by hand, each with its own
/// comment reasoning it out, and two of them reached different conclusions
/// from the same premise.
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
    /// Nodes of this level were merged or dropped, so references INTO it from
    /// the parent changed identity: the parent may now hold twins.
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
/// for its vtree; one built by hand ([`from_levels_unchecked`](Self::from_levels_unchecked)) is
/// not until [`minimize`](crate::reduce::minimize) runs.
#[derive(Clone, Debug)]
pub struct Tdd {
    /// The vtree the diagram is decomposed along. Operands of a binary
    /// operation must share it (`Arc::ptr_eq`).
    pub vtree: Arc<Vtree>,
    /// One level per vtree node: `levels[t.idx()]` is the level of `t`
    /// ([`level`](Self::level)).
    pub levels: Vec<TddLevel>,
    /// The node denoting the function: a node of the root level, or
    /// `local == ZERO` for the constant-false function ([`is_zero`](Self::is_zero)).
    pub output: TddNodeId,
    /// Which levels the reduction passes still have to revisit (not part of
    /// the function denoted).
    pub(crate) dirty: Dirty,
    /// Per-node semiring values for the diagram's weight-marginal levels, when
    /// the caller put the diagram in weighted mode ([`set_weights`]).
    /// `None` is integer mode: marginal levels carry model counts instead.
    ///
    /// [`set_weights`]: Self::set_weights
    pub(crate) weights: Option<WeightStore>,
}

impl Tdd {
    /// Assemble a diagram from levels built by hand, checking the invariants
    /// the [module docs](super) list: one level per vtree node, empty leaf
    /// levels, no stored leaf-label or empty node, every pair side in range
    /// for its child level (decoded through `resolve_marginal_ref` when the
    /// child is marginal, and never with bit 31 set), every overflowed
    /// marginal count backed by an exact value, marginality downward-closed,
    /// and `output` a node of the root level or `ZERO`.
    ///
    /// The result is well-formed but not necessarily canonical: it may hold
    /// unreachable nodes and distinct nodes computing the same function.
    /// [`minimize`](crate::reduce::minimize) makes it canonical.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::Tdd;
    /// use tididi::diagram::{InputPair, NEG_LEAF_IDX, POS_LEAF_IDX, TddLevel, TddNodeId};
    /// use tididi::vtree::Vtree;
    ///
    /// // x1 ∧ ¬x2 over a two-leaf vtree: one root node with one pair.
    /// let vtree = Arc::new(Vtree::balanced(2));
    /// let mut levels = vec![TddLevel::new(); vtree.num_nodes()];
    /// let root = vtree.root();
    /// let node = levels[root.idx()].push_internal_node(&[InputPair { left: POS_LEAF_IDX, right: NEG_LEAF_IDX }]);
    /// let f = Tdd::try_from_levels(vtree, levels, TddNodeId { vtree: root, local: node }).unwrap();
    /// assert_eq!(f.model_count(), 1u32.into());
    /// ```
    ///
    /// # Errors
    ///
    /// The first violation found, as a [`TddBuildError`].
    pub fn try_from_levels(
        vtree: Arc<Vtree>,
        levels: Vec<TddLevel>,
        output: TddNodeId,
    ) -> Result<Self, TddBuildError> {
        let n = vtree.num_nodes();
        if levels.len() != n {
            return Err(TddBuildError::LevelCountMismatch {
                expected: n,
                found: levels.len(),
            });
        }
        for (leaf, _var) in vtree.leaf_bottomup() {
            let lvl = &levels[leaf.idx()];
            if !lvl.nodes.is_empty() || !lvl.pairs.is_empty() || lvl.width() != 0 {
                return Err(TddBuildError::NonEmptyLeafLevel(leaf));
            }
        }
        // The index bound a pair side is checked against: the implicit leaf
        // nodes, the count table of a marginal level, or the stored nodes.
        let bound = |t: VtreeIdx| -> usize {
            let lvl = &levels[t.idx()];
            if lvl.is_marginal() {
                lvl.width()
            } else if vtree.node(t).is_leaf() {
                LEAF_WIDTH
            } else {
                lvl.nodes.len()
            }
        };
        for (t, left, right) in vtree.internal_bottomup() {
            let lvl = &levels[t.idx()];
            if lvl.is_weight_marginal() {
                return Err(TddBuildError::WeightedLevelWithoutStore { level: t });
            }
            if lvl.is_marginal() {
                for child in [left, right] {
                    if !vtree.node(child).is_leaf() && !levels[child.idx()].is_marginal() {
                        return Err(TddBuildError::MarginalNotDownwardClosed { level: t, child });
                    }
                }
                if let Some(counts) = lvl.marginal_counts() {
                    for (slot, &c) in counts.iter().enumerate() {
                        let backed = lvl.marginal_counts_big().and_then(|b| b.get(slot));
                        if c == u128::MAX && backed.is_none() {
                            return Err(TddBuildError::OverflowWithoutValue { level: t, slot });
                        }
                    }
                }
                continue;
            }
            let (lm, rm) = (
                levels[left.idx()].side_view(),
                levels[right.idx()].side_view(),
            );
            let (lb, rb) = (bound(left), bound(right));
            for (i, node) in lvl.nodes.iter().enumerate() {
                let node_idx = NodeIdx(i as u32);
                if node.is_tombstone() {
                    continue;
                }
                if node.is_leaf() {
                    return Err(TddBuildError::LeafNodeStored {
                        level: t,
                        node: node_idx,
                    });
                }
                let pairs = lvl.pairs_of(node);
                if pairs.is_empty() {
                    return Err(TddBuildError::EmptyNode {
                        level: t,
                        node: node_idx,
                    });
                }
                for &pair in pairs {
                    for (side, view, b, child) in
                        [(pair.left, lm, lb, left), (pair.right, rm, rb, right)]
                    {
                        if side.0 & (1 << 31) != 0 {
                            return Err(TddBuildError::ReservedBitSet {
                                level: t,
                                node: node_idx,
                                pair,
                            });
                        }
                        let in_range = match view.child(side) {
                            ChildRef::Value(ValueRef::Inline(_)) => true,
                            r => r.index().unwrap() < b,
                        };
                        if !in_range {
                            return Err(TddBuildError::ChildIndexOutOfRange {
                                level: t,
                                node: node_idx,
                                pair,
                                child,
                            });
                        }
                    }
                }
            }
        }
        let root = vtree.root();
        if output.vtree != root || (output.local != ZERO && output.local.idx() >= bound(root)) {
            return Err(TddBuildError::BadOutput(output));
        }
        Ok(Self::from_levels_unchecked(vtree, levels, output))
    }

    /// Assemble a diagram from levels built by hand, unchecked.
    ///
    /// The caller guarantees the invariants
    /// [`try_from_levels`](Self::try_from_levels) checks; nothing here verifies
    /// them, and a violation surfaces later as a wrong answer or a panic. The result need not be canonical:
    /// [`minimize`](crate::reduce::minimize) makes it so. Every
    /// internal level is marked for twin contraction, so the first minimize
    /// visits all of them.
    pub fn from_levels_unchecked(vtree: Arc<Vtree>, levels: Vec<TddLevel>, output: TddNodeId) -> Self {
        let n = vtree.num_nodes();
        let rebuilt: Vec<VtreeIdx> = (0..n)
            .map(|i| VtreeIdx(i as u32))
            .filter(|&t| !vtree.node(t).is_leaf())
            .collect();
        Self::with_levels_dirty(vtree, levels, output, Dirty::default(), &rebuilt)
    }

    /// Construct a diagram from raw levels with CALLER-SUPPLIED contract worklists,
    /// instead of [`from_levels_unchecked`](Self::from_levels_unchecked)' every-internal-level seed.
    ///
    /// The seeding contract both worklists carry throughout the crate is
    /// "a level absent from the list is at its contraction fixpoint" — every
    /// pair-mutating site marks its own changed levels ([`Tdd::invalidate`]).
    /// `from_levels_unchecked` satisfies it
    /// the blunt way, by naming every internal level; an operation that KNOWS
    /// which levels it rewrote can satisfy it exactly, and the resulting sweep
    /// is identical because the levels it drops were provably going to no-op.
    ///
    /// The caller owes two things, and both must hold for its result to match
    /// `from_levels_unchecked`:
    ///
    /// 1. Every level whose pair list this operation changed is in `rebuilt`;
    /// 2. Every level the INPUT diagram had outstanding is carried over — the
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
        // never drains a list (a contract-only minimize leaves the LEAF list
        // alone; only `contract_leaf_twins` drains it) would otherwise grow it
        // by one spine per clause forever. Entries are level indices into this
        // vtree, so a deduplicated list is at most `n` long and the compaction
        // can fire at most once per `n` pushes: amortized O(1), and the SET the
        // list denotes is unchanged, so it is invisible to both consumers.
        let n = vtree.num_nodes();
        for list in [&mut dirty.contract, &mut dirty.leaf_contract] {
            if list.len() > n {
                list.sort_unstable();
                list.dedup();
            }
        }
        Self { vtree, levels, output, dirty, weights: None }
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
    /// operation relies on — *a weight-marginal level exists only in a diagram
    /// carrying a store* — and the reason the unchecked constructors cannot
    /// assert it: they take no store, so a diagram assembled with
    /// weight-marginal levels is momentarily without one, until this call.
    /// [`try_from_levels`](Self::try_from_levels), which promises a diagram
    /// that is complete when it returns, refuses that shape instead.
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
    /// MULTISET feeding `Σ_pairs c(left)·c(right)` rather than a set, EVERYWHERE
    /// in the diagram — count-bearing duplicate pairs propagate up from a
    /// marginal subtree into levels whose own children are all explicit
    /// (see `reduce::contract::content_twin`).
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
    pub fn max_width(&self) -> usize {
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
    /// single `output`. The gauge audit runs mid-compile, where the root level
    /// can hold several live candidate nodes that are not yet joined into one
    /// output; seeding only from `output` would then mis-classify those as dead.
    /// Shares `propagate_reachability` with [`reachable_nodes`](Self::reachable_nodes). For a ZERO
    /// (UNSAT) diagram the root level is empty, so the result is all-false.
    #[cfg(any(test, debug_assertions))]
    pub(crate) fn reachable_from_root_level(&self) -> Vec<Vec<bool>> {
        let mut reachable = self.empty_reach_matrix();
        for slot in reachable[self.vtree.root().idx()].iter_mut() {
            *slot = true;
        }
        self.propagate_reachability(&mut reachable);
        reachable
    }

    /// Total number of pairs over all stored nodes — the size of the diagram.
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
    /// caller asking a THRESHOLD question about a large accumulator once per
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

// NOTE: diagram node pair lists are *unordered* — there is no sorted invariant,
// globally maintained or otherwise. A node's identity is its (multi)set of pairs.
// In a purely Boolean diagram the list is a set: uniqueness comes from apply's
// injective product construction + determinism, not from sorting (see
// the no-compress proof). Once any level is marginal the
// list is a genuine multiset — pairs feed a sum, so a repeated pair carries real
// multiplicity. The
// conjoin hot path does not sort, and the former arena-sort helpers
// (`sort_arena_tail` / `sort_pair_tail` / `PackedPairs::sort_tail`) were removed
// from the apply emit sites with no effect.
//
// No operation requires a consistent pair order. Twin contraction's exact
// signature comparison (`reduce::contract::find_twin_groups`) canonicalizes
// each node's signature before the `==`, so it is a set comparison regardless
// of the order parents stored their pairs in, and pushing pairs in arbitrary
// order is safe. (`compile_models` sorts, but only to support its own
// adjacent-`dedup`.)
