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
#[derive(Clone, Debug, Default)]
pub(crate) struct Dirty {
    /// Internal vtree node indices whose pair lists changed since the last
    /// `contract_all_twins` pass; it consumes the list to seed its worklist
    /// (children of dirty parents) instead of scanning every level. May hold
    /// duplicates and stale entries (filtered at consume time). A level absent
    /// from the list is at its contraction fixpoint.
    pub(crate) contract: Vec<u32>,
    /// The same, for the leaf-side twin contraction (`contract_leaf_twins`).
    pub(crate) leaf_contract: Vec<u32>,
    /// Worklist for the content-twin fixpoint: vtree indices whose
    /// boundary-parent levels may have gained new content twins since the last
    /// scan round. Only meaningful inside `canonicalize_content_twins`; empty
    /// outside it.
    pub(crate) c2_rescan: Vec<u32>,
}

/// A Tree Decision Diagram: a Boolean function decomposed along a vtree.
///
/// The diagram owns one [`TddLevel`] per vtree node and shares the vtree by
/// `Arc`. The function it denotes is the node `output`; every other stored
/// node is a subfunction over its vtree node's variables. See the
/// [module docs](super) for how to walk it. A minimized diagram is canonical
/// for its vtree; one built by hand ([`with_levels`](Self::with_levels)) is
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
    /// Set when an `ApplyError::OverBudget` unwound from the one contraction
    /// window that mutates before its last fallible push (the parent rewrite
    /// in `contract_twins`): the diagram is structurally inconsistent and its
    /// count unreliable, so consumers must drop it. `query::model_count`
    /// asserts this is `false`.
    pub(crate) poisoned: bool,
    /// Per-node semiring values for the diagram's weight-marginal levels, when
    /// the caller put the diagram in weighted mode ([`attach_weights`]).
    /// `None` is integer mode: marginal levels carry model counts instead.
    ///
    /// [`attach_weights`]: Self::attach_weights
    pub(crate) weights: Option<WeightStore>,
}

impl Tdd {
    /// Assemble a diagram from levels built by hand, checking the invariants
    /// the [module docs](super) list: one level per vtree node, empty leaf
    /// levels, no stored leaf-label or empty node, every pair side in range
    /// for its child level (decoded through `resolve_marg_ref` when the
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
                if let Some(counts) = &lvl.marginal_counts {
                    for (slot, &c) in counts.iter().enumerate() {
                        let backed = lvl.marginal_counts_big.as_ref().and_then(|b| b.get(slot));
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
                            r => r.cell().unwrap() < b,
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
        Ok(Self::with_levels(vtree, levels, output))
    }

    /// Assemble a diagram from levels built by hand, unchecked.
    ///
    /// The invariants [`try_from_levels`](Self::try_from_levels) checks must
    /// hold; nothing here verifies them, and a violation surfaces later as
    /// a wrong answer or a panic. The result need not be canonical:
    /// [`minimize`](crate::reduce::minimize) makes it so. Every
    /// internal level is marked for twin contraction, so the first minimize
    /// visits all of them.
    pub fn with_levels(vtree: Arc<Vtree>, levels: Vec<TddLevel>, output: TddNodeId) -> Self {
        let n = vtree.num_nodes();
        let mut dirty_contract: Vec<u32> = Vec::with_capacity(n);
        for i in 0..n {
            if vtree.node(VtreeIdx(i as u32)).is_leaf() {
                continue;
            }
            dirty_contract.push(i as u32);
        }
        let dirty_leaf_contract = dirty_contract.clone();
        Self::with_levels_dirty(vtree, levels, output, dirty_contract, dirty_leaf_contract)
    }

    /// Construct a TDD from raw levels with CALLER-SUPPLIED contract worklists,
    /// instead of [`with_levels`](Self::with_levels)' every-internal-level seed.
    ///
    /// The seeding contract both worklists carry throughout the crate is
    /// "a level absent from the list is at its contraction fixpoint" — every
    /// pair-mutating site marks its own changed levels (`mark_contract_dirty`,
    /// the rotation fixups, prune, the merge pass). `with_levels` satisfies it
    /// the blunt way, by naming every internal level; an operation that KNOWS
    /// which levels it rewrote can satisfy it exactly, and the resulting sweep
    /// is identical because the levels it drops were provably going to no-op.
    ///
    /// The caller owes two things, and both must hold for its result to match
    /// `with_levels`:
    ///
    /// 1. every level whose pair list this operation changed is in the lists;
    /// 2. every level the INPUT diagram had outstanding is carried over — the
    ///    input's own `dirty_contract` / `dirty_leaf_contract`, which an
    ///    operation that rebuilds a diagram would otherwise silently drop.
    ///
    /// The sole production caller is the clause-specialized apply
    /// (`apply::conjoin_clause::conjoin_clause_into`), which
    /// rewrites exactly the clause's spine and hands both lists straight
    /// through from its accumulator. On a vtree with hundreds of thousands of
    /// levels, seeding a ~10-level spine instead of every internal level is the
    /// difference between an O(vtree) and an O(spine) contraction per clause.
    pub(crate) fn with_levels_dirty(
        vtree: Arc<Vtree>,
        levels: Vec<TddLevel>,
        output: TddNodeId,
        mut dirty_contract: Vec<u32>,
        mut dirty_leaf_contract: Vec<u32>,
    ) -> Self {
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
        for list in [&mut dirty_contract, &mut dirty_leaf_contract] {
            if list.len() > n {
                list.sort_unstable();
                list.dedup();
            }
        }
        Self {
            vtree,
            levels,
            output,
            dirty: Dirty {
                contract: dirty_contract,
                leaf_contract: dirty_leaf_contract,
                c2_rescan: Vec::new(),
            },
            weights: None,
            poisoned: false,
        }
    }

    /// Put the diagram in weighted mode: its weight-marginal levels keep their
    /// per-node semiring values in `ws` instead of model counts.
    ///
    /// Attach the store before the first operation that freezes a level. A
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
    pub fn attach_weights(&mut self, ws: WeightStore) {
        self.weights = Some(ws);
    }

    /// The attached weight store, or `None` in integer mode.
    pub fn weights(&self) -> Option<&WeightStore> {
        self.weights.as_ref()
    }

    /// Detach the weight store, leaving the diagram in integer mode. The values
    /// of any already-frozen level go with it.
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

    /// Seed both contract worklists for a level whose pairs an operation just
    /// changed in place (clause-cascade falsify, marginalize, prune).
    ///
    /// A changed level can hold new twins among its own nodes *and* — because
    /// its children's parent context moved — among its children. Twin
    /// contraction processes a parent to reach its children, so seeding level
    /// `t` here catches `t`'s children; `t`'s own twins are caught by seeding
    /// `t`'s parent (the caller seeds every changed level, so a parent that
    /// also changed is covered, and an unchanged ancestor cannot have gained a
    /// twin). May re-push a level already on the worklist; both consumers dedup
    /// so repeated marks across a cascade are still processed once.
    ///
    /// Every in-place pair mutation must mark its own changed levels, the way
    /// `with_levels` seeds the levels rebuilt by an apply.
    pub(crate) fn mark_contract_dirty(&mut self, t: VtreeIdx) {
        let i = t.idx();
        // Enqueue on both worklists. Pushing unconditionally is safe:
        // `contract_all_twins_topdown` dedups via `needs_check` and leaf
        // contraction always re-checks, so a level enqueued more than once is
        // still processed once.
        self.dirty.contract.push(i as u32);
        self.dirty.leaf_contract.push(i as u32);
        // Feed the content-twin worklist: any level whose pairs changed could be the
        // marg-child of a boundary-parent that now has new content-twins.
        // Only meaningful inside canonicalize_content_twins (empty otherwise).
        self.dirty.c2_rescan.push(i as u32);
    }

    /// True if this diagram denotes the constant-false function: `output.local`
    /// is the [`ZERO`] sentinel, the only representation of ⊥ (no stored node
    /// computes false).
    pub fn is_zero(&self) -> bool {
        self.output.local == ZERO
    }

    /// True if ANY level of the diagram is marginal — the whole-diagram
    /// "marg context" predicate.
    ///
    /// Single source of truth for a question several subsystems ask: once
    /// marginalization has collapsed any level, a node's pair list is a legal
    /// MULTISET feeding `Σ_pairs c(left)·c(right)` rather than a set, EVERYWHERE
    /// in the diagram — count-bearing duplicate pairs propagate up from a
    /// marginal subtree into levels whose own children are all explicit
    /// (see `minimize::contract::content_twin`).
    /// Readers: the content-twin merge's scope gate, its `c2_gated` caller,
    /// rotation's multiset-semantics switch, and contract's debug duplicate
    /// check. O(levels) — a bookkeeping-level sweep, not a hot-path one.
    pub fn has_marginal_level(&self) -> bool {
        self.levels.iter().any(|l| l.is_marginal())
    }

    /// Whether a budget abort left the diagram structurally inconsistent.
    /// A poisoned diagram must not be queried or minimized further.
    pub fn is_poisoned(&self) -> bool {
        self.poisoned
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
                        && let Some(s) = left_view.child(pair.left).cell() {
                            reachable[left_vtree.idx()][s] = true;
                        }
                    if pair.right != ZERO
                        && let Some(s) = right_view.child(pair.right).cell() {
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

    /// Reachability seeded from EVERY node at the vtree root level, not just the
    /// single `output`. The gauge audit runs mid-compile, where the root level
    /// can hold several live candidate nodes that are not yet joined into one
    /// output; seeding only from `output` would then mis-classify those as dead.
    /// Shares `propagate_reachability` with [`reachable_nodes`](Self::reachable_nodes). For a ZERO
    /// (UNSAT) TDD the root level is empty, so the result is all-false.
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

// NOTE: TDD node pair lists are *unordered* — there is no sorted invariant,
// globally maintained or otherwise. A node's identity is its (multi)set of pairs.
// In a purely Boolean diagram the list is a set: uniqueness comes from apply's
// injective product construction + determinism, not from sorting (see
// the no-compress proof). Once any level is marginal the
// list is a genuine multiset — pairs feed a sum, so a repeated pair carries real
// multiplicity. The
// conjoin hot path does NOT sort, and the former arena-sort helpers
// (`sort_arena_tail` / `sort_pair_tail` / `PackedPairs::sort_tail`) were removed
// from the apply emit sites with no effect.
//
// No operation requires a consistent pair order. The one operation that once
// did — twin contraction's exact signature comparison
// (`reduce::contract::find_twin_groups`) — was made order-independent:
// it now canonicalizes each node's signature (sorts the signature slice) before
// the `==`, so the comparison is a set comparison regardless of the order
// parents stored their pairs. Consequently the ad-hoc canonicalizing sorts that
// used to guard this (in `merge_many_internal_twins`, vtree `rotate`, and
// `full::make-full`) were removed — pushing pairs in arbitrary order is safe.
// (`compile_models` still sorts, but only to support its own adjacent-`dedup`,
// not for any downstream order requirement.)
