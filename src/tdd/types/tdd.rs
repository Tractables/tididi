//! The `Tdd` struct.

use std::sync::Arc;

use crate::vtree::{Vtree, VtreeIdx};

use super::level::TddLevel;
use super::marg::resolve_marg_ref;
use super::primitives::{InputPair, LocalNodeIdx, TddNodeId, LEAF_WIDTH, ZERO};
use super::marg::MargResolved;

/// Why [`Tdd::try_from_levels`] rejected a hand-built diagram.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TddBuildError {
    /// `levels.len()` is not the vtree's node count.
    LevelCountMismatch {
        /// `vtree.num_nodes()`.
        expected: usize,
        /// `levels.len()`.
        found: usize,
    },
    /// A leaf level stores nodes or pairs, or has a non-empty count table.
    NonEmptyLeafLevel(VtreeIdx),
    /// A stored node is a leaf label, which only leaf levels denote (implicitly).
    LeafNodeStored {
        /// The level holding the node.
        level: VtreeIdx,
        /// The node.
        node: LocalNodeIdx,
    },
    /// A stored node has no pairs; no stored node may compute false.
    EmptyNode {
        /// The level holding the node.
        level: VtreeIdx,
        /// The node.
        node: LocalNodeIdx,
    },
    /// A pair side has bit 31 set (the `ZERO` sentinel, or a corrupt word).
    ReservedBitSet {
        /// The level holding the node.
        level: VtreeIdx,
        /// The node.
        node: LocalNodeIdx,
        /// The offending pair.
        pair: InputPair,
    },
    /// A pair side is out of range for the child level it refers to.
    ChildIndexOutOfRange {
        /// The level holding the node.
        level: VtreeIdx,
        /// The node.
        node: LocalNodeIdx,
        /// The offending pair.
        pair: InputPair,
        /// The child vtree node whose level was indexed (says which side).
        child: VtreeIdx,
    },
    /// A marginal count slot holds the overflow sentinel but the side table
    /// has no value for it.
    OverflowWithoutValue {
        /// The marginal level.
        level: VtreeIdx,
        /// The slot.
        slot: usize,
    },
    /// A marginal level has a structural (non-leaf, non-marginal) child.
    MarginalNotDownwardClosed {
        /// The marginal level.
        level: VtreeIdx,
        /// Its structural child.
        child: VtreeIdx,
    },
    /// `output` is not a node of the root level (nor the `ZERO` sentinel).
    BadOutput(TddNodeId),
}

impl std::fmt::Display for TddBuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LevelCountMismatch { expected, found } => {
                write!(f, "{found} levels for a vtree with {expected} nodes")
            }
            Self::NonEmptyLeafLevel(t) => write!(f, "leaf level {} stores nodes", t.idx()),
            Self::LeafNodeStored { level, node } => {
                write!(f, "level {} node {} is a leaf label", level.idx(), node.idx())
            }
            Self::EmptyNode { level, node } => {
                write!(f, "level {} node {} has no pairs", level.idx(), node.idx())
            }
            Self::ReservedBitSet { level, node, pair } => write!(
                f,
                "level {} node {} pair ({}, {}) has bit 31 set",
                level.idx(), node.idx(), pair.left.0, pair.right.0
            ),
            Self::ChildIndexOutOfRange { level, node, pair, child } => write!(
                f,
                "level {} node {} pair ({}, {}) indexes past the end of child level {}",
                level.idx(), node.idx(), pair.left.0, pair.right.0, child.idx()
            ),
            Self::OverflowWithoutValue { level, slot } => write!(
                f,
                "marginal level {} slot {slot} is marked overflowed but has no exact value",
                level.idx()
            ),
            Self::MarginalNotDownwardClosed { level, child } => write!(
                f,
                "marginal level {} has structural child {}",
                level.idx(), child.idx()
            ),
            Self::BadOutput(id) => write!(
                f,
                "output ({}, {}) is not a node of the root level",
                id.vtree.idx(), id.local.0
            ),
        }
    }
}

impl std::error::Error for TddBuildError {}

/// A Tree Decision Diagram: a Boolean function decomposed along a vtree.
///
/// The diagram owns one [`TddLevel`] per vtree node and shares the vtree by
/// `Arc`. The function it denotes is the node `output`; every other stored
/// node is a subfunction over its vtree node's variables. See the
/// [module docs](super) for how to walk it. A minimized diagram is canonical
/// for its vtree; one built by hand ([`with_levels`](Self::with_levels)) is
/// not until [`minimize`](crate::tdd::minimize::minimize) runs.
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
    /// Vtree-internal node indices whose pair lists changed since the last
    /// `contract_all_twins` pass. Sites that mutate a level's pair list (rotate,
    /// leaf-twin rewrite, prune-driven full invalidation) push the index here;
    /// `contract_all_twins` consumes the list to seed its worklist (children of
    /// dirty parents) instead of scanning all `num_vtree_nodes` levels per call.
    /// May contain duplicates and stale entries (filtered at consume time).
    ///
    /// Public (rather than `pub(crate)`) only so test crates can build TDDs
    /// with struct-literal syntax. Production code should prefer
    /// `Tdd::with_levels`.
    pub(crate) dirty_contract: Vec<u32>,
    /// Vtree-internal node indices whose pair lists changed since the last
    /// `contract_leaf_twins` pass. Mirrors `dirty_contract` for the leaf-side
    /// twin contraction path: rotate, `contract_twins` (which deduplicates
    /// parent pair lists), prune-driven full invalidation. May contain
    /// duplicates and stale entries (filtered at consume time).
    pub(crate) dirty_leaf_contract: Vec<u32>,
    /// Worklist for the C2 fixpoint: vtree indices whose boundary-parent levels
    /// may have gained new content-twins since the last scan round.  ONLY
    /// meaningful inside `canonicalize_content_twins` — always empty outside
    /// that function.  Cleared at loop entry and at loop exit; populated during
    /// each round by every mutation site that can mint fresh boundary twins
    /// (`mark_contract_dirty`, direct `dirty_contract` pushes in the merge pass,
    /// contract fired-parent marks, slot-prune value-merged levels).
    pub(crate) c2_rescan: Vec<u32>,
    /// Set when an `ApplyError::OverBudget` unwound from a contraction window that
    /// left the diagram structurally inconsistent — specifically the mid-parent-
    /// rewrite W2 window in `contract_twins` (an earlier group's survivor already
    /// grew / earlier parent nodes already remapped, then a fallible push failed).
    /// The remaining diagram's model count is UNRELIABLE (over- or under-counts);
    /// consumers must DROP the diagram, not read a count from it. Model-count
    /// extraction (`query::model_count`) asserts this is `false`. Default `false`;
    /// the W1 window is now transactional (a clean pre-mutation bail leaves this
    /// `false`), so only the W2 backstop ever sets it. NOT serialized.
    pub(crate) poisoned: bool,
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
    /// [`minimize`](crate::tdd::minimize::minimize) makes it canonical.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::tdd::Tdd;
    /// use tididi::tdd::types::{InputPair, NEG_LEAF_IDX, POS_LEAF_IDX, TddLevel, TddNodeId};
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
            return Err(TddBuildError::LevelCountMismatch { expected: n, found: levels.len() });
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
            let (lm, rm) = (levels[left.idx()].is_marginal(), levels[right.idx()].is_marginal());
            let (lb, rb) = (bound(left), bound(right));
            for (i, node) in lvl.nodes.iter().enumerate() {
                let node_idx = LocalNodeIdx(i as u32);
                if node.is_tombstone() {
                    continue;
                }
                if node.is_leaf() {
                    return Err(TddBuildError::LeafNodeStored { level: t, node: node_idx });
                }
                let pairs = lvl.pairs_of(node);
                if pairs.is_empty() {
                    return Err(TddBuildError::EmptyNode { level: t, node: node_idx });
                }
                for &pair in pairs {
                    for (raw, marg, b, child) in
                        [(pair.left.0, lm, lb, left), (pair.right.0, rm, rb, right)]
                    {
                        if raw & (1 << 31) != 0 {
                            return Err(TddBuildError::ReservedBitSet { level: t, node: node_idx, pair });
                        }
                        let in_range = match resolve_marg_ref(raw, marg) {
                            MargResolved::Inline(_) => true,
                            MargResolved::Index(j) => j < b,
                        };
                        if !in_range {
                            return Err(TddBuildError::ChildIndexOutOfRange {
                                level: t, node: node_idx, pair, child,
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
    /// [`minimize`](crate::tdd::minimize::minimize) makes it so. Every
    /// internal level is marked for twin contraction, so the first minimize
    /// visits all of them.
    pub fn with_levels(vtree: Arc<Vtree>, levels: Vec<TddLevel>, output: TddNodeId) -> Self {
        let n = vtree.num_nodes();
        let mut dirty_contract: Vec<u32> = Vec::with_capacity(n);
        for i in 0..n {
            if vtree.node(VtreeIdx(i as u32)).is_leaf() { continue; }
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
    /// (`transform::pairwise::conjoin_clause::try_apply_and_clause`), which
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
            dirty_contract,
            dirty_leaf_contract,
            c2_rescan: Vec::new(),
            poisoned: false,
        }
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
    /// Replaces the old all-internal-levels reseed (`invalidate_contract_caches_full`):
    /// every in-place pair mutation must now mark its own changed levels, the
    /// way `with_levels` seeds the levels rebuilt by an apply.
    pub(crate) fn mark_contract_dirty(&mut self, t: VtreeIdx) {
        let i = t.idx();
        // Enqueue on both worklists. Pushing unconditionally is safe:
        // `contract_all_twins_topdown` dedups via `needs_check` and leaf
        // contraction always re-checks, so a level enqueued more than once is
        // still processed once.
        self.dirty_contract.push(i as u32);
        self.dirty_leaf_contract.push(i as u32);
        // Feed the C2 worklist: any level whose pairs changed could be the
        // marg-child of a boundary-parent that now has new content-twins.
        // Only meaningful inside canonicalize_content_twins (empty otherwise).
        self.c2_rescan.push(i as u32);
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
    /// (maintainer ruling 2026-07-27, `minimize::contract::content_twin`).
    /// Readers: the C2 content-twin merge's scope gate, its `c2_gated` caller,
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

    /// [`TddLevel::width`] of the level of `idx`: 0 on a leaf level. Use
    /// [`effective_width`](Self::effective_width) to size arrays indexed by
    /// child references.
    pub fn width_at(&self, idx: VtreeIdx) -> usize {
        self.levels[idx.idx()].width()
    }

    /// The index bound for references into the level of `idx`:
    /// [`LEAF_WIDTH`] on a leaf level, else [`TddLevel::width`].
    pub fn effective_width(&self, idx: VtreeIdx) -> usize {
        if self.vtree.node(idx).is_leaf() { LEAF_WIDTH } else { self.levels[idx.idx()].width() }
    }

    /// The largest [`TddLevel::live_width`] over all levels; 0 for ⊥.
    pub fn max_width(&self) -> usize {
        self.levels.iter().map(|l| l.live_width()).max().unwrap_or(0)
    }

    /// Number of stored nodes over all levels (implicit leaf nodes excluded).
    pub fn total_nodes(&self) -> usize {
        self.levels.iter().map(|l| l.live_width()).sum()
    }

    /// Monotone tally of marginal-count slots collected by `prune_marg_slots`
    /// across all levels. Strictly non-decreasing over a compile; resets only
    /// when a level is cleared/reset (e.g., at component boundaries).
    ///
    /// Used by the adaptive-minimize gates in the downstream compile driver: each gate baseline
    /// records the retired total at its snapshot instant; at comparison time,
    /// `collected_since = retired_marg_total().saturating_sub(baseline_retired)`
    /// is added to `total_nodes()` so that slot-pruning does not silently
    /// deflate the gate metric and inadvertently delay minimize triggers.
    pub fn retired_marg_total(&self) -> usize {
        self.levels.iter().map(|l| l.retired_marg_width as usize).sum()
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
            let left_marg = self.levels[left_vtree.idx()].is_marginal();
            let right_marg = self.levels[right_vtree.idx()].is_marginal();
            let level = self.level(t);
            for (i, node) in level.nodes.iter().enumerate() {
                if !reachable[t.idx()][i] {
                    continue;
                }
                for pair in level.pairs_of(node) {
                    if pair.left != ZERO {
                        if left_marg {
                            if let MargResolved::Index(s) = resolve_marg_ref(pair.left.0, true) {
                                reachable[left_vtree.idx()][s] = true;
                            }
                        } else {
                            reachable[left_vtree.idx()][pair.left.idx()] = true;
                        }
                    }
                    if pair.right != ZERO {
                        if right_marg {
                            if let MargResolved::Index(s) = resolve_marg_ref(pair.right.0, true) {
                                reachable[right_vtree.idx()][s] = true;
                            }
                        } else {
                            reachable[right_vtree.idx()][pair.right.idx()] = true;
                        }
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
    #[cfg(debug_assertions)]
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
        self.size_capped(usize::MAX)
    }

    /// Input-pair total, abandoned once it reaches `cap` — the ONE sweep behind
    /// [`size`](Self::size), which is this with no early exit.
    ///
    /// The cost is bounded by `cap` rather than by the diagram, which is what a
    /// caller asking a THRESHOLD question about a large accumulator once per
    /// compile step needs: sizing a multi-million-pair diagram at every step is
    /// `O(steps × size)`, while `>= cap` is answered after a few nodes. The one
    /// consumer is the progress-based give-up rule's size factor
    /// (the downstream driver's stall-step deadline check), whose floor test is `>= cap`.
    pub fn size_capped(&self, cap: usize) -> usize {
        let mut total = 0usize;
        for level in &self.levels {
            for i in 0..level.nodes.len() {
                if level.nodes[i].is_internal() {
                    total += level.pair_count_at(i);
                    if total >= cap {
                        return total;
                    }
                }
            }
        }
        total
    }
}

// NOTE: TDD node pair lists are *unordered* — there is no sorted invariant,
// globally maintained or otherwise. A node's identity is its (multi)set of pairs.
// In a purely Boolean diagram the list is a set: uniqueness comes from apply's
// injective product construction + determinism, not from sorting (see
// the no-compress proof). Once any level is marginal the
// list is a genuine multiset — pairs feed a sum, so a repeated pair carries real
// multiplicity (maintainer ruling 2026-07-27). The
// conjoin hot path does NOT sort, and the former arena-sort helpers
// (`sort_arena_tail` / `sort_pair_tail` / `PackedPairs::sort_tail`) were removed
// from the apply emit sites with no effect.
//
// No operation requires a consistent pair order. The one operation that once
// did — twin contraction's exact signature comparison
// (`tdd::minimize::contract::find_twin_groups`) — was made order-independent:
// it now canonicalizes each node's signature (sorts the signature slice) before
// the `==`, so the comparison is a set comparison regardless of the order
// parents stored their pairs. Consequently the ad-hoc canonicalizing sorts that
// used to guard this (in `merge_many_internal_twins`, vtree `rotate`, and
// `full::make-full`) were removed — pushing pairs in arbitrary order is safe.
// (`compile_models` still sorts, but only to support its own adjacent-`dedup`,
// not for any downstream order requirement.)
