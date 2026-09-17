//! Assemble a diagram level by level with [`TddBuilder`].

use std::collections::HashMap;
use std::sync::Arc;

use crate::Engine;
use crate::vtree::{Vtree, VtreeIdx};

use super::build_error::TddBuildError;
use super::level::TddLevel;
use super::pool::{return_levels, take_levels, PoolSlot};
use super::primitives::{ChildPair, NodeIdx, TddNodeId};
use super::tdd::Tdd;
use super::weights::WeightStore;
use super::{ChildRef, ValueRef, LEAF_WIDTH};

/// Hash-cons tables for one level.
///
/// Single-pair nodes, the majority, are keyed by a packed `u64` rather than
/// by the pair list.
#[derive(Default)]
struct InternTable {
    /// Single-pair nodes, keyed by `(left, right)` packed into a `u64`.
    single: HashMap<u64, NodeIdx>,
    /// Nodes with two or more pairs, keyed by the pair list.
    multi: HashMap<Box<[ChildPair]>, NodeIdx>,
}

impl InternTable {
    /// The first node indexed under this pair list.
    fn get(&self, pairs: &[ChildPair]) -> Option<NodeIdx> {
        if let [pair] = pairs {
            self.single.get(&(((pair.left.0 as u64) << 32) | pair.right.0 as u64)).copied()
        } else {
            self.multi.get(pairs).copied()
        }
    }

    /// Index a node without replacing an earlier occurrence of its pair list.
    fn insert(&mut self, pairs: &[ChildPair], index: NodeIdx) {
        if let [pair] = pairs {
            self.single.entry(((pair.left.0 as u64) << 32) | pair.right.0 as u64).or_insert(index);
        } else {
            self.multi.entry(pairs.into()).or_insert(index);
        }
    }
}

/// A borrowed level together with the store interpreting its weighted columns.
#[derive(Clone, Copy, Debug)]
pub struct LevelView<'a> {
    level: &'a TddLevel,
    weights: Option<&'a WeightStore>,
    source: VtreeIdx,
}

impl<'a> LevelView<'a> {
    /// Borrow a structural or count-marginal level; weighted levels need [`Tdd::level_view`].
    pub fn unweighted(level: &'a TddLevel) -> Option<Self> {
        (!level.is_weight_marginal()).then_some(Self { level, weights: None, source: VtreeIdx(0) })
    }

    /// The level's structural data or count column.
    pub fn level(self) -> &'a TddLevel { self.level }
}

/// A diagram under construction: one level per vtree node, filled bottom-up.
///
/// Start with [`Tdd::builder`], add nodes, then call [`finish`](Self::finish)
/// with the output node. The result shares the vtree supplied to the builder.
///
/// Level buffers come from the engine's recycling pool. On success they belong
/// to the returned diagram; [`abandon`](Self::abandon) returns them to the pool,
/// while dropping the builder frees them.
///
/// [`finish`](Self::finish) checks references, storage, and marginal columns.
/// It does not prove determinism: the caller must ensure that nodes at a
/// structural level represent disjoint functions and that a child pair belongs
/// to at most one node at that level. Counting relies on these conditions;
/// minimization cannot turn an arbitrary overlapping circuit into a TDD.
///
/// ```
/// use std::sync::Arc;
/// use tididi::{Engine, Tdd};
/// use tididi::diagram::{ChildPair, NEG_LEAF_IDX, POS_LEAF_IDX, TddNodeId};
/// use tididi::vtree::Vtree;
///
/// // x1 ∧ ¬x2 over a two-leaf vtree: one root node with one pair.
/// let eng = Engine::new();
/// let vtree = Arc::new(Vtree::balanced(2));
/// let root = vtree.root();
/// let mut builder = Tdd::builder(&eng, &vtree);
/// let node = builder.push(root, &[ChildPair::new(POS_LEAF_IDX, NEG_LEAF_IDX)]);
/// let f = builder.finish(TddNodeId { vtree: root, local: node })?;
/// # tididi::test_helpers::assert_canonical(&f);
/// assert_eq!(f.model_count()?, 1u32.into());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct TddBuilder {
    vtree: Arc<Vtree>,
    levels: Vec<TddLevel>,
    /// Per-level hash-cons tables, allocated on the first intern call.
    interned: Vec<Option<InternTable>>,
    /// The store a weighted build carries; none in integer mode.
    weights: Option<WeightStore>,
}

impl std::fmt::Debug for TddBuilder {
    /// The vtree the diagram is being assembled over and how much of it is
    /// filled, rather than the pairs themselves.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TddBuilder")
            .field("vtree_nodes", &self.vtree.num_nodes())
            .field("levels_filled", &self.levels.iter().filter(|l| l.slot_count() > 0).count())
            .field("weights", &self.weights.is_some())
            .finish()
    }
}

impl Tdd {
    /// Borrow a level together with any weighted values needed to copy it.
    pub fn level_view(&self, t: VtreeIdx) -> LevelView<'_> {
        LevelView { level: self.level(t), weights: self.weights(), source: t }
    }

    /// Start a diagram over `vtree`, with one empty level per vtree node.
    ///
    /// See [`TddBuilder`].
    pub fn builder(eng: &Engine, vtree: &Arc<Vtree>) -> TddBuilder {
        TddBuilder {
            levels: take_levels(eng, vtree.num_nodes()),
            vtree: Arc::clone(vtree),
            interned: Vec::new(),
            weights: None,
        }
    }
}

impl TddBuilder {
    /// The level of vtree node `t` as built so far.
    pub fn level(&self, t: VtreeIdx) -> &TddLevel {
        &self.levels[t.idx()]
    }

    /// Append a node with these pairs to level `t`, and return its index.
    ///
    /// Build bottom-up: both sides of a pair name a child node that already
    /// exists, which a debug assertion checks here against the child level as
    /// it stands.
    pub fn push(&mut self, t: VtreeIdx, pairs: &[ChildPair]) -> NodeIdx {
        if cfg!(debug_assertions) {
            debug_assert_pairs(&self.vtree, &self.levels, t, pairs);
        }
        let index = self.levels[t.idx()].push_internal_node(pairs);
        if let Some(Some(table)) = self.interned.get_mut(t.idx()) {
            table.insert(pairs, index);
        }
        index
    }

    /// Return the first node with these pairs, indexing prior pushes lazily and appending if absent.
    pub fn intern(&mut self, t: VtreeIdx, pairs: &[ChildPair]) -> NodeIdx {
        if self.interned.is_empty() {
            self.interned.resize_with(self.levels.len(), || None);
        }
        let table = self.interned[t.idx()].get_or_insert_with(|| {
            let mut table = InternTable::default();
            for (i, _) in self.levels[t.idx()].internal_inputs_iter() {
                table.insert(self.levels[t.idx()].pairs_of_idx(i), NodeIdx(i as u32));
            }
            table
        });
        if let Some(index) = table.get(pairs) { return index; }
        self.push(t, pairs)
    }

    /// Attach literal weights, preserving the configuration of any copied weighted levels.
    ///
    /// # Errors
    ///
    /// Refuses a store inconsistent with the levels already copied in.
    pub fn set_weights(&mut self, ws: WeightStore) -> Result<(), TddBuildError> {
        ws.install(&self.vtree, &self.levels, &mut self.weights)
    }

    /// Replace level `t` and its weighted column together, discarding its intern table.
    ///
    /// Copying any level from a weighted source attaches its weight configuration.
    /// Build bottom-up so that the copied references have matching child levels.
    ///
    /// # Errors
    ///
    /// Refuses incompatible weight configurations and count columns in a weighted build.
    pub fn replace_level(&mut self, t: VtreeIdx, from: LevelView<'_>) -> Result<(), TddBuildError> {
        if let Some(source) = from.weights {
            if self.weights.as_ref().is_some_and(|ws| !ws.compatible(source)) {
                return Err(TddBuildError::IncompatibleWeights);
            }
            if self.weights.is_none()
                && let Some(i) = self.levels.iter().position(|level| level.is_marginal() && !level.is_weight_marginal()) {
                    return Err(TddBuildError::CountLevelWithWeights { level: VtreeIdx(i as u32) });
                }
        }
        if from.level.is_marginal() && !from.level.is_weight_marginal() && self.weights.is_some() {
            return Err(TddBuildError::CountLevelWithWeights { level: t });
        }
        let column = if from.level.is_weight_marginal() {
            let ws = from.weights.ok_or(TddBuildError::WeightedLevelWithoutStore { level: t })?;
            let column = ws.level(from.source.idx()).unwrap_or(&[]);
            if column.len() != from.level.slot_count() {
                return Err(TddBuildError::InvalidWeightColumn { level: t, reason: "does not match the level's slot count" });
            }
            Some(column.to_vec())
        } else { None };
        let level = from.level.clone();
        if self.weights.is_none() { self.weights = from.weights.map(WeightStore::empty_like); }
        if let Some(ws) = self.weights.as_mut() {
            ws.take_level(t.idx());
            if let Some(column) = column { ws.set_level(t.idx(), column); }
        }
        self.levels[t.idx()] = level;
        if let Some(table) = self.interned.get_mut(t.idx()) { *table = None; }
        Ok(())
    }

    /// Seat the diagram on `output` and hand it back.
    ///
    /// The result is well-formed but not necessarily canonical: it may hold
    /// unreachable nodes and uncontracted twins.
    /// [`minimize`](crate::Tdd::minimize) makes a valid TDD canonical.
    ///
    /// Storage validity is checked in every build profile. Semantic determinism
    /// remains the caller's responsibility, as described on [`TddBuilder`].
    ///
    /// # Errors
    ///
    /// The first invariant violated — [`TddBuildError::BadOutput`] when
    /// `output` is neither a live root node nor the false sentinel; other variants
    /// identify invalid references, deleted children, marginal columns or leaf storage.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{Engine, Tdd};
    /// use tididi::diagram::{ChildPair, NodeIdx, POS_LEAF_IDX, NEG_LEAF_IDX, TddNodeId};
    /// use tididi::vtree::Vtree;
    ///
    /// let eng = Engine::new();
    /// let vtree = Arc::new(Vtree::balanced(2));
    /// let root = vtree.root();
    ///
    /// let mut b = Tdd::builder(&eng, &vtree);
    /// let node = b.push(root, &[ChildPair::new(POS_LEAF_IDX, NEG_LEAF_IDX)]);
    /// assert!(b.finish(TddNodeId { vtree: root, local: node }).is_ok());
    ///
    /// // An output naming a node the root level does not hold is refused.
    /// let mut b = Tdd::builder(&eng, &vtree);
    /// b.push(root, &[ChildPair::new(POS_LEAF_IDX, NEG_LEAF_IDX)]);
    /// match b.finish(TddNodeId { vtree: root, local: NodeIdx(7) }) {
    ///     Ok(_) => unreachable!("node 7 was never pushed"),
    ///     Err(e) => assert!(!e.to_string().is_empty()),
    /// }
    /// ```
    pub fn finish(mut self, output: TddNodeId) -> Result<Tdd, TddBuildError> {
        check_levels(&self.vtree, &self.levels, output, self.weights.as_ref())?;
        Ok(self.seat(output))
    }

    /// Finish without the storage validation performed by [`finish`](Self::finish).
    ///
    /// Use this only when construction has already established every invariant
    /// checked by `finish`. The result need not be canonical; it may contain
    /// unreachable nodes and uncontracted twins.
    ///
    /// # Safety
    ///
    /// Every level and child reference must satisfy the storage invariants of
    /// [`TddBuilder`]: references name live nodes or valid marginal values in
    /// the appropriate child levels, marginal regions are downward-closed,
    /// weighted columns match their store, and `output` names a live root node
    /// or the false sentinel. The diagram must also be deterministic.
    /// Invalid references can cause out-of-bounds reads in later operations.
    ///
    /// # Panics
    ///
    /// Debug builds check storage validity and panic if it fails.
    pub unsafe fn finish_unchecked(mut self, output: TddNodeId) -> Tdd {
        debug_assert!(
            check_levels(&self.vtree, &self.levels, output, self.weights.as_ref()).is_ok(),
            "an unchecked seat was handed a diagram the checked one would refuse",
        );
        self.seat(output)
    }

    /// Hand the levels to a [`Tdd`] seated on `output`, re-attaching the store.
    /// The invariants are the caller's to have established.
    fn seat(&mut self, output: TddNodeId) -> Tdd {
        let levels = std::mem::take(&mut self.levels);
        let mut tdd = Tdd::from_levels_unchecked(Arc::clone(&self.vtree), levels, output);
        tdd.weights = self.weights.take();
        tdd
    }

    /// Give up on the diagram, returning its levels to the engine's pool.
    pub fn abandon(mut self, eng: &Engine) {
        return_levels(eng, PoolSlot::First, std::mem::take(&mut self.levels));
    }

}

/// The index bound a pair side pointing at level `t` is checked against: the
/// implicit leaf nodes, the value slots of a marginal level, or the nodes
/// stored so far.
fn bound(vtree: &Vtree, levels: &[TddLevel], t: VtreeIdx) -> usize {
    let lvl = &levels[t.idx()];
    if lvl.is_marginal() {
        lvl.slot_count()
    } else if vtree.node(t).is_leaf() {
        LEAF_WIDTH
    } else {
        lvl.nodes.len()
    }
}

/// Panic if a pair pushed at level `t` names a child that does not exist, or
/// sets the reserved bit.
fn debug_assert_pairs(vtree: &Vtree, levels: &[TddLevel], t: VtreeIdx, pairs: &[ChildPair]) {
    let (left, right) = vtree.children(t);
    let (lv, rv) = (levels[left.idx()].child_decoder(), levels[right.idx()].child_decoder());
    let (lb, rb) = (bound(vtree, levels, left), bound(vtree, levels, right));
    for pair in pairs {
        for (side, view, b) in [(pair.left, lv, lb), (pair.right, rv, rb)] {
            debug_assert!(
                !side.is_reserved(),
                "pair side {side:?} pushed at level {t:?} has the reserved bit set",
            );
            let in_range = match view.child(side) {
                ChildRef::Value(ValueRef::Inline(_)) => true,
                r => r.index().unwrap() < b,
            };
            debug_assert!(
                in_range,
                "pair side {side:?} pushed at level {t:?} is past its child level's {b} entries",
            );
        }
    }
}

/// Check the invariants the [module docs](super) list: one level per vtree
/// node, empty leaf levels, no stored leaf-label or empty node, every pair
/// side naming a live slot in its child level (decoded through `ChildDecoder::child`
/// when the child is marginal, and never with bit 31 set), every overflowed
/// marginal count backed by an exact value, marginality downward-closed, a
/// store behind every weight-marginal level, and `output` a node of the root
/// level or `ZERO`.
///
/// # Errors
///
/// The first violation found.
pub(crate) fn check_levels(
    vtree: &Arc<Vtree>,
    levels: &[TddLevel],
    output: TddNodeId,
    weights: Option<&WeightStore>,
) -> Result<(), TddBuildError> {
    use super::primitives::ZERO;

    let n = vtree.num_nodes();
    if levels.len() != n {
        return Err(TddBuildError::LevelCountMismatch {
            expected: n,
            found: levels.len(),
        });
    }
    if let Some(ws) = weights {
        ws.check_levels(vtree, levels)?;
    } else if let Some(t) = vtree.bottomup().find(|t| levels[t.idx()].is_weight_marginal()) {
        return Err(TddBuildError::WeightedLevelWithoutStore { level: t });
    }
    // A structural leaf level stores nothing. A count-marginal one carries the
    // pinned label column `[2, 1, 1]` or nothing at all (its parents then hold
    // the counts inline); a weight-marginal column is checked by the store.
    for (leaf, _var) in vtree.leaf_bottomup() {
        let lvl = &levels[leaf.idx()];
        let stores_structure = !lvl.nodes.is_empty() || !lvl.pairs.is_empty();
        if stores_structure || (!lvl.is_marginal() && lvl.slot_count() != 0) {
            return Err(TddBuildError::NonEmptyLeafLevel(leaf));
        }
        if let Some(counts) = lvl.marginal_counts()
            && !counts.is_empty()
            && counts != super::leaf_column::LEAF_COUNTS
        {
            return Err(TddBuildError::NonEmptyLeafLevel(leaf));
        }
    }
    for (t, left, right) in vtree.internal_bottomup() {
        let lvl = &levels[t.idx()];
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
            levels[left.idx()].child_decoder(),
            levels[right.idx()].child_decoder(),
        );
        let (lb, rb) = (
            bound(vtree, levels, left),
            bound(vtree, levels, right),
        );
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
                    if side.is_reserved() {
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
                    if !vtree.node(child).is_leaf() && !levels[child.idx()].is_marginal() {
                        let local = NodeIdx(view.child(side).index().unwrap() as u32);
                        if levels[child.idx()].nodes[local.idx()].is_tombstone() {
                            return Err(TddBuildError::DeadChild {
                                level: t, node: node_idx, child: TddNodeId { vtree: child, local },
                            });
                        }
                    }
                }
            }
        }
    }
    let root = vtree.root();
    let dead_output = output.local != ZERO && !vtree.node(root).is_leaf()
        && !levels[root.idx()].is_marginal()
        && levels[root.idx()].nodes.get(output.local.idx()).is_some_and(|node| node.is_tombstone());
    if output.vtree != root || dead_output || (output.local != ZERO && output.local.idx() >= bound(vtree, levels, root)) {
        return Err(TddBuildError::BadOutput(output));
    }
    Ok(())
}
