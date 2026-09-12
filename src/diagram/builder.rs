//! [`TddBuilder`], the one way to assemble a [`Tdd`] level by level.

use std::collections::HashMap;
use std::sync::Arc;

use crate::engine::Engine;
use crate::vtree::{Vtree, VtreeIdx};

use super::build_error::TddBuildError;
use super::level::{LevelKind, TddLevel, ValueKind};
use super::pool::{return_levels, take_levels, PoolSlot};
use super::primitives::{InputPair, NodeIdx, TddNodeId};
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
    multi: HashMap<Box<[InputPair]>, NodeIdx>,
}

/// A diagram under construction: one level per vtree node, filled bottom-up.
///
/// Obtained from [`Tdd::build`], finished with [`finish`](Self::finish) or
/// dropped with [`abandon`](Self::abandon). The builder holds the vtree the
/// result will be seated on, so a finished diagram can never be paired with a
/// tree it was not built against.
///
/// Levels come from the engine's recycling pool and go back to it on either
/// exit, so a caller never handles the pool itself.
///
/// ```
/// use std::sync::Arc;
/// use tididi::{Engine, Tdd};
/// use tididi::diagram::{InputPair, NEG_LEAF_IDX, POS_LEAF_IDX, TddNodeId};
/// use tididi::vtree::Vtree;
///
/// // x1 ∧ ¬x2 over a two-leaf vtree: one root node with one pair.
/// let eng = Engine::new();
/// let vtree = Arc::new(Vtree::balanced(2));
/// let root = vtree.root();
/// let mut b = Tdd::build(&eng, &vtree);
/// let node = b.push(root, &[InputPair { left: POS_LEAF_IDX, right: NEG_LEAF_IDX }]);
/// let f = b.finish(TddNodeId { vtree: root, local: node }).unwrap();
/// assert_eq!(f.model_count(), 1u32.into());
/// ```
pub struct TddBuilder {
    vtree: Arc<Vtree>,
    levels: Vec<TddLevel>,
    /// Per-level hash-cons tables. Empty until the first
    /// [`share`](Self::share) — a builder that never shares pays nothing.
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
            .field("levels_filled", &self.levels.iter().filter(|l| l.width() > 0).count())
            .field("weights", &self.weights.is_some())
            .finish()
    }
}

impl Tdd {
    /// Start a diagram over `vtree`, with one empty level per vtree node.
    ///
    /// See [`TddBuilder`].
    pub fn build(eng: &Engine, vtree: &Arc<Vtree>) -> TddBuilder {
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
    pub fn push(&mut self, t: VtreeIdx, pairs: &[InputPair]) -> NodeIdx {
        if cfg!(debug_assertions) {
            debug_assert_pairs(&self.vtree, &self.levels, t, pairs);
        }
        self.levels[t.idx()].push_internal_node(pairs)
    }

    /// Hash-cons level `t` from here on: [`intern`](Self::intern) on it
    /// returns the existing node when one holds the same pairs.
    ///
    /// Opt-in per level because the tables cost memory a caller that mints
    /// distinct nodes anyway would never get back.
    pub fn share(&mut self, t: VtreeIdx) {
        if self.interned.is_empty() {
            self.interned.resize_with(self.levels.len(), || None);
        }
        self.interned[t.idx()].get_or_insert_with(InternTable::default);
    }

    /// The node of level `t` holding these pairs, appending one only if the
    /// level does not already have it.
    ///
    /// Requires [`share`](Self::share) on `t`; without it the level has no
    /// table and this appends unconditionally.
    pub fn intern(&mut self, t: VtreeIdx, pairs: &[InputPair]) -> NodeIdx {
        let Some(Some(table)) = self.interned.get_mut(t.idx()) else {
            return self.push(t, pairs);
        };
        // The tables are borrowed here, so the range check runs on its own
        // borrow of the levels below rather than through `push`.
        let level = &mut self.levels[t.idx()];
        if let [pair] = pairs {
            let key = ((pair.left.0 as u64) << 32) | pair.right.0 as u64;
            if let Some(&existing) = table.single.get(&key) {
                return existing;
            }
            let idx = level.push_internal_node(pairs);
            table.single.insert(key, idx);
            if cfg!(debug_assertions) {
                debug_assert_pairs(&self.vtree, &self.levels, t, pairs);
            }
            return idx;
        }
        if let Some(&existing) = table.multi.get(pairs) {
            return existing;
        }
        let idx = level.push_internal_node(pairs);
        table.multi.insert(pairs.into(), idx);
        if cfg!(debug_assertions) {
            debug_assert_pairs(&self.vtree, &self.levels, t, pairs);
        }
        idx
    }

    /// Attach the store holding the values of every weight-marginal level the
    /// build copies in; the finished diagram carries it, as after
    /// [`Tdd::set_weights`]. Without one, [`finish`](Self::finish) refuses a
    /// weight-marginal level.
    pub fn set_weights(&mut self, ws: WeightStore) {
        self.weights = Some(ws);
    }

    /// Copy `from` into level `t` whole.
    ///
    /// A marginal level's values and their overflow backing come across as
    /// they are, so the copy keeps the level's marginality. A weight-marginal
    /// level's values live in the store the source diagram carries, which
    /// [`set_weights`](Self::set_weights) attaches to the build.
    pub fn copy_level(&mut self, t: VtreeIdx, from: &TddLevel) {
        let dst = &mut self.levels[t.idx()];
        match from.kind() {
            LevelKind::Marginal(ValueKind::Counts) => {
                let counts = from
                    .marginal_counts()
                    .expect("a count-marginal level holds counts")
                    .to_vec();
                dst.become_marginal(counts, from.marginal_counts_big().cloned());
            }
            LevelKind::Marginal(ValueKind::Weights) => {
                dst.become_marginal_weighted(from.width() as u32);
            }
            LevelKind::Structural => {
                dst.reserve_nodes(from.nodes().len());
                for node in from.nodes() {
                    dst.push_internal_node(from.pairs_of(node));
                }
            }
        }
    }

    /// Seat the diagram on `output` and hand it back.
    ///
    /// The result is well-formed but not necessarily canonical: it may hold
    /// unreachable nodes and distinct nodes computing the same function.
    /// [`minimize`](crate::reduce::minimize) makes it canonical.
    ///
    /// The invariants the [module docs](super) list are checked here, in every
    /// profile, in one walk of the diagram.
    ///
    /// # Errors
    ///
    /// The first invariant violated — [`TddBuildError::BadOutput`] when
    /// `output` is not a node of the root level,
    /// [`TddBuildError::WeightedLevelWithoutStore`] when a level copied in by
    /// [`copy_level`](Self::copy_level) is weight-marginal and no store was
    /// attached with [`set_weights`](Self::set_weights), and the rest of
    /// [`TddBuildError`]'s variants for the structural ones.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{Engine, Tdd};
    /// use tididi::diagram::{InputPair, NodeIdx, POS_LEAF_IDX, NEG_LEAF_IDX, TddNodeId};
    /// use tididi::vtree::Vtree;
    ///
    /// let eng = Engine::new();
    /// let vtree = Arc::new(Vtree::balanced(2));
    /// let root = vtree.root();
    ///
    /// let mut b = Tdd::build(&eng, &vtree);
    /// let node = b.push(root, &[InputPair { left: POS_LEAF_IDX, right: NEG_LEAF_IDX }]);
    /// assert!(b.finish(TddNodeId { vtree: root, local: node }).is_ok());
    ///
    /// // An output naming a node the root level does not hold is refused.
    /// let mut b = Tdd::build(&eng, &vtree);
    /// b.push(root, &[InputPair { left: POS_LEAF_IDX, right: NEG_LEAF_IDX }]);
    /// match b.finish(TddNodeId { vtree: root, local: NodeIdx(7) }) {
    ///     Ok(_) => unreachable!("node 7 was never pushed"),
    ///     Err(e) => assert!(!e.to_string().is_empty()),
    /// }
    /// ```
    pub fn finish(mut self, output: TddNodeId) -> Result<Tdd, TddBuildError> {
        check_levels(&self.vtree, &self.levels, output, self.weights.is_some())?;
        Ok(self.seat(output))
    }

    /// [`finish`](Self::finish) without the invariant walk (debug-asserted
    /// instead). The caller guarantees every invariant the [module docs](super)
    /// list; a violation surfaces later as a wrong answer or a panic.
    pub(crate) fn finish_unchecked(mut self, output: TddNodeId) -> Tdd {
        debug_assert!(
            check_levels(&self.vtree, &self.levels, output, self.weights.is_some()).is_ok(),
            "an unchecked seat was handed a diagram the checked one would refuse",
        );
        self.seat(output)
    }

    /// Hand the levels to a [`Tdd`] seated on `output`, re-attaching the store.
    /// The invariants are the caller's to have established.
    fn seat(&mut self, output: TddNodeId) -> Tdd {
        let levels = std::mem::take(&mut self.levels);
        let mut tdd = Tdd::from_levels_unchecked(Arc::clone(&self.vtree), levels, output);
        if let Some(store) = self.weights.take() {
            tdd.set_weights(store);
        }
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
        lvl.width()
    } else if vtree.node(t).is_leaf() {
        LEAF_WIDTH
    } else {
        lvl.nodes.len()
    }
}

/// Panic if a pair pushed at level `t` names a child that does not exist, or
/// sets the reserved bit.
fn debug_assert_pairs(vtree: &Vtree, levels: &[TddLevel], t: VtreeIdx, pairs: &[InputPair]) {
    let (left, right) = vtree.children(t);
    let (lv, rv) = (levels[left.idx()].side_view(), levels[right.idx()].side_view());
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
/// side in range for its child level (decoded through `SideView::child`
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
    has_weights: bool,
) -> Result<(), TddBuildError> {
    use super::primitives::ZERO;

    let n = vtree.num_nodes();
    if levels.len() != n {
        return Err(TddBuildError::LevelCountMismatch {
            expected: n,
            found: levels.len(),
        });
    }
    // A structural leaf level stores nothing; a marginalized one carries one
    // value per implicit node, which is what its width counts.
    for (leaf, _var) in vtree.leaf_bottomup() {
        let lvl = &levels[leaf.idx()];
        let stores_structure = !lvl.nodes.is_empty() || !lvl.pairs.is_empty();
        if stores_structure || (!lvl.is_marginal() && lvl.width() != 0) {
            return Err(TddBuildError::NonEmptyLeafLevel(leaf));
        }
    }
    for (t, left, right) in vtree.internal_bottomup() {
        let lvl = &levels[t.idx()];
        if lvl.is_weight_marginal() && !has_weights {
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
                }
            }
        }
    }
    let root = vtree.root();
    if output.vtree != root || (output.local != ZERO && output.local.idx() >= bound(vtree, levels, root)) {
        return Err(TddBuildError::BadOutput(output));
    }
    Ok(())
}
