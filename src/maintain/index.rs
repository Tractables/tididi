//! The pair index one maintenance batch builds once and edits as it goes.
//!
//! Two facts about a level are what an update needs and a diagram does not
//! store: **which node owns a given child pair**, so the assignment's path can
//! be walked a level at a time rather than searched; and **whether a node
//! denotes a single assignment**, which is what decides that its block will
//! not split. Both are one bottom-up pass over the diagram.
//!
//! The leaves record a third fact between them: whether any is referenced
//! through `One`. The leaf labels are not disjoint — `One` is `Pos` and `Neg`
//! together — so a diagram that names `One` has no node for a single value at
//! that leaf, and the edit route is unavailable.

use std::sync::Arc;

use rustc_hash::FxHashMap;

use crate::diagram::{ChildPair, NodeIdx, Tdd, ONE_LEAF_IDX, POS_LEAF_IDX, NEG_LEAF_IDX};
use crate::limits::OperationError;
use crate::vtree::VtreeIdx;
use crate::Engine;

/// What a batch knows about the diagram between updates.
pub(super) struct Index {
    /// Per internal level, the node owning each of its stored pairs.
    pub(super) owners: Vec<FxHashMap<(u32, u32), u32>>,
    /// Per internal level, whether node `i` denotes exactly one assignment.
    /// Leaf levels hold an empty vector; their labels decide.
    pub(super) singleton: Vec<Vec<bool>>,
    /// Whether any leaf level is named through `One`.
    pub(super) any_one_mode: bool,
}

impl Index {
    /// Index `tdd` bottom-up. Runs inside the caller's operation scope.
    ///
    /// # Errors
    ///
    /// [`OperationError::MarginalLevel`] for a level that has discarded its
    /// structure, [`OperationError::OverBudget`] for a refused allocation, and
    /// [`OperationError::Stopped`] when a stop request fires during the walk.
    pub(super) fn build(eng: &Engine, tdd: &Tdd) -> Result<Index, OperationError> {
        let lim = eng.limits();
        let vtree = Arc::clone(tdd.vtree());
        let n = vtree.num_nodes();
        let mut owners: Vec<FxHashMap<(u32, u32), u32>> = Vec::new();
        lim.try_resize(&mut owners, n, FxHashMap::default())?;
        let mut singleton: Vec<Vec<bool>> = Vec::new();
        lim.try_resize(&mut singleton, n, Vec::new())?;
        let mut any_one_mode = false;
        let mut gate = lim.gate();

        for (t, left, right) in vtree.internal_bottomup() {
            let level = tdd.level(t);
            if level.is_marginal() { return Err(OperationError::MarginalLevel(t)); }
            let left_leaf = vtree.node(left).is_leaf();
            let right_leaf = vtree.node(right).is_leaf();
            let width = level.slot_count();
            let mut flags = Vec::new();
            lim.try_resize(&mut flags, width, false)?;
            lim.reserve_map(&mut owners[t.idx()], level.live_pairs())?;
            for (i, flag) in flags.iter_mut().enumerate() {
                if !level.nodes()[i].is_internal() { continue; }
                let pairs = level.pairs_of_idx(i);
                for p in pairs {
                    gate.poll(1)?;
                    let previous = owners[t.idx()].insert((p.left.raw(), p.right.raw()), i as u32);
                    debug_assert!(previous.is_none() || previous == Some(i as u32),
                        "two nodes of level {t:?} own one pair; structural determinism broken");
                    any_one_mode |= left_leaf && p.left.raw() == ONE_LEAF_IDX.0;
                    any_one_mode |= right_leaf && p.right.raw() == ONE_LEAF_IDX.0;
                }
                *flag = pairs.len() == 1
                    && child_is_singleton(&singleton, left, left_leaf, pairs[0].left.raw())
                    && child_is_singleton(&singleton, right, right_leaf, pairs[0].right.raw());
            }
            singleton[t.idx()] = flags;
        }
        gate.flush()?;
        Ok(Index { owners, singleton, any_one_mode })
    }

    /// Reserve the index changes before changing diagram storage.
    pub(super) fn reserve_edit(&mut self, eng: &Engine, t: VtreeIdx, new_node: bool) -> Result<(), OperationError> {
        eng.limits().reserve_map(&mut self.owners[t.idx()], 1)?;
        if new_node { eng.limits().reserve(&mut self.singleton[t.idx()], 1)?; }
        Ok(())
    }

    /// Record a new singleton after `reserve_edit` succeeds.
    pub(super) fn note_appended_node(&mut self, t: VtreeIdx, pair: ChildPair, idx: NodeIdx) {
        self.owners[t.idx()].insert((pair.left.raw(), pair.right.raw()), idx.0);
        let flags = &mut self.singleton[t.idx()];
        debug_assert_eq!(flags.len(), idx.idx(), "the node was appended at the level's end");
        flags.push(true);
    }

    /// Record a pair added to an existing node after reserving its index entry.
    pub(super) fn note_appended_pair(&mut self, t: VtreeIdx, pair: ChildPair, idx: NodeIdx) {
        self.owners[t.idx()].insert((pair.left.raw(), pair.right.raw()), idx.0);
        if let Some(flag) = self.singleton[t.idx()].get_mut(idx.idx()) { *flag = false; }
    }

    /// Record a pair dropped from the node at level `t`.
    pub(super) fn note_removed_pair(&mut self, t: VtreeIdx, pair: ChildPair) {
        self.owners[t.idx()].remove(&(pair.left.raw(), pair.right.raw()));
    }
}

/// Whether a child reference denotes exactly one assignment over its subtree.
fn child_is_singleton(singleton: &[Vec<bool>], child: VtreeIdx, is_leaf: bool, slot: u32) -> bool {
    if is_leaf {
        return slot == POS_LEAF_IDX.0 || slot == NEG_LEAF_IDX.0;
    }
    singleton[child.idx()][slot as usize]
}
