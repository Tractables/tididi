//! Adding and removing one assignment at a time, in place.
//!
//! A diagram compiled from a table is maintained rather than rebuilt when a
//! row arrives or leaves. What one assignment does to a level is small: at
//! every vtree node `v` the value the assignment takes over `v`'s subtree
//! gains or loses one extension, and no other value's extensions change. So
//! the level's blocks move by the least a partition can — the value leaves its
//! block, and every other block stands.
//!
//! The cost is what that split forces on the level's *parents*. A block is a
//! set of values, and the diagram's levels are partitions, so a value that
//! leaves its block makes that block cease to exist and every pair naming it
//! has to be rewritten, whether or not the parent's own function changed.
//! Where the block on the path is already a **singleton** there is nothing to
//! split, however many parents name it, and the update is one pair: the
//! chain of singleton nodes for the new value, and one pair added to or
//! removed from the output node.
//!
//! [`Maintenance`] takes that path. It indexes the diagram once, in time
//! linear in its size, and then answers each update in time linear in the
//! vtree's depth. An update whose path meets a block holding more than the
//! one assignment cannot be done by an edit, and falls back to the rebuild
//! [`Tdd::or_cube`] and [`Tdd::and_clause`] perform — correct, and linear in
//! the diagram. Which route an update took is [`Maintenance::rebuilds`].
//!
//! Whether the edit is available is a property of the diagram, not of the
//! update, so a batch that keeps falling back stops paying for an index it
//! cannot use and leaves the rest of its updates to the rebuild. A batch
//! therefore never costs materially more than the operations it replaces.
//!
//! An edit leaves the diagram sound — its levels are partitions and its nodes
//! are satisfiable — but not canonical: a block whose extensions changed may
//! now belong with another, and only [`Tdd::minimize`] decides that. Minimize
//! once at the end of a batch rather than after every update.

use std::sync::Arc;

use rustc_hash::FxHashMap;

use crate::diagram::{
    ChildPair, EncodedChildRef, Literal, NodeIdx, Tdd, ONE_LEAF_IDX, POS_LEAF_IDX, NEG_LEAF_IDX,
};
use crate::limits::OperationError;
use crate::vtree::{Vtree, VtreeIdx};
use crate::{Context, Engine};

mod index;
use index::Index;
mod update;

/// The `path` entry of a level that has no node for the assignment's value.
const FRESH: u32 = u32::MAX;

/// Consecutive rebuilds after which the batch stops re-indexing.
///
/// A rebuild replaces the diagram, so the index goes with it. Where the edit
/// route is open again afterwards that is a pass over the diagram well spent;
/// where the vtree gives the blocks on every path more than one assignment it
/// is a pass per update and nothing to show for it. The batch therefore tries
/// a few times and then leaves the updates to the rebuild, which is what it
/// would have cost without a batch at all.
const GIVE_UP: u32 = 4;

/// What one probe of the assignment's path found.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Probe {
    /// Every level names the assignment's value; the root's own block is this
    /// node. The assignment is a model of the diagram iff it is the output.
    Found(u32),
    /// The assignment leaves the diagram somewhere below the root, so it is
    /// not a model and the levels marked [`FRESH`] are the ones an insertion
    /// gives a node.
    Absent,
    /// A block on the path holds more than this one assignment, so the update
    /// splits it and rewrites its parents. The rebuild answers instead.
    Splits,
}

/// A batch of single-assignment updates to one diagram.
///
/// Built by [`Tdd::maintain`], which indexes the diagram in time linear in its
/// size; each [`insert_model`](Self::insert_model) or
/// [`remove_model`](Self::remove_model) then costs the vtree's depth, unless
/// its path meets a block holding more than that one assignment, where the
/// rebuild answers instead and the index is rebuilt on the next update.
///
/// The diagram is left sound but not canonical — see the module
/// documentation. Minimize it once the batch is over.
///
/// ```
/// use std::sync::Arc;
/// use tididi::{Tdd, Vtree};
/// let vtree = Arc::new(Vtree::balanced(3));
/// let mut f = Tdd::cube(&vtree, [1, 2, 3])?;
/// {
///     let mut batch = f.maintain()?;
///     batch.insert_model([1, 2, -3])?;
///     batch.insert_model([-1, 2, 3])?;
///     batch.remove_model([1, 2, 3])?;
/// }
/// f.minimize()?;
/// assert_eq!(f.model_count()?, 2u32.into());
/// # tididi::test_helpers::assert_canonical(&f);
/// # Ok::<(), tididi::OperationError>(())
/// ```
pub struct Maintenance<'a> {
    /// The diagram under maintenance.
    tdd: &'a mut Tdd,
    /// Its vtree and execution context, held so the updates can borrow the
    /// diagram mutably.
    vtree: Arc<Vtree>,
    context: Arc<Context>,
    /// The pair index, or `None` once a rebuild invalidated it.
    index: Option<Index>,
    /// Per level, the node denoting the assignment's value over that subtree,
    /// or [`FRESH`]. Written by every probe.
    path: Vec<u32>,
    /// Per leaf level, the label the assignment gives it.
    labels: Vec<u32>,
    /// Literal scratch for the rebuild routes.
    literals: Vec<Literal>,
    /// How many updates took the rebuild route.
    rebuilds: u64,
    /// How many took it in a row; see [`GIVE_UP`].
    misses: u32,
}

impl std::fmt::Debug for Maintenance<'_> {
    /// The batch's own state; the diagram is [`diagram`](Self::diagram).
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Maintenance")
            .field("indexed", &self.index.is_some())
            .field("rebuilds", &self.rebuilds)
            .finish_non_exhaustive()
    }
}

impl Tdd {
    /// Begin a batch of single-assignment updates.
    ///
    /// The index this builds costs one pass over the diagram and is reused by
    /// every update of the batch, so a run of updates between queries pays for
    /// it once. The diagram is left sound but not canonical; minimize it when
    /// the batch is over. See the [module documentation](crate::maintain).
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{Tdd, Vtree};
    /// let vtree = Arc::new(Vtree::balanced(4));
    /// let mut f = Tdd::cube(&vtree, [1, 2, 3, 4])?;
    /// f.maintain()?.insert_model([-1, -2, -3, -4])?;
    /// f.minimize()?;
    /// assert_eq!(f.model_count()?, 2u32.into());
    /// # tididi::test_helpers::assert_canonical(&f);
    /// # Ok::<(), tididi::OperationError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`OperationError::MarginalLevel`] if a level has discarded its
    /// structure, or [`OperationError::OverBudget`] if the index is refused.
    pub fn maintain(&mut self) -> Result<Maintenance<'_>, OperationError> {
        let context = Arc::clone(self.context());
        let vtree = Arc::clone(self.vtree());
        let n = vtree.num_nodes();
        let index = context.run(|eng| Index::build(eng, self))?;
        Ok(Maintenance {
            tdd: self,
            vtree,
            context,
            index: Some(index),
            path: vec![FRESH; n],
            labels: vec![ONE_LEAF_IDX.0; n],
            literals: Vec::new(),
            rebuilds: 0,
            misses: 0,
        })
    }

    /// Add one assignment to this diagram's models, in place.
    ///
    /// Equivalent to [`or_cube`](Self::or_cube) with a cube over every
    /// variable, and the same contract: the result counts correctly but may
    /// need [`minimize`](Self::minimize). For more than one update, open a
    /// [`Maintenance`] batch instead — this builds and discards its index.
    ///
    /// # Errors
    ///
    /// Returns [`OperationError::InvalidLiteral`] for integer zero,
    /// [`OperationError::VariableNotInVtree`] for an absent variable,
    /// [`OperationError::MarginalLevel`] for a level that has discarded its
    /// structure, or [`OperationError::OverBudget`] for a refused allocation.
    pub fn insert_model<L: crate::LiteralInput>(&mut self, model: impl AsRef<[L]>) -> Result<(), OperationError> {
        self.maintain()?.insert_model(model)
    }

    /// Drop one assignment from this diagram's models, in place.
    ///
    /// Equivalent to [`and_clause`](Self::and_clause) with the assignment's
    /// negation, and the same contract. For more than one update, open a
    /// [`Maintenance`] batch instead — this builds and discards its index.
    ///
    /// # Errors
    ///
    /// As [`insert_model`](Self::insert_model).
    pub fn remove_model<L: crate::LiteralInput>(&mut self, model: impl AsRef<[L]>) -> Result<(), OperationError> {
        self.maintain()?.remove_model(model)
    }
}

impl Maintenance<'_> {
    /// How many updates of this batch took the rebuild route rather than an
    /// edit. Zero when every path met singleton blocks only.
    ///
    /// A run of rebuilds is also what makes the batch stop indexing: the
    /// diagram's own shape, not the update, decides whether an edit is
    /// available, so once a few updates in a row have taken the rebuild the
    /// rest of the batch takes it too.
    #[must_use]
    pub fn rebuilds(&self) -> u64 {
        self.rebuilds
    }

    /// The diagram under maintenance.
    #[must_use]
    pub fn diagram(&self) -> &Tdd {
        self.tdd
    }

    /// Read the assignment into `labels`, reporting what it is.
    ///
    /// The fast path needs a value at every leaf; anything else is the
    /// rebuild's to answer.
    fn read_model(&mut self, model: &[Literal]) -> Result<ModelShape, OperationError> {
        self.labels.fill(ONE_LEAF_IDX.0);
        self.literals.clear();
        let mut assigned = 0u32;
        for lit in model {
            let leaf = self.vtree.leaf_of(lit.var).ok_or(OperationError::VariableNotInVtree(lit.var))?;
            let want = if lit.sign { POS_LEAF_IDX.0 } else { NEG_LEAF_IDX.0 };
            let slot = &mut self.labels[leaf.idx()];
            if *slot == ONE_LEAF_IDX.0 {
                *slot = want;
                assigned += 1;
                self.literals.push(*lit);
            } else if *slot != want {
                return Ok(ModelShape::Inconsistent);
            }
        }
        if assigned != self.vtree.num_vars() { return Ok(ModelShape::Partial); }
        Ok(ModelShape::Complete)
    }

    /// Walk the assignment's path bottom-up, filling `path`.
    fn probe(&mut self) -> Probe {
        let index = self.index.as_ref().expect("the caller refreshed the index");
        let root = self.vtree.root();
        let mut answer = Probe::Absent;
        for (t, left, right) in self.vtree.internal_bottomup() {
            let sides = (
                child_slot(&self.vtree, &self.path, &self.labels, left),
                child_slot(&self.vtree, &self.path, &self.labels, right),
            );
            let (Some(l), Some(r)) = sides else {
                self.path[t.idx()] = FRESH;
                continue;
            };
            match index.owners[t.idx()].get(&(l, r)) {
                None => self.path[t.idx()] = FRESH,
                Some(&i) => {
                    if t != root && !index.singleton[t.idx()][i as usize] { return Probe::Splits; }
                    self.path[t.idx()] = i;
                    if t == root { answer = Probe::Found(i); }
                }
            }
        }
        answer
    }

    /// The pair naming the assignment at level `t`, whose children's path
    /// entries are already resolved.
    fn path_pair(&self, t: VtreeIdx) -> ChildPair {
        let (left, right) = self.vtree.children(t);
        let l = child_slot(&self.vtree, &self.path, &self.labels, left).expect("the children are resolved");
        let r = child_slot(&self.vtree, &self.path, &self.labels, right).expect("the children are resolved");
        ChildPair::new(EncodedChildRef::from_raw(l), EncodedChildRef::from_raw(r))
    }

    /// Take the diagram out, leaving a false one in its place, for a route
    /// that consumes its operand.
    fn take_diagram(&mut self, eng: &Engine) -> Tdd {
        let placeholder = crate::build::constant_zero(eng, &self.vtree);
        std::mem::replace(self.tdd, placeholder)
    }
}

/// What [`Maintenance::read_model`] found.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ModelShape {
    /// A value for every variable of the vtree.
    Complete,
    /// Some variable left free: the general cube routes answer.
    Partial,
    /// A variable in both polarities: the cube is false.
    Inconsistent,
}

/// The node index a child level contributes to its parent's pair, or `None`
/// when the assignment's value has no node there yet.
fn child_slot(vtree: &Vtree, path: &[u32], labels: &[u32], child: VtreeIdx) -> Option<u32> {
    if vtree.node(child).is_leaf() {
        return Some(labels[child.idx()]);
    }
    match path[child.idx()] {
        FRESH => None,
        i => Some(i),
    }
}

/// Whether a child reference denotes exactly one assignment over its subtree.
fn child_is_singleton(singleton: &[Vec<bool>], child: VtreeIdx, is_leaf: bool, slot: u32) -> bool {
    if is_leaf {
        return slot == POS_LEAF_IDX.0 || slot == NEG_LEAF_IDX.0;
    }
    singleton[child.idx()][slot as usize]
}

#[cfg(test)]
#[path = "tests/mod.rs"]
mod tests;
