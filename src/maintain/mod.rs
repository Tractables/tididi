//! Update a diagram as rows enter or leave a table.
//!
//! [`Maintenance`] indexes a diagram once, then reuses that index for a batch
//! of updates. A complete assignment can often be added or removed by editing
//! the output's pairs and adding singleton nodes below it. When an assignment
//! belongs to a block containing other assignments, the update rebuilds through
//! [`Tdd::or_cube`] or [`Tdd::and_clause`] instead.
//!
//! Indexing traverses the diagram. Each update reads its literals and walks
//! the vtree; an edit can also copy or shift a node's pair list. A rebuild
//! copies the diagram before applying the update, so a failure preserves the
//! previous function. [`Maintenance::rebuilds`] reports how often that route
//! was needed. After consecutive rebuilds, the batch stops rebuilding its index.
//!
//! Updates preserve the represented function on error, including earlier
//! successful updates in the batch. Storage may contain unreachable nodes;
//! the batch remains usable. Successful edits need not be canonical: call
//! [`Tdd::minimize`] after the batch, before comparing diagram shapes.

use std::sync::Arc;

use crate::diagram::{
    ChildPair, EncodedChildRef, Literal, Tdd, ONE_LEAF_IDX, POS_LEAF_IDX, NEG_LEAF_IDX,
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

/// A batch of assignment updates to one diagram.
///
/// Created by [`Tdd::maintain`] or [`Engine::maintain`]. The index is reused
/// when updates can edit singleton nodes; other updates rebuild the diagram.
/// Minimize the diagram after the batch. See the [module](crate::maintain)
/// for costs and recovery after an error.
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
    /// Explicit execution workspace, when the batch is bounded.
    engine: Option<&'a Engine>,
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
    /// Begin a batch of assignment updates.
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
        context.run(|eng| Maintenance::new(eng, self, None))
    }

    /// Add every assignment matching `model` to this diagram, in place.
    ///
    /// A literal for every vtree variable names one model. Omitted variables
    /// are free: `[1]` adds every assignment with variable 1 true, and an empty
    /// input adds all assignments. Repeated literals are harmless; opposite
    /// literals of the same variable name no assignments and change nothing.
    ///
    /// Equivalent to [`or_cube`](Self::or_cube). The result counts correctly
    /// but may need [`minimize`](Self::minimize). For several updates, reuse a
    /// [`Maintenance`] batch instead of building an index for each call.
    /// An error preserves the previous function; the diagram remains usable.
    ///
    /// # Errors
    ///
    /// Returns [`OperationError::InvalidLiteral`] for integer zero,
    /// [`OperationError::VariableNotInVtree`] for an absent variable,
    /// [`OperationError::MarginalLevel`] for a level that has discarded its
    /// structure, [`OperationError::OverBudget`] for a refused allocation,
    /// or [`OperationError::Stopped`] when a stop request fires.
    pub fn insert_model<L: crate::LiteralInput>(&mut self, model: impl AsRef<[L]>) -> Result<(), OperationError> {
        self.maintain()?.insert_model(model)
    }

    /// Remove every assignment matching `model` from this diagram, in place.
    ///
    /// Input semantics follow [`insert_model`](Self::insert_model): partial
    /// input removes every completion, empty input removes all models, and
    /// contradictory input changes nothing. Equivalent to
    /// [`and_clause`](Self::and_clause) with the input's literals negated.
    /// For several updates, reuse a [`Maintenance`] batch.
    /// An error preserves the previous function; the diagram remains usable.
    ///
    /// # Errors
    ///
    /// As [`insert_model`](Self::insert_model).
    pub fn remove_model<L: crate::LiteralInput>(&mut self, model: impl AsRef<[L]>) -> Result<(), OperationError> {
        self.maintain()?.remove_model(model)
    }
}

impl Engine {
    /// Begin [`Tdd::maintain`] using this engine's scratch and limits.
    /// Each update is a separate operation; a failed update preserves the
    /// function and leaves the batch usable.
    ///
    /// # Errors
    ///
    /// As [`Tdd::maintain`], including cancellation and refused allocations.
    pub fn maintain<'a>(&'a self, tdd: &'a mut Tdd) -> Result<Maintenance<'a>, OperationError> {
        Maintenance::new(self, tdd, Some(self))
    }
}

impl<'a> Maintenance<'a> {
    /// Allocate the index and per-level paths before lending the diagram.
    fn new(eng: &Engine, tdd: &'a mut Tdd, engine: Option<&'a Engine>) -> Result<Self, OperationError> {
        let lim = eng.limits();
        let _op = lim.begin_operation();
        lim.check_stop()?;
        tdd.require_structure()?;
        let vtree = Arc::clone(tdd.vtree());
        let context = Arc::clone(tdd.context());
        let index = Index::build(eng, tdd)?;
        let mut path = Vec::new();
        let mut labels = Vec::new();
        lim.try_resize(&mut path, vtree.num_nodes(), FRESH)?;
        lim.try_resize(&mut labels, vtree.num_nodes(), ONE_LEAF_IDX.0)?;
        Ok(Self { tdd, vtree, context, engine, index: Some(index), path, labels,
            literals: Vec::new(), rebuilds: 0, misses: 0 })
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
        let mut assigned = 0u32;
        for lit in model {
            let leaf = self.vtree.leaf_of(lit.var).ok_or(OperationError::VariableNotInVtree(lit.var))?;
            let want = if lit.sign { POS_LEAF_IDX.0 } else { NEG_LEAF_IDX.0 };
            let slot = &mut self.labels[leaf.idx()];
            if *slot == ONE_LEAF_IDX.0 {
                *slot = want;
                assigned += 1;
            } else if *slot != want {
                return Ok(ModelShape::Inconsistent);
            }
        }
        if assigned != self.vtree.num_leaves() { return Ok(ModelShape::Partial); }
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

#[cfg(test)]
#[path = "tests/mod.rs"]
mod tests;
