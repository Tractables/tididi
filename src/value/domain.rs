//! Arithmetic and column access shared by standalone and streaming folds.
//!
//! Two domains exist — integer model counts, and exact semiring weights held
//! in an external [`WeightStore`] — and two folds consume them: the
//! marginalization cascade over a finished diagram, and the streaming column
//! built inside an apply. Both folds are written once against [`ValueDomain`] and reserve columns
//! through the engine, returning allocation refusals as operation errors.

use crate::diagram::{ChildPair, TddLevel, WeightStore};
use crate::Engine;

use crate::limits::OperationError;
use crate::vtree::{Vtree, VtreeIdx};

use super::{walk_bottom_up, Retention, StreamCache};

/// The diagram a fold reads: its vtree, its levels, and the domain's store.
pub(crate) struct FoldInput<'a, D: ValueDomain> {
    pub(crate) vtree: &'a Vtree,
    pub(crate) levels: &'a [TddLevel],
    pub(crate) store: &'a D::Store,
}

impl<D: ValueDomain> Clone for FoldInput<'_, D> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<D: ValueDomain> Copy for FoldInput<'_, D> {}

/// One level of the ensure walk, as [`ValueDomain::fold_node`] sees it: the
/// level and its two children, the diagram, and the columns computed so far.
pub(crate) struct FoldScope<'a, D: ValueDomain> {
    pub(crate) lvl: usize,
    pub(crate) left: usize,
    pub(crate) right: usize,
    pub(crate) input: FoldInput<'a, D>,
    pub(crate) computed: &'a [Option<D::Col>],
}

/// Per-child read view of the child level's fold column, taken before the dense
/// scatter loop.
///
/// The column is borrowed from the child level's marginal storage or the
/// per-apply scratch, not copied, so a wide marginal child's storage is not
/// doubled; the caller splits the output level and its two children apart once
/// so the views coexist with the `&mut` output borrow. In the integer domain
/// the column's `all_u64` certificate lets the cell fold use a widening
/// `u64×u64→u128` multiply, which cannot overflow, in place of `checked_mul`;
/// inline refs decode to at most `MARGINAL_INLINE_MAX`, so the certificate
/// covers every read in the cell loop.
pub(crate) struct StreamChild<'a, D: ValueDomain> {
    pub(crate) col: D::ChildCol<'a>,
    /// True iff this view is of a marginal child level (its refs are
    /// marginal-side slot refs, possibly bit-30 tagged). When set, the per-cell
    /// fold decodes the ref before indexing the column. For a
    /// non-marginal/leaf child the ref is a plain node index (bit 30 may be a
    /// real high bit) — do not decode.
    pub(crate) is_marginal: bool,
}

/// One value domain: the arithmetic, its column, and what the two folds need
/// from it.
///
/// Each method is a place where the two domains differ; what they share (the
/// bottom-up ensure walk and the column fill in [`Self::fold_column`]) is
/// written once here. Each domain's `Σ pairs (left × right)` fold is an
/// inherent method ([`IntFold::fold`](super::IntFold::fold),
/// [`WeightFold::fold`](super::WeightFold::fold)), because the two reader
/// shapes differ: a lazy `CountRead` per side against a `Cow<WeightValue>` per
/// side.
///
/// The ensure walk builds a column with [`Self::alloc_col`] and
/// [`Self::set_col`]; the apply's streaming driver with
/// [`Self::try_with_capacity`] and [`Self::push_col`], one push per alive
/// cell.
pub(crate) trait ValueDomain: Sized {
    /// One per-node fold result (`Count` | `WeightValue`).
    type Scalar;

    /// The column of one level's values.
    type Col;

    /// State the domain carries beside the diagram: the weight store, or
    /// nothing at all.
    type Store;

    /// How this domain reads one child's column for the duration of a row
    /// loop. Stored columns are borrowed; weighted leaf values are computed into an owned column.
    type ChildCol<'a>;

    /// A fresh `width`-element column of the domain's zero, reserved through
    /// the engine.
    fn alloc_col(
        eng: &Engine,
        width: usize,
        store: &Self::Store,
    ) -> Result<Self::Col, OperationError>;

    /// Store one fold result at slot `i` of a pre-sized column.
    fn set_col(
        eng: &Engine,
        col: &mut Self::Col,
        i: usize,
        v: Self::Scalar,
    ) -> Result<(), OperationError>;

    /// An empty column with room for `cap` appends reserved, filled by
    /// [`Self::push_col`].
    fn try_with_capacity(
        eng: &Engine,
        cap: usize,
    ) -> Result<Self::Col, OperationError>;

    /// Append one fold result to a column.
    fn push_col(
        eng: &Engine,
        col: &mut Self::Col,
        v: Self::Scalar,
    ) -> Result<(), OperationError>;

    /// Number of values currently stored in a column.
    fn col_len(col: &Self::Col) -> usize;

    /// This domain's already-computed child columns inside the per-apply cache.
    ///
    /// The cache is one enum because an apply runs a single value kind
    /// throughout; this hook is where that kind is read back out.
    fn stream_columns(cache: &StreamCache) -> &[Option<Self::Col>];

    /// This domain's store, given the apply's weight store.
    ///
    /// The integer domain carries no state and ignores the argument; the
    /// weighted domain requires one, and a weighted column is only ever opened
    /// where a store is attached.
    fn store_of(ws: Option<&WeightStore>) -> &Self::Store;

    /// Fold node `i` of the scope's level: `Σ over its pairs (left × right)`,
    /// with this domain's child readers resolving each `u32` ref against the
    /// levels and the columns computed so far.
    ///
    /// Impls carry `#[inline]`: the ensure walk's per-node loop calls this once
    /// per node, and must not gain a call there.
    fn fold_node(at: &FoldScope<'_, Self>, i: usize) -> Self::Scalar;

    /// Open a read view of child level `left_idx`'s column. `level` is `levels[left_idx]`,
    /// handed in already split off from the output level's `&mut` borrow.
    fn child_view<'a>(
        left_idx: usize,
        vtree: &Vtree,
        level: &'a TddLevel,
        computed: &'a [Option<Self::Col>],
        store: &'a Self::Store,
    ) -> StreamChild<'a, Self>;

    /// Collapse one alive cell's collected pairs to a single scalar:
    /// `Σ left[p.left] × right[p.right]`.
    fn fold_cell(
        pairs: &[ChildPair],
        left: &StreamChild<'_, Self>,
        right: &StreamChild<'_, Self>,
        store: &Self::Store,
    ) -> Self::Scalar;

    /// Compute a complete column without changing the diagram or its cached
    /// columns. Child values must already be available through `input` or `computed`.
    fn fold_column(
        eng: &Engine,
        t: VtreeIdx,
        input: FoldInput<'_, Self>,
        computed: &[Option<Self::Col>],
        mut before_node: impl FnMut(u64) -> Result<(), OperationError>,
    ) -> Result<Self::Col, OperationError> {
        let lvl = t.idx();
        let (left, right) = input.vtree.children(t);
        let level = &input.levels[lvl];
        let mut col = Self::alloc_col(eng, level.slot_count(), input.store)?;
        let at = FoldScope { lvl, left: left.idx(), right: right.idx(), input, computed };
        for (i, pairs) in level.internal_inputs_iter() {
            before_node(1 + pairs.len() as u64)?;
            let value = Self::fold_node(&at, i);
            Self::set_col(eng, &mut col, i, value)?;
        }
        Ok(col)
    }

    /// Populate `computed[root]` and every column below it that a fold at
    /// `root` will read.
    ///
    /// One walk for both folds. `already_marginal` is the caller's answer to
    /// "this level's values are already stored, don't recompute them" — the
    /// cascade and the apply key that on different state, which is why it is
    /// asked of the caller rather than of the domain. `before_node` receives
    /// one unit per node plus its pair count before arithmetic starts.
    fn ensure(
        eng: &Engine,
        root: VtreeIdx,
        input: FoldInput<'_, Self>,
        computed: &mut [Option<Self::Col>],
        already_marginal: &dyn Fn(usize) -> bool,
        retain: Retention,
        mut before_node: impl FnMut(u64) -> Result<(), OperationError>,
    ) -> Result<(), OperationError> {
        let vtree = input.vtree;
        walk_bottom_up(
            vtree,
            root,
            computed,
            // A leaf's values resolve on demand inside the readers, so it
            // never gets a column.
            |computed, i| {
                computed[i].is_some()
                    || already_marginal(i)
                    || vtree.node(VtreeIdx(i as u32)).is_leaf()
            },
            |computed, t| {
                computed[t.idx()] = Some(Self::fold_column(
                    eng, t, input, computed, &mut before_node,
                )?);
                Ok(())
            },
            |computed, i| computed[i] = None,
            // The root is never released, so nothing is exempt.
            retain.frontier(root),
        )
    }
}

mod count;
mod weight;
