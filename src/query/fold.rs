//! The one bottom-up traversal every whole-diagram query runs.
//!
//! Model counting, satisfiability and semiring evaluation ask different
//! questions of the same walk: seed the leaves, fold each internal node's pairs
//! into a value, read the marginal levels' stored values instead of folding them,
//! and release a column once its single parent has consumed it. Only the
//! arithmetic differs, so the per-level fold is written once here, driven by
//! [`walk_bottom_up`], and each query supplies its own [`LevelFold`].

use crate::value::{walk_bottom_up, ColumnRetention};
use crate::diagram::{EncodedChildRef, ChildRef, LeafLabel, PairsIter, ChildDecoder, Tdd, ValueRef, LEAF_WIDTH};
use crate::engine::Engine;
use crate::limits::PollGate;
use crate::limits::OperationError;
use crate::vtree::{VarId, VtreeIdx};

/// A child level as a pair fold reads it: its column, and how to decode a
/// reference into it.
pub(crate) struct Side<'a, C> {
    pub(crate) col: &'a C,
    pub(crate) view: ChildDecoder,
}

impl<C> Clone for Side<'_, C> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<C> Copy for Side<'_, C> {}

/// What one query computes per node, and what column it keeps.
pub(crate) trait LevelFold {
    /// Count work by node slots in bounded batches instead of by pair visits.
    const NODE_WORK: bool = false;

    /// The value of one node.
    type Value;
    /// One level's worth of values.
    type Col: Default;

    /// A fresh `width`-slot column, returning a reservation refusal.
    fn alloc(&self, eng: &Engine, width: usize) -> Result<Self::Col, OperationError>;

    /// Store node `i`'s value.
    fn set(&self, eng: &Engine, col: &mut Self::Col, i: usize, v: Self::Value) -> Result<(), OperationError>;

    /// Drop a completed child's column, releasing any charge owned by this pass.
    fn release(&self, _eng: &Engine, col: &mut Self::Col) {
        *col = Self::Col::default();
    }

    /// The value of leaf `label` for variable `var`. `LeafLabel::Zero` never
    /// reaches a stored leaf slot, but the seed loop passes it, so an
    /// implementation must answer for it.
    fn leaf(&self, var: VarId, label: LeafLabel) -> Self::Value;

    /// Fill `col` from a marginal level's stored values rather than folding it.
    /// A marginal level has no pairs to fold, so its stored column already is
    /// the answer for its whole subtree.
    fn marginal_column(&self, eng: &Engine, tdd: &Tdd, t: VtreeIdx, col: &mut Self::Col) -> Result<(), OperationError>;

    /// Fold node `i` of an internal level: `Σ over pairs (left × right)`.
    fn fold_node(
        &self,
        pairs: PairsIter<'_>,
        left: Side<'_, Self::Col>,
        right: Side<'_, Self::Col>,
    ) -> Self::Value;
}

/// The generic `Σ over pairs (left × right)`, for a fold whose arithmetic is
/// one pass of ordinary semiring operations. A fold that needs the pair list
/// twice — the u128 count that re-reads it exactly on overflow — writes its own
/// [`LevelFold::fold_node`] instead of implementing this.
pub(crate) trait PairAlgebra: LevelFold {
    /// The additive identity.
    fn zero(&self) -> Self::Value;
    /// Node `i`'s value, read back out of a child's column.
    fn read(&self, col: &Self::Col, i: usize) -> Self::Value;
    /// The value a marginal reference carries inline, in its own bits.
    fn inline(&self, count: u32) -> Self::Value;
    /// Accumulate: the semiring `+`.
    fn add_assign(&self, acc: &mut Self::Value, v: &Self::Value);
    /// The semiring `×`.
    fn mul(&self, a: &Self::Value, b: &Self::Value) -> Self::Value;
    /// True once `acc` can no longer change, so the pair loop may stop early.
    fn short_circuit(&self, _acc: &Self::Value) -> bool {
        false
    }

    /// The pair loop itself. `fold_node` delegates here.
    fn sum_over_pairs(
        &self,
        pairs: PairsIter<'_>,
        left: Side<'_, Self::Col>,
        right: Side<'_, Self::Col>,
    ) -> Self::Value {
        let mut acc = self.zero();
        for pair in pairs {
            let l = self.child_value(left, pair.left);
            let r = self.child_value(right, pair.right);
            let product = self.mul(&l, &r);
            self.add_assign(&mut acc, &product);
            if self.short_circuit(&acc) {
                break;
            }
        }
        acc
    }

    /// One side of one pair: a stored value, or one carried inline.
    fn child_value(&self, side: Side<'_, Self::Col>, r: EncodedChildRef) -> Self::Value {
        match side.view.child(r) {
            ChildRef::Value(ValueRef::Inline(c)) => self.inline(c),
            ChildRef::Node(crate::diagram::NodeIdx(i))
            | ChildRef::Value(ValueRef::Slot(i)) => self.read(side.col, i as usize),
        }
    }
}

/// Compute one level's column: seed it if it is a leaf, copy it if it is
/// marginal, fold it otherwise.
///
/// The children's columns must already be complete — the walk order is the
/// caller's to keep. Integer folds charge node slots in bounded batches; other
/// algebras charge pair visits at node boundaries.
pub(crate) fn fold_level<F: LevelFold>(
    f: &F,
    eng: &Engine,
    tdd: &Tdd,
    cols: &mut [F::Col],
    t: VtreeIdx,
    mut poll: Option<&mut PollGate>,
) -> Result<(), OperationError> {
    let ti = t.idx();
    if let Some(gate) = poll.as_deref_mut() {
        eng.limits().poll(gate, 1)?;
    }
    if tdd.vtree.node(t).is_leaf() {
        let var = tdd.vtree.leaf_var(t);
        for i in 0..LEAF_WIDTH {
            let v = f.leaf(var, LeafLabel::from_idx(i));
            f.set(eng, &mut cols[ti], i, v)?;
        }
        return Ok(());
    }
    if tdd.levels[ti].is_marginal() {
        return f.marginal_column(eng, tdd, t, &mut cols[ti]);
    }
    let (left, right) = tdd.vtree.children(t);
    let (left_idx, right_idx) = (left.idx(), right.idx());
    let left_view = tdd.levels[left_idx].child_decoder();
    let right_view = tdd.levels[right_idx].child_decoder();
    let level = &tdd.levels[ti];
    let batch = if F::NODE_WORK { eng.limits().reduce_poll_stride().clamp(1, 256) as usize } else { usize::MAX };
    for start in (0..level.nodes.len()).step_by(batch) {
        let end = start.saturating_add(batch).min(level.nodes.len());
        if F::NODE_WORK && let Some(gate) = poll.as_deref_mut() {
            eng.limits().poll(gate, (end - start) as u64)?;
        }
        for (i, pairs) in level.internal_inputs_range(start..end) {
            if !F::NODE_WORK && let Some(gate) = poll.as_deref_mut() {
                eng.limits().poll(gate, pairs.len() as u64 + 1)?;
            }
            let v = f.fold_node(
                pairs,
                Side { col: &cols[left_idx], view: left_view },
                Side { col: &cols[right_idx], view: right_view },
            );
            f.set(eng, &mut cols[ti], i, v)?;
        }
    }
    Ok(())
}

/// The whole walk: every level of the diagram, children before parents.
///
/// Under [`ColumnRetention::Frontier`] a child's column is released as soon as
/// its parent's is complete; the output level is exempt, being the one column
/// read afterwards. See [`walk_bottom_up`] for the order and the frontier.
///
/// `ensure_col` is the caller's per-level column sizing, called before each
/// level is written. `poll` is the caller's stop-axis gate: with one, the walk
/// is cut at amortized node boundaries and the caller gets the error; without one it runs to
/// the end.
///
/// # Errors
///
/// Propagates allocation refusals and the armed stop at amortized node boundaries.
pub(crate) fn fold_bottom_up<F: LevelFold>(
    f: &F,
    eng: &Engine,
    tdd: &Tdd,
    cols: &mut [F::Col],
    retain: ColumnRetention,
    mut poll: Option<&mut PollGate>,
    mut ensure_col: impl FnMut(&mut [F::Col], usize) -> Result<(), OperationError>,
) -> Result<(), OperationError> {
    walk_bottom_up(
        &tdd.vtree,
        tdd.vtree.root(),
        cols,
        |_, _| false,
        |cols, t| {
            ensure_col(cols, t.idx())?;
            fold_level(f, eng, tdd, cols, t, poll.as_deref_mut())
        },
        |cols, i| f.release(eng, &mut cols[i]),
        retain.frontier(tdd.output.vtree),
    )
}

/// The convenience walk without a stop gate, panicking on a reservation refusal.
pub(crate) fn fold_bottom_up_unpolled<F: LevelFold>(
    f: &F,
    eng: &Engine,
    tdd: &Tdd,
    cols: &mut [F::Col],
    retain: ColumnRetention,
    ensure_col: impl FnMut(&mut [F::Col], usize),
) {
    let mut ensure_col = ensure_col;
    fold_bottom_up(f, eng, tdd, cols, retain, None, |cols, ti| { ensure_col(cols, ti); Ok(()) })
        .expect("query fold: allocation refused");
}
