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
    /// The value of one node.
    type Value;
    /// One level's worth of values.
    type Col;

    /// A fresh `width`-slot column. `width == 0` is how the walk releases one.
    fn alloc(&self, eng: &Engine, width: usize) -> Self::Col;

    /// Store node `i`'s value.
    fn set(&self, eng: &Engine, col: &mut Self::Col, i: usize, v: Self::Value);

    /// The value of leaf `label` for variable `var`. `LeafLabel::Zero` never
    /// reaches a stored leaf slot, but the seed loop passes it, so an
    /// implementation must answer for it.
    fn leaf(&self, var: VarId, label: LeafLabel) -> Self::Value;

    /// Fill `col` from a marginal level's stored values rather than folding it.
    /// A marginal level has no pairs to fold, so its stored column already is
    /// the answer for its whole subtree.
    fn marginal_column(&self, eng: &Engine, tdd: &Tdd, t: VtreeIdx, col: &mut Self::Col);

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
/// caller's to keep.
pub(crate) fn fold_level<F: LevelFold>(
    f: &F,
    eng: &Engine,
    tdd: &Tdd,
    cols: &mut [F::Col],
    t: VtreeIdx,
) {
    let ti = t.idx();
    if tdd.vtree.node(t).is_leaf() {
        let var = tdd.vtree.leaf_var(t);
        for i in 0..LEAF_WIDTH {
            let v = f.leaf(var, LeafLabel::from_idx(i));
            f.set(eng, &mut cols[ti], i, v);
        }
        return;
    }
    if tdd.levels[ti].is_marginal() {
        f.marginal_column(eng, tdd, t, &mut cols[ti]);
        return;
    }
    let (left, right) = tdd.vtree.children(t);
    let (left_idx, right_idx) = (left.idx(), right.idx());
    let left_view = tdd.levels[left_idx].child_decoder();
    let right_view = tdd.levels[right_idx].child_decoder();
    for (i, pairs) in tdd.levels[ti].internal_inputs_iter() {
        let v = f.fold_node(
            pairs,
            Side { col: &cols[left_idx], view: left_view },
            Side { col: &cols[right_idx], view: right_view },
        );
        f.set(eng, &mut cols[ti], i, v);
    }
}

/// The whole walk: every level of the diagram, children before parents.
///
/// Under [`ColumnRetention::Frontier`] a child's column is released as soon as
/// its parent's is complete; the output level is exempt, being the one column
/// read afterwards. See [`walk_bottom_up`] for the order and the frontier.
///
/// `ensure_col` is the caller's per-level column sizing, called before each
/// level is written. `poll` is the caller's stop-axis gate: with one, the walk
/// is cut between levels and the caller gets the error; without one it runs to
/// the end.
///
/// # Errors
///
/// Propagates the armed stop, polled at every internal level boundary.
pub(crate) fn fold_bottom_up<F: LevelFold>(
    f: &F,
    eng: &Engine,
    tdd: &Tdd,
    cols: &mut [F::Col],
    retain: ColumnRetention,
    mut poll: Option<&mut PollGate>,
    mut ensure_col: impl FnMut(&mut [F::Col], usize),
) -> Result<(), OperationError> {
    let lim = eng.limits();
    walk_bottom_up(
        &tdd.vtree,
        tdd.vtree.root(),
        cols,
        |_, _| false,
        |cols, t| {
            // Metered in nodes of the level, the unit the fold scales with.
            // With no gate there is no stop axis to observe and the walk
            // cannot be cut.
            if !tdd.vtree.node(t).is_leaf()
                && let Some(gate) = poll.as_deref_mut()
            {
                lim.poll(gate, tdd.levels[t.idx()].slot_count() as u64 + 1)?;
            }
            ensure_col(cols, t.idx());
            fold_level(f, eng, tdd, cols, t);
            Ok(())
        },
        |cols, i| cols[i] = f.alloc(eng, 0),
        retain.frontier(tdd.output.vtree),
    )
}

/// The walk with no stop axis to observe, for a caller holding an engine that
/// arms none. Structurally infallible: without a gate nothing is polled.
pub(crate) fn fold_bottom_up_unpolled<F: LevelFold>(
    f: &F,
    eng: &Engine,
    tdd: &Tdd,
    cols: &mut [F::Col],
    retain: ColumnRetention,
    ensure_col: impl FnMut(&mut [F::Col], usize),
) {
    fold_bottom_up(f, eng, tdd, cols, retain, None, ensure_col)
        .expect("an unpolled walk observes no stop axis");
}
