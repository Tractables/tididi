//! The scaled-float arm of the streaming fold.
//!
//! Under [`Arithmetic::ScaledFloat`](crate::diagram::Arithmetic::ScaledFloat)
//! a conjunction's summed-out levels fold into bare [`ScaledFloat`] columns,
//! 16 bytes a value where a [`WeightValue`] takes 80, and a column whose
//! parent the same conjunction sums out never enters the weight store: the
//! apply keeps it until the parent has read it (see
//! [`MarginalDomain::commit_in_flight`](crate::marginal::transition::MarginalDomain::commit_in_flight)).
//! The values are those [`WeightFold`] computes on [`WeightValue::Scaled`]
//! values, bit for bit.

use super::*;
use crate::diagram::{ChildDecoder, ChildPair, EncodedChildRef, LeafLabel, PairsIter, ScaledFloat, TddLevel, WeightStore, WeightValue};
use crate::value::FloatFold;
use crate::vtree::{Vtree, VtreeNode};

/// A value of a scaled-float store.
#[inline(always)]
fn scaled(v: &WeightValue) -> ScaledFloat {
    *v.as_scaled().expect("a scaled-float store holds scaled values")
}

/// One child level's values as the float fold reads them.
pub(crate) enum FloatColumn<'a> {
    /// A column this apply folded: a level it summed out, or one folded on
    /// the way to a level it sums out.
    Folded(&'a [ScaledFloat]),
    /// A level an earlier operation made marginal: its weight-store column.
    Stored(&'a [WeightValue]),
    /// A structural vtree leaf's three label values, in label order.
    Leaf([ScaledFloat; crate::diagram::LEAF_WIDTH]),
}

impl FloatColumn<'_> {
    /// The value at slot or node `i`. The variant is one per child level,
    /// so the branch is taken the same way for every read of a row loop.
    #[inline(always)]
    pub(crate) fn get(&self, i: usize) -> ScaledFloat {
        match self {
            FloatColumn::Folded(col) => col[i],
            FloatColumn::Stored(col) => scaled(&col[i]),
            FloatColumn::Leaf(col) => col[i],
        }
    }
}

/// Resolve a child reference of level `level_idx` for the ensure walk, as
/// the weighted domain's reader does: a leaf by label, a level this apply
/// folded from its column, and a level marginal before the apply from the
/// store.
fn read_float(
    level_idx: usize,
    side: EncodedChildRef,
    vtree: &Vtree,
    ws: &WeightStore,
    levels: &[TddLevel],
    computed: &[Option<Vec<ScaledFloat>>],
) -> ScaledFloat {
    if side.is_reserved() {
        return ScaledFloat::ZERO;
    }
    if let VtreeNode::Leaf { var, .. } = *vtree.node(VtreeIdx(level_idx as u32)) {
        return scaled(&ws.leaf_val(var, LeafLabel::from_idx(ChildDecoder::marginal().index(side))));
    }
    let level = &levels[level_idx];
    if let Some(col) = &computed[level_idx] {
        return col[level.child_decoder().index(side)];
    }
    if let Some(values) = level.is_weight_marginal().then(|| ws.level(level_idx)).flatten() {
        return scaled(&values[ChildDecoder::marginal().index(side)]);
    }
    unreachable!("scaled-float value not available for level {}", level_idx);
}

impl ValueDomain for FloatFold {
    /// The weight store: the leaf weights, and the columns of levels made
    /// marginal before this apply.
    type Store = WeightStore;

    /// A folded column, a stored one, or a leaf's three values.
    type ChildCol<'a> = FloatColumn<'a>;

    type Scalar = ScaledFloat;
    type Col = Vec<ScaledFloat>;

    fn alloc_col(
        eng: &Engine,
        width: usize,
        _store: &WeightStore,
    ) -> Result<Vec<ScaledFloat>, OperationError> {
        let mut v: Vec<ScaledFloat> = Vec::new();
        eng.limits().reserve_exact(&mut v, width)?;
        v.resize(width, ScaledFloat::ZERO);
        Ok(v)
    }

    fn set_col(
        _eng: &Engine,
        col: &mut Vec<ScaledFloat>,
        i: usize,
        v: ScaledFloat,
    ) -> Result<(), OperationError> {
        col[i] = v;
        Ok(())
    }

    fn try_with_capacity(
        eng: &Engine,
        cap: usize,
    ) -> Result<Vec<ScaledFloat>, OperationError> {
        let mut col = Vec::new();
        eng.limits().reserve_exact(&mut col, cap)?;
        Ok(col)
    }

    /// The whole bound when it takes at most half the headroom. A column
    /// that doubles its way up leaves each block it outgrew behind, beside a
    /// final block up to twice its length, and the levels a Markov network's
    /// sum-out streams have every cell alive, so the bound is their length.
    #[inline]
    fn stream_capacity(eng: &Engine, cells: usize, widest: usize) -> usize {
        let bytes = (cells as u64).saturating_mul(std::mem::size_of::<ScaledFloat>() as u64);
        if bytes <= eng.limits().headroom() / 2 { cells } else { widest }
    }

    #[inline]
    fn push_col(
        eng: &Engine,
        col: &mut Vec<ScaledFloat>,
        v: ScaledFloat,
    ) -> Result<(), OperationError> {
        eng.limits().reserve(col, 1)?;
        col.push(v);
        Ok(())
    }

    fn col_len(col: &Vec<ScaledFloat>) -> usize {
        col.len()
    }

    #[inline]
    fn stream_columns(cache: &StreamCache) -> &[Option<Vec<ScaledFloat>>] {
        cache.float()
    }

    #[inline]
    fn store_of(ws: Option<&WeightStore>) -> &WeightStore {
        ws.expect("a scaled-float column is only ever built with a store attached")
    }

    #[inline]
    fn fold_node(at: &FoldScope<'_, FloatFold>, pairs: PairsIter<'_>) -> ScaledFloat {
        let FoldInput { vtree, levels, store } = at.input;
        FloatFold::fold(
            pairs,
            |k| read_float(at.left, k, vtree, store, levels, at.computed),
            |k| read_float(at.right, k, vtree, store, levels, at.computed),
        )
    }

    /// A folded column before the store's, which holds none for a level
    /// whose column the apply keeps.
    fn child_view<'a>(
        level_idx: usize,
        vtree: &Vtree,
        level: &'a TddLevel,
        computed: &'a [Option<Vec<ScaledFloat>>],
        store: &'a WeightStore,
    ) -> StreamChild<'a, FloatFold> {
        if let Some(col) = &computed[level_idx] {
            return StreamChild { col: FloatColumn::Folded(col), view: level.child_decoder() };
        }
        if let Some(col) = level.is_weight_marginal().then(|| store.level(level_idx)).flatten() {
            return StreamChild { col: FloatColumn::Stored(col), view: ChildDecoder::marginal() };
        }
        let VtreeNode::Leaf { var, .. } = *vtree.node(VtreeIdx(level_idx as u32)) else {
            unreachable!("FloatFold::child_view: no values for level {level_idx}");
        };
        // The label-ordered triple `diagram::leaf_column_vals` defines.
        let col = std::array::from_fn(|i| scaled(&store.leaf_val(var, LeafLabel::from_idx(i))));
        StreamChild { col: FloatColumn::Leaf(col), view: ChildDecoder::structural() }
    }

    #[inline]
    fn fold_cell(
        pairs: &[ChildPair],
        left: &StreamChild<'_, FloatFold>,
        right: &StreamChild<'_, FloatFold>,
        _store: &WeightStore,
    ) -> ScaledFloat {
        FloatFold::fold(
            pairs.iter().copied(),
            |k| left.col.get(left.view.index(k)),
            |k| right.col.get(right.view.index(k)),
        )
    }
}
