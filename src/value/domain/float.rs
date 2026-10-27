//! The scaled-float arm of the streaming fold.
//!
//! Under [`Arithmetic::ScaledFloat`](crate::diagram::Arithmetic::ScaledFloat)
//! a conjunction's summed-out levels fold into [`FloatCol`] columns, 8
//! bytes a value where a [`WeightValue`] takes 64, and a column whose
//! parent the same conjunction sums out never enters the weight store: the
//! apply keeps it until the parent has read it (see
//! [`MarginalDomain::commit_in_flight`](crate::marginal::transition::MarginalDomain::commit_in_flight)).
//! The values are those [`WeightFold`] computes on [`WeightValue::Scaled`]
//! values, bit for bit. A cell whose two child columns are narrow, with
//! offsets whose products and their sums stay normal floats, is summed in
//! plain floats under the two columns' scales ([`plain_fold`]): the same
//! roundings on the same values, without a scaled float per read.

use super::*;
use crate::diagram::{ChildDecoder, ChildPair, EncodedChildRef, LeafLabel, PairsIter, ScaledFloat, TddLevel, WeightStore, WeightValue};
use crate::value::{FloatCol, FloatFold, FloatSlice};
use crate::vtree::{Vtree, VtreeNode};

/// A value of a scaled-float store.
#[inline(always)]
fn scaled(v: &WeightValue) -> ScaledFloat {
    *v.as_scaled().expect("a scaled-float store holds scaled values")
}

/// One child level's values as the float fold reads them.
pub(crate) enum FloatColumn<'a> {
    /// A column this apply folded, a level it summed out or one folded on
    /// the way to a level it sums out, that holds whole values.
    Folded(FloatSlice<'a>),
    /// Such a column held narrow, with the range of its offsets.
    Narrow(Plain<'a>),
    /// A level an earlier operation made marginal: its weight-store column.
    Stored(&'a [WeightValue]),
    /// A structural vtree leaf's three label values, in label order.
    Leaf([ScaledFloat; crate::diagram::LEAF_WIDTH]),
    /// The same as floats under one scale, with the range of their offsets.
    LeafPlain([f64; crate::diagram::LEAF_WIDTH], Offsets),
}

/// The least and greatest offset `d` of a column's nonzero floats, each
/// `m · 2^d` with `m` in `[0.5, 1)`; `(0, 0)` for a column of zeros.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Offsets {
    scale: i64,
    lo: i64,
    hi: i64,
}

/// Floats `x` standing for `x · 2^scale`, as a narrow column holds them.
#[derive(Clone, Copy)]
pub(crate) struct Plain<'a> {
    vals: &'a [f64],
    at: Offsets,
}

impl Offsets {
    /// The offsets of `vals` under `scale`.
    fn of(vals: &[f64], scale: i64) -> Offsets {
        let (lo, hi) = vals
            .iter()
            .filter(|&&x| x != 0.0)
            .map(|&x| ((x.to_bits() >> 52) & 0x7ff) as i64 - 1022)
            .fold((i64::MAX, i64::MIN), |(lo, hi), d| (lo.min(d), hi.max(d)));
        match lo <= hi {
            true => Offsets { scale, lo, hi },
            false => Offsets { scale, lo: 0, hi: 0 },
        }
    }
}

impl<'a> FloatColumn<'a> {
    /// The column of a folded column's values.
    fn folded(col: FloatSlice<'a>) -> FloatColumn<'a> {
        match col {
            FloatSlice::Narrow(vals, scale) => FloatColumn::Narrow(Plain { vals, at: Offsets::of(vals, scale) }),
            wide => FloatColumn::Folded(wide),
        }
    }

    /// The column of a leaf's values, as floats under the greatest of
    /// their exponents where they all lie within a narrow column's range
    /// of it.
    fn leaf(vals: [ScaledFloat; crate::diagram::LEAF_WIDTH]) -> FloatColumn<'a> {
        let scale = vals.iter().filter(|v| !v.is_zero()).map(ScaledFloat::exponent).max().unwrap_or(0);
        let mut plain = [0.0; crate::diagram::LEAF_WIDTH];
        for (x, v) in plain.iter_mut().zip(&vals) {
            match v.offset_from(scale) {
                Some(f) => *x = f,
                None => return FloatColumn::Leaf(vals),
            }
        }
        FloatColumn::LeafPlain(plain, Offsets::of(&plain, scale))
    }

    /// The value at slot or node `i`. The variant is one per child level,
    /// so the branch is taken the same way for every read of a row loop.
    #[inline(always)]
    pub(crate) fn get(&self, i: usize) -> ScaledFloat {
        match self {
            FloatColumn::Folded(col) => col.get(i),
            FloatColumn::Narrow(col) => ScaledFloat::at_offset(col.vals[i], col.at.scale),
            FloatColumn::Stored(col) => scaled(&col[i]),
            FloatColumn::Leaf(col) => col[i],
            FloatColumn::LeafPlain(vals, at) => ScaledFloat::at_offset(vals[i], at.scale),
        }
    }

    /// The column as plain floats, where it is narrow.
    #[inline(always)]
    fn plain(&self) -> Option<Plain<'_>> {
        match self {
            FloatColumn::Narrow(col) => Some(*col),
            FloatColumn::LeafPlain(vals, at) => Some(Plain { vals, at: *at }),
            _ => None,
        }
    }
}

/// The least sum of two offsets [`plain_fold`] takes: a product of floats
/// at offsets `a` and `b` is at least `2^(a + b - 2)`, a normal float.
const PLAIN_LO: i64 = -1020;
/// The greatest: a sum of fewer than `2^32` products each below
/// `2^(a + b)` stays below `2^1023`.
const PLAIN_HI: i64 = 990;

/// `Σ left × right` over `pairs` in plain floats, the left column's floats
/// times the right's in pair order, standing under the sum of their
/// scales; `None` where a sum cancelled below the normal floats.
///
/// `ScaledSum` rounds each product and each addition once, and moves its
/// own scale only by exact powers of two. With every product a normal float
/// and every partial sum a normal float or an exact difference, the plain
/// floats round the same values the same way, so the result is
/// [`FloatFold::fold`]'s, bit for bit.
#[inline(always)]
fn plain_fold(
    pairs: &[ChildPair],
    left: Plain<'_>,
    lview: ChildDecoder,
    right: Plain<'_>,
    rview: ChildDecoder,
) -> Option<ScaledFloat> {
    let mut sum = 0.0f64;
    for pair in pairs {
        sum += left.vals[lview.index(pair.left)] * right.vals[rview.index(pair.right)];
    }
    (sum == 0.0 || sum.is_normal()).then(|| ScaledFloat::at_offset(sum, left.at.scale + right.at.scale))
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
    computed: &[Option<FloatCol>],
) -> ScaledFloat {
    if side.is_reserved() {
        return ScaledFloat::ZERO;
    }
    if let VtreeNode::Leaf { var, .. } = *vtree.node(VtreeIdx(level_idx as u32)) {
        return scaled(&ws.leaf_val(var, LeafLabel::from_idx(ChildDecoder::marginal().index(side))));
    }
    let level = &levels[level_idx];
    if let Some(col) = &computed[level_idx] {
        return col.get(level.child_decoder().index(side));
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
    type Col = FloatCol;

    fn alloc_col(
        eng: &Engine,
        width: usize,
        _store: &WeightStore,
    ) -> Result<FloatCol, OperationError> {
        FloatCol::zeros(eng, width)
    }

    fn set_col(
        eng: &Engine,
        col: &mut FloatCol,
        i: usize,
        v: ScaledFloat,
    ) -> Result<(), OperationError> {
        col.set(eng, i, v)
    }

    fn try_with_capacity(
        eng: &Engine,
        cap: usize,
    ) -> Result<FloatCol, OperationError> {
        FloatCol::with_capacity(eng, cap)
    }

    /// The whole bound when it takes at most half the headroom. A column
    /// that doubles its way up leaves each block it outgrew behind, beside a
    /// final block up to twice its length, and the levels a Markov network's
    /// sum-out streams have every cell alive, so the bound is their length.
    #[inline]
    fn stream_capacity(eng: &Engine, cells: usize, widest: usize) -> usize {
        let bytes = (cells as u64).saturating_mul(FloatCol::NARROW_BYTES as u64);
        if bytes <= eng.limits().headroom() / 2 { cells } else { widest }
    }

    #[inline]
    fn push_col(
        eng: &Engine,
        col: &mut FloatCol,
        v: ScaledFloat,
    ) -> Result<(), OperationError> {
        col.push(eng, v)
    }

    fn col_len(col: &FloatCol) -> usize {
        col.len()
    }

    #[inline]
    fn stream_columns(cache: &StreamCache) -> &[Option<FloatCol>] {
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
        computed: &'a [Option<FloatCol>],
        store: &'a WeightStore,
    ) -> StreamChild<'a, FloatFold> {
        if let Some(col) = &computed[level_idx] {
            return StreamChild { col: FloatColumn::folded(col.slice()), view: level.child_decoder() };
        }
        if let Some(col) = level.is_weight_marginal().then(|| store.level(level_idx)).flatten() {
            return StreamChild { col: FloatColumn::Stored(col), view: ChildDecoder::marginal() };
        }
        let VtreeNode::Leaf { var, .. } = *vtree.node(VtreeIdx(level_idx as u32)) else {
            unreachable!("FloatFold::child_view: no values for level {level_idx}");
        };
        // The label-ordered triple `diagram::leaf_column_vals` defines.
        let col = std::array::from_fn(|i| scaled(&store.leaf_val(var, LeafLabel::from_idx(i))));
        StreamChild { col: FloatColumn::leaf(col), view: ChildDecoder::structural() }
    }

    #[inline]
    fn fold_cell(
        pairs: &[ChildPair],
        left: &StreamChild<'_, FloatFold>,
        right: &StreamChild<'_, FloatFold>,
        _store: &WeightStore,
    ) -> ScaledFloat {
        if let (Some(l), Some(r)) = (left.col.plain(), right.col.plain())
            && l.at.lo + r.at.lo >= PLAIN_LO
            && l.at.hi + r.at.hi <= PLAIN_HI
            && let Some(v) = plain_fold(pairs, l, left.view, r, right.view)
        {
            return v;
        }
        FloatFold::fold(
            pairs.iter().copied(),
            |k| left.col.get(left.view.index(k)),
            |k| right.col.get(right.view.index(k)),
        )
    }
}

#[cfg(test)]
#[path = "tests/float.rs"]
mod tests;
