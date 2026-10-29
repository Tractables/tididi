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
//! roundings on the same values, without a scaled float per read. Where
//! the two columns' offsets do not fit that range as they are, a column
//! whose offsets lie away from its middle is read times a power of two
//! that centres them ([`Plain::new`], [`plain_fold_centred`]).

use super::*;
use crate::diagram::{ChildDecoder, ChildPair, EncodedChildRef, LeafLabel, PairsIter, ScaledFloat, TddLevel, WeightStore, WeightValue};
use crate::value::{FloatCol, FloatFold, FloatSlice};
use crate::isa::avx2_fn;
#[cfg(all(target_arch = "x86_64", not(miri)))]
use crate::isa::Avx2;
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
/// `m · 2^d` with `m` in `[0.5, 1)`; `(0, 0)` for a column of zeros. `neg`
/// says whether a float has its sign bit set.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Offsets {
    scale: i64,
    lo: i64,
    hi: i64,
    neg: bool,
}

/// Floats `x` standing for `x · 2^scale`, as a narrow column holds them,
/// with the exponent `centre` of the power of two that centres their
/// offsets ([`Plain::new`]).
#[derive(Clone, Copy)]
pub(crate) struct Plain<'a> {
    vals: &'a [f64],
    at: Offsets,
    centre: i64,
}

impl<'a> Plain<'a> {
    /// The floats `vals` under `scale`. Their `centre` moves their offsets
    /// to the middle of the half of [`PLAIN_LO`]`..=`[`PLAIN_HI`] each
    /// factor of a product may take, where they lie outside that half and
    /// span less than it; it is 0 otherwise. A power of two that keeps each
    /// float normal changes no fraction.
    fn new(vals: &'a [f64], scale: i64) -> Plain<'a> {
        let at = Offsets::of(vals, scale);
        let (lo, hi) = (PLAIN_LO / 2, PLAIN_HI / 2);
        let k = (lo + hi) / 2 - (at.lo + at.hi).div_euclid(2);
        let centre = match (at.lo >= lo && at.hi <= hi) || !(-1022..=1023).contains(&k) || at.lo + k < lo || at.hi + k > hi {
            true => 0,
            false => k,
        };
        Plain { vals, at, centre }
    }

    /// The offsets of the floats times `2^centre`.
    #[inline(always)]
    fn centred(&self) -> Offsets {
        let k = self.centre;
        Offsets { scale: self.at.scale - k, lo: self.at.lo + k, hi: self.at.hi + k, neg: self.at.neg }
    }
}

/// `2^k`, for `k` in `-1022..=1023`.
#[inline(always)]
fn pow2(k: i64) -> f64 {
    f64::from_bits(((k + 1023) as u64) << 52)
}

/// The columns [`Offsets::of`] reads with [`avx2_fn`]'s features where
/// the processor has them; a shorter one costs less than the call.
#[cfg(all(target_arch = "x86_64", not(miri)))]
const AVX2_OFFSETS: usize = 32;

impl Offsets {
    /// The offsets of `vals` under `scale`, by [`Offsets::read`] compiled
    /// with [`avx2_fn`]'s features for a column of 32 floats or more where
    /// the processor has them.
    fn of(vals: &[f64], scale: i64) -> Offsets {
        #[cfg(all(target_arch = "x86_64", not(miri)))]
        if vals.len() >= AVX2_OFFSETS
            && let Some(avx2) = Avx2::detect()
        {
            // Safety: an `Avx2` exists only where the processor has the
            // features `avx2_offsets` enables.
            return unsafe { avx2_offsets(avx2, vals, scale) };
        }
        Offsets::read(vals, scale)
    }

    /// The offsets of `vals` under `scale`, read off the floats' exponent
    /// fields with no branch a float: the least field over the nonzero
    /// floats, where a zero reads as `0x800`, above every field, and the
    /// greatest over all, where a zero's field, 0, is at most any other's.
    #[inline(always)]
    fn read(vals: &[f64], scale: i64) -> Offsets {
        let (mut lo, mut hi, mut signs) = (0x800i64, 0i64, 0u64);
        for &x in vals {
            let bits = x.to_bits();
            let field = ((bits >> 52) & 0x7ff) as i64;
            signs |= bits;
            lo = lo.min(field | i64::from(bits << 1 == 0) << 11);
            hi = hi.max(field);
        }
        let neg = signs >> 63 != 0;
        match lo < 0x800 {
            true => Offsets { scale, lo: lo - 1022, hi: hi - 1022, neg },
            false => Offsets { scale, lo: 0, hi: 0, neg },
        }
    }
}

avx2_fn! {
    /// [`Offsets::read`] compiled with [`avx2_fn`]'s features.
    #[inline(never)]
    fn avx2_offsets(_: Avx2, vals: &[f64], scale: i64) -> Offsets {
        Offsets::read(vals, scale)
    }
}

impl<'a> FloatColumn<'a> {
    /// The column of a folded column's values.
    fn folded(col: FloatSlice<'a>) -> FloatColumn<'a> {
        match col {
            FloatSlice::Narrow(vals, scale) => FloatColumn::Narrow(Plain::new(vals, scale)),
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
            FloatColumn::LeafPlain(vals, at) => Some(Plain { vals, at: *at, centre: 0 }),
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

/// [`plain_fold`] with each column's floats times `2^centre`, for a cell
/// whose columns are plain and whose centred offsets fit its range; `None`
/// for any other cell. Kept out of line, so the cells that fit as they are
/// keep their loop.
#[inline(never)]
fn plain_fold_centred(
    pairs: &[ChildPair],
    lchild: &StreamChild<'_, FloatFold>,
    rchild: &StreamChild<'_, FloatFold>,
) -> Option<ScaledFloat> {
    let (Some(left), Some(right)) = (lchild.col.plain(), rchild.col.plain()) else { return None };
    if left.centre == 0 && right.centre == 0 {
        return None;
    }
    let (lview, rview) = (lchild.view, rchild.view);
    let (l, r) = (left.centred(), right.centred());
    if l.lo + r.lo < PLAIN_LO || l.hi + r.hi > PLAIN_HI {
        return None;
    }
    let (by_left, by_right) = (pow2(left.centre), pow2(right.centre));
    let mut sum = 0.0f64;
    for pair in pairs {
        sum += (left.vals[lview.index(pair.left)] * by_left) * (right.vals[rview.index(pair.right)] * by_right);
    }
    (sum == 0.0 || sum.is_normal()).then(|| ScaledFloat::at_offset(sum, l.scale + r.scale))
}

/// Two child columns every cell of whose level [`FloatFold::fold_cell`]
/// sums in plain floats, as they are or centred, with no float negative: a
/// sum of such products, each a normal float or zero, is a normal float or
/// zero, so no cell falls back to the scaled fold. A cell's sum stands under
/// `scale`, and is the float a narrow column under that scale holds for the
/// cell's value, bit for bit.
pub(crate) struct PlainProducts<'a> {
    left: &'a [f64],
    lview: ChildDecoder,
    right: &'a [f64],
    rview: ChildDecoder,
    /// The powers of two each side's floats are read times: 1 unless the
    /// columns are read centred.
    by: (f64, f64),
    pub(crate) scale: i64,
}

impl<'a> PlainProducts<'a> {
    /// The products of `left` and `right`, where both columns are narrow
    /// and nonnegative and their offsets fit the plain range as they are,
    /// or else centred ([`plain_fold_centred`]).
    pub(crate) fn of(left: &'a StreamChild<'_, FloatFold>, right: &'a StreamChild<'_, FloatFold>) -> Option<Self> {
        let (l, r) = (left.col.plain()?, right.col.plain()?);
        if l.at.neg || r.at.neg {
            return None;
        }
        let fits = |a: Offsets, b: Offsets| a.lo + b.lo >= PLAIN_LO && a.hi + b.hi <= PLAIN_HI;
        let (by, scale) = if fits(l.at, r.at) {
            ((1.0, 1.0), l.at.scale + r.at.scale)
        } else if (l.centre != 0 || r.centre != 0) && fits(l.centred(), r.centred()) {
            ((pow2(l.centre), pow2(r.centre)), l.centred().scale + r.centred().scale)
        } else {
            return None;
        };
        Some(PlainProducts { left: l.vals, lview: left.view, right: r.vals, rview: right.view, by, scale })
    }

    /// The product of the left value `lc` names and the right value `rc`
    /// names, as [`plain_fold`] and [`plain_fold_centred`] form it.
    #[inline(always)]
    pub(crate) fn product(&self, lc: u32, rc: u32) -> f64 {
        let l = self.left[self.lview.index(EncodedChildRef::from_raw(lc))];
        let r = self.right[self.rview.index(EncodedChildRef::from_raw(rc))];
        (l * self.by.0) * (r * self.by.1)
    }

    /// The left factor of [`product`](Self::product) for the value `lc`
    /// names, `None` where it names no value of the column: the product is
    /// the two factors' product.
    #[inline(always)]
    pub(crate) fn left_factor(&self, lc: u32) -> Option<f64> {
        self.lview.child(EncodedChildRef::from_raw(lc)).index().and_then(|i| self.left.get(i)).map(|&l| l * self.by.0)
    }

    /// The right factor of [`product`](Self::product), as
    /// [`left_factor`](Self::left_factor).
    #[inline(always)]
    pub(crate) fn right_factor(&self, rc: u32) -> Option<f64> {
        self.rview.child(EncodedChildRef::from_raw(rc)).index().and_then(|i| self.right.get(i)).map(|&r| r * self.by.1)
    }
}

/// The products of two narrow child columns with no float negative, each
/// float times the other as the columns hold them, standing under the sum
/// of the columns' scales: where every product of a cell is a normal float
/// or a product with a zero factor, and the cell's sum stays finite, the
/// float sum in pair order is [`FloatFold::fold`]'s value bit for bit, as
/// [`plain_fold`] argues for a level whose offsets keep every product so.
pub(crate) struct CheckedProducts<'a> {
    left: &'a [f64],
    lview: ChildDecoder,
    right: &'a [f64],
    rview: ChildDecoder,
    scale: i64,
}

impl<'a> CheckedProducts<'a> {
    /// The products of the floats `left.0` under the scale `left.1`, read
    /// through `lview`, and the floats `right.0` under `right.1`, read
    /// through `rview`: each a child column's [`plain_floats`].
    pub(crate) fn new(left: (&'a [f64], i64), lview: ChildDecoder, right: (&'a [f64], i64), rview: ChildDecoder) -> Self {
        CheckedProducts { left: left.0, lview, right: right.0, rview, scale: left.1 + right.1 }
    }

    /// The product of the left value `lc` names and the right value `rc`
    /// names, and whether the float is the product: a normal float above
    /// the least one (a product rounded up to it from below may not be),
    /// or a zero factor's.
    #[inline(always)]
    pub(crate) fn product(&self, lc: u32, rc: u32) -> (f64, bool) {
        let a = self.left[self.lview.index(EncodedChildRef::from_raw(lc))];
        let b = self.right[self.rview.index(EncodedChildRef::from_raw(rc))];
        Self::times(a, b)
    }

    /// The product of the floats `a` and `b`, and whether it is their
    /// product, as [`product`](Self::product) forms it.
    #[inline(always)]
    pub(crate) fn times(a: f64, b: f64) -> (f64, bool) {
        let p = a * b;
        (p, (p > f64::MIN_POSITIVE && p < f64::INFINITY) || a == 0.0 || b == 0.0)
    }

    /// The left float of [`product`](Self::product) for the value `lc`
    /// names, `None` where it names no value of the column.
    #[inline(always)]
    pub(crate) fn left_factor(&self, lc: u32) -> Option<f64> {
        self.lview.child(EncodedChildRef::from_raw(lc)).index().and_then(|i| self.left.get(i)).copied()
    }

    /// The right float of [`product`](Self::product), as
    /// [`left_factor`](Self::left_factor).
    #[inline(always)]
    pub(crate) fn right_factor(&self, rc: u32) -> Option<f64> {
        self.rview.child(EncodedChildRef::from_raw(rc)).index().and_then(|i| self.right.get(i)).copied()
    }

    /// The power of two a cell's float sum of such products stands under:
    /// its value is [`ScaledFloat::at_offset`] of the sum and this scale.
    #[inline(always)]
    pub(crate) fn scale(&self) -> i64 {
        self.scale
    }
}

/// A child column's values as floats `x` standing for `x · 2^scale`, none
/// negative: a narrow column's own floats, and any other column's values
/// each [`ScaledFloat::offset_from`] the middle of their exponents, written
/// into `buf`; `None` where a value is negative or lies too far from that
/// middle for a normal float.
pub(crate) fn plain_floats<'b>(
    eng: &Engine,
    child: &'b StreamChild<'_, FloatFold>,
    buf: &'b mut Vec<f64>,
) -> Result<Option<(&'b [f64], i64)>, OperationError> {
    let col = &child.col;
    if let Some(p) = col.plain() {
        return Ok((!p.at.neg).then_some((p.vals, p.at.scale)));
    }
    let n = match col {
        FloatColumn::Folded(FloatSlice::Wide(vals)) => vals.len(),
        FloatColumn::Stored(vals) => vals.len(),
        FloatColumn::Leaf(vals) => vals.len(),
        _ => return Ok(None),
    };
    let (mut lo, mut hi) = (i64::MAX, i64::MIN);
    for i in 0..n {
        let v = col.get(i);
        if v.mantissa().is_sign_negative() {
            return Ok(None);
        }
        if !v.is_zero() {
            (lo, hi) = (lo.min(v.exponent()), hi.max(v.exponent()));
        }
    }
    let scale = if lo <= hi { lo / 2 + hi / 2 } else { 0 };
    buf.clear();
    eng.limits().reserve(buf, n)?;
    for i in 0..n {
        match col.get(i).offset_from(scale) {
            Some(x) => buf.push(x),
            None => return Ok(None),
        }
    }
    Ok(Some((&buf[..], scale)))
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
        if let Some(v) = plain_fold_centred(pairs, left, right) {
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
