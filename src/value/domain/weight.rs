//! The weighted arm of the streaming fold.

use super::*;
use crate::diagram::{ChildPair, ChildDecoder, EncodedChildRef, LeafLabel, TddLevel, WeightStore, WeightValue};
use crate::vtree::{Vtree, VtreeNode};
use crate::value::{WeightFold};

/// Level `t`'s weighted column, or `None` when this diagram's level is
/// structural and its node indices are therefore not slots of any column.
/// Takes the one level rather than the whole slice for the apply, whose
/// output levels are split apart for the level it is building.
fn column_of<'a>(
    store: &'a WeightStore,
    level: &TddLevel,
    t: usize,
) -> Option<&'a [WeightValue]> {
    level.is_weight_marginal().then(|| store.level(t)).flatten()
}

/// Resolve a child node's semiring value for the weighted marginalization
/// cascade, the weighted counterpart of the integer domain's level reader.
/// Reads, in order:
///   0. a leaf level by label, never through the store column: a leaf-side ref
///      is a bare `LeafLabel` index whether the leaf is structural or
///      weight-marginal (the pinned column is installed in label order by
///      `diagram::leaf_column_vals`),
///      and the shared store may hold a column at this index that belongs to
///      another diagram;
///   1. the [`WeightStore`] column of a level this diagram has already
///      weight-marginalized; marginal-side refs are bare slots in weighted mode;
///   2. the per-batch `computed_weights` buffer for a level computed earlier in
///      this batch but not yet stored.
///
/// Store-slot and per-batch reads borrow; only the `ZERO` sentinel and leaf
/// bases materialize an owned value.
fn read_weight<'a>(
    level_idx: usize,
    side: EncodedChildRef,
    vtree: &Vtree,
    ws: &'a WeightStore,
    levels: &[TddLevel],
    computed_weights: &'a [Option<Vec<WeightValue>>],
) -> std::borrow::Cow<'a, WeightValue> {
    if let VtreeNode::Leaf { var, .. } = *vtree.node(VtreeIdx(level_idx as u32)) {
        if side.is_reserved() {
            // `ZERO` sentinel, as in the integer reader. Leaf levels only ever
            // carry Pos/Neg/One, but the bit is tested before every decode.
            return std::borrow::Cow::Owned(ws.wzero());
        }
        let label_idx = ChildDecoder::marginal().index(side);
        let v = ws.leaf_val(var, LeafLabel::from_idx(label_idx));
        #[cfg(debug_assertions)]
        debug_assert!(
            leaf_column_slot_agrees(ws, level_idx, label_idx, &v),
            "weight-marginal leaf {level_idx}: pinned column disagrees with leaf_val \
             at label slot {label_idx}"
        );
        return std::borrow::Cow::Owned(v);
    }
    // Internal level: `column_of` is gated on this diagram's own marginality,
    // since the shared store can hold a column at this index installed by
    // another live diagram while this one's level is still structural, and a
    // structural level's node indices are not slots of that column.
    if let Some(values) = column_of(ws, &levels[level_idx], level_idx) {
        if side.is_reserved() {
            return std::borrow::Cow::Owned(ws.wzero());
        }
        return std::borrow::Cow::Borrowed(&values[ChildDecoder::marginal().index(side)]);
    }
    if let Some(w) = &computed_weights[level_idx] {
        return std::borrow::Cow::Borrowed(&w[ChildDecoder::structural().node(side).idx()]);
    }
    unreachable!("weighted value not available for level {}", level_idx);
}

/// Debug-only companion to the leaf branch of [`read_weight`]: true unless a
/// pinned leaf column is installed at `level_idx` and its slot `label_idx`
/// differs from `expect`, or the column has other than three slots.
#[cfg(debug_assertions)]
fn leaf_column_slot_agrees(
    ws: &WeightStore,
    level_idx: usize,
    label_idx: usize,
    expect: &WeightValue,
) -> bool {
    use crate::diagram::weight_key;
    let Some(col) = ws.level(level_idx) else { return true };
    if col.len() != crate::diagram::LEAF_WIDTH {
        return false; // some pass compacted, erased or appended to the column
    }
    weight_key(&col[label_idx]) == weight_key(expect)
}

impl ValueDomain for WeightFold {
    /// The external store the marginal columns live in.
    type Store = WeightStore;

    /// Borrow stored columns; own the three computed leaf values.
    type ChildCol<'a> = std::borrow::Cow<'a, [WeightValue]>;

    type Scalar = WeightValue;
    type Col = Vec<WeightValue>;

    fn alloc_col(
        eng: &Engine,
        width: usize,
        store: &WeightStore,
    ) -> Result<Vec<WeightValue>, OperationError> {
        let mut v: Vec<WeightValue> = Vec::new();
        eng.limits().reserve_exact(&mut v, width)?;
        v.resize(width, store.wzero());
        Ok(v)
    }

    fn set_col(
        _eng: &Engine,
        col: &mut Vec<WeightValue>,
        i: usize,
        v: WeightValue,
    ) -> Result<(), OperationError> {
        col[i] = v;
        Ok(())
    }

    fn try_with_capacity(
        eng: &Engine,
        cap: usize,
    ) -> Result<Vec<WeightValue>, OperationError> {
        let mut col = Vec::new();
        eng.limits().reserve_exact(&mut col, cap)?;
        Ok(col)
    }

    fn push_col(
        eng: &Engine,
        col: &mut Vec<WeightValue>,
        v: WeightValue,
    ) -> Result<(), OperationError> {
        eng.limits().reserve(col, 1)?;
        col.push(v);
        Ok(())
    }

    fn col_len(col: &Vec<WeightValue>) -> usize {
        col.len()
    }

    #[inline]
    fn stream_columns(cache: &StreamCache) -> &[Option<Vec<WeightValue>>] {
        cache.weighted()
    }

    #[inline]
    fn store_of(ws: Option<&WeightStore>) -> &WeightStore {
        ws.expect("a weighted column is only ever built with a store attached")
    }

    #[inline]
    fn fold_node(at: &FoldScope<'_, WeightFold>, i: usize) -> WeightValue {
        let FoldInput { vtree, levels, store } = at.input;
        WeightFold::fold(
            levels[at.lvl].pairs_iter_of_idx(i),
            |k| read_weight(at.left, k, vtree, store, levels, at.computed),
            |k| read_weight(at.right, k, vtree, store, levels, at.computed),
            store.wzero(),
        )
    }

    fn child_view<'a>(
        level_idx: usize,
        vtree: &crate::vtree::Vtree,
        level: &'a TddLevel,
        computed: &'a [Option<Vec<WeightValue>>],
        store: &'a WeightStore,
    ) -> StreamChild<'a, WeightFold> {
        if let Some(col) = column_of(store, level, level_idx) {
            return StreamChild { col: std::borrow::Cow::Borrowed(col), is_marginal: true };
        }
        if let crate::vtree::VtreeNode::Leaf { var, .. } = *vtree.node(VtreeIdx(level_idx as u32)) {
            // `LEAF_WIDTH` = 3, ordered {One, Pos, Neg} per `LeafLabel::from_idx` —
            // weighted analogue of `IntFold::child_view`'s `LEAF_COUNTS`, but
            // resolving the semiring leaf bases rather than fixed counts. Built by
            // `diagram::leaf_column_vals`, the one definition of that triple
            // (the same one `marginalize_leaf_weighted` pins into the store), so
            // the structural and marginal branches cannot drift apart. Fixed
            // 3-element alloc, so no budget reservation (the bases are not
            // `const`, hence no static to borrow as the integer twin does).
            let col: Vec<WeightValue> = crate::diagram::leaf_column_vals(store, var);
            return StreamChild { col: std::borrow::Cow::Owned(col), is_marginal: false };
        }
        let col = computed[level_idx]
            .as_ref()
            .expect("WeightFold::child_view: no values for level");
        StreamChild { col: std::borrow::Cow::Borrowed(col), is_marginal: false }
    }

    fn fold_cell(
        pairs: &[ChildPair],
        left: &StreamChild<'_, WeightFold>,
        right: &StreamChild<'_, WeightFold>,
        store: &WeightStore,
    ) -> WeightValue {
        let left_view = if left.is_marginal { ChildDecoder::marginal() } else { ChildDecoder::structural() };
        let right_view = if right.is_marginal { ChildDecoder::marginal() } else { ChildDecoder::structural() };
        WeightFold::fold(
            pairs.iter().copied(),
            |k| std::borrow::Cow::Borrowed(&left.col[left_view.index(k)]),
            |k| std::borrow::Cow::Borrowed(&right.col[right_view.index(k)]),
            store.wzero(),
        )
    }

}


#[cfg(test)]
#[path = "tests/weight/mod.rs"]
mod tests;
