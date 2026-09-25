//! The weighted arm of the streaming fold.

use super::*;
use crate::diagram::{ChildPair, ChildDecoder, TddLevel, WeightStore, WeightValue};
use crate::value::{WeightFold};

/// Weighted analogue of [`super::count::compute_cell_count`]. `Σ left[idx(p.left)] * right[idx(p.right)]`.
/// No overflow handling.
pub(crate) fn compute_cell_weight(
    pairs: &[ChildPair],
    left: &[WeightValue],
    right: &[WeightValue],
    left_is_marginal: bool,
    right_is_marginal: bool,
    ws: &WeightStore,
) -> WeightValue {
    let left_view = if left_is_marginal { ChildDecoder::marginal() } else { ChildDecoder::structural() };
    let right_view = if right_is_marginal { ChildDecoder::marginal() } else { ChildDecoder::structural() };
    WeightFold::fold(
        pairs.iter().copied(),
        |k| std::borrow::Cow::Borrowed(&left[left_view.index(k)]),
        |k| std::borrow::Cow::Borrowed(&right[right_view.index(k)]),
        ws.wzero(),
    )
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
        let cols = crate::value::read::LevelColumns::new(store, levels);
        WeightFold::fold(
            levels[at.lvl].pairs_iter_of_idx(i),
            |k| crate::value::read::read_weight(at.left, k, vtree, &cols, at.computed),
            |k| crate::value::read::read_weight(at.right, k, vtree, &cols, at.computed),
            store.wzero(),
        )
    }

    fn child_view<'a>(
        left_idx: usize,
        vtree: &crate::vtree::Vtree,
        level: &'a TddLevel,
        computed: &'a [Option<Vec<WeightValue>>],
        store: &'a WeightStore,
    ) -> StreamChild<'a, WeightFold> {
        if let Some(col) = crate::value::read::column_of(store, level, left_idx) {
            return StreamChild { col: std::borrow::Cow::Borrowed(col), is_marginal: true };
        }
        if let crate::vtree::VtreeNode::Leaf { var, .. } = *vtree.node(VtreeIdx(left_idx as u32)) {
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
        let col = computed[left_idx]
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
        compute_cell_weight(pairs, &left.col, &right.col, left.is_marginal, right.is_marginal, store)
    }

}


#[cfg(test)]
#[path = "tests/weight/mod.rs"]
mod tests;
