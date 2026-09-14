//! The weighted arm of the streaming fold.

use super::*;
use crate::diagram::{ChildPair, ValueRef, TddLevel, WeightStore, WeightValue};
use crate::value::{WeightFold};
use crate::diagram::MarginalSide;

// ── Weighted payload (algebraic model counting) ──────────────────────────────
//
// The weighted hooks swap the `u128`/`BigUint` model-count payload for an exact
// `BigRational` semiring value carried in the external `WeightStore`, which is
// attached to the diagram for the duration of the compile. BigRational doesn't overflow,
// so there is no big/overflow second pass — a single clean fold. The per-child
// value lookup is the shared `marginal::read_weight`: read from the
// `WeightStore` for a weight-marginal child, the semiring leaf base for a leaf,
// else the per-batch `computed` scratch.

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
    // Resolve a marginal/non-marginal ref to its value; both index the snapshot by
    // reference.
    #[inline(always)]
    fn resolve<'a>(raw: u32, is_marginal: bool, snap: &'a [WeightValue]) -> std::borrow::Cow<'a, WeightValue> {
        if is_marginal {
            match ValueRef::from_raw(MarginalSide(raw)) {
                ValueRef::Inline(_) => unreachable!("weighted marginal-side refs are bare slots"),
                ValueRef::Slot(s) => std::borrow::Cow::Borrowed(&snap[s as usize]),
            }
        } else {
            std::borrow::Cow::Borrowed(&snap[raw as usize])
        }
    }
    WeightFold::fold(
        pairs.iter().copied(),
        |k| resolve(k.raw(), left_is_marginal, left),
        |k| resolve(k.raw(), right_is_marginal, right),
        ws.wzero(),
    )
}

impl ValueDomain for WeightFold {
    /// The external store the marginal columns live in.
    type Store = WeightStore;

    /// `Cow`, not a plain borrow: the `computed` scratch and nothing else can
    /// lend a reference. The `WeightStore` column must be copied out from
    /// under the output level's `&mut`, and the semiring leaf bases are
    /// computed on the spot, so those two arms own.
    type ChildCol<'a> = std::borrow::Cow<'a, [WeightValue]>;

    fn zero(store: &WeightStore) -> WeightValue {
        store.wzero()
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
            at.zero.clone(),
        )
    }

    fn child_view<'a>(
        eng: &Engine,
        left_idx: usize,
        vtree: &crate::vtree::Vtree,
        level: &'a TddLevel,
        computed: &'a [Option<Vec<WeightValue>>],
        store: &WeightStore,
    ) -> Result<StreamChild<'a, WeightFold>, OperationError> {
        if let Some(col) = crate::value::read::column_of(store, level, left_idx) {
            // `column_of` keys on the level's own marginality flag, so a
            // structural level never decodes against a column the store
            // happens to hold at this index. The column is copied because the
            // store is held apart from the level slice for the whole apply and
            // cannot be lent beside the output level's `&mut`; the copy
            // reserves through the budget.
            let mut owned = Vec::new();
            eng.limits().reserve_exact(&mut owned, col.len())?;
            owned.extend_from_slice(col);
            return Ok(StreamChild { col: std::borrow::Cow::Owned(owned), is_marginal: true });
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
            return Ok(StreamChild { col: std::borrow::Cow::Owned(col), is_marginal: false });
        }
        let col = computed[left_idx]
            .as_ref()
            .expect("WeightFold::child_view: no values for level");
        Ok(StreamChild { col: std::borrow::Cow::Borrowed(col), is_marginal: false })
    }

    #[inline(always)]
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
mod tests;
