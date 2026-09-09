//! The weighted arm of the streaming fold.

use super::*;
use crate::diagram::{ValueRef};
use crate::diagram::MargSide;

// ── Weighted payload (algebraic model counting) ──────────────────────────────
//
// The weighted hooks swap the `u128`/`BigUint` model-count payload for an exact
// `BigRational` semiring value carried in the external `WeightStore` (installed
// thread-local for the duration of the compile). BigRational doesn't overflow,
// so there is NO big/overflow second pass — a single clean fold. The per-child
// value lookup is the weighted analogue of `read_level_count`: read from the
// `WeightStore` for a weight-marginal child, the semiring leaf base for a leaf,
// else the per-batch `computed` scratch.

/// Weighted analogue of [`read_level_count`] /
/// `marginal::store::read_marginal_weight`, operating on the in-flight
/// `levels` slice. Resolves a child node's exact semiring value. Returns
/// `Cow`: the per-batch `computed` read borrows (no clone); store slots and
/// leaf bases clone.
#[inline]
pub(crate) fn read_level_weight<'a>(
    child: usize,
    node_ref: usize,
    vtree: &crate::vtree::Vtree,
    levels: &[TddLevel],
    computed_weights: &'a [Option<Vec<WeightVal>>],
    ws: &WeightStore,
) -> std::borrow::Cow<'a, WeightVal> {
    if levels[child].is_weight_marginal() {
        let slot = match ValueRef::from_raw(MargSide(node_ref as u32)) {
            ValueRef::Inline(_) => unreachable!("weighted marg-side refs are bare slots"),
            ValueRef::Slot(s) => s as usize,
        };
        return std::borrow::Cow::Owned(
            ws.level(child).expect("weight-marginal level set")[slot].clone(),
        );
    }
    if let crate::vtree::VtreeNode::Leaf { var, .. } = *vtree.node(VtreeIdx(child as u32)) {
        return std::borrow::Cow::Owned(ws.leaf_val(var, LeafLabel::from_idx(node_ref)));
    }
    if let Some(w) = &computed_weights[child] {
        return std::borrow::Cow::Borrowed(&w[node_ref]);
    }
    unreachable!("weighted value not available for level {}", child);
}

/// Weighted analogue of [`compute_cell_count`]. `Σ left[idx(p.left)] * right[idx(p.right)]`.
/// No overflow handling.
pub(crate) fn compute_cell_weight(
    pairs: &[InputPair],
    left: &[WeightVal],
    right: &[WeightVal],
    left_is_marg: bool,
    right_is_marg: bool,
    ws: &WeightStore,
) -> WeightVal {
    // Resolve a marg/non-marg ref to its value; both index the snapshot by
    // reference.
    #[inline(always)]
    fn resolve<'a>(raw: u32, is_marg: bool, snap: &'a [WeightVal]) -> std::borrow::Cow<'a, WeightVal> {
        if is_marg {
            match ValueRef::from_raw(MargSide(raw)) {
                ValueRef::Inline(_) => unreachable!("weighted marg-side refs are bare slots"),
                ValueRef::Slot(s) => std::borrow::Cow::Borrowed(&snap[s as usize]),
            }
        } else {
            std::borrow::Cow::Borrowed(&snap[raw as usize])
        }
    }
    WeightFold::fold(
        pairs.iter().copied(),
        |k| resolve(k as u32, left_is_marg, left),
        |k| resolve(k as u32, right_is_marg, right),
        ws.wzero(),
    )
}

impl StreamPayload for WeightFold {
    /// `Cow`, not a plain borrow: the `computed_weights` scratch and nothing
    /// else can lend a reference. The `WeightStore` column must be copied out
    /// from under the output level's `&mut`, and the semiring leaf bases are
    /// computed on the spot, so those two arms own.
    type ChildCol<'a> = std::borrow::Cow<'a, [WeightVal]>;

    fn zero(ws: Option<&WeightStore>) -> WeightVal {
        ws.expect("weighted apply without a weight store").wzero()
    }

    #[inline]
    fn fold_node(
        lvl: usize,
        i: usize,
        l_i: usize,
        r_i: usize,
        vtree: &crate::vtree::Vtree,
        levels: &[TddLevel],
        computed: &[Option<Vec<WeightVal>>],
        zero: &WeightVal,
        ws: Option<&WeightStore>,
    ) -> WeightVal {
        let ws = ws.expect("weighted apply without a weight store");
        WeightFold::fold(
            levels[lvl].pairs_of_idx(i).iter().copied(),
            |k| read_level_weight(l_i, k, vtree, levels, computed, ws),
            |k| read_level_weight(r_i, k, vtree, levels, computed, ws),
            zero.clone(),
        )
    }

    fn child_view<'a>(
        eng: &Engine,
        li: usize,
        vtree: &crate::vtree::Vtree,
        level: &'a TddLevel,
        computed_weights: &'a [Option<Vec<WeightVal>>],
        ws: Option<&WeightStore>,
    ) -> Result<StreamChild<'a, WeightFold>, ApplyError> {
        let ws = ws.expect("weighted apply without a weight store");
        if let Some(col) = crate::marginal::column_of(ws, level, li) {
            // Keyed on THIS level's own marginality flag, not on whether the
            // `WeightStore` happens to hold a column for this vtree
            // index — so a level that is structural HERE never decodes against
            // another `Tdd`'s values. For a weight-marginal LEAF the two agree by
            // construction: the pin invariant
            // (`marginalize::marginalize_leaf_weighted`) keeps its column equal,
            // slot for slot, to the label-ordered `leaf_val` triple the structural
            // branch below builds.
            //
            // The ONE column that must still be copied: the store is held apart
            // from the level slice for the whole apply, so its column cannot be
            // lent alongside the output level's `&mut`. Fallible for the same
            // reason the integer path used to be.
            let col = try_clone_counts(eng, col)?;
            return Ok(StreamChild { col: std::borrow::Cow::Owned(col), is_marg: true });
        }
        if let crate::vtree::VtreeNode::Leaf { var, .. } = *vtree.node(VtreeIdx(li as u32)) {
            // LEAF_WIDTH = 3, ordered {One, Pos, Neg} per LeafLabel::from_idx —
            // weighted analogue of `IntFold::child_view`'s `LEAF_COUNTS`, but
            // resolving the semiring leaf bases rather than fixed counts. Built by
            // `marginalize::leaf_column_vals`, the ONE definition of that triple
            // (the same one `marginalize_leaf_weighted` pins into the store), so
            // the structural and marginal branches cannot drift apart. Fixed
            // 3-element alloc, so no budget reservation (the bases are not
            // `const`, hence no static to borrow as the integer twin does).
            let col: Vec<WeightVal> =
                crate::marginal::leaf_column_vals(ws, var);
            return Ok(StreamChild { col: std::borrow::Cow::Owned(col), is_marg: false });
        }
        let col = computed_weights[li]
            .as_ref()
            .expect("WeightFold::child_view: no values for level");
        Ok(StreamChild { col: std::borrow::Cow::Borrowed(col), is_marg: false })
    }

    #[inline(always)]
    fn fold_cell(
        pairs: &[InputPair],
        left: &StreamChild<'_, WeightFold>,
        right: &StreamChild<'_, WeightFold>,
        ws: Option<&WeightStore>,
    ) -> WeightVal {
        compute_cell_weight(
            pairs,
            &left.col,
            &right.col,
            left.is_marg,
            right.is_marg,
            ws.expect("weighted apply without a weight store"),
        )
    }

    #[inline]
    fn store_level(
        levels: &mut [TddLevel],
        li: usize,
        col: Vec<WeightVal>,
        ws: Option<&mut WeightStore>,
    ) {
        // No parent contract-dirty marking: the shared level-state machine
        // establishes C3 at slot-prune, which runs in weighted mode too via
        // `prune_marg_slots_generic::<WeightFold>`; only the integer
        // count-preservation localizer around it is gated off.
        crate::marginal::install_weight_column(
            levels, li, col,
            ws.expect("weighted apply without a weight store"),
        );
    }
}

#[cfg(test)]
#[path = "../stream_overflow_validation_tests.rs"]
mod overflow_validation_tests;
