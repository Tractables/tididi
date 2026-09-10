//! The weighted arm of the streaming fold.

use super::*;
use crate::diagram::{ValueRef};
use crate::diagram::MarginalSide;

// ── Weighted payload (algebraic model counting) ──────────────────────────────
//
// The weighted hooks swap the `u128`/`BigUint` model-count payload for an exact
// `BigRational` semiring value carried in the external `WeightStore` (installed
// thread-local for the duration of the compile). BigRational doesn't overflow,
// so there is no big/overflow second pass — a single clean fold. The per-child
// value lookup is the shared `marginal::read_weight`: read from the
// `WeightStore` for a weight-marginal child, the semiring leaf base for a leaf,
// else the per-batch `computed` scratch.


/// Weighted analogue of [`compute_cell_count`]. `Σ left[idx(p.left)] * right[idx(p.right)]`.
/// No overflow handling.
pub(crate) fn compute_cell_weight(
    pairs: &[InputPair],
    left: &[WeightVal],
    right: &[WeightVal],
    left_is_marginal: bool,
    right_is_marginal: bool,
    ws: &WeightStore,
) -> WeightVal {
    // Resolve a marginal/non-marginal ref to its value; both index the snapshot by
    // reference.
    #[inline(always)]
    fn resolve<'a>(raw: u32, is_marginal: bool, snap: &'a [WeightVal]) -> std::borrow::Cow<'a, WeightVal> {
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
        |k| resolve(k as u32, left_is_marginal, left),
        |k| resolve(k as u32, right_is_marginal, right),
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
    type ChildCol<'a> = std::borrow::Cow<'a, [WeightVal]>;

    fn zero(store: &WeightStore) -> WeightVal {
        store.wzero()
    }

    fn weight_store(store: &mut WeightStore) -> Option<&mut WeightStore> {
        Some(store)
    }

    #[inline]
    fn stream_columns(cache: &StreamCache) -> &[Option<Vec<WeightVal>>] {
        cache.weighted()
    }

    #[inline]
    fn store_of(ws: Option<&WeightStore>) -> &WeightStore {
        ws.expect("a weighted column is only ever built with a store attached")
    }

    #[inline]
    fn fold_node<R: ReservePolicy>(
        lvl: usize,
        i: usize,
        l_i: usize,
        r_i: usize,
        vtree: &crate::vtree::Vtree,
        levels: &[TddLevel],
        computed: &[Option<Vec<WeightVal>>],
        zero: &WeightVal,
        store: &WeightStore,
    ) -> WeightVal {
        let cols = crate::marginal::LevelColumns::new(store, levels);
        WeightFold::fold(
            levels[lvl].pairs_iter_of_idx(i),
            |k| crate::marginal::read_weight(l_i, k, vtree, &cols, computed),
            |k| crate::marginal::read_weight(r_i, k, vtree, &cols, computed),
            zero.clone(),
        )
    }

    fn child_view<'a, R: ReservePolicy>(
        eng: &Engine,
        left_idx: usize,
        vtree: &crate::vtree::Vtree,
        level: &'a TddLevel,
        computed: &'a [Option<Vec<WeightVal>>],
        store: &WeightStore,
    ) -> Result<StreamChild<'a, WeightFold>, ApplyError> {
        if let Some(col) = crate::marginal::column_of(store, level, left_idx) {
            // Keyed on THIS level's own marginality flag, not on whether the
            // `WeightStore` happens to hold a column for this vtree
            // index — so a level that is structural HERE never decodes against
            // another `Tdd`'s values. For a weight-marginal LEAF the two agree by
            // construction: the pin invariant
            // (`marginal::marginalize_leaf_weighted`) keeps its column equal,
            // slot for slot, to the label-ordered `leaf_val` triple the structural
            // branch below builds.
            //
            // The one column that must still be copied: the store is held apart
            // from the level slice for the whole apply, so its column cannot be
            // lent alongside the output level's `&mut`. The clone is fallible
            // because it charges the budget.
            let col = try_clone_counts(eng, col)?;
            return Ok(StreamChild { col: std::borrow::Cow::Owned(col), is_marginal: true });
        }
        if let crate::vtree::VtreeNode::Leaf { var, .. } = *vtree.node(VtreeIdx(left_idx as u32)) {
            // LEAF_WIDTH = 3, ordered {One, Pos, Neg} per LeafLabel::from_idx —
            // weighted analogue of `IntFold::child_view`'s `LEAF_COUNTS`, but
            // resolving the semiring leaf bases rather than fixed counts. Built by
            // `marginal::leaf_column_vals`, the one definition of that triple
            // (the same one `marginalize_leaf_weighted` pins into the store), so
            // the structural and marginal branches cannot drift apart. Fixed
            // 3-element alloc, so no budget reservation (the bases are not
            // `const`, hence no static to borrow as the integer twin does).
            let col: Vec<WeightVal> = crate::marginal::leaf_column_vals(store, var);
            return Ok(StreamChild { col: std::borrow::Cow::Owned(col), is_marginal: false });
        }
        let col = computed[left_idx]
            .as_ref()
            .expect("WeightFold::child_view: no values for level");
        Ok(StreamChild { col: std::borrow::Cow::Borrowed(col), is_marginal: false })
    }

    #[inline(always)]
    fn fold_cell(
        pairs: &[InputPair],
        left: &StreamChild<'_, WeightFold>,
        right: &StreamChild<'_, WeightFold>,
        store: &WeightStore,
    ) -> WeightVal {
        compute_cell_weight(pairs, &left.col, &right.col, left.is_marginal, right.is_marginal, store)
    }

    #[inline]
    fn commit_in_flight<R: ReservePolicy>(
        levels: &mut [TddLevel],
        left_idx: usize,
        col: Vec<WeightVal>,
        store: &mut WeightStore,
    ) {
        // No parent contract-dirty marking: the shared level-state machine
        // establishes invariant 10 at slot-prune, which runs in weighted mode too via
        // `prune_marginal_slots_generic::<WeightFold>`; only the integer
        // count-preservation localizer around it is gated off.
        crate::marginal::install_weight_column(levels, left_idx, col, store);
    }

    /// The weighted store is full width and its references stay bare slots
    /// (slot index == node index), so nothing is minted and the parent's
    /// references need no rewrite.
    fn install(
        tdd: &mut Tdd,
        t: InternalLevel,
        col: Vec<WeightVal>,
        store: &mut WeightStore,
    ) -> Option<Vec<u32>> {
        // The column is full width — one slot per node, tombstones included —
        // which is the slot count the level records.
        crate::marginal::install_weight_column(&mut tdd.levels, t.vtree_idx().idx(), col, store);
        None
    }

    fn sum_out_leaf(
        eng: &Engine,
        tdd: &mut Tdd,
        leaf: VtreeIdx,
        vtree: &crate::vtree::Vtree,
        store: &mut WeightStore,
    ) {
        crate::marginal::marginalize_leaf_weighted(eng, tdd, leaf, vtree, store);
    }

    /// Nothing: weighted marginal-side references are bare slots end to end, so
    /// there is no tag to apply and no snapshot to key it off.
    fn end_sweep(_tdd: &mut Tdd, _was_marginal: &[bool]) {}
}

#[cfg(test)]
#[path = "../streaming_marginal_overflow_tests.rs"]
mod overflow_validation_tests;
