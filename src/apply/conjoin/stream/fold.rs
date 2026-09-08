//! The streaming fold contract and the per-level scratch it drives.

use super::*;

/// The fixed conceptual slots of a LEAF child's column, ordered
/// `{One=2, Pos=1, Neg=1}` per `LeafLabel::from_idx` (LEAF_WIDTH = 3). A
/// `static` so a leaf view borrows it rather than minting a `vec![2, 1, 1]`
/// per level.
pub(crate) static LEAF_COUNTS: [u128; 3] = [2, 1, 1];

/// Per-child READ VIEW of the child level's fold column, taken before the dense
/// scatter loop.
///
/// Borrowed, not copied: the column lives in the child level's marginal storage
/// (or the per-apply computed scratch) and is read there in place. `t` and its
/// two vtree children are distinct nodes, so the caller splits the three level
/// slots apart once (`slice::get_disjoint_mut` in the driver loop) and the
/// child views coexist with the `&mut level` borrow taken for output. Copying
/// them instead doubled a wide marginal child's storage at exactly the moment
/// streaming exists to relieve — a 536M-slot column is an 8 GiB single alloc.
///
/// Integer instantiation (`StreamChild<IntFold>`, aliased
/// [`StreamChildCounts`]): the `all_u64` certificate rides on the [`CountRef`]
/// (see `counts.rs`) — when both children certify, [`compute_cell_count`]
/// takes the widening-multiply fast path (`u64×u64→u128` is a single `mul`
/// that can never overflow the u128 product, max (2^64-1)^2 < 2^128), so the
/// heavy u128 `checked_mul` is skipped and only the running-total `checked_add`
/// guards overflow. Inline bit-30-tagged refs always decode to ≤
/// MARG_INLINE_MAX (u64), so the certificate over the column alone covers
/// every read in the cell loop.
pub(crate) struct StreamChild<'a, F: StreamPayload> {
    pub(crate) col: F::ChildCol<'a>,
    /// True iff this view is of a marginal child level (its refs are
    /// marg-side slot refs, possibly bit-30 tagged). When set, the per-cell
    /// fold decodes the ref before indexing the column. For a
    /// non-marginal/leaf child the ref is a plain node index (bit 30 may be a
    /// real high bit) — do NOT decode.
    pub(crate) is_marg: bool,
}

/// The integer instantiation, spelled out because it is the one the hot path
/// and the overflow validation tests name directly.
pub(crate) type StreamChildCounts<'a> = StreamChild<'a, IntFold>;

/// Per-level streaming state for ONE value kind, live only for the row loop:
/// both child views plus a mutable borrow of the level's output column.
///
/// The output column itself is owned by the driver loop's [`StreamLevelState`]
/// so it outlives the child borrows — the level tail retakes `&mut levels` to
/// commit it, which it could not do while a view into `levels` was alive.
pub(crate) struct StreamState<'a, F: StreamPayload> {
    pub(crate) left: StreamChild<'a, F>,
    pub(crate) right: StreamChild<'a, F>,
    pub(crate) counts: &'a mut F::Col<ApplyBudget>,
    /// The diagram's weight store while `F = WeightFold`; `None` in integer mode.
    pub(crate) ws: Option<&'a WeightStore>,
}

/// The level's in-flight output column, indexed by alive-cell position and
/// handed to [`StreamPayload::store_level`] on commit. The runtime value-kind
/// choice is made once per level in [`build_stream_state`]; everything
/// downstream of the two `match` arms (here and in
/// `cell::run_level_rows_stream_count`) is statically monomorphized — there is
/// no dynamic dispatch and no inert second payload.
///
/// Deliberately BORROW-FREE: it is carried across the whole cell-build route
/// dispatch to the commit, so it must not pin a borrow of `levels`.
pub(crate) enum StreamLevelState {
    Int(CountVec<ApplyBudget>),
    /// Weighted: exact `BigRational` semiring values carried into the
    /// external `WeightStore`.
    Weighted(Vec<WeightVal>),
}

/// The apply-side payload axis: what the ONE streaming driver needs from a
/// value kind on top of [`MargFold`]'s column contract. Five hooks (`zero`,
/// `fold_node`, `child_view`, `fold_cell`, `store_level`), each one a place
/// where the two kinds genuinely differ — see the table in the module
/// comment.
pub(crate) trait StreamPayload: MargFold + Sized {
    /// How this kind reads ONE child's column for the duration of the row loop.
    /// Borrowed wherever the storage can lend a reference ([`CountRef`] over a
    /// level's raw marginal arrays or another column; `Cow::Borrowed` over the
    /// weighted computed scratch); the weighted `WeightStore` column is copied,
    /// so that one case stays owned.
    type ChildCol<'a>;

    /// The additive identity of this value kind, resolved once per ensure
    /// walk (the weighted one reads the diagram's `WeightStore`).
    fn zero(ws: Option<&WeightStore>) -> Self::Scalar;

    /// Fold one node of `levels[lvl]`: `Σ over its pairs (left × right)`, with
    /// this kind's child readers resolving each `u32` ref against the
    /// in-flight `levels` slice and the per-batch `computed` scratch.
    ///
    /// Impls carry `#[inline]`: the pre-unification form was a closure the
    /// ensure walk monomorphized and inlined, and the walk's per-node loop must
    /// not gain a call.
    fn fold_node(
        lvl: usize,
        i: usize,
        l_i: usize,
        r_i: usize,
        vtree: &crate::vtree::Vtree,
        levels: &[TddLevel],
        computed: &[Option<Self::Col<ApplyBudget>>],
        zero: &Self::Scalar,
        ws: Option<&WeightStore>,
    ) -> Self::Scalar;

    /// Open a read view of child level `li`'s column. `level` is `levels[li]`,
    /// handed in already split off from the output level's `&mut` borrow.
    ///
    /// Fallible only for the one case that must still materialize (the
    /// weighted `WeightStore` column): that copy can be multi-GiB on extreme
    /// widths, so it routes through [`try_clone_counts`] and propagates
    /// `OverBudget` (→ v-split / conditioning-deepen recovery) instead of
    /// aborting. Every borrowing case allocates nothing and cannot fail.
    fn child_view<'a>(
        eng: &Engine,
        li: usize,
        vtree: &crate::vtree::Vtree,
        level: &'a TddLevel,
        computed: &'a [Option<Self::Col<ApplyBudget>>],
        ws: Option<&WeightStore>,
    ) -> Result<StreamChild<'a, Self>, ApplyError>;

    /// Collapse one alive cell's collected pairs to a single scalar:
    /// `Σ left[p.left] × right[p.right]`.
    fn fold_cell(
        pairs: &[InputPair],
        left: &StreamChild<'_, Self>,
        right: &StreamChild<'_, Self>,
        ws: Option<&WeightStore>,
    ) -> Self::Scalar;

    /// Commit a finished column into `levels[li]`, turning the level marginal.
    /// The caller has already checked the marginalization precondition
    /// ([`diagram::assert_can_make_marginal`]).
    fn store_level(
        levels: &mut [TddLevel],
        li: usize,
        col: Self::Col<ApplyBudget>,
        ws: Option<&mut WeightStore>,
    );
}

/// Ensure `computed[li]` is populated (or `levels[li]` is already marginal).
/// Mirrors `compile_marginalize.rs::ensure_counts` — both are the shared
/// [`ensure_fold_walk`], here with this context's per-kind readers wired in
/// through [`StreamPayload::fold_node`]. ("counts" is the historical name; the
/// walk carries weighted semiring values under `F = WeightFold` just the same.)
///
/// [`ColumnRetention::All`] is mandatory here and takes no caller knob: every
/// caller pairs this with [`cascade_marginalize_in_apply`], which `take`s the
/// column of EVERY level in the walked subtree to install it as that level's
/// marginal store. Frontier release would free exactly those columns.
pub(crate) fn ensure_level_counts<F: StreamPayload>(
    eng: &Engine,
    li: usize,
    vtree: &crate::vtree::Vtree,
    levels: &[TddLevel],
    computed: &mut [Option<F::Col<ApplyBudget>>],
    ws: Option<&WeightStore>,
) -> Result<(), ApplyError> {
    let zero = F::zero(ws);
    ensure_fold_walk::<F, ApplyBudget, _, _>(
        eng,
        li,
        vtree,
        levels,
        computed,
        &zero,
        &|i| levels[i].is_marginal(),
        &|lvl, i, l_i, r_i, computed| {
            F::fold_node(lvl, i, l_i, r_i, vtree, levels, computed, &zero, ws)
        },
        ColumnRetention::All,
    )
}

/// Cascade marginalization through every explicit non-leaf descendant of
/// `li`, bottom-up. Each level's column must already be populated in
/// `computed` (call [`ensure_level_counts`] first). Mirrors
/// the freeze cascade in `marginal::fold` but operates on the in-flight `levels`
/// slice during apply rather than a finished TDD.
///
/// Soundness: bottom-up order satisfies `assert_can_make_marginal` at each
/// call site (by the time we marginalize `li`, both children of `li` are
/// already marginal or leaves). The caller is responsible for ensuring `li`
/// itself is a sound streaming target (no future references) — within the
/// apply gate this is guaranteed because `marginalize_targets[li] == true` flags
/// any descendant we'd touch, and the schedule monotonicity (a parent's
/// streaming step is at least as late as any descendant's) covers descendants
/// that aren't in this apply call's target set but were targets of an earlier
/// sub-batch and only stayed explicit because of the width gate.
pub(crate) fn cascade_marginalize_in_apply<F: StreamPayload>(
    li: usize,
    vtree: &crate::vtree::Vtree,
    levels: &mut [TddLevel],
    computed: &mut [Option<F::Col<ApplyBudget>>],
    mut ws: Option<&mut WeightStore>,
) {
    if vtree.node(VtreeIdx(li as u32)).is_leaf() || levels[li].is_marginal() {
        return;
    }
    let (left, right) = vtree.children(VtreeIdx(li as u32));
    cascade_marginalize_in_apply::<F>(left.idx(), vtree, levels, computed, ws.as_deref_mut());
    cascade_marginalize_in_apply::<F>(right.idx(), vtree, levels, computed, ws.as_deref_mut());
    let Some(col) = computed[li].take() else {
        // No cached column: `ensure_level_counts` did not visit this branch
        // (cells structurally unreachable from the target's pair lists). Bail
        // rather than fabricate values.
        return;
    };
    diagram::assert_can_make_marginal(levels, vtree, VtreeIdx(li as u32));
    F::store_level(levels, li, col, ws);
}
