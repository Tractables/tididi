//! Choosing how a level is walked and how its output arrays grow.

use super::*;

/// The bottom-up sweep's level walk: the tuned lazy `internal_bottomup` iterator
/// (the byte-identical main-compile order) or the spine-bounded apply's
/// restricted level list. A two-variant enum, not `Box<dyn Iterator>` — that box
/// cost one heap allocation per apply and an indirect `next()` per level, while
/// the loop body below is ONE loop either way.
pub(super) enum LevelWalk<'a, I> {
    Depth(I),
    /// MergeScope-bounded apply: `R` in `topo_pos` order (the `Depth` order with the
    /// levels that would take an identity fast path removed).
    Restricted(std::slice::Iter<'a, VtreeIdx>, &'a crate::vtree::Vtree),
}

impl<'a, I: Iterator<Item = (VtreeIdx, VtreeIdx, VtreeIdx)>> Iterator for LevelWalk<'a, I> {
    type Item = (VtreeIdx, VtreeIdx, VtreeIdx);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            LevelWalk::Depth(it) => it.next(),
            LevelWalk::Restricted(it, vtree) => it.next().map(|&t| {
                let (l, r) = vtree.children(t);
                (t, l, r)
            }),
        }
    }
}

/// Conservative per-cell byte factor for the apply's product grid:
/// pairs (8B) + nodes (8B) + scratch (4–8B) ≈ 24B. Shared by the in-apply
/// predictive budget check (above) and the pre-apply size gate so both agree
/// on the byte conversion.
pub(crate) const APPLY_BYTES_PER_CELL: u64 = 24;

/// Byte ceiling on each of the two per-level exact reserves (`level.nodes` and
/// `level.pairs`, both sized from that level's exact upper bound).
///
/// The bounds — `k1 × right_width` live cells, `|f.pairs| × |g.pairs|` emitted pairs —
/// are exact but loose: most levels have low survival, so an uncapped reserve
/// would routinely grab orders of magnitude more than the level ends up using
/// (and charge every byte of it to the soft budget). Capping bounds the
/// over-allocation per level; a level that outgrows the cap keeps growing
/// through the ordinary fallible push path, and `shrink_arrays` at
/// `finalize_level` hands the unused tail back (it shrinks at cap > 4 × len).
/// ONE definition — the two element-count caps below derive from it.
pub(super) const LEVEL_RESERVE_CAP_BYTES: usize = 64 * 1024;

/// [`LEVEL_RESERVE_CAP_BYTES`] in `TddNodeData`s — the `level.nodes` arm.
pub(super) const LEVEL_RESERVE_NODES_CAP: usize =
    LEVEL_RESERVE_CAP_BYTES / std::mem::size_of::<TddNodeData>();

/// [`LEVEL_RESERVE_CAP_BYTES`] in `InputPair`s — the `level.pairs` arm.
pub(super) const LEVEL_RESERVE_PAIRS_CAP: usize =
    LEVEL_RESERVE_CAP_BYTES / std::mem::size_of::<InputPair>();


/// How one level of the product is built.
///
/// Every per-level boolean the driver used to carry — `use_sparse`,
/// `use_sparse_marg`, `marg_child_dispatch`, `marg_stream_collapse`,
/// `plain_dense` — collapses into this one decision, taken once by
/// [`route_level`] before any of the level's storage is touched. The level
/// loop and the row loop dispatch on it and on nothing else, so a route that
/// is unreachable is unreachable in one place instead of five.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Route {
    /// Scatter-filter-dedup over the children's live products only, never
    /// materializing the `k1 × right_width` grid.
    Sparse,
    /// Exactly one marginal child and a structural output: drive the build
    /// from the structural sibling into a reused `right_width`-row scratch and record
    /// the survivors in a product list, leaving the level tagged sparse for
    /// the grandparent to densify.
    SparseMarg,
    /// A streaming marginalize target: fold `Σ left × right` per cell straight
    /// into the output column, never materializing a product node.
    /// `marg_children` picks the child lookup — marginal sides are read
    /// through `MargLookup`, structural ones positionally.
    Stream { marg_children: bool },
    /// At least one marginal child, materializing output: the shared cell
    /// kernel over the dense grid with `MargLookup` sides.
    MargChild,
    /// No marginal child, no streaming, no NxM masks and no pass-through side:
    /// the dense grid with positional child lookups.
    PlainDense,
    /// The dense grid in its general form — NxM liveness masks, or a
    /// pass-through side to carry across.
    Dense,
}

/// The level's marginality, in the two different senses the routes need.
#[derive(Clone, Copy)]
pub(super) struct LevelMarg {
    /// Left child marginal in the OUTPUT level — or, under a restriction, in
    /// the accumulator's own level, which is where an off-`R` output level
    /// lives until the tail merges it back. What the row-loop dispatch reads:
    /// it decides whether the cell kernel must go through `MargLookup`.
    pub(super) left_now: bool,
    /// Right child, same sense as [`LevelMarg::left_now`].
    pub(super) right_now: bool,
    /// Left child marginal in the output level or in EITHER operand. The
    /// sparse gates read this wider test: the sparse reverse index buckets by
    /// a decoded pair ref, which is not a per-node key once any side of the
    /// level carries count payloads rather than node indices.
    pub(super) left_any: bool,
    /// Right child, same sense as [`LevelMarg::left_any`].
    pub(super) right_any: bool,
    /// The schedule names this level as one to sum out.
    pub(super) is_target: bool,
    /// …and the streaming collapse is armed for it, so the fold can replace
    /// the product construction entirely.
    pub(super) stream_eligible: bool,
}

/// Whether the sparse routes are available at all, and whether this level's
/// children are sparse enough for the scatter walk to beat the grid.
#[derive(Clone, Copy)]
pub(super) struct SparseGate {
    /// The apply is running the bump allocator and product lists the sparse
    /// routes need. Fixed for the whole apply.
    pub(super) available: bool,
    /// The online density check: the children's live product counts are small
    /// enough against their maxima that scattering wins.
    pub(super) density_wins: bool,
    /// Grids at or below this many cells are not worth either sparse route's
    /// setup, whatever the density says.
    pub(super) min_grid: usize,
}

/// Pick this level's [`Route`].
///
/// Pure: it reads the level's shape, its marg plan and the two gates, and
/// touches no storage. Order matters — the sparse routes are checked first
/// because they avoid materializing the grid the dense routes need, and
/// streaming is checked before the marginal-child dense build because a
/// streaming target has no downstream structure to build.
pub(super) fn route_level(
    shape: LevelShape,
    plan: &MargPlan,
    marg: &LevelMarg,
    sparse: SparseGate,
) -> Route {
    let big_grid = shape.k1 * shape.right_width > sparse.min_grid;

    // A marginal child on either side rules the scatter walk out entirely, so
    // the density check never has to hold for a level with count payloads.
    if sparse.available && !(marg.left_any || marg.right_any) && big_grid && sparse.density_wins {
        return Route::Sparse;
    }
    // Exactly one marginal child, and not a target: the marginal side is a
    // pass-through carrier that kills no pair, so the structural sibling alone
    // governs survival and the dense slab would be mostly dead. Targets do
    // occur with one marginal child and are common — they fall through to the
    // streaming fold below.
    if sparse.available && (marg.left_any ^ marg.right_any) && !marg.is_target && big_grid {
        return Route::SparseMarg;
    }
    if marg.stream_eligible {
        return Route::Stream { marg_children: marg.left_now || marg.right_now };
    }
    if marg.left_now || marg.right_now {
        return Route::MargChild;
    }
    if plan.both_multi_pair || plan.sides.left.is_passthrough() || plan.sides.right.is_passthrough() {
        return Route::Dense;
    }
    Route::PlainDense
}

impl Route {
    /// Fault when the level this route was chosen for is not a legal level to
    /// build at all.
    ///
    /// Two invariants, both always on — each is a handful of bool reads against
    /// the product construction that follows.
    ///
    /// **The marginalize schedule.** If either operand's level `t` is marginal
    /// (its pair structure replaced by model counts), an identity fast path
    /// must already have consumed the level: a marginal level conjoins soundly
    /// only with an identity, non-constraining counterpart. Reaching a build
    /// route with a marginal level therefore means the OTHER operand still
    /// constrains node `t` — a variable was summed out of one operand while
    /// still live in the other. The dense build would dereference `nodes[i]`
    /// on an empty vector or silently miscount, so fault instead.
    ///
    /// The violation is the ASYMMETRIC case only. Both operands marginal means
    /// both summed out the same scope, which is sound; marginal against
    /// identity is what the fast paths consume.
    ///
    /// **No marginal leakage onto the structural routes.** Every parent of a
    /// marginal child routes to [`Route::MargChild`], [`Route::Stream`] or
    /// [`Route::SparseMarg`], so [`Route::Dense`] and [`Route::PlainDense`]
    /// must never see a non-empty marginal child: their cell kernel indexes
    /// the output child grids and has no decode for a parent's inline marg
    /// refs. A `width() == 0` marginal child is exempt and legitimate — it has
    /// no cells, so the parent holds no refs into it, and the apply's
    /// streaming commit and leaf marginalization both mint such levels.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn validate(
        self,
        f: &Tdd,
        g: &Tdd,
        shape: LevelShape,
        marg: &LevelMarg,
        c1_identity: &[bool],
        c2_identity: &[bool],
        c1_widths: &[usize],
        c2_widths: &[usize],
        #[cfg_attr(not(debug_assertions), allow(unused_variables))]
        vtree: &crate::vtree::Vtree,
    ) {
        let LevelShape { t, left, right, left_idx, right_idx, .. } = shape;
        let c1_marg = f.level(t).is_marginal();
        let c2_marg = g.level(t).is_marginal();
        let c1_identity_at_t = c1_identity[left_idx] && c1_identity[right_idx];
        let c2_identity_at_t = c2_identity[left_idx] && c2_identity[right_idx];
        let violation = (c1_marg && !c2_marg && !c2_identity_at_t)
            || (c2_marg && !c1_marg && !c1_identity_at_t);
        if violation {
            // The rich subtree dump builds a large string and writes a file, so
            // it is debug-only; the panic below always fires.
            #[cfg(debug_assertions)]
            debug_assert_marg_schedule(
                f, g, t, left, right, vtree,
                shape.k1, shape.right_width, left_idx, right_idx,
                c1_widths, c2_widths, c1_identity, c2_identity,
            );
            panic!(
                "apply_and marginalize-schedule violation at vtree node {t:?} \
                 (left={left:?} right={right:?}): one operand marginalized this node \
                 while the other still constrains it \
                 (f.marg={c1_marg}, g.marg={c2_marg}, c1_id[L,R]={},{}, c2_id[L,R]={},{}). \
                 A variable was summed out of one operand while still live in the \
                 other — a marginalize-schedule bug. This conjoin is invalid and \
                 would corrupt the model count.",
                c1_identity[left_idx],
                c1_identity[right_idx],
                c2_identity[left_idx],
                c2_identity[right_idx],
            );
        }

        if matches!(self, Route::Dense | Route::PlainDense) {
            let marg_wide = |lvl: &TddLevel| lvl.is_marginal() && lvl.width() > 0;
            let (t_idx, left_idx, right_idx) = (shape.t_idx, shape.left_idx, shape.right_idx);
            cheap_assert!(
                !marg.left_now && !marg.right_now
                    && !marg_wide(&f.levels[left_idx]) && !marg_wide(&f.levels[right_idx])
                    && !marg_wide(&g.levels[left_idx]) && !marg_wide(&g.levels[right_idx]),
                "a structural product-grid route was chosen for a level with a \
                 non-empty marginal child (t_idx={t_idx} l={left_idx} r={right_idx})"
            );
        }
    }
}
