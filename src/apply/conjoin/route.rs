//! Choosing how a level is walked and how its output arrays grow.

use super::*;

/// Byte ceiling on each of the two per-level exact reserves (`level.nodes` and
/// `level.pairs`, both sized from that level's exact upper bound).
///
/// The bounds — `left_width × right_width` live cells, `|f.pairs| × |g.pairs|` emitted pairs —
/// are exact but loose: most levels have low survival, so an uncapped reserve
/// would routinely grab orders of magnitude more than the level ends up using
/// (and charge every byte of it to the soft budget). Capping bounds the
/// over-allocation per level; a level that outgrows the cap keeps growing
/// through the ordinary fallible push path, and `shrink_arrays` at
/// `finalize_level` hands the unused tail back (it shrinks at cap > 4 × len).
/// One definition — the two element-count caps below derive from it.
pub(super) const LEVEL_RESERVE_CAP_BYTES: usize = 64 * 1024;

/// [`LEVEL_RESERVE_CAP_BYTES`] in `EncodedNode`s — the `level.nodes` arm.
pub(super) const LEVEL_RESERVE_NODES_CAP: usize =
    LEVEL_RESERVE_CAP_BYTES / std::mem::size_of::<EncodedNode>();

/// [`LEVEL_RESERVE_CAP_BYTES`] in `ChildPair`s — the `level.pairs` arm.
pub(super) const LEVEL_RESERVE_PAIRS_CAP: usize =
    LEVEL_RESERVE_CAP_BYTES / std::mem::size_of::<ChildPair>();


/// How one level of the product is built.
///
/// One decision, taken once by [`route_level`] before any of the level's
/// storage is touched. The level loop and the row loop dispatch on it and on
/// nothing else, so a route that is unreachable is unreachable in one place.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Route {
    /// Scatter-filter-dedup over the children's live products only, never
    /// materializing the `left_width × right_width` grid.
    Sparse,
    /// Exactly one marginal child and a structural output: drive the build
    /// from the structural sibling into a reused `right_width`-row scratch and record
    /// the survivors in a product list, leaving the level tagged sparse for
    /// the grandparent to densify.
    SparseMarg,
    /// A streaming marginalization target: fold `Σ left × right` per cell straight
    /// into the output column, never materializing a product node.
    /// `marginal_children` picks the child lookup — marginal sides are read
    /// through `MarginalLookup`, structural ones positionally.
    Stream { marginal_children: bool },
    /// At least one marginal child or pass-through side, materializing
    /// output: the shared cell kernel over the dense grid with
    /// `MarginalLookup` sides.
    MarginalChild,
    /// No marginal child, no streaming, no dead-pair masks and no pass-through side:
    /// the dense grid with positional child lookups.
    PlainDense,
    /// No marginal child and no pass-through side, but both operands
    /// multi-pair at this level: the dense grid with positional child lookups
    /// and the dead-pair liveness masks.
    Dense,
}

/// The level's marginality, in the two different senses the routes need.
#[derive(Clone, Copy)]
pub(super) struct LevelMarg {
    /// Left child marginal in the output level. What the row-loop dispatch
    /// reads: it decides whether the cell kernel must go through
    /// `MarginalLookup`.
    pub(super) left_now: bool,
    /// Right child, same sense as [`LevelMarg::left_now`].
    pub(super) right_now: bool,
    /// Left child marginal in the output level or in either operand. The
    /// sparse gates read this wider test: the sparse reverse index buckets by
    /// a decoded pair ref, which is not a per-node key once any side of the
    /// level carries count payloads rather than node indices.
    pub(super) left_any: bool,
    /// Right child, same sense as [`LevelMarg::left_any`].
    pub(super) right_any: bool,
    /// The schedule names this level as one to sum out, so the streaming fold
    /// replaces the product construction entirely.
    pub(super) is_target: bool,
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
    /// The child grids the dense route would have to fill outweigh the child
    /// grids the sparse route would have to scan, and one of the former is
    /// over `min_grid` and sparse against its live products: reason enough
    /// to scatter whatever this level's own grid is.
    pub(super) child_grid_wins: bool,
    /// Grids at or below this many cells are not worth either sparse route's
    /// setup, whatever the density says.
    pub(super) min_grid: usize,
}

/// Pick this level's [`Route`].
///
/// Pure: it reads the level's shape, its marginal plan and the two gates, and
/// touches no storage. Order matters — the sparse routes are checked first
/// because they avoid materializing the grid the dense routes need, and
/// streaming is checked before the marginal-child dense build because a
/// streaming target has no downstream structure to build.
pub(super) fn route_level(
    shape: LevelShape,
    plan: &MarginalPlan,
    marginal: &LevelMarg,
    sparse: SparseGate,
) -> Route {
    // The dense routes write this level's grid and, before it, the grid of
    // any child the sparse pipeline left as a product list; the sparse routes
    // walk the live products and write none of the three. A level with a
    // small grid of its own can sit over wide, sparse children — an operand's
    // root does, with one node each side and the whole diagram below — so a
    // child grid the dense route would have to materialize counts as well.
    let big_grid = shape.f.here * shape.g.here > sparse.min_grid || sparse.child_grid_wins;

    // A marginal child on either side rules the scatter walk out entirely, so
    // the density check never has to hold for a level with count payloads.
    if sparse.available && !(marginal.left_any || marginal.right_any) && big_grid && sparse.density_wins {
        return Route::Sparse;
    }
    // Exactly one marginal child, and not a target: the marginal side is a
    // pass-through carrier that kills no pair, so the structural sibling alone
    // governs survival and the dense slab would be mostly dead. Targets do
    // occur with one marginal child and are common — they fall through to the
    // streaming fold below.
    if sparse.available && (marginal.left_any ^ marginal.right_any) && !marginal.is_target && big_grid {
        return Route::SparseMarg;
    }
    if marginal.is_target {
        return Route::Stream { marginal_children: marginal.left_now || marginal.right_now };
    }
    // The positional lookups of the two dense routes cannot carry a side
    // through, so a pass-through side takes the marginal-child build too.
    if marginal.left_now || marginal.right_now
        || plan.sides.left.is_passthrough() || plan.sides.right.is_passthrough()
    {
        return Route::MarginalChild;
    }
    if plan.both_multi_pair {
        return Route::Dense;
    }
    Route::PlainDense
}

impl Route {
    /// Reject unavailable operand structure and assert that the selected route can decode its children.
    ///
    /// Two checks. First, the marginalization schedule: an operand level `t` that
    /// is marginal (pair structure replaced by model counts) conjoins soundly
    /// only with an identity counterpart, which the fast paths consume before
    /// any route is chosen; reaching a route with one operand marginal and the
    /// other constraining `t` means a variable was summed out of one operand
    /// while still live in the other, and the build would index empty `nodes`
    /// or miscount. Both operands marginal is sound and passes.
    ///
    /// Second, no marginal child on a structural route: [`Route::Dense`] and
    /// [`Route::PlainDense`] index the child grids and cannot decode a parent's
    /// marginal refs, so a non-empty marginal child faults there. A marginal
    /// child of `slot_count() == 0` holds no cells and is allowed; the streaming
    /// commit and leaf marginalization both produce such levels.
    pub(super) fn validate(
        self,
        f: &Tdd,
        g: &Tdd,
        shape: LevelShape,
        marginal: &LevelMarg,
        run: &ApplyRun,
    ) -> Result<(), OperationError> {
        let LevelShape { t, left, right, .. } = shape;
        let (left_idx, right_idx) = (left.idx(), right.idx());
        let ApplyRun { left_identity, right_identity, .. } = run;
        let left_marginal = f.level(t).is_marginal();
        let right_marginal = g.level(t).is_marginal();
        let left_identity_at_t = left_identity[left_idx] && left_identity[right_idx];
        let right_identity_at_t = right_identity[left_idx] && right_identity[right_idx];
        let violation = (left_marginal && !right_marginal && !right_identity_at_t)
            || (right_marginal && !left_marginal && !left_identity_at_t);
        if violation {
            return Err(OperationError::MarginalLevel(t));
        }

        if matches!(self, Route::Dense | Route::PlainDense) {
            let marginal_wide = |lvl: &TddLevel| lvl.is_marginal() && lvl.slot_count() > 0;
            let t_idx = t.idx();
            cheap_assert!(
                !marginal.left_now && !marginal.right_now
                    && !marginal_wide(&f.levels[left_idx]) && !marginal_wide(&f.levels[right_idx])
                    && !marginal_wide(&g.levels[left_idx]) && !marginal_wide(&g.levels[right_idx]),
                "a structural product-grid route was chosen for a level with a \
                 non-empty marginal child (t_idx={t_idx} l={left_idx} r={right_idx})"
            );
        }
        Ok(())
    }
}
