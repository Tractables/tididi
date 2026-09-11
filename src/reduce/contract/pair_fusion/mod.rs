//! Same-left pair fusion — production implementation.
//!
//! Entry points: `fuse_pairs_at_parents` (filtered to a parent set) and
//! `fuse_pairs` (unfiltered, test-only).

mod plan;
mod rewrite;
mod slots;


use crate::engine::Engine;
use crate::diagram::WeightVal;
use crate::limits::ApplyError;
use crate::diagram::Tdd;
use crate::vtree::VtreeIdx;

use crate::diagram::{ChildSide, boundary_marginal_levels_into, boundary_marginal_levels_of};
use crate::value::Count;

use super::scratch::{take_scratch, return_scratch, ContractScratch};

use plan::{collect_fusion_plans, resolve_leaf_fusion_refs_by_lookup};
use rewrite::rebuild_parent_level;
use slots::{allocate_fusion_slots, allocate_fusion_slots_weighted};

/// Stats returned by the pair-fusion sweeps.
#[derive(Debug, Clone, Default)]
pub(crate) struct PairFusionStats {
    /// Total number of (parent node, `x_idx`) groups fused (each removes
    /// `group_size - 1` parent pair entries and references one `R_new` slot).
    /// Counts applied rewrites only: a weighted leaf group whose value the pinned
    /// column cannot represent is dropped before Phase 3 and not counted (the
    /// contract fixpoint reads a nonzero count as "the diagram changed").
    pub fusion_groups: usize,
}

/// Destructively apply same-left pair fusion: at every boundary marginal level, for each
/// parent node with a same-X-side group of pairs `(L, R1), (L, R2), …`
/// (same L, distinct marginal-side indices), replace those pairs with a
/// single fused entry `(L, R_new)` where R_new is a newly-pushed
/// marginal slot whose count equals `c(R1) + c(R2) + …`.
///
/// Boolean correctness: at marginal levels, distinct nodes correspond
/// to disjoint Z-assignment sets (partition invariant). So
/// `c(R1 ∨ R2 ∨ …) = c(R1) + c(R2) + …`,
/// and the parent's contribution `c(L)·c(R1) + c(L)·c(R2) + … =
/// c(L)·(f+g+…)` is preserved.
///
/// Weighted mode runs the same rewrite over the semiring: the fused value is the
/// `WeightStore` sum of the group, emitted as a fresh level slot
/// (`allocate_fusion_slots_weighted`) — except at a vtree leaf, whose column is
/// pinned to three label-aliased slots and admits no mint, so there the group
/// folds only onto a value the column already holds
/// (`resolve_leaf_fusion_refs_by_lookup`) and is otherwise left alone. Only the
/// disjointness of the slot reprs, finite additivity over a disjoint union (which
/// holds for signed measures) and distributivity in ℚ are needed — so it is sound
/// in the exact domain and gated off in the bounded-precision `Log` domain. See
/// the diagram's attached store, the single arming predicate.
///
/// Notes:
///   - R1, R2, … are left in the marginal level (they may be referenced
///     from other parent nodes). Subsequent `minimize` will compact any
///     newly-unreferenced slots.
///   - Slot identity is count-keyed: a plan whose `c_new` matches an
///     existing marginal slot — or another plan in this sweep — shares
///     that slot instead of allocating a fresh one. With pair-list dedup
///     removed (pair lists are multisets), the duplicate `(L, R_shared)`
///     pairs this produces at the parent are sound: each occurrence
///     carries one plan's `c(L)·c(R)` contribution and downstream
///     twin-merge preserves the multiset.
///   - Returns stats; does nothing if no boundary level has eligible
///     groups.
///
/// Fallible: every unbounded accumulator grows through `try_push` /
/// `try_resize`, so an allocation that goes over budget, or that exhausts the
/// process address-space limit, returns `Err(ApplyError::OverBudget)` instead of aborting
/// the process. A partially-fused level left behind on early return is
/// still sound (extra unreferenced marginal slots are compacted by
/// minimize; every completed per-node pair rewrite is self-consistent) —
/// with the one exception of the node whose Phase-3 re-encode allocation
/// failed, whose in-place list is left mid-rewrite. The caller discards
/// the diagram on OverBudget regardless, which is what both cases rely on.
// The full unfiltered sweep, for the tests that pin fusion-canonicality on a whole
// diagram; production uses `fuse_pairs_at_parents`.
#[cfg(test)]
pub(crate) fn fuse_pairs(eng: &Engine, tdd: &mut Tdd) -> Result<PairFusionStats, ApplyError> {
    // Test/validate-only full sweep: no caller-held contract scratch reaches
    // here, so borrow the pooled `ContractScratch` for its `pair_fusion` scatter
    // (the weighted gate and all real work live in `fuse_pairs_inner`;
    // production goes through `fuse_pairs_at_parents` or, on the hot contract
    // path, calls the inner directly with its held scratch — see those).
    let mut scratch = take_scratch(eng);
    let r = fuse_pairs_inner(eng, tdd, None, &mut scratch);
    return_scratch(eng, scratch);
    r
}

/// Restricted sweep: only consider boundary-marginal parents whose vtree-parent
/// index is in `parent_vtree_idxs`. Parents not in the filter are skipped
/// entirely. Useful after `marginalize_batch` to restrict the sweep to only
/// the parents of the just-marginalized levels, where new fusion-eligible groups
/// may have been created.
///
/// Pass an empty slice to skip all levels (no-op). Use `fuse_pairs` for
/// the full unfiltered sweep.
///
/// # Errors
///
/// Returns `Err(ApplyError::OverBudget)` if a budget-gated rewrite step fails.
pub(crate) fn fuse_pairs_at_parents(
    eng: &Engine,
    tdd: &mut Tdd,
    parent_vtree_idxs: &[VtreeIdx],
) -> Result<PairFusionStats, ApplyError> {
    // Public entry: no caller-held scratch, so borrow the pooled one. The weighted gate lives in
    // `fuse_pairs_inner`. The hot per-parent contract fixpoint bypasses this
    // wrapper and calls the inner directly to reuse its already-taken scratch.
    let mut scratch = take_scratch(eng);
    let r = fuse_pairs_inner(eng, tdd, Some(parent_vtree_idxs), &mut scratch);
    return_scratch(eng, scratch);
    r
}

/// Per-(node, `x_idx`) fusion plan. Phase 1 builds these; Phase 2 fills `new_ref`.
struct PlanEntry {
    node_idx: usize,
    x_idx: u32,
    c_new: Count,
    // Weighted mode only: the fused semiring value (Σ over the occurrence
    // multiset). `None` in integer mode, where the fused count lives in `c_new`
    // (which is then a dummy `Small(0)` on the weighted arm). Boxed so the
    // integer path pays one pointer rather than a whole inline `WeightVal`
    // (which is sized by its widest variant, the `BigRational` one).
    c_new_w: Option<Box<WeightVal>>,
    // Filled in Phase 2 with the fully-encoded marginal-side ref to write
    // into the fused parent pair: a tagged inline count (bit-30 set)
    // when `c_new` fits the inline threshold under emit mode, else a
    // bare slot index (bit-30 clear). In the slot case plans whose
    // `c_new` matches share a slot (existing or newly allocated).
    // Sound under multiset pair lists.
    new_ref: u32,
}

pub(super) fn fuse_pairs_inner(
    eng: &Engine,
    tdd: &mut Tdd,
    parent_filter: Option<&[VtreeIdx]>,
    scratch: &mut ContractScratch,
) -> Result<PairFusionStats, ApplyError> {
    // Single-source weighted-mode gate for every entry (the pooled wrappers and
    // the hot contract-path direct call). Weighted marginalization carries no
    // integer marginal counts (`marginal_counts` is `None`); its per-slot values
    // live in the external `WeightStore`. Two outcomes:
    //   * Exact domain → run the weighted arm below, which never touches the
    //     `None` integer store;
    //   * otherwise (`Log` domain — signed-log addition is order-dependent and
    //     cancellation-prone) → skip entirely.
    // Integer mode short-circuits on the first test and reaches the body with
    // `weighted == false`.
    let weighted = match tdd.weights.as_ref() {
        // `Log` domain: signed-log addition is order-dependent and
        // cancellation-prone, so a sum over a group is not the value the
        // fusion identity needs. Skip entirely.
        Some(ws) if ws.is_log() => return Ok(PairFusionStats::default()),
        Some(_) => true,
        None => false,
    };
    let mut stats = PairFusionStats::default();
    fill_boundaries(tdd, parent_filter, &mut scratch.boundaries);
    // Indexed so the per-boundary work can borrow `scratch.pair_fusion` (a
    // disjoint field) while this list stays live. Snapshotting the set before
    // the loop is safe: fusion never marginalizes a level, and invariant 5
    // forbids un-marginalizing one, so the snapshot cannot go stale.
    for bi in 0..scratch.boundaries.len() {
        let (v, parent, side) = scratch.boundaries[bi];
        // Weighted: both reading a group's values and minting the fused slot go
        // through this level's `WeightStore` vec, so a level that is marginal but
        // has no store yet is not fusable. It should not arise (the weighted
        // marginalize sets the store as it makes the level marginal), but skipping
        // is the no-op reading, and it keeps the two `expect`s below unreachable.
        if weighted && !tdd.weights.as_ref().is_some_and(|ws| ws.is_set(v.idx())) {
            continue;
        }
        // Phase 1: full-scan parent's nodes; collect per-(node, x_idx) groups
        // with > 1 distinct marginal-side index. Compute c_new for each.
        let mut plans: Vec<PlanEntry> = if weighted {
            collect_fusion_plans::<true>(eng, tdd, parent, v, side, &mut scratch.pair_fusion)?
        } else {
            collect_fusion_plans::<false>(eng, tdd, parent, v, side, &mut scratch.pair_fusion)?
        };
        if plans.is_empty() {
            continue;
        }

        // Phase 2: allocate a count-keyed marginal slot per plan.
        //
        // Two plans whose `c_new` matches share a slot — either an existing slot
        // at the level whose count equals `c_new`, or a single newly-allocated
        // slot reused by all plans in this sweep with the same key. Duplicate
        // `(L, R_shared)` pairs that result at the parent are sound: pair lists
        // are multisets, each occurrence carries
        // one plan's `c(L)·c(R)` contribution, and twin-merge preserves the
        // multiset. Slot sharing is only sound because nothing downstream
        // dedups pair lists — a dedup anywhere below would collapse the shared
        // pairs and drop count.
        // Fusion-inline: carry a small fused count inline in the parent pair
        // instead of allocating a slot for it. A count too wide for a ref takes
        // the slot path instead.
        // Set when at least one plan emits an inline ref: the parent level's
        // marginal-side inline marker must then be raised (below) or readers
        // misdecode the bit-30-tagged ref as a grid coordinate.
        //
        // A weighted leaf boundary allocates nothing at all. A weight-marginal
        // leaf's column is pinned — an immutable, label-ordered, exactly-3-slot
        // cache of `WeightStore::leaf_val`, shared compile-wide and aliased by bare
        // leaf-label refs from every other `Tdd` (`marginalize_leaf_weighted`).
        // Appending a `Slot(3+)` there would break that alias, and would also
        // overflow the flat remap window `prune_unreachable` sizes from
        // `Tdd::effective_width`, which hardcodes `LEAF_WIDTH` for leaf levels —
        // the ref would silently index the neighbouring level's remap region. (The
        // integer arm's escape, a self-describing inline count, has no weighted
        // analogue: a weighted `ValueRef::Inline` is a process-wide intern-table
        // index that dangles across component graft.) So a leaf plan is resolved by
        // lookup in the pinned column and dropped when the column cannot represent
        // its value.
        let leaf_boundary = weighted && tdd.vtree.node(v).is_leaf();
        let any_inline = if leaf_boundary {
            resolve_leaf_fusion_refs_by_lookup(tdd, v, &mut plans);
            if plans.is_empty() {
                // Every group's sum was outside the pinned column: nothing folds
                // at this boundary.
                continue;
            }
            // Slot refs only — the weighted arm never emits an inline marginal ref.
            false
        } else if weighted {
            allocate_fusion_slots_weighted(tdd, v, &mut plans)?
        } else {
            allocate_fusion_slots(eng, tdd, v, &mut plans)?
        };

        // Counted after Phase 2, because the weighted-leaf arm drops the plans
        // whose value the pinned column cannot represent: `fusion_groups` must
        // count applied rewrites only. The contract fixpoint (`strategies.rs`)
        // reads `fusion_groups > 0` as "the diagram changed" and loops again, so
        // counting a dropped plan there would spin it forever. Every other arm
        // applies all of its plans, so the placement is behavior-neutral for them.
        stats.fusion_groups += plans.len();

        // Phase 3: rewrite parent pair lists for affected nodes. `plans` is
        // emitted grouped by ascending `node_idx` in Phase 1 (and the leaf-lookup
        // filter above preserves that order), which is the cursor-walk
        // precondition. See `rebuild_parent_level`.
        rebuild_parent_level(eng, tdd, parent, side, any_inline, &plans)?;
    }
    Ok(stats)
}

/// Fill `out` with the boundary set this sweep covers, scoped to the caller's
/// parents when it named any.
///
/// The hot caller is the contract fixpoint, which calls the sweep once per
/// marginal-boundary parent with a one-element filter — so deriving the
/// boundaries from that parent's own children replaces a full-diagram level
/// scan (and a throwaway `Vec`) per call with two vtree lookups. The unfiltered
/// sweep still scans every level.
#[inline(always)]
fn fill_boundaries(
    tdd: &Tdd,
    parent_filter: Option<&[VtreeIdx]>,
    out: &mut Vec<(VtreeIdx, VtreeIdx, ChildSide)>,
) {
    match parent_filter {
        Some(parents) => boundary_marginal_levels_of(tdd, parents, out),
        None => boundary_marginal_levels_into(tdd, out),
    }
}

#[cfg(test)]
mod tests;
