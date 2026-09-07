//! (P) same-left pair fusion — production implementation.
//!
//! Entry points: `apply_p_fusion_at_parents` (production, invoked by the
//! downstream compile driver) and `apply_p_fusion` (unfiltered, test-only).

use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use crate::tdd::query::semiring::WeightVal;
use crate::tdd::transform::pairwise::conjoin::{try_push, try_resize, ApplyError};
use crate::tdd::types::{InputPair, LocalNodeIdx, MargRef, Tdd};
use crate::vtree::VtreeIdx;

use crate::tdd::marg_slots::{boundary_marginal_levels_into, boundary_marginal_levels_of, push_count_key, sum_marginal_counts, ChildSide, CountKey, SlotInterner};

use super::scratch::{take_scratch, return_scratch, ContractScratch, PFusionScratch};

/// Stats returned by `apply_p_fusion`.
#[derive(Debug, Clone, Default)]
#[doc(hidden)]
pub struct PFusionStats {
    /// Total number of (`parent_node`, `x_idx`) groups fused (each removes
    /// `group_size - 1` parent pair entries and references one `R_new` slot).
    /// Counts APPLIED rewrites only: a weighted LEAF group whose value the pinned
    /// column cannot represent is dropped before Phase 3 and not counted (the
    /// contract fixpoint reads a nonzero count as "the diagram changed").
    pub fusion_groups: usize,
    /// Total parent pair entries removed by fusion (= Σ over groups of
    /// `group_size - 1`).
    pub pairs_eliminated: usize,
    /// Number of new marginal slots pushed. May be LESS than `fusion_groups`:
    /// slot identity is count-keyed, so plans whose `c_new` matches an existing
    /// slot or another plan in the sweep share a slot rather than allocating.
    /// Sound because pair lists are multisets (each shared-slot pair occurrence
    /// carries one plan's contribution). A weighted LEAF boundary contributes
    /// zero by construction — it folds by lookup and never mints.
    pub slots_added: usize,
}

/// Destructively apply (P): at every boundary marginal level, for each
/// parent node with a same-X-side group of pairs `(L, R1), (L, R2), …`
/// (same L, distinct marginal-side indices), replace those pairs with a
/// single fused entry `(L, R_new)` where R_new is a newly-pushed
/// marginal slot whose count equals `c(R1) + c(R2) + …`.
///
/// Boolean correctness: at marginal levels, distinct nodes correspond
/// to disjoint Z-assignment sets (partition invariant). So
/// `c(R1 ∨ R2 ∨ …) = c(R1) + c(R2) + …`,
/// and the parent's contribution `c(L)·c(R1) + c(L)·c(R2) + … =
/// c(L)·(c1+c2+…)` is preserved.
///
/// WEIGHTED mode runs the same rewrite over the semiring: the fused value is the
/// `WeightStore` sum of the group, emitted as a fresh level SLOT
/// (`allocate_fusion_slots_weighted`) — except at a vtree LEAF, whose column is
/// pinned to three label-aliased slots and admits no mint, so there the group
/// folds only onto a value the column already holds
/// (`resolve_leaf_fusion_refs_by_lookup`) and is otherwise left alone. Only the
/// disjointness of the slot reprs, finite additivity over a disjoint union (which
/// holds for SIGNED measures) and distributivity in ℚ are needed — so it is sound
/// in the EXACT domain and gated off in the bounded-precision Log domain. See
/// `marginalize::weighted_fusion_active`, the single arming predicate.
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
/// `try_resize`, so an over-budget or RLIMIT_AS-exhausting
/// allocation returns `Err(ApplyError::OverBudget)` instead of aborting
/// the process. The caller (adaptive_minimize / run_marginalize_at*)
/// propagates this to the vsplit driver / non-vsplit loop, which routes
/// to recovery. A partially-fused level left behind on early return is
/// still sound (extra unreferenced marginal slots are compacted by
/// minimize; every completed per-node pair rewrite is self-consistent) —
/// with the one exception of the node whose Phase-3 re-encode allocation
/// failed, whose in-place list is left mid-rewrite. The caller discards
/// the TDD on OverBudget regardless, which is what both cases rely on.
///
/// Preconditions: same as `apply_h_by_count`.
#[doc(hidden)] // test-support: the full unfiltered sweep is reached only by tests;
               // production uses `apply_p_fusion_at_parents`.
pub fn apply_p_fusion(tdd: &mut Tdd) -> Result<PFusionStats, ApplyError> {
    // Test/validate-only full sweep: no caller-held contract scratch reaches
    // here, so borrow the pooled `ContractScratch` for its `p_fusion` scatter
    // (the weighted gate and all real work live in `apply_p_fusion_inner`;
    // production goes through `apply_p_fusion_at_parents` or, on the hot contract
    // path, calls the inner directly with its held scratch — see those).
    let mut scratch = take_scratch();
    let r = apply_p_fusion_inner(tdd, None, &mut scratch);
    return_scratch(scratch);
    r
}

/// Restricted sweep: only consider boundary-marginal parents whose vtree-parent
/// index is in `parent_vtree_idxs`. Parents not in the filter are skipped
/// entirely. Useful after `marginalize_batch` to restrict the sweep to only
/// the parents of the just-marginalized levels, where new (P)-eligible groups
/// may have been created.
///
/// Pass an empty slice to skip all levels (no-op). Use `apply_p_fusion` for
/// the full unfiltered sweep.
///
/// # Errors
///
/// Returns `Err(ApplyError::OverBudget)` if a budget-gated rewrite step fails.
#[doc(hidden)]
pub fn apply_p_fusion_at_parents(
    tdd: &mut Tdd,
    parent_vtree_idxs: &[VtreeIdx],
) -> Result<PFusionStats, ApplyError> {
    // Public entry (compile driver, e.g. `run_marginalize_at`): no caller-held
    // scratch, so borrow the pooled one. The weighted gate lives in
    // `apply_p_fusion_inner`. The hot per-parent contract fixpoint bypasses this
    // wrapper and calls the inner directly to reuse its already-taken scratch.
    let mut scratch = take_scratch();
    let r = apply_p_fusion_inner(tdd, Some(parent_vtree_idxs), &mut scratch);
    return_scratch(scratch);
    r
}

/// Per-(node, `x_idx`) fusion plan. Phase 1 builds these; Phase 2 fills `new_ref`.
struct PlanEntry {
    node_idx: usize,
    x_idx: u32,
    distinct_margs: Vec<u32>,
    c_new: CountKey,
    // WEIGHTED mode only: the fused semiring value (Σ over the occurrence
    // multiset). `None` in integer mode, where the fused count lives in `c_new`
    // (which is then a dummy `Small(0)` on the weighted arm). BOXED so the
    // integer path pays one pointer rather than a whole inline `WeightVal`
    // (which is sized by its widest variant, the `BigRational` one).
    c_new_w: Option<Box<WeightVal>>,
    // Filled in Phase 2 with the fully-encoded marg-side ref to write
    // into the fused parent pair: a tagged inline count (bit-30 set)
    // when `c_new` fits the inline threshold under emit mode, else a
    // bare slot index (bit-30 clear). In the slot case plans whose
    // `c_new` matches share a slot (existing or newly allocated).
    // Sound under multiset pair lists.
    new_ref: u32,
}

pub(super) fn apply_p_fusion_inner(
    tdd: &mut Tdd,
    parent_filter: Option<&[VtreeIdx]>,
    scratch: &mut ContractScratch,
) -> Result<PFusionStats, ApplyError> {
    // Single-source weighted-mode gate for EVERY entry (the pooled wrappers and
    // the hot contract-path direct call). Weighted marginalization carries NO
    // integer marginal counts (`marginal_counts` is `None`); its per-slot values
    // live in the external `WeightStore`. Two outcomes:
    //   * `weighted_fusion_active()` (Exact domain) → run the WEIGHTED arm
    //     below, which never touches the `None` integer store;
    //   * otherwise (Log domain — signed-log addition is order-dependent and
    //     cancellation-prone) → skip entirely, byte-identical to the pre-port
    //     behavior.
    // Integer mode short-circuits on the first test and reaches the body with
    // `weighted == false`, exactly as before.
    let weighted = if crate::tdd::transform::unary::marginalize::weight_ctx_active() {
        if !crate::tdd::transform::unary::marginalize::weighted_fusion_active() {
            return Ok(PFusionStats::default());
        }
        true
    } else {
        false
    };
    let mut stats = PFusionStats::default();
    fill_boundaries(tdd, parent_filter, &mut scratch.boundaries);
    // Indexed so the per-boundary work can borrow `scratch.p_fusion` (a
    // disjoint field) while this list stays live. The set is snapshotted before
    // the loop, exactly as when it was a local `Vec`: fusion never marginalizes
    // a level, and I1 forbids un-marginalizing one, so it cannot go stale.
    for bi in 0..scratch.boundaries.len() {
        let (v, parent, side) = scratch.boundaries[bi];
        // Weighted: both reading a group's values and minting the fused slot go
        // through this level's `WeightStore` vec, so a level that is marginal but
        // has no store yet is not fusable. It should not arise (the weighted
        // marginalize sets the store as it makes the level marginal), but skipping
        // is the no-op reading, and it keeps the two `expect`s below unreachable.
        if weighted
            && !crate::tdd::transform::unary::marginalize::with_weight_ctx(|ws| ws.is_set(v.idx()))
        {
            continue;
        }
        // Phase 1: full-scan parent's nodes; collect per-(node, x_idx) groups
        // with > 1 distinct marginal-side index. Compute c_new for each.
        let mut plans: Vec<PlanEntry> = if weighted {
            collect_fusion_plans::<true>(tdd, parent, v, side, &mut scratch.p_fusion)?
        } else {
            collect_fusion_plans::<false>(tdd, parent, v, side, &mut scratch.p_fusion)?
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
        // (P)-inline: carry a small fused count inline in the parent pair
        // instead of allocating a slot for it. `MargRef::inline_raw` funnels
        // through `marg_inline_max()`, so the all-slots test regime
        // (threshold 0) keeps the slot path.
        // Set when at least one plan emits an inline ref: the parent level's
        // marg-side inline marker must then be raised (below) or readers
        // misdecode the bit-30-tagged ref as a grid coordinate (#63 corruption).
        //
        // WEIGHTED LEAF BOUNDARY: no allocation at all. A weight-marginal LEAF's
        // column is PINNED — an immutable, label-ordered, exactly-3-slot cache of
        // `WeightStore::leaf_val`, shared compile-wide and aliased by bare
        // leaf-LABEL refs from every other `Tdd` (`marginalize_leaf_weighted`).
        // Appending a `Slot(3+)` there would break that alias AND overflow the
        // flat remap window `prune_unreachable` sizes from `Tdd::effective_width`,
        // which hardcodes LEAF_WIDTH for leaf levels — the ref would silently
        // index the NEIGHBOURING level's remap region. (The integer arm's escape,
        // a self-describing INLINE count, has no weighted analogue: a weighted
        // `MargRef::Inline` is a GLOBAL intern-table index that dangles across
        // component graft.) So a leaf plan is resolved by LOOKUP in the pinned
        // column and dropped when the column cannot represent its value.
        let leaf_boundary = weighted && tdd.vtree.node(v).is_leaf();
        let any_inline = if leaf_boundary {
            resolve_leaf_fusion_refs_by_lookup(tdd, v, &mut plans);
            if plans.is_empty() {
                // Every group's sum was outside the pinned column: nothing folds
                // at this boundary.
                continue;
            }
            // Slot refs only — the weighted arm never emits an inline marg ref.
            false
        } else if weighted {
            allocate_fusion_slots_weighted(tdd, v, &mut plans, &mut stats.slots_added)?
        } else {
            allocate_fusion_slots(tdd, v, &mut plans, &mut stats.slots_added)?
        };

        // Counted AFTER Phase 2, because the weighted-leaf arm DROPS the plans
        // whose value the pinned column cannot represent: `fusion_groups` must
        // count APPLIED rewrites only. The contract fixpoint (`strategies.rs`)
        // reads `fusion_groups > 0` as "the diagram changed" and loops again, so
        // counting a dropped plan there would spin it forever. Every other arm
        // applies all of its plans, so the placement is behavior-neutral for them.
        stats.fusion_groups += plans.len();
        for plan in &plans {
            stats.pairs_eliminated += plan.distinct_margs.len() - 1;
        }

        // Phase 3: rewrite parent pair lists for affected nodes. `plans` is
        // emitted grouped by ascending `node_idx` in Phase 1 (and the leaf-lookup
        // filter above preserves that order), which is the cursor-walk
        // precondition. See `rebuild_parent_level`.
        rebuild_parent_level(tdd, parent, side, any_inline, &plans)?;
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

/// Phase 1: full-scan the parent level's nodes; collect per-(node, `x_idx`) groups
/// with > 1 distinct marginal-side index. Compute `c_new` for each.
///
/// The output plan list is in ascending `node_idx` order (Phase 3 cursor-walk
/// relies on this). Empty return means no boundary at this level is fusable.
///
/// Within a node, plans are emitted in first-occurrence order of their `x_idx`
/// (the order the grouping scatter first saw each explicit-side ref). Phase 2
/// (count-keyed slot interning) and Phase 3 (x_idx-keyed rewrite hashmap) are
/// both insensitive to within-node order, so this ordering is not load-bearing
/// (only ascending `node_idx` across nodes is).
///
/// One const-generic mode selects the per-group ACTION; the grouping walk is
/// shared verbatim by both instantiations. A const generic (not a runtime
/// flag) so the production integer `<false>` instantiation keeps its exact
/// code shape — the weighted branch below folds away — and so the mode can
/// never be varied per node or per pair.
///
/// `WEIGHTED` = weighted (P) fusion: the fused value is summed out of the
/// external `WeightStore` semiring instead of the integer count store. In that
/// mode the level's integer `marginal_counts` is NEVER touched (it is `None`
/// in weight context).
#[inline(always)]
fn collect_fusion_plans<const WEIGHTED: bool>(
    tdd: &Tdd,
    parent: VtreeIdx,
    v: VtreeIdx,
    side: ChildSide,
    scratch: &mut PFusionScratch,
) -> Result<Vec<PlanEntry>, ApplyError> {
    let plevel = &tdd.levels[parent.idx()];
    let vlevel = &tdd.levels[v.idx()];
    // Integer marginal counts exist only on the production integer path: in
    // weight context the values live in the external `WeightStore` and
    // `marginal_counts` is `None`, so the weighted arm may not unwrap it. The
    // empty stand-in is never read (`emit` takes the semiring branch before
    // `sum_marginal_counts`).
    let no_counts: [u128; 0] = [];
    let counts: &[u128] = if WEIGHTED {
        &no_counts
    } else {
        vlevel.marginal_counts.as_ref().unwrap()
    };
    let big = if WEIGHTED { None } else { vlevel.marginal_counts_big.as_ref() };
    let mut out: Vec<PlanEntry> = Vec::new();

    // Grouping key `x_idx` is the EXPLICIT-side ref (opposite the marginal
    // `side`). On the common path it is a DENSE index into the explicit child
    // level — a Boolean node index (`< nodes.len()`) or, if that side is itself
    // a marginal level referenced only through slots, a slot index
    // (`< marginal_counts.len()`). Either is small and dense, so we group with a
    // generation-stamped dense scatter (`scratch`) instead of a hashmap.
    //
    // EXCEPTION: a both-marginal parent (this parent has TWO marginal-child
    // boundaries, one per marginal child) may carry INLINE marg
    // refs on the explicit side, whose bit-30 `MARG_OVERFLOW_TAG` puts `x_idx`
    // outside the dense index space (≈2^30). Indexing a dense array by such a
    // value would demand a multi-GiB allocation, so when the explicit side's
    // inline marker is set we fall back to the opaque-key hashmap for this
    // boundary (rare). Both paths feed ONE `emit` closure below, so the
    // soundness-critical count/plan construction is single-source.
    //
    // The WEIGHTED arm uses the same guard and the same scatter. It once took the
    // hashmap unconditionally, justified by "weight context never sets the inline
    // markers, and mints `MargRef::Inline` refs without raising them" — BOTH
    // halves of which are false. Nothing in weight context mints an inline ref:
    // `scale_weight_ref`'s `Inline` arm is `unreachable!()`, the weighted Phase 2
    // and the leaf sum-lookup emit `slot_raw`, and `emit_marg_side_slots` (the
    // only bit-30 writer) needs integer `marginal_counts`. Weighted marg-side refs
    // are bare slots end to end. And the markers ARE set in weight context —
    // `tag_all_marg_side_slots` runs after every apply and raises them for any
    // marginal-child side — so `explicit_inline` is a conservative SUPERSET here,
    // which is the safe direction: it can only route a boundary to the hashmap
    // that the scatter could have handled.
    let explicit_inline = match side {
        ChildSide::Right => plevel.marg_inlined_left(),
        ChildSide::Left => plevel.marg_inlined_right(),
    };
    let use_scatter = !explicit_inline;

    // Shared per-group emission. `margs` is the FULL occurrence MULTISET of
    // marg-side refs at this x (NO dedup): with count-keyed slot sharing a
    // node's pair list may legitimately contain `(x, M)` more than once, each
    // occurrence carrying one historical plan's `c(M)` contribution, so the
    // fused count sums over OCCURRENCES, not distinct M. Distinct marginal
    // nodes are disjoint Z-sets, so their values add — `c(L)·v1 + c(L)·v2 + … =
    // c(L)·(v1+v2+…)`, the (P) invariant (carried into the semiring in weighted
    // mode; `c_new` is a dummy there).
    let emit = |n: usize, x_idx: u32, margs: &[u32], out: &mut Vec<PlanEntry>|
     -> Result<(), ApplyError> {
        // Weighted: the fused value is the semiring sum over the SAME occurrence
        // multiset; `c_new` is an unread dummy on that arm (Phase 2 reads
        // `c_new_w`). Integer: unchanged.
        let (c_new, c_new_w) = if WEIGHTED {
            (CountKey::Small(0), Some(Box::new(sum_marginal_weights(v, margs))))
        } else {
            (sum_marginal_counts(counts, big, margs), None)
        };
        try_push(out, PlanEntry {
            node_idx: n,
            x_idx,
            distinct_margs: margs.to_vec(),
            c_new,
            c_new_w,
            new_ref: u32::MAX,
        })
    };

    // Per-node body applied to every node in the full scan.
    let process_node = |n: usize,
                        out: &mut Vec<PlanEntry>,
                        sc: &mut PFusionScratch|
     -> Result<(), ApplyError> {
        if plevel.nodes[n].is_leaf() {
            return Ok(());
        }
        // A same-x fusion group needs ≥2 pairs sharing one x_idx, which
        // requires the node to hold ≥2 pairs at all. Single-pair nodes
        // (the common case) can never fuse — skip them before any
        // grouping work. This is the dominant Phase-1 cost: ~94% of
        // scanned nodes produce no plan.
        if plevel.pair_count_at(n) < 2 {
            return Ok(());
        }
        if use_scatter {
            // ── Generation-stamped dense scatter ──
            // Bump the generation instead of clearing `stamp` (O(1) per-node
            // reset). On u32 wrap, zero the stamps and restart at 1 (0 is the
            // "never stamped" sentinel and must never equal a live `gen`).
            sc.generation = match sc.generation.checked_add(1) {
                Some(g) => g,
                None => {
                    for s in sc.stamp.iter_mut() {
                        *s = 0;
                    }
                    1
                }
            };
            let generation = sc.generation;
            sc.touched.clear();
            for p in plevel.pairs_of_idx(n) {
                let (x_idx, marg_idx) = match side {
                    ChildSide::Right => (p.left.0, p.right.0),
                    ChildSide::Left => (p.right.0, p.left.0),
                };
                let xu = x_idx as usize;
                // Pins the premise that lets WEIGHTED share this scatter: a
                // weighted explicit-side ref is a bare slot, never a bit-30
                // inline ref, so `xu` stays in the dense index space and cannot
                // size the arrays into the multi-GiB range. Mirrors the leaf-side
                // pin in `marginalize::inline_leaf_refs_at_parent`. (Grouping is
                // by raw `u32` in both paths, so even a violation would group
                // identically and merely over-allocate — `try_resize` below is
                // fallible, so that degrades to OverBudget, never corruption.)
                debug_assert!(
                    !WEIGHTED || x_idx & crate::tdd::types::MARG_OVERFLOW_TAG == 0,
                    "weighted explicit-side ref {x_idx} carries the inline tag; \
                     weighted marg-side refs are bare slots end to end"
                );
                // Grow the per-x arrays on demand to `max(x)+1` (fallibly — an
                // OOM here becomes OverBudget, not a process abort). New entries
                // are 0 ≠ generation (which is ≥ 1) so they read as "unstamped".
                // `stamp` and `slot_of_x` are kept the SAME length: the OR guard
                // self-heals a prior call that grew one but hit OverBudget before
                // growing the other (both are re-extended to `xu+1`; the already
                // long-enough one's `try_resize` is a no-op), so a later reuse of
                // the pooled scratch can never index the shorter one out of bounds.
                if xu >= sc.stamp.len() || xu >= sc.slot_of_x.len() {
                    try_resize(&mut sc.stamp, xu + 1, 0u32)?;
                    try_resize(&mut sc.slot_of_x, xu + 1, 0u32)?;
                }
                let slot = if sc.stamp[xu] != generation {
                    // First occurrence of this x for this node: open a group.
                    sc.stamp[xu] = generation;
                    let slot = sc.touched.len();
                    sc.slot_of_x[xu] = slot as u32;
                    if sc.touched.len() == sc.touched.capacity() {
                        sc.touched.try_reserve(1).map_err(|_| ApplyError::OverBudget)?;
                    }
                    sc.touched.push(x_idx);
                    // Reuse a retired group slot (keeps its grown capacity) or
                    // allocate one only when this node needs more distinct
                    // x-groups than any prior node.
                    if slot < sc.groups.len() {
                        sc.groups[slot].clear();
                    } else {
                        if sc.groups.len() == sc.groups.capacity() {
                            sc.groups.try_reserve(1).map_err(|_| ApplyError::OverBudget)?;
                        }
                        sc.groups.push(SmallVec::new());
                    }
                    slot
                } else {
                    sc.slot_of_x[xu] as usize
                };
                // Fallible push for SmallVec: while the current buffer (inline
                // OR heap) has spare capacity, push cannot fail. At capacity —
                // the inline→heap spill AND every subsequent heap regrow — use
                // try_reserve so allocation failure becomes OverBudget rather
                // than a process abort. Guarding on `capacity()` keeps every
                // growth fallible for the rare large x-group.
                let g = &mut sc.groups[slot];
                if g.len() == g.capacity() {
                    g.try_reserve(1).map_err(|_| ApplyError::OverBudget)?;
                }
                g.push(marg_idx);
            }
            // Emit in first-occurrence (touched) order. slot i ↔ touched[i].
            for i in 0..sc.touched.len() {
                if sc.groups[i].len() <= 1 {
                    // A single pair at this x cannot fuse.
                    continue;
                }
                emit(n, sc.touched[i], &sc.groups[i], out)?;
            }
        } else {
            // ── Fallback: opaque-key hashmap (explicit side carries inline marg
            // refs; see the `use_scatter` note). Byte-identical grouping to the
            // pre-scatter path; `x_idx` is treated as an opaque key.
            let mut by_x: FxHashMap<u32, SmallVec<[u32; 4]>> = FxHashMap::default();
            for p in plevel.pairs_of_idx(n) {
                let (x_idx, marg_idx) = match side {
                    ChildSide::Right => (p.left.0, p.right.0),
                    ChildSide::Left => (p.right.0, p.left.0),
                };
                let sv = by_x.entry(x_idx).or_default();
                if sv.len() == sv.capacity() {
                    sv.try_reserve(1).map_err(|_| ApplyError::OverBudget)?;
                }
                sv.push(marg_idx);
            }
            for (x_idx, margs) in by_x.drain() {
                if margs.len() <= 1 {
                    continue;
                }
                emit(n, x_idx, &margs, out)?;
            }
        }
        Ok(())
    };
    for n in 0..plevel.nodes.len() {
        process_node(n, &mut out, scratch)?;
    }
    Ok(out)
}

/// Phase 2: allocate a count-keyed marginal slot per plan; fill `plan.new_ref`.
///
/// Returns `true` if at least one plan emitted an inline ref (the parent
/// level's marg-side inline marker must then be raised in Phase 3).
/// Increments `slots_added` for each newly-allocated slot.
#[inline(always)]
fn allocate_fusion_slots(
    tdd: &mut Tdd,
    v: VtreeIdx,
    plans: &mut Vec<PlanEntry>,
    slots_added: &mut usize,
) -> Result<bool, ApplyError> {
    let mut any_inline = false;
    let level = &mut tdd.levels[v.idx()];
    // Seed the interner with existing slot counts so a plan whose
    // c_new equals an existing slot reuses it (C3 preserved on
    // every extension — no duplicate count values are introduced).
    let mut interner = SlotInterner::new();
    {
        let counts = level.marginal_counts.as_ref().unwrap();
        let big = level.marginal_counts_big.as_ref();
        interner.seed(counts, big);
    }
    for plan in plans.iter_mut() {
        // Inline small fused counts: the summed result lives in the pair
        // itself, no slot allocated. Skips count-keyed slot sharing —
        // an inline ref is cheaper than a shared slot.
        if let CountKey::Small(c) = &plan.c_new {
            if let Some(raw) = MargRef::inline_raw(*c) {
                plan.new_ref = raw;
                any_inline = true;
                continue;
            }
        }
        // `interner` checks the map first (read-only); only on miss do we
        // need to push. The hit branch just reads `interner.map` and returns
        // the existing slot.
        if let Some(&existing) = interner.map.get(&plan.c_new) {
            plan.new_ref = MargRef::slot_raw(existing);
            continue;
        }
        // Miss: mint a new slot (`counts` and, for a Big value, the lazily
        // allocated big side-table) via the shared store-push primitive.
        let new_idx = push_count_key(
            level.marginal_counts.as_mut().unwrap(),
            &mut level.marginal_counts_big,
            &plan.c_new,
        )?;
        interner.map.insert(plan.c_new.clone(), new_idx);
        plan.new_ref = MargRef::slot_raw(new_idx);
        *slots_added += 1;
    }
    Ok(any_inline)
}

/// Weighted Phase 1 helper: sum the semiring values of a marg-side occurrence
/// multiset. Mirrors [`sum_marginal_counts`] minus the u128→`BigUint` overflow
/// two-pass — a `BigRational` cannot overflow, so one clean accumulate suffices.
///
/// Each ref is `Inline(gidx)` (value in the store's GLOBAL intern table) or
/// `Slot(s)` (value in the level's `WeightStore` vec); the encoding is
/// self-describing via bit 30, exactly as on the integer side.
///
/// SOUNDNESS. Slots at a marginal level carry pairwise-disjoint model sets
/// (partition invariant), so the values of a group's
/// members are values of disjoint sets and add. Finite additivity over a
/// disjoint union holds for SIGNED measures, so a negative literal weight is not
/// an obstacle; the parent's contribution `Σᵢ W(x)·W(mᵢ) = W(x)·Σᵢ W(mᵢ)` then
/// follows from distributivity in ℚ. This is EXACT-domain reasoning only — the
/// caller's `weighted_fusion_active()` gate excludes the Log domain.
fn sum_marginal_weights(v: VtreeIdx, margs: &[u32]) -> WeightVal {
    crate::tdd::transform::unary::marginalize::with_weight_ctx(|ws| {
        let vals = ws.level(v.idx());
        let mut acc = ws.wzero();
        for &raw in margs {
            // The ZERO sentinel (bit 31) never appears in a pair list (I-invariant;
            // `MargRef::from_raw` debug-asserts the same). Defend anyway: a ZERO
            // child contributes the additive identity, so skipping it is the
            // value-preserving reading — and it keeps `from_raw`'s assert unreached.
            debug_assert!(raw & (1u32 << 31) == 0, "ZERO sentinel must not reach a marg-side pair ref");
            if raw & (1u32 << 31) != 0 {
                continue;
            }
            match MargRef::from_raw(raw) {
                MargRef::Inline(g) => acc.add_assign(ws.interned_value(g)),
                MargRef::Slot(s) => {
                    let v = &vals.expect("weighted p-fusion: marg level has no WeightStore")
                        [s as usize];
                    acc.add_assign(v);
                }
            }
        }
        acc
    })
}

/// Weighted Phase 2 at a vtree LEAF boundary: resolve each plan's fused value to
/// a slot the PINNED column already holds, and DROP the plans it does not.
///
/// The mint-free half of weighted (P) fusion. [`allocate_fusion_slots_weighted`]
/// represents a fused value by appending a slot; at a leaf that is forbidden —
/// the column is the immutable, label-ordered 3-slot `leaf_val` cache every other
/// `Tdd` of the compile aliases by bare leaf-LABEL refs (THE PIN INVARIANT,
/// `marginalize::marginalize_leaf_weighted`). What IS available is the column's
/// own values, and a (P) sum lands on them far more often than a generic lookup
/// would suggest: `(x,Pos) + (x,Neg)` sums to `w⁺+w⁻`, which IS the One slot BY
/// DEFINITION — for every weight table, asymmetric included, which is the lever
/// equal-value ref canonicalization cannot reach — and after that canonicalization
/// `(x,Pos) + (x,Pos)` at `w⁺ = w⁻` sums to `2w⁺ = w⁺+w⁻ = One` as well.
///
/// A MISS drops the plan, which leaves that group's pairs exactly as they were:
/// Phase 3 rewrites only the x-indices a SURVIVING plan names, so an untouched
/// group is a no-op there. The cost is a size residual (one un-fused (P) redex),
/// never a wrong value — and no invariant checker objects, because the C1
/// (P)-saturation checks (`validate::marg::check_no_fusion_redexes`,
/// `debug_assert_p_saturated`) return early in weight context. Order is preserved
/// by `retain_mut`, so Phase 3's ascending-`node_idx` precondition survives.
///
/// The column is never written and `retired_marg_width` is never bumped: the
/// level stays exactly `LEAF_WIDTH` wide, and because
/// `marginalize::find_leaf_slot_by_value` scans ascending, each fused ref is the
/// CANONICAL (smallest) slot of its value class — which is what pin check #4
/// demands of every leaf-side ref.
#[inline(always)]
fn resolve_leaf_fusion_refs_by_lookup(tdd: &Tdd, v: VtreeIdx, plans: &mut Vec<PlanEntry>) {
    use crate::tdd::transform::unary::marginalize::{find_leaf_slot_by_value, with_weight_ctx};
    debug_assert!(
        tdd.vtree.node(v).is_leaf(),
        "leaf-boundary fusion resolution called on internal level {}",
        v.0
    );
    with_weight_ctx(|ws| {
        // Exact domain only: `weight_key` equality is value equality there, while
        // a `WeightKey::Log` compares `f64` bit patterns. The caller's
        // `weighted_fusion_active()` gate already excludes Log — this pins that.
        debug_assert!(
            !ws.log_mode,
            "leaf sum-lookup fold reached in the Log domain (the fusion gate must exclude it)"
        );
        plans.retain_mut(|plan| {
            let val: &WeightVal = plan
                .c_new_w
                .as_deref()
                .expect("weighted p-fusion plan missing fused value");
            match find_leaf_slot_by_value(ws, v.idx(), val) {
                Some(slot) => {
                    plan.new_ref = MargRef::slot_raw(slot);
                    true
                }
                None => false,
            }
        });
    });
}

/// Weighted Phase 2: encode each plan's fused value as a marg-side ref.
///
/// Emission is the per-level `WeightStore` SLOT form — the same
/// `push_value`-then-bump-`retired_marg_width` shape as `scale_weight_ref`'s
/// `Slot` arm (`dup_resolve.rs`), which is the weighted mint path that ships
/// today. On a weight-marginal level `retired_marg_width` IS the live width read
/// by `TddLevel::width()`, and apply sizes its buffers from it, so a missed bump
/// is an out-of-bounds waiting to happen.
///
/// NOT the global intern table / `MargRef::Inline(gidx)` form, even though it
/// would collapse equal values into one identical u32. A `gidx` is keyed to ONE
/// `WeightStore`, and the multi-component weighted path REBUILDS the store after
/// graft (`compile/component.rs`: a fresh `WeightStore`, per-level values copied,
/// intern table dropped) — so an `Inline` ref persisted in a component's pair
/// list becomes a dangling index into an empty table (observed: index-out-of-
/// bounds in `WeightStore::interned_value` on the disconnected-residual canopy
/// fixture). Level slots are carried across that merge and stay valid, because
/// the merge is keyed by level. The value-sharing is not lost, only deferred:
/// `slot_prune`'s C3 value-merge collapses equal-valued slots WITHIN a level on
/// the next prune, which is the sharing the boundary parent's twin merge needs.
///
/// Plans in one sweep that fuse to EQUAL values share a single new slot
/// (`by_value`), so a sweep adds at most one slot per distinct fused value. Sound
/// for the same reason the integer count-keyed sharing is: pair lists are
/// multisets, and each shared-slot pair occurrence carries one plan's
/// contribution.
///
/// ZERO VALUES: signed weights make a fused sum of exactly 0 reachable (e.g.
/// `+a` and `−a`). That is a REAL value and gets a slot like any other — it must
/// NEVER become the bit-31 ZERO sentinel, which denotes the structural FALSE node
/// and would corrupt the Boolean structure. `slot_raw` keeps bit 31 clear by
/// construction; the assert pins it.
///
/// Returns `false`: the parent's marg-side inline marker is never raised, both
/// because this emits no inline ref at all and because `scale_weight_ref` — the
/// precedent — leaves the markers alone. They are an INTEGER-path discriminator
/// (`tag_all_marg_side_slots`, and the grouping scatter's guard).
#[inline(always)]
fn allocate_fusion_slots_weighted(
    tdd: &mut Tdd,
    v: VtreeIdx,
    plans: &mut [PlanEntry],
    slots_added: &mut usize,
) -> Result<bool, ApplyError> {
    use crate::tdd::query::semiring::{weight_key, WeightKey};
    use crate::tdd::transform::unary::marginalize::with_weight_ctx_mut;
    // Never a vtree LEAF: its column is pinned to the 3-slot `leaf_val` cache and
    // this function's `push_value` would append a 4th. Phase 2 in
    // `apply_p_fusion_inner` routes every leaf boundary to the mint-free
    // `resolve_leaf_fusion_refs_by_lookup` instead; this pins that contract at the
    // mint site.
    debug_assert!(
        !tdd.vtree.node(v).is_leaf(),
        "weighted p-fusion must never mint into a pinned leaf column (level {})",
        v.0
    );
    let mut by_value: FxHashMap<WeightKey, u32> = FxHashMap::default();
    for plan in plans.iter_mut() {
        let val: &WeightVal = plan
            .c_new_w
            .as_deref()
            .expect("weighted p-fusion plan missing fused value");
        let key = weight_key(val);
        if let Some(&existing) = by_value.get(&key) {
            plan.new_ref = MargRef::slot_raw(existing);
            continue;
        }
        let s = with_weight_ctx_mut(|ws| ws.push_value(v.idx(), val.clone()));
        let s = u32::try_from(s).map_err(|_| ApplyError::OverBudget)?;
        if s > crate::tdd::types::MARG_INLINE_MAX {
            // A slot index that would not fit the 30-bit marg-ref payload cannot
            // be referenced at all — surface it as OverBudget (routed to
            // recovery) rather than truncate a ref.
            return Err(ApplyError::OverBudget);
        }
        // Keep the weighted level's live width in sync with the store length
        // (the same bump `scale_weight_ref` performs after `push_value`).
        tdd.levels[v.idx()].retired_marg_width = s + 1;
        by_value.insert(key, s);
        plan.new_ref = MargRef::slot_raw(s);
        *slots_added += 1;
        debug_assert!(
            plan.new_ref & (1u32 << 31) == 0,
            "fused weighted marg ref must never alias the ZERO sentinel",
        );
    }
    Ok(false)
}

/// Phase 3: rewrite the parent's pair lists IN PLACE, node by node.
///
/// Fusion strictly SHRINKS every node it touches, which is what makes the
/// in-place form sound: Phase 1 emits a plan only for a group of ≥2 pairs
/// sharing one `x_idx`, and Phase 3 replaces that whole group with ONE fused
/// pair — never splits one. A node carrying `k` plans therefore drops ≥ 2k
/// pairs and gains exactly `k`, so its new list fits strictly inside its own
/// arena range. Per changed node: a write cursor trails the read cursor over
/// that range (dropping the pairs whose x-side carries a plan), then the `k`
/// fused pairs are appended at the cursor — still inside the old range, since
/// `kept + k ≤ old_len − k`. Nothing else on the level is touched, so
/// unchanged nodes (the majority on many instances) cost nothing at all.
///
/// This replaces a move-out + full-size rebuild that held the old level and a
/// fresh full-size copy of it simultaneously — a 2× transient of the whole
/// parent level (which reaches ~10^5 nodes / 10^8 pairs on dense instances)
/// arriving inside the `--mc` minimize loop, i.e. exactly when memory is
/// tightest. Do NOT stage the rewrite through a per-node intermediate either:
/// on dense levels that allocation dominates.
///
/// The shrink leaves the tail of each rewritten range unreferenced; it is
/// charged to `dead_pairs` and reclaimed by the level's own amortized arena
/// sweep at the end (the predecessor got the same effect for free by rebuilding
/// into a fresh arena, at the cost of copying the whole level every time).
///
/// `plans` must keep all of one node's entries CONTIGUOUS (Phase 1 emits them
/// in ascending `node_idx`), so we walk the plan list itself rather than the
/// whole level.
#[inline(always)]
fn rebuild_parent_level(
    tdd: &mut Tdd,
    parent: VtreeIdx,
    side: ChildSide,
    any_inline: bool,
    plans: &[PlanEntry],
) -> Result<(), ApplyError> {
    let level = &mut tdd.levels[parent.idx()];
    // (P)-inline may mint a fresh INLINE marg-side ref (bit-30 tagged) this
    // sweep; the marker for that side must be raised or the end-of-apply tagger
    // and the apply reader misread the ref as a grid coordinate (marg-canon
    // #63). Rewriting in place preserves every other flag — including the
    // marker for a side that was already inlined, and `n_tombstones` — by
    // construction; the fresh-level predecessor had to restore them by hand.
    if any_inline {
        match side {
            ChildSide::Left => level.set_marg_inlined_left(true),
            ChildSide::Right => level.set_marg_inlined_right(true),
        }
    }

    // `fused_x` maps each fused x_idx -> its new marg-side ref. Keyed on a
    // single u32 (the x-side index), NOT on (x_idx, marg) tuples: a plan
    // removes EVERY pair at its x_idx (its distinct_margs is the full marg
    // multiset there), so "this pair is fused away" == "its x_idx has a plan"
    // == `fused_x.contains_key`. This drops the former tuple-keyed `remove`
    // FxHashSet entirely — perf showed that set's construction (one insert per
    // (x,marg)) and its per-pair tuple probe were ~70% of apply_p_fusion_inner
    // self cost. Some nodes carry thousands of plans, so membership must stay a
    // hash lookup (a linear scan over fused entries is O(old_pairs * plans) and
    // regressed 94x on mc2022_track1_081).
    let mut fused_x: FxHashMap<u32, u32> = FxHashMap::default();
    // Arena slots the shrink abandons, noted ONCE below: the counter's only
    // reader is the sweep at the end, so per-node saturating adds buy nothing.
    let mut dead_acc = 0usize;

    let mut cursor = 0usize;
    while cursor < plans.len() {
        let n = plans[cursor].node_idx;
        let plan_start = cursor;
        while cursor < plans.len() && plans[cursor].node_idx == n {
            cursor += 1;
        }
        let this_plans = &plans[plan_start..cursor];

        fused_x.clear();
        // Each plan covers a distinct x_idx (Phase 1 emits one plan per
        // (node, x_idx) group), so the map holds one entry per plan — an
        // x_idx collision here would silently drop a fused pair's count.
        for plan in this_plans {
            fused_x.insert(plan.x_idx, plan.new_ref);
        }
        debug_assert_eq!(fused_x.len(), this_plans.len(), "plans must have distinct x_idx per node");

        // A plan-carrying node held ≥2 pairs (Phase 1 skips `pair_count_at < 2`),
        // so it is arena-backed — never a leaf, a tombstone, or an inline node
        // whose single pair lives in the node word.
        debug_assert!(
            level.nodes[n].is_multi(),
            "rebuild_parent_level: node {n} carries a plan but owns no arena range",
        );
        let start = level.multi_start_at(n);
        let old_len = level.multi_len_at(n);

        // Keep the un-fused pairs, compacting them onto the front of the node's
        // OWN range: `write` never overtakes `read` (it advances at most once
        // per read, from the same origin), so a kept pair only ever moves DOWN
        // onto a slot already read past.
        let mut write = start;
        for read in start..start + old_len {
            let p = level.pairs[read];
            let x_idx = match side {
                ChildSide::Right => p.left.0,
                ChildSide::Left => p.right.0,
            };
            // A pair is fused away iff its x-side index carries a plan: that
            // plan's distinct_margs is the full set of marg values at this
            // x_idx (built from this node's own pairs in Phase 1), so EVERY
            // pair at a fused x_idx is removed and replaced by one fused
            // pair. Hence membership in `fused_x` is the exact removal test
            // — no per-(x,marg) set needed.
            if !fused_x.contains_key(&x_idx) {
                level.pairs[write] = p;
                write += 1;
            }
        }
        // Append one fused pair per plan. Every read is done, and each plan
        // removed ≥2 pairs above, so `write + this_plans.len() ≤ start + old_len`
        // — the appends stay inside the node's own range and cannot reach the
        // next node's slots.
        for (&x_idx, &r_new) in fused_x.iter() {
            // `r_new` is the fully-encoded marg-side ref from Phase 2 —
            // either a tagged inline count (bit-30 set) or a bare slot
            // index (bit-30 clear), self-describing. Write it verbatim;
            // `x_idx` is the non-marg side.
            let fused = match side {
                ChildSide::Right => InputPair {
                    left: LocalNodeIdx(x_idx),
                    right: LocalNodeIdx(r_new),
                },
                ChildSide::Left => InputPair {
                    left: LocalNodeIdx(r_new),
                    right: LocalNodeIdx(x_idx),
                },
            };
            debug_assert!(write < start + old_len, "fusion must shrink the pair list");
            level.pairs[write] = fused;
            write += 1;
        }

        let new_len = write - start;
        // Re-encode via the shared epilogue (`TddLevel::reencode_shrunk_multi`,
        // also used by `contract_leaf::rewrite_level`): shrink in place, inline
        // the sole survivor, or fall back to a length-1 extended multi ALIASING
        // the node's own first slot — reusing its existing `ext` entry when the
        // node is already extended, so nothing here abandons an old `ext` slot
        // as garbage.
        dead_acc += level.reencode_shrunk_multi(n, start, old_len, new_len)?;
    }
    level.note_dead_pairs(dead_acc);
    // Reclaim the abandoned tails once they dominate the arena (the level's own
    // amortized trigger). Safe here and nowhere earlier: the rewrite is done, so
    // no pair-arena offset is held across the call — the caller obligation
    // documented on `compact_pairs_if_stale` (types/level.rs). The boundary loop
    // above holds only vtree indices, so it is unaffected.
    level.compact_pairs_if_stale();
    Ok(())
}

#[cfg(test)]
#[path = "p_fusion_fallible_tests.rs"]
mod fallible_tests;

#[cfg(test)]
#[path = "p_fusion_weighted_tests.rs"]
mod weighted_tests;
