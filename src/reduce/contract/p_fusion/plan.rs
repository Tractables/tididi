//! Phase 1: grouping a parent level's pairs into per-(node, x) fusion plans.

use crate::engine::Engine;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use crate::error::ApplyError;
use crate::diagram::WeightVal;
use crate::diagram::{BigSide, ValueRef, Tdd, TddLevel};
use crate::diagram::WeightStore;
use crate::vtree::VtreeIdx;

use crate::diagram::ChildSide;
use crate::reduce::slots::{CountKey, sum_marginal_counts};

use super::super::scratch::PFusionScratch;
use super::slots::sum_marginal_weights;
use super::PlanEntry;

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
/// `WEIGHTED` = weighted pair fusion: the fused value is summed out of the
/// external `WeightStore` semiring instead of the integer count store. In that
/// mode the level's integer `marginal_counts` is NEVER touched (it is `None`
/// in weight context).
#[inline(always)]
pub(super) fn collect_fusion_plans<const WEIGHTED: bool>(
    eng: &Engine,
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
    let ws = tdd.weights.as_ref();
    let no_counts: [u128; 0] = [];
    let counts: &[u128] = if WEIGHTED {
        &no_counts
    } else {
        vlevel.marginal_counts.as_ref().unwrap()
    };
    let big = if WEIGHTED { None } else { vlevel.marginal_counts_big.as_ref() };
    let values = FusionValues { ws, v, counts, big };
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
    // markers, and mints `ValueRef::Inline` refs without raising them" — BOTH
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

    for n in 0..plevel.nodes.len() {
        group_node_pairs::<WEIGHTED>(eng, plevel, n, side, use_scatter, &values, &mut out, scratch)?;
    }
    Ok(out)
}


/// Where a plan's fused value is read from: the weight store in weighted
/// mode, the level's integer count store otherwise.
struct FusionValues<'a> {
    ws: Option<&'a WeightStore>,
    v: VtreeIdx,
    counts: &'a [u128],
    big: Option<&'a BigSide>,
}

/// Shared per-group emission. `margs` is the FULL occurrence MULTISET of
/// marg-side refs at this x (NO dedup): with count-keyed slot sharing a
/// node's pair list may legitimately contain `(x, M)` more than once, each
/// occurrence carrying one historical plan's `c(M)` contribution, so the
/// fused count sums over OCCURRENCES, not distinct M. Distinct marginal
/// nodes are disjoint Z-sets, so their values add — `c(L)·v1 + c(L)·v2 + … =
/// c(L)·(v1+v2+…)`, the fusion invariant (carried into the semiring in weighted
/// mode; `c_new` is a dummy there).
/// Weighted: the fused value is the semiring sum over the SAME occurrence
/// multiset; `c_new` is an unread dummy on that arm (Phase 2 reads
/// `c_new_w`). Integer: unchanged.
fn emit_fusion_plan<const WEIGHTED: bool>(
    eng: &Engine,
    values: &FusionValues<'_>,
    n: usize,
    x_idx: u32,
    margs: &[u32],
    out: &mut Vec<PlanEntry>,
) -> Result<(), ApplyError> {
        let lim = eng.limits();
        let (c_new, c_new_w) = if WEIGHTED {
            let ws = values.ws.expect("weighted p-fusion without a weight store");
            (CountKey::Small(0), Some(Box::new(sum_marginal_weights(ws, values.v, margs))))
        } else {
            (sum_marginal_counts(values.counts, values.big, margs), None)
        };
        lim.try_push(out, PlanEntry {
            node_idx: n,
            x_idx,
            distinct_margs: margs.to_vec(),
            c_new,
            c_new_w,
            new_ref: u32::MAX,
        })

}


/// Group one parent node's pairs by their explicit-side index and emit a plan
/// for every group holding more than one marg-side ref.
// The contraction scratch buffers are passed separately so they can be
// borrowed independently of the diagram they index into.
#[allow(clippy::too_many_arguments)]
fn group_node_pairs<const WEIGHTED: bool>(
    eng: &Engine,
    plevel: &TddLevel,
    n: usize,
    side: ChildSide,
    use_scatter: bool,
    values: &FusionValues<'_>,
    out: &mut Vec<PlanEntry>,
    sc: &mut PFusionScratch,
) -> Result<(), ApplyError> {
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
        group_by_scatter::<WEIGHTED>(eng, plevel, n, side, values, out, sc)
    } else {
        group_by_hashmap::<WEIGHTED>(eng, plevel, n, side, values, out)
    }
}

/// Group by a generation-stamped dense scatter over the explicit-side index.
fn group_by_scatter<const WEIGHTED: bool>(
    eng: &Engine,
    plevel: &TddLevel,
    n: usize,
    side: ChildSide,
    values: &FusionValues<'_>,
    out: &mut Vec<PlanEntry>,
    sc: &mut PFusionScratch,
) -> Result<(), ApplyError> {
    let lim = eng.limits();
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
            !WEIGHTED || !crate::diagram::ValueRef::is_inline_raw(x_idx),
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
            lim.try_resize(&mut sc.stamp, xu + 1, 0u32)?;
            lim.try_resize(&mut sc.slot_of_x, xu + 1, 0u32)?;
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
        emit_fusion_plan::<WEIGHTED>(eng, values, n, sc.touched[i], &sc.groups[i], out)?;
    }
    Ok(())
}

/// Group through an opaque-key hashmap — the fallback when the explicit side
/// carries inline marg refs, which are outside the dense index space.
fn group_by_hashmap<const WEIGHTED: bool>(
    eng: &Engine,
    plevel: &TddLevel,
    n: usize,
    side: ChildSide,
    values: &FusionValues<'_>,
    out: &mut Vec<PlanEntry>,
) -> Result<(), ApplyError> {
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
        emit_fusion_plan::<WEIGHTED>(eng, values, n, x_idx, &margs, out)?;
    }
    Ok(())
}

/// Weighted Phase 2 at a vtree LEAF boundary: resolve each plan's fused value to
/// a slot the PINNED column already holds, and DROP the plans it does not.
///
/// The mint-free half of weighted pair fusion. [`allocate_fusion_slots_weighted`]
/// represents a fused value by appending a slot; at a leaf that is forbidden —
/// the column is the immutable, label-ordered 3-slot `leaf_val` cache every other
/// `Tdd` of the compile aliases by bare leaf-LABEL refs (THE PIN INVARIANT,
/// `marginalize::marginalize_leaf_weighted`). What IS available is the column's
/// own values, and a fusion sum lands on them far more often than a generic lookup
/// would suggest: `(x,Pos) + (x,Neg)` sums to `w⁺+w⁻`, which IS the One slot BY
/// DEFINITION — for every weight table, asymmetric included, which is the lever
/// equal-value ref canonicalization cannot reach — and after that canonicalization
/// `(x,Pos) + (x,Pos)` at `w⁺ = w⁻` sums to `2w⁺ = w⁺+w⁻ = One` as well.
///
/// A MISS drops the plan, which leaves that group's pairs exactly as they were:
/// Phase 3 rewrites only the x-indices a SURVIVING plan names, so an untouched
/// group is a no-op there. The cost is a size residual (one un-fused fusion redex),
/// never a wrong value — and no invariant checker objects, because the
/// fusion-saturation checks (`check::marg::check_no_fusion_redexes`,
/// `debug_assert_p_saturated`) return early in weight context. Order is preserved
/// by `retain_mut`, so Phase 3's ascending-`node_idx` precondition survives.
///
/// The column is never written and `weight_width` is never bumped: the
/// level stays exactly `LEAF_WIDTH` wide, and because
/// `marginalize::find_leaf_slot_by_value` scans ascending, each fused ref is the
/// CANONICAL (smallest) slot of its value class — which is what pin check #4
/// demands of every leaf-side ref.
#[inline(always)]
pub(super) fn resolve_leaf_fusion_refs_by_lookup(tdd: &Tdd, v: VtreeIdx, plans: &mut Vec<PlanEntry>) {
    use crate::marginal::find_leaf_slot_by_value;
    debug_assert!(
        tdd.vtree.node(v).is_leaf(),
        "leaf-boundary fusion resolution called on internal level {}",
        v.0
    );
    {
        let ws = tdd.weight_store();
        // Exact domain only: `weight_key` equality is value equality there, while
        // a `WeightKey::Log` compares `f64` bit patterns. The caller's
        // caller's Log-domain decline already excludes it — this pins that.
        debug_assert!(
            !ws.is_log(),
            "leaf sum-lookup fold reached in the Log domain (the fusion gate must exclude it)"
        );
        plans.retain_mut(|plan| {
            let val: &WeightVal = plan
                .c_new_w
                .as_deref()
                .expect("weighted p-fusion plan missing fused value");
            match find_leaf_slot_by_value(ws, v.idx(), val) {
                Some(slot) => {
                    plan.new_ref = ValueRef::slot_raw(slot);
                    true
                }
                None => false,
            }
        });
    }
}
