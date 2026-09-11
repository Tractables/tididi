//! Phase 1: grouping a parent level's pairs into per-(node, x) fusion plans.

use crate::engine::Engine;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use crate::limits::ApplyError;
use crate::diagram::{Tdd, TddLevel};
use crate::vtree::VtreeIdx;

use crate::diagram::ChildSide;
use crate::value::slots::SlotValues;

use super::super::scratch::PFusionScratch;
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
/// The value domain `D` sums each group; the grouping walk is shared by both
/// instantiations, and the domain can never vary per node or per pair.
#[inline(always)]
pub(super) fn collect_fusion_plans<D: SlotValues>(
    eng: &Engine,
    tdd: &Tdd,
    parent: VtreeIdx,
    v: VtreeIdx,
    side: ChildSide,
    scratch: &mut PFusionScratch,
) -> Result<Vec<PlanEntry<D::Value>>, ApplyError> {
    let plevel = &tdd.levels[parent.idx()];
    let mut out: Vec<PlanEntry<D::Value>> = Vec::new();

    // Grouping key `x_idx` is the explicit-side ref (opposite the marginal
    // `side`). On the common path it is a dense index into the explicit child
    // level — a Boolean node index (`< nodes.len()`) or, if that side is itself
    // a marginal level referenced only through slots, a slot index
    // (`< marginal_counts.len()`). Either is small and dense, so we group with a
    // generation-stamped dense scatter (`scratch`) instead of a hashmap.
    //
    // One case escapes that: a both-marginal parent (this parent has two
    // marginal-child boundaries, one per marginal child) may carry inline marginal
    // refs on the explicit side, whose bit-30 `MARGINAL_OVERFLOW_TAG` puts `x_idx`
    // outside the dense index space (≈2^30). Indexing a dense array by such a
    // value would demand a multi-GiB allocation, so when the explicit side's
    // inline marker is set we fall back to the opaque-key hashmap for this
    // boundary (rare). Both paths feed one `emit` closure below, so the
    // soundness-critical count/plan construction is single-source.
    //
    // The weighted arm uses the same guard and the same scatter. Nothing in weight
    // context mints an inline ref: the weighted mint and the leaf lookup emit
    // `slot_raw`, and `emit_marginal_side_slots` (the only bit-30 writer) needs
    // integer `marginal_counts`. Weighted marginal-side refs are bare slots end to
    // end. The markers are nonetheless set in weight context —
    // `tag_all_marginal_side_slots` runs after every apply and raises them for any
    // marginal-child side — so `explicit_inline` is a conservative over-estimate
    // here, which is the safe direction: it can only route a boundary to the
    // hashmap that the scatter could have handled.
    let explicit_inline = match side {
        ChildSide::Right => plevel.marginal_inlined_left(),
        ChildSide::Left => plevel.marginal_inlined_right(),
    };
    let use_scatter = !explicit_inline;

    for n in 0..plevel.nodes.len() {
        group_node_pairs::<D>(eng, plevel, n, side, use_scatter, tdd, v, &mut out, scratch)?;
    }
    Ok(out)
}

/// Shared per-group emission. `margs` is the whole occurrence multiset of
/// marginal-side refs at this x (no dedup): with value-keyed slot sharing a
/// node's pair list may legitimately contain `(x, M)` more than once, each
/// occurrence carrying one earlier plan's `c(M)` contribution, so the
/// fused value sums over occurrences, not over distinct M. Distinct marginal
/// nodes are disjoint Z-sets, so their values add — `c(L)·v1 + c(L)·v2 + … =
/// c(L)·(v1+v2+…)`, the fusion invariant, in whichever domain `D` is.
fn emit_fusion_plan<D: SlotValues>(
    eng: &Engine,
    tdd: &Tdd,
    v: VtreeIdx,
    n: usize,
    x_idx: u32,
    margs: &[u32],
    out: &mut Vec<PlanEntry<D::Value>>,
) -> Result<(), ApplyError> {
    eng.limits().try_push(out, PlanEntry {
        node_idx: n,
        x_idx,
        value: D::sum_refs(tdd, v, margs),
        new_ref: u32::MAX,
    })
}


/// Group one parent node's pairs by their explicit-side index and emit a plan
/// for every group holding more than one marginal-side ref.
// The contraction scratch buffers are passed separately so they can be
// borrowed independently of the diagram they index into.
#[allow(clippy::too_many_arguments)]
fn group_node_pairs<D: SlotValues>(
    eng: &Engine,
    plevel: &TddLevel,
    n: usize,
    side: ChildSide,
    use_scatter: bool,
    tdd: &Tdd,
    v: VtreeIdx,
    out: &mut Vec<PlanEntry<D::Value>>,
    sc: &mut PFusionScratch,
) -> Result<(), ApplyError> {
    if plevel.nodes[n].is_leaf() {
        return Ok(());
    }
    // A same-x fusion group needs ≥2 pairs sharing one x_idx, which
    // requires the node to hold ≥2 pairs at all. Single-pair nodes
    // (the common case) can never fuse — skip them before any
    // grouping work; most scanned nodes produce no plan.
    if plevel.pair_count_at(n) < 2 {
        return Ok(());
    }
    if use_scatter {
        group_by_scatter::<D>(eng, plevel, n, side, tdd, v, out, sc)
    } else {
        group_by_hashmap::<D>(eng, plevel, n, side, tdd, v, out)
    }
}

/// Group by a generation-stamped dense scatter over the explicit-side index.
// The contraction scratch buffers are passed separately so they can be
// borrowed independently of the diagram they index into.
#[allow(clippy::too_many_arguments)]
fn group_by_scatter<D: SlotValues>(
    eng: &Engine,
    plevel: &TddLevel,
    n: usize,
    side: ChildSide,
    tdd: &Tdd,
    v: VtreeIdx,
    out: &mut Vec<PlanEntry<D::Value>>,
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
        let (x_idx, marginal_idx) = match side {
            ChildSide::Right => (p.left.0, p.right.0),
            ChildSide::Left => (p.right.0, p.left.0),
        };
        let xu = x_idx as usize;
        // Grow the per-x arrays on demand to `max(x)+1` (fallibly — an
        // OOM here becomes OverBudget, not a process abort). New entries
        // are 0 ≠ generation (which is ≥ 1) so they read as "unstamped".
        // `stamp` and `slot_of_x` are kept the same length: the OR guard
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
            lim.try_push(&mut sc.touched, x_idx)?;
            // Reuse a retired group slot (keeps its grown capacity) or
            // allocate one only when this node needs more distinct
            // x-groups than any prior node.
            if slot < sc.groups.len() {
                sc.groups[slot].clear();
            } else {
                lim.try_push(&mut sc.groups, SmallVec::new())?;
            }
            slot
        } else {
            sc.slot_of_x[xu] as usize
        };
        // Fallible push for SmallVec: while the current buffer, inline
        // or heap, has spare capacity, push cannot fail. At capacity —
        // both the inline-to-heap spill and every subsequent heap regrow — use
        // try_reserve so allocation failure becomes OverBudget rather
        // than a process abort. Guarding on `capacity()` keeps every
        // growth fallible for the rare large x-group.
        let g = &mut sc.groups[slot];
        if g.len() == g.capacity() {
            g.try_reserve(1).map_err(|_| ApplyError::OverBudget)?;
        }
        g.push(marginal_idx);
    }
    // Emit in first-occurrence (touched) order. slot i ↔ touched[i].
    for i in 0..sc.touched.len() {
        if sc.groups[i].len() <= 1 {
            // A single pair at this x cannot fuse.
            continue;
        }
        emit_fusion_plan::<D>(eng, tdd, v, n, sc.touched[i], &sc.groups[i], out)?;
    }
    Ok(())
}

/// Group through an opaque-key hashmap — the fallback when the explicit side
/// carries inline marginal refs, which are outside the dense index space.
fn group_by_hashmap<D: SlotValues>(
    eng: &Engine,
    plevel: &TddLevel,
    n: usize,
    side: ChildSide,
    tdd: &Tdd,
    v: VtreeIdx,
    out: &mut Vec<PlanEntry<D::Value>>,
) -> Result<(), ApplyError> {
    // ── Fallback: opaque-key hashmap (explicit side carries inline marginal
    // refs; see the `use_scatter` note). Byte-identical grouping to the
    // pre-scatter path; `x_idx` is treated as an opaque key.
    let mut by_x: FxHashMap<u32, SmallVec<[u32; 4]>> = FxHashMap::default();
    for p in plevel.pairs_of_idx(n) {
        let (x_idx, marginal_idx) = match side {
            ChildSide::Right => (p.left.0, p.right.0),
            ChildSide::Left => (p.right.0, p.left.0),
        };
        let sv = by_x.entry(x_idx).or_default();
        if sv.len() == sv.capacity() {
            sv.try_reserve(1).map_err(|_| ApplyError::OverBudget)?;
        }
        sv.push(marginal_idx);
    }
    for (x_idx, margs) in by_x.drain() {
        if margs.len() <= 1 {
            continue;
        }
        emit_fusion_plan::<D>(eng, tdd, v, n, x_idx, &margs, out)?;
    }
    Ok(())
}
