//! Phase 1: grouping a parent level's pairs into per-(node, x) fusion plans.

use crate::engine::Engine;
use smallvec::SmallVec;

use crate::limits::ApplyError;
use crate::diagram::{Tdd, TddLevel};
use crate::vtree::VtreeIdx;

use crate::diagram::ChildSide;
use crate::value::slots::SlotValues;

use super::super::scratch::{GroupCell, PFusionScratch};
use super::PlanEntry;

/// Phase 1: full-scan the parent level's nodes; collect per-(node, `x_idx`) groups
/// with > 1 distinct marginal-side index. Compute `c_new` for each.
///
/// The output plan list is in ascending `node_idx` order (Phase 3 cursor-walk
/// relies on this). Empty return means no boundary at this level is fusable.
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

    // The grouping key is the raw explicit-side ref (opposite the marginal
    // `side`). Usually it is a dense index into the explicit child level — a
    // node index or, if that side is itself a marginal level referenced only
    // through slots, a slot index. On a both-marginal parent the explicit side
    // can also carry inline marginal refs, whose bit-30 tag puts the raw value
    // near 2^30. The table in `scratch` hashes the key rather than indexing by
    // it, so both kinds are ordinary keys and the table is sized by the node's
    // pair count.
    for n in 0..plevel.nodes.len() {
        group_node_pairs::<D>(eng, plevel, n, side, tdd, v, &mut out, scratch)?;
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
    group_by_scatter::<D>(eng, plevel, n, side, tdd, v, out, sc)
}

/// Group through the generation-stamped table in `sc`, keyed on the raw
/// explicit-side ref.
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
    // Bump the generation instead of clearing the cells (O(1) per-node
    // reset). On u32 wrap, zero the stamps and restart at 1 (0 is the
    // "never stamped" sentinel and must never equal a live `gen`).
    sc.generation = match sc.generation.checked_add(1) {
        Some(g) => g,
        None => {
            for c in sc.cells.iter_mut() {
                c.stamp = 0;
            }
            1
        }
    };
    let generation = sc.generation;
    sc.touched.clear();
    // At least two cells per pair, so the table is never more than half full
    // and a probe always ends at an unstamped cell. Grown on demand and
    // fallibly — an OOM here becomes OverBudget, not a process abort. New
    // cells are stamped 0 ≠ generation (which is ≥ 1) and read as empty.
    let want = (2 * plevel.pair_count_at(n)).next_power_of_two();
    if sc.cells.len() < want {
        lim.try_resize(&mut sc.cells, want, GroupCell::default())?;
    }
    let mask = sc.cells.len() - 1;
    let hash_shift = 32 - sc.cells.len().trailing_zeros();
    for p in plevel.pairs_of_idx(n) {
        let (x_idx, marginal_idx) = match side {
            ChildSide::Right => (p.left.0, p.right.0),
            ChildSide::Left => (p.right.0, p.left.0),
        };
        // Fibonacci hashing into the table, then linear probing.
        let mut h = (x_idx.wrapping_mul(0x9E37_79B1) >> hash_shift) as usize;
        let slot = loop {
            let cell = sc.cells[h];
            if cell.stamp != generation {
                // First occurrence of this x for this node: open a group.
                let slot = sc.touched.len();
                sc.cells[h] = GroupCell { stamp: generation, key: x_idx, slot: slot as u32 };
                lim.try_push(&mut sc.touched, x_idx)?;
                // Reuse a retired group slot (keeps its grown capacity) or
                // allocate one only when this node needs more distinct
                // x-groups than any prior node.
                if slot < sc.groups.len() {
                    sc.groups[slot].clear();
                } else {
                    lim.try_push(&mut sc.groups, SmallVec::new())?;
                }
                break slot;
            }
            if cell.key == x_idx {
                break cell.slot as usize;
            }
            h = (h + 1) & mask;
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
