//! Group pairs into fusion plans and assign their resulting value references.

use crate::Engine;
use smallvec::SmallVec;
use rustc_hash::FxHashMap;

use crate::limits::OperationError;
use crate::diagram::{Tdd, TddLevel, ValueRef};
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
pub(super) fn collect_fusion_plans<D: SlotValues>(
    eng: &Engine,
    tdd: &Tdd,
    parent: VtreeIdx,
    v: VtreeIdx,
    side: ChildSide,
    scratch: &mut PFusionScratch,
) -> Result<Vec<PlanEntry<D::Value>>, OperationError> {
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
        // A group needs two pairs sharing one explicit-side ref, so a node
        // with fewer than two pairs, the common case, has nothing to group.
        if plevel.pair_count_at(n) < 2 {
            continue;
        }
        group_by_scatter::<D>(eng, tdd, plevel, v, side, n, &mut out, scratch)?;
    }
    Ok(out)
}

/// Group one parent node's pairs by their explicit-side ref through the
/// generation-stamped table in `sc`, and emit a plan for every group holding
/// more than one marginal-side ref.
///
/// A plan's value sums the group's whole occurrence multiset of marginal-side
/// refs (no dedup): with value-keyed slot sharing a node's pair list may
/// legitimately contain `(x, M)` more than once, each occurrence carrying one
/// earlier plan's `c(M)` contribution. Distinct marginal nodes are disjoint
/// Z-sets, so their values add — `c(L)·v1 + c(L)·v2 + … = c(L)·(v1+v2+…)`,
/// the fusion invariant, in whichever domain `D` is.
#[expect(clippy::too_many_arguments)]
fn group_by_scatter<D: SlotValues>(
    eng: &Engine,
    tdd: &Tdd,
    plevel: &TddLevel,
    v: VtreeIdx,
    side: ChildSide,
    n: usize,
    out: &mut Vec<PlanEntry<D::Value>>,
    sc: &mut PFusionScratch,
) -> Result<(), OperationError> {
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
        // `try_reserve` so allocation failure becomes OverBudget rather
        // than a process abort. Guarding on `capacity()` keeps every
        // growth fallible for the rare large x-group.
        let g = &mut sc.groups[slot];
        if g.len() == g.capacity() {
            g.try_reserve(1).map_err(|_| OperationError::OverBudget)?;
        }
        g.push(marginal_idx);
    }
    // Emit in first-occurrence (touched) order. slot i ↔ touched[i].
    for i in 0..sc.touched.len() {
        if sc.groups[i].len() <= 1 {
            // A single pair at this x cannot fuse.
            continue;
        }
        lim.try_push(out, PlanEntry {
            node_idx: n,
            x_idx: sc.touched[i],
            value: D::sum_refs(tdd, v, &sc.groups[i]),
            new_ref: u32::MAX,
        })?;
    }
    Ok(())
}

/// Phase 2: encode each plan's fused value as a marginal-side ref; fill
/// `plan.new_ref`.
///
/// Returns `true` if at least one plan emitted an inline ref (the parent
/// level's marginal-side inline marker must then be raised in Phase 3).
///
/// Slot identity is value-keyed: a plan whose value matches another plan in
/// the sweep — or, where the domain seeds the map, an existing slot — shares
/// that slot rather than minting one. Sound because pair lists are multisets:
/// each shared-slot pair occurrence carries one plan's contribution.
///
/// Never a pinned leaf: the caller resolves that boundary by lookup instead.
pub(super) fn allocate_fusion_slots<D: SlotValues>(
    eng: &Engine,
    tdd: &mut Tdd,
    v: VtreeIdx,
    plans: &mut [PlanEntry<D::Value>],
) -> Result<bool, OperationError> {
    let mut any_inline = false;
    let mut by_value: FxHashMap<D::Key, u32> = FxHashMap::default();
    D::seed(tdd, v, &mut by_value);
    for plan in plans.iter_mut() {
        if let Some(raw) = D::inline_ref(&plan.value) {
            plan.new_ref = raw;
            any_inline = true;
            continue;
        }
        let key = D::key(&plan.value);
        if let Some(&existing) = by_value.get(&key) {
            plan.new_ref = ValueRef::slot_raw(existing);
            continue;
        }
        let s = D::push_slot(eng, tdd, v, plan.value.clone())?;
        by_value.insert(key, s);
        plan.new_ref = ValueRef::slot_raw(s);
    }
    Ok(any_inline)
}
