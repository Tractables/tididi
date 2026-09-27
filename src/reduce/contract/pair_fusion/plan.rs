//! Group pairs into fusion plans and assign their resulting value references.

use crate::Engine;
use smallvec::SmallVec;
use rustc_hash::FxHashMap;

use crate::limits::{Limits, OperationError, Transient};
use crate::diagram::{ChildPair, Tdd, TddLevel, ValueRef};
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
    // A level can hold billions of pairs, so the sweep tests for a stop once
    // per stride of the nodes and pairs it reads, not once per sweep. This
    // phase only reads, so a stop leaves the diagram as it was.
    let lim = eng.limits();
    let stride = lim.reduce_poll_stride();
    let mut read = 0u64;

    // The grouping key is the raw explicit-side ref (opposite the marginal
    // `side`). Usually it is a dense index into the explicit child level — a
    // node index or, if that side is itself a marginal level referenced only
    // through slots, a slot index. On a both-marginal parent the explicit side
    // can also carry inline marginal refs, whose bit-30 tag puts the raw value
    // near 2^30. The table in `scratch` hashes the key rather than indexing by
    // it, so both kinds are ordinary keys and the table is sized by the node's
    // pair count.
    for n in 0..plevel.nodes.len() {
        let pairs = plevel.pair_count_at(n);
        read += 1 + pairs as u64;
        if read >= stride {
            read = 0;
            lim.check_stop()?;
        }
        // A group needs two pairs sharing one explicit-side ref, so a node
        // with fewer than two pairs, the common case, has nothing to group.
        if pairs < 2 {
            continue;
        }
        let node = plevel.pairs_of_idx(n);
        // A node whose pairs ascend by explicit ref holds each group as one
        // run. A wide node that does not is sorted into runs: its table
        // would be too wide for the cache, and a probe per pair would miss.
        if node.windows(2).all(|w| explicit_ref(w[0], side) <= explicit_ref(w[1], side)) {
            group_by_runs::<D>(lim, tdd, v, side, n, node, &mut out, &mut scratch.run)?;
        } else if node.len() >= SORT_MIN {
            group_by_sorting::<D>(lim, tdd, v, side, n, node, &mut out, scratch)?;
        } else {
            group_by_scatter::<D>(eng, tdd, plevel, v, side, n, &mut out, scratch)?;
        }
    }
    Ok(out)
}

/// The explicit-side ref of `pair`: the side opposite the marginal `side`.
#[inline(always)]
fn explicit_ref(pair: ChildPair, side: ChildSide) -> u32 {
    match side {
        ChildSide::Right => pair.left.0,
        ChildSide::Left => pair.right.0,
    }
}

/// The fewest pairs a node sorts into runs rather than hashes: from here its
/// table, two cells of 12 bytes per pair, outgrows a core's cache.
pub(super) const SORT_MIN: usize = 1 << 14;

/// The marginal-side ref of `pair`: the side `side` names.
#[inline(always)]
fn marginal_ref(pair: ChildPair, side: ChildSide) -> u32 {
    match side {
        ChildSide::Right => pair.right.0,
        ChildSide::Left => pair.left.0,
    }
}

/// [`group_by_scatter`] for a wide node whose pairs do not ascend: the
/// positions of its pairs are sorted stably by explicit-side ref, a least
/// significant digit radix sort, so each group is a run of positions in
/// occurrence order. A group's plan carries the position of its first pair,
/// and the plans are put in that order, which is the order the table emits
/// them in. Every pass reads and writes in sequence, where the table misses
/// the cache on nearly every pair of a wide node.
#[expect(clippy::too_many_arguments)]
pub(super) fn group_by_sorting<D: SlotValues>(
    lim: &Limits,
    tdd: &Tdd,
    v: VtreeIdx,
    side: ChildSide,
    n: usize,
    node: &[ChildPair],
    out: &mut Vec<PlanEntry<D::Value>>,
    sc: &mut PFusionScratch,
) -> Result<(), OperationError> {
    const DIGIT: u32 = 11;
    let count = node.len();
    // A node's pairs are counted in `u32` by the level encoding.
    let keyed = &mut sc.keyed;
    keyed.clear();
    lim.reserve(keyed, count)?;
    keyed.extend(node.iter().enumerate().map(|(at, &pair)| (explicit_ref(pair, side), at as u32)));
    let other = &mut sc.keyed_other;
    other.clear();
    lim.try_resize(other, count, (0u32, 0u32))?;
    let widest = keyed.iter().map(|&(x, _)| x).max().unwrap_or(0);
    let bits = u32::BITS - widest.leading_zeros();
    let mut counts = [0u32; 1 << DIGIT];
    let mut shift = 0;
    while shift < bits {
        counts.fill(0);
        for &(x, _) in keyed.iter() {
            counts[((x >> shift) as usize) & ((1 << DIGIT) - 1)] += 1;
        }
        // A digit every pair shares leaves the order as it is.
        if !counts.iter().any(|&c| c as usize == count) {
            let mut sum = 0u32;
            for c in counts.iter_mut() {
                let here = *c;
                *c = sum;
                sum += here;
            }
            for &item in keyed.iter() {
                let d = ((item.0 >> shift) as usize) & ((1 << DIGIT) - 1);
                other[counts[d] as usize] = item;
                counts[d] += 1;
            }
            std::mem::swap(keyed, other);
        }
        // Each pass only reads the diagram, so a stop between passes leaves
        // it as it was.
        lim.check_stop()?;
        shift += DIGIT;
    }
    let mut found: Vec<(u32, PlanEntry<D::Value>)> = Vec::new();
    let run = &mut sc.run;
    let mut at = 0;
    while at < count {
        let x_idx = keyed[at].0;
        let end = at + keyed[at..].iter().take_while(|&&(x, _)| x == x_idx).count();
        if end - at > 1 {
            run.clear();
            lim.reserve(run, end - at)?;
            run.extend(keyed[at..end].iter().map(|&(_, pos)| marginal_ref(node[pos as usize], side)));
            lim.try_push(&mut found, (keyed[at].1, PlanEntry {
                node_idx: n,
                x_idx,
                value: D::sum_refs(tdd, v, run),
                new_ref: u32::MAX,
            }))?;
        }
        at = end;
    }
    // First positions are distinct, so the order is total.
    found.sort_unstable_by_key(|&(first, _)| first);
    lim.reserve(out, found.len())?;
    out.extend(found.into_iter().map(|(_, plan)| plan));
    Ok(())
}

/// [`group_by_scatter`] for a node whose pairs ascend by explicit-side ref:
/// each group is a run of them, and the runs come in the order of their first
/// occurrences, which is the order the table emits its groups in. No table is
/// touched, so a wide node costs one read of its pairs.
#[expect(clippy::too_many_arguments)]
pub(super) fn group_by_runs<D: SlotValues>(
    lim: &Limits,
    tdd: &Tdd,
    v: VtreeIdx,
    side: ChildSide,
    n: usize,
    node: &[ChildPair],
    out: &mut Vec<PlanEntry<D::Value>>,
    run: &mut Vec<u32>,
) -> Result<(), OperationError> {
    let mut at = 0;
    while at < node.len() {
        let x_idx = explicit_ref(node[at], side);
        let end = at + node[at..].iter().take_while(|&&p| explicit_ref(p, side) == x_idx).count();
        if end - at > 1 {
            run.clear();
            lim.reserve(run, end - at)?;
            run.extend(node[at..end].iter().map(|&p| marginal_ref(p, side)));
            lim.try_push(out, PlanEntry {
                node_idx: n,
                x_idx,
                value: D::sum_refs(tdd, v, run),
                new_ref: u32::MAX,
            })?;
        }
        at = end;
    }
    Ok(())
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
pub(super) fn group_by_scatter<D: SlotValues>(
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
    // The node probes only the first `want` cells, a window of its own size,
    // not the whole table: after one wide node the table is as wide as it,
    // and a narrow node hashed across all of it would miss the cache on
    // nearly every pair. Cells past the window keep older stamps unread.
    let mask = want - 1;
    let hash_shift = 32 - want.trailing_zeros();
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
        // than a process abort, and charge the grown capacity to the byte
        // meter as `Limits::reserve` would. Guarding on `capacity()` keeps
        // every growth fallible for the rare large x-group.
        let g = &mut sc.groups[slot];
        if g.len() == g.capacity() {
            let before = g.capacity();
            g.try_reserve(1).map_err(|_| OperationError::OverBudget)?;
            let grown = g.capacity().saturating_sub(before) * std::mem::size_of::<u32>();
            lim.charge_bytes(grown as u64)?;
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
    let lim = eng.limits();
    let mut any_inline = false;
    // Charged for the sweep and handed back on every exit: one entry per
    // plan beyond whatever the domain seeded.
    let mut by_value: Transient<'_, FxHashMap<D::Key, u32>> = Transient::new(lim, FxHashMap::default());
    D::seed(lim, tdd, v, &mut by_value)?;
    lim.reserve_map(&mut by_value, plans.len())?;
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
