//! Prune phase: remove diagram nodes not reachable from the output.
//!
//! Marks reachability top-down from the output node, then compacts each level
//! bottom-up while remapping child references. The remap is monotone (preserves
//! order), so sorted pair lists remain sorted after remapping. Levels that lost
//! a node go onto the contract worklists; `reduce` runs the contraction.

use crate::diagram::{EncodedChildRef, NodeIdx, NodeKind, Tdd};

use crate::Engine;

use crate::vtree::VtreeIdx;
use crate::limits::OperationError;

/// `remap` entry for a slot the pass-1 walk never reached — the whole
/// reachability bitmap, folded into the remap array (they are indexed
/// identically and pass 2 never needs the mark after it has written the
/// compacted index into the same slot). `u32::MAX` can never collide with a
/// real remap value: both a compacted index and an identity index are `< width`,
/// and a level of `u32::MAX` nodes cannot be allocated.
const UNREACHED: u32 = u32::MAX;

/// Placeholder pass 1 stamps on a reached slot; pass 2 overwrites it with the
/// slot's compacted index. Only its inequality with `UNREACHED` is meaningful.
const REACHED: u32 = 0;

/// Remove nodes not reachable from the output.
///
/// Marks reachability top-down, then compacts each level bottom-up, remapping
/// child references in the same pass. The remap is monotone, so sorted pair
/// lists stay sorted. Every level that lost a node is pushed onto the contract
/// worklists (`seed_dirty_levels`), so the caller needs no reseed.
///
/// # Errors
///
/// Returns `Err(OperationError::OverBudget)` if the reservation of `remap`, the
/// one buffer proportional to the summed level width, is refused. It is taken
/// before any mutation of `tdd`, so the diagram is then untouched.
pub(crate) fn prune_unreachable(eng: &Engine, tdd: &mut Tdd) -> Result<(), OperationError> {
    let num_nodes = tdd.vtree.num_nodes();

    // `ZERO` sentinel: the entire diagram computes ⊥ (UNSAT). No nodes are reachable.
    if tdd.is_zero() {
        for level in &mut tdd.levels {
            level.nodes.clear();
            level.n_tombstones = 0;
        }
        return Ok(());
    }

    let pool = eng.reduce_scratch();
    let mut level_base = pool.prune_level_base.take();
    let mut remap = pool.prune_remap.take();

    // Flat offset table: level t occupies remap[level_base[t]..level_base[t+1]].
    // Use `reference_slot_count()` so leaf levels get `LEAF_WIDTH` slots for marginal nodes.
    if level_base.len() < num_nodes + 1 {
        level_base.resize(num_nodes + 1, 0usize);
    }
    level_base[0] = 0;
    for i in 0..num_nodes {
        level_base[i + 1] = level_base[i] + tdd.reference_slot_count(VtreeIdx(i as u32));
    }
    let total = level_base[num_nodes];

    // The exact form leaves the Vec untouched on failure, so the pooled buffer
    // can be returned; through the engine's limits so the reservation is
    // charged against the byte budget.
    let need_remap = total.saturating_sub(remap.len());
    if eng.limits().reserve_exact(&mut remap, need_remap).is_err() {
        pool.prune_level_base.put(level_base);
        pool.prune_remap.put_bounded(eng.limits(), remap);
        return Err(OperationError::OverBudget);
    }

    // Reset the marks. Split so the grown tail is initialized once, by `resize`,
    // rather than written by `resize` and again by the fill.
    let warm = remap.len().min(total);
    remap[..warm].fill(UNREACHED);
    if remap.len() < total {
        remap.resize(total, UNREACHED);
    }
    // ── Pass 1 (top-down): mark reachable nodes ──────────────────────────
    classic_mark(tdd, &level_base, &mut remap[..total]);

    let vtree = std::sync::Arc::clone(&tdd.vtree);
    let level_dirty = compact_levels(tdd, &vtree, &level_base, &mut remap[..total], num_nodes);
    seed_dirty_levels(tdd, &level_dirty);

    tdd.output.local = NodeIdx(
        remap[level_base[tdd.output.vtree.idx()] + tdd.output.local.idx()],
    );

    // Both passes cross every reference slot, and neither can stop partway:
    // `compact_levels` rewrites levels in place, so a diagram abandoned mid-pass
    // has some levels compacted and others still naming their old indices.
    // Charge the walk so a work budget sees it; the cancellation test belongs to
    // the callers, between prunes.
    eng.limits().charge_work(2 * total as u64);

    pool.prune_level_base.put(level_base);
    pool.prune_remap.put_bounded(eng.limits(), remap);

    Ok(())
}

/// Pass 2 (bottom-up): compact unreachable nodes, overwriting the pass-1 marks
/// in `remap` with each surviving node's compacted index and rewriting child
/// references as it goes. Returns the per-level "this level lost a node" flags.
fn compact_levels(
    tdd: &mut Tdd,
    vtree: &crate::vtree::Vtree,
    level_base: &[usize],
    remap: &mut [u32],
    num_nodes: usize,
) -> Vec<bool> {
    // Overwrite the pass-1 marks with the remap (old index → new index) in
    // place, rewrite child references, and drop unreachable nodes in one
    // pass. A slot's mark is only read before its own remap value is written,
    // and `UNREACHED` survives on every slot that is never remapped, so
    // `remap[s] != UNREACHED` stays the reachability predicate throughout.
    //
    // `level_dirty[t]` records whether level `t` lost a node. A level's
    // child-ref rewrite is needed only when a child level shrank (otherwise
    // that child's remap is the identity), and its retain only when the level
    // itself shrank, so all-reachable levels skip both walks.
    let mut level_dirty = vec![false; num_nodes];

    // Bottom-up topological order, so a level's child levels are remapped
    // before it rewrites its child references; raw indices do not encode
    // parent/child order on a rotated vtree.
    for v in vtree.bottomup_slice() {
        let t_idx = v.idx();
        let base = level_base[t_idx];
        let eff_width = tdd.reference_slot_count(VtreeIdx(t_idx as u32));

        if eff_width == 0 {
            continue;
        }

        // Leaf levels: marginal nodes, always identity remap.
        if vtree.node(VtreeIdx(t_idx as u32)).is_leaf() {
            for i in 0..eff_width {
                remap[base + i] = i as u32;
            }
            continue;
        }

        let width = tdd.levels[t_idx].slot_count();
        // A marginal level keeps its content in a value store that parents
        // reference by store-relative slot index, and such refs may be minted
        // after this prune (contract's inline-to-slot redirect). The walk marks
        // only the slots referenced right now, so compacting the store here
        // could drop a slot a later ref reads. Marginal stores keep their full
        // length; `prune_value_slots` collects their orphans.
        let mut new_idx = 0u32;
        if tdd.levels[t_idx].is_marginal() {
            for i in 0..width {
                remap[base + i] = i as u32;
            }
            new_idx = width as u32;
        } else {
            for i in 0..width {
                if remap[base + i] != UNREACHED {
                    remap[base + i] = new_idx;
                    new_idx += 1;
                }
            }
        }
        // Did this level lose any node? (remap stays valid identity either way.)
        let this_dirty = (new_idx as usize) != width;
        level_dirty[t_idx] = this_dirty;

        if tdd.levels[t_idx].is_marginal() {
            // The identity remap above forces `this_dirty == false` here.
            debug_assert!(!this_dirty);
            continue;
        }

        rewrite_child_refs(tdd, VtreeIdx(t_idx as u32), base, width, level_base, &level_dirty, remap);

        // Compact the node Vec in place, O(width) with no allocation. The
        // pair arena is not swept here; a pruned level keeps its arena slack
        // until `shrink_arrays`.
        if this_dirty {
            let mut i = 0;
            tdd.levels[t_idx].nodes.retain(|_| {
                let keep = remap[base + i] != UNREACHED;
                i += 1;
                keep
            });
            // Tombstones are unreferenced, hence unreachable, hence dropped by
            // the retain above.
            tdd.levels[t_idx].n_tombstones = 0;
        }
    }
    level_dirty
}

/// Rewrite level `t`'s child references through its child levels' remaps.
/// `base` and `width` are `t`'s own block in `remap`; `level_base` says where
/// each level's block starts, `level_dirty` which levels shrank.
///
/// A no-op unless a child level actually shrank: otherwise both child remaps
/// are the identity and every write would store a value back onto itself.
///
/// The three tables are parameters rather than one struct: a slice loaded out
/// of a struct loses the aliasing facts a slice parameter carries, and the
/// rewrite loop then re-reads the table pointers on every node.
fn rewrite_child_refs(
    tdd: &mut Tdd,
    t: VtreeIdx,
    base: usize,
    width: usize,
    level_base: &[usize],
    level_dirty: &[bool],
    remap: &[u32],
) {
    let t_idx = t.idx();
    let (left, right) = tdd.vtree.children(t);
    let left_grid_base = level_base[left.idx()];
    let right_grid_base = level_base[right.idx()];
    let left_view = tdd.levels[left.idx()].child_decoder();
    let right_view = tdd.levels[right.idx()].child_decoder();
    if level_dirty[left.idx()] || level_dirty[right.idx()] {
        let left_remap = &remap[left_grid_base..];
        let right_remap = &remap[right_grid_base..];
        for i in 0..width {
            if remap[base + i] == UNREACHED {
                continue;
            }
            match tdd.levels[t_idx].nodes[i].kind() {
                NodeKind::Inline(_) => {
                    let node = &mut tdd.levels[t_idx].nodes[i];
                    node.a = left_view.remap(EncodedChildRef::from_raw(node.a), left_remap).0;
                    node.b = right_view.remap(EncodedChildRef::from_raw(node.b), right_remap).0;
                }
                k if k.pairs_in_arena() => tdd.levels[t_idx]
                    .pairs_remap_indexed(i, left_remap, right_remap, left_view, right_view),
                _ => {}
            }
        }
    }
}

/// Push every level prune shrank onto the contract worklists.
///
/// Removing a node simplifies the parent context of that level's children,
/// which can make two of them twins; contraction processes the parent, so the
/// shrunk level itself is what is pushed. Its own nodes cannot become twins
/// (a removal never equates two survivors), and its parents see only a
/// bijective child remap, so neither needs seeding. `level_dirty` is set on
/// non-leaf levels only, so every index pushed is a parent level.
fn seed_dirty_levels(tdd: &mut Tdd, level_dirty: &[bool]) {
    for (t_idx, dirty) in level_dirty.iter().enumerate() {
        if *dirty {
            tdd.invalidate(VtreeIdx(t_idx as u32));
        }
    }
}

/// Pass 1: mark every slot reachable from the output node, walking the vtree
/// root first (raw indices do not follow topological order on a rotated
/// vtree). `remap` must be `UNREACHED`-filled and `level_base`-indexed; each
/// slot reached is stamped `REACHED`, which pass 2 replaces with the slot's
/// compacted index.
fn classic_mark(tdd: &Tdd, level_base: &[usize], remap: &mut [u32]) {
    let vtree = &tdd.vtree;
    remap[level_base[tdd.output.vtree.idx()] + tdd.output.local.idx()] = REACHED;
    let topo = vtree.bottomup_slice();
    for v in topo.iter().rev() {
        let t_idx = v.idx();
        if vtree.node(VtreeIdx(t_idx as u32)).is_leaf() {
            continue;
        }
        let (left, right) = vtree.children(*v);
        let left_grid_base = level_base[left.idx()];
        let right_grid_base = level_base[right.idx()];
        let output_grid_base = level_base[t_idx];

        let width = tdd.levels[t_idx].slot_count();
        if tdd.levels[t_idx].is_marginal() {
            // A marginal level has no pairs, and by invariant 5 every level
            // beneath it is marginal too, so there is nothing to mark below.
            continue;
        }
        // A side of a marginal child may be an inline count rather than a slot;
        // such a side names no child cell, so `cell()` skips it and only real
        // cells are marked. A structural side is its own cell.
        let left_view = tdd.levels[left.idx()].child_decoder();
        let right_view = tdd.levels[right.idx()].child_decoder();
        let level = &tdd.levels[t_idx];
        for i in 0..width {
            if remap[output_grid_base + i] == UNREACHED {
                continue;
            }
            if level.nodes[i].is_internal() {
                for pair in level.pairs_of_idx(i) {
                    if let Some(s) = left_view.child(pair.left).index() {
                        remap[left_grid_base + s] = REACHED;
                    }
                    if let Some(s) = right_view.child(pair.right).index() {
                        remap[right_grid_base + s] = REACHED;
                    }
                }
            }
        }
    }
}

