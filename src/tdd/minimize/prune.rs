//! Prune phase: remove TDD nodes not reachable from the output.
//!
//! Marks reachability top-down from the output node, then compacts each level
//! bottom-up while remapping child references. The remap is monotone (preserves
//! order), so sorted pair lists remain sorted after remapping.
//!
//! **Interface to the contract phase.** Prune and twin-contraction (`contract/`)
//! are decoupled except through the `Tdd` dirty-contract worklists: pruning marks
//! affected nodes via `Tdd::mark_contract_dirty`, seeding the
//! `dirty_contract`/`dirty_leaf_contract` worklists that the incremental contract
//! strategies drain. That shared state is the only coupling between the phases;
//! the orchestration lives in `minimize/mod.rs`.

use std::cell::Cell;

use crate::vtree::VtreeIdx;
use crate::tdd::limits::ApplyError;
use crate::tdd::types::*;
use crate::tdd::utils::{pool_put, pool_put_bounded, pool_take};
use crate::tdd::types::MAX_LEVEL_ARENA_BYTES;

// Thread-local scratch buffers (grow-only, reused across calls).
// See types.rs for details on this pooling pattern.
thread_local! {
    static SCRATCH_REMAP: Cell<Vec<u32>> = const { Cell::new(Vec::new()) };
    static SCRATCH_OFF: Cell<Vec<usize>> = const { Cell::new(Vec::new()) };
}

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
/// Marks reachability top-down, then compacts each level bottom-up (remapping
/// child references in the same pass). The remap is monotone, so sorted pair
/// lists stay sorted. Levels that lost a node are pushed onto the contract
/// worklists internally (see the seeding loop at the end), so callers need no
/// post-prune reseed.
///
/// **Allocation failure**: the one `total`-proportional scratch buffer
/// (`remap`: 4 B/slot, carrying the pass-1 reachability marks as well — see
/// `UNREACHED`) is `try_reserve_exact`-guarded and returns `Err(OverBudget)`
/// if refused. `total` is the summed effective width of every level, so on a
/// blown-up diagram — exactly when the OOM-recovery path calls minimize hoping
/// to shrink it — this reservation reaches multi-GiB and was the
/// process-aborting allocation on the MCC-2026 recovery churners (017: an
/// 8 GiB `remap`; the separate 1 B/slot reachability array folded into it here
/// cost a further 4.3 GiB on 083). The reservation happens before any mutation
/// of `tdd`, so on `Err` the diagram is untouched: well-formed, not poisoned
/// (`try_minimize`'s B1 error contract).
///
/// # Errors
///
/// Returns `Err(ApplyError::OverBudget)` if the budget-gated `remap`
/// reservation is refused; the diagram is left untouched.
pub(crate) fn prune_unreachable(tdd: &mut Tdd) -> Result<(), ApplyError> {
    let num_nodes = tdd.vtree.num_nodes();

    // ZERO sentinel: the entire TDD computes ⊥ (UNSAT). No nodes are reachable.
    if tdd.is_zero() {
        for level in &mut tdd.levels {
            level.nodes.clear();
            level.n_tombstones = 0;
        }
        return Ok(());
    }

    let mut level_base = pool_take(&SCRATCH_OFF);
    let mut remap = pool_take(&SCRATCH_REMAP);

    // Flat offset table: level t occupies remap[level_base[t]..level_base[t+1]].
    // Use effective_width() so leaf levels get LEAF_WIDTH (3) slots for marginal nodes.
    if level_base.len() < num_nodes + 1 {
        level_base.resize(num_nodes + 1, 0usize);
    }
    level_base[0] = 0;
    for i in 0..num_nodes {
        level_base[i + 1] = level_base[i] + tdd.effective_width(VtreeIdx(i as u32));
    }
    let total = level_base[num_nodes];

    // Fallible reservation of the one big buffer up front (see doc comment).
    // `try_reserve_exact` leaves the Vec untouched on failure, so returning the
    // pooled buffer is safe; the final size is `total` exactly, so the doubling
    // `try_reserve` would over-reserve VAS by up to 2× at GiB scale.
    let need_remap = total.saturating_sub(remap.len());
    if remap.try_reserve_exact(need_remap).is_err() {
        pool_put(&SCRATCH_OFF, level_base);
        pool_put_bounded(&SCRATCH_REMAP, remap, MAX_LEVEL_ARENA_BYTES);
        return Err(ApplyError::OverBudget);
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

    let vtree = &tdd.vtree;

    // ── Pass 2 (bottom-up): compact unreachable nodes ────────────────────
    //
    // Overwrite the pass-1 marks with the remap (old index → new index) in
    // place, update child references, and remove unreachable nodes in a single
    // pass. The remap is monotone (preserves relative order of surviving
    // nodes), so sorted input pair lists remain sorted after remapping — no
    // re-sort needed. A slot's mark is only read before its own remap value is
    // written, and `UNREACHED` survives on every slot that is never remapped,
    // so `remap[s] != UNREACHED` stays the reachability predicate throughout.
    //
    // When all nodes at a level are reachable the remap is the identity, so
    // the child-ref rewrite and the retain become no-ops on that level — the
    // common-case work falls out of the same code that handles the rare case.
    // Per-level "did prune remove a node here" flag (indexed by vtree idx).
    // The child-ref rewrite of a level is only needed when one of its child
    // levels actually shrank (otherwise that child's remap is the identity and
    // the rewrite writes identical values); the retain is only needed when the
    // level itself shrank. This lets the compact skip the O(pairs) rewrite and
    // the O(width) retain on the common all-reachable levels — the dominant
    // cost when the pruned fraction is small (e.g. a level losing 1.6% of nodes
    // still paid a full decode/re-encode over every surviving pair). num_nodes
    // is the vtree node count (≈ #vars), so this Vec is tiny.
    let mut level_dirty = vec![false; num_nodes];

    // Walk in topo bottom-up order so a level's child levels are remapped
    // before this level rewrites its child references. Raw idx no longer
    // encodes parent/child order on a rotated vtree, so `0..n` would be wrong.
    for v in vtree.bottomup_topo() {
        let t_idx = v.idx();
        let base = level_base[t_idx];
        let eff_width = tdd.effective_width(VtreeIdx(t_idx as u32));

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

        let width = tdd.levels[t_idx].width();
        // A marginalized child level keeps its content in its own
        // `marginal_counts` store (nodes=0). Parents reference it by tagged
        // slot indices that are STORE-relative and may be minted *after* this
        // prune (e.g. contract's inline→slot redirect). The reachability walk,
        // which only marks slots referenced by slot-refs present right now, can
        // therefore misclassify still-live slots as dead and truncate them,
        // corrupting a later apply that reads one of those slots → OOB / wrong
        // count. Keep the store at full length with stable indices so any slot
        // ref reads the value it was created against.
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
            // Never compacted here: the identity remap above forces
            // `this_dirty == false` for marginal levels (see the STORE-relative
            // comment). Orphaned slots are collected by `prune_marg_slots`, which
            // runs at post-tagger points, rewrites parent refs itself.
            // (An upstream variant compacted marginal stores here from
            // reachability; that branch is unreachable under the identity
            // remap and slot-prune owns marginal compaction now.)
            debug_assert!(!this_dirty);
            continue;
        }

        let (left, right) = vtree.children(VtreeIdx(t_idx as u32));
        let left_base = level_base[left.idx()];
        let right_base = level_base[right.idx()];

        // Remap child references in the pairs arena (separate pass to avoid
        // borrow conflict between nodes and pairs during retain).
        let left_marg = tdd.levels[left.idx()].is_marginal();
        let right_marg = tdd.levels[right.idx()].is_marginal();
        // Only rewrite child refs when a child level actually shrank — otherwise
        // both remaps are the identity (slot refs re-tag to themselves, inline
        // refs pass through unchanged) and every write would be a self-store.
        if level_dirty[left.idx()] || level_dirty[right.idx()] {
            let left_remap = &remap[left_base..];
            let right_remap = &remap[right_base..];
            // Marg-side refs are slot-tagged: mask before indexing the child remap,
            // re-tag the compacted slot on write. Non-marg side indexes verbatim.
            // An inline ref (bit 30 clear) carries a bare count, not a slot
            // index — it does not point into the child remap, so pass it through
            // verbatim; only slot refs are remapped.
            let remap_left = |raw: u32| -> u32 {
                if left_marg {
                    match MargRef::from_raw(raw) {
                        MargRef::Slot(s) => MargRef::slot_raw(left_remap[s as usize]),
                        MargRef::Inline(_) => {
                            raw
                        }
                    }
                } else {
                    left_remap[raw as usize]
                }
            };
            let remap_right = |raw: u32| -> u32 {
                if right_marg {
                    match MargRef::from_raw(raw) {
                        MargRef::Slot(s) => MargRef::slot_raw(right_remap[s as usize]),
                        MargRef::Inline(_) => {
                            raw
                        }
                    }
                } else {
                    right_remap[raw as usize]
                }
            };
            for i in 0..width {
                if remap[base + i] == UNREACHED {
                    continue;
                }
                if tdd.levels[t_idx].nodes[i].is_inline() {
                    let node = &mut tdd.levels[t_idx].nodes[i];
                    node.a = remap_left(node.a);
                    node.b = remap_right(node.b);
                } else if tdd.levels[t_idx].nodes[i].is_multi() {
                    tdd.levels[t_idx].pairs_remap_indexed(i, left_remap, right_remap, left_marg, right_marg);
                }
            }
        }

        // Compact unreachable nodes in-place. `retain` keeps elements where the
        // closure returns true, shifting survivors left — O(n) with no allocation.
        // Prune deliberately does NOT feed `dead_pairs`: the arena sweep is a
        // contract-path policy (its one call site is contract/merge.rs), and this
        // retain already reclaims the node slots. The price is that a heavily
        // pruned, never-contracted level keeps its arena slack until
        // `shrink_arrays`.
        // Only walk the node Vec when something was actually removed here.
        if this_dirty {
            let mut i = 0;
            tdd.levels[t_idx].nodes.retain(|_| {
                let keep = remap[base + i] != UNREACHED;
                i += 1;
                keep
            });
            // Tombstones are unreferenced, hence unreachable, hence just dropped
            // by the retain above — the level is dense again. (If `this_dirty`
            // is false there were no unreachable nodes, so no tombstones either.)
            tdd.levels[t_idx].n_tombstones = 0;
        }
    }

    // ── Seed the contract worklists for prune-created twins ──────────────
    // Removing a node leaves *that level's children* with a simpler parent
    // context (one fewer parent referencing them), which can equate two
    // children into a new twin. Twin contraction catches those by processing
    // the parent — i.e. the level we just shrank — so every shrunk level is
    // pushed onto both dirty lists. Two cases that do NOT need seeding:
    //   • the shrunk level's own nodes — a node removal can't equate two
    //     surviving siblings, so no twins appear here;
    //   • parents of a shrunk level — they only see a bijective child-index
    //     remap, which preserves pair-list (in)equality, so no twins there.
    // Seeding is deliberately narrow rather than all-internal-levels: levels
    // prune left untouched keep their `contracted` flag (they really are still
    // contracted), so an already-dirty level (e.g. a clause spine, marked
    // `contracted=false` by `with_levels`) is not re-pushed. `level_dirty` is
    // only ever set on non-leaf levels (leaf levels `continue` above before it
    // is written), so every index here is a valid parent level.
    for t_idx in 0..num_nodes {
        if level_dirty[t_idx] {
            tdd.mark_contract_dirty(VtreeIdx(t_idx as u32));
        }
    }

    tdd.output.local = LocalNodeIdx(
        remap[level_base[tdd.output.vtree.idx()] + tdd.output.local.idx()],
    );

    pool_put(&SCRATCH_OFF, level_base);
    pool_put_bounded(&SCRATCH_REMAP, remap, MAX_LEVEL_ARENA_BYTES);

    Ok(())
}

/// Classic Pass-1 mark: start from the output node and follow input pair
/// references downward. Walk in topological order (root first, leaves last)
/// via the side `topo` list — node identity is no longer aligned with topo
/// order after a rotation, so iterating `(0..n).rev()` would be wrong on a
/// rotated vtree. `remap` must be `UNREACHED`-filled and `level_base`-indexed;
/// every slot reached here is stamped `REACHED`, which pass 2 replaces with the
/// slot's compacted index.
fn classic_mark(tdd: &Tdd, level_base: &[usize], remap: &mut [u32]) {
    let vtree = &tdd.vtree;
    remap[level_base[tdd.output.vtree.idx()] + tdd.output.local.idx()] = REACHED;
    let topo = vtree.bottomup_topo();
    for v in topo.iter().rev() {
        let t_idx = v.idx();
        if vtree.node(VtreeIdx(t_idx as u32)).is_leaf() {
            continue;
        }
        let (left, right) = vtree.children(*v);
        let left_base = level_base[left.idx()];
        let right_base = level_base[right.idx()];
        let t_base = level_base[t_idx];

        let width = tdd.levels[t_idx].width();
        if tdd.levels[t_idx].is_marginal() {
            // Marginal levels have no pairs — nothing to propagate. The
            // child level is referenced only by this level's pairs, so once
            // those are gone the child is structurally orphaned. The
            // `marginalize_batch` cascade ensures children of marginal
            // levels are themselves marginal (or leaf), so a `continue`
            // here is correct: there is no reachable structure beneath a
            // marginal parent.
            continue;
        }
        // Marg-side refs are slot-tagged (bit 30): mask to the bare slot before
        // using as a `remap` index, so the same boundary slot is marked
        // reachable as before tagging. Non-marg side indexes verbatim.
        let left_marg = tdd.levels[left.idx()].is_marginal();
        let right_marg = tdd.levels[right.idx()].is_marginal();
        let left_mask = if left_marg { MARG_VALUE_MASK as usize } else { usize::MAX };
        let right_mask = if right_marg { MARG_VALUE_MASK as usize } else { usize::MAX };
        // A marginal side carries inline refs (bit-30 clear) mixed with
        // overflow slots (bit-30 set). An inline ref is a self-contained
        // count, not a child node, so it keeps no `remap` slot — skip it;
        // only genuine slots are marked. Returns the slot to mark, or `None`
        // to skip (inline). Captures only `Copy` state, so it never borrows
        // `remap`.
        let resolve = |raw: usize, is_marg: bool, mask: usize| -> Option<usize> {
            if is_marg {
                match MargRef::from_raw(raw as u32) {
                    MargRef::Slot(s) => Some(s as usize),
                    MargRef::Inline(_) => None,
                }
            } else {
                Some(raw & mask)
            }
        };
        let level = &tdd.levels[t_idx];
        for i in 0..width {
            if remap[t_base + i] == UNREACHED {
                continue;
            }
            if level.nodes[i].is_internal() {
                for pair in level.pairs_of_idx(i) {
                    if let Some(s) = resolve(pair.left.idx(), left_marg, left_mask) {
                        remap[left_base + s] = REACHED;
                    }
                    if let Some(s) = resolve(pair.right.idx(), right_marg, right_mask) {
                        remap[right_base + s] = REACHED;
                    }
                }
            }
        }
    }
}

