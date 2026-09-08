//! Specialized TDD × clause conjunction.
//!
//! The clause is never built as a diagram. At each vtree level `t` it stands for
//! exactly two functions over that subtree's variables:
//!
//! - **`c_t`** — the clause restricted to subtree `t`: at least one of its
//!   literals inside `t` is satisfied;
//! - **`d_t`** — the complement of `c_t` within the subtree: none of them is.
//!
//! Together they partition the subtree's assignments, which is what lets one
//! bottom-up walk carry both branches of the clause at once: a level's output
//! for an accumulator node is its conjunction with `c_t` and with `d_t`, kept
//! side by side in `cd_map`. The whole file is written in terms of this pair.

use std::cell::Cell;
use std::sync::Arc;

use crate::vtree::{Literal, Vtree, VtreeIdx};
use crate::tdd::transform::pairwise::leaf::CONJOIN_GRID;
use crate::tdd::types::{self, *};
use crate::tdd::utils::{pool_put, pool_put_bounded, pool_take};

use crate::tdd::limits::{try_push, ApplyError};
use crate::tdd::transform::pairwise::conjoin::budget::{reserve_pairs_for_emit, try_resize_dead2, DEAD};
use crate::tdd::transform::pairwise::conjoin::decide_emit_growth_mode;

mod spine;
use spine::*;
pub use spine::walk_mark_spine;
mod emit;
use emit::*;
mod pairs;
use pairs::*;
mod rebuild;
use rebuild::*;

// Thread-local scratch buffers for apply_and_clause (pooled via the
// `pool_take`/`pool_put` Cell::take/set pattern).
thread_local! {
    /// Maps accumulator node index → `[ct, dt]` output indices for conjunction
    /// with the clause's c_t / d_t virtual nodes. Interleaved (one `[u32; 2]`
    /// entry per node) so the random per-pair lookup of a node's ct AND dt
    /// remap is a single cache line instead of two — the map loads are the
    /// dominant stall in the batch-1 apply loop (perf: the two loads alone are
    /// >20% of the function's cycles on map-bound CNFs). Same total footprint
    /// as the two flat u32 maps it replaced. Lane 0 = ct, lane 1 = dt; the dt
    /// lane is written iff `need_dt` for the level (stale dt lanes are never
    /// read — see the no-bulk-DEAD-fill note at the sizing site).
    static SCRATCH_CD_MAP: Cell<Vec<[u32; 2]>> = const { Cell::new(Vec::new()) };
    /// Cumulative offsets into cd_map, one per vtree level.
    static SCRATCH_CLAUSE_LEVEL_BASE: Cell<Vec<usize>> = const { Cell::new(Vec::new()) };
    /// Per-level flags: on_spine[t] = clause has variables in subtree t.
    /// Maintained all-false between calls — only
    /// spine entries are ever set, and they are reset on the (single) success
    /// path; error paths drop the taken Vec, so the pooled Vec stays clean.
    static SCRATCH_RELEVANT: Cell<Vec<bool>> = const { Cell::new(Vec::new()) };
    /// Per-level flags: need_dt[t] = must compute complement conjunction at level t.
    /// Same all-false-between-calls discipline as `SCRATCH_RELEVANT`.
    static SCRATCH_NEED_DT: Cell<Vec<bool>> = const { Cell::new(Vec::new()) };
    /// Entailment-skip: per-node "structurally unchanged" flags, indexed like
    /// cd_map (by `level_base[t] + node_idx`). Filled inline during the
    /// leaf-fill and rebuild passes — no separate cone walk.
    static SCRATCH_UNCHANGED: Cell<Vec<bool>> = const { Cell::new(Vec::new()) };
    /// Spine internal nodes in bottom-up (post-order) order.
    static SCRATCH_SPINE_INTERNAL: Cell<Vec<VtreeIdx>> = const { Cell::new(Vec::new()) };
    /// Scratch stack for the post-order spine DFS (node, processed?).
    static SCRATCH_DFS_STACK: Cell<Vec<(VtreeIdx, bool)>> = const { Cell::new(Vec::new()) };
}








#[doc(hidden)] // test-support: production callers use `try_apply_and_clause_owned`
pub fn try_apply_and_clause(acc: &mut Tdd, clause: &[Literal]) -> Result<Tdd, ApplyError> {
    let vtree = &acc.vtree;
    let num_nodes = vtree.num_nodes();

    // Early return for ZERO input.
    if acc.is_zero() {
        let levels = types::take_levels(num_nodes);
        let mut out = Tdd::with_levels(
            Arc::clone(vtree),
            levels,
            TddNodeId { vtree: acc.output.vtree, local: ZERO },
        );
        out.weights = acc.weights.take();
        return Ok(out);
    }

    // ── Build the clause "spine" (Steiner tree of clause-variable leaves) ──
    //
    // `on_spine` is the pooled `relevant` flag array, maintained all-false
    // between calls. `build_clause_spine` marks ancestors of each clause-variable
    // leaf and collects spine internals in post-order.
    let mut on_spine = pool_take(&SCRATCH_RELEVANT);
    if on_spine.len() < num_nodes { on_spine.resize(num_nodes, false); }
    debug_assert!(on_spine[..num_nodes].iter().all(|&b| !b),
        "on_spine scratch not clean on entry — a prior call leaked a set flag");
    let mut spine_internal = pool_take(&SCRATCH_SPINE_INTERNAL);
    let mut dfs_stack = pool_take(&SCRATCH_DFS_STACK);
    build_clause_spine(vtree, clause, &mut on_spine, &mut spine_internal, &mut dfs_stack);

    // need_dt (top-down over the spine): propagated by `propagate_need_dt`.
    let mut need_dt = pool_take(&SCRATCH_NEED_DT);
    if need_dt.len() < num_nodes { need_dt.resize(num_nodes, false); }
    debug_assert!(need_dt[..num_nodes].iter().all(|&b| !b),
        "need_dt scratch not clean on entry — a prior call leaked a set flag");
    propagate_need_dt(vtree, &spine_internal, &on_spine, &mut need_dt);

    // Take ownership of acc's levels. Irrelevant levels stay in place as the
    // identity pass-through (no per-level swap, no fresh num_nodes allocation);
    // spine internal levels are rebuilt in place below. acc is left with empty
    // levels — callers that recycle (the *_owned wrappers) return that empty
    // Vec to the pool.
    let out_vtree = acc.output.vtree;
    let out_local_in = acc.output.local;
    let mut levels = std::mem::take(&mut acc.levels);

    // Compact per-level base offsets into cd_map: only spine levels get
    // storage. Sizing over the spine (leaves via clause literals, internals via
    // spine_internal) keeps the map O(Σ spine widths) — typically ~7 levels —
    // instead of O(total acc nodes). The level_base blocks partition [0,total)
    // with no gaps, so every entry is written exactly once per clause below
    // (no bulk DEAD memset). Irrelevant levels are read via raw pair indices,
    // not the map, so they need no storage.
    //
    // This loop also folds in the marginalization structural-enforcement gate:
    // a relevant internal level that is marginal (pair structure replaced by
    // per-node counts) would make `pairs_of_idx` read an empty `nodes` array.
    // That is a caller-side ordering bug (a clause touching an already-
    // marginalized scope), so panic at the gateway. A caller avoids it by
    // conjoining every clause over a scope before marginalizing that scope.
    let mut level_base = pool_take(&SCRATCH_CLAUSE_LEVEL_BASE);
    if level_base.len() < num_nodes { level_base.resize(num_nodes, 0usize); }
    let total = plan_cd_map_bases(vtree, clause, &spine_internal, &levels, &mut level_base);

    // cd_map: interleaved `[ct, dt]` output node indices for
    // acc_node_i ∧ clause_c_t / ∧ clause_d_t (lane 0 / lane 1). One random
    // per-pair lookup serves both lanes — a single cache line instead of two
    // (the map loads are the dominant stall in the batch-1 apply loop).
    // No bulk DEAD-fill: the leaf loop and the rebuild loop below visit EVERY
    // node index of every spine level and write each map entry exactly once —
    // node-idx when a node is emitted, DEAD otherwise. (dt lanes are written
    // iff need_dt[t]; a dt read implies need_dt on that child, so stale dt
    // lanes are never read.)
    let mut cd_map = pool_take(&SCRATCH_CD_MAP);
    try_resize_dead2(&mut cd_map, total)?;

    fill_leaf_maps(vtree, clause, &level_base, &need_dt, &mut cd_map);

    // Reusable pair buffers for apply_and_clause — hoisted outside the per-level
    // and per-node loops. Retained capacity avoids Vec malloc/free per node.
    let mut clause_dt_pairs: Vec<InputPair> = Vec::new();  // acc × d_t pairs
    // Buffer for "type 3" pairs (dt_L, ct_R) in the both-relevant case.
    // These have larger left indices than type 1/2 pairs, so they're buffered
    // and flushed after the type 1/2 pairs to maintain sorted order. Plain
    // `Vec` so push is fallible via `try_push` — `SmallVec` aborts on heap
    // spill, which an adversarial dense level can blow past.
    let mut clause_t3_buf: Vec<InputPair> = Vec::new();

    // Rebuild each spine internal level bottom-up. Children's maps are fully
    // written before any parent reads them.
    for &t in &spine_internal {
        rebuild_spine_level(
            t, vtree, &mut levels, &mut cd_map, &level_base, &need_dt, &on_spine,
            &mut clause_t3_buf, &mut clause_dt_pairs,
        )?;
    }

    // Output: conjunction of acc's output with c_t at the root. `out_vtree` is
    // an ancestor of every clause leaf, so it is on the spine and its ct_map
    // block is filled.
    let out_base = level_base[out_vtree.idx()];
    let ct_out = cd_map[out_base + out_local_in.idx()][0];
    let out_local = if ct_out != DEAD { LocalNodeIdx(ct_out) } else { ZERO };

    // Reset ONLY the spine entries (preserve the all-false pool invariant for
    // on_spine/need_dt), then return scratch buffers. The spine is exactly
    // {clause leaves} ∪ {spine internals}, so these two loops cover every set
    // flag. (level_base needs no reset — every spine entry is rewritten each
    // call and irrelevant entries are never read.)
    clear_spine_flags(vtree, clause, &spine_internal, &mut on_spine, &mut need_dt);

    // ── Contract seed: this clause's spine, not every internal level ──
    //
    // The rebuild loop above replaced `levels[t]` for `t ∈ spine_internal` and
    // nothing else — every other level rode through as the identity. The spine
    // is ancestor-closed (`walk_mark_spine` walks each clause leaf to the
    // root), so its complement is DESCENDANT-closed: an off-spine level's
    // parent-pairs, its own pairs, and its whole subtree are all bit-identical
    // to the accumulator's.
    //
    // That is exactly what a contraction sweep at an off-spine parent `p`
    // reads: `try_contract_child` looks at `levels[p]` and `levels[child]`, and
    // `contract_twins` writes only those two. So the sweep seeded at `p` here
    // is the same computation, on the same bytes, as the one the accumulator's
    // last sweep already ran to a fixed point — it fires nothing. Seeding the
    // spine alone therefore produces an IDENTICAL diagram, not merely an
    // equivalent one. (Downward cascades need no seeding either way: a child
    // that fires is pushed as a parent by `contract_all_twins_topdown` itself,
    // and the soundness note there rules out a contraction reopening twins at
    // or above its parent.)
    //
    // Whatever the accumulator still owed is carried over rather than dropped,
    // which is what keeps this exact for a caller that does NOT minimize
    // between applies: `with_levels_dirty`'s obligation 2.
    let mut dirty_contract = std::mem::take(&mut acc.scratch.dirty_contract);
    let mut dirty_leaf_contract = std::mem::take(&mut acc.scratch.dirty_leaf_contract);
    dirty_contract.reserve(spine_internal.len());
    dirty_leaf_contract.reserve(spine_internal.len());
    for &t in &spine_internal {
        dirty_contract.push(t.0);
        dirty_leaf_contract.push(t.0);
    }

    pool_put_bounded(&SCRATCH_CD_MAP, cd_map, MAX_LEVEL_ARENA_BYTES);
    pool_put(&SCRATCH_CLAUSE_LEVEL_BASE, level_base);
    pool_put(&SCRATCH_RELEVANT, on_spine);
    pool_put(&SCRATCH_NEED_DT, need_dt);
    pool_put(&SCRATCH_SPINE_INTERNAL, spine_internal);
    pool_put(&SCRATCH_DFS_STACK, dfs_stack);

    // The accumulator's frozen values move to the output along with its levels:
    // a clause carries none of its own, and the output IS the accumulator one
    // clause further on.
    let mut out = Tdd::with_levels_dirty(
        Arc::clone(vtree),
        levels,
        TddNodeId { vtree: out_vtree, local: out_local },
        dirty_contract,
        dirty_leaf_contract,
    );
    out.weights = acc.weights.take();
    Ok(out)
}







/// Infallible wrapper for `try_apply_and_clause` — panics on `OverBudget`.
/// Use only when no soft apply budget is armed; a caller that wants to survive
/// a refusal calls `try_apply_and_clause` and case-splits on `OverBudget`.
///
/// Conjoins a clause into an accumulator without first materializing the clause
/// as a separate TDD — the preferred way to compile a CNF one clause at a time,
/// seeding the accumulator with [`constant_one`](crate::tdd::build::constant_one):
///
/// ```
/// use std::sync::Arc;
/// use num_bigint::BigUint;
/// use tididi::tdd::build::constant_one;
/// use tididi::tdd::transform::pairwise::conjoin_clause::apply_and_clause;
/// use tididi::vtree::Vtree;
///
/// let vtree = Arc::new(Vtree::balanced(3));
/// let cnf = [[1, -2], [2, 3], [-1, 3]]; // DIMACS literals
/// let mut acc = constant_one(&vtree);
/// for clause in &cnf {
///     let lits: Vec<_> = clause.iter().map(|&n| n.into()).collect();
///     acc = apply_and_clause(&mut acc, &lits);
/// }
/// assert_eq!(acc.model_count(), BigUint::from(3u32));
/// ```
///
/// # Panics
///
/// Panics if `try_apply_and_clause` returns `OverBudget` while no soft budget
/// is configured (an internal invariant violation).
pub fn apply_and_clause(acc: &mut Tdd, clause: &[Literal]) -> Tdd {
    try_apply_and_clause(acc, clause)
        .expect("apply_and_clause: OverBudget without budget set")
}

/// Fallible variant of `apply_and_clause` that recycles the accumulator's levels
/// — when it still owns any. Returns `OverBudget` if
/// any internal allocation refuses (OS allocator under `RLIMIT_AS`, or the
/// soft apply budget would be exceeded).
///
/// # Errors
///
/// Returns `Err(ApplyError::OverBudget)` if any internal allocation is refused
/// (OS allocator under `RLIMIT_AS`, or the configured soft budget is exceeded).
pub fn try_apply_and_clause_owned(mut acc: Tdd, clause: &[Literal]) -> Result<Tdd, ApplyError> {
    let result = try_apply_and_clause(&mut acc, clause);
    // Recycle what is left of `acc` — but ONLY if that is a real level array.
    //
    // `try_apply_and_clause` MOVES the accumulator's levels into its own output
    // (the `std::mem::take` above), so on every path but the ZERO early-out it
    // leaves `acc` holding a LENGTH-0 `Vec`. Parking an empty Vec poisons the
    // pool slot: the slot holds one entry, so the empty Vec evicts whatever
    // populated entry was parked there, and the next `take_levels(n)` then finds
    // an entry carrying no level arenas at all — every level of the following
    // `clause_to_tdd` has to regrow its `nodes`/`pairs` from capacity 0. Leaving
    // the slot untouched keeps the previously parked, warm entry available.
    let spent = std::mem::take(&mut acc.levels);
    if !spent.is_empty() {
        types::return_levels(spent);
    }
    result
}

#[cfg(test)]
#[path = "../conjoin_clause_tests.rs"]
mod emit_reserve_tests;
