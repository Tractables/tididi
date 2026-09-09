//! Specialized diagram × clause conjunction.
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

use crate::engine::Engine;
use crate::engine::pool::Pool;
use crate::apply::conjoin::plan::{ClausePlan, OutputPlan};
use crate::apply::scoped_flags::ScopedFlags;
use std::sync::Arc;

use crate::diagram::Literal;
use crate::vtree::{Vtree, VtreeIdx};
use crate::apply::leaf::CONJOIN_GRID;
use crate::diagram::{self, *};

use crate::error::ApplyError;
use crate::apply::conjoin::budget::{reserve_pairs_for_emit, NO_PRODUCT};

mod spine;
use spine::*;
pub use spine::mark_clause_levels;
mod emit;
use emit::*;
mod pairs;
use pairs::*;
mod rebuild;
use rebuild::*;

// Thread-local scratch buffers for apply_and_clause (pooled via the
// `Pool` take/put pattern).
/// Every buffer one engine's clause conjunctions reuse between calls.
///
/// The two flag arrays hold an all-false invariant between calls: only spine
/// entries are ever set and the success path resets them, while an error path
/// drops the taken `Vec` rather than returning it.
#[derive(Default)]
pub(crate) struct ClauseScratch {
    /// Maps accumulator node index → `[ct, dt]` output indices for conjunction
    /// with the clause's c_t / d_t virtual nodes. Interleaved (one `[u32; 2]`
    /// entry per node) so the random per-pair lookup of a node's ct AND dt
    /// remap is a single cache line instead of two; the map loads are the
    /// dominant stall in the single-clause apply loop, and the interleaved
    /// form has the same footprint as two flat `u32` maps. Lane 0 = ct, lane
    /// 1 = dt; the dt lane is written iff `need_dt` for the level (stale dt
    /// lanes are never read — see the no-bulk-`NO_PRODUCT`-fill note at the sizing
    /// site).
    cd_map: Pool<Vec<[u32; 2]>>,
    /// Cumulative offsets into `cd_map`, one per vtree level.
    level_base: Pool<Vec<usize>>,
    /// Per-level flags: `on_spine[t]` = clause has variables in subtree t.
    on_spine: Pool<Vec<bool>>,
    /// Per-level flags: `need_dt[t]` = must compute complement conjunction at t.
    need_dt: Pool<Vec<bool>>,
    /// MergeScope internal nodes in bottom-up (post-order) order.
    spine_internal: Pool<Vec<VtreeIdx>>,
    /// Work stack for the post-order spine walk (node, processed?).
    dfs_stack: Pool<Vec<(VtreeIdx, bool)>>,
}

impl ClauseScratch {
    /// Release every retained buffer, leaving the pools empty.
    pub(crate) fn drain(&self) {
        self.cd_map.drain();
        self.level_base.drain();
        self.on_spine.drain();
        self.need_dt.drain();
        self.spine_internal.drain();
        self.dfs_stack.drain();
    }
}








/// Conjoin `clause` into `acc`, leaving `acc` untouched on failure.
///
/// # Errors
/// Returns the [`ApplyError`] the conjunction stopped on.
pub fn conjoin_clause_into(eng: &Engine, f: &mut Tdd, clause: &[Literal]) -> Result<Tdd, ApplyError> {
    let lim = eng.limits();
    let pool = eng.clause_pool();
    let vtree = &f.vtree;
    let num_nodes = vtree.num_nodes();

    // Early return for ZERO input.
    if f.is_zero() {
        let levels = diagram::take_levels(eng, num_nodes);
        let mut out = Tdd::from_levels_unchecked(
            Arc::clone(vtree),
            levels,
            TddNodeId { vtree: f.output.vtree, local: ZERO },
        );
        out.weights = f.weights.take();
        return Ok(out);
    }

    // The clause spine — the Steiner tree of its variables' leaves — and the
    // `need_dt` flag propagated top-down over it.
    let mut on_spine = ScopedFlags::take(&pool.on_spine, num_nodes);
    let mut spine_internal = pool.spine_internal.take();
    let mut dfs_stack = pool.dfs_stack.take();
    build_clause_spine(vtree, clause, &mut on_spine, &mut spine_internal, &mut dfs_stack);
    let mut need_dt = ScopedFlags::take(&pool.need_dt, num_nodes);
    propagate_need_dt(vtree, &spine_internal, &on_spine, &mut need_dt);

    // Take ownership of f's levels. Irrelevant levels stay in place as the
    // identity pass-through (no per-level swap, no fresh num_nodes allocation);
    // spine internal levels are rebuilt in place below. f is left with empty
    // levels — callers that recycle (the *_owned wrappers) return that empty
    // Vec to the pool.
    let out_vtree = f.output.vtree;
    let out_local_in = f.output.local;
    let mut levels = std::mem::take(&mut f.levels);
    // The accumulator's marginal values move to the output along with its levels:
    // a clause carries none of its own, and the output IS the accumulator one
    // clause further on.
    let f_weights = f.weights.take();

    // Compact per-level base offsets into `cd_map`: only spine levels get
    // storage, which keeps the map `O(Σ spine widths)` — typically a handful of
    // levels — rather than `O(|f|)`. Irrelevant levels are read through raw
    // pair indices, not the map. See `plan_cd_map_bases`.
    let mut level_base = pool.level_base.take();
    if level_base.len() < num_nodes { level_base.resize(num_nodes, 0usize); }
    let total = plan_cd_map_bases(vtree, clause, &spine_internal, &levels, &mut level_base);

    // `cd_map` interleaves the `[c_t, d_t]` output node indices, so one random
    // per-pair lookup serves both lanes off a single cache line — those loads
    // are the dominant stall in this loop. The base blocks partition
    // `[0, total)` with no gaps and every entry is written exactly once below,
    // so there is no bulk `NO_PRODUCT` fill: a node index where one is emitted,
    // `NO_PRODUCT` otherwise. A `d_t` lane is written iff `need_dt[t]`, and a read of
    // one implies `need_dt` on that child, so a stale lane is never read.
    let mut cd_map = pool.cd_map.take();
    lim.try_resize(&mut cd_map, total, [NO_PRODUCT, NO_PRODUCT])?;

    fill_leaf_maps(vtree, clause, &level_base, &need_dt, &mut cd_map);

    // Reusable pair buffers for apply_and_clause — hoisted outside the per-level
    // and per-node loops. Retained capacity avoids Vec malloc/free per node.
    let mut clause_dt_pairs: Vec<InputPair> = Vec::new();  // f × d_t pairs
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
            eng,
            t, vtree, &mut levels, &mut cd_map, &level_base, &need_dt, &on_spine,
            &mut clause_t3_buf, &mut clause_dt_pairs,
        )?;
    }

    // Output: conjunction of f's output with c_t at the root. `out_vtree` is
    // an ancestor of every clause leaf, so it is on the spine and its ct_map
    // block is filled.
    let out_base = level_base[out_vtree.idx()];
    let ct_out = cd_map[out_base + out_local_in.idx()][0];
    let out_local = if ct_out != NO_PRODUCT { NodeIdx(ct_out) } else { ZERO };

    // Contract seed: this clause's spine, not every internal level. The
    // rebuild loop replaced `levels[t]` for `t ∈ spine_internal` and nothing
    // else, and the spine is ancestor-closed (`mark_clause_levels` walks each
    // clause leaf to the root) — the exactness argument is in `finish_rebuilt`,
    // which this shares with the restricted apply.
    let out = ClausePlan(&spine_internal).finish(
        f,
        Arc::clone(vtree),
        levels,
        TddNodeId { vtree: out_vtree, local: out_local },
        f_weights,
    );

    // `level_base` needs no reset — every spine entry is rewritten each call
    // and irrelevant entries are never read.
    pool.cd_map.put_bounded(cd_map, MAX_LEVEL_ARENA_BYTES);
    pool.level_base.put(level_base);
    pool.spine_internal.put(spine_internal);
    pool.dfs_stack.put(dfs_stack);

    Ok(out)
}







/// Infallible wrapper for `conjoin_clause_into` — panics on `OverBudget`.
/// Use only when no soft apply budget is armed; a caller that wants to survive
/// a refusal calls `conjoin_clause_into` and case-splits on `OverBudget`.
///
/// Conjoins a clause into an accumulator without first materializing the clause
/// as a separate diagram — the preferred way to compile a CNF one clause at a time,
/// seeding the accumulator with [`Tdd::one`]:
///
/// ```
/// use std::sync::Arc;
/// use num_bigint::BigUint;
/// use tididi::apply::apply_and_clause;
/// use tididi::vtree::Vtree;
/// use tididi::Tdd;
///
/// let vtree = Arc::new(Vtree::balanced(3));
/// let cnf = [[1, -2], [2, 3], [-1, 3]]; // DIMACS literals
/// let mut acc = Tdd::one(&vtree);
/// for clause in &cnf {
///     let literals: Vec<_> = clause.iter().map(|&n| n.into()).collect();
///     acc = apply_and_clause(&mut acc, &literals);
/// }
/// assert_eq!(acc.model_count(), BigUint::from(3u32));
/// ```
///
/// # Panics
///
/// Panics if `conjoin_clause_into` returns `OverBudget` while no soft budget
/// is configured (an internal invariant violation).
pub fn apply_and_clause(f: &mut Tdd, clause: &[Literal]) -> Tdd {
    let eng = Engine::new();
    conjoin_clause_into(&eng, f, clause)
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
pub fn conjoin_clause_owned(eng: &Engine, mut f: Tdd, clause: &[Literal]) -> Result<Tdd, ApplyError> {
    let result = conjoin_clause_into(eng, &mut f, clause);
    // Recycle what is left of `acc` — but only if that is a real level array.
    //
    // `conjoin_clause_into` MOVES the accumulator's levels into its own output
    // (the `std::mem::take` above), so on every path but the ZERO early-out it
    // leaves `acc` holding a LENGTH-0 `Vec`. Parking an empty Vec poisons the
    // pool slot: the slot holds one entry, so the empty Vec evicts whatever
    // populated entry was parked there, and the next `take_levels(n)` then finds
    // an entry carrying no level arenas at all — every level of the following
    // `Tdd::clause` has to regrow its `nodes`/`pairs` from capacity 0. Leaving
    // the slot untouched keeps the previously parked, warm entry available.
    let spent = std::mem::take(&mut f.levels);
    if !spent.is_empty() {
        diagram::return_levels(eng, diagram::PoolSlot::First, spent);
    }
    result
}

/// The clause-conjunction entry point on a caller's engine.
impl crate::engine::Engine {
    /// Conjoin one clause into a diagram without building the clause as a
    /// diagram of its own: only the levels on the clause's spine are rebuilt.
    ///
    /// The operand is consumed either way, as in [`Engine::and`].
    ///
    /// # Errors
    ///
    /// As [`Engine::and`].
    pub fn and_clause(&self, f: Tdd, clause: &[Literal]) -> Result<Tdd, ApplyError> {
        crate::apply::conjoin_clause::conjoin_clause_owned(self, f, clause)
    }
}

#[cfg(test)]
#[path = "../conjoin_clause_tests.rs"]
mod emit_reserve_tests;
