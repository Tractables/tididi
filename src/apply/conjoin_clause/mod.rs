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
use crate::limits::pool::Pool;
use crate::apply::scoped_flags::ScopedFlags;
use std::sync::Arc;

use crate::diagram::Literal;
use crate::vtree::{Vtree, VtreeIdx};
use crate::apply::leaf::CONJOIN_GRID;
use crate::diagram::{self, *};

use crate::limits::ApplyError;
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

/// Every buffer one engine's clause conjunctions reuse between calls.
///
/// The two flag arrays hold an all-false invariant between calls: only spine
/// entries are ever set and the success path resets them, while an error path
/// drops the taken `Vec` rather than returning it.
#[derive(Default)]
pub(crate) struct ClauseScratch {
    /// Maps accumulator node index → `[ct, dt]` output indices for conjunction
    /// with the clause's `c_t` / `d_t` virtual nodes, interleaved so one
    /// per-pair lookup serves both lanes; the `dt` lane is written iff
    /// `need_dt` for the level.
    cd_map: Pool<Vec<[u32; 2]>>,
    /// Cumulative offsets into `cd_map`, one per vtree level.
    level_base: Pool<Vec<usize>>,
    /// Per-level flags: `on_spine[t]` = clause has variables in subtree t.
    on_spine: Pool<Vec<bool>>,
    /// Per-level flags: `need_dt[t]` = must compute complement conjunction at t.
    need_dt: Pool<Vec<bool>>,
    /// Spine internal nodes in bottom-up (post-order) order.
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

/// Conjoin `clause` into `f`, moving `f`'s levels and weights into the result;
/// `f` is left empty on `Err` as well as on `Ok`.
///
/// # Errors
/// Returns the [`ApplyError`] the conjunction stopped on.
pub(crate) fn conjoin_clause_into(eng: &Engine, f: &mut Tdd, clause: &[Literal]) -> Result<Tdd, ApplyError> {
    let lim = eng.limits();
    let pool = eng.clause_pool();
    let vtree = &f.vtree;
    let num_nodes = vtree.num_nodes();

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

    // A variable named in both polarities satisfies the clause whatever its
    // value, so conjoining it is the identity. The rebuild below keeps one
    // column per variable of the clause and cannot say that.
    if crate::diagram::is_tautological(clause) {
        let vtree = Arc::clone(vtree);
        let output = f.output;
        let mut out =
            Tdd::from_levels_unchecked(vtree, std::mem::take(&mut f.levels), output);
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

    // Take ownership of f's levels: off-spine levels pass through as the
    // identity, spine internal levels are rebuilt in place below, and f is
    // left with empty levels.
    let out_vtree = f.output.vtree;
    let out_local_in = f.output.local;
    let mut levels = std::mem::take(&mut f.levels);
    // The accumulator's marginal values move to the output along with its levels:
    // a clause carries none of its own, and the output is nothing but the
    // accumulator, one clause further on.
    let f_weights = f.weights.take();

    // Per-level base offsets into `cd_map`: only spine levels get storage, so
    // the map is `O(Σ spine widths)` rather than `O(|f|)`; off-spine levels
    // are read through raw pair indices. See `plan_cd_map_bases`.
    let mut level_base = pool.level_base.take();
    if level_base.len() < num_nodes { level_base.resize(num_nodes, 0usize); }
    let total = plan_cd_map_bases(vtree, clause, &spine_internal, &levels, &mut level_base);

    // The base blocks partition `[0, total)` with no gaps and every `c_t`
    // entry is written once below, so no bulk `NO_PRODUCT` fill is needed. A
    // `d_t` lane is written iff `need_dt[t]`, and a read of one implies
    // `need_dt` on that child, so a stale lane is never read.
    let mut cd_map = pool.cd_map.take();
    lim.try_resize(&mut cd_map, total, [NO_PRODUCT, NO_PRODUCT])?;

    fill_leaf_maps(vtree, clause, &level_base, &need_dt, &mut cd_map);

    // Pair buffers reused across the per-level and per-node loops.
    let mut clause_dt_pairs: Vec<InputPair> = Vec::new();  // f × d_t pairs
    // "Type 3" pairs (dt_L, ct_R) of the both-relevant case have larger left
    // indices than type 1/2 pairs, so they are buffered and flushed after
    // them to keep the sorted order.
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

    // Contract seed: this clause's spine, which is ancestor-closed, so an
    // off-spine level and its whole subtree are the accumulator's own bytes;
    // whatever the accumulator still owed is carried over, which keeps this
    // exact for a caller that does not minimize between clauses.
    let vtree = Arc::clone(vtree);
    let carried = f.take_worklists();
    let mut out = Tdd::with_levels_dirty(
        vtree,
        levels,
        TddNodeId { vtree: out_vtree, local: out_local },
        carried,
        &spine_internal,
    );
    out.weights = f_weights;

    // `level_base` needs no reset — every spine entry is rewritten each call
    // and irrelevant entries are never read.
    pool.cd_map.put_bounded(cd_map);
    pool.level_base.put(level_base);
    pool.spine_internal.put(spine_internal);
    pool.dfs_stack.put(dfs_stack);

    Ok(out)
}

/// Conjoin `clause` into `f` on a transient engine with no limits armed — the
/// preferred way to compile a CNF one clause at a time, seeding the accumulator
/// with [`Tdd::one`].
///
/// The clause is never materialized as a diagram of its own: only the levels on
/// its spine are rebuilt. `f` is consumed and its storage recycled into the
/// result, exactly as [`Engine::and_clause`](crate::Engine::and_clause)
/// consumes it; that method is this operation on a caller's engine, and the one
/// that can report a refusal instead of panicking on it.
///
/// The result denotes `f ∧ clause` and counts correctly after every clause,
/// but is not canonical: run [`minimize`](crate::reduce::minimize) when the
/// canonical form is needed. A ⊥ accumulator stays ⊥. Marginal levels off
/// the clause's spine pass through unchanged, and a weight store moves to the
/// result.
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
///     acc = apply_and_clause(acc, &literals);
/// }
/// assert_eq!(acc.model_count(), BigUint::from(3u32));
/// ```
///
/// The literals are a set, as in [`Tdd::clause`](crate::Tdd::clause): a
/// variable repeated in one polarity conjoins the clause the deduplicated
/// literals spell, and a variable in both polarities conjoins ⊤, which is the
/// identity.
///
/// # Panics
///
/// Panics if a literal names a variable the vtree has no leaf for, or if a
/// level on the clause's spine is marginal (its variables were summed out
/// before every clause over them was in). Panics if the rebuild is refused;
/// nothing is armed on the transient engine, so the only refusal left is the
/// allocator's.
#[must_use]
pub fn apply_and_clause(f: Tdd, clause: &[Literal]) -> Tdd {
    let eng = Engine::new();
    conjoin_clause_owned(&eng, f, clause)
        .expect("apply_and_clause: refused with no limits armed")
}

/// Conjoin `clause` into `f` under `eng`'s limits, recycling the accumulator's
/// levels when it still owns any. The implementation behind
/// [`Engine::and_clause`](crate::Engine::and_clause) and [`apply_and_clause`].
///
/// # Errors
///
/// Returns `Err(ApplyError::OverBudget)` if any internal allocation is refused.
pub(crate) fn conjoin_clause_owned(eng: &Engine, mut f: Tdd, clause: &[Literal]) -> Result<Tdd, ApplyError> {
    let result = conjoin_clause_into(eng, &mut f, clause);
    // Recycle what is left of `f` only if it is a real level array: on every
    // path but the zero early-out `f` is left empty, and parking an empty Vec
    // would evict the warm entry the one-slot pool holds.
    let spent = std::mem::take(&mut f.levels);
    if !spent.is_empty() {
        diagram::return_levels(eng, diagram::PoolSlot::First, spent);
    }
    result
}

/// Build a minimal, canonical diagram representing a single clause: the clause
/// conjoined into the constant-true diagram, which rebuilds only the levels on
/// the clause's spine and leaves one identity node at every other level.
///
/// The result passes `test_helpers::check::check_all_fast`.
///
/// Runs with no limit armed: the rebuild touches one node per spine level, and
/// the construction is infallible for every caller.
pub(crate) fn clause_to_tdd(eng: &Engine, vtree: &Arc<Vtree>, clause: &[Literal]) -> Tdd {
    let _unmetered = eng.limits().scope(crate::limits::LimitSet::none());
    conjoin_clause_owned(eng, crate::build::constant_one(eng, vtree), clause)
        .expect("no limit is armed while a clause is built")
}

impl Tdd {
    /// Build a canonical diagram for a single clause from DIMACS-style literals.
    ///
    /// Sugar over [`Engine::clause`](crate::engine::Engine::clause), built on a
    /// transient engine. Each item is converted with [`Into<Literal>`], so plain
    /// integers use the 1-based DIMACS sign convention (`1` → `x1`, `-2` → `¬x2`;
    /// see [`Literal`]).
    ///
    /// The literals are a set: a variable repeated in one polarity builds the
    /// clause the deduplicated literals spell, and a variable in both
    /// polarities builds ⊤.
    ///
    /// # Panics
    ///
    /// Panics if a literal names a variable `vtree` has no leaf for.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::Tdd;
    /// use tididi::vtree::Vtree;
    ///
    /// let vtree = Arc::new(Vtree::balanced(3));
    /// let f = Tdd::clause(&vtree, [1, -2]); // x1 ∨ ¬x2
    /// # let _ = f;
    /// ```
    pub fn clause(vtree: &Arc<Vtree>, literals: impl IntoIterator<Item = impl Into<Literal>>) -> Tdd {
        Engine::new().clause(vtree, literals)
    }
}

/// The clause entry points on a caller's engine.
impl crate::engine::Engine {
    /// A diagram for one clause over `vtree`, built in this engine's pools.
    ///
    /// The engine-owned form of [`Tdd::clause`]; identical result, and the
    /// per-level buffers stay warm for the next clause. The literals are a set,
    /// as in [`Tdd::clause`]. No limit armed on the engine is consulted.
    ///
    /// # Panics
    ///
    /// Panics if a literal names a variable `vtree` has no leaf for.
    #[must_use]
    pub fn clause(
        &self,
        vtree: &Arc<Vtree>,
        literals: impl IntoIterator<Item = impl Into<Literal>>,
    ) -> Tdd {
        let clause: Vec<Literal> = literals.into_iter().map(Into::into).collect();
        clause_to_tdd(self, vtree, &clause)
    }

    /// Conjoin one clause into a diagram without building the clause as a
    /// diagram of its own: only the levels on the clause's spine are rebuilt.
    ///
    /// The operand is consumed on `Err` as well as on `Ok`, as in
    /// [`Engine::and`]. The result counts correctly after every clause but is
    /// not canonical until [`minimize`](crate::reduce::minimize) runs; a ⊥
    /// operand stays ⊥, marginal levels off the spine pass through, and a
    /// weight store moves to the result. The rebuild polls no deadline and
    /// checks no output cap.
    ///
    /// # Errors
    ///
    /// [`ApplyError::OverBudget`] when a reservation is refused by the
    /// allocator or the armed byte budget.
    ///
    /// # Panics
    ///
    /// Panics if a literal names a variable the vtree has no leaf for, or if a
    /// level on the clause's spine is marginal.
    ///
    /// ```
    /// # use std::sync::Arc;
    /// # use tididi::{ApplyError, Engine, Tdd};
    /// # use tididi::limits::LimitSet;
    /// # use tididi::vtree::{VarId, Vtree};
    /// # let vtree = Arc::new(Vtree::balanced(4));
    /// # use tididi::Literal;
    /// let engine = Engine::new();
    /// let clause = [Literal::from(1), Literal::from(-2)];
    /// let f = engine.and_clause(Tdd::clause(&vtree, [2, 3]), &clause).unwrap();
    /// assert_eq!(f.model_count(), 8u32.into());
    ///
    /// // A byte budget of zero refuses the rebuild's first reservation.
    /// let _armed = engine.limits().scope(LimitSet::none().budget(Some(0)));
    /// let g = Tdd::clause(&vtree, [2, 3]);
    /// match engine.and_clause(g, &clause) {
    ///     Ok(_) => unreachable!("no reservation can be granted"),
    ///     Err(e) => assert_eq!(e, ApplyError::OverBudget),
    /// }
    /// ```
    pub fn and_clause(&self, f: Tdd, clause: &[Literal]) -> Result<Tdd, ApplyError> {
        crate::apply::conjoin_clause::conjoin_clause_owned(self, f, clause)
    }
}

#[cfg(test)]
mod tests;
