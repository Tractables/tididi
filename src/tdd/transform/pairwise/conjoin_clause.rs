//! Specialized TDD × clause conjunction.

use std::cell::Cell;
use std::sync::Arc;

use crate::vtree::{Literal, VtreeIdx};
use crate::tdd::transform::pairwise::leaf::CONJOIN_GRID;
use crate::tdd::types::{self, *};
use crate::tdd::utils::{pool_put, pool_put_bounded, pool_take};

use crate::tdd::limits::{try_push, ApplyError};
use crate::tdd::transform::pairwise::conjoin::budget::{reserve_pairs_for_emit, try_resize_dead2, DEAD};
use crate::tdd::transform::pairwise::conjoin::decide_emit_growth_mode;

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

/// If `pairs` is non-empty, emit as a new internal node in the level and
/// record its index in `result_map[base_plus_idx]`.
///
/// No `pairs.dedup()` here, by design.
///
/// The clause-emission paths construct `pairs` from the accumulator's
/// pair list under monotone-injective remaps (the `cd_map` lanes). For
/// non-marginal levels, the input pair list is canonical (sorted,
/// unique) and the output inherits that property — the pair-level
/// corollary of the no-compress proof covers this case.
///
/// For levels whose subtree includes a marginal child, the accumulator's
/// pair list may legitimately be a multiset: count-keyed slot sharing
/// (see `apply_p_fusion`) lets two
/// pairs `(L, R)` co-exist when each carries the `c(L)·c(R)` contribution
/// of one historical marginalization plan. Those duplicates survive
/// through clause apply and must be preserved here — deduplicating would
/// silently lose count.
///
/// **Do not add a defensive dedup here.** Duplicates in a marginal-aware
/// pair list are load-bearing for count correctness; collapsing them
/// reproduces a known count-corruption failure mode.
#[inline]
fn emit_clause_node(
    pairs: &mut Vec<InputPair>,
    level: &mut TddLevel,
    result_map: &mut [[u32; 2]],
    lane: usize,
    base_plus_idx: usize,
) -> Result<(), ApplyError> {
    if !pairs.is_empty() {
        result_map[base_plus_idx][lane] = level.nodes.len() as u32;
        level.try_push_internal_node(pairs).map_err(|_| ApplyError::OverBudget)?;
    } else {
        // Empty c_t/d_t — no node emitted. Write DEAD here (rather than relying
        // on a separate bulk pre-fill) so every map entry in this level's block
        // is written exactly once, in the loop that already visits it. See the
        // "no bulk DEAD-fill" note at the map-sizing site.
        result_map[base_plus_idx][lane] = DEAD;
    }
    Ok(())
}

/// Direct-emission variant of `emit_clause_node` for the `c_t` lane: the pairs
/// were pushed straight onto `level.pairs` starting at `pair_start`, skipping
/// the scratch-buffer staging + `extend_from_slice` copy of the buffered path
/// (a measurable slice of the batch-1 apply loop). Finalizes the node —
/// re-dispatching single-pair lists through `try_push_internal_node` so the
/// inline/ext encodings stay byte-identical to the buffered path — or writes
/// DEAD when no pairs were produced. The same canonicity contract as
/// `emit_clause_node` applies (no defensive dedup — see above).
#[inline]
fn emit_clause_node_direct(
    level: &mut TddLevel,
    pair_start: usize,
    result_map: &mut [[u32; 2]],
    lane: usize,
    base_plus_idx: usize,
) -> Result<(), ApplyError> {
    let pair_len = level.pairs.len() - pair_start;
    // No duplicate-pair assert here: like `emit_clause_node`, this path can
    // legitimately see multiset pair lists on levels whose subtree includes a
    // marginal child (count-keyed slot sharing) — see the multiset note above.
    if pair_len == 0 {
        result_map[base_plus_idx][lane] = DEAD;
    } else if pair_len == 1 {
        // Single pair: pop it back off the arena and re-dispatch so the
        // inline encoding (no arena slot) is preserved exactly.
        let pair = level.pairs[pair_start];
        level.pairs.truncate(pair_start);
        result_map[base_plus_idx][lane] = level.nodes.len() as u32;
        level.try_push_internal_node(&[pair]).map_err(|_| ApplyError::OverBudget)?;
    } else {
        // Invariant for `try_push_multi_by_range`: `pair_len >= 2` here — the
        // single-pair case is re-dispatched through `try_push_internal_node` in
        // the arm above. Its fast path only `debug_assert!`s this.
        result_map[base_plus_idx][lane] = level.nodes.len() as u32;
        level
            .try_push_multi_by_range(pair_start, pair_len)
            .map_err(|_| ApplyError::OverBudget)?;
    }
    Ok(())
}

/// Conjoin a TDD with a single clause directly, without constructing the
/// clause's TDD.
///
/// Equivalent to `apply_and(acc, clause_to_tdd(vtree, clause))` followed by
/// pruning, but faster: avoids the intermediate TDD allocation and prune pass
/// by computing the clause's contribution on-the-fly during the conjunction.
///
/// ## Virtual node model
///
/// At each vtree level, the clause implicitly defines two virtual nodes:
/// - **`c_t`** (clause node): "at least one literal in this subtree satisfies
///   the clause"
/// - **`d_t`** (complement): "no literal in this subtree satisfies the clause"
///
/// Instead of materializing these as TDD nodes, we maintain one flat
/// interleaved map (`cd_map`) indexed by `[level_base[t] + acc_node_index]`:
///   - `cd_map[base + i][0]` = output index for `acc[i] ∧ c_t` (DEAD if zero)
///   - `cd_map[base + i][1]` = output index for `acc[i] ∧ d_t` (DEAD if zero)
///
/// The final output is the conjunction of the accumulator's output with `c_t` at
/// the root level.
///
/// ## Spine-only traversal
///
/// The set of vtree levels that interact with the clause is exactly the union
/// of root-to-leaf paths for the clause's variables — a level is "relevant"
/// iff its subtree contains a clause variable iff it is an ancestor (inclusive)
/// of some clause-variable leaf. This "spine" (the Steiner tree of the clause's
/// leaves) is discovered directly via leaf lookup + parent-pointer walk and
/// processed by a post-order DFS, so the per-clause cost is O(spine) — we never
/// sweep the full vtree/TDD. The pooled flag/offset arrays stay sized to
/// `num_nodes` for O(1) indexing, but only spine entries are written and reset.
/// Walk root-paths from every clause literal, marking visited nodes in `visited`.
///
/// The dedup terminates early as soon as a node is already marked, so the
/// ancestor-closed union of root-paths is computed in O(sum of path lengths)
/// with no revisits. `visited` must already be sized to cover all vtree node
/// indices and have the relevant range zeroed by the caller.
///
/// `newly_marked`, when supplied, collects exactly the nodes this call flipped
/// from false to true — i.e. the clause's own spine minus whatever `visited`
/// already carried. That is what lets a caller accumulate the UNION of several
/// clauses' spines across calls without a second walk or a full-vtree scan:
/// the batch builder folds clauses into one diagram and needs the set of levels
/// those folds can have touched (the downstream driver's batch-build step). The
/// clause-apply path itself passes `None` — it recovers the same set from its
/// own post-order spine list.
#[inline(always)]
#[doc(hidden)]
pub fn walk_mark_spine(
    vtree: &crate::vtree::Vtree,
    clause: &[Literal],
    visited: &mut Vec<bool>,
    mut newly_marked: Option<&mut Vec<VtreeIdx>>,
) {
    for lit in clause {
        let mut cur = vtree.leaf_of(lit.var);
        loop {
            if visited[cur.idx()] { break; }
            visited[cur.idx()] = true;
            if let Some(out) = newly_marked.as_deref_mut() { out.push(cur); }
            match vtree.node(cur).parent() {
                Some(p) => cur = p,
                None => break,
            }
        }
    }
}

/// Phase 1 of `try_apply_and_clause`: build the clause spine.
///
/// Marks every ancestor (inclusive) of each clause-variable leaf in `on_spine`,
/// then collects spine internal nodes in post-order (bottom-up) into
/// `spine_internal` via a DFS over the marked subtree.
///
/// Total work is O(spine), not O(clause-len × height): the walk stops as soon
/// as it meets an already-marked node.
fn build_clause_spine(
    vtree: &crate::vtree::Vtree,
    clause: &[Literal],
    on_spine: &mut Vec<bool>,
    spine_internal: &mut Vec<VtreeIdx>,
    dfs_stack: &mut Vec<(VtreeIdx, bool)>,
) {
    walk_mark_spine(vtree, clause, on_spine, None);

    // Post-order DFS over the marked subtree (rooted at the vtree root, which is
    // always relevant — it is an ancestor of every leaf) collects the spine's
    // INTERNAL nodes bottom-up. The marked set is ancestor-closed, so it is a
    // connected subtree containing the root; descending only into marked
    // children keeps the DFS O(spine).
    spine_internal.clear();
    dfs_stack.clear();
    let root = vtree.root();
    if on_spine[root.idx()] && !vtree.node(root).is_leaf() {
        dfs_stack.push((root, false));
    }
    while let Some((t, processed)) = dfs_stack.pop() {
        if processed {
            spine_internal.push(t);
        } else {
            dfs_stack.push((t, true));
            let (l, r) = vtree.children(t);
            if on_spine[l.idx()] && !vtree.node(l).is_leaf() { dfs_stack.push((l, false)); }
            if on_spine[r.idx()] && !vtree.node(r).is_leaf() { dfs_stack.push((r, false)); }
        }
    }
    // `spine_internal` is now bottom-up (children precede parents).
}

/// Phase 2 of `try_apply_and_clause`: propagate `need_dt` top-down over the spine.
///
/// A level needs the complement conjunction (acc × `d_t`) iff its parent does, OR
/// both siblings are relevant (the both-relevant `c_t` spawns (`d_L,c_R`) and
/// (`c_L,d_R`), each consuming a `d_t` from one side). Iterating `spine_internal`
/// in reverse (top-down) and writing only marked children keeps `need_dt` clean
/// for irrelevant nodes (which are never read).
#[inline(always)]
fn propagate_need_dt(
    vtree: &crate::vtree::Vtree,
    spine_internal: &[VtreeIdx],
    on_spine: &[bool],
    need_dt: &mut Vec<bool>,
) {
    for &t in spine_internal.iter().rev() {
        let (l, r) = vtree.children(t);
        let both = on_spine[l.idx()] && on_spine[r.idx()];
        if on_spine[l.idx()] { need_dt[l.idx()] = need_dt[t.idx()] || both; }
        if on_spine[r.idx()] { need_dt[r.idx()] = need_dt[t.idx()] || both; }
    }
}

/// Inner loop for the both-relevant pair-accumulation step.
///
/// Accumulates `c_t` pairs of types 1/2/3 directly onto `level.pairs` (starting
/// at the caller-recorded `ct_start`) and `d_t` pairs into `clause_dt_pairs`.
/// Callers invoke this after reserving `ct_start = level.pairs.len()` and
/// clearing `clause_t3_buf`/`clause_dt_pairs`, then follow up with their chosen
/// emit variant (`emit_clause_node` / `emit_clause_node_direct`).
///
/// Corresponds to the "3 virtual `c_t` pairs × N acc pairs, FUSED with `d_t`" path
/// described in the main apply loop comment. See `try_apply_and_clause` for the
/// surrounding context.
#[inline(always)]
fn build_both_rel_pairs(
    inputs: &[InputPair],
    left_base: usize,
    right_base: usize,
    compute_dt: bool,
    cd_map: &[[u32; 2]],
    level: &mut TddLevel,
    clause_t3_buf: &mut Vec<InputPair>,
    clause_dt_pairs: &mut Vec<InputPair>,
) -> Result<(), ApplyError> {
    let mut prev_left = u32::MAX;
    for p in inputs {
        if p.left.0 != prev_left {
            level.pairs.extend_from_slice(clause_t3_buf);
            clause_t3_buf.clear();
            prev_left = p.left.0;
        }
        let [l_ct, l_dt] = cd_map[left_base + p.left.idx()];
        let [r_ct, r_dt] = cd_map[right_base + p.right.idx()];
        if l_ct != DEAD {
            if r_ct != DEAD {
                level.pairs.push(InputPair {
                    left: LocalNodeIdx(l_ct),
                    right: LocalNodeIdx(r_ct),
                });
            }
            if r_dt != DEAD {
                level.pairs.push(InputPair {
                    left: LocalNodeIdx(l_ct),
                    right: LocalNodeIdx(r_dt),
                });
            }
        }
        if l_dt != DEAD && r_ct != DEAD {
            try_push(clause_t3_buf, InputPair {
                left: LocalNodeIdx(l_dt),
                right: LocalNodeIdx(r_ct),
            })?;
        }
        if compute_dt && l_dt != DEAD && r_dt != DEAD {
            try_push(clause_dt_pairs, InputPair {
                left: LocalNodeIdx(l_dt),
                right: LocalNodeIdx(r_dt),
            })?;
        }
    }
    level.pairs.extend_from_slice(clause_t3_buf);
    Ok(())
}

/// Inner loop for the single-relevant pair-accumulation step.
///
/// Accumulates `c_t` pairs directly onto `level.pairs` (starting at the caller-
/// recorded `ct_start`) and `d_t` pairs into `clause_dt_pairs`. The caller sets
/// `ct_start = level.pairs.len()` and clears `clause_dt_pairs` before the call,
/// then follows up with its chosen emit variant.
///
/// `left_rel` — true if the left child is the relevant one; false if the right
/// child is. `left_base`/`right_base` are the `cd_map` offsets for each child.
/// The irrelevant side's map is not filled; the raw pair index is used directly.
///
/// Corresponds to the "single virtual pair" path in the main apply loop.
/// See `try_apply_and_clause` for context.
#[inline(always)]
fn build_single_rel_pairs(
    inputs: &[InputPair],
    left_rel: bool,
    left_base: usize,
    right_base: usize,
    compute_dt: bool,
    cd_map: &[[u32; 2]],
    level: &mut TddLevel,
    clause_dt_pairs: &mut Vec<InputPair>,
) -> Result<(), ApplyError> {
    for p in inputs {
        let e = if left_rel {
            cd_map[left_base + p.left.idx()]
        } else {
            cd_map[right_base + p.right.idx()]
        };
        let (l, r) = if left_rel { (e[0], p.right.0) } else { (p.left.0, e[0]) };
        if l != DEAD && r != DEAD {
            level.pairs.push(InputPair {
                left: LocalNodeIdx(l),
                right: LocalNodeIdx(r),
            });
        }
        if compute_dt {
            let (l, r) = if left_rel { (e[1], p.right.0) } else { (p.left.0, e[1]) };
            if l != DEAD && r != DEAD {
                try_push(clause_dt_pairs, InputPair {
                    left: LocalNodeIdx(l),
                    right: LocalNodeIdx(r),
                })?;
            }
        }
    }
    Ok(())
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
    // marginalized scope), so panic at the gateway. Production avoids this via
    // the marginalize schedule (installed by the downstream compile driver).
    let mut level_base = pool_take(&SCRATCH_CLAUSE_LEVEL_BASE);
    if level_base.len() < num_nodes { level_base.resize(num_nodes, 0usize); }
    let mut total = 0usize;
    for lit in clause {
        let ti = vtree.leaf_of(lit.var).idx();
        level_base[ti] = total;
        total += LEAF_WIDTH;
    }
    for &t in &spine_internal {
        let ti = t.idx();
        if levels[ti].is_marginal() {
            panic!(
                "try_apply_and_clause: clause literal under marginal vtree subtree \
                 (vtree t={ti}, marginal-count width={}). The clause references a \
                 variable whose scope has already been marginalized in the accumulator \
                 — callers must marginalize a subtree only after every clause touching \
                 its variables has been applied.",
                levels[ti].width(),
            );
        }
        level_base[ti] = total;
        total += levels[ti].width();
    }

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

    // Leaf fill: only the clause's own leaves (one per literal). Leaf levels are
    // marginal (no node creation) — the clause label index and its complement
    // map directly to CONJOIN_GRID columns. Under the leaf encoding
    // (One=0, Pos=1, Neg=2): positive literal → c_t=Pos(1), d_t=Neg(2);
    // negative literal → c_t=Neg(2), d_t=Pos(1).
    for lit in clause {
        let t = vtree.leaf_of(lit.var);
        let base = level_base[t.idx()];
        let compute_dt = need_dt[t.idx()];
        let (clause_idx, compl_idx) = if lit.positive {
            (POS_LEAF_IDX.0 as usize, NEG_LEAF_IDX.0 as usize)
        } else {
            (NEG_LEAF_IDX.0 as usize, POS_LEAF_IDX.0 as usize)
        };
        for i in 0..LEAF_WIDTH {
            let dt = if compute_dt { CONJOIN_GRID[i][compl_idx] } else { DEAD };
            cd_map[base + i] = [CONJOIN_GRID[i][clause_idx], dt];
        }
    }

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
        let t_idx = t.idx();
        let (left, right) = vtree.children(t);
        let li = left.idx();
        let ri = right.idx();
        let base = level_base[t_idx];
        let compute_dt = need_dt[t_idx];
        let left_rel = on_spine[li];
        let right_rel = on_spine[ri];
        let both_rel = left_rel && right_rel;
        let left_base = level_base[li];
        let right_base = level_base[ri];

        // Swap the old (input) level out so we can rebuild in place. `old` holds
        // the accumulator's pairs for this level; the freshly emptied
        // `levels[t_idx]` receives the conjoined output. (`old` is dropped the
        // moment the emit loop ends — see the drop site below, which must
        // precede the rebuilt level's shrink.)
        let old = std::mem::replace(&mut levels[t_idx], TddLevel::new());
        let k = old.width();
        let in_pairs = old.pairs.len();
        let level = &mut levels[t_idx];
        // Pre-size the output node Vec to skip the per-node doubling reallocs in
        // the emit loop. The bound is cheap — no cross-product recount: <= 2k
        // nodes (a c_t and a d_t per acc node). try_reserve keeps the
        // OverBudget contract.
        let node_cap = if compute_dt { 2 * k } else { k };
        level.nodes.try_reserve(node_cap).map_err(|_| ApplyError::OverBudget)?;
        // Worst-case output pairs PER INPUT PAIR: a both_rel node emits up to 3
        // c_t pairs (type1/2/3), plus 1 d_t pair when compute_dt — so a level's
        // output can reach 4x its input. The slab is sized at 1x (the input-pair
        // count) and the per-node top-up in the loop grows it on demand, so the
        // peak never carries a whole-level worst case beside the still-live
        // `old`. The c_t emit pushes DIRECTLY onto level.pairs — the per-node
        // top-up, not a per-pair reserve, is what makes those pushes safe.
        let pair_mult = (if both_rel { 3 } else { 1 }) + usize::from(compute_dt);
        // Same near-cap growth-mode decision the dense emit walk makes, fed the
        // same kind of sound emit bound, so the top-ups below take bounded
        // headroom-aware increments instead of doubling on a huge level.
        decide_emit_growth_mode(false, (in_pairs as u128).saturating_mul(pair_mult as u128));
        level.pairs.try_reserve(in_pairs).map_err(|_| ApplyError::OverBudget)?;

        for i in 0..k {
            debug_assert!(old.nodes[i].is_internal()
                || old.nodes[i].b == u32::MAX,  // inline pair with right=ZERO (dead node)
                "expected internal node at internal vtree position: t={t:?} i={i}");
            let inputs = old.pairs_of_idx(i);
            if inputs.is_empty() {
                // Dead acc node — no c_t/d_t emitted. Write DEAD so this entry
                // is initialized (no separate bulk fill); a parent referencing
                // this idx must read DEAD.
                cd_map[base + i] = [DEAD, DEAD];
                continue;
            }
            // Guarantee this node's whole worst case before emitting any of it,
            // so the direct c_t pushes stay infallible `Vec::push`es and the d_t
            // extend inside `try_push_internal_node` cannot realloc mid-node.
            // A refused top-up surfaces as `OverBudget` — the same error class
            // every other reserve on this path returns.
            reserve_pairs_for_emit(level, pair_mult * inputs.len())?;

            // ── Conjunction with c_t (clause node) ──
            //
            // The clause's virtual c_t represents "clause satisfied in this
            // subtree". Its shape depends on which children have clause
            // variables:
            //
            //   Only right relevant:  c_t = {(d_L, c_R)}                        — 1 virtual pair
            //   Only left relevant:   c_t = {(c_L, d_R)}                        — 1 virtual pair
            //   Both relevant:        c_t = {(c_L,c_R), (c_L,d_R), (d_L,c_R)}  — 3 virtual pairs
            //
            // The 3-pair case captures: "satisfied iff at least one side is
            // satisfied" = all combos except (d_L, d_R) = 1 − d_L·d_R.
            //
            // Single-pair cases: maps are monotone → output already sorted.
            if both_rel {
                // 3 virtual c_t pairs × N acc pairs, FUSED with the d_t
                // conjunction in one scan over the inputs. See `build_both_rel_pairs`
                // for the full algorithm description; the emit calls are caller-side
                // because they diverge between the allocating and in-place paths.
                let ct_start = level.pairs.len();
                clause_t3_buf.clear();
                if compute_dt { clause_dt_pairs.clear(); }
                build_both_rel_pairs(
                    inputs, left_base, right_base, compute_dt,
                    &cd_map, level, &mut clause_t3_buf, &mut clause_dt_pairs,
                )?;
                emit_clause_node_direct(level, ct_start, &mut cd_map, 0, base + i)?;
                // Emit d_t AFTER c_t so the ct lane < dt lane of cd_map — the
                // index ordering the parent level's both_rel pass relies on.
                if compute_dt {
                    emit_clause_node(
                        &mut clause_dt_pairs, level, &mut cd_map, 1, base + i,
                    )?;
                }
            } else {
                // Single virtual pair: only one child is relevant. See
                // `build_single_rel_pairs` for the fused c_t/d_t algorithm;
                // emit calls are caller-side (diverge between allocating/in-place).
                let ct_start = level.pairs.len();
                if compute_dt { clause_dt_pairs.clear(); }
                build_single_rel_pairs(
                    inputs, left_rel, left_base, right_base, compute_dt,
                    &cd_map, level, &mut clause_dt_pairs,
                )?;
                emit_clause_node_direct(level, ct_start, &mut cd_map, 0, base + i)?;
                if compute_dt {
                    emit_clause_node(
                        &mut clause_dt_pairs, level, &mut cd_map, 1, base + i,
                    )?;
                }
            }
        }
        // The emit loop was the last reader of `old`; only its Copy marg flags
        // are still needed. Free the dead input level HERE, before
        // `shrink_arrays` — that shrink reallocs the rebuilt arenas (alloc +
        // copy + free), so anything still holding `old` pays both arenas plus
        // the realloc's destination copy at the peak.
        let old_marg_flags = old.marg_flags;
        drop(old);
        // Trim the slack the per-node top-up growth left behind.
        level.shrink_arrays();

        // The rebuilt level copied the irrelevant side's pair refs verbatim
        // — including inline marg counts (bit 30) toward a marginal sibling
        // child — but started from a fresh `TddLevel::new()` whose
        // `marg_flags` are zero. Carry the markers over: the relevant side
        // is never marginal (gateway panic above), so its flags are false in
        // the input level and the wholesale copy is exact.
        level.marg_flags = old_marg_flags;
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
    for lit in clause {
        let ti = vtree.leaf_of(lit.var).idx();
        on_spine[ti] = false;
        need_dt[ti] = false;
    }
    for &t in &spine_internal {
        on_spine[t.idx()] = false;
        need_dt[t.idx()] = false;
    }

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
/// Use only when no soft apply budget is armed (tests, bench harnesses);
/// hot-path callers in the downstream compile driver use `try_apply_and_clause` directly so
/// the vsplit driver can catch `OverBudget` and case-split.
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
#[path = "conjoin_clause_tests.rs"]
mod emit_reserve_tests;
