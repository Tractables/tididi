//! One cell of the product: the pair walk and the sinks it writes through.

use super::*;
use crate::limits::PollGate;

/// Fold the per-row alive-column masks for one decoded f row.
///
/// left: pass-through ⇒ always alive (`MAX`); `!both_multi_pair` ⇒ masks unused (`0`);
/// else the OR of the side's `live_cols` over the row's left refs. right mirrors
/// (`MAX` when pass-through OR `!both_multi_pair`). Returns `None` when the row is
/// provably dead under `both_multi_pair` (every cell in it would be culled) — the caller
/// skips the whole row.
#[inline(always)]
pub(crate) fn row_alive_masks(ctx: &CellCtx<'_>, inputs1: &[ChildPair]) -> Option<(u128, u128)> {
    let left_alive_mask: u128 = if ctx.sides.left.plan.is_passthrough() {
        u128::MAX // pass-through side: no grid; always alive
    } else if !ctx.both_multi_pair {
        0u128
    } else {
        inputs1.iter().fold(0u128, |acc, p1| acc | ctx.sides.left.live_cols[p1.left.idx()])
    };
    if ctx.both_multi_pair && left_alive_mask == 0 { return None; }

    let right_alive_mask: u128 = if ctx.sides.right.plan.is_passthrough() || !ctx.both_multi_pair {
        u128::MAX
    } else {
        inputs1.iter().fold(0u128, |acc, p1| acc | ctx.sides.right.live_cols[p1.right.idx()])
    };
    if ctx.both_multi_pair && right_alive_mask == 0 { return None; }

    Some((left_alive_mask, right_alive_mask))
}

/// Emit a product node from pairs accumulated in `level.pairs[pair_start..]`.
/// Handles inline (1 pair) vs multi-pair encoding and updates `node_idx`.
/// For huge cells (pair_start or pair_count ≥ 2^31), uses the extended
/// side-table encoding via `level.try_push_multi_by_range`.
#[inline(always)]
pub(crate) fn emit_product_node(
    eng: &Engine,
    level: &mut TddLevel,
    node_idx: &mut [u32],
    grid_pos: usize,
    pair_start: usize,
    pair_count: usize,
) -> Result<(), OperationError> {
    if pair_count > 0 {
        let nid = level.nodes.len() as u32;
        node_idx[grid_pos] = nid;
        if pair_count == 1 {
            // Phase F: pop from whichever backing is active.
            let pair = level.pop_pair().unwrap();
            emit_single_pair(eng, level, pair)?;
        } else {
            // Invariant for `try_push_multi_by_range`: `pair_count >= 2` here —
            // the single-pair case is dispatched to the inline/extended path in
            // the arm above. Its fast path only `debug_assert!`s this.
            level.try_push_multi_by_range(pair_start, pair_count)
                .map_err(|_| OperationError::OverBudget)?;
        }
    }
    Ok(())
}

/// Emit a node holding exactly `pair`, which is not in the arena yet: inline
/// when the pair fits, else pushed with a one-pair range. Every push is
/// budget-charged.
#[inline(always)]
fn emit_single_pair(eng: &Engine, level: &mut TddLevel, pair: ChildPair) -> Result<(), OperationError> {
    let lim = eng.limits();
    if pair.can_inline() {
        lim.try_push(&mut level.nodes, EncodedNode::inline(pair))
    } else {
        let ps = level.pair_count();
        try_push_pair_into(eng, level, pair)?;
        let ei = level.multi_pairs.len();
        lim.try_push(&mut level.multi_pairs, MultiPairRange { start: ps as u64, len: 1 })?;
        lim.try_push(&mut level.nodes, EncodedNode::multi_ranged(ei as u32))
    }
}

// ============================== Pair sinks ==============================
//
// The per-cell product walk (`process_cell`) is one kernel generic over the
// per-pair action, expressed as a `PairSink`: every action runs the same loop
// and only the sink differs.

/// Per-pair action of the cell product walk. All hooks are `#[inline(always)]`
/// in impls, so each kernel instantiation monomorphizes to a specialized walk.
pub(crate) trait PairSink {
    /// Whether the kernel asserts the g parent node is structurally internal.
    /// True for the emit walks; false for count/collect walks, which may visit
    /// marginal-encoded
    /// operand nodes (e.g. the both-marginal collapse).
    const ASSERT_INTERNAL: bool;

    /// 1×1 cell fast path: the cell's single surviving pair. The emit impl
    /// writes `node_idx[grid_pos]` and builds the node directly, without a
    /// push-then-pop round trip through the pair arena.
    fn single(
        &mut self,
        eng: &Engine,
        node_idx: &mut [u32],
        grid_pos: usize,
        lc: u32,
        rc: u32,
    ) -> Result<(), OperationError>;

    /// Start a multi-pair cell; returns the start token `end` consumes
    /// (the emit impl snapshots `level.pair_count()`).
    fn begin(&mut self) -> usize;

    /// One surviving (lc, rc) pair of a multi-pair cell.
    fn pair(&mut self, eng: &Engine, lc: u32, rc: u32) -> Result<(), OperationError>;

    /// Finish a multi-pair cell; returns the number of pairs committed
    /// (non-emit impls return 0).
    fn end(
        &mut self,
        eng: &Engine,
        node_idx: &mut [u32],
        grid_pos: usize,
        start: usize,
    ) -> Result<usize, OperationError>;
}

/// Build the output level: push pairs, emit product nodes, write `node_idx`.
pub(crate) struct EmitSink<'a> {
    pub(crate) level: &'a mut TddLevel,
}

impl PairSink for EmitSink<'_> {
    const ASSERT_INTERNAL: bool = true;

    #[inline(always)]
    fn single(
        &mut self, eng: &Engine, node_idx: &mut [u32],
        grid_pos: usize,
        lc: u32,
        rc: u32,
    ) -> Result<(), OperationError> {
        let pair = ChildPair { left: NodeIdx(lc), right: NodeIdx(rc) };
        let nid = self.level.nodes.len() as u32;
        node_idx[grid_pos] = nid;
        emit_single_pair(eng, self.level, pair)
    }

    #[inline(always)]
    fn begin(&mut self) -> usize {
        self.level.pair_count()
    }

    #[inline(always)]
    fn pair(&mut self, eng: &Engine, lc: u32, rc: u32) -> Result<(), OperationError> {
        try_push_pair_into(
            eng,
            self.level,
            ChildPair { left: NodeIdx(lc), right: NodeIdx(rc) },
        )
    }

    #[inline(always)]
    fn end(
        &mut self, eng: &Engine, node_idx: &mut [u32],
        grid_pos: usize,
        start: usize,
    ) -> Result<usize, OperationError> {
        let pair_count = self.level.pair_tail_len(start);
        emit_product_node(eng, self.level, node_idx, grid_pos, start, pair_count)?;
        Ok(pair_count)
    }
}

/// Collect surviving `(lc, rc)` pairs into a caller-owned scratch Vec — the
/// streaming collapse-at-source walk (`run_level_rows_stream_count`). Touches
/// neither `level` nor `node_idx[grid_pos]`; the caller feeds the collected
/// refs straight to `compute_cell_count`. Pushes are budget-tracked
/// (`try_push`), matching the emit walk's `try_push_pair_into`: a wide
/// streaming cell's scratch growth charges the apply soft budget and degrades
/// to `Err(OverBudget)` instead of an allocator abort. (The generic route
/// charges the same transient — it materializes these pairs into
/// `level.pairs` before truncating — so this keeps the collapse route's
/// budget accounting equivalent.) The scratch is bounded to one cell's pairs
/// and reused across cells.
pub(crate) struct CollectSink<'a> {
    pub(crate) out: &'a mut Vec<ChildPair>,
}

impl PairSink for CollectSink<'_> {
    // Collect walks may visit marginal-encoded operand nodes (the both-marginal
    // collapse), which the structural-internal assert would reject.
    const ASSERT_INTERNAL: bool = false;

    #[inline(always)]
    fn single(
        &mut self, eng: &Engine, _node_idx: &mut [u32],
        _grid_pos: usize,
        lc: u32,
        rc: u32,
    ) -> Result<(), OperationError> {
        let lim = eng.limits();
        lim.try_push(self.out, ChildPair { left: NodeIdx(lc), right: NodeIdx(rc) })
    }

    #[inline(always)]
    fn begin(&mut self) -> usize {
        0
    }

    #[inline(always)]
    fn pair(&mut self, eng: &Engine, lc: u32, rc: u32) -> Result<(), OperationError> {
        let lim = eng.limits();
        lim.try_push(self.out, ChildPair { left: NodeIdx(lc), right: NodeIdx(rc) })
    }

    #[inline(always)]
    fn end(
        &mut self,
        _eng: &Engine,
        _node_idx: &mut [u32],
        _grid_pos: usize,
        _start: usize,
    ) -> Result<usize, OperationError> {
        Ok(0)
    }
}

/// The one-sided product walk: one operand contributes a single pair, the other
/// is swept. `ITER_C1` says which — `true` sweeps `inputs1` against the lone g
/// pair (N×1), `false` sweeps `inputs2` against the lone f pair (1×N). The
/// grid lookup is ordered `(f field, g field)` in both directions.
///
/// Both directions cull on the reach masks first — if no live left (resp.
/// right) column can reach g-node `j`'s children, every lookup below is `NO_PRODUCT`
/// and the cell emits nothing. The cull is gated on `both_multi_pair` because the reach
/// masks exist only when both levels are multi-pair.
///
/// The pair count is known before the sweep, so the whole cell is charged to
/// the work clock in one go: one branch per cell rather than one per pair.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn cell_one_sided<const ITER_C1: bool, L, R, S>(
    eng: &Engine,
    j: usize,
    inputs1: &[ChildPair],
    inputs2: &[ChildPair],
    left_alive_mask: u128,
    right_alive_mask: u128,
    ctx: &CellCtx<'_>,
    node_idx: &mut [u32],
    grid_pos: usize,
    left: &L,
    right: &R,
    sink: &mut S,
    gate: &mut PollGate,
) -> Result<(), OperationError>
where
    L: ChildLookup,
    R: ChildLookup,
    S: PairSink,
{
    let lim = eng.limits();
    let both_multi_pair = ctx.both_multi_pair;
    if both_multi_pair && !left.passthrough() && left_alive_mask & ctx.sides.left.reach[j] == 0 {
        return Ok(());
    }
    if both_multi_pair && !right.passthrough() && right_alive_mask & ctx.sides.right.reach[j] == 0 {
        return Ok(());
    }
    let n = if ITER_C1 { inputs1.len() } else { inputs2.len() };
    lim.poll(gate, n as u64)?;
    let cell_start = sink.begin();
    for k in 0..n {
        let (p1, p2) = if ITER_C1 {
            (&inputs1[k], &inputs2[0])
        } else {
            (&inputs1[0], &inputs2[k])
        };
        let lc = left.get(node_idx, p1.left.0, p2.left.0);
        if lc == NO_PRODUCT { continue; }
        let rc = right.get(node_idx, p1.right.0, p2.right.0);
        if rc == NO_PRODUCT { continue; }
        sink.pair(eng, lc, rc)?;
    }
    sink.end(eng, node_idx, grid_pos, cell_start)?;
    Ok(())
}

/// The general product walk: every f pair against every g pair, with the
/// dead-pair pre-filter culling rows and columns that cannot contribute.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn cell_prefilter<L, R, S>(
    eng: &Engine,
    j: usize,
    inputs1: &[ChildPair],
    inputs2: &[ChildPair],
    left_alive_mask: u128,
    right_alive_mask: u128,
    ctx: &CellCtx<'_>,
    node_idx: &mut [u32],
    grid_pos: usize,
    left: &L,
    right: &R,
    sink: &mut S,
    gate: &mut PollGate,
) -> Result<(), OperationError>
where
    L: ChildLookup,
    R: ChildLookup,
    S: PairSink,
{
    let lim = eng.limits();
    // ── N×M (implies both_multi_pair: both levels multi-pair ⟹ masks built) ──────
    let left_dead = !left.passthrough()
        && left_alive_mask & ctx.sides.left.reach[j] == 0;
    let right_dead = !right.passthrough()
        && right_alive_mask & ctx.sides.right.reach[j] == 0;
    if left_dead || right_dead {
        return Ok(());
    }

    let cell_start = sink.begin();
    if !left.passthrough() && !right.passthrough()
        && inputs1.len() >= 64 && inputs2.len() >= 64
    {
        // ── Grouped N×M: run-length groups by shared `.left` ──────────
        // Emits the same pair multiset as the general double loop, in
        // grouped order; every consumer is order-independent (pair lists
        // are unordered sets — see diagram/level/mod.rs).
        let n2 = inputs2.len();
        let mut groups2: smallvec::SmallVec<[(usize, usize); 256]> =
            smallvec::SmallVec::new();
        {
            let mut idx = 0;
            while idx < n2 {
                let start = idx;
                let left = inputs2[idx].left;
                idx += 1;
                while idx < n2 && inputs2[idx].left == left { idx += 1; }
                groups2.push((start, idx));
            }
        }
        let n1 = inputs1.len();
        let mut p1_idx = 0;
        while p1_idx < n1 {
            let p1_left = inputs1[p1_idx].left;
            let g1_start = p1_idx;
            p1_idx += 1;
            while p1_idx < n1 && inputs1[p1_idx].left == p1_left { p1_idx += 1; }
            let g1 = &inputs1[g1_start..p1_idx];
            lim.poll(gate, (g1.len() * n2) as u64)?;

            if ctx.sides.left.live_cols[p1_left.idx()] & ctx.sides.left.reach[j] == 0 { continue; }

            for &(g2s, g2e) in &groups2 {
                let p2_left = inputs2[g2s].left;
                let lc = left.get(node_idx, p1_left.0, p2_left.0);
                if lc == NO_PRODUCT { continue; }
                let g2 = &inputs2[g2s..g2e];

                for p1 in g1 {
                    if ctx.sides.right.live_cols[p1.right.idx()] & ctx.sides.right.reach[j] == 0 {
                        continue;
                    }
                    for p2 in g2 {
                        let rc = right.get(node_idx, p1.right.0, p2.right.0);
                        if rc == NO_PRODUCT { continue; }
                        sink.pair(eng, lc, rc)?;
                    }
                }
            }
        }
    } else {
        // ── General N×M ───────────────────────────────────────────────
        for p1 in inputs1 {
            lim.poll(gate, inputs2.len() as u64)?;
            if !left.passthrough()
                && ctx.sides.left.live_cols[p1.left.idx()] & ctx.sides.left.reach[j] == 0 {
                continue;
            }
            if !right.passthrough()
                && ctx.sides.right.live_cols[p1.right.idx()] & ctx.sides.right.reach[j] == 0 {
                continue;
            }
            for p2 in inputs2 {
                let lc = left.get(node_idx, p1.left.0, p2.left.0);
                if lc == NO_PRODUCT { continue; }
                let rc = right.get(node_idx, p1.right.0, p2.right.0);
                if rc == NO_PRODUCT { continue; }
                sink.pair(eng, lc, rc)?;
            }
        }
    }
    sink.end(eng, node_idx, grid_pos, cell_start)?;
    Ok(())
}

/// Merged per-cell product walk — one kernel for every dense cell action.
///
/// The lookups (`L`, `R`) resolve child refs per representation (dense grid /
/// sparse point index / marginal pass-through — see `child_lookup.rs`); the
/// sink (`S`) is the per-pair action (emit / count / collect).
///
/// Arms: 1×1 (single-pair fast path via `sink.single`), N×1 / 1×N (one side
/// single — [`cell_one_sided`], one loop in both directions), N×M (reach-mask
/// culls + the ≥64×64 grouped fast path when neither side is pass-through). A cell in the N×M arm implies `ctx.both_multi_pair` (both sides
/// having >1 pairs means both levels have multi-pair nodes), so the
/// liveness/reach arrays are always built when the culls read them.
///
/// `ChildLookup::passthrough()` is a constant `false` on `DenseLookup`, so the
/// plain instantiations carry no pass-through branch in the inner loops.
///
/// The `inputs2_scratch` lifetime is independent from `node_idx`: `right_level`
/// borrows from a separate `Tdd` operand, and `pairs_view_decoded` borrows
/// `inputs2_scratch` as the decode buffer — neither aliases the output slab.
///
/// Returns `Err(OperationError)`: `OverBudget` on sink allocation failure, and
/// `Deadline` from the intra-cell poll in any arm — so even a
/// count-only sink is not infallible (it can bail mid-cell on a wide cell).
///
/// The parameters stay positional. Bundling them behind a reference was
/// measured: a slice loaded out of a struct carries none of the aliasing
/// facts a reference parameter does, so the inner pair loops re-read the f
/// row's fields and a lookup's geometry on every pair, and the kernel's
/// instruction count rose by about a sixth on dense instances. The two
/// arms below are inlined into it and keep the same parameter list.
#[allow(clippy::too_many_arguments)]
pub(crate) fn process_cell<L, R, S>(
    eng: &Engine,
    j: usize,
    row_base: usize,
    inputs1: &[ChildPair],
    left_alive_mask: u128,
    right_alive_mask: u128,
    ctx: &CellCtx<'_>,
    right_level: &TddLevel,
    inputs2_scratch: &mut Vec<ChildPair>,
    node_idx: &mut [u32],
    left: &L,
    right: &R,
    sink: &mut S,
    gate: &mut PollGate,
) -> Result<(), OperationError>
where
    L: ChildLookup,
    R: ChildLookup,
    S: PairSink,
{
    let lim = eng.limits();
    if S::ASSERT_INTERNAL {
        debug_assert!(
            right_level.nodes[j].is_internal() || right_level.nodes[j].b == u32::MAX,
            "expected internal node at internal vtree position: j={j} right_width={} node_a={:#x} node_b={:#x}",
            ctx.right_width, right_level.nodes[j].a, right_level.nodes[j].b
        );
    }

    // Column `j`'s pairs. Everything about resolving them — masks, encoding,
    // range — depends only on `j` and the level, so it was hoisted into the
    // per-level [`RightColumns`] table and this is two loads. `None` is the
    // fallback for the levels the table declines (marginal-encoded g, or an
    // arena the budget rejected): re-derive per cell, as before.
    let inputs2 = match ctx.right_cols {
        Some(cols) => cols.get(j),
        None => right_level.pairs_view_decoded(j, inputs2_scratch, ctx.sides.left.plan.view, ctx.sides.right.plan.view),
    };
    if inputs2.is_empty() { return Ok(()); }

    // `row_base` is the row's flat slab offset, already computed by the row loop
    // (`ctx.output_grid_base + grid_row * ctx.right_width`) for its `NO_PRODUCT` reset — reuse it instead of
    // re-deriving the same product per cell.
    let grid_pos = row_base + j;

    if inputs1.len() == 1 && inputs2.len() == 1 {
        // ── 1×1 ──────────────────────────────────────────────────────────
        lim.poll(gate, 1)?;
        let p1 = &inputs1[0];
        let p2 = &inputs2[0];
        let lc = left.get(node_idx, p1.left.0, p2.left.0);
        if lc != NO_PRODUCT {
            let rc = right.get(node_idx, p1.right.0, p2.right.0);
            if rc != NO_PRODUCT {
                sink.single(eng, node_idx, grid_pos, lc, rc)?;
            }
        }
    } else if inputs2.len() == 1 {
        cell_one_sided::<true, _, _, _>(
            eng, j, inputs1, inputs2, left_alive_mask, right_alive_mask, ctx,
            node_idx, grid_pos, left, right, sink, gate,
        )?;
    } else if inputs1.len() == 1 {
        cell_one_sided::<false, _, _, _>(
            eng, j, inputs1, inputs2, left_alive_mask, right_alive_mask, ctx,
            node_idx, grid_pos, left, right, sink, gate,
        )?;
    } else {
        cell_prefilter(
            eng,
            j, inputs1, inputs2, left_alive_mask, right_alive_mask, ctx,
            node_idx, grid_pos, left, right, sink, gate,
        )?;
    }
    Ok(())
}
