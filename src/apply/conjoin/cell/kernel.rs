//! One cell of the product: the pair walk and the sinks it writes through.

use super::*;
use crate::engine::PollGate;

/// How much intra-cell product work passes between two asks of the stop.
///
/// Coarser than the between-cell stride because this poll rides inside one
/// cell's product loop: a single very wide cell can push hundreds of millions
/// of pairs without ever reaching a cell boundary, and the ask has to cost at
/// most one branch per outer iteration.
const INTRA_CELL_POLL_STRIDE: u64 = 1 << 20;

/// Fold the per-row alive-column masks for one decoded c1 row.
///
/// left: pass-through ⇒ always alive (`MAX`); `!nxm` ⇒ masks unused (`0`);
/// else the OR of `live_left_cols` over the row's left refs. right mirrors
/// (`MAX` when pass-through OR `!nxm`). Returns `None` when the row is
/// provably dead under `nxm` (every cell in it would be culled) — the caller
/// skips the whole row.
#[inline(always)]
pub(crate) fn row_alive_masks(ctx: &CellCtx<'_>, inputs1: &[InputPair]) -> Option<(u128, u128)> {
    let left_alive_mask: u128 = if ctx.left_passthrough {
        u128::MAX // pass-through side: no grid; always alive
    } else if !ctx.nxm {
        0u128
    } else {
        inputs1.iter().fold(0u128, |acc, p1| acc | ctx.live_left_cols[p1.left.idx()])
    };
    if ctx.nxm && left_alive_mask == 0 { return None; }

    let right_alive_mask: u128 = if ctx.right_passthrough || !ctx.nxm {
        u128::MAX
    } else {
        inputs1.iter().fold(0u128, |acc, p1| acc | ctx.live_right_cols[p1.right.idx()])
    };
    if ctx.nxm && right_alive_mask == 0 { return None; }

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
) -> Result<(), ApplyError> {
    let lim = eng.limits();
    if pair_count > 0 {
        let nid = level.nodes.len() as u32;
        node_idx[grid_pos] = nid;
        if pair_count == 1 {
            // Phase F: pop from whichever backing is active.
            let pair = level.pop_pair().unwrap();
            if pair.can_inline() {
                lim.try_push(&mut level.nodes, TddNodeData::inline(pair))?;
            } else {
                let ps = level.pair_count();
                try_push_pair_into(eng, level, pair)?;
                let ei = level.ext.len();
                lim.try_push(&mut level.ext, ExtMulti { start: ps as u64, len: 1 })?;
                lim.try_push(&mut level.nodes, TddNodeData::multi_extended(ei as u32))?;
            }
        } else {
            // Invariant for `try_push_multi_by_range`: `pair_count >= 2` here —
            // the single-pair case is dispatched to the inline/extended path in
            // the arm above. Its fast path only `debug_assert!`s this.
            level.try_push_multi_by_range(pair_start, pair_count)
                .map_err(|_| ApplyError::OverBudget)?;
        }
    }
    Ok(())
}

// ============================== Pair sinks ==============================
//
// The per-cell product walk (`process_cell`) is ONE kernel generic over the
// per-pair ACTION, expressed as a `PairSink`. Historically the walk was
// copy-pasted per action (emit / streaming-collect) and the copies were kept
// in sync by "mirrors exactly" comments; now all actions run the same loop,
// only the sink differs.

/// Per-pair action of the cell product walk. All hooks are `#[inline(always)]`
/// in impls so each kernel instantiation monomorphizes to the same code the
/// historical hand-written copy produced.
pub(crate) trait PairSink {
    /// Whether the kernel asserts the c2 parent node is structurally internal.
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
    ) -> Result<(), ApplyError>;

    /// Start a multi-pair cell; returns the start token `end` consumes
    /// (the emit impl snapshots `level.pair_count()`).
    fn begin(&mut self) -> usize;

    /// One surviving (lc, rc) pair of a multi-pair cell.
    fn pair(&mut self, eng: &Engine, lc: u32, rc: u32) -> Result<(), ApplyError>;

    /// Finish a multi-pair cell; returns the number of pairs committed
    /// (non-emit impls return 0).
    fn end(
        &mut self,
        eng: &Engine,
        node_idx: &mut [u32],
        grid_pos: usize,
        start: usize,
    ) -> Result<usize, ApplyError>;
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
    ) -> Result<(), ApplyError> {
        let lim = eng.limits();
        let pair = InputPair { left: LocalNodeIdx(lc), right: LocalNodeIdx(rc) };
        let nid = self.level.nodes.len() as u32;
        node_idx[grid_pos] = nid;
        if pair.can_inline() {
            lim.try_push(&mut self.level.nodes, TddNodeData::inline(pair))
        } else {
            let ps = self.level.pair_count();
            try_push_pair_into(eng, self.level, pair)?;
            let ei = self.level.ext.len();
            lim.try_push(&mut self.level.ext, ExtMulti { start: ps as u64, len: 1 })?;
            lim.try_push(&mut self.level.nodes, TddNodeData::multi_extended(ei as u32))
        }
    }

    #[inline(always)]
    fn begin(&mut self) -> usize {
        self.level.pair_count()
    }

    #[inline(always)]
    fn pair(&mut self, eng: &Engine, lc: u32, rc: u32) -> Result<(), ApplyError> {
        try_push_pair_into(
            eng,
            self.level,
            InputPair { left: LocalNodeIdx(lc), right: LocalNodeIdx(rc) },
        )
    }

    #[inline(always)]
    fn end(
        &mut self, eng: &Engine, node_idx: &mut [u32],
        grid_pos: usize,
        start: usize,
    ) -> Result<usize, ApplyError> {
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
/// charges the SAME transient — it materializes these pairs into
/// `level.pairs` before truncating — so this keeps the collapse route's
/// budget accounting equivalent.) The scratch is bounded to one cell's pairs
/// and reused across cells.
pub(crate) struct CollectSink<'a> {
    pub(crate) out: &'a mut Vec<InputPair>,
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
    ) -> Result<(), ApplyError> {
        let lim = eng.limits();
        lim.try_push(self.out, InputPair { left: LocalNodeIdx(lc), right: LocalNodeIdx(rc) })
    }

    #[inline(always)]
    fn begin(&mut self) -> usize {
        0
    }

    #[inline(always)]
    fn pair(&mut self, eng: &Engine, lc: u32, rc: u32) -> Result<(), ApplyError> {
        let lim = eng.limits();
        lim.try_push(self.out, InputPair { left: LocalNodeIdx(lc), right: LocalNodeIdx(rc) })
    }

    #[inline(always)]
    fn end(
        &mut self,
        _eng: &Engine,
        _node_idx: &mut [u32],
        _grid_pos: usize,
        _start: usize,
    ) -> Result<usize, ApplyError> {
        Ok(0)
    }
}

/// Merged per-cell product walk — ONE kernel for every dense cell action.
///
/// The lookups (`L`, `R`) resolve child refs per representation (dense grid /
/// sparse point index / marginal pass-through — see `child_lookup.rs`); the
/// sink (`S`) is the per-pair action (emit / count / collect).
///
/// Arms: 1×1 (single-pair fast path via `sink.single`), N×1 / 1×N (one side
/// single), N×M (reach-mask culls + the ≥64×64 grouped fast path when neither
/// side is pass-through). A cell in the N×M arm implies `ctx.nxm` (both sides
/// having >1 pairs means both levels have multi-pair nodes), so the
/// liveness/reach arrays are always built when the culls read them.
///
/// The pass-through guards fold to constants for the plain lookup
/// (`ChildLookup::passthrough()` is a constant `false` on `DenseLookup` — the
/// only non-marg implementor; a sparse child reaching a dense parent is
/// densified first by `materialize_dense_child`, so there is no sparse
/// `ChildLookup` variant), so the plain instantiations keep branch-free inner
/// loops; only the marg instantiation (`MargLookup`) pays a per-access
/// pass-through branch.
///
/// The `inputs2_scratch` lifetime is independent from `node_idx`: `c2_level`
/// borrows from a separate `Tdd` operand, and `pairs_view_decoded` borrows
/// `inputs2_scratch` as the decode buffer — neither aliases the output slab.
///
/// Returns `Err(ApplyError)`: `OverBudget` on sink allocation failure, and
/// `Deadline` from the intra-cell poll in any arm — so even a
/// count-only sink is NOT infallible (it can bail mid-cell on a wide cell).
#[allow(clippy::too_many_arguments)]
#[inline(always)]
/// The general product walk: every c1 pair against every c2 pair, with the
/// dead-pair pre-filter culling rows and columns that cannot contribute.
#[allow(clippy::too_many_arguments)]
fn cell_nxm<L, R, S>(
    eng: &Engine,
    j: usize,
    inputs1: &[InputPair],
    inputs2: &[InputPair],
    left_alive_mask: u128,
    right_alive_mask: u128,
    ctx: &CellCtx<'_>,
    node_idx: &mut [u32],
    grid_pos: usize,
    left: &L,
    right: &R,
    sink: &mut S,
) -> Result<(), ApplyError>
where
    L: ChildLookup,
    R: ChildLookup,
    S: PairSink,
{
    let lim = eng.limits();
    // ── N×M (implies nxm: both levels multi-pair ⟹ masks built) ──────
    let left_dead = !left.passthrough()
        && left_alive_mask & ctx.reach_c2_left[j] == 0;
    let right_dead = !right.passthrough()
        && right_alive_mask & ctx.reach_c2_right[j] == 0;
    if left_dead || right_dead {
        return Ok(());
    }

    let cell_start = sink.begin();
    // The intra-cell poll. One very wide cell can push hundreds of millions of
    // pairs inside a single call, so a stop armed over the operation has to be
    // asked here as well as between cells — off a stride of its own, coarse
    // enough that the ask costs at most one branch per outer iteration.
    let mut cell_gate = PollGate::new(INTRA_CELL_POLL_STRIDE);
    if !left.passthrough() && !right.passthrough()
        && inputs1.len() >= 64 && inputs2.len() >= 64
    {
        // ── Grouped N×M: run-length groups by shared `.left` ──────────
        // Emits the same pair multiset as the general double loop, in
        // grouped order; every consumer is order-independent (pair lists
        // are unordered sets — see types/level.rs).
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
            lim.poll(&mut cell_gate, inputs2.len() as u64)?;
            let p1_left = inputs1[p1_idx].left;
            let g1_start = p1_idx;
            p1_idx += 1;
            while p1_idx < n1 && inputs1[p1_idx].left == p1_left { p1_idx += 1; }
            let g1 = &inputs1[g1_start..p1_idx];

            if ctx.live_left_cols[p1_left.idx()] & ctx.reach_c2_left[j] == 0 { continue; }

            for &(g2s, g2e) in &groups2 {
                let p2_left = inputs2[g2s].left;
                let lc = left.get(node_idx, p1_left.0, p2_left.0);
                if lc == DEAD { continue; }
                let g2 = &inputs2[g2s..g2e];

                for p1 in g1 {
                    if ctx.live_right_cols[p1.right.idx()] & ctx.reach_c2_right[j] == 0 {
                        continue;
                    }
                    for p2 in g2 {
                        let rc = right.get(node_idx, p1.right.0, p2.right.0);
                        if rc == DEAD { continue; }
                        sink.pair(eng, lc, rc)?;
                    }
                }
            }
        }
    } else {
        // ── General N×M ───────────────────────────────────────────────
        for p1 in inputs1 {
            lim.poll(&mut cell_gate, inputs2.len() as u64)?;
            if !left.passthrough()
                && ctx.live_left_cols[p1.left.idx()] & ctx.reach_c2_left[j] == 0 {
                continue;
            }
            if !right.passthrough()
                && ctx.live_right_cols[p1.right.idx()] & ctx.reach_c2_right[j] == 0 {
                continue;
            }
            for p2 in inputs2 {
                let lc = left.get(node_idx, p1.left.0, p2.left.0);
                if lc == DEAD { continue; }
                let rc = right.get(node_idx, p1.right.0, p2.right.0);
                if rc == DEAD { continue; }
                sink.pair(eng, lc, rc)?;
            }
        }
    }
    sink.end(eng, node_idx, grid_pos, cell_start)?;
    Ok(())
}

pub(crate) fn process_cell<L, R, S>(
    eng: &Engine,
    j: usize,
    row_base: usize,
    inputs1: &[InputPair],
    left_alive_mask: u128,
    right_alive_mask: u128,
    ctx: &CellCtx<'_>,
    c2_level: &TddLevel,
    inputs2_scratch: &mut Vec<InputPair>,
    node_idx: &mut [u32],
    left: &L,
    right: &R,
    sink: &mut S,
) -> Result<(), ApplyError>
where
    L: ChildLookup,
    R: ChildLookup,
    S: PairSink,
{
    let lim = eng.limits();
    if S::ASSERT_INTERNAL {
        debug_assert!(
            c2_level.nodes[j].is_internal() || c2_level.nodes[j].b == u32::MAX,
            "expected internal node at internal vtree position: j={j} k2={} node_a={:#x} node_b={:#x}",
            ctx.k2, c2_level.nodes[j].a, c2_level.nodes[j].b
        );
    }

    // Column `j`'s pairs. Everything about resolving them — masks, encoding,
    // range — depends only on `j` and the level, so it was hoisted into the
    // per-level [`C2Columns`] table and this is two loads. `None` is the
    // fallback for the levels the table declines (marginal-encoded c2, or an
    // arena the budget rejected): re-derive per cell, as before.
    let inputs2 = match ctx.c2_cols {
        Some(cols) => cols.get(j),
        None => c2_level.pairs_view_decoded(j, inputs2_scratch, ctx.left_mask, ctx.right_mask),
    };
    if inputs2.is_empty() { return Ok(()); }

    // `row_base` is the row's flat slab offset, already computed by the row loop
    // (`ctx.t_base + grid_row * ctx.k2`) for its DEAD reset — reuse it instead of
    // re-deriving the same product per cell.
    let grid_pos = row_base + j;
    let nxm = ctx.nxm;

    if inputs1.len() == 1 && inputs2.len() == 1 {
        // ── 1×1 ──────────────────────────────────────────────────────────
        let p1 = &inputs1[0];
        let p2 = &inputs2[0];
        let lc = left.get(node_idx, p1.left.0, p2.left.0);
        if lc != DEAD {
            let rc = right.get(node_idx, p1.right.0, p2.right.0);
            if rc != DEAD {
                sink.single(eng, node_idx, grid_pos, lc, rc)?;
            }
        }
    } else if inputs2.len() == 1 {
        // ── N×1 ──────────────────────────────────────────────────────────
        let p2 = &inputs2[0];
        if nxm && !left.passthrough() && left_alive_mask & ctx.reach_c2_left[j] == 0 {
            return Ok(());
        }
        if nxm && !right.passthrough() && right_alive_mask & ctx.reach_c2_right[j] == 0 {
            return Ok(());
        }
        // A3: an ext-encoded operand can make this single N×1 cell iterate a very
        // large pair list; poll the deadline once for the whole sweep. The pair
        // bound is known up front (`inputs1.len()`), so the check amortizes
        // exactly like the N×M arm's per-OUTER-iteration bump — one branch for
        // the cell instead of one per pair.
        let mut cell_gate = PollGate::new(INTRA_CELL_POLL_STRIDE);
        lim.poll(&mut cell_gate, inputs1.len() as u64)?;
        let cell_start = sink.begin();
        for p1 in inputs1 {
            let lc = left.get(node_idx, p1.left.0, p2.left.0);
            if lc == DEAD { continue; }
            let rc = right.get(node_idx, p1.right.0, p2.right.0);
            if rc == DEAD { continue; }
            sink.pair(eng, lc, rc)?;
        }
        sink.end(eng, node_idx, grid_pos, cell_start)?;
    } else if inputs1.len() == 1 {
        // ── 1×N ──────────────────────────────────────────────────────────
        let p1 = &inputs1[0];
        // Reach-mask cull, mirror of the N×1 arm (P3): if no live left (resp.
        // right) column can reach c2-node j's children, every lookup below is
        // DEAD → the cell emits nothing. Gated on `nxm` because the reach masks
        // are only built when both levels are multi-pair.
        if nxm && !left.passthrough() && left_alive_mask & ctx.reach_c2_left[j] == 0 {
            return Ok(());
        }
        if nxm && !right.passthrough() && right_alive_mask & ctx.reach_c2_right[j] == 0 {
            return Ok(());
        }
        // A3: mirror the N×1 arm — an ext-encoded operand2 can make this single
        // 1×N cell iterate a very large pair list; poll the deadline once for
        // the whole sweep off the up-front pair bound (`inputs2.len()`).
        let mut cell_gate = PollGate::new(INTRA_CELL_POLL_STRIDE);
        lim.poll(&mut cell_gate, inputs2.len() as u64)?;
        let cell_start = sink.begin();
        for p2 in inputs2 {
            let lc = left.get(node_idx, p1.left.0, p2.left.0);
            if lc == DEAD { continue; }
            let rc = right.get(node_idx, p1.right.0, p2.right.0);
            if rc == DEAD { continue; }
            sink.pair(eng, lc, rc)?;
        }
        sink.end(eng, node_idx, grid_pos, cell_start)?;
    } else {
        cell_nxm(
            eng,
            j, inputs1, inputs2, left_alive_mask, right_alive_mask, ctx,
            node_idx, grid_pos, left, right, sink,
        )?;
    }
    Ok(())
}
