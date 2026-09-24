//! One cell of the product: the pair walk and the sinks it writes through.
//!
//! The `#[inline(always)]` in this directory and in `child_lookup` are not
//! decoration: dropping them costs about a tenth of a percent of the
//! instructions a mid-size exact count retires, and the loss shows up as work
//! moving out of `process_cell` into the `CellAction` impls. Anything added
//! here that the walk calls per pair belongs in the same regime.

use crate::diagram::EncodedChildRef;

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
pub(crate) fn row_alive_masks(ctx: &CellCtx<'_>, f_pairs: &[ChildPair]) -> Option<(u128, u128)> {
    let left_alive_mask: u128 = if ctx.sides.left.plan.is_passthrough() {
        u128::MAX // pass-through side: no grid; always alive
    } else if !ctx.both_multi_pair {
        0u128
    } else {
        f_pairs.iter().fold(0u128, |acc, p1| acc | ctx.sides.left.live_cols[p1.left.raw() as usize])
    };
    if ctx.both_multi_pair && left_alive_mask == 0 { return None; }

    let right_alive_mask: u128 = if ctx.sides.right.plan.is_passthrough() || !ctx.both_multi_pair {
        u128::MAX
    } else {
        f_pairs.iter().fold(0u128, |acc, p1| acc | ctx.sides.right.live_cols[p1.right.raw() as usize])
    };
    if ctx.both_multi_pair && right_alive_mask == 0 { return None; }

    Some((left_alive_mask, right_alive_mask))
}

/// Emit a product node from the pairs accumulated in
/// `level.pairs[pair_start..]` and record it at `grid_pos`; a cell that
/// pushed none leaves `NO_PRODUCT` there.
#[inline(always)]
pub(crate) fn emit_product_node(
    eng: &Engine,
    level: &mut TddLevel,
    node_idx: &mut [u32],
    grid_pos: usize,
    pair_start: usize,
) -> Result<(), OperationError> {
    if let Some(node) = finish_node(eng, level, pair_start)? {
        node_idx[grid_pos] = node.0;
    }
    Ok(())
}

/// Emit a node holding exactly `pair`, stored inline. The push is
/// budget-charged.
#[inline(always)]
pub(in crate::apply::conjoin) fn emit_single_pair(eng: &Engine, level: &mut TddLevel, pair: ChildPair) -> Result<(), OperationError> {
    eng.limits().try_push(&mut level.nodes, EncodedNode::inline(pair))
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
    /// True for the emit sink; false for the collect sink, which may visit
    /// marginal-encoded operand nodes (the both-marginal collapse).
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
    /// (the emit impl snapshots `level.arena_len()`).
    fn begin(&mut self) -> usize;

    /// One surviving (lc, rc) pair of a multi-pair cell.
    fn pair(&mut self, eng: &Engine, lc: u32, rc: u32) -> Result<(), OperationError>;

    /// Finish a multi-pair cell.
    fn end(
        &mut self,
        eng: &Engine,
        node_idx: &mut [u32],
        grid_pos: usize,
        start: usize,
    ) -> Result<(), OperationError>;
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
        let pair = ChildPair::new(EncodedChildRef::from_raw(lc), EncodedChildRef::from_raw(rc));
        let nid = self.level.nodes.len() as u32;
        node_idx[grid_pos] = nid;
        emit_single_pair(eng, self.level, pair)
    }

    #[inline(always)]
    fn begin(&mut self) -> usize {
        self.level.arena_len()
    }

    #[inline(always)]
    fn pair(&mut self, eng: &Engine, lc: u32, rc: u32) -> Result<(), OperationError> {
        try_push_pair_into(
            eng,
            self.level,
            ChildPair::new(EncodedChildRef::from_raw(lc), EncodedChildRef::from_raw(rc)),
        )
    }

    #[inline(always)]
    fn end(
        &mut self, eng: &Engine, node_idx: &mut [u32],
        grid_pos: usize,
        start: usize,
    ) -> Result<(), OperationError> {
        emit_product_node(eng, self.level, node_idx, grid_pos, start)
    }
}

/// Collect surviving `(lc, rc)` pairs into a caller-owned scratch Vec — the
/// streaming collapse-at-source walk (`run_level_rows_stream_count`). Touches
/// neither `level` nor `node_idx[grid_pos]`; the caller folds the collected
/// refs through its fold's `fold_cell`. Pushes are budget-tracked
/// (`try_push`), matching the emit walk's `try_push_pair_into`: a wide
/// streaming cell's scratch growth charges the apply soft budget and degrades
/// to `Err(OverBudget)` instead of an allocator abort. The scratch is bounded
/// to one cell's pairs and reused across cells.
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
        lim.try_push(self.out, ChildPair::new(EncodedChildRef::from_raw(lc), EncodedChildRef::from_raw(rc)))
    }

    #[inline(always)]
    fn begin(&mut self) -> usize {
        0
    }

    #[inline(always)]
    fn pair(&mut self, eng: &Engine, lc: u32, rc: u32) -> Result<(), OperationError> {
        let lim = eng.limits();
        lim.try_push(self.out, ChildPair::new(EncodedChildRef::from_raw(lc), EncodedChildRef::from_raw(rc)))
    }

    #[inline(always)]
    fn end(
        &mut self,
        _eng: &Engine,
        _node_idx: &mut [u32],
        _grid_pos: usize,
        _start: usize,
    ) -> Result<(), OperationError> {
        Ok(())
    }
}

/// The one-sided product walk: one operand contributes a single pair, the other
/// is swept. `ITER_F` says which — `true` sweeps `f_pairs` against the lone g
/// pair (N×1), `false` sweeps `g_pairs` against the lone f pair (1×N). The
/// grid lookup is ordered `(f field, g field)` in both directions.
///
/// The pair count is known before the sweep, so the whole cell is charged to
/// the work clock in one go: one branch per cell rather than one per pair.
#[inline(always)]
#[expect(clippy::too_many_arguments)]
fn cell_one_sided<const ITER_F: bool, L, R, S>(
    eng: &Engine,
    f_pairs: &[ChildPair],
    g_pairs: &[ChildPair],
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
    let n = if ITER_F { f_pairs.len() } else { g_pairs.len() };
    gate.poll(n as u64)?;
    let cell_start = sink.begin();
    if ITER_F {
        // N×1: the row changes per pair, so each lookup resolves its own.
        let p2 = &g_pairs[0];
        for p1 in f_pairs {
            let lc = left.get(node_idx, p1.left.0, p2.left.0);
            if lc == NO_PRODUCT { continue; }
            let rc = right.get(node_idx, p1.right.0, p2.right.0);
            if rc == NO_PRODUCT { continue; }
            sink.pair(eng, lc, rc)?;
        }
    } else {
        // 1×N: one f pair fixes both child rows for the whole sweep.
        let p1 = &f_pairs[0];
        let lrow = left.row(p1.left.0);
        let rrow = right.row(p1.right.0);
        for p2 in g_pairs {
            let lc = left.get_in_row(node_idx, lrow, p2.left.0);
            if lc == NO_PRODUCT { continue; }
            let rc = right.get_in_row(node_idx, rrow, p2.right.0);
            if rc == NO_PRODUCT { continue; }
            sink.pair(eng, lc, rc)?;
        }
    }
    sink.end(eng, node_idx, grid_pos, cell_start)?;
    Ok(())
}

/// The general product walk: every f pair against every g pair, with the
/// dead-pair pre-filter culling rows and columns that cannot contribute.
#[inline(always)]
#[expect(clippy::too_many_arguments)]
fn cell_prefilter<L, R, S>(
    eng: &Engine,
    j: usize,
    f_pairs: &[ChildPair],
    g_pairs: &[ChildPair],
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
    // ── N×M (implies both_multi_pair: both levels multi-pair ⟹ masks built) ──────
    // The whole-cell dead test belongs to the row loop, which applies it to
    // every arm before the cell is entered; what is left here is the per-`p1`
    // form of it, which only this arm can use.
    let cell_start = sink.begin();
    if !left.passthrough() && !right.passthrough()
        && f_pairs.len() >= 64 && g_pairs.len() >= 64
    {
        // ── Grouped N×M: run-length groups by shared `.left` ──────────
        // Emits the same pair multiset as the general double loop, in
        // grouped order; every consumer is order-independent (pair lists
        // are unordered sets — see diagram/level/mod.rs).
        let n2 = g_pairs.len();
        let mut groups2: smallvec::SmallVec<[(usize, usize); 256]> =
            smallvec::SmallVec::new();
        {
            let mut idx = 0;
            while idx < n2 {
                let start = idx;
                let left = g_pairs[idx].left;
                idx += 1;
                while idx < n2 && g_pairs[idx].left == left { idx += 1; }
                groups2.push((start, idx));
            }
        }
        let n1 = f_pairs.len();
        let mut p1_idx = 0;
        while p1_idx < n1 {
            let p1_left = f_pairs[p1_idx].left;
            let g1_start = p1_idx;
            p1_idx += 1;
            while p1_idx < n1 && f_pairs[p1_idx].left == p1_left { p1_idx += 1; }
            let g1 = &f_pairs[g1_start..p1_idx];
            gate.poll((g1.len() * n2) as u64)?;

            if ctx.sides.left.live_cols[p1_left.raw() as usize] & ctx.sides.left.reach[j] == 0 { continue; }

            for &(g2s, g2e) in &groups2 {
                let p2_left = g_pairs[g2s].left;
                let lc = left.get(node_idx, p1_left.0, p2_left.0);
                if lc == NO_PRODUCT { continue; }
                let g2 = &g_pairs[g2s..g2e];

                for p1 in g1 {
                    if ctx.sides.right.live_cols[p1.right.raw() as usize] & ctx.sides.right.reach[j] == 0 {
                        continue;
                    }
                    let rrow = right.row(p1.right.0);
                    for p2 in g2 {
                        let rc = right.get_in_row(node_idx, rrow, p2.right.0);
                        if rc == NO_PRODUCT { continue; }
                        sink.pair(eng, lc, rc)?;
                    }
                }
            }
        }
    } else {
        // ── General N×M ───────────────────────────────────────────────
        for p1 in f_pairs {
            gate.poll(g_pairs.len() as u64)?;
            if !left.passthrough()
                && ctx.sides.left.live_cols[p1.left.raw() as usize] & ctx.sides.left.reach[j] == 0 {
                continue;
            }
            if !right.passthrough()
                && ctx.sides.right.live_cols[p1.right.raw() as usize] & ctx.sides.right.reach[j] == 0 {
                continue;
            }
            let lrow = left.row(p1.left.0);
            let rrow = right.row(p1.right.0);
            for p2 in g_pairs {
                let lc = left.get_in_row(node_idx, lrow, p2.left.0);
                if lc == NO_PRODUCT { continue; }
                let rc = right.get_in_row(node_idx, rrow, p2.right.0);
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
/// The lookups (`L`, `R`) resolve child refs per representation — a
/// `DenseLookup` grid read or a `MarginalLookup` pass-through, see
/// `child_lookup.rs`; the sink (`S`) is the per-pair action, `EmitSink` or
/// `CollectSink`.
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
/// The `g_pairs_scratch` lifetime is independent from `node_idx`: `right_level`
/// borrows from a separate `Tdd` operand, and `pairs_view_decoded` borrows
/// `g_pairs_scratch` as the decode buffer — neither aliases the output slab.
///
/// Sink allocation can return [`OperationError::OverBudget`]; every arm polls
/// for [`OperationError::Stopped`], the collect sink included.
///
/// Keep input slices and lookup geometry as direct parameters so the optimizer
/// retains their aliasing information within the pair loops.
#[expect(clippy::too_many_arguments)]
pub(crate) fn process_cell<L, R, S>(
    eng: &Engine,
    j: usize,
    row_base: usize,
    f_pairs: &[ChildPair],
    ctx: &CellCtx<'_>,
    right_level: &TddLevel,
    g_pairs_scratch: &mut Vec<ChildPair>,
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
    let g_pairs = match ctx.right_cols {
        Some(cols) => cols.get(j),
        None => right_level.pairs_view_decoded(j, g_pairs_scratch, ctx.sides.left.plan.view, ctx.sides.right.plan.view),
    };
    if g_pairs.is_empty() { return Ok(()); }

    // `row_base` is the row's flat slab offset, already computed by the row loop
    // (`ctx.output_grid_base + grid_row * ctx.right_width`) for its `NO_PRODUCT` reset — reuse it instead of
    // re-deriving the same product per cell.
    let grid_pos = row_base + j;

    if f_pairs.len() == 1 && g_pairs.len() == 1 {
        // ── 1×1 ──────────────────────────────────────────────────────────
        gate.poll(1)?;
        let p1 = &f_pairs[0];
        let p2 = &g_pairs[0];
        let lc = left.get(node_idx, p1.left.0, p2.left.0);
        if lc != NO_PRODUCT {
            let rc = right.get(node_idx, p1.right.0, p2.right.0);
            if rc != NO_PRODUCT {
                sink.single(eng, node_idx, grid_pos, lc, rc)?;
            }
        }
    } else if g_pairs.len() == 1 {
        cell_one_sided::<true, _, _, _>(
            eng, f_pairs, g_pairs, node_idx, grid_pos, left, right, sink, gate,
        )?;
    } else if f_pairs.len() == 1 {
        cell_one_sided::<false, _, _, _>(
            eng, f_pairs, g_pairs, node_idx, grid_pos, left, right, sink, gate,
        )?;
    } else {
        cell_prefilter(
            eng,
            j, f_pairs, g_pairs, ctx,
            node_idx, grid_pos, left, right, sink, gate,
        )?;
    }
    Ok(())
}
