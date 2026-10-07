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
    /// (the emit impl snapshots the length of its [`buf`](Self::buf)).
    fn begin(&mut self) -> usize;

    /// One surviving (lc, rc) pair of a multi-pair cell, pushed through the
    /// buffer's growth policy.
    fn pair(&mut self, eng: &Engine, lc: u32, rc: u32) -> Result<(), OperationError>;

    /// The buffer [`pair`](Self::pair) pushes onto, which [`push_kept`] writes
    /// into directly while it has [room](Self::room).
    fn buf(&mut self) -> &mut Vec<ChildPair>;

    /// How many pairs [`push_kept`] may write into [`buf`](Self::buf)
    /// before the next one has to go through [`pair`](Self::pair): the
    /// buffer's spare capacity, unless the sink grows its buffer on a
    /// schedule of its own.
    #[inline(always)]
    fn room(&mut self) -> usize {
        let v = self.buf();
        v.capacity() - v.len()
    }

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
        self.buf().len()
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
    fn buf(&mut self) -> &mut Vec<ChildPair> {
        self.level.pairs.stored_mut()
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

/// [`EmitSink`] for a level whose pair arena was reserved before its row
/// loop to hold every pair the loop writes, so the arena never grows.
///
/// The output-pair meter is charged from the arena's capacity as it grows.
/// To keep the meter, and every stop rule that reads it, where the
/// [`EmitSink`] walk would have it, this sink charges it on the schedule that
/// walk's arena grows on: `charged` is the capacity that arena would have
/// now, and when the arena's length reaches it the meter is charged for the
/// next doubling ([`doubled_pairs_capacity`]) before the pair is pushed.
///
/// With `charged` at `usize::MAX` there is no schedule of its own, and the
/// sink is exactly an [`EmitSink`]: the room is the arena's spare capacity
/// and a full arena grows through [`try_push_pair_into`].
pub(crate) struct ReservedEmitSink<'a> {
    pub(crate) level: &'a mut TddLevel,
    /// The pair capacity the meters have been charged for, or `usize::MAX`.
    pub(crate) charged: usize,
}

impl PairSink for ReservedEmitSink<'_> {
    #[inline(always)]
    fn single(
        &mut self, eng: &Engine, node_idx: &mut [u32],
        grid_pos: usize,
        lc: u32,
        rc: u32,
    ) -> Result<(), OperationError> {
        EmitSink { level: &mut *self.level }.single(eng, node_idx, grid_pos, lc, rc)
    }

    #[inline(always)]
    fn begin(&mut self) -> usize {
        self.buf().len()
    }

    #[inline(always)]
    fn pair(&mut self, eng: &Engine, lc: u32, rc: u32) -> Result<(), OperationError> {
        if self.buf().len() == self.charged {
            let grown = doubled_pairs_capacity(self.charged);
            eng.limits().charge_output_pairs(grown - self.charged);
            self.charged = grown;
            super::super::note_scheduled_charge();
            debug_assert!(
                self.buf().len() < self.buf().capacity(),
                "a reserved arena holds every pair its level writes"
            );
        }
        try_push_pair_into(
            eng,
            self.level,
            ChildPair::new(EncodedChildRef::from_raw(lc), EncodedChildRef::from_raw(rc)),
        )
    }

    #[inline(always)]
    fn buf(&mut self) -> &mut Vec<ChildPair> {
        self.level.pairs.stored_mut()
    }

    #[inline(always)]
    fn room(&mut self) -> usize {
        let charged = self.charged;
        let v = self.buf();
        charged.min(v.capacity()) - v.len()
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
    fn buf(&mut self) -> &mut Vec<ChildPair> {
        self.out
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

/// Push the surviving candidates of `items` in order: `cand` maps an item to
/// its `(lc, rc)`, and a candidate survives when neither side is
/// [`NO_PRODUCT`]. `cand` reads the right side only for a live left side, as
/// a walk that tests each side before the next would, and returns
/// `NO_PRODUCT` for it otherwise; so the right side alone says whether a
/// candidate survives. With `kills` false no candidate can die (neither
/// side's lookup [kills](ChildLookup::kills)), and every one is kept without
/// a test.
///
/// While the sink has [room](PairSink::room), survivors are written without
/// a capacity test: each is stored at the buffer's end and the length, held
/// in a register for the run, steps past it. A dead candidate is skipped by a
/// branch, not stored: in a counting diagram's conjunction most candidates
/// die, as few as one in a hundred survives, and a branch that mostly goes
/// one way costs less than a store of every candidate. A run holds no more
/// candidates than the room, so no write passes the buffer's capacity. Once
/// the room is used up, the next survivor goes through [`PairSink::pair`],
/// which grows the buffer; that is the survivor at which pushing each
/// survivor in turn would grow it, so the buffer's capacity, and every meter
/// charged from it, changes at the same pairs as before.
#[inline(always)]
pub(super) fn push_kept<T, S: PairSink>(
    eng: &Engine,
    sink: &mut S,
    items: &[T],
    kills: bool,
    mut cand: impl FnMut(&T) -> (u32, u32),
) -> Result<(), OperationError> {
    let mut rest = items;
    while !rest.is_empty() {
        let room = sink.room();
        if room == 0 {
            let (lc, rc) = cand(&rest[0]);
            rest = &rest[1..];
            debug_assert!(lc != NO_PRODUCT || rc == NO_PRODUCT, "a dead left side stands for the right");
            debug_assert!(kills || rc != NO_PRODUCT, "a side that kills nothing answered no product");
            if !kills || rc != NO_PRODUCT {
                sink.pair(eng, lc, rc)?;
            }
            continue;
        }
        let (run, tail) = rest.split_at(room.min(rest.len()));
        rest = tail;
        let v = sink.buf();
        debug_assert!(run.len() <= v.capacity() - v.len(), "a sink's room is within its buffer's capacity");
        let out = v.as_mut_ptr();
        let mut len = v.len();
        for item in run {
            let (lc, rc) = cand(item);
            debug_assert!(lc != NO_PRODUCT || rc == NO_PRODUCT, "a dead left side stands for the right");
            debug_assert!(kills || rc != NO_PRODUCT, "a side that kills nothing answered no product");
            if !kills || rc != NO_PRODUCT {
                // Safety: `len` starts at the buffer's length and steps at
                // most once per item of `run`, which holds no more items than
                // the sink's room, itself within the spare capacity, so the
                // write stays inside the allocation.
                unsafe { out.add(len).write(ChildPair::new(EncodedChildRef::from_raw(lc), EncodedChildRef::from_raw(rc))) };
                len += 1;
            }
        }
        // Safety: slots below `len` hold the old contents and one written
        // pair per survivor.
        unsafe { v.set_len(len) };
    }
    Ok(())
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
    let kills = left.kills() || right.kills();
    if ITER_F {
        // N×1: the row changes per pair, so each lookup resolves its own.
        let (c, d) = (g_pairs[0].left.0, g_pairs[0].right.0);
        let node_idx = &*node_idx;
        push_kept(eng, sink, f_pairs, kills, |p1| {
            let lc = left.get(node_idx, p1.left.0, c);
            if left.kills() && lc == NO_PRODUCT { return (lc, lc); }
            (lc, right.get(node_idx, p1.right.0, d))
        })?;
    } else {
        // 1×N: one f pair fixes both child rows for the whole sweep.
        let p1 = &f_pairs[0];
        let lrow = left.row(p1.left.0);
        let rrow = right.row(p1.right.0);
        let node_idx = &*node_idx;
        push_kept(eng, sink, g_pairs, kills, |p2| {
            let lc = left.get_in_row(node_idx, lrow, p2.left.0);
            if left.kills() && lc == NO_PRODUCT { return (lc, lc); }
            (lc, right.get_in_row(node_idx, rrow, p2.right.0))
        })?;
    }
    sink.end(eng, node_idx, grid_pos, cell_start)?;
    Ok(())
}

/// The general product walk: every f pair against every g pair, with the
/// dead-pair pre-filter culling rows and columns that cannot contribute.
///
/// On a column the column table grouped (see [`RightColumns::runs`]) and an
/// f row of at least [`GROUPED_MIN_PAIRS`] pairs, the walk takes both sides a
/// run of shared `.left` at a time.
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
    // ── N×M (implies both_multi_pair: both levels multi-pair ⟹ masks built
    // for every side that can kill) ──────
    // The whole-cell dead test belongs to the row loop, which applies it to
    // every arm before the cell is entered; what is left here is the per-`p1`
    // form of it, which only this arm can use.
    let cell_start = sink.begin();
    if f_pairs.len() >= GROUPED_MIN_PAIRS
        && let Some(g_runs) = ctx.right_cols.and_then(|cols| cols.runs(j))
    {
        // ── Grouped N×M: run-length groups by shared `.left` ──────────
        // Emits the same pair multiset as the general double loop, in
        // grouped order; every consumer is order-independent (pair lists
        // are unordered sets — see diagram/level/mod.rs). The column table
        // groups only levels with no pass-through side.
        debug_assert!(!left.passthrough() && !right.passthrough());
        let n2 = g_pairs.len();
        let n1 = f_pairs.len();
        let mut p1_idx = 0;
        while p1_idx < n1 {
            let p1_left = f_pairs[p1_idx].left;
            let g1_start = p1_idx;
            p1_idx += 1;
            while p1_idx < n1 && f_pairs[p1_idx].left == p1_left { p1_idx += 1; }
            let g1 = &f_pairs[g1_start..p1_idx];
            gate.poll((g1.len() * n2) as u64)?;

            if left.kills() && ctx.sides.left.live_cols[p1_left.raw() as usize] & ctx.sides.left.reach[j] == 0 {
                continue;
            }

            let mut g2_start = 0;
            for &g2_end in g_runs {
                let g2 = &g_pairs[g2_start..g2_end as usize];
                g2_start = g2_end as usize;
                let lc = left.get(node_idx, p1_left.0, g2[0].left.0);
                if left.kills() && lc == NO_PRODUCT { continue; }

                for p1 in g1 {
                    if right.kills()
                        && ctx.sides.right.live_cols[p1.right.raw() as usize] & ctx.sides.right.reach[j] == 0
                    {
                        continue;
                    }
                    let rrow = right.row(p1.right.0);
                    let node_idx = &*node_idx;
                    // `lc` is live here, so only the right side can kill.
                    push_kept(eng, sink, g2, right.kills(), |p2| (lc, right.get_in_row(node_idx, rrow, p2.right.0)))?;
                }
            }
        }
    } else {
        // ── General N×M ───────────────────────────────────────────────
        let kills = left.kills() || right.kills();
        for p1 in f_pairs {
            gate.poll(g_pairs.len() as u64)?;
            if !left.passthrough() && left.kills()
                && ctx.sides.left.live_cols[p1.left.raw() as usize] & ctx.sides.left.reach[j] == 0 {
                continue;
            }
            if !right.passthrough() && right.kills()
                && ctx.sides.right.live_cols[p1.right.raw() as usize] & ctx.sides.right.reach[j] == 0 {
                continue;
            }
            let lrow = left.row(p1.left.0);
            let rrow = right.row(p1.right.0);
            let node_idx = &*node_idx;
            push_kept(eng, sink, g_pairs, kills, |p2| {
                let lc = left.get_in_row(node_idx, lrow, p2.left.0);
                if left.kills() && lc == NO_PRODUCT { return (lc, lc); }
                (lc, right.get_in_row(node_idx, rrow, p2.right.0))
            })?;
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
/// culls, grouped by shared `.left` on a long f row against a column the
/// column table grouped). A cell in the N×M arm implies `ctx.both_multi_pair` (both sides
/// having >1 pairs means both levels have multi-pair nodes), so the
/// liveness/reach arrays are always built when the culls read them.
///
/// `ChildLookup::passthrough()` is a constant `false` on `DenseLookup`, so the
/// plain instantiations carry no pass-through branch in the inner loops; and
/// `ChildLookup::kills()` is a constant `false` on `CompleteLookup`, so a
/// complete side carries no `NO_PRODUCT` test and no mask cull.
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
///
/// Inlined into the row loop: most cells are 1×1 or a handful of pairs, and
/// an out-of-line call would cost such a cell more than its pairs do — the
/// arguments spilled and reloaded, the column table and the work-clock gate
/// read back through memory at every cell.
#[inline(always)]
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
    // Column `j`'s pairs. Everything about resolving them — masks, encoding,
    // range — depends only on `j` and the level, so it was hoisted into the
    // per-level [`RightColumns`] table and this is two loads. `None` is the
    // fallback for the levels the table declines (marginal-encoded g, or an
    // arena the budget rejected): re-derive per cell, as before, and walk an
    // N×M cell ungrouped.
    let g_pairs = match ctx.right_cols {
        Some(cols) => cols.get(j),
        None => right_level.pairs_view_decoded(j, g_pairs_scratch, ctx.sides.left.plan.view, ctx.sides.right.plan.view),
    };
    // A dead column's cell holds no product. Its row was reset to that
    // already, except where every live cell is written and no row is reset
    // ([`run_level_rows`](super::rows::run_level_rows)).
    if g_pairs.is_empty() {
        node_idx[row_base + j] = NO_PRODUCT;
        return Ok(());
    }

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
        if !left.kills() || lc != NO_PRODUCT {
            let rc = right.get(node_idx, p1.right.0, p2.right.0);
            if !right.kills() || rc != NO_PRODUCT {
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
