//! Dense cell/row engine for the apply product construction.
//!
//! Contains the merged per-cell product-walk kernel (`process_cell`, generic
//! over a [`ChildLookup`] per side and a [`PairSink`] action), the ONE row-loop
//! driver behind every build route (`run_level_rows`, generic over a
//! [`CellAction`]) with its four route entry points (`run_level_rows_marg`,
//! `run_level_rows_marg_sparse`, `run_level_rows_stream_count`,
//! `run_level_rows_plain`), the streaming per-cell folds ([`StreamCellFold`]),
//! and the product-node emitter (`emit_product_node`).
//!
//! Also houses `CellCtx` (the per-level loop-invariant context struct) and
//! the row-mask fold (`row_alive_masks`). The amortized between-cell
//! cancel/deadline poll is `budget::PollTicker` (shared with the sparse
//! join); see `nxm_deadline_check!` below for the deliberate intra-cell
//! exception that is NOT part of that shared ticker.

use crate::tdd::types::{InputPair, TddLevel, TddNodeData, ExtMulti, LocalNodeIdx,
    MAX_LEVEL_ARENA_BYTES};
use crate::tdd::counts::{ApplyBudget, CountVec, IntFold, WeightFold};
use crate::tdd::query::WeightVal;
use crate::tdd::utils::{pool_put_bounded, pool_take};
use super::{
    ApplyError, DEAD,
    try_push, try_push_pair_into,
};
use crate::tdd::limits::{budget_reserve_exact, unaccount_transient_bytes};
use super::stream::{attach_children, StreamLevelState, StreamPayload, StreamState};
use super::child_lookup::{ChildLookup, MargLookup};
use super::sparse::{ProductEntry, C1NodeIdx, C2NodeIdx, ProdNodeIdx};

/// Amortized wall-clock deadline cut for the intra-cell N×M product loops.
///
/// The per-cell `budget::PollTicker` check in the row loops fires only *between*
/// `process_cell` calls. A single very wide cell can push or count hundreds of
/// millions of (p1,p2) pairs *inside one call*, so a branch deadline
/// (projected-cutset conditioning) can't slice it — the apply grinds
/// minutes/GiB on one cell (track-3 029). This bumps a per-outer-iteration work
/// accumulator and, every ~1M pairs of work, consults the gated `ApplyLimits::deadline`.
///
/// "Outer iteration" is per arm: the N×M arms bump once per c1 pair (by the c2
/// pair count), while the N×1 / 1×N arms have a single sweep whose pair count is
/// known up front, so they bump ONCE before it. Either way the poll costs at
/// most one branch per outer iteration and never one per pair.
///
/// `$armed` is [`CellCtx::deadline_armed`]: whether any stop axis is installed,
/// hoisted once per level, so with none installed this is a single
/// predicted-not-taken local-bool branch per OUTER iteration. On expire it
/// returns `Err(Deadline)`; the partial level is discarded with the aborted
/// apply, never counted.
///
/// Deliberate exception to `PollTicker` (the shared between-cell / sparse
/// ticker): this poll is intra-cell, so it needs a counter the ticker's
/// per-cell granularity cannot give it.
macro_rules! nxm_deadline_check {
    ($armed:expr, $work:expr, $inc:expr) => {
        if $armed {
            $work += ($inc) as u64;
            if $work >= (1u64 << 20) {
                // A TEE of the work this loop already counted, not a second
                // counter: this is the amortization point that already exists
                // for reading it. Charge what the meter HELD, not the cadence it
                // crossed — a single `$inc` can be many cadences wide, and
                // pricing it as one would undercount exactly the large cells the
                // give-up rule exists to catch.
                crate::tdd::limits::charge_compile_work($work);
                $work = 0;
                if crate::tdd::limits::deadline_expired() {
                    return Err(ApplyError::Deadline);
                }
            }
        }
    };
}

#[cfg(test)]
thread_local! {
    /// Test-only override for `bothmarg_collapse_enabled`, scoped by
    /// `with_bothmarg_collapse_forced`.
    static BOTHMARG_COLLAPSE_OVERRIDE: std::cell::Cell<Option<bool>> =
        const { std::cell::Cell::new(None) };
}

/// Run `body` with the streaming gate forced to `enabled` on this thread, so
/// the gate-off parity test can exercise the no-streaming fallback.
#[cfg(test)]
pub(crate) fn with_bothmarg_collapse_forced<T>(enabled: bool, body: impl FnOnce() -> T) -> T {
    crate::tdd::scoped::Scoped::run(&BOTHMARG_COLLAPSE_OVERRIDE, Some(enabled), body)
}

/// Streaming gate for a level whose operands are both marginal: always on
/// (the alternative — materialize, then marginalize after the apply — is
/// count-identical at a higher peak, and exists only as the test comparison).
pub(super) fn bothmarg_collapse_enabled() -> bool {
    #[cfg(test)]
    if let Some(forced) = BOTHMARG_COLLAPSE_OVERRIDE.with(|c| c.get()) {
        return forced;
    }
    true
}

/// Per-level loop-invariant context passed to all cell/row processing functions.
pub(super) struct CellCtx<'a> {
    /// Flat base offset of the output level's grid in `node_idx`.
    pub t_base: usize,
    /// Number of c2 nodes at this level (column count of the product grid).
    pub k2: usize,
    /// Flat base offset of the left child's grid in `node_idx`.
    pub left_base: usize,
    /// Flat base offset of the right child's grid in `node_idx`.
    pub right_base: usize,
    /// c2 column count for the left child grid.
    pub k2_left: u32,
    /// c2 column count for the right child grid.
    pub k2_right: u32,
    /// True when the left child carries a pass-through marginal field
    /// (one operand is identity at left, the other is the marginal carrier).
    pub left_passthrough: bool,
    /// True when the right child carries a pass-through marginal field.
    pub right_passthrough: bool,
    /// Carrier selector for left pass-through: true ⇒ carry `p1.left`, i.e. c1
    /// is the carrier. "Carrier" is defined in `marg_plan`.
    pub left_pt_c1: bool,
    /// Carrier selector for right pass-through: true ⇒ carry `p1.right`.
    pub right_pt_c1: bool,
    /// True when both operands have multi-pair nodes (NxM dead-pair pre-filter active).
    pub nxm: bool,
    /// `limits::any_stop_armed()`, read once per level: with nothing installed
    /// the intra-cell poll is a local-bool test. Hoisting is value-identical
    /// because no poll inside a level can install the first stop axis.
    pub deadline_armed: bool,
    /// Decode mask for the left child's pair fields: `MARG_VALUE_MASK` when that
    /// child is marginal, so a bit-30 inline tag is stripped and the remaining
    /// payload is read as a coordinate; `u32::MAX` otherwise. See
    /// `MARG_OVERFLOW_TAG` for the encoding.
    pub left_mask: u32,
    /// Decode mask for the right child's pair fields, as `left_mask`.
    pub right_mask: u32,
    /// Per-c1-row live-column bitmasks (indexed by c1 left-child node idx).
    pub live_left_cols: &'a [u128],
    /// Per-j reach bitmasks for c2's left child references.
    pub reach_c2_left: &'a [u128],
    /// Per-c1-row live-column bitmasks for the right child.
    pub live_right_cols: &'a [u128],
    /// Per-j reach bitmasks for c2's right child references.
    pub reach_c2_right: &'a [u128],
    /// Per-level c2 column table — every column's pair slice resolved ONCE
    /// (see [`C2Columns`]). `Some` on every level the table could be built
    /// for; `None` ⇒ `process_cell` re-derives column `j`'s slice per cell,
    /// as before (marginal-encoded c2 level, or the marg arena declined by
    /// the budget).
    pub c2_cols: Option<&'a C2Columns>,
}

/// One c2 column's resolved pair slice, held as raw parts.
///
/// Raw rather than `&[InputPair]` so the table can live in a `Cell` scratch
/// pool: `pool_take` requires a `'static` buffer type, which a lifetime-
/// carrying slice is not. Every construction site below writes the parts of a
/// live `&[InputPair]`; [`C2Columns::get`] is the only reader.
#[derive(Clone, Copy)]
pub(super) struct ColSlice {
    ptr: *const InputPair,
    len: usize,
}

/// Per-level c2 column table: column `j`'s pair slice resolved ONCE per LEVEL
/// instead of once per (row, column) CELL.
///
/// Every `process_cell(i, j)` call used to re-derive the same column-`j` slice
/// from scratch — the mask-identity test, the `nodes[j]` bounds check, the
/// leaf/inline/multi encoding tests, the `ext`-sentinel range resolve, and (on
/// marg-mask levels) a full re-decode of the column's pairs into scratch. All
/// of that depends only on `j` and the level, never on the row, so with `k1`
/// rows it ran `k1` times per column. This resolves each column once, before
/// the row sweep; the cell prologue then indexes the table.
///
/// TWO storage regimes behind ONE table — the per-column resolution logic
/// lives here and nowhere else:
/// - **identity masks** (no marginal child): the descriptors are zero-copy
///   borrows of c2's own `nodes`/`pairs` storage, exactly what the per-cell
///   `pairs_view_decoded` fast path handed back. Nothing is copied and `flat`
///   stays empty.
/// - **marg masks**: c2's pairs need decoding, so they are decoded once into
///   `flat` and the descriptors point into it.
///   `flat` is O(Σ c2 pairs) — a real transient the budget must see, so it
///   reserves through `budget_reserve_exact` and un-charges the in-flight
///   accounting on drop (level end). On `OverBudget` the build returns `None`
///   and the walkers fall back to per-cell decode: strictly no worse than the
///   per-cell behavior on the OOM-critical path.
///
/// `cols` is pooled scratch (`SCRATCH_C2_COLS`), not diagram memory, so it is
/// not budget-charged; its retained capacity is capped on return to the pool
/// like every other apply scratch buffer.
pub(super) struct C2Columns {
    /// Decode arena — non-empty ONLY on marg-mask levels. Filled once at
    /// build time and never touched again, so the heap block the descriptors
    /// point into is fixed for the table's whole life (moving the `Vec`, e.g.
    /// out of `build`, moves the 3-word header, never the block).
    ///
    /// Deliberately never read through this field — `build` resolves the
    /// descriptors against the arena's base before handing it over, so the
    /// field's whole job is to OWN the block and free it when the table
    /// drops. Removing it would dangle every marg-level descriptor.
    #[allow(dead_code)]
    flat: Vec<InputPair>,
    /// One descriptor per column `j ∈ 0..k2`.
    cols: Vec<ColSlice>,
    /// Bytes of `flat` charged to `ApplyLimits::budget_in_flight`, released
    /// on drop.
    accounted_bytes: u64,
}

impl C2Columns {
    /// Column `j`'s pairs — the per-level replacement for the per-cell
    /// `pairs_view_decoded` derivation.
    #[inline(always)]
    pub(super) fn get(&self, j: usize) -> &[InputPair] {
        let c = self.cols[j];
        // SAFETY: `c` was built by `build` below out of either (a) a live
        // `&[InputPair]` borrowed from the c2 level, or (b) a subrange of
        // `self.flat`.
        //
        // (b) is owned by `self` and never mutated after `build`, so it is
        // alive and its heap block unmoved for as long as the returned borrow.
        //
        // (a) is alive by the sole caller's shape: the table is a local of ONE
        // iteration of the apply's per-level loop, and for the rest of that
        // iteration `c2` is only ever READ (`c2.level(t)`, `c2.levels[..]`) —
        // there is no `&mut c2` between the table's construction and its drop,
        // so c2's `nodes`/`pairs` cannot be pushed to and cannot reallocate.
        // The row sweep's own writes go to the OUTPUT level, a separate
        // allocation from either operand, and it holds `c2_level_t:
        // &TddLevel` across its full duration.
        //
        // An empty column carries the aligned-non-null pointer of the `&[]`
        // it came from, which `from_raw_parts` accepts at length 0.
        unsafe { std::slice::from_raw_parts(c.ptr, c.len) }
    }

    /// Resolve every column of `c2_level` under `left_mask`/`right_mask`.
    ///
    /// Returns `None` (per-cell derivation fallback) when the table can't be
    /// built: a marginal-encoded c2 level (it stores count payloads, not pair
    /// structure — the marginal-at-t cases are consumed by fast paths before
    /// the cell build, so the walkers never read its pairs, and neither may
    /// we), an arena past `u32::MAX` pairs (one that large has no business
    /// existing), or a budget that rejects the arena / descriptor reservation.
    pub(super) fn build(
        c2_level: &TddLevel,
        k2: usize,
        left_mask: u32,
        right_mask: u32,
    ) -> Option<C2Columns> {
        if c2_level.is_marginal() {
            return None;
        }
        // `k2` is the level width cached before the sweep; resolving a column
        // reads `nodes[j]`, and the table resolves ALL of 0..k2 where the
        // per-cell path only reached the columns of a level with ≥1 live row.
        // If the two ever disagreed, hoisting would index past `nodes` on a
        // level the per-cell path never touched — decline instead, which is
        // exactly the pre-existing per-cell behavior.
        if k2 > c2_level.nodes.len() {
            return None;
        }
        let identity = left_mask == u32::MAX && right_mask == u32::MAX;

        // Identity masks borrow c2's storage directly (the per-cell view was
        // already a zero-copy borrow — never materialize what was borrowed),
        // so the arena and its budget charge exist only for marg masks.
        let mut flat: Vec<InputPair> = Vec::new();
        let mut accounted_bytes: u64 = 0;
        if !identity {
            let mut total: usize = 0;
            for j in 0..k2 {
                if c2_level.nodes[j].is_internal() {
                    total += c2_level.pair_count_at(j);
                }
            }
            if total > u32::MAX as usize {
                return None;
            }
            if budget_reserve_exact(&mut flat, total).is_err() {
                // A reserve can fail AFTER charging (try_reserve succeeds, the
                // soft-budget check trips) — un-charge whatever capacity the
                // vec actually holds before dropping it.
                unaccount_transient_bytes(Self::cap_bytes(&flat));
                return None;
            }
            accounted_bytes = Self::cap_bytes(&flat);
        }

        let mut cols: Vec<ColSlice> = pool_take(&super::SCRATCH_C2_COLS);
        cols.clear();
        if cols.try_reserve(k2).is_err() {
            unaccount_transient_bytes(accounted_bytes);
            pool_put_bounded(&super::SCRATCH_C2_COLS, cols, MAX_LEVEL_ARENA_BYTES);
            return None;
        }

        if identity {
            for j in 0..k2 {
                // THE per-column resolution — the same accessor the per-cell
                // identity fast path (`pairs_view_decoded` → `pairs_view_into`)
                // calls, hoisted out of the row loop.
                let s = c2_level.pairs_of_idx(j);
                cols.push(ColSlice { ptr: s.as_ptr(), len: s.len() });
            }
        } else {
            // Pass 1: decode every column into the arena, recording only
            // lengths — the arena's base is not final until it is full.
            for j in 0..k2 {
                let before = flat.len();
                c2_level.decode_pairs_into(j, &mut flat, left_mask, right_mask);
                cols.push(ColSlice { ptr: std::ptr::null(), len: flat.len() - before });
            }
            // Pass 2: point each descriptor at its subrange of the finished
            // arena. The lengths sum to `flat.len()` by construction, so the
            // running offset never passes the end.
            let base = flat.as_ptr();
            let mut off = 0usize;
            for c in cols.iter_mut() {
                // SAFETY: `off <= flat.len() <= flat.capacity()` at every step,
                // so `base.add(off)` is inside the allocation (or exactly
                // one-past-the-end for a trailing empty column).
                c.ptr = unsafe { base.add(off) };
                off += c.len;
            }
        }

        Some(C2Columns { flat, cols, accounted_bytes })
    }

    fn cap_bytes(flat: &Vec<InputPair>) -> u64 {
        (flat.capacity() as u64) * (std::mem::size_of::<InputPair>() as u64)
    }
}

impl Drop for C2Columns {
    fn drop(&mut self) {
        unaccount_transient_bytes(self.accounted_bytes);
        // Hand the descriptor buffer back to the pool under the module-wide
        // retain cap, so one very wide level can't park its table there and
        // tax every later small apply.
        pool_put_bounded(
            &super::SCRATCH_C2_COLS,
            std::mem::take(&mut self.cols),
            MAX_LEVEL_ARENA_BYTES,
        );
    }
}

/// Fold the per-row alive-column masks for one decoded c1 row.
///
/// left: pass-through ⇒ always alive (`MAX`); `!nxm` ⇒ masks unused (`0`);
/// else the OR of `live_left_cols` over the row's left refs. right mirrors
/// (`MAX` when pass-through OR `!nxm`). Returns `None` when the row is
/// provably dead under `nxm` (every cell in it would be culled) — the caller
/// skips the whole row.
#[inline(always)]
pub(super) fn row_alive_masks(ctx: &CellCtx<'_>, inputs1: &[InputPair]) -> Option<(u128, u128)> {
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
pub(super) fn emit_product_node(
    level: &mut TddLevel,
    node_idx: &mut [u32],
    grid_pos: usize,
    pair_start: usize,
    pair_count: usize,
) -> Result<(), ApplyError> {
    if pair_count > 0 {
        let nid = level.nodes.len() as u32;
        node_idx[grid_pos] = nid;
        if pair_count == 1 {
            // Phase F: pop from whichever backing is active.
            let pair = level.pop_pair().unwrap();
            if pair.can_inline() {
                try_push(&mut level.nodes, TddNodeData::inline(pair))?;
            } else {
                let ps = level.pair_count();
                try_push_pair_into(level, pair)?;
                let ei = level.ext.len();
                try_push(&mut level.ext, ExtMulti { start: ps as u64, len: 1 })?;
                try_push(&mut level.nodes, TddNodeData::multi_extended(ei as u32))?;
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
pub(super) trait PairSink {
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
        node_idx: &mut [u32],
        grid_pos: usize,
        lc: u32,
        rc: u32,
    ) -> Result<(), ApplyError>;

    /// Start a multi-pair cell; returns the start token `end` consumes
    /// (the emit impl snapshots `level.pair_count()`).
    fn begin(&mut self) -> usize;

    /// One surviving (lc, rc) pair of a multi-pair cell.
    fn pair(&mut self, lc: u32, rc: u32) -> Result<(), ApplyError>;

    /// Finish a multi-pair cell; returns the number of pairs committed
    /// (non-emit impls return 0).
    fn end(
        &mut self,
        node_idx: &mut [u32],
        grid_pos: usize,
        start: usize,
    ) -> Result<usize, ApplyError>;
}

/// Build the output level: push pairs, emit product nodes, write `node_idx`.
pub(super) struct EmitSink<'a> {
    pub(super) level: &'a mut TddLevel,
}

impl PairSink for EmitSink<'_> {
    const ASSERT_INTERNAL: bool = true;

    #[inline(always)]
    fn single(
        &mut self,
        node_idx: &mut [u32],
        grid_pos: usize,
        lc: u32,
        rc: u32,
    ) -> Result<(), ApplyError> {
        let pair = InputPair { left: LocalNodeIdx(lc), right: LocalNodeIdx(rc) };
        let nid = self.level.nodes.len() as u32;
        node_idx[grid_pos] = nid;
        if pair.can_inline() {
            try_push(&mut self.level.nodes, TddNodeData::inline(pair))
        } else {
            let ps = self.level.pair_count();
            try_push_pair_into(self.level, pair)?;
            let ei = self.level.ext.len();
            try_push(&mut self.level.ext, ExtMulti { start: ps as u64, len: 1 })?;
            try_push(&mut self.level.nodes, TddNodeData::multi_extended(ei as u32))
        }
    }

    #[inline(always)]
    fn begin(&mut self) -> usize {
        self.level.pair_count()
    }

    #[inline(always)]
    fn pair(&mut self, lc: u32, rc: u32) -> Result<(), ApplyError> {
        try_push_pair_into(
            self.level,
            InputPair { left: LocalNodeIdx(lc), right: LocalNodeIdx(rc) },
        )
    }

    #[inline(always)]
    fn end(
        &mut self,
        node_idx: &mut [u32],
        grid_pos: usize,
        start: usize,
    ) -> Result<usize, ApplyError> {
        let pair_count = self.level.pair_tail_len(start);
        emit_product_node(self.level, node_idx, grid_pos, start, pair_count)?;
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
pub(super) struct CollectSink<'a> {
    pub(super) out: &'a mut Vec<InputPair>,
}

impl PairSink for CollectSink<'_> {
    // Collect walks may visit marginal-encoded operand nodes (the both-marginal
    // collapse), which the structural-internal assert would reject.
    const ASSERT_INTERNAL: bool = false;

    #[inline(always)]
    fn single(
        &mut self,
        _node_idx: &mut [u32],
        _grid_pos: usize,
        lc: u32,
        rc: u32,
    ) -> Result<(), ApplyError> {
        try_push(self.out, InputPair { left: LocalNodeIdx(lc), right: LocalNodeIdx(rc) })
    }

    #[inline(always)]
    fn begin(&mut self) -> usize {
        0
    }

    #[inline(always)]
    fn pair(&mut self, lc: u32, rc: u32) -> Result<(), ApplyError> {
        try_push(self.out, InputPair { left: LocalNodeIdx(lc), right: LocalNodeIdx(rc) })
    }

    #[inline(always)]
    fn end(
        &mut self,
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
/// `Deadline` from the amortized `nxm_deadline_check` in any arm — so even a
/// count-only sink is NOT infallible (it can bail mid-cell on a wide cell).
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(super) fn process_cell<L, R, S>(
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
                sink.single(node_idx, grid_pos, lc, rc)?;
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
        let _dl_armed = ctx.deadline_armed;
        let mut _dl_work: u64 = 0;
        nxm_deadline_check!(_dl_armed, _dl_work, inputs1.len());
        let cell_start = sink.begin();
        for p1 in inputs1 {
            let lc = left.get(node_idx, p1.left.0, p2.left.0);
            if lc == DEAD { continue; }
            let rc = right.get(node_idx, p1.right.0, p2.right.0);
            if rc == DEAD { continue; }
            sink.pair(lc, rc)?;
        }
        sink.end(node_idx, grid_pos, cell_start)?;
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
        let _dl_armed = ctx.deadline_armed;
        let mut _dl_work: u64 = 0;
        nxm_deadline_check!(_dl_armed, _dl_work, inputs2.len());
        let cell_start = sink.begin();
        for p2 in inputs2 {
            let lc = left.get(node_idx, p1.left.0, p2.left.0);
            if lc == DEAD { continue; }
            let rc = right.get(node_idx, p1.right.0, p2.right.0);
            if rc == DEAD { continue; }
            sink.pair(lc, rc)?;
        }
        sink.end(node_idx, grid_pos, cell_start)?;
    } else {
        // ── N×M (implies nxm: both levels multi-pair ⟹ masks built) ──────
        let left_dead = !left.passthrough()
            && left_alive_mask & ctx.reach_c2_left[j] == 0;
        let right_dead = !right.passthrough()
            && right_alive_mask & ctx.reach_c2_right[j] == 0;
        if left_dead || right_dead {
            return Ok(());
        }

        let cell_start = sink.begin();
        let _dl_armed = ctx.deadline_armed;
        let mut _dl_work = 0u64;
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
                nxm_deadline_check!(_dl_armed, _dl_work, inputs2.len());
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
                            sink.pair(lc, rc)?;
                        }
                    }
                }
            }
        } else {
            // ── General N×M ───────────────────────────────────────────────
            for p1 in inputs1 {
                nxm_deadline_check!(_dl_armed, _dl_work, inputs2.len());
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
                    sink.pair(lc, rc)?;
                }
            }
        }
        sink.end(node_idx, grid_pos, cell_start)?;
    }
    Ok(())
}

// ============================= Row-loop driver =============================
//
// The four build routes — marginal-child emit (`run_level_rows_marg`),
// sparse-marg emit (`run_level_rows_marg_sparse`), streaming collapse
// (`stream_collapse_rows`) and plain emit (`run_level_rows_plain`) — differ
// only in what they do per row and per cell. Everything around that (the
// per-row DEAD reset, the c1 pair decode, the empty/dead-row skips, the alive
// masks, the k2 column sweep, the between-cell poll) is ONE loop body, living
// once in `run_level_rows` and parameterized by a [`CellAction`] — the same
// shape as `process_cell`, which is one product walk parameterized by a
// [`PairSink`].

/// Everything one cell of the row loop needs, bundled and passed by value.
///
/// A bundle rather than a dozen `cell` parameters: restating the parameters in
/// every impl measured about twice the added source lines at identical codegen.
struct CellArgs<'a, 'c, L, R> {
    /// Column index — the c2 node.
    j: usize,
    /// TRUE c1 row index — NOT necessarily the grid row: the sparse-marg route
    /// builds every row at grid row 0 and needs this as the product entry's
    /// `c1_idx`.
    i: usize,
    /// Flat slab offset of the grid row this cell writes into —
    /// `ctx.t_base + CellAction::grid_row(i) * ctx.k2`, computed ONCE per row by
    /// the driver for its DEAD reset, so `grid_pos == row_base + j` and the reset
    /// and the kernel cannot drift.
    row_base: usize,
    /// Decoded pairs of c1 row `i` (never empty — empty rows are skipped).
    inputs1: &'a [InputPair],
    left_alive_mask: u128,
    right_alive_mask: u128,
    ctx: &'a CellCtx<'c>,
    c2_level_t: &'a TddLevel,
    inputs2_scratch: &'a mut Vec<InputPair>,
    node_idx: &'a mut [u32],
    left: &'a L,
    right: &'a R,
}

/// Per-row / per-cell action of the shared row loop ([`run_level_rows`]).
///
/// All hooks are `#[inline(always)]` in impls, so each instantiation
/// monomorphizes to what the hand-written per-route loop produced and the hooks
/// three of the four routes leave at their empty defaults vanish entirely.
trait CellAction<L: ChildLookup, R: ChildLookup> {
    /// Whether the driver debug-asserts that each c1 row node is structurally
    /// internal before decoding its pairs. Mirrors [`PairSink::ASSERT_INTERNAL`],
    /// and for the same reason cannot be a shared unconditional assert: the
    /// collapse and marginal-child routes legitimately walk marginal-encoded
    /// operand nodes.
    const ASSERT_INTERNAL: bool;

    /// Whether this action owns a DENSE `k1 × k2` slab — one grid row per c1 row,
    /// so the per-row DEAD resets tile the level's whole slab exactly once and
    /// can be replaced by a single fill (see [`run_level_rows`]). FALSE for
    /// [`SparseMargEmit`], which reuses grid row 0 for EVERY structural row and
    /// therefore must re-fill that one row between rows.
    const DENSE_SLAB: bool;

    /// Output grid row for c1 row `i`. Drives BOTH the per-row DEAD reset and
    /// the kernel's `grid_pos` (through `CellArgs::row_base`), so the two can
    /// never drift. Deliberately has no default — "row `i` of a dense slab" vs
    /// "the one reused row scratch" is exactly the distinction a default would
    /// paper over.
    fn grid_row(&self, i: usize) -> usize;

    /// Fires once per LIVE row, after the empty-row skip and after the alive
    /// masks resolve — the seam for per-row state an action needs pinned before
    /// its first cell but only for rows that are actually built.
    #[inline(always)]
    fn begin_row(&mut self, _i: usize, _inputs1: &[InputPair]) {}

    /// One cell of the row.
    fn cell(&mut self, a: CellArgs<'_, '_, L, R>) -> Result<(), ApplyError>;

    /// Fires once after the last row, on the success path only — a failing cell
    /// short-circuits out of the driver, so an action's end-of-level bookkeeping
    /// is skipped for a level the apply is abandoning.
    #[inline(always)]
    fn finish(&mut self, _k1: usize, _ctx: &CellCtx<'_>, _node_idx: &[u32]) {}
}

/// Size gate for [`run_level_rows`]'s one-shot DEAD slab fill (A3). At or below
/// this many cells the whole `k1 × k2` slab is filled once before the row loop;
/// above it the fill stays row-wise so the reset of the row about to be built
/// keeps that row in L1. THE gate constant — defined once, read once.
const DEAD_SLAB_FILL_MAX_CELLS: usize = 1 << 16;

/// The ONE row/cell loop of the dense product build.
///
/// `const DENSE` skips the per-row alive-mask fold on levels where it provably
/// cannot skip anything: `ctx.nxm` false and neither side a pass-through, where
/// `row_alive_masks` always returns `(0, u128::MAX)`. Only the plain route
/// reaches that regime; the other three always fold.
///
/// `k2` and `t_base` are read from `ctx` rather than passed alongside it, so
/// the row reset and the kernel's `grid_pos` derive from the same values by
/// construction.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn run_level_rows<const DENSE: bool, L, R, A>(
    k1: usize,
    c1_level_t: &TddLevel,
    c2_level_t: &TddLevel,
    ctx: &CellCtx<'_>,
    inputs1_scratch: &mut Vec<InputPair>,
    inputs2_scratch: &mut Vec<InputPair>,
    node_idx: &mut [u32],
    left: &L,
    right: &R,
    action: &mut A,
) -> Result<(), ApplyError>
where
    L: ChildLookup,
    R: ChildLookup,
    A: CellAction<L, R>,
{
    // Amortized wall-deadline/cancel poll: one TLS read per ~65k cell iterations
    // so an expired deadline cuts within a fraction of a level rather than
    // waiting for the next vtree-level boundary (20+ s on the widest levels).
    let mut poll = crate::tdd::limits::PollTicker::new(super::budget::DENSE_CELL_POLL_STRIDE);
    let k2 = ctx.k2;

    // A3 — one slab fill instead of `k1` row fills. On a dense-slab action the
    // per-row resets below tile `t_base .. t_base + k1*k2` exactly once each
    // (`grid_row(i) == i`), so hoisting them into a single `fill` writes the same
    // cells with the same byte pattern; only the order changes, and nothing reads
    // this level's slab between rows (the child lookups read the CHILD levels'
    // disjoint slabs). Above the size gate the per-row form stays: interleaving
    // the reset with the row's cell work is what keeps the active row L1-resident
    // on a large grid.
    let slab_fill = A::DENSE_SLAB && k1.saturating_mul(k2) <= DEAD_SLAB_FILL_MAX_CELLS;
    if slab_fill {
        node_idx[ctx.t_base..ctx.t_base + k1 * k2].fill(DEAD);
    }

    for i in 0..k1 {
        let row_base = ctx.t_base + action.grid_row(i) * k2;
        if !slab_fill {
            node_idx[row_base..row_base + k2].fill(DEAD);
        }

        if A::ASSERT_INTERNAL {
            debug_assert!(c1_level_t.nodes[i].is_internal()
                || c1_level_t.nodes[i].b == u32::MAX,
                "expected internal node at internal vtree position: t_base={} i={i} k1={k1} node_a={:#x} node_b={:#x}",
                ctx.t_base, c1_level_t.nodes[i].a, c1_level_t.nodes[i].b);
        }

        let inputs1 = c1_level_t.pairs_view_decoded(
            i, inputs1_scratch, ctx.left_mask, ctx.right_mask,
        );
        // Empty pairs means dead (ZERO-containing) node — skip this row.
        if inputs1.is_empty() { continue; }

        let (left_alive_mask, right_alive_mask) = if DENSE {
            (0u128, u128::MAX)
        } else {
            match row_alive_masks(ctx, inputs1) {
                Some(masks) => masks,
                // Row skip: if c1[i]'s pairs all reference dead child rows,
                // no cell in this row can produce output.
                None => continue,
            }
        };

        action.begin_row(i, inputs1);

        for j in 0..k2 {
            action.cell(CellArgs {
                j, i, row_base, inputs1, left_alive_mask, right_alive_mask,
                ctx, c2_level_t, left, right,
                inputs2_scratch: &mut *inputs2_scratch,
                node_idx: &mut *node_idx,
            })?;
        }
        // One `tick_by(k2)` per row instead of `tick()` per cell: the ticker only
        // meters accumulated work, so the same total is booked either way. A row
        // wider than the stride now polls once rather than once per stride's worth
        // of cells — the poll is an idempotent deadline read, so firing once per
        // crossing is equivalent, and a row that skipped the `j` loop (empty or
        // dead) books nothing, exactly as before (`k2 == 0` books nothing either).
        poll.tick_by(k2 as u64)?;
    }
    action.finish(k1, ctx, node_idx);
    Ok(())
}

/// Route A action: materialize each cell as a product node in the dense
/// `k1 × k2` output slab.
struct MargEmit<'a> {
    level: &'a mut TddLevel,
}

impl<L: ChildLookup, R: ChildLookup> CellAction<L, R> for MargEmit<'_> {
    /// A marginal-child level's c1 rows may be marginal-encoded.
    const ASSERT_INTERNAL: bool = false;

    const DENSE_SLAB: bool = true;

    /// Dense slab: one grid row per c1 row.
    #[inline(always)]
    fn grid_row(&self, i: usize) -> usize { i }

    #[inline(always)]
    fn cell(&mut self, a: CellArgs<'_, '_, L, R>) -> Result<(), ApplyError> {
        process_cell::<_, _, _>(
            a.j, a.row_base, a.inputs1, a.left_alive_mask, a.right_alive_mask,
            a.ctx, a.c2_level_t, a.inputs2_scratch, a.node_idx, a.left, a.right,
            &mut EmitSink { level: &mut *self.level },
        )
    }
}

/// Route A row-loop: forward-order scatter for levels with at least one marginal child.
///
/// Called when `marg_child_dispatch` is true. Iterates rows 0..k1 in forward order
/// (lever-8 reverse iteration is a no-op on this path — see the per-level comment),
/// running the emit kernel for each (i,j) cell.
///
/// Never streams: streaming marginal-child levels take the collapse-at-source
/// walker ([`run_level_rows_stream_count`]) unconditionally — there is no
/// materialize-then-convert fallback for them.
///
/// Takes `c1_level_t` as a pre-taken immutable borrow into c1.levels[t_idx] so the
/// caller can keep its `vtree = &c1.vtree` borrow live simultaneously.
pub(super) fn run_level_rows_marg(
    k1: usize,
    c1_level_t: &TddLevel,
    c2_level_t: &TddLevel,
    cell_ctx: &CellCtx<'_>,
    inputs1_scratch: &mut Vec<InputPair>,
    inputs2_scratch: &mut Vec<InputPair>,
    level: &mut TddLevel,
    node_idx: &mut [u32],
) -> Result<(), ApplyError> {
    let left = MargLookup::left(cell_ctx);
    let right = MargLookup::right(cell_ctx);
    run_level_rows::<false, _, _, _>(
        k1, c1_level_t, c2_level_t, cell_ctx,
        inputs1_scratch, inputs2_scratch, node_idx,
        &left, &right,
        &mut MargEmit { level },
    )
}

/// Per-cell scalar fold for the streaming collapse walker
/// ([`stream_collapse_rows`]): resolves one alive cell's collected pairs to a
/// single scalar and records it in the streaming state, remapping
/// `node_idx[grid_pos]` from DEAD to the new slot index. ONE impl, generic
/// over the value kind, so the ONE row/cell loop serves both the integer count
/// fold and the weighted (`BigRational`) fold (D2 stage 1).
pub(super) trait StreamCellFold {
    fn fold_cell(
        &mut self,
        pairs: &[InputPair],
        node_idx: &mut [u32],
        grid_pos: usize,
    ) -> Result<(), ApplyError>;
}

/// The single source of truth for the fold / column-push / `node_idx` remap
/// step, for both value kinds.
///
/// Growth past the output column's initial `k1.max(k2)` reserve must stay
/// fallible — the column can grow up to alive cells (≤ k1*k2), well past the
/// upfront reserve. The push discipline is the value kind's: `CountVec::push`
/// stores a `Count::Big` as the `STREAM_OVERFLOW` sentinel with the exact
/// `BigUint` in the lazily-built, `None`-backfilled side table; the weighted
/// column is an ordinary `Vec` whose per-pair transient is budget-charged by
/// [`CollectSink`] instead.
impl<F: StreamPayload> StreamCellFold for StreamState<'_, F> {
    #[inline(always)]
    fn fold_cell(
        &mut self,
        pairs: &[InputPair],
        node_idx: &mut [u32],
        grid_pos: usize,
    ) -> Result<(), ApplyError> {
        let v = F::fold_cell(pairs, &self.left, &self.right, self.ws);
        let cell_idx = F::col_len::<ApplyBudget>(self.counts);
        F::push_col::<ApplyBudget>(self.counts, v)?;
        node_idx[grid_pos] = cell_idx as u32;
        Ok(())
    }
}

/// Streaming collapse-at-source entry — the ONE driver for streaming-
/// marginalize levels, generic over the child lookups:
///
/// - Marginal-child shapes (Route A): at least one child marginal
///   (`MargLookup` sides — a `MargLookup` degrades to the plain dense grid
///   read on a non-pass-through side, so both-marginal, and
///   one-marginal × leaf all route here with the same lookups). The level is
///   a marginalize target whose every alive cell collapses to a scalar
///   `Σ left × right` — there is no downstream structure to keep.
/// - Plain shape (Route B): a streaming target with no marginal child
///   (leaf children at the lowest levels), served by `DenseLookup` sides.
///
/// Monomorphizes the per-cell fold once per level on the state's value kind —
/// integer or weighted — binds that kind's two child column VIEWS
/// ([`attach_children`]; the columns are read in place in `left_level` /
/// `right_level`, never copied), then runs the shared [`stream_collapse_rows`]
/// loop. Which side is marginal is carried by the views themselves
/// (`StreamChild::is_marg`), so the fold needs no shape-specific wiring.
///
/// The views live only for this call: `stream_state` owns the output column and
/// outlives them, so the caller can retake `&mut levels` to commit it.
///
/// This is the ONLY streaming build path — there is no materialize-then-fold
/// alternative to fall back on: [`bothmarg_collapse_enabled`] disables
/// streaming *eligibility* rather than switching routes.
#[allow(clippy::too_many_arguments)]
pub(super) fn run_level_rows_stream_count<L: ChildLookup, R: ChildLookup>(
    k1: usize,
    c1_level_t: &TddLevel,
    c2_level_t: &TddLevel,
    cell_ctx: &CellCtx<'_>,
    inputs1_scratch: &mut Vec<InputPair>,
    inputs2_scratch: &mut Vec<InputPair>,
    node_idx: &mut [u32],
    left: &L,
    right: &R,
    stream_state: &mut StreamLevelState,
    left_idx: usize,
    right_idx: usize,
    vtree: &crate::vtree::Vtree,
    left_level: &TddLevel,
    right_level: &TddLevel,
    computed: &[Option<CountVec<ApplyBudget>>],
    computed_weights: &[Option<Vec<WeightVal>>],
    ws: Option<&crate::tdd::weight_store::WeightStore>,
) -> Result<(), ApplyError> {
    match stream_state {
        StreamLevelState::Weighted(counts) => {
            let mut st = attach_children::<WeightFold>(
                left_idx, right_idx, vtree, left_level, right_level, computed_weights, counts, ws,
            )?;
            stream_collapse_rows(
                k1, c1_level_t, c2_level_t, cell_ctx,
                inputs1_scratch, inputs2_scratch, node_idx, left, right, &mut st,
            )
        }
        StreamLevelState::Int(counts) => {
            let mut st = attach_children::<IntFold>(
                left_idx, right_idx, vtree, left_level, right_level, computed, counts, None,
            )?;
            stream_collapse_rows(
                k1, c1_level_t, c2_level_t, cell_ctx,
                inputs1_scratch, inputs2_scratch, node_idx, left, right, &mut st,
            )
        }
    }
}

/// Collapse-at-source action: enumerate each alive cell's surviving `(lc, rc)`
/// refs into a reused scratch `Vec<InputPair>` and feed them straight to the
/// fold, never touching `level`.
struct StreamCollapse<'a, F> {
    fold: &'a mut F,
    /// Reused across all cells — bounds the transient peak to one cell's pairs.
    cell_pairs: Vec<InputPair>,
}

impl<L: ChildLookup, R: ChildLookup, F: StreamCellFold> CellAction<L, R>
    for StreamCollapse<'_, F>
{
    /// Collapse walks may visit marginal-encoded operand nodes.
    const ASSERT_INTERNAL: bool = false;

    const DENSE_SLAB: bool = true;

    /// Dense slab: one grid row per c1 row (the collapsed scalars live in the
    /// streaming column; `node_idx` still carries this level's cell→slot map).
    #[inline(always)]
    fn grid_row(&self, i: usize) -> usize { i }

    #[inline(always)]
    fn cell(&mut self, a: CellArgs<'_, '_, L, R>) -> Result<(), ApplyError> {
        self.cell_pairs.clear();
        process_cell::<_, _, _>(
            a.j, a.row_base, a.inputs1, a.left_alive_mask, a.right_alive_mask,
            a.ctx, a.c2_level_t, a.inputs2_scratch, a.node_idx, a.left, a.right,
            &mut CollectSink { out: &mut self.cell_pairs },
        )?;
        // An empty cell stays DEAD (no slot) — mirrors the emit walk, where
        // `emit_product_node` produces no node for zero pairs. Same `row_base + j`
        // the kernel used, not a second derivation of it.
        if !self.cell_pairs.is_empty() {
            self.fold.fold_cell(&self.cell_pairs, a.node_idx, a.row_base + a.j)?;
        }
        Ok(())
    }
}

/// The collapse route's entry into the shared row loop (see
/// [`run_level_rows_stream_count`] for the route/shape documentation).
///
/// Count-identical to the materializing emit walk by construction — same
/// per-cell pair multiset (the kernel IS the emit walk minus node
/// materialization; its row/reach culls prune only provably-dead pairs, and
/// the ≥64×64 grouped N×M path emits the same multiset in a different order
/// under an order-independent fold).
#[allow(clippy::too_many_arguments)]
fn stream_collapse_rows<L: ChildLookup, R: ChildLookup, F: StreamCellFold>(
    k1: usize,
    c1_level_t: &TddLevel,
    c2_level_t: &TddLevel,
    cell_ctx: &CellCtx<'_>,
    inputs1_scratch: &mut Vec<InputPair>,
    inputs2_scratch: &mut Vec<InputPair>,
    node_idx: &mut [u32],
    left: &L,
    right: &R,
    fold: &mut F,
) -> Result<(), ApplyError> {
    // A4: the per-cell scratch is pooled, not rebuilt from empty at every
    // streaming level — `cell` clears it before each cell, so pooled capacity can
    // carry nothing but capacity. Returned on the error path too, under the
    // module's byte cap, so one huge level can't park its arena in the pool.
    let mut action = StreamCollapse {
        fold,
        cell_pairs: pool_take(&super::SCRATCH_CELL_PAIRS),
    };
    let result = run_level_rows::<false, _, _, _>(
        k1, c1_level_t, c2_level_t, cell_ctx,
        inputs1_scratch, inputs2_scratch, node_idx,
        left, right,
        &mut action,
    );
    pool_put_bounded(
        &super::SCRATCH_CELL_PAIRS,
        std::mem::take(&mut action.cell_pairs),
        MAX_LEVEL_ARENA_BYTES,
    );
    result
}

/// Sparse-output action: emit into the reused row scratch, then record each
/// surviving cell as a `ProductEntry` instead of leaving it in a dense slab.
struct SparseMargEmit<'a> {
    level: &'a mut TddLevel,
    product_list: &'a mut Vec<ProductEntry>,
}

impl<L: ChildLookup, R: ChildLookup> CellAction<L, R> for SparseMargEmit<'_> {
    /// A marginal-child level's c1 rows may be marginal-encoded.
    const ASSERT_INTERNAL: bool = false;

    /// NOT a dense slab — every structural row is rebuilt in the SAME `k2`-wide
    /// scratch row, so its DEAD reset must fire between rows and cannot be
    /// hoisted into a one-shot slab fill.
    const DENSE_SLAB: bool = false;

    /// ONE reused `k2`-wide row scratch instead of a dense slab: every
    /// structural row builds at grid row 0, so `row_base == ctx.t_base`.
    /// The true row index survives only in `CellArgs::i`, which the product
    /// entry's `c1_idx` reads below.
    #[inline(always)]
    fn grid_row(&self, _i: usize) -> usize { 0 }

    #[inline(always)]
    fn cell(&mut self, a: CellArgs<'_, '_, L, R>) -> Result<(), ApplyError> {
        let row_pos = a.row_base + a.j;
        process_cell::<_, _, _>(
            a.j, a.row_base, a.inputs1, a.left_alive_mask, a.right_alive_mask,
            a.ctx, a.c2_level_t, a.inputs2_scratch, a.node_idx, a.left, a.right,
            &mut EmitSink { level: &mut *self.level },
        )?;
        let nid = a.node_idx[row_pos];
        if nid != DEAD {
            try_push(self.product_list, ProductEntry {
                c1_idx: C1NodeIdx(a.i as u32),
                c2_idx: C2NodeIdx(a.j as u32),
                prod_idx: ProdNodeIdx(nid),
            })?;
        }
        Ok(())
    }
}

/// Sparse-output variant of Route A for an *exactly-one*-marginal-child level
/// whose output is structural (never a marginalize target).
///
/// Identical per-cell math to [`run_level_rows_marg`] — it runs the same emit
/// kernel — but instead of writing into a dense `k1*k2` slab it reuses a single
/// `k2`-wide row scratch (`cell_ctx.t_base .. +k2`) and records each surviving
/// cell into `product_list`. The marginal child is a pass-through carrier (it
/// never kills a pair), so the *structural* sibling alone governs which cells
/// are alive; the dense slab the other path allocates is therefore mostly DEAD
/// and pure overhead. The grandparent densifies the emitted `product_list`
/// lazily via `ensure_grid`, reproducing exactly the grid the dense path would
/// have built.
///
/// Every cell is built at grid row 0 so the kernel's
/// `grid_pos == cell_ctx.t_base + j` (one row), then read back from the scratch
/// and, if alive, pushed as
/// `ProductEntry { c1_idx: row i, c2_idx: col j, prod_idx: node }`.
///
/// No streaming: the `use_sparse_marg` gate excludes marginalize targets
/// explicitly (`!is_marg_target`), so a streaming target never routes here.
/// (A one-marginal-child marginalize target does exist; it takes the streaming
/// dispatch, not this sparse path.)
#[allow(clippy::too_many_arguments)]
pub(super) fn run_level_rows_marg_sparse(
    k1: usize,
    c1_level_t: &TddLevel,
    c2_level_t: &TddLevel,
    cell_ctx: &CellCtx<'_>,
    inputs1_scratch: &mut Vec<InputPair>,
    inputs2_scratch: &mut Vec<InputPair>,
    level: &mut TddLevel,
    node_idx: &mut [u32],
    product_list: &mut Vec<ProductEntry>,
) -> Result<(), ApplyError> {
    let left = MargLookup::left(cell_ctx);
    let right = MargLookup::right(cell_ctx);
    run_level_rows::<false, _, _, _>(
        k1, c1_level_t, c2_level_t, cell_ctx,
        inputs1_scratch, inputs2_scratch, node_idx,
        &left, &right,
        &mut SparseMargEmit { level, product_list },
    )
}

/// Route B action: materializing emit.
struct PlainEmit<'a> {
    level: &'a mut TddLevel,
}

impl<L: ChildLookup, R: ChildLookup> CellAction<L, R> for PlainEmit<'_> {
    /// Route B's operand invariant: no marginal child, so every c1 row node at
    /// an internal vtree position is structurally internal.
    const ASSERT_INTERNAL: bool = true;

    const DENSE_SLAB: bool = true;

    /// Dense slab: one grid row per c1 row.
    #[inline(always)]
    fn grid_row(&self, i: usize) -> usize { i }

    #[inline(always)]
    fn cell(&mut self, a: CellArgs<'_, '_, L, R>) -> Result<(), ApplyError> {
        process_cell::<_, _, _>(
            a.j, a.row_base, a.inputs1, a.left_alive_mask, a.right_alive_mask,
            a.ctx, a.c2_level_t, a.inputs2_scratch, a.node_idx, a.left, a.right,
            &mut EmitSink { level: &mut *self.level },
        )
    }
}

/// Route B row-loop: forward pass for plain (non-marginal-child) levels.
///
/// Called when `marg_child_dispatch` is false. Iterates rows in forward order.
/// Runs the emit kernel with the caller's plain lookups (no pass-through, no
/// mask decode — the caller guarantees no marginal child).
///
/// `const DENSE: bool` selects the branch-hoisted fast path for the common case where
/// both level-invariant guards hold simultaneously:
///   1. `cell_ctx.nxm == false` — no liveness-mask filtering.
///   2. `!cell_ctx.left_passthrough && !cell_ctx.right_passthrough` — no pass-through sides.
///
/// When `DENSE = true` the inner loop is free of branches on those constants;
/// when `DENSE = false`, nxm row-skip checks are active.
///
/// Never streams: streaming levels take the collapse-at-source walker
/// ([`run_level_rows_stream_count`]) unconditionally — there is no post-cell
/// snapshot conversion.
#[inline(always)]
pub(super) fn run_level_rows_plain<const DENSE: bool, L: ChildLookup, R: ChildLookup>(
    k1: usize,
    c1_level_t: &TddLevel,
    c2_level_t: &TddLevel,
    cell_ctx: &CellCtx<'_>,
    inputs1_scratch: &mut Vec<InputPair>,
    inputs2_scratch: &mut Vec<InputPair>,
    level: &mut TddLevel,
    node_idx: &mut [u32],
    left_lookup: &L,
    right_lookup: &R,
) -> Result<(), ApplyError> {
    // When DENSE, the alive masks are constants (nxm is false, no pass-through):
    //   left_alive_mask  = 0u128      (the !nxm branch of `row_alive_masks`)
    //   right_alive_mask = u128::MAX  (the `|| !nxm` branch of `row_alive_masks`)
    // The driver passes those directly to the kernel, skipping the fold.
    let mut action = PlainEmit { level };
    run_level_rows::<DENSE, _, _, _>(
        k1, c1_level_t, c2_level_t, cell_ctx,
        inputs1_scratch, inputs2_scratch, node_idx,
        left_lookup, right_lookup,
        &mut action,
    )
}

#[cfg(test)]
#[path = "cell_tests.rs"]
mod tests;
