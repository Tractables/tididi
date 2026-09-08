//! Apply setup: phases 1-3 of `apply_and_fallible_inner` (width/marginal-entry
//! snapshot, sparse/budget pre-scan, grid/product-list allocation), bundled into
//! `ApplySetup` and produced by `apply_and_setup`. Pure code motion out of
//! `conjoin/mod.rs`; the driver destructures `ApplySetup` back into its locals.
//! Scratch pools, `MARG_ENTRY_*`, `APPLY_BYTES_PER_CELL`, and `APPLY_LIMITS`
//! stay in `mod.rs`/`budget` and are reached via `super::`.

use crate::vtree::VtreeIdx;
use crate::tdd::types::{self, *};
use crate::tdd::counts::{ApplyBudget, CountVec};
use crate::tdd::utils::pool_take;
use super::{ApplyError, LevelGrid, APPLY_BYTES_PER_CELL,
    SCRATCH_GRIDS, SCRATCH_C1_WIDTHS, SCRATCH_C2_WIDTHS, SCRATCH_STREAM_COUNTS,
    SCRATCH_PRODUCT_LISTS, SCRATCH_LIVE_COUNTS, SCRATCH_HAS_PL, SCRATCH_NODE_IDX,
    MARG_ENTRY_C1, MARG_ENTRY_C2};
use super::budget::try_resize_dead;
use crate::tdd::limits::APPLY_LIMITS;
use super::sparse::{sparse_config, ProductEntry};

/// Bundled result of `apply_and_setup` — the per-apply working state produced
/// before the bottom-up level sweep. (Was a 14-tuple.)
pub(super) struct ApplySetup {
    pub(super) levels: Vec<TddLevel>,
    pub(super) grids: Vec<LevelGrid>,
    pub(super) c1_widths: Vec<usize>,
    pub(super) c2_widths: Vec<usize>,
    pub(super) min_grid: usize,
    pub(super) sparsity_factor: u128,
    pub(super) might_use_sparse: bool,
    pub(super) stream_computed: Vec<Option<CountVec<ApplyBudget>>>,
    pub(super) node_idx: Vec<u32>,
    pub(super) grid_end: usize,
    pub(super) product_lists: Vec<Vec<ProductEntry>>,
    pub(super) live_counts: Vec<usize>,
    pub(super) has_pl: Vec<bool>,
    /// True iff some operand level was marginal at apply entry — i.e. iff the
    /// `MARG_ENTRY_C1`/`MARG_ENTRY_C2` snapshots below were actually filled.
    /// When false they were CLEARED, so every `plan_marg_level` lookup into them
    /// is provably `None`; threading the flag out lets that path skip the four
    /// per-level `RefCell` borrows entirely.
    pub(super) any_entry_marginal: bool,
}

/// Phases 1–3 of `apply_and_fallible_inner`: width/marginal-entry snapshot,
/// sparse/budget pre-scan, grid/product-list allocation.
///
/// Fills the `MARG_ENTRY_C1`/`MARG_ENTRY_C2` thread-locals as a side effect
/// (the pass-through carrier selector in the main loop reads them).
///
/// Returns owned scratch vectors so the caller can destructure them into the
/// same local names, leaving phases 4–6 untouched.
#[inline(always)]
#[allow(clippy::type_complexity)]
/// Snapshot both operands' per-level widths, note whether either carries a
/// marginal level at entry, and accumulate the dense-route cell count the
/// predictive budget check below reads.
///
/// The widths must be read before the bottom-up sweep's identity swaps steal
/// levels, which zero `effective_width` and clear `is_marginal`. Under a
/// restriction only `R ∪ children(R)` is ever indexed, so only those levels are
/// visited and `any_entry_marginal` stays false: the snapshot it guards exists
/// to recover an operand child that an identity fast path stole mid-sweep, and
/// a restricted apply takes no fast path.
fn snapshot_widths(
    c1: &Tdd,
    c2: &Tdd,
    num_nodes: usize,
    min_grid: usize,
    restrict: Option<&super::Restrict<'_>>,
    c1_widths: &mut [usize],
    c2_widths: &mut [usize],
) -> (u64, bool) {
    let mut any_entry_marginal = false;
    let mut total_cells: u64 = 0;
    let mut width_at = |i: usize, total_cells: &mut u64, any: &mut bool| {
        let w1 = c1.effective_width(VtreeIdx(i as u32));
        let w2 = c2.effective_width(VtreeIdx(i as u32));
        c1_widths[i] = w1;
        c2_widths[i] = w2;
        *any |= c1.levels[i].is_marginal() | c2.levels[i].is_marginal();
        let cells = (w1 as u64).saturating_mul(w2 as u64);
        if cells <= min_grid as u64 {
            *total_cells = total_cells.saturating_add(cells);
        }
    };
    if let Some(r) = restrict {
        let mut ignored = false;
        for &t in r.touched {
            width_at(t.idx(), &mut total_cells, &mut ignored);
        }
    } else {
        for i in 0..num_nodes {
            width_at(i, &mut total_cells, &mut any_entry_marginal);
        }
    }
    drop(width_at);
    (total_cells, any_entry_marginal)
}

/// Record which of each operand's levels are marginal at entry, for the
/// pass-through carrier selector to consult after an identity swap has stolen a
/// level (which clears the flag on the level itself).
///
/// With nothing marginal at entry the snapshot is all-false and the selector's
/// default answers identically, so the two passes are skipped — but the
/// thread-locals are still cleared, so a previous marginal apply on this thread
/// leaves nothing stale behind.
fn snapshot_entry_marginality(c1: &Tdd, c2: &Tdd, num_nodes: usize, any_entry_marginal: bool) {
    if any_entry_marginal {
        MARG_ENTRY_C1.with(|v| {
            let mut v = v.borrow_mut();
            v.clear();
            for i in 0..num_nodes { v.push(c1.levels[i].is_marginal()); }
        });
        MARG_ENTRY_C2.with(|v| {
            let mut v = v.borrow_mut();
            v.clear();
            for i in 0..num_nodes { v.push(c2.levels[i].is_marginal()); }
        });
    } else {
        MARG_ENTRY_C1.with(|v| v.borrow_mut().clear());
        MARG_ENTRY_C2.with(|v| v.borrow_mut().clear());
    }
}

/// Clear the per-level sparse bookkeeping this apply can read.
///
/// A restricted apply's reachable set is `R ∪ children(R)`; unrestricted, it is
/// every level. Entries outside the set are unreachable by construction, so
/// leaving them stale is what turns whole-array memsets into `O(|R|)` writes.
fn reset_level_tracking(
    restrict: Option<&super::Restrict<'_>>,
    num_nodes: usize,
    live_counts: &mut [usize],
    product_lists: &mut [Vec<ProductEntry>],
    has_pl: &mut [bool],
) {
    match restrict {
        Some(r) => {
            for &t in r.touched {
                live_counts[t.idx()] = 0;
                product_lists[t.idx()].clear();
                has_pl[t.idx()] = false;
            }
        }
        None => {
            live_counts[..num_nodes].fill(0);
            for pl in &mut product_lists[..num_nodes] { pl.clear(); }
            has_pl[..num_nodes].fill(false);
        }
    }
}

/// Give every level its grid descriptor and, where the whole product is dense,
/// the flat `node_idx` arena behind them.
///
/// With any sparse level present the arena is bump-allocated as levels are
/// reached, so every level starts as `Sparse` and grid space is claimed later;
/// otherwise the layout is computed up front and the arena sized once.
fn layout_grids(
    might_use_sparse: bool,
    restrict: Option<&super::Restrict<'_>>,
    num_nodes: usize,
    c1_widths: &[usize],
    c2_widths: &[usize],
    grids: &mut [LevelGrid],
) -> Result<(Vec<u32>, usize), ApplyError> {
    let mut node_idx: Vec<u32>;
    let grid_end: usize;
    if might_use_sparse {
        // ── Bump allocator mode ──────────────────────────────────────────
        // Allocate grid space incrementally. Sparse levels skip grids entirely;
        // their parents consume product_lists instead of node_idx lookups.
        node_idx = pool_take(&SCRATCH_NODE_IDX);
        match restrict {
            Some(r) => for &t in r.touched { grids[t.idx()] = LevelGrid::Sparse; },
            None => grids[..num_nodes + 1].fill(LevelGrid::Sparse),
        }
        grid_end = 0;
    } else {
        // ── Pre-computed layout ──────────────────────────────────────────
        // All levels get grids. No sparse infrastructure needed. Variant is
        // overwritten by each producer below (Leaf / DenseStrict); we seed
        // each entry with the right base here and update the kind in-place.
        let mut cursor = 0usize;
        match restrict {
            Some(r) => for &t in r.touched {
                grids[t.idx()] = LevelGrid::DenseWeak { base: cursor };
                cursor += c1_widths[t.idx()] * c2_widths[t.idx()];
            },
            None => for i in 0..num_nodes {
                grids[i] = LevelGrid::DenseWeak { base: cursor };
                cursor += c1_widths[i] * c2_widths[i];
            },
        }
        grids[num_nodes] = LevelGrid::DenseWeak { base: cursor };
        grid_end = cursor;

        node_idx = pool_take(&SCRATCH_NODE_IDX);
        try_resize_dead(&mut node_idx, grid_end)?;
    }
    Ok((node_idx, grid_end))
}

pub(super) fn apply_and_setup(
    c1: &mut Tdd,
    c2: &mut Tdd,
    vtree: &crate::vtree::Vtree,
    num_nodes: usize,
    marginalize_targets: Option<&[bool]>,
    restrict: Option<&super::Restrict<'_>>,
) -> Result<ApplySetup, ApplyError> {
    let levels: Vec<TddLevel> = types::take_levels(num_nodes);

    // ── Product grid: node_idx + grids descriptors ───────────────────────
    //
    // `node_idx` is a flat array: per level, (i, j) cells live at
    //     base[t] + i * k2[t] + j
    // and hold the output index for c1[i] ∧ c2[j] at that level, or DEAD
    // if that product was zero. u32 (not u16) because widths exceed 65k on
    // hard benchmarks. The base for each level comes from `grids[t]`; the
    // LevelGrid variant also records *what produced* the level (leaf grid
    // from CONJOIN_GRID, dense-emit row-major-monotone, scatter-materialised,
    // or sparse-only with no materialised grid).
    //
    // VALIDITY. A cell holds a meaningful value only where its producer wrote
    // one. The dense routes fill a level's whole grid, DEAD included; the sparse
    // route writes only the cells its scatter produced and leaves the rest as
    // whatever the arena's previous tenant left — the arena is bump-allocated
    // and reclaimed, never zeroed on reuse. So a read of `node_idx` is sound
    // only for a level whose `LevelGrid` variant says the grid was
    // materialised, which is what the stale-grid guard on the output path
    // checks before trusting the root cell.
    let mut grids: Vec<LevelGrid> = pool_take(&SCRATCH_GRIDS);
    if grids.len() < num_nodes + 1 {
        grids.resize(num_nodes + 1, LevelGrid::Sparse);
    }

    // Snapshot widths *before* the identity swaps below — swaps steal levels
    // from c1 / c2, making their effective_width return 0 afterwards.
    let mut c1_widths = pool_take(&SCRATCH_C1_WIDTHS);
    let mut c2_widths = pool_take(&SCRATCH_C2_WIDTHS);
    if c1_widths.len() < num_nodes { c1_widths.resize(num_nodes, 0); }
    if c2_widths.len() < num_nodes { c2_widths.resize(num_nodes, 0); }
    // Snapshot widths AND detect whether any operand level is marginal at entry,
    // in one pass — both must be read before the bottom-up sweep's identity swaps
    // steal levels (which zero `effective_width` and flip `is_marginal`).
    //
    // The predictive budget's dense-cell sum is accumulated in this same sweep.
    // It reads exactly the two widths this loop already has in registers, over
    // exactly the same `0..num_nodes` range, so folding it in is a verbatim
    // move of the loop body — one fewer pass over the width arrays. (The sparse
    // pre-scan below is deliberately NOT fused: it ranges over
    // `internal_bottomup()`, which walks the cached topo order — reachable
    // internal nodes only — and that is not the same set as
    // `(0..num_nodes).filter(|i| !nodes[i].is_leaf())`.) `sparse_config()` moves
    // above the loop to supply `min_grid`; it is a pure config read with no
    // bearing on the operands.
    let cfg = sparse_config();
    let min_grid = cfg.min_grid;
    let sparsity_factor = cfg.sparsity_factor;
    let (total_cells, any_entry_marginal) = snapshot_widths(
        c1, c2, num_nodes, min_grid, restrict, &mut c1_widths, &mut c2_widths,
    );

    // Snapshot per-level is_marginal BEFORE the bottom-up sweep mutates operands
    // (an identity-swap steals levels → is_marginal flips true→false). The
    // pass-through carrier selector reads this to recover a child that was
    // marginal at entry but got stolen into the output store mid-sweep.
    //
    // When NO operand level is marginal at entry (the dominant pure-Boolean / MC
    // fold path), the snapshot is definitionally all-false and the carrier
    // selector's `get(idx).unwrap_or(false)` returns the same false for every
    // index — so skip the two O(num_nodes) RefCell-push passes entirely and just
    // clear the thread-locals so a prior marginal apply on this thread leaves no
    // stale entries. Behaviour-identical to always snapshotting.
    snapshot_entry_marginality(c1, c2, num_nodes, any_entry_marginal);

    // Predictive soft-budget check. Sum the product cells we are *guaranteed*
    // to materialize, then bail before any allocation if that exceeds the
    // remaining budget. Per-cell factor is intentionally conservative — pairs
    // (8B) + nodes (8B) + scratch (4-8B) ≈ 24B.
    //
    // Only DENSE-path levels (`cells <= min_grid`) actually allocate a
    // `width1 × width2` product grid. Levels above the sparse threshold take a
    // sparse conjoin that never materializes that grid — their dense width
    // product is a worst-case fiction (a single wide level can be ~width²,
    // e.g. 240070² ≈ 5.8e10 cells ≈ 1.4 TB), so counting it here trips
    // OverBudget on memory we will never allocate. Sparse levels' real cost is
    // the surviving-pair count, which is bounded per-push at the call site
    // (`try_push` / `try_resize` / `budget_reserve`) during the apply — so the
    // soft budget is still enforced for them, just not by this predictor.
    //
    // Do NOT bulk-pre-reserve `pairs`/`nodes`/`ext` here: under `ulimit -v`
    // that consumes VAS the apply never uses (Linux lazy commit bounds RSS,
    // but VAS is what ulimit measures). Budget coverage for those arenas is
    // per-push — every `Vec` growth in the apply body is fallible at its own
    // call site (`try_push` / `try_resize`). The sum itself is accumulated in
    // the width sweep above.
    if let Some(rem) = APPLY_LIMITS.with(|l| l.budget_remaining.get()) {
        let predicted = total_cells.saturating_mul(APPLY_BYTES_PER_CELL);
        if predicted > rem {
            return Err(ApplyError::OverBudget);
        }
    }

    // Pre-scan: check if any internal level's product grid exceeds the sparse
    // threshold. When no level qualifies, skip all sparse infrastructure
    // (product lists, live counts, bump allocator) for zero overhead.
    //
    // Under a restriction the caller supplies the same predicate's value —
    // computed in O(|R|) from the accumulator's already-known `max_width` plus
    // the spine levels, since every off-spine level is `k1 × 1`. It is MATCHED
    // rather than forced false so the sparse / sparse-marg routes fire at
    // exactly the levels the unrestricted apply would fire them at.
    let might_use_sparse = match restrict {
        Some(r) => r.might_use_sparse,
        None => vtree.internal_bottomup()
            .any(|(t, _, _)| c1_widths[t.idx()].saturating_mul(c2_widths[t.idx()]) > min_grid),
    };

    // Streaming-marginal scratch (only when marginalize_targets is provided).
    // Holds lazily computed child counts for streaming-target levels whose
    // children are still explicit (not yet marginalized).
    let mut stream_computed = if marginalize_targets.is_some() {
        pool_take(&SCRATCH_STREAM_COUNTS)
    } else {
        Vec::new()
    };
    if marginalize_targets.is_some() {
        if stream_computed.len() < num_nodes {
            stream_computed.resize_with(num_nodes, || None);
        }
        // Clear any stale entries from prior calls within `num_nodes`.
        for slot in stream_computed[..num_nodes].iter_mut() { *slot = None; }
    }

    // Product lists, live counts, and has_pl are only used when might_use_sparse.
    let mut product_lists = pool_take(&SCRATCH_PRODUCT_LISTS);
    let mut live_counts = pool_take(&SCRATCH_LIVE_COUNTS);
    let mut has_pl = pool_take(&SCRATCH_HAS_PL);

    // Zero `live_counts[0..num_nodes]` unconditionally: 0 is the correct
    // "no output nodes built yet" seed for every level, and pooled reuse can
    // retain stale entries (`resize` only appends/truncates, never clears the
    // live prefix). Stale values would corrupt both the parent density reads
    // and the running `out_nodes_so_far` sum the output-node-cap relies on.
    live_counts.resize(num_nodes, 0);
    if product_lists.len() < num_nodes { product_lists.resize_with(num_nodes, Vec::new); }
    has_pl.resize(num_nodes, false);

    // Both layouts below reset only the entries this apply can read. Restricted
    // mode's reachable index set is `R ∪ children(R)`; unrestricted, it is every
    // level. Stale values outside the set are unreachable by construction, so
    // leaving them is what turns four O(levels) memsets into O(|R|) writes.
    reset_level_tracking(restrict, num_nodes, &mut live_counts, &mut product_lists, &mut has_pl);

    let (node_idx, grid_end) = layout_grids(
        might_use_sparse, restrict, num_nodes, &c1_widths, &c2_widths, &mut grids,
    )?;

    Ok(ApplySetup {
        levels, grids, c1_widths, c2_widths,
        min_grid, sparsity_factor, might_use_sparse,
        stream_computed,
        node_idx, grid_end,
        product_lists, live_counts, has_pl,
        any_entry_marginal,
    })
}
