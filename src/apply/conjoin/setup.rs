//! Apply setup: phases 1-3 of `apply_and_fallible_inner` (width/marginal-entry
//! snapshot, sparse/budget pre-scan, grid/product-list allocation), bundled into
//! `ApplyRun` and produced by `apply_and_setup`. Pure code motion out of
//! `conjoin/mod.rs`; the driver destructures `ApplyRun` back into its locals.
//! Scratch pools, `MARG_ENTRY_*`, `APPLY_BYTES_PER_CELL`, and `APPLY_LIMITS`
//! stay in `mod.rs`/`budget` and are reached via `super::`.

use crate::engine::Engine;
use crate::vtree::VtreeIdx;
use crate::diagram::{self, *};
use crate::value_fold::CountVec;
use crate::engine::ApplyBudget;
use super::{liveness, ApplyError, LevelGrid, APPLY_BYTES_PER_CELL};
use super::grid_arena::GridArena;
use super::sparse::{sparse_config, ProductEntry};
use super::route::{LevelMarg, SparseGate};
use super::plan::ApplyPlan;
use super::stream::stream_marginal_eligible;

/// Bundled result of `apply_and_setup` — the per-apply working state produced
/// before the bottom-up level sweep. (Was a 14-tuple.)
pub(super) struct ApplyRun {
    pub(super) levels: Vec<TddLevel>,
    pub(super) c1_widths: Vec<usize>,
    pub(super) c2_widths: Vec<usize>,
    pub(super) min_grid: usize,
    pub(super) sparsity_factor: u128,
    pub(super) stream_computed: Vec<Option<CountVec<ApplyBudget>>>,
    /// The flat product-grid slab and every level's claim on it. See
    /// [`GridArena`].
    pub(super) arena: GridArena,
    pub(super) product_lists: Vec<Vec<ProductEntry>>,
    pub(super) live_counts: Vec<usize>,
    pub(super) has_pl: Vec<bool>,
    /// True iff some operand level was marginal at apply entry — i.e. iff the
    /// `eng.apply().marg_entry_c1`/`eng.apply().marg_entry_c2` snapshots below were actually filled.
    /// When false they were CLEARED, so every `plan_marg_level` lookup into them
    /// is provably `None`; threading the flag out lets that path skip the four
    /// per-level `RefCell` borrows entirely.
    pub(super) any_entry_marginal: bool,
    /// Weighted mirror of `stream_computed`, empty when not marginalizing.
    pub(super) stream_computed_weights: Vec<Option<Vec<crate::diagram::WeightVal>>>,
    /// `c2_identity[t]` — c2 computes constant-true over subtree `t`, so c1's
    /// nodes pass through unchanged. Lazily accreted, so a false reading only
    /// costs a fallback to the dense grid.
    pub(super) c2_identity: Vec<bool>,
    /// The symmetric flag for c1.
    pub(super) c1_identity: Vec<bool>,
    /// Decode buffers for one cell's pairs, one per operand.
    pub(super) inputs1_scratch: Vec<InputPair>,
    pub(super) inputs2_scratch: Vec<InputPair>,
    /// The four NxM dead-pair pre-filter masks, reused across internal levels.
    pub(super) nxm_masks: liveness::NxmMaskScratch,
    /// Output nodes built so far, kept in step with `live_counts` by
    /// `bump_live_count` so the output-node cap reads it without re-summing.
    pub(super) out_nodes_so_far: u64,
}

/// One internal vtree level's identity: the node, its two children, and both
/// operands' widths at each of the three.
///
/// The widths come from the entry snapshot, not from the levels themselves —
/// an identity fast path steals an operand's level mid-sweep, which zeroes the
/// width the level would report.
#[derive(Clone, Copy)]
pub(super) struct LevelShape {
    pub(super) t: VtreeIdx,
    pub(super) left: VtreeIdx,
    pub(super) right: VtreeIdx,
    pub(super) t_idx: usize,
    pub(super) left_idx: usize,
    pub(super) right_idx: usize,
    /// c1's width at `t`, at `left`, and at `right`.
    pub(super) k1: usize,
    pub(super) k1_left: usize,
    pub(super) k1_right: usize,
    /// c2's, likewise.
    pub(super) k2: usize,
    pub(super) k2_left: usize,
    pub(super) k2_right: usize,
}

impl ApplyRun {
    /// The shape of the level at `t`, read off the entry width snapshot.
    pub(super) fn shape(&self, t: VtreeIdx, left: VtreeIdx, right: VtreeIdx) -> LevelShape {
        let (t_idx, left_idx, right_idx) = (t.idx(), left.idx(), right.idx());
        LevelShape {
            t, left, right,
            t_idx, left_idx, right_idx,
            k1: self.c1_widths[t_idx],
            k1_left: self.c1_widths[left_idx],
            k1_right: self.c1_widths[right_idx],
            k2: self.c2_widths[t_idx],
            k2_left: self.c2_widths[left_idx],
            k2_right: self.c2_widths[right_idx],
        }
    }

    /// This level's marginality, in the two senses [`route_level`] needs.
    ///
    /// `restrict` matters to the `_now` pair only: under a restriction an
    /// off-`R` output level is never copied into the fresh array — the output
    /// array IS the accumulator's, merged at the tail — so the output-level
    /// question has to be asked of `c1` as well. Levels inside `R` are
    /// structural in the accumulator by construction, so the extra disjunct is
    /// inert for them.
    pub(super) fn level_marg(
        &self,
        c1: &Tdd,
        c2: &Tdd,
        shape: LevelShape,
        marginalize_targets: Option<&[bool]>,
        restricted: bool,
    ) -> LevelMarg {
        let LevelShape { t_idx, left_idx, right_idx, .. } = shape;
        let now = |i: usize| {
            self.levels[i].is_marginal() || (restricted && c1.levels[i].is_marginal())
        };
        let any = |i: usize| {
            self.levels[i].is_marginal()
                || c1.levels[i].is_marginal()
                || c2.levels[i].is_marginal()
        };
        LevelMarg {
            left_now: now(left_idx),
            right_now: now(right_idx),
            left_any: any(left_idx),
            right_any: any(right_idx),
            is_target: marginalize_targets.is_some_and(|a| a[t_idx]),
            stream_eligible: stream_marginal_eligible(marginalize_targets, t_idx),
        }
    }

    /// Whether the sparse routes are open at this level, and whether the
    /// children's live products are sparse enough for the scatter walk.
    ///
    /// The density test is exact arithmetic on `u128`: the two maxima are
    /// products of widths and overflow `u64` on wide levels.
    pub(super) fn sparse_gate(&self, shape: LevelShape) -> SparseGate {
        let LevelShape { left_idx, right_idx, k1_left, k2_left, k1_right, k2_right, .. } = shape;
        let max_left = (k1_left * k2_left) as u128;
        let max_right = (k1_right * k2_right) as u128;
        let live_l = self.live_counts[left_idx] as u128;
        let live_r = self.live_counts[right_idx] as u128;
        SparseGate {
            available: self.arena.is_bump(),
            density_wins: max_left > 0
                && max_right > 0
                && self.sparsity_factor * live_l * live_r < max_left * max_right,
            min_grid: self.min_grid,
        }
    }

    /// Hand every pooled buffer back to the engine and return the built levels.
    ///
    /// Heavy buffers are capped at `MAX_LEVEL_ARENA_BYTES` on the way out, so a
    /// single wide conjunction cannot park GiB-scale allocations in the pools.
    pub(super) fn finish(mut self, eng: &Engine, marginalizing: bool) -> Vec<TddLevel> {
        let pool = eng.apply();
        let (slab, grids) = self.arena.into_parts();
        pool.node_idx.put_bounded(slab, MAX_LEVEL_ARENA_BYTES);
        pool.grids.put(grids);
        pool.c2_identity.put(self.c2_identity);
        pool.c1_identity.put(self.c1_identity);
        for pl in &mut self.product_lists {
            crate::engine::pool::release_if_oversized(pl, MAX_LEVEL_ARENA_BYTES);
        }
        pool.product_lists.put(self.product_lists);
        pool.live_counts.put(self.live_counts);
        pool.has_pl.put(self.has_pl);
        pool.c1_widths.put(self.c1_widths);
        pool.c2_widths.put(self.c2_widths);
        pool.inputs1.put_bounded(self.inputs1_scratch, MAX_LEVEL_ARENA_BYTES);
        pool.inputs2.put_bounded(self.inputs2_scratch, MAX_LEVEL_ARENA_BYTES);
        // Same retention rule, applied to the bundle's four fields.
        self.nxm_masks.release_oversized();
        pool.nxm_masks.put(self.nxm_masks);
        if marginalizing {
            pool.stream_counts.put(self.stream_computed);
            pool.stream_weights.put(self.stream_computed_weights);
        }
        self.levels
    }
}

/// Phases 1–3 of `apply_and_fallible_inner`: width/marginal-entry snapshot,
/// sparse/budget pre-scan, grid/product-list allocation.
///
/// Fills the engine's `marg_entry_c1` / `marg_entry_c2` snapshots as a side effect
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
/// All three come out of ONE pass. The cell sum reads exactly the two widths
/// the loop already has in registers over exactly the same range, so folding it
/// in costs nothing and saves a pass. The sparse pre-scan is deliberately NOT
/// fused: it ranges over the cached topo order — reachable internal nodes only
/// — which is a different set.
///
/// The widths must be read before the bottom-up sweep's identity swaps steal
/// levels, which zero `effective_width` and clear `is_marginal`. Under a
/// restriction only `R ∪ children(R)` is ever indexed, so only those levels are
/// visited and `any_entry_marginal` stays false: the snapshot it guards exists
/// to recover an operand child that an identity fast path stole mid-sweep, and
/// a restricted apply takes no fast path.
fn snapshot_widths<P: ApplyPlan>(
    c1: &Tdd,
    c2: &Tdd,
    num_nodes: usize,
    min_grid: usize,
    plan: &P,
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
    let mut ignored = false;
    let tracks_marginal = plan.tracks_entry_marginal();
    for i in plan.touched(num_nodes) {
        let any = if tracks_marginal { &mut any_entry_marginal } else { &mut ignored };
        width_at(i, &mut total_cells, any);
    }
    (total_cells, any_entry_marginal)
}

/// Record which of each operand's levels are marginal at entry, for the
/// pass-through carrier selector to consult after an identity swap has stolen a
/// level (which clears the flag on the level itself).
///
/// With nothing marginal at entry the snapshot is all-false and the selector's
/// default answers identically, so the two passes are skipped — but the
/// snapshots are still cleared, so a previous marginal conjunction on this
/// engine leaves nothing stale behind.
fn snapshot_entry_marginality(
    eng: &Engine,
    c1: &Tdd,
    c2: &Tdd,
    num_nodes: usize,
    any_entry_marginal: bool,
) {
    let mut e1 = eng.apply().marg_entry_c1.borrow_mut();
    let mut e2 = eng.apply().marg_entry_c2.borrow_mut();
    e1.clear();
    e2.clear();
    if any_entry_marginal {
        for i in 0..num_nodes {
            e1.push(c1.levels[i].is_marginal());
            e2.push(c2.levels[i].is_marginal());
        }
    }
}

/// Clear the per-level sparse bookkeeping this apply can read.
///
/// A restricted apply's reachable set is `R ∪ children(R)`; unrestricted, it is
/// every level. Entries outside the set are unreachable by construction, so
/// leaving them stale is what turns whole-array memsets into `O(|R|)` writes.
fn reset_level_tracking<P: ApplyPlan>(
    plan: &P,
    num_nodes: usize,
    live_counts: &mut [usize],
    product_lists: &mut [Vec<ProductEntry>],
    has_pl: &mut [bool],
) {
    for i in plan.touched(num_nodes) {
        live_counts[i] = 0;
        product_lists[i].clear();
        has_pl[i] = false;
    }
}

/// Build the product-grid arena in the shape this apply needs.
///
/// With any sparse level possible the arena bumps: every level starts
/// ungridded and claims space when it is reached. Otherwise every level's base
/// is computed up front and the slab is sized once.
fn layout_grids<P: ApplyPlan>(
    eng: &Engine,
    might_use_sparse: bool,
    plan: &P,
    num_nodes: usize,
    c1_widths: &[usize],
    c2_widths: &[usize],
    grids: Vec<LevelGrid>,
) -> Result<GridArena, ApplyError> {
    let cells = eng.apply().node_idx.take();
    if might_use_sparse {
        Ok(GridArena::bump(cells, grids, plan.touched(num_nodes).chain(std::iter::once(num_nodes))))
    } else {
        GridArena::preplanned(
            eng, cells, grids,
            plan.touched(num_nodes)
                .map(|i| (i, c1_widths[i] * c2_widths[i]))
                .chain(std::iter::once((num_nodes, 0))),
        )
    }
}

/// Take one of the streaming column caches and clear its first `num_nodes`
/// slots, or leave it empty when nothing is being marginalized — the streaming
/// routes are then unreachable and nothing reads it.
///
/// The cache holds lazily computed child counts for streaming-target levels
/// whose children are still explicit. There are two of them, integer and
/// weighted, differing only in the value they hold.
fn take_stream_cache<T>(
    pool: &crate::engine::pool::Pool<Vec<Option<T>>>,
    num_nodes: usize,
    marginalizing: bool,
) -> Vec<Option<T>> {
    if !marginalizing {
        return Vec::new();
    }
    let mut cache = pool.take();
    if cache.len() < num_nodes {
        cache.resize_with(num_nodes, || None);
    }
    for slot in cache[..num_nodes].iter_mut() {
        *slot = None;
    }
    cache
}

/// Refuse before allocating anything if the cells this apply is *guaranteed* to
/// materialize already exceed the remaining soft budget.
///
/// `total_cells` counts DENSE-path levels only. A level above the sparse
/// threshold takes a conjoin that never materializes its grid, so its dense
/// width product is a worst-case fiction — a single wide level can be ~width²
/// (240070² ≈ 5.8e10 cells ≈ 1.4 TB) — and counting it here would refuse over
/// memory that is never allocated. A sparse level's real cost is its surviving
/// pair count, which the soft budget still sees, just per-push at each call
/// site rather than through this predictor.
///
/// The per-cell byte factor is deliberately conservative: pairs (8B) + nodes
/// (8B) + scratch (4–8B) ≈ 24B.
///
/// The arenas themselves are deliberately NOT bulk-reserved anywhere near
/// here. Under `ulimit -v` that consumes address space the apply never uses —
/// Linux's lazy commit bounds RSS, but the limit measures VAS — and every
/// `Vec` growth in the apply body is fallible at its own call site anyway.
///
/// # Errors
///
/// [`ApplyError::OverBudget`] when the prediction does not fit.
fn preflight_dense_budget(lim: &crate::engine::Limits, total_cells: u64) -> Result<(), ApplyError> {
    if let Some(rem) = lim.budget()
        && total_cells.saturating_mul(APPLY_BYTES_PER_CELL) > rem {
            return Err(ApplyError::OverBudget);
        }
    Ok(())
}

pub(super) fn apply_and_setup<P: ApplyPlan>(
    eng: &Engine,
    c1: &mut Tdd,
    c2: &mut Tdd,
    vtree: &crate::vtree::Vtree,
    num_nodes: usize,
    marginalize_targets: Option<&[bool]>,
    plan: &P,
) -> Result<ApplyRun, ApplyError> {
    let lim = eng.limits();
    let levels: Vec<TddLevel> = diagram::take_levels(eng, num_nodes);

    let mut grids: Vec<LevelGrid> = eng.apply().grids.take();
    if grids.len() < num_nodes + 1 {
        grids.resize(num_nodes + 1, LevelGrid::Sparse);
    }

    let mut c1_widths = eng.apply().c1_widths.take();
    let mut c2_widths = eng.apply().c2_widths.take();
    if c1_widths.len() < num_nodes { c1_widths.resize(num_nodes, 0); }
    if c2_widths.len() < num_nodes { c2_widths.resize(num_nodes, 0); }
    let cfg = sparse_config();
    let min_grid = cfg.min_grid;
    let sparsity_factor = cfg.sparsity_factor;
    let (total_cells, any_entry_marginal) = snapshot_widths(
        c1, c2, num_nodes, min_grid, plan, &mut c1_widths, &mut c2_widths,
    );

    snapshot_entry_marginality(eng, c1, c2, num_nodes, any_entry_marginal);

    preflight_dense_budget(lim, total_cells)?;

    // With no level over the threshold, all the sparse infrastructure — product
    // lists, live counts, bump allocator — is skipped outright.
    let might_use_sparse = plan.might_use_sparse(vtree, &c1_widths, &c2_widths, min_grid);

    // Streaming-marginal scratch (only when marginalize_targets is provided).
    // Holds lazily computed child counts for streaming-target levels whose
    // children are still explicit (not yet marginalized).
    let stream_computed =
        take_stream_cache(&eng.apply().stream_counts, num_nodes, marginalize_targets.is_some());

    // Product lists, live counts, and has_pl are only used when might_use_sparse.
    let mut product_lists = eng.apply().product_lists.take();
    let mut live_counts = eng.apply().live_counts.take();
    let mut has_pl = eng.apply().has_pl.take();

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
    reset_level_tracking(plan, num_nodes, &mut live_counts, &mut product_lists, &mut has_pl);

    let arena = layout_grids(
        eng,
        might_use_sparse, plan, num_nodes, &c1_widths, &c2_widths, grids,
    )?;

    // Weighted streaming scratch: the concrete weighted mirror of
    // `stream_computed`, with the same take/clear/return discipline, so pooled
    // reuse cannot leak a stale weight into a later apply.
    let stream_computed_weights =
        take_stream_cache(&eng.apply().stream_weights, num_nodes, marginalize_targets.is_some());

    let mut inputs1_scratch: Vec<InputPair> = eng.apply().inputs1.take();
    let mut inputs2_scratch: Vec<InputPair> = eng.apply().inputs2.take();
    inputs1_scratch.clear();
    inputs2_scratch.clear();

    Ok(ApplyRun {
        levels, c1_widths, c2_widths,
        min_grid, sparsity_factor,
        stream_computed,
        arena,
        product_lists, live_counts, has_pl,
        any_entry_marginal,
        stream_computed_weights,
        c2_identity: eng.apply().c2_identity.take(),
        c1_identity: eng.apply().c1_identity.take(),
        inputs1_scratch,
        inputs2_scratch,
        nxm_masks: eng.apply().nxm_masks.take(),
        out_nodes_so_far: 0,
    })
}
