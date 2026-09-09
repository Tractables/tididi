use super::*;
use crate::diagram::SideView;
use crate::engine::Engine;

/// A2 regression guard: `CollectSink` pushes must charge the apply soft
/// budget (`try_push`), like the emit walk's `try_push_pair_into`. The
/// pre-fix collectors used plain `Vec::push`, so a wide streaming cell's
/// scratch grew invisibly to the budget — and an allocator failure aborted
/// the process instead of degrading to `Err(OverBudget)`.
///
/// Installs a tiny per-thread soft budget, then feeds pairs through the
/// sink until the scratch's capacity growth must exceed it. On the
/// unbudgeted sink every push returns `Ok` and the assert fails.
#[test]
fn collect_sink_pushes_charge_the_soft_budget() {
    // 64 bytes ≈ 8 `InputPair`s of capacity; 1024 pushes must trip it.
    let eng = Engine::new();
    let lim = eng.limits();
    lim.set_budget(Some(64));
    let mut out: Vec<InputPair> = Vec::new();
    let mut sink = CollectSink { out: &mut out };
    let mut result = Ok(());
    for _ in 0..1024 {
        result = sink.pair(&eng, 1, 2);
        if result.is_err() { break; }
    }
    assert!(
        matches!(result, Err(ApplyError::OverBudget)),
        "CollectSink pushes bypass the apply soft budget"
    );
}

fn pair(l: u32, r: u32) -> InputPair {
    InputPair { left: NodeIdx(l), right: NodeIdx(r) }
}

/// Fixture level exercising every decode shape the arena must mirror:
/// inline (1-pair) and multi (≥2-pair) nodes, bit-30 inline-count tags and
/// bit-31 dead sentinels on the marg side, plus a leaf/dead cell.
fn marg_shaped_level() -> (TddLevel, usize) {
    use crate::diagram::ValueRef;
    let mut lvl = TddLevel::new();
    lvl.push_internal_node(&[pair(3, ValueRef::inline_raw(7).expect("test inline count must fit inline encoding"))]);
    lvl.push_internal_node(&[
        pair(0, 1),
        pair(1, (1 << 31) | 5),
        pair(2, ValueRef::inline_raw(2).expect("test inline count must fit inline encoding")),
    ]);
    lvl.push_internal_node(&[pair(5, 0), pair(6, 2)]);
    let k2 = lvl.nodes.len();
    (lvl, k2)
}

/// Equivalence guard, marg masks: the per-level column table must hand back,
/// for every column, exactly the slice the per-cell `pairs_view_decoded`
/// derivation produces (same masks, same order).
#[test]
fn c2_columns_match_per_cell_decode() {
    let eng = Engine::new();
    
    let (lvl, k2) = marg_shaped_level();
    let (lm, rm) = (SideView::structural(), SideView::valued()); // right child marginal
    let cols = C2Columns::build(&eng, &lvl, k2, lm, rm).expect("a valued side must build");
    let mut scratch: Vec<InputPair> = Vec::new();
    for j in 0..k2 {
        let want = lvl.pairs_view_decoded(j, &mut scratch, lm, rm).to_vec();
        assert_eq!(cols.get(j), &want[..], "column {j} diverges from per-cell decode");
    }
}

/// Equivalence guard, identity masks: the table must hand back the SAME
/// zero-copy borrow the per-cell fast path returns — same contents *and* the
/// same backing storage (nothing may be materialized that used to be
/// borrowed), across both the inline and the multi-pair node encodings.
#[test]
fn c2_columns_borrow_identity_mask_storage() {
    let eng = Engine::new();
    let (lvl, k2) = marg_shaped_level();
    let cols = C2Columns::build(&eng, &lvl, k2, SideView::structural(), SideView::structural())
        .expect("identity masks must build a borrowing table");
    let mut scratch: Vec<InputPair> = Vec::new();
    for j in 0..k2 {
        let want = lvl.pairs_view_decoded(j, &mut scratch, SideView::structural(), SideView::structural());
        let got = cols.get(j);
        assert_eq!(got, want, "column {j} diverges from the per-cell view");
        assert_eq!(
            got.as_ptr(), want.as_ptr(),
            "column {j} must be the same borrow, not a copy",
        );
    }
}

/// A marginal-encoded c2 level stores count payloads, not pair structure —
/// the walkers never read its pairs, so the table must decline rather than
/// resolve columns out of it.
#[test]
fn c2_columns_skip_marginal_levels() {
    let eng = Engine::new();
    let (mut lvl, k2) = marg_shaped_level();
    lvl.set_counts_state(vec![0u128; k2], None);
    assert!(C2Columns::build(&eng, &lvl, k2, SideView::structural(), SideView::structural()).is_none());
}

/// Budget guard: the marg-mask decode arena charges the apply soft budget
/// while alive and releases its exact charge on drop (it is a per-level
/// transient — `ApplyLimits::budget_in_flight` is otherwise monotone within an
/// apply, so a leak here would permanently eat headroom). Over-budget builds
/// fall back to `None` without retaining any charge. The identity-mask table
/// borrows, so it charges nothing.
#[test]
fn c2_columns_charge_and_release_the_soft_budget() {
    
    let (lvl, k2) = marg_shaped_level();

    {
        let eng = Engine::new();
        let lim = eng.limits();
    lim.set_budget(Some(1 << 20));
        let h0 = lim.budget_headroom().expect("budget installed");
        let cols = C2Columns::build(&eng, &lvl, k2, SideView::structural(), SideView::valued())
            .expect("within budget");
        let h_alive = lim.budget_headroom().unwrap();
        assert!(h_alive < h0, "arena reservation must charge the soft budget");
        drop(cols);
        assert_eq!(
            lim.budget_headroom().unwrap(),
            h0,
            "arena drop must release exactly its charge"
        );
    }

    // Identity masks borrow c2's storage — no arena, so no charge.
    {
        let eng = Engine::new();
        let lim = eng.limits();
    lim.set_budget(Some(1 << 20));
        let h0 = lim.budget_headroom().expect("budget installed");
        let cols = C2Columns::build(&eng, &lvl, k2, SideView::structural(), SideView::structural()).expect("within budget");
        assert_eq!(
            lim.budget_headroom().unwrap(),
            h0,
            "a borrowing table must not charge the soft budget",
        );
        drop(cols);
    }

    // Over-budget: build declines, nothing stays charged.
    {
        let eng = Engine::new();
        let lim = eng.limits();
    lim.set_budget(Some(8));
        let h1 = lim.budget_headroom().unwrap();
        assert!(C2Columns::build(&eng, &lvl, k2, SideView::structural(), SideView::valued()).is_none());
        assert_eq!(
            lim.budget_headroom().unwrap(),
            h1,
            "declined build must not retain a charge"
        );
    }
}

/// A2 regression: the pair-collecting sink must honor the soft memory
/// budget — `process_cell` with a `CollectSink` returns `Err(OverBudget)`
/// when its per-cell pair collection would exceed the installed budget,
/// rather than pushing unbudgeted. On unfixed `main` the (pre-unification)
/// collector used infallible `out.push`, growing `cell_pairs` with zero
/// accounting (→ process abort under a memory cap); every push now routes
/// through `try_push` like the rest of the module.
#[test]
fn collect_sink_respects_soft_budget() {
    let eng = Engine::new();
    let lim = eng.limits();
    use crate::diagram::{InputPair, NodeIdx, TddLevel, TddNodeData};
    use super::{process_cell, CellCtx, CollectSink, ApplyError};
    use crate::apply::conjoin::child_lookup::ChildLookup;

    // Always resolves children to a live (non-DEAD) node, so every
    // (p1, p2) combination emits one pair.
    struct AliveLookup;
    impl ChildLookup for AliveLookup {
        fn get(&self, _node_idx: &[u32], _row: u32, _col: u32) -> u32 { 1 }
    }

    // c2 level: a single inline node → exactly one decoded pair for j = 0,
    // putting an N-pair inputs1 into the N×1 arm.
    let mut c2 = TddLevel::new();
    c2.nodes.push(TddNodeData::inline(InputPair {
        left: NodeIdx(2),
        right: NodeIdx(3),
    }));

    let ctx = CellCtx {
        t_base: 0, k2: 1,
        left_base: 0, right_base: 0,
        k2_left: 1, k2_right: 1,
        left_passthrough: false, right_passthrough: false,
        left_pt_c1: false, right_pt_c1: false,
        nxm: false,
        left_view: SideView::structural(), right_view: SideView::structural(),
        live_left_cols: &[], reach_c2_left: &[],
        live_right_cols: &[], reach_c2_right: &[],
        c2_cols: None,
    };
    let mut scratch: Vec<InputPair> = Vec::new();
    let mut node_idx: Vec<u32> = Vec::new();

    // Control: no budget installed → the collector completes and emits one
    // pair per left input.
    lim.reset_meters();
    let small: Vec<InputPair> =
        (0..8).map(|_| InputPair { left: NodeIdx(2), right: NodeIdx(3) }).collect();
    let mut out: Vec<InputPair> = Vec::new();
    let ok = {
        let eng = Engine::new();
        process_cell::<_, _, _>(
            &eng,
            0, 0, &small, 0, 0, &ctx, &c2, &mut scratch, &mut node_idx,
            &AliveLookup, &AliveLookup, &mut CollectSink { out: &mut out },
            &mut crate::engine::PollGate::new(u64::MAX),
        )
    };
    assert!(ok.is_ok(), "no budget: collector should complete, got {:?}", ok.err());
    assert_eq!(out.len(), small.len(), "no budget: every combo emits one pair");

    // Tiny budget → the collector's fallible push trips OverBudget instead
    // of growing `out` without accounting.
    lim.reset_meters();
    let big: Vec<InputPair> =
        (0..8192).map(|_| InputPair { left: NodeIdx(2), right: NodeIdx(3) }).collect();
    let mut out: Vec<InputPair> = Vec::new();
    let res = {
        let eng = Engine::new();
        let lim = eng.limits();
    lim.set_budget(Some(4096));
        process_cell::<_, _, _>(
            &eng,
            0, 0, &big, 0, 0, &ctx, &c2, &mut scratch, &mut node_idx,
            &AliveLookup, &AliveLookup, &mut CollectSink { out: &mut out },
            &mut crate::engine::PollGate::new(u64::MAX),
        )
    };
    lim.reset_meters();
    assert_eq!(
        res.err(), Some(ApplyError::OverBudget),
        "tiny budget: collector must bail OverBudget instead of pushing unbudgeted",
    );
}

/// The work clock must count the PAIRS a level walked, not the cells.
///
/// `StopAt::Work` is a public stop axis and the only reproducible one, so what
/// the clock counts is observable. A level of N×1 cells walks `k1 * n` pairs
/// through `n` cells; the clock has to reflect the pairs. The pre-fix kernel
/// gave every one-sided cell a fresh `PollGate` of its own with a stride wider
/// than the cell, so the gate never came due and the cell charged nothing —
/// the clock reported the cell count and a level could walk hundreds of
/// millions of pairs while the work stop sat still.
///
/// 256 rows of 1024 pairs against a single-pair c2 node: four strides' worth of
/// pairs through 256 N×1 cells. On unfixed `main` the clock reads 256.
#[test]
fn the_work_clock_counts_the_pairs_a_level_walks_not_its_cells() {
    use crate::apply::conjoin::child_lookup::ChildLookup;
    use crate::diagram::{InputPair, NodeIdx, TddLevel};
    use super::{CellAction, CellArgs, CellCtx, CollectSink, process_cell, run_level_rows};

    const K1: usize = 256;
    const PAIRS_PER_ROW: usize = 1024;
    let expected_pairs = (K1 * PAIRS_PER_ROW) as u64;

    // Every child ref resolves live, so no cull short-circuits the walk.
    struct AliveLookup;
    impl ChildLookup for AliveLookup {
        fn get(&self, _node_idx: &[u32], _row: u32, _col: u32) -> u32 { 1 }
    }

    // The pair-collecting sink, driven through the shared row loop so the
    // level's residual charge is flushed the way the real routes flush it.
    struct Collect<'a> { out: &'a mut Vec<InputPair> }
    impl<L: ChildLookup, R: ChildLookup> CellAction<L, R> for Collect<'_> {
        const ASSERT_INTERNAL: bool = false;
        const DENSE_SLAB: bool = true;
        #[inline(always)]
        fn grid_row(&self, i: usize) -> usize { i }
        #[inline(always)]
        fn cell(&mut self, eng: &Engine, a: CellArgs<'_, '_, L, R>) -> Result<(), ApplyError> {
            process_cell::<_, _, _>(
                eng,
                a.j, a.row_base, a.inputs1, a.left_alive_mask, a.right_alive_mask,
                a.ctx, a.c2_level_t, a.inputs2_scratch, a.node_idx, a.left, a.right,
                &mut CollectSink { out: &mut *self.out },
                a.gate,
            )
        }
    }

    let pair = InputPair { left: NodeIdx(0), right: NodeIdx(0) };
    let mut c1 = TddLevel::new();
    for _ in 0..K1 {
        c1.push_internal_node(&vec![pair; PAIRS_PER_ROW]);
    }
    let mut c2 = TddLevel::new();
    c2.push_internal_node(&[pair]);

    let ctx = CellCtx {
        t_base: 0, k2: 1,
        left_base: 0, right_base: 0,
        k2_left: 1, k2_right: 1,
        left_passthrough: false, right_passthrough: false,
        left_pt_c1: false, right_pt_c1: false,
        nxm: false,
        left_view: SideView::structural(), right_view: SideView::structural(),
        live_left_cols: &[], reach_c2_left: &[],
        live_right_cols: &[], reach_c2_right: &[],
        c2_cols: None,
    };

    let eng = Engine::new();
    eng.limits().reset_meters();
    let mut inputs1_scratch: Vec<InputPair> = Vec::new();
    let mut inputs2_scratch: Vec<InputPair> = Vec::new();
    let mut node_idx: Vec<u32> = vec![0; K1];
    let mut out: Vec<InputPair> = Vec::new();
    run_level_rows::<true, _, _, _>(
        &eng, K1, &c1, &c2, &ctx,
        &mut inputs1_scratch, &mut inputs2_scratch, &mut node_idx,
        &AliveLookup, &AliveLookup, &mut Collect { out: &mut out },
    )
    .expect("nothing is armed, so the level completes");

    assert_eq!(out.len(), K1 * PAIRS_PER_ROW, "every (p1, p2) combination emits one pair");
    let work = eng.limits().work_units();
    assert!(
        work.abs_diff(expected_pairs) < crate::apply::conjoin::budget::DENSE_CELL_POLL_STRIDE,
        "work clock read {work}, expected the {expected_pairs} pairs walked within one stride",
    );
}
