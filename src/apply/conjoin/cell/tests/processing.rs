use super::*;
use crate::diagram::ChildDecoder;
use crate::Engine;
use crate::test_helpers::pair;

/// `CollectSink` pushes must charge the apply soft budget (`try_push`), like
/// the emit walk's `try_push_pair_into`: an uncharged push lets a wide
/// streaming cell's scratch grow invisibly to the budget, so an allocator
/// failure aborts the process instead of degrading to `Err(OverBudget)`.
///
/// Installs a tiny per-thread soft budget, then feeds pairs through the
/// sink until the scratch's capacity growth must exceed it. On the
/// unbudgeted sink every push returns `Ok` and the assert fails.
#[test]
fn collect_sink_pushes_charge_the_soft_budget() {
    // 64 bytes ≈ 8 `ChildPair`s of capacity; 1024 pushes must trip it.
    let eng = Engine::new();
    let lim = eng.limits();
    lim.set_budget(Some(64));
    let mut out: Vec<ChildPair> = Vec::new();
    let mut sink = CollectSink { out: &mut out };
    let mut result = Ok(());
    for _ in 0..1024 {
        result = sink.pair(&eng, 1, 2);
        if result.is_err() { break; }
    }
    assert!(
        matches!(result, Err(OperationError::OverBudget)),
        "CollectSink pushes bypass the apply soft budget"
    );
}

/// Fixture level exercising every decode shape the arena must mirror:
/// inline (1-pair) and multi (≥2-pair) nodes, bit-30 inline-count tags and
/// bit-31 dead sentinels on the marginal side, plus a leaf/dead cell.
fn marginal_shaped_level() -> (TddLevel, usize) {
    use crate::diagram::ValueRef;
    let mut lvl = TddLevel::new();
    lvl.push_internal_node(&[pair(3, ValueRef::inline_raw(7).expect("test inline count must fit inline encoding"))]);
    lvl.push_internal_node(&[
        pair(0, 1),
        pair(1, (1 << 31) | 5),
        pair(2, ValueRef::inline_raw(2).expect("test inline count must fit inline encoding")),
    ]);
    lvl.push_internal_node(&[pair(5, 0), pair(6, 2)]);
    let right_width = lvl.nodes.len();
    (lvl, right_width)
}

/// Equivalence guard, marginal masks: the per-level column table must hand back,
/// for every column, exactly the slice the per-cell `pairs_view_decoded`
/// derivation produces (same masks, same order).
#[test]
fn columns_match_per_cell_decode() {
    let eng = Engine::new();
    
    let (lvl, right_width) = marginal_shaped_level();
    let (lm, rm) = (ChildDecoder::structural(), ChildDecoder::marginal()); // right child marginal
    let cols = RightColumns::build(&eng, &lvl, right_width, lm, rm).expect("a marginal side must build");
    let mut scratch: Vec<ChildPair> = Vec::new();
    for j in 0..right_width {
        let want = lvl.pairs_view_decoded(j, &mut scratch, lm, rm).to_vec();
        assert_eq!(cols.get(j), &want[..], "column {j} diverges from per-cell decode");
    }
}

/// Equivalence guard, identity masks: the table must hand back the same
/// zero-copy borrow the per-cell fast path returns — same contents *and* the
/// same backing storage (nothing borrowed may be materialized), across both
/// the inline and the multi-pair node encodings.
#[test]
fn columns_borrow_identity_mask_storage() {
    let eng = Engine::new();
    let (lvl, right_width) = marginal_shaped_level();
    let cols = RightColumns::build(&eng, &lvl, right_width, ChildDecoder::structural(), ChildDecoder::structural())
        .expect("identity masks must build a borrowing table");
    let mut scratch: Vec<ChildPair> = Vec::new();
    for j in 0..right_width {
        let want = lvl.pairs_view_decoded(j, &mut scratch, ChildDecoder::structural(), ChildDecoder::structural());
        let got = cols.get(j);
        assert_eq!(got, want, "column {j} diverges from the per-cell view");
        assert_eq!(
            got.as_ptr(), want.as_ptr(),
            "column {j} must be the same borrow, not a copy",
        );
    }
}

/// A marginal-encoded g level stores count payloads, not pair structure —
/// the walkers never read its pairs, so the table must decline rather than
/// resolve columns out of it.
#[test]
fn columns_skip_marginal_levels() {
    let eng = Engine::new();
    let (mut lvl, right_width) = marginal_shaped_level();
    lvl.set_counts_state(vec![0u128; right_width], None);
    assert!(RightColumns::build(&eng, &lvl, right_width, ChildDecoder::structural(), ChildDecoder::structural()).is_none());
}

/// Budget guard: the marginal-mask decode arena charges the apply soft budget
/// while alive and releases its exact charge on drop (it is a per-level
/// transient — `Limits`'s in-flight byte count is otherwise monotone within an
/// apply, so a leak here would permanently eat headroom). Over-budget builds
/// fall back to `None` without retaining any charge. The identity-mask table
/// borrows, so it charges nothing.
#[test]
fn columns_charge_and_release_the_soft_budget() {
    
    let (lvl, right_width) = marginal_shaped_level();

    {
        let eng = Engine::new();
        let lim = eng.limits();
    lim.set_budget(Some(1 << 20));
        let h0 = lim.budget_headroom().expect("budget installed");
        let cols = RightColumns::build(&eng, &lvl, right_width, ChildDecoder::structural(), ChildDecoder::marginal())
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

    // Identity masks borrow g's storage — no arena, so no charge.
    {
        let eng = Engine::new();
        let lim = eng.limits();
    lim.set_budget(Some(1 << 20));
        let h0 = lim.budget_headroom().expect("budget installed");
        let cols = RightColumns::build(&eng, &lvl, right_width, ChildDecoder::structural(), ChildDecoder::structural()).expect("within budget");
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
        assert!(RightColumns::build(&eng, &lvl, right_width, ChildDecoder::structural(), ChildDecoder::marginal()).is_none());
        assert_eq!(
            lim.budget_headroom().unwrap(),
            h1,
            "declined build must not retain a charge"
        );
    }
}

/// The pair-collecting sink must honor the soft memory
/// budget — `process_cell` with a `CollectSink` returns `Err(OverBudget)`
/// when its per-cell pair collection would exceed the installed budget,
/// rather than pushing unbudgeted: an infallible `out.push` would grow
/// `cell_pairs` with zero accounting and abort the process under a memory
/// cap, so every push routes
/// through `try_push` like the rest of the module.
#[test]
fn collect_sink_respects_soft_budget() {
    let eng = Engine::new();
    let lim = eng.limits();
    use crate::diagram::{ChildPair, NodeIdx, TddLevel, EncodedNode};
    use super::{process_cell, CellCtx, CollectSink, OperationError};
    use crate::apply::conjoin::child_lookup::ChildLookup;

    // Always resolves children to a live (non-`NO_PRODUCT`) node, so every
    // (p1, p2) combination emits one pair.
    struct AliveLookup;
    impl ChildLookup for AliveLookup {
        type Row = ();
        fn row(&self, _row: u32) {}
        fn get_in_row(&self, _node_idx: &[u32], _row: (), _col: u32) -> u32 { 1 }
    }

    // g level: a single inline node → exactly one decoded pair for j = 0,
    // putting an N-pair inputs1 into the N×1 arm.
    let mut g = TddLevel::new();
    g.nodes.push(EncodedNode::inline(ChildPair::new(NodeIdx(2), NodeIdx(3))));

    let side = ChildPlan {
        plan: SidePlan { carrier: None, view: ChildDecoder::structural() },
        base: 0, stride: 1, live_cols: &[], reach: &[],
    };
    let ctx = CellCtx {
        output_grid_base: 0, right_width: 1,
        both_multi_pair: false,
        sides: Sides { left: side, right: side },
        right_cols: None,
    };
    let mut scratch: Vec<ChildPair> = Vec::new();
    let mut node_idx: Vec<u32> = Vec::new();

    // Control: no budget installed → the collector completes and emits one
    // pair per left input.
    lim.reset_meters();
    let small: Vec<ChildPair> =
        (0..8).map(|_| ChildPair::new(NodeIdx(2), NodeIdx(3))).collect();
    let mut out: Vec<ChildPair> = Vec::new();
    let ok = {
        let eng = Engine::new();
        process_cell::<_, _, _>(
            &eng,
            0, 0, &small, &ctx, &g, &mut scratch, &mut node_idx,
            &AliveLookup, &AliveLookup, &mut CollectSink { out: &mut out },
            &mut eng.limits().gate_with(u64::MAX),
        )
    };
    assert!(ok.is_ok(), "no budget: collector should complete, got {:?}", ok.err());
    assert_eq!(out.len(), small.len(), "no budget: every combo emits one pair");

    // Tiny budget → the collector's fallible push trips OverBudget instead
    // of growing `out` without accounting.
    lim.reset_meters();
    let big: Vec<ChildPair> =
        (0..8192).map(|_| ChildPair::new(NodeIdx(2), NodeIdx(3))).collect();
    let mut out: Vec<ChildPair> = Vec::new();
    let res = {
        let eng = Engine::new();
        let lim = eng.limits();
    lim.set_budget(Some(4096));
        process_cell::<_, _, _>(
            &eng,
            0, 0, &big, &ctx, &g, &mut scratch, &mut node_idx,
            &AliveLookup, &AliveLookup, &mut CollectSink { out: &mut out },
            &mut eng.limits().gate_with(u64::MAX),
        )
    };
    lim.reset_meters();
    assert_eq!(
        res.err(), Some(OperationError::OverBudget),
        "tiny budget: collector must bail OverBudget instead of pushing unbudgeted",
    );
}

/// The work clock must count the PAIRS a level walked, not the cells.
///
/// `StopAt::WorkUnits` is a public stop axis and the only reproducible one, so what
/// the clock counts is observable. A level of N×1 cells walks `left_width * n` pairs
/// through `n` cells; the clock has to reflect the pairs. Giving each one-sided
/// cell its own `PollGate` with a stride wider than the cell would leave the
/// gate never due and the cell charging nothing, so a level could walk
/// arbitrarily many pairs while the work stop sat still.
///
/// 256 rows of 1024 pairs against a single-pair g node: four strides' worth of
/// pairs through 256 N×1 cells. On unfixed `main` the clock reads 256.
#[test]
fn the_work_clock_counts_the_pairs_a_level_walks_not_its_cells() {
    use crate::apply::conjoin::child_lookup::ChildLookup;
    use crate::diagram::{ChildPair, NodeIdx, TddLevel};
    use super::{CellAction, CellArgs, CellCtx, CollectSink, process_cell, run_level_rows};

    const K1: usize = 256;
    const PAIRS_PER_ROW: usize = 1024;
    let expected_pairs = (K1 * PAIRS_PER_ROW) as u64;

    // Every child ref resolves live, so no cull short-circuits the walk.
    struct AliveLookup;
    impl ChildLookup for AliveLookup {
        type Row = ();
        fn row(&self, _row: u32) {}
        fn get_in_row(&self, _node_idx: &[u32], _row: (), _col: u32) -> u32 { 1 }
    }

    // The pair-collecting sink, driven through the shared row loop so the
    // level's residual charge is flushed the way the real routes flush it.
    struct Collect<'a> { out: &'a mut Vec<ChildPair> }
    impl<L: ChildLookup, R: ChildLookup> CellAction<L, R> for Collect<'_> {
        const ASSERT_INTERNAL: bool = false;
        const DENSE_SLAB: bool = true;
        #[inline(always)]
        fn grid_row(&self, i: usize) -> usize { i }
        #[inline(always)]
        fn cell(&mut self, eng: &Engine, a: CellArgs<'_, '_, L, R>) -> Result<(), OperationError> {
            process_cell::<_, _, _>(
                eng,
                a.j, a.row_base, a.inputs1,
                a.ctx, a.right_level_t, a.inputs2_scratch, a.node_idx, a.left, a.right,
                &mut CollectSink { out: &mut *self.out },
                a.gate,
            )
        }
    }

    let pair = ChildPair::new(NodeIdx(0), NodeIdx(0));
    let mut f = TddLevel::new();
    for _ in 0..K1 {
        f.push_internal_node(&vec![pair; PAIRS_PER_ROW]);
    }
    let mut g = TddLevel::new();
    g.push_internal_node(&[pair]);

    let side = ChildPlan {
        plan: SidePlan { carrier: None, view: ChildDecoder::structural() },
        base: 0, stride: 1, live_cols: &[], reach: &[],
    };
    let ctx = CellCtx {
        output_grid_base: 0, right_width: 1,
        both_multi_pair: false,
        sides: Sides { left: side, right: side },
        right_cols: None,
    };

    let eng = Engine::new();
    eng.limits().reset_meters();
    let mut inputs1_scratch: Vec<ChildPair> = Vec::new();
    let mut inputs2_scratch: Vec<ChildPair> = Vec::new();
    let mut node_idx: Vec<u32> = vec![0; K1];
    let mut out: Vec<ChildPair> = Vec::new();
    run_level_rows::<true, _, _, _>(
        &eng,
        // The collecting action never streams, so the child levels stand in for
        // themselves — nothing on this route reads them.
        RowLoop { f_level: &f, g_level: &g, children: Sides { left: &f, right: &g }, ctx: &ctx, f_width: K1 },
        RowScratch { inputs1: &mut inputs1_scratch, inputs2: &mut inputs2_scratch, node_idx: &mut node_idx },
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

/// The kernel's per-cell fallback must hand the walk exactly the columns the
/// hoisted table would have.
///
/// The table declines when the soft budget refuses its decode arena, and the
/// cell then re-derives column `j` itself. Both routes have to produce the same
/// pairs — marginal masks included, which is where the two derivations differ
/// most (the table decodes into an arena, the fallback into a per-cell
/// scratch).
#[test]
fn the_per_cell_column_fallback_walks_what_the_table_would_have() {
    use super::{process_cell, CellCtx, CollectSink};
    use crate::apply::conjoin::child_lookup::ChildLookup;

    // A child resolution that depends on both refs, so an emitted pair carries
    // which column entries the walk actually read.
    struct RefLookup;
    impl ChildLookup for RefLookup {
        type Row = u32;
        fn row(&self, row: u32) -> u32 { row }
        fn get_in_row(&self, _node_idx: &[u32], row: u32, col: u32) -> u32 {
            row.wrapping_mul(31).wrapping_add(col) & 0x00ff_ffff
        }
    }

    let eng = Engine::new();
    let (lvl, right_width) = marginal_shaped_level();
    let (lm, rm) = (ChildDecoder::structural(), ChildDecoder::marginal());
    let cols = RightColumns::build(&eng, &lvl, right_width, lm, rm)
        .expect("a marginal side must build");

    // Nothing culled: every column is reachable and every f row live, so the
    // walk visits each cell and the two derivations are compared in full.
    let reach = vec![u128::MAX; right_width];
    let live_cols = vec![u128::MAX; 8];
    let side = |view| ChildPlan {
        plan: SidePlan { carrier: None, view },
        base: 0, stride: right_width as u32,
        live_cols: &live_cols, reach: &reach,
    };
    let inputs1 = [pair(1, 2), pair(4, 5)];
    let walk = |right_cols| {
        let ctx = CellCtx {
            output_grid_base: 0, right_width,
            both_multi_pair: true,
            sides: Sides { left: side(lm), right: side(rm) },
            right_cols,
        };
        let mut out: Vec<ChildPair> = Vec::new();
        let mut scratch: Vec<ChildPair> = Vec::new();
        let mut node_idx: Vec<u32> = Vec::new();
        for j in 0..right_width {
            process_cell(
                &eng, j, 0, &inputs1, &ctx, &lvl,
                &mut scratch, &mut node_idx,
                &RefLookup, &RefLookup, &mut CollectSink { out: &mut out },
                &mut eng.limits().gate_with(u64::MAX),
            ).expect("an unbudgeted walk completes");
        }
        out
    };

    let hoisted = walk(Some(&cols));
    assert!(!hoisted.is_empty(), "the fixture must emit pairs to compare");
    assert_eq!(
        walk(None), hoisted,
        "the per-cell fallback must walk the same columns as the hoisted table",
    );
}

/// Alternating borrowed and decoded columns must not leave stale pointers in the pool.
#[test]
fn columns_reuse_descriptors_after_the_source_level_is_dropped() {
    let eng = Engine::new();
    for iteration in 0..6 {
        let (mut level, _) = marginal_shaped_level();
        if iteration % 2 == 0 { level.push_internal_node(&[pair(2, 3)]); }
        let decoder = if iteration % 3 == 0 { ChildDecoder::marginal() } else { ChildDecoder::structural() };
        let columns = RightColumns::build(&eng, &level, level.nodes.len(), ChildDecoder::structural(), decoder).unwrap();
        let mut scratch = Vec::new();
        for j in 0..level.nodes.len() {
            assert_eq!(columns.get(j), level.pairs_view_decoded(j, &mut scratch, ChildDecoder::structural(), decoder));
        }
        drop(columns);
        drop(level);
    }
}
