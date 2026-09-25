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
    let cols = RightColumns::build(&eng, &lvl, right_width, lm, rm, false).expect("a marginal side must build");
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
    let cols = RightColumns::build(&eng, &lvl, right_width, ChildDecoder::structural(), ChildDecoder::structural(), false)
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
    assert!(RightColumns::build(&eng, &lvl, right_width, ChildDecoder::structural(), ChildDecoder::structural(), false).is_none());
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
        let cols = RightColumns::build(&eng, &lvl, right_width, ChildDecoder::structural(), ChildDecoder::marginal(), false)
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
        let cols = RightColumns::build(&eng, &lvl, right_width, ChildDecoder::structural(), ChildDecoder::structural(), false).expect("within budget");
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
        assert!(RightColumns::build(&eng, &lvl, right_width, ChildDecoder::structural(), ChildDecoder::marginal(), false).is_none());
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
    // putting an N-pair f_pairs into the N×1 arm.
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
        const DENSE_SLAB: bool = true;
        #[inline(always)]
        fn grid_row(&self, i: usize) -> usize { i }
        #[inline(always)]
        fn cell(&mut self, eng: &Engine, a: CellArgs<'_, '_, L, R>) -> Result<(), OperationError> {
            process_cell::<_, _, _>(
                eng,
                a.j, a.row_base, a.f_pairs,
                a.ctx, a.right_level_t, a.g_pairs_scratch, a.node_idx, a.left, a.right,
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
    let mut f_pairs_scratch: Vec<ChildPair> = Vec::new();
    let mut g_pairs_scratch: Vec<ChildPair> = Vec::new();
    let mut node_idx: Vec<u32> = vec![0; K1];
    let mut out: Vec<ChildPair> = Vec::new();
    run_level_rows::<true, _, _, _>(
        &eng,
        // The collecting action never streams, so the child levels stand in for
        // themselves — nothing on this route reads them.
        RowLoop { f_level: &f, g_level: &g, children: Sides { left: &f, right: &g }, ctx: &ctx, f_width: K1 },
        RowScratch { f_pairs: &mut f_pairs_scratch, g_pairs: &mut g_pairs_scratch, node_idx: &mut node_idx },
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
    let cols = RightColumns::build(&eng, &lvl, right_width, lm, rm, false)
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
    let f_pairs = [pair(1, 2), pair(4, 5)];
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
                &eng, j, 0, &f_pairs, &ctx, &lvl,
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
        let columns = RightColumns::build(&eng, &level, level.nodes.len(), ChildDecoder::structural(), decoder, false).unwrap();
        let mut scratch = Vec::new();
        for j in 0..level.nodes.len() {
            assert_eq!(columns.get(j), level.pairs_view_decoded(j, &mut scratch, ChildDecoder::structural(), decoder));
        }
        drop(columns);
        drop(level);
    }
}

/// A level with one long column in three runs of shared `.left`, and one
/// column too short to group.
fn long_and_short_columns() -> TddLevel {
    let mut level = TddLevel::new();
    let long: Vec<ChildPair> = (0..70u32).map(|k| pair(if k < 10 { 0 } else if k < 40 { 1 } else { 2 }, k)).collect();
    level.push_internal_node(&long);
    level.push_internal_node(&[pair(0, 1), pair(0, 2), pair(1, 3)]);
    level
}

/// The run ends of `pairs`, found the plain way.
fn run_ends_of(pairs: &[ChildPair]) -> Vec<u32> {
    (1..=pairs.len())
        .filter(|&i| i == pairs.len() || pairs[i].left != pairs[i - 1].left)
        .map(|i| i as u32)
        .collect()
}

/// A grouped table records the runs of each column long enough to group and
/// of no other; an ungrouped table records none.
#[test]
fn columns_record_the_runs_of_long_columns_on_grouped_levels() {
    let eng = Engine::new();
    let level = long_and_short_columns();
    let views = (ChildDecoder::structural(), ChildDecoder::structural());
    let grouped = RightColumns::build(&eng, &level, 2, views.0, views.1, true).expect("unbudgeted");
    let want = run_ends_of(grouped.get(0));
    assert_eq!(want.len(), 3, "the fixture's long column holds three runs");
    assert_eq!(grouped.runs(0), Some(&want[..]));
    assert_eq!(grouped.runs(1), None, "a column under the grouping length is not grouped");
    drop(grouped);
    let plain = RightColumns::build(&eng, &level, 2, views.0, views.1, false).expect("unbudgeted");
    assert_eq!((plain.runs(0), plain.runs(1)), (None, None));
}

/// The recorded runs charge the soft budget while the table lives and give
/// the charge back on drop. A budget too small for them leaves the column
/// ungrouped rather than refusing the table.
#[test]
fn column_runs_charge_the_soft_budget_and_degrade_when_refused() {
    let level = long_and_short_columns();
    let views = (ChildDecoder::structural(), ChildDecoder::structural());

    let eng = Engine::new();
    let lim = eng.limits();
    lim.set_budget(Some(1 << 20));
    let h0 = lim.budget_headroom().expect("budget installed");
    let table = RightColumns::build(&eng, &level, 2, views.0, views.1, true).expect("within budget");
    assert!(table.runs(0).is_some());
    assert!(lim.budget_headroom().unwrap() < h0, "the runs must charge the soft budget");
    drop(table);
    assert_eq!(lim.budget_headroom().unwrap(), h0, "dropping the table must release the runs' charge");

    let eng = Engine::new();
    let lim = eng.limits();
    lim.set_budget(Some(4));
    let h1 = lim.budget_headroom().unwrap();
    let table = RightColumns::build(&eng, &level, 2, views.0, views.1, true)
        .expect("identity columns borrow and need no budget");
    assert_eq!(table.runs(0), None, "a refused run push leaves the column ungrouped");
    assert_eq!(table.get(0).len(), 70, "the table itself still serves the column");
    drop(table);
    assert_eq!(lim.budget_headroom().unwrap(), h1, "a refused recording must not retain a charge");
}

/// The grouped N×M walk emits exactly the pairs of the ungrouped one.
///
/// A long f row against a long column, through a lookup that kills some
/// `(f, g)` child combinations on each side, so both the per-run left
/// test and the per-pair right test skip entries.
#[test]
fn the_grouped_walk_emits_the_pairs_of_the_ungrouped_walk() {
    use super::{process_cell, CellCtx, CollectSink};
    use crate::apply::conjoin::child_lookup::ChildLookup;

    struct SomeDead;
    impl ChildLookup for SomeDead {
        type Row = u32;
        fn row(&self, row: u32) -> u32 { row }
        fn get_in_row(&self, _node_idx: &[u32], row: u32, col: u32) -> u32 {
            if (row + col).is_multiple_of(3) { NO_PRODUCT } else { row * 1000 + col }
        }
    }

    let eng = Engine::new();
    let level = long_and_short_columns();
    let views = (ChildDecoder::structural(), ChildDecoder::structural());
    let reach = vec![u128::MAX; 2];
    let live_cols = vec![u128::MAX; 256];
    let side = ChildPlan {
        plan: SidePlan { carrier: None, view: ChildDecoder::structural() },
        base: 0, stride: 2, live_cols: &live_cols, reach: &reach,
    };
    let f_pairs: Vec<ChildPair> = (0..66u32).map(|k| pair(k / 6, 2 * k)).collect();
    let walk = |grouped| {
        let cols = RightColumns::build(&eng, &level, 2, views.0, views.1, grouped).expect("unbudgeted");
        assert_eq!(cols.runs(0).is_some(), grouped);
        let ctx = CellCtx {
            output_grid_base: 0, right_width: 2,
            both_multi_pair: true,
            sides: Sides { left: side, right: side },
            right_cols: Some(&cols),
        };
        let mut out: Vec<ChildPair> = Vec::new();
        process_cell(
            &eng, 0, 0, &f_pairs, &ctx, &level,
            &mut Vec::new(), &mut Vec::<u32>::new(),
            &SomeDead, &SomeDead, &mut CollectSink { out: &mut out },
            &mut eng.limits().gate_with(u64::MAX),
        ).expect("an unbudgeted walk completes");
        out.sort_by_key(|p| (p.left.raw(), p.right.raw()));
        out
    };

    let ungrouped = walk(false);
    assert!(!ungrouped.is_empty(), "the fixture must emit pairs to compare");
    assert!(ungrouped.len() < f_pairs.len() * 70, "the fixture must kill some combinations");
    assert_eq!(walk(true), ungrouped);
}
