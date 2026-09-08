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
    lvl.marginal_counts = Some(vec![0u128; k2]);
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
        )
    };
    lim.reset_meters();
    assert_eq!(
        res.err(), Some(ApplyError::OverBudget),
        "tiny budget: collector must bail OverBudget instead of pushing unbudgeted",
    );
}
