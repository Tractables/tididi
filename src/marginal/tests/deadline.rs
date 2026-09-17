//! The ∃-forget batch's mid-loop preemption point.
//!
//! `marginalize_batch` is where a compile forgets the variables a step has
//! finished with, and it runs BETWEEN two applies of one bottom-up step — so
//! before the poll a caller's wall was observed only at the seam past it. These
//! tests pin what the poll is worth having for: it fires when the wall has
//! passed, it is amortized rather than per-target, and the diagram a cut hands
//! back is still readable — same count, same invariants — because the batch is
//! only ever interrupted between two targets, with the end-sweep tagger run.
//!
//! The DISARMED property (a wall in the past is invisible without the arming
//! call) is pinned once, on the reduce walk
//! (`reduce::contract::strategies::tests::deadline`): all three post-apply walks
//! consult the one arming cell through the one `PollGate`, so re-testing it
//! here would pin nothing new and would race the flag, which is process-global.

use super::*;

use crate::Engine;
use crate::diagram::Literal;
use crate::vtree::{VarId};
use crate::test_helpers::clause_to_tdd;


use crate::test_helpers::check::marginal::{check_slot_count_uniqueness, check_inline_discipline};
use crate::apply::apply_and;
use crate::test_helpers::deadline_probe;
use std::sync::Arc;

/// A four-variable diagram and the two internal levels under its root, in the
/// bottom-up order a batch requires.
///
/// Both targets carry real nodes, so a batch over them is two units of metered
/// work — which is what lets a stride cut between them rather than only before
/// the first.
fn two_target_tdd() -> (Tdd, Arc<Vtree>, [VtreeIdx; 2]) {
    let eng = &crate::Engine::new();
    let vtree = Arc::new(Vtree::balanced(4));
    let lit = |v: u32, sign: bool| Literal::new(VarId(v + 1), sign);
    let clauses = [
        vec![lit(0, true), lit(1, true)],
        vec![lit(1, false), lit(2, true)],
        vec![lit(2, true), lit(3, true)],
        vec![lit(0, false), lit(3, false)],
    ];
    let mut acc: Option<Tdd> = None;
    for c in &clauses {
        let clause = clause_to_tdd(eng, &vtree, c);
        acc = Some(match acc {
            Some(prev) => {
                let mut r = apply_and(prev, clause);
                r.minimize().unwrap();
                r
            }
            None => clause,
        });
    }
    let mut tdd = acc.expect("four clauses build a diagram");
    tdd.minimize().unwrap();

    let (left, right) = vtree.children(vtree.root());
    let mut targets = [left, right];
    targets.sort_by_key(|t| vtree.topo_pos(*t));
    for t in targets {
        assert!(
            !tdd.levels[t.idx()].is_marginal() && tdd.levels[t.idx()].slot_count() > 0,
            "test setup: target {} must be an explicit level with nodes to forget",
            t.0,
        );
    }
    (tdd, vtree, targets)
}

/// Armed, with a wall already in the past, the batch cuts at its first metered
/// target and reports it through `OperationError::Stopped` — the same arm the
/// compile already takes when an apply runs out of wall. Stride 1 makes every
/// target a poll, which is what "the first target is metered" means.
#[test]
fn an_expired_wall_cuts_the_forget_batch() {
    let (mut tdd, vtree, targets) = two_target_tdd();

    let r = deadline_probe(Some(1), |eng| marginalize_batch(eng, &mut tdd, &targets, &vtree));

    assert!(
        matches!(r, Err(OperationError::Stopped)),
        "a wall in the past must surface Deadline, not forget the whole batch; got {r:?}",
    );
    assert!(
        targets.iter().all(|t| !tdd.levels[t.idx()].is_marginal()),
        "a cut at the first target must leave every target explicit",
    );
}

/// The state a MID-batch cut leaves behind. The stride is pinned so the meter
/// comes due on the second target: the first is forgotten, the second is not, and
/// what is handed back must be exactly the diagram a batch over that prefix
/// alone would have produced — it counts the same formula, and the marginal
/// invariants the end-sweep tagger is responsible for still hold. This is the
/// property the tagger call on the error path buys.
#[test]
fn a_cut_batch_leaves_a_readable_diagram() {
    let (mut tdd, vtree, targets) = two_target_tdd();
    let before = tdd.model_count().unwrap();
    // One past the first target's metered work (`width + 1`), so the first tick
    // does not poll and the second does.
    let stride = tdd.levels[targets[0].idx()].slot_count() as u64 + 2;

    let r = deadline_probe(Some(stride), |eng| marginalize_batch(eng, &mut tdd, &targets, &vtree));

    assert!(
        matches!(r, Err(OperationError::Stopped)),
        "the second target's tick must poll an expired wall; got {r:?}",
    );
    assert!(
        tdd.levels[targets[0].idx()].is_marginal(),
        "the target processed before the cut must keep its marginal store",
    );
    assert!(
        !tdd.levels[targets[1].idx()].is_marginal(),
        "test setup: the stride must cut BEFORE the second target, not after it",
    );
    assert_eq!(
        before,
        tdd.model_count().unwrap(),
        "forgetting is count-preserving, so a partly-forgotten diagram must still count the formula",
    );
    check_slot_count_uniqueness(&tdd).expect("invariant 10 must hold on a cut batch's stores");
    check_inline_discipline(&tdd).expect("invariant 7 must hold: the cut path runs the end-sweep tagger");
}

/// Armed but with no wall installed, the batch runs to completion: the poll reads
/// the caller's deadline cell, and `None` there is the shield an untimed
/// compile path holds.
#[test]
fn no_wall_installed_completes() {
    let (mut tdd, vtree, targets) = two_target_tdd();
    let before = tdd.model_count().unwrap();

    // Not `deadline_probe`: this is the one case whose engine must carry NO
    // wall, which is exactly what the probe installs.
    let r = {
        let eng = Engine::new();
        eng.limits().pin_reduce_poll_stride(Some(1));
        marginalize_batch(&eng, &mut tdd, &targets, &vtree)
    };

    r.expect("no wall → the batch must complete");
    assert!(
        targets.iter().all(|t| tdd.levels[t.idx()].is_marginal()),
        "the completed batch must forget every target",
    );
    assert_eq!(before, tdd.model_count().unwrap(), "forgetting is count-preserving");
}

/// The poll is amortized, not per-target: with a stride wider than the whole
/// batch's work, an expired wall goes unnoticed and the batch completes. Paired
/// with `an_expired_wall_cuts_the_forget_batch` (same fixture, same expired
/// wall, stride 1) this pins that the stride is what decides
/// when the clock is read, which is the property the production cadence rests on.
/// The production stride is untouched; the cadence is pinned per-test.
#[test]
fn a_stride_wider_than_the_batch_never_polls() {
    let (mut tdd, vtree, targets) = two_target_tdd();

    let r = deadline_probe(Some(u64::MAX), |eng| marginalize_batch(eng, &mut tdd, &targets, &vtree));

    r.expect("a stride the batch never reaches must not read the clock at all");
    assert!(
        targets.iter().all(|t| tdd.levels[t.idx()].is_marginal()),
        "the unpolled batch must still forget every target",
    );
}
