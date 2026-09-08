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
//! (`minimize::contract::strategies_deadline_tests`): all three post-apply walks
//! consult the ONE arming cell through the ONE `PollTicker`, so re-testing it
//! here would pin nothing new and would race the flag, which is process-global.

use super::*;
use crate::diagram::Literal;
use crate::vtree::{VarId};
use crate::build::clause_to_tdd;
use crate::reduce::minimize;
use crate::query::model_count;
use crate::check::marg::{check_slot_count_uniqueness, check_tdd_marg_invariants};
use crate::limits::{
    apply_limits, with_reduce_poll_stride,
};
use crate::apply::conjoin::apply_and;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// A four-variable diagram and the two internal levels under its root, in the
/// bottom-up order a batch requires.
///
/// Both targets carry real nodes, so a batch over them is TWO units of metered
/// work — which is what lets a stride cut between them rather than only before
/// the first.
fn two_target_tdd() -> (Tdd, Arc<Vtree>, [VtreeIdx; 2]) {
    let vtree = Arc::new(Vtree::balanced(4));
    let lit = |v: u32, sign: bool| Literal::new(VarId(v), sign);
    let clauses = [
        vec![lit(0, true), lit(1, true)],
        vec![lit(1, false), lit(2, true)],
        vec![lit(2, true), lit(3, true)],
        vec![lit(0, false), lit(3, false)],
    ];
    let mut acc: Option<Tdd> = None;
    for c in &clauses {
        let clause = clause_to_tdd(&vtree, c);
        acc = Some(match acc {
            Some(prev) => {
                let mut r = apply_and(prev, clause);
                minimize(&mut r);
                r
            }
            None => clause,
        });
    }
    let mut tdd = acc.expect("four clauses build a diagram");
    minimize(&mut tdd);

    let (left, right) = vtree.children(vtree.root());
    let mut targets = [left, right];
    targets.sort_by_key(|t| vtree.topo_pos(*t));
    for t in targets {
        assert!(
            !tdd.levels[t.idx()].is_marginal() && tdd.levels[t.idx()].width() > 0,
            "test setup: target {} must be an explicit level with nodes to forget",
            t.0,
        );
    }
    (tdd, vtree, targets)
}

/// Armed, with a wall already in the past, the batch cuts at its first metered
/// target and reports it through `ApplyError::Deadline` — the same arm the
/// compile already takes when an apply runs out of wall. Stride 1 makes every
/// target a poll, which is what "the first target is metered" means.
#[test]
fn an_expired_wall_cuts_the_forget_batch() {
    let (mut tdd, vtree, targets) = two_target_tdd();

    let r = {
        let _lim = apply_limits().deadline(Some(Instant::now() - Duration::from_secs(1))).apply();
        with_reduce_poll_stride(1, || marginalize_batch(&mut tdd, &targets, &vtree))
    };

    assert!(
        matches!(r, Err(ApplyError::Deadline)),
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
    let before = model_count(&tdd);
    // One past the first target's metered work (`width + 1`), so the first tick
    // does not poll and the second does.
    let stride = tdd.levels[targets[0].idx()].width() as u64 + 2;

    let r = {
        let _lim = apply_limits().deadline(Some(Instant::now() - Duration::from_secs(1))).apply();
        with_reduce_poll_stride(stride, || marginalize_batch(&mut tdd, &targets, &vtree))
    };

    assert!(
        matches!(r, Err(ApplyError::Deadline)),
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
        model_count(&tdd),
        "forgetting is count-preserving, so a partly-forgotten diagram must still count the formula",
    );
    check_slot_count_uniqueness(&tdd).expect("C3 must hold on a cut batch's stores");
    check_tdd_marg_invariants(&tdd).expect("I2 must hold: the cut path runs the end-sweep tagger");
}

/// Armed but with no wall installed, the batch runs to completion: the poll reads
/// the caller's deadline cell, and `None` there is the shield an untimed
/// compile path holds.
#[test]
fn no_wall_installed_completes() {
    let (mut tdd, vtree, targets) = two_target_tdd();
    let before = model_count(&tdd);

    let r = {
        let _lim = apply_limits().deadline(None).apply();
        with_reduce_poll_stride(1, || marginalize_batch(&mut tdd, &targets, &vtree))
    };

    r.expect("no wall → the batch must complete");
    assert!(
        targets.iter().all(|t| tdd.levels[t.idx()].is_marginal()),
        "the completed batch must forget every target",
    );
    assert_eq!(before, model_count(&tdd), "forgetting is count-preserving");
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

    let r = {
        let _lim = apply_limits().deadline(Some(Instant::now() - Duration::from_secs(1))).apply();
        with_reduce_poll_stride(u64::MAX, || marginalize_batch(&mut tdd, &targets, &vtree))
    };

    r.expect("a stride the batch never reaches must not read the clock at all");
    assert!(
        targets.iter().all(|t| tdd.levels[t.idx()].is_marginal()),
        "the unpolled batch must still forget every target",
    );
}
