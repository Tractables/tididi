//! The clustering rotation pass's mid-loop preemption point.
//!
//! `cluster_marginal_rotations_in_subtree` runs after every step's forget, tens
//! of times per canopy leaf compile, and one attempt restructures two levels as a
//! multiset — so before the poll a caller's wall was observed only at the seam
//! past the whole pass. These tests pin that it fires when the wall has passed,
//! that the diagram a cut hands back still counts the formula it was given, and
//! that the poll is amortized rather than per-candidate.
//!
//! The DISARMED property is pinned once, on the reduce walk
//! (`minimize::contract::strategies_deadline_tests`) — see the note in
//! `transform::unary::marginalize_deadline_tests`.

use super::*;
use crate::tdd::build::clause_to_tdd;
use crate::tdd::minimize::minimize;
use crate::tdd::query::model_count;
use crate::tdd::query::validate_marg::check_slot_count_uniqueness;
use crate::tdd::transform::pairwise::conjoin::{
    apply_and_both_owned, apply_limits, enable_reduce_deadline_check,
    reset_reduce_deadline_check_for_test, with_reduce_poll_stride,
};
use crate::tdd::transform::unary::marginalize::marginalize_batch;
use crate::vtree::{Literal, VarId};
use std::time::{Duration, Instant};

/// The shape the pass exists for, from `rotate_tests`'s parent-of-marginal
/// fixture: `root = (A, w)` and `w = (B, C)` with A and B already forgotten, so a
/// LEFT rotation at the root would bring the two marginal levels under one
/// parent. Returns the diagram, its root, and its model count.
fn one_candidate_tdd() -> (Tdd, VtreeIdx) {
    let vt_str = "vtree 9\n\
        L 0 1\nL 1 2\nI 2 0 1\n\
        L 3 3\nL 4 4\nI 5 3 4\n\
        L 6 5\nI 7 5 6\n\
        I 8 2 7\n";
    let vtree = Arc::new(Vtree::from_vtree_text(vt_str).expect("a well-formed vtree literal"));
    let lit = |v: i32| Literal::new(VarId(v.unsigned_abs() - 1), v > 0);
    let clauses = [
        vec![lit(1), lit(2)],
        vec![lit(-2), lit(3)],
        vec![lit(3), lit(4)],
        vec![lit(-4), lit(5)],
        vec![lit(1), lit(-5)],
    ];
    let mut acc: Option<Tdd> = None;
    for c in &clauses {
        let clause = clause_to_tdd(&vtree, c);
        acc = Some(match acc {
            Some(prev) => {
                let mut r = apply_and_both_owned(prev, clause);
                minimize(&mut r);
                r
            }
            None => clause,
        });
    }
    let mut tdd = acc.expect("five clauses build a diagram");
    minimize(&mut tdd);

    let root = vtree.root();
    let (a_idx, w_idx) = vtree.children(root);
    let (b_idx, _) = vtree.children(w_idx);
    let mut targets = [a_idx, b_idx];
    targets.sort_by_key(|t| t.idx());
    marginalize_batch(&mut tdd, &targets, &vtree).expect("no wall is installed here");
    assert!(
        !collect_cluster_candidates(&tdd, &subtree_allow_mask(&tdd.vtree, root)).is_empty(),
        "test setup: the fixture must offer the pass something to cluster",
    );
    (tdd, root)
}

/// Armed, with a wall already in the past, the pass cuts at its first metered
/// candidate and reports it through `ApplyError::Deadline`. Nothing is rotated,
/// and what is handed back is the diagram it was given — same count, same
/// marginal invariants — because the cut lands between two attempts and an
/// attempt is all-or-nothing.
#[test]
fn armed_expired_wall_cuts_the_clustering_pass() {
    let (mut tdd, root) = one_candidate_tdd();
    let before = model_count(&tdd);
    let mut tried = vec![0u8; tdd.vtree.num_nodes()];

    let r = {
        let _lim = apply_limits().deadline(Some(Instant::now() - Duration::from_secs(1))).apply();
        enable_reduce_deadline_check();
        with_reduce_poll_stride(1, || {
            cluster_marginal_rotations_in_subtree(&mut tdd, root, 8, &mut tried)
        })
    };
    reset_reduce_deadline_check_for_test();

    assert!(
        matches!(r, Err(ApplyError::Deadline)),
        "a wall in the past must surface Deadline, not run the pass to its fixpoint; got {r:?}",
    );
    assert_eq!(
        before,
        model_count(&tdd),
        "a cut pass must leave a diagram that still counts the formula",
    );
    check_slot_count_uniqueness(&tdd).expect("C3 must hold on a cut pass's stores");
}

/// The poll is amortized, not per-candidate: with a stride wider than the whole
/// pass's work, an expired wall goes unnoticed and the pass runs to its fixpoint.
/// Paired with the test above (same fixture, same expired wall, stride 1) this
/// pins that the stride — not the arming — decides when the clock is read.
#[test]
fn a_stride_wider_than_the_pass_never_polls() {
    let (mut tdd, root) = one_candidate_tdd();
    let before = model_count(&tdd);
    let mut tried = vec![0u8; tdd.vtree.num_nodes()];

    let r = {
        let _lim = apply_limits().deadline(Some(Instant::now() - Duration::from_secs(1))).apply();
        enable_reduce_deadline_check();
        with_reduce_poll_stride(u64::MAX, || {
            cluster_marginal_rotations_in_subtree(&mut tdd, root, 8, &mut tried)
        })
    };
    reset_reduce_deadline_check_for_test();

    r.expect("a stride the pass never reaches must not read the clock at all");
    assert_eq!(
        before,
        model_count(&tdd),
        "the pass is a size optimization — clustering never moves the count",
    );
}
