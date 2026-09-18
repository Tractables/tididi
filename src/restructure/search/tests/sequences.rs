//! [`Tdd::try_rotations`]: what a kept sequence leaves behind, and what a
//! declined one does not.

use std::sync::Arc;

use crate::restructure::search::RotationMove;
use crate::test_helpers::{assert_canonical, compile_clauses};
use crate::vtree::{RotationKind, Vtree, VtreeIdx};
use crate::Tdd;

/// A diagram with room to rotate: six variables, a cycle of clauses, and a
/// balanced vtree.
fn fixture() -> (Arc<Vtree>, Tdd) {
    let vtree = Arc::new(Vtree::balanced(6));
    let mut tdd = compile_clauses(
        &vtree,
        &[vec![1, 2], vec![-2, 3], vec![3, 4], vec![-4, 5], vec![5, -6], vec![1, -6]],
    );
    tdd.minimize().expect("an unarmed engine refuses nothing");
    (vtree, tdd)
}

/// Every sequence of `k` moves the internal nodes admit, for small `k`.
fn sequences(vtree: &Vtree, k: usize) -> Vec<Vec<RotationMove>> {
    let pivots: Vec<VtreeIdx> = (0..vtree.num_nodes())
        .map(|i| VtreeIdx(i as u32))
        .filter(|&v| !vtree.node(v).is_leaf())
        .collect();
    let mut out = vec![Vec::new()];
    for _ in 0..k {
        let mut next = Vec::new();
        for prefix in &out {
            for &pivot in &pivots {
                for kind in [RotationKind::Left, RotationKind::Right] {
                    let mut seq = prefix.clone();
                    seq.push(RotationMove { pivot, kind });
                    next.push(seq);
                }
            }
        }
        out = next;
    }
    out
}

#[test]
fn a_declined_sequence_leaves_the_diagram_and_its_vtree_exactly_as_they_were() {
    for k in 1..=3 {
        let (vtree, tdd) = fixture();
        let count = tdd.model_count().unwrap();
        let pairs = tdd.pair_count();
        let nodes = tdd.node_count();
        for moves in sequences(&vtree, k) {
            let mut probed = tdd.clone();
            let kept = probed.try_rotations(&moves, usize::MAX, |_| false).unwrap();
            assert!(!kept, "a closure that declines cannot keep a sequence");
            assert!(probed.vtree().same_tree(&vtree), "k = {k}, moves {moves:?}");
            assert!(probed.equivalent(&tdd).unwrap());
            assert_eq!(probed.pair_count(), pairs);
            assert_eq!(probed.node_count(), nodes);
            assert_eq!(probed.model_count().unwrap(), count);
            assert_canonical(&probed);
        }
    }
}

#[test]
fn a_kept_sequence_preserves_the_function_and_stays_canonical() {
    for k in 1..=3 {
        let (vtree, tdd) = fixture();
        let count = tdd.model_count().unwrap();
        for moves in sequences(&vtree, k) {
            let mut probed = tdd.clone();
            probed.try_rotations(&moves, usize::MAX, |_| true).unwrap();
            probed.minimize().unwrap();
            assert_canonical(&probed);
            assert_eq!(probed.model_count().unwrap(), count, "k = {k}, moves {moves:?}");
        }
    }
}

#[test]
fn a_sequence_and_its_inverse_return_the_vtree_and_the_diagram() {
    let (vtree, tdd) = fixture();
    for moves in sequences(&vtree, 2) {
        let mut probed = tdd.clone();
        if !probed.try_rotations(&moves, usize::MAX, |_| true).unwrap() {
            continue;
        }
        let inverse: Vec<RotationMove> = moves.iter().rev().map(|mv| mv.inverse()).collect();
        assert!(probed.try_rotations(&inverse, usize::MAX, |_| true).unwrap());
        probed.minimize().unwrap();
        assert!(probed.vtree().same_tree(&vtree), "moves {moves:?}");
        assert_eq!(probed.pair_count(), tdd.pair_count());
        assert_eq!(probed.node_count(), tdd.node_count());
        assert_canonical(&probed);
    }
}

#[test]
fn a_two_move_sequence_gives_what_the_two_moves_give_one_at_a_time() {
    let (vtree, tdd) = fixture();
    for moves in sequences(&vtree, 2) {
        let mut together = tdd.clone();
        if !together.try_rotations(&moves, usize::MAX, |_| true).unwrap() {
            continue;
        }
        let mut apart = tdd.clone();
        for mv in &moves {
            assert!(apart.try_rotations(&[*mv], usize::MAX, |_| true).unwrap());
        }
        assert!(together.vtree().same_tree(apart.vtree()), "moves {moves:?}");
        assert_eq!(together.pair_count(), apart.pair_count());
        assert_eq!(together.node_count(), apart.node_count());
    }
}

#[test]
fn a_probe_reads_the_state_before_the_first_move() {
    let (vtree, tdd) = fixture();
    let moves = sequences(&vtree, 2)
        .into_iter()
        .find(|seq| {
            let mut probed = tdd.clone();
            probed.try_rotations(seq, usize::MAX, |_| true).unwrap()
        })
        .expect("the fixture admits a two-move sequence");

    let mut probed = tdd.clone();
    let seen = std::cell::RefCell::new(Vec::new());
    probed
        .try_rotations(&moves, usize::MAX, |probe| {
            for &level in probe.changed() {
                seen.borrow_mut().push((
                    level,
                    probe.before(level).unwrap().live_pairs(),
                    probe.after(level).unwrap().live_pairs(),
                ));
            }
            false
        })
        .unwrap();
    let seen = seen.into_inner();
    assert!(!seen.is_empty(), "a scored sequence rebuilt at least two levels");
    // Declined, so the diagram is the one the probe called `before`.
    for (level, before, _) in seen {
        assert_eq!(tdd.level(level).live_pairs(), before);
    }
}

#[test]
fn a_move_that_does_not_apply_declines_without_scoring() {
    let (vtree, mut tdd) = fixture();
    // A leaf has no children to promote, so no rotation applies at it.
    let leaf = (0..vtree.num_nodes())
        .map(|i| VtreeIdx(i as u32))
        .find(|&v| vtree.node(v).is_leaf())
        .expect("a vtree has leaves");
    let moves = [RotationMove { pivot: leaf, kind: RotationKind::Left }];
    let kept = tdd
        .try_rotations(&moves, usize::MAX, |_| panic!("an inapplicable move must not be scored"))
        .unwrap();
    assert!(!kept);
    assert!(tdd.vtree().same_tree(&vtree));
}

#[test]
fn an_empty_sequence_keeps_nothing() {
    let (vtree, mut tdd) = fixture();
    let kept = tdd.try_rotations(&[], usize::MAX, |_| panic!("nothing to score")).unwrap();
    assert!(!kept);
    assert!(tdd.vtree().same_tree(&vtree));
}
