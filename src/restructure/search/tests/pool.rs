//! [`Engine::rotate_pool_if`] and [`Engine::pool_search`]: every member keeps
//! its function under every move, a declined move puts every member back,
//! and the pool always shares one vtree allocation.

use std::sync::Arc;

use crate::restructure::search::{PoolMove, PoolSearchConfig, RotationMove};
use crate::test_helpers::{and2, assert_canonical, compile_clauses, cube};
use crate::vtree::{RotationKind, Vtree, VtreeIdx};
use crate::Tdd;

const VARS: u32 = 6;

/// Five functions of six variables on one balanced vtree, a constant and a
/// literal among them.
fn pool() -> Vec<Tdd> {
    let vtree = Arc::new(Vtree::balanced(VARS));
    let mut members = vec![
        compile_clauses(&vtree, &[vec![1, 2], vec![-2, 3], vec![3, 4], vec![-4, 5], vec![5, -6], vec![1, -6]]),
        compile_clauses(&vtree, &[vec![1, 6], vec![2, -5], vec![-3, 4]]),
        compile_clauses(&vtree, &[vec![-1, -4], vec![2, 5], vec![3, 6], vec![-2, -3]]),
        compile_clauses(&vtree, &[]),
        compile_clauses(&vtree, &[vec![-4]]),
    ];
    for m in &mut members {
        m.minimize().expect("an unarmed engine refuses nothing");
    }
    members
}

/// Which of the 2^VARS assignments satisfy `f`, found by conjoining `f` with
/// each assignment's cube on `f`'s own vtree.
fn truth_table(f: &Tdd) -> Vec<bool> {
    (0..1u32 << VARS)
        .map(|bits| {
            let literals: Vec<(u32, bool)> = (0..VARS).map(|i| (i + 1, bits >> i & 1 == 1)).collect();
            !and2(f, &cube(f.vtree(), &literals)).is_zero()
        })
        .collect()
}

fn moves(vtree: &Vtree) -> Vec<PoolMove> {
    let mut out = Vec::new();
    for i in 0..vtree.num_nodes() {
        let pivot = VtreeIdx(i as u32);
        if vtree.node(pivot).is_leaf() {
            continue;
        }
        for kind in [RotationKind::Left, RotationKind::Right] {
            for crossed in [false, true] {
                out.push(PoolMove { rotation: RotationMove { pivot, kind }, crossed });
            }
        }
    }
    out
}

fn with_members<R>(members: &mut [Tdd], f: impl FnOnce(&mut [&mut Tdd]) -> R) -> R {
    let mut refs: Vec<&mut Tdd> = members.iter_mut().collect();
    f(&mut refs)
}

#[test]
fn a_kept_move_preserves_every_function_and_leaves_one_shared_vtree() {
    let original = pool();
    let tables: Vec<Vec<bool>> = original.iter().map(truth_table).collect();
    let eng = crate::Engine::new();
    let mut kept_moves = [0usize; 2];
    for mv in moves(original[0].vtree()) {
        let mut members = original.clone();
        let kept = with_members(&mut members, |refs| eng.rotate_pool_if(refs, mv, usize::MAX, |_| true)).unwrap();
        if !kept {
            continue;
        }
        kept_moves[usize::from(mv.crossed)] += 1;
        assert!(!members[0].vtree().same_tree(original[0].vtree()), "{mv:?} kept, and the tree did not change");
        let first = Arc::clone(members[0].vtree());
        for (m, table) in members.iter_mut().zip(&tables) {
            assert!(Arc::ptr_eq(m.vtree(), &first), "{mv:?}");
            m.minimize().unwrap();
            assert_canonical(m);
            assert_eq!(&truth_table(m), table, "{mv:?}");
        }
    }
    assert!(kept_moves[0] > 0 && kept_moves[1] > 0, "{kept_moves:?}");
}

#[test]
fn a_declined_move_leaves_every_member_exactly_as_it_was() {
    let original = pool();
    let eng = crate::Engine::new();
    for mv in moves(original[0].vtree()) {
        let mut members = original.clone();
        let kept = with_members(&mut members, |refs| eng.rotate_pool_if(refs, mv, usize::MAX, |_| false)).unwrap();
        assert!(!kept);
        for (m, o) in members.iter().zip(&original) {
            assert!(Arc::ptr_eq(m.vtree(), o.vtree()), "{mv:?}");
            assert_eq!(m.pair_count(), o.pair_count(), "{mv:?}");
            assert_eq!(m.node_count(), o.node_count(), "{mv:?}");
            assert!(m.equivalent(o).unwrap(), "{mv:?}");
            assert_canonical(m);
        }
    }
}

#[test]
fn a_crossed_move_changes_the_order_of_the_leaves() {
    let original = pool();
    let eng = crate::Engine::new();
    let order = |t: &Tdd| -> Vec<u32> {
        let vt = t.vtree();
        let mut out = Vec::new();
        let mut stack = vec![vt.root()];
        while let Some(n) = stack.pop() {
            match *vt.node(n) {
                crate::vtree::VtreeNode::Leaf { var, .. } => out.push(var.0),
                crate::vtree::VtreeNode::Internal { left, right, .. } => {
                    stack.push(right);
                    stack.push(left);
                }
            }
        }
        out
    };
    let before = order(&original[0]);
    let mut changed = 0;
    for mv in moves(original[0].vtree()) {
        let mut members = original.clone();
        if with_members(&mut members, |refs| eng.rotate_pool_if(refs, mv, usize::MAX, |_| true)).unwrap() {
            let after = order(&members[0]);
            if mv.crossed {
                changed += usize::from(after != before);
            } else {
                assert_eq!(after, before, "{mv:?}: a plain rotation keeps the leaf order");
            }
        }
    }
    assert!(changed > 0);
}

#[test]
fn members_on_different_vtrees_are_refused() {
    let mut a = pool();
    let other = Arc::new(Vtree::balanced(VARS));
    let mut b = compile_clauses(&other, &[vec![1, 2]]);
    let eng = crate::Engine::new();
    let mut refs: Vec<&mut Tdd> = vec![&mut a[0], &mut b];
    let mv = moves(refs[0].vtree())[0];
    assert_eq!(
        eng.rotate_pool_if(&mut refs, mv, usize::MAX, |_| true),
        Err(crate::OperationError::VtreeMismatch)
    );
}

#[test]
fn a_pool_search_keeps_every_function_and_never_grows_the_pool() {
    let original = pool();
    let tables: Vec<Vec<bool>> = original.iter().map(truth_table).collect();
    let eng = crate::Engine::new();
    for crossed in [false, true] {
        let mut members = original.clone();
        let config = PoolSearchConfig { max_sweeps: 6, max_inner_pairs: usize::MAX, crossed };
        let stats = with_members(&mut members, |refs| eng.pool_search(refs, &config)).unwrap();
        assert!(stats.pairs_after <= stats.pairs_before, "{stats:?}");
        assert_eq!(stats.pairs_after, members.iter().map(Tdd::pair_count).sum::<usize>());
        for (m, table) in members.iter().zip(&tables) {
            assert!(Arc::ptr_eq(m.vtree(), members[0].vtree()));
            assert_canonical(m);
            assert_eq!(&truth_table(m), table, "crossed {crossed}");
        }
    }
}
