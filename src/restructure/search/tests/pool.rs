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
            assert!(crate::test_helpers::same_storage(m, o), "{mv:?}");
            assert_canonical(m);
        }
    }
}

/// The pairs a probe shows `accept` are the pairs the kept move's rebuilt
/// levels hold: the probe counts them without building the levels.
#[test]
fn a_kept_move_rebuilds_the_pairs_its_probe_counted() {
    let original = pool();
    let eng = crate::Engine::new();
    let mut kept = 0;
    for mv in moves(original[0].vtree()) {
        let (left, right) = original[0].vtree().children(mv.rotation.pivot);
        let promoted = match mv.rotation.kind {
            RotationKind::Left => right,
            RotationKind::Right => left,
        };
        let mut members = original.clone();
        let mut counted = Vec::new();
        let accept = |p: &super::PoolProbe<'_>| {
            counted = p.after().to_vec();
            true
        };
        if !with_members(&mut members, |refs| eng.rotate_pool_if(refs, mv, usize::MAX, accept)).unwrap() {
            continue;
        }
        kept += 1;
        for (m, &pairs) in members.iter().zip(&counted) {
            let levels = m.levels();
            assert_eq!(levels[mv.rotation.pivot.idx()].live_pairs() + levels[promoted.idx()].live_pairs(), pairs, "{mv:?}");
        }
    }
    assert!(kept > 0);
}

/// A bound the count of a member's levels would pass abandons the move
/// before `accept` is asked; a move counted within the bound is rebuilt
/// within it.
#[test]
fn a_bound_abandons_a_move_before_it_is_asked_or_not_at_all() {
    let original = pool();
    let tables: Vec<Vec<bool>> = original.iter().map(truth_table).collect();
    let eng = crate::Engine::new();
    let (mut abandoned, mut kept) = (0, 0);
    for bound in [0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32] {
        for mv in moves(original[0].vtree()) {
            let mut members = original.clone();
            let mut asked = false;
            let outcome = with_members(&mut members, |refs| eng.rotate_pool_if(refs, mv, bound, |_| {
                asked = true;
                true
            })).unwrap();
            assert_eq!(outcome, asked, "bound {bound}, {mv:?}");
            if outcome {
                kept += 1;
                for (m, table) in members.iter_mut().zip(&tables) {
                    m.minimize().unwrap();
                    assert_eq!(&truth_table(m), table, "bound {bound}, {mv:?}");
                }
            } else {
                abandoned += 1;
                for (m, o) in members.iter().zip(&original) {
                    assert!(crate::test_helpers::same_storage(m, o), "bound {bound}, {mv:?}");
                }
            }
        }
    }
    assert!(abandoned > 0 && kept > 0, "{abandoned} abandoned, {kept} kept");
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
        let config = PoolSearchConfig { max_sweeps: 6, crossed, ..PoolSearchConfig::default() };
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

#[test]
fn a_probe_charges_its_rebuilds_to_the_work_clock() {
    let original = pool();
    let eng = crate::Engine::new();
    let mv = moves(original[0].vtree()).into_iter().find(|&mv| {
        let mut members = original.clone();
        with_members(&mut members, |refs| eng.rotate_pool_if(refs, mv, usize::MAX, |_| true)).unwrap()
    });
    let mv = mv.expect("some move applies");
    let mut members = original.clone();
    let start = eng.limits().work_units();
    assert!(!with_members(&mut members, |refs| eng.rotate_pool_if(refs, mv, usize::MAX, |_| false)).unwrap());
    assert!(eng.limits().work_units() > start);
    for m in &members {
        assert_canonical(m);
    }
}

#[test]
fn a_search_with_no_work_to_spend_probes_nothing() {
    let original = pool();
    let eng = crate::Engine::new();
    let mut members = original.clone();
    let config = PoolSearchConfig { max_work_units: 0, ..PoolSearchConfig::default() };
    let stats = with_members(&mut members, |refs| eng.pool_search(refs, &config)).unwrap();
    assert_eq!((stats.probes, stats.accepts, stats.work_units), (0, 0, 0), "{stats:?}");
    for (m, o) in members.iter().zip(&original) {
        assert!(Arc::ptr_eq(m.vtree(), o.vtree()));
        assert!(m.equivalent(o).unwrap());
        assert_canonical(m);
    }
}

#[test]
fn invalid_pivot_is_refused_without_changing_members() {
    let mut members = pool();
    let original = members.clone();
    let eng = crate::Engine::new();
    let pivot = VtreeIdx(u32::MAX);
    let mv = PoolMove { rotation: RotationMove { pivot, kind: RotationKind::Left }, crossed: false };
    let result = with_members(&mut members, |refs| eng.rotate_pool_if(refs, mv, usize::MAX, |_| panic!("invalid move scored")));
    assert_eq!(result, Err(crate::OperationError::LevelNotInVtree(pivot)));
    for (m, before) in members.iter().zip(&original) {
        assert!(crate::test_helpers::same_storage(m, before));
        assert_canonical(m);
    }
}

#[test]
fn a_stop_before_acceptance_restores_the_pool() {
    use crate::limits::{LimitConfig, StopAt, StopRules};
    let eng = crate::Engine::new();
    let vtree = Arc::new(Vtree::balanced(6));
    let original = Tdd::one(&vtree);
    let mut f = original.clone();
    let start = eng.limits().work_units();
    let rules = StopRules { unconditional: Some(StopAt::WorkUnits(start + 1)), after_pairs: None };
    let result = {
        let _scope = eng.limits().scope(LimitConfig::none().with_stop_rules(rules));
        let mv = PoolMove { rotation: RotationMove { pivot: vtree.root(), kind: RotationKind::Left }, crossed: false };
        eng.rotate_pool_if(&mut [&mut f], mv, usize::MAX, |_| panic!("stopped move scored"))
    };
    assert_eq!(result, Err(crate::OperationError::Stopped));
    assert!(crate::test_helpers::same_storage(&f, &original));
    assert!(Arc::ptr_eq(f.vtree(), &vtree));
    assert_canonical(&f);
}

#[test]
fn search_stops_before_another_applicable_move() {
    let eng = crate::Engine::new();
    let vtree = Arc::new(Vtree::balanced(6));
    let mut f = Tdd::one(&vtree);
    let config = PoolSearchConfig { max_work_units: 1, ..PoolSearchConfig::default() };
    let stats = eng.pool_search(&mut [&mut f], &config).unwrap();
    // The move, and the one member's two levels of a pair each, read and
    // not rebuilt: the move does not lower the pairs, so it is not kept.
    assert_eq!(stats.work_units, super::PROBE_UNITS + 3, "only the first applicable probe may overshoot");
    assert_canonical(&f);
}

#[test]
fn a_declined_move_rebuilds_no_member_whose_levels_keep_their_size() {
    let eng = crate::Engine::new();
    let vtree = Arc::new(Vtree::balanced(VARS));
    let mut original = vec![Tdd::one(&vtree), Tdd::one(&vtree), compile_clauses(&vtree, &[vec![-4]])];
    for m in &mut original {
        m.minimize().unwrap();
    }
    let mv = PoolMove { rotation: RotationMove { pivot: vtree.root(), kind: RotationKind::Left }, crossed: false };
    let mut members = original.clone();
    let start = eng.limits().work_units();
    assert!(!with_members(&mut members, |refs| eng.rotate_pool_if(refs, mv, usize::MAX, |p| {
        assert_eq!(p.before(), p.after());
        false
    })).unwrap());
    // Each member's two levels at the root and its right child hold a pair
    // each: read, charged, and left alone.
    let copy = super::PROBE_UNITS + vtree.num_nodes() as u64;
    assert_eq!(eng.limits().work_units() - start, copy + 3 * 3);
    for (m, before) in members.iter().zip(&original) {
        assert!(crate::test_helpers::same_storage(m, before));
        assert!(Arc::ptr_eq(m.vtree(), &vtree));
        assert_canonical(m);
    }
    // Kept, every member is rebuilt onto the rotated tree.
    let tables: Vec<Vec<bool>> = original.iter().map(truth_table).collect();
    assert!(with_members(&mut members, |refs| eng.rotate_pool_if(refs, mv, usize::MAX, |_| true)).unwrap());
    let rotated = Arc::clone(members[0].vtree());
    assert!(!rotated.same_tree(&vtree));
    for (m, table) in members.iter_mut().zip(&tables) {
        assert!(Arc::ptr_eq(m.vtree(), &rotated));
        m.minimize().unwrap();
        assert_canonical(m);
        assert_eq!(&truth_table(m), table);
    }
}

#[test]
fn every_refused_pool_reservation_restores_all_members() {
    let eng = crate::Engine::new();
    let original = pool();
    let mv = PoolMove { rotation: RotationMove { pivot: original[0].vtree().root(), kind: RotationKind::Left }, crossed: true };
    let mut granted = false;
    for nth in 0..1000 {
        let mut members = original.clone();
        eng.limits().refuse_nth_reserve(nth);
        let result = with_members(&mut members, |refs| eng.rotate_pool_if(refs, mv, usize::MAX, |_| true));
        eng.limits().grant_every_reserve();
        match result {
            Err(crate::OperationError::OverBudget) => {
                for (m, before) in members.iter().zip(&original) {
                    assert!(crate::test_helpers::same_storage(m, before), "reservation {nth}");
                    assert!(Arc::ptr_eq(m.vtree(), before.vtree()));
                    assert_canonical(m);
                }
            }
            Ok(true) => { granted = true; break; }
            other => panic!("unexpected probe result at reservation {nth}: {other:?}"),
        }
    }
    assert!(granted, "every finite probe eventually has enough reservations");
}

#[test]
fn every_pool_stop_restores_all_members() {
    use crate::limits::{LimitConfig, StopAt, StopRules};
    let eng = crate::Engine::new();
    let original = pool();
    let mv = PoolMove { rotation: RotationMove { pivot: original[0].vtree().root(), kind: RotationKind::Left }, crossed: true };
    let mut completed = false;
    for cut in 0..5000 {
        let mut members = original.clone();
        let outcome = {
            let stop = StopRules { unconditional: Some(StopAt::WorkUnits(eng.limits().work_units() + cut)), after_pairs: None };
            let _scope = eng.limits().scope(LimitConfig::none().with_stop_rules(stop));
            with_members(&mut members, |refs| eng.rotate_pool_if(refs, mv, usize::MAX, |_| true))
        };
        match outcome {
            Ok(true) => { completed = true; break; }
            Err(crate::OperationError::Stopped) => {
                for (m, before) in members.iter().zip(&original) {
                    assert!(crate::test_helpers::same_storage(m, before), "stop {cut}");
                    assert!(Arc::ptr_eq(m.vtree(), before.vtree()));
                    assert_canonical(m);
                }
            }
            other => panic!("unexpected pool result: {other:?}"),
        }
    }
    assert!(completed);
}

/// Every member after a pool search that `outcome` ended: its function, on
/// the vtree every member shares. A search that fails keeps the moves it
/// kept, and the probe under way puts back every member it rebuilt.
fn assert_search_kept_every_function(members: &mut [Tdd], tables: &[Vec<bool>], label: &str) {
    let first = Arc::clone(members[0].vtree());
    for (m, table) in members.iter_mut().zip(tables) {
        assert!(Arc::ptr_eq(m.vtree(), &first), "{label}");
        m.minimize().unwrap();
        assert_canonical(m);
        assert_eq!(&truth_table(m), table, "{label}");
    }
}

#[test]
fn every_stop_in_a_pool_search_keeps_every_function() {
    use crate::limits::{LimitConfig, StopAt, StopRules};
    let eng = crate::Engine::new();
    let original = pool();
    let tables: Vec<Vec<bool>> = original.iter().map(truth_table).collect();
    let config = PoolSearchConfig { max_sweeps: 6, ..PoolSearchConfig::default() };
    let (mut stopped, mut cut) = (0, 0);
    loop {
        let mut members = original.clone();
        let outcome = {
            let stop = StopRules { unconditional: Some(StopAt::WorkUnits(eng.limits().work_units() + cut)), after_pairs: None };
            let _scope = eng.limits().scope(LimitConfig::none().with_stop_rules(stop));
            with_members(&mut members, |refs| eng.pool_search(refs, &config))
        };
        let done = match outcome {
            Ok(_) => true,
            Err(crate::OperationError::Stopped) => {
                stopped += 1;
                false
            }
            other => panic!("unexpected pool search result at stop {cut}: {other:?}"),
        };
        assert_search_kept_every_function(&mut members, &tables, &format!("stop {cut}"));
        if done {
            break;
        }
        cut += 1 + cut / 32;
    }
    assert!(stopped > 0);
}

#[test]
fn every_refused_reservation_in_a_pool_search_keeps_every_function() {
    let eng = crate::Engine::new();
    let original = pool();
    let tables: Vec<Vec<bool>> = original.iter().map(truth_table).collect();
    let config = PoolSearchConfig { max_sweeps: 6, ..PoolSearchConfig::default() };
    let (mut refused, mut nth) = (0, 0);
    loop {
        let mut members = original.clone();
        eng.limits().refuse_nth_reserve(nth);
        let outcome = with_members(&mut members, |refs| eng.pool_search(refs, &config));
        eng.limits().grant_every_reserve();
        let done = match outcome {
            Ok(_) => true,
            Err(crate::OperationError::OverBudget) => {
                refused += 1;
                false
            }
            other => panic!("unexpected pool search result at reservation {nth}: {other:?}"),
        };
        assert_search_kept_every_function(&mut members, &tables, &format!("reservation {nth}"));
        if done {
            break;
        }
        nth += 1 + nth / 32;
    }
    assert!(refused > 0);
}

/// The descent without the refused pivots skipped: every pivot of every
/// sweep probed, in the search's order. Returns the moves kept and probed.
fn full_descent(eng: &crate::Engine, members: &mut [Tdd], config: &PoolSearchConfig) -> (usize, usize) {
    for m in members.iter_mut() {
        eng.reduce(m, crate::reduce::ReductionPlan::default()).unwrap();
    }
    let crossings: &[bool] = if config.crossed { &[false, true] } else { &[false] };
    let (mut accepts, mut probes) = (0, 0);
    for _ in 0..config.max_sweeps {
        let internals: Vec<VtreeIdx> = members[0].vtree().internal_bottomup().map(|(v, _, _)| v).collect();
        let mut kept = 0;
        for v in internals {
            'pivot: for kind in [RotationKind::Left, RotationKind::Right] {
                for &crossed in crossings {
                    let mv = PoolMove { rotation: RotationMove { pivot: v, kind }, crossed };
                    probes += 1;
                    if with_members(members, |refs| eng.rotate_pool_if(refs, mv, usize::MAX, |p| p.live_pairs_delta() < 0)).unwrap() {
                        accepts += 1;
                        kept += 1;
                        break 'pivot;
                    }
                }
            }
        }
        if kept == 0 {
            break;
        }
    }
    for m in members.iter_mut() {
        eng.reduce(m, crate::reduce::ReductionPlan::default()).unwrap();
    }
    (accepts, probes)
}

#[test]
fn skipping_refused_pivots_keeps_the_moves_of_the_full_descent() {
    let eng = crate::Engine::new();
    for crossed in [false, true] {
        let config = PoolSearchConfig { max_sweeps: 6, crossed, ..PoolSearchConfig::default() };
        let mut searched = pool();
        let stats = with_members(&mut searched, |refs| eng.pool_search(refs, &config)).unwrap();
        let mut full = pool();
        let (accepts, probes) = full_descent(&eng, &mut full, &config);
        assert_eq!(stats.accepts, accepts, "crossed {crossed}");
        // The pool takes more than one sweep, and the later ones skip.
        assert!(stats.sweeps > 1 && stats.probes < probes, "crossed {crossed}: {stats:?} against {probes} probes");
        assert_eq!(searched[0].vtree().to_text(), full[0].vtree().to_text(), "crossed {crossed}");
        for (s, f) in searched.iter().zip(&full) {
            assert_eq!(s.pair_count(), f.pair_count(), "crossed {crossed}");
        }
    }
}
