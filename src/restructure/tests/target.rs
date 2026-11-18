use std::sync::Arc;

use crate::restructure::RestructureError;
use crate::test_helpers::{CnfShape, assert_canonical, compile_clauses, rand_cnf};
use crate::vtree::rng::Lcg;
use crate::vtree::{VarId, Vtree};
use crate::{OperationError, Tdd};

/// A random tree over `vars` in a random leaf order: random merges of a
/// shuffled list, so every shape is reachable.
fn random_tree(vars: &[VarId], rng: &mut Lcg) -> Arc<Vtree> {
    let mut parts: Vec<Vtree> = vars.iter().map(|&v| Vtree::leaf(v)).collect();
    while parts.len() > 1 {
        let i = rng.below(parts.len() as u64) as usize;
        let a = parts.swap_remove(i);
        let j = rng.below(parts.len() as u64) as usize;
        let b = parts.swap_remove(j);
        parts.push(Vtree::join(&a, &b).expect("disjoint variables"));
    }
    Arc::new(parts.pop().expect("at least one variable"))
}

/// `f` moved to `target` must be the diagram compiled there directly: the
/// same function, canonical, and — canonical form being unique — the same
/// size.
fn check_move(clauses: &[Vec<i32>], source: &Arc<Vtree>, target: &Arc<Vtree>) {
    let mut moved = compile_clauses(source, clauses);
    let direct = compile_clauses(target, clauses);
    let stats = moved.restructure_to(target, usize::MAX).expect("an unbounded move completes");
    assert!(Arc::ptr_eq(moved.vtree(), target));
    assert_canonical(&moved);
    assert!(moved.equivalent(&direct).unwrap(), "{} -> {}", source.to_text(), target.to_text());
    assert_eq!(moved.model_count().unwrap(), direct.model_count().unwrap());
    assert_eq!(moved.pair_count(), direct.pair_count(), "{} -> {}", source.to_text(), target.to_text());
    assert!(stats.peak_pairs >= moved.pair_count());
}

#[test]
fn random_moves_keep_the_function_and_land_canonical() {
    let mut rng = Lcg::new(7);
    for round in 0..60 {
        let n = 2 + (round % 9) as u32;
        let vars: Vec<VarId> = (1..=n).map(VarId).collect();
        let clauses = rand_cnf(&mut rng, n, CnfShape { clauses: n as usize + 2, width: 3 });
        let source = random_tree(&vars, &mut rng);
        let target = random_tree(&vars, &mut rng);
        check_move(&clauses, &source, &target);
    }
}

/// A right comb onto the left comb of the same order takes n - 2 rotations,
/// the fewest possible.
#[test]
fn a_comb_turns_in_n_minus_two_rotations() {
    for n in [8u32, 12, 20] {
        let leaves: Vec<Vtree> = (1..=n).map(|v| Vtree::leaf(VarId(v))).collect();
        let right = leaves.iter().rev().cloned().reduce(|t, l| Vtree::join(&l, &t).unwrap()).unwrap();
        let left = leaves.iter().cloned().reduce(|t, l| Vtree::join(&t, &l).unwrap()).unwrap();
        let (source, target) = (Arc::new(right), Arc::new(left));
        let clauses: Vec<Vec<i32>> = (1..n as i32).map(|i| vec![i, -(i + 1)]).collect();
        let mut moved = compile_clauses(&source, &clauses);
        let stats = moved.restructure_to(&target, usize::MAX).unwrap();
        assert_eq!(stats.rotations, n as usize - 2, "n = {n}");
        check_move(&clauses, &source, &target);
    }
}

/// Blocks of three variables kept whole in both trees: only the levels above
/// them are rebuilt, and three blocks need at most one rotation.
#[test]
fn shared_blocks_move_as_units() {
    let block = |k: u32| {
        Vtree::linear_from_order(&[VarId(3 * k + 1), VarId(3 * k + 2), VarId(3 * k + 3)]).unwrap()
    };
    let (a, b, c) = (block(0), block(1), block(2));
    let ab_c = Arc::new(Vtree::join(&Vtree::join(&a, &b).unwrap(), &c).unwrap());
    let a_bc = Arc::new(Vtree::join(&a, &Vtree::join(&b, &c).unwrap()).unwrap());
    let ac_b = Arc::new(Vtree::join(&Vtree::join(&c, &a).unwrap(), &b).unwrap());
    let clauses = vec![vec![1, 4], vec![-2, 7], vec![5, -8, 3], vec![6, 9], vec![-1, -9]];
    for target in [&a_bc, &ac_b] {
        let mut f = compile_clauses(&ab_c, &clauses);
        let stats = f.restructure_to(target, usize::MAX).unwrap();
        assert_eq!(stats.rotations, 1);
        assert_canonical(&f);
        assert!(f.equivalent(&compile_clauses(target, &clauses)).unwrap());
    }
}

#[test]
fn a_mirrored_target_costs_no_rotation() {
    let order: Vec<VarId> = (1..=5).map(VarId).collect();
    let mut reversed = order.clone();
    reversed.reverse();
    let stick = Arc::new(Vtree::linear_from_order(&order).unwrap());
    let clauses = vec![vec![1, -2], vec![3, 4, -5], vec![-1, 5]];
    // The right-leaning stick over the reversed order is the left-leaning
    // one's mirror image.
    let mut nodes: Vec<Vtree> = reversed.iter().map(|&v| Vtree::leaf(v)).collect();
    let mut acc = nodes.remove(0);
    for leaf in nodes {
        acc = Vtree::join(&leaf, &acc).unwrap();
    }
    let mirror = Arc::new(acc);
    let mut f = compile_clauses(&stick, &clauses);
    let stats = f.restructure_to(&mirror, usize::MAX).unwrap();
    assert_eq!(stats.rotations, 0);
    assert_canonical(&f);
    assert!(f.equivalent(&compile_clauses(&mirror, &clauses)).unwrap());
}

#[test]
fn a_move_past_its_bound_keeps_the_function() {
    let vars: Vec<VarId> = (1..=8).map(VarId).collect();
    let mut rng = Lcg::new(3);
    let clauses = rand_cnf(&mut rng, 8, CnfShape { clauses: 10, width: 3 });
    let source = Arc::new(Vtree::linear_from_order(&vars).unwrap());
    let target = Arc::new(Vtree::balanced_over(&[VarId(1), VarId(8), VarId(2), VarId(7), VarId(3), VarId(6), VarId(4), VarId(5)]).unwrap());
    let mut f = compile_clauses(&source, &clauses);
    let count = f.model_count().unwrap();
    match f.restructure_to(&target, 1) {
        Err(RestructureError::Bound { .. }) => {}
        other => panic!("a bound of one pair refuses the first rotation: {other:?}"),
    }
    assert_eq!(f.model_count().unwrap(), count);
    assert_canonical(&f);
}

/// A rehome refused after kept rotations used to leave the diagram on a root
/// vtree of its own leaves, its counts short by the home's other variables.
#[test]
fn a_refused_rehome_stays_on_a_home_with_its_count() {
    use crate::vtree::Home;
    let mut rng = Lcg::new(11);
    let home = Home::new(Vtree::linear(10));
    let kept = [1u32, 2, 3, 5, 6, 8, 9];
    let own = Vtree::projection(&home, kept.map(VarId)).unwrap();
    let f = crate::test_helpers::rand_conj_over(&own, &kept, 8, 3, false, &mut rng);
    let count = f.model_count().unwrap();
    let order = [1, 10, 2, 9, 3, 8, 4, 7, 5, 6].map(VarId);
    let other = Home::new(Vtree::balanced_over(&order).unwrap());
    let (mut refused, mut moved) = (0, 0);
    for cut in 0.. {
        let eng = crate::Engine::new();
        eng.refuse_nth_reserve(cut);
        let result = eng.rehome(&f, &other, usize::MAX);
        let pending = eng.limits().refusal_pending();
        match result {
            Ok(g) => {
                assert_eq!(g.model_count().unwrap(), count);
                if pending {
                    break;
                }
            }
            Err(r) => {
                refused += 1;
                let home_of = r.tdd.vtree().home().expect("a refused diagram stays on a home");
                moved += usize::from(!Arc::ptr_eq(home_of.vtree(), home.vtree()));
                assert_eq!(r.tdd.model_count().unwrap(), count, "refused at reserve {cut}");
            }
        }
    }
    assert!(moved > 0 && refused > moved, "{refused} refusals, {moved} after a kept move");
}

#[test]
fn other_variables_are_refused() {
    let f = Tdd::clause(&Arc::new(Vtree::balanced(3)), [1, 2]).unwrap();
    let wider = Arc::new(Vtree::balanced(4));
    let mut g = f.clone();
    assert_eq!(g.restructure_to(&wider, usize::MAX), Err(RestructureError::Variables { variable: VarId(4) }));
}

#[test]
fn the_empty_function_moves() {
    let source = Arc::new(Vtree::linear(4));
    let target = Arc::new(Vtree::balanced(4));
    let mut f = compile_clauses(&source, &[vec![1], vec![-1]]);
    f.restructure_to(&target, usize::MAX).unwrap();
    assert!(f.is_zero());
    assert!(Arc::ptr_eq(f.vtree(), &target));
}

/// A summed-out block moves with its parent as long as no rotation turns a
/// summed-out level.
#[test]
fn summed_out_blocks_move_as_units() {
    let block = |k: u32| Vtree::linear_from_order(&[VarId(2 * k + 1), VarId(2 * k + 2)]).unwrap();
    let (a, b, c) = (block(0), block(1), block(2));
    let ab_c = Arc::new(Vtree::join(&Vtree::join(&a, &b).unwrap(), &c).unwrap());
    let ac_b = Arc::new(Vtree::join(&Vtree::join(&a, &c).unwrap(), &b).unwrap());
    let clauses = vec![vec![1, 3], vec![-2, 5], vec![4, -6], vec![1, -5]];
    let f = compile_clauses(&ab_c, &clauses);
    let summed = f.vtree().leaf_of(VarId(3)).and_then(|leaf| f.vtree().node(leaf).parent()).unwrap();
    let eng = crate::Engine::new();
    let g = eng.and_marginalizing(f.clone(), crate::Tdd::one(&ab_c), &[summed]).unwrap();
    let count = g.model_count().unwrap();
    let mut moved = g.clone();
    moved.restructure_to(&ac_b, usize::MAX).unwrap();
    assert_eq!(moved.model_count().unwrap(), count);
    assert!(Arc::ptr_eq(moved.vtree(), &ac_b));
}

#[test]
fn a_rotation_at_a_summed_out_level_is_refused() {
    let vars: Vec<VarId> = (1..=4).map(VarId).collect();
    let source = Arc::new(Vtree::linear_from_order(&vars).unwrap());
    let f = compile_clauses(&source, &[vec![1, 2], vec![-3, 4]]);
    let eng = crate::Engine::new();
    let g = eng.and_marginalizing(f, crate::Tdd::one(&source), &[source.root()]).unwrap();
    let mut moved = g;
    let target = Arc::new(Vtree::balanced(4));
    match moved.restructure_to(&target, usize::MAX) {
        Err(RestructureError::Operation(OperationError::MarginalLevel(_))) => {}
        other => panic!("{other:?}"),
    }
}

#[test]
fn the_move_leaves_both_vtrees_untouched() {
    let vars: Vec<VarId> = (1..=6).map(VarId).collect();
    let mut rng = Lcg::new(5);
    let source = random_tree(&vars, &mut rng);
    let target = random_tree(&vars, &mut rng);
    let (source_text, target_text) = (source.to_text(), target.to_text());
    let clauses = rand_cnf(&mut rng, 6, CnfShape { clauses: 7, width: 3 });
    let f = compile_clauses(&source, &clauses);
    let mut g = f.clone();
    g.restructure_to(&target, usize::MAX).unwrap();
    assert_eq!(source.to_text(), source_text);
    assert_eq!(target.to_text(), target_text);
    assert!(Arc::ptr_eq(f.vtree(), &source));
    assert_canonical(&f);
}

#[test]
fn moves_more_than_thirty_two_independent_units() {
    for n in [33, 40, 65] {
        let order: Vec<_> = (1..=n).map(VarId).collect();
        let reversed: Vec<_> = order.iter().rev().copied().collect();
        let source = Arc::new(Vtree::linear_from_order(&order).unwrap());
        let target = Arc::new(Vtree::linear_from_order(&reversed).unwrap());
        check_move(&[vec![1, 2], vec![-2, n as i32]], &source, &target);
    }
}

#[test]
fn reseating_counts_its_mirrors() {
    let source = Arc::new(Vtree::join(&Vtree::leaf(VarId(1)), &Vtree::leaf(VarId(2))).unwrap());
    let target = Arc::new(Vtree::join(&Vtree::leaf(VarId(2)), &Vtree::leaf(VarId(1))).unwrap());
    let mut f = compile_clauses(&source, &[vec![1]]);
    let stats = f.restructure_to(&target, usize::MAX).unwrap();
    assert_canonical(&f);
    assert_eq!((stats.rotations, stats.mirrors), (0, 1));
    assert_eq!(f.model_count().unwrap(), 2u32.into());
}

#[test]
fn every_refusal_or_stop_keeps_a_move_retryable() {
    use crate::limits::{LimitConfig, StopAt, StopRules};
    for n in [5, 9] {
        let source = Arc::new(Vtree::linear(n));
        let target = Arc::new(Vtree::random(n, 17));
        let clauses = vec![vec![1, -3], vec![2, n as i32]];
        let original = compile_clauses(&source, &clauses);
        for stopping in [false, true] {
            let eng = crate::Engine::new();
            let mut completed = false;
            let mut refusals = 0;
            for cut in 0..5000 {
                let mut f = original.clone();
                let scope = if stopping {
                    let stop = StopRules { unconditional: Some(StopAt::WorkUnits(eng.limits().work_units() + cut)), after_pairs: None };
                    Some(eng.limits().scope(LimitConfig::none().with_stop_rules(stop)))
                } else {
                    eng.refuse_nth_reserve(cut as u32);
                    None
                };
                let outcome = eng.restructure_to(&mut f, &target, usize::MAX);
                drop(scope);
                eng.limits().grant_every_reserve();
                assert_canonical(&f);
                let direct = compile_clauses(f.vtree(), &clauses);
                assert!(f.equivalent(&direct).unwrap(), "n={n}, stop={stopping}, cut={cut}");
                match outcome {
                    Ok(_) => { completed = true; break; }
                    Err(RestructureError::Operation(OperationError::OverBudget | OperationError::Stopped)) => {
                        refusals += 1;
                        eng.restructure_to(&mut f, &target, usize::MAX).unwrap();
                        assert_canonical(&f);
                        assert!(f.equivalent(&compile_clauses(&target, &clauses)).unwrap());
                    }
                    other => panic!("unexpected move result: {other:?}"),
                }
            }
            assert!(completed && refusals > 0, "n={n}, stop={stopping}");
        }
    }
}
