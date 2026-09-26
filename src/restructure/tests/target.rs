use std::sync::Arc;

use super::{Turn, children_of, shortest_turns, split_turns};
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

#[test]
fn wide_trees_take_the_split_construction() {
    // Eleven single-variable units exceed the exact search.
    let mut rng = Lcg::new(11);
    for _ in 0..6 {
        let vars: Vec<VarId> = (1..=11).map(VarId).collect();
        let clauses = rand_cnf(&mut rng, 11, CnfShape { clauses: 9, width: 3 });
        let source = random_tree(&vars, &mut rng);
        let target = random_tree(&vars, &mut rng);
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

/// The two planners reach the target tree from every tree over five units.
#[test]
fn both_planners_reach_every_tree_over_five_units() {
    let trees = all_trees(0b11111);
    for from in &trees {
        for to in &trees {
            let exact = shortest_turns(from, to);
            assert_eq!(replay(from, &exact), *to);
            let split = |c: u32| children_of(to, c);
            let built = split_turns(from, &split);
            assert_eq!(replay(from, &built), *to);
            assert!(built.len() >= exact.len());
        }
    }
    // Neighbouring trees are one turn apart.
    let a = vec![0b011, 0b111];
    let b = vec![0b101, 0b111];
    assert_eq!(shortest_turns(&a, &b).len(), 1);
}

/// Every rooted binary tree over the units of `set`, as sorted cluster lists.
fn all_trees(set: u32) -> Vec<Vec<u32>> {
    if set.count_ones() == 1 {
        return vec![Vec::new()];
    }
    let low = set & set.wrapping_neg();
    let mut out = Vec::new();
    // Splits containing the lowest unit on the left, so each is counted once.
    let rest = set & !low;
    let mut sub = rest;
    loop {
        let left = low | sub;
        let right = set & !left;
        if right != 0 {
            for l in all_trees(left) {
                for r in all_trees(right) {
                    let mut tree: Vec<u32> = l.iter().chain(r.iter()).copied().collect();
                    tree.push(set);
                    tree.sort_unstable();
                    out.push(tree);
                }
            }
        }
        if sub == 0 {
            break;
        }
        sub = (sub - 1) & rest;
    }
    out
}

fn replay(from: &[u32], turns: &[Turn]) -> Vec<u32> {
    let mut set = from.to_vec();
    for turn in turns {
        let v = turn.w | turn.x;
        assert!(set.contains(&v) && set.contains(&turn.w));
        let (a, b) = children_of(&set, turn.w);
        assert!(turn.y == a || turn.y == b);
        set.retain(|&c| c != turn.w);
        set.push(turn.x | turn.y);
        set.sort_unstable();
    }
    set
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
