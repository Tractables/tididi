//! Acceptance policies, wider neighborhoods and the multistart driver.

use std::sync::Arc;

use crate::Engine;
use crate::limits::OperationError;
use crate::restructure::search::{
    Annealing, Greedy, MinimizePairs, Neighborhood, RotationSearchConfig, Tabu,
};
use crate::test_helpers::{assert_canonical, compile_clauses};
use crate::vtree::Vtree;
use crate::diagram::Tdd;

/// A bounded configuration: every policy below may keep a worsening sequence,
/// and those are refused without one.
fn bounded() -> RotationSearchConfig {
    RotationSearchConfig { max_inner_pairs: 1 << 20, ..RotationSearchConfig::default() }
}

/// Seven variables on a balanced vtree, minimized. A single-rotation descent
/// stops here with pairs a wider neighborhood and a tabu walk both remove.
fn plateau() -> Tdd {
    let vtree = Arc::new(Vtree::balanced(7));
    let mut tdd = compile_clauses(
        &vtree,
        &[vec![5], vec![3, -4], vec![5, 1, 7], vec![-6, 7, 5], vec![-7, -2]],
    );
    tdd.minimize().expect("an unarmed engine refuses nothing");
    tdd
}

/// Nine variables on a linear vtree. Three connected rotations reach a shape
/// that pairs of them do not.
fn triple_plateau() -> Tdd {
    let vtree = Arc::new(Vtree::linear(9));
    let mut tdd = compile_clauses(
        &vtree,
        &[
            vec![2, 6],
            vec![4, -3],
            vec![1, 8, 4],
            vec![-9, -5],
            vec![-7, 8, 5],
            vec![2, 7, -3],
        ],
    );
    tdd.minimize().expect("an unarmed engine refuses nothing");
    tdd
}

/// Run `policy` over `tdd` and check what every search here has to preserve.
fn search_with<A: crate::restructure::search::AcceptancePolicy>(
    tdd: &mut Tdd,
    policy: &mut A,
    config: &RotationSearchConfig,
) {
    let eng = Engine::new();
    let count = tdd.model_count().unwrap();
    eng.rotation_search_with(tdd, &mut MinimizePairs, policy, config).unwrap();
    assert_canonical(tdd);
    assert_eq!(tdd.model_count().unwrap(), count);
}

#[test]
fn the_greedy_policy_is_what_the_plain_search_runs() {
    for mut tdd in [plateau(), triple_plateau()] {
        let eng = Engine::new();
        let mut explicit = tdd.clone();
        let plain =
            eng.rotation_search(&mut tdd, &mut MinimizePairs, &RotationSearchConfig::default())
                .unwrap();
        let with = eng
            .rotation_search_with(
                &mut explicit,
                &mut MinimizePairs,
                &mut Greedy,
                &RotationSearchConfig::default(),
            )
            .unwrap();
        assert_eq!((plain.probes, plain.accepts, plain.sweeps), (with.probes, with.accepts, with.sweeps));
        assert!(tdd.vtree().same_tree(explicit.vtree()));
        assert_eq!(tdd.pair_count(), explicit.pair_count());
    }
}

#[test]
fn a_pair_of_rotations_crosses_a_hill_one_rotation_stops_at() {
    let mut single = plateau();
    search_with(&mut single, &mut Greedy, &bounded());
    let mut pair = single.clone();
    search_with(&mut pair, &mut Greedy, &RotationSearchConfig {
        neighborhood: Neighborhood::Pair,
        ..bounded()
    });
    assert!(
        pair.pair_count() < single.pair_count(),
        "pairs {} did not improve on singles {}",
        pair.pair_count(),
        single.pair_count(),
    );
}

#[test]
fn three_rotations_cross_a_hill_two_stop_at() {
    let mut pair = triple_plateau();
    search_with(&mut pair, &mut Greedy, &RotationSearchConfig {
        neighborhood: Neighborhood::Pair,
        ..bounded()
    });
    let mut triple = pair.clone();
    search_with(&mut triple, &mut Greedy, &RotationSearchConfig {
        neighborhood: Neighborhood::Triple,
        ..bounded()
    });
    assert!(
        triple.pair_count() < pair.pair_count(),
        "triples {} did not improve on pairs {}",
        triple.pair_count(),
        pair.pair_count(),
    );
}

#[test]
fn tabu_leaves_a_local_minimum_the_descent_cannot() {
    let mut greedy = plateau();
    search_with(&mut greedy, &mut Greedy, &bounded());
    let mut tabu = plateau();
    search_with(&mut tabu, &mut Tabu::default(), &bounded());
    assert!(
        tabu.pair_count() < greedy.pair_count(),
        "tabu {} did not improve on the descent's {}",
        tabu.pair_count(),
        greedy.pair_count(),
    );
}

#[test]
fn tabu_returns_the_best_diagram_it_passed_through_not_the_last() {
    // The walk keeps worsening sequences, so without the rewind the result
    // would be wherever patience ran out.
    let start = plateau();
    let mut tabu = start.clone();
    search_with(&mut tabu, &mut Tabu::new(4, 1), &bounded());
    assert!(tabu.pair_count() <= start.pair_count());
}

#[test]
fn annealing_is_the_same_search_every_time_for_one_seed() {
    let mut first = plateau();
    search_with(&mut first, &mut Annealing::new(11, 4.0, 0.5), &bounded());
    let mut second = plateau();
    search_with(&mut second, &mut Annealing::new(11, 4.0, 0.5), &bounded());
    assert!(first.vtree().same_tree(second.vtree()));
    assert_eq!(first.pair_count(), second.pair_count());
    assert_eq!(first.node_count(), second.node_count());
}

#[test]
fn a_policy_that_keeps_worsening_moves_needs_a_bound() {
    let eng = Engine::new();
    let unbounded = RotationSearchConfig::default();
    assert_eq!(unbounded.max_inner_pairs, usize::MAX);
    for (name, error) in [
        ("tabu", {
            let mut tdd = plateau();
            eng.rotation_search_with(&mut tdd, &mut MinimizePairs, &mut Tabu::default(), &unbounded)
        }),
        ("annealing", {
            let mut tdd = plateau();
            eng.rotation_search_with(
                &mut tdd,
                &mut MinimizePairs,
                &mut Annealing::default(),
                &unbounded,
            )
        }),
    ] {
        let Err(OperationError::UnboundedSearch { option, needed_by }) = error else {
            panic!("{name} accepted an unbounded rebuild");
        };
        assert_eq!(option, "RotationSearchConfig::max_inner_pairs");
        assert!(needed_by.contains("worsening"), "{needed_by}");
    }
    // The descent is fine without one: it never keeps a sequence that grows.
    let mut tdd = plateau();
    assert!(eng.rotation_search_with(&mut tdd, &mut MinimizePairs, &mut Greedy, &unbounded).is_ok());
}

#[test]
fn tabu_is_never_worse_than_the_descent_over_a_seeded_corpus() {
    use crate::test_helpers::r#gen::{CnfShape, Lcg, rand_cnf};

    let eng = Engine::new();
    let mut better = 0usize;
    for seed in 0..40u64 {
        let num_vars = 5 + (seed % 5) as u32;
        let mut rng = Lcg::new(seed);
        let clauses = rand_cnf(&mut rng, num_vars, CnfShape { clauses: 8, width: 3 });
        let vtree = Arc::new(Vtree::balanced(num_vars));
        let mut start = compile_clauses(&vtree, &clauses);
        eng.minimize(&mut start).unwrap();

        let mut greedy = start.clone();
        search_with(&mut greedy, &mut Greedy, &bounded());
        let mut tabu = start.clone();
        search_with(&mut tabu, &mut Tabu::default(), &bounded());

        assert!(
            tabu.pair_count() <= greedy.pair_count(),
            "seed {seed}: tabu {} is worse than the descent's {}",
            tabu.pair_count(),
            greedy.pair_count(),
        );
        if tabu.pair_count() < greedy.pair_count() {
            better += 1;
        }
    }
    assert!(better > 0, "tabu matched the descent everywhere, so the corpus proves nothing");
}
