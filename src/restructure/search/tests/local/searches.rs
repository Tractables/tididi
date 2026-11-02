use super::*;
use crate::vtree::Vtree;
use std::sync::Arc;

use crate::test_helpers::{assert_canonical, compile_clauses, literals};

/// The size-objective descent, spelled once for the tests below.
fn size_descent(tdd: &mut Tdd) -> RotationSearchStats {
    crate::Engine::new()
        .rotation_search(tdd, &mut MinimizePairs, &RotationSearchConfig::default())
        .expect("an unarmed engine stops nothing")
}

fn level_snapshot(tdd: &Tdd) -> Vec<crate::diagram::TddLevel> {
    tdd.levels.to_vec()
}

fn six_var_cycle() -> (Arc<Vtree>, Tdd) {
    let vtree = Arc::new(Vtree::balanced(6));
    let tdd = compile_clauses(
        &vtree,
        &[
            vec![1, 2],
            vec![-2, 3],
            vec![3, 4],
            vec![-4, 5],
            vec![5, -6],
            vec![1, -6],
        ],
    );
    (vtree, tdd)
}

/// The size search preserves the model count, never grows the diagram, and a
/// second call is a fixpoint (accepts nothing).
#[test]
fn size_search_preserves_count_shrinks_and_is_idempotent() {
    let (_vtree, mut tdd) = six_var_cycle();
    let mc_before = tdd.model_count().unwrap();
    let size_before = tdd.pair_count();

    let _ = size_descent(&mut tdd);
    assert_canonical(&tdd);
    assert_eq!(
        mc_before,
        tdd.model_count().unwrap(),
        "rotation search must preserve #F"
    );
    assert!(
        tdd.pair_count() <= size_before,
        "{} > {size_before}",
        tdd.pair_count()
    );

    let stats2 = size_descent(&mut tdd);
    assert_eq!(
        stats2.accepts, 0,
        "second search must accept nothing at a local minimum"
    );
}

/// The public rotation search must not panic — and must preserve the model
/// count — when handed a diagram compiled clause-by-clause without an
/// intervening full minimize (`Tdd::one` + `and_clause`). `and_clause` only
/// rebuilds the clause spine; it does not run a global twin contraction, so a
/// rotation on a non-canonical diagram can expose an unresolved twin at a
/// level above `w_idx`. The contract cascade is count-exact regardless of
/// where it fires, so the count must survive the whole search.
#[test]
fn rotation_search_on_non_canonical_clause_build_preserves_count() {
    for cnf in [
        vec![vec![1, -2], vec![2, 3], vec![-1, 3]],
        vec![vec![1, -2], vec![2, 3], vec![-1, 3], vec![4, -3]],
    ] {
        let vtree = Arc::new(Vtree::balanced(4));
        let mut acc = Tdd::one(&vtree);
        for clause in &cnf {
            acc = acc.and_clause(literals(clause)).unwrap();
        }
        let count_before = acc.model_count().unwrap();
        let _ = size_descent(&mut acc);
        assert_eq!(count_before, acc.model_count().unwrap(), "cnf={cnf:?}");
    }
}

#[test]
fn reject_all_objective_leaves_tdd_untouched() {
    struct RejectAll;
    impl RotationObjective for RejectAll {
        fn delta(&mut self, _: &RotationProbe<'_>) -> i64 {
            0
        }
    }

    let (vtree, mut tdd) = six_var_cycle();
    let mc_before = tdd.model_count().unwrap();
    let snap = level_snapshot(&tdd);

    let stats = crate::Engine::new()
        .rotation_search(&mut tdd, &mut RejectAll, &RotationSearchConfig::default())
        .expect("an unarmed engine stops nothing");

    assert!(Arc::ptr_eq(tdd.vtree(), &vtree));
    assert_eq!(stats.accepts, 0);
    assert!(stats.probes > 0);
    assert_eq!(mc_before, tdd.model_count().unwrap());
    let after = level_snapshot(&tdd);
    assert_eq!(snap.len(), after.len());
    for (a, b) in snap.iter().zip(&after) {
        assert!(a.nodes == b.nodes && a.pairs == b.pairs);
    }
    assert_canonical(&tdd);
}

#[test]
fn a_search_keeps_its_private_tree_between_rejected_probes() {
    struct Reject;
    impl ProbeRule for Reject {
        fn keeps(&mut self, _: &RotationProbe<'_>, _: &crate::vtree::rotate::RotationInfo) -> bool {
            false
        }
    }
    let eng = Engine::new();
    let tree = Arc::new(Vtree::balanced(8));
    let mut f = Tdd::one(&tree);
    assert_canonical(&f);
    {
        let mut search = crate::restructure::search::SearchTree::new(&mut f);
        let private = Arc::as_ptr(search.tdd.vtree());
        assert_ne!(private, Arc::as_ptr(&tree));
        let mut scratch = RestructureScratch::default();
        for kind in [RotationKind::Left, RotationKind::Right] {
            let turn = RotationMove {
                pivot: tree.root(),
                kind,
            };
            assert!(
                !search
                    .probe_moves(&eng, &[turn], &mut Reject, &mut scratch, usize::MAX)
                    .unwrap()
            );
            assert_eq!(private, Arc::as_ptr(search.tdd.vtree()));
        }
    }
    assert!(Arc::ptr_eq(f.vtree(), &tree));
    assert_canonical(&f);
}

#[test]
fn a_panicking_search_restores_shared_tree_identity() {
    struct Panic;
    impl RotationObjective for Panic {
        fn delta(&mut self, _: &RotationProbe<'_>) -> i64 {
            panic!("objective failed")
        }
    }
    let tree = Arc::new(Vtree::balanced(8));
    let mut f = Tdd::one(&tree);
    assert_canonical(&f);
    f.minimize().unwrap();
    let before = format!("{f:?}");
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        Engine::new().rotation_search(&mut f, &mut Panic, &RotationSearchConfig::default())
    }));
    assert!(result.is_err());
    assert!(Arc::ptr_eq(f.vtree(), &tree));
    assert_eq!(format!("{f:?}"), before);
    assert_canonical(&f);
}
