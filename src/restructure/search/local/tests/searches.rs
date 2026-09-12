use super::*;
use std::sync::Arc;
use crate::vtree::Vtree;
use crate::query::model_count;
use crate::test_helpers::{assert_canonical, compile_clauses, literals};

/// The size-objective descent, spelled once for the tests below.
fn size_descent(tdd: &mut Tdd) -> RotationSearchStats {
    crate::engine::Engine::new()
        .rotation_search(tdd, &mut SizeDelta, &RotationSearchConfig::default())
        .expect("an unarmed engine stops nothing")
}

fn level_snapshot(tdd: &Tdd) -> Vec<(Vec<crate::diagram::EncodedNode>, Vec<crate::diagram::ChildPair>)> {
    tdd.levels.iter().map(|l| (l.nodes.clone(), l.pairs.clone())).collect()
}

/// The size search preserves the model count, never grows the diagram, and a
/// second call is a fixpoint (accepts nothing).
#[test]
fn size_search_preserves_count_shrinks_and_is_idempotent() {
    let vtree = Arc::new(Vtree::balanced(6));
    let mut tdd = compile_clauses(
        &vtree,
        &[
            vec![1, 2], vec![-2, 3], vec![3, 4],
            vec![-4, 5], vec![5, -6], vec![1, -6],
        ],
    );
    let mc_before = model_count(&tdd);
    let size_before = tdd.pair_count();

    let stats = size_descent(&mut tdd);

    assert_canonical(&tdd);
    assert_eq!(mc_before, model_count(&tdd), "rotation search must preserve #F");
    assert!(
        tdd.pair_count() <= size_before,
        "size search must not increase size ({} > {})",
        tdd.pair_count(), size_before,
    );

    // Fixpoint idempotence: already at a local minimum ⇒ no further accepts.
    let stats2 = size_descent(&mut tdd);
    assert_eq!(stats2.accepts, 0, "second search must accept nothing at a local minimum");
    let _ = stats; // stats.probes/accepts/sweeps are informational here.
}

/// Regression: the public rotation search must not panic — and must preserve
/// the model count — when handed a diagram compiled clause-by-clause WITHOUT an
/// intervening full minimize (the `Tdd::one` + `apply_and_clause` pattern
/// from the public api-guide). `apply_and_clause` only rebuilds the clause
/// spine; it does not run a global twin contraction, so the accumulator is
/// correct-count but NON-canonical (residual twins survive). A rotation on a
/// non-canonical diagram can expose an unresolved twin at a level *above*
/// `w_idx` — twin equivalence is semantic, so the single-level locality
/// tightening holds only for a canonical input. The contract cascade is
/// count-exact regardless of where it fires, so the count must survive the
/// whole search.
#[test]
fn rotation_search_on_non_canonical_clause_build_preserves_count() {
    use crate::apply::apply_and_clause;

    // Both reproducer CNFs from the onboarding bug report, over balanced(4).
    for cnf in [
        vec![vec![1, -2], vec![2, 3], vec![-1, 3]],
        vec![vec![1, -2], vec![2, 3], vec![-1, 3], vec![4, -3]],
    ] {
        let vtree = Arc::new(Vtree::balanced(4));
        let mut acc = Tdd::one(&vtree);
        for clause in &cnf {
            let lits = literals(clause);
            acc = apply_and_clause(acc, &lits);
        }
        let count_before = model_count(&acc);

        // Must not panic.
        let _ = size_descent(&mut acc);

        assert_eq!(
            count_before,
            model_count(&acc),
            "rotation search on a non-canonical clause-built TDD must preserve #F (cnf={cnf:?})",
        );
    }
}

/// A custom objective that never improves (delta ≥ 0 always) pins the generic
/// contract: the search accepts zero rotations and leaves the diagram
/// bit-identical.
#[test]
fn reject_all_objective_leaves_tdd_untouched() {
    struct RejectAll;
    impl RotationObjective for RejectAll {
        fn delta(&mut self, _before: (&TddLevel, &TddLevel), _after: (&TddLevel, &TddLevel)) -> i64 {
            0 // never < 0 ⇒ every probe reverts
        }
    }

    let vtree = Arc::new(Vtree::balanced(6));
    let mut tdd = compile_clauses(
        &vtree,
        &[
            vec![1, 2], vec![-2, 3], vec![3, 4],
            vec![-4, 5], vec![5, -6], vec![1, -6],
        ],
    );
    let mc_before = model_count(&tdd);
    let snap = level_snapshot(&tdd);

    let stats = crate::engine::Engine::new()
        .rotation_search(&mut tdd, &mut RejectAll, &RotationSearchConfig::default())
        .expect("an unarmed engine stops nothing");

    assert_eq!(stats.accepts, 0, "reject-all objective must accept nothing");
    assert!(stats.probes > 0, "test must actually exercise probes");
    assert_eq!(mc_before, model_count(&tdd), "count unchanged");
    assert_eq!(snap, level_snapshot(&tdd), "every probe must revert bit-identically");
    assert_canonical(&tdd);
}
