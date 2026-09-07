use super::*;
use crate::vtree::{Literal, VarId, Vtree};
use crate::tdd::build::clause_to_tdd;
use crate::tdd::transform::pairwise::conjoin::apply_and_both_owned;
use crate::tdd::minimize::minimize;
use crate::tdd::query::model_count;

fn make_clause(lits: &[i32]) -> Vec<Literal> {
    lits.iter()
        .map(|&l| Literal::new(VarId(l.unsigned_abs() - 1), l > 0))
        .collect()
}

fn compile(clauses: &[Vec<i32>], vtree: Arc<Vtree>) -> Tdd {
    let clauses: Vec<Vec<Literal>> = clauses.iter().map(|c| make_clause(c)).collect();
    let mut tdd: Option<Tdd> = None;
    for clause in &clauses {
        let c = clause_to_tdd(&vtree, clause);
        tdd = Some(match tdd {
            Some(acc) => {
                let mut r = apply_and_both_owned(acc, c);
                minimize(&mut r);
                r
            }
            None => c,
        });
    }
    let mut tdd = tdd.unwrap();
    minimize(&mut tdd);
    tdd
}

fn level_snapshot(tdd: &Tdd) -> Vec<(Vec<crate::tdd::types::TddNodeData>, Vec<crate::tdd::types::InputPair>)> {
    tdd.levels.iter().map(|l| (l.nodes.clone(), l.pairs.clone())).collect()
}

/// The size search preserves the model count, never grows the diagram, and a
/// second call is a fixpoint (accepts nothing).
#[test]
fn size_search_preserves_count_shrinks_and_is_idempotent() {
    let vtree = Arc::new(Vtree::balanced(6));
    let mut tdd = compile(
        &[
            vec![1, 2], vec![-2, 3], vec![3, 4],
            vec![-4, 5], vec![5, -6], vec![1, -6],
        ],
        vtree,
    );
    let mc_before = model_count(&tdd);
    let size_before = tdd.size();

    let stats = search_to_local_min(&mut tdd);

    assert_eq!(mc_before, model_count(&tdd), "rotation search must preserve #F");
    assert!(
        tdd.size() <= size_before,
        "size search must not increase size ({} > {})",
        tdd.size(), size_before,
    );

    // Fixpoint idempotence: already at a local minimum ⇒ no further accepts.
    let stats2 = search_to_local_min(&mut tdd);
    assert_eq!(stats2.accepts, 0, "second search must accept nothing at a local minimum");
    let _ = stats; // stats.probes/accepts/sweeps are informational here.
}

/// Regression: the public rotation search must not panic — and must preserve
/// the model count — when handed a TDD compiled clause-by-clause WITHOUT an
/// intervening full minimize (the `constant_one` + `apply_and_clause` pattern
/// from the public api-guide). `apply_and_clause` only rebuilds the clause
/// spine; it does not run a global twin contraction, so the accumulator is
/// correct-count but NON-canonical (residual twins survive). A rotation on a
/// non-canonical diagram can expose an unresolved twin at a level *above*
/// `w_idx` (§9 "Twin equivalence is semantic": the single-level locality
/// tightening holds only for a canonical input), which used to trip the
/// debug-only `contract_all_twins_with_locality` assertion in
/// `minimize_after_rotation`. The contract cascade is count-exact regardless
/// of where it fires, so the count must survive the whole search.
#[test]
fn rotation_search_on_non_canonical_clause_build_preserves_count() {
    use crate::tdd::build::constant_one;
    use crate::tdd::transform::pairwise::conjoin_clause::apply_and_clause;

    // Both reproducer CNFs from the onboarding bug report, over balanced(4).
    for cnf in [
        vec![vec![1, -2], vec![2, 3], vec![-1, 3]],
        vec![vec![1, -2], vec![2, 3], vec![-1, 3], vec![4, -3]],
    ] {
        let vtree = Arc::new(Vtree::balanced(4));
        let mut acc = constant_one(&vtree);
        for clause in &cnf {
            let lits = make_clause(clause);
            acc = apply_and_clause(&mut acc, &lits);
        }
        let count_before = model_count(&acc);

        // Must not panic.
        let _ = search_to_local_min(&mut acc);

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
    let mut tdd = compile(
        &[
            vec![1, 2], vec![-2, 3], vec![3, 4],
            vec![-4, 5], vec![5, -6], vec![1, -6],
        ],
        vtree,
    );
    let mc_before = model_count(&tdd);
    let snap = level_snapshot(&tdd);

    let stats = rotation_search(&mut tdd, &mut RejectAll, &RotationSearchConfig::default());

    assert_eq!(stats.accepts, 0, "reject-all objective must accept nothing");
    assert!(stats.probes > 0, "test must actually exercise probes");
    assert_eq!(mc_before, model_count(&tdd), "count unchanged");
    assert_eq!(snap, level_snapshot(&tdd), "every probe must revert bit-identically");
}
