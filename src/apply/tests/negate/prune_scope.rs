//! The prune a negation ends with, seeded from the root, against the walk over
//! the whole diagram.

use crate::diagram::ChildPair;
use crate::reduce::prune::{PruneScope, below_root_walk_applies};
use crate::reduce::ReductionPlan;
use crate::test_helpers::{CnfShape, Lcg, compile_clauses_on, rand_cnf, vtree_shapes};
use crate::{Engine, Tdd};

use super::plan::shape;

/// Negate `f`, ending with `plan` under `scope`. The seeded scope is what
/// `Engine::negate_with` uses; `Whole` is the same negation with the walk that
/// assumes nothing.
fn negate_scoped(eng: &Engine, f: Tdd, plan: ReductionPlan<'_>, scope: PruneScope) -> Tdd {
    let mut result = crate::apply::negate::negate_tdd_owned(eng, f).unwrap();
    assert!(
        below_root_walk_applies(&result) || result.is_zero(),
        "the seeded walk has to be the one under test"
    );
    eng.reduce_scoped(&mut result, plan, scope).unwrap();
    result
}

/// Both scopes on the same operand, under both plans: the diagrams agree node
/// for node and pair for pair, not merely as functions.
fn assert_same_negation(eng: &Engine, f: &Tdd) {
    for prune_only in [true, false] {
        let plan = || if prune_only { ReductionPlan::Prune } else { ReductionPlan::default() };
        let seeded = negate_scoped(eng, f.clone(), plan(), PruneScope::BelowRoot);
        let whole = negate_scoped(eng, f.clone(), plan(), PruneScope::Whole);
        assert_eq!(shape(&seeded), shape(&whole), "the seeded prune left a different diagram");
        assert_eq!(seeded.node_count(), whole.node_count());
        assert_eq!(seeded.pair_count(), whole.pair_count());
        assert_eq!(seeded.model_count().unwrap(), whole.model_count().unwrap());
    }
}

#[test]
fn the_seeded_prune_leaves_the_diagram_the_whole_walk_leaves() {
    let eng = &Engine::new();
    let mut rng = Lcg::new(0x51ed_2b17);
    let mut cases = 0usize;
    for num_vars in [3u32, 5, 8] {
        for (_name, vtree) in vtree_shapes(num_vars) {
            for _ in 0..12 {
                let ca = rand_cnf(&mut rng, num_vars, CnfShape { clauses: 6, width: 3 });
                let cb = rand_cnf(&mut rng, num_vars, CnfShape { clauses: 3, width: 2 });
                let f = compile_clauses_on(eng, &vtree, &ca);
                let g = compile_clauses_on(eng, &vtree, &cb);
                if f.is_zero() || g.is_zero() {
                    continue;
                }
                // A minimized operand and an unminimized conjunction: the two
                // shapes a Tp-compilation consumer negates.
                let conj = crate::and(f.clone(), g).unwrap();
                for operand in [f, conj] {
                    if operand.is_zero() {
                        continue;
                    }
                    cases += 1;
                    assert_same_negation(eng, &operand);
                }
            }
        }
    }
    assert!(cases > 100, "expected a corpus, got {cases} cases");
}

/// The seeded walk stops at the first level that loses nothing, which is sound
/// because `expand_full` leaves every node named from the level above. An
/// operand that arrives with unreachable nodes of its own does not change that:
/// the fill covers their columns too, so they are reachable in the complement
/// and neither walk removes them. The two scopes have to agree on such an
/// operand as well.
#[test]
fn an_operand_with_unreachable_nodes_negates_the_same_under_either_scope() {
    let eng = &Engine::new();
    let mut rng = Lcg::new(0x2c9a_0f41);
    let mut cases = 0usize;
    for num_vars in [4u32, 6] {
        for (_name, vtree) in vtree_shapes(num_vars) {
            for _ in 0..8 {
                let clauses = rand_cnf(&mut rng, num_vars, CnfShape { clauses: 5, width: 3 });
                let mut f = compile_clauses_on(eng, &vtree, &clauses);
                if f.is_zero() {
                    continue;
                }
                // An orphan: a copy of a node that no parent names.
                let mut orphaned = false;
                for t in 0..f.levels.len() {
                    if f.levels[t].is_marginal() || f.levels[t].slot_count() == 0 {
                        continue;
                    }
                    let pairs: Vec<ChildPair> = f.levels[t].pairs_of_idx(0).to_vec();
                    if pairs.is_empty() {
                        continue;
                    }
                    f.levels[t].push_node(eng.limits(), &pairs).unwrap();
                    orphaned = true;
                    break;
                }
                if !orphaned {
                    continue;
                }
                cases += 1;
                assert_same_negation(eng, &f);
            }
        }
    }
    assert!(cases > 10, "expected a corpus, got {cases} cases");
}
