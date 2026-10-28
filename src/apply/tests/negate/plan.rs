//! The reduction a negation ends with, chosen explicitly.

use crate::Tdd;
use crate::reduce::ReductionPlan;

/// Every node's pairs, level by level: two diagrams agree here iff they are
/// the same diagram.
pub(super) fn shape(t: &Tdd) -> String {
    let mut s = format!("out={}:{}", t.output.vtree.idx(), t.output.local.idx());
    for (i, lv) in t.levels.iter().enumerate() {
        if lv.slot_count() == 0 || lv.is_marginal() {
            continue;
        }
        s.push_str(&format!("|L{i}"));
        for j in 0..lv.slot_count() {
            s.push_str(&format!(";{j}:"));
            for p in lv.pairs_vec(j) {
                s.push_str(&format!("({},{})", p.left.0, p.right.0));
            }
        }
    }
    s
}

#[test]
fn a_prune_only_negation_minimizes_to_the_negation() {
    use crate::test_helpers::{CnfShape, Lcg, compile_clauses_on, rand_cnf};
    let eng = &crate::Engine::new();
    let mut rng = Lcg::new(0x6c07_8965);
    let mut cases = 0usize;
    for num_vars in [3u32, 5, 8] {
        for (_name, vtree) in crate::test_helpers::vtree_shapes(num_vars) {
            for _ in 0..12 {
                let ca = rand_cnf(&mut rng, num_vars, CnfShape { clauses: 6, width: 3 });
                let cb = rand_cnf(&mut rng, num_vars, CnfShape { clauses: 3, width: 2 });
                let f = compile_clauses_on(eng, &vtree, &ca);
                let g = compile_clauses_on(eng, &vtree, &cb);
                if f.is_zero() || g.is_zero() {
                    continue;
                }
                // Both a minimized operand and an unminimized conjunction, the
                // shape a Tp-compilation consumer actually negates.
                let conj = crate::and(f.clone(), g).unwrap();
                for operand in [f, conj] {
                    if operand.is_zero() {
                        continue;
                    }
                    cases += 1;
                    let full = eng.negate(operand.clone()).unwrap();
                    let mut pruned = eng.negate_with(operand, ReductionPlan::Prune).unwrap();
                    assert!(
                        pruned.equivalent(&full).unwrap(),
                        "prune-only changed the function"
                    );
                    assert!(pruned.pair_count() >= full.pair_count());
                    eng.minimize(&mut pruned).unwrap();
                    assert_eq!(shape(&pruned), shape(&full), "minimizing afterwards did not agree");
                }
            }
        }
    }
    assert!(cases > 0, "nothing was tested");
}

#[test]
fn a_prune_only_negation_leaves_no_unreachable_node() {
    use crate::test_helpers::{CnfShape, Lcg, compile_clauses_on, rand_cnf};
    let eng = &crate::Engine::new();
    let mut rng = Lcg::new(0x3ff0_1234);
    for num_vars in [3u32, 6] {
        for (_name, vtree) in crate::test_helpers::vtree_shapes(num_vars) {
            for _ in 0..8 {
                let clauses = rand_cnf(&mut rng, num_vars, CnfShape { clauses: 5, width: 3 });
                let f = compile_clauses_on(eng, &vtree, &clauses);
                if f.is_zero() {
                    continue;
                }
                let mut pruned = eng.negate_with(f, ReductionPlan::Prune).unwrap();
                let before = pruned.node_count();
                eng.reduce(&mut pruned, ReductionPlan::Prune).unwrap();
                assert_eq!(before, pruned.node_count(), "a second prune still found something");
            }
        }
    }
}

/// A negation of a contracted operand that is not certified canonical owes
/// the contraction what its prune changed and the levels whose fill named
/// a node no pair named before, not every level: the next contraction finds
/// no twin anywhere else (debug builds check each negation), and minimizing
/// from those worklists reaches the canonical negation.
#[test]
fn a_negation_owes_the_contraction_only_what_it_changed() {
    use crate::diagram::Pass;
    use crate::test_helpers::{CnfShape, Lcg, compile_clauses_on, rand_cnf};
    let eng = &crate::Engine::new();
    let mut rng = Lcg::new(0x51ab_0e1d);
    let (mut cases, mut fewer) = (0usize, 0usize);
    for num_vars in [5u32, 8, 11] {
        for (_name, vtree) in crate::test_helpers::vtree_shapes(num_vars) {
            for _ in 0..10 {
                let f = compile_clauses_on(eng, &vtree, &rand_cnf(&mut rng, num_vars, CnfShape { clauses: 6, width: 3 }));
                let g = compile_clauses_on(eng, &vtree, &rand_cnf(&mut rng, num_vars, CnfShape { clauses: 3, width: 2 }));
                let mut conj = eng.and(f, g).unwrap();
                eng.reduce(&mut conj, ReductionPlan::Contract).unwrap();
                if conj.is_zero() || conj.levels.is_canonical(conj.output) {
                    continue;
                }
                cases += 1;
                let full = eng.negate(&conj).unwrap();
                for borrowed in [true, false] {
                    let mut neg = if borrowed {
                        eng.negate_with(&conj, ReductionPlan::Prune).unwrap()
                    } else {
                        eng.negate_with(conj.clone(), ReductionPlan::Prune).unwrap()
                    };
                    let mut owed = neg.dirty.levels(Pass::Contract).to_vec();
                    owed.sort_unstable();
                    owed.dedup();
                    fewer += usize::from(owed.len() < vtree.internal_bottomup().count());
                    eng.minimize(&mut neg).unwrap();
                    assert_eq!(shape(&neg), shape(&full), "minimizing from the negation's worklists did not agree");
                }
            }
        }
    }
    assert!(cases > 0 && fewer > 0, "cases {cases}, owing fewer levels {fewer}");
}
