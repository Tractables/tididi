use super::*;
use crate::test_helpers::clause_to_tdd;
use crate::build::constant_one;


use crate::vtree::Vtree;
use num_bigint::BigUint;

fn balanced_vtree(n: u32) -> Arc<Vtree> {
    Arc::new(Vtree::balanced(n))
}

// ── Explicit `expand_full` tests ─────────────────────────────────────────

#[test]
fn expand_full_constant_one() {
    let eng = &crate::Engine::new();
    let vtree = balanced_vtree(4);
    let mut tdd = constant_one(eng, &vtree);
    expand_full(&crate::Engine::new(), &mut tdd).unwrap();
}

#[test]
fn expand_full_single_clause() {
    let eng = &crate::Engine::new();
    let vtree = balanced_vtree(4);
    let mut tdd = clause_to_tdd(eng, &vtree, &crate::test_helpers::clause(&[(1, true), (2, false)]));
    let count_before = tdd.model_count().unwrap();
    expand_full(&crate::Engine::new(), &mut tdd).unwrap();
    assert_eq!(count_before, tdd.model_count().unwrap());
}

#[test]
fn expand_full_preserves_determinism() {
    let eng = &crate::Engine::new();
    let vtree = balanced_vtree(4);
    let mut tdd = clause_to_tdd(eng, &vtree, &crate::test_helpers::clause(&[(1, true), (3, false)]));
    tdd.minimize().unwrap();
    let widths_before: Vec<usize> = tdd.levels.iter().map(|l| l.slot_count()).collect();
    expand_full(&crate::Engine::new(), &mut tdd).unwrap();

    let counts = crate::test_helpers::node_counts(&tdd);
    let mut subtree_vars = vec![0u32; vtree.num_nodes()];
    for (t, _var) in vtree.leaf_bottomup() {
        subtree_vars[t.idx()] = 1;
    }
    for (t, left, right) in vtree.internal_bottomup() {
        subtree_vars[t.idx()] = subtree_vars[left.idx()] + subtree_vars[right.idx()];
    }
    for (t, _left, _right) in vtree.internal_bottomup() {
        let ti = t.idx();
        if tdd.levels[ti].slot_count() <= widths_before[ti] { continue; }
        let expected = BigUint::from(1u32) << subtree_vars[ti] as usize;
        let total: BigUint = counts[ti].iter().sum();
        assert_eq!(total, expected);
    }
}

// ── Randomized sweeps ────────────────────────────────────────────────────

/// Every internal structural level of `tdd` covers every cell of its
/// `lefts × rights` basis. Returns the first level that does not.
fn first_not_full(tdd: &Tdd) -> Option<(usize, usize, usize)> {
    let vtree = Arc::clone(tdd.vtree());
    for (t, left, right) in vtree.internal_bottomup() {
        let lefts = ChildBasis::of(&vtree, &tdd.levels, left);
        let rights = ChildBasis::of(&vtree, &tdd.levels, right);
        let level = &tdd.levels[t.idx()];
        let mut seen = vec![false; lefts.len() * rights.len()];
        for node in &level.nodes {
            if !node.is_internal() {
                continue;
            }
            for pair in level.pairs_of(node) {
                let (l, r) = (pair.left.0, pair.right.0);
                if lefts.contains(l) && rights.contains(r) {
                    seen[(l - lefts.start) as usize * rights.len() + (r - rights.start) as usize] = true;
                }
            }
        }
        let covered = seen.iter().filter(|&&b| b).count();
        if covered != seen.len() {
            return Some((t.idx(), covered, seen.len()));
        }
    }
    None
}

#[test]
fn expand_full_fills_every_level_on_random_formulas() {
    use crate::test_helpers::{compile_clauses_on, rand_cnf, CnfShape, Lcg};
    let eng = &crate::Engine::new();
    let mut rng = Lcg::new(0x9e37_79b9);
    for num_vars in [3u32, 5, 8] {
        for (_name, vtree) in crate::test_helpers::vtree_shapes(num_vars) {
            for _ in 0..12 {
                let clauses = rand_cnf(&mut rng, num_vars, CnfShape { clauses: 6, width: 3 });
                let mut tdd = compile_clauses_on(eng, &vtree, &clauses);
                if tdd.is_zero() {
                    continue;
                }
                let before = tdd.model_count().unwrap();
                expand_full(eng, &mut tdd).unwrap();
                assert_eq!(before, tdd.model_count().unwrap(), "expand_full changed the count");
                assert_eq!(first_not_full(&tdd), None, "a level is not full: {clauses:?}");
            }
        }
    }
}

#[test]
fn negate_matches_the_truth_table_on_random_formulas() {
    use crate::test_helpers::{compile_clauses_on, eval, rand_cnf, CnfShape, Lcg};
    let eng = &crate::Engine::new();
    let mut rng = Lcg::new(0x5f37_59df);
    for num_vars in [3u32, 5, 7] {
        for (_name, vtree) in crate::test_helpers::vtree_shapes(num_vars) {
            for _ in 0..12 {
                let clauses = rand_cnf(&mut rng, num_vars, CnfShape { clauses: 6, width: 3 });
                let f = compile_clauses_on(eng, &vtree, &clauses);
                let not_f = eng.negate(f.clone()).unwrap();
                for mask in 0u32..(1 << num_vars) {
                    let asn: Vec<bool> = (0..num_vars).map(|i| mask >> i & 1 == 1).collect();
                    assert_eq!(
                        eval(&not_f, &asn),
                        !eval(&f, &asn),
                        "negate disagrees at {asn:?} on {clauses:?}"
                    );
                }
                // Double negation returns the function.
                let back = eng.negate(not_f).unwrap();
                assert!(crate::test_helpers::equiv(&f, &back), "!!f != f on {clauses:?}");
            }
        }
    }
}
