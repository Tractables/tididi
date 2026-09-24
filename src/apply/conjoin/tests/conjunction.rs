use super::*;
use crate::Engine;
use crate::limits::LimitConfig;
// Named explicitly, not through the glob above, so it resolves however
// `conjoin` routes its own import of it.
use crate::test_helpers::assert_canonical;
use crate::test_helpers::clause_to_tdd;
use crate::build::constant_one;


use crate::diagram::Tdd;
use crate::diagram::Literal;
use crate::vtree::{VarId, Vtree};
use num_bigint::BigUint;

#[test]
fn test_apply_and_with_constant_one() {
    let eng = &crate::Engine::new();
    let vtree = Arc::new(Vtree::balanced(3));
    let one = constant_one(eng, &vtree);
    let clause = vec![Literal::pos(VarId(1))];
    let clause_tdd = clause_to_tdd(eng, &vtree, &clause);

    // 1 ∧ clause = clause (after minimize)
    let mut expected = clause_tdd.clone();
    let mut result = apply_and(one, clause_tdd);
    result.minimize().unwrap();
    expected.minimize().unwrap();
    assert_canonical(&result);
    // Both should have the same model count (2^2 = 4 models satisfying x0)
    assert_eq!(result.model_count().unwrap(), expected.model_count().unwrap());

    // Structural check (not just count-equality): a count-preserving bug that
    // returns a re-canonicalized-but-DIFFERENT diagram for `1 ∧ clause` would
    // still pass the `model_count` assertion above. `is_self_conjunction`
    // (the same predicate `apply_and` itself consults for its structural
    // shortcut, `conjoin/sparse/mod.rs`) is a genuine canonical-equality
    // check: after `minimize`, two operands representing the same function
    // on the same vtree must have identical output + identical per-level
    // nodes/pairs/ranges (diagram canonicity). Neither operand here carries a
    // marginal level, so the check is meaningful (see its doc comment for
    // the marginal-level caveat).
    assert!(
        is_self_conjunction(&result, &expected),
        "1 \u{2227} clause must be structurally identical (canonical form) to clause alone, \
         not just count-equal"
    );
}

#[test]
fn test_apply_and_two_clauses() {
    let eng = &crate::Engine::new();
    let vtree = Arc::new(Vtree::balanced(3));
    let f = vec![Literal::pos(VarId(1))];
    let g = vec![Literal::neg(VarId(2))];

    let t1 = clause_to_tdd(eng, &vtree, &f);
    let t2 = clause_to_tdd(eng, &vtree, &g);
    let mut result = apply_and(t1, t2);
    result.minimize().unwrap();
    assert_canonical(&result);
    // x0=1 AND x1=0: 2 models (x2 can be 0 or 1)
    assert_eq!(result.model_count().unwrap(), BigUint::from(2u32));
}

#[test]
fn test_apply_and_contradictory() {
    let eng = &crate::Engine::new();
    // x0 ∧ ¬x0 should have no models
    let vtree = Arc::new(Vtree::balanced(1));
    let f = vec![Literal::pos(VarId(1))];
    let g = vec![Literal::neg(VarId(1))];

    let t1 = clause_to_tdd(eng, &vtree, &f);
    let t2 = clause_to_tdd(eng, &vtree, &g);
    let mut result = apply_and(t1, t2);
    result.minimize().unwrap();
    assert_canonical(&result);
    assert_eq!(result.model_count().unwrap(), BigUint::ZERO);
}


#[test]
fn test_apply_and_self_conjunction() {
    let eng = &crate::Engine::new();
    // f ∧ f = f for a non-trivial diagram.
    let vtree = Arc::new(Vtree::balanced(4));
    let f = vec![Literal::pos(VarId(1)), Literal::pos(VarId(3))];
    let g = vec![Literal::neg(VarId(2)), Literal::pos(VarId(4))];
    let mut tdd = clause_to_tdd(eng, &vtree, &f);
    let t2 = clause_to_tdd(eng, &vtree, &g);
    tdd = apply_and(tdd, t2);
    tdd.minimize().unwrap();

    let expected_mc = tdd.model_count().unwrap();
    let expected_size = tdd.pair_count();

    // Conjoin with a clone of itself.
    let copy = tdd.clone();
    let mut result = apply_and(tdd, copy);
    result.minimize().unwrap();

    assert_canonical(&result);
    assert_eq!(result.model_count().unwrap(), expected_mc);
    assert_eq!(result.pair_count(), expected_size);
}

#[test]
fn test_apply_and_self_conjunction_owned() {
    let eng = &crate::Engine::new();
    // f ∧ f = f via the owned variant (avoids clone).
    let vtree = Arc::new(Vtree::balanced(4));
    let f = vec![Literal::pos(VarId(1)), Literal::neg(VarId(3))];
    let g = vec![Literal::pos(VarId(2)), Literal::pos(VarId(4))];
    let mut tdd = clause_to_tdd(eng, &vtree, &f);
    let t2 = clause_to_tdd(eng, &vtree, &g);
    tdd = apply_and(tdd, t2);
    tdd.minimize().unwrap();

    let expected_mc = tdd.model_count().unwrap();
    let copy = tdd.clone();
    let mut result = apply_and(tdd, copy);
    result.minimize().unwrap();

    assert_canonical(&result);
    assert_eq!(result.model_count().unwrap(), expected_mc);
}

#[test]
fn test_apply_and_stick_vtree_reachability() {
    let eng = &crate::Engine::new();
    // Exercise the top-down reachability path on a stick (right-linear) vtree.
    // On sticks, every internal level has a leaf left child (3×3 grid),
    // triggering reachability gating at every level.
    //
    // Build diagrams from individual clauses (not compile_cnf) to keep both on
    // the same vtree Arc — compile_cnf grafts multi-component formulas.
    let vtree = Arc::new(Vtree::linear(8));

    let clauses1 = [
        vec![Literal::pos(VarId(1)), Literal::neg(VarId(3))],
        vec![Literal::neg(VarId(2)), Literal::pos(VarId(4))],
        vec![Literal::pos(VarId(5)), Literal::neg(VarId(6))],
        vec![Literal::neg(VarId(7)), Literal::pos(VarId(8))],
    ];
    let mut f = constant_one(eng, &vtree);
    for clause in &clauses1 {
        let cl = clause_to_tdd(eng, &vtree, clause);
        f = apply_and(f, cl);
        f.minimize().unwrap();
    }

    let clauses2 = [
        vec![Literal::neg(VarId(1)), Literal::pos(VarId(2))],
        vec![Literal::pos(VarId(3)), Literal::pos(VarId(5))],
        vec![Literal::neg(VarId(4)), Literal::neg(VarId(6))],
        vec![Literal::pos(VarId(7)), Literal::neg(VarId(8))],
    ];
    let mut g = constant_one(eng, &vtree);
    for clause in &clauses2 {
        let cl = clause_to_tdd(eng, &vtree, clause);
        g = apply_and(g, cl);
        g.minimize().unwrap();
    }

    let mut result = apply_and(f, g);
    result.minimize().unwrap();
    assert_canonical(&result);

    // Brute-force: count assignments satisfying both formulas.
    let expected = (0u64..256).filter(|&a| {
        let v = |i: u32| (a >> i) & 1 == 1;
        (v(0) || !v(2)) && (!v(1) || v(3)) && (v(4) || !v(5)) && (!v(6) || v(7))
        && (!v(0) || v(1)) && (v(2) || v(4)) && (!v(3) || !v(5)) && (v(6) || !v(7))
    }).count();

    assert_eq!(result.model_count().unwrap(), BigUint::from(expected as u64));
}

/// A canonical marginal operand cannot meet another that still constrains its discarded structure.
#[test]
fn conjunction_rejects_a_constrained_marginal_level() {
    let eng = crate::Engine::new();
    let tree = Arc::new(Vtree::balanced(4));
    let left = tree.children(tree.root()).0;
    let mut marginal = Tdd::clause(&tree, [1, 3]).unwrap();
    eng.marginalize_levels(&mut marginal, &[left]).unwrap();
    crate::test_helpers::assert_canonical(&marginal);
    let structural = Tdd::clause(&tree, [1, 2]).unwrap();
    crate::test_helpers::assert_canonical(&structural);
    for (f, g) in [(&marginal, &structural), (&structural, &marginal)] {
        assert_eq!(eng.and(f.clone(), g.clone()).unwrap_err(), OperationError::MarginalLevel(left));
    }
}

/// Regression for the segment-conjoin output-size cap: when
/// `LimitConfig::output_node_cap` is armed, `apply_and_fallible` must abort with
/// `OperationError::OutputCap` the moment the cumulative output node count crosses
/// the cap — a deliberate cut of a product ballooning past the intended limit,
/// distinguishable from an OOM by the typed variant itself. The same conjoin
/// with no cap must complete. On unfixed `main` (no cap machinery) the armed
/// run would also complete — the cap is what makes it bail.
#[test]
fn test_apply_output_node_cap_bails_cleanly() {
    // Build a diagram by folding clauses with `apply_and` — every operand shares the
    // same `vtree` Arc (`clause_to_tdd` / `constant_one` clone it), so the final
    // conjoin's pointer-identical-vtree precondition holds.
    fn build(vtree: &Arc<Vtree>, clauses: &[&[i32]]) -> Tdd {
        let eng = &crate::Engine::new();
        let mut acc = constant_one(eng, vtree);
        for literals in clauses {
            let clause: Vec<Literal> = literals.iter()
                .map(|&l| Literal::new(VarId(l.unsigned_abs()), l > 0))
                .collect();
            let c = clause_to_tdd(eng, vtree, &clause);
            acc = apply_and(acc, c);
        }
        acc
    }

    // Two non-trivial functions whose conjunction has a multi-node intermediate
    // diagram (well above a 1-node cap).
    let vtree = Arc::new(Vtree::balanced(14));
    // Cross-linking clauses over interleaved variables force a wide intermediate
    // diagram on the balanced vtree (hundreds of nodes), so a modest cap clearly
    // trips mid-build while the uncapped conjoin completes.
    let fa: &[&[i32]] = &[
        &[1, 8], &[2, 9], &[3, 10], &[4, 11], &[5, 12], &[6, 13], &[7, 14],
        &[-1, -9], &[-2, -10], &[-3, -11], &[-4, -12], &[-5, -13], &[-6, -14],
    ];
    let fb: &[&[i32]] = &[
        &[1, -8], &[2, -9], &[3, -10], &[4, -11], &[5, -12], &[6, -13], &[7, -14],
        &[8, 2], &[9, 3], &[10, 4], &[11, 5], &[12, 6], &[13, 7],
    ];

    // Control: no cap → the conjoin completes.
    let mut a = build(&vtree, fa);
    let mut b = build(&vtree, fb);
    let uncapped = {
        let eng = Engine::new();
        apply_and_fallible(&eng, &mut a, &mut b, VtreeMask::default(), VtreeMask::default(), None)
    };
    assert!(uncapped.is_ok(), "no cap: conjoin should complete, got {:?}", uncapped.err());

    // Armed with a tiny cap → the apply bails as a deliberate output cut.
    let mut a = build(&vtree, fa);
    let mut b = build(&vtree, fb);
    let capped = {
        let eng = Engine::new();
        let _prior = eng.limits().install(LimitConfig::none().with_output_node_cap(Some(1)));
        apply_and_fallible(&eng, &mut a, &mut b, VtreeMask::default(), VtreeMask::default(), None)
    };
    assert_eq!(
        capped.err(),
        Some(OperationError::OutputCap),
        "tiny cap: apply must bail OutputCap once output exceeds the cap",
    );
}

#[test]
fn conjunction_checks_the_final_root_on_the_preallocated_path() {
    use crate::{Engine, OperationError, Tdd};
    use crate::limits::LimitConfig;
    let tree = Arc::new(Vtree::balanced(2));
    let f = Tdd::clause(&tree, [1, 2]).unwrap();
    let g = Tdd::clause(&tree, [1, -2]).unwrap();
    let eng = Engine::new();
    {
        let _scope = eng
            .limits()
            .scope(LimitConfig::none().with_output_node_cap(Some(0)));
        assert_eq!(
            eng.and(f.clone(), g.clone()).unwrap_err(),
            OperationError::OutputCap
        );
    }
    let _scope = eng
        .limits()
        .scope(LimitConfig::none().with_output_node_cap(Some(1)));
    let result = eng.and(f, g).unwrap();
    for row in 0..4 {
        assert_eq!(crate::test_helpers::eval(&result, &[row & 1 != 0, row & 2 != 0]), row & 1 != 0);
    }
}
