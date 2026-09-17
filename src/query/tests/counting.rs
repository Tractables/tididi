use super::*;
use crate::test_helpers::stopping_engine;

#[test]
fn test_model_count_constant_one() {
    let eng = &crate::Engine::new();
    let vtree = Arc::new(Vtree::balanced(3));
    let tdd = constant_one(eng, &vtree);
    // 3 variables → 2^3 = 8 models
    assert_eq!(tdd.model_count().unwrap(), BigUint::from(8u32));
}

#[test]
fn test_model_count_single_positive_literal() {
    let eng = &crate::Engine::new();
    let vtree = Arc::new(Vtree::balanced(3));
    let clause = vec![Literal::pos(VarId(1))];
    let tdd = clause_to_tdd(eng, &vtree, &clause);
    // x0: satisfied when x0=1. 4 assignments for x1,x2 → 4 models
    assert_eq!(tdd.model_count().unwrap(), BigUint::from(4u32));
}

#[test]
fn test_model_count_two_literal_clause() {
    let eng = &crate::Engine::new();
    let vtree = Arc::new(Vtree::balanced(3));
    // x0 ∨ ¬x1: satisfied unless x0=0 and x1=1
    let clause = vec![
        Literal::pos(VarId(1)),
        Literal::neg(VarId(2)),
    ];
    let tdd = clause_to_tdd(eng, &vtree, &clause);
    // 8 - 2 = 6 models (2 assignments with x0=0,x1=1, times 2 for x2)
    assert_eq!(tdd.model_count().unwrap(), BigUint::from(6u32));
}

#[test]
fn test_model_count_conjunction() {
    let eng = &crate::Engine::new();
    let vtree = Arc::new(Vtree::balanced(3));
    // (x0) ∧ (x1): both must be true, x2 free → 2 models
    let f = vec![Literal::pos(VarId(1))];
    let g = vec![Literal::pos(VarId(2))];
    let t1 = clause_to_tdd(eng, &vtree, &f);
    let t2 = clause_to_tdd(eng, &vtree, &g);
    let result = apply_and(t1, t2);
    assert_eq!(result.model_count().unwrap(), BigUint::from(2u32));
}

#[test]
fn test_model_count_unsat() {
    let eng = &crate::Engine::new();
    let vtree = Arc::new(Vtree::balanced(1));
    // (x0) ∧ (¬x0) = UNSAT
    let f = vec![Literal::pos(VarId(1))];
    let g = vec![Literal::neg(VarId(1))];
    let t1 = clause_to_tdd(eng, &vtree, &f);
    let t2 = clause_to_tdd(eng, &vtree, &g);
    let result = apply_and(t1, t2);
    assert_eq!(result.model_count().unwrap(), BigUint::ZERO);
    assert!(!result.is_sat().unwrap());
}

#[test]
fn test_model_count_single_var() {
    let eng = &crate::Engine::new();
    let vtree = Arc::new(Vtree::balanced(1));
    let tdd = constant_one(eng, &vtree);
    assert_eq!(tdd.model_count().unwrap(), BigUint::from(2u32));
}

#[test]
fn test_model_count_clause_all_vars() {
    let eng = &crate::Engine::new();
    // 4 variables, clause x0 ∨ x1 ∨ x2 ∨ x3
    // Unsatisfied only when all are 0: 2^4 - 1 = 15 models
    let vtree = Arc::new(Vtree::balanced(4));
    let clause = vec![
        Literal::pos(VarId(1)),
        Literal::pos(VarId(2)),
        Literal::pos(VarId(3)),
        Literal::pos(VarId(4)),
    ];
    let tdd = clause_to_tdd(eng, &vtree, &clause);
    assert_eq!(tdd.model_count().unwrap(), BigUint::from(15u32));
}

// --- node_counts ---

#[test]
fn test_node_counts_basic() {
    let eng = &crate::Engine::new();
    let vtree = Arc::new(Vtree::balanced(3));
    let tdd = constant_one(eng, &vtree);
    let counts = node_counts(&tdd);
    // Output node should have count = 2^3 = 8
    let out_count = &counts[tdd.output.vtree.idx()][tdd.output.local.idx()];
    assert_eq!(*out_count, BigUint::from(8u32));
}

// --- Overflow tests ---

/// Differential invariant underpinning conditioning's false-output canonicalization:
/// `is_sat_structural(t)` must agree with `t.model_count()? != 0` for every diagram,
/// including non-canonical ⊥ (structurally-false output node that still carries pairs).
/// The canonicalization relies on this equivalence to collapse a dead diagram to ZERO
/// without ever changing a live count. Covers SAT, plain UNSAT, and a multi-conjoin
/// UNSAT that produces an in-range (non-root-stale) false output.
#[test]
fn test_output_is_satisfiable_agrees_with_model_count() {
    let eng = &crate::Engine::new();
    let check = |t: &Tdd, what: &str| {
        let sat = is_sat_structural(eng, t).unwrap();
        let nonzero = t.model_count().unwrap() != BigUint::ZERO;
        assert_eq!(sat, nonzero, "is_sat_structural disagrees with model_count>0 for {what}");
    };

    // SAT: tautology, single literal, satisfiable conjunction.
    check(&constant_one(eng, &Arc::new(Vtree::balanced(3))), "constant_one");
    let vtree = Arc::new(Vtree::balanced(3));
    check(&clause_to_tdd(eng, &vtree, &[Literal::pos(VarId(1))]), "single literal");
    {
        let t1 = clause_to_tdd(eng, &vtree, &[Literal::pos(VarId(1))]);
        let t2 = clause_to_tdd(eng, &vtree, &[Literal::pos(VarId(2))]);
        check(&apply_and(t1, t2), "x0 ∧ x1 (SAT)");
    }

    // UNSAT: direct contradiction.
    {
        let v1 = Arc::new(Vtree::balanced(1));
        let t1 = clause_to_tdd(eng, &v1, &[Literal::pos(VarId(1))]);
        let t2 = clause_to_tdd(eng, &v1, &[Literal::neg(VarId(1))]);
        check(&apply_and(t1, t2), "x0 ∧ ¬x0 (UNSAT)");
    }

    // UNSAT via a chain of conjoins over a wider vtree — exercises a deeper false output,
    // not just the root-stale-grid case the existing FALSE guard already caught.
    {
        let v = Arc::new(Vtree::balanced(4));
        let mut acc = clause_to_tdd(eng, &v, &[Literal::pos(VarId(1))]);
        for lit in [
            Literal::pos(VarId(2)),
            Literal::pos(VarId(3)),
            Literal::neg(VarId(1)), // contradicts the seed → UNSAT
        ] {
            let step = clause_to_tdd(eng, &v, &[lit]);
            acc = apply_and(acc, step);
        }
        check(&acc, "chained conjoin → UNSAT");
    }

    // Marginal level: its column comes from the summed counts (it reads no child
    // column at all), and its own parent reads it as a marginal child. Covers the
    // walk's column-release rule on both sides of a marginal level.
    {
        let v = Arc::new(Vtree::balanced(4));
        let marginal_root = (0..v.num_nodes())
            .find(|&vi| !v.node(VtreeIdx(vi as u32)).is_leaf() && vi != v.root().idx())
            .map(|vi| VtreeIdx(vi as u32))
            .expect("balanced(4) has a non-root internal node");
        let t1 = clause_to_tdd(eng, &v, &[Literal::pos(VarId(1)), Literal::pos(VarId(3))]);
        let t2 = clause_to_tdd(eng, &v, &[Literal::neg(VarId(2)), Literal::pos(VarId(4))]);
        let mut t = apply_and(t1, t2);
        crate::test_helpers::marginalize_subtree(&mut t, marginal_root);
        t.minimize().unwrap();
        assert!(
            t.levels[marginal_root.idx()].is_marginal(),
            "fixture must carry a marginal level"
        );
        check(&t, "marginalized subtree (SAT)");
    }
}

/// T3 — operand-consumption contract of `apply_and_fallible`.
///
/// `apply_and_fallible` drains dead operand-child levels in place as its
/// bottom-up loop ascends (`drop_dead_operand_level`), so a COMPLETED
/// conjoin leaves both operands consumed — every level below each root has been
/// stolen. This is the guaranteed, observable half of the "operands are
/// consumed / unspecified after the call" contract documented on
/// `apply_and_fallible` (the `Err` path is even less specified: an early
/// budget/cap trip may ascend little and leave operands nearly intact — which is
/// exactly why callers must never reuse operands and must rebuild from a clone).
#[test]
fn test_apply_fallible_consumes_operands() {
    let eng = Engine::new();
    // Fold clauses into a diagram; every operand shares the same vtree Arc so the
    // conjoin's pointer-identical-vtree precondition holds.
    fn build(eng: &Engine, vtree: &Arc<Vtree>, clauses: &[&[i32]]) -> Tdd {
        let mut acc = constant_one(eng, vtree);
        for clause in clauses {
            let c = clause_to_tdd(eng, vtree, &crate::test_helpers::literals(clause));
            acc = apply_and(acc, c);
        }
        acc
    }

    // Two multi-level functions over interleaved vars — a genuine multi-node
    // intermediate diagram, so the ascending loop drains many operand levels.
    let vtree = Arc::new(Vtree::balanced(14));
    let fa: &[&[i32]] = &[
        &[1, 8], &[2, 9], &[3, 10], &[4, 11], &[5, 12], &[6, 13], &[7, 14],
        &[-1, -9], &[-2, -10], &[-3, -11], &[-4, -12], &[-5, -13], &[-6, -14],
    ];
    let fb: &[&[i32]] = &[
        &[1, -8], &[2, -9], &[3, -10], &[4, -11], &[5, -12], &[6, -13], &[7, -14],
        &[8, 2], &[9, 3], &[10, 4], &[11, 5], &[12, 6], &[13, 7],
    ];

    let mut a = build(&eng, &vtree, fa);
    let mut b = build(&eng, &vtree, fb);
    let a_before = a.node_count();
    let b_before = b.node_count();
    assert!(a_before > 1 && b_before > 1, "operands should be multi-node to make consumption observable");

    // A completed (uncapped) conjoin: must succeed, and consume both operands.
    let result = apply_and_fallible(&eng, &mut a, &mut b, MarginalTargets::None);
    assert!(result.is_ok(), "uncapped conjoin should complete: {:?}", result.err());
    assert!(
        a.node_count() < a_before && b.node_count() < b_before,
        "a completed apply must consume both operands (drain levels below root) — \
         a: {a_before} -> {}, b: {b_before} -> {}",
        a.node_count(),
        b.node_count(),
    );
}

/// `Engine::model_count` is the counted model count: with nothing armed it
/// agrees with `model_count`, and under an armed stop it cuts instead of
/// running to the end.
#[test]
fn batch_model_count_matches_ordinary_count_and_honors_the_stop_axis() {
    let vtree = Arc::new(Vtree::balanced(3));
    let f = Tdd::clause(&vtree, [1, 2, 3]).unwrap();
    assert_eq!(
        Engine::new().model_count(&f).expect("nothing armed"),
        f.model_count().unwrap()
    );
    let stopped = stopping_engine();
    stopped.limits().pin_reduce_poll_stride(Some(1));
    assert!(matches!(
        stopped.model_count(&f),
        Err(crate::limits::OperationError::Stopped)
    ));
}



