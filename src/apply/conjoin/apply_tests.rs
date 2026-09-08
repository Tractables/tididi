use super::*;
use crate::engine::{Engine, LimitSet};
// Explicit (not just via the `use super::*` glob above): `is_self_conjunction`
// is `pub(super)` in the `sparse` submodule (= visible throughout `conjoin`
// and its descendants, which `apply_tests` is one of), so this path resolves
// regardless of how `conjoin::mod`'s own private re-import of it is routed.
use super::sparse::is_self_conjunction;
use crate::build::{clause_to_tdd, constant_one};
use crate::reduce::minimize;
use crate::query::model_count;
use crate::diagram::{
    InputPair, LeafLabel, NodeIdx, Tdd, TddNodeId,
    assert_can_make_marginal, take_levels,
};
use crate::diagram::Literal;
use crate::vtree::{VarId, Vtree, VtreeIdx, VtreeNode};
use num_bigint::BigUint;

/// PROBE (ignored; run explicitly): measure the TDD size of a dumped feasible
/// show-projection set over a VTREE, to test diagram-compactness in TiDiDi's
/// native class (vs the CUDD OBDD measurement, which only searches linear orders).
///
///   TIDIDI_TDD_MINTERM_FILE=path  (lines of 0/1, one minterm per line)
///   TIDIDI_TDD_MINTERM_LIMIT=N    (optional cap)
///   TIDIDI_TDD_MINTERM_VTREE=balanced|linear  (default balanced)
///   cargo test --release -p tididi tdd_minterm_compactness -- --ignored --nocapture
#[test]
#[ignore]
fn tdd_minterm_compactness() {
    let eng = &crate::engine::Engine::new();
    use crate::apply::apply_or;
    use std::time::Instant;

    let Ok(path) = std::env::var("TIDIDI_TDD_MINTERM_FILE") else {
        eprintln!("skipped: set TIDIDI_TDD_MINTERM_FILE=<path> to run this manual harness");
        return;
    };
    let limit: usize = std::env::var("TIDIDI_TDD_MINTERM_LIMIT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(usize::MAX);
    let which = std::env::var("TIDIDI_TDD_MINTERM_VTREE").unwrap_or_else(|_| "balanced".into());

    let raw = std::fs::read_to_string(&path).expect("read minterm file");
    let mut rows: Vec<Vec<bool>> = Vec::new();
    for line in raw.lines() {
        let l = line.trim();
        if l.is_empty() {
            continue;
        }
        rows.push(l.bytes().map(|b| b == b'1').collect());
        if rows.len() >= limit {
            break;
        }
    }
    let ncols = rows[0].len();
    let npts = rows.len();
    println!("loaded {} minterms, {} cols, vtree={}", npts, ncols, which);

    let vtree = match which.as_str() {
        "linear" => Arc::new(Vtree::linear(ncols as u32)),
        _ => Arc::new(Vtree::balanced(ncols as u32)),
    };

    // single-literal TDDs cached per (col, polarity)
    let lit_tdd = |col: usize, b: bool| {
        clause_to_tdd(eng, &vtree, &[Literal::new(VarId(col as u32), b)])
    };

    let start = Instant::now();
    // Build all cube TDDs, then OR-reduce pairwise (tournament): O(N) applies on
    // small intermediates instead of O(N) applies on one growing accumulator.
    let mut layer: Vec<Tdd> = rows
        .iter()
        .map(|row| {
            let mut cube = constant_one(eng, &vtree);
            for (col, &b) in row.iter().enumerate() {
                let lt = lit_tdd(col, b);
                cube = apply_and(cube, lt);
            }
            cube
        })
        .collect();
    println!("  built {} cube TDDs in {:?}", layer.len(), start.elapsed());
    let mut round = 0;
    while layer.len() > 1 {
        let mut next: Vec<Tdd> = Vec::with_capacity(layer.len() / 2 + 1);
        let mut it = layer.into_iter();
        while let Some(a) = it.next() {
            if let Some(b) = it.next() {
                let mut t = apply_or(a.clone(), b.clone());
                minimize(&mut t);
                next.push(t);
            } else {
                next.push(a);
            }
        }
        round += 1;
        layer = next;
        if round % 4 == 0 || layer.len() <= 4 {
            let maxsz = layer.iter().map(|t| t.size()).max().unwrap_or(0);
            println!(
                "  round {} -> {} TDDs, max size={} ({:?})",
                round,
                layer.len(),
                maxsz,
                start.elapsed()
            );
        }
    }
    let mut acc = layer.pop().unwrap();
    minimize(&mut acc);
    println!(
        "FINAL {} pts -> size={} node_count={} max_width={} model_count={} (nodes/pt={:.3}) [{:?}]",
        npts,
        acc.size(),
        acc.node_count(),
        acc.max_width(),
        model_count(&acc),
        acc.node_count() as f64 / npts as f64,
        start.elapsed()
    );
    println!("  vs CUDD OBDD on same set: 44038 nodes (sift-stable). TDD node_count << that => diagram-block ALIVE.");
}

#[test]
fn test_apply_and_with_constant_one() {
    let eng = &crate::engine::Engine::new();
    let vtree = Arc::new(Vtree::balanced(3));
    let one = constant_one(eng, &vtree);
    let clause = vec![Literal::pos(VarId(0))];
    let clause_tdd = clause_to_tdd(eng, &vtree, &clause);

    // 1 ∧ clause = clause (after minimize)
    let mut expected = clause_tdd.clone();
    let mut result = apply_and(one, clause_tdd);
    minimize(&mut result);
    minimize(&mut expected);
    // Both should have the same model count (2^2 = 4 models satisfying x0)
    assert_eq!(model_count(&result), model_count(&expected));

    // Structural check (not just count-equality): a count-preserving bug that
    // returns a re-canonicalized-but-DIFFERENT diagram for `1 ∧ clause` would
    // still pass the model_count assertion above. `is_self_conjunction`
    // (the same predicate `apply_and` itself consults for its structural
    // shortcut, `conjoin/sparse.rs`) is a genuine canonical-equality
    // check: after `minimize`, two operands representing the same function
    // on the same vtree must have identical output + identical per-level
    // nodes/pairs/ext (TDD canonicity). Neither operand here carries a
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
    let eng = &crate::engine::Engine::new();
    let vtree = Arc::new(Vtree::balanced(3));
    let c1 = vec![Literal::pos(VarId(0))];
    let c2 = vec![Literal::neg(VarId(1))];

    let t1 = clause_to_tdd(eng, &vtree, &c1);
    let t2 = clause_to_tdd(eng, &vtree, &c2);
    let mut result = apply_and(t1, t2);
    minimize(&mut result);
    // x0=1 AND x1=0: 2 models (x2 can be 0 or 1)
    assert_eq!(model_count(&result), BigUint::from(2u32));
}

#[test]
fn test_apply_and_contradictory() {
    let eng = &crate::engine::Engine::new();
    // x0 ∧ ¬x0 should have no models
    let vtree = Arc::new(Vtree::balanced(1));
    let c1 = vec![Literal::pos(VarId(0))];
    let c2 = vec![Literal::neg(VarId(0))];

    let t1 = clause_to_tdd(eng, &vtree, &c1);
    let t2 = clause_to_tdd(eng, &vtree, &c2);
    let mut result = apply_and(t1, t2);
    minimize(&mut result);
    assert_eq!(model_count(&result), BigUint::ZERO);
}


#[test]
fn test_apply_and_self_conjunction() {
    let eng = &crate::engine::Engine::new();
    // f ∧ f = f for a non-trivial TDD.
    let vtree = Arc::new(Vtree::balanced(4));
    let c1 = vec![Literal::pos(VarId(0)), Literal::pos(VarId(2))];
    let c2 = vec![Literal::neg(VarId(1)), Literal::pos(VarId(3))];
    let mut tdd = clause_to_tdd(eng, &vtree, &c1);
    let t2 = clause_to_tdd(eng, &vtree, &c2);
    tdd = apply_and(tdd, t2);
    minimize(&mut tdd);

    let expected_mc = model_count(&tdd);
    let expected_size = tdd.size();

    // Conjoin with a clone of itself.
    let copy = tdd.clone();
    let mut result = apply_and(tdd, copy);
    minimize(&mut result);

    assert_eq!(model_count(&result), expected_mc);
    assert_eq!(result.size(), expected_size);
}

#[test]
fn test_apply_and_self_conjunction_owned() {
    let eng = &crate::engine::Engine::new();
    // f ∧ f = f via the owned variant (avoids clone).
    let vtree = Arc::new(Vtree::balanced(4));
    let c1 = vec![Literal::pos(VarId(0)), Literal::neg(VarId(2))];
    let c2 = vec![Literal::pos(VarId(1)), Literal::pos(VarId(3))];
    let mut tdd = clause_to_tdd(eng, &vtree, &c1);
    let t2 = clause_to_tdd(eng, &vtree, &c2);
    tdd = apply_and(tdd, t2);
    minimize(&mut tdd);

    let expected_mc = model_count(&tdd);
    let copy = tdd.clone();
    let mut result = apply_and(tdd, copy);
    minimize(&mut result);

    assert_eq!(model_count(&result), expected_mc);
}

#[test]
fn test_apply_and_stick_vtree_reachability() {
    let eng = &crate::engine::Engine::new();
    // Exercise the top-down reachability path on a stick (right-linear) vtree.
    // On sticks, every internal level has a leaf left child (3×3 grid),
    // triggering reachability gating at every level.
    //
    // Build TDDs from individual clauses (not compile_cnf) to keep both on
    // the same vtree Arc — compile_cnf grafts multi-component formulas.
    let vtree = Arc::new(Vtree::linear(8));

    let clauses1 = [
        vec![Literal::pos(VarId(0)), Literal::neg(VarId(2))],
        vec![Literal::neg(VarId(1)), Literal::pos(VarId(3))],
        vec![Literal::pos(VarId(4)), Literal::neg(VarId(5))],
        vec![Literal::neg(VarId(6)), Literal::pos(VarId(7))],
    ];
    let mut c1 = constant_one(eng, &vtree);
    for clause in &clauses1 {
        let cl = clause_to_tdd(eng, &vtree, clause);
        c1 = apply_and(c1, cl);
        minimize(&mut c1);
    }

    let clauses2 = [
        vec![Literal::neg(VarId(0)), Literal::pos(VarId(1))],
        vec![Literal::pos(VarId(2)), Literal::pos(VarId(4))],
        vec![Literal::neg(VarId(3)), Literal::neg(VarId(5))],
        vec![Literal::pos(VarId(6)), Literal::neg(VarId(7))],
    ];
    let mut c2 = constant_one(eng, &vtree);
    for clause in &clauses2 {
        let cl = clause_to_tdd(eng, &vtree, clause);
        c2 = apply_and(c2, cl);
        minimize(&mut c2);
    }

    let mut result = apply_and(c1, c2);
    minimize(&mut result);

    // Brute-force: count assignments satisfying both formulas.
    let expected = (0u64..256).filter(|&a| {
        let v = |i: u32| (a >> i) & 1 == 1;
        (v(0) || !v(2)) && (!v(1) || v(3)) && (v(4) || !v(5)) && (!v(6) || v(7))
        && (!v(0) || v(1)) && (v(2) || v(4)) && (!v(3) || !v(5)) && (v(6) || !v(7))
    }).count();

    assert_eq!(model_count(&result), BigUint::from(expected as u64));
}

/// Regression test: apply_and must not panic when one operand has a marginal
/// (`make_marginal`'d) level at a vtree position where the other operand is
/// non-identity.
///
/// `apply_and`'s c1/c2-identity fast paths only fire when the OTHER operand is
/// identity (width 1, propagating c1_identity/c2_identity) at every level
/// inside the marginal subtree. A marginalization schedule is what guarantees
/// that; once a vtree rotation or any other reshape breaks it, the other
/// operand can be non-identity at the marginal level and apply_and falls
/// through to the dense path, which reads `nodes[idx]` on an empty Vec.
///
/// This test pins the invariant: `apply_and` requires that whenever one
/// operand is marginal at vtree node t, the other operand is identity at t
/// (i.e. the conjunction at t is a no-op). Violating this is a soundness
/// error — the marginal form has discarded the pair structure needed to
/// compute the cross-product. The test deliberately violates the invariant
/// and asserts that `apply_and` panics with a recognisable diagnostic
/// (debug builds) rather than the cryptic `index out of bounds` from
/// `pairs_of_idx`.
// The diagnostic panic asserted below is `#[cfg(debug_assertions)]`-gated
// in `conjoin/mod.rs`. Under `cargo test
// --release` the gate is off and apply_and falls through to a cryptic
// `index out of bounds` from `pairs_of_idx`, which doesn't match
// `should_panic`. Gate the test to debug-builds so the default
// `cargo test` convention (testing.md) keeps it active and release-mode
// runs don't surface a spurious failure.
#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "apply_and: c1 marginal at vtree node")]
fn test_apply_and_panics_on_marginal_invariant_violation() {
    let eng = &crate::engine::Engine::new();
    // 4-leaf balanced vtree: root → (v_left, v_right), each width-2 internal.
    let vtree = Arc::new(Vtree::balanced(4));
    let root = VtreeIdx((vtree.num_nodes() - 1) as u32);
    let (v_left, v_right) = vtree.children(root);
    assert!(matches!(*vtree.node(v_left), VtreeNode::Internal { .. }));
    assert!(matches!(*vtree.node(v_right), VtreeNode::Internal { .. }));

    let pos = NodeIdx(LeafLabel::Pos as u32);
    let neg = NodeIdx(LeafLabel::Neg as u32);
    let one = NodeIdx(LeafLabel::One as u32);

    // ── TDD A: width-2 at v_left, made marginal ─────────────────────────
    let mut levels_a = take_levels(eng, vtree.num_nodes());
    let a0 = levels_a[v_left.idx()].push_internal_node(&[InputPair { left: pos, right: one }]);
    let a1 = levels_a[v_left.idx()].push_internal_node(&[InputPair { left: neg, right: one }]);
    let r0 = levels_a[v_right.idx()].push_internal_node(&[InputPair { left: pos, right: one }]);
    let r1 = levels_a[v_right.idx()].push_internal_node(&[InputPair { left: one, right: pos }]);
    let root_a = levels_a[root.idx()].push_internal_node(&[
        InputPair { left: a0, right: r0 },
        InputPair { left: a1, right: r1 },
    ]);
    let mut tdd_a = Tdd::with_levels(
        vtree.clone(),
        levels_a,
        TddNodeId { vtree: root, local: root_a },
    );

    // Freeze v_left into marginal form. Each entry has x1 free (mc = 2).
    assert_can_make_marginal(&tdd_a.levels, &vtree, v_left);
    tdd_a.levels[v_left.idx()].make_marginal(vec![2u128, 2u128], None);
    // Hand-rolled make_marginal bypasses production marginalization; tag the
    // now-marginal level's persisted parent refs so the 0=inline decode
    // invariant holds for the model_count below (mirrors marginalize_batch).
    crate::diagram::tag_all_marg_side_slots(&mut tdd_a, None);
    assert!(tdd_a.levels[v_left.idx()].is_marginal());
    assert_eq!(tdd_a.levels[v_left.idx()].width(), 2);

    // Model count of A = 8 (4 + 4 from the two disjoint root pairs).
    let mc_a = model_count(&tdd_a);
    assert_eq!(mc_a, BigUint::from(8u32), "tdd_a baseline model count");

    // ── TDD B: same shape, NOT marginal at v_left ──
    //
    // Width >1 at v_left means apply_and's k2==1 fast-path can't fire on B
    // as the c2 operand. v_left in B is explicit (not marginal), so this is
    // the "two width-2 operands meeting at a marginal level" shape that
    // bypasses both fast-paths and falls through to the dense path.
    let mut levels_b = take_levels(eng, vtree.num_nodes());
    let b0 = levels_b[v_left.idx()].push_internal_node(&[InputPair { left: pos, right: one }]);
    let b1 = levels_b[v_left.idx()].push_internal_node(&[InputPair { left: neg, right: one }]);
    let s0 = levels_b[v_right.idx()].push_internal_node(&[InputPair { left: pos, right: one }]);
    let s1 = levels_b[v_right.idx()].push_internal_node(&[InputPair { left: one, right: pos }]);
    let root_b = levels_b[root.idx()].push_internal_node(&[
        InputPair { left: b0, right: s0 },
        InputPair { left: b1, right: s1 },
    ]);
    let tdd_b = Tdd::with_levels(
        vtree.clone(),
        levels_b,
        TddNodeId { vtree: root, local: root_b },
    );
    let mc_b = model_count(&tdd_b);
    assert_eq!(mc_b, BigUint::from(8u32), "tdd_b baseline model count");

    // ── apply_and(A, B) ────────────────────────────────────────────────
    //
    // On unfixed main this panics:
    //   `index out of bounds: the len is 0 but the index is 0`
    //   at tididi/src/tdd/types.rs:549 (pairs_of_idx)
    let mut result = apply_and(tdd_a, tdd_b);
    minimize(&mut result);

    // A and B represent the same Boolean function, so A ∧ B = A → 8 models.
    assert_eq!(model_count(&result), BigUint::from(8u32));
}

/// Regression for the segment-conjoin output-size cap: when
/// `ApplyLimits::output_node_cap` is armed, `apply_and_fallible` must abort with
/// `ApplyError::OutputCap` the moment the cumulative output node count crosses
/// the cap — a deliberate cut of a product ballooning past the intended limit,
/// distinguishable from an OOM by the typed variant itself. The same conjoin
/// with no cap must complete. On unfixed `main` (no cap machinery) the armed
/// run would also complete — the cap is what makes it bail.
#[test]
fn test_apply_output_node_cap_bails_cleanly() {
    // Build a TDD by folding clauses with apply_and — every operand shares the
    // same `vtree` Arc (clause_to_tdd / constant_one clone it), so the final
    // conjoin's pointer-identical-vtree precondition holds.
    fn build(vtree: &Arc<Vtree>, clauses: &[&[i32]]) -> Tdd {
        let eng = &crate::engine::Engine::new();
        let mut acc = constant_one(eng, vtree);
        for lits in clauses {
            let clause: Vec<Literal> = lits.iter()
                .map(|&l| Literal::new(VarId(l.unsigned_abs() - 1), l > 0))
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
        apply_and_fallible(&eng, &mut a, &mut b, None)
    };
    assert!(uncapped.is_ok(), "no cap: conjoin should complete, got {:?}", uncapped.err());

    // Armed with a tiny cap → the apply bails as a deliberate output cut.
    let mut a = build(&vtree, fa);
    let mut b = build(&vtree, fb);
    let capped = {
        let eng = Engine::with_limits(LimitSet::none().output_cap(Some(1)));
        apply_and_fallible(&eng, &mut a, &mut b, None)
    };
    assert_eq!(
        capped.err(),
        Some(ApplyError::OutputCap),
        "tiny cap: apply must bail OutputCap once output exceeds the cap",
    );
}

// ── T5: `level_marginal_is_constant_true` (identity-fast-path eligibility) ──
//
// `level_marginal_is_constant_true` gates a structural-identity fast path
// (see its doc comment, `conjoin/mod.rs`): a wrong `true` silently
// drops operand content. These tests hand-roll `TddLevel`s via
// `TddLevel::new()` + `make_marginal(counts, big)` into the exact shapes
// that exercise its two guarded branches: the `subvars >= 128` BigUint
// side-table branch, and the `c0 == u128::MAX` overflow sentinel.
