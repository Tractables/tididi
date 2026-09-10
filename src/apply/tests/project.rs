//! Existential projection, in both the plain and the scoped form.
//!
//! Fixtures come from `crate::test_helpers`, re-exported by the parent.

use super::*;


#[test]
fn project_var_of_constant_one_is_one() {
    let eng = &crate::engine::Engine::new();
    let vtree = Arc::new(Vtree::balanced(3));
    let tdd = constant_one(eng, &vtree);
    let result = project_var(&tdd, VarId(0), Projection::Automatic);
    assert!(!result.is_zero());
    assert_eq!(model_count(&result), BigUint::from(8u32));
}

#[test]
fn project_var_of_constant_zero_is_zero() {
    let eng = &crate::engine::Engine::new();
    let vtree = Arc::new(Vtree::balanced(3));
    let tdd = constant_zero(eng, &vtree);
    let result = project_var(&tdd, VarId(0), Projection::Automatic);
    assert!(result.is_zero());
}

#[test]
fn project_var_of_literal_is_one() {
    let eng = &crate::engine::Engine::new();
    let vtree = Arc::new(Vtree::balanced(1));
    let tdd = clause_to_tdd(eng, &vtree, &crate::test_helpers::clause(&[(0, true)]));
    assert_eq!(model_count(&tdd), BigUint::from(1u32));

    let result = project_var(&tdd, VarId(0), Projection::Automatic);
    assert!(!result.is_zero());
    assert_eq!(model_count(&result), BigUint::from(2u32));
}

#[test]
fn project_var_of_x_and_y_drops_x() {
    let eng = &crate::engine::Engine::new();
    let vtree = Arc::new(Vtree::balanced(2));
    let tdd_x = clause_to_tdd(eng, &vtree, &crate::test_helpers::clause(&[(0, true)]));
    let tdd_y = clause_to_tdd(eng, &vtree, &crate::test_helpers::clause(&[(1, true)]));

    let tdd_xy = apply_and(tdd_x, tdd_y);
    assert_eq!(model_count(&tdd_xy), BigUint::from(1u32));

    let result = project_var(&tdd_xy, VarId(0), Projection::Automatic);
    assert!(!result.is_zero());
    assert_eq!(model_count(&result), BigUint::from(2u32));
}

#[test]
fn project_var_soundness_brute_force() {
    let eng = &crate::engine::Engine::new();
    // F = (x ∨ y) ∧ (¬y ∨ z), vars 0=x 1=y 2=z. Project out y.
    let vtree = Arc::new(Vtree::balanced(3));

    let tdd1 = clause_to_tdd(eng, &vtree, &crate::test_helpers::clause(&[(0, true), (1, true)]));
    let tdd2 = clause_to_tdd(eng, &vtree, &crate::test_helpers::clause(&[(1, false), (2, true)]));

    let tdd_f = apply_and(tdd1, tdd2);

    let brute = {
        let mut seen = std::collections::HashSet::new();
        for assign in 0u32..(1u32 << 3) {
            let x = assign & 1 == 1;
            let y = (assign >> 1) & 1 == 1;
            let z = (assign >> 2) & 1 == 1;
            let f = x || y;
            let g = !y || z;
            if f && g {
                seen.insert(assign & !(1u32 << 1));
            }
        }
        seen.len() as u64
    };

    let result = project_var(&tdd_f, VarId(1), Projection::Automatic);

    // The projected diagram still lives in the 3-var vtree. y's leaf becomes
    // "all One", so model_count counts over all 3 bits — each surviving
    // (x,z) pair appears twice (once for y=T, once for y=F).
    let result_count = model_count(&result);
    assert_eq!(
        result_count,
        BigUint::from(brute * 2),
        "brute projected count = {brute}; TDD model_count (3-var) = {result_count}"
    );
}

/// A zero-width marginal level must not crash the marginalize cascade.
///
/// `ensure_counts` needs the same `width() == 0` guard `marginalize_batch`
/// has. When an internal vtree level
/// has 0 pair nodes, `ensure_counts` computes empty counts → the cascade
/// calls `become_marginal(vec![], None)` → 0-width marginal. Later, `project_var`
/// calls `apply_or(pos_cofactor, neg_cofactor)` where both cofactors inherit this
/// 0-width marginal (the level is disjoint from the projected variable's leaf).
/// `apply_and` then encounters left_width=right_width=0 with both levels marginal, which neither
/// identity fast-path (both require k==1) handles — dense path panics at
/// `pairs_of_idx(0)` on an empty nodes Vec.
///
/// Fix: 0-width marginal fast-path added to `apply_and_fallible` before the
/// debug-assertions block in `apply::conjoin`.
///
/// This test constructs the crashing state directly and calls `apply_and`,
/// because the state is unreachable through the public compile API with a
/// static vtree: within one driver marginalize step, projection runs
/// before `marginalize_batch`, and the marginal-carrying accumulator only
/// merges with the other vtree half at their LCA — but any cross-half
/// clause that schedules the projection trigger at that LCA step also
/// delays the marginal's creation to the same step, where projection wins.
/// Production reached the state via mid-compile vtree rotations
/// (the marginal-cluster rotation pass), which aren't deterministic
/// enough for a test.
///
/// Construction: balanced(8), f={var4}, g={var5} — clauses confined to
/// the right half, so the left half holds only trivial structure. Mirror
/// production's marginalized left half in both operands:
///   A = Internal(var0,var1) → `become_marginal(vec![], None)` — the 0-width
///       orphan, exactly what the marginalize cascade / `ensure_counts` emits
///       for a 0-node level (it lacks `marginalize_batch`'s width()==0 guard);
///   B = Internal(var2,var3) → marginal [4]  (vars 2,3 free);
///   C = parent(A,B)         → marginal [16] (vars 0..3 free).
/// C being marginal is what makes A a true orphan (marginal levels carry
/// counts, not pair references), matching the production dump where the
/// 0-width level had no live parents. `apply_and` then hits A with
/// left_width=right_width=0, both marginal: without the fix the debug assert (debug builds)
/// or `pairs_of_idx(0)` (release) panics; with it the level passes
/// through empty and the conjunction's count is unchanged.
#[test]
fn apply_and_zero_width_marginal_levels() {
    let eng = &crate::engine::Engine::new();
    use crate::vtree::VtreeIdx;

    let vtree = Arc::new(Vtree::balanced(8));
    // Baseline: var4 ∧ var5 over 8 vars = 2^6 models.
    let b1 = clause_to_tdd(eng, &vtree, &crate::test_helpers::clause(&[(4, true)]));
    let b2 = clause_to_tdd(eng, &vtree, &crate::test_helpers::clause(&[(5, true)]));
    let baseline = model_count(&apply_and(b1, b2));
    assert_eq!(baseline, BigUint::from(64u32));

    // Locate Internal(varX,varY) nodes by their leaf children.
    let leaf_parent = |x: u32, y: u32| {
        (0..vtree.num_nodes() as u32)
            .map(VtreeIdx)
            .find(|&i| {
                !vtree.node(i).is_leaf() && {
                    let (l, r) = vtree.children(i);
                    matches!(*vtree.node(l), crate::vtree::VtreeNode::Leaf { var, .. } if var == VarId(x))
                        && matches!(*vtree.node(r), crate::vtree::VtreeNode::Leaf { var, .. } if var == VarId(y))
                }
            })
            .expect("balanced(8) must have this Internal(leaf,leaf) node")
    };
    let a = leaf_parent(0, 1);
    let b = leaf_parent(2, 3);
    let c = vtree.node(a).parent().expect("A has a parent");
    assert_eq!(vtree.node(b).parent(), Some(c), "C must be Internal(A,B)");

    let mut f = clause_to_tdd(eng, &vtree, &crate::test_helpers::clause(&[(4, true)]));
    let mut g = clause_to_tdd(eng, &vtree, &crate::test_helpers::clause(&[(5, true)]));
    for t in [&mut f, &mut g] {
        t.levels[a.idx()].become_marginal(vec![], None); // 0-width orphan
        t.levels[b.idx()].become_marginal(vec![4], None);
        t.levels[c.idx()].become_marginal(vec![16], None);
    }
    assert!(f.levels[a.idx()].is_marginal() && f.levels[a.idx()].width() == 0);
    assert!(g.levels[a.idx()].is_marginal() && g.levels[a.idx()].width() == 0);

    // Unfixed: panics inside apply_and_fallible at the 0-width marginal level.
    let result = apply_and(f, g);
    assert_eq!(
        model_count(&result),
        baseline,
        "orphan 0-width marginal level changed the conjunction's model count"
    );
}

/// Guard for the always-on marginalize-schedule invariant in `apply_and`
/// (`conjoin/mod.rs`): conjoining a diagram that has marginalized a vtree node with
/// one that still constrains a variable under that node is INVALID. Before the guard
/// this dereferenced a bad marginal-side reference and SIGSEGV'd in release (the debug
/// assert that should have caught it had been compiled out); now it must panic
/// cleanly so a marginalize-schedule bug surfaces loudly instead of corrupting the
/// model count.
///
/// Minimal hand-checkable case (6-var balanced vtree): `fm` = f with vtree node 7's
/// subtree (vars {4,5}) marginalized via `marginalize_subtree` (production-faithful:
/// mirrors the marginalize pass, tags marginal-side slots). `partner`
/// still references x5, so `and2(partner, fm)` is the invalid conjoin and must be
/// rejected. (In a correct run the schedule only marginalizes PRIVATE vars — vars no
/// partner references — so this never arises; the test deliberately constructs it.)
#[test]
fn apply_and_rejects_marginalize_schedule_violation() {
    let eng = &crate::engine::Engine::new();
    use crate::test_helpers::marginalize_subtree;
    use crate::vtree::VtreeIdx;
    let nvars = 6u32;
    let vtree = Arc::new(Vtree::balanced(nvars));
    let build = |cls: &[&[(u32, bool)]]| -> Tdd {
        let mut acc: Option<Tdd> = None;
        for literals in cls {
            let cl = clause_to_tdd(eng, &vtree, &crate::test_helpers::clause(literals));
            acc = Some(match acc {
                None => cl,
                Some(a) => and2(&a, &cl),
            });
        }
        acc.unwrap()
    };
    let ra = VtreeIdx(7);
    let f_raw = build(&[
        &[(0, false), (3, false), (5, true)],
        &[(0, true), (3, true)],
        &[(1, true), (4, false)],
    ]);
    let partner = build(&[
        &[(1, true), (5, true)],
        &[(0, true), (5, false)],
        &[(0, true), (2, true), (3, false)],
        &[(3, false)],
    ]);

    let mut fm = f_raw.clone();
    marginalize_subtree(&mut fm, ra);
    crate::reduce::minimize(&mut fm);
    let has_marginal = (0..vtree.num_nodes())
        .any(|i| fm.levels[i].is_marginal());
    // Which variables does marginalizing node `ra` sum out (the leaves under ra)?
    // Production only marginalizes PRIVATE vars — vars no partner references. If any
    // of these is in partner's support, this is the de-marginalize-a-needed-var case,
    // which production does not create.
    let mut marginal_vars: Vec<u32> = Vec::new();
    for vi in 0..vtree.num_nodes() {
        if let crate::vtree::VtreeNode::Leaf { var, .. } = *vtree.node(VtreeIdx(vi as u32)) {
            let mut cur = VtreeIdx(vi as u32);
            let mut under = cur == ra;
            while let Some(p) = vtree.node(cur).parent() {
                if p == ra {
                    under = true;
                    break;
                }
                cur = p;
            }
            if under {
                marginal_vars.push(var.0);
            }
        }
    }
    marginal_vars.sort_unstable();
    let partner_support: std::collections::BTreeSet<u32> =
        [0u32, 1, 2, 3, 5].into_iter().collect();
    let overlap: Vec<u32> = marginal_vars
        .iter()
        .copied()
        .filter(|v| partner_support.contains(v))
        .collect();
    println!(
        "MINIMAL setup: f.size={} fm.size={} fm_has_marginal={} partner.size={} same_vtree={}",
        f_raw.size(),
        fm.size(),
        has_marginal,
        partner.size(),
        partner.output.vtree == fm.output.vtree,
    );
    println!(
        "MINIMAL marginal_vars(under node {})={:?} partner_support={:?} OVERLAP={:?}",
        ra.idx(),
        marginal_vars,
        partner_support,
        overlap,
    );

    assert!(!overlap.is_empty(), "test fixture must marginalize a var partner uses");

    // The invalid conjoin: partner constrains x5, but fm summed x5 out. Before the
    // always-on marginalize-schedule guard this SIGSEGV'd (or silently miscounted)
    // in the apply's dense path. With the guard it must PANIC cleanly — catchable,
    // diagnostic, never a silent wrong answer. Silence the hook so the expected
    // panic's backtrace doesn't spam test output.
    std::panic::set_hook(Box::new(|_| {}));
    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| and2(&partner, &fm)));
    let _ = std::panic::take_hook();
    // An error must be thrown (the conjoin must not silently return a value).
    let payload = res.expect_err(
        "apply_and must REJECT conjoining a marginalized operand with one that still \
         constrains the summed-out var, but the conjoin returned a value",
    );
    // And it must be OUR invariant violation, not some incidental panic.
    let msg = payload
        .downcast_ref::<String>()
        .map(|s| s.as_str())
        .or_else(|| payload.downcast_ref::<&str>().copied())
        .unwrap_or("");
    // apply_and may reject this invalid conjoin at either of two equivalent
    // marginal-conjoin guards: the deep "marginalize-schedule violation"
    // (conjoin/mod.rs:2587) or the earlier structural "Marginal pair
    // structure cannot conjoin with a non-trivial operand" sanity block
    // (conjoin/mod.rs:2166-2184), which catches this fixture first. Both
    // enforce the same property — a marginalized operand must not conjoin
    // with one still constraining the summed-out var — so accept either.
    assert!(
        msg.contains("marginalize-schedule violation")
            || msg.contains("Marginal pair structure cannot conjoin"),
        "expected apply_and to reject the marginal×constraining conjoin \
         (marginal_vars={marginal_vars:?}, overlap={overlap:?}), got a different panic: {msg:?}"
    );
}

#[test]
fn scoped_constant_one_is_one() {
    let eng = &crate::engine::Engine::new();
    let vtree = Arc::new(Vtree::balanced(3));
    let tdd = constant_one(eng, &vtree);
    let r = project_var(&tdd, VarId(0), Projection::Structural);
    assert!(!r.is_zero());
    assert_eq!(model_count(&r), BigUint::from(8u32));
}

#[test]
fn scoped_constant_zero_is_zero() {
    let eng = &crate::engine::Engine::new();
    let vtree = Arc::new(Vtree::balanced(3));
    let tdd = constant_zero(eng, &vtree);
    let r = project_var(&tdd, VarId(0), Projection::Structural);
    assert!(r.is_zero());
}

#[test]
fn scoped_single_literal_is_one() {
    let eng = &crate::engine::Engine::new();
    let vtree = Arc::new(Vtree::balanced(1));
    let tdd = clause_to_tdd(eng, &vtree, &crate::test_helpers::clause(&[(0, true)]));
    let r = project_var(&tdd, VarId(0), Projection::Structural);
    assert!(!r.is_zero());
    assert_eq!(model_count(&r), BigUint::from(2u32));
}

#[test]
fn scoped_x_and_y_drops_x() {
    let eng = &crate::engine::Engine::new();
    let vtree = Arc::new(Vtree::balanced(2));
    let tx = clause_to_tdd(eng, &vtree, &crate::test_helpers::clause(&[(0, true)]));
    let ty = clause_to_tdd(eng, &vtree, &crate::test_helpers::clause(&[(1, true)]));
    let txy = apply_and(tx, ty);
    let r = project_var(&txy, VarId(0), Projection::Structural);
    assert_eq!(model_count(&r), BigUint::from(2u32));
}

/// Marginal-sibling case: `project_var` panics here; `project_var_scoped`
/// must succeed and give the correct count.
///
/// Build F over balanced(4) with two FULLY DISJOINT halves:
///   left half (a,b): {a∨b}      (3 models over {a,b})
///   right half (x,w): {x∨w}     (3 models over {x,w})
/// so F = (a∨b) ∧ (x∨w), 9 models. Marginalize the LEFT half (a,b) — disjoint
/// from x's leaf-to-root path — into a real width>1 marginal carrying the
/// left subtree's per-node counts. Correctly marginalizing a disjoint
/// subtree preserves the model count, so `project_var_scoped` on the
/// marginalized diagram must equal `project_var` on the non-marginal diagram.
#[test]
fn scoped_marginal_sibling_succeeds() {
    let eng = &crate::engine::Engine::new();
    use crate::vtree::{VtreeIdx, VtreeNode};

    let vtree = Arc::new(Vtree::balanced(4));
    let t1 = clause_to_tdd(eng, &vtree, &crate::test_helpers::clause(&[(0, true), (1, true)])); // a∨b
    let t2 = clause_to_tdd(eng, &vtree, &crate::test_helpers::clause(&[(2, true), (3, true)])); // x∨w
    let f = apply_and(t1, t2);
    assert_eq!(model_count(&f), BigUint::from(9u32));

    // Reference: project x on the non-marginal diagram.
    let ref_count = model_count(&project_var(&f, VarId(2), Projection::Automatic));

    // Find Internal(a,b) — the root's left child, disjoint from x's path.
    let ab = (0..vtree.num_nodes() as u32)
        .map(VtreeIdx)
        .find(|&i| {
            !vtree.node(i).is_leaf() && {
                let (l, r) = vtree.children(i);
                matches!(*vtree.node(l), VtreeNode::Leaf { var, .. } if var == VarId(0))
                    && matches!(*vtree.node(r), VtreeNode::Leaf { var, .. } if var == VarId(1))
            }
        })
        .expect("balanced(4) has Internal(a,b)");

    // Marginalize the disjoint left subtree (a,b) via the test helper, which
    // installs the correct per-node counts. This produces a real width>1
    // marginal sibling on x's path (the root's left child) — exactly the
    // shape that crashes `project_var`'s cofactor-OR.
    let mut fm = f.clone();
    crate::test_helpers::marginalize_subtree(&mut fm, ab);
    assert!(fm.levels[ab.idx()].is_marginal());
    assert!(fm.levels[ab.idx()].width() > 0);
    // Marginalizing a disjoint subtree preserves the model count.
    assert_eq!(model_count(&fm), BigUint::from(9u32));

    // Must not panic crossing the marginal sibling, and must match the count.
    let g = project_var(&fm, VarId(2), Projection::Structural);
    assert_eq!(
        model_count(&g),
        ref_count,
        "scoped projection over marginal sibling gave wrong count"
    );
}

/// Regression for the path-side `One` reference at an INTERNAL ancestor level.
///
/// `regroup_internal` indexes `child_remap[path_child.idx()]`. On internal
/// levels `NodeIdx(0)` is the constant-true (One) representative, so a
/// pair whose path-side (x's subtree) is UNCONSTRAINED references it as One.
/// This test forces exactly that shape to confirm the scoped forget handles a
/// path-side One ref at the root (not just at the leaf-parent).
///
/// F = (v0∨v3) ∧ (v2∨v3) = (v0∧v2) ∨ v3 over balanced(4) [L=(v0,v1), R=(v2,v3)].
/// When v3=1, F=1 regardless of the left half (v0,v1) → the root has a branch
/// where the left (path) subtree is free = a path-side One ref. Project v0:
/// ∃v0.F = v2∨v3; with v1 free and v0's leaf→One, model_count = 6·2 = 12, and
/// PMC onto {v1,v2,v3} = 6.
#[test]
fn scoped_path_side_one_ref_at_root() {
    let eng = &crate::engine::Engine::new();
    let vtree = Arc::new(Vtree::balanced(4));
    let t1 = clause_to_tdd(eng, &vtree, &crate::test_helpers::clause(&[(0, true), (3, true)])); // v0∨v3
    let t2 = clause_to_tdd(eng, &vtree, &crate::test_helpers::clause(&[(2, true), (3, true)])); // v2∨v3
    let f = apply_and(t1, t2);

    let g_scoped = project_var(&f, VarId(0), Projection::Structural);
    let g_ref = project_var(&f, VarId(0), Projection::Automatic);
    assert_eq!(
        model_count(&g_scoped),
        model_count(&g_ref),
        "scoped != cofactor projecting v0 from (v0∨v3)∧(v2∨v3)"
    );
    assert_eq!(model_count(&g_scoped), BigUint::from(12u32));
    crate::check::check_determinism(&g_scoped).unwrap();

    // PMC onto show={v1,v2,v3}: project v0, >>1, vs brute force.
    let clauses = vec![vec![1, 4], vec![3, 4]]; // DIMACS 1-indexed: (v0∨v3)∧(v2∨v3)
    let pmc = model_count(&project_vars(&f, &[VarId(0)], Projection::Structural)) >> 1usize;
    assert_eq!(pmc, brute_force_pmc(&clauses, 4, &[1, 2, 3]));
}

/// A weighted diagram stays weighted across a projection.
///
/// The cofactor rewrite reaches its result through a disjunction, and negation
/// — which a disjunction is built from — copies levels without the side table.
/// A caller that projects a weighted accumulator and reads its values
/// afterwards depends on this, and would otherwise have to detach and reattach
/// the store around every call.
#[test]
fn projecting_a_weighted_diagram_keeps_its_weight_store() {
    use crate::diagram::{Arithmetic, RationalWeights, WeightStore};
    use num_rational::BigRational;

    let vtree = Arc::new(Vtree::balanced(3));
    let semiring = RationalWeights::from_weights(&[
        (BigRational::from_integer(1.into()), BigRational::from_integer(2.into())),
        (BigRational::from_integer(1.into()), BigRational::from_integer(3.into())),
        (BigRational::from_integer(1.into()), BigRational::from_integer(5.into())),
    ]);
    let mut tdd = Tdd::clause(&vtree, [1, 2]);
    tdd.set_weights(WeightStore::new(semiring, Arithmetic::ExactRational));
    // No level is marginal, so this takes the cofactor route, not the
    // structural one that clones the whole diagram.
    assert!(tdd.levels.iter().all(|l| !l.is_marginal()));

    let projected = project_var(&tdd, VarId(0), Projection::Automatic);
    assert!(
        projected.weights().is_some(),
        "the projection dropped the weight store",
    );
}

/// A projection whose cofactor copy the armed budget cannot pay for gives the
/// refusal back, and the engine it ran on is usable for the next call.
///
/// The copy is the first reservation the cofactor rewrite makes, so a budget
/// below one copy of the level array stops the projection there.
#[test]
fn a_projection_refuses_a_cofactor_copy_it_cannot_afford() {
    use crate::engine::LimitSet;
    use crate::error::ApplyError;

    let eng = Engine::new();
    let vtree = Arc::new(Vtree::balanced(8));
    let mut f = clause_to_tdd(&eng, &vtree, &crate::test_helpers::clause(&[(0, true), (1, false)]));
    for v in 1..7u32 {
        let c = clause_to_tdd(
            &eng,
            &vtree,
            &crate::test_helpers::clause(&[(v, true), (v + 1, false)]),
        );
        f = apply_and(f, c);
    }
    assert!(!f.is_zero());
    let expected = model_count(&project_var(&f, VarId(0), Projection::Automatic));

    let one_copy = std::mem::size_of_val(f.levels()) as u64;
    let budget = 64u64;
    assert!(budget < one_copy, "the budget has to be below one copy of the level array");

    eng.limits().reset_meters();
    let refused = {
        let _armed = eng.limits().scope(LimitSet::none().budget(Some(budget)));
        eng.project_var(f.clone(), VarId(0), Projection::Automatic)
    };
    assert!(
        matches!(refused, Err(ApplyError::OverBudget)),
        "a projection that cannot copy its operand must report the refusal"
    );
    // The copy is what asked, so the whole level array was charged before the
    // refusal. An unreserved copy charges nothing and the refusal lands
    // somewhere downstream, well short of this.
    assert!(
        eng.limits().meters().in_flight_bytes >= one_copy,
        "the refusal has to come from the copy's own reservation"
    );

    let out = eng
        .project_var(f, VarId(0), Projection::Automatic)
        .expect("the engine takes the next projection after a refusal");
    assert_eq!(model_count(&out), expected);
}
