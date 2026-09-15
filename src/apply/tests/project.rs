//! Existential projection, in both the plain and the scoped form.
//!
//! Fixtures come from `crate::test_helpers`, re-exported by the parent.

use super::*;

#[test]
fn batch_projection_accumulates_charges_across_variables() {
    let vtree = Arc::new(Vtree::balanced(16));
    let f = Tdd::clause(&vtree, [1, 2]).unwrap() & Tdd::clause(&vtree, [-1, 3]).unwrap()
        & Tdd::clause(&vtree, [4, 5]).unwrap() & Tdd::clause(&vtree, [-4, 6]).unwrap();
    crate::test_helpers::assert_canonical(&f);
    let eng = crate::Engine::new();
    let first = eng.exists_var(f.clone(), VarId(0)).unwrap();
    crate::test_helpers::assert_canonical(&first);
    let first_charge = eng.limits().meters().in_flight_bytes;
    let second = eng.exists_var(first, VarId(3)).unwrap();
    crate::test_helpers::assert_canonical(&second);
    let second_charge = eng.limits().meters().in_flight_bytes;
    let bounded = crate::Engine::new();
    let _prior = bounded.limits().install(crate::limits::LimitConfig::none().with_memory_budget_bytes(Some(first_charge.max(second_charge))));
    assert!(matches!(bounded.exists_vars(f, &[VarId(0), VarId(3)]), Err(crate::OperationError::OverBudget)));
    let empty_batch = bounded.exists_vars(second, &[]).unwrap();
    assert_eq!(bounded.limits().meters().in_flight_bytes, 0);
    crate::test_helpers::assert_canonical(&empty_batch);
}


#[test]
fn exists_var_of_constant_one_is_one() {
    let eng = &crate::engine::Engine::new();
    let vtree = Arc::new(Vtree::balanced(3));
    let tdd = constant_one(eng, &vtree);
    let result = (tdd).clone().exists_var(VarId(0)).unwrap();
    assert!(!result.is_zero());
    assert_eq!(result.model_count().unwrap(), BigUint::from(8u32));
}

#[test]
fn exists_var_of_constant_zero_is_zero() {
    let eng = &crate::engine::Engine::new();
    let vtree = Arc::new(Vtree::balanced(3));
    let tdd = constant_zero(eng, &vtree);
    let result = (tdd).clone().exists_var(VarId(0)).unwrap();
    assert!(result.is_zero());
}

#[test]
fn exists_var_of_literal_is_one() {
    let eng = &crate::engine::Engine::new();
    let vtree = Arc::new(Vtree::balanced(1));
    let tdd = clause_to_tdd(eng, &vtree, &crate::test_helpers::clause(&[(0, true)]));
    assert_eq!(tdd.model_count().unwrap(), BigUint::from(1u32));

    let result = (tdd).clone().exists_var(VarId(0)).unwrap();
    assert!(!result.is_zero());
    assert_eq!(result.model_count().unwrap(), BigUint::from(2u32));
}

#[test]
fn exists_var_of_x_and_y_drops_x() {
    let eng = &crate::engine::Engine::new();
    let vtree = Arc::new(Vtree::balanced(2));
    let tdd_x = clause_to_tdd(eng, &vtree, &crate::test_helpers::clause(&[(0, true)]));
    let tdd_y = clause_to_tdd(eng, &vtree, &crate::test_helpers::clause(&[(1, true)]));

    let tdd_xy = apply_and(tdd_x, tdd_y);
    assert_eq!(tdd_xy.model_count().unwrap(), BigUint::from(1u32));

    let result = (tdd_xy).clone().exists_var(VarId(0)).unwrap();
    assert!(!result.is_zero());
    assert_eq!(result.model_count().unwrap(), BigUint::from(2u32));
}

#[test]
fn exists_var_soundness_brute_force() {
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

    let result = (tdd_f).clone().exists_var(VarId(1)).unwrap();

    // The projected diagram still lives in the 3-var vtree. y's leaf becomes
    // "all One", so model_count counts over all 3 bits — each surviving
    // (x,z) pair appears twice (once for y=T, once for y=F).
    let result_count = result.model_count().unwrap();
    assert_eq!(
        result_count,
        BigUint::from(brute * 2),
        "brute projected count = {brute}; TDD (3-var).model_count().unwrap() = {result_count}"
    );
}

/// A zero-width marginal level must not crash the marginalization cascade.
///
/// `ensure_counts` needs the same `slot_count() == 0` guard `marginalize_batch`
/// has. When an internal vtree level
/// has 0 pair nodes, `ensure_counts` computes empty counts → the cascade
/// calls `become_marginal(vec![], None)` → 0-width marginal. Later, `exists_var`
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
/// static vtree: within one driver marginalization step, projection runs
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
///       orphan, exactly what the marginalization cascade / `ensure_counts` emits
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
    let baseline = (apply_and(b1, b2)).model_count().unwrap();
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
    assert!(f.levels[a.idx()].is_marginal() && f.levels[a.idx()].slot_count() == 0);
    assert!(g.levels[a.idx()].is_marginal() && g.levels[a.idx()].slot_count() == 0);

    // Unfixed: panics inside apply_and_fallible at the 0-width marginal level.
    let result = apply_and(f, g);
    assert_eq!(
        result.model_count().unwrap(),
        baseline,
        "orphan 0-width marginal level changed the conjunction's model count"
    );
}

/// Guard for the always-on marginalization-schedule invariant in `apply_and`
/// (`conjoin/mod.rs`): conjoining a diagram that has marginalized a vtree node with
/// one that still constrains a variable under that node is INVALID. Before the guard
/// this dereferenced a bad marginal-side reference and SIGSEGV'd in release (the debug
/// assert that should have caught it had been compiled out); now it must panic
/// cleanly so a marginalization-schedule bug surfaces loudly instead of corrupting the
/// model count.
///
/// Minimal hand-checkable case (6-var balanced vtree): `fm` = f with vtree node 7's
/// subtree (vars {4,5}) marginalized via `marginalize_subtree` (production-faithful:
/// mirrors the marginalization pass, tags marginal-side slots). `partner`
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
    fm.minimize().unwrap();
    let has_marginal = (0..vtree.num_nodes())
        .any(|i| fm.levels[i].is_marginal());
    // Which variables does marginalizing node `ra` sum out (the leaves under ra)?
    // Production only marginalizes PRIVATE vars — vars no partner references. If any
    // of these is in partner's support, this is the restore-a-needed-variable case,
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
        "MINIMAL setup: f.pair_count={} fm.pair_count={} fm_has_marginal={} partner.pair_count={} same_vtree={}",
        f_raw.pair_count(),
        fm.pair_count(),
        has_marginal,
        partner.pair_count(),
        partner.output.vtree == fm.output.vtree,
    );
    println!(
        "MINIMAL marginal_vars(under node {})={:?} partner_support={:?} OVERLAP={:?}",
        ra.idx(),
        marginal_vars,
        partner_support,
        overlap,
    );

    assert!(!overlap.is_empty(), "test fixture must marginalize_levels a var partner uses");

    assert_canonical(&partner);
    fm.minimize().unwrap();
    assert_canonical(&fm);
    assert!(matches!(eng.and(partner, fm), Err(crate::OperationError::MarginalLevel(_))));
}

#[test]
fn scoped_constant_one_is_one() {
    let eng = &crate::engine::Engine::new();
    let vtree = Arc::new(Vtree::balanced(3));
    let tdd = constant_one(eng, &vtree);
    let r = (tdd).clone().exists_var_with_strategy(VarId(0), QuantificationStrategy::Structural).unwrap();
    assert!(!r.is_zero());
    assert_eq!(r.model_count().unwrap(), BigUint::from(8u32));
}

#[test]
fn scoped_constant_zero_is_zero() {
    let eng = &crate::engine::Engine::new();
    let vtree = Arc::new(Vtree::balanced(3));
    let tdd = constant_zero(eng, &vtree);
    let r = (tdd).clone().exists_var_with_strategy(VarId(0), QuantificationStrategy::Structural).unwrap();
    assert!(r.is_zero());
}

#[test]
fn scoped_single_literal_is_one() {
    let eng = &crate::engine::Engine::new();
    let vtree = Arc::new(Vtree::balanced(1));
    let tdd = clause_to_tdd(eng, &vtree, &crate::test_helpers::clause(&[(0, true)]));
    let r = (tdd).clone().exists_var_with_strategy(VarId(0), QuantificationStrategy::Structural).unwrap();
    assert!(!r.is_zero());
    assert_eq!(r.model_count().unwrap(), BigUint::from(2u32));
}

#[test]
fn scoped_x_and_y_drops_x() {
    let eng = &crate::engine::Engine::new();
    let vtree = Arc::new(Vtree::balanced(2));
    let tx = clause_to_tdd(eng, &vtree, &crate::test_helpers::clause(&[(0, true)]));
    let ty = clause_to_tdd(eng, &vtree, &crate::test_helpers::clause(&[(1, true)]));
    let txy = apply_and(tx, ty);
    let r = (txy).clone().exists_var_with_strategy(VarId(0), QuantificationStrategy::Structural).unwrap();
    assert_eq!(r.model_count().unwrap(), BigUint::from(2u32));
}

/// Marginal-sibling case: `exists_var` panics here; `exists_var_scoped`
/// must succeed and give the correct count.
///
/// Build F over balanced(4) with two FULLY DISJOINT halves:
///   left half (a,b): {a∨b}      (3 models over {a,b})
///   right half (x,w): {x∨w}     (3 models over {x,w})
/// so F = (a∨b) ∧ (x∨w), 9 models. Marginalize the LEFT half (a,b) — disjoint
/// from x's leaf-to-root path — into a real width>1 marginal carrying the
/// left subtree's per-node counts. Correctly marginalizing a disjoint
/// subtree preserves the model count, so `exists_var_scoped` on the
/// marginalized diagram must equal `exists_var` on the non-marginal diagram.
#[test]
fn scoped_marginal_sibling_succeeds() {
    let eng = &crate::engine::Engine::new();
    use crate::vtree::{VtreeIdx, VtreeNode};

    let vtree = Arc::new(Vtree::balanced(4));
    let t1 = clause_to_tdd(eng, &vtree, &crate::test_helpers::clause(&[(0, true), (1, true)])); // a∨b
    let t2 = clause_to_tdd(eng, &vtree, &crate::test_helpers::clause(&[(2, true), (3, true)])); // x∨w
    let f = apply_and(t1, t2);
    assert_eq!(f.model_count().unwrap(), BigUint::from(9u32));

    // Reference: project x on the non-marginal diagram.
    let ref_count = ((f).clone().exists_var(VarId(2)).unwrap()).model_count().unwrap();

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
    // shape that crashes `exists_var`'s cofactor-OR.
    let mut fm = f.clone();
    crate::test_helpers::marginalize_subtree(&mut fm, ab);
    assert!(fm.levels[ab.idx()].is_marginal());
    assert!(fm.levels[ab.idx()].slot_count() > 0);
    // Marginalizing a disjoint subtree preserves the model count.
    assert_eq!(fm.model_count().unwrap(), BigUint::from(9u32));

    // Must not panic crossing the marginal sibling, and must match the count.
    let g = (fm).clone().exists_var_with_strategy(VarId(2), QuantificationStrategy::Structural).unwrap();
    assert_eq!(
        g.model_count().unwrap(),
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

    let g_scoped = (f).clone().exists_var_with_strategy(VarId(0), QuantificationStrategy::Structural).unwrap();
    let g_ref = (f).clone().exists_var(VarId(0)).unwrap();
    assert_eq!(
        g_scoped.model_count().unwrap(),
        g_ref.model_count().unwrap(),
        "scoped != cofactor projecting v0 from (v0∨v3)∧(v2∨v3)"
    );
    assert_eq!(g_scoped.model_count().unwrap(), BigUint::from(12u32));
    crate::test_helpers::check::check_determinism(&g_scoped).unwrap();

    // PMC onto show={v1,v2,v3}: project v0, >>1, vs brute force.
    let clauses = vec![vec![1, 4], vec![3, 4]]; // DIMACS 1-indexed: (v0∨v3)∧(v2∨v3)
    let pmc = ((f).clone().exists_vars_with_strategy(&[VarId(0)], QuantificationStrategy::Structural).unwrap()).model_count().unwrap() >> 1usize;
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
    use crate::diagram::{Arithmetic, LiteralWeights, RationalWeights, WeightStore};
    use num_rational::BigRational;

    let vtree = Arc::new(Vtree::balanced(3));
    let algebra = RationalWeights::from_literals(&[
        LiteralWeights { negative: BigRational::from_integer(1.into()), positive: BigRational::from_integer(2.into()) },
        LiteralWeights { negative: BigRational::from_integer(1.into()), positive: BigRational::from_integer(3.into()) },
        LiteralWeights { negative: BigRational::from_integer(1.into()), positive: BigRational::from_integer(5.into()) },
    ]);
    let mut tdd = Tdd::clause(&vtree, [1, 2]).unwrap();
    tdd.set_weights(WeightStore::new(algebra, Arithmetic::ExactRational)).unwrap();
    // No level is marginal, so this takes the cofactor route, not the
    // structural one that clones the whole diagram.
    assert!(tdd.levels.iter().all(|l| !l.is_marginal()));

    let projected = (tdd).clone().exists_var(VarId(0)).unwrap();
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
    use crate::limits::LimitConfig;
    use crate::limits::OperationError;

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
    let expected = ((f).clone().exists_var(VarId(0)).unwrap()).model_count().unwrap();

    let one_copy = std::mem::size_of_val(f.levels()) as u64;
    let budget = 64u64;
    assert!(budget < one_copy, "the budget has to be below one copy of the level array");

    eng.limits().reset_meters();
    let refused = {
        let _armed = eng.limits().scope(LimitConfig::none().with_memory_budget_bytes(Some(budget)));
        eng.exists_var(f.clone(), VarId(0))
    };
    assert!(
        matches!(refused, Err(OperationError::OverBudget)),
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
        .exists_var(f, VarId(0))
        .expect("the engine takes the next projection after a refusal");
    assert_eq!(out.model_count().unwrap(), expected);
}

/// A variable the vtree does not carry is the caller's input, so the engine
/// form names it in an error rather than aborting the process.
#[test]
fn projecting_a_variable_outside_the_vtree_is_an_error() {
    use crate::OperationError;
    let eng = &crate::engine::Engine::new();
    let vtree = Arc::new(Vtree::balanced(3));
    let f = Tdd::clause(&vtree, [1, -2]).unwrap();
    assert!(matches!(
        eng.exists_var(f.clone(), VarId(7)),
        Err(OperationError::VariableNotInVtree(VarId(7))),
    ));
    assert!(matches!(
        eng.exists_vars(f, &[VarId(0), VarId(7)]),
        Err(OperationError::VariableNotInVtree(VarId(7))),
    ));
}

/// The structural rewrite answers the same request the same way, and so does a
/// request against the constant-false diagram, whose shortcut must not swallow
/// the bad variable.
#[test]
fn projecting_a_variable_outside_the_vtree_is_an_error_on_every_route() {
    use crate::OperationError;
    let eng = &crate::engine::Engine::new();
    let vtree = Arc::new(Vtree::balanced(3));
    assert!(matches!(
        eng.exists_var_with_strategy(Tdd::clause(&vtree, [1, -2]).unwrap(), VarId(3), QuantificationStrategy::Structural),
        Err(OperationError::VariableNotInVtree(VarId(3))),
    ));
    assert!(matches!(
        eng.exists_var(Tdd::zero(&vtree), VarId(3)),
        Err(OperationError::VariableNotInVtree(VarId(3))),
    ));
}

/// The structural rewrite reduces on the caller's engine, so its byte budget
/// reaches the reduction.
#[test]
fn a_structural_projection_is_refused_by_the_engines_budget() {
    let vtree = Arc::new(Vtree::balanced(4));
    let eng = Engine::new();
    let f = Tdd::clause(&vtree, [1, -2]).unwrap() & Tdd::clause(&vtree, [2, 3]).unwrap() & Tdd::clause(&vtree, [-3, 4]).unwrap();
    let _armed = eng.limits().scope(crate::limits::LimitConfig::none().with_memory_budget_bytes(Some(0)));
    assert_eq!(eng.exists_var_with_strategy(f, VarId(1), QuantificationStrategy::Structural).err(), Some(crate::OperationError::OverBudget));
}

#[test]
fn bulk_quantification_validates_late_variables_before_rewriting() {
    use crate::{Engine, OperationError};
    use crate::limits::LimitConfig;
    let eng = Engine::new();
    let tree = Arc::new(Vtree::balanced(3));
    let f = eng.clause(&tree, [1, 2, 3]).unwrap();
    crate::test_helpers::assert_canonical(&f);
    let vars = [VarId(0), VarId(99)];
    let _scope = eng.limits().scope(LimitConfig::none().with_output_node_cap(Some(0)));
    assert_eq!(eng.exists_vars(f.clone(), &vars).unwrap_err(), OperationError::VariableNotInVtree(VarId(99)));
    assert_eq!(eng.and_exists(f.clone(), f.clone(), &vars).unwrap_err(), OperationError::VariableNotInVtree(VarId(99)));
    for how in [QuantificationStrategy::Automatic, QuantificationStrategy::Structural] {
        assert_eq!(eng.exists_vars_with_strategy(f.clone(), &vars, how).unwrap_err(), OperationError::VariableNotInVtree(VarId(99)));
        assert_eq!(eng.and_exists_with_strategy(f.clone(), f.clone(), &vars, how).unwrap_err(), OperationError::VariableNotInVtree(VarId(99)));
    }
}

#[test]
fn bulk_quantification_preserves_first_occurrence_order_and_skips_repeated_rewrites() {
    use crate::Engine;
    let vars = [VarId(19), VarId(2), VarId(8)];
    let tree = Arc::new(Vtree::balanced_over(&vars));
    let f = Engine::new().clause(&tree, vars.map(crate::Literal::pos)).unwrap();
    crate::test_helpers::assert_canonical(&f);
    let order = [vars[2], vars[0], vars[1]];
    let duplicates = [vars[2], vars[0], vars[2], vars[1], vars[0]];
    for how in [QuantificationStrategy::Automatic, QuantificationStrategy::Structural] {
        let eng = Engine::new();
        let prepared = crate::apply::project::quantification_targets(&eng, &tree, &duplicates).unwrap();
        assert_eq!(prepared, order.map(|var| tree.leaf_of(var).unwrap()));
        let expected = eng.exists_vars_with_strategy(f.clone(), &order, how).unwrap();
        let actual = eng.exists_vars_with_strategy(f.clone(), &duplicates, how).unwrap();
        assert!(eng.equivalent(&actual, &expected).unwrap());
        assert_eq!(actual.model_count().unwrap(), BigUint::from(8u32));
        crate::test_helpers::assert_canonical(&actual);
        crate::test_helpers::assert_canonical(&expected);
    }
}
