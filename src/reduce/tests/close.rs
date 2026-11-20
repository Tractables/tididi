//! A closing reduction on a home: it reaches the diagram that writing the
//! true labels and then trimming reaches, and minimizing it again changes
//! nothing.

use crate::test_helpers::check::reachable_true_labels;
use crate::test_helpers::{assert_same_shape, rand_conj, trim_by_rebuild, vtree_shapes};
use crate::vtree::rng::Lcg;
use crate::vtree::Home;

use super::*;

fn leaf_vars(t: &Tdd) -> Vec<VarId> {
    let v = t.vtree();
    v.bottomup().filter(|&n| v.node(n).is_leaf()).map(|n| v.leaf_var(n)).collect()
}

#[test]
fn a_closing_reduce_is_a_plain_reduce_then_a_trim() {
    let eng = Engine::new();
    let mut dropped = 0;
    for seed in 0..24u64 {
        let mut rng = Lcg::new(seed ^ 0xc105e);
        let n = 5 + rng.below(5) as u32;
        for (shape, vtree) in vtree_shapes(n).into_iter().filter(|(s, _)| matches!(*s, "balanced" | "linear")) {
            let home = Home::new(vtree);
            let make = |rng: &mut Lcg| {
                let mut f = rand_conj(home.vtree(), n, 1 + rng.below(5), 3, false, rng);
                f.minimize().unwrap();
                f.trim(&home).unwrap();
                f
            };
            let (f, g) = (make(&mut rng), make(&mut rng));
            let case = format!("seed {seed}, {n} variables, {shape}");
            for f in [&f, &g] {
                assert_eq!(reachable_true_labels(f), Ok(None), "{case}: operand");
            }
            let product = eng.and(&f, &g).unwrap();
            for full in [true, false] {
                let plan = || if full { ReductionPlan::default() } else { ReductionPlan::Contract };
                let mut closed = product.clone();
                eng.reduce(&mut closed, plan()).unwrap();
                let mut plain = product.clone();
                eng.reduce_plain(&mut plain, plan()).unwrap();
                let before = leaf_vars(&plain);
                trim_by_rebuild(&mut plain, &home).unwrap();
                dropped += before.len() - leaf_vars(&plain).len();
                assert_eq!(reachable_true_labels(&closed), Ok(None), "{case}");
                assert!(eng.xor(closed.clone(), product.clone()).unwrap().is_zero(), "{case}");
                if !full {
                    continue;
                }
                assert_eq!(leaf_vars(&closed), leaf_vars(&plain), "{case}");
                assert_same_shape(&closed, &plain, &case);
                let mut again = closed.clone();
                again.minimize().unwrap();
                assert_eq!(leaf_vars(&again), leaf_vars(&closed), "{case}: second minimize");
                assert_same_shape(&again, &closed, &case);
                assert!(!trim_by_rebuild(&mut again, &home).unwrap(), "{case}: trim after minimize");
            }
        }
    }
    assert!(dropped > 0, "no case contracted a leaf");
}

#[test]
fn minimize_after_embed_onto_a_home() {
    let eng = Engine::new();
    for seed in 0..12u64 {
        let mut rng = Lcg::new(seed ^ 0xe3b);
        let small = Arc::new(Vtree::balanced(4));
        let f = rand_conj(&small, 4, 1 + rng.below(4), 3, false, &mut rng);
        let home = Home::new(Vtree::balanced(8));
        let shift = 4 * rng.below(2) as u32;
        let (mut placed, _) = eng.embed(&f, home.vtree(), |v| VarId(v.0 + shift)).unwrap();
        let before = placed.clone();
        eng.minimize(&mut placed).unwrap();
        assert!(eng.xor(placed, before).unwrap().is_zero(), "seed {seed}");
    }
}

/// A random I1 diagram on the projection of `home` on its support.
fn i1(home: &Home, n: u32, rng: &mut Lcg) -> Tdd {
    let mut f = rand_conj(home.vtree(), n, 1 + rng.below(5), 3, false, rng);
    f.minimize().unwrap();
    f.trim(home).unwrap();
    f
}

#[test]
fn quantification_and_conditioning_close_as_plain_then_trim() {
    let eng = Engine::new();
    let mut dropped = 0;
    for seed in 0..40u64 {
        let mut rng = Lcg::new(seed ^ 0x5eed);
        let n = 4 + rng.below(6) as u32;
        let home = Home::new(Vtree::random(n, seed));
        let f = i1(&home, n, &mut rng);
        if f.vtree().num_nodes() == 0 {
            continue;
        }
        let vars: Vec<VarId> = (1..=n).map(VarId).filter(|_| rng.below(3) == 0).collect();
        let case = format!("seed {seed}, {n} variables, {vars:?}");
        let mut closed = eng.exists_vars(f.clone(), &vars).unwrap();
        eng.minimize(&mut closed).unwrap();
        let mut plain = crate::apply::project::exists_vars_on(&eng, f.clone(), &vars, Some(ReductionPlan::default()), false).unwrap();
        dropped += usize::from(trim_by_rebuild(&mut plain, &home).unwrap());
        assert_eq!(reachable_true_labels(&closed), Ok(None), "{case}");
        assert_eq!(leaf_vars(&closed), leaf_vars(&plain), "{case}");
        if closed.vtree().num_nodes() > 0 {
            assert_same_shape(&closed, &plain, &case);
        }
        for prune in [true, false] {
            let plan = if prune { ReductionPlan::Prune } else { ReductionPlan::Contract };
            let other = crate::apply::with_plan::exists_vars_with(&eng, f.clone(), &vars, plan).unwrap();
            assert_eq!(reachable_true_labels(&other), Ok(None), "{case}, prune {prune}");
            assert!(eng.equivalent(&other, &closed).unwrap(), "{case}, prune {prune}");
        }
        let var = VarId(1 + rng.below(u64::from(n)) as u32);
        let value = rng.coin();
        let mut conditioned = eng.condition_var(f.clone(), var, value).unwrap();
        eng.minimize(&mut conditioned).unwrap();
        let literal = eng.literal(home.vtree(), Literal { var, sign: value }).unwrap();
        let mut by_exists = crate::apply::project::exists_vars_on(&eng, eng.and(&f, &literal).unwrap(), &[var], Some(ReductionPlan::default()), false).unwrap();
        trim_by_rebuild(&mut by_exists, &home).unwrap();
        assert_eq!(reachable_true_labels(&conditioned), Ok(None), "{case}, {var:?} = {value}");
        assert_eq!(leaf_vars(&conditioned), leaf_vars(&by_exists), "{case}, {var:?} = {value}");
        if conditioned.vtree().num_nodes() > 0 {
            assert_same_shape(&conditioned, &by_exists, &case);
        }
    }
    assert!(dropped > 0, "no plain result had a leaf to trim");
}

#[test]
fn a_seed_variable_the_diagram_lacks_is_skipped() {
    let eng = Engine::new();
    let home = Home::new(Vtree::balanced(4));
    let f = eng.clause(home.vtree(), [1, -3]).unwrap();
    assert_eq!(leaf_vars(&f), [VarId(1), VarId(3)]);
    let quantified = eng.exists_vars(f.clone(), &[VarId(2), VarId(3)]).unwrap();
    assert!(quantified.is_one() && quantified.vtree().num_nodes() == 0);
    let conditioned = eng.condition(f.clone(), [Literal::pos(VarId(4)), Literal::neg(VarId(1))]).unwrap();
    assert_eq!(leaf_vars(&conditioned), [VarId(3)]);
    assert!(eng.equivalent(&conditioned, &eng.literal(home.vtree(), -3).unwrap()).unwrap());
    let forall = eng.forall_vars(f, &[VarId(2), VarId(1)]).unwrap();
    assert_eq!(leaf_vars(&forall), [VarId(3)]);
}

/// ((x y) z), ¬x ∧ ¬y ∧ ¬z: under `Contract` the root level holds the parts
/// of the output and their union, which name the same nodes of (x y).
#[test]
fn exists_under_contract_closes_on_the_union() {
    let eng = Engine::new();
    let leaf = |v| Vtree::leaf(VarId(v));
    let home = Home::new(Vtree::join(&Vtree::join(&leaf(1), &leaf(2)).unwrap(), &leaf(3)).unwrap());
    let f = eng.cube(home.vtree(), [-1, -2, -3]).unwrap();
    let g = crate::apply::with_plan::exists_vars_with(&eng, f, &[VarId(3)], ReductionPlan::Contract).unwrap();
    assert_eq!(leaf_vars(&g), [VarId(1), VarId(2)]);
    assert_eq!(reachable_true_labels(&g), Ok(None));
    assert!(eng.equivalent(&g, &eng.cube(home.vtree(), [-1, -2]).unwrap()).unwrap());
}

#[test]
fn forall_of_an_embedded_false_is_false() {
    let eng = Engine::new();
    let home = Home::new(Vtree::balanced(3));
    let zero = crate::test_helpers::padded(Tdd::zero(home.vtree()), home.vtree());
    assert!(zero.vtree().num_nodes() > 0);
    let g = eng.forall_vars(zero, &[VarId(1)]).unwrap();
    assert!(g.is_zero() && g.vtree().num_nodes() == 0);
}

/// (x ∨ y) ∧ (¬x ∨ y) is y: the product's reduction leaves x before the
/// quantification of y.
#[test]
fn and_exists_after_a_product_that_narrows() {
    let eng = Engine::new();
    let home = Home::new(Vtree::balanced(2));
    let f = eng.clause(home.vtree(), [1, 2]).unwrap();
    let g = eng.clause(home.vtree(), [-1, 2]).unwrap();
    let h = eng.and_exists(f, g, &[VarId(2)]).unwrap();
    assert!(h.is_one() && h.vtree().num_nodes() == 0);
}

#[test]
fn projections_and_counts_of_a_diagram_its_quantification_narrows() {
    let eng = Engine::new();
    let home = Home::new(Vtree::balanced(2));
    let f = eng.xor(eng.literal(home.vtree(), 1).unwrap(), eng.literal(home.vtree(), 2).unwrap()).unwrap();
    assert_eq!(leaf_vars(&f), [VarId(1), VarId(2)]);
    let kept = eng.project_to_vars(&f, |v| (v == VarId(2)).then_some(VarId(1)), 1).unwrap();
    assert_eq!(kept.vtree().num_leaves(), 1, "the projection stays on the vtree it was asked for");
    assert_eq!(kept.model_count().unwrap(), 2u32.into(), "x xor y is true over y");
    assert_eq!(eng.projected_model_count(&f, &[VarId(2)]).unwrap(), 2u32.into());
}

#[test]
fn renaming_builds_on_the_variables_used() {
    let eng = Engine::new();
    let leaf = |v| Vtree::leaf(VarId(v));
    let home = Home::new(Vtree::join(&leaf(1), &Vtree::join(&leaf(2), &leaf(3)).unwrap()).unwrap());
    let f = eng.cube(home.vtree(), [1, 2, 3]).unwrap();
    let g = eng.rename_vars(f, &[(VarId(2), VarId(3))]).unwrap();
    assert_eq!(reachable_true_labels(&g), Ok(None));
    assert_eq!(leaf_vars(&g), [VarId(1), VarId(3)]);
    assert!(eng.equivalent(&g, &eng.cube(home.vtree(), [1, 3]).unwrap()).unwrap());

    let home = Home::new(Vtree::balanced(2));
    let x = eng.literal(home.vtree(), 1).unwrap();
    let y = eng.rename_vars(x, &[(VarId(1), VarId(2))]).unwrap();
    assert_eq!(leaf_vars(&y), [VarId(2)]);
    assert!(eng.equivalent(&y, &eng.literal(home.vtree(), 2).unwrap()).unwrap());
}

#[test]
fn a_constructor_given_the_root_or_a_projection_returns_the_same_vtree() {
    let eng = Engine::new();
    let home = Home::new(Vtree::balanced(6));
    let own = Vtree::projection(&home, [1, 2, 4, 5].map(VarId)).unwrap();
    for vtree in [home.vtree(), &own] {
        let built = [
            eng.literal(vtree, -2).unwrap(),
            eng.cube(vtree, [1, -6]).unwrap(),
            eng.clause(vtree, [3, -5]).unwrap(),
            eng.one(vtree),
            eng.zero(vtree),
            Tdd::from_models(vtree, &[VarId(1), VarId(5)], &[0b01, 0b11]).unwrap(),
        ];
        let expected: [&[u32]; 6] = [&[2], &[1, 6], &[3, 5], &[], &[], &[1]];
        for (t, vars) in built.iter().zip(expected) {
            let want: Vec<VarId> = vars.iter().map(|&v| VarId(v)).collect();
            assert_eq!(leaf_vars(t), want);
            assert!(t.vtree().is_on(&home));
            assert_eq!(reachable_true_labels(t), Ok(None));
            if t.vtree().num_nodes() > 0 {
                let alone = Vtree::projection(&home, want.iter().copied()).unwrap();
                assert_eq!(t.vtree().num_nodes(), alone.num_nodes());
                assert_eq!(t.vtree().root(), alone.root());
            }
        }
    }
}

/// On a home of four variables, a rename onto variables of the home answers
/// as on a bare root of the same shape, on the projection of the variables
/// the result uses: onto a variable `f` has no leaf for, a swap with one,
/// onto a leaf `f` keeps. A target outside the home is refused on both.
#[test]
fn a_rename_onto_a_home_variable_answers_as_on_a_bare_root() {
    use crate::test_helpers::check::check_trimmed;
    use crate::test_helpers::eval;
    let eng = Engine::new();
    let home = Home::new(Vtree::balanced(4));
    let bare = Arc::new(Vtree::balanced(4));
    let f_on = |vtree: &Arc<Vtree>| eng.or(eng.literal(vtree, 1).unwrap(), eng.literal(vtree, 2).unwrap()).unwrap();
    let f = f_on(home.vtree());
    assert_eq!(leaf_vars(&f), [VarId(1), VarId(2)]);
    let v = VarId;
    type Case<'a> = (&'a [(VarId, VarId)], &'a [VarId]);
    let cases: [Case<'_>; 3] = [
        (&[(v(1), v(3))], &[v(2), v(3)]),
        (&[(v(1), v(4)), (v(4), v(1))], &[v(2), v(4)]),
        (&[(v(1), v(2))], &[v(2)]),
    ];
    for (renames, leaves) in cases {
        let got = eng.rename_vars(f.clone(), renames).unwrap();
        check_trimmed(&got, true).unwrap_or_else(|e| panic!("{renames:?}: {e}"));
        let mut got_leaves = leaf_vars(&got);
        got_leaves.sort_unstable();
        assert_eq!(got_leaves, leaves, "{renames:?}");
        let want = eng.rename_vars(f_on(&bare), renames).unwrap();
        for a in 0..16u32 {
            let asn: Vec<bool> = (0..4).map(|i| a >> i & 1 == 1).collect();
            assert_eq!(eval(&got, &asn), eval(&want, &asn), "{renames:?} at {asn:?}");
        }
    }
    for vtree in [home.vtree(), &bare] {
        assert_eq!(eng.rename_vars(f_on(vtree), &[(v(1), v(9))]).err(), Some(OperationError::VariableNotInVtree(v(9))));
    }
}
