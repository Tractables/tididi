use std::sync::Arc;

use num_bigint::BigUint;

use super::*;
use crate::diagram::{Arithmetic, RationalWeights, WeightStore};
use crate::test_helpers::{assert_canonical, compile_clauses_on, node_value, rand_cnf, under, vtree_shapes, CnfShape, Lcg};
use crate::{and, or};

/// The count `f` gives the assignment `bits` of variables `1..=n`, bit `i`
/// the value of variable `i + 1`; summed-out variables are read past.
fn value(f: &Tdd, n: u32, bits: u64) -> BigUint {
    let assignment: Vec<bool> = (0..n).map(|i| bits >> i & 1 == 1).collect();
    node_value(f, f.output, &|_| false, &assignment)
}

/// Assert that `sum` counts every assignment as `f` and `g` together do, and
/// is canonical.
fn assert_sum(sum: &Tdd, f: &Tdd, g: &Tdd, n: u32, label: &str) {
    assert_canonical(sum);
    for bits in 0..1u64 << n {
        assert_eq!(value(sum, n, bits), value(f, n, bits) + value(g, n, bits), "{label}: assignment {bits:b}");
    }
    assert_eq!(sum.model_count().unwrap(), f.model_count().unwrap() + g.model_count().unwrap(), "{label}: total");
}

/// The subtrees a test sums out on `vtree`: one internal subtree below the
/// root, that subtree and a leaf outside it, two disjoint internal subtrees,
/// or one leaf.
fn targets(rng: &mut Lcg, vtree: &Vtree) -> Vec<VtreeIdx> {
    let inner: Vec<VtreeIdx> = vtree.internal_bottomup().map(|(t, _, _)| t).filter(|&t| t != vtree.root()).collect();
    let leaves = |outside: Option<VtreeIdx>| -> Vec<VtreeIdx> {
        (0..vtree.num_nodes() as u32)
            .map(VtreeIdx)
            .filter(|&t| vtree.node(t).is_leaf() && outside.is_none_or(|s| !under(vtree, t, s)))
            .collect()
    };
    let pick = |rng: &mut Lcg, from: &[VtreeIdx]| from[rng.below(from.len() as u64) as usize];
    match (rng.below(4), inner.is_empty()) {
        (0, false) => vec![pick(rng, &inner)],
        (1, false) => {
            let summed = pick(rng, &inner);
            vec![summed, pick(rng, &leaves(Some(summed)))]
        }
        (2, false) => {
            let summed = pick(rng, &inner);
            let apart: Vec<VtreeIdx> = inner.iter().copied().filter(|&t| !under(vtree, t, summed) && !under(vtree, summed, t)).collect();
            match apart.is_empty() {
                true => vec![summed],
                false => vec![summed, pick(rng, &apart)],
            }
        }
        _ => vec![pick(rng, &leaves(None))],
    }
}

/// Random functions summed out over the same subtrees add to the sum of
/// their counts, on every vtree shape and assignment.
#[test]
fn adds_the_counts_of_functions_summed_out_alike() {
    let mut rng = Lcg::new(17);
    for round in 0..24 {
        let n = 3 + (round % 6) as u32;
        let shape = CnfShape { clauses: n as usize, width: 2 + (round % 3) };
        for (name, vtree) in vtree_shapes(n) {
            let eng = Engine::new();
            let summed = targets(&mut rng, &vtree);
            let mut f = compile_clauses_on(&eng, &vtree, &rand_cnf(&mut rng, n, shape));
            let mut g = compile_clauses_on(&eng, &vtree, &rand_cnf(&mut rng, n, shape));
            for d in [&mut f, &mut g] {
                eng.marginalize_levels(d, &summed).unwrap();
                eng.minimize(d).unwrap();
            }
            let sum = eng.add(f.clone(), g.clone()).unwrap();
            assert_sum(&sum, &f, &g, n, &format!("round {round}, {name}, summed {summed:?}"));
            if !f.is_zero() && !g.is_zero() {
                for t in vtree.bottomup() {
                    assert_eq!(sum.level(t).is_marginal(), f.level(t).is_marginal(), "{name}: level {t:?}");
                }
            }
        }
    }
}

/// A diagram added to itself doubles every count, the models of both
/// operands meeting everywhere.
#[test]
fn adds_a_diagram_to_itself() {
    let mut rng = Lcg::new(5);
    for n in 3..=6 {
        for (name, vtree) in vtree_shapes(n) {
            let eng = Engine::new();
            let mut f = compile_clauses_on(&eng, &vtree, &rand_cnf(&mut rng, n, CnfShape { clauses: n as usize, width: 3 }));
            eng.marginalize_levels(&mut f, &targets(&mut rng, &vtree)).unwrap();
            eng.minimize(&mut f).unwrap();
            let twice = eng.add(f.clone(), f.clone()).unwrap();
            assert_sum(&twice, &f, &f, n, name);
        }
    }
}

/// With no summed-out level, functions with no common model add to their
/// disjunction.
#[test]
fn boolean_operands_with_no_common_model_add_to_their_disjunction() {
    let mut rng = Lcg::new(23);
    for n in 2..=6 {
        for (name, vtree) in vtree_shapes(n) {
            let eng = Engine::new();
            let split = 1 + rng.below(u64::from(n)) as i32;
            let shape = CnfShape { clauses: n as usize, width: 3 };
            let f = and(compile_clauses_on(&eng, &vtree, &rand_cnf(&mut rng, n, shape)), Tdd::cube(&vtree, [split]).unwrap()).unwrap();
            let g = and(compile_clauses_on(&eng, &vtree, &rand_cnf(&mut rng, n, shape)), Tdd::cube(&vtree, [-split]).unwrap()).unwrap();
            let sum = eng.add(f.clone(), g.clone()).unwrap();
            assert!(!sum.has_marginal_level(), "{name}");
            assert!(sum.equivalent(&or(f.clone(), g.clone()).unwrap()).unwrap(), "{name}");
            assert_sum(&sum, &f, &g, n, name);
        }
    }
}

/// With no summed-out level a shared model would count twice, which no
/// structural diagram holds.
#[test]
fn boolean_operands_sharing_a_model_are_refused() {
    let mut rng = Lcg::new(29);
    for n in 2..=6 {
        for (name, vtree) in vtree_shapes(n) {
            let eng = Engine::new();
            let shape = CnfShape { clauses: n as usize, width: 3 };
            let f = compile_clauses_on(&eng, &vtree, &rand_cnf(&mut rng, n, shape));
            let g = compile_clauses_on(&eng, &vtree, &rand_cnf(&mut rng, n, shape));
            let shared = !eng.and(f.clone(), g.clone()).unwrap().is_zero();
            match eng.add(f.clone(), g.clone()) {
                Err(OperationError::SharedModel) => assert!(shared, "{name}: refused with no shared model"),
                Ok(sum) => {
                    assert!(!shared, "{name}: a shared model was summed");
                    assert_sum(&sum, &f, &g, n, name);
                }
                Err(e) => panic!("{name}: {e}"),
            }
        }
    }
    // On one variable the root is a leaf.
    let vtree = Arc::new(Vtree::balanced(1));
    let x = || Tdd::cube(&vtree, [1]).unwrap();
    assert_eq!(add(x(), x()).unwrap_err(), OperationError::SharedModel);
    assert_eq!(add(x(), Tdd::one(&vtree)).unwrap_err(), OperationError::SharedModel);
    let both = add(x(), Tdd::cube(&vtree, [-1]).unwrap()).unwrap();
    assert!(both.equivalent(&Tdd::one(&vtree)).unwrap());
}

/// A level summed out in one operand only is refused, naming a level under
/// the subtree summed out.
#[test]
fn a_level_summed_out_on_one_side_only_is_refused() {
    let vtree = Arc::new(Vtree::balanced(4));
    let (left, right) = vtree.children(vtree.root());
    let mut f = Tdd::clause(&vtree, [1, 2]).unwrap();
    let mut g = Tdd::clause(&vtree, [1, 3]).unwrap();
    f.marginalize_levels(&[left]).unwrap();
    match add(f.clone(), g.clone()) {
        Err(OperationError::MarginalLevel(t)) => assert!(under(&vtree, t, left)),
        other => panic!("expected a marginal level, got {other:?}"),
    }
    g.marginalize_levels(&[right]).unwrap();
    assert!(matches!(add(f, g), Err(OperationError::MarginalLevel(_))));
}

/// A false operand is the sum's identity, whatever levels the other summed.
#[test]
fn a_false_operand_returns_the_other() {
    let vtree = Arc::new(Vtree::balanced(4));
    let (left, _) = vtree.children(vtree.root());
    let mut f = Tdd::clause(&vtree, [1, 3]).unwrap();
    f.marginalize_levels(&[left]).unwrap();
    for sum in [add(f.clone(), Tdd::zero(&vtree)).unwrap(), add(Tdd::zero(&vtree), f.clone()).unwrap()] {
        assert_sum(&sum, &f, &Tdd::zero(&vtree), 4, "false operand");
    }
}

/// Operands on different vtree allocations are refused.
#[test]
fn operands_on_different_vtrees_are_refused() {
    let a = Arc::new(Vtree::balanced(3));
    let b = Arc::new(Vtree::balanced(3));
    let err = add(Tdd::cube(&a, [1]).unwrap(), Tdd::cube(&b, [-1]).unwrap()).unwrap_err();
    assert_eq!(err, OperationError::VtreeMismatch);
}

/// Diagrams summed out to one value add their values.
#[test]
fn wholly_summed_out_diagrams_add_their_values() {
    let vtree = Arc::new(Vtree::balanced(4));
    let root = vtree.root();
    let mut f = Tdd::clause(&vtree, [1, 2]).unwrap();
    let mut g = Tdd::cube(&vtree, [3, -4]).unwrap();
    f.marginalize_levels(&[root]).unwrap();
    g.marginalize_levels(&[root]).unwrap();
    let sum = add(f, g).unwrap();
    assert_eq!(sum.model_count().unwrap(), BigUint::from(12u32 + 4));
    assert_canonical(&sum);
}

/// Counts past `u128` are added exactly, `g`'s overflowed values held past
/// `f`'s.
#[test]
fn counts_past_u128_add_exactly() {
    let vtree = Arc::new(Vtree::balanced(260));
    let (left, _) = vtree.children(vtree.root());
    let mut f = Tdd::one(&vtree);
    let mut g = Tdd::cube(&vtree, [1]).unwrap();
    f.marginalize_levels(&[left]).unwrap();
    g.marginalize_levels(&[left]).unwrap();
    let two = BigUint::from(2u32);
    let sum = add(f.clone(), g).unwrap();
    assert_eq!(sum.model_count().unwrap(), two.pow(260) + two.pow(259));
    assert_canonical(&sum);
    let twice = add(f.clone(), f).unwrap();
    assert_eq!(twice.model_count().unwrap(), two.pow(261));
    assert_canonical(&twice);
}

/// Summed-out levels that hold weights are not added.
#[test]
fn weights_on_summed_out_levels_are_refused() {
    let vtree = Arc::new(Vtree::balanced(4));
    let (left, _) = vtree.children(vtree.root());
    let weighted = |mut f: Tdd| {
        f.set_weights(WeightStore::new(RationalWeights::unit(4), Arithmetic::ExactRational)).unwrap();
        f.marginalize_levels(&[left]).unwrap();
        f
    };
    let f = weighted(Tdd::clause(&vtree, [1, 2]).unwrap());
    let g = weighted(Tdd::cube(&vtree, [1, 3]).unwrap());
    assert_eq!(add(f, g).unwrap_err(), OperationError::IncompatibleWeights);
}

/// A sum conjoins with a function of its structural variables as any
/// summed-out diagram does: the counts the function keeps.
#[test]
fn a_sum_conjoins_with_a_filter_on_its_structural_variables() {
    let mut rng = Lcg::new(41);
    let n = 6;
    for (name, vtree) in vtree_shapes(n) {
        let eng = Engine::new();
        let summed = targets(&mut rng, &vtree);
        let shape = CnfShape { clauses: n as usize, width: 3 };
        let mut f = compile_clauses_on(&eng, &vtree, &rand_cnf(&mut rng, n, shape));
        let mut g = compile_clauses_on(&eng, &vtree, &rand_cnf(&mut rng, n, shape));
        for d in [&mut f, &mut g] {
            eng.marginalize_levels(d, &summed).unwrap();
            eng.minimize(d).unwrap();
        }
        let sum = eng.add(f, g).unwrap();
        let free = crate::test_helpers::free_vars(&sum);
        let Some(v) = (1..=n).find(|&v| free[v as usize - 1]) else { continue };
        let kept = eng.and(sum.clone(), Tdd::cube(&vtree, [v as i32]).unwrap()).unwrap();
        for bits in 0..1u64 << n {
            let on = bits >> (v - 1) & 1 == 1;
            let expected = if on { value(&sum, n, bits) } else { BigUint::ZERO };
            assert_eq!(value(&kept, n, bits), expected, "{name}: x{v}, assignment {bits:b}");
        }
    }
}
