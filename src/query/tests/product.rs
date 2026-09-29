//! `Engine::marginal_product_count` against enumeration, against the
//! conjunction's count where the two operands summed out disjoint scopes, and
//! on the refusals and big values its contract names.

use super::*;
use crate::test_helpers::{
    assert_canonical, compile_clauses_on, free_vars, marginal_diagrams, rand_cnf, stopping_engine, truth_table,
    vtree_shapes, CnfShape, Lcg,
};
use crate::vtree::Vtree;
use std::sync::Arc;

/// `m_T` of clauses whose variables outside `kept` are summed out: each
/// assignment's projection onto `kept` (a bit mask, variable `i + 1` in bit
/// `i`) with its number of models.
fn marginal(n: u32, clauses: &[Vec<i32>], kept: u64) -> Vec<u64> {
    let mut m = vec![0u64; 1 << n];
    for (a, sat) in truth_table(n, clauses).into_iter().enumerate() {
        if sat {
            m[a & kept as usize] += 1;
        }
    }
    m
}

/// The variables of `t` that neither summed out, as a bit mask.
fn kept_mask(t: &Tdd) -> u64 {
    free_vars(t).iter().enumerate().filter(|&(_, &f)| f).map(|(i, _)| 1u64 << i).sum()
}

/// `Σ_{x ⊆ kept_l ∩ kept_r} m_l(x) · m_r(x)`, or `None` where one operand's
/// count depends on a variable only the other summed out.
fn brute(n: u32, cl: &[Vec<i32>], kl: u64, cr: &[Vec<i32>], kr: u64) -> Option<BigUint> {
    let (ml, mr) = (marginal(n, cl, kl), marginal(n, cr, kr));
    let both = kl & kr;
    for x in 0..1u64 << n {
        if ml[(x & kl) as usize] != ml[(x & both) as usize] || mr[(x & kr) as usize] != mr[(x & both) as usize] {
            return None;
        }
    }
    let mut sum = BigUint::ZERO;
    for x in 0..1u64 << n {
        if x & !both == 0 {
            sum += BigUint::from(ml[x as usize]) * BigUint::from(mr[x as usize]);
        }
    }
    Some(sum)
}

/// A random downward-closed set of summed-out subtrees: up to two vtree
/// nodes other than the root.
fn draw_targets(rng: &mut Lcg, vtree: &Vtree) -> Vec<VtreeIdx> {
    let nodes: Vec<VtreeIdx> = (0..vtree.num_nodes() as u32).map(VtreeIdx).filter(|&t| t != vtree.root()).collect();
    (0..rng.below(3)).map(|_| nodes[rng.below(nodes.len() as u64) as usize]).collect()
}

fn summed(eng: &Engine, vtree: &Arc<Vtree>, clauses: &[Vec<i32>], targets: &[VtreeIdx]) -> Tdd {
    let mut t = compile_clauses_on(eng, vtree, clauses);
    assert_canonical(&t);
    if !t.is_zero() && !targets.is_empty() {
        eng.marginalize_levels(&mut t, targets).unwrap();
        eng.minimize(&mut t).unwrap();
        assert_canonical(&t);
    }
    t
}

/// Clauses over `n` variables that avoid every variable in `avoid`.
fn cnf_avoiding(rng: &mut Lcg, n: u32, avoid: u64) -> Vec<Vec<i32>> {
    let shape = CnfShape { clauses: n as usize, width: 3 };
    rand_cnf(rng, n, shape)
        .into_iter()
        .map(|c| c.into_iter().filter(|l| avoid >> (l.unsigned_abs() - 1) & 1 == 0).collect::<Vec<_>>())
        .filter(|c: &Vec<i32>| !c.is_empty())
        .collect()
}

/// Every drawn pair of operands, on every vtree shape: the count equals
/// enumeration whenever it is given, is given whenever the two summed out
/// the same scope, and the general walk and the one-diagram walk agree.
#[test]
fn marginal_products_match_enumeration() {
    let mut rng = Lcg::new(7);
    let (mut same_scope, mut mixed, mut refused) = (0, 0, 0);
    for round in 0..40 {
        let n = 3 + (round % 5) as u32;
        for (name, vtree) in vtree_shapes(n) {
            let eng = Engine::new();
            let shape = CnfShape { clauses: n as usize, width: 3 };
            let (cl, cr) = (rand_cnf(&mut rng, n, shape), rand_cnf(&mut rng, n, shape));
            let tl = draw_targets(&mut rng, &vtree);
            let tr = if rng.coin() { tl.clone() } else { draw_targets(&mut rng, &vtree) };
            let (l, r) = (summed(&eng, &vtree, &cl, &tl), summed(&eng, &vtree, &cr, &tr));
            let (kl, kr) = (kept_mask(&l), kept_mask(&r));
            let want = brute(n, &cl, kl, &cr, kr);
            match eng.marginal_product_count(&l, &r) {
                Ok(got) => {
                    assert_eq!(Some(&got), want.as_ref(), "{name} n={n} round {round}");
                    let general = product::<BigUint>(eng.limits(), &l, &r, false).ok().unwrap();
                    assert_eq!(got, general, "{name} n={n} round {round}: the general walk");
                    if kl == kr { same_scope += 1 } else { mixed += 1 }
                }
                Err(e) => {
                    assert_eq!(std::mem::discriminant(&e), std::mem::discriminant(&OperationError::MarginalLevel(VtreeIdx(0))));
                    assert!(kl != kr, "{name} n={n} round {round}: refused over one scope");
                    refused += 1;
                }
            }
            // One diagram against itself: the sum of its squared counts.
            let got = eng.marginal_product_count(&l, &l).unwrap();
            assert_eq!(Some(got.clone()), brute(n, &cl, kl, &cl, kl), "{name} n={n} round {round}: square");
            let general = product::<BigUint>(eng.limits(), &l, &l.clone(), false).ok().unwrap();
            assert_eq!(got, general, "{name} n={n} round {round}: the square's general walk");
        }
    }
    assert!(same_scope > 50 && mixed > 20 && refused > 0, "{same_scope} {mixed} {refused}");
}

/// Operands that summed out disjoint scopes, each free of the other's summed
/// variables: the product is the conjunction's count.
#[test]
fn disjoint_scopes_count_the_conjunction() {
    let mut rng = Lcg::new(11);
    let mut compared = 0;
    for round in 0..30 {
        let n = 4 + (round % 4) as u32;
        for (name, vtree) in vtree_shapes(n) {
            let eng = Engine::new();
            let nodes: Vec<VtreeIdx> = (0..vtree.num_nodes() as u32).map(VtreeIdx).filter(|&t| t != vtree.root()).collect();
            let tl = nodes[rng.below(nodes.len() as u64) as usize];
            let under = |t: VtreeIdx| -> u64 {
                (1..=n).filter(|&v| crate::test_helpers::under(&vtree, vtree.leaf_of(crate::vtree::VarId(v)).unwrap(), t)).map(|v| 1u64 << (v - 1)).sum()
            };
            let yl = under(tl);
            let Some(&tr) = nodes.iter().find(|&&t| under(t) & yl == 0 && rng.coin()) else { continue };
            let yr = under(tr);
            let cl = cnf_avoiding(&mut rng, n, yr);
            let cr = cnf_avoiding(&mut rng, n, yl);
            let (l, r) = (summed(&eng, &vtree, &cl, &[tl]), summed(&eng, &vtree, &cr, &[tr]));
            let got = eng.marginal_product_count(&l, &r).unwrap();
            assert_eq!(Some(&got), brute(n, &cl, kept_mask(&l), &cr, kept_mask(&r)).as_ref(), "{name} round {round}");
            assert_eq!(got, eng.and_model_count(l, r, &[]).unwrap(), "{name} round {round}: the conjunction");
            compared += 1;
        }
    }
    assert!(compared > 40, "{compared}");
}

/// Structural operands: the model count of their conjunction.
#[test]
fn structural_operands_count_their_conjunction() {
    let mut rng = Lcg::new(3);
    for round in 0..20 {
        let n = 3 + (round % 6) as u32;
        for (name, vtree) in vtree_shapes(n) {
            let eng = Engine::new();
            let shape = CnfShape { clauses: n as usize, width: 3 };
            let (l, r) = (compile_clauses_on(&eng, &vtree, &rand_cnf(&mut rng, n, shape)), compile_clauses_on(&eng, &vtree, &rand_cnf(&mut rng, n, shape)));
            assert_canonical(&l);
            assert_canonical(&r);
            let want = eng.model_count(&eng.and(l.clone(), r.clone()).unwrap()).unwrap();
            assert_eq!(eng.marginal_product_count(&l, &r).unwrap(), want, "{name} round {round}");
            assert_eq!(eng.marginal_product_count(&l, &l).unwrap(), eng.model_count(&l).unwrap(), "{name}: f ∧ f");
        }
    }
}

/// Diagrams whose marginal references carry their counts inline, squared.
#[test]
fn inline_marginal_references_are_read_as_values() {
    for (t, _) in marginal_diagrams(5, 30, 4..8) {
        let eng = Engine::new();
        let general = product::<BigUint>(eng.limits(), &t, &t, false).ok().unwrap();
        let fast = eng.marginal_product_count(&t, &t).unwrap();
        assert_eq!(fast, general);
        // Enumerated: the square of each kept assignment's count.
        let n = t.vtree().num_vars();
        let kept = kept_mask(&t);
        let mut sum = BigUint::ZERO;
        for x in 0..1u64 << n {
            if x & !kept == 0 {
                let a: Vec<bool> = (0..n).map(|i| x >> i & 1 == 1).collect();
                let m = crate::test_helpers::node_value(&t, t.output(), &|_| false, &a);
                sum += &m * &m;
            }
        }
        assert_eq!(fast, sum);
    }
}

/// Counts past `u128`: a scope of 130 summed variables, squared, is exact.
#[test]
fn counts_past_u128_are_exact() {
    let eng = Engine::new();
    let vtree = Arc::new(Vtree::balanced(260));
    let (left, right) = vtree.children(vtree.root());
    let mut f = eng.clause(&vtree, [1]).unwrap();
    eng.marginalize_levels(&mut f, &[right]).unwrap();
    assert_canonical(&f);
    // m(x) = [x1] · 2^130 over the 130 kept variables: 2^129 assignments of
    // 2^260 each.
    assert_eq!(eng.marginal_product_count(&f, &f).unwrap(), BigUint::from(1u32) << 389usize);
    // The left half summed out too: one value, 2^129 · 2^130, squared.
    eng.marginalize_levels(&mut f, &[left]).unwrap();
    assert_eq!(eng.marginal_product_count(&f, &f).unwrap(), BigUint::from(1u32) << 518usize);
}

/// The refusals the contract names, and the false diagram.
#[test]
fn refusals_and_the_false_diagram() {
    let eng = Engine::new();
    let vtree = Arc::new(Vtree::balanced(4));
    let left = vtree.children(vtree.root()).0;
    let mut m = eng.clause(&vtree, [1, 3]).unwrap();
    eng.marginalize_levels(&mut m, &[left]).unwrap();
    assert_canonical(&m);
    let s = eng.clause(&vtree, [1, 2]).unwrap();
    for (f, g) in [(&m, &s), (&s, &m)] {
        assert_eq!(eng.marginal_product_count(f, g).unwrap_err(), OperationError::MarginalLevel(left));
    }
    // A structural operand free of the summed-out variables is read as its
    // constant.
    let free = eng.clause(&vtree, [3, 4]).unwrap();
    let want = eng.and_model_count(m.clone(), free.clone(), &[]).unwrap();
    assert_eq!(eng.marginal_product_count(&m, &free).unwrap(), want);
    let zero = crate::diagram::Tdd::zero(&vtree);
    assert_eq!(eng.marginal_product_count(&m, &zero).unwrap(), BigUint::ZERO);
    let other = Arc::new(Vtree::balanced(4));
    let elsewhere = eng.clause(&other, [1]).unwrap();
    assert_eq!(eng.marginal_product_count(&s, &elsewhere).unwrap_err(), OperationError::VtreeMismatch);
    let stopped = stopping_engine();
    assert_eq!(stopped.marginal_product_count(&m, &m).unwrap_err(), OperationError::Stopped);
    // Weights attached to either operand.
    use crate::diagram::{Arithmetic, LiteralWeights, RationalWeights, WeightStore};
    let half = num_rational::BigRational::new(1.into(), 2.into());
    let weights = RationalWeights::from_literals(&vec![LiteralWeights { negative: half.clone(), positive: half }; 4]);
    let mut weighted = s.clone();
    weighted.set_weights(WeightStore::new(weights, Arithmetic::ExactRational)).unwrap();
    for (f, g) in [(&weighted, &s), (&s, &weighted)] {
        assert_eq!(eng.marginal_product_count(f, g).unwrap_err(), OperationError::IncompatibleWeights);
    }
}

