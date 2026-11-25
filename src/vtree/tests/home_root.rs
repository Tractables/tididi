//! A home's root is the projection of its home on every leaf: each entry
//! point that asks whether a diagram is on a home answers as on any other
//! projection.

use std::sync::Arc;

use num_rational::BigRational;

use crate::diagram::{Arithmetic, LiteralWeights, RationalWeights, Weights};
use crate::vtree::rng::Lcg;
use crate::vtree::{Home, VarId, Vtree};
use crate::{Engine, OperationError, Tdd};

/// Home (x1, x2) and x1 ∧ x2 on its root.
fn x_and_y() -> (Home, Tdd) {
    let home = Home::new(Vtree::balanced(2));
    let f = Tdd::cube(&home, [1, 2]).unwrap();
    (home, f)
}

// Padded form: a bare root, placed by tree matching.
/// Random clauses over `vars`, two or three literals each, and their
/// conjunction on `vtree`.
fn cnf(rng: &mut Lcg, vtree: &Arc<Vtree>, vars: &[u32], clauses: usize) -> (Vec<Vec<i32>>, Tdd) {
    let mut cnf = Vec::new();
    let on = Home::unchecked(vtree);
    let mut f = Tdd::cube(&on, std::iter::empty::<i32>()).unwrap();
    for _ in 0..clauses {
        let mut clause: Vec<i32> = Vec::new();
        while clause.len() < 2 + rng.below(2) as usize {
            let var = vars[rng.below(vars.len() as u64) as usize] as i32;
            if !clause.iter().any(|l| l.abs() == var) {
                clause.push(if rng.coin() { var } else { -var });
            }
        }
        f = crate::and(f, Tdd::clause(&on, clause.iter().copied()).unwrap()).unwrap();
        cnf.push(clause);
    }
    (cnf, f)
}

/// With weights attached, `and_model_count` with targets gives the integer
/// count of the conjunction, literal weights ignored: on a weighted root
/// vtree, on a weighted home's root and on projections of that home.
#[test]
fn weighted_and_model_count_with_targets_counts_models() {
    const N: u32 = 6;
    let eng = Engine::new();
    let weights = || {
        let w = |i: u32| LiteralWeights { negative: BigRational::new((i + 1).into(), 3.into()), positive: BigRational::new(2.into(), 7.into()) };
        Weights::new(RationalWeights::from_literals(&(0..N).map(w).collect::<Vec<_>>()), Arithmetic::ExactRational)
    };
    let home = Home::weighted(Vtree::random(N, 3), weights()).unwrap();
    let plain = Arc::new(Vtree::random(N, 3).with_weights(weights()).unwrap());
    let all: Vec<u32> = (1..=N).collect();
    let mut rng = Lcg::new(11);
    let mut summed = 0;
    for round in 0..240 {
        let (fv, gv): (Vec<u32>, Vec<u32>) = match round % 3 {
            0 | 1 => (all.clone(), all.clone()),
            _ => (all.iter().copied().filter(|_| rng.coin()).collect(), all.iter().copied().filter(|_| rng.coin()).collect()),
        };
        if fv.len() < 3 || gv.len() < 3 {
            continue;
        }
        let on = |vars: &[u32]| match round % 3 {
            0 => Arc::clone(&plain),
            1 => Arc::clone(&home),
            _ => Vtree::projection(&home, vars.iter().map(|&v| VarId(v))).unwrap(),
        };
        let (nf, ng) = (1 + rng.below(3) as usize, 1 + rng.below(3) as usize);
        let (fc, f) = cnf(&mut rng, &on(&fv), &fv, nf);
        let (gc, g) = cnf(&mut rng, &on(&gv), &gv, ng);
        let v = Arc::clone(f.vtree());
        let targets: Vec<_> = v.bottomup().filter(|&t| !v.node(t).is_leaf() && rng.coin()).collect();
        summed += usize::from(!targets.is_empty());
        let expected = crate::test_helpers::brute_force_count(N, &[fc, gc].concat());
        let count = crate::test_helpers::and_model_count_under(&eng, f, g, &targets).unwrap_or_else(|e| panic!("round {round}: {e}"));
        assert_eq!(count, expected.into(), "round {round}");
    }
    assert!(summed > 100, "{summed} rounds with targets");
}

/// `marginalize` refuses an internal summed-out level that is not a region
/// on a home's root, as on a projection (the smooth API no longer makes
/// one; the internal sum makes it here).
#[test]
fn marginalize_refuses_an_internal_summed_level_on_a_home_root() {
    let (home, mut f) = x_and_y();
    let root = home.root();
    crate::marginal::tests::marginalize_levels(&Engine::new(), &mut f, &[root]).unwrap();
    assert_eq!(f.marginalize(|_| true), Err(OperationError::MarginalLevel(root)));
}

/// A summed variable leaf on a home's root is placed as on a projection
/// (the smooth API no longer makes one; the internal sum makes it here).
#[test]
fn embed_places_a_summed_leaf_of_a_home_root() {
    let (home, mut f) = x_and_y();
    let eng = Engine::new();
    let x = home.leaf_of(VarId(1)).unwrap();
    crate::marginal::tests::marginalize_levels(&eng, &mut f, &[x]).unwrap();
    let count = f.model_count().unwrap();
    let into = Arc::new(Vtree::balanced(2));
    let (placed, _) = crate::test_helpers::moved(&eng, f, &into, |v| v).unwrap();
    assert_eq!(placed.model_count().unwrap(), count);
}

/// `tag_level` into a home's root reads the root as a projection: the
/// code's side follows the leaf order, here on the left.
#[test]
fn tag_level_into_a_home_root_with_the_code_on_the_left() {
    let eng = Engine::new();
    let pair = Home::new(Vtree::balanced(2));
    let f = Tdd::cube(&pair, [1, 2]).unwrap();
    let code = Vtree::balanced_over(&[VarId(3), VarId(4)]).unwrap();
    let home = Home::new(Vtree::join(&code, &Vtree::balanced(2)).unwrap());
    let g = eng.tag_level(&f, pair.root(), &home, |v| v, &[Some(0)], &[VarId(3), VarId(4)], &[0b10]).unwrap();
    let expected = eng.and(Tdd::cube(&home, [1, 2]).unwrap(), Tdd::cube(&home, [-3, 4]).unwrap()).unwrap();
    assert!(eng.equivalent(&g, &expected).unwrap());
}
