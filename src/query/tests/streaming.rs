//! The streaming fold against the materialized count.

use std::sync::Arc;

use num_bigint::BigUint;
use num_rational::BigRational;
use num_traits::Zero;

use crate::Engine;
use crate::apply::conjoin::tests::apply_and_fallible;
use crate::apply::conjoin::{VtreeMask, apply_and};
use crate::build::constant_one;
use crate::diagram::{Arithmetic, LiteralWeights, RationalWeights, Tdd, WeightStore};
use crate::test_helpers::{CnfShape, Lcg, clause_to_tdd, exact_weight, literals, rand_cnf, rat};
use crate::vtree::{Vtree, VtreeIdx};

fn interior_targets(vtree: &Vtree) -> Vec<bool> {
    (0..vtree.num_nodes())
        .map(|i| !vtree.node(VtreeIdx(i as u32)).is_leaf())
        .collect()
}

fn rand_unminimized_tdd(eng: &Engine, vtree: &Arc<Vtree>, rng: &mut Lcg, nvars: u32) -> Tdd {
    let shape = CnfShape {
        clauses: 4,
        width: nvars as usize,
    };
    let mut acc = constant_one(eng, vtree);
    for clause in rand_cnf(rng, nvars, shape) {
        acc = apply_and(acc, clause_to_tdd(vtree, &literals(&clause)));
    }
    acc
}

#[test]
fn streaming_fold_count_matches_materialized_randomized() {
    let eng = Engine::new();
    let mut rng = Lcg::new(0x0bad_c0de_1337_f00d);
    let (mut checked, mut nonzero) = (0u32, 0u32);
    for &nvars in &[2u32, 3, 4, 5, 6] {
        let vtree = Arc::new(Vtree::balanced(nvars));
        let targets = interior_targets(&vtree);
        for _ in 0..200 {
            let a = rand_unminimized_tdd(&eng, &vtree, &mut rng, nvars);
            let b = rand_unminimized_tdd(&eng, &vtree, &mut rng, nvars);
            let run = |mask: VtreeMask<'_>| {
                let (mut a_c, mut b_c) = (a.clone(), b.clone());
                apply_and_fallible(&eng, &mut a_c, &mut b_c, mask, VtreeMask::default())
                    .unwrap()
                    .model_count()
                    .unwrap()
            };
            let oracle = run(VtreeMask::default());
            let fold = run(VtreeMask::new(Some(&targets)));
            assert_eq!(
                fold, oracle,
                "nvars={nvars}: streaming fold != materialized"
            );
            checked += 1;
            nonzero += u32::from(oracle != BigUint::ZERO);
        }
    }
    assert!(checked > 0 && nonzero > 0);
}

#[test]
fn streaming_fold_weighted_matches_materialized_randomized() {
    let eng = Engine::new();
    let mut rng = Lcg::new(0x5eed_5eed_c0ff_ee11);
    let (mut checked, mut nonzero) = (0u32, 0u32);
    let rand_weight =
        |rng: &mut Lcg| -> BigRational { rat(1 + rng.below(9) as i64, 1 + rng.below(9) as i64) };
    for &nvars in &[2u32, 3, 4, 5, 6] {
        let vtree = Arc::new(Vtree::balanced(nvars));
        let targets = interior_targets(&vtree);
        for _ in 0..50 {
            let a = rand_unminimized_tdd(&eng, &vtree, &mut rng, nvars);
            let b = rand_unminimized_tdd(&eng, &vtree, &mut rng, nvars);
            let weights: Vec<LiteralWeights<BigRational>> = (0..nvars)
                .map(|_| LiteralWeights {
                    negative: rand_weight(&mut rng),
                    positive: rand_weight(&mut rng),
                })
                .collect();
            let run = |mask: VtreeMask<'_>| {
                let (mut a_c, mut b_c) = (a.clone(), b.clone());
                a_c.set_weights(WeightStore::new(
                    RationalWeights::from_literals(&weights),
                    Arithmetic::ExactRational,
                ))
                .unwrap();
                let res =
                    apply_and_fallible(&eng, &mut a_c, &mut b_c, mask, VtreeMask::default())
                        .unwrap();
                exact_weight(
                    &res.weighted_value()
                        .unwrap()
                        .expect("store follows the result"),
                )
            };
            let oracle = run(VtreeMask::default());
            let fold = run(VtreeMask::new(Some(&targets)));
            assert_eq!(
                fold, oracle,
                "nvars={nvars}: weighted streaming fold != materialized"
            );
            checked += 1;
            nonzero += u32::from(!oracle.is_zero());
        }
    }
    assert!(checked > 0 && nonzero > 0);
}
