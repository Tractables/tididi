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
