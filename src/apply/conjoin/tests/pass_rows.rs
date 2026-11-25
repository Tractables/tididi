//! Levels where the first operand passes through over the second's nodes,
//! listed by the first operand's rows: random CNFs on overlapping
//! projections of random homes, either operand first, against the
//! conjunction on the home.

use std::sync::Arc;

use crate::vtree::rng::Lcg;
use crate::vtree::{Home, VarId, Vtree};
use crate::{Engine, Tdd};

fn cnf(eng: &Engine, vtree: &Arc<Vtree>, vars: &[u32], clauses: usize, rng: &mut Lcg) -> Tdd {
    let mut f = Tdd::one(vtree);
    for _ in 0..clauses {
        let clause: Vec<i32> = (0..2 + rng.below(2))
            .map(|_| {
                let v = vars[rng.below(vars.len() as u64) as usize] as i32;
                if rng.coin() { v } else { -v }
            })
            .collect();
        f = eng.and_clause(f, &clause).unwrap();
    }
    eng.minimize(&mut f).unwrap();
    f
}

#[test]
fn a_conjunction_on_overlapping_projections_is_the_one_on_the_home() {
    let eng = Engine::new();
    let n = 26u32;
    for seed in 0..40u64 {
        let mut rng = Lcg::new(seed ^ 0x9e37_79b9);
        let home = Home::new(Arc::new(Vtree::random(n, seed)));
        let on_home = |t: &Tdd| {
            crate::test_helpers::embedded(t, home.vtree(), |v| v)
                .unwrap()
                .0
        };
        for (low, high) in [(16, 9), (20, 5), (13, 13)] {
            let fv: Vec<u32> = (1..=low).collect();
            let gv: Vec<u32> = (high..=n).collect();
            let fp = Vtree::projection(&home, fv.iter().map(|&v| VarId(v))).unwrap();
            let gp = Vtree::projection(&home, gv.iter().map(|&v| VarId(v))).unwrap();
            let f = cnf(&eng, &fp, &fv, fv.len() * 2 / 3, &mut rng);
            let g = cnf(&eng, &gp, &gv, gv.len() * 2 / 3, &mut rng);
            let expected = eng.and(on_home(&f), on_home(&g)).unwrap();
            for (x, y) in [(&f, &g), (&g, &f)] {
                let h = eng.and_restoring(x, y).map_err(|r| r.error).unwrap();
                assert!(
                    eng.equivalent(&on_home(&h), &expected).unwrap(),
                    "seed {seed} split {low}/{high}"
                );
            }
        }
    }
}
