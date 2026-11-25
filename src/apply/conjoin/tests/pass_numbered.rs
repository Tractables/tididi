//! Levels where one operand passes through and the passed child's products
//! are the carrier's nodes in order: random CNFs on overlapping projections
//! of random homes that share some clauses, either operand first, checked
//! against the conjunction on the home and, by the work meter, against the
//! route that lists the level's cells.

use std::sync::Arc;

use super::{pass_numbered_census, pass_numbered_off};
use crate::limits::{LimitConfig, SparseRoute};
use crate::vtree::rng::Lcg;
use crate::vtree::{Home, VarId, Vtree};
use crate::{Engine, Tdd};

fn clauses(vars: &[u32], count: usize, rng: &mut Lcg) -> Vec<Vec<i32>> {
    (0..count)
        .map(|_| {
            (0..2 + rng.below(2))
                .map(|_| {
                    let v = vars[rng.below(vars.len() as u64) as usize] as i32;
                    if rng.coin() { v } else { -v }
                })
                .collect()
        })
        .collect()
}

fn cnf(eng: &Engine, vtree: &Arc<Vtree>, clauses: &[Vec<i32>]) -> Tdd {
    let mut f = Tdd::one(vtree);
    for clause in clauses {
        f = eng.and_clause(f, clause).unwrap();
    }
    eng.minimize(&mut f).unwrap();
    f
}

/// A pass level whose passed child's products are the carrier's nodes in
/// order, each with one passing node, is the carrier's level: the
/// conjunction is the one that lists the level's cells, of the same shape.
/// It costs no more work units unless a level's carrier pairs were read in
/// vain, a carrier node meeting two passing nodes, and fewer net of those.
#[test]
fn a_pass_level_that_keeps_the_carrier_numbering_is_the_carrier_level() {
    let eng = Engine::new();
    let n = 24u32;
    let (mut fired, mut vain, mut net) = (0, 0, 0);
    for seed in 0..40u64 {
        let mut rng = Lcg::new(seed ^ 0x51ed_270b);
        let home = Home::new(Arc::new(Vtree::random(n, seed)));
        let on_home = |t: &Tdd| crate::test_helpers::embedded(t, home.vtree(), |v| v).unwrap().0;
        for (a, b) in [(8u32, 16u32), (12, 18), (6, 12)] {
            // f reads 1..=b and g reads 1..=a with b+1..=n: both hold the
            // clauses `base` over the shared 1..=a.
            let shared: Vec<u32> = (1..=a).collect();
            let fv: Vec<u32> = (1..=b).collect();
            let own: Vec<u32> = (b + 1..=n).collect();
            let gv: Vec<u32> = shared.iter().chain(&own).copied().collect();
            let base = clauses(&shared, shared.len() * 2 / 3, &mut rng);
            let fc: Vec<_> = base.iter().cloned().chain(clauses(&fv, fv.len() / 2, &mut rng)).collect();
            let gc: Vec<_> = base.iter().cloned().chain(clauses(&own, own.len() / 2, &mut rng)).collect();
            let fp = Vtree::projection(&home, fv.iter().map(|&v| VarId(v))).unwrap();
            let gp = Vtree::projection(&home, gv.iter().map(|&v| VarId(v))).unwrap();
            let f = cnf(&eng, &fp, &fc);
            let g = cnf(&eng, &gp, &gc);
            let expected = eng.and(on_home(&f), on_home(&g)).unwrap();
            for (x, y) in [(&f, &g), (&g, &f)] {
                let at = format!("seed {seed} split {a}/{b}");
                let _scope = eng.limits().scope(LimitConfig::none().with_sparse_route(SparseRoute::new(1, 0)));
                let census = pass_numbered_census();
                let w = eng.limits().work_units();
                let h = eng.and_restoring(x, y).map_err(|r| r.error).unwrap();
                let numbered = eng.limits().work_units() - w;
                let [levels, failed] = std::array::from_fn(|k| pass_numbered_census()[k] - census[k]);
                let w = eng.limits().work_units();
                let listed = pass_numbered_off(|| eng.and_restoring(x, y).map_err(|r| r.error).unwrap());
                let listed_units = eng.limits().work_units() - w;
                assert!(eng.equivalent(&on_home(&h), &expected).unwrap(), "{at}");
                crate::test_helpers::assert_same_shape(&h, &listed, &at);
                assert!(failed > 0 || numbered <= listed_units, "{at}: {numbered} units carried, {listed_units} listed");
                fired += levels;
                vain += failed;
                net += listed_units as i64 - numbered as i64;
            }
        }
    }
    assert!(fired > 50, "{fired} pass levels carried whole");
    assert!(net > 0, "{net} units saved net of {vain} levels read in vain");
}
