//! Levels where the first operand passes through over the second's nodes:
//! random CNFs on overlapping projections of random homes, either operand
//! first, checked against the conjunction on the home, and by the work
//! meter against the route that emits every cell of such a level.

use std::sync::Arc;

use super::{pass_listed_g_census, pass_listed_off};
use crate::limits::{LimitConfig, SparseRoute};
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

/// No conjunction costs more work units than with every cell of its pass
/// levels emitted, and the conjunctions whose listed pass levels with
/// carrier `g` have at most a quarter of their cells live save, together,
/// at least one unit per dead cell. A small level can save less than that:
/// its buckets cost about what its few dead cells do.
#[test]
fn a_listed_pass_level_is_the_conjunction_on_the_home_and_costs_its_live_cells() {
    let eng = Engine::new();
    let n = 26u32;
    let (mut sparse, mut saved, mut dead) = (0, 0, 0);
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
                let at = format!("seed {seed} split {low}/{high}");
                let _scope = eng
                    .limits()
                    .scope(LimitConfig::none().with_sparse_route(SparseRoute::new(1, 0)));
                let census = pass_listed_g_census();
                let w = eng.limits().work_units();
                let h = eng.and_restoring(x, y).map_err(|r| r.error).unwrap();
                let listed = eng.limits().work_units() - w;
                let [levels, emitted, cells] =
                    std::array::from_fn(|k| pass_listed_g_census()[k] - census[k]);
                let w = eng.limits().work_units();
                pass_listed_off(|| eng.and_restoring(x, y).map_err(|r| r.error).unwrap());
                let every = eng.limits().work_units() - w;
                assert!(eng.equivalent(&on_home(&h), &expected).unwrap(), "{at}");
                assert!(
                    listed <= every,
                    "{at}: {listed} units listed, {every} emitting every cell"
                );
                if levels > 0 && 4 * emitted <= cells {
                    sparse += 1;
                    saved += every - listed;
                    dead += cells - emitted;
                }
            }
        }
    }
    assert!(
        sparse > 0,
        "no pass level with carrier g had a quarter of its cells live"
    );
    assert!(
        saved >= dead,
        "{sparse} conjunctions with sparse pass levels saved {saved} units over {dead} dead cells"
    );
}
