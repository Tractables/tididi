//! Levels where one operand passes through, its passed child is complete
//! and the carrier's level is affine: conjunctions of operands on
//! overlapping projections of random homes, each a CNF of equalities that
//! cross the vtree's cuts so that their levels are complete products,
//! checked against the conjunction on the home and, by shape,
//! against the route that emits the level's cells.

use std::sync::Arc;

use super::{pass_composed_census, pass_composed_off};
use crate::vtree::rng::Lcg;
use crate::vtree::{Home, VarId, Vtree};
use crate::{Engine, Tdd};

/// A CNF over `vars`, in the home's leaf order, that makes the outermost
/// remaining two equal, or unequal, `pairs` times inward: each pair crosses
/// the cuts between them, so the levels there count the settings of the
/// pairs, and are complete products.
fn crossing(vars: &[u32], pairs: usize, rng: &mut Lcg) -> Vec<Vec<i32>> {
    let mut out = Vec::new();
    for k in 0..pairs.min(vars.len() / 2) {
        let (x, y) = (vars[k] as i32, vars[vars.len() - 1 - k] as i32);
        let y = if rng.coin() { y } else { -y };
        out.push(vec![x, -y]);
        out.push(vec![-x, y]);
    }
    out
}

fn cnf(eng: &Engine, vtree: &Arc<Vtree>, clauses: &[Vec<i32>]) -> Tdd {
    let mut f = Tdd::one(&Home::new(Arc::clone(vtree)));
    for clause in clauses {
        f = eng.and_clause(f, clause).unwrap();
    }
    eng.minimize(&mut f).unwrap();
    f
}

/// A pass level over a complete passed child whose carrier's level is
/// affine is written as its description: the conjunction equals the one
/// on the home, has the shape of the one that emits the level's cells and
/// costs the same work units.
#[test]
fn a_pass_level_over_a_complete_child_is_written_as_its_description() {
    let eng = Engine::new();
    let n = 28u32;
    let mut fired = 0;
    for seed in 0..40u64 {
        let mut rng = Lcg::new(seed ^ 0x7a55_c0de);
        let home = Home::new(Arc::new(Vtree::random(n, seed)));
        let mut order: Vec<u32> = (1..=n).collect();
        order.sort_by_key(|&v| home.leaf_of(VarId(v)).unwrap().idx());
        let on_home = |t: &Tdd| crate::test_helpers::embedded(t, &home, |v| v).unwrap().0;
        for (a, b, pairs) in [(14usize, 20usize, 7usize), (8, 18, 6), (10, 24, 8)] {
            let fv = &order[..b];
            let gv: Vec<u32> = order[..a].iter().chain(&order[b..]).copied().collect();
            let fp = Vtree::projection(&home, fv.iter().map(|&v| VarId(v))).unwrap();
            let gp = Vtree::projection(&home, gv.iter().map(|&v| VarId(v))).unwrap();
            let f = cnf(&eng, &fp, &crossing(fv, pairs, &mut rng));
            let g = cnf(&eng, &gp, &crossing(&gv, pairs, &mut rng));
            let expected = eng.and(on_home(&f), on_home(&g)).unwrap();
            for (x, y) in [(&f, &g), (&g, &f)] {
                let at = format!("seed {seed} split {a}/{b} pairs {pairs}");
                let census = pass_composed_census();
                let w = eng.limits().work_units();
                let h = eng.and_restoring(x, y).map_err(|r| r.error).unwrap();
                let described = eng.limits().work_units() - w;
                fired += pass_composed_census() - census;
                let w = eng.limits().work_units();
                let emitted = pass_composed_off(|| eng.and_restoring(x, y).map_err(|r| r.error).unwrap());
                assert_eq!(described, eng.limits().work_units() - w, "{at}: work units");
                assert!(eng.equivalent(&on_home(&h), &expected).unwrap(), "{at}");
                crate::test_helpers::assert_same_shape(&h, &emitted, &at);
            }
        }
    }
    assert!(fired > 20, "{fired} pass levels written as their description");
}
