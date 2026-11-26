//! Levels where one operand passes through, the cells of its passed child
//! that the carrier reads are affine and the carrier's level is too: conjunctions of operands on
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
    let mut f = Tdd::one(vtree);
    for clause in clauses {
        f = eng.and_clause(f, clause).unwrap();
    }
    eng.minimize(&mut f).unwrap();
    f
}

/// Conjoin `x` and `y`, both ways round, as the pass route writes their
/// levels and as it emits them, and check both against `expected` on the
/// home: equal, of one shape, at equal work units. Returns the pass levels
/// written as their description, over a complete child and over a slab.
fn check(eng: &Engine, home: &Home, (f, g): (&Tdd, &Tdd), at: &str) -> [u64; 2] {
    let on_home = |t: &Tdd| crate::test_helpers::embedded(t, home.vtree(), |v| v).unwrap().0;
    let expected = eng.and(on_home(f), on_home(g)).unwrap();
    let mut fired = [0; 2];
    for (x, y) in [(f, g), (g, f)] {
        let census = pass_composed_census();
        let w = eng.limits().work_units();
        let h = eng.and_restoring(x, y).map_err(|r| r.error).unwrap();
        let described = eng.limits().work_units() - w;
        let now = pass_composed_census();
        fired = [fired[0] + now[0] - census[0], fired[1] + now[1] - census[1]];
        let w = eng.limits().work_units();
        let emitted = pass_composed_off(|| eng.and_restoring(x, y).map_err(|r| r.error).unwrap());
        assert_eq!(described, eng.limits().work_units() - w, "{at}: work units");
        assert!(eng.equivalent(&on_home(&h), &expected).unwrap(), "{at}");
        crate::test_helpers::assert_same_shape(&h, &emitted, at);
    }
    fired
}

/// Random homes of `n` variables, each with its variables in leaf order.
fn homes(n: u32, seeds: u64) -> impl Iterator<Item = (u64, Home, Vec<u32>)> {
    (0..seeds).map(move |seed| {
        let home = Home::new(Arc::new(Vtree::random(n, seed)));
        let mut order: Vec<u32> = (1..=n).collect();
        order.sort_by_key(|&v| home.vtree().leaf_of(VarId(v)).unwrap().idx());
        (seed, home, order)
    })
}

/// A pass level over a complete passed child whose carrier's level is
/// affine is written as its description: the conjunction equals the one
/// on the home, has the shape of the one that emits the level's cells and
/// costs the same work units.
#[test]
fn a_pass_level_over_a_complete_child_is_written_as_its_description() {
    let eng = Engine::new();
    let mut fired = 0;
    for (seed, home, order) in homes(28, 40) {
        let mut rng = Lcg::new(seed ^ 0x7a55_c0de);
        for (a, b, pairs) in [(14usize, 20usize, 7usize), (8, 18, 6), (10, 24, 8)] {
            let fv = &order[..b];
            let gv: Vec<u32> = order[..a].iter().chain(&order[b..]).copied().collect();
            let fp = Vtree::projection(&home, fv.iter().map(|&v| VarId(v))).unwrap();
            let gp = Vtree::projection(&home, gv.iter().map(|&v| VarId(v))).unwrap();
            let f = cnf(&eng, &fp, &crossing(fv, pairs, &mut rng));
            let g = cnf(&eng, &gp, &crossing(&gv, pairs, &mut rng));
            fired += check(&eng, &home, (&f, &g), &format!("seed {seed} split {a}/{b} pairs {pairs}"))[0];
        }
    }
    assert!(fired > 20, "{fired} pass levels written as their description");
}

/// A pass level whose passed child has dead cells, a passing node dead
/// with every slot the carrier names and the live cells affine, is written
/// as its description: the operands' crossing equalities, with clauses on
/// the variables they share that leave some settings there false.
#[test]
fn a_pass_level_over_a_slab_child_is_written_as_its_description() {
    let eng = Engine::new();
    let mut fired = [0; 2];
    for (seed, home, order) in homes(28, 240) {
        let mut rng = Lcg::new(seed ^ 0x51ab_c0de);
        for (a, b, pairs) in [(14usize, 20usize, 7usize), (8, 18, 6), (10, 24, 8)] {
            let fv = &order[..b];
            let gv: Vec<u32> = order[..a].iter().chain(&order[b..]).copied().collect();
            let fp = Vtree::projection(&home, fv.iter().map(|&v| VarId(v))).unwrap();
            let gp = Vtree::projection(&home, gv.iter().map(|&v| VarId(v))).unwrap();
            let mut fc = crossing(fv, pairs, &mut rng);
            let mut gc = crossing(&gv, pairs, &mut rng);
            let shared = |rng: &mut Lcg| {
                let v = order[rng.below(a as u64) as usize] as i32;
                if rng.coin() { v } else { -v }
            };
            fc.push(vec![shared(&mut rng), shared(&mut rng)]);
            gc.push(vec![shared(&mut rng), shared(&mut rng)]);
            let at = format!("seed {seed} split {a}/{b} pairs {pairs}");
            let n = crate::test_helpers::with_floor(4, || {
                let (f, g) = (cnf(&eng, &fp, &fc), cnf(&eng, &gp, &gc));
                check(&eng, &home, (&f, &g), &at)
            });
            fired = [fired[0] + n[0], fired[1] + n[1]];
        }
    }
    assert!(fired[1] > 20, "{fired:?} pass levels written as their description");
}
