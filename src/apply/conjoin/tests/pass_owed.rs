//! What the cells of a pass level leave the reductions after the
//! conjunction: random CNFs on projections of random homes, overlapping or
//! on disjoint variables, with the pass levels written cell by cell (no
//! description, no carried level), checked by the loose levels the result
//! lists, by the debug build's check of the levels it owes no contraction,
//! and by the work meter against the result that lists every pass level as
//! any built level.

use std::sync::Arc;

use super::{pass_composed_off, pass_levels_census, pass_levels_off, pass_numbered_off};
use crate::limits::{LimitConfig, SparseRoute};
use crate::vtree::rng::Lcg;
use crate::vtree::{Home, VarId, Vtree};
use crate::{Engine, Tdd};

fn cnf(eng: &Engine, home: &Home, vars: &[u32], count: usize, rng: &mut Lcg) -> Tdd {
    let vtree = Vtree::projection(home, vars.iter().map(|&v| VarId(v))).unwrap();
    let mut f = Tdd::one(&Home::new(vtree));
    for _ in 0..count {
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

/// `x ∧ y` with the pass levels written cell by cell, by the listed route
/// where `listed`, and with the pass levels recorded unless `off`.
fn conjoin(eng: &Engine, (x, y): (&Tdd, &Tdd), listed: bool, off: bool) -> Tdd {
    let config = match listed {
        true => LimitConfig::none().with_sparse_route(SparseRoute::new(1, 0)),
        false => LimitConfig::none(),
    };
    let _scope = eng.limits().scope(config);
    let and = || pass_numbered_off(|| pass_composed_off(|| eng.and_restoring(x, y).map_err(|r| r.error).unwrap()));
    match off {
        true => pass_levels_off(and),
        false => and(),
    }
}

fn minimized(eng: &Engine, mut h: Tdd) -> (Tdd, u64) {
    let w = eng.limits().work_units();
    eng.minimize(&mut h).unwrap();
    (h, eng.limits().work_units() - w)
}

fn loose(h: &Tdd) -> usize {
    h.dirty.loose().map_or(usize::MAX, |l| l.len())
}

/// A pass level's passed child is tight where both operands have it tight,
/// and gains no twins; its other child too where no carrier pair lost a
/// product: the result's loose levels hold (and, in the debug build, the
/// levels it owes no contraction have no twins), it lists no more loose
/// levels than with the pass levels listed as any built level, and its
/// minimization is the same diagram for fewer work units in all. One
/// minimization can cost a few units more: the prune settles the fewer
/// loose levels it is given where it would walk past every one listed.
#[test]
fn a_pass_level_owes_its_children_only_what_its_lost_products_can_leave() {
    let eng = Engine::new();
    let n = 24u32;
    let (mut fewer, mut saved) = (0, 0);
    let before = pass_levels_census();
    for seed in 0..30u64 {
        let mut rng = Lcg::new(seed ^ 0x2545_f491);
        let home = Home::new(Arc::new(Vtree::random(n, seed)));
        for (a, b) in [(8u32, 16u32), (12, 18), (4, 20)] {
            let fv: Vec<u32> = (1..=b).collect();
            let gv: Vec<u32> = (1..=a).chain(b + 1..=n).collect();
            let f = cnf(&eng, &home, &fv, fv.len() / 2, &mut rng);
            let g = cnf(&eng, &home, &gv, gv.len() / 2, &mut rng);
            for (x, y) in [(&f, &g), (&g, &f)] {
                for listed in [false, true] {
                    let at = format!("seed {seed} split {a}/{b} listed {listed}");
                    let h = conjoin(&eng, (x, y), listed, false);
                    let every = conjoin(&eng, (x, y), listed, true);
                    crate::test_helpers::check::check_loose_levels(&h).unwrap_or_else(|e| panic!("{at}: {e}"));
                    let (kept, all) = (loose(&h), loose(&every));
                    assert!(kept <= all, "{at}: {kept} loose levels, {all} with the pass levels listed");
                    let ((h, units), (every, every_units)) = (minimized(&eng, h), minimized(&eng, every));
                    crate::test_helpers::assert_same_shape(&h, &every, &at);
                    fewer += all - kept;
                    saved += every_units as i64 - units as i64;
                }
            }
        }
    }
    let [levels, lost] = std::array::from_fn(|k| pass_levels_census()[k] - before[k]);
    assert!(levels > 100 && lost > 20 && lost < levels, "{levels} pass levels recorded, {lost} lost a product");
    assert!(fewer > 50, "{fewer} loose levels left out");
    assert!(saved > 0, "the minimizations saved no work");
}

/// On disjoint variables no product is lost, so the conjunction of two
/// minimized diagrams owes nothing at its pass levels: the minimization
/// after it leaves it unchanged, for no more work units than with the pass
/// levels listed as any built level, and fewer in all. A level the
/// conjunction built otherwise can still be owed a walk.
#[test]
fn the_minimization_after_a_conjunction_on_disjoint_variables_changes_nothing() {
    let eng = Engine::new();
    let n = 24u32;
    let (mut fired, mut saved) = (0, 0);
    for seed in 0..30u64 {
        let mut rng = Lcg::new(seed ^ 0x6a09_e667);
        let home = Home::new(Arc::new(Vtree::random(n, seed)));
        for a in [8u32, 12, 16] {
            let fv: Vec<u32> = (1..=a).collect();
            let gv: Vec<u32> = (a + 1..=n).collect();
            let f = cnf(&eng, &home, &fv, fv.len() / 2, &mut rng);
            let g = cnf(&eng, &home, &gv, gv.len() / 2, &mut rng);
            for (x, y) in [(&f, &g), (&g, &f)] {
                for listed in [false, true] {
                    let at = format!("seed {seed} split {a} listed {listed}");
                    let census = pass_levels_census();
                    let h = conjoin(&eng, (x, y), listed, false);
                    let [levels, lost] = std::array::from_fn(|k| pass_levels_census()[k] - census[k]);
                    assert_eq!(lost, 0, "{at}: a product lost on disjoint variables");
                    let every = conjoin(&eng, (x, y), listed, true);
                    let (reduced, units) = minimized(&eng, h.clone());
                    let (_, every_units) = minimized(&eng, every);
                    crate::test_helpers::assert_same_shape(&reduced, &h, &at);
                    assert!(units <= every_units, "{at}: minimize {units} units, {every_units} with the pass levels listed");
                    fired += (levels > 0) as usize;
                    saved += every_units - units;
                }
            }
        }
    }
    assert!(fired > 50, "{fired} conjunctions with a pass level");
    assert!(saved > 0, "the minimizations saved no work");
}

/// A chain of conjunctions whose result is only contracted between them,
/// so that its passed children hold nodes no pair names, with the pass
/// levels written cell by cell: the minimized result is canonical and the
/// diagram of the conjunction that lists every pass level as any built
/// level. (The contraction can leave two nodes no pair names under a pass
/// level, which a whole search of it drops: they share the empty context,
/// and the prune drops them.)
#[test]
fn a_chain_of_contracted_conjunctions_through_the_pass_cells_stays_canonical() {
    use crate::reduce::ReductionPlan;
    use crate::test_helpers::{assert_canonical, assert_same_shape};
    let eng = Engine::new();
    let n = 16u32;
    let before = pass_levels_census();
    let subset = |rng: &mut Lcg| -> Vec<u32> {
        let vars: Vec<u32> = (1..=n).filter(|_| rng.below(3) > 0).collect();
        if vars.len() < 2 { vec![1, n] } else { vars }
    };
    for seed in 0..60u64 {
        let mut rng = Lcg::new(seed ^ 0x3c6e_f372);
        let home = Home::new(Arc::new(Vtree::random(n, seed)));
        let vars = subset(&mut rng);
        let mut acc = cnf(&eng, &home, &vars, vars.len() / 2, &mut rng);
        for step in 0..6 {
            let vars = subset(&mut rng);
            let g = cnf(&eng, &home, &vars, vars.len() / 3, &mut rng);
            let at = format!("seed {seed} step {step}");
            let listed = step % 2 == 1;
            let mut h = conjoin(&eng, (&acc, &g), listed, false);
            let mut every = conjoin(&eng, (&acc, &g), listed, true);
            eng.reduce(&mut h, ReductionPlan::Contract).unwrap();
            eng.reduce(&mut every, ReductionPlan::Contract).unwrap();
            let ((m, _), (every, _)) = (minimized(&eng, h.clone()), minimized(&eng, every));
            assert_canonical(&m);
            assert_same_shape(&m, &every, &at);
            acc = h;
        }
    }
    let [levels, lost] = std::array::from_fn(|k| pass_levels_census()[k] - before[k]);
    assert!(levels > 100 && lost > 20, "{levels} pass levels recorded, {lost} lost a product");
}
