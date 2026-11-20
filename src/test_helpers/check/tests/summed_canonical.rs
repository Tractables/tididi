//! On random weighted and summed diagrams, with regions: `minimize` reaches
//! the form `check_canonicity` accepts, and `check_all_fast` agrees with it
//! on a certified result.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;

use crate::diagram::{Arithmetic, LiteralWeights, RationalWeights, Weights};
use crate::test_helpers::{assert_canonical, canonical_engine, compile_clauses_on, rand_cnf, rat, vtree_shapes, CnfShape, Lcg};
use crate::vtree::{Home, VarId, Vtree, VtreeIdx};
use crate::{Engine, Tdd};
use super::*;

fn weights(rng: &mut Lcg, n: u32, arithmetic: Arithmetic) -> Weights {
    let literals: Vec<_> = (0..n)
        .map(|_| LiteralWeights { negative: rat(1 + rng.below(4) as i64, 1 + rng.below(3) as i64), positive: rat(1 + rng.below(4) as i64, 1 + rng.below(3) as i64) })
        .collect();
    Weights::new(RationalWeights::from_literals(&literals), arithmetic)
}

/// One summed diagram: on a root (integer or weighted) by level targets, or
/// on a home (integer or weighted) by variables, which leaves regions.
fn summed(rng: &mut Lcg, eng: &Engine) -> Option<(Tdd, String)> {
    let n = 3 + rng.below(5) as u32;
    let shapes = vtree_shapes(n);
    let (shape, base) = &shapes[rng.below(shapes.len() as u64) as usize];
    let arithmetic = [None, Some(Arithmetic::ExactRational), Some(Arithmetic::ExactRational), Some(Arithmetic::SignedLog)][rng.below(4) as usize];
    let table = arithmetic.map(|a| weights(rng, n, a));
    let cnf = CnfShape { clauses: n as usize, width: 3 };
    let (f_cnf, g_cnf) = (rand_cnf(rng, n, cnf), rand_cnf(rng, n, cnf));
    let on_home = rng.below(2) == 0;
    let label = format!("n {n} {shape} {arithmetic:?} home {on_home}");
    if on_home {
        let home = match table {
            Some(w) => Home::weighted(Vtree::clone(base), w).ok()?,
            None => Home::new(Vtree::clone(base)),
        };
        let vtree = home.vtree();
        let f = compile_clauses_on(eng, vtree, &f_cnf);
        let g = compile_clauses_on(eng, vtree, &g_cnf);
        let mask: Vec<bool> = (0..=n).map(|_| rng.below(2) == 0).collect();
        let r = if rng.below(2) == 0 {
            eng.and_marginalizing_where(&f, &g, |v: VarId| mask[v.idx()]).ok()?
        } else {
            let mut h = eng.and(&f, &g).ok()?;
            eng.marginalize(&mut h, |v: VarId| mask[v.idx()]).ok()?;
            h
        };
        return Some((r, label));
    }
    let vtree = Arc::new(match table {
        Some(w) => Vtree::clone(base).with_weights(w).ok()?,
        None => Vtree::clone(base),
    });
    let f = compile_clauses_on(eng, &vtree, &f_cnf);
    let g = compile_clauses_on(eng, &vtree, &g_cnf);
    let below_root: Vec<VtreeIdx> = (0..vtree.num_nodes() as u32).map(VtreeIdx).filter(|&t| t != vtree.root()).collect();
    let targets: Vec<VtreeIdx> = below_root.iter().copied().filter(|_| rng.below(4) == 0).collect();
    let r = if rng.below(2) == 0 {
        eng.and_marginalizing(&f, &g, &targets).ok()?
    } else {
        let mut h = eng.and(&f, &g).ok()?;
        eng.marginalize_levels(&mut h, &targets).ok()?;
        h
    };
    Some((r, label))
}

fn fails(check: impl FnOnce()) -> Option<String> {
    catch_unwind(AssertUnwindSafe(check)).err().map(|e| {
        e.downcast_ref::<String>().cloned().or_else(|| e.downcast_ref::<&str>().map(|s| (*s).to_string())).unwrap_or_default()
    })
}

/// Every result passes `assert_canonical`, minimized or not; minimized, it
/// passes `check_canonicity`; and on a certified result `check_all_fast`
/// decides what `check_canonicity` decides.
#[test]
fn minimize_reaches_canonical_form_on_summed_diagrams() {
    let mut rng = Lcg::new(0x5eed_ca50);
    let (mut cases, mut summed_cases, mut regions, mut certified) = (0, 0, 0, 0);
    while cases < 1200 {
        let eng = if rng.below(2) == 0 { Engine::new() } else { canonical_engine() };
        let Some((r, label)) = summed(&mut rng, &eng) else { continue };
        cases += 1;
        summed_cases += usize::from(r.has_marginal_level());
        regions += usize::from(r.vtree().bottomup().any(|t| r.vtree().node(t).is_region()));
        if let Some(e) = fails(|| assert_canonical(&r)) {
            panic!("case {cases} ({label}): {e}");
        }
        let mut m = r.clone();
        eng.minimize(&mut m).unwrap();
        check_canonicity(&m, 3).unwrap_or_else(|e| panic!("case {cases} ({label}) minimized: {e}"));
        if r.is_canonical() {
            certified += 1;
            let fast = fails(|| check_all_fast(&r, "certified"));
            assert_eq!(fast.is_none(), check_canonicity(&r, 3).is_ok(), "case {cases} ({label}): {fast:?}");
        }
    }
    assert!(summed_cases >= 600 && regions >= 200 && certified >= 100, "{summed_cases} summed, {regions} with regions, {certified} certified");
}
