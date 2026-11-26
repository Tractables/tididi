//! A conjunction meters every pair its sweep puts in the output, built,
//! copied or carried, one-pair nodes included: under a pair floor at the
//! pairs of its result it stops, its operands on one vtree or read through
//! views on projections of one home.

use std::sync::Arc;

use crate::limits::{LimitConfig, OperationError, StopAt, StopRules};
use crate::vtree::rng::Lcg;
use crate::vtree::{Home, VarId, Vtree};
use crate::{Engine, Tdd};

/// A random CNF of `clauses` clauses of two or three literals over `vars`,
/// built on `vtree`.
fn cnf(eng: &Engine, vtree: &Arc<Vtree>, vars: &[u32], clauses: usize, rng: &mut Lcg) -> Tdd {
    let mut f = Tdd::one(&Home::new(Arc::clone(vtree)));
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

/// A conjunction run on the engine it is given.
type Run<'a> = &'a dyn Fn(&Engine) -> Result<Tdd, OperationError>;

/// `run` on a fresh engine under a pair floor of `floor`, past which the
/// stop falls at once.
fn under_floor(floor: u64, run: Run<'_>) -> Result<Tdd, OperationError> {
    let eng = Engine::new();
    let rules = StopRules::default().with_after_pairs(Some((floor, StopAt::WorkUnits(0))));
    let _scope = eng.limits().scope(LimitConfig::none().with_stop_rules(rules));
    run(&eng)
}

/// The routes of `f ∧ g` that fail to stop under a floor at the pairs of
/// their result, by name.
fn unstopped(f: &Tdd, g: &Tdd) -> Vec<String> {
    let out = Engine::new().and(f, g).unwrap();
    let floor = out.pair_count() as u64;
    let runs: [(&str, Run<'_>); 4] = [
        ("and borrowed", &|eng| eng.and(f, g)),
        ("and owned", &|eng| eng.and(f.clone(), g.clone())),
        ("and_restoring owned", &|eng| eng.and_restoring(f.clone(), g.clone()).map_err(|r| r.error)),
        ("and_restoring borrowed", &|eng| eng.and_restoring(f, g).map_err(|r| r.error)),
    ];
    let mut failed = Vec::new();
    if out.is_zero() || floor == 0 {
        return failed;
    }
    for (name, run) in runs {
        if let Ok(t) = under_floor(floor, run) {
            failed.push(format!("{name}: {} pairs under a floor of {floor}", t.pair_count()));
        }
    }
    failed
}

#[test]
fn a_floor_at_the_result_stops_the_conjunction() {
    let eng = Engine::new();
    let n = 26u32;
    let mut failed = Vec::new();
    for seed in 0..40u64 {
        let mut rng = Lcg::new(seed ^ 0x2545_f491);
        let home = Home::new(Arc::new(Vtree::random(n, seed)));
        let all: Vec<u32> = (1..=n).collect();
        let some = |rng: &mut Lcg| -> Vec<u32> { all.iter().copied().filter(|_| rng.below(4) != 0).collect() };
        let low: Vec<u32> = (1..=16).collect();
        let high: Vec<u32> = (9..=n).collect();
        let (a, b) = (some(&mut rng), some(&mut rng));
        let layouts = [("one vtree", all.clone(), all.clone()), ("nested", all.clone(), a.clone()), ("overlapping", low, high), ("random", a, b)];
        for (name, fv, gv) in layouts {
            let fp = Vtree::projection(&home, fv.iter().map(|&v| VarId(v))).unwrap();
            let gp = match name {
                "one vtree" => Arc::clone(&fp),
                _ => Vtree::projection(&home, gv.iter().map(|&v| VarId(v))).unwrap(),
            };
            let f = cnf(&eng, &fp, &fv, fv.len() * 2 / 3, &mut rng);
            let g = cnf(&eng, &gp, &gv, gv.len() * 2 / 3, &mut rng);
            for (x, y) in [(&f, &g), (&g, &f)] {
                failed.extend(unstopped(x, y).into_iter().map(|e| format!("seed {seed} {name}: {e}")));
            }
        }
    }
    assert!(failed.is_empty(), "{} runs did not stop:\n{}", failed.len(), failed.join("\n"));
}
