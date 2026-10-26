//! The sparse leaf arm's levels whose nodes are f's as f stores them, moved
//! from f instead of written: the same diagram, stored alike, for the same
//! work, and both operands given back as they were on a refusal after such
//! a level.

use std::sync::Arc;

use super::*;
use crate::Engine;
use crate::limits::{LimitConfig, SparseRoute, StopAt, StopRules};
use crate::test_helpers::{compile_clauses_on, rand_cnf, same_storage, vtree_shapes, CnfShape, Lcg};
use crate::vtree::Vtree;

/// Every level the sparse route can take, takes it.
const SPARSE: SparseRoute = SparseRoute { sparsity: 1, min_grid: 0 };

/// Pairs of formulas on every vtree shape over `nvars` variables, the
/// first implying the second: a random formula, and it with a few more
/// clauses. Where the first's nodes refine the second's, the leaf arm's
/// product level is the first's.
fn cases(nvars: u32, seed: u64) -> Vec<(String, Arc<Vtree>, Tdd, Tdd)> {
    let mut rng = Lcg::new(seed);
    let mut out = Vec::new();
    for (name, vtree) in vtree_shapes(nvars) {
        for case in 0..6 {
            let eng = Engine::new();
            let base = rand_cnf(&mut rng, nvars, CnfShape { clauses: 2 * nvars as usize, width: 3 });
            let more = rand_cnf(&mut rng, nvars, CnfShape { clauses: 1 + case % 3, width: 2 });
            let g = compile_clauses_on(&eng, &vtree, &base);
            let f = compile_clauses_on(&eng, &vtree, &[base, more].concat());
            out.push((format!("{name}, {nvars} variables, case {case}"), Arc::clone(&vtree), f, g));
        }
    }
    out
}

/// Conjoin `f` and `g` on a fresh engine, every level on the sparse route
/// it can take, and the work it took.
fn conjoin(f: &Tdd, g: &Tdd) -> (Tdd, u64) {
    let eng = Engine::new();
    let _sparse = eng.limits().scope(LimitConfig::none().with_sparse_route(SPARSE));
    let mark = eng.limits().mark();
    let out = eng.and(f.clone(), g.clone()).unwrap();
    (out, eng.limits().work_since(mark))
}

/// Both ways round, the levels moved from f against the same levels
/// written: the same nodes and pairs on every level, and the same work.
#[test]
fn a_level_moved_from_f_is_the_level_written() {
    let before = lent_levels();
    let mut conjunctions = 0;
    for nvars in [10, 14] {
        for (what, _, f, g) in cases(nvars, 0x1e47_0f00 + u64::from(nvars)) {
            for (a, b, order) in [(&f, &g, "f and g"), (&g, &f, "g and f")] {
                let (out, work) = conjoin(a, b);
                let (oracle, oracle_work) = no_lend(|| conjoin(a, b));
                assert!(same_storage(&out, &oracle), "{what}, {order}: the levels moved from f are not the levels written");
                assert_eq!(work, oracle_work, "{what}, {order}: the levels moved from f took other work");
                conjunctions += 1;
            }
        }
    }
    let lent = lent_levels() - before;
    eprintln!("levels moved from f: {lent} over {conjunctions} conjunctions");
    assert!(lent > 0, "expected levels moved from f, got none over {conjunctions} conjunctions");
}

/// What a refused conjunction gave back of an operand besides its
/// levels: whether they are closed, whether it is certified canonical, and
/// whether it keeps level counts.
fn kept_state(t: &Tdd) -> (bool, bool, bool) {
    (t.levels.is_closed(), t.levels.is_canonical(t.output), t.has_level_counts())
}

/// A restoring conjunction refused at each reserve and at each unit of
/// work in turn, on operands keeping level counts of which it moves levels
/// from f, against the same refusal with those levels written: every
/// refusal gives both operands back as they were given, levels and all,
/// and with what they keep besides as the written levels leave it, those
/// refused after a level was moved among them.
#[test]
fn a_refused_conjunction_gives_f_the_levels_moved_from_it_back() {
    let mut refused_after = 0;
    for (what, _, mut f, mut g) in cases(10, 0x1e47_0f0a) {
        let before = lent_levels();
        let _ = conjoin(&f, &g);
        if lent_levels() == before {
            continue;
        }
        let eng = Engine::new();
        eng.attach_level_counts(&mut f).unwrap();
        eng.attach_level_counts(&mut g).unwrap();
        let expected = conjoin(&f, &g).0;
        for arm in ["reserve", "work"] {
            for n in 0u64.. {
                // A fresh engine each time, so that both arms meet the
                // same reserves: a pooled scratch reserves on its first use.
                #[expect(clippy::result_large_err, reason = "the refusal hands back the operands")]
                let attempt = || {
                    let eng = Engine::new();
                    let stop = StopRules { unconditional: Some(StopAt::WorkUnits(eng.limits().work_units() + n)), after_pairs: None };
                    let config = LimitConfig::none().with_sparse_route(SPARSE);
                    let _scope = eng.limits().scope(if arm == "work" { config.with_stop_rules(stop) } else { config });
                    if arm == "reserve" {
                        eng.limits().refuse_nth_reserve(n as u32);
                    }
                    eng.and_restoring(f.clone(), g.clone())
                };
                let lent = lent_levels();
                let outcome = attempt();
                let moved = lent_levels() > lent;
                match (outcome, no_lend(attempt)) {
                    (Ok(out), Ok(_)) => {
                        assert!(eng.equivalent(&out, &expected).unwrap(), "{what}: granted at {arm} {n}, a different function");
                        break;
                    }
                    (Err(refused), Err(oracle)) => {
                        assert!(
                            same_storage(&refused.f, &f) && same_storage(&refused.g, &g),
                            "{what}: refused at {arm} {n}, an operand changed",
                        );
                        assert_eq!(
                            (kept_state(&refused.f), kept_state(&refused.g)),
                            (kept_state(&oracle.f), kept_state(&oracle.g)),
                            "{what}: refused at {arm} {n}, an operand kept other than the written levels leave it",
                        );
                        refused_after += u64::from(moved);
                    }
                    (outcome, oracle) => panic!(
                        "{what}: at {arm} {n}, granted {} with levels moved from f and {} with them written",
                        outcome.is_ok(), oracle.is_ok(),
                    ),
                }
                assert!(n < 100_000, "{what}: never granted");
            }
        }
    }
    eprintln!("refusals after a level moved from f: {refused_after}");
    assert!(refused_after > 0, "expected refusals after a level moved from f, got none");
}
