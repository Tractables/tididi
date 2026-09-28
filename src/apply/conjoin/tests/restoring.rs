//! `Engine::and_restoring` gives both operands back as they were at every
//! point the conjunction can be refused: before the sweep, between levels
//! after identity levels have moved into the output, and when the result's
//! worklists are seeded.
use std::sync::Arc;

use crate::limits::{LimitConfig, StopAt, StopRules};
use crate::test_helpers::same_storage as same;
use crate::{Engine, Tdd, Vtree};

/// Two operands whose supports overlap on a few variables of a balanced
/// vtree over 12, so the sweep takes identity levels from both before it
/// reaches the shared ones.
fn operands(eng: &Engine, vtree: &Arc<Vtree>) -> (Tdd, Tdd) {
    let build = |clauses: &[&[i32]]| {
        let mut f = eng.cube(vtree, std::iter::empty::<i32>()).unwrap();
        for c in clauses {
            let c = eng.clause(vtree, c.iter().copied()).unwrap();
            f = eng.and(f, c).unwrap();
        }
        eng.minimize(&mut f).unwrap();
        f
    };
    let f = build(&[&[1, 2, -7], &[-1, 3], &[2, -3, 8], &[4, 5], &[-5, 6, 7]]);
    let g = build(&[&[7, 8, -9], &[-8, 10], &[9, -10, 11], &[-11, 12, 1], &[6, -12]]);
    (f, g)
}

/// Refuse the conjunction at each point `arm` picks in turn, checking that
/// every refusal hands both operands back unchanged, until one is granted.
fn refuse_at_every_point(arm: impl Fn(&Engine, u64) -> Option<crate::limits::LimitScope<'_>>) {
    let eng = Engine::new();
    let vtree = Arc::new(Vtree::balanced(12));
    let (f, g) = operands(&eng, &vtree);
    let expected = eng.and(f.clone(), g.clone()).unwrap();
    for (a, b) in [(&f, &g), (&g, &f)] {
        let mut refusals = 0;
        for n in 0.. {
            let scope = arm(&eng, n);
            let outcome = eng.and_restoring(a.clone(), b.clone());
            drop(scope);
            eng.limits().grant_every_reserve();
            match outcome {
                Ok(out) => {
                    assert!(eng.equivalent(&out, &expected).unwrap(), "granted at {n}: a different function");
                    break;
                }
                Err(refused) => {
                    assert!(same(&refused.f, a) && same(&refused.g, b), "refused at {n}: an operand changed");
                    refusals += 1;
                }
            }
            assert!(n < 10_000, "never granted");
        }
        assert!(refusals > 3, "only {refusals} refusal points");
    }
}

#[test]
fn every_refused_reserve_gives_the_operands_back() {
    refuse_at_every_point(|eng, n| {
        eng.limits().refuse_nth_reserve(n as u32);
        None
    });
}

#[test]
fn every_stop_gives_the_operands_back() {
    refuse_at_every_point(|eng, n| {
        let at = eng.limits().work_units() + n;
        let stop = StopRules { unconditional: Some(StopAt::WorkUnits(at)), after_pairs: None };
        Some(eng.limits().scope(LimitConfig::none().with_stop_rules(stop)))
    });
}

#[test]
fn a_granted_conjunction_matches_and() {
    let eng = Engine::new();
    let vtree = Arc::new(Vtree::balanced(12));
    let (f, g) = operands(&eng, &vtree);
    let mut restored = eng.and_restoring(f.clone(), g.clone()).map_err(|r| r.error).unwrap();
    let mut plain = eng.and(f, g).unwrap();
    eng.minimize(&mut restored).unwrap();
    eng.minimize(&mut plain).unwrap();
    assert!(same(&restored, &plain));
}
