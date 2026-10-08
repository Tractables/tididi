//! `Engine::and_restoring` gives both operands back as they were at every
//! point the conjunction can be refused: before the sweep, between levels
//! after identity levels have moved into the output, and when the result's
//! worklists are seeded. The table of the level counts the result keeps is
//! not such a point: refused, the result keeps none.
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

/// Operands keeping their level counts, the conjunction refused at each
/// reserve in turn: each refusal gives both back unchanged, and the last
/// reserve, the table of the counts the result keeps for the levels it
/// moved from them, is granted without it, the result keeping no counts.
#[test]
fn a_refused_count_table_keeps_the_operands_and_drops_the_counts() {
    let eng = Engine::new();
    let vtree = Arc::new(Vtree::balanced(12));
    let (mut f, mut g) = operands(&eng, &vtree);
    eng.attach_level_counts(&mut f).unwrap();
    eng.attach_level_counts(&mut g).unwrap();
    let expected = eng.and(f.clone(), g.clone()).unwrap();
    let count = eng.model_count(&expected).unwrap();
    for (a, b) in [(&f, &g), (&g, &f)] {
        let granted = eng.and_restoring(a.clone(), b.clone()).map_err(|r| r.error).unwrap();
        assert!(granted.has_level_counts(), "the conjunction kept no counts: nothing to refuse");
        let (mut refusals, mut dropped) = (0, 0);
        for n in 0.. {
            eng.limits().refuse_nth_reserve(n);
            let outcome = eng.and_restoring(a.clone(), b.clone());
            let fired = !eng.limits().refusal_pending();
            eng.limits().grant_every_reserve();
            match outcome {
                Ok(out) => {
                    assert!(eng.equivalent(&out, &expected).unwrap(), "granted at {n}: a different function");
                    assert_eq!(eng.model_count(&out).unwrap(), count, "granted at {n}: count");
                    dropped += usize::from(!out.has_level_counts());
                }
                Err(refused) => {
                    // Their levels as given; a moved level dropped its
                    // operand's counts, a cache, on the way out.
                    assert!(same(&refused.f, a) && same(&refused.g, b), "refused at {n}: an operand changed");
                    refusals += 1;
                }
            }
            if !fired {
                break;
            }
            assert!(n < 10_000, "never past the last reserve");
        }
        assert!(refusals > 3, "only {refusals} refusal points");
        assert_eq!(dropped, 1, "the count table's refusal kept the counts, or another one dropped them");
    }
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
