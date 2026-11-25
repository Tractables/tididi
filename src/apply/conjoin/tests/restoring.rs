//! `Engine::and_restoring` gives both operands back as they were at every
//! point the conjunction can be refused: before the sweep, between levels
//! after identity levels have moved into the output, and when the result's
//! worklists are seeded. Each comes back stored as given and known as
//! given: certified canonical, closed and keeping level counts where it
//! was, so that what is done with it next costs what it would have.
use std::sync::Arc;

use crate::limits::{LimitConfig, StopAt, StopRules};
use crate::test_helpers::{known, same_storage as same};
use crate::vtree::{Home, VarId};
use crate::{Engine, Tdd, Vtree};

/// Two operands whose supports overlap on a few variables of a balanced
/// vtree over 12, so the sweep takes identity levels from both before it
/// reaches the shared ones.
fn operands(eng: &Engine, vtree: &Arc<Vtree>) -> (Tdd, Tdd) {
    (build(eng, vtree, F_CLAUSES), build(eng, vtree, G_CLAUSES))
}

const F_CLAUSES: &[&[i32]] = &[&[1, 2, -7], &[-1, 3], &[2, -3, 8], &[4, 5], &[-5, 6, 7]];
const G_CLAUSES: &[&[i32]] = &[&[7, 8, -9], &[-8, 10], &[9, -10, 11], &[-11, 12, 1], &[6, -12]];

/// The conjunction of `clauses` on `vtree`, minimized.
fn build(eng: &Engine, vtree: &Arc<Vtree>, clauses: &[&[i32]]) -> Tdd {
    let mut f = eng.cube(&crate::vtree::Home::unchecked(vtree), std::iter::empty::<i32>()).unwrap();
    for c in clauses {
        let c = eng.clause(&crate::vtree::Home::unchecked(vtree), c.iter().copied()).unwrap();
        f = eng.and(f, c).unwrap();
    }
    eng.minimize(&mut f).unwrap();
    f
}

/// `t` minimized and then negated on a fresh engine, with the work each
/// took: what the certificate and the closed form save.
fn reduced_and_negated(t: &Tdd) -> (Tdd, u64, Tdd, u64) {
    let eng = Engine::new();
    let mut m = t.clone();
    let mark = eng.limits().mark();
    eng.minimize(&mut m).unwrap();
    let minimized = eng.limits().work_since(mark);
    let mark = eng.limits().mark();
    let n = crate::apply::with_plan::negate_with(&eng, t.clone(), crate::reduce::ReductionPlan::Prune).unwrap();
    (m, minimized, n, eng.limits().work_since(mark))
}

/// Whether `back`, handed back by a refusal, is `given`: stored and known
/// alike, and minimized and negated alike for the same work.
fn as_given(back: &Tdd, given: &Tdd) -> bool {
    let (m, mw, n, nw) = reduced_and_negated(back);
    let (gm, gmw, gn, gnw) = reduced_and_negated(given);
    same(back, given) && known(back) == known(given) && same(&m, &gm) && mw == gmw && same(&n, &gn) && nw == gnw
}

/// Refuse the conjunction at each point `arm` picks in turn, on operands
/// keeping level counts where `counts` says, checking that every refusal
/// hands both operands back as given, until one is granted.
fn refuse_at_every_point(
    counts: bool,
    given: impl Fn(&Engine) -> (Tdd, Tdd),
    arm: impl Fn(&Engine, u64) -> Option<crate::limits::LimitScope<'_>>,
) {
    let eng = Engine::new();
    let (mut f, mut g) = given(&eng);
    if counts {
        eng.attach_level_counts(&mut f).unwrap();
        eng.attach_level_counts(&mut g).unwrap();
    }
    for t in [&f, &g] {
        let (canonical, closed, kept, _) = known(t);
        assert!(canonical && closed && kept == counts, "an operand certified, closed and keeping counts as the test needs");
    }
    let expected = eng.and(f.clone(), g.clone()).unwrap();
    for (a, b) in [(&f, &g), (&g, &f)] {
        let mut refusals = 0;
        for n in 0.. {
            let scope = arm(&eng, n);
            let outcome = eng.and_restoring(a.clone(), b.clone());
            let pending = scope.is_some() || eng.limits().refusal_pending();
            drop(scope);
            eng.limits().grant_every_reserve();
            match outcome {
                Ok(out) => {
                    assert!(eng.equivalent(&out, &expected).unwrap(), "granted at {n}: a different function");
                    if pending {
                        break;
                    }
                    continue;
                }
                Err(refused) => {
                    let count = |t: &Tdd| Engine::new().model_count(t).map_err(|e| format!("{e:?}"));
                    assert_eq!((count(&refused.f), count(&refused.g)), (count(a), count(b)), "refused at {n}: another model count");
                    assert!(as_given(&refused.f, a) && as_given(&refused.g, b), "refused at {n}: an operand changed");
                    assert!(
                        Arc::ptr_eq(refused.f.vtree(), a.vtree()) && Arc::ptr_eq(refused.g.vtree(), b.vtree()),
                        "refused at {n}: an operand on another vtree",
                    );
                    refusals += 1;
                }
            }
            assert!(n < 10_000, "never granted");
        }
        assert!(refusals > 3, "only {refusals} refusal points");
    }
}

/// [`operands`] on one balanced vtree over 12.
fn on_one_vtree(eng: &Engine) -> (Tdd, Tdd) {
    operands(eng, &Home::new(Vtree::balanced(12)))
}

/// The variables `vars` as a projection of `home`.
fn projection(home: &Home, vars: impl IntoIterator<Item = u32>) -> Arc<Vtree> {
    Vtree::projection(home, vars.into_iter().map(VarId)).unwrap()
}

/// [`operands`] on projections of a balanced home over 13: `f` on the
/// variables 1 to 12, `g` on its own, inside them.
fn nested(eng: &Engine) -> (Tdd, Tdd) {
    let home = Home::new(Arc::new(Vtree::balanced(13)));
    let f = build(eng, &projection(&home, 1..=12), F_CLAUSES);
    let g = build(eng, &projection(&home, std::iter::once(1).chain(6..=12)), G_CLAUSES);
    (f, g)
}

/// [`operands`] on two allocations of one projection.
fn equal_layouts(eng: &Engine) -> (Tdd, Tdd) {
    let home = Home::new(Arc::new(Vtree::balanced(13)));
    let (f, _) = operands(eng, &projection(&home, 1..=12));
    let (_, g) = operands(eng, &projection(&home, 1..=12));
    (f, g)
}

/// [`operands`] on one allocation of a projection.
fn one_projection(eng: &Engine) -> (Tdd, Tdd) {
    let home = Home::new(Arc::new(Vtree::balanced(13)));
    operands(eng, &projection(&home, 1..=12))
}

/// [`operands`] on projections of a balanced home over 13 with neither
/// leaf set inside the other: `f` on the variables 1 to 9, `g` on 1 and 6
/// to 12.
fn union(eng: &Engine) -> (Tdd, Tdd) {
    let home = Home::new(Arc::new(Vtree::balanced(13)));
    let f = build(eng, &projection(&home, 1..=9), F_CLAUSES);
    let g = build(eng, &projection(&home, std::iter::once(1).chain(6..=12)), G_CLAUSES);
    (f, g)
}

type Layout = fn(&Engine) -> (Tdd, Tdd);

/// Each layout of operands the restoring conjunction takes owned, and the
/// union layout, which it reads borrowed.
const LAYOUTS: [Layout; 5] = [on_one_vtree, nested, equal_layouts, one_projection, union];

#[test]
fn every_refused_reserve_gives_the_operands_back() {
    for (given, counts) in LAYOUTS.into_iter().flat_map(|given| [(given, false), (given, true)]) {
        refuse_at_every_point(counts, given, |eng, n| {
            eng.refuse_nth_reserve(n as u32);
            None
        });
    }
}

#[test]
fn every_stop_gives_the_operands_back() {
    for (given, counts) in LAYOUTS.into_iter().flat_map(|given| [(given, false), (given, true)]) {
        refuse_at_every_point(counts, given, |eng, n| {
            let at = eng.limits().work_units() + n;
            let stop = StopRules { unconditional: Some(StopAt::WorkUnits(at)), after_pairs: None };
            Some(eng.limits().scope(LimitConfig::none().with_stop_rules(stop)))
        });
    }
}

/// A conjunction that adopts an owned `f`, refused at each reserve in
/// turn, leaves the engine's workspace so that the next one is right: `g`
/// spread over `f`'s vtree, and `g` on two variables deep in it, where the
/// sweep may stop below the root.
#[test]
fn a_refused_adopting_conjunction_leaves_the_next_one_right() {
    let eng = Engine::new();
    let home = Home::new(Arc::new(Vtree::linear(13)));
    let (f, spread) = nested(&eng);
    let deep = build(&eng, &projection(&home, 11..=12), &[&[11, -12]]);
    let on_linear = build(&eng, &projection(&home, 1..=12), F_CLAUSES);
    for (f, g) in [(&f, &spread), (&on_linear, &deep)] {
        let expected = eng.and(f.clone(), g).unwrap();
        let mut refusals = 0;
        for n in 0.. {
            let given = f.clone();
            eng.refuse_nth_reserve(n);
            let outcome = eng.and(given, g);
            let pending = eng.limits().refusal_pending();
            eng.limits().grant_every_reserve();
            let next = eng.and(f.clone(), g).unwrap();
            assert!(eng.equivalent(&next, &expected).unwrap(), "after a refusal at {n}");
            match outcome {
                Ok(out) => {
                    assert!(eng.equivalent(&out, &expected).unwrap(), "granted at {n}");
                    if pending {
                        break;
                    }
                }
                Err(_) => refusals += 1,
            }
            assert!(n < 10_000, "never granted");
        }
        assert!(refusals > 3, "only {refusals} refusal points");
    }
}

/// `g` on the two deepest variables of an implication chain forces every
/// variable above it, so the sweep reaches each level of the chain to the
/// root; a second `g` the result implies stops it at once.
#[test]
fn a_forced_chain_is_reached_to_the_root() {
    let eng = Engine::new();
    let home = Home::new(Arc::new(Vtree::linear(14)));
    let chain: Vec<[i32; 2]> = (1..13).map(|v| [v, -(v + 1)]).collect();
    let clauses: Vec<&[i32]> = chain.iter().map(|c| &c[..]).collect();
    let f = build(&eng, &projection(&home, 1..=13), &clauses);
    let g = build(&eng, &projection(&home, 12..=13), &[&[12, 13]]);
    let mut with_g = clauses.clone();
    with_g.push(&[12, 13]);
    let expected = build(&eng, &projection(&home, 1..=13), &with_g);
    let out = eng.and(f, &g).unwrap();
    assert!(eng.equivalent(&out, &expected).unwrap());
    let again = eng.and(out, &g).unwrap();
    assert!(eng.equivalent(&again, &expected).unwrap());
}

/// Owned operands with a summed-out level are read in place, not copied
/// outside the budget: refused at each reserve in turn, both come back as
/// given, and the granted conjunction is `and`'s.
#[test]
fn a_summed_out_conjunction_refused_at_each_reserve_gives_the_operands_back() {
    let eng = Engine::new();
    let vtree = Arc::new(Vtree::balanced(8));
    let f = build(&eng, &vtree, &[&[5, 6], &[-7, 8], &[6, -8]]);
    let mut g = build(&eng, &vtree, &[&[2, 5], &[-4, 6], &[7, 8, -2], &[1, -6]]);
    eng.marginalize_levels(&mut g, &[vtree.children(vtree.root()).0]).unwrap();
    assert!(g.has_marginal_level());
    // `equivalent` refuses a summed-out level: the counts stand in for it.
    let count = |t: &Tdd| Engine::new().model_count(t).unwrap();
    let expected = count(&eng.and(f.clone(), g.clone()).unwrap());
    let mut refusals = 0;
    for n in 0.. {
        eng.refuse_nth_reserve(n);
        let outcome = eng.and_restoring(f.clone(), g.clone());
        let pending = eng.limits().refusal_pending();
        eng.limits().grant_every_reserve();
        match outcome {
            Ok(out) if pending => {
                assert_eq!(count(&out), expected);
                break;
            }
            Ok(out) => assert_eq!(count(&out), expected, "granted at {n}: another count"),
            Err(refused) => {
                assert!(same(&refused.f, &f) && same(&refused.g, &g), "refused at {n}: an operand changed");
                refusals += 1;
            }
        }
        assert!(n < 10_000, "never granted");
    }
    assert!(refusals > 0, "no refusal point");
}

#[test]
fn a_granted_conjunction_matches_and() {
    let eng = Engine::new();
    let vtree = Home::new(Vtree::balanced(12));
    let (f, g) = operands(&eng, &vtree);
    let mut restored = eng.and_restoring(f.clone(), g.clone()).map_err(|r| r.error).unwrap();
    let mut plain = eng.and(f, g).unwrap();
    eng.minimize(&mut restored).unwrap();
    eng.minimize(&mut plain).unwrap();
    assert!(same(&restored, &plain));
}
