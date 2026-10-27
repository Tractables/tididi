use super::*;
use crate::diagram::LevelState;
use crate::test_helpers::check::check_implicit_levels;
use crate::vtree::VarId;

/// The live pairs each level keeps, where it keeps them.
fn kept(f: &Tdd) -> Vec<Option<u64>> {
    f.levels
        .iter()
        .map(|l| match &l.state {
            LevelState::Structural(held) => held.get(),
            _ => None,
        })
        .collect()
}

/// Every level's live pairs, read off its nodes.
fn census(f: &Tdd) -> usize {
    f.levels.iter().map(TddLevel::live_pairs).sum()
}

#[test]
fn pair_count_keeps_each_closed_level_once() {
    let vtree = Arc::new(Vtree::balanced(6));
    let mut f = Tdd::clause(&vtree, [1, 2, -3]).unwrap() & Tdd::clause(&vtree, [-1, 4, 5]).unwrap();
    f.minimize().unwrap();
    assert!(f.levels.is_closed());
    let pairs = census(&f);
    assert!(pairs > 0);
    assert_eq!(f.pair_count(), pairs);
    // Each stored level now keeps its count, and the count reads them.
    for (level, n) in f.levels.iter().zip(kept(&f)) {
        if level.implicit().is_none() {
            assert_eq!(n, Some(level.live_pairs() as u64));
        }
    }
    assert_eq!(f.pair_count(), pairs);
    assert!(check_implicit_levels(&f).is_ok());

    // An operation's result counts its changed levels again.
    let g = f.clone().condition_var(VarId(1), true).unwrap();
    assert_eq!(g.pair_count(), census(&g));
    assert_eq!(g.pair_count(), census(&g));

    // A mutable access opens the levels: the count reads every level.
    let mut m = f.clone();
    let _ = &mut m.levels[0];
    assert!(!m.levels.is_closed());
    assert_eq!(m.pair_count(), pairs);

    // A kept count that is not the level's pairs fails the check.
    let t = f.levels.iter().position(|l| l.live_pairs() > 0 && l.implicit().is_none()).unwrap();
    if let LevelState::Structural(held) = &f.levels[t].state {
        held.set(f.levels[t].live_pairs() as u64 + 1);
    }
    assert!(check_implicit_levels(&f).is_err());
}

/// `f` with no level keeping a count, its levels known closed still where
/// they were.
fn forget_counts(f: &mut Tdd) {
    let closed = f.levels.is_closed();
    for level in f.levels.iter_mut() {
        level.forget_held_pairs();
    }
    if closed {
        f.levels.reinstate_closed();
    }
}

/// The stored structural levels of `f` that keep a count.
fn counted(f: &Tdd) -> usize {
    f.levels.iter().zip(kept(f)).filter(|(l, n)| l.implicit().is_none() && n.is_some()).count()
}

/// The counts `f`'s levels keep are their pairs, none is left for a close
/// that did not take it, and the count reads them.
fn assert_counts_hold(f: &Tdd, what: &str) {
    if let Err(e) = check_implicit_levels(f).and_then(|()| crate::test_helpers::check::check_left_pairs_taken(f)) {
        panic!("{what}: {e}");
    }
    assert_eq!(f.pair_count(), census(f), "{what}");
}

/// Conjoined unreduced, one random clause at a time; every other
/// conjunction's operands keep the counts a read left them, which the
/// levels it moves bring along.
fn conjunctions(eng: &crate::Engine, seed: u64) -> Vec<Tdd> {
    use crate::test_helpers::{rand_cnf, vtree_shapes, CnfShape, Lcg};
    let mut rng = Lcg::new(seed);
    let mut out = Vec::new();
    for (_, vtree) in vtree_shapes(8) {
        for _ in 0..6 {
            let mut f = Tdd::one(&vtree);
            for (k, clause) in rand_cnf(&mut rng, 8, CnfShape { clauses: 8, width: 3 }).into_iter().enumerate() {
                let mut g = eng.clause(&vtree, clause.iter().copied()).unwrap();
                if k % 2 == 0 {
                    forget_counts(&mut f);
                    forget_counts(&mut g);
                } else {
                    let _ = (f.pair_count(), g.pair_count());
                }
                f = eng.and(f, g).unwrap();
                out.push(f.clone());
            }
        }
    }
    out
}

#[test]
fn the_operations_leave_the_levels_they_change_counted() {
    let left = left_counted(0x9a17);
    assert!(left.iter().all(|&n| n > 0), "levels left counted: {left:?}");
    // The same where small levels are held as descriptions.
    crate::test_helpers::with_floor(crate::test_helpers::LOW_FLOOR, || left_counted(0x9a18));
}

/// The stored levels that keep a count after a conjunction, a negation, a
/// prune of levels counted before it and a minimization, of diagrams
/// drawn from `seed`, each operation's counts checked.
fn left_counted(seed: u64) -> [usize; 4] {
    let eng = crate::Engine::new();
    let mut left = [0usize; 4];
    for f in conjunctions(&eng, seed) {
        left[0] += counted(&f);
        assert_counts_hold(&f, "conjunction");

        let mut g = f.clone();
        forget_counts(&mut g);
        let n = eng.negate(g).unwrap();
        left[1] += counted(&n);
        assert_counts_hold(&n, "negation");

        let mut p = f.clone();
        let _ = p.pair_count();
        let before: Vec<usize> = p.levels.iter().map(TddLevel::live_pairs).collect();
        p.reduce(crate::reduce::ReductionPlan::Prune).unwrap();
        left[2] += p.levels.iter().zip(kept(&p)).zip(&before)
            .filter(|((l, n), b)| l.implicit().is_none() && n.is_some() && l.live_pairs() != **b)
            .count();
        assert_counts_hold(&p, "prune");

        let mut m = f;
        forget_counts(&mut m);
        m.minimize().unwrap();
        left[3] += counted(&m);
        assert_counts_hold(&m, "minimization");
    }
    left
}

#[test]
fn a_refused_reduction_leaves_no_count_for_a_close() {
    use crate::limits::{LimitConfig, StopCallback, StopDecision};
    use std::sync::atomic::{AtomicUsize, Ordering};
    let eng = crate::Engine::new();
    let mut refused = 0;
    for f in conjunctions(&eng, 0x51c3).into_iter().step_by(5) {
        let polls = Arc::new(AtomicUsize::new(0));
        let seen = Arc::clone(&polls);
        let counting = StopCallback::new(move |_, _| {
            seen.fetch_add(1, Ordering::Relaxed);
            StopDecision::Continue
        });
        {
            let _scope = eng.limits().scope(LimitConfig::none().with_stop_callback(Some(counting)));
            let mut m = f.clone();
            let _ = m.pair_count();
            eng.minimize(&mut m).unwrap();
        }
        for allowed in 0..polls.load(Ordering::Relaxed) {
            let calls = Arc::new(AtomicUsize::new(0));
            let seen = Arc::clone(&calls);
            let stop = StopCallback::new(move |_, _| {
                if seen.fetch_add(1, Ordering::Relaxed) >= allowed { StopDecision::Stop } else { StopDecision::Continue }
            });
            let mut m = f.clone();
            let _ = m.pair_count();
            let done = {
                let _scope = eng.limits().scope(LimitConfig::none().with_stop_callback(Some(stop)));
                eng.minimize(&mut m)
            };
            if done.is_err() {
                refused += 1;
                assert_counts_hold(&m, "a refused minimization");
                eng.minimize(&mut m).unwrap();
                assert_counts_hold(&m, "a minimization after a refused one");
            }
        }
    }
    assert!(refused > 0);
}

/// A conjunction that borrows an operand leaves the borrow's counts as they
/// were and the result's counts holding; a restoring one refused at each
/// unit of work hands an owned operand back with the counts it kept.
#[test]
fn borrowed_and_refused_conjunctions_keep_the_counts_they_were_given() {
    use crate::limits::{LimitConfig, StopAt, StopRules};
    use std::borrow::Cow;
    let eng = crate::Engine::new();
    let diagrams = conjunctions(&eng, 0xb0e5);
    let mut refusals = 0;
    for (i, j) in (0..diagrams.len() - 9).step_by(7).map(|i| (i, i + 9)) {
        if !Arc::ptr_eq(diagrams[i].vtree(), diagrams[j].vtree()) {
            continue;
        }
        for read in [true, false] {
            let mut ab = [diagrams[i].clone(), diagrams[j].clone()];
            for t in &mut ab {
                if read {
                    let _ = t.pair_count();
                } else {
                    forget_counts(t);
                }
            }
            let [a, b] = &ab;
            let given = [kept(a), kept(b)];
            for (x, y) in [(a, b), (b, a)] {
                for borrow in [(true, false), (false, true), (true, true)] {
                    let side = |t, borrowed| if borrowed { Cow::Borrowed(t) } else { Cow::Owned(Tdd::clone(t)) };
                    let out = eng.and(side(x, borrow.0), side(y, borrow.1)).unwrap();
                    assert_counts_hold(&out, "a conjunction of a borrowed operand");
                }
                assert_eq!([kept(a), kept(b)], given, "a borrowed operand's counts changed");
                for n in 0u64.. {
                    let stop = StopRules { unconditional: Some(StopAt::WorkUnits(eng.limits().work_units() + n)), after_pairs: None };
                    let done = {
                        let _scope = eng.limits().scope(LimitConfig::none().with_stop_rules(stop));
                        eng.and_restoring(x.clone(), y.clone())
                    };
                    match done {
                        Ok(out) => {
                            assert_counts_hold(&out, "a granted restoring conjunction");
                            break;
                        }
                        Err(r) => {
                            refusals += 1;
                            for (back, was) in [(r.f, x), (r.g, y)] {
                                let back = back.into_owned();
                                assert_eq!(kept(&back), kept(was), "refused at work {n}: an operand's counts changed");
                                assert_eq!(back.levels.is_closed(), was.levels.is_closed());
                                assert_counts_hold(&back, "a refused restoring conjunction's operand");
                            }
                        }
                    }
                    assert!(n < 100_000, "never granted");
                }
            }
        }
    }
    assert!(refusals > 100, "only {refusals} refusals");
}
