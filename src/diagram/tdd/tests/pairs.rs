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

/// Conjoined unreduced, one random clause at a time.
fn conjunctions(eng: &crate::Engine, seed: u64) -> Vec<Tdd> {
    use crate::test_helpers::{rand_cnf, vtree_shapes, CnfShape, Lcg};
    let mut rng = Lcg::new(seed);
    let mut out = Vec::new();
    for (_, vtree) in vtree_shapes(8) {
        for _ in 0..6 {
            let mut f = Tdd::one(&vtree);
            for clause in rand_cnf(&mut rng, 8, CnfShape { clauses: 8, width: 3 }) {
                let mut g = eng.clause(&vtree, clause.iter().copied()).unwrap();
                forget_counts(&mut f);
                forget_counts(&mut g);
                f = eng.and(f, g).unwrap();
                out.push(f.clone());
            }
        }
    }
    out
}

#[test]
fn the_operations_leave_the_levels_they_change_counted() {
    let eng = crate::Engine::new();
    // Stored levels that keep a count after a conjunction, a negation, a
    // prune of levels counted before it, a minimization.
    let mut left = [0usize; 4];
    for f in conjunctions(&eng, 0x9a17) {
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
    assert!(left[1] > 0 && left[2] > 0, "levels left counted: {left:?}");
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
