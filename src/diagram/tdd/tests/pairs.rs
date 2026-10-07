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
