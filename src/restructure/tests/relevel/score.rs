//! [`score_levels`] against [`prepare_levels`]: on every move at every
//! internal node of seeded diagrams, crossed or not, the score is the pairs
//! the preparation gives, and both give up at the same bound.

use super::*;
use crate::test_helpers::{random_diagrams, rotate_left, rotate_right, with_floor};

/// Every move of `tdd` scored and prepared under `bound`; the moves compared.
fn compare_every_move(tdd: &Tdd, bound: usize, scratch: &mut RestructureScratch) -> usize {
    let lim = Limits::new();
    let mut compared = 0;
    for (v, _, _) in tdd.vtree.internal_bottomup() {
        for kind in [RotationKind::Left, RotationKind::Right] {
            let mut tree = (*tdd.vtree).clone();
            let rotated = match kind {
                RotationKind::Left => rotate_left(&mut tree, v),
                RotationKind::Right => rotate_right(&mut tree, v),
            };
            let Some(info) = rotated else { continue };
            for crossed in [false, true] {
                let mut prepared = Prepared::default();
                let exact = prepare_levels(&lim, tdd, &info, kind, crossed, false, scratch, &mut prepared, bound)
                    .expect("nothing is armed");
                let score = score_levels(&lim, tdd, &info, kind, crossed, &mut scratch.score, bound)
                    .expect("nothing is armed");
                assert_eq!(score, exact, "{kind:?} at {v:?}, crossed {crossed}, bound {bound}");
                compared += 1;
            }
        }
    }
    compared
}

#[test]
fn a_score_is_the_pairs_a_preparation_gives() {
    let mut scratch = RestructureScratch::default();
    let mut compared = 0;
    for tdd in random_diagrams(0x5c0e, 120, 3..10) {
        let mut minimized = tdd.clone();
        minimized.minimize().expect("an unarmed engine refuses nothing");
        for f in [&tdd, &minimized] {
            compared += compare_every_move(f, usize::MAX, &mut scratch);
        }
    }
    assert!(compared > 2000, "{compared} moves compared");
}

#[test]
fn a_score_gives_up_where_a_preparation_does() {
    let mut scratch = RestructureScratch::default();
    let mut gave_up = 0;
    for tdd in random_diagrams(0xb0d, 16, 4..8) {
        for bound in [0, 1, 2, 3, 5, 8, 13, 21, 34] {
            compare_every_move(&tdd, bound, &mut scratch);
        }
        let lim = Limits::new();
        let root = tdd.vtree.root();
        let mut tree = (*tdd.vtree).clone();
        if let Some(info) = rotate_left(&mut tree, root) {
            gave_up += usize::from(score_levels(&lim, &tdd, &info, RotationKind::Left, false, &mut scratch.score, 1).unwrap().is_none());
        }
    }
    assert!(gave_up > 0);
}

/// Implicit levels are read through their description.
#[test]
fn a_score_reads_implicit_levels() {
    with_floor(4, || {
        let mut scratch = RestructureScratch::default();
        let mut implicit = 0;
        for mut tdd in random_diagrams(0x1e7e1, 24, 5..10) {
            tdd.minimize().expect("an unarmed engine refuses nothing");
            implicit += tdd.levels.iter().filter(|l| l.implicit().is_some()).count();
            compare_every_move(&tdd, usize::MAX, &mut scratch);
        }
        assert!(implicit > 0, "no diagram held an implicit level");
    });
}
