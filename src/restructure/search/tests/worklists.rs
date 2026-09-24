//! What a rotation trial does with the reduction obligations it inherits.

use std::sync::Arc;

use crate::Engine;
use crate::diagram::{Pass, Tdd};
use crate::restructure::search::RotationMove;
use crate::restructure::search::probe::rotate_if_on;
use crate::test_helpers::assert_canonical;
use crate::vtree::{RotationKind, Vtree};

/// The levels the diagram still owes each reduction pass, read without
/// consuming them.
fn owed(tdd: &Tdd) -> [Vec<u32>; 3] {
    let mut dirty = tdd.dirty.clone();
    [dirty.take(Pass::Contract), dirty.take(Pass::LeafContract), dirty.take(Pass::ContentTwin)]
}

#[test]
fn an_accepted_rotation_keeps_what_the_diagram_already_owed() {
    let eng = Engine::new();
    let vtree = Arc::new(Vtree::balanced(4));
    // A conjunction puts the levels it rebuilt on the worklists, and nothing
    // drains them before the probe, so the diagram enters the trial owing work
    // the way a mid-compile caller's does.
    let mut f = Tdd::clause(&vtree, [1, 2]).unwrap() & Tdd::clause(&vtree, [3, 4]).unwrap();
    let count = f.model_count().unwrap();
    let before = owed(&f);
    assert!(before.iter().any(|list| !list.is_empty()), "the fixture owes no pass anything");

    let turn = RotationMove { pivot: vtree.root(), kind: RotationKind::Left };
    let kept = rotate_if_on(&eng, &mut f, &[turn], usize::MAX, |_| true);
    assert_eq!(kept, Ok(true));

    // A level absent from a pass's list is taken to be at that pass's fixpoint,
    // so a level dropped here is one the next reduce walks past for good.
    let after = owed(&f);
    for (pass, (was, now)) in before.iter().zip(after.iter()).enumerate() {
        for level in was {
            assert!(now.contains(level), "pass {pass} lost level {level}");
        }
    }
    // Underneath, not instead of: the rebuilt level is on the list as well.
    assert!(after[0].contains(&vtree.root().0), "the rotation recorded no level of its own");

    eng.minimize(&mut f).unwrap();
    assert_canonical(&f);
    assert_eq!(f.model_count().unwrap(), count);
}
