//! What a rotation trial does with the reduction obligations it inherits.

use std::sync::Arc;

use crate::Engine;
use crate::diagram::{Pass, Tdd};
use crate::restructure::search::RotationMove;
use crate::restructure::search::probe::rotate_if_on;
use crate::test_helpers::check::marginal::check_pair_fusion_saturation;
use crate::test_helpers::{assert_canonical, assert_same_shape, compile_clauses_on, rand_cnf, vtree_shapes, CnfShape};
use crate::vtree::rng::Lcg;
use crate::vtree::{RotationKind, Vtree, VtreeIdx};

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

/// The levels whose pair fusion record is set.
fn recorded(tdd: &Tdd) -> Vec<VtreeIdx> {
    (0..tdd.vtree.num_nodes() as u32).map(VtreeIdx).filter(|&t| tdd.dirty.is_fused(t.0)).collect()
}

/// Random formulas on every vtree shape with a level marginalized and the
/// result minimized, then rotated at random pivots: an accepted rotation
/// keeps the records of levels it neither rebuilt nor pushed, each of those
/// levels has no fusion left to find, and minimizing gives the diagram
/// minimizing with every record forgotten gives.
#[test]
fn an_accepted_rotation_keeps_the_fusion_records_it_left_alone() {
    let mut rng = Lcg::new(41);
    let mut kept_records = 0;
    for num_vars in 5..9u32 {
        for (shape, vtree) in vtree_shapes(num_vars) {
            let eng = Engine::new();
            let clauses = rand_cnf(&mut rng, num_vars, CnfShape { clauses: 2 * num_vars as usize, width: 3 });
            let mut f = compile_clauses_on(&eng, &vtree, &clauses);
            if f.is_zero() {
                continue;
            }
            let inner: Vec<VtreeIdx> = (0..vtree.num_nodes() as u32)
                .map(VtreeIdx)
                .filter(|&t| t != vtree.root() && !vtree.node(t).is_leaf())
                .collect();
            let target = inner[rng.below(inner.len() as u64) as usize];
            eng.marginalize_levels(&mut f, &[target]).unwrap();
            eng.minimize(&mut f).unwrap();
            let count = f.model_count().unwrap();
            for step in 0..6 {
                let label = format!("{shape} over {num_vars} variables, step {step}");
                let before = recorded(&f);
                let pivots: Vec<VtreeIdx> = (0..f.vtree.num_nodes() as u32)
                    .map(VtreeIdx)
                    .filter(|&t| !f.vtree.node(t).is_leaf())
                    .collect();
                let pivot = pivots[rng.below(pivots.len() as u64) as usize];
                let kind = if rng.below(2) == 0 { RotationKind::Left } else { RotationKind::Right };
                if rotate_if_on(&eng, &mut f, &[RotationMove { pivot, kind }], usize::MAX, |_| true) != Ok(true) {
                    continue;
                }
                let after = recorded(&f);
                kept_records += before.iter().filter(|t| after.contains(t)).count();
                if let Err(e) = check_pair_fusion_saturation(&f, Some(&after)) {
                    panic!("{label}: {e}");
                }
                let mut forced = f.clone();
                forced.dirty.forget_all_fused();
                eng.minimize(&mut f).unwrap();
                eng.minimize(&mut forced).unwrap();
                forced.dirty.forget_all_fused();
                assert_same_shape(&f, &forced, &label);
                assert_eq!(f.model_count().unwrap(), count, "{label}");
            }
        }
    }
    assert!(kept_records > 0, "no rotation kept a record");
}
