//! Level counts kept with a diagram: each column is what the fold computes
//! again, a count that reads them is the count that folds, and a
//! conjunction or any other change to the levels drops them.

use std::sync::Arc;

use super::*;
use super::relabel::{cases, function_of};
use crate::Engine;
use crate::test_helpers::{vtree_shapes, Lcg};
use crate::value::CountRead;
use crate::vtree::{VarId, Vtree};

/// Every kept column of `f` against the fold over `f` as it is now; the
/// number of columns checked.
fn kept_columns_are_folded_counts(eng: &Engine, f: &Tdd, what: &str) -> usize {
    let Some(counts) = f.levels.counts() else { return 0 };
    let folded = eng.node_counts_u128(f).unwrap();
    let mut checked = 0;
    for t in f.vtree.bottomup() {
        let Some(column) = counts.column(t) else { continue };
        assert!(!f.vtree.node(t).is_leaf() && !f.levels[t.idx()].is_marginal(), "{what}: a column at {t:?}");
        assert_eq!(column.len(), f.levels[t.idx()].slot_count(), "{what}: width at {t:?}");
        for (i, &expected) in folded[t.idx()].iter().enumerate() {
            match column.get(i) {
                CountRead::Fast(c) => assert_eq!(c, expected, "{what}: node {i} of {t:?}"),
                CountRead::Big(_) => panic!("{what}: an overflow at {t:?} under a few variables"),
            }
        }
        checked += 1;
    }
    checked
}

/// A copy of `f` keeping its level counts.
fn counted(eng: &Engine, f: &Tdd) -> Tdd {
    let mut f = f.clone();
    eng.attach_level_counts(&mut f).unwrap();
    assert!(f.has_level_counts());
    f
}

#[test]
fn kept_columns_are_the_folds_columns() {
    let mut rng = Lcg::new(0x01e7_e1c0);
    let eng = Engine::new();
    for (shape, vtree) in vtree_shapes(9) {
        let all: Vec<u32> = (1..=9).collect();
        for _ in 0..4 {
            let f = function_of(&vtree, &all, &mut rng);
            let kept = counted(&eng, &f);
            // The false diagram keeps no column: its count needs none.
            let internal = if f.is_zero() { 0 } else { vtree.internal_bottomup().count() };
            assert_eq!(kept_columns_are_folded_counts(&eng, &kept, shape), internal, "{shape}: a level without a column");
            assert_eq!(eng.model_count(&kept).unwrap(), eng.model_count(&f).unwrap(), "{shape}: model count");
        }
    }
}

/// A conjunction of operands that keep their level counts keeps none, but
/// `f ∧ f`, which returns `f` as it was.
#[test]
fn a_conjunction_keeps_no_counts() {
    let eng = Engine::new();
    for (what, f, g) in cases(9, 0x3c_a221) {
        let plain = eng.and(f.clone(), g.clone()).unwrap();
        for (a, b) in [(counted(&eng, &f), g.clone()), (f.clone(), counted(&eng, &g)), (counted(&eng, &f), counted(&eng, &g))] {
            let out = eng.and(a.clone(), b.clone()).unwrap();
            assert!(!out.has_level_counts() || super::is_self_conjunction(&a, &b), "{what}: a conjunction kept counts");
            assert_eq!(eng.model_count(&out).unwrap(), eng.model_count(&plain).unwrap(), "{what}");
            assert_eq!(eng.and_model_count(a, b, &[]).unwrap(), eng.model_count(&plain).unwrap(), "{what}: and_model_count");
        }
    }
}

/// Any change to the levels drops the counts; a copy keeps them.
#[test]
fn a_change_to_the_levels_drops_the_counts() {
    let vtree = Arc::new(Vtree::balanced(6));
    let eng = Engine::new();
    let mut f = eng.clause(&vtree, [1, -2, 5]).unwrap();
    f = eng.and(f, eng.clause(&vtree, [2, 3, -6]).unwrap()).unwrap();
    eng.minimize(&mut f).unwrap();
    let f = counted(&eng, &f);
    assert!(f.clone().has_level_counts(), "a clone");
    assert!(f.try_clone_on(&eng).unwrap().has_level_counts(), "a copy");
    let changed = [
        ("condition", eng.condition_var(f.clone(), VarId(2), true).unwrap()),
        ("exists", eng.exists_vars(f.clone(), &[VarId(3)]).unwrap()),
        ("or", eng.or(f.clone(), eng.clause(&vtree, [4]).unwrap()).unwrap()),
    ];
    for (what, g) in changed {
        assert!(!g.has_level_counts() || kept_columns_are_folded_counts(&eng, &g, what) > 0, "{what}: stale counts");
    }
    let mut g = f.clone();
    g.levels[0].clear();
    assert!(!g.has_level_counts(), "a mutable access");
    let mut g = f.clone();
    eng.marginalize_levels(&mut g, &[vtree.children(vtree.root()).0]).unwrap();
    assert!(!g.has_level_counts(), "marginalize");
}
