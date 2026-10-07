//! Level counts kept with a diagram: each column is what the fold computes
//! again, a count that reads them is the count that folds, a conjunction
//! keeps exactly the columns of the levels it moved untouched, and a change
//! to the levels drops them.

use std::sync::Arc;

use super::*;
use super::relabel::{cases, function_of, vars_under};
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

/// `and_model_count` with the counts kept on either operand, both or
/// neither, on every vtree shape with one operand's support under each
/// subtree in turn: the same count, read from kept columns somewhere.
#[test]
fn a_count_that_reads_kept_counts_is_the_folded_count() {
    let eng = Engine::new();
    let before = kept_counts_census();
    for (what, f, g) in cases(9, 0xc0_17ed) {
        let expected = eng.and_model_count(f.clone(), g.clone(), &[]).unwrap();
        let (fk, gk) = (counted(&eng, &f), counted(&eng, &g));
        for (a, b) in [(&fk, &g), (&f, &gk), (&fk, &gk), (&gk, &fk)] {
            assert_eq!(eng.and_model_count(a.clone(), b.clone(), &[]).unwrap(), expected, "{what}");
        }
    }
    assert!(kept_counts_census() > before, "no count read a kept column");
}

/// The same with a target summed on the way, the levels under it made
/// marginal: a kept column is read only where the output's levels are the
/// operand's as counted.
#[test]
fn a_count_with_targets_reads_only_unchanged_levels() {
    let eng = Engine::new();
    for (what, f, g) in cases(8, 0x7a_47e7) {
        let vtree = Arc::clone(f.vtree());
        for (t, _, _) in vtree.internal_bottomup() {
            if t == vtree.root() {
                continue;
            }
            let expected = eng.and_model_count(f.clone(), g.clone(), &[t]).unwrap();
            let got = eng.and_model_count(counted(&eng, &f), counted(&eng, &g), &[t]).unwrap();
            assert_eq!(got, expected, "{what}, target {t:?}");
        }
    }
}

/// A conjunction keeps the columns of the levels an identity fast path moved
/// into it, each equal to the fold over the result, and nothing else.
#[test]
fn a_conjunction_keeps_the_moved_levels_columns() {
    let eng = Engine::new();
    let mut kept_any = false;
    for (what, f, g) in cases(9, 0x3c_a221) {
        for (a, b) in [(counted(&eng, &f), g.clone()), (f.clone(), counted(&eng, &g)), (counted(&eng, &f), counted(&eng, &g))] {
            let out = eng.and(a, b).unwrap();
            kept_any |= kept_columns_are_folded_counts(&eng, &out, &what) > 0;
            let plain = eng.and(f.clone(), g.clone()).unwrap();
            assert!(!plain.has_level_counts(), "{what}: counts from operands that kept none");
            assert_eq!(eng.model_count(&out).unwrap(), eng.model_count(&plain).unwrap(), "{what}");
        }
    }
    assert!(kept_any, "no conjunction kept a column");
}

/// A chain of conjunctions over one counted operand: the count of the second
/// reads the columns the first kept, whether it counts its root or builds it
/// and counts the result.
#[test]
fn a_chain_counts_through_kept_columns() {
    let mut rng = Lcg::new(0x000c_4a14);
    let eng = Engine::new();
    let before = kept_counts_census();
    for (shape, vtree) in vtree_shapes(10) {
        let internal: Vec<VtreeIdx> = vtree.internal_bottomup().map(|(t, _, _)| t).filter(|&t| t != vtree.root()).collect();
        for pair in internal.windows(2) {
            let (u, w) = (vars_under(&vtree, pair[0]), vars_under(&vtree, pair[1]));
            let all: Vec<u32> = (1..=10).filter(|v| !u.contains(v) || !w.contains(v)).collect();
            let f = function_of(&vtree, &all, &mut rng);
            let (d1, d2) = (function_of(&vtree, &u, &mut rng), function_of(&vtree, &w, &mut rng));
            let expected = eng.and_model_count(eng.and(f.clone(), d1.clone()).unwrap(), d2.clone(), &[]).unwrap();
            let first = eng.and(counted(&eng, &f), d1).unwrap();
            kept_columns_are_folded_counts(&eng, &first, shape);
            assert_eq!(eng.and_model_count(first, d2, &[]).unwrap(), expected, "{shape}");
        }
    }
    assert!(kept_counts_census() > before, "no count read a column a conjunction kept");
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
