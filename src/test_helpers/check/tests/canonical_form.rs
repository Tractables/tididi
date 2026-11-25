//! `check_canonicity` accepts a structural diagram exactly when `minimize`
//! leaves it unchanged.


use crate::reduce::ReductionPlan;
use crate::diagram::{ChildPair, Dirty, Tdd, TddNodeId, NEG_LEAF_IDX, POS_LEAF_IDX};
use crate::test_helpers::{chain, normalized_levels, random_diagrams};
use crate::vtree::Vtree;
use crate::Engine;
use super::*;

/// `x1 x2 x3 (x4 x5 ∨ ¬x4 ¬x5)` on the right-linear vtree, built as two
/// chains joined only at the root: each level below it holds one node of
/// each chain, and the root names both tops with `x1`.
fn two_chains() -> Tdd {
    let eng = Engine::new();
    let vtree = std::sync::Arc::clone(&crate::vtree::Home::new(Vtree::linear(5)));
    let root = vtree.root();
    let a = vtree.children(root).1;
    let b = vtree.children(a).1;
    let c = vtree.children(b).1;
    let mut builder = Tdd::builder(&eng, &vtree).unwrap();
    let mut top = Vec::new();
    for (l, r) in [(POS_LEAF_IDX, POS_LEAF_IDX), (NEG_LEAF_IDX, NEG_LEAF_IDX)] {
        let n = builder.push(&eng, c, &[ChildPair::new(l, r)]).unwrap();
        let n = builder.push(&eng, b, &[ChildPair::new(POS_LEAF_IDX, n)]).unwrap();
        top.push(builder.push(&eng, a, &[ChildPair::new(POS_LEAF_IDX, n)]).unwrap());
    }
    let pairs: Vec<ChildPair> = top.iter().map(|&n| ChildPair::new(POS_LEAF_IDX, n)).collect();
    let out = builder.push(&eng, root, &pairs).unwrap();
    builder.finish(TddNodeId { vtree: root, local: out }).unwrap()
}

/// `f` minimized with every level owed to the contraction passes.
fn minimized(f: &Tdd) -> Tdd {
    let mut m = f.clone();
    m.levels.forget();
    let internal = f.vtree.internal_bottomup().map(|(t, _, _)| t);
    m.dirty = Dirty::default().seeded(&f.vtree, internal, None).unwrap();
    m.minimize().unwrap();
    m
}

#[test]
fn two_chains_joined_at_the_root_are_twins() {
    let f = two_chains();
    check_all_fast(&f, "two chains");
    check_determinism(&f).unwrap();
    let err = check_canonicity(&f, 3).expect_err("the two tops are named in the same place");
    assert!(err.contains("twins"), "{err}");
    let m = minimized(&f);
    assert!(m.node_count() < f.node_count());
    check_canonicity(&m, 3).unwrap();
}

#[test]
fn an_unreachable_node_is_not_canonical() {
    let (f, _) = chain();
    let err = check_canonicity(&f, 3).expect_err("chain keeps unreachable nodes");
    assert!(err.contains("not reachable"), "{err}");
    check_canonicity(&minimized(&f), 3).unwrap();
}

/// `x1 (x2 x3 ∨ ¬x2 x3)` on the right-linear vtree: the level over `x2`
/// names it through both literals with one partner.
#[test]
fn a_leaf_read_alike_through_both_literals_is_not_canonical() {
    let eng = Engine::new();
    let vtree = std::sync::Arc::clone(&crate::vtree::Home::new(Vtree::linear(3)));
    let root = vtree.root();
    let a = vtree.children(root).1;
    let mut builder = Tdd::builder(&eng, &vtree).unwrap();
    let n = builder
        .push(&eng, a, &[ChildPair::new(POS_LEAF_IDX, POS_LEAF_IDX), ChildPair::new(NEG_LEAF_IDX, POS_LEAF_IDX)])
        .unwrap();
    let out = builder.push(&eng, root, &[ChildPair::new(POS_LEAF_IDX, n)]).unwrap();
    let f = builder.finish(TddNodeId { vtree: root, local: out }).unwrap();
    check_all_fast(&f, "leaf twins");
    let err = check_canonicity(&f, 3).expect_err("x2 is read alike through both literals");
    assert!(err.contains("pairs every Pos"), "{err}");
    check_canonicity(&minimized(&f), 3).unwrap();
}

/// Over unminimized diagrams and their pruned copies, so that unreachable
/// nodes do not hide the twins behind them.
#[test]
fn the_check_accepts_exactly_what_minimize_leaves_unchanged() {
    let (mut kept, mut seen) = (0, [0; 3]);
    for f in random_diagrams(0x00c4_ec4c, 200, 3..8) {
        let mut pruned = f.clone();
        pruned.reduce(ReductionPlan::Prune).unwrap();
        for f in [f, pruned] {
            let unchanged = normalized_levels(&minimized(&f)) == normalized_levels(&f);
            let accepted = check_canonicity(&f, 3);
            assert_eq!(accepted.is_ok(), unchanged, "check says {accepted:?}, minimize changes it: {}", !unchanged);
            match accepted {
                Ok(()) => kept += 1,
                Err(e) => seen[["not reachable", "twins", "pairs every Pos"].iter().position(|r| e.contains(r)).unwrap()] += 1,
            }
        }
    }
    assert!(kept > 20 && seen.iter().all(|&n| n > 5), "{kept} kept; unreachable, twins, leaf twins: {seen:?}");
}
