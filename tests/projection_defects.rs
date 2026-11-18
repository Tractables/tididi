//! Defects of diagrams on projections of a home vtree, one failing test each.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Arc;

use num_bigint::BigUint;
use tididi::diagram::{ChildPair, NodeIdx, TddNodeId, NEG_LEAF_IDX, ONE_LEAF_IDX, POS_LEAF_IDX};
use tididi::vtree::{Home, VarId, Vtree};
use tididi::{Engine, Tdd};

/// `(x1 or x3 or x5) and (not x2 or x4 or not x5)` on the projection of a
/// linear home of eight variables on x1 to x5: 24 models of five variables,
/// 192 over the home.
fn two_clauses(home: &Home) -> Tdd {
    let own = Vtree::projection(home, [1, 2, 3, 4, 5].map(VarId)).unwrap();
    Engine::new().and(Tdd::clause(&own, [1, 3, 5]).unwrap(), Tdd::clause(&own, [-2, 4, -5]).unwrap()).unwrap()
}

#[test]
fn pair_index_of_true_on_no_leaf_answers() {
    let eng = Engine::new();
    let home = Home::new(Vtree::balanced(4));
    let one = Tdd::one(&Vtree::empty_projection(&home));
    let index = catch_unwind(AssertUnwindSafe(|| eng.pair_index(&one).map(drop)));
    assert!(index.is_ok(), "pair_index panics on the constant true with no leaf");
}

#[test]
fn a_rehome_refused_under_a_bound_keeps_the_count_over_the_home() {
    let eng = Engine::new();
    let home = Home::new(Vtree::linear(8));
    let other = Home::new(Vtree::balanced(8));
    let f = two_clauses(&home);
    assert_eq!(f.model_count().unwrap(), BigUint::from(192u32));
    for bound in 1..=64 {
        if let Err(refused) = eng.rehome(&f, &other, bound) {
            assert_eq!(
                refused.tdd.model_count().unwrap(),
                BigUint::from(192u32),
                "the diagram handed back under bound {bound}, on a projection: {}",
                refused.tdd.vtree().home().is_some(),
            );
        }
    }
}

#[test]
fn a_restructure_refused_under_a_bound_keeps_the_count_over_the_home() {
    let home = Home::new(Vtree::linear(8));
    let other = Home::new(Vtree::balanced(8));
    let target = Vtree::projection(&other, [1, 2, 3, 4, 5].map(VarId)).unwrap();
    for bound in 1..=64 {
        let mut f = two_clauses(&home);
        if f.restructure_to(&target, bound).is_err() {
            assert_eq!(f.model_count().unwrap(), BigUint::from(192u32), "after a refusal under bound {bound}");
        }
    }
}

#[test]
fn summing_out_the_root_follows_the_output_when_equal_counts_merge() {
    let eng = Engine::new();
    let home = Home::new(Vtree::balanced(2));
    let not_x1 = Tdd::clause(&Vtree::projection(&home, [VarId(1)]).unwrap(), [-1]).unwrap();
    let root = home.vtree().root();
    let summed = eng.and_marginalizing(Tdd::one(home.vtree()), &not_x1, &[root]).unwrap();
    let count = catch_unwind(AssertUnwindSafe(|| summed.model_count().unwrap()));
    assert_eq!(count.ok(), Some(BigUint::from(2u32)), "the count of not x1 over two variables, its root summed out");
}

#[test]
fn summing_out_a_root_of_several_nodes_follows_the_output() {
    // Three root nodes with 2, 2 and 3 models; the output is the second.
    let eng = Engine::new();
    let vtree = Arc::new(Vtree::balanced(2));
    let root = vtree.root();
    let mut b = Tdd::builder(&eng, &vtree).unwrap();
    b.push(&eng, root, &[ChildPair::new(POS_LEAF_IDX, ONE_LEAF_IDX)]).unwrap();
    let second = b.push(&eng, root, &[ChildPair::new(NEG_LEAF_IDX, ONE_LEAF_IDX)]).unwrap();
    b.push(&eng, root, &[ChildPair::new(POS_LEAF_IDX, ONE_LEAF_IDX), ChildPair::new(NEG_LEAF_IDX, NEG_LEAF_IDX)]).unwrap();
    let mut f = b.finish(TddNodeId { vtree: root, local: second }).unwrap();
    assert_eq!(f.model_count().unwrap(), BigUint::from(2u32));
    f.marginalize_levels(&[root]).unwrap();
    let count = catch_unwind(AssertUnwindSafe(|| f.model_count().unwrap()));
    assert_eq!(count.ok(), Some(BigUint::from(2u32)));
}

#[test]
fn trim_reads_every_pair_of_a_level_that_names_a_leaf_both_ways() {
    // `x1 or x2` as the pairs (any x1, x2) and (x1, not x2): the first names
    // the leaf of x1 as free, the second as a literal.
    let eng = Engine::new();
    let vtree = Arc::new(Vtree::balanced(2));
    let root = vtree.root();
    let mut b = Tdd::builder(&eng, &vtree).unwrap();
    let node = b.push(&eng, root, &[ChildPair::new(ONE_LEAF_IDX, POS_LEAF_IDX), ChildPair::new(POS_LEAF_IDX, NEG_LEAF_IDX)]).unwrap();
    let mut f = b.finish(TddNodeId { vtree: root, local: node }).unwrap();
    assert_eq!(f.model_count().unwrap(), BigUint::from(3u32));
    f.trim(&Home::new(Arc::clone(&vtree))).unwrap();
    assert_eq!(f.model_count().unwrap(), BigUint::from(3u32), "after trim, on {} leaves", f.vtree().num_leaves());
}

#[test]
fn finish_on_no_leaf_answers_or_refuses_an_output() {
    let eng = Engine::new();
    let home = Home::new(Vtree::balanced(2));
    let empty = Vtree::empty_projection(&home);
    for local in [NodeIdx(u32::MAX - 1), NodeIdx(0)] {
        let output = TddNodeId { vtree: empty.root(), local };
        let built = catch_unwind(AssertUnwindSafe(|| Tdd::builder(&eng, &empty).unwrap().finish(output).map(|f| f.model_count())));
        assert!(built.is_ok(), "finish panics on output {local:?} over a vtree with no node");
    }
}
