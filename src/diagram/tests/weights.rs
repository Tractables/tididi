//! Attaching a weight store: refused once a level holds integer counts.

use std::sync::Arc;

use crate::diagram::{Arithmetic, LiteralWeights, RationalWeights, WeightStore};

use crate::test_helpers::{exact_weight, rat};
use crate::vtree::Vtree;
use crate::{Engine, Tdd};

#[test]
fn a_weighted_leaf_requires_its_store() {
    let eng = Engine::new();
    let vtree = Arc::new(Vtree::balanced(1));
    let mut f = Tdd::clause(&vtree, [1]).unwrap();
    f.set_weights(WeightStore::new(RationalWeights::unit(1), Arithmetic::ExactRational)).unwrap();
    eng.marginalize_levels(&mut f, &[vtree.root()]).unwrap();
    crate::test_helpers::assert_canonical(&f);
    assert!(crate::diagram::check_levels(&vtree, &f.levels, f.output(), None).is_err());
}

#[test]
fn weighted_levels_require_their_columns() {
    let eng = Engine::new();
    let vtree = Arc::new(Vtree::balanced(4));
    let f = weighted_with_a_marginal_level(&eng, &vtree);
    crate::test_helpers::assert_canonical(&f);
    let mut b = Tdd::builder(&eng, &vtree).unwrap();
    for t in vtree.bottomup() {
        b.replace_level(&eng, t, f.level_view(t)).unwrap();
    }
    assert!(b.set_weights(f.weights().unwrap().empty_like()).is_err());
    let g = b.finish(f.output()).unwrap();
    crate::test_helpers::assert_canonical(&g);
    assert_eq!(exact_weight(&g.weighted_value().unwrap().unwrap()), exact_weight(&f.weighted_value().unwrap().unwrap()));
}

#[test]
fn replacing_weights_cannot_erase_live_columns() {
    let eng = Engine::new();
    let vtree = Arc::new(Vtree::balanced(4));
    let mut f = weighted_with_a_marginal_level(&eng, &vtree);
    crate::test_helpers::assert_canonical(&f);
    let before = exact_weight(&f.weighted_value().unwrap().unwrap());
    let empty = f.weights().unwrap().empty_like();
    assert!(f.set_weights(empty).is_err());
    assert_eq!(exact_weight(&f.weighted_value().unwrap().unwrap()), before);
    crate::test_helpers::assert_canonical(&f);
}

#[test]
fn a_store_is_refused_after_a_level_was_summed_out_as_counts() {
    let eng = Engine::new();
    let vtree = Arc::new(Vtree::balanced(4));
    let (left, _) = vtree.children(vtree.root());
    let mut f = Tdd::clause(&vtree, [1, -2]).unwrap() & Tdd::clause(&vtree, [2, 3]).unwrap();
    eng.marginalize_levels(&mut f, &[left]).unwrap();
    let weights: Vec<_> = (0..4).map(|_| LiteralWeights { negative: rat(1, 2), positive: rat(1, 3) }).collect();
    assert!(f.set_weights(WeightStore::new(RationalWeights::from_literals(&weights), Arithmetic::ExactRational)).is_err());
}

/// A weighted diagram, with the level under the root's left child summed out.
fn weighted_with_a_marginal_level(eng: &Engine, vtree: &Arc<Vtree>) -> Tdd {
    let (left, _) = vtree.children(vtree.root());
    let mut f = Tdd::clause(vtree, [1, -2]).unwrap() & Tdd::clause(vtree, [2, 3]).unwrap();
    let weights: Vec<_> = (0..4).map(|_| LiteralWeights { negative: rat(1, 2), positive: rat(1, 3) }).collect();
    f.set_weights(WeightStore::new(RationalWeights::from_literals(&weights), Arithmetic::ExactRational)).unwrap();
    eng.marginalize_levels(&mut f, &[left]).unwrap();
    assert!(f.level(left).is_weight_marginal());
    f
}

#[test]
fn a_build_copying_a_weight_marginal_level_finishes_with_its_store() {
    let eng = Engine::new();
    let vtree = Arc::new(Vtree::balanced(4));
    let f = weighted_with_a_marginal_level(&eng, &vtree);

    let mut b = Tdd::builder(&eng, &vtree).unwrap();
    for (t, _, _) in vtree.internal_bottomup() {
        b.replace_level(&eng, t, f.level_view(t)).unwrap();
    }
    let g = b.finish(f.output()).expect("the store holds the copied level's values");
    let value = |t: &Tdd| exact_weight(&t.weighted_value().unwrap().expect("the diagram is weighted"));
    assert_eq!(value(&g), value(&f));
}

#[test]
fn a_weighted_level_cannot_be_copied_without_its_store() {
    let eng = Engine::new();
    let vtree = Arc::new(Vtree::balanced(4));
    let f = weighted_with_a_marginal_level(&eng, &vtree);
    crate::test_helpers::assert_canonical(&f);
    let (left, _) = vtree.children(vtree.root());
    assert!(crate::diagram::LevelView::unweighted(f.level(left)).is_none());
}

#[test]
fn subsumed_empty_columns_remain_valid_in_both_arithmetics() {
    for arithmetic in [Arithmetic::ExactRational, Arithmetic::SignedLog] {
        let eng = Engine::new();
        let vtree = Arc::new(Vtree::balanced(8));
        let mut f = Tdd::clause(&vtree, [1, 2]).unwrap();
        f.set_weights(WeightStore::new(RationalWeights::unit(8), arithmetic)).unwrap();
        eng.marginalize_levels(&mut f, &[vtree.root()]).unwrap();
        crate::test_helpers::assert_canonical(&f);
        let store = f.weights().unwrap().clone();
        assert!(vtree.internal_bottomup().any(|(t, _, _)| t != vtree.root() && f.level(t).slot_count() == 0));
        f.set_weights(store).unwrap();
        crate::test_helpers::assert_canonical(&f);
    }
}

#[test]
fn an_incompatible_column_does_not_replace_the_store() {
    let eng = Engine::new();
    let vtree = Arc::new(Vtree::balanced(4));
    let mut f = weighted_with_a_marginal_level(&eng, &vtree);
    crate::test_helpers::assert_canonical(&f);
    let expected = exact_weight(&f.weighted_value().unwrap().unwrap());
    let (left, _) = vtree.children(vtree.root());
    let mut store = f.weights().unwrap().clone();
    let col = store.level_vals_mut(left.idx()).unwrap();
    col[0] = crate::diagram::WeightValue::Log(crate::diagram::SignedLog::zero());
    assert!(f.set_weights(store).is_err());
    assert_eq!(exact_weight(&f.weighted_value().unwrap().unwrap()), expected);
    crate::test_helpers::assert_canonical(&f);
}

#[test]
fn a_weight_table_must_cover_the_vtree() {
    let vtree = Arc::new(Vtree::balanced(4));
    let mut f = Tdd::clause(&vtree, [1]).unwrap();
    crate::test_helpers::assert_canonical(&f);
    assert!(f.set_weights(WeightStore::new(RationalWeights::unit(1), Arithmetic::ExactRational)).is_err());
    assert!(f.weights().is_none());
    crate::test_helpers::assert_canonical(&f);
}

#[test]
fn copied_pinned_leaf_columns_are_checked_in_both_arithmetics() {
    for arithmetic in [Arithmetic::ExactRational, Arithmetic::SignedLog] {
        let eng = Engine::new();
        let tree = Arc::new(Vtree::balanced(2));
        let mut f = Tdd::clause(&tree, [1]).unwrap();
        f.set_weights(WeightStore::new(RationalWeights::from_literals(&vec![LiteralWeights { negative: rat(1, 2), positive: rat(1, 3) }; 2]), arithmetic)).unwrap();
        let leaf = tree.leaf_bottomup().next().unwrap().0;
        eng.marginalize_levels(&mut f, &[leaf]).unwrap();
        crate::test_helpers::assert_canonical(&f);
        let mut builder = Tdd::builder(&eng, &tree).unwrap();
        for t in tree.bottomup() { builder.replace_level(&eng, t, f.level_view(t)).unwrap(); }
        let copied = builder.finish(f.output()).unwrap();
        crate::test_helpers::assert_canonical(&copied);
        let mut malformed = f.weights().unwrap().clone();
        malformed.level_vals_mut(leaf.idx()).unwrap().swap(0, 1);
        assert!(f.set_weights(malformed).is_err());
        crate::test_helpers::assert_canonical(&f);
    }
}

#[test]
fn negation_preserves_weights_for_structural_and_constant_results() {
    for n in [1, 4] {
        let tree = Arc::new(Vtree::balanced(n));
        let eng = Engine::new();
        for f in [Tdd::clause(&tree, [1]).unwrap(), Tdd::zero(&tree), Tdd::one(&tree)] {
            let mut f = f;
            f.set_weights(WeightStore::new(RationalWeights::unit(n as usize), Arithmetic::ExactRational)).unwrap();
            crate::test_helpers::assert_canonical(&f);
            let before = f.weighted_value().unwrap().unwrap().as_rational().into_owned();
            let g = eng.negate(f).unwrap();
            crate::test_helpers::assert_canonical(&g);
            let after = eng.weighted_value(&g).unwrap().unwrap().as_rational().into_owned();
            assert_eq!(before + after, num_rational::BigRational::from_integer((1u64 << n).into()));
        }
    }
}

/// Both input layouts preserve variable order and named literal polarities.
#[test]
fn named_weight_layouts_evaluate_the_same_literals() {
    let rows = [
        LiteralWeights { negative: rat(-2, 5), positive: rat(3, 7) },
        LiteralWeights { negative: rat(5, 11), positive: rat(0, 1) },
    ];
    let from_rows = RationalWeights::from_literals(&rows);
    let from_columns = RationalWeights::from_polarities(LiteralWeights {
        negative: vec![rat(-2, 5), rat(5, 11)],
        positive: vec![rat(3, 7), rat(0, 1)],
    }).unwrap();
    assert_eq!(from_rows, from_columns);
    let tree = Arc::new(Vtree::balanced(2));
    for (literals, expected) in [
        ([1, -2], rat(15, 77)),
        ([-1, -2], rat(-2, 11)),
        ([1, 2], rat(0, 1)),
    ] {
        let mut f = Tdd::clause(&tree, [literals[0]]).unwrap() & Tdd::clause(&tree, [literals[1]]).unwrap();
        f.minimize().unwrap();
        crate::test_helpers::assert_canonical(&f);
        assert_eq!(f.evaluate(&from_rows).unwrap(), expected);
    }
}

/// Neither polarity may have a longer column than the other.
#[test]
fn weight_columns_reject_either_length_mismatch() {
    for (negative, positive) in [(vec![rat(1, 2)], vec![]), (vec![], vec![rat(1, 2)])] {
        assert!(RationalWeights::from_polarities(LiteralWeights { negative, positive }).is_none());
    }
    let empty = RationalWeights::from_polarities(LiteralWeights { negative: vec![], positive: vec![] }).unwrap();
    assert_eq!(empty, RationalWeights::from_literals(&[]));
}
