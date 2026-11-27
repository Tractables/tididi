//! Diagrams on a weighted vtree, and the weight values a level holds.

use crate::vtree::Home;
use std::sync::Arc;

use crate::diagram::{Arithmetic, LiteralWeights, RationalWeights, Weights};

use crate::test_helpers::{exact_weight, rat};
use crate::vtree::Vtree;
use crate::{Engine, Tdd};

fn value_of(f: &Tdd) -> num_rational::BigRational {
    exact_weight(&f.weighted_value().unwrap().unwrap())
}

fn copy_levels(eng: &Engine, vtree: &Arc<Vtree>, f: &Tdd) -> crate::diagram::TddBuilder {
    let mut b = Tdd::builder(eng, vtree).unwrap();
    for t in vtree.bottomup() {
        b.replace_level(eng, t, f.level_view(t)).unwrap();
    }
    b
}

#[test]
fn a_weighted_leaf_requires_its_store() {
    let eng = Engine::new();
    let vtree = crate::vtree::Home::weighted(Vtree::balanced(1), Weights::new(RationalWeights::unit(1), Arithmetic::ExactRational)).unwrap();
    let mut f = Tdd::clause(&vtree, [1]).unwrap();
    crate::marginal::tests::marginalize_levels(&eng, &mut f, &[vtree.root()]).unwrap();
    crate::test_helpers::assert_canonical(&f);
    assert!(crate::diagram::builder::check_levels(&vtree, &f.levels, f.output(), None).is_err());
}

#[test]
fn copied_weighted_levels_keep_their_columns() {
    let eng = Engine::new();
    let vtree = weighted_vtree();
    let f = weighted_with_a_marginal_level(&eng, &vtree);
    crate::test_helpers::assert_canonical(&f);
    let g = copy_levels(&eng, &vtree, &f).finish(f.output()).unwrap();
    crate::test_helpers::assert_canonical(&g);
    assert_eq!(value_of(&g), value_of(&f));
}

fn weighted_vtree() -> Arc<Vtree> {
    let weights = vec![LiteralWeights { negative: rat(1, 2), positive: rat(1, 3) }; 4];
    Arc::new(
        Vtree::balanced(4)
            .with_weights(Weights::new(RationalWeights::from_literals(&weights), Arithmetic::ExactRational))
            .unwrap(),
    )
}

// Padded form: the padded sum-out.
/// A weighted diagram, with the level under the root's left child summed out.
fn weighted_with_a_marginal_level(eng: &Engine, vtree: &Arc<Vtree>) -> Tdd {
    let (left, _) = vtree.children(vtree.root());
    let mut f = Tdd::clause(&crate::vtree::Home::unchecked(vtree), [1, -2]).unwrap() & Tdd::clause(&crate::vtree::Home::unchecked(vtree), [2, 3]).unwrap();
    crate::marginal::tests::marginalize_levels(eng, &mut f, &[left]).unwrap();
    assert!(f.level(left).is_weight_marginal());
    f
}

#[test]
fn a_build_copying_a_weight_marginal_level_finishes_with_its_weights() {
    let eng = Engine::new();
    let vtree = weighted_vtree();
    let f = weighted_with_a_marginal_level(&eng, &vtree);

    let mut b = Tdd::builder(&eng, &vtree).unwrap();
    for (t, _, _) in vtree.internal_bottomup() {
        b.replace_level(&eng, t, f.level_view(t)).unwrap();
    }
    let g = b
        .finish(f.output())
        .expect("the vtree's weights interpret the copied level's values");
    assert_eq!(value_of(&g), value_of(&f));
}

#[test]
fn a_weighted_level_cannot_be_copied_without_its_weights() {
    let eng = Engine::new();
    let vtree = weighted_vtree();
    let f = weighted_with_a_marginal_level(&eng, &vtree);
    crate::test_helpers::assert_canonical(&f);
    let (left, _) = vtree.children(vtree.root());
    assert!(crate::diagram::LevelView::unweighted(f.level(left)).is_none());
}

/// On a home a summed-out node is a region holding the summed values, with
/// no empty level under it, whether it is the projection's root or under an
/// unsummed level.
#[test]
fn summed_out_regions_pass_the_level_check_in_every_arithmetic() {
    for arithmetic in [
        Arithmetic::ExactRational,
        Arithmetic::SignedLog,
        Arithmetic::ScaledFloat,
    ] {
        let eng = Engine::new();
        let home = Home::weighted(Vtree::balanced(8), Weights::new(RationalWeights::unit(8), arithmetic)).unwrap();
        for clause in [&[1, 2][..], &[1, 2, 3, 4]] {
            let mut f = Tdd::clause(&home, clause.iter().copied()).unwrap();
            eng.marginalize(&mut f, |v| v.0 <= 2).unwrap();
            crate::test_helpers::assert_canonical(&f);
            let vtree = f.vtree();
            let regions: Vec<_> = vtree.bottomup().filter(|&t| f.level(t).is_marginal()).collect();
            assert_eq!(regions.len(), 1, "{clause:?}");
            assert!(vtree.node(regions[0]).is_region() && f.level(regions[0]).slot_count() > 0, "{clause:?}");
            crate::diagram::builder::check_levels(vtree, &f.levels, f.output(), f.weights()).unwrap();
        }
    }
}

#[test]
fn a_column_of_another_arithmetic_fails_the_level_check() {
    let eng = Engine::new();
    let vtree = weighted_vtree();
    let f = weighted_with_a_marginal_level(&eng, &vtree);
    crate::test_helpers::assert_canonical(&f);
    let expected = value_of(&f);
    let (left, _) = vtree.children(vtree.root());
    let mut g = f.clone();
    g.levels[left.idx()].weight_values_mut().unwrap()[0] = crate::diagram::WeightValue::Log(crate::diagram::SignedLog::zero());
    assert!(crate::diagram::builder::check_levels(&vtree, &g.levels, g.output(), g.weights()).is_err());
    assert_eq!(value_of(&f), expected);
    crate::test_helpers::assert_canonical(&f);
}

#[test]
fn copied_pinned_leaf_columns_are_checked_in_both_arithmetics() {
    for arithmetic in [
        Arithmetic::ExactRational,
        Arithmetic::SignedLog,
        Arithmetic::ScaledFloat,
    ] {
        let eng = Engine::new();
        let weights = vec![LiteralWeights { negative: rat(1, 2), positive: rat(1, 3) }; 2];
        let home = Home::weighted(Vtree::balanced(2), Weights::new(RationalWeights::from_literals(&weights), arithmetic)).unwrap();
        let mut f = Tdd::clause(&home, [1, 2]).unwrap();
        let leaf = f.vtree().leaf_of(crate::vtree::VarId(1)).unwrap();
        crate::marginal::tests::marginalize_levels(&eng, &mut f, &[leaf]).unwrap();
        let tree = Arc::clone(f.vtree());
        crate::test_helpers::assert_canonical(&f);
        let copied = copy_levels(&eng, &tree, &f).finish(f.output()).unwrap();
        crate::test_helpers::assert_canonical(&copied);
        let mut malformed = f.clone();
        malformed.levels[leaf.idx()].weight_values_mut().unwrap().swap(0, 1);
        assert!(copy_levels(&eng, &tree, &malformed).finish(malformed.output()).is_err());
        crate::test_helpers::assert_canonical(&f);
    }
}

#[test]
fn negation_preserves_weights_for_structural_and_constant_results() {
    for n in [1, 4] {
        let tree = crate::vtree::Home::weighted(Vtree::balanced(n), Weights::new(RationalWeights::unit(n as usize), Arithmetic::ExactRational)).unwrap();
        let eng = Engine::new();
        for f in [
            Tdd::clause(&tree, [1]).unwrap(),
            Tdd::zero(&tree),
            Tdd::one(&tree),
        ] {
            crate::test_helpers::assert_canonical(&f);
            let before = f
                .weighted_value()
                .unwrap()
                .unwrap()
                .as_rational()
                .into_owned();
            let g = eng.negate(f).unwrap();
            crate::test_helpers::assert_canonical(&g);
            let after = eng
                .weighted_value(&g)
                .unwrap()
                .unwrap()
                .as_rational()
                .into_owned();
            assert_eq!(
                before + after,
                num_rational::BigRational::from_integer((1u64 << n).into())
            );
        }
    }
}

/// Both input layouts preserve variable order and named literal polarities.
#[test]
fn named_weight_layouts_evaluate_the_same_literals() {
    let rows = [
        LiteralWeights {
            negative: rat(-2, 5),
            positive: rat(3, 7),
        },
        LiteralWeights {
            negative: rat(5, 11),
            positive: rat(0, 1),
        },
    ];
    let from_rows = RationalWeights::from_literals(&rows);
    let from_columns = RationalWeights::from_polarities(LiteralWeights {
        negative: vec![rat(-2, 5), rat(5, 11)],
        positive: vec![rat(3, 7), rat(0, 1)],
    })
    .unwrap();
    assert_eq!(from_rows, from_columns);
    let tree = Home::new(Vtree::balanced(2));
    for (literals, expected) in [
        ([1, -2], rat(15, 77)),
        ([-1, -2], rat(-2, 11)),
        ([1, 2], rat(0, 1)),
    ] {
        let mut f =
            Tdd::clause(&tree, [literals[0]]).unwrap() & Tdd::clause(&tree, [literals[1]]).unwrap();
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
    let empty = RationalWeights::from_polarities(LiteralWeights {
        negative: vec![],
        positive: vec![],
    })
    .unwrap();
    assert_eq!(empty, RationalWeights::from_literals(&[]));
}

/// The false result of a conjunction of weighted operands, carried on by
/// `or` and `negate`, then conjoined again in both orders with a diagram
/// summed out at the same level.
#[test]
fn a_false_result_carries_no_column_into_a_later_conjunction() {
    let eng = Engine::new();
    let literals = [
        LiteralWeights { positive: rat(1, 3), negative: rat(1, 2) },
        LiteralWeights { positive: rat(1, 5), negative: rat(1, 7) },
        LiteralWeights { positive: rat(2, 3), negative: rat(1, 4) },
        LiteralWeights { positive: rat(3, 11), negative: rat(5, 13) },
    ];
    let weights = Weights::new(RationalWeights::from_literals(&literals), Arithmetic::ExactRational);
    let vtree = Home::weighted(Vtree::balanced(4), weights).unwrap();
    let a = Tdd::clause(&vtree, [3]).unwrap();
    let x12 = Tdd::clause(&vtree, [1]).unwrap() & Tdd::clause(&vtree, [2]).unwrap();
    let g = eng.and_marginalizing_where(a.clone(), x12, |v| v.0 <= 2).unwrap();
    let z = eng.and(Tdd::zero(&vtree), g).unwrap();
    assert!(z.is_zero());
    let h = eng.or(z.clone(), a).unwrap();
    let top = eng.negate(&z).unwrap();
    let mut k = Tdd::clause(&vtree, [1, 2]).unwrap();
    eng.marginalize(&mut k, |v| v.0 <= 2).unwrap();

    let total = |l: &LiteralWeights<num_rational::BigRational>| &l.positive + &l.negative;
    let [w1, w2, w3, w4] = &literals;
    let either = &w1.positive * &w2.positive + &w1.positive * &w2.negative + &w1.negative * &w2.positive;
    let with_a = &w3.positive * &either * total(w4);
    let with_top = &either * total(w3) * total(w4);
    let orders = [
        ("and(h,k)", eng.and(&h, &k).unwrap(), &with_a),
        ("and(k,h)", eng.and(&k, &h).unwrap(), &with_a),
        ("and(T,k)", eng.and(&top, &k).unwrap(), &with_top),
        ("and(k,T)", eng.and(&k, &top).unwrap(), &with_top),
    ];
    let wrong: Vec<_> = orders
        .into_iter()
        .filter_map(|(name, f, expected)| (value_of(&f) != *expected).then(|| format!("{name} = {}, expected {expected}", value_of(&f))))
        .collect();
    assert!(wrong.is_empty(), "{wrong:?}");
}

/// A copy of a weighted diagram reserves each level's values: every refused
/// reservation returns `OverBudget`, and the copy that is granted carries the
/// values.
#[test]
fn a_refused_copy_of_weight_values_is_over_budget() {
    let eng = Engine::new();
    let vtree = weighted_vtree();
    let f = weighted_with_a_marginal_level(&eng, &vtree);
    let (left, _) = vtree.children(vtree.root());
    let mut refused = 0;
    let copy = loop {
        eng.refuse_nth_reserve(refused);
        let result = eng.try_clone(&f);
        eng.limits().grant_every_reserve();
        match result {
            Ok(copy) => break copy,
            Err(e) => assert_eq!(e, crate::OperationError::OverBudget),
        }
        refused += 1;
    };
    assert!(refused > 0);
    assert_eq!(copy.level(left).weight_values().map(<[_]>::len), f.level(left).weight_values().map(<[_]>::len));
    assert_eq!(value_of(&copy), value_of(&f));
}
