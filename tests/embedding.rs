//! Placing one compiled circuit at several positions of a wider vtree.

use std::sync::Arc;

use tididi::restructure::GraftError;
use tididi::test_helpers::{assert_canonical, brute_force_count};
use tididi::vtree::{VarId, Vtree};
use tididi::{and, Engine, OperationError, Tdd};

/// `(xa ∨ xb) ∧ (¬xc ∨ xd)` on `vtree`, minimized.
fn fixture(vtree: &Arc<Vtree>, names: [i32; 4]) -> Tdd {
    let mut f = and(
        Tdd::clause(vtree, [names[0], names[1]]).unwrap(),
        Tdd::clause(vtree, [-names[2], names[3]]).unwrap(),
    )
    .unwrap();
    f.minimize().unwrap();
    assert_canonical(&f);
    f
}

#[test]
fn a_circuit_lands_on_a_subtree_of_the_destination() {
    let small = Arc::new(Vtree::balanced(4));
    let f = fixture(&small, [1, 2, 3, 4]);

    let image = Vtree::balanced_over(&[VarId(5), VarId(6), VarId(7), VarId(8)]).unwrap();
    let free: Vec<VarId> = [1, 2, 3, 4, 9, 10, 11, 12].iter().map(|&v| VarId(v)).collect();
    let big = Arc::new(Vtree::join(&image, &Vtree::balanced_over(&free).unwrap()).unwrap());
    assert_eq!(big.num_leaves(), 12);

    let (embedded, levels) = f.embed(&big, |v| VarId(v.0 + 4)).unwrap();
    assert_canonical(&embedded);
    assert!(embedded.equivalent(&fixture(&big, [5, 6, 7, 8])).unwrap());
    assert_eq!(embedded.model_count().unwrap(), (9u32 << 8).into());
    assert_eq!(levels.as_slice().len(), small.num_nodes());
}

#[test]
fn a_circuit_lands_on_leaves_spread_along_a_spine() {
    let small = Arc::new(Vtree::linear(4));
    let f = fixture(&small, [1, 2, 3, 4]);

    let big = Arc::new(Vtree::linear(10));
    let positions = [2u32, 4, 7, 9];
    let (embedded, levels) = f.embed(&big, |v| VarId(positions[v.idx()])).unwrap();
    assert_canonical(&embedded);
    assert!(embedded.equivalent(&fixture(&big, [2, 4, 7, 9])).unwrap());
    assert_eq!(embedded.model_count().unwrap(), (9u32 << 6).into());
    for (leaf, var) in small.leaf_bottomup() {
        assert_eq!(levels.level_of(leaf), big.leaf_of(VarId(positions[var.idx()])).unwrap());
    }
}

#[test]
fn one_relation_placed_twice_and_conjoined() {
    let triple = Arc::new(Vtree::linear(3));
    let mut r = Tdd::clause(&triple, [1, -2, 3]).unwrap();
    r.minimize().unwrap();

    let big = Arc::new(Vtree::linear(5));
    let (first, _) = r.embed(&big, |v| v).unwrap();
    let (second, _) = r.embed(&big, |v| VarId(v.0 + 2)).unwrap();
    assert_canonical(&first);
    assert_canonical(&second);

    let joined = and(first, second).unwrap();
    assert_canonical(&joined);
    assert_eq!(
        joined.model_count().unwrap(),
        brute_force_count(5, &[vec![1, -2, 3], vec![3, -4, 5]]).into(),
    );
}

#[test]
fn a_renaming_the_destination_cannot_carry_is_refused() {
    let small = Arc::new(Vtree::linear(2));
    let mut f = Tdd::clause(&small, [1, 2]).unwrap();
    f.minimize().unwrap();
    let big = Arc::new(Vtree::linear(4));

    // Two variables cannot share one destination leaf.
    assert!(matches!(f.embed(&big, |_| VarId(2)), Err(GraftError::Vtree(_))));
    // Nor can a variable land outside the destination.
    assert!(matches!(
        f.embed(&big, |v| VarId(v.0 + 4)),
        Err(GraftError::VariableOutOfRange { .. }),
    ));
    // Nor may the renaming cross the destination's leaf order.
    let reversed = |v: VarId| if v == VarId(1) { VarId(3) } else { VarId(1) };
    assert!(matches!(f.embed(&big, reversed), Err(GraftError::NotIsomorphic { .. })));

    // A level whose structure has been summed out cannot be copied.
    Engine::new().marginalize_levels(&mut f, &[small.root()]).unwrap();
    assert!(matches!(
        f.embed(&big, |v| v),
        Err(GraftError::Operation(OperationError::MarginalLevel(_))),
    ));
}
