//! The shape of a diagram after a sum-out at a home node, on two small
//! homes whose results are known node for node.

use std::sync::Arc;

use num_bigint::BigUint;

use super::region::{check_form, on_root_copy};
use crate::diagram::{ChildRef, LeafLabel, ValueRef};
use crate::vtree::{Home, VarId, Vtree, VtreeIdx};
use crate::{Engine, Tdd};

/// The home (((1,2),3),4).
fn left_linear_home() -> Home {
    Home::new(Vtree::from_text("vtree 7\nL 0 1\nL 1 2\nI 2 0 1\nL 3 3\nI 4 2 3\nL 5 4\nI 6 4 5\n").unwrap())
}

/// The disjunction of `cubes` on `vtree`, trimmed to `home`.
fn from_cubes(eng: &Engine, home: &Home, vtree: &Arc<Vtree>, cubes: &[&[i32]]) -> Tdd {
    let on = Home::unchecked(vtree);
    let mut f = Tdd::zero(&on);
    for cube in cubes {
        f = eng.or(f, eng.cube(&on, cube.iter().copied()).unwrap()).unwrap();
    }
    eng.trim(&mut f, home).unwrap();
    f
}

fn var_leaves(f: &Tdd) -> Vec<u32> {
    let vtree = f.vtree();
    let mut vars: Vec<u32> = vtree.bottomup().filter(|&t| vtree.node(t).is_var_leaf()).map(|t| vtree.leaf_var(t).0).collect();
    vars.sort_unstable();
    vars
}

/// The pairs of the root's one node, each side named by the count of a
/// region, the label of a variable leaf, or the index of an internal node.
fn root_pairs(f: &Tdd) -> Vec<(String, String)> {
    let vtree = f.vtree();
    let root = vtree.root();
    let (l, r) = vtree.children(root);
    let side = |child: VtreeIdx, side| {
        let level = f.level(child);
        match level.child_decoder().child(side) {
            ChildRef::Value(ValueRef::Inline(c)) => c.to_string(),
            ChildRef::Value(ValueRef::Slot(s)) => level.marginal_counts().unwrap()[s as usize].to_string(),
            ChildRef::Node(n) if vtree.node(child).is_leaf() => format!("{:?}", LeafLabel::from_idx(n.idx())),
            ChildRef::Node(n) => format!("n{}", n.idx()),
        }
    };
    assert_eq!(f.level(root).slot_count(), 1);
    let mut pairs: Vec<_> = f.level(root).pairs_vec(0).iter().map(|p| (side(l, p.left), side(r, p.right))).collect();
    pairs.sort();
    pairs
}

fn pairs(list: &[(&str, &str)]) -> Vec<(String, String)> {
    let mut v: Vec<_> = list.iter().map(|&(a, b)| (a.to_string(), b.to_string())).collect();
    v.sort();
    v
}

/// The model count of `f` with the variables of `cube` set, as a multiple of
/// the assignments to the home's other variables.
fn count_at(f: &Tdd, cube: &[(u32, bool)], free: u32) -> BigUint {
    let mut g = f.clone();
    for &(x, b) in cube {
        g = g.condition_var(VarId(x), b).unwrap();
    }
    g.model_count().unwrap() >> free
}

/// Summing out 1 and 2 at their home node keeps the classes over 3 that
/// differ in their residual over 4 apart: nine pairs. The same sum on a
/// function that does not depend on 3 leaves two.
#[test]
fn a_sum_out_keeps_classes_with_different_residuals_apart() {
    let eng = Engine::new();
    let home = left_linear_home();
    let f = &[
        &[-1, -2, 3][..],
        &[-1, 2, 3, 4],
        &[1, -2, 3, 4],
        &[1, 2, 3, -4],
        &[-1, -2, -3],
        &[-1, 2, -3],
        &[1, -2, -3, 4],
    ];
    let mut f = from_cubes(&eng, &home, &home, f);
    assert_eq!(f.model_count().unwrap(), BigUint::from(10u32));
    f.marginalize(|v| v.0 <= 2).unwrap();
    eng.minimize(&mut f).unwrap();
    check_form(&f, "f");
    assert_eq!((f.pair_count(), var_leaves(&f)), (9, vec![3, 4]));

    let projection = Vtree::projection(&home, [1, 2, 4].map(VarId)).unwrap();
    let mut g = from_cubes(&eng, &home, &projection, &[&[-1], &[1, -2, 4]]);
    g.marginalize(|v| v.0 <= 2).unwrap();
    eng.minimize(&mut g).unwrap();
    check_form(&g, "f'");
    assert_eq!((g.pair_count(), var_leaves(&g)), (2, vec![4]));

    assert_eq!(f.model_count().unwrap(), g.model_count().unwrap());
    for x in [false, true] {
        for (y, count) in [(true, 3u32), (false, 2)] {
            assert_eq!(count_at(&f, &[(3, x), (4, y)], 2), BigUint::from(count), "f at x={x} y={y}");
        }
    }
    for (y, count) in [(true, 3u32), (false, 2)] {
        assert_eq!(count_at(&g, &[(4, y)], 2), BigUint::from(count), "f' at y={y}");
    }
}

/// A sum-out at (1,2) on the projection on {1, 2, 4} merges at its own
/// parent; on the home, the pass-through level over 3 keeps two nodes.
#[test]
fn a_pass_through_level_keeps_apart_what_the_projection_merges() {
    let eng = Engine::new();
    let home = left_linear_home();
    let projection = Vtree::projection(&home, [1, 2, 4].map(VarId)).unwrap();
    let mut f = from_cubes(&eng, &home, &projection, &[&[1, 2], &[-1, 4]]);
    let home_vtree = Arc::clone(&home);
    let mut g = on_root_copy(&f, &home_vtree);
    let summed = home_vtree.node(home_vtree.leaf_of(VarId(1)).unwrap()).parent().unwrap();

    f.marginalize(|v| v.0 <= 2).unwrap();
    check_form(&f, "projection");
    assert_eq!(root_pairs(&f), pairs(&[("1", "Neg"), ("3", "Pos")]));

    crate::test_helpers::marginalize_under(&eng, &mut g, &[summed]).unwrap();
    let middle = home_vtree.node(summed).parent().unwrap();
    let root = g.vtree().root();
    assert_eq!((g.level(middle).slot_count(), g.level(root).pair_count_at(0)), (2, 3));

    assert_eq!(f.model_count().unwrap(), BigUint::from(8u32));
    assert_eq!(g.model_count().unwrap(), BigUint::from(8u32));
}
