use std::sync::Arc;

use crate::{and, Tdd};
use crate::diagram::LEAF_WIDTH;
use crate::test_helpers::{assert_canonical, compile_clauses, test_cases, vtree_shapes};
use crate::vtree::{VarId, Vtree, VtreeIdx, VtreeNode};

/// `vtree`'s shape with every variable renamed through `rename`, over an id
/// space of `num_vars`.
fn renamed_shape(vtree: &Vtree, rename: impl Fn(VarId) -> VarId, num_vars: u32) -> Vtree {
    let mut nodes = Vec::with_capacity(vtree.num_nodes());
    let mut copy_of = vec![VtreeIdx(0); vtree.num_nodes()];
    for t in vtree.bottomup() {
        copy_of[t.idx()] = VtreeIdx(nodes.len() as u32);
        nodes.push(match *vtree.node(t) {
            VtreeNode::Leaf { var, .. } => VtreeNode::Leaf { var: rename(var), parent: None },
            VtreeNode::Internal { left, right, .. } => VtreeNode::Internal {
                left: copy_of[left.idx()],
                right: copy_of[right.idx()],
                parent: None,
            },
        });
    }
    Vtree::from_nodes(nodes, copy_of[vtree.root().idx()], num_vars)
        .expect("a renamed copy of a tree is a tree")
}

/// `(x1 ∨ x2) ∧ (¬x3 ∨ x4)` renamed through `names`, on `vtree`.
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

/// Minimizing the result changes nothing, which is what "canonical when the
/// source is" claims.
fn assert_at_fixpoint(f: &Tdd) {
    assert_canonical(f);
    let mut minimized = f.clone();
    minimized.minimize().unwrap();
    assert_eq!(minimized.node_count(), f.node_count());
}

#[test]
fn a_whole_subtree_image_matches_a_direct_build() {
    let small = Arc::new(Vtree::balanced(4));
    let f = fixture(&small, [1, 2, 3, 4]);

    let image = Vtree::balanced_over(&[VarId(5), VarId(6), VarId(7), VarId(8)]).unwrap();
    let free: Vec<VarId> = [1, 2, 3, 4, 9, 10, 11, 12].iter().map(|&v| VarId(v)).collect();
    let big = Arc::new(Vtree::join(&image, &Vtree::balanced_over(&free).unwrap()).unwrap());
    assert_eq!(big.num_leaves(), 12);

    let (embedded, levels) = f.embed(&big, |v| VarId(v.0 + 4)).unwrap();
    assert_at_fixpoint(&embedded);
    assert!(embedded.equivalent(&fixture(&big, [5, 6, 7, 8])).unwrap());
    // Nine models over four variables, times the eight variables left free.
    assert_eq!(embedded.model_count().unwrap(), (9u32 << 8).into());

    // The image is a subtree, so every source level keeps its width.
    for t in f.vtree().bottomup() {
        assert_eq!(
            embedded.reference_slot_count(levels.level_of(t)),
            f.reference_slot_count(t),
        );
    }
}

#[test]
fn leaves_spread_along_a_spine_match_a_direct_build() {
    let small = Arc::new(Vtree::linear(4));
    let f = fixture(&small, [1, 2, 3, 4]);

    let big = Arc::new(Vtree::linear(10));
    let positions = [2u32, 4, 7, 9];
    let (embedded, levels) = f.embed(&big, |v| VarId(positions[v.idx()])).unwrap();
    assert_at_fixpoint(&embedded);
    assert!(embedded.equivalent(&fixture(&big, [2, 4, 7, 9])).unwrap());
    // Nine models over four variables, times the six variables left free.
    assert_eq!(embedded.model_count().unwrap(), (9u32 << 6).into());

    for t in f.vtree().bottomup() {
        assert_eq!(
            embedded.reference_slot_count(levels.level_of(t)),
            f.reference_slot_count(t),
        );
    }
    // Every leaf landed on the leaf carrying its image.
    for (leaf, var) in f.vtree().leaf_bottomup() {
        assert_eq!(levels.level_of(leaf), big.leaf_of(VarId(positions[var.idx()])).unwrap());
        assert_eq!(embedded.reference_slot_count(levels.level_of(leaf)), LEAF_WIDTH);
    }
}

#[test]
fn two_placements_of_one_relation_conjoin() {
    let pair = Arc::new(Vtree::linear(2));
    let mut r = Tdd::clause(&pair, [1, 2]).unwrap();
    r.minimize().unwrap();

    let big = Arc::new(Vtree::linear(3));
    let (first, _) = r.embed(&big, |v| v).unwrap();
    let (second, _) = r.embed(&big, |v| VarId(v.0 + 1)).unwrap();
    assert_at_fixpoint(&first);
    assert_at_fixpoint(&second);

    let joined = and(first, second).unwrap();
    assert_canonical(&joined);
    assert_eq!(
        joined.model_count().unwrap(),
        crate::test_helpers::brute_force_count(3, &[vec![1, 2], vec![2, 3]]).into(),
    );
}

#[test]
fn a_false_diagram_embeds_to_false() {
    let small = Arc::new(Vtree::balanced(2));
    let f = Tdd::zero(&small);
    let big = Arc::new(Vtree::balanced(6));
    let (embedded, levels) = f.embed(&big, |v| VarId(v.0 + 2)).unwrap();
    assert_canonical(&embedded);
    assert!(embedded.is_zero());
    assert!(Arc::ptr_eq(embedded.vtree(), &big));
    // The correspondence is reported even though no level was copied.
    assert_eq!(levels.as_slice().len(), small.num_nodes());
}

#[test]
fn a_one_variable_diagram_embeds() {
    let small = Arc::new(Vtree::leaf(VarId(1)));
    let f = crate::literal(&small, -1).unwrap();
    assert_canonical(&f);
    let big = Arc::new(Vtree::linear(4));
    let (embedded, levels) = f.embed(&big, |_| VarId(3)).unwrap();
    assert_at_fixpoint(&embedded);
    assert!(embedded.equivalent(&crate::literal(&big, -3).unwrap()).unwrap());
    assert_eq!(levels.level_of(small.root()), big.leaf_of(VarId(3)).unwrap());
}

#[test]
fn a_constant_true_diagram_embeds() {
    let small = Arc::new(Vtree::balanced(3));
    let f = Tdd::one(&small);
    let big = Arc::new(Vtree::join(&Vtree::balanced(3), &Vtree::balanced_over(&[VarId(4), VarId(5)]).unwrap()).unwrap());
    let (embedded, _) = f.embed(&big, |v| v).unwrap();
    assert_at_fixpoint(&embedded);
    assert_eq!(embedded.model_count().unwrap(), 32u32.into());
}

#[test]
fn every_shape_embeds_beside_the_variables_it_leaves_free() {
    for (num_vars, clauses) in test_cases() {
        if num_vars > 4 {
            continue;
        }
        let shift = num_vars;
        let rename = |v: VarId| VarId(v.0 + shift);
        let renamed: Vec<Vec<i32>> =
            clauses.iter().map(|c| c.iter().map(|&l| l + l.signum() * shift as i32).collect()).collect();
        let free: Vec<VarId> = (1..=num_vars).map(VarId).collect();

        for (label, small) in vtree_shapes(num_vars) {
            let mut f = compile_clauses(&small, &clauses);
            f.minimize().unwrap();
            assert_canonical(&f);

            let image = renamed_shape(&small, rename, 2 * num_vars);
            let big = Arc::new(Vtree::join(&image, &Vtree::balanced_over(&free).unwrap()).unwrap());
            let (embedded, levels) = f.embed(&big, rename).unwrap();

            assert_at_fixpoint(&embedded);
            let expected = compile_clauses(&big, &renamed);
            assert!(embedded.equivalent(&expected).unwrap(), "{label}");
            assert_eq!(
                embedded.model_count().unwrap(),
                f.model_count().unwrap() << num_vars,
                "{label}",
            );
            for t in f.vtree().bottomup() {
                assert_eq!(
                    embedded.reference_slot_count(levels.level_of(t)),
                    f.reference_slot_count(t),
                    "{label}",
                );
            }
        }
    }
}
