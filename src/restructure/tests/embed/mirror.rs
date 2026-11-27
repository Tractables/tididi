use crate::test_helpers::{assert_canonical, compile_clauses, test_cases, vtree_shapes};
use crate::vtree::{Home, VarId, Vtree, VtreeIdx, VtreeNode};

/// `vtree`'s shape with every variable renamed through `rename` and the two
/// children of every node `swap` names exchanged.
pub(super) fn mirrored_shape(
    vtree: &Vtree,
    rename: impl Fn(VarId) -> VarId,
    swap: impl Fn(VtreeIdx) -> bool,
    num_vars: u32,
) -> Vtree {
    let mut nodes = Vec::with_capacity(vtree.num_nodes());
    let mut copy_of = vec![VtreeIdx(0); vtree.num_nodes()];
    for t in vtree.bottomup() {
        copy_of[t.idx()] = VtreeIdx(nodes.len() as u32);
        nodes.push(match *vtree.node(t) {
            VtreeNode::Leaf { var, .. } => VtreeNode::Leaf { var: rename(var), parent: None },
            VtreeNode::Internal { left, right, .. } => {
                let (left, right) = match swap(t) {
                    true => (right, left),
                    false => (left, right),
                };
                VtreeNode::Internal { left: copy_of[left.idx()], right: copy_of[right.idx()], parent: None }
            }
        });
    }
    Vtree::from_nodes(nodes, copy_of[vtree.root().idx()], num_vars)
        .expect("a mirrored copy of a tree is a tree")
}

fn renamed(clauses: &[Vec<i32>], rename: impl Fn(VarId) -> VarId) -> Vec<Vec<i32>> {
    clauses
        .iter()
        .map(|clause| {
            clause.iter().map(|&l| l.signum() * rename(VarId(l.unsigned_abs())).0 as i32).collect()
        })
        .collect()
}

#[test]
fn a_mirrored_copy_is_the_diagram_compiled_on_the_mirrored_tree() {
    for (num_vars, clauses) in test_cases() {
        for (name, home) in vtree_shapes(num_vars) {
            let f = compile_clauses(&home, &clauses);
            let rename = |v: VarId| VarId(num_vars + 1 - v.0);
            let vars: Vec<VarId> =
                f.vtree().bottomup().filter(|&t| f.vtree().node(t).is_leaf()).map(|t| rename(f.vtree().leaf_var(t))).collect();
            for pick in 0..3u32 {
                let swap = |t: VtreeIdx| match pick {
                    0 => t.0.is_multiple_of(2),
                    1 => !t.0.is_multiple_of(2),
                    _ => true,
                };
                let into = Home::new(mirrored_shape(&home, rename, swap, num_vars));
                // The mirror of f's projection is the projection of the
                // mirrored home on the renamed leaves.
                let on = match vars.is_empty() {
                    true => Vtree::empty_projection(&into),
                    false => Vtree::projection(&into, vars.iter().copied()).unwrap(),
                };
                let (g, levels) = crate::test_helpers::embedded(&f, &on, rename).unwrap();
                assert_canonical(&g);
                let direct = compile_clauses(&into, &renamed(&clauses, rename));
                assert!(g.equivalent(&direct).unwrap(), "{name}, mirror set {pick}");
                assert_eq!(g.node_count(), direct.node_count(), "{name}, mirror set {pick}");
                // Every level is copied pair for pair.
                assert_eq!(g.pair_count(), f.pair_count(), "{name}, mirror set {pick}");
                for t in f.vtree().bottomup() {
                    assert_eq!(
                        g.reference_slot_count(levels.level_of(t)),
                        f.reference_slot_count(t),
                        "{name}, mirror set {pick}",
                    );
                }
            }
        }
    }
}
