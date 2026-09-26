use std::sync::Arc;

use crate::Tdd;
use crate::restructure::EmbedError;
use crate::test_helpers::{assert_canonical, compile_clauses, test_cases, vtree_shapes};
use crate::vtree::{VarId, Vtree, VtreeIdx, VtreeNode};

/// `vtree`'s shape with every variable renamed through `rename` and the two
/// children of every node `swap` names exchanged.
fn mirrored_shape(
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
        for (name, vtree) in vtree_shapes(num_vars) {
            let f = compile_clauses(&vtree, &clauses);
            let rename = |v: VarId| VarId(num_vars + 1 - v.0);
            for pick in 0..3u32 {
                let swap = |t: VtreeIdx| match pick {
                    0 => t.0.is_multiple_of(2),
                    1 => !t.0.is_multiple_of(2),
                    _ => true,
                };
                let into = Arc::new(mirrored_shape(&vtree, rename, swap, num_vars));
                let (g, levels) = f.embed_mirrored(&into, rename).unwrap();
                assert_canonical(&g);
                let direct = compile_clauses(&into, &renamed(&clauses, rename));
                assert!(g.equivalent(&direct).unwrap(), "{name}, mirror set {pick}");
                assert_eq!(g.node_count(), direct.node_count(), "{name}, mirror set {pick}");
                // Every level is copied pair for pair.
                assert_eq!(g.pair_count(), f.pair_count(), "{name}, mirror set {pick}");
                for t in vtree.bottomup() {
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

#[test]
fn only_the_mirrored_embedding_accepts_swapped_children() {
    let vtree = Arc::new(Vtree::balanced(4));
    let f = compile_clauses(&vtree, &[vec![1, 2], vec![-3, 4]]);
    let into = Arc::new(mirrored_shape(&vtree, |v| v, |t| t == vtree.root(), 4));
    assert!(matches!(f.embed(&into, |v| v), Err(EmbedError::NotIsomorphic { .. })));
    let (g, _) = f.embed_mirrored(&into, |v| v).unwrap();
    assert_canonical(&g);
    assert!(g.equivalent(&compile_clauses(&into, &[vec![1, 2], vec![-3, 4]])).unwrap());
}

#[test]
fn a_mirrored_plan_places_like_the_one_shot_call() {
    let vtree = Arc::new(Vtree::random(5, 7));
    let clauses = [vec![1, -2], vec![2, 3, -4], vec![-1, 5]];
    let f = compile_clauses(&vtree, &clauses);
    let into = Arc::new(mirrored_shape(&vtree, |v| v, |t| t.0.is_multiple_of(3), 5));
    let plan = crate::restructure::EmbeddingPlan::new_mirrored(&vtree, &into, |v| v).unwrap();
    let placed: Tdd = plan.apply(&f).unwrap();
    let (once, _) = f.embed_mirrored(&into, |v| v).unwrap();
    assert!(placed.equivalent(&once).unwrap());
    assert_eq!(placed.node_count(), once.node_count());
}
