use std::sync::Arc;

use super::*;
use crate::test_helpers::{rotate_left, rotate_right};
use crate::vtree::home::{HomeCell, LeafSets, Role};
use crate::vtree::rng::Lcg;

/// A random tree over `n` variables, rotated at random so that its node
/// indices are no longer in the order a construction leaves them.
fn rotated_home(n: u32, seed: u64) -> Arc<Vtree> {
    let mut vtree = Vtree::random(n, seed);
    let mut rng = Lcg::new(seed ^ 0x9e37_79b9);
    for _ in 0..n {
        let v = VtreeIdx(rng.below(vtree.num_nodes() as u64) as u32);
        if rng.coin() {
            rotate_left(&mut vtree, v);
        } else {
            rotate_right(&mut vtree, v);
        }
    }
    Arc::new(vtree)
}

fn subset(n: u32, rng: &mut Lcg) -> Vec<VarId> {
    let keep: Vec<VarId> = (1..=n).filter(|_| rng.coin()).map(VarId).collect();
    if keep.is_empty() { vec![VarId(1 + rng.below(n as u64) as u32)] } else { keep }
}

fn cases(mut each: impl FnMut(&Arc<Vtree>, &[VarId], &mut Lcg)) {
    for seed in 0..40u64 {
        let n = 1 + (seed % 20) as u32 * 3;
        let home = rotated_home(n, seed);
        let mut rng = Lcg::new(seed ^ 0x5bd1_e995);
        for _ in 0..6 {
            let vars = subset(n, &mut rng);
            each(&home, &vars, &mut rng);
        }
    }
}

/// The projection is the tree `project_to_vars` makes when it keeps the ids,
/// node for node, and it validates.
#[test]
fn projection_is_project_to_vars_keeping_ids() {
    cases(|home, vars, _| {
        let own = Vtree::projection(home, vars.iter().copied()).unwrap();
        let mut keep = vec![false; home.num_vars() as usize + 1];
        for v in vars {
            keep[v.0 as usize] = true;
        }
        let expected = home.project_to_vars(|v| keep[v.0 as usize].then_some(v), home.num_vars()).unwrap();
        assert!(own.same_tree(&expected));
        assert_eq!(own.nodes, expected.nodes);
        assert_eq!(own.validate(), Ok(()));
        assert!(Arc::ptr_eq(own.home().unwrap(), home));
        assert!(Arc::ptr_eq(own.context(), home.context()));
        assert_eq!(own.num_vars(), home.num_vars());
        assert!(own.var_to_leaf.is_empty());
    });
}

/// Each own node stands for the home node with the same leaves below it: a
/// leaf for its leaf, an internal node for the lowest common ancestor of its
/// two sides.
#[test]
fn home_node_is_the_ancestor_in_the_home() {
    cases(|home, vars, _| {
        let own = Vtree::projection(home, vars.iter().copied()).unwrap();
        for t in own.bottomup() {
            let h = own.home_node(t);
            match *own.node(t) {
                VtreeNode::Leaf { var, .. } => assert_eq!(home.leaf_of(var), Some(h)),
                VtreeNode::Internal { left, right, .. } => {
                    assert_eq!(home.lca(own.home_node(left), own.home_node(right)), h);
                }
            }
        }
    });
}

/// The index's ancestor of two leaves is [`Vtree::lca`]'s.
#[test]
fn index_ancestor_is_lca() {
    for seed in 0..30u64 {
        let n = 2 + (seed % 15) as u32 * 5;
        let home = rotated_home(n, seed);
        let index = home.home_index();
        let leaf_at = |r: u32| index.leaf_at[r as usize];
        for a in 0..n {
            for b in a + 1..n {
                let gap = index.ancestor_gap(a, b);
                assert_eq!(index.gap_node[gap as usize], home.lca(leaf_at(a), leaf_at(b)), "seed {seed}, ranks {a} {b}");
            }
        }
    }
}

#[test]
fn leaf_of_answers_through_the_home() {
    cases(|home, vars, _| {
        let own = Vtree::projection(home, vars.iter().copied()).unwrap();
        for v in 0..=home.num_vars() + 2 {
            let var = VarId(v);
            match own.leaf_of(var) {
                Some(t) => {
                    assert!(vars.contains(&var));
                    assert_eq!(own.leaf_var(t), var);
                }
                None => assert!(!vars.contains(&var), "variable {v}"),
            }
        }
    });
}

/// `node_at_home` inverts `home_node`, and names nothing else.
#[test]
fn node_at_home_inverts_home_node() {
    cases(|home, vars, _| {
        let own = Vtree::projection(home, vars.iter().copied()).unwrap();
        let mut expected = vec![None; home.num_nodes()];
        for t in own.bottomup() {
            expected[own.home_node(t).idx()] = Some(t);
        }
        for h in 0..home.num_nodes() + 2 {
            let h = VtreeIdx(h as u32);
            assert_eq!(own.node_at_home(h), expected.get(h.idx()).copied().flatten(), "home node {}", h.0);
            assert_eq!(home.node_at_home(h), (h.idx() < home.num_nodes()).then_some(h));
        }
    });
}

/// The layout is a function of the home and the leaf set: order and
/// repetition of the variables do not matter, nor does projecting from a
/// projection.
#[test]
fn equal_leaf_sets_give_equal_layouts() {
    cases(|home, vars, rng| {
        let own = Vtree::projection(home, vars.iter().copied()).unwrap();
        let mut shuffled: Vec<VarId> = vars.iter().chain(vars).copied().collect();
        for i in (1..shuffled.len()).rev() {
            shuffled.swap(i, rng.below(i as u64 + 1) as usize);
        }
        let again = Vtree::projection(home, shuffled).unwrap();
        assert_eq!(own.nodes, again.nodes);
        let fewer: Vec<VarId> = vars.iter().copied().filter(|_| rng.coin()).collect();
        if !fewer.is_empty() {
            let nested = Vtree::projection(&own, fewer.iter().copied()).unwrap();
            let direct = Vtree::projection(home, fewer).unwrap();
            assert_eq!(nested.nodes, direct.nodes);
            assert!(Arc::ptr_eq(nested.home().unwrap(), home));
            assert_eq!(nested.validate(), Ok(()));
            assert!(matches!(Vtree::leaf_sets(&nested, &own), Some(LeafSets::AInB | LeafSets::Equal)));
        }
    });
}

#[test]
fn leaf_sets_compare_and_unite() {
    cases(|home, vars, rng| {
        let other = subset(home.num_vars(), rng);
        let (a, b) = (Vtree::projection(home, vars.iter().copied()).unwrap(), Vtree::projection(home, other.iter().copied()).unwrap());
        let (in_a, in_b) = (|v: &VarId| vars.contains(v), |v: &VarId| other.contains(v));
        let got = Vtree::leaf_sets(&a, &b).unwrap();
        match got {
            LeafSets::Equal => assert_eq!(a.nodes, b.nodes),
            LeafSets::AInB => assert!(vars.iter().all(in_b) && !other.iter().all(in_a)),
            LeafSets::BInA => assert!(other.iter().all(in_a) && !vars.iter().all(in_b)),
            LeafSets::Union(ref union) => {
                assert!(!vars.iter().all(in_b) && !other.iter().all(in_a));
                let direct = Vtree::projection(home, vars.iter().chain(&other).copied()).unwrap();
                assert_eq!(union.nodes, direct.nodes);
                assert_eq!(union.validate(), Ok(()));
            }
        }
        if matches!(got, LeafSets::Equal) {
            assert_eq!(vars.len(), other.len());
        }
        assert!(matches!(Vtree::leaf_sets(&a, home), Some(LeafSets::AInB)));
        assert!(matches!(Vtree::leaf_sets(home, &b), Some(LeafSets::BInA)));
        assert!(matches!(Vtree::leaf_sets(home, home), Some(LeafSets::Equal)));
        let stranger = Arc::new((**home).clone());
        assert!(Vtree::leaf_sets(&a, &stranger).is_none());
        assert!(Vtree::leaf_sets(&stranger, home).is_none());
    });
}

#[test]
fn projection_refuses_no_variables_and_foreign_ones() {
    let home = Arc::new(Vtree::balanced_over(&[VarId(2), VarId(5), VarId(9)]).unwrap());
    assert!(matches!(Vtree::projection(&home, []), Err(VtreeError::Invalid(_))));
    for absent in [0, 1, 3, 10, u32::MAX] {
        assert!(matches!(Vtree::projection(&home, [VarId(2), VarId(absent)]), Err(VtreeError::Invalid(_))));
    }
    let own = Vtree::projection(&home, [VarId(5), VarId(9)]).unwrap();
    assert!(matches!(Vtree::projection(&own, [VarId(2)]), Err(VtreeError::Invalid(_))));
    let lone = Vtree::projection(&home, [VarId(9)]).unwrap();
    assert_eq!((lone.num_nodes(), lone.leaf_var(lone.root()), lone.num_vars()), (1, VarId(9), 9));
    assert_eq!(lone.home_node(lone.root()), home.leaf_of(VarId(9)).unwrap());
}

/// A rotation changes the shape the home link describes, so it makes the
/// projection a root vtree with the same leaves; a root's index is rebuilt.
#[test]
fn rotating_clears_the_home() {
    let home = rotated_home(12, 7);
    let own = Vtree::projection(&home, [1, 3, 4, 8, 11].map(VarId)).unwrap();
    let mut rotated = (*own).clone();
    let root = rotated.root();
    rotate_right(&mut rotated, root).or_else(|| rotate_left(&mut rotated, root)).unwrap();
    assert!(rotated.home().is_none());
    assert_eq!(rotated.validate(), Ok(()));
    assert_eq!(rotated.num_vars(), home.num_vars());
    for v in [1, 3, 4, 8, 11] {
        assert_eq!(rotated.leaf_var(rotated.leaf_of(VarId(v)).unwrap()), VarId(v));
    }
    let mut swapped = (*own).clone();
    swapped.swap_children(swapped.root());
    assert!(swapped.home().is_none());
    assert_eq!(swapped.validate(), Ok(()));

    let mut moved = (*home).clone();
    assert!(matches!(moved.home.role(), Some(Role::Home(_))));
    let root = moved.root();
    rotate_right(&mut moved, root).or_else(|| rotate_left(&mut moved, root)).unwrap();
    assert!(moved.home.role().is_none());
    let moved = Arc::new(moved);
    let again = Vtree::projection(&moved, [1, 3, 4, 8, 11].map(VarId)).unwrap();
    assert_eq!(again.validate(), Ok(()));
}

/// A broken link is refused.
#[test]
fn validate_checks_the_link() {
    let home = Arc::new(Vtree::balanced(8));
    let own = Vtree::projection(&home, [2, 3, 6].map(VarId)).unwrap();
    let mut broken = (*own).clone();
    let mut link = broken.home.link().unwrap().clone();
    link.node_of.swap(0, 1);
    broken.home = HomeCell::projection(link);
    assert!(matches!(broken.validate(), Err(VtreeError::Invalid(_))));
}
