//! A union or a sub-projection built from an operand's link is the
//! projection on the same variables, node for node and link for link.

use std::sync::Arc;

use super::*;
use crate::vtree::home::LeafSets;
use crate::vtree::rng::Lcg;

pub(super) fn assert_same(got: &Vtree, want: &Vtree, what: &str) {
    assert_eq!(got.nodes, want.nodes, "{what}");
    assert_eq!(got.root, want.root, "{what}");
    assert_eq!(got.leaf_count, want.leaf_count, "{what}");
    assert_eq!(got.topo.all(), want.topo.all(), "{what}");
    assert!(got.var_to_leaf.is_empty() && want.var_to_leaf.is_empty(), "{what}");
    assert!(Arc::ptr_eq(got.context(), want.context()), "{what}");
    let (a, b) = (got.home.as_deref().unwrap(), want.home.as_deref().unwrap());
    assert!(Arc::ptr_eq(a.home.vtree(), b.home.vtree()), "{what}");
    assert_eq!((&a.ranks, &a.node_of, &a.in_order, &a.regions), (&b.ranks, &b.node_of, &b.in_order, &b.regions), "{what}");
    assert_eq!(got.validate(), Ok(()), "{what}");
}

/// `count` distinct random variables of `1..=n`, or each with probability
/// one half when `count` is zero.
fn pick(n: u32, count: u32, rng: &mut Lcg) -> Vec<VarId> {
    if count == 0 {
        let some: Vec<VarId> = (1..=n).filter(|_| rng.coin()).map(VarId).collect();
        return if some.is_empty() { vec![VarId(1)] } else { some };
    }
    let mut chosen = vec![false; n as usize + 1];
    let mut vars = Vec::new();
    while (vars.len() as u32) < count.min(n) {
        let v = 1 + rng.below(u64::from(n)) as u32;
        if !std::mem::replace(&mut chosen[v as usize], true) {
            vars.push(VarId(v));
        }
    }
    vars
}

/// Homes from one leaf to a few thousand, so that the range minimum's
/// blocks and its sparse table are both crossed; sets dense and sparse.
fn cases(mut each: impl FnMut(&Home, Vec<VarId>, Vec<VarId>, &mut Lcg)) {
    for seed in 0..120u64 {
        let mut rng = Lcg::new(seed ^ 0x7072_6f6a);
        let n = match seed % 4 {
            0 => 1 + rng.below(20) as u32,
            1 => 20 + rng.below(80) as u32,
            2 => 100 + rng.below(400) as u32,
            _ => 2000 + rng.below(2000) as u32,
        };
        let home = Home::new(Vtree::random(n, seed));
        for _ in 0..4 {
            let count = match rng.below(3) {
                0 => 0,
                1 => 1 + rng.below(4) as u32,
                _ => 1 + rng.below(u64::from(n)) as u32,
            };
            let a = pick(n, count, &mut rng);
            let b = pick(n, if rng.coin() { count } else { 1 + rng.below(u64::from(n)) as u32 }, &mut rng);
            each(&home, a, b, &mut rng);
        }
    }
}

#[test]
fn a_union_is_the_projection_on_both() {
    let mut unions = 0;
    cases(|home, a, b, _| {
        let (pa, pb) = (Vtree::projection(home, a.iter().copied()).unwrap(), Vtree::projection(home, b.iter().copied()).unwrap());
        let union = Vtree::union(&pa, &pb).unwrap();
        let want = Vtree::projection(home, a.iter().chain(&b).copied()).unwrap();
        match Vtree::leaf_sets(&pa, &pb).unwrap() {
            LeafSets::Union(_) => {
                unions += 1;
                assert_same(&union, &want, "union");
            }
            LeafSets::Equal(_) | LeafSets::BInA => assert!(Arc::ptr_eq(&union, &pa)),
            LeafSets::AInB => assert!(Arc::ptr_eq(&union, &pb)),
        }
        assert!(Arc::ptr_eq(&Vtree::union(&pa, home.vtree()).unwrap(), home.vtree()));
    });
    assert!(unions > 300, "{unions} unions");
}

#[test]
fn a_sub_projection_is_the_projection_on_the_kept_leaves() {
    cases(|home, a, _, rng| {
        let own = Vtree::projection(home, a.iter().copied()).unwrap();
        for from in [&own, home.vtree()] {
            let keep: Vec<bool> = (0..from.num_nodes()).map(|_| rng.below(4) != 0).collect();
            let kept: Vec<VarId> = from.leaf_bottomup().filter(|&(t, _)| keep[t.idx()]).map(|(_, v)| v).collect();
            let sub = from.sub_projection(home, |t| keep[t.idx()]);
            if kept.is_empty() {
                assert!(sub.is_err());
            } else if kept.len() == from.num_leaves() as usize {
                assert!(Arc::ptr_eq(&sub.unwrap(), from));
            } else {
                assert_same(&sub.unwrap(), &Vtree::projection(home, kept).unwrap(), "sub-projection");
            }
        }
        let stranger = Home::new((**home.vtree()).clone());
        assert!(own.sub_projection(&stranger, |_| true).is_err());
    });
}

/// The lowest common ancestor by range minimum, on homes past one block of
/// gaps.
#[test]
fn node_over_is_the_lowest_common_ancestor_on_large_homes() {
    for seed in 0..12u64 {
        let n = 40 + 300 * seed as u32;
        let home = Home::new(Vtree::random(n, seed));
        let tree = home.vtree();
        let mut rng = Lcg::new(seed);
        for _ in 0..400 {
            let (a, b) = (rng.below(u64::from(n)) as u32, rng.below(u64::from(n)) as u32);
            let lca = tree.lca(home.node_over(a, a), home.node_over(b, b));
            assert_eq!(home.node_over(a, b), lca, "seed {seed}, ranks {a} {b}");
        }
    }
}
