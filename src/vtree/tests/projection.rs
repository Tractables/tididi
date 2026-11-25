//! A union or a sub-projection built from an operand's link is the
//! projection on the same variables, node for node and link for link.

use std::sync::Arc;

use super::*;
use crate::vtree::home::LeafSets;
use crate::vtree::rng::Lcg;

pub(super) fn assert_same(got: &Vtree, want: &Vtree, what: &str) {
    if got.home.is_none() || want.home.is_none() {
        assert!(std::ptr::eq(got, want), "{what}: the projection on every leaf is the home");
        return;
    }
    assert_eq!(got.nodes, want.nodes, "{what}");
    assert_eq!(got.root, want.root, "{what}");
    assert_eq!(got.leaf_count, want.leaf_count, "{what}");
    assert_eq!(got.topo.all(), want.topo.all(), "{what}");
    assert!(got.var_to_leaf.is_empty() && want.var_to_leaf.is_empty(), "{what}");
    assert!(Arc::ptr_eq(got.context(), want.context()), "{what}");
    let (a, b) = (got.home.as_ref().unwrap(), want.home.as_ref().unwrap());
    assert!(Arc::ptr_eq(&a.home, &b.home), "{what}");
    assert_eq!((&a.slot, &a.regions), (&b.slot, &b.regions), "{what}");
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
        assert!(Arc::ptr_eq(&Vtree::union(&pa, home).unwrap(), home));
    });
    assert!(unions > 300, "{unions} unions");
}

/// Extend `own` by `added`, check the result against the projection on
/// both, and its map and fresh nodes against the home nodes they stand for.
fn check_extension(home: &Home, own: &Arc<Vtree>, added: &[VarId], what: &str) {
    let mut ranked: Vec<(u32, VarId)> = added.iter().map(|&v| (home.rank(v).unwrap(), v)).collect();
    ranked.sort_unstable();
    let mut map = vec![VtreeIdx(u32::MAX); own.num_nodes()];
    let mut fresh = Vec::new();
    let got = own.extended(&ranked, &mut map, |t, reading| fresh.push((t, reading)));
    let all: Vec<VarId> = own.leaf_bottomup().map(|(_, v)| v).chain(added.iter().copied()).collect();
    let want = Vtree::projection(home, all).unwrap();
    match want.home.is_some() {
        true => assert_same(&got, &want, what),
        false => assert!(got.same_tree(&want), "{what}: the full extension is the home's tree"),
    }
    let mut seen = vec![false; got.num_nodes()];
    for (s, &t) in map.iter().enumerate() {
        assert_eq!(got.home_node(t), own.home_node(VtreeIdx(s as u32)), "{what}: image of {s}");
        assert!(!std::mem::replace(&mut seen[t.idx()], true), "{what}: two nodes on {t:?}");
    }
    let own_under = |t: VtreeIdx| {
        let mut stack = vec![t];
        while let Some(u) = stack.pop() {
            match got.node(u) {
                VtreeNode::Leaf { var, .. } if own.leaf_of(*var).is_some() => return true,
                VtreeNode::Leaf { .. } => {}
                VtreeNode::Internal { left, right, .. } => stack.extend([*left, *right]),
            }
        }
        false
    };
    assert!(fresh.windows(2).all(|w| w[0].0 < w[1].0), "{what}: fresh in closing order");
    for &(t, reading) in &fresh {
        let (left, right) = got.children(t);
        match reading {
            Reading::Pass(c) => {
                let other = if c == left { right } else { left };
                assert!(c == left || c == right, "{what}");
                assert!(own_under(c) && !own_under(other), "{what}: pass at {t:?}");
            }
            Reading::Free => assert!(!own_under(t), "{what}: free at {t:?}"),
            Reading::Own(_) => panic!("{what}: an image among the fresh"),
        }
        assert!(!std::mem::replace(&mut seen[t.idx()], true), "{what}: {t:?} fresh and an image");
    }
    for t in got.bottomup() {
        assert!(seen[t.idx()] || got.node(t).is_leaf(), "{what}: {t:?} neither an image nor fresh");
    }
}

#[test]
fn an_extension_is_the_projection_on_both() {
    let mut rng = Lcg::new(0x6578_7465);
    let mut full = 0;
    for seed in 0..400u64 {
        let n = 1 + rng.below(12) as u32;
        let home = Home::new(Vtree::random(n, seed));
        let mut by_rank: Vec<VarId> = (1..=n).map(VarId).collect();
        by_rank.sort_unstable_by_key(|&v| home.rank(v).unwrap());
        let k = match seed % 5 {
            0 => 0,
            1 => 1.min(n - 1),
            _ => rng.below(u64::from(n)) as u32,
        };
        let mut own_vars = pick(n, k.max(1), &mut rng);
        if k == 0 {
            own_vars.clear();
        }
        let own = match own_vars.is_empty() {
            true => Vtree::empty_projection(&home),
            false => Vtree::projection(&home, own_vars.iter().copied()).unwrap(),
        };
        let rest: Vec<VarId> = by_rank.iter().copied().filter(|v| !own_vars.contains(v)).collect();
        if rest.is_empty() {
            continue;
        }
        for m in [1, 2, 5] {
            let mut added: Vec<VarId> = rest.clone();
            while added.len() > m {
                added.remove(rng.below(added.len() as u64) as usize);
            }
            full += usize::from(added.len() == rest.len());
            check_extension(&home, &own, &added, &format!("seed {seed}, k {k}, m {m}"));
        }
        // Before the first own rank and after the last.
        check_extension(&home, &own, &rest[..1], &format!("seed {seed}, first"));
        check_extension(&home, &own, &rest[rest.len() - 1..], &format!("seed {seed}, last"));
    }
    assert!(full > 50, "{full} full extensions");
}

#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "region")]
fn an_extension_with_a_region_is_refused() {
    let home = Home::new(Vtree::balanced(4));
    let mut g = crate::Tdd::clause(&crate::vtree::Home::unchecked(&Vtree::projection(&home, [1, 2, 3].map(VarId)).unwrap()), [1, 2, 3]).unwrap();
    g.marginalize(|v| v.0 <= 2).unwrap();
    assert!(g.vtree().has_regions());
    let rank = home.rank(VarId(4)).unwrap();
    let _ = g.vtree().extended(&[(rank, VarId(4))], &mut vec![VtreeIdx(0); g.vtree().num_nodes()], |_, _| {});
}

#[test]
fn a_sub_projection_is_the_projection_on_the_kept_leaves() {
    cases(|home, a, _, rng| {
        let own = Vtree::projection(home, a.iter().copied()).unwrap();
        for from in [&own, home] {
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
        let stranger = Home::new(Vtree::clone(home));
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
        let tree = &home;
        let mut rng = Lcg::new(seed);
        for _ in 0..400 {
            let (a, b) = (rng.below(u64::from(n)) as u32, rng.below(u64::from(n)) as u32);
            let lca = tree.lca(home.node_over(a, a), home.node_over(b, b));
            assert_eq!(home.node_over(a, b), lca, "seed {seed}, ranks {a} {b}");
        }
    }
}

#[test]
fn a_join_of_two_projections_joins_their_shapes() {
    for seed in 0..40u64 {
        let mut rng = Lcg::new(seed);
        let n = 4 + rng.below(9) as u32;
        let home = Home::new(Vtree::random(n, seed));
        let (mut left, mut right) = (Vec::new(), Vec::new());
        for v in 1..=n {
            match rng.below(3) {
                0 => left.push(VarId(v)),
                1 => right.push(VarId(v)),
                _ => {}
            }
        }
        if left.is_empty() || right.is_empty() {
            continue;
        }
        let (a, b) = (Vtree::projection(&home, left).unwrap(), Vtree::projection(&home, right).unwrap());
        let joined = Vtree::join(&a, &b).unwrap();
        assert!(joined.link_home().is_none(), "seed {seed}");
        assert_eq!(joined.validate(), Ok(()), "seed {seed}");
        let (ra, rb) = (a.renamed(|v| v, n).unwrap(), b.renamed(|v| v, n).unwrap());
        assert!(ra.link_home().is_none() && ra.same_tree(&a) && rb.same_tree(&b), "seed {seed}");
        assert!(joined.same_tree(&Vtree::join(&ra, &rb).unwrap()), "seed {seed}");
        let (jl, jr) = joined.children(joined.root());
        for (side, own) in [(jl, &a), (jr, &b)] {
            let under: Vec<VarId> = joined.leaf_bottomup().filter(|&(t, _)| joined.in_subtree(t, side)).map(|(_, v)| v).collect();
            let mut want: Vec<VarId> = own.leaf_bottomup().map(|(_, v)| v).collect();
            let mut got = under;
            got.sort();
            want.sort();
            assert_eq!(got, want, "seed {seed}");
        }
    }
}

#[test]
fn renaming_keeps_the_shape() {
    let home = Home::new(Vtree::balanced(8));
    let own = Vtree::projection(&home, [VarId(2), VarId(3), VarId(7)]).unwrap();
    let copy = own.renamed(|v| VarId(9 - v.0), 8).unwrap();
    assert!(copy.link_home().is_none());
    assert!(copy.same_tree(&Vtree::join(&Vtree::balanced_over(&[VarId(7), VarId(6)]).unwrap(), &Vtree::leaf(VarId(2))).unwrap()));
    assert!(matches!(own.renamed(|_| VarId(1), 8), Err(VtreeError::OverlappingVariable(VarId(1)))));
    assert!(own.renamed(|v| v, 6).is_err());
}
