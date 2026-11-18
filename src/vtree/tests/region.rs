use std::sync::Arc;

use super::*;
use crate::vtree::home::{LeafSets, Region};
use crate::vtree::REGION_VAR;
use crate::vtree::rng::Lcg;
use crate::OperationError;

fn vars(ids: &[u32]) -> Vec<VarId> {
    ids.iter().copied().map(VarId).collect()
}

/// The extent as ascending intervals of ranks, from ranks in any order.
fn intervals(mut ranks: Vec<u32>) -> Box<[(u32, u32)]> {
    ranks.sort_unstable();
    ranks.dedup();
    let mut extent: Vec<(u32, u32)> = Vec::new();
    for r in ranks {
        match extent.last_mut() {
            Some((_, end)) if *end == r => *end += 1,
            _ => extent.push((r, r + 1)),
        }
    }
    extent.into_boxed_slice()
}

/// The projection of `home` on the variable leaves `leaves` and one region
/// per entry of `regions`, its extent's variables and its anchor; not
/// validated, so that a test can see what validation refuses.
fn with_regions(home: &Home, leaves: &[u32], regions: &[(&[u32], u32)]) -> Vtree {
    let rank = |var: u32| home.rank(VarId(var)).expect("a variable of the home");
    let mut positions: Vec<(u32, Option<usize>)> = leaves.iter().map(|&var| (rank(var), None)).collect();
    positions.extend(regions.iter().enumerate().map(|(j, &(_, anchor))| (rank(anchor), Some(j))));
    positions.sort_unstable_by_key(|&(r, _)| r);
    let mut ranks = Vec::with_capacity(positions.len());
    let mut found = Vec::new();
    for (i, (r, region)) in positions.into_iter().enumerate() {
        ranks.push(r);
        if let Some(j) = region {
            let extent = intervals(regions[j].0.iter().map(|&var| rank(var)).collect());
            let count = extent.iter().map(|&(lo, hi)| hi - lo).sum();
            found.push(Region { at: i as u32, extent, count });
        }
    }
    Vtree::build_projection(home, ranks, found)
}

/// The region at `anchor`'s home leaf.
fn region_at(own: &Vtree, anchor: u32) -> VtreeIdx {
    let home = own.home().unwrap();
    let t = own.node_at_home(home.vtree().leaf_of(VarId(anchor)).unwrap()).unwrap();
    assert!(own.node(t).is_region());
    t
}

#[test]
fn a_projection_with_extents_holds() {
    let home = Home::new(Vtree::balanced(8));
    let own = with_regions(&home, &[1, 5], &[(&[2, 3, 4], 2), (&[7, 8], 7)]);
    assert_eq!(own.validate(), Ok(()));
    assert_eq!((own.num_leaves(), own.num_nodes()), (2, 7));
    for (i, node) in own.nodes.iter().enumerate() {
        assert_eq!((node.is_leaf(), node.is_region()), (i < 4, (2..4).contains(&i)), "node {i}");
    }
    assert_eq!(own.leaf_bottomup().map(|(_, var)| var).collect::<Vec<_>>().len(), 2);
    assert_eq!(own.internal_bottomup().len(), 3);
    assert!(own.leaf_of(VarId(1)).is_some() && own.leaf_of(VarId(2)).is_none() && own.leaf_of(VarId(3)).is_none());
    assert_eq!(own.free_leaves(), 1);
    assert_eq!(own.free_vars(), vars(&[6]));
    assert_eq!((1..=8).filter(|&v| own.home_has(VarId(v))).collect::<Vec<_>>(), [6]);
    let first = region_at(&own, 2);
    let h = home.node_over(home.rank(VarId(3)).unwrap(), home.rank(VarId(4)).unwrap());
    assert_eq!(own.own_at(h), Some(first));
    assert_eq!(own.own_at(home.vtree().leaf_of(VarId(4)).unwrap()), Some(first));
    assert_eq!(own.own_at(home.vtree().leaf_of(VarId(6)).unwrap()), None);
}

#[test]
#[cfg_attr(not(debug_assertions), ignore = "leaf_var refuses a region in debug builds only")]
#[should_panic(expected = "region")]
fn a_region_has_no_variable() {
    let home = Home::new(Vtree::balanced(4));
    let own = with_regions(&home, &[1], &[(&[3, 4], 3)]);
    own.leaf_var(region_at(&own, 3));
}

/// A region read as a leaf named `Pos` would later be read with `leaf_var`,
/// which has no variable to give.
#[test]
fn the_builder_refuses_pairs_at_a_region() {
    use crate::diagram::{ChildPair, POS_LEAF_IDX, TddBuildError, TddNodeId};
    let eng = crate::Engine::new();
    let home = Home::new(Vtree::balanced(8));
    let own = Arc::new(with_regions(&home, &[1, 5], &[(&[2, 3, 4], 2), (&[7, 8], 7)]));
    let regions = [region_at(&own, 2), region_at(&own, 7)];
    let mut b = crate::Tdd::builder(&eng, &own).unwrap();
    let mut below = Vec::new();
    for r in regions {
        let parent = own.node(r).parent().unwrap();
        below.push(b.push(&eng, parent, &[ChildPair::new(POS_LEAF_IDX, POS_LEAF_IDX)]).unwrap());
    }
    let root = b.push(&eng, own.root(), &[ChildPair::new(below[0], below[1])]).unwrap();
    match b.finish(TddNodeId { vtree: own.root(), local: root }) {
        Err(TddBuildError::StructuralRegion { level }) => assert!(regions.contains(&level)),
        other => panic!("pairs naming a region were not refused: {other:?}"),
    }
}

#[test]
fn validation_refuses_overlaps_and_a_stray_anchor() {
    let home = Home::new(Vtree::balanced(8));
    for (leaves, regions) in [
        (&[1, 3][..], &[(&[2, 3, 4][..], 2)][..]),
        (&[1], &[(&[2, 3], 2), (&[3, 4], 4)]),
        (&[1], &[(&[2, 3], 5)]),
    ] {
        assert!(matches!(with_regions(&home, leaves, regions).validate(), Err(VtreeError::Invalid(_))), "{leaves:?} {regions:?}");
    }
}

/// Home ((e1, x1), (x2, e2)) and its mirror. Anchored at e1, the region
/// keeps x1 as its sibling on both; anchored at the leftmost extent
/// variable of the mirror, it would take x2.
#[test]
fn an_anchor_that_is_not_leftmost_keeps_the_own_shape() {
    let (e1, x1, x2, e2) = (1, 2, 3, 4);
    let home = Home::new(Vtree::from_in_order(&vars(&[e1, x1, x2, e2]), &[2, 1, 2], 4).unwrap());
    let mirror = Home::new(Vtree::from_in_order(&vars(&[e2, x2, x1, e1]), &[2, 1, 2], 4).unwrap());
    let sibling = |home: &Home, anchor: u32| {
        let own = with_regions(home, &[x1, x2], &[(&[e1, e2], anchor)]);
        assert_eq!(own.validate(), Ok(()));
        own.leaf_var(own.sibling(region_at(&own, anchor)))
    };
    assert_eq!(sibling(&home, e1), VarId(x1));
    assert_eq!(sibling(&mirror, e1), VarId(x1));
    assert_eq!(sibling(&mirror, e2), VarId(x2));
}

#[test]
fn leaf_sets_refuse_overlaps_and_unite_the_rest() {
    let home = Home::new(Vtree::balanced(8));
    let a = Arc::new(with_regions(&home, &[1], &[(&[2, 3], 2)]));
    let region = region_at(&a, 2);
    for (leaves, regions) in [(&[3][..], &[][..]), (&[5], &[(&[3, 4][..], 4)]), (&[5], &[(&[2, 3], 2)])] {
        let b = Arc::new(with_regions(&home, leaves, regions));
        assert_eq!(Vtree::leaf_sets(&a, &b).err(), Some(OperationError::MarginalLevel(region)), "{leaves:?}");
    }
    assert_eq!(Vtree::leaf_sets(&a, home.vtree()).err(), Some(OperationError::MarginalLevel(region)));
    let fewer = Vtree::projection(&home, [VarId(1)]).unwrap();
    assert!(matches!(Vtree::leaf_sets(&a, &fewer), Ok(LeafSets::BInA)));
    assert!(matches!(Vtree::leaf_sets(&fewer, &a), Ok(LeafSets::AInB)));
    let b = Arc::new(with_regions(&home, &[1, 5], &[(&[7, 8], 8)]));
    let Ok(LeafSets::Union(union)) = Vtree::leaf_sets(&a, &b) else { panic!("a union") };
    assert_eq!(union.validate(), Ok(()));
    let mut free = union.free_vars();
    free.sort();
    assert_eq!((union.num_leaves(), free), (2, vars(&[4, 6])));
    let regions = &union.home.as_deref().unwrap().regions;
    assert_eq!(regions.len(), 2);
    region_at(&union, 2);
    region_at(&union, 8);
}

/// What each variable of the home is to a random projection.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Role {
    Leaf,
    Free,
    Region(usize),
}

fn random_roles(n: u32, rng: &mut Lcg) -> Vec<Role> {
    loop {
        let roles: Vec<Role> = (0..n)
            .map(|_| match rng.below(5) {
                0 => Role::Leaf,
                1 => Role::Free,
                j => Role::Region(j as usize - 2),
            })
            .collect();
        if roles.iter().any(|&role| role != Role::Free) {
            return roles;
        }
    }
}

/// The projection with `roles`, each region anchored at a random variable
/// of its extent.
fn build(home: &Home, roles: &[Role], rng: &mut Lcg) -> (Vtree, Vec<u32>) {
    let of = |want: Role| (1..=roles.len() as u32).filter(|&v| roles[v as usize - 1] == want).collect::<Vec<_>>();
    let leaves = of(Role::Leaf);
    let extents: Vec<Vec<u32>> = (0..3).map(|j| of(Role::Region(j))).filter(|e| !e.is_empty()).collect();
    let anchors: Vec<u32> = extents.iter().map(|e| e[rng.below(e.len() as u64) as usize]).collect();
    let regions: Vec<(&[u32], u32)> = extents.iter().zip(&anchors).map(|(e, &a)| (e.as_slice(), a)).collect();
    (with_regions(home, &leaves, &regions), anchors)
}

#[test]
fn random_regions_hold_and_count() {
    for seed in 0..60u64 {
        let mut rng = Lcg::new(seed);
        let n = 2 + rng.below(14) as u32;
        let home = Home::new(Vtree::random(n, seed));
        let roles = random_roles(n, &mut rng);
        let (own, anchors) = build(&home, &roles, &mut rng);
        assert_eq!(own.validate(), Ok(()), "seed {seed}");
        let leaves: Vec<VarId> = own.leaf_bottomup().map(|(_, var)| var).collect();
        assert_eq!(leaves.len(), own.num_leaves() as usize, "seed {seed}");
        assert!(leaves.iter().all(|&var| var != REGION_VAR && roles[var.idx()] == Role::Leaf), "seed {seed}");
        assert_eq!(own.nodes.iter().filter(|node| node.is_region()).count(), anchors.len(), "seed {seed}");
        let free: Vec<VarId> = {
            let mut free: Vec<u32> = (1..=n).filter(|&v| roles[v as usize - 1] == Role::Free).collect();
            free.sort_by_key(|&v| home.rank(VarId(v)));
            vars(&free)
        };
        assert_eq!(own.free_vars(), free);
        assert_eq!(own.free_leaves() as usize, free.len());
        for v in 1..=n {
            let role = roles[v as usize - 1];
            assert_eq!(own.home_has(VarId(v)), role == Role::Free);
            assert_eq!(own.leaf_of(VarId(v)).is_some(), role == Role::Leaf);
        }
        let region_of = |var: u32| match roles[var as usize - 1] {
            Role::Region(_) => anchors.iter().find(|&&a| roles[a as usize - 1] == roles[var as usize - 1]).map(|&a| region_at(&own, a)),
            _ => None,
        };
        for h in home.vtree().bottomup() {
            let under: Vec<u32> = home.rank_range(h).map(|r| home.vtree().leaf_var(home.index().leaf_at[r as usize]).0).collect();
            let one = region_of(under[0]).filter(|_| under.iter().all(|&v| roles[v as usize - 1] == roles[under[0] as usize - 1]));
            if one.is_some() {
                assert_eq!(own.own_at(h), one, "seed {seed}");
            }
        }
    }
}

#[test]
fn random_leaf_sets_refuse_exactly_the_overlaps() {
    for seed in 0..80u64 {
        let mut rng = Lcg::new(seed ^ 0x5eed);
        let n = 2 + rng.below(12) as u32;
        let home = Home::new(Vtree::random(n, seed));
        let (ra, rb) = (random_roles(n, &mut rng), random_roles(n, &mut rng));
        let (a, b) = (Arc::new(build(&home, &ra, &mut rng).0), Arc::new(build(&home, &rb, &mut rng).0));
        let used = |role: Role| role != Role::Free;
        let summed = |role: Role| matches!(role, Role::Region(_));
        let overlap = ra.iter().zip(&rb).any(|(&x, &y)| used(x) && used(y) && (summed(x) || summed(y)));
        match Vtree::leaf_sets(&a, &b) {
            Err(OperationError::MarginalLevel(t)) => {
                assert!(overlap, "seed {seed}");
                assert!(a.nodes.get(t.idx()).is_some_and(VtreeNode::is_region) || b.nodes.get(t.idx()).is_some_and(VtreeNode::is_region));
            }
            Err(other) => panic!("seed {seed}: {other:?}"),
            Ok(sets) => {
                assert!(!overlap, "seed {seed}");
                if let LeafSets::Union(union) = sets {
                    assert_eq!(union.validate(), Ok(()), "seed {seed}");
                    let neither = ra.iter().zip(&rb).filter(|&(&x, &y)| !used(x) && !used(y)).count();
                    assert_eq!(union.free_leaves() as usize, neither, "seed {seed}");
                }
            }
        }
    }
}

#[test]
fn a_sub_projection_keeps_the_regions() {
    for seed in 0..80u64 {
        let mut rng = Lcg::new(seed ^ 0x5b_9e0c);
        let n = 2 + rng.below(40) as u32;
        let home = Home::new(Vtree::random(n, seed));
        let roles = random_roles(n, &mut rng);
        let (own, anchors) = build(&home, &roles, &mut rng);
        let own = Arc::new(own);
        let keep: Vec<bool> = (0..own.num_nodes()).map(|_| rng.coin()).collect();
        let kept: Vec<u32> = own.leaf_bottomup().filter(|&(t, _)| keep[t.idx()]).map(|(_, v)| v.0).collect();
        let sub = own.sub_projection(&home, |t| keep[t.idx()]);
        if anchors.is_empty() && kept.is_empty() {
            assert!(sub.is_err(), "seed {seed}");
            continue;
        }
        let sub = sub.unwrap();
        if kept.len() == own.num_leaves() as usize {
            assert!(Arc::ptr_eq(&sub, &own), "seed {seed}");
            continue;
        }
        let extents: Vec<Vec<u32>> = (0..3)
            .map(|j| (1..=n).filter(|&v| roles[v as usize - 1] == Role::Region(j)).collect::<Vec<_>>())
            .filter(|e| !e.is_empty())
            .collect();
        let regions: Vec<(&[u32], u32)> = extents.iter().zip(&anchors).map(|(e, &a)| (e.as_slice(), a)).collect();
        super::projection::assert_same(&sub, &with_regions(&home, &kept, &regions), &format!("seed {seed}"));
    }
}
