use std::sync::Arc;

use super::*;
use crate::vtree::home::{LeafSets, Region};
use crate::vtree::REGION_VAR;
use crate::vtree::rng::Lcg;
use crate::OperationError;

fn vars(ids: &[u32]) -> Vec<VarId> {
    ids.iter().copied().map(VarId).collect()
}

/// The variables of the home leaves `own` has neither as a leaf nor in an
/// extent, by a walk over the home's leaves: the reference for
/// [`Vtree::free_set`].
fn free_vars(own: &Vtree) -> Vec<VarId> {
    let Some(link) = own.home.as_ref() else { return Vec::new() };
    let mut ranks = link.ranks().iter().copied().peekable();
    let mut extents: Vec<(u32, u32)> = link.regions.iter().flat_map(|region| region.extent.iter().copied()).collect();
    extents.sort_unstable();
    let mut extents = extents.into_iter().peekable();
    let mut free = Vec::new();
    for (rank, &leaf) in link.home.index().leaf_at.iter().enumerate() {
        let rank = rank as u32;
        while extents.next_if(|&(_, end)| end <= rank).is_some() {}
        let in_extent = extents.peek().is_some_and(|&(start, _)| start <= rank);
        if ranks.next_if_eq(&rank).is_none() && !in_extent {
            free.push(link.home.vtree().leaf_var(leaf));
        }
    }
    free
}

/// The variables of the home leaves under own node `t`'s home node that
/// `own` does not have as a leaf, right to left; none when every variable
/// outside the extents is a leaf: the reference for [`Vtree::free_set_under`].
fn free_vars_under(own: &Vtree, t: VtreeIdx) -> Vec<VarId> {
    let Some(home) = own.link_home().map(Home::vtree).filter(|_| own.free_leaves() > 0) else { return Vec::new() };
    home.subtree(own.home_node(t))
        .filter(|&h| home.node(h).is_leaf())
        .map(|h| home.leaf_var(h))
        .filter(|&var| own.leaf_of(var).is_none())
        .collect()
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
    Vtree::linked(home, Vtree::build_ranks(home, &ranks, &found), found)
}

/// The region at `anchor`'s home leaf.
fn region_at(own: &Vtree, anchor: u32) -> VtreeIdx {
    let home = own.link_home().unwrap();
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
    assert_eq!(free_vars(&own), vars(&[6]));
    assert_eq!((1..=8).filter(|&v| own.home_has(VarId(v))).collect::<Vec<_>>(), [6]);
    let first = region_at(&own, 2);
    let h = home.node_over(home.rank(VarId(3)).unwrap(), home.rank(VarId(4)).unwrap());
    assert_eq!(own.own_at(h), None, "ranks of the extent without its anchor");
    assert_eq!(own.own_at(home.vtree().leaf_of(VarId(2)).unwrap()), Some(first));
    assert_eq!(own.own_at(home.vtree().leaf_of(VarId(4)).unwrap()), None);
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
    let mut free = free_vars(&union);
    free.sort();
    assert_eq!((union.num_leaves(), free), (2, vars(&[4, 6])));
    let regions = &union.home.as_ref().unwrap().regions;
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
        assert_eq!(free_vars(&own), free);
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
            let anchored = |a: u32| under.contains(&a);
            let one = region_of(under[0]).filter(|_| under.iter().all(|&v| roles[v as usize - 1] == roles[under[0] as usize - 1]))
                .filter(|_| anchors.iter().any(|&a| roles[a as usize - 1] == roles[under[0] as usize - 1] && anchored(a)));
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

#[test]
fn renaming_refuses_a_region_leaf() {
    let home = Home::new(Vtree::balanced(6));
    let own = with_regions(&home, &[1, 4], &[(&[2, 3], 2)]);
    assert!(matches!(own.renamed(|v| v, 6), Err(VtreeError::Invalid(_))));
    assert!(matches!(Vtree::empty_projection(&home).renamed(|v| v, 6), Err(VtreeError::Invalid(_))));
}

mod free_set {
    use num_bigint::BigInt;
    use num_rational::BigRational;

    use super::*;
    use crate::diagram::{EvalAlgebra, LeafLabel, LiteralWeights, RationalWeights};
    use crate::vtree::FreeVars;

    /// Weights read from a table, rounding at every product.
    struct Floats(Vec<f64>);

    impl EvalAlgebra for Floats {
        type Value = f64;
        fn zero(&self) -> f64 { 0.0 }
        fn one(&self) -> f64 { 1.0 }
        fn leaf(&self, var: VarId, label: LeafLabel) -> f64 {
            let pos = self.0[var.idx()];
            match label {
                LeafLabel::Pos => pos,
                LeafLabel::Neg => 1.0 - pos / 3.0,
                LeafLabel::One => pos + (1.0 - pos / 3.0),
                LeafLabel::Zero => 0.0,
            }
        }
        fn add_assign(&self, acc: &mut f64, other: &f64) { *acc += other; }
        fn mul(&self, a: &f64, b: &f64) -> f64 { a * b }
    }

    /// Model counts that wrap.
    struct Wrapping;

    impl EvalAlgebra for Wrapping {
        type Value = u64;
        fn zero(&self) -> u64 { 0 }
        fn one(&self) -> u64 { 1 }
        fn leaf(&self, var: VarId, label: LeafLabel) -> u64 {
            match label {
                LeafLabel::One => 3 + u64::from(var.0),
                LeafLabel::Zero => 0,
                _ => 1,
            }
        }
        fn add_assign(&self, acc: &mut u64, other: &u64) { *acc = acc.wrapping_add(*other); }
        fn mul(&self, a: &u64, b: &u64) -> u64 { a.wrapping_mul(*b) }
    }

    /// The free factor as the library computed it from the materialized set.
    fn listed<A: EvalAlgebra>(algebra: &A, vars: &[VarId]) -> Option<A::Value> {
        let (&first, rest) = vars.split_first()?;
        let one = |var| algebra.leaf(var, LeafLabel::One);
        Some(rest.iter().fold(one(first), |acc, &var| algebra.mul(&acc, &one(var))))
    }

    fn check(set: Option<FreeVars<'_>>, list: &[VarId], algebras: (&Floats, &RationalWeights), at: &str) {
        let Some(set) = set else {
            assert!(list.is_empty(), "{at}: no set, {} listed", list.len());
            return;
        };
        assert_eq!(set.vars().collect::<Vec<_>>(), list, "{at}");
        assert_eq!(set.len() as usize, list.len(), "{at}");
        let intervals: Vec<_> = set.intervals().collect();
        assert!(intervals.iter().all(|r| !r.is_empty()), "{at}");
        assert!(intervals.windows(2).all(|w| w[0].end < w[1].start), "{at}");
        let (floats, rationals) = algebras;
        assert_eq!(floats.free(&set).map(f64::to_bits), listed(floats, list).map(f64::to_bits), "{at}");
        assert_eq!(rationals.free(&set), listed(rationals, list), "{at}");
        assert_eq!(Wrapping.free(&set), listed(&Wrapping, list), "{at}");
    }

    #[test]
    fn the_interval_form_gives_the_values_of_the_listed_set() {
        for seed in 0..400u64 {
            let mut rng = Lcg::new(seed);
            let n = 1 + rng.below(64) as u32;
            let home = Home::new(match seed % 3 {
                0 => Vtree::random(n, seed),
                1 => Vtree::balanced(n),
                _ => Vtree::linear(n),
            });
            let roles: Vec<Role> = match seed % 2 {
                0 if n >= 2 => random_roles(n, &mut rng),
                _ => (0..n).map(|_| if rng.below(3) == 0 { Role::Leaf } else { Role::Free }).collect(),
            };
            let own = match roles.iter().all(|&role| role == Role::Free) {
                true => (*Vtree::empty_projection(&home)).clone(),
                false => build(&home, &roles, &mut rng).0,
            };
            let floats = Floats((0..n).map(|_| (1 + rng.below(1000)) as f64 / 997.0).collect());
            let mut rational = || BigRational::new(BigInt::from(1 + rng.below(9)), BigInt::from(1 + rng.below(7)));
            let literals: Vec<LiteralWeights<BigRational>> =
                (0..n).map(|_| LiteralWeights { negative: rational(), positive: rational() }).collect();
            let rationals = RationalWeights::from_literals(&literals);
            let algebras = (&floats, &rationals);
            let free = free_vars(&own);
            check(own.free_set(), &free, algebras, &format!("seed {seed}"));
            // The listed set under a node came right to left; the interval form is in rank order.
            for t in own.bottomup() {
                let mut under = free_vars_under(&own, t);
                under.sort_by_key(|&var| home.rank(var));
                check(own.free_set_under(t), &under, algebras, &format!("seed {seed} under {}", t.0));
            }
            let pinned: Vec<VarId> = (1..=n).map(VarId).filter(|_| rng.below(4) == 0).collect();
            let pins = own.free_ranks(pinned.iter().copied());
            let unpinned: Vec<VarId> = free.iter().copied().filter(|v| !pinned.contains(v)).collect();
            let set = own.free_set().map(|set| set.without(&pins)).filter(|set| !set.is_empty());
            check(set, &unpinned, algebras, &format!("seed {seed} pinned"));
        }
    }
}

/// A projection by definition: an own node for each home node with a kept
/// home leaf under each child.
enum Def {
    Childless(VtreeIdx),
    Node(VtreeIdx, Box<Def>, Box<Def>),
}

fn by_definition(home: &Vtree, h: VtreeIdx, kept: &[bool]) -> Option<Def> {
    match *home.node(h) {
        VtreeNode::Leaf { .. } => kept[h.idx()].then_some(Def::Childless(h)),
        VtreeNode::Internal { left, right, .. } => {
            match (by_definition(home, left, kept), by_definition(home, right, kept)) {
                (Some(l), Some(r)) => Some(Def::Node(h, Box::new(l), Box::new(r))),
                (l, r) => l.or(r),
            }
        }
    }
}

/// Whether `own` is the projection of its home on the home leaves `kept`,
/// node for node, each own node standing for its home node and found from
/// it.
fn is_by_definition(own: &Vtree, kept: &[bool], what: &str) {
    assert_eq!(own.validate(), Ok(()), "{what}");
    let home = own.link_home().expect("a projection").vtree();
    let def = by_definition(home, home.root(), kept).expect("a kept leaf");
    let mut stack = vec![(own.root(), &def)];
    while let Some((t, def)) = stack.pop() {
        let h = match def {
            Def::Childless(h) => {
                assert!(own.node(t).is_leaf(), "{what}: node {} is not childless", t.0);
                assert!(own.node(t).is_region() || own.leaf_var(t) == home.leaf_var(*h), "{what}: leaf {}", t.0);
                *h
            }
            Def::Node(h, l, r) => {
                let (a, b) = own.children(t);
                stack.extend([(a, &**l), (b, &**r)]);
                *h
            }
        };
        assert_eq!(own.home_node(t), h, "{what}: node {}", t.0);
        assert_eq!(own.node_at_home(h), Some(t), "{what}: home node {}", h.0);
    }
}

/// The home leaves of `ranks` as a table by home node.
fn kept_of(home: &Home, ranks: impl IntoIterator<Item = u32>) -> Vec<bool> {
    let mut kept = vec![false; home.vtree().num_nodes()];
    for r in ranks {
        kept[home.index().leaf_at[r as usize].idx()] = true;
    }
    kept
}

/// The four producers, `projection`, `union`, `sub_projection` and
/// `from_in_order`, and the cold builder of regions, against the projection
/// by definition on random homes of up to 12 leaves.
#[test]
fn the_producers_build_the_projection_by_definition() {
    // Unions, sub-projections, sub-projections with regions, unions with
    // regions.
    let mut seen = [0; 4];
    for seed in 0..400u64 {
        let mut rng = Lcg::new(seed ^ 0x6c69_6e6b);
        let n = 1 + rng.below(12) as u32;
        let home = Home::new(Vtree::random(n, seed));
        let index = home.index();
        let pick = |rng: &mut Lcg| -> Vec<u32> {
            let ranks: Vec<u32> = (0..n).filter(|_| rng.coin()).collect();
            if ranks.is_empty() { vec![rng.below(u64::from(n)) as u32] } else { ranks }
        };
        let var = |r: u32| index.var_at[r as usize];
        let (ra, rb) = (pick(&mut rng), pick(&mut rng));
        let pa = Vtree::projection(&home, ra.iter().rev().map(|&r| var(r))).unwrap();
        let pb = Vtree::projection(&home, rb.iter().map(|&r| var(r))).unwrap();
        for (p, ranks) in [(&pa, &ra), (&pb, &rb)] {
            if ranks.len() == n as usize {
                assert!(Arc::ptr_eq(p, home.vtree()));
                continue;
            }
            is_by_definition(p, &kept_of(&home, ranks.iter().copied()), "projection");
            let locals: Vec<VarId> = (1..=ranks.len() as u32).map(VarId).collect();
            let depth = |mut h: VtreeIdx| {
                let mut d = 0;
                while let Some(parent) = home.vtree().node(h).parent() {
                    (h, d) = (parent, d + 1);
                }
                d
            };
            let depths: Vec<u32> = ranks.windows(2).map(|w| depth(home.node_over(w[0], w[1]))).collect();
            let local = Vtree::from_in_order(&locals, &depths, ranks.len() as u32).unwrap();
            let twin = p.local_twin(|t| VarId(t.0 + 1), ranks.len() as u32);
            assert!(local.same_tree(&twin), "from_in_order, seed {seed}");
        }

        let mut both: Vec<u32> = ra.iter().chain(&rb).copied().collect();
        both.sort_unstable();
        both.dedup();
        let union = Vtree::union(&pa, &pb).unwrap();
        match both.len() == n as usize {
            true => assert!(Arc::ptr_eq(&union, home.vtree())),
            false => {
                seen[0] += usize::from(!Arc::ptr_eq(&union, &pa) && !Arc::ptr_eq(&union, &pb));
                is_by_definition(&union, &kept_of(&home, both.iter().copied()), "union");
            }
        }

        if pa.home.is_some() {
            let keep: Vec<bool> = (0..pa.num_nodes()).map(|_| rng.coin()).collect();
            let mut map = Vec::new();
            if let Ok(sub) = pa.sub_projection_mapped(&home, |t| keep[t.idx()], Some(&mut map)) {
                let kept = ra.iter().enumerate().filter(|&(i, _)| keep[i]).map(|(_, &r)| r);
                if !Arc::ptr_eq(&sub, &pa) {
                    seen[1] += 1;
                    is_by_definition(&sub, &kept_of(&home, kept), "sub-projection");
                }
                for (t, &to) in map.iter().enumerate() {
                    let h = pa.home_node(VtreeIdx(t as u32));
                    assert_eq!(to, sub.node_at_home(h), "sub-projection map, seed {seed}, node {t}");
                }
            }
        }

        // Regions: some positions of `ra` anchor an extent reaching towards
        // the next position.
        if ra.len() == n as usize {
            continue;
        }
        let mut regions = Vec::new();
        for (i, &r) in ra.iter().enumerate() {
            if rng.below(3) == 0 {
                let end = ra.get(i + 1).copied().unwrap_or(n).min(r + 1 + rng.below(3) as u32);
                let count = end - r;
                regions.push(Region { at: i as u32, extent: Box::new([(r, end)]), count });
            }
        }
        if regions.is_empty() {
            continue;
        }
        let own = Vtree::project_regions(
            &home,
            ra.iter().enumerate().filter(|&(i, _)| regions.iter().all(|g| g.at as usize != i)).map(|(_, &r)| r).collect(),
            regions.iter().map(|g| (ra[g.at as usize], g.extent.clone())).collect(),
        );
        is_by_definition(&own, &kept_of(&home, ra.iter().copied()), "regions");
        let keep: Vec<bool> = (0..own.num_nodes()).map(|_| rng.coin()).collect();
        let mut map = Vec::new();
        if let Ok(sub) = own.sub_projection_mapped(&home, |t| keep[t.idx()], Some(&mut map)) {
            let anchors: Vec<u32> = regions.iter().map(|g| ra[g.at as usize]).collect();
            let kept = own.projected_leaves().zip(&ra).filter(|&(t, r)| anchors.contains(r) || keep[t.idx()]).map(|(_, &r)| r);
            seen[2] += usize::from(!Arc::ptr_eq(&sub, &own));
            is_by_definition(&sub, &kept_of(&home, kept.collect::<Vec<_>>()), "sub-projection with regions");
            assert_eq!(sub.regions().count(), regions.len());
            for (t, &to) in map.iter().enumerate() {
                assert_eq!(to, sub.node_at_home(own.home_node(VtreeIdx(t as u32))), "sub map with regions, node {t}");
            }
        }
        let outside: Vec<u32> = (0..n).filter(|&r| regions.iter().all(|g| !(g.extent[0].0..g.extent[0].1).contains(&r))).filter(|_| rng.coin()).collect();
        if let Some(&first) = outside.first() {
            let other = Vtree::projection(&home, outside.iter().map(|&r| var(r))).unwrap();
            if let Ok(LeafSets::Union(union)) = Vtree::leaf_sets(&own, &other) {
                let mut all: Vec<u32> = ra.iter().chain(&outside).copied().collect();
                all.sort_unstable();
                all.dedup();
                seen[3] += 1;
                is_by_definition(&union, &kept_of(&home, all), "union with regions");
                assert_eq!(union.regions().count(), regions.len(), "seed {seed}, first {first}");
            }
        }
    }
    assert!(seen.iter().all(|&count| count > 20), "{seen:?}");
}
