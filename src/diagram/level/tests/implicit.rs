//! Descriptions of complete products: read back from a level, multiplied as
//! the conjunction's row loop multiplies levels, and written out again.

use super::*;
use crate::test_helpers::Lcg;

/// The pairs of a level, node by node, as child slots.
type Pairs = Vec<Vec<(i64, i64)>>;

/// An explicit level holding `pairs`.
fn level_of(pairs: &Pairs) -> TddLevel {
    let mut level = TddLevel::new();
    for node in pairs {
        let node: Vec<ChildPair> = node.iter().map(|&(l, r)| pair(l, r)).collect();
        level.push_internal_node(&node);
    }
    level
}

/// The pairs `d` describes, node by node.
fn described(d: &ImplicitLevel) -> Pairs {
    let mut all = Vec::new();
    d.for_each_pair(|l, r| all.push((l, r)));
    all.chunks(d.pairs_per_node()).map(<[_]>::to_vec).collect()
}

/// A random affine level: `nodes` nodes of `per_node` pairs, the radices
/// random factors of each, the steps random, the slots from 0.
fn affine(rng: &mut Lcg, nodes: usize, per_node: usize) -> Pairs {
    fn factors(rng: &mut Lcg, mut n: usize) -> Vec<usize> {
        let mut out = Vec::new();
        while n > 1 {
            let divisors: Vec<usize> = (2..=n).filter(|d| n.is_multiple_of(*d)).collect();
            let d = divisors[rng.below(divisors.len() as u64) as usize];
            out.push(d);
            n /= d;
        }
        out
    }
    let step = |rng: &mut Lcg| (rng.below(7) as i64, rng.below(7) as i64);
    let within: Vec<(usize, (i64, i64))> = factors(rng, per_node).into_iter().map(|r| (r, step(rng))).collect();
    let across: Vec<(usize, (i64, i64))> = factors(rng, nodes).into_iter().map(|r| (r, step(rng))).collect();
    let value = |digits: &[(usize, (i64, i64))], mut n: usize| {
        digits.iter().fold((0, 0), |(l, r), &(radix, (dl, dr))| {
            let c = (n % radix) as i64;
            n /= radix;
            (l + c * dl, r + c * dr)
        })
    };
    (0..nodes)
        .map(|i| {
            let (fl, fr) = value(&across, i);
            (0..per_node).map(|m| {
                let (l, r) = value(&within, m);
                (fl + l, fr + r)
            }).collect()
        })
        .collect()
}

/// A random affine level of up to `nodes` nodes and of `least` to
/// `least + per_node - 1` pairs a node.
fn random_affine(rng: &mut Lcg, nodes: u64, per_node: u64, least: usize) -> Pairs {
    let n = 1 + rng.below(nodes) as usize;
    let k = least + rng.below(per_node) as usize;
    affine(rng, n, k)
}

/// The row loop's pairs for `f` and `g` with child strides `stride`: node
/// `i · |g| + j` of every pair of `f`'s node `i` with every pair of `g`'s
/// node `j`, in the row loop's order: `f`'s pairs outer, or on a grouped
/// level a run of `f`'s pairs sharing a left slot, then a run of `g`'s,
/// then the pairs of each.
fn row_loop(f: &Pairs, g: &Pairs, stride: (i64, i64), grouped: bool) -> Pairs {
    fn runs(pairs: &[(i64, i64)]) -> Vec<&[(i64, i64)]> {
        pairs.chunk_by(|a, b| a.0 == b.0).collect()
    }
    let mut out = Vec::new();
    for fi in f {
        for gj in g {
            let mut node = Vec::new();
            let mut emit = |p1: &(i64, i64), p2: &(i64, i64)| node.push((p1.0 * stride.0 + p2.0, p1.1 * stride.1 + p2.1));
            if grouped {
                for r1 in runs(fi) {
                    for r2 in runs(gj) {
                        for p1 in r1 {
                            for p2 in r2 {
                                emit(p1, p2);
                            }
                        }
                    }
                }
            } else {
                for p1 in fi {
                    for p2 in gj {
                        emit(p1, p2);
                    }
                }
            }
            out.push(node);
        }
    }
    out
}

#[test]
fn an_affine_level_is_read_back_and_written_out() {
    let mut rng = Lcg::new(0x1d1e_0001);
    for _ in 0..300 {
        let nodes = 1 + rng.below(12) as usize;
        let per_node = 1 + rng.below(12) as usize;
        let pairs = affine(&mut rng, nodes, per_node);
        let level = level_of(&pairs);
        let d = ImplicitLevel::fit(&level).expect("an affine level fits");
        assert_eq!(described(&d), pairs);
        for i in 0..nodes {
            let mut out = Vec::new();
            d.pairs_of(i, &mut out);
            assert_eq!(out, level.pairs_vec(i));
        }
    }
}

#[test]
fn a_level_that_is_not_affine_does_not_fit() {
    // Two nodes whose pairs differ by other offsets.
    let pairs: Pairs = vec![vec![(0, 0), (1, 0)], vec![(2, 0), (4, 0)]];
    assert!(ImplicitLevel::fit(&level_of(&pairs)).is_none());
    // Nodes with different numbers of pairs.
    let pairs: Pairs = vec![vec![(0, 0), (1, 0)], vec![(2, 0)]];
    assert!(ImplicitLevel::fit(&level_of(&pairs)).is_none());
    // Offsets that are no mixed radix: 0, 1, 3.
    let pairs: Pairs = vec![vec![(0, 0), (1, 0), (3, 0)]];
    assert!(ImplicitLevel::fit(&level_of(&pairs)).is_none());
}

#[test]
fn a_product_is_the_row_loops_pairs() {
    let mut rng = Lcg::new(0x1d1e_0002);
    let mut grouped_seen = 0;
    for _ in 0..400 {
        let f = random_affine(&mut rng, 6, 9, 1);
        let g = random_affine(&mut rng, 6, 9, 1);
        let df = ImplicitLevel::fit(&level_of(&f)).unwrap();
        let dg = ImplicitLevel::fit(&level_of(&g)).unwrap();
        let stride = (1 + rng.below(40) as i64, 1 + rng.below(40) as i64);
        let s = (stride.0 as usize, stride.1 as usize);
        let plain = ImplicitLevel::product(&df, &dg, s, false).expect("an ungrouped product is described");
        assert_eq!(described(&plain), row_loop(&f, &g, stride, false));
        if let Some(grouped) = ImplicitLevel::product(&df, &dg, s, true) {
            grouped_seen += 1;
            assert_eq!(described(&grouped), row_loop(&f, &g, stride, true));
        }
        // The product of products: the description of a description.
        let h = random_affine(&mut rng, 3, 4, 1);
        let dh = ImplicitLevel::fit(&level_of(&h)).unwrap();
        let twice = ImplicitLevel::product(&plain, &dh, (3, 5), false).unwrap();
        assert_eq!(described(&twice), row_loop(&described(&plain), &h, (3, 5), false));
    }
    assert!(grouped_seen > 50, "the grouped products went unchecked");
}

#[test]
fn a_pair_is_read_where_it_lies() {
    let mut rng = Lcg::new(0x1d1e_0003);
    for _ in 0..100 {
        let pairs = random_affine(&mut rng, 8, 8, 2);
        let d = ImplicitLevel::fit(&level_of(&pairs)).unwrap();
        let i = rng.below(pairs.len() as u64) as usize;
        let m = rng.below(pairs[i].len() as u64) as usize;
        let mut places = d.places(i);
        assert_eq!(places.len(), pairs[i].len());
        assert_eq!(places.nth(m), Some(pair(pairs[i][m].0, pairs[i][m].1)));
        assert_eq!(places.len(), pairs[i].len() - m - 1);
        assert!(places.eq(pairs[i][m + 1..].iter().map(|&(l, r)| pair(l, r))));
    }
}

#[test]
fn an_implicit_arena_reads_as_the_stored_one() {
    let mut rng = Lcg::new(0x1d1e_0004);
    for _ in 0..50 {
        let pairs = random_affine(&mut rng, 8, 8, 2);
        let stored = level_of(&pairs);
        let d = ImplicitLevel::fit(&stored).unwrap();
        let mut level = TddLevel::new();
        d.write_nodes(&Limits::new(), &mut level).unwrap();
        level.pairs.describe(d.clone(), 3 * d.pairs());
        assert_eq!(level.nodes, stored.nodes);
        assert_eq!(level.implicit(), Some(&d));
        assert_eq!(level.pairs.len(), stored.pairs.len());
        assert_eq!(level.pairs.capacity(), 3 * d.pairs());
        // A reader sees the pairs; the description stays.
        let mut buf = Vec::new();
        for i in 0..level.nodes.len() {
            assert_eq!(level.pairs_read(i, &mut buf), stored.pairs_vec(i));
            assert!(level.pairs_iter_of_idx(i).eq(stored.pairs_iter_of_idx(i)));
        }
        assert!(level.pairs.implicit().is_some() && level.pairs.stored().is_none());
        // Clearing leaves an empty stored arena of the capacity held.
        level.clear();
        assert!(level.pairs.is_empty() && level.pairs.implicit().is_none());
        assert_eq!(level.pairs.capacity(), 3 * d.pairs());
    }
}

#[test]
#[should_panic(expected = "an implicit level's pairs are not stored")]
fn an_implicit_arena_is_not_written_in_place() {
    let pairs: Pairs = (0..8).map(|i| (0..8).map(|m| (8 * i + m, m)).collect()).collect();
    let stored = level_of(&pairs);
    let d = ImplicitLevel::fit(&stored).unwrap();
    let mut level = TddLevel::new();
    d.write_nodes(&Limits::new(), &mut level).unwrap();
    level.pairs.describe(d, 64);
    level.pairs.stored_mut();
}

/// The `j`th digit of node `i` of `d`, counting its node digits only.
fn node_digit(d: &ImplicitLevel, i: usize, j: usize) -> usize {
    let g = &d.digits()[d.within() + j];
    (i / g.node as usize) % g.radix
}

/// What a prune leaves of an affine level, read off the description and
/// the children's new indices, is the description of the pairs written out
/// and moved, when they have one, and nothing when they have none: the
/// nodes a prune keeps on a diagonal of two digits, in a box, or at random,
/// with each child's slots renumbered by their rank among a random set
/// holding every slot the kept nodes name, as a prune renumbers a child.
#[test]
fn what_a_prune_leaves_is_read_off_the_description() {
    let mut rng = Lcg::new(0x1d1e_0005);
    let (mut fitted, mut diagonals) = (0, 0);
    for round in 0..600 {
        let pairs = random_affine(&mut rng, 24, 6, 2);
        let d = ImplicitLevel::fit(&level_of(&pairs)).unwrap();
        let n = d.nodes();
        let digits = d.digits().len() - d.within();
        let kept: Vec<usize> = match round % 4 {
            0 if digits >= 2 => {
                let (a, b) = (rng.below(digits as u64) as usize, rng.below(digits as u64) as usize);
                (0..n).filter(|&i| node_digit(&d, i, a) == node_digit(&d, i, b)).collect()
            }
            1 if digits >= 1 => {
                let a = rng.below(digits as u64) as usize;
                let at = rng.below(d.digits()[d.within() + a].radix as u64) as usize;
                (0..n).filter(|&i| node_digit(&d, i, a) == at).collect()
            }
            2 => (0..n).collect(),
            _ => (0..n).filter(|_| rng.below(3) != 0).collect(),
        };
        if kept.is_empty() {
            continue;
        }
        // A side is renumbered, or kept as it is.
        let mut renumber = |side: fn(&(i64, i64)) -> i64| -> Option<Vec<i64>> {
            if rng.below(3) == 0 {
                return None;
            }
            let named: Vec<i64> = kept.iter().flat_map(|&i| pairs[i].iter().map(side)).collect();
            let top = named.iter().copied().max().unwrap() + 3;
            let mut rank = vec![-1i64; top as usize + 1];
            let mut next = 0;
            for (x, r) in rank.iter_mut().enumerate() {
                if named.contains(&(x as i64)) || rng.below(4) == 0 {
                    *r = next;
                    next += 1;
                }
            }
            Some(rank)
        };
        let (left, right) = (renumber(|p| p.0), renumber(|p| p.1));
        let moved = |rank: &Option<Vec<i64>>, x: i64| rank.as_ref().map_or(x, |r| r[x as usize]);
        let oracle: Pairs = kept.iter().map(|&i| pairs[i].iter().map(|&(l, r)| (moved(&left, l), moved(&right, r))).collect()).collect();
        let read = d.pruned(kept.len(), |j| kept.get(j).copied(), |x| moved(&left, x), |x| moved(&right, x));
        assert_eq!(read, ImplicitLevel::fit(&level_of(&oracle)), "round {round}");
        if let Some(r) = read {
            assert_eq!(described(&r), oracle);
            fitted += 1;
            diagonals += usize::from(round % 4 == 0 && kept.len() < n);
        }
    }
    assert!(fitted > 100 && diagonals > 10, "{fitted} fitted, {diagonals} on a diagonal");
}

/// An arena a prune kept described keeps the length and capacity the
/// written one keeps, reads its nodes' pairs from the new description, is
/// copied at its length, and drops the slots past them on a sweep's
/// truncation without writing them; no truncation cuts into the described
/// pairs.
#[test]
fn a_redescribed_arena_keeps_the_written_length() {
    let pairs: Pairs = (0..4).map(|i| (0..3).map(|m| (3 * i + m, i)).collect()).collect();
    let d = ImplicitLevel::fit(&level_of(&pairs)).unwrap();
    let mut level = TddLevel::new();
    d.write_nodes(&Limits::new(), &mut level).unwrap();
    level.pairs.describe(d.clone(), 20);
    // Keep nodes 1 and 3, renumbered 0 and 1.
    let kept = [1usize, 3];
    let left_of = d.pruned(2, |j| kept.get(j).copied(), |x| x, |x| x).unwrap();
    level.nodes.truncate(2);
    level.pairs.redescribe(left_of.clone());
    assert_eq!((level.pairs.len(), level.pairs.capacity()), (12, 20));
    assert_eq!(level.implicit(), Some(&left_of));
    let mut buf = Vec::new();
    for (j, &i) in kept.iter().enumerate() {
        let want: Vec<ChildPair> = pairs[i].iter().map(|&(l, r)| pair(l, r)).collect();
        assert_eq!(level.pairs_read(j, &mut buf), &want[..]);
        assert_eq!(level.pairs_vec(j), &want[..]);
    }
    // Read whole, the arena has the written length.
    assert_eq!(level.pairs.len(), 12);
    // A copy has its length as its capacity, as a copy of a stored arena
    // does.
    let mut swept = level.clone();
    assert_eq!((swept.pairs.len(), swept.pairs.capacity()), (12, 12));
    swept.pairs.truncate(6);
    assert!(swept.pairs.implicit().is_some());
    assert_eq!((swept.pairs.len(), swept.pairs.capacity()), (6, 12));
    swept.pairs.shrink_to_fit();
    assert_eq!(swept.pairs.capacity(), 6);
    let cut = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| level.pairs.truncate(5)));
    assert!(cut.is_err(), "a truncation into the described pairs");
}
