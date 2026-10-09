//! Descriptions of complete products: read back from a level, multiplied as
//! the conjunction's row loop multiplies levels, and written out again.

use super::*;
use crate::limits::Limits;
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

/// The pairs `d` describes, node by node: every setting of its digits,
/// from the first pair.
fn described(d: &ImplicitLevel) -> Pairs {
    let mut all = Vec::new();
    each_place(&d.digits, d.first, |l, r| all.push((l, r)));
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
    let within = factors(rng, per_node).into_iter().map(|r| (r, step(rng))).collect::<Vec<_>>();
    let across = factors(rng, nodes).into_iter().map(|r| (r, step(rng))).collect::<Vec<_>>();
    affine_of(&within, &across)
}

/// A random step on each side.
fn step(rng: &mut Lcg) -> (i64, i64) {
    (rng.below(7) as i64, rng.below(7) as i64)
}

/// The level whose pairs `within` and `across`, digits of a radix and a
/// step each, fastest first, number from the slots 0: pair `m` of node `i`
/// at the digits of `m` in `within` plus those of `i` in `across`.
fn affine_of(within: &[(usize, (i64, i64))], across: &[(usize, (i64, i64))]) -> Pairs {
    let value = |digits: &[(usize, (i64, i64))], mut n: usize| {
        digits.iter().fold((0, 0), |(l, r), &(radix, (dl, dr))| {
            let c = (n % radix) as i64;
            n /= radix;
            (l + c * dl, r + c * dr)
        })
    };
    let (nodes, per_node) = (across.iter().map(|d| d.0).product(), within.iter().map(|d| d.0).product());
    (0..nodes)
        .map(|i| {
            let (fl, fr) = value(across, i);
            (0..per_node).map(|m| {
                let (l, r) = value(within, m);
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

/// Slice reads replace scratch on implicit levels, append reads retain its
/// prefix, and a stored read borrows its arena without touching scratch.
#[test]
fn pair_buffers_agree_across_stored_and_implicit_levels() {
    for k in [1, 2, 63, 64, 65, 257] {
        let pairs = affine_of(&[(k, (1, 2))], &[(3, (1000, 2000))]);
        let stored = level_of(&pairs);
        let mut implicit = TddLevel::new();
        let d = ImplicitLevel::fit(&stored).unwrap();
        implicit.pairs.describe(d.clone(), d.arena_len(), d.nodes());
        let prefix = pair(99, 101);
        let mut scratch = vec![prefix];
        for i in [2, 0, 1, 2] {
            let expected = stored.pairs_vec(i);
            assert_eq!(stored.pairs_read(i, &mut scratch), expected);
            assert_eq!(scratch, [prefix]);
            assert_eq!(implicit.pairs_read(i, &mut scratch), expected);
            assert_eq!(scratch, expected);
            scratch.clear();
            scratch.push(prefix);
            d.pairs_of(i, &mut scratch);
            assert_eq!(scratch[0], prefix);
            assert_eq!(scratch[1..], expected);
            scratch.truncate(1);
        }
    }
}

/// Bounds do not depend on storage or on debug assertions being enabled.
#[test]
fn pair_slice_reads_reject_indices_past_the_level() {
    for k in [1, 2, 65] {
        let stored = level_of(&affine_of(&[(k, (1, 2))], &[(3, (1000, 2000))]));
        let mut implicit = TddLevel::new();
        let d = ImplicitLevel::fit(&stored).unwrap();
        implicit.pairs.describe(d.clone(), d.arena_len(), d.nodes());
        for level in [&stored, &implicit] {
            for i in [3, usize::MAX] {
                assert!(std::panic::catch_unwind(|| {
                    level.pairs_read(i, &mut Vec::new());
                }).is_err());
            }
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

/// A level of one pair a node implies the nodes its description gives,
/// inline, read in order, folded or one at a time, and stores them back:
/// node digits of short and long runs, more digits past a run than the
/// odometer counts, a fastest digit of more places than a run holds, and
/// steps of either sign.
#[test]
fn a_one_pair_level_implies_the_nodes_it_describes() {
    let mut rng = Lcg::new(0x1d1e_0031);
    let shapes: [&[usize]; 9] =
        [&[64], &[300], &[600, 3], &[257, 2], &[2; 14], &[3, 5, 7], &[5, 100], &[2, 2, 3, 4, 2, 3, 5], &[16, 16, 2]];
    for radices in shapes {
        for _ in 0..4 {
            let across: Vec<(usize, (i64, i64))> =
                radices.iter().map(|&r| (r, (rng.below(9) as i64 - 4, rng.below(9) as i64 - 4))).collect();
            // The first pair as far from 0 as the steps down reach.
            let low = |side: fn((i64, i64)) -> i64| -> i64 {
                -across.iter().map(|&(r, s)| (r as i64 - 1) * side(s).min(0)).sum::<i64>()
            };
            let first = (low(|s| s.0), low(|s| s.1));
            let n = radices.iter().product();
            let d = ImplicitLevel::assemble(n, 1, first, &[], &across);
            let want: Vec<EncodedNode> = described(&d).iter().map(|p| EncodedNode::inline(pair(p[0].0, p[0].1))).collect();
            let mut level = TddLevel::new();
            level.pairs.describe(d.clone(), 0, n);
            assert!(level.nodes.stored().is_empty() && level.pairs.is_empty());
            assert!(level.nodes().iter().eq(want.iter().copied()), "{across:?}");
            let mut folded = Vec::new();
            level.nodes().iter().for_each(|node| folded.push(node));
            assert_eq!(folded, want, "{across:?}, folded");
            for _ in 0..8 {
                let i = rng.below(n as u64) as usize;
                assert_eq!(level.node(i), want[i]);
                assert_eq!(level.nodes().iter().nth(i), Some(want[i]));
            }
            assert!(!level.has_multi_pair());
            level.store_if_implicit(&Limits::new()).unwrap();
            assert_eq!(level.nodes.stored(), &want[..], "{across:?}, stored");
            assert!(level.pairs.implicit().is_none() && level.pairs.is_empty());
        }
    }
}

/// An affine stored level of one pair a node closes to the description of
/// its pairs, its nodes implied and its arenas' lengths and capacities
/// kept, and built stored again it holds the words it held.
#[test]
fn a_one_pair_level_closes_and_is_stored_back() {
    let mut rng = Lcg::new(0x1d1e_0033);
    for radices in [&[64][..], &[300], &[257, 2], &[2; 9], &[16, 16, 2]] {
        let across: Vec<(usize, (i64, i64))> =
            radices.iter().map(|&r| (r, (1 + rng.below(5) as i64, rng.below(5) as i64))).collect();
        let stored = level_of(&affine_of(&[], &across));
        let mut level = stored.clone();
        // A clone holds its own capacities, which closing keeps.
        let held = (level.pairs.len(), level.pairs.capacity(), level.node_capacity());
        level.close();
        let d = level.implicit().expect("an affine level of one pair a node closes").clone();
        assert_eq!((d.pairs_per_node(), d.nodes()), (1, stored.nodes().len()));
        assert!(level.nodes.stored().is_empty());
        assert!(level.nodes().iter().eq(stored.nodes().iter()));
        assert_eq!((level.pairs.len(), level.pairs.capacity(), level.node_capacity()), held);
        level.store_if_implicit(&Limits::new()).unwrap();
        assert_eq!(level.nodes.stored(), stored.nodes.stored());
        assert_eq!((level.pairs.len(), level.pairs.capacity(), level.node_capacity()), held);
    }
}

/// A rewrite of an implicit level of one pair a node builds it stored as
/// the in-place rewrite leaves a stored one: a node whose pair changed holds
/// the new pair inline, one whose pair was dropped the empty placeholder,
/// and the arena no pair; a node's only pair is not removed.
#[test]
fn a_one_pair_level_is_rewritten_inline() {
    let pairs = affine_of(&[], &[(16, (1, 0)), (8, (0, 1))]);
    let mut level = level_of(&pairs);
    level.close();
    assert!(level.implicit().is_some());
    let only = level.pairs_vec(3)[0];
    let lim = Limits::new();
    assert!(!level.remove_pair_from_node(&lim, 3, only).unwrap());
    assert!(level.implicit().is_some(), "a node's only pair stays");
    let moved = |p: ChildPair| pair(i64::from(p.left.raw()) + 1000, i64::from(p.right.raw()));
    let emptied = level.rewrite_described(&lim, true, |i, _, _, p| match i % 5 {
        0 => None,
        1 => Some(moved(p)),
        _ => Some(p),
    }).unwrap();
    assert!(emptied);
    let empty = level.encode_multi(0, 0);
    let want: Vec<EncodedNode> = pairs
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let p = pair(p[0].0, p[0].1);
            match i % 5 {
                0 => empty,
                1 => EncodedNode::inline(moved(p)),
                _ => EncodedNode::inline(p),
            }
        })
        .collect();
    assert!(level.pairs.implicit().is_none());
    assert_eq!(level.nodes.stored(), &want[..]);
    assert!(level.pairs.is_empty() && level.dead_pairs == 0);
}

/// A level of one pair a node, its pairs inline, fits the description of
/// its pairs, its words compared in runs; with one node's pair moved it
/// fits only a description of its pairs.
#[test]
fn a_one_pair_level_fits_by_its_words() {
    let mut rng = Lcg::new(0x1d1e_0032);
    for radices in [&[64][..], &[300], &[257, 2], &[2; 9], &[3, 5, 7], &[5, 100], &[16, 16, 2]] {
        for _ in 0..4 {
            let across: Vec<(usize, (i64, i64))> =
                radices.iter().map(|&r| (r, (1 + rng.below(5) as i64, rng.below(5) as i64))).collect();
            let pairs = affine_of(&[], &across);
            let level = level_of(&pairs);
            let d = ImplicitLevel::fit(&level).expect("an affine level fits");
            assert_eq!(described(&d), pairs);
            assert!(d.holds_inline(level.nodes.stored()));
            let mut moved = pairs.clone();
            let i = rng.below(pairs.len() as u64) as usize;
            moved[i][0].0 += 1 + rng.below(3) as i64;
            if let Some(d) = ImplicitLevel::fit(&level_of(&moved)) {
                assert_eq!(described(&d), moved, "node {i} moved");
            }
        }
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
        level.pairs.describe(d.clone(), 3 * d.pairs(), d.nodes());
        // The description implies the nodes the stored level holds.
        assert!(level.nodes.stored().is_empty());
        assert!(level.nodes().iter().eq(stored.nodes().iter()));
        assert_eq!(level.implicit(), Some(&d));
        assert_eq!(level.pairs.len(), stored.pairs.len());
        assert_eq!(level.pairs.capacity(), 3 * d.pairs());
        // A reader sees the pairs; the description stays.
        let mut buf = Vec::new();
        for i in 0..level.nodes().len() {
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

/// A level held as a description stands for arenas of a capacity it does
/// not allocate. Cleared where the allocator refuses that capacity, it
/// leaves empty arenas, not an abort, and the level pool, which keeps no
/// arena past its cap, gives it none.
#[test]
fn an_implicit_arena_past_the_allocator_clears_to_an_empty_one() {
    let pairs: Pairs = (0..8).map(|i| (0..8).map(|m| (8 * i + m, m)).collect()).collect();
    let stored = level_of(&pairs);
    let d = ImplicitLevel::fit(&stored).unwrap();
    // More bytes than an allocation may have: `Vec::with_capacity` panics
    // on it, as it aborts on an address-space limit's refusal.
    let huge = isize::MAX as usize / std::mem::size_of::<ChildPair>() + 1;
    for pooled in [false, true] {
        let mut level = TddLevel::new();
        level.pairs.describe(d.clone(), huge, huge);
        assert_eq!((level.pairs.capacity(), level.node_capacity()), (huge, huge));
        let kept = match pooled {
            true => crate::diagram::pool::reset_level(&mut level),
            false => {
                level.clear();
                0
            }
        };
        assert!(level.pairs.is_empty() && level.pairs.implicit().is_none() && level.nodes().is_empty());
        assert_eq!((level.pairs.capacity(), level.node_capacity(), kept), (0, 0, 0));
    }
}

#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "an implicit level's pairs are not stored")]
fn an_implicit_arena_is_not_written_in_place() {
    let pairs: Pairs = (0..8).map(|i| (0..8).map(|m| (8 * i + m, m)).collect()).collect();
    let stored = level_of(&pairs);
    let d = ImplicitLevel::fit(&stored).unwrap();
    let mut level = TddLevel::new();
    level.pairs.describe(d.clone(), 64, d.nodes());
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
        let read = d.pruned(&Limits::new(), kept.len(), |j| kept.get(j).copied(), || kept.iter().copied(), |x| moved(&left, x), |x| moved(&right, x)).unwrap();
        assert_eq!(read, ImplicitLevel::fit(&level_of(&oracle)), "round {round}");
        if let Some(r) = read {
            assert_eq!(described(&r), oracle);
            fitted += 1;
            diagonals += usize::from(round % 4 == 0 && kept.len() < n);
        }
    }
    assert!(fitted > 100 && diagonals > 10, "{fitted} fitted, {diagonals} on a diagonal");
}

/// What a prune keeps of a box of an affine level, each child kept whole or
/// renumbered onto exactly the slots the kept pairs name, is derived from
/// the description as the fit of the pairs written out and moved gives it;
/// on a kept set that is no box, or a child that kept a slot no kept pair
/// names, the derivation gives nothing, and never another description. At
/// one pair a node and more, the nodes kept in a box of random ranges of
/// the node digits, at random, or all.
#[test]
fn what_a_prune_keeps_of_a_box_is_derived_from_the_description() {
    let mut rng = Lcg::new(0x1d1e_0006);
    let (mut derived, mut renumbered, mut one_pair) = (0, 0, 0);
    for round in 0..1500 {
        let pairs = random_affine(&mut rng, 36, 4, 1);
        let d = ImplicitLevel::fit(&level_of(&pairs)).unwrap();
        let n = d.nodes();
        let node_digits = &d.digits()[d.within()..];
        let in_box = round % 3 != 1;
        let kept: Vec<usize> = match round % 3 {
            0 => {
                let ranges: Vec<(usize, usize)> = node_digits.iter().map(|g| {
                    let a = rng.below(g.radix as u64) as usize;
                    (a, a + rng.below((g.radix - a) as u64) as usize)
                }).collect();
                (0..n).filter(|&i| ranges.iter().enumerate().all(|(j, &(a, b))| (a..=b).contains(&node_digit(&d, i, j)))).collect()
            }
            1 => (0..n).filter(|_| rng.below(4) != 0).collect(),
            _ => (0..n).collect(),
        };
        if kept.is_empty() {
            continue;
        }
        let mut own = vec![0u64; n.div_ceil(64)];
        for &i in &kept {
            own[i >> 6] |= 1 << (i & 63);
        }
        // A side is kept whole, renumbered onto the slots the kept pairs
        // name, or renumbered onto those and one more, a slot another
        // reference keeps.
        let mut child = |side: fn(&(i64, i64)) -> i64| -> (ChildKept, Option<Vec<i64>>, bool) {
            let how = rng.below(5);
            if how < 2 {
                return (ChildKept::Whole, None, false);
            }
            let named: std::collections::BTreeSet<i64> = kept.iter().flat_map(|&i| pairs[i].iter().map(side)).collect();
            let top = named.iter().copied().max().unwrap() + 2;
            let extra = (how == 4).then(|| (0..=top).find(|x| !named.contains(x)).unwrap());
            let mut rank = vec![-1i64; top as usize + 1];
            let mut next = 0;
            for (x, r) in rank.iter_mut().enumerate() {
                if named.contains(&(x as i64)) || extra == Some(x as i64) {
                    *r = next;
                    next += 1;
                }
            }
            (ChildKept::Renumbered(next as usize), Some(rank), extra.is_some())
        };
        let ((lk, left, l_extra), (rk, right, r_extra)) = (child(|p| p.0), child(|p| p.1));
        let moved = |rank: &Option<Vec<i64>>, x: i64| rank.as_ref().map_or(x, |r| r[x as usize]);
        let oracle: Pairs = kept.iter().map(|&i| pairs[i].iter().map(|&(l, r)| (moved(&left, l), moved(&right, r))).collect()).collect();
        let fitted = ImplicitLevel::fit(&level_of(&oracle));
        let got = d.kept_box(&own, kept.len(), lk, rk);
        if let Some(g) = &got {
            assert_eq!(got, fitted, "round {round}: a derived description differs from the fit");
            assert_eq!(described(g), oracle);
            derived += 1;
            renumbered += usize::from(lk != ChildKept::Whole || rk != ChildKept::Whole);
            one_pair += usize::from(d.pairs_per_node() == 1);
        }
        // A box with both children whole is always derived; a child that
        // kept a slot no kept pair names never is.
        if in_box && lk == ChildKept::Whole && rk == ChildKept::Whole {
            assert!(got.is_some(), "round {round}: a box of whole children was not derived");
        }
        if l_extra || r_extra {
            assert!(got.is_none(), "round {round}: derived past a slot another reference keeps");
        }
    }
    assert!(derived > 300 && renumbered > 100 && one_pair > 50, "{derived} derived, {renumbered} renumbered, {one_pair} at one pair a node");
}

/// What a prune keeps of any nodes of an affine level is derived from the
/// description, as the fit of the pairs written out and moved gives it,
/// where each side is one no drop bends: moved by no node digit, or
/// renumbered onto exactly the slots the kept pairs name with its slot
/// strictly monotone in the pair's position; and nothing is derived past a
/// slot a child keeps that no kept pair names.
#[test]
fn what_a_prune_keeps_of_any_nodes_is_derived_where_no_drop_bends_a_side() {
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
    let mut rng = Lcg::new(0x1d1e_0007);
    let (mut derived, mut bent) = (0, 0);
    for round in 0..2000 {
        let (n, k) = (2 + rng.below(35) as usize, 1 + rng.below(4) as usize);
        let (within, across) = (factors(&mut rng, k), factors(&mut rng, n));
        // The left slot rising or falling with the position, or at random;
        // the right moved by no node digit, or at random.
        let monotone = rng.below(3) != 0;
        let (sign, scale) = (if rng.below(2) == 0 { 1 } else { -1 }, 1 + rng.below(3) as i64);
        let constant = rng.below(3) != 0;
        let mut unit = 1i64;
        let mut digit = |radix: usize, node: bool, rng: &mut Lcg| {
            let l = if monotone { sign * scale * unit } else { rng.below(7) as i64 - 3 };
            let r = if constant && node { 0 } else { rng.below(7) as i64 - 3 };
            unit *= radix as i64;
            (radix, (l, r))
        };
        let within: Vec<_> = within.into_iter().map(|r| digit(r, false, &mut rng)).collect();
        let across: Vec<_> = across.into_iter().map(|r| digit(r, true, &mut rng)).collect();
        let raw = affine_of(&within, &across);
        let low = |side: fn(&(i64, i64)) -> i64| raw.iter().flatten().map(side).min().unwrap();
        let (bl, br) = (low(|p| p.0), low(|p| p.1));
        let pairs: Pairs = raw.iter().map(|node| node.iter().map(|&(l, r)| (l - bl, r - br)).collect()).collect();
        let d = ImplicitLevel::fit(&level_of(&pairs)).unwrap();
        let kept: Vec<usize> = (0..n).filter(|_| rng.below(4) != 0).collect();
        if kept.is_empty() {
            continue;
        }
        let mut child = |side: fn(&(i64, i64)) -> i64| -> (ChildKept, Option<Vec<i64>>, bool) {
            let how = rng.below(6);
            if how < 2 {
                return (ChildKept::Whole, None, false);
            }
            let named: std::collections::BTreeSet<i64> = kept.iter().flat_map(|&i| pairs[i].iter().map(side)).collect();
            let top = named.iter().copied().max().unwrap() + 2;
            let extra = (how == 5).then(|| (0..=top).find(|x| !named.contains(x)).unwrap());
            let mut rank = vec![-1i64; top as usize + 1];
            let mut next = 0;
            for (x, r) in rank.iter_mut().enumerate() {
                if named.contains(&(x as i64)) || extra == Some(x as i64) {
                    *r = next;
                    next += 1;
                }
            }
            (ChildKept::Renumbered(next as usize), Some(rank), extra.is_some())
        };
        let ((lk, left, l_extra), (rk, right, r_extra)) = (child(|p| p.0), child(|p| p.1));
        let moved = |rank: &Option<Vec<i64>>, x: i64| rank.as_ref().map_or(x, |r| r[x as usize]);
        let oracle: Pairs = kept.iter().map(|&i| pairs[i].iter().map(|&(l, r)| (moved(&left, l), moved(&right, r))).collect()).collect();
        let got = d.kept_any(kept.len(), lk, rk);
        if let Some(got) = &got {
            assert_eq!(Some(got), ImplicitLevel::fit(&level_of(&oracle)).as_ref(), "round {round}: a derived description differs from the fit");
            assert_eq!(&described(got), &oracle);
            derived += 1;
            bent += usize::from(kept.len() < n && lk != ChildKept::Whole && d.digits()[d.within()..].iter().any(|g| g.left != 0));
        }
        // A renumbered constant side needs its place digits one to one; at
        // one pair a node it has none.
        if monotone && lk != ChildKept::Whole && !l_extra && constant && (rk == ChildKept::Whole || k == 1 && !r_extra) {
            assert!(got.is_some(), "round {round}: a rising side and a constant one were not derived");
        }
        if l_extra && d.digits()[d.within()..].iter().any(|g| g.left != 0) || r_extra && d.digits()[d.within()..].iter().any(|g| g.right != 0) {
            assert!(got.is_none(), "round {round}: derived past a slot another reference keeps");
        }
    }
    assert!(derived > 400 && bent > 100, "{derived} derived, {bent} past dropped nodes on a moving side");
}

/// The derivation reads a box whose nodes are not consecutive, a run of
/// nodes a slower digit's step apart, and refuses one node missing from
/// any run; a renumbered side whose digits name a slot twice is refused,
/// one whose steps fall counts down.
#[test]
fn a_kept_box_is_read_in_runs() {
    // Nodes 0..24 numbered by digits of radices 2, 3, 4; one pair a node,
    // what a unit of each adds to the slots (1, 0), (0, 1) and (2, 5).
    let pairs = affine_of(&[], &[(2, (1, 0)), (3, (0, 1)), (4, (2, 5))]);
    let d = ImplicitLevel::fit(&level_of(&pairs)).unwrap();
    let box_of = |a: [std::ops::RangeInclusive<usize>; 3]| -> Vec<usize> {
        (0..24).filter(|&i| a[0].contains(&(i % 2)) && a[1].contains(&(i / 2 % 3)) && a[2].contains(&(i / 6))).collect()
    };
    let own_of = |kept: &[usize]| vec![kept.iter().fold(0u64, |w, &i| w | 1 << i)];
    let kept = box_of([0..=1, 1..=2, 1..=3]);
    let got = d.kept_box(&own_of(&kept), kept.len(), ChildKept::Whole, ChildKept::Whole).unwrap();
    let oracle: Pairs = kept.iter().map(|&i| pairs[i].clone()).collect();
    assert_eq!(described(&got), oracle);
    // A node missing from the last run.
    let mut holed = kept.clone();
    holed.pop();
    assert_eq!(d.kept_box(&own_of(&holed), holed.len(), ChildKept::Whole, ChildKept::Whole), None);
    // As many nodes as the box, one of them outside it.
    let mut shifted = holed.clone();
    shifted.push(0);
    shifted.sort_unstable();
    assert_eq!(d.kept_box(&own_of(&shifted), shifted.len(), ChildKept::Whole, ChildKept::Whole), None);
    // The left slots a + 2·b over a of radix 3 and b of radix 2 name slot 2
    // twice, five slots in all: refused, by the count or by the steps.
    let twice = affine_of(&[], &[(3, (1, 0)), (2, (2, 1))]);
    let d2 = ImplicitLevel::fit(&level_of(&twice)).unwrap();
    for named in [5, 6] {
        assert_eq!(d2.kept_box(&[0b11_1111], 6, ChildKept::Renumbered(named), ChildKept::Whole), None);
    }
    // A falling step counts down: the slots 9 - 3·a over a = 0..2 are
    // renumbered 2, 1, 0.
    let pairs = affine_of(&[], &[(3, (-3, 1))]).into_iter().map(|n| n.into_iter().map(|(l, r)| (l + 9, r)).collect()).collect::<Pairs>();
    let d = ImplicitLevel::fit(&level_of(&pairs)).unwrap();
    let got = d.kept_box(&[0b111], 3, ChildKept::Renumbered(3), ChildKept::Whole).unwrap();
    assert_eq!(described(&got), vec![vec![(2, 0)], vec![(1, 1)], vec![(0, 2)]]);
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
    level.pairs.describe(d.clone(), 20, d.nodes());
    // Keep nodes 1 and 3, renumbered 0 and 1.
    let kept = [1usize, 3];
    let left_of = d.pruned(&Limits::new(), 2, |j| kept.get(j).copied(), || kept.iter().copied(), |x| x, |x| x).unwrap().unwrap();
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

/// The pairs a node's [`Places`] steps through, from its first or from a
/// place skipped to, and the first pairs a [`NodeCursor`] steps or reads
/// off the digits, in node order or not, are the pairs read off the digits
/// one by one: on levels of up to six place digits and six node digits,
/// more than the odometer counts in place, on nodes read in several runs,
/// and on nodes whose fastest digit has more places than a run.
#[test]
fn stepped_pairs_are_the_pairs_read_off_the_digits() {
    let mut rng = Lcg::new(0x1d1e_0007);
    let (mut long, mut runs, mut wide) = (0, 0, 0);
    let digits = |rng: &mut Lcg, n: u64, radices: u64| {
        (0..n).map(|_| (2 + rng.below(radices) as usize, step(rng))).collect::<Vec<_>>()
    };
    for round in 0..400 {
        let pairs = match round % 4 {
            0 | 2 => {
                let (w, a) = (5 + rng.below(2), rng.below(7));
                let (within, across) = (digits(&mut rng, w, 2), digits(&mut rng, a, 1));
                affine_of(&within, &across)
            }
            1 => {
                let (n, k) = (1 + rng.below(40) as usize, 2 + rng.below(30) as usize);
                affine(&mut rng, n, k)
            }
            _ => {
                let fastest = (RUN_PAIRS + 1 + rng.below(40) as usize, (1 + rng.below(6) as i64, rng.below(7) as i64));
                let within = [fastest, (2 + rng.below(2) as usize, step(&mut rng))];
                let a = rng.below(3);
                affine_of(&within, &digits(&mut rng, a, 1))
            }
        };
        let (nodes, per_node) = (pairs.len(), pairs[0].len());
        let d = ImplicitLevel::fit(&level_of(&pairs)).unwrap();
        long += usize::from(d.within() > 4 && d.digits().len() - d.within() > 4);
        runs += usize::from(d.run.len() < per_node);
        wide += usize::from(d.run_digits == 0);
        for (i, node) in pairs.iter().enumerate() {
            let want = || node.iter().map(|&(l, r)| pair(l, r));
            assert!(d.places(i).eq(want()));
            let m = rng.below(per_node as u64) as usize;
            let mut places = d.places(i);
            places.next();
            assert!(places.clone().eq(want().skip(1)));
            let mut written = vec![pair(0, 0)];
            places.clone().write_into(&mut written);
            assert!(written[1..].iter().copied().eq(want().skip(1)));
            assert_eq!(places.nth(m), want().nth(m + 1));
            assert!(places.eq(want().skip(m + 2)));
        }
        let mut cursor = d.cursor();
        let mut i = 0;
        for _ in 0..3 * nodes {
            assert_eq!(cursor.first_of(i), d.node_first(i));
            assert_eq!(d.node_first(i), pairs[i][0]);
            i = match rng.below(4) {
                0 => rng.below(nodes as u64) as usize,
                1 => (i + 1 + rng.below(12) as usize).min(nodes - 1),
                _ => (i + 1).min(nodes - 1),
            };
        }
        let first = rng.below(nodes as u64 + 1) as usize;
        let mut flat = Vec::new();
        d.pairs_of_first(first, &mut flat);
        assert!(flat.into_iter().eq(pairs[..first].iter().flatten().map(|&(l, r)| pair(l, r))));
        // An implicit level's nodes read in order are the stored level's.
        let stored = level_of(&pairs);
        let mut level = TddLevel::new();
        level.pairs.describe(d.clone(), d.pairs(), d.nodes());
        let a = rng.below(nodes as u64) as usize;
        let b = a + rng.below((nodes - a) as u64 + 1) as usize;
        let read = |l: &TddLevel| -> Vec<(usize, Vec<ChildPair>)> {
            l.internal_inputs_range(a..b).map(|(i, p)| (i, p.collect())).collect()
        };
        assert_eq!(read(&level), read(&stored));
    }
    assert!(long > 20, "the levels of more than four place digits went unchecked");
    assert!(runs > 100 && wide > 50, "the nodes read in several runs went unchecked");
}

/// A [`NodeCursor`] on a level of more nodes than a run reads the first
/// pairs off the digits, in node order, ahead in jumps and back: runs of
/// the fastest node digits read off a table, and a fastest digit of more
/// places than a run read as one unit of it times the place.
#[test]
fn a_cursor_reads_the_first_pairs_in_runs() {
    let mut rng = Lcg::new(0x1d1e_0316);
    let (mut tabled, mut led) = (0, 0);
    for round in 0..40 {
        let across: Vec<(usize, (i64, i64))> = if round % 2 == 0 {
            (0..9 + rng.below(2)).map(|_| (2 + rng.below(2) as usize, step(&mut rng))).collect()
        } else {
            let lead = (RUN_PAIRS + 1 + rng.below(40) as usize, (1 + rng.below(6) as i64, rng.below(7) as i64));
            std::iter::once(lead).chain((0..1 + rng.below(2)).map(|_| (2 + rng.below(2) as usize, step(&mut rng)))).collect()
        };
        let within: Vec<(usize, (i64, i64))> = (0..rng.below(2)).map(|_| (2, (1, 1))).collect();
        let pairs = affine_of(&within, &across);
        let nodes = pairs.len();
        let d = ImplicitLevel::fit(&level_of(&pairs)).unwrap();
        let mut cursor = d.cursor();
        assert!(cursor.cycle < nodes, "round {round}: one run holds every node");
        tabled += usize::from(cursor.span == cursor.cycle);
        led += usize::from(cursor.span < cursor.cycle);
        let mut i = 0;
        for _ in 0..2 * nodes {
            assert_eq!(cursor.first_of(i), (pairs[i][0].0, pairs[i][0].1), "round {round}, node {i}");
            i = match rng.below(8) {
                0 => rng.below(nodes as u64) as usize,
                1 | 2 => (i + 1 + rng.below(3 * cursor.cycle as u64) as usize).min(nodes - 1),
                _ => (i + 1) % nodes,
            };
        }
    }
    assert!(tabled > 10 && led > 10, "{tabled} cursors in tabled runs, {led} with a lead digit");
}

/// A pass over every pair of an implicit level reads them a chunk of nodes
/// at a time ([`NodeChunks`], [`LevelPairs`],
/// [`TddLevel::try_for_each_node`]): the pairs and their nodes are the
/// stored level's, from any node, after any pairs read one at a time, over
/// any range of nodes up to an error, at one pair a node in runs of a
/// table and of a lead digit wider than a run, and at several.
#[test]
fn a_level_is_folded_in_runs_from_any_node() {
    let mut rng = Lcg::new(0x1d1e_0412);
    let (mut tabled, mut led, mut several) = (0, 0, 0);
    for round in 0..120 {
        let across: Vec<(usize, (i64, i64))> = match round % 3 {
            0 => (0..1 + rng.below(10)).map(|_| (2 + rng.below(2) as usize, step(&mut rng))).collect(),
            1 => {
                let lead = (RUN_PAIRS + 1 + rng.below(300) as usize, (1 + rng.below(6) as i64, rng.below(7) as i64));
                std::iter::once(lead).chain((0..rng.below(3)).map(|_| (2 + rng.below(2) as usize, step(&mut rng)))).collect()
            }
            _ => (0..1 + rng.below(5)).map(|_| (2 + rng.below(3) as usize, step(&mut rng))).collect(),
        };
        let within: Vec<(usize, (i64, i64))> = match round % 3 {
            2 => (0..1 + rng.below(3)).map(|_| (2 + rng.below(3) as usize, step(&mut rng))).collect(),
            _ => Vec::new(),
        };
        let pairs = affine_of(&within, &across);
        let nodes = pairs.len();
        let d = ImplicitLevel::fit(&level_of(&pairs)).unwrap();
        if d.pairs_per_node() == 1 {
            let cursor = d.cursor();
            tabled += usize::from(cursor.cycle < nodes && cursor.span == cursor.cycle);
            led += usize::from(cursor.span < cursor.cycle);
        } else {
            several += 1;
        }
        let want: Vec<(u32, ChildPair)> = pairs.iter().enumerate()
            .flat_map(|(i, node)| node.iter().map(move |&(l, r)| (i as u32, pair(l, r))))
            .collect();
        // From any node, a chunk at a time.
        let from = rng.below(nodes as u64 + 1) as usize;
        let (mut chunks, mut buf, mut written) = (NodeChunks::new(&d, from..nodes), Vec::new(), Vec::new());
        let k = d.pairs_per_node();
        while let Some(start) = chunks.fill(&mut buf) {
            assert!(!buf.is_empty() && buf.len() % k == 0, "round {round}: a chunk of whole nodes");
            assert_eq!(start, from + written.len() / k, "round {round}: chunks in order");
            written.extend(buf.chunks_exact(k).enumerate().flat_map(|(j, ps)| ps.iter().map(move |&p| ((start + j) as u32, p))));
        }
        let skip = from * k;
        assert_eq!(written, want[skip..], "round {round}: the chunks from node {from}");
        // After any pairs read one at a time.
        let mut read = LevelPairs::new(&d);
        let t = rng.below(want.len() as u64 + 1) as usize;
        let first: Vec<_> = read.by_ref().take(t).collect();
        assert_eq!(first, want[..t], "round {round}: the first {t} pairs read one at a time");
        assert_eq!(read.len(), want.len() - t);
        assert_eq!(read.clone().fold(Vec::new(), |mut v, p| {
            v.push(p);
            v
        }), want[t..], "round {round}: the fold after {t} pairs");
        assert!(read.eq(want[t..].iter().copied()));
        // A level that describes them reads as the stored level.
        let stored = level_of(&pairs);
        let mut level = TddLevel::new();
        level.pairs.describe(d.clone(), d.pairs(), d.nodes());
        assert!(level.pairs_with_parent().eq(stored.pairs_with_parent()));
        let each = |l: &TddLevel| {
            let mut v = Vec::new();
            l.for_each_node_pair(|i, p| v.push((i as u32, p)));
            v
        };
        assert_eq!(each(&level), want);
        assert_eq!(each(&stored), want);
        // Each node of a range with its pairs as a slice, up to an error.
        let a = rng.below(nodes as u64 + 1) as usize;
        let b = a + rng.below((nodes - a) as u64 + 1) as usize;
        let stop = (rng.below(3) == 0).then(|| a + rng.below((b - a) as u64 + 1) as usize);
        let slices = |l: &TddLevel| {
            let mut v = Vec::new();
            let r = l.try_for_each_node(a..b, |i, pairs| {
                if Some(i) == stop {
                    return Err(i);
                }
                v.push((i, pairs.to_vec()));
                Ok(())
            });
            (v, r)
        };
        let (got, end) = slices(&level);
        assert_eq!(end, stop.filter(|&i| i < b).map_or(Ok(()), Err), "round {round}: the pass over {a}..{b} ends");
        assert_eq!((got, end), slices(&stored), "round {round}: the nodes {a}..{b}");
    }
    assert!(tabled > 10 && led > 10 && several > 10, "{tabled} tabled, {led} led, {several} of several pairs a node");
}

/// A description's normal form, its digits each merged into the one before
/// it where its step is on that one's line, is the greedy read of the pairs
/// it describes: over digits of random radices whose steps are on the line
/// of the digits before them or not, zero steps among them.
#[test]
fn the_normal_form_merges_the_digits_on_a_line() {
    let mut rng = Lcg::new(0x1d1e_0501);
    let mut merges = 0;
    for round in 0..2000 {
        let digits = |rng: &mut Lcg, n: u64| {
            let mut out: Vec<(usize, (i64, i64))> = Vec::new();
            let mut line: Option<(usize, (i64, i64))> = None;
            for _ in 0..n {
                let radix = 2 + rng.below(4) as usize;
                let step = match (line, rng.below(3)) {
                    (Some((places, (l, r))), 0) => (places as i64 * l, places as i64 * r),
                    (Some((places, (l, _))), 1) => (places as i64 * l, rng.below(3) as i64),
                    _ => (rng.below(5) as i64 - 1, rng.below(5) as i64 - 1),
                };
                line = match line {
                    Some((places, s)) if step == (places as i64 * s.0, places as i64 * s.1) => Some((places * radix, s)),
                    _ => Some((radix, step)),
                };
                out.push((radix, step));
            }
            out
        };
        let (w, a) = (rng.below(4), rng.below(5));
        let within = digits(&mut rng, w);
        let across = digits(&mut rng, a);
        let (k, n) = (within.iter().map(|d| d.0).product(), across.iter().map(|d| d.0).product());
        let first = (20 + rng.below(9) as i64, 20 + rng.below(9) as i64);
        let d = ImplicitLevel::assemble(n, k, first, &within, &across);
        let normal = d.normal();
        assert_eq!(normal, d.read_normal(), "round {round}: {d:?}");
        assert_eq!(normal.normal(), normal);
        merges += usize::from(normal.digits().len() < d.digits().len());
    }
    assert!(merges > 500, "{merges} descriptions had digits to merge");
}

#[test]
fn twins_are_read_off_the_digits() {
    // Node i holds (3i + m, m) for m < 3: every left slot named once, in its
    // own context, and every right slot named in two contexts of its own.
    let pairs: Pairs = (0..2).map(|i| (0..3).map(|m| (3 * i + m, m)).collect()).collect();
    let d = ImplicitLevel::fit(&level_of(&pairs)).unwrap();
    assert!(d.twin_free(ChildSide::Left, 6));
    assert!(d.twin_free(ChildSide::Right, 3));
    // Slots that do not cover the child: not shown.
    assert!(!d.twin_free(ChildSide::Left, 7));
    // Node i holds (3i + m, 0): the left slots of a node share its one
    // context, so they are twins, and the digits must not say otherwise.
    let pairs: Pairs = (0..2).map(|i| (0..3).map(|m| (3 * i + m, 0)).collect()).collect();
    let d = ImplicitLevel::fit(&level_of(&pairs)).unwrap();
    assert!(!d.twin_free(ChildSide::Left, 6));
    // Two left slots each with right slots 0 and 1 in one node: twins.
    let pairs: Pairs = vec![vec![(0, 0), (1, 0), (0, 1), (1, 1)]];
    let d = ImplicitLevel::fit(&level_of(&pairs)).unwrap();
    assert!(!d.twin_free(ChildSide::Left, 2));
}

/// Where the digits say a child has no twins, it has none: every slot of
/// the child is named, and no two share the multiset of their contexts,
/// the pairs' nodes with their slots on the other side. On levels whose
/// slots on one side count a random subset of the digits in a mixed radix,
/// so that they cover the child, and whose other side steps at random.
#[test]
fn a_child_the_digits_call_twin_free_has_no_twins() {
    let mut rng = Lcg::new(0x1d1e_0008);
    let (mut free, mut twins) = (0, 0);
    for _ in 0..2000 {
        let radices = |rng: &mut Lcg| -> Vec<usize> { (0..rng.below(4)).map(|_| 2 + rng.below(3) as usize).collect() };
        let (within, across) = (radices(&mut rng), radices(&mut rng));
        if within.is_empty() {
            continue;
        }
        // The counted side's steps: each chosen digit, in a random order, the
        // product of the radices of those before it.
        let mut steps: Vec<(i64, i64)> = Vec::new();
        let mut width = 1usize;
        let mut order: Vec<usize> = (0..within.len() + across.len()).collect();
        for j in (1..order.len()).rev() {
            order.swap(j, rng.below(j as u64 + 1) as usize);
        }
        steps.resize(order.len(), (0, 0));
        for j in order {
            let radix = if j < within.len() { within[j] } else { across[j - within.len()] };
            let counted = if rng.below(3) > 0 {
                let s = width as i64;
                width *= radix;
                s
            } else {
                0
            };
            steps[j] = (counted, rng.below(5) as i64);
        }
        let digits = |radices: &[usize], steps: &[(i64, i64)]| radices.iter().copied().zip(steps.iter().copied()).collect::<Vec<_>>();
        let pairs = affine_of(&digits(&within, &steps[..within.len()]), &digits(&across, &steps[within.len()..]));
        let flip = rng.below(2) == 1;
        let pairs: Pairs = if flip { pairs.iter().map(|p| p.iter().map(|&(l, r)| (r, l)).collect()).collect() } else { pairs };
        let side = if flip { ChildSide::Right } else { ChildSide::Left };
        let Some(d) = ImplicitLevel::fit(&level_of(&pairs)) else { continue };
        let mut contexts = vec![Vec::new(); width];
        for (i, node) in pairs.iter().enumerate() {
            for &(l, r) in node {
                let (t, other) = if flip { (r, l) } else { (l, r) };
                contexts[t as usize].push((i, other));
            }
        }
        for c in &mut contexts {
            c.sort_unstable();
        }
        let no_twins = contexts.iter().all(|c| !c.is_empty()) && {
            let mut sorted = contexts.clone();
            sorted.sort();
            sorted.windows(2).all(|w| w[0] != w[1])
        };
        if d.twin_free(side, width) {
            assert!(no_twins, "twins under a level the digits call twin free: {pairs:?}");
            free += 1;
        } else if !no_twins {
            twins += 1;
        }
    }
    assert!(free > 200 && twins > 200, "too few cases either way: {free} twin free, {twins} with twins");
}

/// The divisor search gives what counting down from `m` gives, at every
/// `m` of every `n` up to 400.
#[test]
fn divisor_at_most_counts_down() {
    for n in 1..=400usize {
        for m in 1..=n + 1 {
            let mut d = m.min(n);
            while d > 1 && !n.is_multiple_of(d) {
                d -= 1;
            }
            assert_eq!(divisor_at_most(n, m), d, "n {n}, m {m}");
        }
    }
}

/// Node 0's check answers as a reading of every place off its digits by
/// division does, on affine nodes and on nodes with one pair moved.
#[test]
fn digits_hold_reads_every_place() {
    let at = |digits: &[(usize, (i64, i64))], mut m: usize| {
        digits.iter().fold((0, 0), |(l, r), &(radix, (dl, dr))| {
            let c = (m % radix) as i64;
            m /= radix;
            (l + c * dl, r + c * dr)
        })
    };
    let mut rng = Lcg::new(7);
    let (mut held, mut failed) = (0, 0);
    for _ in 0..400 {
        let per_node = [2, 4, 6, 12, 30, 64, 90][rng.below(7) as usize];
        let mut node = affine(&mut rng, 1, per_node).swap_remove(0);
        if rng.below(2) == 0 {
            node[1 + rng.below(per_node as u64 - 1) as usize].0 += 1;
        }
        let offset = |m: usize| Some((node[m].0 - node[0].0, node[m].1 - node[0].1));
        let Some(within) = read_digits(per_node, offset) else { continue };
        let every = (0..per_node).all(|m| offset(m) == Some(at(&within, m)));
        assert_eq!(digits_hold(&within, per_node, offset), every, "{node:?}");
        if every { held += 1 } else { failed += 1 }
    }
    assert!(held > 0 && failed > 0, "held {held}, failed {failed}");
}

/// A node of 2^31 pairs, each a slot further on the left and all at one
/// slot on the right, has as many offsets on the left: they are counted
/// without being written, and refused room is an error, not an abort.
#[test]
fn the_offsets_of_a_node_of_billions_of_pairs_are_counted_and_reserved() {
    let pairs = 1usize << 31;
    let d = ImplicitLevel::assemble(1, pairs, (0, 0), &[(pairs, (1, 0))], &[]);
    assert_eq!(d.side_offset_count(ChildSide::Left), pairs);
    assert_eq!(d.side_offset_count(ChildSide::Right), 1);
    let lim = crate::limits::Limits::new();
    let mut out = Vec::new();
    d.side_offsets(ChildSide::Right, &lim, &mut out).expect("one offset");
    assert_eq!(out, [0]);
    lim.refuse_nth_reserve(0);
    let mut out = Vec::new();
    assert_eq!(d.side_offsets(ChildSide::Left, &lim, &mut out), Err(crate::limits::OperationError::OverBudget));
    assert_eq!(out.capacity(), 0);
}

/// The pairs of `level`, node by node, as child slots, read whatever its
/// form.
fn pairs_of(level: &TddLevel) -> Pairs {
    (0..level.nodes().len())
        .map(|i| level.pairs_vec(i).iter().map(|p| (i64::from(p.left.raw()), i64::from(p.right.raw()))).collect())
        .collect()
}

/// A level held as the description of its pairs is built stored in room
/// reserved before any of it is written: a refused node or pair arena
/// leaves the level described as it was, and the granted room holds the
/// stored level. The room is not charged, before a refusal or after: the
/// meters count the capacities the description stands for already.
#[test]
fn a_refused_room_leaves_the_description() {
    let pairs = affine_of(&[(4, (1, 0))], &[(16, (4, 0)), (8, (0, 1))]);
    let stored = level_of(&pairs);
    let mut level = stored.clone();
    level.close();
    let d = level.implicit().expect("an affine level closes").clone();
    let held = (level.pairs.len(), level.pairs.capacity(), level.node_capacity());
    for n in 0..2 {
        let lim = Limits::new();
        lim.refuse_nth_reserve(n);
        assert_eq!(level.store_if_implicit(&lim), Err(OperationError::OverBudget), "reserve {n}");
        assert_eq!(level.implicit(), Some(&d), "reserve {n}");
        assert_eq!((level.pairs.len(), level.pairs.capacity(), level.node_capacity()), held);
        assert_eq!(lim.meters().in_flight_bytes, 0, "reserve {n}");
    }
    let lim = Limits::new();
    level.store_if_implicit(&lim).unwrap();
    assert_eq!(lim.meters().in_flight_bytes, 0);
    assert!(level.implicit().is_none());
    assert_eq!(pairs_of(&level), pairs);
    assert!(level.nodes().iter().eq(stored.nodes().iter()));
    assert_eq!((level.pairs.len(), level.pairs.capacity(), level.node_capacity()), held);
}

/// A rewrite that changes a pair of a level held as the description of its
/// pairs, refused the room the level is built stored in, leaves it as it
/// was; one that changes nothing asks for no room.
#[test]
fn a_refused_rewrite_leaves_the_description() {
    let pairs = affine_of(&[(4, (1, 0))], &[(16, (4, 0)), (8, (0, 1))]);
    let mut level = level_of(&pairs);
    level.close();
    let d = level.implicit().expect("an affine level closes").clone();
    let lim = Limits::new();
    lim.refuse_nth_reserve(0);
    assert_eq!(level.rewrite_described(&lim, true, |_, _, _, p| Some(p)), Ok(false), "an identity rewrite");
    assert_eq!(level.implicit(), Some(&d));
    assert_eq!(level.rewrite_described(&lim, true, |i, r, _, p| (i != 5 || r != 2).then_some(p)), Err(OperationError::OverBudget));
    assert_eq!(level.implicit(), Some(&d));
    lim.refuse_nth_reserve(0);
    let dropped = level.pairs_vec(7)[1];
    assert_eq!(level.remove_pair_from_node(&lim, 7, dropped), Err(OperationError::OverBudget));
    assert_eq!(level.implicit(), Some(&d));
    assert_eq!(lim.meters().in_flight_bytes, 0);
    assert_eq!(level.remove_pair_from_node(&lim, 7, dropped), Ok(true));
    let mut want = pairs.clone();
    want[7].remove(1);
    assert_eq!(pairs_of(&level), want);
}

/// The lists a description is checked with, a node's offsets and the
/// slots a move reads, are reserved: a refusal is an error and leaves the
/// level as it was. Where the place digits show a node's pairs distinct,
/// no list is made.
#[test]
fn the_lists_a_description_is_checked_with_are_reserved() {
    let lim = Limits::new();
    // Offsets 0, 1, 1, 2 on the left: a node repeats a pair.
    let repeating = ImplicitLevel::assemble(4, 4, (0, 0), &[(2, (1, 0)), (2, (1, 0))], &[(4, (3, 0))]);
    lim.refuse_nth_reserve(0);
    assert_eq!(repeating.repeats_a_pair(&lim), Err(OperationError::OverBudget));
    assert_eq!(repeating.repeats_a_pair(&lim), Ok(true));
    let distinct = ImplicitLevel::assemble(4, 4, (0, 0), &[(2, (1, 0)), (2, (2, 0))], &[(4, (4, 0))]);
    lim.refuse_nth_reserve(0);
    assert_eq!(distinct.repeats_a_pair(&lim), Ok(false));
    assert_eq!(lim.reserve_exact(&mut Vec::<u8>::new(), 1), Err(OperationError::OverBudget), "nothing was listed");
    lim.refuse_nth_reserve(0);
    assert_eq!(distinct.pruned(&lim, 2, |j| (j < 2).then_some(j), || 0..2, |x| x, |x| x), Err(OperationError::OverBudget));
    lim.grant_every_reserve();

    // A move one to one and affine keeps the description; one that is not
    // stores the level. Every reservation either takes, refused, is an
    // error that leaves the level as it was.
    let pairs = affine_of(&[(4, (1, 0))], &[(16, (4, 0)), (8, (0, 1))]);
    for (name, f) in [("one to one", (|x: i64| 2 * x) as fn(i64) -> i64), ("merging", |x: i64| x / 2)] {
        let mut level = level_of(&pairs);
        level.close();
        let d = level.implicit().expect("an affine level closes").clone();
        let mut refused = 0;
        loop {
            let lim = Limits::new();
            lim.refuse_nth_reserve(refused);
            let mut moved = level.clone();
            match moved.move_described(&lim, ChildSide::Left, f) {
                Err(e) => {
                    assert_eq!(e, OperationError::OverBudget);
                    assert_eq!(moved.implicit(), Some(&d), "{name}, reserve {refused}");
                    assert_eq!(lim.meters().in_flight_bytes, 0, "{name}, reserve {refused}");
                    refused += 1;
                }
                Ok(()) => {
                    let want: Pairs = pairs.iter().map(|node| node.iter().map(|&(l, r)| (f(l), r)).collect()).collect();
                    assert_eq!(pairs_of(&moved), want, "{name}");
                    assert_eq!(moved.implicit().is_some(), name == "one to one");
                    break;
                }
            }
        }
        assert!(refused >= 2, "{name}: {refused} reservations");
    }
}
