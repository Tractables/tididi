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
            assert_eq!(out, level.pairs_of_idx(i));
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
fn a_range_of_pairs_is_read_where_it_lies() {
    let mut rng = Lcg::new(0x1d1e_0003);
    for _ in 0..100 {
        let pairs = random_affine(&mut rng, 8, 8, 2);
        let d = ImplicitLevel::fit(&level_of(&pairs)).unwrap();
        let flat: Vec<ChildPair> = pairs.iter().flatten().map(|&(l, r)| pair(l, r)).collect();
        let a = rng.below(flat.len() as u64) as usize;
        let b = a + rng.below((flat.len() - a) as u64 + 1) as usize;
        let mut out = Vec::new();
        d.write_range(a..b, &mut out);
        assert_eq!(out, &flat[a..b]);
    }
}

#[test]
fn an_implicit_arena_reads_and_writes_as_the_written_one() {
    let mut rng = Lcg::new(0x1d1e_0004);
    for _ in 0..50 {
        let pairs = random_affine(&mut rng, 8, 8, 2);
        let explicit = level_of(&pairs);
        let d = ImplicitLevel::fit(&explicit).unwrap();
        let mut level = TddLevel::new();
        d.write_nodes(&Limits::new(), &mut level).unwrap();
        level.pairs.describe(d.clone(), 3 * d.pairs());
        assert_eq!(level.nodes, explicit.nodes);
        assert_eq!(level.implicit(), Some(&d));
        assert_eq!(level.pairs.len(), explicit.pairs.len());
        assert_eq!(level.pairs.capacity(), 3 * d.pairs());
        // A reader sees the pairs; the description stays.
        assert_eq!(&*level.pairs, &*explicit.pairs);
        assert!(level.pairs.implicit().is_some());
        let mut buf = Vec::new();
        for i in 0..level.nodes.len() {
            assert_eq!(level.pairs_read(i, &mut buf), explicit.pairs_of_idx(i));
        }
        // A writer gets them written, at the capacity held.
        let mut written = level.clone();
        written.pairs.push(pair(0, 0));
        assert!(written.pairs.implicit().is_none());
        assert_eq!(written.pairs.capacity(), 3 * d.pairs());
        written.pairs.pop();
        assert_eq!(written.pairs, explicit.pairs);
        // Clearing keeps the capacity, written.
        level.clear();
        assert!(level.pairs.is_empty() && level.pairs.implicit().is_none());
        assert_eq!(level.pairs.capacity(), 3 * d.pairs());
    }
    assert!(materialized().iter().any(|m| m.copied > 0 && m.reader.file().ends_with("implicit.rs")));
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
