use super::*;
use crate::vtree::rng::Lcg;
use crate::Engine;

/// Groups of cells as the Boolean inner build leaves them: each inner node
/// opened by its first group in `group_info`'s order, the groups that share a
/// node carrying the same cell list, every list ascending and distinct, and
/// the groups' runs of triples in another order than the groups'.
fn clustered_groups(rng: &mut Lcg, n_v: u32, axes: &[u32]) -> (Vec<u128>, Vec<PairGroup>) {
    let nodes = 1 + rng.next_u64() % 12;
    let mut lists: Vec<Vec<u64>> = Vec::new();
    for _ in 0..nodes {
        let mut cells: Vec<u64> = (0..1 + rng.next_u64() % 9)
            .map(|_| {
                let src = rng.next_u64() % u64::from(n_v);
                let axis = axes[(rng.next_u64() % axes.len() as u64) as usize];
                (src << 32) | u64::from(axis)
            })
            .collect();
        // In a group's order: by axis, then by src.
        cells.sort_unstable_by_key(|&c| (c as u32, (c >> 32) as u32));
        cells.dedup();
        lists.push(cells);
    }
    // Each node's groups: the first opens it, the rest join it later on.
    let mut order: Vec<usize> = Vec::new();
    for node in 0..lists.len() {
        let extra = (rng.next_u64() % 3) as usize;
        let at = order.len();
        order.push(node);
        for _ in 0..extra {
            let pos = at + 1 + (rng.next_u64() as usize) % (order.len() - at);
            order.insert(pos.min(order.len()), node);
        }
    }
    let mut triples: Vec<u128> = Vec::new();
    let mut groups: Vec<PairGroup> = Vec::new();
    for (k, &node) in order.iter().enumerate().rev() {
        let start = triples.len() as u32;
        let inner = ChildPair::new(NodeIdx(k as u32), NodeIdx(0));
        for &cell in &lists[node] {
            triples.push(u128::pack(&Layout::WIDE, inner, (cell >> 32) as u32, EncodedChildRef::from_raw(cell as u32)));
        }
        groups.push(PairGroup { hash: node as u64, inner, start, end: triples.len() as u32, node: NodeIdx(node as u32) });
    }
    groups.reverse();
    (triples, groups)
}

/// The outer lists as filing every group's cells, sorting each list and
/// dropping repeats made them.
fn sorted_and_deduplicated(triples: &[u128], groups: &[PairGroup], n_v: usize, dir: RotationKind) -> Vec<Vec<ChildPair>> {
    let wide = &Layout::WIDE;
    let mut lists = vec![Vec::new(); n_v];
    for g in groups {
        for &p in &triples[g.start as usize..g.end as usize] {
            let pair = match dir {
                RotationKind::Left => ChildPair::new(g.node, tri_axis(p, wide)),
                RotationKind::Right => ChildPair::new(tri_axis(p, wide), g.node),
            };
            lists[tri_src(p, wide) as usize].push(pair);
        }
    }
    for list in &mut lists {
        list.sort_unstable();
        list.dedup();
    }
    lists
}

/// The outer lists `file_outer_pairs` files from `triples` under `layout`.
fn filed<W: Word>(triples: &[W], layout: &Layout, groups: &[PairGroup], n_v: usize, dir: RotationKind) -> Vec<Vec<ChildPair>> {
    let eng = Engine::new();
    let mut scratch = RestructureScratch { group_info: groups.to_vec(), ..Default::default() };
    file_outer_pairs(eng.limits(), triples, layout, &mut scratch, n_v, dir, false).unwrap();
    let mut begin = 0;
    let mut lists = Vec::new();
    for &end in &scratch.outer_ends[..n_v] {
        lists.push(scratch.outer_pairs[begin as usize..end as usize].to_vec());
        begin = end;
    }
    lists
}

/// Filing only each node's first group, in order of the axes where they are
/// dense and by sorting where they are sparse, gives each old v-node the pairs
/// that filing every group, sorting and dropping repeats gave, from wide
/// triples and from the same triples in a fitted word.
#[test]
fn the_filed_outer_lists_are_the_sorted_distinct_pairs() {
    let mut rng = Lcg::new(5);
    let dense: Vec<u32> = (0..10).collect();
    let sparse: Vec<u32> = (0..10).map(|a| 1000 * a + 7).collect();
    for round in 0..200 {
        let n_v = 1 + (round % 6) as u32;
        let axes = if round % 2 == 0 { &dense } else { &sparse };
        let (triples, groups) = clustered_groups(&mut rng, n_v, axes);
        // A v-node no cell names has no pairs; in a rotation every one has some.
        let wide = &Layout::WIDE;
        let named: std::collections::BTreeSet<u32> = triples.iter().map(|&p| tri_src(p, wide)).collect();
        if named.len() != n_v as usize {
            continue;
        }
        let mut bound = [0u32; 4];
        for &p in &triples {
            for (k, b) in bound.iter_mut().enumerate() {
                *b |= p.field(wide, k);
            }
        }
        let fitted = Layout::fitted(bound);
        assert!(fitted.bits() <= 64, "round {round}: the test's fields fit one word");
        let narrow: Vec<u64> = triples.iter().map(|&p| u64::pack(&fitted, tri_inner(p, wide), tri_src(p, wide), tri_axis(p, wide))).collect();
        for dir in [RotationKind::Left, RotationKind::Right] {
            let want = sorted_and_deduplicated(&triples, &groups, n_v as usize, dir);
            assert_eq!(filed(&triples, wide, &groups, n_v as usize, dir), want, "round {round}, {dir:?}, wide");
            assert_eq!(filed(&narrow, &fitted, &groups, n_v as usize, dir), want, "round {round}, {dir:?}, fitted");
        }
    }
}
