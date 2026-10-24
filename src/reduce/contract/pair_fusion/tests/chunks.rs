//! A boundary fused a chunk of plans at a time leaves the level and the
//! store one chunk leaves, pair for pair and slot for slot.

use super::*;
use crate::test_helpers::{compile_clauses_on, rand_cnf, vtree_shapes, CnfShape};
use crate::vtree::rng::Lcg;

/// A level's nodes as their pairs' raw refs, left then right, and its count
/// store.
type LevelImage = (Vec<Vec<(u32, u32)>>, Option<Vec<u128>>);

/// Every level's nodes, pair for pair, and every count store, slot for slot.
fn levels(tdd: &Tdd) -> Vec<LevelImage> {
    let mut buf = Vec::new();
    tdd.levels
        .iter()
        .map(|level| {
            let nodes = match level.is_marginal() {
                true => Vec::new(),
                false => (0..level.nodes().len())
                    .map(|n| level.pairs_read(n, &mut buf).iter().map(|p| (p.left.0, p.right.0)).collect())
                    .collect(),
            };
            (nodes, level.marginal_counts().map(<[u128]>::to_vec))
        })
        .collect()
}

/// Random formulas on every vtree shape, a level marginalized at a time and
/// minimized after each, fused one plan's node a chunk and in one chunk: the
/// diagrams are the same after every step, and some step fused more than one
/// chunk at a boundary.
#[test]
fn a_level_fused_in_chunks_is_the_level_fused_whole() {
    let mut rng = Lcg::new(31);
    let chunked_before = chunked_boundaries();
    for num_vars in 4..9u32 {
        for (shape, vtree) in vtree_shapes(num_vars) {
            let eng = Engine::new();
            let clauses = rand_cnf(&mut rng, num_vars, CnfShape { clauses: 2 * num_vars as usize, width: 3 });
            let mut chunked = compile_clauses_on(&eng, &vtree, &clauses);
            if chunked.is_zero() {
                continue;
            }
            let mut whole = chunked.clone();
            let below_root: Vec<VtreeIdx> =
                (0..vtree.num_nodes() as u32).map(VtreeIdx).filter(|&t| t != vtree.root()).collect();
            for step in 0..4 {
                let target = below_root[rng.below(below_root.len() as u64) as usize];
                let prior = pin_chunk_plans(Some(1));
                eng.marginalize_levels(&mut chunked, &[target]).unwrap();
                eng.minimize(&mut chunked).unwrap();
                pin_chunk_plans(Some(usize::MAX));
                eng.marginalize_levels(&mut whole, &[target]).unwrap();
                eng.minimize(&mut whole).unwrap();
                pin_chunk_plans(prior);
                assert!(levels(&chunked) == levels(&whole), "{shape} over {num_vars} variables, step {step}");
            }
        }
    }
    assert!(chunked_boundaries() > chunked_before, "no boundary was fused in two chunks");
}
