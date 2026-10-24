//! The sparse route's walks against the grid: random operands over random
//! subsets of the variables, so that leaves hold One as well as literals, on
//! every vtree shape, conjoined with every level forced onto the sparse route
//! (its walks asking the cache ahead or not, and its walks of f's nodes
//! handing every node's candidates to the emit as a list or only those of
//! large nodes) and with none. All give the same levels, the same minimized
//! diagram and the same count.

use std::sync::Arc;

use super::{ForcedPrefetch, ForcedThresholds, SparseThresholds};
use crate::apply::conjoin::{conjoin_checked, VtreeMask};
use crate::diagram::Tdd;
use crate::test_helpers::{assert_canonical, normalized_levels, rand_conj_over, vtree_shapes, Lcg};
use crate::vtree::Vtree;
use crate::Engine;

/// Every level of every conjunction takes the sparse route.
const ALWAYS: SparseThresholds = SparseThresholds { min_grid: 1, sparsity_factor: 1, ..SparseThresholds::PRODUCTION };

/// [`ALWAYS`], with every f node a walk of f's nodes visits handing its
/// candidates to the emit as a list.
const ALWAYS_COLLECTED: SparseThresholds = SparseThresholds { collected_pairs: 1, ..ALWAYS };

/// No level takes it.
const NEVER: SparseThresholds = SparseThresholds { min_grid: usize::MAX, ..SparseThresholds::PRODUCTION };

fn and_under(eng: &Engine, thresholds: SparseThresholds, f: &Tdd, g: &Tdd) -> Tdd {
    let _forced = ForcedThresholds::install(thresholds);
    conjoin_checked(eng, f.clone(), g.clone(), VtreeMask::new(None), VtreeMask::default())
        .expect("an unarmed engine refuses nothing").0
}

fn count_under(eng: &Engine, thresholds: SparseThresholds, f: &Tdd, g: &Tdd) -> num_bigint::BigUint {
    let _forced = ForcedThresholds::install(thresholds);
    eng.and_model_count(f.clone(), g.clone(), &[]).expect("an unarmed engine refuses nothing")
}

/// A random subset of `1..=n` of at least two variables.
fn some_vars(n: u32, rng: &mut Lcg) -> Vec<u32> {
    loop {
        let vars: Vec<u32> = (1..=n).filter(|_| rng.below(3) != 0).collect();
        if vars.len() >= 2 {
            return vars;
        }
    }
}

/// A random conjunction of clauses over a random subset of the variables,
/// minimized.
fn operand(vtree: &Arc<Vtree>, n: u32, rng: &mut Lcg) -> Tdd {
    let vars = some_vars(n, rng);
    let mut t = rand_conj_over(vtree, &vars, 8, 4, false, rng);
    t.minimize().unwrap();
    t
}

/// Conjoin both ways round under every threshold, the sparse walks asking
/// the cache ahead or not, and require the same diagram, level by level and
/// minimized, and the same count.
fn same_as_grid(eng: &Engine, what: &str, f: &Tdd, g: &Tdd) {
    for (a, b) in [(f, g), (g, f)] {
        let dense = and_under(eng, NEVER, a, b);
        let mut minimized = dense.clone();
        eng.minimize(&mut minimized).unwrap();
        for (gate, always) in [(Some(false), ALWAYS), (Some(true), ALWAYS), (Some(false), ALWAYS_COLLECTED)] {
            let _pinned = ForcedPrefetch::install(gate);
            let how = format!("gate {gate:?}, collected from {} pairs", always.collected_pairs);
            let sparse = and_under(eng, always, a, b);
            assert_eq!(normalized_levels(&sparse), normalized_levels(&dense), "{what}, {how}: levels");
            let mut sparse = sparse;
            eng.minimize(&mut sparse).unwrap();
            assert_canonical(&sparse);
            assert_eq!(normalized_levels(&sparse), normalized_levels(&minimized), "{what}, {how}: minimized");
            assert_eq!(count_under(eng, always, a, b), count_under(eng, NEVER, a, b), "{what}, {how}: count");
        }
    }
}

#[test]
fn sparse_walks_build_the_grids_levels() {
    let mut rng = Lcg::new(0x1eaf_1abe);
    let eng = Engine::new();
    for n in [6, 9] {
        for (shape, vtree) in vtree_shapes(n) {
            for round in 0..10 {
                let f = operand(&vtree, n, &mut rng);
                let g = operand(&vtree, n, &mut rng);
                same_as_grid(&eng, &format!("{n} vars, {shape}, round {round}"), &f, &g);
            }
        }
    }
}
