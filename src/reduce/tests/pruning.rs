//! Pruning.
//!
//! The fixtures these read are in `mod.rs`.

use std::sync::Arc;

use crate::reduce::prune::{PruneScope, below_root_walk_applies, prune_unreachable};

use crate::test_helpers::compile_clauses;
use crate::vtree::Vtree;


/// The prune's one diagram-proportional scratch buffer is reserved through the
/// engine, so both halves of the tracked reserve reach it: the allocation-
/// failure injection refuses it, and a granted reservation is charged to the
/// in-flight byte meter.
#[test]
fn the_prune_scratch_reservation_goes_through_the_engine() {
    let vtree = Arc::new(Vtree::balanced(4));
    let clauses = [vec![1, 2], vec![-2, 3], vec![3, -4]];

    // Refused: the diagram is left exactly as it was.
    let eng = &crate::Engine::new();
    let mut tdd = compile_clauses(&vtree, &clauses);
    tdd.minimize().unwrap();
    let (size_before, mc_before) = (tdd.pair_count(), tdd.model_count().unwrap());
    eng.limits().refuse_nth_reserve(0);
    let refused = prune_unreachable(eng, &mut tdd, PruneScope::Whole);
    eng.limits().grant_every_reserve();
    assert!(refused.is_err(), "the armed injection must refuse the scratch reservation");
    assert_eq!(tdd.pair_count(), size_before, "a refused prune must not touch the diagram");
    assert_eq!(tdd.model_count().unwrap(), mc_before, "a refused prune must not touch the count");

    // Granted: the bytes the scratch takes are charged.
    let eng = &crate::Engine::new();
    let mut tdd = compile_clauses(&vtree, &clauses);
    tdd.minimize().unwrap();
    eng.limits().reset_meters();
    prune_unreachable(eng, &mut tdd, PruneScope::Whole).expect("a tiny scratch reservation cannot fail");
    assert!(
        eng.limits().meters().in_flight_bytes > 0,
        "the scratch reservation must be charged against the byte budget",
    );
}

/// The nodes of each structural level the output reaches, by a walk that
/// shares nothing with the prune: index `t` holds level `t`'s reached slots.
fn reached_by_level(tdd: &crate::Tdd) -> Vec<std::collections::BTreeSet<usize>> {
    let vtree = &tdd.vtree;
    let mut reached = vec![std::collections::BTreeSet::new(); vtree.num_nodes()];
    reached[tdd.output.vtree.idx()].insert(tdd.output.local.idx());
    for &t in vtree.bottomup_slice().iter().rev() {
        if !tdd.is_structural_internal(t) {
            continue;
        }
        let (left, right) = vtree.children(t);
        let (lv, rv) = (tdd.levels[left.idx()].child_decoder(), tdd.levels[right.idx()].child_decoder());
        let nodes: Vec<usize> = reached[t.idx()].iter().copied().collect();
        for i in nodes {
            for pair in tdd.levels[t.idx()].pairs_of_idx(i) {
                if let Some(s) = lv.child(pair.left).index() {
                    reached[left.idx()].insert(s);
                }
                if let Some(s) = rv.child(pair.right).index() {
                    reached[right.idx()].insert(s);
                }
            }
        }
    }
    reached
}

/// Every node's pairs, level by level, and the output: two diagrams agree
/// here iff they are the same diagram.
fn layout(tdd: &crate::Tdd) -> Vec<Vec<Vec<(u32, u32)>>> {
    let mut out = vec![vec![vec![(tdd.output.vtree.0, tdd.output.local.0)]]];
    for level in &tdd.levels {
        if level.is_marginal() {
            continue;
        }
        out.push(
            (0..level.slot_count())
                .map(|i| level.pairs_of_idx(i).iter().map(|p| (p.left.0, p.right.0)).collect())
                .collect(),
        );
    }
    out
}

/// A conjunction leaves nodes no parent names, on levels wide enough that
/// their marks take several words. The walk down from the root through the
/// levels the conjunction built and the walk over the whole diagram leave the
/// same diagram, and it holds exactly the nodes an independent walk reaches.
#[test]
fn the_walk_from_the_root_and_the_whole_walk_keep_exactly_the_reached_nodes() {
    use crate::test_helpers::{CnfShape, Lcg, compile_clauses_on, rand_cnf, vtree_shapes};

    let eng = &crate::Engine::new();
    let mut rng = Lcg::new(0x7b31_c4e9);
    let (mut cases, mut seeded, mut wide_losses) = (0usize, 0usize, 0usize);
    for shape in [CnfShape { clauses: 16, width: 6 }, CnfShape { clauses: 32, width: 8 }] {
        for (_name, vtree) in vtree_shapes(20) {
            for _ in 0..6 {
                let ca = rand_cnf(&mut rng, 20, shape);
                let cb = rand_cnf(&mut rng, 20, shape);
                let f = compile_clauses_on(eng, &vtree, &ca);
                let g = compile_clauses_on(eng, &vtree, &cb);
                if f.is_zero() || g.is_zero() {
                    continue;
                }
                let conj = crate::and(f, g).unwrap();
                if conj.is_zero() || conj.dirty.loose().is_none() || conj.has_marginal_level() {
                    continue;
                }
                let reached = reached_by_level(&conj);
                for &t in conj.vtree.bottomup_slice() {
                    let width = conj.levels[t.idx()].slot_count();
                    if conj.is_structural_internal(t) && width > 64 && reached[t.idx()].len() < width {
                        wide_losses += 1;
                    }
                }
                let count = conj.model_count().unwrap();
                seeded += usize::from(below_root_walk_applies(&conj));

                let mut walked = conj.clone();
                prune_unreachable(eng, &mut walked, PruneScope::Whole).unwrap();
                let mut whole = conj;
                whole.dirty.set_loose(None);
                prune_unreachable(eng, &mut whole, PruneScope::Whole).unwrap();

                assert_eq!(layout(&walked), layout(&whole), "the two walks left different diagrams");
                for &t in walked.vtree.bottomup_slice() {
                    if walked.is_structural_internal(t) {
                        assert_eq!(
                            walked.levels[t.idx()].slot_count(),
                            reached[t.idx()].len(),
                            "level {} keeps other than the nodes the output reaches",
                            t.0
                        );
                    }
                }
                assert_eq!(walked.model_count().unwrap(), count);
                cases += 1;
            }
        }
    }
    assert!(cases > 40, "expected a corpus, got {cases} cases");
    assert!(seeded > 40, "expected the walk from the root on most cases, got {seeded}");
    assert!(wide_losses > 30, "expected losses on levels of several words, got {wide_losses}");
}
