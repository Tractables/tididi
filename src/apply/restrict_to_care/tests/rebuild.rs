use std::sync::Arc;

use super::*;
use crate::apply::restrict_to_care::pairs::PairMarks;
use crate::diagram::{TddNodeId, POS_LEAF_IDX};
use crate::limits::{LimitConfig, StopAt, StopRules};
use crate::test_helpers::{assert_canonical, assert_same_shape};
use crate::vtree::Vtree;

/// Keep every node and pair of a structural fixture.
fn all_live(f: &Tdd) -> Marking {
    Marking {
        alive: f.levels.iter().map(|l| vec![true; l.nodes.len()]).collect(),
        pair_alive: PairMarks::all(),
        root_live: true,
    }
}

/// Keep every node and every pair except the output node's pairs.
fn all_live_but_the_output_pairs(eng: &Engine, f: &Tdd) -> Marking {
    let mut pair_alive = PairMarks::new(eng, f).unwrap();
    for (v, _, _) in f.vtree.internal_bottomup() {
        let level = &f.levels[v.idx()];
        for j in 0..level.nodes.len() {
            if (v, j) == (f.output.vtree, f.output.local.idx()) { continue; }
            let count = level.pairs_of_idx(j).len();
            for k in 0..count { pair_alive.mark(eng, v, NodeIdx(j as u32), k, count).unwrap(); }
        }
    }
    Marking { pair_alive, ..all_live(f) }
}

#[test]
fn marks_that_kill_nothing_leave_the_operand_as_it_is() {
    let tree = Arc::new(Vtree::linear(8192));
    let f = Tdd::one(&tree);
    assert_canonical(&f);
    let result = all_live(&f).rebuild(&Engine::new(), f.clone()).unwrap();
    assert_same_shape(&result, &f, "nothing dead");
    assert_canonical(&result);
    assert_eq!(result.model_count().unwrap(), num_bigint::BigUint::from(1u32) << 8192);
}

/// Killing the node for `x1 ∧ x2` under `(x1 ∧ x2) ∨ (x3 ∧ x4)` drops the root
/// pair naming it, in place, so the rest of the diagram keeps its indices.
#[test]
fn a_dead_node_takes_the_pairs_naming_it_and_nothing_else() {
    let tree = Arc::new(Vtree::balanced(4));
    let eng = Engine::new();
    let f = eng.or(Tdd::cube(&tree, [1, 2]).unwrap(), Tdd::cube(&tree, [3, 4]).unwrap()).unwrap();
    assert_canonical(&f);
    let (left, _) = tree.children(tree.root());
    let both_positive = f.levels[left.idx()].pairs_iter_of_idx(0).all(|p| p.left == POS_LEAF_IDX.into() && p.right == POS_LEAF_IDX.into());
    let target = if both_positive { 0 } else { 1 };
    let mut marks = all_live(&f);
    marks.alive[left.idx()][target] = false;

    let g = marks.rebuild(&eng, f.clone()).unwrap();
    assert_eq!(g.output(), f.output(), "the output node keeps its index");
    assert_eq!(g.model_count().unwrap(), 3u32.into());
    assert!(g.pair_count() < f.pair_count());
    let mut expected = eng.and(f, eng.negate(Tdd::cube(&tree, [1, 2]).unwrap()).unwrap()).unwrap();
    expected.minimize().unwrap();
    let mut g = g;
    g.minimize().unwrap();
    assert_same_shape(&g, &expected, "x3 ∧ x4 ∧ ¬(x1 ∧ x2)");
}

#[test]
fn masking_out_every_pair_of_the_output_empties_the_diagram() {
    let tree = Arc::new(Vtree::balanced(4));
    let f = Tdd::cube(&tree, [1, 2, 3, 4]).unwrap();
    let eng = Engine::new();
    let marks = all_live_but_the_output_pairs(&eng, &f);
    let g = marks.rebuild(&eng, f).unwrap();
    assert!(g.is_zero());
    assert_canonical(&g);
}

#[test]
fn the_rebuild_polls_once_per_pair_it_reads() {
    let tree = Arc::new(Vtree::linear(8));
    let f = Tdd::one(&tree);
    let eng = Engine::new();
    eng.limits().pin_reduce_poll_stride(Some(1));
    let _scope = eng.limits().scope(LimitConfig::none().with_stop_rules(StopRules {
        unconditional: Some(StopAt::WorkUnits(3)), ..StopRules::default()
    }));
    assert_eq!(all_live(&f).rebuild(&eng, f).err(), Some(OperationError::Stopped));
    assert_eq!(eng.limits().meters().work_units, 3);
}

#[test]
fn the_rebuild_recovers_after_each_refused_reservation() {
    let tree = Arc::new(Vtree::balanced(4));
    let f = Tdd::clause(&tree, [1, -2, 3]).unwrap();
    assert_canonical(&f);
    let mut reached_success = false;
    for nth in 0..256 {
        let eng = Engine::new();
        eng.limits().refuse_nth_reserve(nth);
        // Masking out every pair of the output empties the diagram.
        let marks = all_live_but_the_output_pairs(&Engine::new(), &f);
        match marks.rebuild(&eng, f.clone()) {
            Ok(result) => { assert!(result.is_zero()); reached_success = true; break; }
            Err(error) => assert_eq!(error, OperationError::OverBudget),
        }
        eng.limits().grant_every_reserve();
        let result = all_live(&f).rebuild(&eng, f.clone()).unwrap();
        assert_canonical(&result);
        assert_eq!(result.model_count().unwrap(), f.model_count().unwrap());
    }
    assert!(reached_success, "the sweep must cover every reservation");
}

/// The node filter's bottom-up sweep and this depth-first rebuild keep the
/// same function once both are minimized, on structural operands and on
/// operands with a summed-out subtree.
#[test]
fn the_node_filter_agrees_with_the_depth_first_rebuild() {
    use crate::apply::FilterOutcome;
    use crate::test_helpers::{assert_same_shape, compile_clauses_on, rand_cnf, vtree_shapes, CnfShape, Lcg};
    let mut rng = Lcg::new(0xdf5);
    let mut compared = 0;
    for round in 0..16u32 {
        let n = 4 + round % 5;
        for (_, vtree) in vtree_shapes(n) {
            let eng = Engine::new();
            let f = compile_clauses_on(&eng, &vtree, &rand_cnf(&mut rng, n, CnfShape { clauses: 2 * n as usize, width: 3 }));
            let mut summed = f.clone();
            let inner: Vec<_> = vtree.internal_bottomup().map(|(t, _, _)| t).filter(|&t| t != vtree.root()).collect();
            if let Some(&t) = inner.get(round as usize % inner.len().max(1)) {
                eng.marginalize_levels(&mut summed, &[t]).unwrap();
                eng.minimize(&mut summed).unwrap();
            }
            for f in [f, summed] {
                if f.is_zero() { continue; }
                let root = f.output();
                let keep = |id: TddNodeId| id == root || !(u64::from(id.vtree.0) * 31 + u64::from(id.local.0) + u64::from(round)).is_multiple_of(5);
                let sweep = match eng.filter_nodes_with(&f, keep, ReductionPlan::default()).unwrap() {
                    FilterOutcome::Unchanged => f.clone(),
                    FilterOutcome::Filtered { tdd, .. } | FilterOutcome::Unsatisfiable { tdd, .. } => tdd,
                };
                let mut marks = Marking::trivial(&eng, &f, true).unwrap();
                for (i, level) in f.levels.iter().enumerate() {
                    if level.is_marginal() { continue; }
                    for (j, _) in level.nodes.iter().enumerate() {
                        if !keep(TddNodeId { vtree: VtreeIdx(i as u32), local: NodeIdx(j as u32) }) {
                            marks.alive[i][j] = false;
                        }
                    }
                }
                let mut dfs = marks.rebuild(&eng, f.clone()).unwrap();
                eng.minimize(&mut dfs).unwrap();
                if sweep.is_zero() {
                    assert!(dfs.is_zero());
                    continue;
                }
                assert_same_shape(&sweep, &dfs, "the sweep against the depth-first rebuild");
                assert_eq!(sweep.model_count().unwrap(), dfs.model_count().unwrap());
                let counts = |g: &Tdd| -> Vec<Option<Vec<u128>>> {
                    g.levels.iter().map(|level| level.marginal_counts().map(|c| { let mut c = c.to_vec(); c.sort_unstable(); c })).collect()
                };
                assert_eq!(counts(&sweep), counts(&dfs));
                compared += 1;
            }
        }
    }
    assert!(compared >= 60, "too few operands compared: {compared}");
}
