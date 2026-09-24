use std::sync::Arc;

use super::*;
use crate::diagram::POS_LEAF_IDX;
use crate::limits::{LimitConfig, StopAt, StopRules};
use crate::test_helpers::{assert_canonical, assert_same_shape};
use crate::vtree::Vtree;

/// Keep every node and pair of a structural fixture.
fn all_live(f: &Tdd) -> Marking {
    Marking {
        alive: f.levels.iter().map(|l| vec![true; l.nodes.len()]).collect(),
        pair_alive: f.levels.iter().map(|l| vec![u64::MAX; l.nodes.len()]).collect(),
        root_live: true,
    }
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
    let mut marks = all_live(&f);
    marks.pair_alive[f.output.vtree.idx()][f.output.local.idx()] = 0;
    let g = marks.rebuild(&Engine::new(), f).unwrap();
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
        let mut marks = all_live(&f);
        marks.pair_alive[f.output.vtree.idx()][f.output.local.idx()] = 0;
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
