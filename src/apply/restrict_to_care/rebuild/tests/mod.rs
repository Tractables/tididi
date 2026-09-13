use super::*;
use crate::limits::{LimitConfig, StopAt, StopRules};
use crate::test_helpers::assert_canonical;
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
fn care_rebuild_handles_a_deep_linear_vtree() {
    let tree = Arc::new(Vtree::linear(8192));
    let f = Tdd::one(&tree);
    assert_canonical(&f);
    let result = all_live(&f).rebuild(&Engine::new(), f).unwrap();
    assert_canonical(&result);
    assert_eq!(result.model_count(), num_bigint::BigUint::from(1u32) << 8192);
}

#[test]
fn care_rebuild_polls_before_emitting_the_root() {
    let tree = Arc::new(Vtree::linear(8));
    let f = Tdd::one(&tree);
    assert_canonical(&f);
    let eng = Engine::new();
    eng.limits().pin_reduce_poll_stride(Some(1));
    // Initialization visits each level; the next polls descend through frames.
    let stop = tree.num_nodes() as u64 + 3;
    let _scope = eng.limits().scope(LimitConfig::none().with_stop_rules(StopRules {
        unconditional: Some(StopAt::WorkUnits(stop)), ..StopRules::default()
    }));
    assert_eq!(all_live(&f).rebuild(&eng, f).err(), Some(OperationError::Stopped));
    assert_eq!(eng.limits().meters().work_units, stop);
}

#[test]
fn care_rebuild_recovers_after_each_refused_reservation() {
    let tree = Arc::new(Vtree::balanced(4));
    let f = Tdd::clause(&tree, [1, -2, 3]);
    assert_canonical(&f);
    let mut reached_success = false;
    for nth in 0..256 {
        let eng = Engine::new();
        eng.limits().refuse_nth_reserve(nth);
        match all_live(&f).rebuild(&eng, f.clone()) {
            Ok(result) => { assert_canonical(&result); reached_success = true; break; }
            Err(error) => assert_eq!(error, OperationError::OverBudget),
        }
        eng.limits().grant_every_reserve();
        let result = all_live(&f).rebuild(&eng, f.clone()).unwrap();
        assert_canonical(&result);
        assert_eq!(result.model_count(), f.model_count());
    }
    assert!(reached_success, "the sweep must cover every reservation");
}
