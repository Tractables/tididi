//! Which routes open a level's arenas and arm its growth mode.
use super::*;
use crate::diagram::{Tdd, TddLevel, TddNodeId, NodeIdx};
use crate::limits::growth::DENSE_GROWTH_DECISION_THRESHOLD;
use crate::test_helpers::pair;
use crate::vtree::Vtree;
use crate::apply::conjoin::setup::OperandWidths;
use std::sync::Arc;

/// An operand whose root level holds `pairs` pairs in one node, so that two
/// such operands bound the root's emit at `pairs²`.
fn operand(vtree: &Arc<Vtree>, pairs: usize) -> Tdd {
    let mut levels: Vec<TddLevel> = (0..vtree.num_nodes()).map(|_| TddLevel::new()).collect();
    let root = &mut levels[vtree.root().idx()];
    root.pairs.extend((0..pairs).map(|_| pair(0, 0)));
    root.nodes.push(crate::diagram::EncodedNode::multi_pair(0, pairs as u32));
    Tdd::from_levels_unchecked(Arc::clone(vtree), levels, TddNodeId { vtree: vtree.root(), local: NodeIdx(0) })
}

/// A width-one root level over `vtree`.
fn root_shape(vtree: &Vtree) -> LevelShape {
    let t = vtree.root();
    let (left, right) = vtree.children(t);
    let one = OperandWidths { here: 1, left: 1, right: 1 };
    LevelShape { t, left, right, f: one, g: one }
}

/// A root whose two operands together bound the emit past the growth
/// decision threshold, under a budget that makes the bound unaffordable.
fn tight_level() -> (Engine, Arc<Vtree>, Tdd, Tdd) {
    let vtree = Arc::new(Vtree::balanced(2));
    let pairs = (DENSE_GROWTH_DECISION_THRESHOLD as f64).sqrt() as usize + 1;
    assert!((pairs as u128) * (pairs as u128) > DENSE_GROWTH_DECISION_THRESHOLD);
    let f = operand(&vtree, pairs);
    let g = operand(&vtree, pairs);
    let eng = Engine::new();
    eng.limits().set_budget(Some(1 << 20));
    (eng, vtree, f, g)
}

#[test]
fn every_emitting_route_arms_the_growth_mode() {
    let (eng, vtree, f, g) = tight_level();
    let shape = root_shape(&vtree);
    for route in [Route::MarginalChild, Route::PlainDense, Route::Dense, Route::SparseMarg] {
        let _op = eng.limits().begin_operation();
        let mut level = TddLevel::new();
        open_level_arenas(eng.limits(), &f, &g, shape, &mut level, route).unwrap();
        assert!(eng.limits().bounded_growth(), "{route:?} emits pairs and must arm bounded growth");
        assert!(level.pairs.capacity() > 0 && level.nodes.capacity() > 0, "{route:?} must seed both arenas");
    }
}

#[test]
fn the_streaming_routes_open_nothing() {
    let (eng, vtree, f, g) = tight_level();
    let shape = root_shape(&vtree);
    for marginal_children in [false, true] {
        let _op = eng.limits().begin_operation();
        let mut level = TddLevel::new();
        open_level_arenas(eng.limits(), &f, &g, shape, &mut level, Route::Stream { marginal_children }).unwrap();
        assert!(!eng.limits().bounded_growth(), "a streaming level never emits into the pair arena");
        assert_eq!(level.pairs.capacity(), 0);
        assert_eq!(level.nodes.capacity(), 0);
        assert_eq!(eng.limits().meters().in_flight_bytes, 0, "a streaming level reserves nothing");
    }
}
