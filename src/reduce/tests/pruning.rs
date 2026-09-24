//! Pruning.
//!
//! The fixtures these read are in `mod.rs`.

use std::sync::Arc;

use crate::reduce::prune::{PruneScope, prune_unreachable};

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
