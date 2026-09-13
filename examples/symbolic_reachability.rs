//! Compute reachable states using image, renaming and semantic equivalence.
//! Run with `cargo run --example symbolic_reachability`.

use std::sync::Arc;

use tididi::apply::QuantificationStrategy;
use tididi::vtree::VarId;
use tididi::{Engine, OperationError, Tdd, Vtree};

fn main() -> Result<(), OperationError> {
    let engine = Engine::new();
    let tree = Arc::new(Vtree::balanced(4));
    // Variables 1,2 encode the current state; 3,4 encode the next state.
    // The first bit in each pair is least significant. Edges: 0 -> 1 -> 2 -> 1.
    let mut transition = Tdd::zero(&tree);
    for edge in [[-1, -2, 3, -4], [1, -2, -3, 4], [-1, 2, 3, -4]] {
        transition = engine.or(transition, engine.cube(&tree, edge)?)?;
    }
    let mut reached = engine.cube(&tree, [-1, -2])?; // start at state 0
    let current = [VarId(0), VarId(1)];
    let next_to_current = [(VarId(2), VarId(0)), (VarId(3), VarId(1))];
    let mut iterations = 0;

    loop {
        let successors = engine.and_exists(
            reached.clone(),
            transition.clone(),
            &current,
            QuantificationStrategy::Automatic,
        )?;
        let successors = engine.rename_vars(successors, &next_to_current)?;
        let enlarged = engine.or(reached.clone(), successors)?;
        iterations += 1;

        // Each state has four assignments to the two free next-state variables.
        let state_count = engine.model_count(&enlarged)? / 4u32;
        println!("Iteration {iterations}: {state_count} reachable states");
        if engine.equivalent(&enlarged, &reached)? {
            break;
        }
        reached = enlarged;
        assert!(
            iterations < 4,
            "a four-state system must converge within four images"
        );
    }
    assert_eq!(iterations, 3);

    // States 0,1,2 are reachable; state 3 (both current bits true) is not.
    let forbidden = engine.cube(&tree, [1, 2])?;
    let safe = engine.negate(forbidden)?;
    assert!(engine.equivalent(&reached, &safe)?);
    assert!(engine.implies(&reached, &safe)?);
    assert_eq!(engine.model_count(&reached)?, 12u32.into());
    println!("State 3 is unreachable");

    let target = engine.cube(&tree, [-1, 2])?; // state 2
    let reachable_target = engine.and(reached, target)?;
    let witness = engine
        .satisfying_assignment(&reachable_target)?
        .expect("state 2 is reachable");
    assert!(engine.implies(&engine.cube(&tree, &witness)?, &reachable_target)?);
    let state = witness
        .iter()
        .filter(|literal| literal.var.0 < 2 && literal.positive)
        .fold(0u32, |bits, literal| bits | (1 << literal.var.0));
    assert_eq!(state, 2);
    // This is a state assignment; reconstructing a path needs predecessor tracking.
    println!("Reachable target witness: state {state}");
    Ok(())
}
