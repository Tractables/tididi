//! Compute reachable states using image, renaming and semantic equivalence.
//! Run with `cargo run --example symbolic_reachability`.

use std::sync::Arc;

use tididi::vtree::VarId;
use tididi::{and, and_exists, or, OperationError, Tdd, Vtree};

fn main() -> Result<(), OperationError> {
    let vtree = Arc::new(Vtree::balanced(4));
    // Variables 1,2 encode the current state; 3,4 encode the next state.
    // The first bit in each pair is least significant. Edges: 0 -> 1 -> 2 -> 1.
    let mut transition = Tdd::zero(&vtree);
    for edge in [[-1, -2, 3, -4], [1, -2, -3, 4], [-1, 2, 3, -4]] {
        transition = or(transition, Tdd::cube(&vtree, edge)?)?;
    }
    let mut reached = Tdd::cube(&vtree, [-1, -2])?; // start at state 0
    let current = [VarId(0), VarId(1)];
    let next_to_current = [(VarId(2), VarId(0)), (VarId(3), VarId(1))];
    let mut iterations = 0;

    loop {
        let possible_steps = and(reached.clone(), transition.clone())?;
        let successors = possible_steps.exists_vars(&current)?;

        // The combined operation expresses the same image in one call.
        let combined = and_exists(reached.clone(), transition.clone(), &current)?;
        assert!(successors.equivalent(&combined)?);
        let successors = successors.rename_vars(&next_to_current)?;
        let enlarged = or(reached.clone(), successors)?;
        iterations += 1;

        // Each state has four assignments to the two free next-state variables.
        let state_count = enlarged.model_count()? / 4u32;
        println!("Iteration {iterations}: {state_count} reachable states");
        if enlarged.equivalent(&reached)? {
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
    let forbidden = Tdd::cube(&vtree, [1, 2])?;
    let safe = forbidden.negate()?;
    assert!(reached.equivalent(&safe)?);
    assert!(reached.implies(&safe)?);
    assert_eq!(reached.model_count()?, 12u32.into());
    println!("State 3 is unreachable");

    let target = Tdd::cube(&vtree, [-1, 2])?; // state 2
    let reachable_target = and(reached, target)?;
    let witness = reachable_target
        .satisfying_assignment()?
        .expect("state 2 is reachable");
    assert!(Tdd::cube(&vtree, &witness)?.implies(&reachable_target)?);
    let state = witness
        .iter()
        .filter(|literal| literal.var.0 < 2 && literal.positive)
        .fold(0u32, |bits, literal| bits | (1 << literal.var.0));
    assert_eq!(state, 2);
    // This is a state assignment; reconstructing a path needs predecessor tracking.
    println!("Reachable target witness: state {state}");
    Ok(())
}
