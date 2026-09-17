//! Compute reachable states using image, renaming and semantic equivalence.
//! Run with `cargo run --example symbolic_reachability`.

use std::sync::Arc;

use tididi::vtree::VarId;
use tididi::{and, and_exists, or, OperationError, Tdd, Vtree};

fn main() -> Result<(), OperationError> {
    let vtree = Arc::new(Vtree::balanced(8));
    // Indicators a,b,c,d use literals 1..4; next-state indicators use 5..8.
    let at_a = Tdd::cube(&vtree, [1, -2, -3, -4])?;
    let at_b = Tdd::cube(&vtree, [-1, 2, -3, -4])?;
    let at_c = Tdd::cube(&vtree, [-1, -2, 3, -4])?;
    let at_d = Tdd::cube(&vtree, [-1, -2, -3, 4])?;
    let next_b = Tdd::cube(&vtree, [-5, 6, -7, -8])?;
    let next_c = Tdd::cube(&vtree, [-5, -6, 7, -8])?;

    let a_to_b = and(at_a.clone(), next_b.clone())?;
    let b_to_c = and(at_b.clone(), next_c)?;
    let c_to_b = and(at_c.clone(), next_b)?;
    let transition = or(a_to_b, or(b_to_c, c_to_b)?)?;

    let mut reached = at_a.clone();
    let current = [VarId(1), VarId(2), VarId(3), VarId(4)];
    let next_to_current = [
        (VarId(5), VarId(1)), (VarId(6), VarId(2)),
        (VarId(7), VarId(3)), (VarId(8), VarId(4)),
    ];
    let mut iterations = 0;

    // Check the first image using the individual operations.
    let possible_steps = and(reached.clone(), transition.clone())?;
    let successors = possible_steps.exists_vars(&current)?;
    let successors = successors.rename_vars(&next_to_current)?;
    let combined = image(reached.clone(), transition.clone(), &current, &next_to_current)?;
    assert!(successors.equivalent(&combined)?);
    assert!(successors.equivalent(&at_b)?);

    loop {
        let successors = image(reached.clone(), transition.clone(), &current, &next_to_current)?;
        let enlarged = or(reached.clone(), successors)?;
        iterations += 1;

        // Count distinct current states, regardless of next-state assignments.
        let state_count = enlarged.projected_model_count(&current)?;
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

    // Check the whole reachable set, including exclusion of invalid encodings.
    let expected = or(at_a, or(at_b, at_c.clone())?)?;
    assert!(reached.equivalent(&expected)?);
    assert_eq!(reached.projected_model_count(&current)?, 3u32.into());

    let safe = at_d.negate()?;
    assert!(reached.implies(&safe)?);
    println!("D is unreachable");

    let reachable_target = and(reached, at_c)?;
    let witness = reachable_target
        .satisfying_assignment()?
        .expect("C is reachable");
    assert!(Tdd::cube(&vtree, &witness)?.implies(&reachable_target)?);
    let active = witness
        .iter()
        .filter(|literal| literal.var.0 < 4 && literal.positive)
        .map(|literal| literal.var)
        .collect::<Vec<_>>();
    assert_eq!(active, [VarId(3)]);
    // This is a state assignment; reconstructing a path needs predecessor tracking.
    println!("Reachable target witness: a=false, b=false, c=true, d=false");
    Ok(())
}

/// Compute successors and express them in current-state coordinates.
fn image(
    states: Tdd,
    transition: Tdd,
    current: &[VarId],
    next_to_current: &[(VarId, VarId)],
) -> Result<Tdd, OperationError> {
    and_exists(states, transition, current)?.rename_vars(next_to_current)
}
