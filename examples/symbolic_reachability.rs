// scenario: docs/scenarios.md#reachability

//! Compute reachable states using image, renaming and semantic equivalence.
//! Run with `cargo run --example symbolic_reachability`.

const NODES: usize = 16;
const EDGES: &[(usize, usize)] = &[
    (0, 1), (1, 2), (3, 2), (4, 5), (5, 6), (6, 7), (8, 9), (9, 10), (10, 11),
    (0, 4), (1, 5), (2, 6), (3, 7), (4, 8), (5, 9), (6, 10), (7, 11),
    (4, 0), (5, 1), (10, 6), (11, 7), (12, 13), (13, 15), (15, 14), (14, 12),
];

fn main() -> Result<(), tididi::OperationError> {
    use std::sync::Arc;

    use tididi::vtree::VarId;
    use tididi::{and_exists, Literal, OperationError, Tdd, Vtree};

    let vtree = Arc::new(Vtree::balanced(2 * NODES as u32));
    let current_vars: Vec<_> = (1..=NODES as u32).map(VarId).collect();
    let next_vars: Vec<_> = (NODES as u32 + 1..=2 * NODES as u32).map(VarId).collect();

    fn state(vtree: &Arc<Vtree>, indicators: &[VarId], node: usize)
        -> Result<Tdd, OperationError> {
        Tdd::cube(vtree, indicators.iter().enumerate()
            .map(|(i, &var)| Literal::new(var, i == node)))
    }

    let mut at_current = Vec::new();
    let mut at_next = Vec::new();
    for node in 0..NODES {
        at_current.push(state(&vtree, &current_vars, node)?);
        at_next.push(state(&vtree, &next_vars, node)?);
    }

    let mut transition = Tdd::zero(&vtree);
    for &(from, to) in EDGES {
        let step = at_current[from].clone() & at_next[to].clone();
        transition = transition | step;
    }

    let mut reached = at_current[0].clone();

    // Check the first image using the individual operations.
    let possible_steps = reached.clone() & transition.clone();
    let successors = possible_steps.exists_vars(&current_vars)?;

    let next_to_current: Vec<_> = next_vars.iter().copied()
        .zip(current_vars.iter().copied()).collect();
    let successors = successors.rename_vars(&next_to_current)?;

    /// Compute successors and express them in current-state coordinates.
    fn image(
        states: Tdd,
        transition: Tdd,
        current_vars: &[VarId],
        next_to_current: &[(VarId, VarId)],
    ) -> Result<Tdd, OperationError> {
        and_exists(states, transition, current_vars)?.rename_vars(next_to_current)
    }

    let combined = image(reached.clone(), transition.clone(), &current_vars, &next_to_current)?;
    assert!(successors.equivalent(&combined)?);
    assert!(successors.equivalent(&(at_current[1].clone() | at_current[4].clone()))?);

    let mut iterations = 0;
    loop {
        let successors = image(reached.clone(), transition.clone(), &current_vars, &next_to_current)?;
        let enlarged = reached.clone() | successors;
        iterations += 1;

        let state_count = enlarged.projected_model_count(&current_vars)?;
        let nodes = enlarged.node_count();
        let pairs = enlarged.pair_count();
        println!(
            "Iteration {iterations}: {state_count} states, {nodes} circuit nodes, {pairs} pairs"
        );
        if enlarged.equivalent(&reached)? {
            println!("Fixed point reached");
            break;
        }
        reached = enlarged;
    }
    assert_eq!(iterations, 6);

    // Compare against an ordinary graph traversal, including every state indicator.
    let mut visited = [false; NODES];
    let mut pending = vec![0];
    while let Some(node) = pending.pop() {
        if visited[node] { continue; }
        visited[node] = true;
        pending.extend(EDGES.iter().filter(|&&(from, _)| from == node).map(|&(_, to)| to));
    }
    let mut expected = Tdd::zero(&vtree);
    for (node, &is_reached) in visited.iter().enumerate() {
        if is_reached { expected = expected | at_current[node].clone(); }
    }
    assert!(reached.equivalent(&expected)?);
    assert_eq!(reached.projected_model_count(&current_vars)?, 11u32.into());

    println!("Node 3 unreachable: {}", reached.implies(&!at_current[3].clone())?);

    let mut forbidden = Tdd::zero(&vtree);
    for node in &at_current[12..16] {
        forbidden = forbidden | node.clone();
    }
    println!("Nodes 12–15 unreachable: {}", reached.implies(&!forbidden)?);

    let reachable_target = reached & at_current[11].clone();
    let witness = reachable_target.satisfying_assignment()?.expect("node 11 is reachable");
    let active: Vec<_> = current_vars.iter().enumerate()
        .filter(|&(_, &var)| witness.contains(&Literal::pos(var)))
        .map(|(node, _)| node)
        .collect();
    println!("Reachable target: {active:?}");
    assert_eq!(active, [11]);
    assert!(Tdd::cube(&vtree, &witness)?.implies(&reachable_target)?);
    Ok(())
}
