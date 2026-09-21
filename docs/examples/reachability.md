<!-- scenario: docs/scenarios.md#reachability -->

# Explore reachable states

Starting at node 0, which nodes can we reach by following these arrows?

![A directed graph with 16 nodes. Node 0 starts a three-by-four grid. Node 3 has edges to nodes 2 and 7 but no incoming edges. Nodes 12 through 15 form a separate cycle.](https://raw.githubusercontent.com/Tractables/tididi/v0.1.0/docs/reachability.svg)

We will turn this graph into a Boolean formula, then compute successors until
no new states appear. The graph has 16 nodes and 25 edges; a helper constructs
the state formulas, and a loop builds the transition relation from this list:

```rust,ignore,{class=tested-example}
const NODES: usize = 16;
const EDGES: &[(usize, usize)] = &[
    (0, 1), (1, 2), (3, 2), (4, 5), (5, 6), (6, 7), (8, 9), (9, 10), (10, 11),
    (0, 4), (1, 5), (2, 6), (3, 7), (4, 8), (5, 9), (6, 10), (7, 11),
    (4, 0), (5, 1), (10, 6), (11, 7), (12, 13), (13, 15), (15, 14), (14, 12),
];
```

Run it with `cargo run --example symbolic_reachability`.

## Describe a state with Boolean indicators

Allocate one Boolean variable per node for the current state, and another
group for the next state:

```rust,ignore,{class=tested-example}
use std::sync::Arc;

use tididi::vtree::VarId;
use tididi::{and_exists, Literal, OperationError, Tdd, Vtree};

let vtree = Arc::new(Vtree::balanced(2 * NODES as u32));
let current_vars: Vec<_> = (1..=NODES as u32).map(VarId).collect();
let next_vars: Vec<_> = (NODES as u32 + 1..=2 * NODES as u32).map(VarId).collect();
```

The variables in `current_vars` are our indicators `x₀, …, x₁₅`: `x₀` means “we
are at node 0”, `x₁` means “we are at node 1”, and so on. A state formula
makes exactly one indicator true. For example:

```text
S₀(x₀, …, x₁₅) =  x₀ ∧ ¬x₁ ∧ ¬x₂ ∧ … ∧ ¬x₁₅
S₁(x₀, …, x₁₅) = ¬x₀ ∧  x₁ ∧ ¬x₂ ∧ … ∧ ¬x₁₅
```

In general, `Sᵢ` makes indicator `i` true and every other indicator false.

Now construct the state formulas. [`Tdd::cube`](crate::Tdd::cube) conjoins
literals; [`Literal::new`](crate::Literal::new) makes each literal positive
only when its index is the selected node. Use the helper for both groups:

```rust,ignore,{class=tested-example}
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
```

`at_current[i]` now represents `Sᵢ(x)` and `at_next[i]` represents `Sᵢ(x′)`, where
`x = (x₀, …, x₁₅)` and `x′ = (x′₀, …, x′₁₅)`.

## Build the transition relation

An edge `(i, j)` allows a move from state `i` now to state `j` next. Conjoin
those two state formulas, then take the disjunction over all edges:

```text
T(x, x′) = ⋁_{(i, j) ∈ EDGES} (Sᵢ(x) ∧ Sⱼ(x′))
```

The loop follows this formula directly. It starts with false (no allowed
moves) and adds one term per edge:

```rust,ignore,{class=tested-example}
let mut transition = Tdd::zero(&vtree);
for &(from, to) in EDGES {
    let step = at_current[from].clone() & at_next[to].clone();
    transition = transition | step;
}
```

Initially only node 0 has been reached: `R₀(x) = S₀(x)`.

```rust,ignore,{class=tested-example}
let mut reached = at_current[0].clone();
```

## Compute successor states

A next state is a successor if **some** reached state has an edge to it:

```text
S(x′) = ∃x₀, …, x₁₅. (R(x) ∧ T(x, x′))
```

The existential quantifier removes the source indicators, keeping the
possible destinations. From node 0, the first image contains nodes 1 and 4.

Conjoin the reached set with the relation, then quantify the current variables:

```rust,ignore,{class=tested-example}
let possible_steps = reached.clone() & transition.clone();
let successors = possible_steps.exists_vars(&current_vars)?;
```

Pair each next-state indicator with its current counterpart, then rename
the result so it can serve as a current-state set in another step:

```rust,ignore,{class=tested-example}
let next_to_current: Vec<_> = next_vars.iter().copied()
    .zip(current_vars.iter().copied()).collect();
let successors = successors.rename_vars(&next_to_current)?;
```

[`and_exists`](crate::and_exists) combines the first two operations above into
one call: conjoin the reached states with the transition relation, then
existentially quantify the current-state variables. Renaming is still a separate
step. Put both calls in an image helper for the search:

```rust,ignore,{class=tested-example}
fn image(
    states: Tdd,
    transition: Tdd,
    current_vars: &[VarId],
    next_to_current: &[(VarId, VarId)],
) -> Result<Tdd, OperationError> {
    and_exists(states, transition, current_vars)?.rename_vars(next_to_current)
}
```

## Repeat until the set stops growing

Add each image to the states already reached. The first update gives
`S₀(x) ∨ S₁(x) ∨ S₄(x)`, allowing any of those three states. Stop when
[`equivalent`](crate::Tdd::equivalent) says the set has not changed:

```rust,ignore,{class=tested-example}
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
```

Output:

```text
Iteration 1: 3 states, 35 circuit nodes, 37 pairs
Iteration 2: 6 states, 40 circuit nodes, 45 pairs
Iteration 3: 8 states, 41 circuit nodes, 48 pairs
Iteration 4: 10 states, 42 circuit nodes, 51 pairs
Iteration 5: 11 states, 42 circuit nodes, 52 pairs
Iteration 6: 11 states, 42 circuit nodes, 52 pairs
Fixed point reached
```

[`projected_model_count`](crate::Tdd::projected_model_count) counts only the
current-state assignments. Ordinary counting would also count the free
next-state indicators.

[`node_count`](crate::Tdd::node_count) counts stored circuit nodes;
[`pair_count`](crate::Tdd::pair_count) counts their child pairs. These describe
the circuit representing the reachable set, rather than the graph's nodes and edges.

Iterations 5 and 6 have the same reachable set and circuit size: 42 nodes and
52 pairs. The stopping condition remains equivalence; matching sizes alone
would not prove stability.

## Check a safety property

Node 3 belongs to the same component as node 0 if we ignore arrow direction,
but its edges only lead away from it: `3 → 2` and `3 → 7`. No path from node 0
can enter it. Check that it is absent from the reached set:

```rust,ignore,{class=tested-example}
println!("Node 3 unreachable: {}", reached.implies(&!at_current[3].clone())?);
```

Output:

```text
Node 3 unreachable: true
```

The four nodes in the separate component are also unreachable. Form the
union of their state formulas and check that every reached state lies outside it:

```rust,ignore,{class=tested-example}
let mut forbidden = Tdd::zero(&vtree);
for node in &at_current[12..16] {
    forbidden = forbidden | node.clone();
}
println!("Nodes 12–15 unreachable: {}", reached.implies(&!forbidden)?);
```

Output:

```text
Nodes 12–15 unreachable: true
```

## Find a reachable target

Intersect the reached set with `S₁₁(x)` and ask for a satisfying assignment.
The positive current-state indicator identifies the target node:

```rust,ignore,{class=tested-example}
let reachable_target = reached & at_current[11].clone();
let witness = reachable_target.satisfying_assignment()?.expect("node 11 is reachable");
let active: Vec<_> = current_vars.iter().enumerate()
    .filter(|&(_, &var)| witness.contains(&Literal::pos(var)))
    .map(|(node, _)| node)
    .collect();
println!("Reachable target: {active:?}");
```

Output:

```text
Reachable target: [11]
```

This finds a state assignment. Recovering a path to it would also require
retaining predecessor information during the search.

The [complete program](https://github.com/Tractables/tididi/blob/v0.1.0/examples/symbolic_reachability.rs)
also checks the symbolic result against an ordinary graph traversal and
verifies the witness.

When the terms of a union are already available as circuits,
[`or_many`](crate::or_many) combines them in one batch.
