# Explore reachable states

A transition relation is a Boolean function of a current state and a next
state. We can represent a whole set of states with another diagram, then use
Boolean operations and quantification to compute its successors.

This example starts at state 0 in the following system:

```text
start
  │
  ▼
  0 ───▶ 1 ◀───▶ 2       3
                       isolated
```

The edges are `0 → 1`, `1 → 2`, and `2 → 1`. We will find all reachable states,
prove that state 3 is unreachable, and obtain an assignment for state 2.
Run it with `cargo run --example symbolic_reachability`; only `tididi` is
needed as a dependency.

```rust,ignore,{class=tested-example}
use std::sync::Arc;

use tididi::vtree::VarId;
use tididi::{and, and_exists, or, OperationError, Tdd, Vtree};
```

## Encode current and next states

Two bits describe four states. We give the current state variables 1 and 2,
and the next state variables 3 and 4, with the first bit least significant
in each pair:

| State | First bit | Second bit |
|---|---|---|
| 0 | false | false |
| 1 | true | false |
| 2 | false | true |
| 3 | true | true |

An edge is a cube: the conjunction of the four literals specifying its
source and destination. The relation is the union of the three edge cubes.
Build the relation on one shared vtree:

```rust,ignore,{class=tested-example}
let vtree = Arc::new(Vtree::balanced(4));
// Variables 1,2 encode the current state; 3,4 encode the next state.
// The first bit in each pair is least significant. Edges: 0 -> 1 -> 2 -> 1.
let mut transition = Tdd::zero(&vtree);
for edge in [[-1, -2, 3, -4], [1, -2, -3, 4], [-1, 2, 3, -4]] {
    transition = or(transition, Tdd::cube(&vtree, edge)?)?;
}
```

For example, `[-1, -2, 3, -4]` means current state 0 and next state 1.
Integer literals start at 1, with a negative sign for false. The `VarId`
values used to quantify and rename variables start at 0:

```rust,ignore,{class=tested-example}
let mut reached = Tdd::cube(&vtree, [-1, -2])?; // start at state 0
let current = [VarId(0), VarId(1)];
let next_to_current = [(VarId(2), VarId(0)), (VarId(3), VarId(1))];
let mut iterations = 0;
```

Only current-state variables constrain `reached`; its next-state variables
are free.

## Take one step

The image of a state set `R` under transition relation `T` is
`∃current. (R(current) AND T(current, next))`. This retains a next state
exactly when some state in `R` can reach it. Inside the loop:

```rust,ignore,{class=tested-example}
let possible_steps = and(reached.clone(), transition.clone())?;
let successors = possible_steps.exists_vars(&current)?;
```

The successors are expressed using next-state variables. Rename them to
current-state variables so they can be used in the next iteration:

```rust,ignore,{class=tested-example}
let successors = successors.rename_vars(&next_to_current)?;
```

Add them to the states already reached:

```rust,ignore,{class=tested-example}
let enlarged = or(reached.clone(), successors)?;
iterations += 1;
```

## Stop when the state set no longer grows

Count distinct assignments to the current-state variables with
[`projected_model_count`](crate::Tdd::projected_model_count):

```rust,ignore,{class=tested-example}
let state_count = enlarged.projected_model_count(&current)?;
println!("Iteration {iterations}: {state_count} reachable states");
```

The loop compares the represented functions, rather than their storage:

```rust,ignore,{class=tested-example}
if enlarged.equivalent(&reached)? {
    break;
}
reached = enlarged;
assert!(
    iterations < 4,
    "a four-state system must converge within four images"
);
```

The progression is:

| Image step | Successors | Accumulated states |
|---|---|---|
| 1 | {1} | {0, 1} |
| 2 | {1, 2} | {0, 1, 2} |
| 3 | {1, 2} | {0, 1, 2}: fixed point |

## Check a safety property

State 3 is forbidden. Its complement should equal the reachable set:

```rust,ignore,{class=tested-example}
let forbidden = Tdd::cube(&vtree, [1, 2])?;
let safe = forbidden.negate()?;
assert!(reached.equivalent(&safe)?);
assert!(reached.implies(&safe)?);
assert_eq!(reached.projected_model_count(&current)?, 3u32.into());
println!("State 3 is unreachable");
```

Equivalence checks the complete expected set in this example; implication is
enough to prove a general safety property, even when some safe states are
unreachable.

## Find a reachable target

To find an assignment for state 2, intersect the target with the reachable set:

```rust,ignore,{class=tested-example}
let target = Tdd::cube(&vtree, [-1, 2])?; // state 2
let reachable_target = and(reached, target)?;
let witness = reachable_target
    .satisfying_assignment()?
    .expect("state 2 is reachable");
```

The witness is a state assignment, not a sequence of transitions. Recovering a
path requires retaining predecessor information during the search.

## Combine the image operations

Once the separate steps are familiar, [`and_exists`](crate::and_exists)
expresses conjunction and quantification in one call. This helper takes a state
set and a transition relation, then returns the successors in current-state
coordinates:

```rust,ignore,{class=tested-example}
fn image(
    states: Tdd,
    transition: Tdd,
    current: &[VarId],
    next_to_current: &[(VarId, VarId)],
) -> Result<Tdd, OperationError> {
    and_exists(states, transition, current)?.rename_vars(next_to_current)
}
```

The complete program checks this helper against the separate steps at each
iteration. It can replace those steps once the image computation is familiar.

Ordinary quantification selects its strategy automatically. If a particular
workload needs the structural rewrite, use
[`Tdd::exists_vars_with_strategy`](crate::Tdd::exists_vars_with_strategy) or
[`and_exists_with_strategy`](crate::and_exists_with_strategy)
with [`QuantificationStrategy::Structural`](crate::apply::QuantificationStrategy::Structural);
the operation contracts describe its requirements. The represented Boolean
function is the same.

The [complete program](https://github.com/Tractables/tididi/blob/main/examples/symbolic_reachability.rs)
includes the loop and decodes the witness back to state 2. For the contracts
of the image and renaming operations, see
[`and_exists`](crate::and_exists) and
[`Tdd::rename_vars`](crate::Tdd::rename_vars).
