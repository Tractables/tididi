<!-- scenario: docs/scenarios.md#reusable-components -->

# Reuse a circuit in a larger model

Two servers each need a backup destination: local storage, remote storage, or
both. We can build the rule **L ∨ R** once and reuse it for each server. Then
we will add a constraint connecting them: the shared remote service can serve
at most one server.

Run `cargo run --example composition`.

## Build one component

The component has just two variables, L = 1 and R = 2, on a
[home](crate::vtree::Home) of its own:

```rust,ignore,{class=tested-example}
use tididi::{literal, Engine, Tdd, Vtree};
use tididi::vtree::{Home, VarId};

let component = Home::new(Vtree::balanced(2));
let backup = Tdd::clause(component.vtree(), [1, 2])?;
println!("Choices for one server: {}", backup.model_count()?);
```

Output:

```text
Choices for one server: 3
```

## Place it twice

The complete model needs four variables. Group each server's variables together
in a balanced vtree:

| Component variable | Server A | Server B |
|---|---|---|
| Local, 1 | L_A, 1 | L_B, 3 |
| Remote, 2 | R_A, 2 | R_B, 4 |

[`transport`](crate::Engine::transport) moves a circuit onto the home of the
complete model under a variable mapping. Here the first placement keeps the
variable numbers, and the second adds two:

```rust,ignore,{class=tested-example}
let engine = Engine::new();
let home = Home::new(Vtree::balanced(4));
let (server_a, _) = engine.transport(&backup, &home, |var| var, 0).map_err(|r| r.error)?;
let (server_b, _) = engine.transport(&backup, &home, |var| VarId(var.0 + 2), 0).map_err(|r| r.error)?;
println!("Server A rule over both servers: {}", server_a.model_count()?);
```

Output:

```text
Server A rule over both servers: 12
```

The first copy represents **L_A ∨ R_A**, leaving L_B and R_B free. Its count
is over the home's variables, therefore 3 × 2 × 2 = 12. Both copies are on the
same home, so we can combine them with the usual Boolean operators. The result represents
**(L_A ∨ R_A) ∧ (L_B ∨ R_B)**:

```rust,ignore,{class=tested-example}
let independent = server_a & server_b;
println!("Independent backup choices: {}", independent.model_count()?);
```

Output:

```text
Independent backup choices: 9
```

Each server has three choices, giving 3 × 3 = 9 combinations.

## Connect the components

The capacity constraint is **¬(R_A ∧ R_B)**. Build it on the same home and
conjoin it with the two server rules:

```rust,ignore,{class=tested-example}
let remote_a = literal(home.vtree(), 2)?;
let remote_b = literal(home.vtree(), 4)?;
let shared_capacity = !(remote_a & remote_b);
let system = independent & shared_capacity;
println!("Choices with shared capacity: {}", system.model_count()?);
println!("Reusable component still available: {}", backup.model_count()?);
```

Output:

```text
Choices with shared capacity: 5
Reusable component still available: 3
```

Four of the nine combinations use remote storage for both servers; excluding
them leaves five. Transport borrowed `backup`, so the original component remains
available for another placement.

## Choose the destination grouping

Transport keeps the component's variable grouping where the home restricted to
the mapped variables has it, and mirrors a node whose children the mapping puts
in the other order. Our home gives each copy its own matching two-leaf subtree.
A home that groups the variables otherwise needs rotations, which the last
argument bounds; with 0 such a placement is refused. Weights belong to a home:
the copy reads the destination's, so weigh the home for the complete model.

## Reuse a placement as rules change

Suppose server B must now use local storage only: **L ∧ ¬R**. Its variables
still occupy the same positions, so the same placement applies to successive
rules:

```rust,ignore,{class=tested-example}
let place_b = |rule: &Tdd| engine.transport(rule, &home, |var| VarId(var.0 + 2), 0)
    .map(|(placed, _)| placed).map_err(|r| r.error);
let local_only = Tdd::cube(component.vtree(), [1, -2])?;
let original_b = place_b(&backup)?;
let updated_b = place_b(&local_only)?;
println!("Server B choices before and after: {} -> {}",
    original_b.model_count()?, updated_b.model_count()?);
```

Output:

```text
Server B choices before and after: 12 -> 4
```

Server B now has one choice, while server A's two variables remain free.

The [complete program](https://github.com/Tractables/tididi/blob/v0.2.0/examples/composition.rs)
builds and connects both server models.
