<!-- scenario: docs/scenarios.md#reusable-components -->

# Reuse a circuit in a larger model

Two servers each need a backup destination: local storage, remote storage, or
both. We can build the rule **L ∨ R** once and reuse it for each server. Then
we will add a constraint connecting them: the shared remote service can serve
at most one server.

Run `cargo run --example reusable_components`.

## Build one component

The component has just two variables, L = 1 and R = 2:

```rust,ignore,{class=tested-example}
use std::sync::Arc;
use tididi::{literal, Tdd, Vtree};
use tididi::vtree::VarId;

let component_vtree = Arc::new(Vtree::balanced(2));
let backup = Tdd::clause(&component_vtree, [1, 2])?;
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

[`embed`](crate::Tdd::embed) copies a circuit into a destination vtree under a
variable mapping. Here the first placement keeps the variable numbers, and the
second adds two:

```rust,ignore,{class=tested-example}
let vtree = Arc::new(Vtree::balanced(4));
let (server_a, _) = backup.embed(&vtree, |var| var)?;
let (server_b, _) = backup.embed(&vtree, |var| VarId(var.0 + 2))?;
println!("Server A rule over both servers: {}", server_a.model_count()?);
```

Output:

```text
Server A rule over both servers: 12
```

The first copy represents **L_A ∨ R_A**, leaving L_B and R_B free. Its count is
therefore 3 × 2 × 2 = 12. Both copies share the destination `vtree`, so we can
combine them with the usual Boolean operators. The result represents
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

The capacity constraint is **¬(R_A ∧ R_B)**. Build it on the same destination
vtree and conjoin it with the two server rules:

```rust,ignore,{class=tested-example}
let remote_a = literal(&vtree, 2)?;
let remote_b = literal(&vtree, 4)?;
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
them leaves five. Embedding borrowed `backup`, so the original component remains
available for another placement.

## Choose the destination grouping

Embedding preserves the component's variable grouping and left-to-right order.
Our destination gives each copy its own matching two-leaf subtree. More general
placements can leave gaps for extra variables, provided the destination restricted
to the mapped variables still matches the source vtree; see the
[`embed` contract](crate::Tdd::embed). Embedding copies structure but drops
attached weights, so supply weights for the complete model when evaluating it.

If independently built components already use disjoint variable identifiers,
[`graft`](crate::Tdd::graft) can conjoin them while constructing a combined vtree.
Embedding is useful when you choose that shared destination in advance, including
when different components need to refer to the same variable.

The [complete program](https://github.com/Tractables/tididi/blob/v0.1.0/examples/reusable_components.rs)
builds and connects both server models.
