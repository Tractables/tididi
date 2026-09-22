<!-- scenario: docs/scenarios.md#execution -->

# Control execution and release working memory

Ordinary operations reuse scratch buffers attached to the vtree. Use an
explicit batch when you need resource limits; release idle scratch when you
no longer need the retained memory.

We use four backup options: local L, remote R, encryption E, and notifications N.
First build L ∨ R, then require encryption to obtain (L ∨ R) ∧ E.
Notifications remain free. Run the [complete program](https://github.com/Tractables/tididi/blob/v0.1.0/examples/execution_limits.rs)
with `cargo run --example execution_limits`.

## Bound a batch of operations

Use the vtree's [`Context`](crate::Context) to set a memory budget. A zero-byte
budget lets us demonstrate a refused allocation:

```rust,ignore,{class=tested-example}
use std::sync::Arc;

use tididi::limits::LimitConfig;
use tididi::{literal, OperationError, Tdd, Vtree};

let vtree = Arc::new(Vtree::balanced(4));
let context = Arc::clone(vtree.context());
let limit = LimitConfig::none().with_memory_budget_bytes(Some(0));
let attempt = context.with_limits(limit, |operations| {
    operations.clause(&vtree, [1, 2])
});
```

Call through `operations` inside the batch: ordinary diagram methods and free
functions do not inherit its limits. The budget covers charged allocation
growth per operation; [`LimitConfig`](crate::limits::LimitConfig) describes
what it measures. Handle a refusal like any other error:

```rust,ignore,{class=tested-example}
match attempt {
    Ok(diagram) => println!("Destination choices: {}", diagram.model_count()?),
    Err(OperationError::OverBudget) => println!("Not enough budget to build the destination rule"),
    Err(error) => return Err(error),
}
```

Output:

```text
Not enough budget to build the destination rule
```

The budget ends with the batch. A later, unrestricted operation succeeds:

```rust,ignore,{class=tested-example}
let destination = Tdd::clause(&vtree, [1, 2])?;
println!("Destination choices: {}", destination.model_count()?);
```

Output:

```text
Destination choices: 12
```

Use [`Context::run`](crate::Context::run) for a batch with no
initial limits; its example shows several checked operations in one checkout.

## Keep inputs for a retry

Conjunction consumes its operands even when it returns an error. Pass clones
if you need to preserve them for another attempt. Here we retry without a
budget; an application could instead choose a larger limit or defer the work.

```rust,ignore,{class=tested-example}
let encrypted = literal(&vtree, 3)?;
let limit = LimitConfig::none().with_memory_budget_bytes(Some(0));
let attempt = context.with_limits(limit, |operations| {
    operations.and(destination.clone(), encrypted.clone())
});
let secured = match attempt {
    Ok(diagram) => diagram,
    Err(OperationError::OverBudget) => {
        println!("Not enough budget; retrying with the original operands");
        tididi::and(destination, encrypted)?
    }
    Err(error) => return Err(error),
};
println!("Secured choices: {}", secured.model_count()?);
```

Output:

```text
Not enough budget; retrying with the original operands
Secured choices: 6
```

The three choices of destination each allow notifications on or off, giving
six configurations with encryption enabled.

## Bound repeated queries

A counter can use batch limits while keeping its cached counts between
queries. Bind it to the supplied engine for the duration of the query:

```rust,ignore,{class=tested-example}
let mut counter = secured.counter()?;
let query_limit = LimitConfig::none().with_memory_budget_bytes(Some(1_000_000));
let bounded_count = context.with_limits(query_limit, |operations| {
    counter.bind(operations).model_count()
})?;
println!("Count with a budget: {bounded_count}");
```

Output:

```text
Count with a budget: 6
```

After the batch, the counter keeps its observations and cached counts.
[`Engine::counter`](crate::Engine::counter) creates one bound to an engine
from the start.

## Release idle scratch

Call [`clear_scratch`](crate::Context::clear_scratch) between batches to
release idle buffers while keeping the diagrams:

```rust,ignore,{class=tested-example}
context.clear_scratch();
println!("Circuit remains usable: {}", secured.is_sat()?);
```

Output:

```text
Circuit remains usable: true
```

Dropping the last reference to a context also frees its idle buffers.
