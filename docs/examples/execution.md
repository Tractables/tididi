# Control execution and release working memory

Ordinary operations reuse scratch buffers attached to the vtree. Use an
explicit batch when you need resource limits; release idle scratch when you
no longer need the retained memory.

This continues the [configuration walkthrough](crate::guide::examples::configurations),
using its `configurations`, `vtree` and `count`. Run both parts with
`cargo run --example build_minimize_count`.

## Bound a batch of operations

Use the vtree's [`Context`](crate::Context) to set a memory budget. A zero-byte
budget lets us demonstrate a refused allocation:

```rust,ignore,{class=tested-example}
use tididi::OperationError;
use tididi::limits::LimitConfig;
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
assert!(matches!(attempt, Err(OperationError::OverBudget)));
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

The budget ends with the batch. A later operation succeeds, and the original
rules remain available:

```rust,ignore,{class=tested-example}
let destination = Tdd::clause(&vtree, [1, 2])?;
assert_eq!(destination.model_count()?, 12u32.into());
assert_eq!(configurations.model_count()?, count);
```

Use [`Context::run`](crate::Context::run) for a batch with no
initial limits; its example shows several checked operations in one checkout.

## Remove redundant storage

Use [`minimize`](crate::Tdd::minimize) to remove redundant storage without
changing the function or vtree:

```rust,ignore,{class=tested-example}
configurations.minimize()?;
assert_eq!(configurations.model_count()?, count);
println!("Minimized representation: {} pairs", configurations.pair_count());
```

Output:

```text
Minimized representation: 8 pairs
```

For a different variable grouping, see the
[vtree walkthrough](crate::guide::examples::vtrees).

## Bound repeated queries

A counter can use batch limits while keeping its cached counts between
queries. Bind it to the supplied engine for the duration of the query:

```rust,ignore,{class=tested-example}
let mut counter = configurations.counter()?;
let query_limit = LimitConfig::none().with_memory_budget_bytes(Some(1_000_000));
let bounded_count = context.with_limits(query_limit, |operations| {
    counter.bind(operations).model_count()
})?;
assert_eq!(bounded_count, count);
```

After the batch, the counter keeps its observations and cached counts.
[`Engine::counter`](crate::Engine::counter) creates one bound to an engine
from the start.

## Release idle scratch

Call [`clear_scratch`](crate::Context::clear_scratch) between batches to
release idle buffers while keeping the diagrams:

```rust,ignore,{class=tested-example}
context.clear_scratch();
```

Dropping the last reference to a context also frees its idle buffers.

The [complete program](https://github.com/Tractables/tididi/blob/main/examples/build_minimize_count.rs)
contains the basic workflow and these execution controls.
