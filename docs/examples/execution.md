# Control execution and release working memory

The [configuration walkthrough](crate::guide::examples::configurations) builds
and queries a diagram without an explicit engine. Those operations already
reuse the workspace attached to the shared vtree. This page continues the same
program when an application needs to bound work or release retained buffers.

The variables `configurations`, `tree` and `count` come from the first part.
Run both parts with `cargo run --example build_minimize_count`.

## Handle errors directly

The ordinary Boolean functions, such as [`and`](crate::and) and
[`or`](crate::or), already return `Result`. Constructors and queries also return `Result`: use `Tdd::clause(...)`, `f.model_count()` and
`f.satisfying_assignment()` to handle their errors.
The optional operators `&`, `|` and `!` panic on failure.

An operation taking diagrams by value consumes them even on error. Keep a copy
before calling it if recovery needs the original. Queries borrow their inputs.

## Bound a batch of operations

The tree's [`Context`](crate::Context) lends a working engine for a batch.
Here we deliberately allow zero bytes of charged allocation growth, so the
attempt to rebuild the destination rule is refused:

```rust,ignore
use tididi::OperationError;
use tididi::limits::LimitConfig;
let context = Arc::clone(tree.context());
let limit = LimitConfig::none().with_memory_budget_bytes(Some(0));
let attempt = context.with_limits(limit, |operations| {
    operations.clause(&tree, [1, 2])
});
```

Call through `operations` throughout the bounded batch. Those calls share its
limit configuration; each top-level call starts new work measurements.
Free functions, ordinary diagram methods and nested context calls are
independent operations and do not inherit a batch's limits.

This is a soft budget for charged allocation growth in each operation, not a
bound on the application's total memory. The example verifies its deliberate
refusal, then handles the result as an application would:

```rust,ignore
assert!(matches!(attempt, Err(OperationError::OverBudget)));
match attempt {
    Ok(diagram) => println!("Destination choices: {}", diagram.model_count()?),
    Err(OperationError::OverBudget) => println!("Not enough budget to build the destination rule"),
    Err(error) => return Err(error),
}
```

The context retains reusable buffers after the batch, but clears its limits
and callbacks. The next operation succeeds, and the original rules are intact:

```rust,ignore
let destination = Tdd::clause(&tree, [1, 2])?;
assert_eq!(destination.model_count()?, 12u32.into());
assert_eq!(configurations.model_count()?, count);
```

The complete program returns `Result<(), tididi::OperationError>` so `?` can
propagate errors. Use [`Context::run`](crate::Context::run) for a batch with no
initial limits; its example shows several checked operations in one checkout.

## Specialize storage when needed

The preceding queries work without an explicit minimization step. To remove
redundancy under the current vtree, minimize the diagram:

```rust,ignore
configurations.minimize()?;
assert_eq!(configurations.model_count()?, count);
```

For a different variable grouping, continue with the
[vtree walkthrough](crate::guide::examples::vtrees). For many counts under
changing observations, [`Tdd::counter`](crate::Tdd::counter) retains counting
state and updates evidence without rebuilding the diagram:

```rust,ignore
let mut counter = configurations.counter()?;
counter.set_pin(tididi::vtree::VarId(1), Some(true))?;
assert_eq!(counter.model_count()?, 4u32.into());
```

Variable 1 is remote backups; add disabled notifications as a second observation
with [`set_pins`](crate::query::ModelCounter::set_pins), which validates the whole
update before changing any pins:

```rust,ignore
counter.set_pins(&[
    (tididi::vtree::VarId(1), Some(true)),
    (tididi::vtree::VarId(3), Some(false)),
])?;
assert_eq!(counter.model_count()?, 2u32.into());
counter.clear_pins();
assert_eq!(counter.model_count()?, count);
```

[`clear_pins`](crate::query::ModelCounter::clear_pins) removes all observations;
[`Tdd::counter_with`](crate::Tdd::counter_with) selects another storage policy or
cofactor semantics.

To count under batch limits, temporarily bind the existing counter to the
supplied engine:

```rust,ignore
let query_limit = LimitConfig::none().with_memory_budget_bytes(Some(1_000_000));
let bounded_count = context.with_limits(query_limit, |operations| {
    counter.bind(operations).model_count()
})?;
assert_eq!(bounded_count, count);
```

The binding borrows the counter and engine; after the batch, the counter keeps
its pins and cached state. [`Engine::counter`](crate::Engine::counter) creates
a counter bound to that engine from the start.

## Release idle scratch

When a batch of work ends, keeping the diagrams also keeps their shared context
alive. Release its idle buffers when the application no longer needs that
capacity:

```rust,ignore
context.clear_scratch();
```

The diagrams keep their results. An operation still running can return buffers
after this call, so clear between batches when all scratch must be released.
Dropping the last reference to a context also frees its idle buffers.

The [complete program](https://github.com/Tractables/tididi/blob/main/examples/build_minimize_count.rs)
contains the basic workflow and these execution controls.
