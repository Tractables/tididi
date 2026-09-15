# Count and query valid configurations

Suppose a backup service offers four on/off options: local backups, remote
backups, encryption, and notifications. A configuration must choose at least
one backup destination, and remote backups require encryption. We want to count
valid configurations and find one we can offer to a user.

This example needs only `tididi` as a dependency. Its code sections follow the
order of the complete program, which you can run from the repository with
`cargo run --example build_minimize_count`.

## Give each option a variable

A vtree arranges the variables used by our diagrams. Start with a balanced tree
over four variables and share it between the option diagrams:

```rust,ignore
use std::sync::Arc;

use tididi::{Engine, OperationError, Tdd, Vtree};
use tididi::limits::LimitConfig;
use tididi::reduce::try_minimize;
```

```rust,ignore
let tree = Arc::new(Vtree::balanced(4));
let names = ["local backups", "remote backups", "encryption", "notifications"];
let local = Tdd::literal(&tree, 1);
let remote = Tdd::literal(&tree, 2);
let encrypted = Tdd::literal(&tree, 3);
```

Integer literals start at 1; a negative integer means the option is off.
The `VarId` values returned by queries start at 0, matching positions in
`names`. Notifications need no literal here: they remain an unconstrained
variable in the tree.

## Write the rules as Boolean expressions

Use `|` for OR, `&` for AND, and `!` for NOT. The implication “remote requires
encryption” is `!remote | encrypted`.

```rust,ignore
let destination = local | remote.clone();
let encryption_rule = !remote.clone() | encrypted;
let mut configurations = destination & encryption_rule;
```

Each `Tdd` owns its circuit. Operators consume their operands, so we clone
`remote` where we will need it again. Cloning copies the diagram storage and
shares the vtree; borrow diagrams for queries that do not transform them.
These operations allocate temporary
working memory and release it on return; no persistent engine is needed.
The diagrams must share the same `Arc<Vtree>` allocation, as these do.

## Count configurations

```rust,ignore
let count = configurations.model_count();
assert_eq!(count, 8u32.into());
println!("Valid configurations: {count}");
```

The four valid choices for the constrained options are:

| Local | Remote | Encryption | Notification choices |
|---|---|---|---|
| on | off | off | off or on |
| on | off | on | off or on |
| off | on | on | off or on |
| on | on | on | off or on |

Every row has two choices for notifications, giving eight assignments over
the full vtree. Counting returns an arbitrary-precision integer.

## Add a requirement without losing the original

If a user selects remote backups, conjoin that option with a copy of the rules:

```rust,ignore
let with_remote = configurations.clone() & remote;
let remote_count = with_remote.model_count();
assert_eq!(remote_count, 4u32.into());
println!("Configurations with remote backups: {remote_count}");
```

Only the last two rows remain. The original `configurations` still represents
all eight choices. This is evidence expressed as another constraint;
[`Engine::condition`](crate::engine::Engine::condition) instead substitutes values
into a function, with different counting semantics.

## Handle a resource refusal

The operators above panic on failure. When an application needs to handle an
error, use the corresponding checked operation on an [`Engine`](crate::Engine).
An engine also keeps working buffers for reuse between calls.

```rust,ignore
let engine = Engine::new();
```

Here we deliberately give construction a zero-byte allocation budget, so the
attempt to rebuild the destination rule returns `OverBudget`. In an application,
choose a budget appropriate to the work and handle either outcome:

```rust,ignore
{
    let _limit = engine.limits().scope(
        LimitConfig::none().with_memory_budget_bytes(Some(0)),
    );
    let attempt = engine.clause(&tree, [1, 2]);
    assert!(matches!(attempt, Err(OperationError::OverBudget)));
    match attempt {
        Ok(diagram) => println!("Destination choices: {}", diagram.model_count()),
        Err(OperationError::OverBudget) => println!("Not enough budget to build the destination rule"),
        Err(error) => return Err(error),
    }
}
```

The assertion checks this example's deliberate refusal. The `match` shows the
application's choices: use the result, report a resource refusal, or propagate
another error. This is a soft budget for charged allocation growth in each
operation, not a bound on the application's total memory.

The guard restores the previous limits when the block ends. Construction now
succeeds, and our original configuration diagram still has eight models:

```rust,ignore
let destination = engine.clause(&tree, [1, 2])?;
assert_eq!(destination.model_count(), 12u32.into());
assert_eq!(configurations.model_count(), count);
```

An operation that takes diagrams by value consumes them even when it returns
an error. Keep a copy before such a call if a retry needs the original; that
copy is outside the engine's allocation budget.

## Ask for one concrete configuration

Use the same engine to borrow the configuration diagram and find a witness:

```rust,ignore
let witness = engine.satisfying_assignment(&configurations)?
    .expect("the backup rules have a solution");
println!("One valid configuration:");
for literal in &witness {
    println!("  {}: {}", names[literal.var.idx()], literal.positive);
}
```

The witness assigns every vtree variable. There can be many correct witnesses,
so the program verifies that its returned assignment satisfies the rules:

```rust,ignore
assert!(engine.implies(&engine.cube(&tree, &witness)?, &configurations)?);
```

The complete program returns `Result<(), tididi::OperationError>` so the `?`
operator propagates errors from these checked operations.

## Minimize when the representation needs it

All the preceding queries work without an explicit minimization step. If
further edits leave redundant storage, minimization removes it without changing
the represented configurations:

```rust,ignore
try_minimize(&engine, &mut configurations)?;
assert_eq!(engine.model_count(&configurations)?, count);
```

The [complete program](https://github.com/Tractables/tididi/blob/main/examples/build_minimize_count.rs)
contains these sections together. Continue with the
[probability walkthrough](crate::guide::examples::probability) to assign different weights to
configurations, or the [task guide](crate::guide::api) to find another operation.
