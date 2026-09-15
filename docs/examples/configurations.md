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

use tididi::{Engine, Tdd, Vtree};
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
`remote` where we will need it again. These operations allocate temporary
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
[`Engine::condition`](https://docs.rs/tididi/latest/tididi/engine/struct.Engine.html#method.condition) instead substitutes values
into a function, with different counting semantics.

## Ask for one concrete configuration

An [`Engine`](https://docs.rs/tididi/latest/tididi/engine/struct.Engine.html) supplies reusable scratch buffers and optional
resource limits. We create one now for the remaining operations; it can use
the diagrams already built above.

```rust,ignore
let engine = Engine::new();
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
operator propagates errors from these checked operations. The convenience
operators above panic on errors; use engine methods when you need to handle
errors or install limits.

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
[probability walkthrough](https://docs.rs/tididi/latest/tididi/guide/examples/probability/index.html) to assign different weights to
configurations, or the [task guide](https://docs.rs/tididi/latest/tididi/guide/api/index.html) to find another operation.
