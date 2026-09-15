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

use tididi::{and, or, Tdd, Vtree};
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

Use [`or`](crate::or) for alternatives, [`and`](crate::and) for simultaneous
requirements, and [`negate`](crate::Tdd::negate) to complement a diagram.
“Remote requires encryption” means either remote backups are off or encryption
is on:

```rust,ignore
let destination = or(local, remote.clone())?;
let encryption_rule = or(remote.clone().negate()?, encrypted)?;
let mut configurations = and(destination, encryption_rule)?;
```

These operations return `Result`; `?` propagates an error from `main`, whose
return type is `Result<(), tididi::OperationError>`.

Each `Tdd` owns its circuit. Boolean operations consume their operands, so we clone
`remote` where we will need it again. Cloning copies the diagram storage and
shares the vtree; borrow diagrams for queries that do not transform them.
These operations reuse the working buffers attached to the shared vtree; no
explicit engine is needed.
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
let with_remote = and(configurations.clone(), remote)?;
let remote_count = with_remote.model_count();
assert_eq!(remote_count, 4u32.into());
println!("Configurations with remote backups: {remote_count}");
```

Only the last two rows remain. The original `configurations` still represents
all eight choices. This is evidence expressed as another constraint;
[`Tdd::condition`](crate::Tdd::condition) instead substitutes values
into a function, with different counting semantics.

## Ask for one concrete configuration

Borrow the configuration diagram to find one complete assignment:

```rust,ignore
let witness = configurations.satisfying_assignment()
    .expect("the backup rules have a solution");
println!("One valid configuration:");
for literal in &witness {
    println!("  {}: {}", names[literal.var.idx()], literal.positive);
}
```

The witness assigns every vtree variable. There can be many correct witnesses,
so the program verifies that its returned assignment satisfies the rules:

```rust,ignore
let selected = and(configurations.clone(), Tdd::cube(&tree, &witness))?;
assert_eq!(selected.model_count(), 1u32.into());
```

The rules now support counting, additional constraints and finding a solution.
The constructor and query conveniences shown here panic on failure; their
`try_` forms return errors. Boolean operators `&`, `|` and `!` are also
available as shorthand that panics on failure.

Continue with [execution controls](crate::guide::examples::execution) when you
need resource limits, or with [probability queries](crate::guide::examples::probability)
to weight the valid assignments. The
[complete program](https://github.com/Tractables/tididi/blob/main/examples/build_minimize_count.rs)
continues with the execution-control example after these steps.
