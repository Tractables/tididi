# Count and query valid configurations

Suppose a backup service offers four on/off options: local backups, remote
backups, encryption, and notifications. A configuration must choose at least
one backup destination, and remote backups require encryption. We want to count
valid configurations and find one we can offer to a user.

Run this example with `cargo run --example build_minimize_count`.

## Give each option a variable

A vtree arranges the variables used by our diagrams. Start with a balanced vtree
over four variables and share it between the option diagrams:

```rust,ignore,{class=tested-example}
use std::sync::Arc;

use tididi::{and, literal, or, Tdd, Vtree};
```

```rust,ignore,{class=tested-example}
let vtree = Arc::new(Vtree::balanced(4));
let names = ["local backups", "remote backups", "encryption", "notifications"];
let local = literal(&vtree, 1)?;
let remote = literal(&vtree, 2)?;
let encrypted = literal(&vtree, 3)?;
```

Integer literals start at 1; a negative integer means the option is off.
Notifications need no literal here: they remain an unconstrained variable
in the vtree.

## Write the rules as Boolean expressions

Combine alternatives with [`or`](crate::or) and requirements with
[`and`](crate::and). "Remote requires encryption" means either remote backups
are off or encryption is on:

```rust,ignore,{class=tested-example}
let destination = or(local, remote.clone())?;
let encryption_rule = or(remote.clone().negate()?, encrypted)?;
let mut configurations = and(destination, encryption_rule)?;
```

Boolean operations consume their operands, so `remote.clone()` keeps a copy
for later use. Queries borrow the circuit. All these diagrams share one vtree.
The `?` operator propagates errors from `main`, which returns
`Result<(), tididi::OperationError>`.

## Count configurations

```rust,ignore,{class=tested-example}
let count = configurations.model_count()?;
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

```rust,ignore,{class=tested-example}
let with_remote = and(configurations.clone(), remote)?;
let remote_count = with_remote.model_count()?;
assert_eq!(remote_count, 4u32.into());
println!("Configurations with remote backups: {remote_count}");
```

Only the last two rows remain; the original still represents all eight
choices. Conjunction keeps the selected option in the count.
[`Tdd::condition`](crate::Tdd::condition) substitutes its value, which is a
different query.

## Find forced choices and conflicts

With remote backups selected, encryption is mandatory. Use
[`implied_literals`](crate::Tdd::implied_literals) to find the choices shared
by every remaining configuration:

```rust,ignore,{class=tested-example}
let forced = with_remote.implied_literals()?;
assert!(forced.contains(&3.try_into()?));
for literal in &forced {
    println!("Required choice: {} = {}", names[literal.var.idx()], literal.positive);
}
```

The result includes both remote backups, which we selected, and encryption,
which follows from the rules. Local backups and notifications remain optional.
If the user also disables encryption, no configuration satisfies the choices:

```rust,ignore,{class=tested-example}
let conflicting = and(with_remote, literal(&vtree, -3)?)?;
assert!(!conflicting.is_sat()?);
```

Check satisfiability before displaying forced choices for arbitrary user input;
an empty list can mean either a conflict or that no choice is forced.

## Ask for one concrete configuration

Borrow the configuration diagram to find one complete assignment:

```rust,ignore,{class=tested-example}
let witness = configurations.satisfying_assignment()?
    .expect("the backup rules have a solution");
println!("One valid configuration:");
for literal in &witness {
    println!("  {}: {}", names[literal.var.idx()], literal.positive);
}
```

The witness assigns every vtree variable. Its variable identifiers index
`names` starting at zero. Check that this particular configuration satisfies
the rules:

```rust,ignore,{class=tested-example}
let selected = and(configurations.clone(), Tdd::cube(&vtree, &witness)?)?;
assert_eq!(selected.model_count()?, 1u32.into());
```

## Reuse the circuit as choices change

An interactive configurator asks many questions about the same rules. Create a
[`counter`](crate::Tdd::counter) to retain counting state while observations
change. First the user selects remote backups:

```rust,ignore,{class=tested-example}
let mut counter = configurations.counter()?;
counter.observe([2])?;
assert_eq!(counter.model_count()?, 4u32.into());
```

Observations use the same signed literal numbers as circuit construction.
Turning notifications off adds a second choice while keeping remote backups on:

```rust,ignore,{class=tested-example}
counter.observe([-4])?;
assert_eq!(counter.model_count()?, 2u32.into());
```

Now the user switches remote backups off and disables encryption. Notifications
remain off, and local backups must be on: only one configuration remains.

```rust,ignore,{class=tested-example}
counter.observe([-2, -3])?;
assert_eq!(counter.model_count()?, 1u32.into());
```

Each [`observe`](crate::query::ModelCounter::observe) call changes only the
listed choices. Clear them all to recover the original count:

```rust,ignore,{class=tested-example}
counter.clear_pins();
assert_eq!(counter.model_count()?, count);
```

The circuit stays unchanged. Use [`set_pin`](crate::query::ModelCounter::set_pin)
to clear one observation.

Continue with [minimum costs](crate::guide::examples::optimization),
[probabilities](crate::guide::examples::probability), or
[execution limits](crate::guide::examples::execution).
The [complete program](https://github.com/Tractables/tididi/blob/main/examples/build_minimize_count.rs)
includes the queries above and the execution example.
