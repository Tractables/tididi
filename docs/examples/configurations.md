# Count and query valid configurations

Suppose a backup service offers four on/off options: local backups, remote
backups, encryption, and notifications. A configuration must choose at least
one backup destination, and remote backups require encryption. We want to count
valid configurations and find one we can offer to a user.

Run this example with `cargo run --example build_minimize_count`.

## Give each option a variable

A vtree arranges the variables used by our diagrams. Start with a balanced vtree
over four variables. Name the literals so we can use them both to build option
diagrams and to record user choices:

```rust,ignore,{class=tested-example}
use std::sync::Arc;

use tididi::{literal, Literal, Tdd, Vtree};

let vtree = Arc::new(Vtree::balanced(4));
let names = ["local backups", "remote backups", "encryption", "notifications"];
let local_choice = Literal::try_from(1)?;
let remote_choice = Literal::try_from(2)?;
let encrypted_choice = Literal::try_from(3)?;
let notifications_choice = Literal::try_from(4)?;
let local = literal(&vtree, local_choice)?;
let remote = literal(&vtree, remote_choice)?;
let encrypted = literal(&vtree, encrypted_choice)?;
let notifications = literal(&vtree, notifications_choice)?;
```

A `Literal` names one variable and its sign; `literal` builds the corresponding
circuit. Integer literals start at 1; a negative integer means the option is off.

## Write the rules as Boolean expressions

Use `|` for OR, `&` for AND, and `!` for NOT. "Remote requires encryption"
means either remote backups are off or encryption is on:

```rust,ignore,{class=tested-example}
let destination = local | remote.clone();
let encryption_rule = !remote.clone() | encrypted.clone();
let mut configurations = destination & encryption_rule;
```

Notifications remain optional because neither rule constrains them.

Boolean operators consume their operands. Cloning supplies a copy to the
operation, leaving the original `remote` or `encrypted` available for later use.
Queries borrow the circuit. All these diagrams share one vtree.

## Count configurations

```rust,ignore,{class=tested-example}
let count = configurations.model_count()?;
println!("Valid configurations: {count}");
```

Output:

```text
Valid configurations: 8
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
let with_remote = configurations.clone() & remote;
let remote_count = with_remote.model_count()?;
println!("Configurations with remote backups: {remote_count}");
```

Output:

```text
Configurations with remote backups: 4
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
for literal in &forced {
    println!("Required choice: {} = {}", names[literal.var.idx()], literal.sign);
}
```

Output:

```text
Required choice: remote backups = true
Required choice: encryption = true
```

Local backups and notifications remain optional.
If the user also disables encryption, no configuration satisfies the choices:

```rust,ignore,{class=tested-example}
let remote_without_encryption = with_remote & !encrypted.clone();
println!("Satisfiable: {}", remote_without_encryption.is_sat()?);
```

Output:

```text
Satisfiable: false
```

## Ask for one concrete configuration

Borrow the configuration diagram to find one complete assignment:

```rust,ignore,{class=tested-example}
let witness = configurations.satisfying_assignment()?
    .expect("the backup rules have a solution");
println!("One valid configuration:");
for literal in &witness {
    println!("  {}: {}", names[literal.var.idx()], literal.sign);
}
```

Output:

```text
One valid configuration:
  local backups: true
  remote backups: true
  encryption: true
  notifications: false
```

The witness assigns every option. Check that this configuration satisfies
the rules:

```rust,ignore,{class=tested-example}
let selected = configurations.clone() & Tdd::cube(&vtree, &witness)?;
println!("Selected valid configurations: {}", selected.model_count()?);
```

Output:

```text
Selected valid configurations: 1
```

## Reuse the circuit as choices change

An interactive configurator asks many questions about the same rules. Create a
[`counter`](crate::Tdd::counter) to retain counting state while observations
change. First the user selects remote backups:

```rust,ignore,{class=tested-example}
let mut counter = configurations.counter()?;
counter.observe([remote_choice])?;
println!("Matching configurations: {}", counter.model_count()?);
```

Output:

```text
Matching configurations: 4
```

Turning notifications off adds a second choice while keeping remote backups on:

```rust,ignore,{class=tested-example}
counter.observe([notifications_choice.negated()])?;
println!("Matching configurations: {}", counter.model_count()?);
```

Output:

```text
Matching configurations: 2
```

Now the user switches remote backups off and disables encryption. Notifications
remain off, and local backups must be on: only one configuration remains.

```rust,ignore,{class=tested-example}
counter.observe([remote_choice.negated(), encrypted_choice.negated()])?;
println!("Matching configurations: {}", counter.model_count()?);
```

Output:

```text
Matching configurations: 1
```

Each [`observe`](crate::query::ModelCounter::observe) call changes only the
listed choices. Clear them all to recover the original count:

```rust,ignore,{class=tested-example}
counter.clear_pins();
println!("Matching configurations: {}", counter.model_count()?);
```

Output:

```text
Matching configurations: 8
```

The circuit stays unchanged. Use [`set_pin`](crate::query::ModelCounter::set_pin)
to clear one observation.

## Handle operation errors

The Boolean operators panic if an operation fails. To handle failures, use
[`and`](crate::and), [`or`](crate::or), and [`negate`](crate::Tdd::negate),
which return `Result`. Here is the same model written that way:

```rust,ignore,{class=tested-example}
use tididi::{and, or};

let local = literal(&vtree, local_choice)?;
let remote = literal(&vtree, remote_choice)?;
let encrypted = literal(&vtree, encrypted_choice)?;
let destination = or(local, remote.clone())?;
let encryption_rule = or(remote.negate()?, encrypted)?;
let checked = and(destination, encryption_rule)?;
```

The `?` operator returns an error to the caller; use `match` if you want to
handle it here. Constructors and queries already return `Result`, which is
why their calls use `?` throughout this example. The
[execution example](crate::guide::examples::execution) shows how to handle
a memory-budget error.

Continue with [minimum costs](crate::guide::examples::optimization),
[probabilities](crate::guide::examples::probability), or
[execution limits](crate::guide::examples::execution).
The [complete program](https://github.com/Tractables/tididi/blob/main/examples/build_minimize_count.rs)
includes the queries above and the execution example.
