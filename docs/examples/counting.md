<!-- scenario: docs/scenarios.md#counting-choices -->

# What are we counting?

A backup configuration has four choices: local storage **L**, remote storage **R**,
encryption **E**, and notifications **N**. The rules are **(L ∨ R) ∧ (¬R ∨ E)**.
Notifications are optional. We will use the same model to answer three questions:
how many configurations match a choice, what rules remain after substituting it,
and how many distinct choices are possible for selected options.

Run this example with `cargo run --example counting_choices`.

## Build the rules

```rust,ignore,{class=tested-example}
use std::sync::Arc;
use tididi::{literal, Literal, Tdd, Vtree};

let vtree = Arc::new(Vtree::balanced(4));
let local = Literal::try_from(1)?;
let remote = Literal::try_from(2)?;
let encrypted = Literal::try_from(3)?;
let notifications = Literal::try_from(4)?;
let rules = Tdd::clause(&vtree, [local, remote])?
    & Tdd::clause(&vtree, [remote.negated(), encrypted])?;
println!("Valid configurations: {}", rules.model_count()?);
```

Output:

```text
Valid configurations: 8
```

## Keep configurations consistent with a choice

If the user selects remote backups, count configurations satisfying **rules ∧ R**.
Conjunction builds that set; a counter answers the same question without building
another circuit:

```rust,ignore,{class=tested-example}
let selected = rules.clone() & literal(&vtree, remote)?;
println!("With remote backups: {}", selected.model_count()?);
let mut counter = rules.counter()?;
counter.observe([remote])?;
println!("Observed remote backups: {}", counter.model_count()?);
```

Output:

```text
With remote backups: 4
Observed remote backups: 4
```

Remote and encryption are true. Local storage and notifications each have two
choices, giving four configurations.

## Substitute a value into the rules

Substituting **R = true** simplifies **(L ∨ R) ∧ (¬R ∨ E)** to **E**.
The resulting function no longer depends on R. Its vtree still includes R,
so an ordinary count includes both values of that now-free variable:

```rust,ignore,{class=tested-example}
let residual = rules.clone().condition([remote])?;
println!("After substituting remote = true: {}", residual.model_count()?);
let remaining = [local.var, encrypted.var, notifications.var];
println!("Distinct remaining choices: {}", residual.projected_model_count(&remaining)?);
```

Output:

```text
After substituting remote = true: 8
Distinct remaining choices: 4
```

Use [`condition`](crate::Tdd::condition) when you need the remaining function;
use observation or conjunction when you want to retain the selected value.

## Count distinct choices for a subset of options

Suppose we care only about storage destinations. Local only, remote only, and
both are possible: three choices. Encryption and notifications should not
multiply this count. A projected count keeps only the named variables:

```rust,ignore,{class=tested-example}
let destinations = [local.var, remote.var];
println!("Valid destination choices: {}", rules.projected_model_count(&destinations)?);
let destination_rule = rules.clone().exists_vars(&[encrypted.var, notifications.var])?;
println!("Destination rule over the full vtree: {}", destination_rule.model_count()?);
println!("Destination rule projected: {}", destination_rule.projected_model_count(&destinations)?);
```

Output:

```text
Valid destination choices: 3
Destination rule over the full vtree: 12
Destination rule projected: 3
```

Existential quantification builds the destination rule **L ∨ R**. Like
substitution, it leaves the vtree unchanged; E and N are now free. Use
[`projected_model_count`](crate::Tdd::projected_model_count) when you need only
the number of distinct destination choices, and
[`exists_vars`](crate::Tdd::exists_vars) when you need the rule itself for further
composition, as in the [reachability example](crate::guide::examples::reachability).

The [complete program](https://github.com/Tractables/tididi/blob/v0.1.0/examples/counting_choices.rs)
contains these queries.
