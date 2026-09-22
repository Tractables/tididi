<!-- scenario: docs/scenarios.md#care-sets -->

# Simplify under an assumption

A backup rule requires local storage **L** or remote storage **R**:
**F = L ∨ R**. Suppose the deployment already guarantees that remote storage
always accompanies local storage: **C = R → L = ¬R ∨ L**.

Within that deployment, the backup rule can be simplified to **L**. Remote-only
backups would satisfy the original rule, but the deployment excludes them.
The assignments where an assumption holds are called a *care set*.

[`restrict_to_care`](crate::Tdd::restrict_to_care) removes parts of a circuit
that cannot contribute under a care set. It returns a function **G** satisfying
**G ∧ C = F ∧ C**. This is useful when another part of the application already
enforces C and repeated work can use a smaller circuit.

Run `cargo run --example care_sets`.

## Express the rule and the assumption

Build both functions on the same vtree:

```rust,ignore,{class=tested-example}
use std::sync::Arc;
use tididi::{literal, Vtree};

let vtree = Arc::new(Vtree::balanced(2));
let local = literal(&vtree, 1)?;
let remote = literal(&vtree, 2)?;
let rule = local.clone() | remote.clone();
let care = !remote | local;
println!("Original backup choices: {}", rule.model_count()?);
```

Output:

```text
Original backup choices: 3
```

## Remove choices that cannot occur

Keep copies of the original rule and care set so we can compare the results.
Extract the resulting circuit, then minimize it:

```rust,ignore,{class=tested-example}
let mut simplified = rule.clone().restrict_to_care(care.clone())?.into_tdd();
simplified.minimize()?;
println!("Pairs before: {}; after: {}", rule.pair_count(), simplified.pair_count());
```

Output:

```text
Pairs before: 3; after: 1
```

[`RestrictionOutcome`](crate::apply::RestrictionOutcome) also tells a caller
whether restriction changed the diagram; this example only needs the circuit.

## Check what was preserved

Conjoin the assumption with each version to compare the functions on the care
set. Both allow local-only backups and local-plus-remote backups:

```rust,ignore,{class=tested-example}
let valid_before = rule.clone() & care.clone();
let valid_after = simplified.clone() & care;
println!("Equivalent under the assumption: {}", valid_before.equivalent(&valid_after)?);
println!("Choices under the assumption: {}", valid_after.model_count()?);
```

Output:

```text
Equivalent under the assumption: true
Choices under the assumption: 2
```

The two functions need not agree outside the care set. Remote-only backups
provide a concrete difference here:

```rust,ignore,{class=tested-example}
println!("Equivalent everywhere: {}", rule.equivalent(&simplified)?);
let remote_only = [-1, 2];
println!("Remote only, original: {}", rule.condition(remote_only)?.is_sat()?);
println!("Remote only, simplified: {}", simplified.condition(remote_only)?.is_sat()?);
```

Output:

```text
Equivalent everywhere: false
Remote only, original: true
Remote only, simplified: false
```

Use the simplified circuit only while the assumption holds. If the deployment
later permits remote-only backups, return to the original rule. Use conjunction
when you want the exact valid configurations under an assumption; use
[`condition`](crate::Tdd::condition) to substitute specific variable values,
as in the [counting tutorial](crate::guide::examples::counting).

The [complete program](https://github.com/Tractables/tididi/blob/v0.1.0/examples/care_sets.rs)
compares the circuits both inside and outside the care set.
