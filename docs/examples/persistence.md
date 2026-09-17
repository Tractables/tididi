# Save, reload, and combine diagrams

A service might compile its rules once and load them for each new session.
Here we save two rules separately, discard their original diagrams, then
restore and combine them. The saved vtree tells the reader how both diagrams
interpret their variables.

Run `cargo run --example save_reload`. This example saves to byte buffers;
it does not create files.

```rust,ignore,{class=tested-example}
use std::sync::Arc;

use tididi::{and, Tdd, Vtree};
use tididi::io::{read_tdd, write_tdd};
```

## Build two rules in one variable domain

Variables 1, 2, and 3 mean local backups, remote backups, and encryption.
Require a destination, and require encryption whenever remote backups are on:

```rust,ignore,{class=tested-example}
let vtree = Arc::new(Vtree::balanced(3));
let destination = Tdd::clause(&vtree, [1, 2])?;
let encryption_rule = Tdd::clause(&vtree, [-2, 3])?;
```

Both diagrams share the same vtree. Save that vtree once alongside the two
serialized diagrams:

```text
saved session
  ├─ vtree text
  ├─ destination diagram
  └─ encryption diagram
```

## Write the structure

[`write_tdd`](crate::io::write_tdd) writes to any Rust `Write` stream; a byte
vector is one such stream. The vtree has its own text representation:

```rust,ignore,{class=tested-example}
let vtree_text = vtree.to_text();
let mut destination_bytes = Vec::new();
let mut encryption_bytes = Vec::new();
write_tdd(&mut destination_bytes, &destination)?;
write_tdd(&mut encryption_bytes, &encryption_rule)?;
drop((destination, encryption_rule, vtree));
```

For files, use [`save_tdd`](crate::io::save_tdd) and
[`load_tdd`](crate::io::load_tdd). Save application variable names alongside
them if you need those names later.

Save before marginalizing, and store any weights separately;
[`write_tdd`](crate::io::write_tdd) preserves the Boolean structure.

## Restore the vtree once

Read the saved vtree into one `Arc`, then pass that same allocation to both
readers:

```rust,ignore,{class=tested-example}
let restored_vtree = Arc::new(Vtree::from_text(&vtree_text)?);
let destination = read_tdd(&mut destination_bytes.as_slice(), &restored_vtree)?;
let encryption_rule = read_tdd(&mut encryption_bytes.as_slice(), &restored_vtree)?;
assert!(Arc::ptr_eq(destination.vtree(), encryption_rule.vtree()));
```

Both readers receive the same `Arc<Vtree>`, so their results can be combined.
Reading the vtree separately for each diagram would create incompatible allocations.

## Combine and check the result

The loaded diagrams support the same operations as freshly built ones:

```rust,ignore,{class=tested-example}
let configurations = and(destination, encryption_rule)?;
assert_eq!(configurations.model_count()?, 4u32.into());
println!("Restored rules allow {} configurations", configurations.model_count()?);
```

Output:

```text
Restored rules allow 4 configurations
```

The four choices are local-only backups with encryption off or on,
remote-only encrypted backups, and both destinations with encryption.

Rebuild the rules to check that loading preserved the whole function:

```rust,ignore,{class=tested-example}
let expected = and(
    Tdd::clause(&restored_vtree, [1, 2])?,
    Tdd::clause(&restored_vtree, [-2, 3])?,
)?;
assert!(configurations.equivalent(&expected)?);
```

See [`read_tdd`](crate::io::read_tdd) for format requirements and the
[complete program](https://github.com/Tractables/tididi/blob/main/examples/save_reload.rs)
for the runnable example.
