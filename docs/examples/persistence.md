# Save, reload, and combine diagrams

A service might compile its rules once and load them for each new session.
Here we save two rules separately, discard their original diagrams, then
restore and combine them. The saved vtree tells the reader how both diagrams
interpret their variables.

Run `cargo run --example save_reload`. This example saves to byte buffers;
it does not create files.

## Build two rules in one variable domain

Variables 1, 2, and 3 mean local backups, remote backups, and encryption.
The rules are **local ∨ remote** and **¬remote ∨ encrypted**:

```rust,ignore,{class=tested-example}
use std::sync::Arc;

use tididi::{literal, Vtree};
use tididi::io::{read_tdd, write_tdd};

let vtree = Arc::new(Vtree::balanced(3));
let local = literal(&vtree, 1)?;
let remote = literal(&vtree, 2)?;
let encrypted = literal(&vtree, 3)?;
let destination = local | remote.clone();
let encryption_rule = !remote | encrypted;
```

Save their shared vtree once alongside the two diagrams.

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
```

Both readers receive the same `Arc<Vtree>`, so their results can be combined.
Reading the vtree separately for each diagram would create incompatible allocations.

## Combine and check the result

The loaded diagrams support the same operations as freshly built ones:

```rust,ignore,{class=tested-example}
let configurations = destination & encryption_rule;
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
let local = literal(&restored_vtree, 1)?;
let remote = literal(&restored_vtree, 2)?;
let encrypted = literal(&restored_vtree, 3)?;
let expected = (local | remote.clone()) & (!remote | encrypted);
println!("Equivalent to the original rules: {}", configurations.equivalent(&expected)?);
```

Output:

```text
Equivalent to the original rules: true
```

See [`read_tdd`](crate::io::read_tdd) for format requirements and the
[complete program](https://github.com/Tractables/tididi/blob/main/examples/save_reload.rs)
for the runnable example.
