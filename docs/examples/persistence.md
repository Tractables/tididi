# Save, reload, and combine diagrams

A service might compile its rules once and load them for each new session.
Here we save two rules separately, discard their original diagrams, then
restore and combine them. The saved vtree tells the reader how both diagrams
interpret their variables.

Run the complete program with `cargo run --example save_reload`; only `tididi`
is needed as a dependency. It uses byte buffers so you can run it without
creating files.

```rust,ignore
use std::sync::Arc;

use tididi::{and, Tdd, Vtree};
use tididi::io::{read_tdd, write_tdd};
```

## Build two rules in one variable domain

Variables 1, 2, and 3 mean local backups, remote backups, and encryption.
Require a destination, and require encryption whenever remote backups are on:

```rust,ignore
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

```rust,ignore
let vtree_text = vtree.to_text();
let mut destination_bytes = Vec::new();
let mut encryption_bytes = Vec::new();
write_tdd(&mut destination_bytes, &destination)?;
write_tdd(&mut encryption_bytes, &encryption_rule)?;
drop((destination, encryption_rule, vtree));
```

The serialized data is now all that remains. For files, use
[`save_tdd`](crate::io::save_tdd) and [`load_tdd`](crate::io::load_tdd), and
write the vtree text alongside them. Store application variable names with
that data too: the diagram format identifies variables by number.

Serialization preserves Boolean structure. It does not save attached weights;
restore those separately. Save before marginalizing, since discarded structure
cannot be reconstructed from its count or weighted value.

## Restore the vtree once

Read the saved vtree into one `Arc`, then pass that same allocation to both
readers:

```rust,ignore
let restored_vtree = Arc::new(Vtree::from_text(&vtree_text)?);
let destination = read_tdd(&mut destination_bytes.as_slice(), &restored_vtree)?;
let encryption_rule = read_tdd(&mut encryption_bytes.as_slice(), &restored_vtree)?;
assert!(Arc::ptr_eq(destination.vtree(), encryption_rule.vtree()));
```

Reading the vtree twice would make two separate allocations. Even if their
text is identical, diagrams on those separate vtrees cannot be conjoined
directly. Use the same restored vtree for diagrams you intend to combine.

## Combine and check the result

The loaded diagrams support the same operations as freshly built ones:

```rust,ignore
let configurations = and(destination, encryption_rule)?;
assert_eq!(configurations.model_count()?, 4u32.into());
```

There are four valid assignments: local-only backups with either encryption
setting, remote-only encrypted backups, and both destinations with encryption.
Unlike the first walkthrough, this vtree has no notification variable, so there
is no additional factor of two.

The program also checks functional equality against freshly constructed rules;
a matching model count alone would not establish that the rules survived:

```rust,ignore
let expected = and(
    Tdd::clause(&restored_vtree, [1, 2])?,
    Tdd::clause(&restored_vtree, [-2, 3])?,
)?;
assert!(configurations.equivalent(&expected)?);
```

The complete program returns `Result<(), Box<dyn std::error::Error>>` so `?`
can propagate vtree, I/O, and operation errors. See
[`read_tdd`](crate::io::read_tdd) for the format checks made during loading.

The [complete program](https://github.com/Tractables/tididi/blob/main/examples/save_reload.rs)
puts these steps together; the [API overview](crate::guide::api) lists the other
operations available on restored diagrams.
