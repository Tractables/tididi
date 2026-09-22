<!-- scenario: docs/scenarios.md#tables -->

# Build and update a table of allowed choices

Suppose an application stores the permission combinations it allows:

| Read | Write | Share |
| --- | --- | --- |
| true | false | false |
| true | true | false |
| true | false | true |
| true | true | false |

The duplicate row does not add another choice: a Boolean function describes a
set of assignments. We can build that function directly from the table with
[`Tdd::from_models`](crate::Tdd::from_models).

The [complete program](https://github.com/Tractables/tididi/blob/v0.1.0/examples/table_updates.rs)
runs with `cargo run --example table_updates`.

## Build the circuit

Each column becomes a variable. Pack a row into a word with read in bit 0,
write in bit 1 and share in bit 2; for example, `0b101` enables read and share.

```rust,ignore,{class=tested-example}
use std::sync::Arc;

use tididi::vtree::VarId;
use tididi::{literal, Tdd, Vtree};

let vtree = Arc::new(Vtree::balanced(3));
let vars = [VarId(1), VarId(2), VarId(3)];
// Low to high bits: read, write, share.
let rows = [0b001, 0b011, 0b101, 0b011];
let mut permissions = Tdd::from_models(&vtree, &vars, &rows)?;
println!("Distinct permission sets: {}", permissions.model_count()?);
```

Output:

```text
Distinct permission sets: 3
```

## Change the allowed combinations

Now allow all three permissions together and withdraw read-and-write without
sharing. A [`Maintenance`](crate::maintain::Maintenance) batch reuses an index
across these edits; minimize after the batch to remove any redundancy.

```rust,ignore,{class=tested-example}
{
    let mut batch = permissions.maintain()?;
    batch.insert_model([1, 2, 3])?;
    batch.remove_model([1, 2, -3])?;
}
permissions.minimize()?;
println!("After updates: {}", permissions.model_count()?);
```

Output:

```text
After updates: 3
```

The count is still three, but the allowed combinations have changed:

| Read | Write | Share |
| --- | --- | --- |
| true | false | false |
| true | false | true |
| true | true | true |

## Query the updated table

The result supports the same operations as any other circuit. Conjoin a copy
with sharing to select the permission sets that allow it, keeping the original
available for further updates:

```rust,ignore,{class=tested-example}
let share = literal(&vtree, 3)?;
let sharing = permissions.clone() & share;
println!("Permission sets allowing sharing: {}", sharing.model_count()?);
```

Output:

```text
Permission sets allowing sharing: 2
```

## Remove a group of rows

Suppose sharing is withdrawn entirely. A partial assignment names just the
permissions to match: `[3]` selects every row where sharing is true, whatever
its read and write values. Remove those rows in one call:

```rust,ignore,{class=tested-example}
permissions.remove_model([3])?;
println!("After withdrawing sharing: {}", permissions.model_count()?);
```

Output:

```text
After withdrawing sharing: 1
```

Only read access without write or share remains. This changes the allowed table;
it does not change sharing to false in existing rows. See
[`remove_model`](crate::Tdd::remove_model) and [`insert_model`](crate::Tdd::insert_model)
for partial assignments and empty inputs.

To combine relations and eliminate columns, see
[`and_exists`](crate::and_exists) and the [reachability example](crate::guide::examples::reachability).
