<!-- scenario: docs/scenarios.md#minimum-cost -->

# Find the cost of the cheapest configuration

The [configuration walkthrough](crate::guide::examples::configurations) counts
valid backup configurations. Suppose each enabled option also has a cost.
Compute the minimum total cost by evaluating the same circuit with costs.

This advanced example implements [`EvalAlgebra`](crate::diagram::EvalAlgebra).
Run it with `cargo run --example minimum_cost`.

## Keep the rules, assign costs

Use the same four variables: local backups, remote backups, encryption and
notifications. At least one destination is required, and remote backups
require encryption: **(local ∨ remote) ∧ (¬remote ∨ encrypted)**.

```rust,ignore,{class=tested-example}
use std::sync::Arc;
use tididi::{literal, Vtree};

let vtree = Arc::new(Vtree::balanced(4));
let local = literal(&vtree, 1)?;
let remote = literal(&vtree, 2)?;
let encrypted = literal(&vtree, 3)?;
let configurations = (local | remote.clone()) & (!remote.clone() | encrypted.clone());
```

## Define the cost calculation

Within the stored circuit, each OR chooses among alternatives and each AND
combines disjoint variable sets. Choose the cheaper feasible alternative at an
OR and add costs at an AND; each option is then charged once. These rules form a *min-plus algebra*:

| Circuit component | Cost |
|---|---|
| False | No feasible assignment (`None`) |
| Enabled option | Its table entry |
| Disabled option | Zero |
| Free option | The cheaper setting: zero here |
| OR | Minimum feasible cost |
| AND | Sum, if both children are feasible |

Store one enabling cost per option, in variable order. Disabling an option
costs zero. `None` means that no configuration is possible:

```rust,ignore,{class=tested-example}
use tididi::diagram::{EvalAlgebra, LeafLabel};
use tididi::vtree::VarId;

struct Costs([u32; 4]);

impl EvalAlgebra for Costs {
    type Value = Option<u64>;

    fn zero(&self) -> Self::Value { None }

    fn leaf(&self, var: VarId, label: LeafLabel) -> Self::Value {
        match label {
            LeafLabel::Pos => Some(u64::from(self.0[var.idx()])),
            LeafLabel::Neg | LeafLabel::One => Some(0),
            LeafLabel::Zero => None,
        }
    }

    fn add_assign(&self, best: &mut Self::Value, candidate: &Self::Value) {
        *best = match (*best, *candidate) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        };
    }

    fn mul(&self, left: &Self::Value, right: &Self::Value) -> Self::Value {
        // A valid circuit combines disjoint variables; four u32 costs fit in u64.
        Some((*left)? + (*right)?)
    }
}
```

## Query the minimum

With costs 5 for local backups, 2 for remote backups and 1 for encryption,
the cheapest valid configuration enables remote backups and encryption:

```rust,ignore,{class=tested-example}
let costs = Costs([5, 2, 1, 0]);
let minimum = configurations.evaluate(&costs)?.expect("the rules have a solution");
println!("Minimum configuration cost: {minimum}");
```

Output:

```text
Minimum configuration cost: 3
```

To recover a cheapest assignment as well as its cost, an evaluator would
also need to track which alternatives attain the minimum.

## Change prices or add a requirement

If local backups fall in price, reevaluate the existing circuit:

```rust,ignore,{class=tested-example}
let local_discount = Costs([1, 2, 1, 0]);
println!("Minimum with discount: {:?}", configurations.evaluate(&local_discount)?);
```

Output:

```text
Minimum with discount: Some(1)
```

Selecting remote backups still incurs its enabling and encryption costs.
Disabling encryption at the same time makes the choices infeasible:

```rust,ignore,{class=tested-example}
let with_remote = configurations.clone() & remote;
println!("Minimum with remote backups: {:?}", with_remote.evaluate(&local_discount)?);
let remote_without_encryption = with_remote & !encrypted;
println!("Minimum without encryption: {:?}", remote_without_encryption.evaluate(&costs)?);
```

Output:

```text
Minimum with remote backups: Some(3)
Minimum without encryption: None
```

The [complete program](https://github.com/Tractables/tididi/blob/v0.1.0/examples/minimum_cost.rs)
also checks both price scenarios by enumerating the sixteen assignments.
For evaluation by weighted sums instead of minima, see the
[probability walkthrough](crate::guide::examples::probability).
