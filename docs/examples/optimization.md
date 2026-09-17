# Find the cost of the cheapest configuration

The [configuration walkthrough](crate::guide::examples::configurations) counts
valid backup configurations. Suppose each enabled option also has a cost.
Compute the minimum total cost by evaluating the same circuit with costs.

This advanced example implements [`EvalAlgebra`](crate::diagram::EvalAlgebra).
Run it with `cargo run --example minimum_cost`.

```rust,ignore,{class=tested-example}
use std::sync::Arc;
use tididi::diagram::{EvalAlgebra, LeafLabel};
use tididi::vtree::VarId;
use tididi::{and, literal, Tdd, Vtree};
```

## Keep the rules, assign costs

Use the same four variables: local backups, remote backups, encryption and
notifications. At least one destination is required, and remote backups
require encryption:

```rust,ignore,{class=tested-example}
let vtree = Arc::new(Vtree::balanced(4));
let configurations = and(
    Tdd::clause(&vtree, [1, 2])?,
    Tdd::clause(&vtree, [-2, 3])?,
)?;
```

Our cost table has one nonnegative enabling cost per option; disabling an
option costs zero. `None` will mean that no configuration is possible, keeping
that case distinct from a valid configuration with zero cost.

```rust,ignore,{class=tested-example}
struct Costs([u32; 4]);
```

## Define the cost calculation

For an OR, choose the cheaper feasible alternative. For an AND, add the costs
of its children. TDD conjunctions combine disjoint variable sets, so each
option is charged once. These rules form a *min-plus algebra*:

| Circuit component | Cost |
|---|---|
| False | No feasible assignment (`None`) |
| Enabled option | Its table entry |
| Disabled option | Zero |
| Free option | The cheaper setting: zero here |
| OR | Minimum feasible cost |
| AND | Sum, if both children are feasible |

Implement those rules on the table:

```rust,ignore,{class=tested-example}
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

The result uses `u64` so the sum of the four `u32` costs fits.

## Query the minimum

With costs 5 for local backups, 2 for remote backups and 1 for encryption,
the cheapest valid configuration enables remote backups and encryption:

```rust,ignore,{class=tested-example}
let costs = Costs([5, 2, 1, 0]);
assert_eq!(configurations.evaluate(&costs)?, Some(3));
println!("Minimum configuration cost: 3");
```

To recover a cheapest assignment as well as its cost, an evaluator would
also need to track which alternatives attain the minimum.

## Change prices or add a requirement

If local backups fall in price, reevaluate the existing circuit:

```rust,ignore,{class=tested-example}
let local_discount = Costs([1, 2, 1, 0]);
assert_eq!(configurations.evaluate(&local_discount)?, Some(1));
```

Selecting remote backups still incurs its enabling and encryption costs.
Disabling encryption at the same time makes the choices infeasible:

```rust,ignore,{class=tested-example}
let with_remote = and(configurations.clone(), literal(&vtree, 2)?)?;
assert_eq!(with_remote.evaluate(&local_discount)?, Some(3));
let conflicting = and(with_remote, literal(&vtree, -3)?)?;
assert_eq!(conflicting.evaluate(&costs)?, None);
```

The [complete program](https://github.com/Tractables/tididi/blob/main/examples/minimum_cost.rs)
also checks both price scenarios by enumerating the sixteen assignments.
For evaluation by weighted sums instead of minima, see the
[probability walkthrough](crate::guide::examples::probability).
