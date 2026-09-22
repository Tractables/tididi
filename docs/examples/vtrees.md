<!-- scenario: docs/scenarios.md#vtrees -->

# Choose a variable grouping

The same Boolean function can need different amounts of storage under different
vtrees. Consider `(x1 ↔ x3) ∧ (x2 ↔ x4)`: two independent pairs of variables
must agree. We will build it under two balanced vtrees and compare the minimized
representations.

Run `cargo run --example vtree_grouping`.

## Keep each equality together, or split both

The vtrees have the same shape and variables. Only the leaf order changes:

<img src="https://raw.githubusercontent.com/Tractables/tididi/v0.1.0/docs/vtree-grouping.svg" alt="Two balanced vtrees: the grouped vtree pairs x1 with x3 and x2 with x4; the split vtree pairs x1 with x2 and x3 with x4." width="760">

[`Vtree::balanced_over`](crate::Vtree::balanced_over) takes the
variable identifiers in left-to-right leaf order. Changing their positions
does not rename them: `VarId(3)` still means `x3`.

```rust,ignore,{class=tested-example}
use std::sync::Arc;

use tididi::{literal, xor, OperationError, Tdd, Vtree};
use tididi::vtree::VarId;

let grouped_vtree = Arc::new(Vtree::balanced_over(&[
    VarId(1), VarId(3), VarId(2), VarId(4),
])?);
let split_vtree = Arc::new(Vtree::balanced(4));
```

## Build the same formula on each vtree

Two variables agree when their exclusive OR is false: `x1 ↔ x3` is
`¬(x1 XOR x3)`. Use [`xor`](crate::xor) to build each disagreement, negate it,
then conjoin the equalities. The helper minimizes before comparing storage:

```rust,ignore,{class=tested-example}
fn equal_pairs(vtree: &Arc<Vtree>) -> Result<Tdd, OperationError> {
    let x1 = literal(vtree, 1)?;
    let x2 = literal(vtree, 2)?;
    let x3 = literal(vtree, 3)?;
    let x4 = literal(vtree, 4)?;
    let first_equal = !xor(x1, x3)?;
    let second_equal = !xor(x2, x4)?;
    let mut f = first_equal & second_equal;
    f.minimize()?;
    Ok(f)
}

let grouped = equal_pairs(&grouped_vtree)?;
let split = equal_pairs(&split_vtree)?;
println!("Models: grouped = {}, split = {}",
    grouped.model_count()?, split.model_count()?);
println!("Grouped equalities: {} pairs; split equalities: {} pairs",
    grouped.pair_count(), split.pair_count());
```

Output:

```text
Models: grouped = 4, split = 4
Grouped equalities: 5 pairs; split equalities: 12 pairs
```

Each equality allows both variables to be false or both to be true, giving
four models in either case. The grouped representation stores fewer pairs.

## Account for the stored pairs

In the grouped vtree, each child of the root represents a complete equality.
Each equality needs two pairs, and the root conjoins them with one pair.
In the split vtree, the root must match each of four left assignments with its
corresponding right assignment:

| Grouping | Pairs at left level | Pairs at right level | Pairs at root | Total |
|---|---|---|---|---|
| Equalities grouped | 2 | 2 | 1 | 5 |
| Equalities split | 4 | 4 | 4 | 12 |

Both representations describe the same four models; the table measures
the pairs needed to store them.

## Try other groupings

Start with a balanced vtree, then try keeping related variables in the same
subtree. Compare the resulting sizes for your own constraints.

[`Tdd::rotation_search`](crate::Tdd::rotation_search) can search vtree
changes on an existing diagram. The [data model](crate::guide::model) explains
how those decompositions represent functions, and the
[complete program](https://github.com/Tractables/tididi/blob/v0.1.0/examples/vtree_grouping.rs)
runs this comparison.
