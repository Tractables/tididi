# Choose a variable grouping

The same Boolean function can need different amounts of storage under different
vtrees. Consider `(x1 ↔ x3) ∧ (x2 ↔ x4)`: two independent pairs of variables
must agree. We will build it under two balanced vtrees and compare the minimized
representations.

Run `cargo run --example vtree_grouping`.

```rust,ignore,{class=tested-example}
use std::sync::Arc;

use tididi::{and, OperationError, Tdd, Vtree};
use tididi::vtree::VarId;
```

## Keep each equality together, or split both

The vtrees have the same shape and variables. Only the leaf order changes:

```text
Equalities grouped                  Equalities split

         root                                root
        /    \                              /    \
       /      \                            /      \
      •        •                          •        •
     / \      / \                        / \      / \
    x1  x3   x2  x4                      x1  x2   x3  x4
```

[`Vtree::balanced_over`](crate::Vtree::balanced_over) takes the zero-based
variable identifiers in left-to-right leaf order. Changing their positions
does not rename them: `VarId(2)` still means `x3`.

```rust,ignore,{class=tested-example}
let grouped_vtree = Arc::new(Vtree::balanced_over(&[
    VarId(0), VarId(2), VarId(1), VarId(3),
])?);
let split_vtree = Arc::new(Vtree::balanced(4));
```

## Build the same formula on each vtree

Equality is a pair of implications: `x1 ↔ x3` is
`(¬x1 ∨ x3) ∧ (x1 ∨ ¬x3)`. The helper uses the same literal numbers for either
vtree, then minimizes before comparing storage:

```rust,ignore,{class=tested-example}
fn equal_pairs(vtree: &Arc<Vtree>) -> Result<Tdd, OperationError> {
    let first_equal = and(Tdd::clause(vtree, [-1, 3])?, Tdd::clause(vtree, [1, -3])?)?;
    let second_equal = and(Tdd::clause(vtree, [-2, 4])?, Tdd::clause(vtree, [2, -4])?)?;
    let mut f = and(first_equal, second_equal)?;
    f.minimize()?;
    Ok(f)
}
```

```rust,ignore,{class=tested-example}
let grouped = equal_pairs(&grouped_vtree)?;
let split = equal_pairs(&split_vtree)?;
assert_eq!(grouped.model_count()?, 4u32.into());
assert_eq!(split.model_count()?, 4u32.into());
assert_eq!(grouped.pair_count(), 5);
assert_eq!(split.pair_count(), 12);
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
[complete program](https://github.com/Tractables/tididi/blob/main/examples/vtree_grouping.rs)
checks the counts and pair totals above.
