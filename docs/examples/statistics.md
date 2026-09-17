# Inspect the stored circuit

Find the circuit node with the most child pairs and the vtree level where
it lives. This helps locate the largest decomposition in a stored diagram.


Run the complete program with `cargo run --example statistic`. Read the
[data model](crate::guide::model) first
if the distinction between a vtree level, a TDD node, and a child pair is new.

```rust,ignore,{class=tested-example}
use std::sync::Arc;

use tididi::{and, literal, OperationError, Tdd};
use tididi::vtree::{Vtree, VtreeIdx};
```

## Walk levels, then nodes

The traversal uses three nested views:

```text
vtree level
  └─ stored TDD node
       ├─ (left child, right child)
       └─ (left child, right child)
```

At each level, `internal_inputs_iter()` yields each stored node's identifier
and an iterator over its pairs; a leaf level or a marginalized level yields
none. The iterator knows its length, so obtaining the pair count does not
require decoding the children:

```rust,ignore,{class=tested-example}
fn widest_node(t: &Tdd) -> (VtreeIdx, usize) {
    let mut best = (t.vtree().root(), 0usize);
    for v in t.vtree().bottomup() {
        // Leaf levels store nothing and marginal levels have dropped their
        // pairs; neither yields a node here.
        for (_i, pairs) in t.level(v).internal_inputs_iter() {
            let n = pairs.len(); // PairsIter is ExactSizeIterator
            if n > best.1 {
                best = (v, n);
            }
        }
    }
    best
}
```

The helper returns `(root, 0)` if there are no stored nodes and keeps the
first maximum on ties.

## Inspect an exclusive disjunction

Use a balanced vtree over four variables. The exclusive OR of its first two
variables has two alternatives: `(x1, NOT x2)` and `(NOT x1, x2)`.
Each alternative is one pair at the level containing those variables:

```rust,ignore,{class=tested-example}
let vtree = Arc::new(Vtree::balanced(4));
// x1 ⊕ x2 needs two pairs at the level over {x1, x2}: (x1, ¬x2) and (¬x1, x2).
let xor = and(Tdd::clause(&vtree, [1, 2])?, Tdd::clause(&vtree, [-1, -2])?)?;
let (level, pairs) = widest_node(&xor);
let (left, _) = vtree.children(vtree.root());
assert_eq!((level, pairs), (left, 2));
```

The other two variables are free. For comparison, a single positive literal
needs just one pair per stored node:

```rust,ignore,{class=tested-example}
let unit = literal(&vtree, 1)?;
assert_eq!(widest_node(&unit).1, 1);
```

## Compare with total storage

The built-in `pair_count()` sums pairs over all stored nodes. Our largest
individual node cannot exceed that total:

```rust,ignore,{class=tested-example}
assert!(widest_node(&xor).1 <= xor.pair_count());
println!("statistic: widest node has {pairs} pairs at vtree node {}", level.idx());
```

Output:

```text
statistic: widest node has 2 pairs at vtree node 4
```

Adapt the same traversal for a histogram of node sizes or a per-level report.
The [complete program](https://github.com/Tractables/tididi/blob/main/examples/statistic.rs)
includes the runnable entry point.
