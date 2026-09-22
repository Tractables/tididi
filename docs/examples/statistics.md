<!-- scenario: docs/scenarios.md#statistics -->

# Inspect the stored circuit

Find the circuit node with the most child pairs and the vtree level where
it lives. This helps locate the largest decomposition in a stored diagram.

Run the complete program with `cargo run --example statistic`. Read the
[data model](crate::guide::model) first
if the distinction between a vtree level, a TDD node, and a child pair is new.

## Inspect an exclusive disjunction

Use a balanced vtree over four variables. The exclusive OR of its first two
variables is **(x1 ∧ ¬x2) ∨ (¬x1 ∧ x2)**. Each alternative is one child pair
at the level containing those variables. Build it with [`xor`](crate::xor),
copying `x1` so we can inspect the literal separately:

```rust,ignore,{class=tested-example}
use std::sync::Arc;

use tididi::{literal, xor, Tdd};
use tididi::vtree::{Vtree, VtreeIdx};

let vtree = Arc::new(Vtree::balanced(4));
let x1 = literal(&vtree, 1)?;
let x2 = literal(&vtree, 2)?;
let exclusive = xor(x1.clone(), x2)?;
```

## Find the widest node

At each level, `internal_inputs_iter()` yields each stored node's identifier
and an iterator over its pairs; a leaf level or a marginalized level yields
none. The iterator knows its length, so obtaining the pair count does not
require decoding the children.

The helper returns `(root, 0)` if there are no stored nodes and keeps the
first maximum on ties.

```rust,ignore,{class=tested-example}
fn widest_node(circuit: &Tdd) -> (VtreeIdx, usize) {
    let mut best = (circuit.vtree().root(), 0usize);
    for v in circuit.vtree().bottomup() {
        for (_, pairs) in circuit.level(v).internal_inputs_iter() {
            let n = pairs.len();
            if n > best.1 {
                best = (v, n);
            }
        }
    }
    best
}

let (level, pairs) = widest_node(&exclusive);
println!("Widest XOR node: {pairs} pairs at vtree node {}", level.idx());
```

Output:

```text
Widest XOR node: 2 pairs at vtree node 4
```

The other two variables are free. For comparison, a single positive literal
needs just one pair per stored node:

```rust,ignore,{class=tested-example}
println!("Widest literal node: {} pair", widest_node(&x1).1);
```

Output:

```text
Widest literal node: 1 pair
```

## Compare with total storage

The built-in [`pair_count`](crate::Tdd::pair_count) sums pairs over all stored
nodes. Compare that storage with the number of satisfying assignments:

```rust,ignore,{class=tested-example}
println!("Total XOR pairs: {}", exclusive.pair_count());
println!("Satisfying assignments: {}", exclusive.model_count()?);
```

Output:

```text
Total XOR pairs: 4
Satisfying assignments: 8
```

Exactly one of `x1` and `x2` is true, giving two choices. Each of the other
two variables is free, so the model count is 2 × 2 × 2 = 8. The four stored
pairs describe those eight assignments.

Adapt the same traversal for a histogram of node sizes or a per-level report.
The [complete program](https://github.com/Tractables/tididi/blob/v0.1.0/examples/statistic.rs)
includes the runnable entry point.
