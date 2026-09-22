<!-- scenario: docs/scenarios.md#keeping-values -->

# Keep values instead of structure

Suppose two servers each require local or remote backups:
**(L_A ∨ R_A) ∧ (L_B ∨ R_B)**. Each server has three choices, giving nine
configurations. If future queries concern only server B, we can replace server
A's circuit structure with the number of choices it contributes.

This operation is called *marginalization*. It preserves the total count, but
forgets which server A assignments produced it. We will keep the original
circuit as well, so we can compare what each version can answer.

Run `cargo run --example marginalize_components`. The probability section uses
`num-rational = "0.4"` alongside `tididi` for exact fractions.

## Build the two-server model

A balanced four-variable vtree groups server A's options in its left subtree
and server B's in its right subtree:

```rust,ignore,{class=tested-example}
use std::sync::Arc;
use tididi::{Literal, OperationError, Tdd, Vtree};

let vtree = Arc::new(Vtree::balanced(4));
let local_a = Literal::try_from(1)?;
let remote_a = Literal::try_from(2)?;
let local_b = Literal::try_from(3)?;
let remote_b = Literal::try_from(4)?;
let rules = Tdd::clause(&vtree, [local_a, remote_a])?
    & Tdd::clause(&vtree, [local_b, remote_b])?;
println!("Original configurations: {}", rules.model_count()?);
```

Output:

```text
Original configurations: 9
```

## Replace one subtree with counts

[`marginalize_levels`](crate::Tdd::marginalize_levels) takes vtree node indices.
Select the left subtree and summarize it in a copy of the circuit:

```rust,ignore,{class=tested-example}
let (server_a, _) = vtree.children(vtree.root());
let mut counted = rules.clone();
counted.marginalize_levels(&[server_a])?;
println!("After summarizing server A: {}", counted.model_count()?);
println!("Distinct server B choices: {}",
    rules.projected_model_count(&[local_b.var, remote_b.var])?);
```

Output:

```text
After summarizing server A: 9
Distinct server B choices: 3
```

Marginalization keeps server A's three alternatives as a multiplicity, so the
total stays nine. Projection instead counts each server B choice once.
Use the [counting lesson](crate::guide::examples::counting) when your goal is
to count distinct choices for selected variables.

## Ask about the retained variables

A counter can still observe server B. Selecting its remote option leaves two
choices for B, each with three extensions on A, giving six configurations:

```rust,ignore,{class=tested-example}
let mut counter = counted.counter()?;
counter.observe([remote_b])?;
println!("With remote backups on B: {}", counter.model_count()?);
let cannot_observe_a = matches!(counter.observe([remote_a]),
    Err(OperationError::MarginalLevel(_)));
println!("Server A observations need discarded structure: {cannot_observe_a}");
let mut original_counter = rules.counter()?;
original_counter.observe([remote_a, remote_b])?;
println!("Both servers remote, using the original: {}", original_counter.model_count()?);
```

Output:

```text
With remote backups on B: 6
Server A observations need discarded structure: true
Both servers remote, using the original: 4
```

The stored count cannot tell us how many server A alternatives use remote
storage. That query needs the original circuit, which still distinguishes them.

## Keep a probability instead

For independent options that are each enabled with probability 1/2, either
server has a backup with probability 3/4. The combined probability is 9/16.
Attach those weights to a fresh structural copy **before** marginalizing:

```rust,ignore,{class=tested-example}
use num_rational::BigRational;
use tididi::diagram::{Arithmetic, LiteralWeights, RationalWeights, WeightStore};

let half = BigRational::new(1.into(), 2.into());
let weights = RationalWeights::from_literals(&vec![
    LiteralWeights { negative: half.clone(), positive: half }; 4
]);
let mut weighted = rules.clone();
weighted.set_weights(WeightStore::new(weights, Arithmetic::ExactRational))?;
println!("Probability before: {}", weighted.weighted_value()?.unwrap().into_rational());
weighted.marginalize_levels(&[server_a])?;
println!("Probability after: {}", weighted.weighted_value()?.unwrap().into_rational());
```

Output:

```text
Probability before: 9/16
Probability after: 9/16
```

[`WeightStore`](crate::diagram::WeightStore) retains the weighted values and
their interpretation. Read them with [`weighted_value`](crate::Tdd::weighted_value);
ordinary model counting cannot recover counts from weighted summaries.

## Change the probabilities

If each option's probability becomes 1/3, the old summary no longer applies.
The original circuit can evaluate the new table: each server's backup probability
is now 5/9, giving 25/81 overall.

```rust,ignore,{class=tested-example}
let new_weights = RationalWeights::from_literals(&vec![
    LiteralWeights {
        negative: BigRational::new(2.into(), 3.into()),
        positive: BigRational::new(1.into(), 3.into()),
    }; 4
]);
println!("New probabilities, using the original: {}", rules.evaluate(&new_weights)?);
let needs_structure = matches!(weighted.evaluate(&new_weights),
    Err(OperationError::MarginalLevel(_)));
println!("Reevaluation needs discarded structure: {needs_structure}");
```

Output:

```text
New probabilities, using the original: 25/81
Reevaluation needs discarded structure: true
```

Keep structure for changing weights, recovering assignments, or observing the
variables you would otherwise discard. Marginalize a subtree when retaining its
contribution is sufficient. The [complete program](https://github.com/Tractables/tididi/blob/v0.1.0/examples/marginalize_components.rs)
compares both representations.
