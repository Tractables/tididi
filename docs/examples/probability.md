# Reevaluate a probability model

Suppose rain and a sprinkler independently make the grass wet. After observing
wet grass, how likely is rain? We will build the Boolean events once, then
evaluate them under several choices of the input probabilities.

The model is deliberately small:

```text
rain ───────┐
            OR ─── wet grass
sprinkler ──┘
```

It assumes either cause always makes the grass wet. A third variable, wind,
will stay free so we can see how an irrelevant variable affects the result.

Run the complete program with `cargo run --example probabilistic_query`.
In a separate application, add `num-rational = "0.4"` and `num-traits = "0.2"`
alongside `tididi` to use the exact arithmetic shown here.

```rust,ignore,{class=tested-example}
use std::sync::Arc;

use num_rational::BigRational;
use num_traits::{One, Zero};
use tididi::diagram::{LiteralWeights, RationalWeights};
use tididi::{and, literal, or, OperationError, Tdd, Vtree};
```

## Build the events once

Build the events on one shared vtree.

```rust,ignore,{class=tested-example}
let vtree = Arc::new(Vtree::balanced(3));
let rain = literal(&vtree, 1)?;
let sprinkler = literal(&vtree, 2)?;
// Wet grass is the observation: rain OR sprinkler. Variable 3 (wind) is free.
let wet = or(rain.clone(), sprinkler)?;
let rain_and_wet = and(rain.clone(), wet.clone())?;
```

For any query event `Q` and evidence `E`, conditional probability is
`P(Q AND E) / P(E)`. Here `Q` is rain and `E` is wet grass. Rain implies wet
grass in this model, but building their conjunction also works for queries
that do not imply the evidence.

## Assign weights to true and false

Each independent Boolean input has two weights: its probability of being true
and its probability of being false. These helpers keep the values exact:

```rust,ignore,{class=tested-example}
fn fraction(numerator: i64, denominator: i64) -> BigRational {
    BigRational::new(numerator.into(), denominator.into())
}

fn bernoulli(positive: BigRational) -> LiteralWeights<BigRational> {
    LiteralWeights {
        negative: BigRational::one() - &positive,
        positive,
    }
}
```

The program loops over three choices of input probabilities and an expected
answer. The final scenario makes the evidence impossible:

```rust,ignore,{class=tested-example}
let scenarios = [
    (fraction(1, 5), fraction(1, 10), Some(fraction(5, 7))),
    (fraction(3, 5), fraction(1, 10), Some(fraction(15, 16))),
    (fraction(0, 1), fraction(0, 1), None),
];
```

For each `(rain_probability, sprinkler_probability, expected)` in `scenarios`,
build a table from the current probabilities:

```rust,ignore,{class=tested-example}
let weights = RationalWeights::from_literals(&[
    bernoulli(rain_probability.clone()),
    bernoulli(sprinkler_probability.clone()),
    bernoulli(fraction(2, 5)), // free wind contributes 2/5 + 3/5 = 1
]);
```

Entries follow variable order. Every satisfying assignment contributes the
product of its literal weights; evaluation sums those products. Wind appears
in neither event, and its two weights sum to one, so it does not change either
probability. Independence is a modeling assumption of this weight table.

## Evaluate an event

Call [`evaluate`](crate::Tdd::evaluate) on an event to obtain its probability
under the current weights:

```rust,ignore,{class=tested-example}
let wet_probability = wet.evaluate(&weights)?;
assert_eq!(rain.evaluate(&weights)?, rain_probability);
```

The result is an exact rational number. The `?` propagates an evaluation error
from the enclosing function.

## Divide by the evidence mass

Zero-probability evidence has no conditional probability. The helper returns
`None` for that case instead of dividing by zero:

```rust,ignore,{class=tested-example}
fn conditional_probability(
    query_and_evidence: &Tdd,
    evidence: &Tdd,
    weights: &RationalWeights,
) -> Result<Option<BigRational>, OperationError> {
    let evidence_mass = evidence.evaluate(weights)?;
    if evidence_mass.is_zero() {
        Ok(None)
    } else {
        Ok(Some(query_and_evidence.evaluate(weights)? / evidence_mass))
    }
}
```

The caller supplies the same weights for the numerator and denominator:

```rust,ignore,{class=tested-example}
let conditional = conditional_probability(&rain_and_wet, &wet, &weights)?;
assert_eq!(conditional, expected);
```

## Change probabilities, keep the diagrams

The program checks three scenarios:

| P(rain) | P(sprinkler) | P(wet) | P(rain given wet) |
|---|---|---|---|
| 1/5 | 1/10 | 7/25 | 5/7 |
| 3/5 | 1/10 | 16/25 | 15/16 |
| 0 | 0 | 0 | undefined |

Each [`Tdd::evaluate`](crate::Tdd::evaluate) call reads the new table
without changing the structural diagram. This lets an application update
probabilities while keeping the compiled logical events.

See the [complete program](https://github.com/Tractables/tididi/blob/main/examples/probabilistic_query.rs)
for the scenario loop and its assertions, and the
[API overview](crate::guide::api) for other weighted queries.
