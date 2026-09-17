# Reevaluate a probability model

Suppose rain and a sprinkler independently make the grass wet. After observing
wet grass, how likely is rain? We will build the Boolean events once, then
evaluate them under several choices of the input probabilities.

The model is:

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

## Build the events

```rust,ignore,{class=tested-example}
let vtree = Arc::new(Vtree::balanced(3));
let rain = literal(&vtree, 1)?;
let sprinkler = literal(&vtree, 2)?;
// Wet grass is the evidence: rain OR sprinkler. Variable 3 (wind) is free.
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

## Compute a conditional probability

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

The caller supplies the same weight table for both events. Evaluation
multiplies the literal weights in each satisfying assignment, then sums over
assignments.

## Try different probabilities

Here are three scenarios, with the expected conditional probability in the
last column. In the third case, wet grass has probability zero:

```rust,ignore,{class=tested-example}
let scenarios = [
    (fraction(1, 5), fraction(1, 10), Some(fraction(5, 7))),
    (fraction(3, 5), fraction(1, 10), Some(fraction(15, 16))),
    (fraction(0, 1), fraction(0, 1), None),
];
```

Build a weight table for each scenario, in variable order: rain, sprinkler,
then wind. The wind weights sum to one, so this unused variable leaves the
probabilities unchanged. The table assumes independent inputs.

```rust,ignore,{class=tested-example}
for (rain_probability, sprinkler_probability, expected) in scenarios {
    // Rain, sprinkler and wind have independent priors.
    let weights = RationalWeights::from_literals(&[
        bernoulli(rain_probability.clone()),
        bernoulli(sprinkler_probability.clone()),
        bernoulli(fraction(2, 5)), // free wind contributes 2/5 + 3/5 = 1
    ]);

    // The diagrams stay structural and unchanged; each call performs a fresh fold.
    let wet_probability = wet.evaluate(&weights)?;
    assert_eq!(rain.evaluate(&weights)?, rain_probability);
    assert_eq!(
        wet_probability,
        &rain_probability + &sprinkler_probability - &rain_probability * &sprinkler_probability
    );
    let conditional = conditional_probability(&rain_and_wet, &wet, &weights)?;
    assert_eq!(conditional, expected);

    println!("P(rain) = {rain_probability}, P(wet) = {wet_probability}");
    match conditional {
        Some(value) => println!("P(rain | wet) = {value}"),
        None => println!("P(rain | wet) is undefined: the evidence has probability zero"),
    }
}
```

The same diagrams give all three results:

| P(rain) | P(sprinkler) | P(wet) | P(rain given wet) |
|---|---|---|---|
| 1/5 | 1/10 | 7/25 | 5/7 |
| 3/5 | 1/10 | 16/25 | 15/16 |
| 0 | 0 | 0 | undefined |

[`evaluate`](crate::Tdd::evaluate) reads each new weight table without changing
the diagrams. The [complete program](https://github.com/Tractables/tididi/blob/main/examples/probabilistic_query.rs)
puts the helpers and scenario loop together.
