<!-- scenario: docs/scenarios.md#probability -->

# Reevaluate a probability model

Suppose rain and a sprinkler independently make the grass wet. After observing
wet grass, how likely is rain? We will build the Boolean events once, then
evaluate them under several choices of the input probabilities.

The Boolean event is **wet = rain ∨ sprinkler**: either cause makes the grass
wet. We assume independent inputs. A third variable, wind, stays free; its
probability should not affect the answer.

Run the complete program with `cargo run --example probability`.
The external crate `num-rational` provides exact fractions; `num-traits`
provides numeric operations such as `one()`. In a separate application, add `num-rational = "0.4"` and `num-traits = "0.2"`
alongside `tididi` to use the exact arithmetic shown here.

## Build the events

Variables 1 and 2 represent rain and the sprinkler; variable 3 represents wind.

```rust,ignore,{class=tested-example}
use std::sync::Arc;

use num_rational::BigRational;
use num_traits::One;
use tididi::diagram::{LiteralWeights, RationalWeights};
use tididi::{literal, Vtree};

let vtree = Arc::new(Vtree::balanced(3));
let rain = literal(&vtree, 1)?;
let sprinkler = literal(&vtree, 2)?;
let wet = rain.clone() | sprinkler;
let rain_and_wet = rain & wet.clone();
```

For any query event `Q` and evidence `E`, conditional probability is
`P(Q ∧ E) / P(E)`. Here `Q` is rain and `E` is wet grass. Rain implies wet
grass in this model, but building their conjunction also works for queries
that do not imply the evidence.

## Assign weights to true and false

Each input has a weight for true and one for false. They sum to one. The
helpers below keep the arithmetic exact. Compare two rain probabilities while
keeping the sprinkler probability fixed:

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

let scenarios = [
    (fraction(1, 5), fraction(1, 10)),
    (fraction(3, 5), fraction(1, 10)),
];
```

## Compute a conditional probability

Evaluation sums the probabilities of the event’s satisfying assignments.
The weight table lists rain, sprinkler, then wind; wind’s two weights sum to
one, so it leaves the result unchanged.

For each scenario, compute `P(rain ∧ wet) / P(wet)`:

```rust,ignore,{class=tested-example}
for (rain_probability, sprinkler_probability) in scenarios {
    // Rain, sprinkler and wind have independent priors.
    let weights = RationalWeights::from_literals(&[
        bernoulli(rain_probability.clone()),
        bernoulli(sprinkler_probability),
        bernoulli(fraction(2, 5)), // free wind contributes 2/5 + 3/5 = 1
    ]);

    let wet_probability = wet.evaluate(&weights)?;
    let conditional = rain_and_wet.evaluate(&weights)? / &wet_probability;

    println!("P(rain) = {rain_probability}, P(wet) = {wet_probability}");
    println!("P(rain | wet) = {conditional}");
}
```

Output:

```text
P(rain) = 1/5, P(wet) = 7/25
P(rain | wet) = 5/7
P(rain) = 3/5, P(wet) = 16/25
P(rain | wet) = 15/16
```

## Change observations

[`evaluator`](crate::Tdd::evaluator) retains values for repeated queries under
evidence; here we observe that it did not rain, then remove that observation.

```rust,ignore,{class=tested-example}
let weights = RationalWeights::from_literals(&[
    bernoulli(fraction(3, 5)),
    bernoulli(fraction(1, 10)),
    bernoulli(fraction(2, 5)),
]);
let mut evaluator = wet.evaluator(weights)?;
evaluator.observe([-1])?;
println!("P(wet and no rain) = {}", evaluator.value()?);
evaluator.clear_pins();
println!("P(wet) = {}", evaluator.value()?);
```

Output:

```text
P(wet and no rain) = 1/25
P(wet) = 16/25
```

The [complete program](https://github.com/Tractables/tididi/blob/v0.1.0/examples/probability.rs)
includes both the changing priors and observations.
