// scenario: docs/scenarios.md#probability

//! Compile a query and evidence once, then reevaluate them with new probabilities.
//! Run with `cargo run --example probabilistic_query`.

fn main() -> Result<(), tididi::OperationError> {
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
    Ok(())
}
