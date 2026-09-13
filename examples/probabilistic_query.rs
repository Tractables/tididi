//! Compile a query and evidence once, then reevaluate them with new probabilities.
//! Run with `cargo run --example probabilistic_query`.

use std::sync::Arc;

use num_rational::BigRational;
use num_traits::{One, Zero};
use tididi::diagram::{LiteralWeights, RationalWeights};
use tididi::query::evaluate;
use tididi::{Engine, OperationError, Tdd, Vtree};

fn fraction(numerator: i64, denominator: i64) -> BigRational {
    BigRational::new(numerator.into(), denominator.into())
}

fn bernoulli(positive: BigRational) -> LiteralWeights<BigRational> {
    LiteralWeights {
        negative: BigRational::one() - &positive,
        positive,
    }
}

// Both masses use the same weight table; zero-mass evidence has no conditional
// probability. The caller represents that case explicitly with None.
fn conditional_probability(
    query_and_evidence: &Tdd,
    evidence: &Tdd,
    weights: &RationalWeights,
) -> Option<BigRational> {
    let evidence_mass = evaluate(evidence, weights);
    if evidence_mass.is_zero() {
        None
    } else {
        Some(evaluate(query_and_evidence, weights) / evidence_mass)
    }
}

fn main() -> Result<(), OperationError> {
    let engine = Engine::new();
    let tree = Arc::new(Vtree::balanced(3));
    let rain = engine.literal(&tree, 1)?;
    let sprinkler = engine.literal(&tree, 2)?;
    // Wet grass is the observation: rain OR sprinkler. Variable 3 (wind) is free.
    let wet = engine.or(rain.clone(), sprinkler)?;
    let rain_and_wet = engine.and(rain.clone(), wet.clone())?;

    for (rain_probability, sprinkler_probability, expected) in [
        (fraction(1, 5), fraction(1, 10), Some(fraction(5, 7))),
        (fraction(3, 5), fraction(1, 10), Some(fraction(15, 16))),
        (fraction(0, 1), fraction(0, 1), None),
    ] {
        // Rain, sprinkler and wind have independent priors.
        let weights = RationalWeights::from_literals(&[
            bernoulli(rain_probability.clone()),
            bernoulli(sprinkler_probability.clone()),
            bernoulli(fraction(2, 5)), // free wind contributes 2/5 + 3/5 = 1
        ]);

        // The diagrams stay structural and unchanged; each call performs a fresh fold.
        let wet_probability = evaluate(&wet, &weights);
        assert_eq!(evaluate(&rain, &weights), rain_probability);
        assert_eq!(
            wet_probability,
            &rain_probability + &sprinkler_probability - &rain_probability * &sprinkler_probability
        );
        let conditional = conditional_probability(&rain_and_wet, &wet, &weights);
        assert_eq!(conditional, expected);

        println!("P(rain) = {rain_probability}, P(wet) = {wet_probability}");
        match conditional {
            Some(value) => println!("P(rain | wet) = {value}"),
            None => println!("P(rain | wet) is undefined: the observation has probability zero"),
        }
    }
    Ok(())
}
