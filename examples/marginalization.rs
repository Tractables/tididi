// scenario: docs/scenarios.md#keeping-values
// Dependencies beyond tididi: num-rational = "0.4"

fn main() -> Result<(), Box<dyn std::error::Error>> {
    use tididi::vtree::Home;
    use tididi::{Literal, OperationError, Tdd, Vtree};

    let vtree = Home::new(Vtree::balanced(4));
    let local_a = Literal::try_from(1)?;
    let remote_a = Literal::try_from(2)?;
    let local_b = Literal::try_from(3)?;
    let remote_b = Literal::try_from(4)?;
    let rules = Tdd::clause(&vtree, [local_a, remote_a])?
        & Tdd::clause(&vtree, [local_b, remote_b])?;
    println!("Original configurations: {}", rules.model_count()?);

    let server_a = |v| v == local_a.var || v == remote_a.var;
    let mut counted = rules.clone();
    counted.marginalize(server_a)?;
    println!("After summarizing server A: {}", counted.model_count()?);
    println!("Distinct server B choices: {}",
        rules.projected_model_count(&[local_b.var, remote_b.var])?);

    let mut counter = counted.counter()?;
    counter.observe([remote_b])?;
    println!("With remote backups on B: {}", counter.model_count()?);
    let cannot_observe_a = matches!(counter.observe([remote_a]),
        Err(OperationError::MarginalLevel(_)));
    println!("Server A observations need discarded structure: {cannot_observe_a}");
    let mut original_counter = rules.counter()?;
    original_counter.observe([remote_a, remote_b])?;
    println!("Both servers remote, using the original: {}", original_counter.model_count()?);

    use num_rational::BigRational;
    use tididi::diagram::{Arithmetic, LiteralWeights, RationalWeights, Weights};

    let half = BigRational::new(1.into(), 2.into());
    let weights = RationalWeights::from_literals(&vec![
        LiteralWeights { negative: half.clone(), positive: half }; 4
    ]);
    let weighted_vtree = Home::weighted(Vtree::balanced(4), Weights::new(weights, Arithmetic::ExactRational))?;
    let mut weighted = Tdd::clause(&weighted_vtree, [local_a, remote_a])?
        & Tdd::clause(&weighted_vtree, [local_b, remote_b])?;
    println!("Probability before: {}", weighted.weighted_value()?.unwrap().into_rational());
    weighted.marginalize(server_a)?;
    println!("Probability after: {}", weighted.weighted_value()?.unwrap().into_rational());

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
    Ok(())
}
