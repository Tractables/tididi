// scenario: docs/scenarios.md#counting-choices

fn main() -> Result<(), tididi::OperationError> {
    use std::sync::Arc;
    use tididi::{literal, Literal, Tdd, Vtree};

    let vtree = Arc::new(Vtree::balanced(4));
    let local = Literal::try_from(1)?;
    let remote = Literal::try_from(2)?;
    let encrypted = Literal::try_from(3)?;
    let notifications = Literal::try_from(4)?;
    let rules = Tdd::clause(&vtree, [local, remote])?
        & Tdd::clause(&vtree, [remote.negated(), encrypted])?;
    println!("Valid configurations: {}", rules.model_count()?);

    let selected = rules.clone() & literal(&vtree, remote)?;
    println!("With remote backups: {}", selected.model_count()?);
    let mut counter = rules.counter()?;
    counter.observe([remote])?;
    println!("Observed remote backups: {}", counter.model_count()?);

    let residual = rules.clone().condition([remote])?;
    println!("After substituting remote = true: {}", residual.model_count()?);
    let remaining = [local.var, encrypted.var, notifications.var];
    println!("Distinct remaining choices: {}", residual.projected_model_count(&remaining)?);

    let destinations = [local.var, remote.var];
    println!("Valid destination choices: {}", rules.projected_model_count(&destinations)?);
    let destination_rule = rules.clone().exists_vars(&[encrypted.var, notifications.var])?;
    println!("Destination rule over the full vtree: {}", destination_rule.model_count()?);
    println!("Destination rule projected: {}", destination_rule.projected_model_count(&destinations)?);

    Ok(())
}
