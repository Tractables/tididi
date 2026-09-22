// scenario: docs/scenarios.md#care-sets

fn main() -> Result<(), tididi::OperationError> {
    use std::sync::Arc;
    use tididi::{literal, Vtree};

    let vtree = Arc::new(Vtree::balanced(2));
    let local = literal(&vtree, 1)?;
    let remote = literal(&vtree, 2)?;
    let rule = local.clone() | remote.clone();
    let care = !remote | local;
    println!("Original backup choices: {}", rule.model_count()?);

    let mut simplified = rule.clone().restrict_to_care(care.clone())?.into_tdd();
    simplified.minimize()?;
    println!("Pairs before: {}; after: {}", rule.pair_count(), simplified.pair_count());

    let valid_before = rule.clone() & care.clone();
    let valid_after = simplified.clone() & care;
    println!("Equivalent under the assumption: {}", valid_before.equivalent(&valid_after)?);
    println!("Choices under the assumption: {}", valid_after.model_count()?);

    println!("Equivalent everywhere: {}", rule.equivalent(&simplified)?);
    let remote_only = [-1, 2];
    println!("Remote only, original: {}", rule.condition(remote_only)?.is_sat()?);
    println!("Remote only, simplified: {}", simplified.condition(remote_only)?.is_sat()?);
    Ok(())
}
