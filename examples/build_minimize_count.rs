//! Count backup configurations, find a solution, and apply an observation.
//! Run with `cargo run --example build_minimize_count`.

/// Build the backup rules and query the configurations they permit.
fn main() -> Result<(), tididi::OperationError> {
    use std::sync::Arc;

    use tididi::{literal, Literal, Tdd, Vtree};

    let vtree = Arc::new(Vtree::balanced(4));
    let names = ["local backups", "remote backups", "encryption", "notifications"];
    let local_choice = Literal::try_from(1)?;
    let remote_choice = Literal::try_from(2)?;
    let encrypted_choice = Literal::try_from(3)?;
    let notifications_choice = Literal::try_from(4)?;
    let local = literal(&vtree, local_choice)?;
    let remote = literal(&vtree, remote_choice)?;
    let encrypted = literal(&vtree, encrypted_choice)?;
    let notifications = literal(&vtree, notifications_choice)?;

    // Require at least one destination; remote backups require encryption.
    let destination = local | remote.clone();
    let encryption_rule = !remote.clone() | encrypted.clone();
    let mut configurations = destination & encryption_rule;

    // There are four choices for the first three options and two for notifications.
    let count = configurations.model_count()?;
    println!("Valid configurations: {count}");
    assert_eq!(count, 8u32.into());

    // Notifications are optional, so requiring them leaves half the configurations.
    let with_notifications = configurations.clone() & notifications;
    assert_eq!(with_notifications.model_count()?, 4u32.into());

    // Keep the original and require remote backups in a second diagram.
    let with_remote = configurations.clone() & remote;
    let remote_count = with_remote.model_count()?;
    println!("Configurations with remote backups: {remote_count}");
    assert_eq!(remote_count, 4u32.into());

    // Remote backups force encryption; disabling it leaves no valid configuration.
    let forced = with_remote.implied_literals()?;
    for literal in &forced {
        println!("Required choice: {} = {}", names[literal.var.idx()], literal.sign);
    }
    assert!(forced.contains(&3.try_into()?));
    let remote_without_encryption = with_remote & !encrypted.clone();
    println!("Satisfiable: {}", remote_without_encryption.is_sat()?);
    assert!(!remote_without_encryption.is_sat()?);

    let witness = configurations.satisfying_assignment()?
        .expect("the backup rules have a solution");
    println!("One valid configuration:");
    for literal in &witness {
        println!("  {}: {}", names[literal.var.idx()], literal.sign);
    }
    let selected = configurations.clone() & Tdd::cube(&vtree, &witness)?;
    println!("Selected valid configurations: {}", selected.model_count()?);

    // Reuse the circuit while a user adds and changes choices.
    let mut counter = configurations.counter()?;
    counter.observe([remote_choice])?;
    println!("Matching configurations: {}", counter.model_count()?);
    counter.observe([notifications_choice.negated()])?;
    println!("Matching configurations: {}", counter.model_count()?);
    counter.observe([remote_choice.negated(), encrypted_choice.negated()])?;
    println!("Matching configurations: {}", counter.model_count()?);
    counter.clear_pins();
    println!("Matching configurations: {}", counter.model_count()?);
    // The counter borrows the rules; release it before they are minimized below.
    drop(counter);

    // Build the same rules with operations that return errors.
    use tididi::{and, or};

    let local = literal(&vtree, local_choice)?;
    let remote = literal(&vtree, remote_choice)?;
    let encrypted = literal(&vtree, encrypted_choice)?;
    let destination = or(local, remote.clone())?;
    let encryption_rule = or(remote.negate()?, encrypted)?;
    let checked = and(destination, encryption_rule)?;
    assert!(checked.equivalent(&configurations)?);

    // Reuse the attached context for a bounded batch.
    use tididi::OperationError;
    use tididi::limits::LimitConfig;
    let context = Arc::clone(vtree.context());
    let limit = LimitConfig::none().with_memory_budget_bytes(Some(0));
    let attempt = context.with_limits(limit, |operations| {
        operations.clause(&vtree, [local_choice, remote_choice])
    });
    match attempt {
        Ok(diagram) => println!("Destination choices: {}", diagram.model_count()?),
        Err(OperationError::OverBudget) => println!("Not enough budget to build the destination rule"),
        Err(error) => return Err(error),
    }
    // The completed batch leaves no limits installed on later operations.
    let destination = Tdd::clause(&vtree, [local_choice, remote_choice])?;
    println!("Destination choices: {}", destination.model_count()?);
    println!("Valid configurations: {}", configurations.model_count()?);

    // Minimization removes redundancy; the earlier queries need no explicit pass.
    configurations.minimize()?;
    println!("Valid configurations: {}", configurations.model_count()?);
    println!("Minimized representation: {} pairs", configurations.pair_count());
    // Temporarily use batch limits while retaining the counter for later queries.
    let mut counter = configurations.counter()?;
    let query_limit = LimitConfig::none().with_memory_budget_bytes(Some(1_000_000));
    let bounded_count = context.with_limits(query_limit, |operations| {
        counter.bind(operations).model_count()
    })?;
    println!("Count with a budget: {bounded_count}");

    context.clear_scratch();
    Ok(())
}
