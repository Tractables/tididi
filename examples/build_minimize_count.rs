//! Count backup configurations, find a solution, and apply an observation.
//! Run with `cargo run --example build_minimize_count`.

use std::sync::Arc;

use tididi::{and, or, Tdd, Vtree};

/// Build the backup rules and query the configurations they permit.
fn main() -> Result<(), tididi::OperationError> {
    // Each variable is one on/off option. Notifications are independent of the rules.
    let tree = Arc::new(Vtree::balanced(4));
    let names = ["local backups", "remote backups", "encryption", "notifications"];
    let local = Tdd::literal(&tree, 1);
    let remote = Tdd::literal(&tree, 2);
    let encrypted = Tdd::literal(&tree, 3);

    // Require at least one destination; remote backups require encryption.
    let destination = or(local, remote.clone())?;
    let encryption_rule = or(remote.clone().negate()?, encrypted)?;
    let mut configurations = and(destination, encryption_rule)?;

    // There are four choices for the first three options and two for notifications.
    let count = configurations.model_count();
    assert_eq!(count, 8u32.into());
    println!("Valid configurations: {count}");

    // Keep the original and require remote backups in a second diagram.
    let with_remote = and(configurations.clone(), remote)?;
    let remote_count = with_remote.model_count();
    assert_eq!(remote_count, 4u32.into());
    println!("Configurations with remote backups: {remote_count}");

    let witness = configurations.satisfying_assignment()
        .expect("the backup rules have a solution");
    println!("One valid configuration:");
    for literal in &witness {
        println!("  {}: {}", names[literal.var.idx()], literal.positive);
    }
    let selected = and(configurations.clone(), Tdd::cube(&tree, &witness))?;
    assert_eq!(selected.model_count(), 1u32.into());

    // Reuse the attached context for a bounded batch.
    use tididi::OperationError;
    use tididi::limits::LimitConfig;
    let context = Arc::clone(tree.context());
    let limit = LimitConfig::none().with_memory_budget_bytes(Some(0));
    let attempt = context.with_limits(limit, |operations| {
        operations.clause(&tree, [1, 2])
    });
    assert!(matches!(attempt, Err(OperationError::OverBudget)));
    match attempt {
        Ok(diagram) => println!("Destination choices: {}", diagram.model_count()),
        Err(OperationError::OverBudget) => println!("Not enough budget to build the destination rule"),
        Err(error) => return Err(error),
    }
    // The completed batch leaves no limits installed on later operations.
    let destination = Tdd::try_clause(&tree, [1, 2])?;
    assert_eq!(destination.model_count(), 12u32.into());
    assert_eq!(configurations.model_count(), count);

    // Minimization removes redundancy; the earlier queries need no explicit pass.
    configurations.minimize()?;
    assert_eq!(configurations.model_count(), count);
    println!("Minimized representation: {} pairs", configurations.pair_count());
    context.clear_scratch();
    Ok(())
}
