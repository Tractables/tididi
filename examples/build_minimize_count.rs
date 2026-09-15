//! Count backup configurations, find a solution, and apply an observation.
//! Run with `cargo run --example build_minimize_count`.

use std::sync::Arc;

use tididi::{Engine, Tdd, Vtree};
use tididi::reduce::try_minimize;

/// Build the backup rules and query the configurations they permit.
fn main() -> Result<(), tididi::OperationError> {
    // Each variable is one on/off option. Notifications are independent of the rules.
    let tree = Arc::new(Vtree::balanced(4));
    let names = ["local backups", "remote backups", "encryption", "notifications"];
    let local = Tdd::literal(&tree, 1);
    let remote = Tdd::literal(&tree, 2);
    let encrypted = Tdd::literal(&tree, 3);

    // Require at least one destination; remote backups require encryption.
    let destination = local | remote.clone();
    let encryption_rule = !remote.clone() | encrypted;
    let mut configurations = destination & encryption_rule;

    // There are four choices for the first three options and two for notifications.
    let count = configurations.model_count();
    assert_eq!(count, 8u32.into());
    println!("Valid configurations: {count}");

    // Keep the original and require remote backups in a second diagram.
    let with_remote = configurations.clone() & remote;
    let remote_count = with_remote.model_count();
    assert_eq!(remote_count, 4u32.into());
    println!("Configurations with remote backups: {remote_count}");

    // Reuse one workspace for the following queries and minimization.
    let engine = Engine::new();
    let witness = engine.satisfying_assignment(&configurations)?
        .expect("the backup rules have a solution");
    println!("One valid configuration:");
    for literal in &witness {
        println!("  {}: {}", names[literal.var.idx()], literal.positive);
    }
    assert!(engine.implies(&engine.cube(&tree, &witness)?, &configurations)?);

    // Minimization removes redundancy; all the queries above work before this step.
    try_minimize(&engine, &mut configurations)?;
    assert_eq!(engine.model_count(&configurations)?, count);
    println!("Minimized representation: {} pairs", configurations.pair_count());
    Ok(())
}
