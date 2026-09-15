//! Count backup configurations, find a solution, and apply an observation.
//! Run with `cargo run --example build_minimize_count`.

use std::sync::Arc;

use tididi::{Engine, Literal, Vtree};
use tididi::reduce::try_minimize;
use tididi::vtree::VarId;

/// Build the backup rules and query the configurations they permit.
fn main() -> Result<(), tididi::OperationError> {
    let engine = Engine::new();
    // Each variable is one on/off option. Notifications are independent of the rules.
    let tree = Arc::new(Vtree::balanced(4));
    let names = ["local backups", "remote backups", "encryption", "notifications"];
    let local = Literal::pos(VarId(0));
    let remote = Literal::pos(VarId(1));
    let encrypted = Literal::pos(VarId(2));

    // Require at least one destination; remote backups require encryption.
    let mut configurations = engine.one(&tree);
    for clause in [[local, remote], [remote.negated(), encrypted]] {
        configurations = engine.and_clause(configurations, &clause)?;
    }

    // There are four choices for the first three options and two for notifications.
    let count = engine.model_count(&configurations)?;
    assert_eq!(count, 8u32.into());
    println!("Valid configurations: {count}");

    let witness = engine.satisfying_assignment(&configurations)?
        .expect("the backup rules have a solution");
    println!("One valid configuration:");
    for literal in &witness {
        println!("  {}: {}", names[literal.var.idx()], literal.positive);
    }
    assert!(engine.implies(&engine.cube(&tree, &witness)?, &configurations)?);

    // Keep the original; conditioning consumes its copy and substitutes remote = true.
    let with_remote = engine.condition(configurations.clone(), [remote])?;
    // The tree still contains remote, now free, so remove its two choices from the count.
    let remote_count = engine.model_count(&with_remote)? / 2u32;
    assert_eq!(remote_count, 4u32.into());
    println!("Configurations with remote backups: {remote_count}");

    // Minimization removes redundancy; all the queries above work before this step.
    try_minimize(&engine, &mut configurations)?;
    assert_eq!(engine.model_count(&configurations)?, count);
    println!("Minimized representation: {} pairs", configurations.pair_count());
    Ok(())
}
