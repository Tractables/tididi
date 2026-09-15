//! Save two diagrams, restore their shared domain, and combine them.
//! Run with `cargo run --example save_reload`.

use std::sync::Arc;

use tididi::{and, Tdd, Vtree};
use tididi::io::{read_tdd, write_tdd};

/// Round-trip two rules and check their conjunction over the restored tree.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let tree = Arc::new(Vtree::balanced(3));
    let destination = Tdd::clause(&tree, [1, 2])?;
    let encryption_rule = Tdd::clause(&tree, [-2, 3])?;

    let tree_text = tree.to_text();
    let mut destination_bytes = Vec::new();
    let mut encryption_bytes = Vec::new();
    write_tdd(&mut destination_bytes, &destination)?;
    write_tdd(&mut encryption_bytes, &encryption_rule)?;
    drop((destination, encryption_rule, tree));

    let restored_tree = Arc::new(Vtree::from_text(&tree_text)?);
    let destination = read_tdd(&mut destination_bytes.as_slice(), &restored_tree)?;
    let encryption_rule = read_tdd(&mut encryption_bytes.as_slice(), &restored_tree)?;
    assert!(Arc::ptr_eq(destination.vtree(), encryption_rule.vtree()));

    let configurations = and(destination, encryption_rule)?;
    assert_eq!(configurations.model_count()?, 4u32.into());
    let expected = and(
        Tdd::clause(&restored_tree, [1, 2])?,
        Tdd::clause(&restored_tree, [-2, 3])?,
    )?;
    assert!(configurations.equivalent(&expected)?);
    println!("Restored rules allow {} configurations", configurations.model_count()?);
    Ok(())
}
