//! Save two diagrams, restore their shared domain, and combine them.
//! Run with `cargo run --example save_reload`.

use std::sync::Arc;

use tididi::{and, Tdd, Vtree};
use tididi::io::{read_tdd, write_tdd};

/// Round-trip two rules and check their conjunction over the restored vtree.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let vtree = Arc::new(Vtree::balanced(3));
    let destination = Tdd::clause(&vtree, [1, 2])?;
    let encryption_rule = Tdd::clause(&vtree, [-2, 3])?;

    let vtree_text = vtree.to_text();
    let mut destination_bytes = Vec::new();
    let mut encryption_bytes = Vec::new();
    write_tdd(&mut destination_bytes, &destination)?;
    write_tdd(&mut encryption_bytes, &encryption_rule)?;
    drop((destination, encryption_rule, vtree));

    let restored_vtree = Arc::new(Vtree::from_text(&vtree_text)?);
    let destination = read_tdd(&mut destination_bytes.as_slice(), &restored_vtree)?;
    let encryption_rule = read_tdd(&mut encryption_bytes.as_slice(), &restored_vtree)?;
    assert!(Arc::ptr_eq(destination.vtree(), encryption_rule.vtree()));

    let configurations = and(destination, encryption_rule)?;
    assert_eq!(configurations.model_count()?, 4u32.into());
    println!("Restored rules allow {} configurations", configurations.model_count()?);
    let expected = and(
        Tdd::clause(&restored_vtree, [1, 2])?,
        Tdd::clause(&restored_vtree, [-2, 3])?,
    )?;
    assert!(configurations.equivalent(&expected)?);
    Ok(())
}
