//! Save two diagrams, restore their shared domain, and combine them.
//! Run with `cargo run --example save_reload`.

fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::sync::Arc;

    use tididi::{literal, Vtree};
    use tididi::io::{read_tdd, write_tdd};

    let vtree = Arc::new(Vtree::balanced(3));
    let local = literal(&vtree, 1)?;
    let remote = literal(&vtree, 2)?;
    let encrypted = literal(&vtree, 3)?;
    let destination = local | remote.clone();
    let encryption_rule = !remote | encrypted;

    let vtree_text = vtree.to_text();
    let mut destination_bytes = Vec::new();
    let mut encryption_bytes = Vec::new();
    write_tdd(&mut destination_bytes, &destination)?;
    write_tdd(&mut encryption_bytes, &encryption_rule)?;
    drop((destination, encryption_rule, vtree));

    let restored_vtree = Arc::new(Vtree::from_text(&vtree_text)?);
    let destination = read_tdd(&mut destination_bytes.as_slice(), &restored_vtree)?;
    let encryption_rule = read_tdd(&mut encryption_bytes.as_slice(), &restored_vtree)?;

    let configurations = destination & encryption_rule;
    println!("Restored rules allow {} configurations", configurations.model_count()?);

    let local = literal(&restored_vtree, 1)?;
    let remote = literal(&restored_vtree, 2)?;
    let encrypted = literal(&restored_vtree, 3)?;
    let expected = (local | remote.clone()) & (!remote | encrypted);
    println!("Equivalent to the original rules: {}", configurations.equivalent(&expected)?);
    Ok(())
}
