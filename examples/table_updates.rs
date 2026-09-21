//! Build and maintain a table of permitted access combinations.
use std::sync::Arc;

use tididi::vtree::VarId;
use tididi::{literal, Tdd, Vtree};

fn main() -> Result<(), tididi::OperationError> {
    let vtree = Arc::new(Vtree::balanced(3));
    let vars = [VarId(1), VarId(2), VarId(3)];
    // Low to high bits: read, write, share.
    let rows = [0b001, 0b011, 0b101, 0b011];
    let mut permissions = Tdd::from_models(&vtree, &vars, &rows)?;
    println!("Distinct permission sets: {}", permissions.model_count()?);

    {
        let mut batch = permissions.maintain()?;
        batch.insert_model([1, 2, 3])?;
        batch.remove_model([1, 2, -3])?;
    }
    permissions.minimize()?;
    println!("After updates: {}", permissions.model_count()?);

    let share = literal(&vtree, 3)?;
    let sharing = permissions & share;
    println!("Permission sets allowing sharing: {}", sharing.model_count()?);
    Ok(())
}
