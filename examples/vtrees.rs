// scenario: docs/scenarios.md#vtrees

//! Compare two variable groupings for the same pair of equality constraints.
//! Run with `cargo run --example vtrees`.

fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::sync::Arc;

    use tididi::{literal, xor, OperationError, Tdd, Vtree};
    use tididi::vtree::VarId;

    let grouped_vtree = Arc::new(Vtree::balanced_over(&[
        VarId(1), VarId(3), VarId(2), VarId(4),
    ])?);
    let split_vtree = Arc::new(Vtree::balanced(4));

    fn equal_pairs(vtree: &Arc<Vtree>) -> Result<Tdd, OperationError> {
        let x1 = literal(vtree, 1)?;
        let x2 = literal(vtree, 2)?;
        let x3 = literal(vtree, 3)?;
        let x4 = literal(vtree, 4)?;
        let first_equal = !xor(x1, x3)?;
        let second_equal = !xor(x2, x4)?;
        let mut f = first_equal & second_equal;
        f.minimize()?;
        Ok(f)
    }

    let grouped = equal_pairs(&grouped_vtree)?;
    let split = equal_pairs(&split_vtree)?;
    println!("Models: grouped = {}, split = {}",
        grouped.model_count()?, split.model_count()?);
    println!("Grouped equalities: {} pairs; split equalities: {} pairs",
        grouped.pair_count(), split.pair_count());
    Ok(())
}
