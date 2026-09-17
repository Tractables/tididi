//! Compare two variable groupings for the same pair of equality constraints.
//! Run with `cargo run --example vtree_grouping`.

use std::sync::Arc;

use tididi::{and, OperationError, Tdd, Vtree};
use tididi::vtree::VarId;

/// Build two independent equalities and minimize under the supplied vtree.
fn equal_pairs(vtree: &Arc<Vtree>) -> Result<Tdd, OperationError> {
    let first_equal = and(Tdd::clause(vtree, [-1, 3])?, Tdd::clause(vtree, [1, -3])?)?;
    let second_equal = and(Tdd::clause(vtree, [-2, 4])?, Tdd::clause(vtree, [2, -4])?)?;
    let mut f = and(first_equal, second_equal)?;
    f.minimize()?;
    Ok(f)
}

/// Check equal model counts and compare storage after minimization.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let grouped_vtree = Arc::new(Vtree::balanced_over(&[
        VarId(0), VarId(2), VarId(1), VarId(3),
    ])?);
    let split_vtree = Arc::new(Vtree::balanced(4));
    let grouped = equal_pairs(&grouped_vtree)?;
    let split = equal_pairs(&split_vtree)?;
    assert_eq!(grouped.model_count()?, 4u32.into());
    assert_eq!(split.model_count()?, 4u32.into());
    assert_eq!(grouped.pair_count(), 5);
    assert_eq!(split.pair_count(), 12);
    println!("Grouped equalities: {} pairs; split equalities: {} pairs",
        grouped.pair_count(), split.pair_count());
    Ok(())
}
