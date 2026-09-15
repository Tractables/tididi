//! Add clauses to a function, minimize it, count models, and take a cofactor.
//! Run with `cargo run --example build_minimize_count`.

use std::sync::Arc;

use tididi::{Engine, Literal, Vtree};
use tididi::io::tdd_to_dot;
use tididi::reduce::try_minimize;
use tididi::vtree::VarId;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // (x1 ∨ x2) ∧ (¬x2 ∨ x3) ∧ (x1 ∨ ¬x3), with x4 free.
    let clauses = [[1, 2], [-2, 3], [1, -3]];
    let engine = Engine::new();
    let tree = Arc::new(Vtree::balanced(4));
    let mut f = engine.one(&tree);
    for clause in clauses {
        let literals = clause.into_iter().map(Literal::try_from).collect::<Result<Vec<_>, _>>()?;
        f = engine.and_clause(f, &literals)?;
    }

    // Counting accepts the current representation; minimization removes redundancy.
    let count = engine.model_count(&f)?;
    try_minimize(&engine, &mut f)?;
    assert_eq!(engine.model_count(&f)?, count);
    assert_eq!(count, 6u32.into());
    println!("size: {} pairs over {} nodes", f.pair_count(), f.node_count());
    println!("models: {count}");

    // Keep the original for later queries; the transformation consumes its copy.
    let cofactor = engine.condition_var(f.clone(), VarId(0), true)?;
    // x1 remains free in the cofactor's vtree, so divide out its two choices.
    let observed_count = engine.model_count(&cofactor)? / 2u32;
    assert_eq!(observed_count, 6u32.into());
    println!("models with x1 = true: {observed_count}");

    // Graphviz can render this text; the diagram must still be structural.
    let dot = tdd_to_dot(&cofactor)?;
    println!("cofactor as Graphviz text:");
    print!("{dot}");
    Ok(())
}
