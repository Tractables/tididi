//! Build a CNF one clause at a time, reduce it, count its models, condition a
//! variable, and render the result.
//!
//! The end-to-end path through the crate: pick a vtree, conjoin the clauses
//! into an accumulator, minimize once at the end, read the count off the
//! canonical diagram, then take a cofactor and write it out as DOT. Run it with
//! `cargo run --example build_minimize_count`.

use std::sync::Arc;

use num_bigint::BigUint;
use tididi::Tdd;
use tididi::apply::apply_and_clause;
use tididi::reduce::minimize;
use tididi::apply::condition_var;
use tididi::io::tdd_to_dot;
use tididi::vtree::{VarId, Vtree};

fn main() {
    // (x1 v x2) ^ (!x2 v x3) ^ (x1 v !x3) over four variables; x4 is free.
    let cnf = [[1, 2], [-2, 3], [1, -3]];

    let vtree = Arc::new(Vtree::balanced(4));
    let mut f = Tdd::one(&vtree);
    for clause in &cnf {
        let lits: Vec<_> = clause.iter().map(|&n| n.into()).collect();
        f = apply_and_clause(f, &lits);
    }
    minimize(&mut f);

    let count = f.model_count();
    println!("size: {} pairs over {} nodes", f.pair_count(), f.node_count());
    println!("models: {count}");
    assert_eq!(count, BigUint::from(6u32));

    // Conditioning on x1 = true removes x1 from the diagram, so its models are
    // counted over the remaining variables — and x1 itself becomes free, which
    // is why the cofactor's count still carries a factor of two.
    let cofactor = condition_var(&f, VarId(0), true);
    println!("models with x1 = true: {}", cofactor.model_count() / 2u32);

    // The DOT rendering is what to paste into Graphviz to look at the diagram.
    let dot = tdd_to_dot(&cofactor).expect("an explicit diagram renders");
    println!("--- cofactor as DOT ---");
    print!("{dot}");
}
