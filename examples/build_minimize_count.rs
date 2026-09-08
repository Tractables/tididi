//! Build a CNF one clause at a time, reduce it, and count its models.
//!
//! The shortest end-to-end path through the crate: pick a vtree, conjoin the
//! clauses into an accumulator, minimize once at the end, then read the count
//! off the canonical diagram. Run it with
//! `cargo run --example build_minimize_count`.

use std::sync::Arc;

use num_bigint::BigUint;
use tididi::Tdd;
use tididi::apply::apply_and_clause;
use tididi::reduce::minimize;
use tididi::query::model_count;
use tididi::vtree::Vtree;

fn main() {
    // (x1 v x2) ^ (!x2 v x3) ^ (x1 v !x3) over four variables; x4 is free.
    let cnf = [[1, 2], [-2, 3], [1, -3]];

    let vtree = Arc::new(Vtree::balanced(4));
    let mut f = Tdd::one(&vtree);
    for clause in &cnf {
        let lits: Vec<_> = clause.iter().map(|&n| n.into()).collect();
        f = apply_and_clause(&mut f, &lits);
    }
    minimize(&mut f);

    let count = model_count(&f);
    println!("size: {} pairs over {} nodes", f.size(), f.node_count());
    println!("models: {count}");
    assert_eq!(count, BigUint::from(6u32));
}
