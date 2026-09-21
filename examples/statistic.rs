// scenario: docs/scenarios.md#statistics

//! A custom statistic over the stored encoding: the node with the most pairs
//! and the vtree level it sits at. Run with `cargo run --example statistic`.

fn main() -> Result<(), tididi::OperationError> {
    use std::sync::Arc;

    use tididi::{literal, Tdd};
    use tididi::vtree::{Vtree, VtreeIdx};

    let vtree = Arc::new(Vtree::balanced(4));
    let x1 = literal(&vtree, 1)?;
    let x2 = literal(&vtree, 2)?;
    let xor = (x1.clone() & !x2.clone()) | (!x1.clone() & x2);

    fn widest_node(circuit: &Tdd) -> (VtreeIdx, usize) {
        let mut best = (circuit.vtree().root(), 0usize);
        for v in circuit.vtree().bottomup() {
            for (_i, pairs) in circuit.level(v).internal_inputs_iter() {
                let n = pairs.len();
                if n > best.1 {
                    best = (v, n);
                }
            }
        }
        best
    }

    let (level, pairs) = widest_node(&xor);
    println!("Widest XOR node: {pairs} pairs at vtree node {}", level.idx());

    println!("Widest literal node: {} pair", widest_node(&x1).1);

    println!("Total XOR pairs: {}", xor.pair_count());
    Ok(())
}
