//! A custom statistic over the stored encoding: the node with the most pairs
//! and the vtree level it sits at. Run with `cargo run --example statistic`.

use std::sync::Arc;

use tididi::Tdd;
use tididi::vtree::{Vtree, VtreeIdx};

/// `(vtree node, pair count)` of the widest node; `(root, 0)` for a diagram
/// with no stored nodes.
fn widest_node(t: &Tdd) -> (VtreeIdx, usize) {
    let mut best = (t.vtree().root(), 0usize);
    for v in t.vtree().bottomup() {
        let lvl = t.level(v);
        // Leaf levels store nothing and marginal levels have dropped their
        // pairs; both yield no nodes here.
        if lvl.is_marginal() {
            continue;
        }
        for (_i, pairs) in lvl.internal_inputs_iter() {
            let n = pairs.len(); // PairsIter is ExactSizeIterator
            if n > best.1 {
                best = (v, n);
            }
        }
    }
    best
}

fn main() {
    let vtree = Arc::new(Vtree::balanced(4));
    // x1 ⊕ x2 needs two pairs at the level over {x1, x2}: (x1, ¬x2) and (¬x1, x2).
    let xor = Tdd::clause(&vtree, [1, 2]) & Tdd::clause(&vtree, [-1, -2]);
    let (level, pairs) = widest_node(&xor);
    let (left, _) = vtree.children(vtree.root());
    assert_eq!((level, pairs), (left, 2));

    // A unit clause is a cube: one pair per node everywhere.
    let unit = Tdd::clause(&vtree, [1]);
    assert_eq!(widest_node(&unit).1, 1);

    // The diagram's own size metric is the sum of every node's pair count.
    assert!(widest_node(&xor).1 <= xor.pair_count());
    println!("statistic: widest node has {pairs} pairs at vtree node {}", level.idx());
}
