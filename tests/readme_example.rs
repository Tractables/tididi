//! The runnable version of the usage example in the top-level README.
//! Keep this file and the README snippet in sync — this test is what guarantees
//! the documented sequence compiles and produces the right counts.

use std::sync::Arc;
use num_bigint::BigUint;
use tididi::tdd::Tdd;
use tididi::tdd::transform::unary::project::project_var;
use tididi::vtree::{VarId, Vtree};

#[test]
fn readme_example() {
    let vtree = Arc::new(Vtree::balanced(3)); // variables x1, x2, x3

    // Arbitrary Boolean combinations — conjunction, disjunction:
    let f = (Tdd::clause(&vtree, [1]) & Tdd::clause(&vtree, [2])) | Tdd::clause(&vtree, [3]);
    assert_eq!(f.model_count(), BigUint::from(5u32)); // (x1 ∧ x2) ∨ x3

    // Exact canonical negation:
    let g = !f;
    assert_eq!(g.model_count(), BigUint::from(3u32));

    // Forgetting (existential projection): ∃x1. ¬f
    let h = project_var(&g, VarId(0)); // VarId is 0-based: x1
    assert_eq!(h.model_count(), BigUint::from(4u32)); // x1 now free: 2 × 2 don't-care assignments
}
