//! The runnable version of the example in the top-level README. Keep this
//! file and the README snippet in sync: this test is what guarantees the
//! documented sequence compiles and produces the stated counts. The only
//! difference is the output path, which here is a temporary file.

use num_bigint::BigUint;
use std::sync::Arc;
use tididi::Tdd;
use tididi::io::save_tdd;
use tididi::reduce::minimize;
use tididi::apply::apply_and_clause;
use tididi::vtree::{VarId, Vtree};

#[test]
fn readme_example() {
    // A vtree over x1..x4 that groups {x1, x2} and {x3, x4}.
    let left = Vtree::balanced_over(&[VarId(0), VarId(1)]);
    let right = Vtree::balanced_over(&[VarId(2), VarId(3)]);
    let vtree = Arc::new(Vtree::join(&left, &right).unwrap());

    // (x1 ∨ ¬x2) ∧ (x2 ∨ x3) ∧ (¬x3 ∨ x4), one clause at a time; integers are
    // DIMACS literals (1 → x1, -2 → ¬x2).
    let mut f = Tdd::one(&vtree);
    for clause in [[1, -2], [2, 3], [-3, 4]] {
        let lits: Vec<_> = clause.iter().map(|&n| n.into()).collect();
        f = apply_and_clause(&mut f, &lits);
    }
    minimize(&mut f); // canonical form for this vtree
    assert_eq!(f.model_count(), BigUint::from(5u32));

    // Conjoin with x1 ⊕ x4, built from clauses with the operators.
    let g = Tdd::clause(&vtree, [1, 4]) & Tdd::clause(&vtree, [-1, -4]);
    let h = f & g; // apply results are already canonical
    assert_eq!(h.model_count(), BigUint::from(2u32));

    let path = std::env::temp_dir().join("tididi_readme_example.tdd");
    save_tdd(&h, path.to_str().unwrap()).unwrap();
    assert!(
        std::fs::read_to_string(&path)
            .unwrap()
            .starts_with("c TiDiDi TDD circuit")
    );
    let _ = std::fs::remove_file(&path);
}
