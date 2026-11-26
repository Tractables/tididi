//! The leaf form a negation emits in.
//!
//! Determinism allows a level to name a leaf child either as `One` or as
//! `Pos`/`Neg`, never both, so a fill node and the complement at the root are
//! emitted in whichever form the level already uses. These tests pin that:
//! mixing the two would be an invalid diagram, and it is also what blocks
//! leaf-twin contraction, which is all-or-nothing per level.

use super::*;
use crate::diagram::Tdd;
use crate::test_helpers::check::assert_no_mixed_leaf_side;

#[test]
fn the_fill_joins_a_level_in_the_form_that_level_already_uses() {
    let eng = &crate::Engine::new();
    let vtree = crate::vtree::Home::new(crate::Vtree::linear(4));
    // A cube over variable 1 leaves every level below it naming its leaf as
    // `One`; the levels above name theirs as `Pos`.
    let f = Tdd::cube(&vtree, [1]).unwrap();
    let mut full = f.clone();
    expand_full_unmarked(eng, &mut full).unwrap();
    assert_no_mixed_leaf_side(&full, "expand_full");
    assert_no_mixed_leaf_side(&eng.negate(f).unwrap(), "negate");
}

#[test]
fn no_negation_mixes_the_two_leaf_forms_on_random_formulas() {
    use crate::test_helpers::{CnfShape, Lcg, compile_clauses_on, rand_cnf};
    let eng = &crate::Engine::new();
    let mut rng = Lcg::new(0x2545_f491);
    for num_vars in [3u32, 5, 8] {
        for (_name, vtree) in crate::test_helpers::vtree_shapes(num_vars) {
            for _ in 0..12 {
                let clauses = rand_cnf(&mut rng, num_vars, CnfShape { clauses: 6, width: 3 });
                let f = compile_clauses_on(eng, &vtree, &clauses);
                if f.is_zero() {
                    continue;
                }
                let mut full = f.clone();
                expand_full_unmarked(eng, &mut full).unwrap();
                assert_no_mixed_leaf_side(&full, "expand_full");
                // Before `reduce`, so the complement at the root is checked in
                // the form it was emitted in rather than after contraction.
                let raw = crate::apply::negate::negate_on(eng, f.clone()).unwrap();
                assert_no_mixed_leaf_side(&raw, "negate_on");
                assert_no_mixed_leaf_side(&eng.negate(f).unwrap(), "negate");
            }
        }
    }
}
