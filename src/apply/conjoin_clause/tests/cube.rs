use super::*;
use std::sync::Arc;

use crate::test_helpers::check::check_determinism;
use crate::test_helpers::{
    assert_canonical, assert_same_shape, assignment, compile_clauses, rand_cnf, sweep_shapes,
    test_cases, vtree_shapes, CnfShape, Lcg,
};
use crate::vtree::VarId;

/// The disjunction the specialized walk is checked against: build the cube as
/// a diagram and take the general `or`.
fn or_with_cube_diagram(vtree: &Arc<Vtree>, f: &Tdd, cube: &[i32]) -> Tdd {
    let cube = Tdd::cube(vtree, cube).expect("the cube is over the vtree");
    let mut out = crate::or(f.clone(), cube).expect("the disjunction fits");
    out.minimize().expect("minimizing fits");
    out
}

/// `or_cube` with its result minimized, so it is comparable with the oracle's.
fn or_cube_minimized(f: &Tdd, cube: &[i32]) -> Tdd {
    let mut out = f.clone().or_cube(cube).expect("the disjunction fits");
    out.minimize().expect("minimizing fits");
    out
}

/// Structural determinism: the level is a partition. The checker is quadratic
/// in the level widths, so callers keep the variable count small.
fn assert_partition(f: &Tdd, what: &str) {
    check_determinism(f).unwrap_or_else(|e| panic!("{what}: a level is not a partition: {e}"));
}

/// The cases an exhaustive sweep over assignments can afford. The oracle is
/// exponential in the variable count and the sweep multiplies it by `2^n`.
fn small_cases() -> Vec<(u32, Vec<Vec<i32>>)> {
    test_cases().into_iter().filter(|(num_vars, _)| *num_vars <= 5).collect()
}

#[test]
fn disjoining_a_model_matches_the_general_disjunction() {
    for (num_vars, clauses) in small_cases() {
        for (shape, vtree) in sweep_shapes(num_vars) {
            let f = compile_clauses(&vtree, &clauses);
            for bits in 0..(1u64 << num_vars) {
                let cube = assignment(num_vars, bits);
                let got = or_cube_minimized(&f, &cube);
                let want = or_with_cube_diagram(&vtree, &f, &cube);
                assert_canonical(&got);
                assert_same_shape(&got, &want, &format!("{shape}: or_cube {cube:?}"));
            }
        }
    }
}

#[test]
fn disjoining_a_model_keeps_every_level_a_partition() {
    let mut rng = Lcg::new(0x5eed);
    for round in 0..40u32 {
        let num_vars = 4 + rng.below(2) as u32;
        let clauses = rand_cnf(&mut rng, num_vars, CnfShape { clauses: 6, width: 3 });
        for (shape, vtree) in sweep_shapes(num_vars) {
            let f = compile_clauses(&vtree, &clauses);
            let cube = assignment(num_vars, rng.below(1u64 << num_vars));
            let out = f.clone().or_cube(&cube).expect("the disjunction fits");
            assert_partition(&out, &format!("round {round} {shape}"));
        }
    }
}

#[test]
fn a_partial_cube_matches_the_general_disjunction() {
    for (num_vars, clauses) in small_cases() {
        if num_vars < 2 { continue; }
        for (shape, vtree) in sweep_shapes(num_vars) {
            let f = compile_clauses(&vtree, &clauses);
            // Cubes that leave at least one variable free.
            for mask in 1..((1u64 << num_vars) - 1) {
                for bits in (mask & 1..(1u64 << num_vars)).step_by(3) {
                    let cube: Vec<i32> = assignment(num_vars, bits)
                        .into_iter()
                        .filter(|lit| mask >> (lit.unsigned_abs() - 1) & 1 == 1)
                        .collect();
                    let got = or_cube_minimized(&f, &cube);
                    let want = or_with_cube_diagram(&vtree, &f, &cube);
                    assert_canonical(&got);
                    assert_same_shape(&got, &want, &format!("{shape}: or_cube {cube:?}"));
                }
            }
        }
    }
}

#[test]
fn removing_a_model_and_putting_it_back_restores_the_diagram() {
    for (num_vars, clauses) in small_cases() {
        for (shape, vtree) in sweep_shapes(num_vars) {
            let mut f = compile_clauses(&vtree, &clauses);
            f.minimize().expect("minimizing fits");
            for bits in 0..(1u64 << num_vars) {
                let cube = assignment(num_vars, bits);
                let clause: Vec<i32> = cube.iter().map(|lit| -lit).collect();
                let mut without = f.clone().and_clause(&clause).expect("the conjunction fits");
                without.minimize().expect("minimizing fits");
                let mut again = without.or_cube(&cube).expect("the disjunction fits");
                again.minimize().expect("minimizing fits");
                assert_canonical(&again);
                // Removing then re-adding restores the operand when it had the
                // model, and adds it when it did not; both are `f ∨ cube`.
                let want = or_with_cube_diagram(&vtree, &f, &cube);
                assert_same_shape(&again, &want, &format!("{shape}: round trip {cube:?}"));
            }
        }
    }
}

#[test]
fn inserting_a_row_matches_rebuilding_the_relation() {
    // Two two-bit attributes: the shape a dictionary-coded relation compiles to.
    let vars: Vec<VarId> = (1..=4).map(VarId).collect();
    let row = |a: u64, b: u64| (a >> 1) | ((a & 1) << 1) | ((b >> 1) << 2) | ((b & 1) << 3);
    let mut rng = Lcg::new(7);
    for (shape, vtree) in vtree_shapes(4) {
        for round in 0..24u32 {
            let n = 1 + rng.below(6) as usize;
            let mut rows: Vec<u64> = (0..n).map(|_| row(rng.below(4), rng.below(4))).collect();
            rows.sort_unstable();
            rows.dedup();
            let f = Tdd::from_models(&vtree, &vars, &rows).expect("the rows fit");
            let extra = row(rng.below(4), rng.below(4));
            let mut got = f.or_cube(assignment(4, extra)).expect("the disjunction fits");
            got.minimize().expect("minimizing fits");
            assert_canonical(&got);

            let mut extended = rows.clone();
            extended.push(extra);
            extended.sort_unstable();
            extended.dedup();
            let want = Tdd::from_models(&vtree, &vars, &extended).expect("the rows fit");
            assert_same_shape(&got, &want, &format!("{shape} round {round}: insert {extra}"));
        }
    }
}

#[test]
fn the_false_cube_and_the_true_cube_are_the_two_constants() {
    let vtree = Arc::new(Vtree::balanced(3));
    let f = Tdd::clause(&vtree, [1, 2]).expect("the clause is over the vtree");

    // Both polarities of a variable: the cube is false and the operand stands.
    let unchanged = f.clone().or_cube([1, -1, 2, 3]).expect("the disjunction fits");
    assert_same_shape(&unchanged, &f, "a false cube");

    // The empty cube is true.
    let everything = f.clone().or_cube::<i32>([]).expect("the disjunction fits");
    assert_eq!(everything.model_count().expect("counting fits"), 8u32.into());
    assert_canonical(&everything);

    // A repeated literal is ignored.
    let mut repeated = f.clone().or_cube([1, 1, -2, -3]).expect("the disjunction fits");
    repeated.minimize().expect("minimizing fits");
    assert_same_shape(&repeated, &or_with_cube_diagram(&vtree, &f, &[1, -2, -3]), "a repeated literal");
}

#[test]
fn disjoining_into_the_false_diagram_builds_the_cube() {
    let vtree = Arc::new(Vtree::balanced(4));
    for cube in [vec![1, -2, 3, -4], vec![1, -3]] {
        let zero = Tdd::zero(&vtree);
        let got = zero.or_cube(&cube).expect("the disjunction fits");
        assert_canonical(&got);
        assert_same_shape(&got, &Tdd::cube(&vtree, &cube).expect("the cube is over the vtree"),
            "false ∨ cube");
    }
}

#[test]
fn an_absent_variable_and_a_zero_literal_are_refused() {
    let vtree = Arc::new(Vtree::balanced(3));
    let f = Tdd::one(&vtree);
    assert_eq!(f.clone().or_cube([1, 9]).err(), Some(OperationError::VariableNotInVtree(VarId(9))));
    assert_eq!(f.or_cube([1, 0]).err(), Some(OperationError::InvalidLiteral(0)));
}

#[test]
fn the_typed_and_integer_cubes_agree() {
    let vtree = Arc::new(Vtree::balanced(5));
    let f = compile_clauses(&vtree, &[vec![1, 2], vec![-3, 4], vec![5]]);
    let integers = [1, -2, 3, -4, 5];
    let typed: Vec<Literal> = integers.iter().map(|&n| Literal::try_from(n).expect("nonzero")).collect();
    let mut from_integers = f.clone().or_cube(integers).expect("the disjunction fits");
    let mut from_typed = f.or_cube(typed.as_slice()).expect("the disjunction fits");
    from_integers.minimize().expect("minimizing fits");
    from_typed.minimize().expect("minimizing fits");
    assert_same_shape(&from_integers, &from_typed, "typed and integer cubes");
}
