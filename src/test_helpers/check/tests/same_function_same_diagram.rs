//! Canonicity: one Boolean function, one diagram.
//!
//! Two CNF encodings of the same function, compiled against the same vtree,
//! must produce structurally identical diagrams — clause order, subsumed
//! clauses, resolvents, duplicated clauses and transitive closures are all
//! invisible to the result. Each case is asserted twice: once structurally,
//! by comparing the two diagrams up to node numbering, and once against every
//! invariant a finished diagram carries — canonicity among them, which hashes
//! each node to a semiring signature and demands that no two nodes on one
//! level collide.
//!
//! Operands come from [`compile_clauses`], which conjoins the clauses one at a
//! time against a fixed vtree. A driver that preprocesses the formula first —
//! splitting it into components, grafting them together, choosing a vtree per
//! component — reaches these diagrams by other routes; that variety belongs to
//! the driver's own tests, not here.

use std::sync::Arc;

use crate::query::model_count;
use crate::test_helpers::{
    assert_canonical, assert_same_shape, brute_force_count, compile_clauses, queens_clauses,
    test_cases, vtree_shapes,
};
use crate::vtree::Vtree;

/// Reordering the clauses cannot reach a different diagram.
#[test]
fn clause_order_does_not_reach_a_different_diagram() {
    for (num_vars, clauses) in test_cases() {
        if clauses.len() < 2 {
            continue;
        }
        let mut reversed = clauses.clone();
        reversed.reverse();
        let mut interleaved: Vec<Vec<i32>> =
            clauses.iter().step_by(2).cloned().collect();
        interleaved.extend(clauses.iter().skip(1).step_by(2).cloned());

        for (shape, vtree) in vtree_shapes(num_vars) {
            let what = format!("{shape} vtree, {num_vars} vars, {} clauses", clauses.len());
            let original = compile_clauses(&vtree, &clauses);
            assert_same_shape(&original, &compile_clauses(&vtree, &reversed), &format!("{what}: reversed"));
            assert_same_shape(&original, &compile_clauses(&vtree, &interleaved), &format!("{what}: interleaved"));
            assert_canonical(&original);
        }
    }
}

/// A subsumed clause, a resolvent, a duplicate and a transitive closure all
/// leave the function alone, so they all leave the diagram alone.
#[test]
fn logically_equivalent_encodings_reach_the_same_diagram() {
    /// One function, written two ways: the lean encoding and a redundant one.
    struct Encodings {
        what: &'static str,
        lean: Vec<Vec<i32>>,
        redundant: Vec<Vec<i32>>,
    }

    let equivalent = |what, lean: &[&[i32]], redundant: &[&[i32]]| Encodings {
        what,
        lean: lean.iter().map(|c| c.to_vec()).collect(),
        redundant: redundant.iter().map(|c| c.to_vec()).collect(),
    };

    let cases = [
        equivalent(
            "subsumed clause",
            &[&[1, 2], &[-3, 4]],
            &[&[1, 2], &[-3, 4], &[1, 2, 3]],
        ),
        equivalent("resolvent", &[&[1, 2], &[-1, 3]], &[&[1, 2], &[-1, 3], &[2, 3]]),
        equivalent(
            "duplicated clauses",
            &[&[1, 2], &[-2, 3], &[-1, -3]],
            &[&[1, 2], &[-2, 3], &[-1, -3], &[1, 2], &[-2, 3]],
        ),
        equivalent(
            "transitive closure of an implication chain",
            &[&[-1, 2], &[-2, 3], &[-3, 4]],
            &[&[-1, 2], &[-2, 3], &[-3, 4], &[-1, 3], &[-2, 4], &[-1, 4]],
        ),
    ];

    for Encodings { what, lean, redundant } in &cases {
        assert_eq!(
            brute_force_count(4, lean),
            brute_force_count(4, redundant),
            "{what}: the two encodings are not the same function",
        );
        for (shape, vtree) in vtree_shapes(4) {
            let what = format!("{what}, {shape} vtree");
            let a = compile_clauses(&vtree, lean);
            let b = compile_clauses(&vtree, redundant);
            assert_eq!(model_count(&a), model_count(&b), "{what}: model count differs");
            assert_same_shape(&a, &b, &what);
            assert_canonical(&a);
        }
    }
}

/// Compiling one formula twice reaches one diagram: nothing in the fold
/// depends on state left over from an earlier compile.
#[test]
fn compiling_the_same_formula_twice_is_deterministic() {
    for (num_vars, clauses) in test_cases() {
        for (shape, vtree) in vtree_shapes(num_vars) {
            let what = format!("{shape} vtree, {num_vars} vars");
            assert_same_shape(
                &compile_clauses(&vtree, &clauses),
                &compile_clauses(&vtree, &clauses),
                &what,
            );
        }
    }
}

/// The same statement on a formula wide enough that the fold builds real
/// intermediate diagrams rather than a handful of nodes.
#[test]
fn clause_order_does_not_reach_a_different_diagram_on_four_queens() {
    let (num_vars, clauses) = queens_clauses(4);
    let vtree = Arc::new(Vtree::balanced(num_vars));

    let original = compile_clauses(&vtree, &clauses);
    let mut reversed = clauses;
    reversed.reverse();

    assert_same_shape(&original, &compile_clauses(&vtree, &reversed), "four queens");
    assert_canonical(&original);
}

/// A weight-marginal node's identity is its weight row, so two nodes on one
/// weight-marginal level that carry different values are different nodes and
/// must reach the signature check with different signatures.
#[test]
fn weighted_marginal_nodes_with_different_values_do_not_collide() {
    use crate::diagram::{Arithmetic, LiteralWeights, RationalWeights, WeightStore};
    use crate::test_helpers::{rat, toy_weighted};

    let store = WeightStore::new(
        RationalWeights::from_literals(&[LiteralWeights { negative: rat(2, 5), positive: rat(3, 11) }, LiteralWeights { negative: rat(1, 3), positive: rat(-4, 9) }]),
        Arithmetic::ExactRational,
    );
    let tdd = toy_weighted(store, vec![rat(3, 7), rat(1, 2)], &[&[(0, 0), (1, 1)]]);

    crate::test_helpers::check::check_canonicity(&tdd, 3).expect("distinct weight rows are distinct nodes");
}
