use std::sync::Arc;

use num_bigint::BigUint;

use crate::diagram::Tdd;
use crate::limits::{LimitConfig, OperationError};
use crate::test_helpers::{assert_canonical, assert_same_shape, brute_force_count, vtree_shapes};
use crate::vtree::rng::Lcg;
use crate::vtree::{VarId, Vtree};
use crate::Engine;

/// The variables `1..=n`.
fn vars(n: u32) -> Vec<VarId> {
    (1..=n).map(VarId).collect()
}

/// A flat row buffer from one value per constrained variable per row.
fn rows_of(table: &[Vec<bool>]) -> Vec<u64> {
    let w = super::words_per_row(table.first().map_or(0, Vec::len));
    let mut out = vec![0u64; w * table.len()];
    for (k, bits) in table.iter().enumerate() {
        for (i, &b) in bits.iter().enumerate() {
            if b {
                out[k * w + i / 64] |= 1u64 << (i % 64);
            }
        }
    }
    out
}

/// The `width` bits of `code`, most significant first.
fn code_bits(code: u32, width: u32) -> Vec<bool> {
    (0..width).map(|j| (code >> (width - 1 - j)) & 1 == 1).collect()
}

/// Clauses blocking every assignment to the first `width` variables that the
/// table does not hold, which is the row set read as a formula.
fn blocking_clauses(table: &[Vec<bool>], width: u32) -> Vec<Vec<i32>> {
    let mut clauses = Vec::new();
    for assignment in 0..1u32 << width {
        let bits: Vec<bool> = (0..width).map(|i| (assignment >> i) & 1 == 1).collect();
        if table.contains(&bits) {
            continue;
        }
        clauses.push(
            (0..width as i32)
                .map(|i| if bits[i as usize] { -(i + 1) } else { i + 1 })
                .collect(),
        );
    }
    clauses
}

/// The same function built as a disjunction of cubes.
fn or_of_cubes(vtree: &Arc<Vtree>, constrained: &[VarId], table: &[Vec<bool>]) -> Tdd {
    let mut f = Tdd::zero(vtree);
    for bits in table {
        let cube: Vec<i32> = bits
            .iter()
            .zip(constrained)
            .map(|(&b, v)| if b { v.0 as i32 } else { -(v.0 as i32) })
            .collect();
        f = crate::or(f, Tdd::cube(vtree, cube).unwrap()).unwrap();
    }
    f
}

/// The rows deduplicated, which is how many models the constrained variables
/// account for.
fn distinct(table: &[Vec<bool>]) -> usize {
    let mut sorted = table.to_vec();
    sorted.sort();
    sorted.dedup();
    sorted.len()
}

#[test]
fn no_rows_is_false_and_no_variables_is_true() {
    let vtree = Arc::new(Vtree::balanced(3));
    let none = Tdd::from_models(&vtree, &vars(2), &[]).unwrap();
    assert!(none.is_zero());
    assert_canonical(&none);

    let all = Tdd::from_models(&vtree, &[], &[0]).unwrap();
    assert_canonical(&all);
    assert_eq!(all.model_count().unwrap(), 8u32.into());

    // No variables and no rows is still the empty relation.
    let empty = Tdd::from_models(&vtree, &[], &[]).unwrap();
    assert!(empty.is_zero());
}

#[test]
fn bits_past_the_last_variable_and_repeated_rows_are_ignored() {
    let vtree = Arc::new(Vtree::balanced(4));
    let clean = Tdd::from_models(&vtree, &vars(3), &[0b101, 0b010]).unwrap();
    let noisy = Tdd::from_models(&vtree, &vars(3), &[!0b010u64, 0b010, 0b101, 0b010]).unwrap();
    assert_canonical(&clean);
    assert_canonical(&noisy);
    assert_same_shape(&clean, &noisy, "padding bits and repeats");
    assert_eq!(clean.model_count().unwrap(), 4u32.into());
}

#[test]
fn a_ragged_row_buffer_is_refused() {
    let vtree = Arc::new(Vtree::balanced(70));
    assert!(matches!(
        Tdd::from_models(&vtree, &vars(70), &[0, 0, 0]),
        Err(OperationError::RaggedRows { words: 3, per_row: 2 })
    ));
}

#[test]
fn an_unknown_or_repeated_variable_is_refused() {
    let vtree = Arc::new(Vtree::balanced(3));
    assert_eq!(
        Tdd::from_models(&vtree, &[VarId(1), VarId(9)], &[0]).unwrap_err(),
        OperationError::VariableNotInVtree(VarId(9)),
    );
    assert_eq!(
        Tdd::from_models(&vtree, &[VarId(2), VarId(2)], &[0]).unwrap_err(),
        OperationError::DuplicateVariable(VarId(2)),
    );
}

#[test]
fn free_variables_in_the_middle_of_the_vtree_multiply_the_count() {
    let constrained = [VarId(2), VarId(5), VarId(7)];
    let table = vec![vec![false, false, true], vec![true, true, true], vec![true, false, false]];
    for (name, vtree) in vtree_shapes(8) {
        let f = Tdd::from_models(&vtree, &constrained, &rows_of(&table)).unwrap();
        assert_canonical(&f);
        assert_eq!(f.model_count().unwrap(), (3u32 * 32).into(), "{name}");
        assert_same_shape(&f, &or_of_cubes(&vtree, &constrained, &table), name);
    }
}

#[test]
fn the_canonicity_gap_of_four_nodes_and_six_edges() {
    // The relation a direct construction first got wrong, emitting overlapping
    // nodes: four values, two bits per endpoint, six edges.
    let edges = [(0u32, 1u32), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3)];
    let table: Vec<Vec<bool>> = edges
        .iter()
        .map(|&(u, v)| {
            let mut bits = code_bits(u, 2);
            bits.extend(code_bits(v, 2));
            bits
        })
        .collect();
    for (name, vtree) in vtree_shapes(4) {
        let f = Tdd::from_models(&vtree, &vars(4), &rows_of(&table)).unwrap();
        assert_canonical(&f);
        assert_eq!(f.model_count().unwrap(), 6u32.into(), "{name}");
        let mut minimized = f.clone();
        minimized.minimize().unwrap();
        assert_eq!(minimized.node_count(), f.node_count(), "{name} was not already minimal");
        assert_same_shape(&f, &or_of_cubes(&vtree, &vars(4), &table), name);
    }
}

#[test]
fn small_relations_match_an_or_of_cubes_and_the_enumerated_count() {
    let mut rng = Lcg::new(20260917);
    for width in 1..=5u32 {
        for free in 0..=2u32 {
            let num_vars = width + free;
            for (name, vtree) in vtree_shapes(num_vars) {
                for round in 0..4 {
                    let table: Vec<Vec<bool>> = (0..1u32 << width)
                        .filter(|_| !rng.next_u64().is_multiple_of(3))
                        .map(|code| (0..width).map(|i| (code >> i) & 1 == 1).collect())
                        .collect();
                    let constrained = vars(width);
                    let f = Tdd::from_models(&vtree, &constrained, &rows_of(&table)).unwrap();
                    assert_canonical(&f);
                    let label = format!("{name}, {width} constrained, {free} free, round {round}");
                    let want = brute_force_count(num_vars, &blocking_clauses(&table, width));
                    assert_eq!(f.model_count().unwrap(), want.into(), "{label}");
                    assert_same_shape(&f, &or_of_cubes(&vtree, &constrained, &table), &label);
                }
            }
        }
    }
}

#[test]
fn a_relation_wider_than_one_word_reads_both_words() {
    // Seventy constrained variables scattered through a hundred-leaf vtree, so
    // a row spans two words and the leaf order is not the input order.
    let vtree = Arc::new(Vtree::balanced(100));
    let constrained: Vec<VarId> = (1..=70).map(|i| VarId(i * 100 / 70)).collect();
    let mut rng = Lcg::new(7);
    let table: Vec<Vec<bool>> =
        (0..40).map(|_| (0..70).map(|_| rng.coin()).collect()).collect();
    let rows = rows_of(&table);
    assert_eq!(rows.len(), 2 * table.len());

    let f = Tdd::from_models(&vtree, &constrained, &rows).unwrap();
    assert_canonical(&f);
    assert_eq!(f.model_count().unwrap(), BigUint::from(distinct(&table)) << 30u32);
    assert_same_shape(
        &f,
        &or_of_cubes(&vtree, &constrained, &table),
        "seventy variables over two words",
    );
}

#[test]
fn a_tiny_budget_refuses_without_leaving_a_diagram() {
    let vtree = Arc::new(Vtree::linear(12));
    let mut rng = Lcg::new(3);
    let table: Vec<Vec<bool>> =
        (0..200).map(|_| (0..12).map(|_| rng.coin()).collect()).collect();
    let rows = rows_of(&table);

    let refused = vtree.context().with_limits(
        LimitConfig::none().with_memory_budget_bytes(Some(64)),
        |eng| eng.from_models(&vtree, &vars(12), &rows),
    );
    assert_eq!(refused.unwrap_err(), OperationError::OverBudget);

    // The same rows build once the budget is out of the way.
    let f = Tdd::from_models(&vtree, &vars(12), &rows).unwrap();
    assert_canonical(&f);
    assert_eq!(f.model_count().unwrap(), BigUint::from(distinct(&table)));
}

#[test]
fn every_refusal_point_answers_over_budget_and_returns_the_buffers() {
    let vtree = Arc::new(Vtree::balanced(6));
    let table: Vec<Vec<bool>> = (0..20u32)
        .map(|code| (0..6).map(|i| ((code * 7) >> i) & 1 == 1).collect())
        .collect();
    let rows = rows_of(&table);
    let mut refused = 0;
    for cut in 0..40u32 {
        let eng = Engine::new();
        // Park a level array so the pool holds exactly one either way: a build
        // that takes it and then refuses has to hand it back.
        Tdd::builder(&eng, &vtree).abandon(&eng);
        eng.limits().refuse_nth_reserve(cut);
        match eng.from_models(&vtree, &vars(6), &rows) {
            Ok(f) => assert_canonical(&f),
            Err(e) => {
                assert_eq!(e, OperationError::OverBudget, "cut {cut}");
                assert_eq!(eng.levels().occupancy(), 1, "cut {cut} lost a level buffer");
                refused += 1;
            }
        }
        eng.limits().grant_every_reserve();
    }
    assert!(refused > 0, "no reservation was refused across the sweep");
}
