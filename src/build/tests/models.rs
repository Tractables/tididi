use std::sync::Arc;

use num_bigint::BigUint;

use crate::diagram::Tdd;
use crate::limits::{LimitConfig, OperationError};
use crate::test_helpers::{assert_canonical, assert_same_shape, brute_force_count, packed_rows, vtree_shapes};
use crate::vtree::rng::Lcg;
use crate::vtree::{VarId, Vtree};
use crate::Engine;

/// The variables `1..=n`.
fn vars(n: u32) -> Vec<VarId> {
    (1..=n).map(VarId).collect()
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

/// The rows of `table` as a function of `constrained`, built as a disjunction
/// of one cube per row. Row-driven rather than truth-table driven, so a
/// relation over seventy variables is as cheap as its row count.
fn table_function(vtree: &Arc<Vtree>, constrained: &[VarId], table: &[Vec<bool>]) -> Tdd {
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
        let f = Tdd::from_models(&vtree, &constrained, &packed_rows(table.first().map_or(0, Vec::len), &table)).unwrap();
        assert_canonical(&f);
        assert_eq!(f.model_count().unwrap(), (3u32 * 32).into(), "{name}");
        assert_same_shape(&f, &table_function(&vtree, &constrained, &table), name);
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
        let f = Tdd::from_models(&vtree, &vars(4), &packed_rows(table.first().map_or(0, Vec::len), &table)).unwrap();
        assert_canonical(&f);
        assert_eq!(f.model_count().unwrap(), 6u32.into(), "{name}");
        let mut minimized = f.clone();
        minimized.minimize().unwrap();
        assert_eq!(minimized.node_count(), f.node_count(), "{name} was not already minimal");
        assert_same_shape(&f, &table_function(&vtree, &vars(4), &table), name);
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
                    let f = Tdd::from_models(&vtree, &constrained, &packed_rows(table.first().map_or(0, Vec::len), &table)).unwrap();
                    assert_canonical(&f);
                    let label = format!("{name}, {width} constrained, {free} free, round {round}");
                    let want = brute_force_count(num_vars, &blocking_clauses(&table, width));
                    assert_eq!(f.model_count().unwrap(), want.into(), "{label}");
                    assert_same_shape(&f, &table_function(&vtree, &constrained, &table), &label);
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
    let rows = packed_rows(table.first().map_or(0, Vec::len), &table);
    assert_eq!(rows.len(), 2 * table.len());

    let f = Tdd::from_models(&vtree, &constrained, &rows).unwrap();
    assert_canonical(&f);
    assert_eq!(f.model_count().unwrap(), BigUint::from(distinct(&table)) << 30u32);
    assert_same_shape(
        &f,
        &table_function(&vtree, &constrained, &table),
        "seventy variables over two words",
    );
}

#[test]
fn the_radix_pass_leaves_the_order_a_comparison_sort_leaves() {
    let mut rng = Lcg::new(5);
    for num_vars in [1usize, 11, 12, 34, 63, 64] {
        let mask = if num_vars == 64 { !0u64 } else { (1u64 << num_vars) - 1 };
        let words: Vec<u64> =
            (0..super::rows::RADIX_MIN_ROWS + 37).map(|_| rng.next_u64() & mask).collect();
        let mut want = words.clone();
        want.sort_unstable();
        let mut got = words;
        let eng = Engine::new();
        super::rows::sort_words(eng.limits(), &mut got, num_vars).unwrap();
        assert_eq!(got, want, "{num_vars} variables");
    }
}

#[test]
fn leaf_partitions_match_every_small_relation() {
    // Exhaustive row sets include independence, constants and correlations.
    // Free leaves exercise carrying a lazy bit through a one-sided subtree.
    for subset in 1u32..256 {
        let table: Vec<Vec<bool>> = (0..8).filter(|row| subset & (1 << row) != 0)
            .map(|row| (0..3).map(|bit| row & (1 << bit) != 0).collect()).collect();
        for (name, vtree) in vtree_shapes(5) {
            let constrained = [VarId(1), VarId(3), VarId(5)];
            let f = Tdd::from_models(&vtree, &constrained, &packed_rows(table.first().map_or(0, Vec::len), &table)).unwrap();
            assert_canonical(&f);
            assert_same_shape(&f, &table_function(&vtree, &constrained, &table), name);
        }
    }
}

#[test]
fn leaf_independence_checks_the_complete_external_assignment() {
    let eng = Engine::new();
    // A shifted two-word encoding moves the toggled bit across word edges;
    // equal counts of zeros and ones alone do not imply equal completions.
    for bit in [0usize, 31, 63, 64, 65, 70] {
        let word = bit / 64;
        let mask = 1u64 << (bit % 64);
        let mut rng = Lcg::new(317);
        for correlated in [false, true] {
            let mut rows = Vec::new();
            for _ in 0..13 {
                let mut row = [rng.next_u64(), rng.next_u64()];
                row[word] &= !mask;
                rows.push(row);
                row[word] |= mask;
                if correlated { row[1 - word] ^= 1; }
                rows.push(row);
            }
            rows.sort_unstable_by_key(|r| (r[1], r[0]));
            rows.dedup();
            let expected = rows.iter().all(|r| {
                let mut toggled = *r;
                toggled[word] ^= mask;
                rows.contains(&toggled)
            });
            let packed: Vec<_> = rows.into_iter().flatten().collect();
            assert_eq!(super::independent_bit(eng.limits(), &packed, 2, word, mask).unwrap(), expected);
        }
    }
}

#[test]
fn a_closed_subtree_merges_duplicate_child_pairs_below_free_ancestors() {
    let vtree = Arc::new(Vtree::balanced(16));
    // The constrained columns occupy a proper subtree. Cartesian blocks
    // repeat child-atom pairs, while correlations keep other atoms separate.
    for constrained in [vec![VarId(1), VarId(2), VarId(3), VarId(4)],
                        vec![VarId(1), VarId(3), VarId(5), VarId(7)]] {
        let table: Vec<Vec<bool>> = (0u32..16)
            .filter(|r| r & 0b1100 == 0 || r & 0b0011 == 0b0011)
            .map(|r| (0..4).map(|b| r & (1 << b) != 0).collect()).collect();
        let f = Tdd::from_models(&vtree, &constrained, &packed_rows(table.first().map_or(0, Vec::len), &table)).unwrap();
        assert_canonical(&f);
        assert_same_shape(&f, &table_function(&vtree, &constrained, &table), "closed proper subtree");
        assert_eq!(f.model_count().unwrap(), BigUint::from(table.len()) << 12);
    }
}

#[test]
fn unique_and_shared_completions_match_cube_construction() {
    // A bijection gives each value its own completion; a many-to-one map
    // merges some values into atoms with multiple child pairs.
    for merge in [false, true] {
        let table: Vec<Vec<bool>> = (0u32..16).map(|x| {
            let y = if merge { x % 4 } else { (x * 7) % 16 };
            let mut bits = code_bits(x, 4);
            bits.extend(code_bits(y, 4));
            bits
        }).collect();
        for (name, vtree) in vtree_shapes(12) {
            for constrained in [vars(8), vec![VarId(1), VarId(2), VarId(4), VarId(5),
                VarId(7), VarId(8), VarId(10), VarId(11)]] {
                let f = Tdd::from_models(&vtree, &constrained, &packed_rows(table.first().map_or(0, Vec::len), &table)).unwrap();
                assert_canonical(&f);
                assert_same_shape(&f, &table_function(&vtree, &constrained, &table), name);
                assert_eq!(f.model_count().unwrap(), 256u32.into());
            }
        }
    }
}

#[test]
fn a_tiny_budget_refuses_without_leaving_a_diagram() {
    let vtree = Arc::new(Vtree::linear(12));
    let mut rng = Lcg::new(3);
    let table: Vec<Vec<bool>> =
        (0..200).map(|_| (0..12).map(|_| rng.coin()).collect()).collect();
    let rows = packed_rows(table.first().map_or(0, Vec::len), &table);

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
    let rows = packed_rows(table.first().map_or(0, Vec::len), &table);
    let mut refused = 0;
    for cut in 0..40u32 {
        let eng = Engine::new();
        // Park a level array so the pool holds exactly one either way: a build
        // that takes it and then refuses has to hand it back.
        Tdd::builder(&eng, &vtree).unwrap().abandon(&eng);
        eng.limits().refuse_nth_reserve(cut);
        match eng.from_models(&vtree, &vars(6), &rows) {
            Ok(f) => assert_canonical(&f),
            Err(e) => {
                assert_eq!(e, OperationError::OverBudget, "cut {cut}");
                assert_eq!(eng.scratch.levels.occupancy(), 1, "cut {cut} lost a level buffer");
                refused += 1;
            }
        }
        eng.limits().grant_every_reserve();
    }
    assert!(refused > 0, "no reservation was refused across the sweep");
}

#[test]
fn cached_layout_tracks_column_order_and_vtree_identity() {
    let eng = Engine::new();
    let a = Arc::new(Vtree::balanced(4));
    let b = Arc::new(Vtree::linear(4));
    for vtree in [&a, &b, &a] {
        for columns in [[VarId(1), VarId(4)], [VarId(4), VarId(1)]] {
            for rows in [&[1u64][..], &[2u64, 3][..], &[][..]] {
                let f = eng.from_models(vtree, &columns, rows).unwrap();
                let table: Vec<_> = rows.iter().map(|&r| vec![r & 1 != 0, r & 2 != 0]).collect();
                let expected = table_function(vtree, &columns, &table);
                assert_canonical(&f);
                assert_canonical(&expected);
                assert!(f.equivalent(&expected).unwrap());
            }
        }
    }
}

#[test]
fn layout_cache_reuses_storage_and_does_not_retain_vtree() {
    let eng = Engine::new();
    let vtree = Arc::new(Vtree::balanced(4));
    let weak = Arc::downgrade(&vtree);
    let columns = [VarId(3), VarId(1)];
    let f = eng.from_models(&vtree, &columns, &[1, 2]).unwrap();
    assert_canonical(&f);
    drop(f);
    let allocation = eng.scratch.model_layout.checkout(eng.limits()).position.as_ptr();
    let f = eng.from_models(&vtree, &columns, &[3]).unwrap();
    assert_canonical(&f);
    assert_eq!(eng.scratch.model_layout.checkout(eng.limits()).position.as_ptr(), allocation);
    drop(f);
    assert_eq!(eng.from_models(&vtree, &[VarId(2), VarId(2)], &[0]).unwrap_err(),
        OperationError::DuplicateVariable(VarId(2)));
    let f = eng.from_models(&vtree, &columns, &[2]).unwrap();
    assert_canonical(&f);
    assert_eq!(eng.scratch.model_layout.checkout(eng.limits()).position.as_ptr(), allocation);
    drop(f);
    drop(vtree);
    // A context parks an engine, so its layout must not keep the vtree alive.
    assert!(weak.upgrade().is_none());
}
