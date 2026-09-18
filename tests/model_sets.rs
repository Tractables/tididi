//! Building a circuit from a table of assignments, as a consumer sees it.

use std::sync::Arc;

use tididi::limits::{LimitConfig, OperationError};
use tididi::test_helpers::assert_canonical;
use tididi::vtree::VarId;
use tididi::{and, Tdd, Vtree};

/// A block of `width` variables starting at `first`.
fn block(first: u32, width: u32) -> Vec<VarId> {
    (first..first + width).map(VarId).collect()
}

/// `code` written most significant bit first onto `width` bits at `offset` of
/// a row that is `words` words long.
fn write_code(row: &mut [u64], offset: u32, width: u32, code: u32) {
    for j in 0..width {
        if (code >> (width - 1 - j)) & 1 == 1 {
            let bit = (offset + j) as usize;
            row[bit / 64] |= 1u64 << (bit % 64);
        }
    }
}

/// Pairs packed as two `width`-bit codes, the first in the low bits.
fn pair_rows(edges: &[(u32, u32)], width: u32) -> Vec<u64> {
    let words = (2 * width as usize).div_ceil(64).max(1);
    let mut rows = vec![0u64; words * edges.len()];
    for (k, &(u, v)) in edges.iter().enumerate() {
        let row = &mut rows[k * words..(k + 1) * words];
        write_code(row, 0, width, u);
        write_code(row, width, width, v);
    }
    rows
}

#[test]
fn a_relation_and_its_reverse_meet_on_a_shared_vtree() {
    // Two blocks of three variables each hold a node of a small graph; the
    // circuit is the set of edges, and conjoining it with the same relation
    // over the second and third block gives the paths of length two.
    let width = 3;
    let edges = [(0u32, 1u32), (1, 2), (2, 3), (3, 0), (0, 3), (5, 5)];
    let vtree = Arc::new(Vtree::linear(3 * width));

    let first = [block(1, width), block(1 + width, width)].concat();
    let second = [block(1 + width, width), block(1 + 2 * width, width)].concat();
    let rows = pair_rows(&edges, width);

    let f = Tdd::from_models(&vtree, &first, &rows).unwrap();
    let g = Tdd::from_models(&vtree, &second, &rows).unwrap();
    assert_canonical(&f);
    assert_canonical(&g);
    // Each atom leaves the third block free.
    assert_eq!(f.model_count().unwrap(), (edges.len() as u32 * 8).into());

    let paths = and(f, g).unwrap();
    assert_canonical(&paths);
    let want = edges
        .iter()
        .flat_map(|&(a, b)| edges.iter().map(move |&(c, d)| (a, b, c, d)))
        .filter(|&(_, b, c, _)| b == c)
        .count() as u32;
    assert_eq!(paths.model_count().unwrap(), want.into());
}

#[test]
fn a_hundred_variables_span_two_words_per_row() {
    let vtree = Arc::new(Vtree::balanced(100));
    let vars: Vec<VarId> = (1..=100).map(VarId).collect();
    // Row k sets the variables whose position is a multiple of k + 2.
    let mut rows = vec![0u64; 2 * 8];
    for k in 0..8usize {
        for i in 0..100usize {
            if i % (k + 2) == 0 {
                rows[k * 2 + i / 64] |= 1u64 << (i % 64);
            }
        }
    }
    let f = Tdd::from_models(&vtree, &vars, &rows).unwrap();
    assert_canonical(&f);
    assert_eq!(f.model_count().unwrap(), 8u32.into());
    assert!(f.is_sat().unwrap());
}

#[test]
fn the_degenerate_sizes_are_the_constants() {
    let vtree = Arc::new(Vtree::balanced(4));
    assert!(Tdd::from_models(&vtree, &[VarId(1)], &[]).unwrap().is_zero());
    assert_eq!(Tdd::from_models(&vtree, &[], &[0]).unwrap().model_count().unwrap(), 16u32.into());
}

#[test]
fn a_budget_that_cannot_hold_the_rows_refuses() {
    let vtree = Arc::new(Vtree::linear(10));
    let rows: Vec<u64> = (0..500u64).map(|k| k * 7 % 1024).collect();
    let vars: Vec<VarId> = (1..=10).map(VarId).collect();
    let refused = vtree
        .context()
        .with_limits(LimitConfig::none().with_memory_budget_bytes(Some(128)), |engine| {
            engine.from_models(&vtree, &vars, &rows)
        });
    assert_eq!(refused.unwrap_err(), OperationError::OverBudget);
}
