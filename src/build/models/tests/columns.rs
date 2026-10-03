use std::sync::Arc;

use super::*;
use crate::test_helpers::{assert_canonical, assert_same_shape, vtree_shapes, Lcg};
use crate::vtree::Vtree;

/// The chosen rows of a table of `codes` on `columns`, packed for
/// `from_models` over the variables `columns` lists, column after column.
fn packed(columns: &[&[VarId]], codes: &[&[u32]], chosen: &[usize]) -> Vec<u64> {
    let width: usize = columns.iter().map(|c| c.len()).sum();
    let w = width.div_ceil(64).max(1);
    let mut out = Vec::new();
    for &r in chosen {
        let start = out.len();
        out.resize(start + w, 0u64);
        let mut at = 0;
        for (column, code) in columns.iter().zip(codes) {
            for i in 0..column.len() {
                if (code[r] >> (column.len() - 1 - i)) & 1 == 1 {
                    out[start + (at + i) / 64] |= 1u64 << ((at + i) % 64);
                }
            }
            at += column.len();
        }
    }
    out
}

/// The rows `rows` chooses of a table of `len` rows, in the order it reads them.
fn chosen(rows: RowSelection<'_>, len: usize) -> Vec<usize> {
    match rows {
        RowSelection::All => (0..len).collect(),
        RowSelection::Listed(listed) => listed.iter().map(|&r| r as usize).collect(),
        RowSelection::Marked(marked) => (0..len).filter(|&r| marked[r]).collect(),
    }
}

/// `from_columns` against `from_models` on the same rows packed by hand.
fn check(vtree: &Arc<Vtree>, columns: &[&[VarId]], codes: &[&[u32]], rows: RowSelection<'_>, what: &str) {
    let got = Tdd::from_columns(vtree, columns, codes, rows).unwrap();
    assert_canonical(&got);
    let vars: Vec<VarId> = columns.iter().flat_map(|c| c.iter().copied()).collect();
    let len = codes.first().map_or(0, |c| c.len());
    let want = Tdd::from_models(vtree, &vars, &packed(columns, codes, &chosen(rows, len))).unwrap();
    assert!(got.equivalent(&want).unwrap(), "{what}");
    assert_same_shape(&got, &want, what);
}

/// A random table on the variables `1..=num_vars` cut into columns of at
/// most 32 bits, with codes that set bits past their column's width too,
/// and few enough distinct codes that rows repeat.
fn random_table(rng: &mut Lcg, num_vars: u32, len: usize) -> (Vec<Vec<VarId>>, Vec<Vec<u32>>) {
    let mut order: Vec<VarId> = (1..=num_vars).map(VarId).collect();
    for i in (1..order.len()).rev() {
        order.swap(i, rng.below(i as u64 + 1) as usize);
    }
    let mut columns = Vec::new();
    let mut at = 0;
    while at < order.len() {
        let width = (1 + rng.below(32) as usize).min(order.len() - at);
        columns.push(order[at..at + width].to_vec());
        at += width;
    }
    let codes = columns
        .iter()
        .map(|_| {
            let values = 1 + rng.below(6);
            let spread = rng.next_u64() as u32;
            (0..len).map(|_| (rng.below(values) as u32).wrapping_mul(spread | 1) ^ rng.below(2) as u32).collect()
        })
        .collect();
    (columns, codes)
}

#[test]
fn random_tables_build_what_their_packed_rows_build() {
    // Columns of one to 32 bits laid on every vtree shape, so a column's
    // bits are next to each other in some and scattered in others, every
    // selection of the rows, and tables past one word of variables.
    let mut rng = Lcg::new(0xc01d_2026);
    for num_vars in [3u32, 9, 40, 70, 100] {
        for (name, vtree) in vtree_shapes(num_vars) {
            for round in 0..4 {
                let len = 1 + rng.below(300) as usize;
                let (columns, codes) = random_table(&mut rng, num_vars, len);
                let columns: Vec<&[VarId]> = columns.iter().map(|c| &c[..]).collect();
                let codes: Vec<&[u32]> = codes.iter().map(|c| &c[..]).collect();
                let listed: Vec<u32> = (0..rng.below(2 * len as u64)).map(|_| rng.below(len as u64) as u32).collect();
                let marked: Vec<bool> = (0..len).map(|_| rng.below(3) > 0).collect();
                for (how, rows) in
                    [("all", RowSelection::All), ("listed", RowSelection::Listed(&listed)), ("marked", RowSelection::Marked(&marked))]
                {
                    check(&vtree, &columns, &codes, rows, &format!("{num_vars} vars on {name}, round {round}, {how}"));
                }
            }
        }
    }
}

#[test]
fn columns_in_leaf_order_shift_into_place_and_sorted_rows_are_only_checked() {
    // Each column's variables left to right in the vtree, the columns in
    // order: every code is shifted into its row, and rows sorted on the
    // columns come in the build's order already.
    let mut rng = Lcg::new(0x0dd_c01);
    let widths = [3usize, 8, 1, 17, 30];
    let total: usize = widths.iter().sum();
    let order: Vec<VarId> = (1..=total as u32).map(VarId).collect();
    for vtree in [Vtree::balanced_over(&order).unwrap(), Vtree::linear_from_order(&order).unwrap()] {
        let vtree = Arc::new(vtree);
        let mut columns = Vec::new();
        let mut at = 0;
        for &w in &widths {
            columns.push(&order[at..at + w]);
            at += w;
        }
        let mut rows: Vec<Vec<u32>> = (0..500)
            .map(|_| widths.iter().map(|&w| (rng.next_u64() as u32) & ((1u64 << w) - 1) as u32 & 0x3f).collect())
            .collect();
        rows.sort();
        rows.dedup();
        let codes: Vec<Vec<u32>> = (0..widths.len()).map(|j| rows.iter().map(|row| row[j]).collect()).collect();
        let codes: Vec<&[u32]> = codes.iter().map(|c| &c[..]).collect();
        check(&vtree, &columns, &codes, RowSelection::All, "sorted");
        let reversed: Vec<u32> = (0..rows.len() as u32).rev().collect();
        check(&vtree, &columns, &codes, RowSelection::Listed(&reversed), "reversed");
    }
}

#[test]
fn no_rows_is_false_and_no_columns_is_true() {
    let vtree = Arc::new(Vtree::balanced(3));
    let column: &[VarId] = &[VarId(1), VarId(2)];
    let none = Tdd::from_columns(&vtree, &[column], &[&[1, 2]], RowSelection::Marked(&[false, false])).unwrap();
    assert!(none.is_zero());
    assert_canonical(&none);
    let empty = Tdd::from_columns(&vtree, &[column], &[&[]], RowSelection::All).unwrap();
    assert!(empty.is_zero());
    let all = Tdd::from_columns(&vtree, &[], &[], RowSelection::Listed(&[])).unwrap();
    assert!(all.is_zero(), "no row chosen");
    let one: &[VarId] = &[];
    let all = Tdd::from_columns(&vtree, &[one], &[&[7, 9]], RowSelection::All).unwrap();
    assert_canonical(&all);
    assert_eq!(all.model_count().unwrap(), 8u32.into());
}

#[test]
fn ragged_columns_unknown_rows_and_bad_variables_are_refused() {
    let vtree = Arc::new(Vtree::balanced(40));
    let (a, b): (&[VarId], &[VarId]) = (&[VarId(1), VarId(2)], &[VarId(3)]);
    let err = |columns: &[&[VarId]], codes: &[&[u32]], rows| Tdd::from_columns(&vtree, columns, codes, rows).unwrap_err();
    assert_eq!(err(&[a, b], &[&[1, 2]], RowSelection::All), OperationError::RaggedColumns { column: 1, len: 0, expected: 2 });
    assert_eq!(err(&[a, b], &[&[1, 2], &[0]], RowSelection::All), OperationError::RaggedColumns { column: 1, len: 1, expected: 2 });
    assert_eq!(
        err(&[a, b], &[&[1, 2], &[0, 1]], RowSelection::Marked(&[true])),
        OperationError::RaggedColumns { column: 2, len: 1, expected: 2 }
    );
    assert_eq!(err(&[a, b], &[&[1, 2], &[0, 1]], RowSelection::Listed(&[0, 2])), OperationError::RowOutOfRange { row: 2, rows: 2 });
    let wide: Vec<VarId> = (1..=33).map(VarId).collect();
    assert_eq!(err(&[&wide], &[&[0]], RowSelection::All), OperationError::ColumnTooWide { column: 0, bits: 33 });
    assert_eq!(err(&[a, &[VarId(41)]], &[&[1], &[0]], RowSelection::All), OperationError::VariableNotInVtree(VarId(41)));
    assert_eq!(err(&[a, &[VarId(2)]], &[&[1], &[0]], RowSelection::All), OperationError::DuplicateVariable(VarId(2)));
}
