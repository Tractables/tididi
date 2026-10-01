//! `Tdd::model_columns` against the models read off a truth table.

use std::collections::BTreeSet;
use std::sync::Arc;

use super::*;
use crate::reduce::ReductionPlan;
use crate::test_helpers::{
    assert_canonical, compile_clauses, rand_cnf, truth_table, vtree_shapes, CnfShape, Lcg,
};
use crate::vtree::{Vtree, VtreeIdx, VtreeNode};

/// The rows of a column table, one vector per row.
fn rows_of(columns: &[Vec<u32>]) -> Vec<Vec<u32>> {
    let n = columns.first().map_or(0, Vec::len);
    (0..n).map(|r| columns.iter().map(|c| c[r]).collect()).collect()
}

/// The distinct rows the truth table's true assignments give on `columns`.
fn expected(num_vars: u32, truth: &[bool], columns: &[Vec<VarId>]) -> BTreeSet<Vec<u32>> {
    (0..1u64 << num_vars)
        .filter(|&mask| truth[mask as usize])
        .map(|mask| {
            columns
                .iter()
                .map(|vars| vars.iter().fold(0u32, |code, v| code << 1 | ((mask >> (v.0 - 1)) & 1) as u32))
                .collect()
        })
        .collect()
}

/// A random split of a random subset of `1..=num_vars` into one to three
/// columns, each in a random order.
fn random_columns(rng: &mut Lcg, num_vars: u32) -> Vec<Vec<VarId>> {
    let ncols = 1 + rng.below(3) as usize;
    let mut columns = vec![Vec::new(); ncols];
    let mut vars: Vec<u32> = (1..=num_vars).collect();
    for i in (1..vars.len()).rev() {
        vars.swap(i, rng.below(i as u64 + 1) as usize);
    }
    for v in vars {
        // About one variable in four is left unlisted.
        let pick = rng.below(4 * ncols as u64 / 3 + 1) as usize;
        if pick < ncols {
            columns[pick].push(VarId(v));
        }
    }
    columns
}

/// Checks `f`'s table on `columns` against the projection of `truth`, and
/// checks that batches written over random ranges tile the whole table.
fn check(f: &Tdd, num_vars: u32, truth: &[bool], columns: &[Vec<VarId>], rng: &mut Lcg, what: &str) {
    let refs: Vec<&[VarId]> = columns.iter().map(Vec::as_slice).collect();
    let mut table = f.model_columns(&refs).unwrap_or_else(|e| panic!("{what}: {e}"));
    let want = expected(num_vars, truth, columns);
    assert_eq!(table.rows(), want.len() as u64, "{what}: rows");
    let whole = table.to_columns();
    let rows = rows_of(&whole);
    let got: BTreeSet<Vec<u32>> = rows.iter().cloned().collect();
    assert_eq!(got.len(), rows.len(), "{what}: a row repeats");
    assert_eq!(got, want, "{what}: rows differ");
    // Batches over random ranges, in a fresh table and in the used one.
    for fresh in [true, false] {
        let mut table = match fresh {
            true => f.model_columns(&refs).unwrap(),
            false => table.clone(),
        };
        let mut batched: Vec<Vec<u32>> = vec![Vec::new(); columns.len()];
        let mut at = 0u64;
        while at < table.rows() {
            let end = (at + 1 + rng.below(7)).min(table.rows());
            let mut bufs: Vec<Vec<u32>> = vec![vec![u32::MAX; (end - at) as usize]; columns.len()];
            let mut out: Vec<&mut [u32]> = bufs.iter_mut().map(Vec::as_mut_slice).collect();
            table.write(at..end, &mut out);
            for (b, part) in batched.iter_mut().zip(bufs) {
                b.extend(part);
            }
            at = end;
        }
        assert_eq!(batched, whole, "{what}: batches differ from the whole table (fresh {fresh})");
    }
}

#[test]
fn columns_match_projected_truth_tables_on_every_shape() {
    let mut rng = Lcg::new(7);
    for num_vars in [1u32, 2, 3, 5, 7, 9] {
        for (shape, vtree) in vtree_shapes(num_vars) {
            for round in 0..12 {
                let clauses = rand_cnf(&mut rng, num_vars, CnfShape { clauses: 5, width: 3 });
                let f = compile_clauses(&vtree, &clauses);
                assert_canonical(&f);
                let columns = random_columns(&mut rng, num_vars);
                let listed: BTreeSet<VarId> = columns.iter().flatten().copied().collect();
                let unlisted: Vec<VarId> = (1..=num_vars).map(VarId).filter(|v| !listed.contains(v)).collect();
                let truth = truth_table(num_vars, &clauses);
                let what = format!("{num_vars} vars, {shape}, round {round}, columns {columns:?}");
                // Minimized, then pruned without contracting twins.
                let g = vtree.context().run(|e| e.exists_vars(f.clone(), &unlisted)).unwrap();
                assert_canonical(&g);
                check(&g, num_vars, &truth, &columns, &mut rng, &format!("{what}, minimized"));
                let g = vtree.context().run(|e| e.exists_vars_with(f.clone(), &unlisted, ReductionPlan::Prune)).unwrap();
                match g.model_columns(&columns.iter().map(Vec::as_slice).collect::<Vec<_>>()) {
                    Ok(_) => check(&g, num_vars, &truth, &columns, &mut rng, &format!("{what}, pruned")),
                    Err(OperationError::UnlistedLiteral(v)) => assert!(!listed.contains(&v), "{what}"),
                    Err(e) => panic!("{what}: {e}"),
                }
            }
        }
    }
}

#[test]
fn columns_of_unminimized_conjunctions_are_distinct() {
    // A conjunction leaves its result unminimized, twins and all; every
    // variable is listed, so determinism alone keeps the rows distinct.
    let mut rng = Lcg::new(11);
    for num_vars in [4u32, 6, 8] {
        for (shape, vtree) in vtree_shapes(num_vars) {
            for round in 0..8 {
                let clauses = rand_cnf(&mut rng, num_vars, CnfShape { clauses: 6, width: 3 });
                let mut f = Tdd::one(&vtree);
                for c in &clauses {
                    f = crate::and(f, Tdd::clause(&vtree, c.iter().copied()).unwrap()).unwrap();
                }
                let mut columns = random_columns(&mut rng, num_vars);
                let listed: BTreeSet<VarId> = columns.iter().flatten().copied().collect();
                columns[0].extend((1..=num_vars).map(VarId).filter(|v| !listed.contains(v)));
                columns[0].truncate(MAX_COLUMN_BITS);
                let truth = truth_table(num_vars, &clauses);
                check(&f, num_vars, &truth, &columns, &mut rng, &format!("{num_vars} vars, {shape}, round {round}"));
            }
        }
    }
}

#[test]
fn a_listed_block_left_free_yields_every_code() {
    // x1 and nothing about x2..x4: the column over x2..x4 holds all eight
    // codes beside x1 = 1.
    for vtree in [Arc::new(Vtree::balanced(4)), Arc::new(Vtree::linear(4)), Arc::new(Vtree::random(4, 3))] {
        let f = crate::literal(&vtree, 1).unwrap();
        assert_canonical(&f);
        let mut table = f.model_columns(&[&[VarId(1)], &[VarId(2), VarId(3), VarId(4)]]).unwrap();
        let columns = table.to_columns();
        let rows: BTreeSet<Vec<u32>> = rows_of(&columns).into_iter().collect();
        assert_eq!(rows, (0..8).map(|c| vec![1, c]).collect());
        // Free below one branch only: x1 → x2, so x1 = 0 leaves x2 free.
        let g = Tdd::clause(&vtree, [-1, 2]).unwrap();
        let mut table = g.model_columns(&[&[VarId(1), VarId(2)]]).unwrap();
        let mut codes = table.to_columns().remove(0);
        codes.sort_unstable();
        assert_eq!(codes, [0, 1, 3]);
    }
}

#[test]
fn constants_and_degenerate_tables() {
    let vtree = Arc::new(Vtree::balanced(3));
    let mut zero = Tdd::zero(&vtree).model_columns(&[&[VarId(1)]]).unwrap();
    assert_eq!(zero.rows(), 0);
    assert_eq!(zero.to_columns(), vec![Vec::<u32>::new()]);
    let mut one = Tdd::one(&vtree).model_columns(&[]).unwrap();
    assert_eq!((one.rows(), one.width()), (1, 0));
    assert!(one.to_columns().is_empty());
    let mut one = Tdd::one(&vtree).model_columns(&[&[], &[VarId(2)]]).unwrap();
    assert_eq!(one.to_columns(), vec![vec![0, 0], vec![0, 1]]);
    // A one-variable vtree: the output is a leaf node.
    let leaf = Arc::new(Vtree::leaf(VarId(5)));
    for (f, want) in [(Tdd::one(&leaf), vec![0, 1]), (crate::literal(&leaf, -5).unwrap(), vec![0])] {
        assert_eq!(f.model_columns(&[&[VarId(5)]]).unwrap().to_columns(), vec![want]);
    }
}

#[test]
fn malformed_columns_are_refused() {
    let vtree = Arc::new(Vtree::balanced(40));
    let f = Tdd::clause(&vtree, [1, 2]).unwrap();
    assert_eq!(f.model_columns(&[&[VarId(41)]]).unwrap_err(), OperationError::VariableNotInVtree(VarId(41)));
    assert_eq!(f.model_columns(&[&[VarId(1)], &[VarId(2), VarId(1)]]).unwrap_err(), OperationError::DuplicateVariable(VarId(1)));
    let wide: Vec<VarId> = (1..=33).map(VarId).collect();
    assert_eq!(f.model_columns(&[&wide]).unwrap_err(), OperationError::ColumnTooWide { column: 0, bits: 33 });
    assert_eq!(f.model_columns(&[&[VarId(1)]]).unwrap_err(), OperationError::UnlistedLiteral(VarId(2)));
    // Thirty-two bits is the widest column.
    let full: Vec<VarId> = (1..=32).map(VarId).collect();
    let g = Tdd::clause(&vtree, [32]).unwrap();
    let codes = g.model_columns(&[&full]).unwrap().rows();
    assert_eq!(codes, 1u64 << 31);
}


/// The variables below vtree node `t`, left to right.
fn vars_below(vtree: &Vtree, t: VtreeIdx) -> Vec<VarId> {
    match *vtree.node(t) {
        VtreeNode::Leaf { var, .. } => vec![var],
        VtreeNode::Internal { left, right, .. } => {
            let mut vars = vars_below(vtree, left);
            vars.extend(vars_below(vtree, right));
            vars
        }
    }
}

#[test]
fn columns_on_whole_subtrees_match_projected_truth_tables() {
    // Columns that are exactly a vtree node's variables write whole words
    // (`emit`), with shared nodes' code lists kept; checked on minimized,
    // pruned and unminimized diagrams, whole and in batches.
    let mut rng = Lcg::new(23);
    for num_vars in [4u32, 6, 8, 10] {
        for (shape, vtree) in vtree_shapes(num_vars) {
            let root = vtree.root();
            let VtreeNode::Internal { left, right, .. } = *vtree.node(root) else { continue };
            let mut splits = vec![vec![vars_below(&vtree, left), vars_below(&vtree, right)]];
            if let VtreeNode::Internal { left: ll, right: lr, .. } = *vtree.node(left) {
                splits.push(vec![vars_below(&vtree, ll), vars_below(&vtree, lr), vars_below(&vtree, right)]);
            }
            for columns in splits {
                for round in 0..10 {
                    let clauses = rand_cnf(&mut rng, num_vars, CnfShape { clauses: 4, width: 3 });
                    let truth = truth_table(num_vars, &clauses);
                    let what = format!("{num_vars} vars, {shape}, round {round}, columns {columns:?}");
                    let f = compile_clauses(&vtree, &clauses);
                    check(&f, num_vars, &truth, &columns, &mut rng, &format!("{what}, canonical"));
                    let mut g = Tdd::one(&vtree);
                    for c in &clauses {
                        g = crate::and(g, Tdd::clause(&vtree, c.iter().copied()).unwrap()).unwrap();
                    }
                    check(&g, num_vars, &truth, &columns, &mut rng, &format!("{what}, conjoined"));
                    // One column dropped: the rest projected, minimized.
                    let kept = &columns[1..];
                    let dropped = columns[0].clone();
                    let h = vtree.context().run(|e| e.exists_vars(f.clone(), &dropped)).unwrap();
                    let kept_truth: Vec<bool> = truth.clone();
                    let refs: Vec<&[VarId]> = kept.iter().map(Vec::as_slice).collect();
                    let mut table = h.model_columns(&refs).unwrap();
                    let want = expected(num_vars, &kept_truth, kept);
                    let rows: BTreeSet<Vec<u32>> = rows_of(&table.to_columns()).into_iter().collect();
                    assert_eq!(table.rows(), want.len() as u64, "{what}, projected: rows");
                    assert_eq!(rows, want, "{what}, projected");
                }
            }
        }
    }
}

#[test]
fn write_zeroed_matches_write_on_zeroed_buffers() {
    let mut rng = Lcg::new(31);
    for (shape, vtree) in vtree_shapes(7) {
        let clauses = rand_cnf(&mut rng, 7, CnfShape { clauses: 4, width: 3 });
        let f = compile_clauses(&vtree, &clauses);
        let columns: Vec<Vec<VarId>> = vec![(1..=3).map(VarId).collect(), (4..=7).map(VarId).collect()];
        let refs: Vec<&[VarId]> = columns.iter().map(Vec::as_slice).collect();
        let mut table = f.model_columns(&refs).unwrap();
        let rows = table.rows();
        let mut dirty: Vec<Vec<u32>> = vec![vec![u32::MAX; rows as usize]; 2];
        let mut clean: Vec<Vec<u32>> = vec![vec![0; rows as usize]; 2];
        table.write(0..rows, &mut dirty.iter_mut().map(Vec::as_mut_slice).collect::<Vec<_>>());
        table.write_zeroed(0..rows, &mut clean.iter_mut().map(Vec::as_mut_slice).collect::<Vec<_>>());
        assert_eq!(dirty, clean, "{shape}");
    }
}

/// Whether `rows` strictly ascend, column by column.
fn strictly_ascending(rows: &[Vec<u32>]) -> bool {
    rows.windows(2).all(|w| w[0] < w[1])
}

#[test]
fn ascending_is_exact_on_lexicographic_layouts() {
    // Columns read off the leaves left to right, most significant bit
    // first: `ascending` says exactly whether the rows come out sorted,
    // before and after `sort_pairs`, which keeps the rows and never undoes
    // an ascending order. Any other layout reports `false` past one row.
    let mut rng = Lcg::new(41);
    let (mut cases, mut before, mut after) = (0, 0, 0);
    for num_vars in [2u32, 3, 5, 7, 9] {
        for (shape, vtree) in vtree_shapes(num_vars) {
            let leaves = vars_below(&vtree, vtree.root());
            for round in 0..12 {
                let clauses = rand_cnf(&mut rng, num_vars, CnfShape { clauses: 4, width: 3 });
                let truth = truth_table(num_vars, &clauses);
                let f = compile_clauses(&vtree, &clauses);
                let listed: Vec<VarId> = leaves.iter().copied().filter(|_| rng.below(4) != 0).collect();
                if listed.is_empty() {
                    continue;
                }
                // Consecutive runs of the listed leaves, one to three columns.
                let ncols = 1 + rng.below(listed.len().min(3) as u64) as usize;
                let mut cuts: Vec<usize> = (1..listed.len()).collect();
                for i in (1..cuts.len()).rev() {
                    cuts.swap(i, rng.below(i as u64 + 1) as usize);
                }
                let mut cuts: Vec<usize> = cuts.into_iter().take(ncols - 1).collect();
                cuts.sort_unstable();
                let mut columns = Vec::new();
                let mut from = 0;
                for &c in cuts.iter().chain([listed.len()].iter()) {
                    columns.push(listed[from..c].to_vec());
                    from = c;
                }
                let unlisted: Vec<VarId> = (1..=num_vars).map(VarId).filter(|v| !listed.contains(v)).collect();
                let what = format!("{num_vars} vars, {shape}, round {round}, columns {columns:?}");
                let minimized = vtree.context().run(|e| e.exists_vars(f.clone(), &unlisted)).unwrap();
                let pruned = vtree
                    .context()
                    .run(|e| e.exists_vars_with(f.clone(), &unlisted, ReductionPlan::Prune))
                    .unwrap();
                for (g, form) in [(&minimized, "minimized"), (&pruned, "pruned")] {
                    let refs: Vec<&[VarId]> = columns.iter().map(Vec::as_slice).collect();
                    let mut table = match g.model_columns(&refs) {
                        Ok(t) => t,
                        Err(OperationError::UnlistedLiteral(_)) => continue,
                        Err(e) => panic!("{what}, {form}: {e}"),
                    };
                    let rows = rows_of(&table.to_columns());
                    let sorted = strictly_ascending(&rows);
                    assert_eq!(table.ascending(), sorted, "{what}, {form}: before sorting pairs");
                    table.sort_pairs();
                    let whole = table.to_columns();
                    let resorted = rows_of(&whole);
                    let set: BTreeSet<Vec<u32>> = rows.iter().cloned().collect();
                    assert_eq!(resorted.iter().cloned().collect::<BTreeSet<_>>(), set, "{what}, {form}: rows changed");
                    assert_eq!(table.ascending(), strictly_ascending(&resorted), "{what}, {form}: after sorting pairs");
                    assert!(!sorted || table.ascending(), "{what}, {form}: sorting pairs undid the order");
                    // The batches still tile the renumbered table.
                    check(g, num_vars, &truth, &columns, &mut rng, &format!("{what}, {form}"));
                    let mut batched: Vec<Vec<u32>> = vec![Vec::new(); columns.len()];
                    let mut at = 0u64;
                    while at < table.rows() {
                        let end = (at + 1 + rng.below(5)).min(table.rows());
                        let mut bufs: Vec<Vec<u32>> = vec![vec![u32::MAX; (end - at) as usize]; columns.len()];
                        table.write(at..end, &mut bufs.iter_mut().map(Vec::as_mut_slice).collect::<Vec<_>>());
                        for (b, part) in batched.iter_mut().zip(bufs) {
                            b.extend(part);
                        }
                        at = end;
                    }
                    assert_eq!(batched, whole, "{what}, {form}: batches after sorting pairs");
                    cases += 1;
                    before += usize::from(sorted);
                    after += usize::from(table.ascending());
                    // The columns in another order, or one read least
                    // significant bit first, is not the layout.
                    let mut other = columns.clone();
                    if other.len() > 1 {
                        other.reverse();
                    } else if other[0].len() > 1 {
                        other[0].reverse();
                    } else {
                        continue;
                    }
                    let refs: Vec<&[VarId]> = other.iter().map(Vec::as_slice).collect();
                    let t = g.model_columns(&refs).unwrap();
                    assert_eq!(t.ascending(), t.rows() <= 1, "{what}, {form}: columns {other:?}");
                }
            }
        }
    }
    assert!(cases > 0 && after > before, "sorting pairs orders some tables: {before} -> {after} of {cases}");
}
