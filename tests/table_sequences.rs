//! Table operations checked against an independent set of bit assignments.
use std::collections::BTreeSet;
use std::sync::Arc;

use tididi::io::{read_tdd, write_tdd};
use tididi::test_helpers::{assert_canonical, eval, Lcg};
use tididi::vtree::VarId;
use tididi::{nor_many, or_many, Engine, Literal, Tdd, Vtree};

/// Compare every assignment, not just the number of models.
fn check(f: &Tdd, vars: &[VarId], rows: &BTreeSet<u64>) {
    assert_canonical(f);
    assert_eq!(f.model_count().unwrap(), (rows.len() as u64).into());
    let mut assignment = vec![false; f.vtree().num_vars() as usize];
    for row in 0..1u64 << vars.len() {
        for (bit, var) in vars.iter().enumerate() { assignment[var.idx()] = row & (1 << bit) != 0; }
        assert_eq!(eval(f, &assignment), rows.contains(&row), "row {row}");
    }
}

/// Complete input in the table's column order, including sparse variable IDs.
fn literals(vars: &[VarId], row: u64) -> Vec<Literal> {
    vars.iter().enumerate().map(|(bit, &var)| Literal::new(var, row & (1 << bit) != 0)).collect()
}

#[test]
fn table_update_project_rename_combine_and_reload_sequence() {
    let vars = [VarId(9), VarId(2), VarId(14), VarId(5), VarId(7), VarId(11)];
    let vtree = Arc::new(Vtree::balanced_over(&vars).unwrap());
    let mut rng = Lcg::new(0x41ac_98ee_5532);
    let mut rows: BTreeSet<u64> = (0..18).map(|_| rng.next_u64() & 63).collect();
    let mut f = Tdd::from_models(&vtree, &vars, &rows.iter().copied().collect::<Vec<_>>()).unwrap();
    check(&f, &vars, &rows);
    let engine = Engine::new();
    for round in 0..20 {
        {
            let mut batch = engine.maintain(&mut f).unwrap();
            for _ in 0..5 {
                let row = rng.next_u64() & 63;
                if rng.next_u64() & 1 == 0 {
                    batch.insert_model(literals(&vars, row)).unwrap();
                    rows.insert(row);
                } else {
                    batch.remove_model(literals(&vars, row)).unwrap();
                    rows.remove(&row);
                }
            }
        }
        f.minimize().unwrap();
        check(&f, &vars, &rows);

        let bit = round % vars.len();
        f = f.exists_vars(&[vars[bit], vars[bit]]).unwrap();
        rows = rows.iter().flat_map(|&row| [row & !(1 << bit), row | (1 << bit)]).collect();
        check(&f, &vars, &rows);

        f = f.rename_vars(&[(vars[0], vars[4]), (vars[4], vars[0])]).unwrap();
        rows = rows.iter().map(|&r| (r & !17) | ((r & 1) << 4) | ((r & 16) >> 4)).collect();
        check(&f, &vars, &rows);

        let other: BTreeSet<u64> = (0..64).filter(|_| rng.next_u64() & 3 == 0).collect();
        let g = Tdd::from_models(&vtree, &vars, &other.iter().copied().collect::<Vec<_>>()).unwrap();
        assert_canonical(&g);
        if round % 3 == 0 {
            f = nor_many([f, g]).unwrap();
            rows = (0..64).filter(|r| !rows.contains(r) && !other.contains(r)).collect();
        } else if round % 3 == 1 {
            f = or_many([f, g]).unwrap();
            rows.extend(other);
        } else {
            f = engine.and(f, g).unwrap();
            rows = rows.intersection(&other).copied().collect();
        }
        check(&f, &vars, &rows);
        let mut text = Vec::new();
        write_tdd(&mut text, &f).unwrap();
        f = read_tdd(&mut text.as_slice(), &vtree).unwrap();
        check(&f, &vars, &rows);
    }
}

#[test]
fn row_construction_crosses_word_and_sort_boundaries() {
    let mut rng = Lcg::new(0x1234_5678_abcd);
    for width in [16, 17, 63, 64, 65] {
        let vars: Vec<_> = (1..=width).map(VarId).collect();
        let vtree = Arc::new(Vtree::balanced(width));
        let words = (width as usize).div_ceil(64);
        let mut rows = BTreeSet::new();
        for _ in 0..40 {
            let mut row: Vec<_> = (0..words).map(|_| rng.next_u64()).collect();
            if width % 64 != 0 { row[words - 1] &= (1 << (width % 64)) - 1; }
            rows.insert(row);
        }
        let packed: Vec<_> = rows.iter().flatten().copied().collect();
        let f = Tdd::from_models(&vtree, &vars, &packed).unwrap();
        assert_canonical(&f);
        assert_eq!(f.model_count().unwrap(), (rows.len() as u64).into());
        for row in rows {
            let assignment: Vec<_> = (0..width as usize).map(|bit| row[bit / 64] & (1 << (bit % 64)) != 0).collect();
            assert!(eval(&f, &assignment));
        }
    }
    let vars: Vec<_> = (1..=15).map(VarId).collect();
    let vtree = Arc::new(Vtree::balanced(15));
    for len in [16_383, 16_384, 16_385] {
        let rows: Vec<u64> = (0..len).rev().collect();
        let f = Tdd::from_models(&vtree, &vars, &rows).unwrap();
        assert_canonical(&f);
        assert_eq!(f.model_count().unwrap(), len.into());
        for row in [0, len - 1, len, 32_767] {
            let assignment: Vec<_> = (0..15).map(|bit| row & (1 << bit) != 0).collect();
            assert_eq!(eval(&f, &assignment), row < len);
        }
    }
}

#[test]
fn invariant_checkers_are_active_in_release_integration_tests() {
    use tididi::diagram::{ChildPair, NEG_LEAF_IDX, POS_LEAF_IDX, TddNodeId};
    let vtree = Arc::new(Vtree::balanced(2));
    let eng = Engine::new();
    let mut builder = Tdd::builder(&eng, &vtree).unwrap();
    let root = vtree.root();
    let output = builder.push(&eng, root, &[ChildPair::new(POS_LEAF_IDX, NEG_LEAF_IDX)]).unwrap();
    builder.push(&eng, root, &[ChildPair::new(POS_LEAF_IDX, NEG_LEAF_IDX)]).unwrap();
    let f = builder.finish(TddNodeId { vtree: root, local: output }).unwrap();
    // Storage-valid but deliberately noncanonical: two nodes have identical content.
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| assert_canonical(&f))).is_err());
    let valid = Tdd::cube(&vtree, [1, -2]).unwrap();
    assert_canonical(&valid);
}
