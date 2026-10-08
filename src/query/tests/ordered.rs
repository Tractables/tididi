//! `Tdd::ordered_models` against the models read off a truth table, sorted.

use std::collections::BTreeSet;
use std::sync::Arc;

use super::*;
use crate::reduce::ReductionPlan;
use crate::test_helpers::{compile_clauses, rand_cnf, truth_table, vtree_shapes, CnfShape, Lcg};
use crate::vtree::Vtree;

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

/// A row's key: its first `keys` codes, each negated where descending.
fn key(row: &[u32], keys: usize, descending: &[bool]) -> Vec<i64> {
    (0..keys).map(|j| if descending[j] { -i64::from(row[j]) } else { i64::from(row[j]) }).collect()
}

/// A random vtree whose leaves, left to right, are `order`.
fn random_over(order: &[VarId], rng: &mut Lcg) -> Vtree {
    if let [v] = order {
        return Vtree::leaf(*v);
    }
    let cut = 1 + rng.below(order.len() as u64 - 1) as usize;
    Vtree::join(&random_over(&order[..cut], rng), &random_over(&order[cut..], rng)).unwrap()
}

/// A random vtree that reads `keys` first, in order, then `others` in a
/// random order, with each of `unlisted` anywhere.
fn keys_first(keys: &[VarId], others: &[VarId], unlisted: &[VarId], rng: &mut Lcg) -> Arc<Vtree> {
    let mut order: Vec<VarId> = keys.to_vec();
    let mut rest = others.to_vec();
    for i in (1..rest.len()).rev() {
        rest.swap(i, rng.below(i as u64 + 1) as usize);
    }
    order.extend(rest);
    for &u in unlisted {
        let at = rng.below(order.len() as u64 + 1) as usize;
        order.insert(at, u);
    }
    Arc::new(random_over(&order, rng))
}

/// The key bits as a right spine over a random vtree of the other
/// variables.
fn spine_over(num_vars: u32, key_bits: &[VarId], seed: u64) -> Arc<Vtree> {
    let rest = Vtree::random(num_vars, seed).project_to_vars(|v| (!key_bits.contains(&v)).then_some(v), num_vars);
    let (mut t, skip) = match rest {
        Some(rest) => (rest, 0),
        None => (Vtree::leaf(*key_bits.last().unwrap()), 1),
    };
    for &k in key_bits.iter().rev().skip(skip) {
        t = Vtree::join(&Vtree::leaf(k), &t).unwrap();
    }
    Arc::new(t)
}

#[test]
fn the_first_models_are_the_least_on_a_key_spine() {
    // Random functions on a key spine over a random rest: the first
    // `limit` models are the `limit` least on the keys, in order, ties in
    // any order but members of the tie group, every row a model, none
    // twice; minimized and pruned, ascending and descending.
    let mut rng = Lcg::new(71);
    let mut cases = 0;
    for num_vars in [2u32, 3, 5, 7, 9] {
        for round in 0..40 {
            let mut vars: Vec<VarId> = (1..=num_vars).map(VarId).collect();
            for i in (1..vars.len()).rev() {
                vars.swap(i, rng.below(i as u64 + 1) as usize);
            }
            let nkey_bits = 1 + rng.below(u64::from(num_vars).min(5)) as usize;
            let key_bits: Vec<VarId> = vars[..nkey_bits].to_vec();
            let vtree = spine_over(num_vars, &key_bits, rng.below(1 << 20));
            // The key bits in one or two columns; the rest in up to two,
            // some unlisted.
            let split = if nkey_bits > 1 && rng.below(2) == 1 { 1 + rng.below(nkey_bits as u64 - 1) as usize } else { nkey_bits };
            let mut columns: Vec<Vec<VarId>> = vec![key_bits[..split].to_vec()];
            if split < nkey_bits {
                columns.push(key_bits[split..].to_vec());
            }
            let keys = columns.len();
            let rest: Vec<VarId> = vars[nkey_bits..].iter().copied().filter(|_| rng.below(4) != 0).collect();
            let cut = rng.below(rest.len() as u64 + 1) as usize;
            if cut > 0 {
                columns.push(rest[..cut].to_vec());
            }
            if cut < rest.len() {
                columns.push(rest[cut..].to_vec());
            }
            let descending: Vec<bool> = columns.iter().map(|_| rng.below(2) == 1).collect();
            let listed: BTreeSet<VarId> = columns.iter().flatten().copied().collect();
            let unlisted: Vec<VarId> = (1..=num_vars).map(VarId).filter(|v| !listed.contains(v)).collect();
            let clauses = rand_cnf(&mut rng, num_vars, CnfShape { clauses: 4, width: 3 });
            let truth = truth_table(num_vars, &clauses);
            let f = compile_clauses(&vtree, &clauses);
            let all = expected(num_vars, &truth, &columns);
            let mut sorted: Vec<Vec<u32>> = all.iter().cloned().collect();
            sorted.sort_by_key(|r| key(r, keys, &descending));
            let refs: Vec<&[VarId]> = columns.iter().map(Vec::as_slice).collect();
            let minimized = vtree.context().run(|e| e.exists_vars(f.clone(), &unlisted)).unwrap();
            let pruned = vtree.context().run(|e| e.exists_vars_with(f.clone(), &unlisted, ReductionPlan::Prune)).unwrap();
            for (g, form) in [(&minimized, "minimized"), (&pruned, "pruned")] {
                for limit in [0u64, 1, 2, 5, all.len() as u64, all.len() as u64 + 3] {
                    let what = format!("{num_vars} vars, round {round}, {form}, columns {columns:?}, descending {descending:?}, limit {limit}");
                    let got = match g.ordered_models(&refs, &descending, keys, limit) {
                        Ok(Some(c)) => c,
                        Ok(None) => panic!("{what}: a spine refused"),
                        Err(OperationError::UnlistedLiteral(v)) => {
                            assert!(form == "pruned" && !listed.contains(&v), "{what}");
                            continue;
                        }
                        Err(e) => panic!("{what}: {e}"),
                    };
                    let n = got.first().map_or(0, Vec::len);
                    let k = (limit as usize).min(all.len());
                    assert_eq!(n, k, "{what}: rows");
                    let rows: Vec<Vec<u32>> = (0..n).map(|r| got.iter().map(|c| c[r]).collect()).collect();
                    let distinct: BTreeSet<Vec<u32>> = rows.iter().cloned().collect();
                    assert_eq!(distinct.len(), n, "{what}: a row repeats");
                    assert!(distinct.is_subset(&all), "{what}: not a model");
                    for i in 0..n {
                        assert_eq!(key(&rows[i], keys, &descending), key(&sorted[i], keys, &descending), "{what}: row {i}; got {rows:?} want {sorted:?} vtree {}", vtree.to_text());
                    }
                    cases += 1;
                }
            }
            // The layout refused: a key bit off the spine.
            if num_vars >= 3 && !rest.is_empty() {
                let other = [vec![rest[0]], key_bits.clone()];
                let refs: Vec<&[VarId]> = other.iter().map(Vec::as_slice).collect();
                let full = vtree.context().run(|e| e.exists_vars(f.clone(), &(1..=num_vars).map(VarId).filter(|v| !other.iter().flatten().any(|w| w == v)).collect::<Vec<_>>())).unwrap();
                assert_eq!(full.ordered_models(&refs, &[false, false], 1, 3).unwrap(), None, "{num_vars} vars, round {round}");
            }
        }
    }
    assert!(cases > 500, "{cases} cases");
}

#[test]
fn no_key_columns_is_any_order_on_any_vtree() {
    // With no key, every vtree is a spine of nothing: the first models are
    // any `limit` distinct models.
    let mut rng = Lcg::new(73);
    for (shape, vtree) in vtree_shapes(6) {
        let clauses = rand_cnf(&mut rng, 6, CnfShape { clauses: 4, width: 3 });
        let truth = truth_table(6, &clauses);
        let f = compile_clauses(&vtree, &clauses);
        let columns: Vec<Vec<VarId>> = vec![(1..=3).map(VarId).collect(), (4..=6).map(VarId).collect()];
        let all = expected(6, &truth, &columns);
        let refs: Vec<&[VarId]> = columns.iter().map(Vec::as_slice).collect();
        let got = f.ordered_models(&refs, &[false, true], 0, 1000).unwrap().unwrap();
        let rows: BTreeSet<Vec<u32>> = (0..got[0].len()).map(|r| vec![got[0][r], got[1][r]]).collect();
        assert_eq!(rows, all, "{shape}");
        assert_eq!(got[0].len(), all.len(), "{shape}: a row repeats");
    }
}

#[test]
fn constants_and_refusals() {
    let vtree = Arc::new(Vtree::linear(3));
    let key: &[VarId] = &[VarId(1), VarId(2)];
    assert_eq!(Tdd::zero(&vtree).ordered_models(&[key, &[VarId(3)]], &[false, false], 1, 5).unwrap(), Some(vec![vec![], vec![]]));
    let one = Tdd::one(&vtree).ordered_models(&[key, &[VarId(3)]], &[true, false], 1, 5).unwrap().unwrap();
    assert_eq!(one, vec![vec![3, 3, 2, 2, 1], vec![0, 1, 0, 1, 0]]);
    // A key read least significant bit first is not the spine.
    assert_eq!(Tdd::one(&vtree).ordered_models(&[&[VarId(2), VarId(1)]], &[false], 1, 5).unwrap(), None);
    let f = Tdd::clause(&vtree, [1, 3]).unwrap();
    assert_eq!(f.ordered_models(&[key], &[false], 1, 5).unwrap_err(), OperationError::UnlistedLiteral(VarId(3)));
}

#[test]
fn the_first_keys_are_the_least_distinct_values_on_a_key_spine() {
    // Random functions on a key spine over a random rest that no column
    // lists: the first `limit` values the keys take, distinct and in
    // order, both directions, before and after the rest is projected away.
    let mut rng = Lcg::new(29);
    let mut cases = 0;
    for num_vars in [1u32, 2, 4, 6, 8] {
        for round in 0..40 {
            let mut vars: Vec<VarId> = (1..=num_vars).map(VarId).collect();
            for i in (1..vars.len()).rev() {
                vars.swap(i, rng.below(i as u64 + 1) as usize);
            }
            let nkey_bits = 1 + rng.below(u64::from(num_vars).min(5)) as usize;
            let key_bits: Vec<VarId> = vars[..nkey_bits].to_vec();
            let vtree = spine_over(num_vars, &key_bits, rng.below(1 << 20));
            let split = if nkey_bits > 1 && rng.below(2) == 1 { 1 + rng.below(nkey_bits as u64 - 1) as usize } else { nkey_bits };
            let mut columns: Vec<Vec<VarId>> = vec![key_bits[..split].to_vec()];
            if split < nkey_bits {
                columns.push(key_bits[split..].to_vec());
            }
            let descending: Vec<bool> = columns.iter().map(|_| rng.below(2) == 1).collect();
            let clauses = rand_cnf(&mut rng, num_vars, CnfShape { clauses: 3, width: 3 });
            let truth = truth_table(num_vars, &clauses);
            let f = compile_clauses(&vtree, &clauses);
            let mut sorted: Vec<Vec<u32>> = expected(num_vars, &truth, &columns).into_iter().collect();
            sorted.sort_by_key(|r| key(r, columns.len(), &descending));
            let refs: Vec<&[VarId]> = columns.iter().map(Vec::as_slice).collect();
            let rest: Vec<VarId> = vars[nkey_bits..].to_vec();
            let projected = vtree.context().run(|e| e.exists_vars(f.clone(), &rest)).unwrap();
            for (g, form) in [(&f, "whole"), (&projected, "projected")] {
                for limit in [0u64, 1, 2, 3, sorted.len() as u64, sorted.len() as u64 + 2] {
                    let what = format!("{num_vars} vars, round {round}, {form}, columns {columns:?}, descending {descending:?}, limit {limit}");
                    let got = g.ordered_keys(&refs, &descending, limit).unwrap().unwrap_or_else(|| panic!("{what}: a spine refused"));
                    let n = got.first().map_or(0, Vec::len);
                    let rows: Vec<Vec<u32>> = (0..n).map(|r| got.iter().map(|c| c[r]).collect()).collect();
                    let k = (limit as usize).min(sorted.len());
                    assert_eq!(rows, sorted[..k].to_vec(), "{what}: vtree {}", vtree.to_text());
                    cases += 1;
                }
            }
        }
    }
    assert!(cases > 500, "{cases} cases");
    // Keys read out of order: refused.
    let vtree = Arc::new(Vtree::balanced(4));
    assert_eq!(Tdd::one(&vtree).ordered_keys(&[&[VarId(2)], &[VarId(1)]], &[false, false], 2).unwrap(), None);
}

#[test]
fn the_first_rows_are_the_least_on_any_layout_that_reads_the_keys_first() {
    // Random functions on random vtrees that read the keys first, in
    // order, every other listed variable after them and unlisted ones
    // anywhere: `ordered_models` (the unlisted projected first) and
    // `ordered_keys` (projecting them itself) against the sorted truth table.
    let mut rng = Lcg::new(53);
    let mut cases = 0;
    for num_vars in [2u32, 3, 5, 7, 9] {
        for round in 0..60 {
            let mut vars: Vec<VarId> = (1..=num_vars).map(VarId).collect();
            for i in (1..vars.len()).rev() {
                vars.swap(i, rng.below(i as u64 + 1) as usize);
            }
            let nkey_bits = 1 + rng.below(u64::from(num_vars).min(6)) as usize;
            let key_bits: Vec<VarId> = vars[..nkey_bits].to_vec();
            let split = if nkey_bits > 1 && rng.below(2) == 1 { 1 + rng.below(nkey_bits as u64 - 1) as usize } else { nkey_bits };
            let mut columns: Vec<Vec<VarId>> = vec![key_bits[..split].to_vec()];
            if split < nkey_bits {
                columns.push(key_bits[split..].to_vec());
            }
            let keys = columns.len();
            let (mut others, mut unlisted) = (Vec::new(), Vec::new());
            for &v in &vars[nkey_bits..] {
                if rng.below(3) == 0 { unlisted.push(v) } else { others.push(v) }
            }
            if !others.is_empty() {
                columns.push(others.clone());
            }
            let descending: Vec<bool> = columns.iter().map(|_| rng.below(2) == 1).collect();
            let vtree = keys_first(&key_bits, &others, &unlisted, &mut rng);
            let clauses = rand_cnf(&mut rng, num_vars, CnfShape { clauses: 4, width: 3 });
            let truth = truth_table(num_vars, &clauses);
            let f = compile_clauses(&vtree, &clauses);
            let refs: Vec<&[VarId]> = columns.iter().map(Vec::as_slice).collect();
            let all = expected(num_vars, &truth, &columns);
            let mut sorted: Vec<Vec<u32>> = all.iter().cloned().collect();
            sorted.sort_by_key(|r| key(r, keys, &descending));
            let minimized = vtree.context().run(|e| e.exists_vars(f.clone(), &unlisted)).unwrap();
            let pruned = vtree.context().run(|e| e.exists_vars_with(f.clone(), &unlisted, ReductionPlan::Prune)).unwrap();
            let key_columns: Vec<Vec<VarId>> = columns[..keys].to_vec();
            let mut values: Vec<Vec<u32>> = expected(num_vars, &truth, &key_columns).into_iter().collect();
            values.sort_by_key(|r| key(r, keys, &descending));
            let key_refs: Vec<&[VarId]> = key_columns.iter().map(Vec::as_slice).collect();
            for limit in [0u64, 1, 2, 5, all.len() as u64, all.len() as u64 + 3] {
                let what = format!("{num_vars} vars, round {round}, columns {columns:?}, unlisted {unlisted:?}, descending {descending:?}, limit {limit}, vtree {}", vtree.to_text());
                for (g, form) in [(&minimized, "minimized"), (&pruned, "pruned")] {
                    let got = match g.ordered_models(&refs, &descending, keys, limit) {
                        Ok(Some(c)) => c,
                        Ok(None) => panic!("{what}: {form}: the layout refused"),
                        Err(OperationError::UnlistedLiteral(v)) => {
                            assert!(form == "pruned" && unlisted.contains(&v), "{what}");
                            continue;
                        }
                        Err(e) => panic!("{what}: {e}"),
                    };
                    let n = got.first().map_or(0, Vec::len);
                    assert_eq!(n, (limit as usize).min(all.len()), "{what}: {form}: rows");
                    let rows: Vec<Vec<u32>> = (0..n).map(|r| got.iter().map(|c| c[r]).collect()).collect();
                    let distinct: BTreeSet<Vec<u32>> = rows.iter().cloned().collect();
                    assert_eq!(distinct.len(), n, "{what}: {form}: a row repeats");
                    assert!(distinct.is_subset(&all), "{what}: {form}: not a model");
                    for i in 0..n {
                        assert_eq!(key(&rows[i], keys, &descending), key(&sorted[i], keys, &descending), "{what}: {form}: row {i}; got {rows:?} want {sorted:?}");
                    }
                    cases += 1;
                }
                for (g, form) in [(&f, "whole"), (&minimized, "projected")] {
                    let got = g.ordered_keys(&key_refs, &descending[..keys], limit).unwrap().unwrap_or_else(|| panic!("{what}: {form}: keys refused"));
                    let n = got.first().map_or(0, Vec::len);
                    let rows: Vec<Vec<u32>> = (0..n).map(|r| got.iter().map(|c| c[r]).collect()).collect();
                    assert_eq!(rows, values[..(limit as usize).min(values.len())].to_vec(), "{what}: {form}: keys");
                    cases += 1;
                }
            }
        }
    }
    assert!(cases > 2000, "{cases} cases");
}

#[test]
fn ordered_queries_accept_128_key_bits_and_decline_more() {
    let vtree = Arc::new(Vtree::linear(129));
    let f = Tdd::one(&vtree);
    crate::test_helpers::assert_canonical(&f);
    let bits: Vec<VarId> = (1..=129).map(VarId).collect();
    let columns: Vec<&[VarId]> = bits.chunks(32).collect();
    let directions = [false; 5];
    let keys = &columns[..4];
    assert_eq!(f.ordered_keys(keys, &directions[..4], 1).unwrap(), Some(vec![vec![0]; 4]));
    assert_eq!(f.ordered_models(&columns, &directions, 4, 1).unwrap(), Some(vec![vec![0]; 5]));
    assert_eq!(f.ordered_keys(&columns, &directions, 1).unwrap(), None);
    assert_eq!(f.ordered_models(&columns, &directions, 5, 1).unwrap(), None);
}
