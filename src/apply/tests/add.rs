use std::sync::Arc;

use num_bigint::BigUint;

use super::*;
use crate::diagram::{Arithmetic, RationalWeights, Weights};
use crate::test_helpers::{assert_canonical, compile_clauses_on, node_value, rand_cnf, under, vtree_shapes, CnfShape, Lcg};
use crate::vtree::{Home, VarId};
use crate::{and, or};

/// The count `f` gives the assignment `bits` of variables `1..=n`, bit `i`
/// the value of variable `i + 1`; summed-out variables are read past.
fn value(f: &Tdd, n: u32, bits: u64) -> BigUint {
    if f.vtree().num_nodes() == 0 {
        return BigUint::from(u8::from(f.is_one()));
    }
    let assignment: Vec<bool> = (0..n).map(|i| bits >> i & 1 == 1).collect();
    node_value(f, f.output, &|_| false, &assignment)
}

/// Assert that `sum` counts every assignment as `f` and `g` together do, and
/// is canonical.
fn assert_sum(sum: &Tdd, f: &Tdd, g: &Tdd, n: u32, label: &str) {
    // A region the sum fills is dropped, its variables read as free.
    let dropped = width(f).max(width(g)) - width(sum);
    if dropped == 0 {
        assert_canonical(sum);
    }
    for bits in 0..1u64 << n {
        assert_eq!(value(sum, n, bits) << dropped, value(f, n, bits) + value(g, n, bits), "{label}: assignment {bits:b}");
    }
    assert_eq!(sum.model_count().unwrap(), f.model_count().unwrap() + g.model_count().unwrap(), "{label}: total");
}

/// The number of variables of the regions `d` records.
fn width(d: &Tdd) -> u32 {
    d.vtree().regions().map(|t| d.vtree().region_at(t).1.iter().map(|&(lo, hi)| hi - lo).sum::<u32>()).sum()
}

/// The regions `d` records, each by its anchor and extent.
fn recorded(d: &Tdd) -> Vec<impl PartialEq + std::fmt::Debug> {
    d.vtree().regions().map(|t| d.vtree().region_at(t)).collect()
}

/// `add` of `f` and `g` as it answers on projections of a home: their sum
/// where both record the same regions, a refusal naming a level where they
/// record different ones, and, where neither records one, their
/// disjunction or a refusal of their shared model. The sum, if any.
fn checked_add(eng: &Engine, f: &Tdd, g: &Tdd, n: u32, label: &str) -> Option<Tdd> {
    let out = eng.add(f.clone(), g.clone());
    if !f.is_zero() && !g.is_zero() {
        match (recorded(f), recorded(g)) {
            (rf, rg) if rf != rg => {
                assert!(matches!(out, Err(OperationError::MarginalLevel(_))), "{label}: {out:?}");
                return None;
            }
            (rf, _) if rf.is_empty() && matches!(out, Err(OperationError::SharedModel)) => {
                assert!(!eng.and(f.clone(), g.clone()).unwrap().is_zero(), "{label}: refused with no shared model");
                return None;
            }
            _ => {}
        }
    }
    let sum = out.unwrap_or_else(|e| panic!("{label}: {e}"));
    assert_sum(&sum, f, g, n, label);
    Some(sum)
}

/// The subtrees a test sums out on `vtree`: one internal subtree below the
/// root, that subtree and a leaf outside it, two disjoint internal subtrees,
/// or one leaf.
fn targets(rng: &mut Lcg, vtree: &Vtree) -> Vec<VtreeIdx> {
    let inner: Vec<VtreeIdx> = vtree.internal_bottomup().map(|(t, _, _)| t).filter(|&t| t != vtree.root()).collect();
    let leaves = |outside: Option<VtreeIdx>| -> Vec<VtreeIdx> {
        (0..vtree.num_nodes() as u32)
            .map(VtreeIdx)
            .filter(|&t| vtree.node(t).is_leaf() && outside.is_none_or(|s| !under(vtree, t, s)))
            .collect()
    };
    let pick = |rng: &mut Lcg, from: &[VtreeIdx]| from[rng.below(from.len() as u64) as usize];
    match (rng.below(4), inner.is_empty()) {
        (0, false) => vec![pick(rng, &inner)],
        (1, false) => {
            let summed = pick(rng, &inner);
            vec![summed, pick(rng, &leaves(Some(summed)))]
        }
        (2, false) => {
            let summed = pick(rng, &inner);
            let apart: Vec<VtreeIdx> = inner.iter().copied().filter(|&t| !under(vtree, t, summed) && !under(vtree, summed, t)).collect();
            match apart.is_empty() {
                true => vec![summed],
                false => vec![summed, pick(rng, &apart)],
            }
        }
        _ => vec![pick(rng, &leaves(None))],
    }
}

/// Compile two random formulas on `vtree`, marginalize `summed`, and minimize both.
fn random_marginal_pair(
    eng: &Engine,
    vtree: &Arc<Vtree>,
    rng: &mut Lcg,
    n: u32,
    shape: CnfShape,
    summed: &[VtreeIdx],
) -> (Tdd, Tdd) {
    let mut f = compile_clauses_on(eng, vtree, &rand_cnf(rng, n, shape));
    let mut g = compile_clauses_on(eng, vtree, &rand_cnf(rng, n, shape));
    for d in [&mut f, &mut g] {
        eng.marginalize(d, crate::test_helpers::summing(vtree, summed)).unwrap();
        eng.minimize(d).unwrap();
    }
    (f, g)
}

/// Random functions summed out over the same subtrees add to the sum of
/// their counts, on every vtree shape and assignment.
#[test]
fn adds_the_counts_of_functions_summed_out_alike() {
    let mut rng = Lcg::new(17);
    let mut added = 0;
    for round in 0..24 {
        let n = 3 + (round % 6) as u32;
        let shape = CnfShape { clauses: n as usize, width: 2 + (round % 3) };
        for (name, vtree) in vtree_shapes(n) {
            let eng = Engine::new();
            let summed = targets(&mut rng, &vtree);
            let (f, g) = random_marginal_pair(&eng, &vtree, &mut rng, n, shape, &summed);
            let label = format!("round {round}, {name}, summed {summed:?}");
            let Some(sum) = checked_add(&eng, &f, &g, n, &label) else { continue };
            if !f.is_zero() && !g.is_zero() && !recorded(&f).is_empty() {
                assert!(recorded(&sum).iter().all(|r| recorded(&f).contains(r)), "{label}");
                added += 1;
            }
        }
    }
    assert!(added > 0, "no pair with a region added");
}

/// A diagram added to itself doubles every count, the models of both
/// operands meeting everywhere.
#[test]
fn adds_a_diagram_to_itself() {
    let mut rng = Lcg::new(5);
    let mut added = 0;
    for n in 3..=6 {
        for (name, vtree) in vtree_shapes(n) {
            let eng = Engine::new();
            let mut f = compile_clauses_on(&eng, &vtree, &rand_cnf(&mut rng, n, CnfShape { clauses: n as usize, width: 3 }));
            eng.marginalize(&mut f, crate::test_helpers::summing(&vtree, &targets(&mut rng, &vtree))).unwrap();
            eng.minimize(&mut f).unwrap();
            if checked_add(&eng, &f, &f, n, name).is_some() && !recorded(&f).is_empty() {
                added += 1;
            }
        }
    }
    assert!(added > 0, "no diagram with a region added");
}

/// With no summed-out level, functions with no common model add to their
/// disjunction.
#[test]
fn boolean_operands_with_no_common_model_add_to_their_disjunction() {
    let mut rng = Lcg::new(23);
    for n in 2..=6 {
        for (name, vtree) in vtree_shapes(n) {
            let eng = Engine::new();
            let split = 1 + rng.below(u64::from(n)) as i32;
            let shape = CnfShape { clauses: n as usize, width: 3 };
            let f = and(compile_clauses_on(&eng, &vtree, &rand_cnf(&mut rng, n, shape)), Tdd::cube(&vtree, [split]).unwrap()).unwrap();
            let g = and(compile_clauses_on(&eng, &vtree, &rand_cnf(&mut rng, n, shape)), Tdd::cube(&vtree, [-split]).unwrap()).unwrap();
            let sum = eng.add(f.clone(), g.clone()).unwrap();
            assert!(!sum.has_marginal_level(), "{name}");
            assert!(sum.equivalent(&or(f.clone(), g.clone()).unwrap()).unwrap(), "{name}");
            assert_sum(&sum, &f, &g, n, name);
        }
    }
}

/// With no summed-out level a shared model would count twice, which no
/// structural diagram holds.
#[test]
fn boolean_operands_sharing_a_model_are_refused() {
    let mut rng = Lcg::new(29);
    for n in 2..=6 {
        for (name, vtree) in vtree_shapes(n) {
            let eng = Engine::new();
            let shape = CnfShape { clauses: n as usize, width: 3 };
            let f = compile_clauses_on(&eng, &vtree, &rand_cnf(&mut rng, n, shape));
            let g = compile_clauses_on(&eng, &vtree, &rand_cnf(&mut rng, n, shape));
            let shared = !eng.and(f.clone(), g.clone()).unwrap().is_zero();
            match eng.add(f.clone(), g.clone()) {
                Err(OperationError::SharedModel) => assert!(shared, "{name}: refused with no shared model"),
                Ok(sum) => {
                    assert!(!shared, "{name}: a shared model was summed");
                    assert_sum(&sum, &f, &g, n, name);
                }
                Err(e) => panic!("{name}: {e}"),
            }
        }
    }
    // On one variable the root is a leaf.
    let vtree = Home::new(Vtree::balanced(1));
    let x = || Tdd::cube(&vtree, [1]).unwrap();
    assert_eq!(add(x(), x()).unwrap_err(), OperationError::SharedModel);
    assert_eq!(add(x(), Tdd::one(&vtree)).unwrap_err(), OperationError::SharedModel);
    let both = add(x(), Tdd::cube(&vtree, [-1]).unwrap()).unwrap();
    assert!(both.equivalent(&Tdd::one(&vtree)).unwrap());
}

/// A level summed out in one operand only is refused, naming a level under
/// the subtree summed out.
#[test]
fn a_level_summed_out_on_one_side_only_is_refused() {
    let vtree = Home::new(Vtree::balanced(4));
    let (left, right) = vtree.children(vtree.root());
    let mut f = Tdd::clause(&vtree, [1, 2]).unwrap();
    let mut g = Tdd::clause(&vtree, [1, 3]).unwrap();
    f.marginalize(crate::test_helpers::summing(&vtree, &[left])).unwrap();
    match add(f.clone(), g.clone()) {
        Err(OperationError::MarginalLevel(t)) => assert!(under(&vtree, t, left)),
        other => panic!("expected a marginal level, got {other:?}"),
    }
    g.marginalize(crate::test_helpers::summing(&vtree, &[right])).unwrap();
    assert!(matches!(add(f, g), Err(OperationError::MarginalLevel(_))));
}

/// A false operand is the sum's identity, whatever levels the other summed.
#[test]
fn a_false_operand_returns_the_other() {
    let vtree = Home::new(Vtree::balanced(4));
    let (left, _) = vtree.children(vtree.root());
    let mut f = Tdd::clause(&vtree, [1, 3]).unwrap();
    f.marginalize(crate::test_helpers::summing(&vtree, &[left])).unwrap();
    for sum in [add(f.clone(), Tdd::zero(&vtree)).unwrap(), add(Tdd::zero(&vtree), f.clone()).unwrap()] {
        assert_sum(&sum, &f, &Tdd::zero(&vtree), 4, "false operand");
    }
}

/// Operands on different vtree allocations are refused.
#[test]
fn operands_on_different_vtrees_are_refused() {
    let a = Home::new(Vtree::balanced(3));
    let b = Home::new(Vtree::balanced(3));
    let err = add(Tdd::cube(&a, [1]).unwrap(), Tdd::cube(&b, [-1]).unwrap()).unwrap_err();
    assert_eq!(err, OperationError::VtreeMismatch);
}

/// Diagrams summed out to one value add their values.
#[test]
fn wholly_summed_out_diagrams_add_their_values() {
    let vtree = Home::new(Vtree::balanced(4));
    let root = vtree.root();
    let mut f = Tdd::clause(&vtree, [1, 2, 3, 4]).unwrap();
    let mut g = Tdd::cube(&vtree, [1, 2, 3, -4]).unwrap();
    f.marginalize(crate::test_helpers::summing(&vtree, &[root])).unwrap();
    g.marginalize(crate::test_helpers::summing(&vtree, &[root])).unwrap();
    let sum = add(f, g).unwrap();
    assert_eq!(sum.model_count().unwrap(), BigUint::from(15u32 + 1));
    assert_canonical(&sum);
}

/// Counts past `u128` are added exactly, `g`'s overflowed values held past
/// `f`'s.
#[test]
fn counts_past_u128_add_exactly() {
    let vtree = Home::new(Vtree::balanced(260));
    let (left, _) = vtree.children(vtree.root());
    let mut f = Tdd::clause(&vtree, 1..=130).unwrap();
    let mut g = Tdd::clause(&vtree, -130..=-1).unwrap();
    f.marginalize(crate::test_helpers::summing(&vtree, &[left])).unwrap();
    g.marginalize(crate::test_helpers::summing(&vtree, &[left])).unwrap();
    let two = BigUint::from(2u32);
    let free = two.pow(130);
    let sum = add(f.clone(), g).unwrap();
    assert_eq!(sum.model_count().unwrap(), (two.pow(131) - 2u32) * &free);
    assert_canonical(&sum);
    let twice = add(f.clone(), f).unwrap();
    assert_eq!(twice.model_count().unwrap(), (two.pow(131) - 2u32) * &free);
    assert_canonical(&twice);
}

/// Summed-out levels that hold weights are not added.
#[test]
fn weights_on_summed_out_levels_are_refused() {
    let weights = Weights::new(RationalWeights::unit(4), Arithmetic::ExactRational);
    let vtree = Home::weighted(Vtree::balanced(4), weights).unwrap();
    let (left, _) = vtree.children(vtree.root());
    let weighted = |mut f: Tdd| {
        f.marginalize(crate::test_helpers::summing(&vtree, &[left])).unwrap();
        f
    };
    let f = weighted(Tdd::clause(&vtree, [1, 2]).unwrap());
    let g = weighted(Tdd::cube(&vtree, [1, 2]).unwrap());
    assert_eq!(add(f, g).unwrap_err(), OperationError::IncompatibleWeights);
}

/// A sum conjoins with a function of its structural variables as any
/// summed-out diagram does: the counts the function keeps.
#[test]
fn a_sum_conjoins_with_a_filter_on_its_structural_variables() {
    let mut rng = Lcg::new(41);
    let n = 6;
    for (name, vtree) in vtree_shapes(n) {
        let eng = Engine::new();
        let summed = targets(&mut rng, &vtree);
        let shape = CnfShape { clauses: n as usize, width: 3 };
        let (f, g) = random_marginal_pair(&eng, &vtree, &mut rng, n, shape, &summed);
        let Some(sum) = checked_add(&eng, &f, &g, n, name) else { continue };
        let Some(v) = sum.vtree().leaf_bottomup().map(|(_, v)| v.0).min() else { continue };
        let kept = eng.and(sum.clone(), Tdd::cube(&vtree, [v as i32]).unwrap()).unwrap();
        for bits in 0..1u64 << n {
            let on = bits >> (v - 1) & 1 == 1;
            let expected = if on { value(&sum, n, bits) } else { BigUint::ZERO };
            assert_eq!(value(&kept, n, bits), expected, "{name}: x{v}, assignment {bits:b}");
        }
    }
}

/// A count level whose nodes outgrow the 32-bit offsets that list them is
/// refused with [`OperationError::IndexOverflow`], here under a pinned list
/// cap; the sum that goes through counts as both operands do. On a
/// right-linear vtree with its deepest internal subtree summed out, every
/// level between that subtree and the root is a count level.
#[test]
fn count_levels_past_the_list_cap_are_refused() {
    use crate::limits::Limits;
    let mut rng = Lcg::new(53);
    let n = 6;
    let vtree = Home::new(Vtree::linear(n));
    let (deepest, _, _) = vtree.internal_bottomup().next().unwrap();
    let eng = Engine::new();
    let shape = CnfShape { clauses: n as usize, width: 3 };
    let (f, g) = loop {
        let (f, g) = random_marginal_pair(&eng, &vtree, &mut rng, n, shape, &[deepest]);
        if !f.is_zero() && !g.is_zero() && recorded(&f) == recorded(&g) && !recorded(&f).is_empty() {
            break (f, g);
        }
    };
    let (sum, refused) = eng.limits().through_caps(Limits::pin_list_cap, || eng.add(f.clone(), g.clone()));
    assert!(refused > 0, "no cap refused the sum");
    assert_sum(&sum, &f, &g, n, "under the list caps");
}

/// A sum refused at each reserve in turn: each refusal answers
/// `OverBudget` with the operands as they were given, the engine adds
/// after it, and every granted attempt is the sum.
#[test]
fn every_refused_reserve_of_a_sum_leaves_the_operands() {
    let mut rng = Lcg::new(6);
    let mut refused = 0;
    for n in [4u32, 6] {
        for (name, vtree) in vtree_shapes(n).into_iter().take(3) {
            let eng = Engine::new();
            let mut f = compile_clauses_on(&eng, &vtree, &rand_cnf(&mut rng, n, CnfShape { clauses: n as usize, width: 3 }));
            eng.marginalize(&mut f, crate::test_helpers::summing(&vtree, &targets(&mut rng, &vtree))).unwrap();
            eng.minimize(&mut f).unwrap();
            if recorded(&f).is_empty() {
                continue;
            }
            let before = format!("{f:?}");
            refused += eng
                .sweep_reserves(
                    &crate::OperationError::OverBudget,
                    || {
                        let out = eng.add(f.clone(), f.clone());
                        assert_eq!(format!("{f:?}"), before, "{name}");
                        out
                    },
                    |sum, at| assert_sum(&sum, &f, &f, n, &format!("{name}, {at}")),
                )
                .refused;
        }
    }
    assert!(refused > 0, "no sum refused");
}

/// The values `f` gives the assignments of variables `1..=n`, times
/// `2^free`, agree with `root`'s, `f` on a projection and `root` on a copy of
/// its home's tree: `free` variables `root` summed out are free in `f`.
fn assert_reads_as(f: &Tdd, root: &Tdd, n: u32, free: u32, label: &str) {
    crate::test_helpers::check::check_all_fast(f, label);
    for bits in 0..1u64 << n {
        let reading = match f.vtree().num_nodes() {
            0 => BigUint::from(u8::from(f.is_one())),
            _ => value(f, n, bits),
        };
        assert_eq!(reading << free, value(root, n, bits), "{label}: assignment {bits:b}");
    }
}

/// On the projection of `((x1 x2) (x3 x4))` on `{x1, x3}`, `x1` summed out
/// of `x1 ∨ x3` is a region; the diagram added to itself counts as the sum
/// on the root does.
#[test]
fn a_diagram_with_a_region_adds_to_itself() {
    let eng = Engine::new();
    let home = Home::new(Vtree::balanced(4));
    let vtree = Home::new(Vtree::projection(&home, [VarId(1), VarId(3)]).unwrap());
    let mut f = Tdd::clause(&vtree, [1, 3]).unwrap();
    let root = Arc::new(Vtree::clone(&home));
    let mut r = crate::marginal::tests::region::on_home(&f, &root);
    eng.marginalize(&mut f, |v| v == VarId(1)).unwrap();
    assert_eq!(f.vtree().regions().count(), 1);
    crate::marginal::tests::marginalize_levels(&eng, &mut r, &[root.leaf_of(VarId(1)).unwrap()]).unwrap();
    let twice = eng.add(f.clone(), f).unwrap();
    assert_eq!(twice.vtree().regions().count(), 1);
    assert_reads_as(&twice, &eng.add(r.clone(), r).unwrap(), 4, 0, "x1 + x3, twice");
}

/// Two random CNFs on projections of a random home of 3 to 8 variables,
/// each with the variables under one home node summed out, a node whose
/// variables both use: where the two record the same regions, the sum reads
/// as the sum on the root; where they record different ones, for different
/// nodes or for one whose sum one operand no longer records, it is refused.
#[test]
fn operands_on_two_projections_add_where_they_summed_out_alike() {
    use crate::marginal::tests::region::{check_form, on_home, operand, under as within};
    let eng = Engine::new();
    let (mut added, mut refused, mut apart) = (0, 0, 0);
    for seed in 0..400u64 {
        let mut rng = Lcg::new(seed);
        let n = 3 + rng.below(6) as u32;
        let home = Home::new(Vtree::random(n, seed));
        let mut vars = || -> Vec<VarId> { (1..=n).filter(|_| rng.below(4) != 0).map(VarId).collect() };
        let (fv, gv) = (vars(), vars());
        let (f, g) = (operand(&eng, &home, &fv, &mut rng), operand(&eng, &home, &gv, &mut rng));
        let h: &Vtree = &home;
        let both: Vec<VtreeIdx> = h
            .bottomup()
            .filter(|&t| h.subtree(t).filter(|&u| h.node(u).is_leaf()).all(|u| [&f, &g].iter().all(|d| d.vtree().leaf_of(h.leaf_var(u)).is_some())))
            .collect();
        if both.is_empty() {
            continue;
        }
        let at = |rng: &mut Lcg| both[rng.below(both.len() as u64) as usize];
        let hf = at(&mut rng);
        let hg = if rng.below(4) == 0 { at(&mut rng) } else { hf };
        let what = format!("seed {seed}, {n} variables, {hf:?} and {hg:?}");
        let root = Arc::new(Vtree::clone(h));
        let (mut fr, mut gr) = (on_home(&f, &root), on_home(&g, &root));
        let (mut f, mut g) = (f, g);
        eng.marginalize(&mut f, within(&home, &[hf])).unwrap();
        eng.marginalize(&mut g, within(&home, &[hg])).unwrap();
        let recorded = |d: &Tdd| d.vtree().regions().map(|t| d.vtree().region_at(t)).collect::<Vec<_>>();
        let same = recorded(&f) == recorded(&g);
        let width = |d: &Tdd| d.vtree().regions().map(|t| d.vtree().region_at(t).1.iter().map(|&(lo, hi)| hi - lo).sum::<u32>()).sum::<u32>();
        let summed = width(&g);
        // Operands of which neither records a sum add as Boolean ones.
        if hf != hg && same || summed == 0 && width(&f) == 0 {
            continue;
        }
        let differ = !Arc::ptr_eq(f.vtree(), g.vtree());
        let sum = eng.add(f, g);
        if !same {
            assert!(matches!(sum, Err(OperationError::MarginalLevel(_))), "{what}: {sum:?}");
            refused += 1;
            continue;
        }
        let sum = sum.unwrap_or_else(|e| panic!("{what}: {e:?}"));
        check_form(&sum, &what);
        crate::marginal::tests::marginalize_levels(&eng, &mut fr, &[hf]).unwrap();
        crate::marginal::tests::marginalize_levels(&eng, &mut gr, &[hf]).unwrap();
        // A region the sum fills is dropped, its variables free.
        assert_reads_as(&sum, &eng.add(fr, gr).unwrap(), n, summed - width(&sum), &what);
        added += 1;
        apart += usize::from(differ);
    }
    assert!(apart > 100 && refused > 10,"{added} added, {apart} on two projections, {refused} refused");
}

/// A sum that fills a region of its operands (`2^k` everywhere for its `k`
/// variables) drops it only where no level above holds a multiple the
/// region carried: the sum is minimized, on every vtree shape.
#[test]
fn a_sum_that_drops_its_region_is_canonical() {
    use crate::marginal::tests::region::under as within;
    let width = |d: &Tdd| d.vtree().regions().map(|t| d.vtree().region_at(t).1.iter().map(|&(lo, hi)| hi - lo).sum::<u32>()).sum::<u32>();
    let (mut dropped, mut kept) = (0, 0);
    for seed in 0..30 {
        let mut rng = Lcg::new(seed);
        for n in 3..=7 {
            for (name, vtree) in vtree_shapes(n) {
                let eng = Engine::new();
                let home = Home::new(Vtree::clone(&vtree));
                let mut f = compile_clauses_on(&eng, &home, &rand_cnf(&mut rng, n, CnfShape { clauses: n as usize, width: 3 }));
                let hs = targets(&mut rng, &home);
                eng.marginalize(&mut f, within(&home, &hs)).unwrap();
                eng.minimize(&mut f).unwrap();
                let label = format!("seed {seed}, {n} variables, {name}, summed {hs:?}");
                let Ok(twice) = eng.add(f.clone(), f.clone()) else { continue };
                let gone = width(&f) - width(&twice);
                for bits in 0..1u64 << n {
                    assert_eq!(value(&twice, n, bits) << gone, value(&f, n, bits) << 1, "{label}: assignment {bits:b}");
                }
                crate::test_helpers::assert_minimized(&twice, &label);
                dropped += usize::from(gone > 0);
                kept += usize::from(gone == 0 && width(&f) > 0);
            }
        }
    }
    assert!(dropped > 100 && kept > 100, "{dropped} sums dropped a region, {kept} kept one");
}
