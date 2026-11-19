//! `trim` on diagrams with regions: a leaf every pair names true goes, also
//! where the chain above it reaches a region and its nodes stand for sums of
//! that region's values. Checked against a pointwise evaluation of the
//! untrimmed diagram on homes of at most ten variables.

use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{ToPrimitive, Zero};

use crate::diagram::{Arithmetic, ChildDecoder, ChildPair, EncodedChildRef, TddBuilder, TddLevel, ValueRef, Weights, WeightValue, ONE_LEAF_IDX, POS_LEAF_IDX};
use crate::marginal::tests::region::{operand, reweighted, under, weights};
use crate::test_helpers::check::check_trimmed;
use crate::test_helpers::deadline_probe;
use crate::value::CountRead;
use crate::vtree::rng::Lcg;
use crate::vtree::Home;
use crate::OperationError;

use super::*;

fn rational(w: &WeightValue) -> BigRational {
    match w.as_scaled() {
        Some(s) => BigRational::from_float(s.to_f64()).expect("a finite weight"),
        None => w.as_rational().into_owned(),
    }
}

/// The value of `f` where its variable leaves take `x`: a node is the sum
/// over its pairs of the product of what they name, a leaf named true is 1,
/// a region reads its value.
fn value_at(f: &Tdd, x: &dyn Fn(VarId) -> bool) -> BigRational {
    let v = f.vtree();
    if v.num_nodes() == 0 {
        return BigRational::from_integer(BigInt::from(u8::from(f.is_one())));
    }
    let mut cols: Vec<Vec<BigRational>> = vec![Vec::new(); v.num_nodes()];
    let read = |cols: &[Vec<BigRational>], c: VtreeIdx, side: EncodedChildRef| -> BigRational {
        if !f.level(c).is_marginal() {
            return cols[c.idx()][side.raw() as usize].clone();
        }
        if side.is_reserved() {
            return BigRational::zero();
        }
        match ChildDecoder::marginal().value(side) {
            ValueRef::Inline(n) => BigRational::from_integer(BigInt::from(n)),
            ValueRef::Slot(s) => cols[c.idx()][s as usize].clone(),
        }
    };
    for t in v.bottomup() {
        let level = f.level(t);
        cols[t.idx()] = if level.is_marginal() {
            match level.count_column() {
                Some(c) => (0..c.len())
                    .map(|i| match c.get(i) {
                        CountRead::Fast(n) => BigRational::from_integer(BigInt::from(n)),
                        CountRead::Big(n) => BigRational::from_integer(BigInt::from(n.clone())),
                    })
                    .collect(),
                None => level.weight_values().expect("a summed level holds weights").iter().map(rational).collect(),
            }
        } else if v.node(t).is_leaf() {
            let on = x(v.leaf_var(t));
            let mut c = vec![BigRational::zero(); 3];
            c[POS_LEAF_IDX.idx()] = BigRational::from_integer(BigInt::from(u8::from(on)));
            c[crate::diagram::NEG_LEAF_IDX.idx()] = BigRational::from_integer(BigInt::from(u8::from(!on)));
            c[ONE_LEAF_IDX.idx()] = BigRational::from_integer(BigInt::from(1));
            c
        } else {
            let (l, r) = v.children(t);
            (0..level.slot_count())
                .map(|j| level.pairs_iter_of_idx(j).map(|p| read(&cols, l, p.left) * read(&cols, r, p.right)).sum())
                .collect()
        };
    }
    cols[v.root().idx()][f.output().local.idx()].clone()
}

fn variables(f: &Tdd) -> Vec<VarId> {
    let v = f.vtree();
    v.bottomup().filter(|&t| v.node(t).is_var_leaf()).map(|t| v.leaf_var(t)).collect()
}

/// The value of `f` where the variables `vars` take `x`, summed over its
/// other variable leaves and counting each home variable it does not hold,
/// as a leaf or in a region, twice.
fn value_over(f: &Tdd, n: u32, vars: &[VarId], x: u32) -> BigRational {
    let v = f.vtree();
    let summed: u32 = v.regions().map(|t| v.region_at(t).1.iter().map(|&(lo, hi)| hi - lo).sum::<u32>()).sum();
    let others: Vec<VarId> = variables(f).into_iter().filter(|w| !vars.contains(w)).collect();
    let free = n - variables(f).len() as u32 - summed;
    let mut total = BigRational::zero();
    for y in 0..1u32 << others.len() {
        let at = |var: VarId| match vars.iter().position(|&w| w == var) {
            Some(i) => x >> i & 1 == 1,
            None => others.iter().position(|&w| w == var).is_some_and(|i| y >> i & 1 == 1),
        };
        total += value_at(f, &at);
    }
    total * BigRational::from_integer(BigInt::from(1u64 << free))
}

/// `f` and `g`, `g`'s variable leaves a subset of `f`'s, agree at every
/// assignment of `g`'s variable leaves, the others summed; exactly or, in
/// floats, to rounding.
fn same_pointwise(f: &Tdd, g: &Tdd, n: u32, floats: bool, what: &str) {
    let vars = variables(g);
    for bits in 0..1u32 << vars.len() {
        let (a, b) = (value_over(f, n, &vars, bits), value_over(g, n, &vars, bits));
        if floats {
            let (a, b) = (a.to_f64().unwrap(), b.to_f64().unwrap());
            assert!((a - b).abs() <= 1e-9 * a.abs().max(b.abs()), "{what}: at {bits:b} {a} against {b}");
        } else {
            assert_eq!(a, b, "{what}: at {bits:b}");
        }
    }
}

/// `f` conditioned on each leaf of `leaves` taking its literal, as levels for
/// a builder, each leaf left named true: at its parent a pair naming the
/// literal names the leaf true and a pair naming the other literal goes. A
/// node that held the literal in two pairs stands for the sum of what they
/// named, the pairs' products being disjoint. `None` when a node is left
/// with no pair or two nodes of a level become equal, which no operation
/// leaves and the trim's chain merge does not read.
fn named_true(f: &Tdd, leaves: &[(VtreeIdx, u32)]) -> Option<(TddBuilder, crate::diagram::TddNodeId)> {
    let eng = Engine::new();
    let v = Arc::clone(f.vtree());
    let mut levels = f.clone().levels.into_vec();
    for &(x, lit) in leaves {
        let p = v.node(x).parent().expect("a leaf under a node");
        let left = v.children(p).0 == x;
        let old = &levels[p.idx()];
        let mut level = TddLevel::new();
        for j in 0..old.slot_count() {
            let side = |q: &ChildPair| if left { q.left } else { q.right }.raw();
            let pairs: Vec<ChildPair> = old
                .pairs_iter_of_idx(j)
                .filter(|q| [lit, ONE_LEAF_IDX.0].contains(&side(q)))
                .map(|q| match left {
                    true => ChildPair::new(ONE_LEAF_IDX, q.right),
                    false => ChildPair::new(q.left, ONE_LEAF_IDX),
                })
                .collect();
            if pairs.is_empty() {
                return None;
            }
            level.push_node(eng.limits(), &pairs).unwrap();
        }
        let mut nodes: Vec<Vec<(u32, u32)>> = (0..level.slot_count())
            .map(|j| level.pairs_iter_of_idx(j).map(|q| (q.left.raw(), q.right.raw())).collect())
            .collect();
        nodes.iter_mut().for_each(|n| n.sort_unstable());
        nodes.sort_unstable();
        if nodes.windows(2).any(|w| w[0] == w[1]) {
            return None;
        }
        levels[p.idx()] = level;
    }
    Some((TddBuilder::from_levels(v, levels), f.output()))
}

/// One or two variable leaves of `f` at random, no two siblings: a node over
/// two leaves both named true would stand for the number of its pairs, which
/// no Boolean diagram holds.
/// Each with the literal it is conditioned on: one whose every node at the
/// parent holds a pair it names.
fn pick_leaves(f: &Tdd, rng: &mut Lcg) -> Vec<(VtreeIdx, u32)> {
    let v = f.vtree();
    let mut leaves: Vec<VtreeIdx> = v.bottomup().filter(|&t| v.node(t).is_var_leaf() && v.node(t).parent().is_some()).collect();
    let mut picked: Vec<(VtreeIdx, u32)> = Vec::new();
    for _ in 0..1 + rng.below(2) {
        if leaves.is_empty() {
            break;
        }
        let x = leaves.swap_remove(rng.below(leaves.len() as u64) as usize);
        let p = v.node(x).parent().unwrap();
        let left = v.children(p).0 == x;
        let level = f.level(p);
        let held = |lit: u32| {
            (0..level.slot_count()).all(|j| level.pairs_iter_of_idx(j).any(|q| [lit, ONE_LEAF_IDX.0].contains(&if left { q.left } else { q.right }.raw())))
        };
        let lits: Vec<u32> = [POS_LEAF_IDX.0, crate::diagram::NEG_LEAF_IDX.0].into_iter().filter(|&lit| held(lit)).collect();
        if !lits.is_empty() && picked.iter().all(|&(y, _)| v.node(y).parent() != Some(p)) {
            picked.push((x, lits[rng.below(lits.len() as u64) as usize]));
        }
    }
    picked
}

/// Whether a node at the parent of a leaf of `leaves` whose sibling is a
/// region holds two pairs: the trim must then add values.
fn sums_over_a_region(f: &Tdd, leaves: &[(VtreeIdx, u32)]) -> bool {
    let v = f.vtree();
    leaves.iter().any(|&(x, _)| {
        let p = v.node(x).parent().unwrap();
        let (l, r) = v.children(p);
        let sibling = if l == x { r } else { l };
        v.node(sibling).is_region() && (0..f.level(p).slot_count()).any(|j| f.level(p).pairs_iter_of_idx(j).len() > 1)
    })
}

/// A diagram with a region on a random home of at most ten variables, and
/// the arithmetic of its values.
fn with_region(case: u64, rng: &mut Lcg) -> Option<(Home, Tdd, Option<Arithmetic>)> {
    let eng = Engine::new();
    let modes = [None, Some(Arithmetic::ExactRational), Some(Arithmetic::ScaledFloat)];
    let arithmetic = modes[case as usize % modes.len()];
    let n = 3 + rng.below(8) as u32;
    let mut home = Home::new(Vtree::random(n, case));
    let all: Vec<VarId> = home.vtree().leaf_bottomup().map(|(_, var)| var).collect();
    let used: Vec<VarId> = all.iter().copied().filter(|_| rng.below(4) != 0).collect();
    let mut f = operand(&eng, &home, &used, rng);
    if let Some(arithmetic) = arithmetic {
        let weighted = Home::weighted(Vtree::random(n, case), Weights::new(weights(n, rng), arithmetic)).unwrap();
        f = reweighted(&f, &home, &weighted);
        home = weighted;
    }
    let h = VtreeIdx(rng.below(home.vtree().num_nodes() as u64) as u32);
    eng.marginalize(&mut f, under(&home, &[h])).ok()?;
    f.vtree().has_regions().then_some((home, f, arithmetic))
}

#[test]
fn a_dropped_leaf_over_a_region_adds_the_values_its_nodes_name() {
    let mut rng = Lcg::new(0x7a1d_5eed);
    let (mut cases, mut sums, mut kept, mut built) = (0, 0, 0, 0);
    for case in 0..6000u64 {
        let Some((home, f, arithmetic)) = with_region(case, &mut rng) else { continue };
        let leaves = pick_leaves(&f, &mut rng);
        if leaves.is_empty() {
            continue;
        }
        let what = format!("case {case} leaves {leaves:?} arithmetic {arithmetic:?}");
        let Some((b, out)) = named_true(&f, &leaves) else { continue };
        if b.check(out).is_err() {
            continue;
        }
        let untrimmed = unsafe { b.finish_unchecked(out) };
        let n = home.vtree().num_leaves();
        let floats = arithmetic == Some(Arithmetic::ScaledFloat);
        let mut t = untrimmed.clone();
        let moved = Engine::new().trim(&mut t, &home).unwrap_or_else(|e| panic!("{what}: {e:?}"));
        same_pointwise(&untrimmed, &t, n, floats, &what);
        cases += 1;
        sums += usize::from(sums_over_a_region(&untrimmed, &leaves));
        if moved {
            check_trimmed(&t, false).unwrap_or_else(|e| panic!("{what}: {e}"));
        } else {
            // A repeated pair above a summed-out level, which a set would
            // count once.
            kept += 1;
        }
        let Some((b, out)) = named_true(&f, &leaves) else { continue };
        if let Ok(g) = b.finish(out) {
            same_pointwise(&untrimmed, &g, n, floats, &format!("{what}, built"));
            built += usize::from(check_trimmed(&g, false).is_ok());
        }
    }
    println!("{cases} cases, {sums} adding values over a region, {kept} kept, {built} built trimmed");
    assert!(cases > 400 && sums > 40 && built > 400, "{cases} cases, {sums} adding values, {built} built trimmed");
}

/// A refused allocation or a stop at any point of a trim that adds values
/// leaves the diagram as it was.
#[test]
fn a_refused_trim_over_a_region_leaves_the_diagram() {
    let mut rng = Lcg::new(0x51ed);
    let mut checked = 0;
    for case in 0..3000u64 {
        if checked == 6 {
            break;
        }
        let Some((home, f, arithmetic)) = with_region(case, &mut rng) else { continue };
        let leaves = pick_leaves(&f, &mut rng);
        let Some((b, out)) = named_true(&f, &leaves) else { continue };
        if leaves.is_empty() || b.check(out).is_err() {
            continue;
        }
        let untrimmed = unsafe { b.finish_unchecked(out) };
        if !sums_over_a_region(&untrimmed, &leaves) {
            continue;
        }
        checked += 1;
        let n = home.vtree().num_leaves();
        let floats = arithmetic == Some(Arithmetic::ScaledFloat);
        let unchanged = |g: &Tdd, what: &str| {
            assert!(Arc::ptr_eq(g.vtree(), untrimmed.vtree()), "{what}: the vtree moved");
            same_pointwise(&untrimmed, g, n, floats, what);
        };
        let mut refused = 0;
        for cut in 0.. {
            let eng = Engine::new();
            let mut g = untrimmed.clone();
            eng.limits().refuse_nth_reserve(cut);
            let result = eng.trim(&mut g, &home);
            let pending = eng.limits().refusal_pending();
            match result {
                Err(e) => {
                    assert_eq!(e, OperationError::OverBudget, "case {case} cut {cut}");
                    unchanged(&g, &format!("case {case} refused at reserve {cut}"));
                    refused += 1;
                }
                Ok(_) if pending => break,
                Ok(_) => {}
            }
        }
        assert!(refused > 0, "case {case}: no reserve refused");
        for stride in [1, 2, 3, 5, 8] {
            let mut g = untrimmed.clone();
            if deadline_probe(Some(stride), |eng| eng.trim(&mut g, &home)).is_err() {
                unchanged(&g, &format!("case {case} stopped at stride {stride}"));
            }
        }
    }
    assert_eq!(checked, 6);
}
