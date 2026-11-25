//! Summing out on weighted diagrams on projections keeps the weighted value,
//! checked against enumeration over the home.

use num_rational::BigRational;
use num_traits::{One, ToPrimitive, Zero};

use crate::diagram::{Arithmetic, LiteralWeights, RationalWeights, Weights};
use crate::test_helpers::{eval, exact_weight};
use crate::vtree::rng::Lcg;
use crate::vtree::{Home, VarId, Vtree};
use crate::{Engine, Tdd};

fn operand(eng: &Engine, home: &Home, vars: &[VarId], rng: &mut Lcg) -> (Tdd, String) {
    let vtree = Vtree::projection(home, vars.iter().copied()).unwrap();
    let mut f = Tdd::one(&crate::vtree::Home::new(std::sync::Arc::clone(&vtree)));
    let mut name = String::new();
    for _ in 0..1 + rng.below(4) {
        let clause: Vec<i32> = (0..1 + rng.below(3))
            .map(|_| {
                let var = vars[rng.below(vars.len() as u64) as usize].0 as i32;
                if rng.coin() { var } else { -var }
            })
            .collect();
        name += &format!("{clause:?}");
        f = eng.and_clause(f, &clause).unwrap();
    }
    if rng.coin() {
        f.minimize().unwrap();
        name += " minimized";
    }
    if rng.below(3) == 0 {
        eng.trim(&mut f, home).unwrap();
        name += " trimmed";
    }
    (f, name)
}

fn subset(all: &[VarId], rng: &mut Lcg) -> Vec<VarId> {
    let s: Vec<VarId> = all.iter().copied().filter(|_| rng.coin()).collect();
    if s.is_empty() { vec![all[rng.below(all.len() as u64) as usize]] } else { s }
}

/// The weighted model count of `t`, a diagram without a summed-out level,
/// over every variable of the home.
fn enumerated(t: &Tdd, literals: &[LiteralWeights<BigRational>]) -> BigRational {
    let n = literals.len();
    let mut sum = BigRational::zero();
    for mask in 0..1u32 << n {
        let asn: Vec<bool> = (0..n).map(|i| mask >> i & 1 == 1).collect();
        if t.is_zero() || !(t.is_one() || eval(t, &asn)) {
            continue;
        }
        let mut term = BigRational::one();
        for (w, &x) in literals.iter().zip(&asn) {
            term *= if x { &w.positive } else { &w.negative };
        }
        sum += term;
    }
    sum
}

fn value(t: &Tdd) -> BigRational {
    let v = t.weighted_value().unwrap().unwrap();
    match v.as_scaled() {
        Some(s) => BigRational::from_float(s.to_f64()).unwrap(),
        None => exact_weight(&v),
    }
}

fn close(a: &BigRational, b: &BigRational, exact: bool) -> bool {
    if exact {
        return a == b;
    }
    let (x, y) = (a.to_f64().unwrap(), b.to_f64().unwrap());
    (x - y).abs() <= 1e-9 * x.abs().max(y.abs())
}

/// `marginalize` and `and_marginalizing_where` on random weighted diagrams
/// on projections, weights not summing to one, some operands summed out
/// before, each result also minimized.
#[test]
fn summing_out_on_a_weighted_projection_keeps_its_value() {
    let eng = Engine::new();
    let mut rng = Lcg::new(0x77e1_6b7d);
    let mut failures = std::collections::BTreeMap::<String, (usize, String)>::new();
    let mut joined = 0usize;
    for case in 0..6000u64 {
        let n = 3 + rng.below(6) as u32;
        let all: Vec<VarId> = (1..=n).map(VarId).collect();
        let arithmetic = if rng.coin() { Arithmetic::ExactRational } else { Arithmetic::ScaledFloat };
        let exact = arithmetic == Arithmetic::ExactRational;
        let mut w = || BigRational::new((1 + rng.below(6) as i64).into(), (1 + rng.below(4) as i64).into());
        let literals: Vec<_> = (0..n).map(|_| LiteralWeights { negative: w(), positive: w() }).collect();
        let linear = rng.coin();
        let tree = if linear { Vtree::linear(n) } else { Vtree::random(n, case) };
        let home = Home::weighted(tree, Weights::new(RationalWeights::from_literals(&literals), arithmetic)).unwrap();
        let (pf, pg) = (subset(&all, &mut rng), subset(&all, &mut rng));
        let (f0, fname) = operand(&eng, &home, &pf, &mut rng);
        let (g0, gname) = operand(&eng, &home, &pg, &mut rng);
        let mut f = f0.clone();
        let first = if rng.below(3) == 0 { subset(&all, &mut rng) } else { Vec::new() };
        if !first.is_empty() {
            eng.marginalize(&mut f, |v| first.contains(&v)).unwrap();
        }
        let summed = subset(&all, &mut rng);
        let summable = |v: VarId| summed.contains(&v);
        let what = format!(
            "case {case} n {n} {arithmetic:?} f {fname} on {pf:?} summed on {first:?}, g {gname} on {pg:?}, summing {summed:?}, home linear {linear}"
        );
        let mut check = |op: &str, t: &mut Tdd, want: &BigRational| {
            for (form, got) in [("", value(t)), (", minimized", { eng.minimize(t).unwrap(); value(t) })] {
                if !close(&got, want, exact) {
                    let e = failures.entry(format!("{op}{form} {arithmetic:?} presummed {}", !first.is_empty())).or_default();
                    e.0 += 1;
                    if e.1.is_empty() {
                        e.1 = format!("{what}: got {got} want {want}");
                    }
                }
            }
        };
        let fv = enumerated(&f0, &literals);
        check("f", &mut f.clone(), &fv);
        let mut s = f.clone();
        eng.marginalize(&mut s, summable).unwrap();
        check("marginalize", &mut s, &fv);

        let joint = enumerated(&eng.and(&f0, &g0).unwrap(), &literals);
        if first.is_empty() {
            let mut and = eng.and(&f, &g0).unwrap();
            check("and", &mut and.clone(), &joint);
            eng.marginalize(&mut and, summable).unwrap();
            check("and, marginalize", &mut and, &joint);
        }
        match eng.and_marginalizing_where(&f, &g0, summable) {
            Ok(mut h) => {
                check("and_marginalizing_where", &mut h, &joint);
                joined += 1;
            }
            // A summed-out operand meeting a leaf of a variable it summed.
            Err(crate::OperationError::MarginalLevel(_)) if !first.is_empty() => {}
            Err(e) => panic!("{what}: and_marginalizing_where {e:?}"),
        }
    }
    let report: Vec<String> = failures.iter().map(|(k, (count, first))| format!("{count} x {k}\n  first: {first}")).collect();
    assert!(report.is_empty(), "{}", report.join("\n"));
    assert!(joined > 4000, "{joined} conjunctions summed out");
}

/// A sum-out that makes a region one of whose parent's nodes names two of
/// its values beside one partner, values 1 and 2: leaf twin contraction
/// read them as the labels `Pos` and `Neg` and merged them into value 0.
#[test]
fn minimizing_keeps_two_region_values_beside_one_partner() {
    let eng = Engine::new();
    let r = |a: i64, b: i64| BigRational::new(a.into(), b.into());
    let literals = [(3, 2, 4, 1), (3, 2, 5, 2), (1, 1, 6, 1), (1, 1, 3, 4), (5, 2, 2, 1), (5, 3, 5, 3), (4, 3, 3, 1)]
        .map(|(a, b, c, d)| LiteralWeights { negative: r(a, b), positive: r(c, d) });
    let home = Home::weighted(Vtree::random(7, 3258), Weights::new(RationalWeights::from_literals(&literals), Arithmetic::ExactRational)).unwrap();
    let on = |vars: &[u32]| Vtree::projection(&home, vars.iter().map(|&v| VarId(v))).unwrap();
    let mut f = eng.and_clause(Tdd::one(&crate::vtree::Home::new(std::sync::Arc::clone(&on(&[1, 2, 3, 5, 7])))), &[7, 2]).unwrap();
    f.minimize().unwrap();
    eng.trim(&mut f, &home).unwrap();
    let mut g = Tdd::one(&crate::vtree::Home::new(std::sync::Arc::clone(&on(&[3, 4, 5, 7]))));
    for clause in [&[4, 7, 4][..], &[5, 7, -3], &[7, -3], &[7, 4]] {
        g = eng.and_clause(g, clause).unwrap();
    }
    let mut h = eng.and(&f, &g).unwrap();
    let want = enumerated(&h, &literals);
    eng.marginalize(&mut h, |v| v.0 >= 4).unwrap();
    assert!(h.vtree().bottomup().any(|t| h.vtree().node(t).is_region()));
    assert_eq!(value(&h), want);
    eng.minimize(&mut h).unwrap();
    assert_eq!(value(&h), want);
}
