//! `marginalize` on projections, checked against `marginalize_levels` of the
//! same diagram placed on the home.

use std::sync::Arc;

use num_bigint::BigUint;
use num_rational::BigRational;

use crate::diagram::{Arithmetic, EvalAlgebra, LeafLabel, LiteralWeights, RationalWeights, WeightStore};
use crate::test_helpers::check::marginal::{check_inline_discipline, check_no_orphan_slots, check_slot_count_uniqueness};
use crate::vtree::rng::Lcg;
use crate::vtree::{Home, VarId, Vtree, VtreeIdx};
use crate::{Engine, OperationError, Tdd};

/// A random trimmed diagram on the projection of `home` on `vars`: a CNF of
/// one to five clauses, a constant on the empty projection when `vars` is
/// empty.
fn operand(eng: &Engine, home: &Home, vars: &[VarId], rng: &mut Lcg) -> Tdd {
    if vars.is_empty() {
        let empty = Vtree::empty_projection(home);
        return if rng.below(4) == 0 { Tdd::zero(&empty) } else { Tdd::one(&empty) };
    }
    let vtree = Vtree::projection(home, vars.iter().copied()).unwrap();
    let mut f = Tdd::one(&vtree);
    for _ in 0..1 + rng.below(vars.len().min(12) as u64) {
        let clause: Vec<i32> = (0..1 + rng.below(3))
            .map(|_| {
                let var = vars[rng.below(vars.len() as u64) as usize].0 as i32;
                if rng.coin() { var } else { -var }
            })
            .collect();
        f = eng.and_clause(f, &clause).unwrap();
    }
    eng.trim(&mut f, home).unwrap();
    f
}

fn on_home(eng: &Engine, f: &Tdd, home: &Arc<Vtree>) -> Tdd {
    if f.vtree().num_nodes() == 0 {
        return if f.is_one() { Tdd::one(home) } else { Tdd::zero(home) };
    }
    eng.embed(f, home, |v| v).unwrap().0
}

fn weights(n: u32, rng: &mut Lcg) -> RationalWeights {
    let mut rat = || BigRational::new((1 + rng.below(6) as i64).into(), (1 + rng.below(4) as i64).into());
    let literals: Vec<_> = (0..n).map(|_| LiteralWeights { negative: rat(), positive: rat() }).collect();
    RationalWeights::from_literals(&literals)
}

/// One to three home nodes; for a second sum-out, each related to one of
/// `before` with probability two in three, an ancestor or a descendant.
fn targets(home: &Vtree, before: &[VtreeIdx], rng: &mut Lcg) -> Vec<VtreeIdx> {
    (0..1 + rng.below(3))
        .map(|_| {
            let mut h = VtreeIdx(rng.below(home.num_nodes() as u64) as u32);
            if !before.is_empty() {
                let b = before[rng.below(before.len() as u64) as usize];
                match rng.below(3) {
                    0 => h = home.node(b).parent().unwrap_or(b),
                    1 if !home.node(b).is_leaf() => {
                        let (l, r) = home.children(b);
                        h = if rng.coin() { l } else { r };
                    }
                    _ => {}
                }
            }
            h
        })
        .collect()
}

fn check_form(f: &Tdd, what: &str) {
    let vtree = f.vtree();
    if vtree.num_nodes() != 0 {
        vtree.validate().unwrap_or_else(|e| panic!("{what}: {e}"));
    }
    assert_eq!(f.levels.len(), vtree.num_nodes(), "{what}");
    for t in vtree.bottomup() {
        assert!(!f.level(t).is_marginal() || vtree.node(t).is_leaf(), "{what}: internal summed-out level {}", t.0);
    }
    if vtree.num_nodes() != 0 {
        assert_eq!(f.output().vtree, vtree.root(), "{what}");
    }
    if f.weights().is_none() {
        check_inline_discipline(f).unwrap_or_else(|e| panic!("{what}: {e}"));
        check_slot_count_uniqueness(f).unwrap_or_else(|e| panic!("{what}: {e}"));
    }
    check_no_orphan_slots(f).unwrap_or_else(|e| panic!("{what}: {e}"));
}

/// Model counts as an algebra that values counts.
struct Counting;

impl EvalAlgebra for Counting {
    type Value = BigUint;
    fn zero(&self) -> BigUint {
        BigUint::ZERO
    }
    fn leaf(&self, _: VarId, label: LeafLabel) -> BigUint {
        BigUint::from(if label == LeafLabel::One { 2u32 } else { 1 })
    }
    fn add_assign(&self, acc: &mut BigUint, other: &BigUint) {
        *acc += other;
    }
    fn mul(&self, a: &BigUint, b: &BigUint) -> BigUint {
        a * b
    }
    fn count(&self, n: &BigUint) -> Option<BigUint> {
        Some(n.clone())
    }
}

fn check_values(f: &Tdd, g: &Tdd, n: u32, arithmetic: Option<Arithmetic>, what: &str) {
    match arithmetic {
        None => {
            assert_eq!(f.model_count().unwrap(), g.model_count().unwrap(), "{what}: model count");
            assert_eq!(f.evaluate(&Counting).unwrap(), g.evaluate(&Counting).unwrap(), "{what}: evaluate");
            let unit = RationalWeights::unit(n as usize);
            if !g.has_marginal_level() {
                assert_eq!(f.evaluate(&unit).unwrap(), g.evaluate(&unit).unwrap(), "{what}: evaluate");
            }
        }
        Some(Arithmetic::ExactRational) => {
            let (a, b) = (f.weighted_value().unwrap().unwrap(), g.weighted_value().unwrap().unwrap());
            assert_eq!(a.into_rational_opt(), b.into_rational_opt(), "{what}: weighted value");
        }
        Some(_) => {
            let value = |f: &Tdd| f.weighted_value().unwrap().unwrap().as_scaled().expect("a scaled float").to_f64();
            let (a, b) = (value(f), value(g));
            assert!((a - b).abs() <= 1e-9 * a.abs().max(b.abs()), "{what}: weighted value {a} against {b}");
        }
    }
}

#[test]
fn marginalize_matches_the_smooth_form_on_the_home() {
    let eng = Engine::new();
    let modes = [None, Some(Arithmetic::ExactRational), Some(Arithmetic::ScaledFloat)];
    let mut rng = Lcg::new(0x5eed_0bad);
    let mut seen = [0usize; 3];
    for case in 0..600u64 {
        let n = 1 + rng.below(64) as u32;
        let home_vtree = Arc::new(Vtree::random(n, case));
        let home = Home::new(Arc::clone(&home_vtree));
        let all: Vec<VarId> = home_vtree.leaf_bottomup().map(|(_, var)| var).collect();
        let used: Vec<VarId> = match rng.below(8) {
            0 => Vec::new(),
            1 => all.clone(),
            _ => all.iter().copied().filter(|_| rng.coin()).collect(),
        };
        let arithmetic = modes[case as usize % modes.len()];
        let mut f = operand(&eng, &home, &used, &mut rng);
        let mut g = on_home(&eng, &f, &home_vtree);
        if let Some(arithmetic) = arithmetic {
            let w = weights(n, &mut rng);
            f.set_weights(WeightStore::new(w.clone(), arithmetic)).unwrap();
            g.set_weights(WeightStore::new(w, arithmetic)).unwrap();
        }
        let mut before = Vec::new();
        for step in 0..1 + rng.below(2) {
            let hs = targets(&home_vtree, &before, &mut rng);
            let what = format!("case {case} step {step} targets {hs:?} arithmetic {arithmetic:?}");
            let had = count_regions(&f);
            let free = hs.iter().any(|&h| f.vtree().own_at(h).is_none());
            eng.marginalize(&mut f, &hs).unwrap_or_else(|e| panic!("{what}: {e:?}"));
            eng.marginalize_levels(&mut g, &hs).unwrap();
            check_form(&f, &what);
            check_values(&f, &g, n, arithmetic, &what);
            seen[0] += usize::from(count_regions(&f) > had);
            seen[1] += usize::from(free);
            seen[2] += usize::from(had > 0);
            before = hs;
        }
    }
    let [built, free, again] = seen;
    assert!(built > 150 && free > 300 && again > 40, "{built} sum-outs built a region, {free} had a free target, {again} ran on a region");
}

fn count_regions(f: &Tdd) -> usize {
    f.vtree().bottomup().filter(|&t| f.vtree().node(t).is_region()).count()
}

#[test]
fn marginalize_refuses_a_summed_out_internal_level_that_is_not_a_region() {
    let home = Home::new(Vtree::balanced(4));
    let own = Vtree::projection(&home, [VarId(1), VarId(2), VarId(3)]).unwrap();
    let mut f = Tdd::clause(&own, [1, 2, 3]).unwrap();
    let (left, _) = own.children(own.root());
    f.marginalize_levels(&[left]).unwrap();
    assert_eq!(f.marginalize(&[home.vtree().root()]), Err(OperationError::MarginalLevel(left)));
    assert_eq!(f.marginalize(&[VtreeIdx(99)]), Err(OperationError::LevelNotInVtree(VtreeIdx(99))));
}
