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
pub(crate) fn operand(eng: &Engine, home: &Home, vars: &[VarId], rng: &mut Lcg) -> Tdd {
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

pub(crate) fn on_home(eng: &Engine, f: &Tdd, home: &Arc<Vtree>) -> Tdd {
    if f.vtree().num_nodes() == 0 {
        return if f.is_one() { Tdd::one(home) } else { Tdd::zero(home) };
    }
    eng.embed(f, home, |v| v).unwrap().0
}

pub(crate) fn weights(n: u32, rng: &mut Lcg) -> RationalWeights {
    let mut rat = || BigRational::new((1 + rng.below(6) as i64).into(), (1 + rng.below(4) as i64).into());
    let literals: Vec<_> = (0..n).map(|_| LiteralWeights { negative: rat(), positive: rat() }).collect();
    RationalWeights::from_literals(&literals)
}

/// One to three home nodes; for a second sum-out, each related to one of
/// `before` with probability two in three, an ancestor or a descendant.
pub(crate) fn targets(home: &Vtree, before: &[VtreeIdx], rng: &mut Lcg) -> Vec<VtreeIdx> {
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

/// The predicate "under one of the home nodes `hs`".
pub(crate) fn under<'a>(home: &'a Home, hs: &'a [VtreeIdx]) -> impl Fn(VarId) -> bool + 'a {
    move |v| home.rank(v).is_some_and(|r| hs.iter().any(|&h| home.rank_range(h).contains(&r)))
}

/// The largest number of intervals an extent of `f` has.
pub(crate) fn max_intervals(f: &Tdd) -> usize {
    f.vtree().regions().map(|t| f.vtree().region_at(t).1.len()).max().unwrap_or(0)
}

pub(crate) fn check_form(f: &Tdd, what: &str) {
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

pub(crate) fn check_values(f: &Tdd, g: &Tdd, n: u32, arithmetic: Option<Arithmetic>, what: &str) {
    match arithmetic {
        None => {
            assert_eq!(f.model_count().unwrap(), g.model_count().unwrap(), "{what}: model count");
            assert_eq!(f.evaluate(&Counting).unwrap(), g.evaluate(&Counting).unwrap(), "{what}: evaluate");
            let unit = RationalWeights::unit(n as usize);
            if !f.has_marginal_level() && !g.has_marginal_level() {
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
    let mut seen = [0usize; 4];
    for case in 0..600u64 {
        let n = 1 + rng.below(64) as u32;
        let home = Home::new(Vtree::random(n, case));
        let home_vtree = Arc::clone(home.vtree());
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
            eng.marginalize(&mut f, under(&home, &hs)).unwrap_or_else(|e| panic!("{what}: {e:?}"));
            eng.marginalize_levels(&mut g, &hs).unwrap();
            check_form(&f, &what);
            check_values(&f, &g, n, arithmetic, &what);
            seen[0] += usize::from(count_regions(&f) > had);
            seen[1] += usize::from(free);
            seen[2] += usize::from(had > 0);
            seen[3] = seen[3].max(max_intervals(&f));
            before = hs;
        }
    }
    let [built, free, again, intervals] = seen;
    println!("largest number of intervals in an extent: {intervals}");
    assert!(built > 150 && free > 300 && again > 40, "{built} sum-outs built a region, {free} had a free target, {again} ran on a region");
}

fn count_regions(f: &Tdd) -> usize {
    f.vtree().bottomup().filter(|&t| f.vtree().node(t).is_region()).count()
}

#[test]
fn marginalize_refuses_a_summed_out_internal_level_that_is_not_a_region() {
    let eng = Engine::new();
    let home = Home::new(Vtree::balanced(4));
    let own = Vtree::projection(&home, [VarId(1), VarId(2), VarId(3)]).unwrap();
    let mut f = Tdd::clause(&own, [1, 2, 3]).unwrap();
    let (left, _) = own.children(own.root());
    crate::marginal::marginalize_levels(&eng, &mut f, &[left]).unwrap();
    assert_eq!(f.marginalize(|_| true), Err(OperationError::MarginalLevel(left)));
}

/// Home node ids index another tree than a projection's own nodes.
#[test]
fn marginalize_levels_refuses_a_projection() {
    let home = Home::new(Vtree::balanced(4));
    let own = Vtree::projection(&home, [VarId(1), VarId(2), VarId(3)]).unwrap();
    let mut f = Tdd::clause(&own, [1, 2, 3]).unwrap();
    let before = f.clone();
    let (left, _) = home.vtree().children(home.vtree().root());
    assert!(matches!(f.marginalize_levels(&[left]), Err(OperationError::InertOption { .. })));
    assert_eq!(f.levels.len(), before.levels.len());
    assert!(!f.has_marginal_level());
}

/// A summable leaf whose parent has a variable that is not summable is not
/// summed out.
#[test]
fn a_lone_summable_leaf_stays_a_variable_leaf() {
    let home = Home::new(Vtree::balanced(4));
    let own = Vtree::projection(&home, [VarId(1), VarId(3)]).unwrap();
    let mut f = Tdd::clause(&own, [1, 3]).unwrap();
    f.marginalize(|v| v == VarId(1)).unwrap();
    assert!(!f.has_marginal_level());
    assert_eq!(f.vtree().num_leaves(), 2);
    assert_eq!(f.model_count().unwrap(), BigUint::from(12u32));
}

/// Summing a node with nodes under it leaves one region and no internal
/// summed-out level.
#[test]
fn a_node_under_a_summed_node_leaves_no_internal_summed_level() {
    let eng = Engine::new();
    let home = Home::new(Vtree::balanced(8));
    let mut rng = Lcg::new(7);
    let all: Vec<VarId> = (1..=8).map(VarId).collect();
    let mut f = operand(&eng, &home, &all, &mut rng);
    let before = f.model_count().unwrap();
    f.marginalize(|v| v.0 <= 4).unwrap();
    check_form(&f, "sum of 1 to 4");
    assert!(count_regions(&f) <= 1);
    assert_eq!(f.model_count().unwrap(), before);
}

/// Home `(m, (((x, u), z), y))` and `f = (m ∨ x) ∧ (x ∨ z) ∧ y` on the
/// projection without `u`, `m` summed in place and `y` conditioned: a
/// sum-out over `x` and `z` that is refused leaves the count as it was.
#[test]
fn a_refused_marginalize_leaves_the_count_as_it_was() {
    let eng = Engine::new();
    let leaf = |v: u32| Vtree::leaf(VarId(v));
    let join = |a: &Vtree, b: &Vtree| Vtree::join(a, b).unwrap();
    let xuz = join(&join(&leaf(2), &leaf(3)), &leaf(4));
    let home = Home::new(join(&leaf(1), &join(&xuz, &leaf(5))));
    let own = Vtree::projection(&home, [1, 2, 4, 5].map(VarId)).unwrap();
    let mut f = Tdd::one(&own);
    for clause in [&[1, 2][..], &[2, 4], &[5]] {
        f = f.and_clause(clause).unwrap();
    }
    let m = f.vtree().leaf_bottomup().find(|&(_, v)| v == VarId(1)).unwrap().0;
    crate::marginal::marginalize_levels(&eng, &mut f, &[m]).unwrap();
    let mut f = f.condition_var(VarId(5), true).unwrap();
    let before = f.model_count().unwrap();
    let _ = f.marginalize(|v| (2..=4).contains(&v.0));
    assert_eq!(f.model_count().unwrap(), before);
}

/// A refusal at any reserve of a sum-out that builds a region leaves the
/// count as it was.
#[test]
fn every_refused_reserve_of_a_region_sum_out_leaves_the_count() {
    let home = Home::new(Vtree::balanced(6));
    let own = Vtree::projection(&home, [1, 2, 4, 5].map(VarId)).unwrap();
    let mut f = Tdd::one(&own);
    for clause in [[1, -2], [2, 4], [-4, 5]] {
        f = f.and_clause(clause).unwrap();
    }
    let before = f.model_count().unwrap();
    let (mut refused, mut built) = (0, false);
    for cut in 0.. {
        let eng = Engine::new();
        let mut g = f.clone();
        eng.limits().refuse_nth_reserve(cut);
        let result = eng.marginalize(&mut g, |_| true);
        let pending = eng.limits().refusal_pending();
        refused += usize::from(result.is_err());
        built |= result.is_ok() && count_regions(&g) > 0;
        assert_eq!(g.model_count().unwrap(), before, "refused at reserve {cut}: {result:?}");
        if result.is_ok() && pending {
            break;
        }
    }
    assert!(built && refused > 0, "{refused} refusals");
}

#[test]
fn summing_out_a_root_of_several_nodes_follows_the_output() {
    use crate::diagram::{ChildPair, TddNodeId, NEG_LEAF_IDX, ONE_LEAF_IDX, POS_LEAF_IDX};
    // Three root nodes with 2, 2 and 3 models; the output is the second.
    let eng = Engine::new();
    let vtree = Arc::new(Vtree::balanced(2));
    let root = vtree.root();
    let mut b = Tdd::builder(&eng, &vtree).unwrap();
    b.push(&eng, root, &[ChildPair::new(POS_LEAF_IDX, ONE_LEAF_IDX)]).unwrap();
    let second = b.push(&eng, root, &[ChildPair::new(NEG_LEAF_IDX, ONE_LEAF_IDX)]).unwrap();
    b.push(&eng, root, &[ChildPair::new(POS_LEAF_IDX, ONE_LEAF_IDX), ChildPair::new(NEG_LEAF_IDX, NEG_LEAF_IDX)]).unwrap();
    let mut f = b.finish(TddNodeId { vtree: root, local: second }).unwrap();
    assert_eq!(f.model_count().unwrap(), BigUint::from(2u32));
    f.marginalize_levels(&[root]).unwrap();
    assert_eq!(f.model_count().unwrap(), BigUint::from(2u32));
}
