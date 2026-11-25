//! `marginalize` on projections, checked against `marginalize_levels` of the
//! same diagram placed on the home.

use std::sync::Arc;

use num_bigint::BigUint;
use num_rational::BigRational;

use crate::diagram::{Arithmetic, EvalAlgebra, LeafLabel, LiteralWeights, RationalWeights, Weights};
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
        return if rng.below(4) == 0 { Tdd::zero(&crate::vtree::Home::unchecked(&empty)) } else { Tdd::one(&crate::vtree::Home::unchecked(&empty)) };
    }
    let vtree = Vtree::projection(home, vars.iter().copied()).unwrap();
    let mut f = crate::test_helpers::padded(Tdd::one(&crate::vtree::Home::unchecked(&vtree)), &vtree);
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

pub(crate) fn on_home(f: &Tdd, home: &Arc<Vtree>) -> Tdd {
    crate::test_helpers::embedded(f, home, |v| v).unwrap().0
}

/// `f`, on `home` or a projection of it, with the same levels on
/// `weighted`, a weighted home of the same tree: on its root, or on its
/// projection on the same variables, whose node ids are those of `f`'s.
pub(crate) fn reweighted(f: &Tdd, home: &Home, weighted: &Home) -> Tdd {
    let vars: Vec<VarId> = f.vtree().leaf_bottomup().map(|(_, var)| var).collect();
    let vtree = if Arc::ptr_eq(f.vtree(), home) {
        Arc::clone(weighted)
    } else if vars.is_empty() {
        Vtree::empty_projection(weighted)
    } else {
        Vtree::projection(weighted, vars).unwrap()
    };
    assert_eq!(vtree.to_text(), f.vtree().to_text());
    Tdd::from_levels_unchecked(vtree, f.levels.to_vec(), f.output)
}

/// `f` on a copy of `home`'s tree that is not a home, where the smooth API
/// sums out by `home`'s node ids.
pub(crate) fn on_root_copy(f: &Tdd, home: &Arc<Vtree>) -> Tdd {
    on_home(f, &Arc::new(Vtree::clone(home)))
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
    if vtree.home().is_some() {
        crate::test_helpers::check::check_trimmed(f, false).unwrap_or_else(|e| panic!("{what}: {e}"));
    }
}

/// Model counts as an algebra that values counts.
struct Counting;

impl EvalAlgebra for Counting {
    type Value = BigUint;
    fn zero(&self) -> BigUint {
        BigUint::ZERO
    }
    fn one(&self) -> BigUint {
        BigUint::from(1u32)
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
    let mut seen = [0usize; 5];
    for case in 0..600u64 {
        let n = 1 + rng.below(64) as u32;
        let mut home = Home::new(Vtree::random(n, case));
        let all: Vec<VarId> = home.leaf_bottomup().map(|(_, var)| var).collect();
        let used: Vec<VarId> = match rng.below(8) {
            0 => Vec::new(),
            1 => all.clone(),
            _ => all.iter().copied().filter(|_| rng.coin()).collect(),
        };
        let arithmetic = modes[case as usize % modes.len()];
        let mut f = operand(&eng, &home, &used, &mut rng);
        if let Some(arithmetic) = arithmetic {
            let weighted = Home::weighted(Vtree::random(n, case), Weights::new(weights(n, &mut rng), arithmetic)).unwrap();
            f = reweighted(&f, &home, &weighted);
            home = weighted;
        }
        let home_vtree = Arc::clone(&home);
        let mut g = on_root_copy(&f, &home_vtree);
        let mut before = Vec::new();
        for step in 0..1 + rng.below(2) {
            let hs = targets(&home_vtree, &before, &mut rng);
            let what = format!("case {case} step {step} targets {hs:?} arithmetic {arithmetic:?}");
            let had = count_regions(&f);
            let free = hs.iter().any(|&h| f.vtree().own_at(h).is_none());
            eng.marginalize(&mut f, under(&home, &hs)).unwrap_or_else(|e| panic!("{what}: {e:?}"));
            crate::test_helpers::marginalize_under(&eng, &mut g, &hs).unwrap();
            check_form(&f, &what);
            check_values(&f, &g, n, arithmetic, &what);
            // The closing step of `condition` on the summed-out form.
            let mut pick = Lcg::new(case << 2 | step);
            // A lone summable leaf stays a variable leaf of `f` while `g`
            // sums it.
            let summed_in_g = |x: VarId| {
                let leaf = g.vtree().leaf_of(x).expect("a leaf of the home");
                std::iter::successors(Some(leaf), |&t| g.vtree().node(t).parent()).any(|t| g.level(t).is_marginal())
            };
            let vars: Vec<VarId> = f.vtree().bottomup().filter(|&t| f.vtree().node(t).is_var_leaf()).map(|t| f.vtree().leaf_var(t)).filter(|&x| !summed_in_g(x)).collect();
            if f.has_marginal_level() && !vars.is_empty() {
                let (x, b) = (vars[pick.below(vars.len() as u64) as usize], pick.coin());
                let fx = f.clone().condition_var(x, b).unwrap_or_else(|e| panic!("{what}: condition {x:?}: {e:?}"));
                let gx = g.clone().condition_var(x, b).unwrap();
                check_form(&fx, &format!("{what}, condition {x:?} {b}"));
                // Or a repeated pair kept it, which `check_form` admitted.
                let kept = crate::test_helpers::check::check_trimmed(&fx, false).is_err();
                assert!(fx.vtree().leaf_of(x).is_none() || kept, "{what}: condition {x:?} {b} left its leaf");
                seen[4] += usize::from(!kept);
                check_values(&fx, &gx, n, arithmetic, &format!("{what}, condition {x:?} {b}"));
            }
            seen[0] += usize::from(count_regions(&f) > had);
            seen[1] += usize::from(free);
            seen[2] += usize::from(had > 0);
            seen[3] = seen[3].max(max_intervals(&f));
            before = hs;
        }
    }
    let [built, free, again, intervals, conditioned] = seen;
    println!("largest number of intervals in an extent: {intervals}; {conditioned} conditioned on a summed-out form");
    assert!(
        built > 150 && free > 300 && again > 40 && conditioned > 100,
        "{built} sum-outs built a region, {free} had a free target, {again} ran on a region, {conditioned} conditioned"
    );
}

/// Home `((e, f), (x, y))` and `e ∧ f ∧ x ∧ y` with `e` and `f` summed into a
/// region: conditioning on `x` drops its leaf and `∃x` is refused, as on any
/// diagram with a summed-out level; so is `∃x` of `¬y ∧ (¬e ∨ x) ∧ (e ∨ f)`
/// with the region on either side of the root.
#[test]
fn a_closing_step_beside_a_region_drops_the_leaf_it_names_true() {
    let leaf = |v: u32| Vtree::leaf(VarId(v));
    let join = |a: &Vtree, b: &Vtree| Vtree::join(a, b).unwrap();
    let (ef, xy) = (join(&leaf(1), &leaf(2)), join(&leaf(3), &leaf(4)));
    let home = Home::new(join(&ef, &xy));
    let mut f = Tdd::cube(&home, [1, 2, 3, 4]).unwrap();
    f.marginalize(|v| v.0 <= 2).unwrap();
    assert_eq!(count_regions(&f), 1);
    for b in [true, false] {
        let fx = f.clone().condition_var(VarId(3), b).unwrap();
        check_form(&fx, &format!("condition x {b}"));
        assert!(fx.vtree().leaf_of(VarId(3)).is_none(), "condition x {b} left its leaf");
        assert_eq!(fx.model_count().unwrap(), BigUint::from(if b { 2u32 } else { 0 }), "condition x {b}");
    }
    assert!(matches!(f.exists_var(VarId(3)), Err(OperationError::MarginalLevel(_))));
    for home in [Home::new(join(&ef, &xy)), Home::new(join(&xy, &ef))] {
        let mut g = Tdd::cube(&home, [-4]).unwrap();
        g = g.and_clause([-1, 3]).unwrap().and_clause([1, 2]).unwrap();
        g.marginalize(|v| v.0 <= 2).unwrap();
        assert_eq!(count_regions(&g), 1);
        assert!(matches!(g.clone().exists_var(VarId(3)), Err(OperationError::MarginalLevel(_))));
        assert!(matches!(g.exists_vars(&[VarId(3)]), Err(OperationError::MarginalLevel(_))));
    }
}

fn count_regions(f: &Tdd) -> usize {
    f.vtree().bottomup().filter(|&t| f.vtree().node(t).is_region()).count()
}

#[test]
fn marginalize_refuses_a_summed_out_internal_level_that_is_not_a_region() {
    let eng = Engine::new();
    let home = Home::new(Vtree::balanced(4));
    let own = Vtree::projection(&home, [VarId(1), VarId(2), VarId(3)]).unwrap();
    let mut f = Tdd::clause(&crate::vtree::Home::unchecked(&own), [1, 2, 3]).unwrap();
    let (left, _) = own.children(own.root());
    crate::marginal::tests::marginalize_levels(&eng, &mut f, &[left]).unwrap();
    assert_eq!(f.marginalize(|_| true), Err(OperationError::MarginalLevel(left)));
}

/// A summable leaf whose parent has a variable that is not summable becomes
/// a region of its one variable.
#[test]
fn a_lone_summable_leaf_becomes_a_region() {
    let home = Home::new(Vtree::balanced(4));
    let own = Vtree::projection(&home, [VarId(1), VarId(3)]).unwrap();
    let mut f = Tdd::clause(&crate::vtree::Home::unchecked(&own), [1, 3]).unwrap();
    f.marginalize(|v| v == VarId(1)).unwrap();
    check_form(&f, "sum of 1");
    assert_eq!(f.vtree().regions().count(), 1);
    assert_eq!(f.vtree().num_leaves(), 1);
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
    let mut f = Tdd::one(&crate::vtree::Home::unchecked(&own));
    for clause in [&[1, 2][..], &[2, 4], &[5]] {
        f = f.and_clause(clause).unwrap();
    }
    let m = f.vtree().leaf_bottomup().find(|&(_, v)| v == VarId(1)).unwrap().0;
    crate::marginal::tests::marginalize_levels(&eng, &mut f, &[m]).unwrap();
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
    let mut f = Tdd::one(&crate::vtree::Home::unchecked(&own));
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
    use crate::diagram::{ChildPair, TddNodeId, NEG_LEAF_IDX, POS_LEAF_IDX};
    // Three root nodes with 1, 1 and 2 models; the output is the second.
    let eng = Engine::new();
    let vtree = Arc::new(Vtree::balanced(2));
    let root = vtree.root();
    let mut b = Tdd::builder(&eng, &vtree).unwrap();
    b.push(&eng, root, &[ChildPair::new(POS_LEAF_IDX, NEG_LEAF_IDX)]).unwrap();
    let second = b.push(&eng, root, &[ChildPair::new(NEG_LEAF_IDX, POS_LEAF_IDX)]).unwrap();
    b.push(&eng, root, &[ChildPair::new(POS_LEAF_IDX, POS_LEAF_IDX), ChildPair::new(NEG_LEAF_IDX, NEG_LEAF_IDX)]).unwrap();
    let mut f = b.finish(TddNodeId { vtree: root, local: second }).unwrap();
    assert_eq!(f.model_count().unwrap(), BigUint::from(1u32));
    f.marginalize(crate::test_helpers::summing(f.vtree(), &[root])).unwrap();
    assert_eq!(f.model_count().unwrap(), BigUint::from(1u32));
}

/// Trim of a summed diagram can leave a region in its pass lists, the end of
/// a dropped chain the lists named; the conjunction that follows must read
/// it as the leaf it is.
#[test]
fn a_restricted_summed_diagram_conjoins_as_it_was() {
    let eng = Engine::new();
    let mut cases = 0;
    for seed in 0..600u64 {
        let mut rng = Lcg::new(seed ^ 0x7e61_0a4d);
        let n = 2 + rng.below(9) as u32;
        let mut home = Home::new(Vtree::random(n, seed));
        let used: Vec<VarId> = (1..=n).filter(|_| rng.below(4) != 0).map(VarId).collect();
        let mut g = operand(&eng, &home, &used, &mut rng);
        if g.vtree().num_nodes() == 0 {
            continue;
        }
        if rng.below(3) == 0 {
            let weighted = Home::weighted(Vtree::random(n, seed), Weights::new(weights(n, &mut rng), Arithmetic::ExactRational)).unwrap();
            g = reweighted(&g, &home, &weighted);
            home = weighted;
        }
        let h = &home;
        let full: Vec<VtreeIdx> = h
            .bottomup()
            .filter(|&t| !h.node(t).is_leaf() && h.subtree(t).filter(|&u| h.node(u).is_leaf()).all(|u| g.vtree().leaf_of(h.leaf_var(u)).is_some()))
            .collect();
        let mut hs: Vec<VtreeIdx> = Vec::new();
        for _ in 0..1 + rng.below(2) {
            if let Some(&t) = full.get(rng.below(full.len().max(1) as u64) as usize)
                && hs.iter().all(|&c| !h.in_subtree(c, t) && !h.in_subtree(t, c))
            {
                hs.push(t);
            }
        }
        if hs.is_empty() {
            continue;
        }
        eng.marginalize(&mut g, |v| hs.iter().any(|&t| h.in_subtree(h.leaf_of(v).unwrap(), t))).unwrap();
        if count_regions(&g) == 0 {
            continue;
        }
        let free = (1..=n).map(VarId).filter(|&v| hs.iter().all(|&t| !h.in_subtree(h.leaf_of(v).unwrap(), t)));
        let mut rng = Lcg::new(seed ^ 0xb1de);
        let vars: Vec<VarId> = free.filter(|_| rng.below(3) != 0).collect();
        if vars.is_empty() {
            continue;
        }
        let care = operand(&eng, &home, &vars, &mut rng);
        cases += 1;
        let r = eng.restrict_to_care(g.clone(), &care).unwrap().into_tdd();
        let want = conjoined_value(&eng, &g, &care);
        assert_eq!(conjoined_value(&eng, &r, &care), want, "seed {seed}");
    }
    assert!(cases > 150, "{cases} cases");
}

/// The value of `f ∧ g` in each order, each operand borrowed and owned, all
/// four equal.
fn conjoined_value(eng: &Engine, f: &Tdd, g: &Tdd) -> String {
    let value = |t: Tdd| format!("{:?} {:?}", t.model_count(), t.weighted_value());
    let forms = [eng.and(f, g), eng.and(g, f), eng.and(f.clone(), g.clone()), eng.and(g.clone(), f.clone())];
    let values: Vec<String> = forms.into_iter().map(|t| value(t.unwrap())).collect();
    assert!(values.iter().all(|v| *v == values[0]), "{values:?}");
    values[0].clone()
}

/// Home `((x1, x2), (x3, x4))`, `f = x1 ∧ (x3 ∨ x4)` summed over `(x3, x4)`:
/// beside `x2`, the reader of `x1` names it as a literal through a
/// pass-through, with the region in `f`'s leaf pass list.
#[test]
fn a_literal_beside_a_region_conjoins_with_a_pass_through() {
    let eng = Engine::new();
    let home = Home::new(Vtree::balanced(4));
    let own = Vtree::projection(&home, [1, 3, 4].map(VarId)).unwrap();
    let mut f = Tdd::one(&crate::vtree::Home::unchecked(&own));
    for clause in [&[1][..], &[3, 4]] {
        f = eng.and_clause(f, clause).unwrap();
    }
    eng.marginalize(&mut f, |v| v.0 >= 3).unwrap();
    let care = Tdd::clause(&crate::vtree::Home::unchecked(&Vtree::projection(&home, [VarId(2)]).unwrap()), [2]).unwrap();
    let r = eng.restrict_to_care(f.clone(), &care).unwrap().into_tdd();
    assert_eq!(conjoined_value(&eng, &f, &care), "Ok(3) Ok(None)");
    assert_eq!(conjoined_value(&eng, &r, &care), "Ok(3) Ok(None)");
}
