//! `restrict_to_care` on projections of one home where either operand has
//! summed out variables the other reads, checked pointwise on homes of at
//! most eight variables.

use num_rational::BigRational;
use num_traits::Zero;

use crate::apply::restrict_to_care::RestrictionOutcome;
use crate::marginal::tests::region::operand;
use crate::reduce::tests::trim_regions::value_at;
use crate::vtree::rng::Lcg;
use crate::vtree::{Home, VarId, Vtree};
use crate::{Engine, Tdd};

/// The variables of `f`'s leaves that it has not summed out.
fn structural(f: &Tdd) -> Vec<VarId> {
    f.vtree().leaf_bottomup().filter(|&(t, _)| !f.level(t).is_marginal()).map(|(_, var)| var).collect()
}

/// The number of variables `f`'s regions sum.
fn in_regions(f: &Tdd) -> u32 {
    let v = f.vtree();
    v.regions().map(|t| v.region_at(t).1.iter().map(|&(lo, hi)| hi - lo).sum::<u32>()).sum()
}

/// The value of `f` where the variables `fixed` take `x`, summed over every
/// other variable of the home: its other leaves, its summed leaves and
/// regions as they hold them, and each variable it does not hold counted
/// twice.
fn value_over(f: &Tdd, n: u32, fixed: &[VarId], x: u32) -> BigRational {
    let leaves: Vec<VarId> = f.vtree().leaf_bottomup().map(|(_, var)| var).collect();
    let others: Vec<VarId> = structural(f).into_iter().filter(|w| !fixed.contains(w)).collect();
    let held = fixed.len() as u32 + leaves.iter().filter(|w| !fixed.contains(w)).count() as u32 + in_regions(f);
    let mut total = BigRational::zero();
    for y in 0..1u32 << others.len() {
        let at = |var: VarId| match fixed.iter().position(|&w| w == var) {
            Some(i) => x >> i & 1 == 1,
            None => others.iter().position(|&w| w == var).is_some_and(|i| y >> i & 1 == 1),
        };
        total += value_at(f, &at);
    }
    total * BigRational::from_integer((1u64 << (n - held)).into())
}

/// `f` with the variables of `vars` summed out, or `f` itself when none is
/// one of its leaves.
fn summing(eng: &Engine, mut f: Tdd, vars: &[VarId]) -> Tdd {
    if vars.iter().any(|&var| f.vtree().leaf_of(var).is_some()) {
        eng.marginalize(&mut f, |var| vars.contains(&var)).unwrap();
    }
    f
}

/// `g` is `f` restricted to `care` as on a root vtree, where a summed level
/// of `f` is live under any `care` and a summed level of `care` is true: at
/// each value of `f`'s structural variables, `g` is at most `f`, and equal
/// where `care` holds for some value of the other variables.
fn check_restricted(f: &Tdd, care: &Tdd, g: &Tdd, n: u32, what: &str) {
    let fixed = structural(f);
    let rest: Vec<VarId> = (1..=n).map(VarId).filter(|w| !fixed.contains(w)).collect();
    for x in 0..1u32 << fixed.len() {
        let (fx, gx) = (value_over(f, n, &fixed, x), value_over(g, n, &fixed, x));
        assert!(gx <= fx, "{what}: at {x:b} {gx} above {fx}");
        let live = (0..1u32 << rest.len()).any(|y| {
            let at = |var: VarId| match fixed.iter().position(|&w| w == var) {
                Some(i) => x >> i & 1 == 1,
                None => rest.iter().position(|&w| w == var).is_some_and(|i| y >> i & 1 == 1),
            };
            value_at(care, &at) != BigRational::zero()
        });
        if live {
            assert_eq!(gx, fx, "{what}: at {x:b}, where care holds");
        }
    }
}

/// Random pairs of trimmed diagrams on projections of one home, each with
/// some of its variables summed out, at least one with a summed level:
/// `(seed, n, f, care)`.
fn summed_pairs(eng: &Engine) -> impl Iterator<Item = (u64, u32, Tdd, Tdd)> + '_ {
    (0..400u64).filter_map(move |seed| {
        let mut rng = Lcg::new(seed ^ 0x2e57_c0de);
        let n = 3 + rng.below(6) as u32;
        let home = Home::new(Vtree::random(n, seed));
        let pick = |rng: &mut Lcg, of: &[VarId]| -> Vec<VarId> { of.iter().copied().filter(|_| rng.coin()).collect() };
        let all: Vec<VarId> = (1..=n).map(VarId).collect();
        let (a, b) = (pick(&mut rng, &all), pick(&mut rng, &all));
        if a.is_empty() || b.is_empty() {
            return None;
        }
        let f = operand(eng, &home, &a, &mut rng);
        let care = operand(eng, &home, &b, &mut rng);
        let (sum_f, sum_care) = (pick(&mut rng, &a), pick(&mut rng, &b));
        let (f, care) = (summing(eng, f, &sum_f), summing(eng, care, &sum_care));
        (f.has_marginal_level() || care.has_marginal_level()).then_some((seed, n, f, care))
    })
}

/// Whether `f` and `g` have one projection holding both: no extent of one
/// meets the other's leaves or extents.
fn share_a_projection(f: &Tdd, g: &Tdd) -> bool {
    Vtree::leaf_sets(f.vtree(), g.vtree()).is_ok()
}

/// A summed level of the care set beside variables `f` reads is true, and
/// leaves no level in the result: the result's regions are `f`'s, summed.
#[test]
fn a_restriction_across_projections_reads_summed_levels_as_on_a_root() {
    let eng = Engine::new();
    let (mut cases, mut shrunk) = (0, 0);
    for (seed, n, f, care) in summed_pairs(&eng).filter(|(_, _, f, care)| share_a_projection(f, care)) {
        cases += 1;
        let what = format!("seed {seed}");
        let outcome = eng.restrict_to_care(f.clone(), &care).unwrap_or_else(|e| panic!("{what}: {e}"));
        shrunk += usize::from(matches!(outcome, RestrictionOutcome::Shrunk(_) | RestrictionOutcome::Unsatisfiable(_)));
        let mut g = outcome.into_tdd();
        let v = g.vtree();
        assert!(v.regions().all(|t| g.level(t).is_marginal()), "{what}: a region of the result is not summed out");
        assert!(v.regions().count() <= f.vtree().regions().count(), "{what}: a region the result does not have from f");
        check_restricted(&f, &care, &g, n, &what);
        eng.minimize(&mut g).unwrap();
        check_restricted(&f, &care, &g, n, &what);
    }
    assert!(cases > 150 && shrunk > 20, "{cases} cases, {shrunk} shrunk");
}

/// Operands with no common projection, where an extent of one meets the
/// other's leaves or extents, give `f` back unchanged, which agrees with the
/// root reading.
#[test]
fn a_restriction_across_meeting_extents_gives_f_unchanged() {
    let eng = Engine::new();
    let mut cases = 0;
    for (seed, n, f, care) in summed_pairs(&eng).filter(|(_, _, f, care)| !share_a_projection(f, care)) {
        cases += 1;
        let what = format!("seed {seed}");
        let outcome = eng.restrict_to_care(f.clone(), &care).unwrap_or_else(|e| panic!("{what}: {e}"));
        let RestrictionOutcome::Unchanged(g) = outcome else { panic!("{what}: {outcome:?}") };
        assert!(std::sync::Arc::ptr_eq(g.vtree(), f.vtree()), "{what}: not on f's vtree");
        check_restricted(&f, &care, &g, n, &what);
    }
    assert!(cases > 40, "{cases} cases");
}
