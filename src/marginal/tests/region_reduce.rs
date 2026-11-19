//! The reduction passes on diagrams with regions.

use num_bigint::BigUint;

use crate::marginal::tests::region::check_form;
use crate::vtree::{Home, VarId, Vtree};
use crate::{Engine, OperationError, Tdd};

/// Example E9c: home `(x, (((e1, e2), e3), z))`, `Φ₁(x, e)` with `e = 000`
/// only beside `x = 0`, `001` and `010` only beside `x = 1`, `011`, `100`
/// and `101` beside both, and `Φ = Φ₁ ∧ z`.
fn e9c() -> (Tdd, Tdd) {
    let leaf = |v: u32| Vtree::leaf(VarId(v));
    let join = |a: &Vtree, b: &Vtree| Vtree::join(a, b).unwrap();
    let e = join(&join(&leaf(2), &leaf(3)), &leaf(4));
    let home = Home::new(join(&leaf(1), &join(&e, &leaf(5))));
    let own = Vtree::projection(&home, [1, 2, 3, 4].map(VarId)).unwrap();
    let mut phi1 = Tdd::one(&own);
    for clause in [&[-1, 2, 3, 4][..], &[1, 2, 3, -4], &[1, 2, -3, 4], &[-2, -3]] {
        phi1 = phi1.and_clause(clause).unwrap();
    }
    let z = Tdd::clause(&Vtree::projection(&home, [VarId(5)]).unwrap(), [5]).unwrap();
    (phi1, z)
}

/// Each pass of the reduction on a diagram with regions, before and after a
/// condition and a conjunction with a literal: the value stays, the form is
/// one `marginalize` makes, a trim follows, and an `exists` answers as on
/// the diagram before the pass or is refused at a level holding a pair twice.
#[test]
fn every_reduction_pass_keeps_a_diagram_with_regions() {
    use crate::marginal::tests::region_entries::{region_case, value};
    use crate::reduce::{ContentTwinPolicy, ReductionPlan};
    let eng = Engine::new();
    let (mut cases, mut refused, mut cleared) = (0, 0, 0);
    for seed in 0..600u64 {
        let Some((home, n, _, g)) = region_case(&eng, seed, 10) else { continue };
        let held: Vec<VarId> = (1..=n).map(VarId).filter(|&v| g.vtree().leaf_of(v).is_some()).collect();
        let mut inputs = vec![g.clone()];
        if let Some(&v) = held.first() {
            inputs.push(g.clone().condition_var(v, seed % 2 == 0).unwrap());
            if let Ok(h) = g.clone().and_clause([crate::Literal { var: v, sign: seed % 3 == 0 }]) {
                inputs.push(h);
            }
        }
        for (k, input) in inputs.into_iter().enumerate() {
            for index in 0..4 {
                let mut r = input.clone();
                let plan = match index {
                    0 => ReductionPlan::Prune,
                    1 => ReductionPlan::Contract,
                    2 => ReductionPlan::Full(ContentTwinPolicy::Skip),
                    _ => ReductionPlan::default(),
                };
                r.reduce(plan).unwrap();
                cases += 1;
                let what = format!("seed {seed}, input {k}, plan {index}");
                check_form(&r, &what);
                assert_eq!(format!("{:?}", value(&r)), format!("{:?}", value(&input)), "{what}");
                let mut t = r.clone();
                t.trim(&home).unwrap();
                assert_eq!(format!("{:?}", value(&t)), format!("{:?}", value(&input)), "{what}: trim");
                for &v in &held {
                    let before = input.clone().exists_var(v).map(|t| format!("{:?}", value(&t)));
                    match r.clone().exists_var(v) {
                        // The prune left only full counts at the region,
                        // which the trim dropped with it.
                        Ok(_) if matches!(before, Err(OperationError::MarginalLevel(_))) && !r.has_marginal_level() => cleared += 1,
                        Ok(after) => assert_eq!(Ok(format!("{:?}", value(&after))), before, "{what}: exists {v:?}"),
                        Err(OperationError::MarginalLevel(_)) => refused += 1,
                        Err(e) => panic!("{what}: exists {v:?}: {e:?}"),
                    }
                }
            }
        }
    }
    println!("{cases} cases, {refused} refused, {cleared} answered once the region went");
    assert!(cases > 1000 && refused > 0, "{cases} cases, {refused} refused");
}

/// E9c: summing after the conjunction merges the nodes the sum makes equal
/// above the region, up to the root: the study's 7 pairs. Summing first,
/// the conjunction keeps a node no pair reaches, which a prune drops.
#[test]
fn e9c_summing_after_a_conjunction_merges_up_to_the_root() {
    let eng = Engine::new();
    let (phi1, z) = e9c();
    let is_e = |v: VarId| (2..=4).contains(&v.0);
    let mut before = phi1.clone();
    before.marginalize(is_e).unwrap();
    let mut before = eng.and(&before, &z).unwrap();
    let mut after = eng.and(&phi1, &z).unwrap();
    after.marginalize(is_e).unwrap();
    let stored = before.pair_count();
    before.reduce(crate::reduce::ReductionPlan::Prune).unwrap();
    println!("E9c pairs: sum then conjoin {stored}, pruned {}; conjoin then sum {}", before.pair_count(), after.pair_count());
    for f in [&before, &after] {
        check_form(f, "E9c");
        assert_eq!(f.model_count().unwrap(), BigUint::from(9u32));
    }
    assert_eq!((before.pair_count(), after.pair_count()), (4, 7));
}
