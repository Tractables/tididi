//! `exists`, `condition`, `substitute`, `transport` and the cluster merge on
//! diagrams with regions, against the smooth form on the home, and the
//! refusals the region form keeps.

use std::sync::Arc;

use num_bigint::BigUint;

use super::region::check_form;
use super::region::under;
use super::region_entries::{region_case_and_source, value};
use crate::diagram::{NodeIdx, TddBuildError, TddNodeId};
use crate::restructure::search::{MinimizePairs, RotationSearchConfig};
use crate::vtree::rng::Lcg;
use crate::vtree::{Home, VarId, Vtree};
use crate::{Engine, OperationError, Tdd};

fn same(x: &Tdd, y: &Tdd) -> bool {
    format!("{:?}", value(x)) == format!("{:?}", value(y))
}

/// `∃x` after a sum-out: the region form, once fused, cannot tell which
/// pre-sum events its values came from, so it once answered 24 where the sum
/// taken after `∃x4` gives 40 (and, at 62 leaves, seed 127 with `∃x6`). Both
/// forms refuse.
#[test]
fn exists_on_a_summed_diagram_is_refused() {
    let eng = Engine::new();
    for (seed, max, x, count) in [(4795, 8, 4, 40u64), (127, 64, 6, 88_664_617_663_856_640)] {
        let (home, _, f, g, source, hs) = region_case_and_source(&eng, seed, max).unwrap();
        let mut summed_after = source.exists_var(VarId(x)).unwrap();
        summed_after.marginalize(under(&home, &hs)).unwrap();
        assert_eq!(summed_after.model_count().unwrap(), BigUint::from(count), "seed {seed}");
        assert!(matches!(g.exists_var(VarId(x)), Err(OperationError::MarginalLevel(_))), "seed {seed}");
        assert!(matches!(f.exists_var(VarId(x)), Err(OperationError::MarginalLevel(_))), "seed {seed}");
    }
    // Home (e, (x, y)), ¬y ∧ (¬e ∨ x), e summed out: the fused form would
    // count 6 for ∃x where Σe ∃x gives 4.
    let leaf = |v: u32| Vtree::leaf(VarId(v));
    let vtree = Arc::new(Vtree::join(&leaf(1), &Vtree::join(&leaf(2), &leaf(3)).unwrap()).unwrap());
    let mut f = eng.and(Tdd::cube(&vtree, [-3]).unwrap(), Tdd::clause(&vtree, [-1, 2]).unwrap()).unwrap();
    f.marginalize_levels(&[vtree.leaf_of(VarId(1)).unwrap()]).unwrap();
    assert!(f.has_marginal_level());
    assert!(matches!(f.exists_var(VarId(2)), Err(OperationError::MarginalLevel(_))));
}

/// Every operation but `exists` on a region form against the smooth form;
/// `exists` of a leaf variable is refused on both.
#[test]
fn operations_on_regions_answer_as_the_smooth_form_on_homes_of_up_to_64_leaves() {
    let eng = Engine::new();
    let (mut cases, mut big) = (0, 0);
    for seed in 0..400u64 {
        let Some((home, n, f, g, _, _)) = region_case_and_source(&eng, seed, 64) else { continue };
        cases += 1;
        big += usize::from(n > 32);
        let mut rng = Lcg::new(seed);
        let what = format!("seed {seed}, {n} variables");
        let leaves: Vec<VarId> = (1..=n).map(VarId).filter(|&v| g.vtree().leaf_of(v).is_some()).collect();
        let inside: Vec<VarId> = (1..=n).map(VarId).filter(|&v| g.vtree().region_of(v).is_some()).collect();
        assert!(!inside.is_empty(), "{what}");
        let region = |v: VarId| OperationError::MarginalLevel(g.vtree().region_of(v).unwrap());
        for &v in &inside {
            let b = rng.coin();
            assert_eq!(g.clone().exists_var(v).err(), Some(region(v)), "{what}: exists {v:?}");
            assert_eq!(g.clone().condition_var(v, b).err(), Some(region(v)), "{what}: condition {v:?}");
        }
        if let Some(&x) = leaves.get(rng.below(leaves.len().max(1) as u64) as usize) {
            let b = rng.coin();
            let (got, expected) = (g.clone().condition_var(x, b).unwrap(), f.clone().condition_var(x, b).unwrap());
            check_form(&got, &what);
            assert!(same(&got, &expected), "{what}: condition {x:?}");
            assert!(matches!(g.clone().exists_var(x), Err(OperationError::MarginalLevel(_))), "{what}: exists {x:?}");
            assert!(matches!(f.clone().exists_var(x), Err(OperationError::MarginalLevel(_))), "{what}: exists {x:?}");
            let literal = crate::literal(g.vtree(), i32::try_from(x.0).unwrap()).unwrap();
            assert!(matches!(g.clone().substitute(&[(x, &literal)]), Err(OperationError::MarginalLevel(_))), "{what}: substitute");
        }
        let moved = eng.transport(&g, &home, |v| v, 0).map_err(|r| r.error).unwrap().0;
        check_form(&moved, &what);
        assert!(same(&moved, &f), "{what}: transport");
        if g.weights().is_none() {
            let mirrored: Vec<String> = home
                .to_text()
                .lines()
                .map(|line| match line.strip_prefix("L ").and_then(|rest| rest.split_once(' ')) {
                    Some((id, var)) => format!("L {id} {}", n + 1 - var.parse::<u32>().unwrap()),
                    None => line.to_string(),
                })
                .collect();
            let other = Home::new(Vtree::from_text(&mirrored.join("\n")).unwrap());
            let renamed = eng.transport(&g, &other, |v| VarId(n + 1 - v.0), 0).map_err(|r| r.error).unwrap().0;
            check_form(&renamed, &what);
            assert_eq!(renamed.model_count().unwrap(), g.model_count().unwrap(), "{what}: transport renamed");
            assert_eq!(renamed.vtree().regions().count(), g.vtree().regions().count(), "{what}");
        }
        for t in g.vtree().bottomup().filter(|&t| !g.vtree().node(t).is_leaf()) {
            let mut c = g.clone();
            eng.rotate_marginal_cluster(&mut c, t, 8, &mut Vec::new()).unwrap();
            check_form(&c, &what);
            assert!(Arc::ptr_eq(&c.vtree().home().unwrap(), &home), "{what}: the home rotated");
            assert!(same(&c, &f), "{what}: cluster at {t:?}");
        }
    }
    assert!(cases > 150 && big > 30, "{cases} cases, {big} over 32 leaves");
}

/// Home `((((1, 2), (3, 4)), 5), 6)`; `a` holds a region over `{1, 2}`, `b`
/// one over `{3, 4}`, so their conjunction holds two regions that are own
/// siblings.
fn sibling_regions(eng: &Engine) -> (Home, Tdd) {
    let leaf = |v: u32| Vtree::leaf(VarId(v));
    let join = |a: &Vtree, b: &Vtree| Vtree::join(a, b).unwrap();
    let four = join(&join(&leaf(1), &leaf(2)), &join(&leaf(3), &leaf(4)));
    let home = Home::new(join(&join(&four, &leaf(5)), &leaf(6)));
    let on = |vars: [u32; 3]| Vtree::projection(&home, vars.map(VarId)).unwrap();
    let mut a = Tdd::clause(&on([1, 2, 5]), [1, 2, 5]).unwrap();
    a.marginalize(|v| v.0 <= 2).unwrap();
    let mut b = Tdd::clause(&on([3, 4, 6]), [-3, 4, -6]).unwrap();
    b.marginalize(|v| (3..=4).contains(&v.0)).unwrap();
    (home, eng.and(&a, &b).unwrap())
}

#[test]
fn two_sibling_regions_merge_into_one_and_the_home_stays() {
    let eng = Engine::new();
    let (home, g) = sibling_regions(&eng);
    let regions: Vec<_> = g.vtree().regions().collect();
    assert_eq!(regions.len(), 2);
    let parent = g.vtree().node(regions[0]).parent().unwrap();
    assert_eq!(g.vtree().node(regions[1]).parent(), Some(parent), "the regions are not siblings");
    let mut merged = g.clone();
    eng.rotate_marginal_cluster(&mut merged, parent, 8, &mut Vec::new()).unwrap();
    check_form(&merged, "merged");
    let region = merged.vtree().regions().collect::<Vec<_>>();
    assert_eq!(region.len(), 1);
    let (_, extent) = merged.vtree().region_vars(region[0]);
    assert_eq!(extent, [1, 2, 3, 4].map(VarId));
    assert!(Arc::ptr_eq(&merged.vtree().home().unwrap(), &home));
    assert_eq!(merged.model_count().unwrap(), g.model_count().unwrap());
}

/// R3: once a condition leaves a region holding 2^|extent| in every
/// reference, trim drops it, and its variables take an operand again.
#[test]
fn trim_drops_a_region_that_holds_every_assignment_of_its_extent() {
    let leaf = |v: u32| Vtree::leaf(VarId(v));
    let home = Home::new(Vtree::join(&Vtree::join(&leaf(1), &leaf(2)).unwrap(), &leaf(3)).unwrap());
    let mut g = Tdd::clause(&Vtree::projection(&home, [1, 2, 3].map(VarId)).unwrap(), [1, 2, 3]).unwrap();
    g.marginalize(|v| v.0 <= 2).unwrap();
    assert_eq!(g.vtree().regions().count(), 1);
    assert!(matches!(g.clone().and_clause([1]), Err(OperationError::MarginalLevel(_))));
    let mut g = g.condition_var(VarId(3), true).unwrap();
    let count = g.model_count().unwrap();
    g.trim(&home).unwrap();
    check_form(&g, "R3");
    assert_eq!(g.vtree().regions().count(), 0);
    assert_eq!(g.model_count().unwrap(), count);
    assert_eq!(g.and_clause([1]).unwrap().model_count().unwrap(), count / BigUint::from(2u32));
}

#[test]
fn a_diagram_with_a_region_refuses_a_search_a_build_and_a_text_record() {
    let eng = Engine::new();
    let (_, g) = sibling_regions(&eng);
    let first = g.vtree().regions().next().unwrap();
    let mut s = g.clone();
    let searched = s.rotation_search(&mut MinimizePairs, &RotationSearchConfig::default());
    assert_eq!(searched.err(), Some(OperationError::MarginalLevel(first)));
    let builder = Tdd::builder(&eng, g.vtree()).unwrap();
    let built = builder.finish(TddNodeId { vtree: g.vtree().root(), local: NodeIdx(0) });
    assert!(matches!(built, Err(TddBuildError::StructuralRegion { level }) if level == first), "{built:?}");
    let text = g.vtree().to_text();
    assert!(text.contains(&format!("R {}", first.0)), "{text}");
    assert!(Vtree::from_text(&text).is_err());
}

