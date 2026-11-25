//! `Plan::build` onto a projection matches leaves by home rank, onto a root
//! vtree by `leaf_of`, and climbs from the leaves; `Plan::build_as` without
//! the climb matches by `leaf_of` and passes over all of `into`. All give
//! the same plan, error and work.
use std::sync::Arc;

use crate::Engine;
use crate::restructure::embed::{Free, Plan, Target, climb_from_leaves};
use crate::vtree::rng::Lcg;
use crate::vtree::{Home, VarId, Vtree, VtreeNode};

/// `vars` in random order.
fn shuffled(mut vars: Vec<VarId>, rng: &mut Lcg) -> Vec<VarId> {
    for i in (1..vars.len()).rev() {
        vars.swap(i, rng.below(i as u64 + 1) as usize);
    }
    vars
}

/// A random subset of `vars`, of at least `min` of them.
fn some_of(vars: &[VarId], min: usize, rng: &mut Lcg) -> Vec<VarId> {
    let some: Vec<VarId> = vars.iter().copied().filter(|_| rng.coin()).collect();
    if some.len() >= min { some } else { vars[..min].to_vec() }
}

/// `into` as a root vtree.
fn root_copy(into: &Vtree) -> Vtree {
    let nodes = (0..into.num_nodes())
        .map(|i| match *into.node(crate::vtree::VtreeIdx(i as u32)) {
            VtreeNode::Leaf { var, .. } => VtreeNode::Leaf { var, parent: None },
            VtreeNode::Internal { left, right, .. } => VtreeNode::Internal { left, right, parent: None },
        })
        .collect();
    Vtree::from_nodes(nodes, into.root(), into.num_vars()).unwrap()
}

/// Plans built and refused onto a projection, and built with a mirrored
/// image.
#[derive(Default, Debug)]
struct Seen {
    built: u32,
    refused: u32,
    mirrored: u32,
}

/// Asserts that the climb from `want`'s childless images takes the plan
/// `want` and does not leave it to the pass.
fn assert_climbs(source: &Vtree, into: &Vtree, mirror: bool, want: &Plan, what: &str) {
    let mut targets = want.targets.clone();
    for &t in into.internal_bottomup_slice() {
        targets[t.idx()] = Target::Free;
    }
    let mut embedding = want.embedding.levels.clone();
    for &s in source.internal_bottomup_slice() {
        embedding[s.idx()] = into.root();
    }
    let matched = climb_from_leaves(source, into, mirror, &mut targets, &mut embedding).unwrap_or_else(|| panic!("{what}: the climb gave up"));
    assert_eq!(targets, want.targets, "{what}: climbed targets");
    assert_eq!(embedding, want.embedding.levels, "{what}: climbed embedding");
    assert_eq!((matched.mirrors, matched.pass_throughs), (want.mirrors, want.pass_throughs), "{what}: climbed counts");
}

/// Asserts that `Plan::build` and `Plan::build_as` without the climb agree,
/// work included, onto `into` and onto a root copy of it.
fn assert_same_build(eng: &Engine, source: &Vtree, into: &Vtree, map: impl Fn(VarId) -> VarId + Copy, seen: &mut Seen, what: &str) {
    let copy = root_copy(into);
    for (onto, how) in [(into, "projection"), (&copy, "root copy")] {
        for mirror in [false, true] {
            let what = format!("{what}, {how}, mirror {mirror}");
            let mark = eng.limits().mark();
            let want = Plan::build_as(eng, source, onto, map, mirror, Free::Leave, false);
            let want_work = eng.limits().work_since(mark);
            let mark = eng.limits().mark();
            let got = Plan::build(eng, source, onto, map, mirror, Free::Leave);
            assert_eq!(eng.limits().work_since(mark), want_work, "{what}: work");
            match (got, &want) {
                (Ok(got), Ok(want)) => {
                    assert_eq!(got.targets, want.targets, "{what}: targets");
                    assert_eq!(got.embedding.levels, want.embedding.levels, "{what}: embedding");
                    assert_eq!(got.mirrors, want.mirrors, "{what}: mirrors");
                    assert_eq!(got.pass_throughs, want.pass_throughs, "{what}: pass-throughs");
                }
                (Err(got), Err(want)) => assert_eq!(&got, want, "{what}: error"),
                (got, want) => panic!("{what}: {got:?} against {want:?}"),
            }
            if let Ok(want) = &want {
                assert_climbs(source, onto, mirror, want, &what);
            }
            match want {
                Ok(_) if !std::ptr::eq(onto, into) => {}
                Ok(plan) => {
                    seen.built += 1;
                    seen.mirrored += u32::from(plan.mirrors);
                }
                Err(_) => seen.refused += u32::from(std::ptr::eq(onto, into)),
            }
        }
    }
}

#[test]
fn a_match_by_rank_is_the_match_by_leaf() {
    let eng = Engine::new();
    let mut seen = Seen::default();
    for seed in 0..400u64 {
        let mut rng = Lcg::new(seed ^ 0x706c_616e);
        let n = 1 + rng.below(12) as u32;
        let home = Home::new(Vtree::random(n, seed));
        let all: Vec<VarId> = (1..=n).map(VarId).collect();
        let into_vars = some_of(&shuffled(all.clone(), &mut rng), 1, &mut rng);
        let into = Vtree::projection(&home, into_vars.iter().copied()).unwrap();
        let outside = all.iter().copied().find(|v| !into_vars.contains(v)).unwrap_or(VarId(n + 3));
        let what = format!("seed {seed}, home of {n}, into {into_vars:?}");

        // A projection of the same home, of another home of the same shape,
        // and the home's root, each under the identity.
        let inside = some_of(&shuffled(into_vars.clone(), &mut rng), 1, &mut rng);
        let stranger = Home::new(Vtree::clone(&home));
        let mut sources: Vec<(Arc<Vtree>, String)> = vec![
            (Vtree::projection(&home, inside.iter().copied()).unwrap(), "same home".into()),
            (Vtree::projection(&stranger, inside.iter().copied()).unwrap(), "other home".into()),
            (Arc::clone(&home), "home root".into()),
        ];
        let over = some_of(&shuffled(all.clone(), &mut rng), 1, &mut rng);
        sources.push((Vtree::projection(&home, over.iter().copied()).unwrap(), format!("same home {over:?}")));
        for (source, label) in &sources {
            assert_same_build(&eng, source, &into, |v| v, &mut seen, &format!("{what}, {label}, identity"));
        }

        // A renaming onto `into`, from a projection of the home and from
        // another root, then the same with one variable sent outside `into`
        // and with two sent onto one.
        let k = 1 + rng.below(into_vars.len() as u64) as usize;
        let images = shuffled(into_vars.clone(), &mut rng)[..k].to_vec();
        let from = shuffled(all.clone(), &mut rng)[..k].to_vec();
        let mut from_sorted = from.clone();
        from_sorted.sort();
        let renamed: Vec<(Arc<Vtree>, Vec<VarId>, String)> = vec![
            (Vtree::projection(&home, from.iter().copied()).unwrap(), from_sorted, "projection".into()),
            (Arc::new(Vtree::random(k as u32, seed + 7)), (1..=k as u32).map(VarId).collect(), "other root".into()),
        ];
        for (source, vars, label) in &renamed {
            let image = |v: VarId| images[vars.iter().position(|&w| w == v).unwrap()];
            assert_same_build(&eng, source, &into, image, &mut seen, &format!("{what}, {label}, renaming {images:?}"));
            let bad = rng.below(k as u64) as usize;
            let escape = |v: VarId| if v == vars[bad] { outside } else { image(v) };
            assert_same_build(&eng, source, &into, escape, &mut seen, &format!("{what}, {label}, {:?} to {outside:?}", vars[bad]));
            if k > 1 {
                let other = (bad + 1 + rng.below(k as u64 - 1) as usize) % k;
                let clash = |v: VarId| if v == vars[bad] { images[other] } else { image(v) };
                assert_same_build(&eng, source, &into, clash, &mut seen, &format!("{what}, {label}, {:?} onto {:?}", vars[bad], images[other]));
            }
        }
    }
    assert!(seen.built > 400 && seen.refused > 400 && seen.mirrored > 20, "{seen:?}");
}
