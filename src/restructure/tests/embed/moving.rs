//! `Engine::embed_moving` builds the diagram `Engine::embed` builds, up to
//! node numbering and unread literal nodes, deterministic before any
//! minimization, and gives its input back as it was at every point it can
//! be refused.
use std::sync::Arc;

use crate::limits::{LimitConfig, StopAt, StopRules};
use crate::test_helpers::check::check_determinism;
use crate::test_helpers::{assert_same_shape, compile_clauses, same_storage, test_cases, vtree_shapes};
use crate::vtree::{VarId, Vtree};
use crate::{Engine, Tdd};

/// Move `f` into `into` and check it against the copying embedding: the same
/// diagram once minimized, and the same placement of levels.
fn check_against_copy(eng: &Engine, f: &Tdd, into: &Arc<Vtree>, map: impl Fn(VarId) -> VarId + Copy, what: &str) {
    let (copied, copied_levels) = eng.embed(f, into, map).unwrap();
    let (mut moved, moved_levels) = eng.embed_moving(f.clone(), into, map).map_err(|r| r.error).unwrap();
    assert_eq!(moved_levels.levels, copied_levels.levels, "{what}");
    // Unminimized, the result is already a diagram every operation takes.
    check_determinism(&moved).unwrap_or_else(|e| panic!("{what}: {e}"));
    eng.minimize(&mut moved).unwrap();
    let mut copied = copied;
    eng.minimize(&mut copied).unwrap();
    assert_same_shape(&moved, &copied, what);
}

#[test]
fn every_shape_moves_to_the_diagram_embed_copies() {
    let eng = Engine::new();
    for (num_vars, clauses) in test_cases() {
        if num_vars > 4 {
            continue;
        }
        let shift = num_vars;
        let rename = move |v: VarId| VarId(v.0 + shift);
        let free: Vec<VarId> = (1..=num_vars).map(VarId).collect();
        for (label, small) in vtree_shapes(num_vars) {
            let mut f = compile_clauses(&small, &clauses);
            f.minimize().unwrap();
            let image = super::placement::renamed_shape(&small, rename, 2 * num_vars);
            let beside = Arc::new(Vtree::join(&image, &Vtree::balanced_over(&free).unwrap()).unwrap());
            check_against_copy(&eng, &f, &beside, rename, label);
        }
    }
}

#[test]
fn leaves_spread_along_a_spine_move_to_the_diagram_embed_copies() {
    // The spine's last node joins x9 to a free x10, a pass-through over a
    // leaf, so the moved result carries wrappers the copy prunes.
    let eng = Engine::new();
    let small = Arc::new(Vtree::linear(4));
    let big = Arc::new(Vtree::linear(10));
    let positions = [2u32, 4, 7, 9];
    for clauses in [vec![vec![1, 2], vec![-3, 4]], vec![vec![1, -4], vec![2, 3, 4]], vec![vec![4]]] {
        let mut f = compile_clauses(&small, &clauses);
        f.minimize().unwrap();
        check_against_copy(&eng, &f, &big, |v| VarId(positions[v.idx()]), "spine");
    }
}

#[test]
fn false_and_true_move() {
    let eng = Engine::new();
    let small = Arc::new(Vtree::linear(2));
    let big = Arc::new(Vtree::linear(5));
    for f in [eng.clause(&small, std::iter::empty::<i32>()).unwrap(), eng.cube(&small, std::iter::empty::<i32>()).unwrap()] {
        check_against_copy(&eng, &f, &big, |v| VarId(v.0 + 2), "constant");
    }
}

/// A diagram over x1..x6 on a spine, and a spine over x1..x12 that carries
/// it on the even variables: a pass-through at every other level, the
/// lowest over a leaf.
fn source_and_destination() -> (Tdd, Arc<Vtree>, impl Fn(VarId) -> VarId + Copy) {
    let small = Arc::new(Vtree::linear(6));
    let mut f = compile_clauses(&small, &[vec![1, 2, -5], vec![-1, 3], vec![2, -3, 6], vec![4, -6]]);
    f.minimize().unwrap();
    (f, Arc::new(Vtree::linear(12)), |v: VarId| VarId(2 * v.0))
}

/// Refuse the embedding at each point `arm` picks in turn, checking that
/// every refusal hands the diagram back unchanged, until one is granted.
fn refuse_at_every_point(arm: impl Fn(&Engine, u64) -> Option<crate::limits::LimitScope<'_>>) -> usize {
    let eng = Engine::new();
    let (f, big, map) = source_and_destination();
    let (expected, _) = eng.embed(&f, &big, map).unwrap();
    let mut refusals = 0;
    for n in 0.. {
        let scope = arm(&eng, n);
        let outcome = eng.embed_moving(f.clone(), &big, map);
        drop(scope);
        eng.limits().grant_every_reserve();
        match outcome {
            Ok((out, _)) => {
                assert!(eng.equivalent(&out, &expected).unwrap(), "granted at {n}: a different function");
                break;
            }
            Err(refused) => {
                assert!(same_storage(&refused.tdd, &f), "refused at {n}: the diagram changed");
                refusals += 1;
            }
        }
        assert!(n < 10_000, "never granted");
    }
    refusals
}

#[test]
fn every_refused_reserve_gives_the_diagram_back() {
    let refusals = refuse_at_every_point(|eng, n| {
        eng.limits().refuse_nth_reserve(n as u32);
        None
    });
    assert!(refusals >= 2, "only {refusals} refusal points");
}

#[test]
fn every_stop_gives_the_diagram_back() {
    let refusals = refuse_at_every_point(|eng, n| {
        let at = eng.limits().work_units() + n;
        let stop = StopRules { unconditional: Some(StopAt::WorkUnits(at)), after_pairs: None };
        Some(eng.limits().scope(LimitConfig::none().with_stop_rules(stop)))
    });
    assert!(refusals >= 1, "no stop point");
}

#[test]
fn a_structure_check_refusal_gives_the_diagram_back() {
    let eng = Engine::new();
    let (f, big, map) = source_and_destination();
    let narrow = Arc::new(Vtree::linear(3));
    let refused = eng.embed_moving(f.clone(), &narrow, map).unwrap_err();
    assert!(same_storage(&refused.tdd, &f));
    let refused = eng.embed_moving(f.clone(), &big, |v| VarId(v.0 + 100)).unwrap_err();
    assert!(same_storage(&refused.tdd, &f));
}

#[test]
fn a_moved_diagram_minimized_under_every_stop_keeps_its_function() {
    let eng = Engine::new();
    let (f, big, map) = source_and_destination();
    let (moved, _) = eng.embed_moving(f.clone(), &big, map).map_err(|r| r.error).unwrap();
    let (copied, _) = eng.embed(&f, &big, map).unwrap();
    let count = eng.model_count(&copied).unwrap();
    for n in 0..2000u64 {
        let mut g = moved.clone();
        let outcome = {
            let at = eng.limits().work_units() + n;
            let stop = StopRules { unconditional: Some(StopAt::WorkUnits(at)), after_pairs: None };
            let _scope = eng.limits().scope(LimitConfig::none().with_stop_rules(stop));
            eng.minimize(&mut g)
        };
        assert_eq!(eng.model_count(&g).unwrap(), count, "stopped at {n}");
        let mut h = g.clone();
        eng.minimize(&mut h).unwrap();
        assert!(eng.equivalent(&h, &copied).unwrap(), "stopped at {n}");
        if outcome.is_ok() {
            break;
        }
    }
}

#[test]
fn moved_diagrams_negate_and_disjoin_like_copies() {
    use crate::reduce::ReductionPlan;
    let eng = Engine::new();
    let mut failures = Vec::new();
    for (small, big, positions) in [
        (Arc::new(Vtree::linear(6)), Arc::new(Vtree::linear(12)), vec![2u32, 4, 6, 8, 10, 12]),
        (Arc::new(Vtree::linear(3)), Arc::new(Vtree::linear(4)), vec![1, 2, 3]),
        (Arc::new(Vtree::linear(3)), Arc::new(Vtree::linear(4)), vec![2, 3, 4]),
        (Arc::new(Vtree::linear(2)), Arc::new(Vtree::balanced(4)), vec![1, 3]),
        (Arc::new(Vtree::linear(2)), Arc::new(Vtree::balanced(4)), vec![2, 4]),
        // Chains of three pass-throughs on each side of the root.
        (Arc::new(Vtree::linear(2)), Arc::new(Vtree::balanced(8)), vec![1, 8]),
        // A chain from the source's only leaf up to the root.
        (Arc::new(Vtree::linear(1)), Arc::new(Vtree::linear(3)), vec![3]),
        (Arc::new(Vtree::linear(1)), Arc::new(Vtree::balanced(4)), vec![2]),
    ] {
        let n = small.num_vars() as i32;
        let mut sets: Vec<Vec<Vec<i32>>> = vec![vec![vec![n]], vec![vec![-n]], vec![]];
        if n >= 2 {
            sets.extend([vec![vec![1, 2]], vec![vec![-1, -2]], vec![vec![1], vec![-n]], vec![vec![1, n], vec![-1, -n]]]);
        }
        if n >= 3 { sets.push(vec![vec![1, 2, -3], vec![-1, 3]]); }
        for clauses in sets {
            let mut f = compile_clauses(&small, &clauses);
            f.minimize().unwrap();
            let map = |v: VarId| VarId(positions[v.idx()]);
            let (copied, _) = eng.embed(&f, &big, map).unwrap();
            let (moved, _) = eng.embed_moving(f.clone(), &big, map).map_err(|r| r.error).unwrap();
            if let Err(e) = check_determinism(&moved) {
                failures.push(format!("determinism {clauses:?} -> {positions:?}: {e}"));
            }
            let expect = eng.negate(copied.clone()).unwrap();
            let got = eng.negate_with(moved.clone(), ReductionPlan::Prune).unwrap();
            if !eng.equivalent(&got, &expect).unwrap() {
                failures.push(format!("negate_with {clauses:?} -> {positions:?}"));
            }
            let got = eng.negate(moved.clone()).unwrap();
            if !eng.equivalent(&got, &expect).unwrap() {
                failures.push(format!("negate {clauses:?} -> {positions:?}"));
            }
            let c = eng.clause(&big, [1, -2]).unwrap();
            if !eng.equivalent(&eng.and(moved.clone(), c.clone()).unwrap(), &eng.and(copied.clone(), c.clone()).unwrap()).unwrap() {
                failures.push(format!("and {clauses:?} -> {positions:?}"));
            }
            if !eng.equivalent(&eng.or(moved.clone(), c.clone()).unwrap(), &eng.or(copied.clone(), c).unwrap()).unwrap() {
                failures.push(format!("or {clauses:?} -> {positions:?}"));
            }
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// Each half of the clauses compiled on the vtree `vtree` projects to its
/// variables, minimized with `settled` and else a bare conjunction that
/// still owes its contraction, then moved onto `vtree`. A moved half owes
/// what its source owed and nothing for the levels its embedding built, and
/// it, its conjunction with a unit clause and the conjunction of the two
/// halves minimize from those worklists to the canonical diagram.
#[test]
fn moved_halves_conjoin_to_the_canonical_diagram() {
    use crate::test_helpers::{CnfShape, Lcg, rand_cnf};
    let eng = Engine::new();
    for seed in 0..60u64 {
        let num_vars = 6 + (seed % 5) as u32;
        let mut rng = Lcg::new(seed);
        let clauses = rand_cnf(&mut rng, num_vars, CnfShape { clauses: 14, width: 3 });
        let (a, b) = clauses.split_at(clauses.len() / 2);
        if a.is_empty() {
            continue;
        }
        for (label, vtree) in vtree_shapes(num_vars) {
            let expected = compile_clauses(&vtree, &clauses);
            let moved = |part: &[Vec<i32>], settled: bool| -> Tdd {
                let mut vars: Vec<u32> = part.iter().flatten().map(|l| l.unsigned_abs()).collect();
                vars.sort_unstable();
                vars.dedup();
                let local = |v: u32| vars.iter().position(|&w| w == v).unwrap() as i32 + 1;
                let small = Arc::new(
                    vtree
                        .project_to_vars(|v| vars.iter().position(|&w| w == v.0).map(|i| VarId(i as u32 + 1)), vars.len() as u32)
                        .unwrap(),
                );
                let mut f = eng.cube(&small, std::iter::empty::<i32>()).unwrap();
                for c in part {
                    let c = c.iter().map(|&l| if l < 0 { -local(l.unsigned_abs()) } else { local(l.unsigned_abs()) });
                    f = eng.and(f, eng.clause(&small, c).unwrap()).unwrap();
                }
                if settled {
                    eng.minimize(&mut f).unwrap();
                }
                eng.embed_moving(f, &vtree, |v| VarId(vars[v.0 as usize - 1])).map_err(|r| r.error).unwrap().0
            };
            let alone = compile_clauses(&vtree, a);
            // A unit clause leaves the conjunction most of the moved half's
            // levels to carry as they are.
            let unit = b[0][0];
            let mut with_unit = a.to_vec();
            with_unit.push(vec![unit]);
            let with_unit = compile_clauses(&vtree, &with_unit);
            for settled in [true, false] {
                let what = format!("seed {seed}, {label}, settled {settled}");
                let mut out = moved(a, settled);
                eng.minimize(&mut out).unwrap();
                assert_same_shape(&out, &alone, &format!("{what}, alone"));
                let mut out = eng.and(moved(a, settled), eng.clause(&vtree, [unit]).unwrap()).unwrap();
                eng.minimize(&mut out).unwrap();
                assert_same_shape(&out, &with_unit, &format!("{what}, with a unit"));
                let mut out = eng.and(moved(a, settled), moved(b, settled)).unwrap();
                eng.minimize(&mut out).unwrap();
                assert_same_shape(&out, &expected, &what);
            }
        }
    }
}
