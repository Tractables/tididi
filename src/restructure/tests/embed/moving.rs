//! `Engine::embed_moving` builds the diagram `Engine::embed` builds, up to
//! the leaf wrappers minimization removes, and gives its input back as it
//! was at every point it can be refused.
use std::sync::Arc;

use crate::limits::{LimitConfig, StopAt, StopRules};
use crate::test_helpers::{assert_same_shape, compile_clauses, same_storage, test_cases, vtree_shapes};
use crate::vtree::{VarId, Vtree};
use crate::{Engine, Tdd};

/// Move `f` into `into` and check it against the copying embedding: the same
/// diagram once minimized, and the same placement of levels.
fn check_against_copy(eng: &Engine, f: &Tdd, into: &Arc<Vtree>, map: impl Fn(VarId) -> VarId + Copy, what: &str) {
    let (copied, copied_levels) = eng.embed(f, into, map).unwrap();
    let (mut moved, moved_levels) = eng.embed_moving(f.clone(), into, map).map_err(|r| r.error).unwrap();
    assert_eq!(moved_levels.levels, copied_levels.levels, "{what}");
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
