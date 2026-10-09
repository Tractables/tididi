//! `Engine::and_onto` builds what `Engine::embed_moving` on each operand and
//! `Engine::and_restoring` on the two build, node for node, for their work
//! less the free levels it does not charge, and gives each operand back, as
//! it was given or as it was placed, at every point it can be refused.
use std::sync::Arc;

use crate::limits::{LimitConfig, LimitScope, StopAt, StopRules};
use crate::test_helpers::{compile_clauses, same_storage, test_cases, vtree_shapes};
use crate::vtree::{VarId, Vtree};
use crate::{Engine, Tdd};

use super::placement::renamed_shape;

/// A renaming onto the destination.
trait Map: Fn(VarId) -> VarId + Copy {}
impl<M: Fn(VarId) -> VarId + Copy> Map for M {}

/// `tdd` on `into`: as it is when it is there already, else moved there.
fn placed(eng: &Engine, tdd: Tdd, map: impl Map, into: &Arc<Vtree>) -> Tdd {
    if Arc::ptr_eq(tdd.vtree(), into) {
        return tdd;
    }
    eng.embed_moving(tdd, into, map).map_err(|r| r.error).unwrap().0
}

/// What `and_onto` stands for: each operand placed, then conjoined.
fn by_embeddings(eng: &Engine, f: &Tdd, f_map: impl Map, g: &Tdd, g_map: impl Map, into: &Arc<Vtree>) -> Tdd {
    let (f, g) = (placed(eng, f.clone(), f_map, into), placed(eng, g.clone(), g_map, into));
    eng.and_restoring(f, g).map_err(|r| r.error).unwrap()
}

/// The levels of `into` a placement of `tdd` by `map` leaves free: the
/// internal nodes none of its variables is under. None for a diagram on
/// `into` already, which is not placed, or for the false diagram, which has
/// no levels.
fn free_levels(tdd: &Tdd, map: impl Map, into: &Arc<Vtree>) -> u64 {
    if Arc::ptr_eq(tdd.vtree(), into) || tdd.is_zero() {
        return 0;
    }
    let source = tdd.vtree();
    let mut held = vec![false; into.num_nodes()];
    for t in into.bottomup() {
        held[t.idx()] = match into.node(t).is_leaf() {
            true => source.bottomup().any(|s| source.node(s).is_leaf() && map(source.leaf_var(s)) == into.leaf_var(t)),
            false => {
                let (left, right) = into.children(t);
                held[left.idx()] || held[right.idx()]
            }
        };
    }
    into.bottomup().filter(|&t| !into.node(t).is_leaf() && !held[t.idx()]).count() as u64
}

/// Check `and_onto` against the embeddings and conjunction it stands for:
/// the same storage and worklists, for their work less the free levels.
fn check(eng: &Engine, f: &Tdd, f_map: impl Map, g: &Tdd, g_map: impl Map, into: &Arc<Vtree>, what: &str) {
    let mark = eng.limits().mark();
    let expected = by_embeddings(eng, f, f_map, g, g_map, into);
    let expected_work = eng.limits().work_since(mark);
    let mark = eng.limits().mark();
    let got = eng.and_onto(f.clone(), f_map, g.clone(), g_map, into).map_err(|r| r.error).unwrap();
    let work = eng.limits().work_since(mark);
    assert!(same_storage(&got, &expected), "{what}: a different diagram");
    assert_eq!(format!("{:?}", got.dirty), format!("{:?}", expected.dirty), "{what}: worklists");
    let free = free_levels(f, f_map, into) + free_levels(g, g_map, into);
    assert_eq!(work + free, expected_work, "{what}: work");
}

/// `clauses` with every literal negated.
fn negated(clauses: &[Vec<i32>]) -> Vec<Vec<i32>> {
    clauses.iter().map(|c| c.iter().map(|l| -l).collect()).collect()
}

#[test]
fn every_shape_conjoins_onto_as_embeddings_and_conjunction_do() {
    let eng = Engine::new();
    for (num_vars, clauses) in test_cases() {
        if num_vars > 4 {
            continue;
        }
        let space = 2 * num_vars + 2;
        let spare = Vtree::balanced_over(&[VarId(space - 1), VarId(space)]).unwrap();
        for (label, small) in vtree_shapes(num_vars) {
            let minimized = |clauses: &[Vec<i32>]| {
                let mut f = compile_clauses(&small, clauses);
                f.minimize().unwrap();
                f
            };
            let (f, g) = (minimized(&clauses), minimized(&negated(&clauses)));
            let low = |v: VarId| v;
            let high = move |v: VarId| VarId(v.0 + num_vars);
            let image_low = renamed_shape(&small, low, space);
            let image_high = renamed_shape(&small, high, space);

            // Disjoint supports, with two variables neither has.
            let apart = Arc::new(Vtree::join(&Vtree::join(&image_low, &spare).unwrap(), &image_high).unwrap());
            check(&eng, &f, low, &g, high, &apart, &format!("{label}: apart"));
            check(&eng, &g, high, &f, low, &apart, &format!("{label}: apart, swapped"));
            check(&eng, &f, low, &f, high, &apart, &format!("{label}: apart, one function"));

            // One support, free levels in both operands at the same places.
            let shared = Arc::new(Vtree::join(&spare, &image_low).unwrap());
            check(&eng, &f, low, &g, low, &shared, &format!("{label}: shared"));
            check(&eng, &f, low, &f, low, &shared, &format!("{label}: shared, one function"));

            // An operand on the destination already: its levels all built.
            let on = placed(&eng, f.clone(), low, &shared);
            check(&eng, &f, low, &on, low, &shared, &format!("{label}: one placed, one function"));
            check(&eng, &on, low, &f, low, &shared, &format!("{label}: one placed, one function, swapped"));
            check(&eng, &g, low, &on, low, &shared, &format!("{label}: one placed"));
            let other = placed(&eng, g.clone(), low, &shared);
            check(&eng, &on, low, &other, low, &shared, &format!("{label}: both placed"));
        }
    }
}

#[test]
fn constants_conjoin_onto() {
    let eng = Engine::new();
    let small = Arc::new(Vtree::linear(3));
    let mut f = compile_clauses(&small, &[vec![1, -2], vec![2, 3]]);
    f.minimize().unwrap();
    let falsum = eng.clause(&small, std::iter::empty::<i32>()).unwrap();
    let verum = eng.cube(&small, std::iter::empty::<i32>()).unwrap();
    let low = |v: VarId| v;
    let shift = |v: VarId| VarId(v.0 + 5);
    let spare = Vtree::balanced_over(&[VarId(4), VarId(5)]).unwrap();
    let low_on = Vtree::join(&renamed_shape(&small, low, 8), &spare).unwrap();
    let into = Arc::new(Vtree::join(&low_on, &renamed_shape(&small, shift, 8)).unwrap());
    for (c, what) in [(&falsum, "false"), (&verum, "true")] {
        check(&eng, c, low, &f, shift, &into, what);
        check(&eng, &f, shift, c, low, &into, what);
        check(&eng, c, low, c, shift, &into, what);
    }
}

/// A diagram over x1..x8 and one over x5..x8 and x13..x16, both on a
/// balanced vtree over 16 that each covers in part: free levels in both,
/// and product levels where they share variables.
fn overlapping() -> (Tdd, Tdd, Arc<Vtree>) {
    let vars = |range: std::ops::RangeInclusive<u32>| range.map(VarId).collect::<Vec<_>>();
    let a = Arc::new(Vtree::balanced_over(&vars(1..=8)).unwrap());
    let b = Arc::new(
        Vtree::join(&Vtree::balanced_over(&vars(5..=8)).unwrap(), &Vtree::balanced_over(&vars(13..=16)).unwrap())
            .unwrap(),
    );
    let mut f = compile_clauses(&a, &[vec![1, 2, -7], vec![-1, 3], vec![2, -3, 8], vec![4, 5], vec![-5, 6, 7]]);
    let mut g = compile_clauses(&b, &[vec![7, 8, -13], vec![-8, 14], vec![13, -14, 15], vec![-15, 16, 5], vec![6, -16]]);
    f.minimize().unwrap();
    g.minimize().unwrap();
    (f, g, Arc::new(Vtree::balanced(16)))
}

#[test]
fn overlapping_supports_conjoin_onto() {
    let eng = Engine::new();
    let (f, g, into) = overlapping();
    let same = |v: VarId| v;
    check(&eng, &f, same, &g, same, &into, "f, g");
    check(&eng, &g, same, &f, same, &into, "g, f");
}

/// Where a refusal left the two operands: whether each is on the
/// destination.
type Placed = (bool, bool);

/// Refuse `and_onto` at each point `arm` picks in turn until it is granted:
/// every refusal gives each operand back as it was given or as it was
/// placed, and the grant is the diagram the embeddings and conjunction make.
/// With `no_earlier`, the embeddings and conjunction refused at the same
/// point are refused at the same phase or an earlier one: `and_onto` charges
/// less work. Returns where the refusals left the operands.
fn refuse_at_every_point(arm: impl Fn(&Engine, u64) -> Option<LimitScope<'_>>, no_earlier: bool) -> Vec<Placed> {
    let eng = Engine::new();
    let (f, g, into) = overlapping();
    let same = |v: VarId| v;
    let expected = by_embeddings(&eng, &f, same, &g, same, &into);
    let (f_on, g_on) = (placed(&eng, f.clone(), same, &into), placed(&eng, g.clone(), same, &into));
    let mut seen = Vec::new();
    for n in 0.. {
        let scope = arm(&eng, n);
        let outcome = eng.and_onto(f.clone(), same, g.clone(), same, &into);
        drop(scope);
        eng.limits().grant_every_reserve();
        let old = {
            let scope = arm(&eng, n);
            let old = eng
                .embed_moving(f.clone(), &into, same)
                .map_err(|_| (false, false))
                .and_then(|(f, _)| match eng.embed_moving(g.clone(), &into, same) {
                    Ok((g, _)) => eng.and_restoring(f, g).map_err(|_| (true, true)),
                    Err(_) => Err((true, false)),
                });
            drop(scope);
            eng.limits().grant_every_reserve();
            old
        };
        match outcome {
            Ok(out) => {
                assert!(same_storage(&out, &expected), "granted at {n}: a different diagram");
                break;
            }
            Err(refused) => {
                let back = |got: &Tdd, given: &Tdd, on: &Tdd| match Arc::ptr_eq(got.vtree(), &into) {
                    true => same_storage(got, on),
                    false => same_storage(got, given),
                };
                assert!(back(&refused.f, &f, &f_on) && back(&refused.g, &g, &g_on), "refused at {n}: an operand changed");
                let at = (Arc::ptr_eq(refused.f.vtree(), &into), Arc::ptr_eq(refused.g.vtree(), &into));
                if no_earlier {
                    let old = old.err().unwrap_or_else(|| panic!("refused at {n}, where the embeddings and conjunction are not"));
                    let phase = |(f, g): Placed| u8::from(f) + u8::from(g);
                    assert!(phase(old) <= phase(at), "refused at {n}: before the embeddings and conjunction are");
                }
                seen.push(at);
            }
        }
        assert!(n < 100_000, "never granted");
    }
    seen
}

#[test]
fn every_refused_reserve_gives_the_operands_back() {
    let seen = refuse_at_every_point(
        |eng, n| {
            eng.limits().refuse_nth_reserve(n as u32);
            None
        },
        false,
    );
    for phase in [(false, false), (true, false), (true, true)] {
        assert!(seen.contains(&phase), "no reserve refused with the operands at {phase:?}");
    }
}

#[test]
fn every_stop_gives_the_operands_back_no_earlier_than_the_embeddings_and_conjunction_stop() {
    let seen = refuse_at_every_point(
        |eng, n| {
            let at = eng.limits().work_units() + n;
            let stop = StopRules { unconditional: Some(StopAt::WorkUnits(at)), after_pairs: None };
            Some(eng.limits().scope(LimitConfig::none().with_stop_rules(stop)))
        },
        true,
    );
    for phase in [(false, false), (true, false), (true, true)] {
        assert!(seen.contains(&phase), "no stop with the operands at {phase:?}");
    }
}

/// Every output-node cap refuses `and_onto` where it refuses the embeddings
/// and conjunction it stands for, after their work less the free levels:
/// the conjunction counts the nodes of the levels it takes from an operand's
/// free regions as if it had carried them in their places.
#[test]
fn every_output_cap_refuses_onto_where_it_refuses_the_embeddings_and_conjunction() {
    let eng = Engine::new();
    for (f, g, into) in [overlapping(), {
        let (f, g, into) = overlapping();
        (g, f, into)
    }] {
        let same = |v: VarId| v;
        let free = free_levels(&f, same, &into) + free_levels(&g, same, &into);
        let mut refused = 0;
        for cap in 0.. {
            let scope = eng.limits().scope(LimitConfig::none().with_output_node_cap(Some(cap)));
            let mark = eng.limits().mark();
            let (f_on, g_on) = (placed(&eng, f.clone(), same, &into), placed(&eng, g.clone(), same, &into));
            let expected = eng.and_restoring(f_on, g_on);
            let expected_work = eng.limits().work_since(mark);
            let mark = eng.limits().mark();
            let got = eng.and_onto(f.clone(), same, g.clone(), same, &into);
            let work = eng.limits().work_since(mark);
            drop(scope);
            match (got, expected) {
                (Ok(got), Ok(expected)) => {
                    assert!(same_storage(&got, &expected), "cap {cap}: a different diagram");
                    break;
                }
                (Err(got), Err(expected)) => {
                    assert_eq!(got.error, expected.error.into(), "cap {cap}");
                    assert_eq!(work + free, expected_work, "cap {cap}: work");
                    refused += 1;
                }
                (got, expected) => panic!(
                    "cap {cap}: and_onto refused {}, the embeddings and conjunction {}",
                    got.is_err(),
                    expected.is_err(),
                ),
            }
            assert!(cap < 100_000, "never granted");
        }
        assert!(refused > 0, "no cap refused");
    }
}
