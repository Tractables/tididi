//! A conjunction that borrows an operand builds what the conjunction of the
//! same operands owned builds: the same diagram, stored alike, with the same
//! worklists, kept counts, certificate and closed form, for the same work,
//! minimized alike, whichever operand is borrowed and on whichever route; it
//! leaves a borrowed operand as it was. A refused conjunction hands an owned
//! operand back as it was given, certificate and closed form included, and
//! a borrowed one as the borrow, refused at the same work as the owned one.

use std::borrow::Cow;
use std::sync::Arc;

use super::*;
use crate::Engine;
use crate::diagram::{Arithmetic, LiteralWeights, RationalWeights, WeightStore};
use crate::limits::{LimitConfig, SparseRoute, StopAt, StopRules};
use crate::test_helpers::{compile_clauses_on, rand_cnf, rat, same_storage, vtree_shapes, CnfShape, Lcg};
use crate::vtree::{VarId, Vtree};

/// Every level the sparse route can take, takes it.
const SPARSE: SparseRoute = SparseRoute { sparsity: 1, min_grid: 0 };

/// Which operands a conjunction borrows: f, g, both.
const BORROWS: [(bool, bool); 3] = [(true, false), (false, true), (true, true)];

/// `t` owned (a clone) or borrowed.
fn given(t: &Tdd, borrowed: bool) -> Cow<'_, Tdd> {
    match borrowed {
        true => Cow::Borrowed(t),
        false => Cow::Owned(t.clone()),
    }
}

/// Whether `back` is what was given as `given(t, borrowed)`: the borrow
/// itself, or an owned diagram stored as `t` and known as `t` is
/// ([`outcome`]).
#[expect(clippy::ptr_arg, reason = "which form came back is what is tested")]
fn handed_back(back: &Cow<'_, Tdd>, t: &Tdd, borrowed: bool) -> bool {
    match (back, borrowed) {
        (Cow::Borrowed(back), true) => std::ptr::eq(*back, t),
        (Cow::Owned(back), false) => same_storage(back, t) && outcome(back, 0) == outcome(t, 0),
        _ => false,
    }
}

/// Pairs of operands on every vtree shape over `nvars` variables, an even
/// number: random formulas; the first implying the second, where the leaf
/// arm builds levels that are f's; formulas over the two halves of the
/// variables, whose levels the identity fast paths carry; a cube against a
/// formula, for the relabelling route; a formula and itself; and a formula
/// and the false diagram.
fn cases(nvars: u32, seed: u64) -> Vec<(String, Tdd, Tdd)> {
    let mut rng = Lcg::new(seed);
    let mut out = Vec::new();
    let half = (nvars / 2) as i32;
    let cnf = |rng: &mut Lcg, clauses: usize, width: usize| rand_cnf(rng, nvars, CnfShape { clauses, width });
    let shift = |clauses: &[Vec<i32>], by: i32| -> Vec<Vec<i32>> {
        clauses.iter().map(|c| c.iter().map(|l| l.signum() * ((l.abs() - 1) % half + 1 + by)).collect()).collect()
    };
    for (name, vtree) in vtree_shapes(nvars) {
        let eng = Engine::new();
        let minimized = |clauses: &[Vec<i32>]| {
            let mut f = compile_clauses_on(&eng, &vtree, clauses);
            eng.minimize(&mut f).unwrap();
            f
        };
        for case in 0..3 {
            let (a, b) = (cnf(&mut rng, 2 * nvars as usize, 3), cnf(&mut rng, 2 * nvars as usize, 3));
            out.push((format!("{name}, {nvars} variables, random {case}"), minimized(&a), minimized(&b)));
            let more = cnf(&mut rng, 1 + case, 2);
            out.push((format!("{name}, {nvars} variables, implying {case}"), minimized(&[a.clone(), more].concat()), minimized(&a)));
            out.push((format!("{name}, {nvars} variables, halves {case}"), minimized(&shift(&a, 0)), minimized(&shift(&b, half))));
        }
        let f = minimized(&cnf(&mut rng, 2 * nvars as usize, 3));
        let cube = eng.cube(&vtree, [1, -(nvars as i32)]).unwrap();
        let zero = eng.and(eng.literal(&vtree, 1).unwrap(), eng.literal(&vtree, -1).unwrap()).unwrap();
        out.push((format!("{name}, {nvars} variables, cube"), f.clone(), cube));
        out.push((format!("{name}, {nvars} variables, itself"), f.clone(), f.clone()));
        out.push((format!("{name}, {nvars} variables, false"), f, zero));
    }
    out
}

/// What a conjunction leaves besides its levels: its worklists, whether it
/// keeps level counts, whether it is certified canonical and its levels
/// closed, and the work it took.
fn outcome(out: &Tdd, work: u64) -> (String, bool, bool, bool, u64) {
    (format!("{:?}", out.dirty), out.has_level_counts(), out.levels.is_canonical(out.output), out.levels.is_closed(), work)
}

/// Run `op` on a fresh engine, every level on the sparse route it can take
/// where `sparse` says, with the work it took.
fn on_fresh<R>(sparse: bool, op: impl FnOnce(&Engine) -> R) -> (R, u64) {
    let eng = Engine::new();
    let config = if sparse { LimitConfig::none().with_sparse_route(SPARSE) } else { LimitConfig::none() };
    let _scope = eng.limits().scope(config);
    let mark = eng.limits().mark();
    let out = op(&eng);
    (out, eng.limits().work_since(mark))
}

/// `out` minimized on a fresh engine, with the work it took.
fn minimized(out: &Tdd) -> (Tdd, u64) {
    on_fresh(false, |eng| {
        let mut out = out.clone();
        eng.minimize(&mut out).unwrap();
        out
    })
}

/// [`Engine::and`] and [`Engine::and_restoring`] with each operand borrowed
/// in turn and both, both ways round, against the same call with both
/// owned; with `counts`, on operands keeping level counts. The number of
/// conjunctions checked.
fn check_against_owned(cases: &[(String, Tdd, Tdd)], counts: bool) -> usize {
    let mut checked = 0;
    for (what, f, g) in cases {
        let (mut f, mut g) = (f.clone(), g.clone());
        if counts {
            let eng = Engine::new();
            eng.attach_level_counts(&mut f).unwrap();
            eng.attach_level_counts(&mut g).unwrap();
        }
        for (a, b, order) in [(&f, &g, "f and g"), (&g, &f, "g and f")] {
            let (a_before, b_before) = (a.clone(), b.clone());
            for sparse in [false, true] {
                for restoring in [false, true] {
                    let conjoin = |eng: &Engine, x: Cow<'_, Tdd>, y: Cow<'_, Tdd>| match restoring {
                        true => eng.and_restoring(x, y).map_err(|r| r.error).unwrap(),
                        false => eng.and(x, y).unwrap(),
                    };
                    let (owned, owned_work) = on_fresh(sparse, |eng| conjoin(eng, given(a, false), given(b, false)));
                    let (owned_min, owned_min_work) = minimized(&owned);
                    for (borrow_a, borrow_b) in BORROWS {
                        let at = format!("{what}, {order}, sparse {sparse}, restoring {restoring}, borrowing ({borrow_a}, {borrow_b})");
                        let (out, work) = on_fresh(sparse, |eng| conjoin(eng, given(a, borrow_a), given(b, borrow_b)));
                        assert!(same_storage(&out, &owned), "{at}: a different diagram");
                        assert_eq!(outcome(&out, work), outcome(&owned, owned_work), "{at}: worklists, kept counts, certificate, closed form or work");
                        let (out_min, out_min_work) = minimized(&out);
                        assert!(same_storage(&out_min, &owned_min), "{at}: minimized to a different diagram");
                        assert_eq!(out_min_work, owned_min_work, "{at}: minimized for other work");
                        assert!(same_storage(a, &a_before) && same_storage(b, &b_before), "{at}: a borrowed operand changed");
                        checked += 1;
                    }
                }
            }
        }
    }
    checked
}

#[test]
fn a_borrowed_operand_conjoins_as_an_owned_one() {
    let before = lent_levels();
    let mut checked = 0;
    for nvars in [6, 10, 14] {
        checked += check_against_owned(&cases(nvars, 0xb0_77ed + u64::from(nvars)), false);
    }
    assert!(checked > 1000, "only {checked} conjunctions");
    assert!(lent_levels() > before, "no level the leaf arm made f's: the copy of a borrowed f went untested");
}

#[test]
fn a_borrowed_operand_conjoins_as_an_owned_one_keeping_counts() {
    assert!(check_against_owned(&cases(10, 0xc0_0475), true) > 100);
}

/// A weighted operand, or one with a marginal level, is copied whole first:
/// the conjunction is the owned one's, for the same work.
#[test]
fn a_weighted_or_marginal_borrowed_operand_is_copied_first() {
    let eng = Engine::new();
    let vtree = Arc::new(Vtree::balanced(8));
    let f = compile_clauses_on(&eng, &vtree, &[vec![1, 2, -3], vec![3, 4], vec![-5, 6, 7], vec![8, -1]]);
    let g = compile_clauses_on(&eng, &vtree, &[vec![2, 5], vec![-4, 6], vec![7, 8, -2], vec![1, -6]]);
    let values: Vec<LiteralWeights<_>> = (0..8).map(|i| LiteralWeights { negative: rat(i + 1, 1), positive: rat(i + 2, 1) }).collect();
    let store = WeightStore::new(RationalWeights::from_literals(&values), Arithmetic::ExactRational);
    let (mut f_weighted, mut g_weighted) = (f.clone(), g.clone());
    f_weighted.set_weights(store.clone()).unwrap();
    g_weighted.set_weights(store).unwrap();
    // g with the variables 1 to 4 summed out, and a formula over the others.
    let mut marginal = g.clone();
    eng.marginalize_levels(&mut marginal, &[vtree.children(vtree.root()).0]).unwrap();
    assert!(marginal.has_marginal_level());
    let right = compile_clauses_on(&eng, &vtree, &[vec![5, 6], vec![-7, 8], vec![6, -8]]);
    for (a, b, what) in [(&f_weighted, &g_weighted, "weighted"), (&right, &marginal, "marginal")] {
        for (x, y) in [(a, b), (b, a)] {
            let (owned, owned_work) = on_fresh(false, |eng| eng.and(x.clone(), y.clone()).unwrap());
            for (borrow_x, borrow_y) in BORROWS {
                let at = format!("{what}, borrowing ({borrow_x}, {borrow_y})");
                let (out, work) = on_fresh(false, |eng| eng.and(given(x, borrow_x), given(y, borrow_y)).unwrap());
                assert!(same_storage(&out, &owned), "{at}: a different diagram");
                assert_eq!(outcome(&out, work), outcome(&owned, owned_work), "{at}: worklists, kept counts, certificate, closed form or work");
            }
        }
    }
}

/// A restoring conjunction refused at each unit of work in turn, and at
/// each reserve in turn: each refusal hands back what it was given, and at
/// each unit of work it is refused where the owned one is.
#[test]
fn a_refused_conjunction_hands_back_what_it_was_given() {
    let mut refusals = 0;
    for (what, f, g) in cases(10, 0x7ef_05ed).into_iter().step_by(4) {
        for (a, b) in [(&f, &g), (&g, &f)] {
            for (borrow_a, borrow_b) in BORROWS {
                for arm in ["work", "reserve"] {
                    for n in 0u64.. {
                        let attempt = |borrow: (bool, bool)| {
                            let eng = Engine::new();
                            let stop = StopRules { unconditional: Some(StopAt::WorkUnits(eng.limits().work_units() + n)), after_pairs: None };
                            let config = if arm == "work" { LimitConfig::none().with_stop_rules(stop) } else { LimitConfig::none() };
                            let _scope = eng.limits().scope(config);
                            if arm == "reserve" {
                                eng.limits().refuse_nth_reserve(n as u32);
                            }
                            eng.and_restoring(given(a, borrow.0), given(b, borrow.1)).map(drop).map_err(|r| {
                                assert!(
                                    handed_back(&r.f, a, borrow.0) && handed_back(&r.g, b, borrow.1),
                                    "{what}, borrowing {borrow:?}: refused at {arm} {n}, not handed back as given",
                                );
                            })
                        };
                        let outcome = attempt((borrow_a, borrow_b));
                        if arm == "work" {
                            assert_eq!(outcome.is_ok(), attempt((false, false)).is_ok(), "{what}: at work {n}, refused unlike the owned one");
                        }
                        if outcome.is_ok() {
                            break;
                        }
                        refusals += 1;
                        assert!(n < 100_000, "{what}: never granted");
                    }
                }
            }
        }
    }
    assert!(refusals > 100, "only {refusals} refusals");
}

/// [`Engine::and_onto`] with each operand borrowed in turn and both, one on
/// the destination and one placed there, against the same call with both
/// owned; refused at each unit of work, it is refused where the owned one
/// is and hands back what it was given: a borrowed operand as the borrow,
/// an owned one as it was or placed.
#[test]
fn a_borrowed_operand_conjoins_onto_as_an_owned_one() {
    let eng = Engine::new();
    let small = Arc::new(Vtree::balanced(4));
    let into = Arc::new(Vtree::balanced(8));
    let mut rng = Lcg::new(0x0170);
    let mut checked = 0;
    for case in 0..8u32 {
        let on_small = compile_clauses_on(&eng, &small, &rand_cnf(&mut rng, 4, CnfShape { clauses: 6, width: 3 }));
        let on_into = compile_clauses_on(&eng, &into, &rand_cnf(&mut rng, 8, CnfShape { clauses: 12, width: 3 }));
        let offset = 4 * (case % 2);
        let map = |t: &Tdd| {
            let there = Arc::ptr_eq(t.vtree(), &into);
            move |v: VarId| if there { v } else { VarId(v.0 + offset) }
        };
        let placed = |t: &Tdd| match Arc::ptr_eq(t.vtree(), &into) {
            true => t.clone(),
            false => eng.embed_moving(t.clone(), &into, map(t)).map_err(|r| r.error).unwrap().0,
        };
        for (a, b, what) in [(&on_small, &on_into, "placed and on into"), (&on_into, &on_small, "on into and placed")] {
            let (owned, owned_work) = on_fresh(false, |eng| eng.and_onto(given(a, false), map(a), given(b, false), map(b), &into).map_err(|r| r.error).unwrap());
            for (borrow_a, borrow_b) in BORROWS {
                let at = format!("case {case}, {what}, borrowing ({borrow_a}, {borrow_b})");
                let (out, work) = on_fresh(false, |eng| eng.and_onto(given(a, borrow_a), map(a), given(b, borrow_b), map(b), &into).map_err(|r| r.error).unwrap());
                assert!(same_storage(&out, &owned), "{at}: a different diagram");
                assert_eq!(outcome(&out, work), outcome(&owned, owned_work), "{at}: worklists, kept counts, certificate, closed form or work");
                for n in 0..=owned_work {
                    let refused_at = |borrow: (bool, bool)| {
                        let eng = Engine::new();
                        let stop = StopRules { unconditional: Some(StopAt::WorkUnits(eng.limits().work_units() + n)), after_pairs: None };
                        let _scope = eng.limits().scope(LimitConfig::none().with_stop_rules(stop));
                        let refused = eng.and_onto(given(a, borrow.0), map(a), given(b, borrow.1), map(b), &into).err();
                        if let Some(r) = &refused {
                            for (back, t, borrowed) in [(&r.f, a, borrow.0), (&r.g, b, borrow.1)] {
                                let as_placed = !borrowed && matches!(back, Cow::Owned(back) if same_storage(back, &placed(t)));
                                assert!(handed_back(back, t, borrowed) || as_placed, "{at}: refused at {n}, an operand handed back neither as given nor placed");
                            }
                        }
                        refused.is_some()
                    };
                    assert_eq!(refused_at((borrow_a, borrow_b)), refused_at((false, false)), "{at}: at work {n}, refused unlike the owned one");
                }
                checked += 1;
            }
        }
    }
    assert_eq!(checked, 48);
}
