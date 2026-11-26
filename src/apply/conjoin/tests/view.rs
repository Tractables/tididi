//! [`conjoin_onto`] reads its plain operands through views: it builds
//! what placing them and conjoining the placed operands builds, for the same
//! work, is refused where that is refused, and hands back every operand as
//! it was given or, owned and placed rather than viewed, placed.

use std::borrow::Cow;
use std::sync::Arc;

use super::*;
use crate::Engine;
use crate::limits::{LimitConfig, StopAt, StopRules};
use crate::test_helpers::{compile_clauses_on, rand_cnf, same_storage, CnfShape, Lcg};
use crate::vtree::{Home, VarId, Vtree};

/// Which operands a call borrows.
const BORROWS: [(bool, bool); 4] = [(false, false), (true, false), (false, true), (true, true)];

/// An operand on `home`: on its vtree, or on its projection on a random
/// set of its variables.
fn operand(eng: &Engine, rng: &mut Lcg, home: &Home) -> Tdd {
    let n = home.num_vars();
    let kept: Vec<u32> = match rng.below(4) {
        0 => (1..=n).collect(),
        _ => {
            let mut kept: Vec<u32> = (1..=n).filter(|_| rng.below(2) == 0).collect();
            if kept.is_empty() {
                kept.push(1 + rng.below(u64::from(n)) as u32);
            }
            kept
        }
    };
    let vtree = match kept.len() as u32 == n {
        true => Arc::clone(home),
        false => Vtree::projection(home, kept.iter().map(|&v| VarId(v))).unwrap(),
    };
    let k = kept.len() as u32;
    let cnf = |rng: &mut Lcg| compile_clauses_on(eng, home, &renamed_cnf(rand_cnf(rng, k, CnfShape { clauses: 1 + k as usize, width: 3 }), &kept));
    let f = cnf(rng);
    // A conjunction's result, which owes contraction passes.
    match rng.below(2) {
        0 => f,
        _ => {
            let g = cnf(rng);
            and_on(eng, f, g, &vtree).map_err(|r| r.error).unwrap()
        }
    }
}

fn given(t: &Tdd, borrowed: bool) -> Cow<'_, Tdd> {
    match borrowed {
        true => Cow::Borrowed(t),
        false => Cow::Owned(t.clone()),
    }
}

/// Whether `back` is `t` as `given(t, borrowed)` gave it.
#[expect(clippy::ptr_arg, reason = "which form came back is what is tested")]
fn as_given(back: &Cow<'_, Tdd>, t: &Tdd, borrowed: bool) -> bool {
    match (back, borrowed) {
        (Cow::Borrowed(back), true) => std::ptr::eq(*back, t),
        (Cow::Owned(back), false) => {
            same_storage(back, t)
                && format!("{:?}", back.dirty) == format!("{:?}", t.dirty)
                && back.levels.is_closed() == t.levels.is_closed()
        }
        _ => false,
    }
}

/// Whether a refused call handed back both operands as given; an owned
/// operand that is not viewed (false, or with many pass-through levels)
/// may come back placed.
fn handed_back(r: &AndRefused<'_>, (f, g, u): Case<'_>, (bf, bg): (bool, bool)) -> bool {
    let placed = |back: &Cow<'_, Tdd>, t: &Tdd, borrowed: bool| {
        !borrowed && Arc::ptr_eq(back.vtree(), u) && same_storage(back, &placed_on(&Engine::new(), t, u))
    };
    [(&r.f, f, bf), (&r.g, g, bg)]
        .into_iter()
        .all(|(back, t, b)| as_given(back, t, b) || placed(back, t, b))
}

type Case<'t> = (&'t Tdd, &'t Tdd, &'t Arc<Vtree>);

#[derive(Clone, Copy)]
enum Limit {
    None,
    StopAfter(u64),
    RefuseReserve(u32),
}

/// The conjunction of `f` and `g` on `u` on a fresh
/// engine under `limit`, through views where `viewing`, and the work it
/// took.
fn run<'t>(
    (f, g, u): Case<'t>,
    (bf, bg): (bool, bool),
    viewing: bool,
    limit: Limit,
) -> (Result<Tdd, AndRefused<'t>>, u64) {
    let eng = Engine::new();
    let config = match limit {
        Limit::StopAfter(n) => {
            let at = eng.limits().work_units() + n;
            LimitConfig::none().with_stop_rules(StopRules { unconditional: Some(StopAt::WorkUnits(at)), after_pairs: None })
        }
        Limit::None | Limit::RefuseReserve(_) => LimitConfig::none(),
    };
    let _scope = eng.limits().scope(config);
    if let Limit::RefuseReserve(n) = limit {
        eng.limits().refuse_nth_reserve(n);
    }
    let mark = eng.limits().mark();
    let out = conjoin_onto(&eng, given(f, bf), given(g, bg), u, viewing);
    let work = eng.limits().work_since(mark);
    (out, work)
}

/// The view of a plain operand holds at every node of `U` the level that
/// placing the operand stores there, its output and its worklists.
#[test]
fn a_view_reads_as_the_placed_operand() {
    let eng = Engine::new();
    let mut rng = Lcg::new(0x51e1);
    let (mut viewed, mut owing) = (0, 0);
    for case in 0..300u64 {
        let home = Home::new(Vtree::random(3 + rng.below(8) as u32, case));
        let u = home.clone();
        let f = operand(&eng, &mut rng, &home);
        if Arc::ptr_eq(f.vtree(), &u) || f.is_zero() {
            continue;
        }
        let plan = Plan::build(&eng, f.vtree(), &u, |v| v, false, Free::Leave).unwrap();
        let view = super::super::view::View::build(&eng, &f, &u, &plan).unwrap();
        let placed = place_by(&eng, Cow::Borrowed(&f), &u, &plan, Free::Leave).map_err(|e| e.0).unwrap();
        let at = format!("case {case}");
        assert_eq!(view.output(), placed.output, "{at}: output");
        assert_eq!(format!("{:?}", view.dirty()), format!("{:?}", placed.dirty), "{at}: worklists");
        for t in 0..u.num_nodes() {
            let (x, y) = (view.level(&f, t), &placed.levels[t]);
            assert!(x.nodes == y.nodes && x.pairs == y.pairs && x.ranges == y.ranges, "{at}: level {t}: {x:?}\n  placed {y:?}");
        }
        viewed += 1;
        owing += usize::from(!f.dirty.is_empty());
    }
    assert!(viewed > 100, "only {viewed} viewed operands");
    assert!(owing > 20, "only {owing} viewed operands owe contraction passes");
}

#[test]
fn views_conjoin_onto_as_placed_operands_do() {
    let eng = Engine::new();
    let mut rng = Lcg::new(0x51e0);
    let (mut refusals, mut viewed) = (0, 0);
    for case in 0..120u64 {
        let home = Home::new(Vtree::random(3 + rng.below(8) as u32, case));
        let u = home.clone();
        let f = operand(&eng, &mut rng, &home);
        let g = operand(&eng, &mut rng, &home);
        let c: Case<'_> = (&f, &g, &u);
        viewed += usize::from(!Arc::ptr_eq(f.vtree(), &u)) + usize::from(!Arc::ptr_eq(g.vtree(), &u));
        // Placed operands that are one diagram return `f` placed; two views
        // are not compared, and conjoin.
        let self_conjunction = same_storage(&placed_on(&eng, &f, &u), &placed_on(&eng, &g, &u));
        for borrow in BORROWS {
            let at = format!("case {case}, borrowing {borrow:?}");
            let (placed, placed_work) = run(c, borrow, false, Limit::None);
            let (got, work) = run(c, borrow, true, Limit::None);
            let (placed, got) = (placed.map_err(|r| r.error).unwrap(), got.map_err(|r| r.error).unwrap());
            assert!(same_storage(&got, &placed), "{at}: a different diagram");
            if !self_conjunction {
                assert_eq!(format!("{:?}", got.dirty), format!("{:?}", placed.dirty), "{at}: worklists");
                assert_eq!(work, placed_work, "{at}: work");
            }
            // Stopped at each unit of work, refused where the placed form is,
            // each operand handed back as it was given.
            for n in 0..=work {
                let (got, _) = run(c, borrow, true, Limit::StopAfter(n));
                let (placed, _) = run(c, borrow, false, Limit::StopAfter(n));
                assert!(self_conjunction || got.is_err() == placed.is_err(), "{at}: stopped at {n}, refused unlike the placed form");
                if let Err(r) = got {
                    refusals += 1;
                    assert!(handed_back(&r, c, borrow), "{at}: stopped at {n}, an operand changed");
                }
            }
            // Refused at each reserve in turn.
            for n in 0u32.. {
                let Err(r) = run(c, borrow, true, Limit::RefuseReserve(n)).0 else { break };
                refusals += 1;
                assert!(handed_back(&r, c, borrow), "{at}: refused at reserve {n}, an operand changed");
            }
        }
    }
    assert!(viewed > 120, "only {viewed} viewed operands");
    assert!(refusals > 2000, "only {refusals} refusals");
}

