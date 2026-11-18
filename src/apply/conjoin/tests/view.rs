//! [`Engine::and_onto`] reads its plain operands through views: it builds
//! what placing them and conjoining the placed operands builds, for the same
//! work, is refused where that is refused, and hands back every operand as
//! it was given or, owned and placed rather than viewed, placed.

use std::borrow::Cow;
use std::sync::Arc;

use super::*;
use crate::Engine;
use crate::limits::{LimitConfig, StopAt, StopRules};
use crate::test_helpers::{compile_clauses_on, rand_cnf, same_storage, CnfShape, Lcg};
use crate::vtree::{VarId, Vtree};

/// Which operands a call borrows.
const BORROWS: [(bool, bool); 4] = [(false, false), (true, false), (false, true), (true, true)];

/// An operand on `u`: on `u` itself, or on the projection of `u` on a
/// random set of its variables, with the map from its variables to `u`'s.
fn operand(eng: &Engine, rng: &mut Lcg, u: &Arc<Vtree>) -> (Tdd, Vec<VarId>) {
    let n = u.num_vars();
    let (vtree, map) = match rng.below(4) {
        0 => (Arc::clone(u), (1..=n).map(VarId).collect::<Vec<_>>()),
        _ => {
            let mut kept: Vec<u32> = (1..=n).filter(|_| rng.below(2) == 0).collect();
            if kept.is_empty() {
                kept.push(1 + rng.below(u64::from(n)) as u32);
            }
            let local_of = |v: VarId| kept.iter().position(|&k| k == v.0).map(|i| VarId(i as u32 + 1));
            let own = u.project_to_vars(local_of, kept.len() as u32).expect("a variable is kept");
            (Arc::new(own), kept.into_iter().map(VarId).collect())
        }
    };
    let k = vtree.num_vars();
    let cnf = |rng: &mut Lcg| compile_clauses_on(eng, &vtree, &rand_cnf(rng, k, CnfShape { clauses: 1 + k as usize, width: 3 }));
    let f = cnf(rng);
    // A conjunction's result, which owes contraction passes.
    let f = match rng.below(2) {
        0 => f,
        _ => {
            let g = cnf(rng);
            eng.and_onto(Cow::Owned(f), |v| v, Cow::Owned(g), |v| v, &vtree).map_err(|r| r.error).unwrap()
        }
    };
    (f, map)
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
fn handed_back(r: &AndOntoRefused<'_>, ((f, f_map), (g, g_map), u): Case<'_>, (bf, bg): (bool, bool)) -> bool {
    let placed = |back: &Cow<'_, Tdd>, t: &Tdd, map: &[VarId], borrowed: bool| {
        !borrowed && Arc::ptr_eq(back.vtree(), u) && {
            let eng = Engine::new();
            let embedded = eng.embed_moving(t.clone(), u, |v| map[v.0 as usize - 1]).map_err(|r| r.error).unwrap().0;
            same_storage(back, &embedded)
        }
    };
    [(&r.f, f, f_map, bf), (&r.g, g, g_map, bg)]
        .into_iter()
        .all(|(back, t, map, b)| as_given(back, t, b) || placed(back, t, map, b))
}

type Case<'t> = ((&'t Tdd, &'t [VarId]), (&'t Tdd, &'t [VarId]), &'t Arc<Vtree>);

#[derive(Clone, Copy)]
enum Limit {
    None,
    StopAfter(u64),
    RefuseReserve(u32),
}

/// The conjunction of `(f, f_map)` and `(g, g_map)` on `u` on a fresh
/// engine under `limit`, through views where `viewing`, and the work it
/// took.
fn run<'t>(
    ((f, f_map), (g, g_map), u): Case<'t>,
    (bf, bg): (bool, bool),
    viewing: bool,
    limit: Limit,
) -> (Result<Tdd, AndOntoRefused<'t>>, u64) {
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
    let out = conjoin_onto_as(
        &eng,
        (given(f, bf), |v: VarId| f_map[v.0 as usize - 1]),
        (given(g, bg), |v: VarId| g_map[v.0 as usize - 1]),
        u,
        viewing,
    );
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
        let u = Arc::new(Vtree::random(3 + rng.below(8) as u32, case));
        let (f, map) = operand(&eng, &mut rng, &u);
        if Arc::ptr_eq(f.vtree(), &u) || f.is_zero() {
            continue;
        }
        let plan = crate::restructure::embed::Plan::build(&eng, f.vtree(), &u, |v| map[v.0 as usize - 1], false).unwrap();
        let view = super::super::view::View::build(&eng, &f, &u, &plan).unwrap();
        let placed = place_by(&eng, Cow::Borrowed(&f), &u, &plan, crate::restructure::embed::Free::Leave).map_err(|e| e.0).unwrap();
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
        let u = Arc::new(Vtree::random(3 + rng.below(8) as u32, case));
        let (f, f_map) = operand(&eng, &mut rng, &u);
        let (g, g_map) = operand(&eng, &mut rng, &u);
        let c: Case<'_> = ((&f, &f_map), (&g, &g_map), &u);
        viewed += usize::from(!Arc::ptr_eq(f.vtree(), &u)) + usize::from(!Arc::ptr_eq(g.vtree(), &u));
        for borrow in BORROWS {
            let at = format!("case {case}, borrowing {borrow:?}");
            let (placed, placed_work) = run(c, borrow, false, Limit::None);
            let (got, work) = run(c, borrow, true, Limit::None);
            let (placed, got) = (placed.map_err(|r| r.error).unwrap(), got.map_err(|r| r.error).unwrap());
            assert!(same_storage(&got, &placed), "{at}: a different diagram");
            assert_eq!(format!("{:?}", got.dirty), format!("{:?}", placed.dirty), "{at}: worklists");
            assert_eq!(work, placed_work, "{at}: work");
            // Stopped at each unit of work, refused where the placed form is,
            // each operand handed back as it was given.
            for n in 0..=work {
                let (got, _) = run(c, borrow, true, Limit::StopAfter(n));
                let (placed, _) = run(c, borrow, false, Limit::StopAfter(n));
                assert_eq!(got.is_err(), placed.is_err(), "{at}: stopped at {n}, refused unlike the placed form");
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
