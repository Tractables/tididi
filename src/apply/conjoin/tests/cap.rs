//! Levels of two complete child sides refused before they are built for
//! passing the output-node cap, checked against the build they replace,
//! which fails the cap at the next level boundary: under every cap, the
//! same outcome at the same work, alone and under stops by work and by
//! output pairs.

use super::*;
use crate::Engine;
use crate::limits::{LimitConfig, StopAt, StopRules};
use crate::test_helpers::{assert_same_shape, compile_clauses_on, rand_cnf, rand_conj_over, vtree_shapes, CnfShape, Lcg};
use crate::vtree::Vtree;

/// The stops by work a refusing cap is run under: one at each `k / CUTS`
/// of its work.
const CUTS: u64 = 12;

/// The units the dense levels' gates come due at, so that stops fall
/// inside the levels of small diagrams.
const STRIDE: u64 = 1 << 6;

/// No stop rule.
const NO_STOP: StopRules = StopRules { unconditional: None, after_pairs: None };

/// A conjunction run on a fresh engine under a cap and stop rules: whether
/// it finished or why not, and the work it took.
type Conjoin<'a> = &'a dyn Fn(Option<u64>, StopRules) -> (Result<(), OperationError>, u64);

/// `run` under `cap` and `rules`, refused before the build and built past
/// the cap: the same outcome at the same work. Whether a level was refused.
fn same_outcome(run: Conjoin<'_>, cap: Option<u64>, rules: StopRules, what: &str) -> bool {
    let want = builds_past_cap(|| run(cap, rules));
    let refused = refused_past_cap();
    let got = run(cap, rules);
    assert_eq!(got, want, "{what}: under a cap of {cap:?} nodes and {rules:?}");
    refused_past_cap() > refused
}

/// `run` under each cap from none up to past its output, and under each cap
/// a level was refused at, under stops by work at [`CUTS`] points of the
/// work it took and at output-pair floors up to past its pairs: the same
/// outcome at the same work as the levels built. The caps a level was
/// refused at.
fn check(run: Conjoin<'_>, what: &str) -> usize {
    let (done, _) = run(None, NO_STOP);
    assert_eq!(done, Ok(()), "{what}: the conjunction failed with no cap");
    let mut refusing = Vec::new();
    let mut cap = 0u64;
    loop {
        if same_outcome(run, Some(cap), NO_STOP, what) {
            refusing.push(cap);
        }
        if run(Some(cap), NO_STOP).0.is_ok() {
            break;
        }
        cap = cap * 5 / 4 + 1;
    }
    for &cap in &refusing {
        let (_, total) = builds_past_cap(|| run(Some(cap), NO_STOP));
        for k in 1..=CUTS {
            let at = StopAt::WorkUnits(total * k / CUTS);
            same_outcome(run, Some(cap), StopRules { unconditional: Some(at), after_pairs: None }, what);
        }
        let mut floor = 1u64;
        while floor <= 1 << 20 {
            let rules = StopRules { unconditional: None, after_pairs: Some((floor, StopAt::WorkUnits(0))) };
            same_outcome(run, Some(cap), rules, what);
            floor = floor * 4 + 1;
        }
    }
    refusing.len()
}

/// [`check`] of `f ∧ g` by [`Engine::and`].
fn check_and(f: &Tdd, g: &Tdd, what: &str) -> usize {
    check(
        &|cap, rules| {
            let eng = Engine::new();
            let _scope = eng.limits().scope(LimitConfig::none().with_output_node_cap(cap).with_stop_rules(rules));
            let out = with_dense_poll_stride(STRIDE, || eng.and(f.clone(), g.clone())).map(drop);
            (out, eng.limits().work_units())
        },
        what,
    )
}

/// A random function of `vars`, minimized, on `vtree`.
fn function_of(vtree: &Arc<Vtree>, vars: &[u32], rng: &mut Lcg) -> Tdd {
    let mut f = rand_conj_over(vtree, vars, 10, 3, false, rng);
    f.minimize().unwrap();
    f
}

/// `x_a ↔ x_b` for each pair of `pairs`.
fn equalities(vtree: &Arc<Vtree>, pairs: &[(i32, i32)]) -> Tdd {
    let eng = Engine::new();
    let mut f = Tdd::one(vtree);
    for &(a, b) in pairs {
        f = eng.and(f, Tdd::clause(vtree, [-a, b]).unwrap()).unwrap();
        f = eng.and(f, Tdd::clause(vtree, [a, -b]).unwrap()).unwrap();
    }
    f
}

/// `f` over the odd variables and `g` over the even ones of a balanced
/// vtree: every level over two internal children has two complete sides,
/// and a cap between the levels' sizes refuses the level that passes it.
#[test]
fn disjoint_supports_refuse_the_level_past_the_cap() {
    let vtree = Arc::new(Vtree::balanced(16));
    let odd: Vec<u32> = (1..=16).filter(|v| v % 2 == 1).collect();
    let even: Vec<u32> = (1..=16).filter(|v| v % 2 == 0).collect();
    let mut rng = Lcg::new(0x5eed_ca91);
    let mut refusing = 0;
    for i in 0..6 {
        let f = function_of(&vtree, &odd, &mut rng);
        let g = function_of(&vtree, &even, &mut rng);
        refusing += check_and(&f, &g, &format!("disjoint supports {i}"));
    }
    assert!(refusing > 0, "no cap refused a level before it was built");
}

/// The children of the root of two equality chains over disjoint
/// variables hold 128 one-pair nodes each, and their products a full grid
/// of 16,384 one-pair nodes, past the seeded arena.
#[test]
fn a_grid_of_one_pair_nodes_is_refused_whole() {
    let vtree = Arc::new(Vtree::balanced(64));
    let f = equalities(&vtree, &[(1, 33), (3, 35), (5, 37), (7, 39), (9, 41), (11, 43), (13, 45)]);
    let g = equalities(&vtree, &[(2, 34), (4, 36), (6, 38), (8, 40), (10, 42), (12, 44), (14, 46)]);
    assert!(check_and(&f, &g, "equalities") > 0, "no cap refused a level before it was built");
}

/// Random formulas on every vtree shape, whose levels are rarely complete
/// on both sides: no cap refuses a level whose build would end otherwise.
#[test]
fn a_cap_refuses_no_level_the_build_would_finish_otherwise() {
    let mut rng = Lcg::new(0x5eed_ca92);
    for shape in [CnfShape { clauses: 12, width: 2 }, CnfShape { clauses: 30, width: 3 }] {
        for (name, vtree) in vtree_shapes(16) {
            let eng = Engine::new();
            let f = compile_clauses_on(&eng, &vtree, &rand_cnf(&mut rng, 16, shape));
            let g = compile_clauses_on(&eng, &vtree, &rand_cnf(&mut rng, 16, shape));
            check_and(&f, &g, &format!("{name}, {shape:?}"));
        }
    }
}

/// [`Engine::and_onto`] counts the nodes of the levels a placement leaves
/// free under a refused level: `f` over the odd variables of its own
/// balanced vtree, placed on the left half of a wider one, against `g`
/// over the even variables of the wider vtree.
#[test]
fn a_placement_counts_its_free_levels_under_a_refused_level() {
    let narrow = Arc::new(Vtree::balanced(16));
    let wide = Arc::new(Vtree::balanced(32));
    let odd: Vec<u32> = (1..=16).filter(|v| v % 2 == 1).collect();
    let even: Vec<u32> = (1..=32).filter(|v| v % 2 == 0).collect();
    let mut rng = Lcg::new(0x5eed_ca93);
    let mut refusing = 0;
    for i in 0..4 {
        let f = function_of(&narrow, &odd, &mut rng);
        let g = function_of(&wide, &even, &mut rng);
        refusing += check(
            &|cap, rules| {
                let eng = Engine::new();
                let _scope = eng.limits().scope(LimitConfig::none().with_output_node_cap(cap).with_stop_rules(rules));
                let out = with_dense_poll_stride(STRIDE, || {
                    eng.and_onto(f.clone(), |v| v, g.clone(), |v| v, &wide).map(drop).map_err(|r| match r.error {
                        crate::restructure::EmbedError::Operation(e) => e,
                        other => panic!("placed {i}: {other:?}"),
                    })
                });
                (out, eng.limits().work_units())
            },
            &format!("placed {i}"),
        );
    }
    assert!(refusing > 0, "no cap refused a level before it was built");
}

/// A conjunction refused before a level was built hands both operands back
/// unchanged, and the engine conjoins them afterwards as a fresh one does.
#[test]
fn a_refused_conjunction_restores_its_operands() {
    let vtree = Arc::new(Vtree::balanced(16));
    let odd: Vec<u32> = (1..=16).filter(|v| v % 2 == 1).collect();
    let even: Vec<u32> = (1..=16).filter(|v| v % 2 == 0).collect();
    let mut rng = Lcg::new(0x5eed_ca94);
    let f = function_of(&vtree, &odd, &mut rng);
    let g = function_of(&vtree, &even, &mut rng);
    let want = Engine::new().and(f.clone(), g.clone()).unwrap();
    let eng = Engine::new();
    let mut refusals = 0;
    for cap in 0..want.levels.iter().map(TddLevel::slot_count).sum::<usize>() as u64 {
        let refused = refused_past_cap();
        let out = {
            let _scope = eng.limits().scope(LimitConfig::none().with_output_node_cap(Some(cap)));
            eng.and_restoring(f.clone(), g.clone())
        };
        if refused_past_cap() == refused {
            continue;
        }
        refusals += 1;
        let back = out.expect_err("a level was refused");
        assert_eq!(back.error, OperationError::OutputCap);
        assert_same_shape(&back.f, &f, "the first operand");
        assert_same_shape(&back.g, &g, "the second operand");
        assert_same_shape(&eng.and(f.clone(), g.clone()).unwrap(), &want, "the conjunction after a refusal");
        if refusals == 8 {
            break;
        }
    }
    assert!(refusals > 0, "no cap refused a level before it was built");
}
