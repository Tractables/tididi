//! Levels that read a complete child side by arithmetic, checked against
//! the grid reads they replace: the same diagram, the same work, and the
//! output-pair meter where the grid route has it.

use super::*;
use crate::Engine;
use crate::limits::{LimitConfig, StopAt, StopRules};
use crate::test_helpers::{assert_canonical, rand_conj_over, same_as_stored, Lcg};
use crate::vtree::Vtree;

/// A random function of `vars`, minimized, on `vtree`.
fn function_of(vtree: &Arc<Vtree>, vars: &[u32], rng: &mut Lcg) -> Tdd {
    let mut f = rand_conj_over(vtree, vars, 10, 3, false, rng);
    f.minimize().unwrap();
    f
}

/// Conjoin `f` and `g` on a fresh engine, reading every child side from the
/// grid when `grid` holds, under a stop at `floor` output pairs when given:
/// the result and the work units it took.
fn conjoin(f: &Tdd, g: &Tdd, grid: bool, floor: Option<u64>) -> (Result<Tdd, OperationError>, u64) {
    let eng = Engine::new();
    let run = || {
        let rules = StopRules { unconditional: None, after_pairs: floor.map(|p| (p, StopAt::WorkUnits(0))) };
        let _scope = eng.limits().scope(LimitConfig::none().with_stop_rules(rules));
        eng.and(f.clone(), g.clone())
    };
    let out = if grid { grid_lookups(run) } else { run() };
    (out, eng.limits().work_units())
}

/// Conjoin `f` and `g` both ways and require the same diagram and the same
/// work; then, under a stop at each output-pair floor up to the result's
/// size, the same stop at the same work. Returns the conjunction.
fn same_both_ways(f: &Tdd, g: &Tdd) -> Tdd {
    let (oracle, oracle_work) = conjoin(f, g, true, None);
    let (out, work) = conjoin(f, g, false, None);
    let (oracle, out) = (oracle.unwrap(), out.unwrap());
    assert_canonical(&out);
    assert!(is_self_conjunction(&out, &oracle), "the arithmetic lookups built another diagram");
    assert_eq!(work, oracle_work, "the arithmetic lookups did other work");
    let total: usize = out.levels.iter().map(|l| l.pairs.len()).sum();
    let mut floor = 1u64;
    while floor <= 4 * total as u64 {
        let (oracle, oracle_work) = conjoin(f, g, true, Some(floor));
        let (stopped, work) = conjoin(f, g, false, Some(floor));
        assert_eq!(stopped.is_ok(), oracle.is_ok(), "a stop at {floor} pairs fell differently");
        assert_eq!(work, oracle_work, "a stop at {floor} pairs fell at other work");
        floor = floor * 3 / 2 + 1;
    }
    out
}

/// `f` over the odd variables and `g` over the even ones of a balanced
/// vtree: every product of two of their nodes is satisfiable, so every level
/// over two internal children reads both by arithmetic.
#[test]
fn disjoint_supports_read_both_sides_by_arithmetic() {
    let vtree = Arc::new(Vtree::balanced(16));
    let odd: Vec<u32> = (1..=16).filter(|v| v % 2 == 1).collect();
    let even: Vec<u32> = (1..=16).filter(|v| v % 2 == 0).collect();
    let mut rng = Lcg::new(0x5eed_a901);
    let before = complete_census();
    for _ in 0..6 {
        let f = function_of(&vtree, &odd, &mut rng);
        let g = function_of(&vtree, &even, &mut rng);
        same_both_ways(&f, &g);
    }
    let census = complete_census();
    assert!(census[0] > before[0], "no level read both sides by arithmetic");
}

/// `x_a ↔ x_b` for each pair of `pairs`: on a balanced vtree with every `a`
/// left of the root and every `b` right of it, the root holds one pair for
/// each assignment of the `a`s.
fn equalities(vtree: &Arc<Vtree>, pairs: &[(i32, i32)]) -> Tdd {
    let eng = Engine::new();
    let mut f = Tdd::one(vtree);
    for &(a, b) in pairs {
        f = eng.and(f, Tdd::clause(vtree, [-a, b]).unwrap()).unwrap();
        f = eng.and(f, Tdd::clause(vtree, [a, -b]).unwrap()).unwrap();
    }
    f
}

/// Two roots of 128 pairs each, over disjoint supports, make a root of
/// 16,384 pairs, past the cap on a level's seeded arena: the arithmetic
/// route's arena, reserved whole, charges the meter on the schedule the
/// grid route's arena grows by.
#[test]
fn an_arena_past_the_seed_cap_charges_on_the_schedule() {
    let vtree = Arc::new(Vtree::balanced(64));
    let f = equalities(&vtree, &[(1, 33), (3, 35), (5, 37), (7, 39), (9, 41), (11, 43), (13, 45)]);
    let g = equalities(&vtree, &[(2, 34), (4, 36), (6, 38), (8, 40), (10, 42), (12, 44), (14, 46)]);
    let root = vtree.root().idx();
    assert_eq!((f.levels[root].live_pairs(), g.levels[root].live_pairs()), (128, 128));
    let before = complete_census();
    let fg = same_both_ways(&f, &g);
    assert_eq!(fg.levels[root].live_pairs(), 128 * 128);
    let census = complete_census();
    assert!(census[3] > before[3], "no reserved arena outgrew its seed, so the meter's schedule went unchecked");
}

/// A level of one-pair nodes holds its pairs inline, none in its arena: a
/// product with it is seeded at its pairs all the same, and no level of a
/// conjunction under the seed's cap grows its arena. `g` over the leftmost
/// quarter is one node of one pair on every level outside it.
#[test]
fn a_level_of_one_pair_nodes_seeds_its_products_arena() {
    let vtree = Arc::new(Vtree::balanced(16));
    let all: Vec<u32> = (1..=16).collect();
    let mut rng = Lcg::new(0x5eed_a905);
    let before = pairs_grown();
    for _ in 0..6 {
        let f = function_of(&vtree, &all, &mut rng);
        let g = function_of(&vtree, &[1, 2, 3, 4], &mut rng);
        same_both_ways(&f, &g);
        same_both_ways(&g, &f);
    }
    assert_eq!(pairs_grown(), before, "a level's arena outgrew its seed");
}

/// Shared variables under one child leave products unsatisfiable there: that
/// child has dead cells and is read from the grid, its sibling by
/// arithmetic, on the left at one level and on the right at another. On
/// implicit levels and on stored ones alike ([`same_as_stored`]).
#[test]
fn shared_variables_leave_one_side_on_the_grid() {
    let vtree = Arc::new(Vtree::balanced(16));
    // `f` also reads 2 and 4, under the leftmost quarter, and `g` 13 and 15,
    // under the rightmost.
    let fv: Vec<u32> = (1..=16).filter(|v| v % 2 == 1 || *v == 2 || *v == 4).collect();
    let gv: Vec<u32> = (1..=16).filter(|v| v % 2 == 0 || *v == 13 || *v == 15).collect();
    let before = complete_census();
    same_as_stored(|| {
        let mut rng = Lcg::new(0x5eed_a902);
        let mut out = Vec::new();
        for _ in 0..8 {
            let f = function_of(&vtree, &fv, &mut rng);
            let g = function_of(&vtree, &gv, &mut rng);
            let fg = same_both_ways(&f, &g);
            out.extend([f, g, fg]);
        }
        out
    });
    let census = complete_census();
    assert!(census[1] > before[1], "no level read only its left side by arithmetic");
    assert!(census[2] > before[2], "no level read only its right side by arithmetic");
}

/// `g` over the leftmost quarter of the variables is one node of one pair
/// on every level outside it, and a level of two complete sides there reads
/// that pair in every cell, both ways round.
#[test]
fn a_lone_g_pair_is_read_in_every_cell() {
    let vtree = Arc::new(Vtree::balanced(16));
    let all: Vec<u32> = (1..=16).collect();
    let mut rng = Lcg::new(0x5eed_a904);
    let before = complete_census();
    for _ in 0..6 {
        let f = function_of(&vtree, &all, &mut rng);
        let g = function_of(&vtree, &[1, 2, 3, 4], &mut rng);
        same_both_ways(&f, &g);
        same_both_ways(&g, &f);
    }
    let census = complete_census();
    assert!(census[4] > before[4], "no level read a lone g pair in every cell");
}

/// The grid route stands in for the arithmetic one only in tests: the
/// census counts no level while it is forced.
#[test]
fn the_oracle_reads_every_side_from_the_grid() {
    let vtree = Arc::new(Vtree::balanced(8));
    let mut rng = Lcg::new(0x5eed_a903);
    let f = function_of(&vtree, &[1, 3, 5, 7], &mut rng);
    let g = function_of(&vtree, &[2, 4, 6, 8], &mut rng);
    let before = complete_census();
    conjoin(&f, &g, true, None).0.unwrap();
    assert_eq!(complete_census(), before);
}
