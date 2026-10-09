//! `and_in_place` and `and_cube` against the conjunction they stand for.

use super::*;
use crate::apply::conjoin_in_place::and_in_place_grid;
use crate::diagram::Literal;
use crate::test_helpers::check::{check_all_fast, check_determinism, check_no_false_nodes};
use crate::vtree::VtreeIdx;

/// The variables under vtree node `t`.
fn vars_under(vtree: &Vtree, t: VtreeIdx) -> Vec<VarId> {
    let mut out = Vec::new();
    let mut stack = vec![t];
    while let Some(t) = stack.pop() {
        match vtree.node(t).is_leaf() {
            true => out.push(vtree.leaf_var(t)),
            false => {
                let (l, r) = vtree.children(t);
                stack.extend([l, r]);
            }
        }
    }
    out.sort();
    out
}

/// A random set of codes over the variables under a random internal node,
/// as the disjunction of one cube per code, `⊤` over every other variable.
fn local_set(vtree: &Arc<Vtree>, rng: &mut Lcg) -> Tdd {
    let internal: Vec<VtreeIdx> = vtree.internal_bottomup().map(|(t, _, _)| t).collect();
    let t = internal[rng.below(internal.len() as u64) as usize];
    let mut vars = vars_under(vtree, t);
    vars.truncate(4);
    let rows = 1usize << vars.len();
    let keep: Vec<bool> = (0..rows).map(|_| rng.below(3) != 0).collect();
    or_of_cubes(vtree, &vars, |r| keep[r])
}

/// A random cube over up to three variables.
fn random_cube(nvars: u32, rng: &mut Lcg) -> Vec<Literal> {
    let mut lits: Vec<Literal> = Vec::new();
    for _ in 0..1 + rng.below(3) {
        let v = VarId(1 + rng.below(u64::from(nvars)) as u32);
        if lits.iter().all(|l| l.var != v) {
            lits.push(Literal::new(v, rng.coin()));
        }
    }
    lits
}

/// Every model of `got` and of `f ∧ g` agree, over the whole truth table,
/// and the result keeps the structural invariants.
fn agrees(f: &Tdd, g: &Tdd, got: &Tdd, nvars: u32, label: &str) {
    for mask in 0..1u32 << nvars {
        let asn: Vec<bool> = (0..nvars).map(|i| (mask >> i) & 1 == 1).collect();
        assert_eq!(eval(got, &asn), eval(f, &asn) && eval(g, &asn), "{label}: models differ at {asn:?}");
    }
    check_no_false_nodes(got).unwrap_or_else(|e| panic!("{label}: {e}"));
    check_determinism(got).unwrap_or_else(|e| panic!("{label}: {e}"));
    let mut minimized = got.clone();
    minimized.minimize().unwrap();
    check_all_fast(&minimized, label);
}

/// [`agrees`] without its determinism check. That check conjoins every two
/// nodes of a level as diagrams rooted there, and on these twelve-variable
/// operands `Engine::and` panics on such a pair of `from_models`' own
/// diagrams (the product lookup at a level with no stored products), so it
/// says nothing about what this module builds.
fn agrees_wide(f: &Tdd, g: &Tdd, got: &Tdd, nvars: u32, label: &str) {
    for mask in 0..1u32 << nvars {
        let asn: Vec<bool> = (0..nvars).map(|i| (mask >> i) & 1 == 1).collect();
        assert_eq!(eval(got, &asn), eval(f, &asn) && eval(g, &asn), "{label}: models differ at {asn:?}");
    }
    check_no_false_nodes(got).unwrap_or_else(|e| panic!("{label}: {e}"));
    let mut minimized = got.clone();
    minimized.minimize().unwrap();
    check_all_fast(&minimized, label);
}

/// The cases: per vtree shape and size, random diagrams conjoined in place
/// with cubes, local code sets, two of them at once, and arbitrary
/// functions, pruned and loose; the results, raw and minimized.
fn cases() -> Vec<Tdd> {
    let eng = Engine::new();
    let mut rng = Lcg::new(0x7e3_f11e_2026_1008);
    let mut out = Vec::new();
    for nvars in [3u32, 5, 7] {
        for (shape, vtree) in vtree_shapes(nvars) {
            for round in 0..12 {
                let f = rand_conj(&vtree, nvars, 4, 3, round % 2 == 0, &mut rng);
                let label = format!("{shape} n={nvars} round {round}");
                let lits = random_cube(nvars, &mut rng);
                let cube = Tdd::cube(&vtree, lits.clone()).unwrap();
                let got = eng.and_cube(f.clone(), lits).unwrap();
                agrees(&f, &cube, &got, nvars, &format!("{label} cube"));
                out.push(got);
                let loose = eng.and_in_place_loose(f.clone(), &cube).unwrap();
                agrees(&f, &cube, &loose, nvars, &format!("{label} cube, loose"));
                out.push(loose);
                let set = local_set(&vtree, &mut rng);
                let got = eng.and_in_place(f.clone(), &set).unwrap();
                agrees(&f, &set, &got, nvars, &format!("{label} set"));
                out.push(got);
                let loose = eng.and_in_place_loose(f.clone(), &set).unwrap();
                agrees(&f, &set, &loose, nvars, &format!("{label} set, loose"));
                // Conjoined again, the loose levels are settled there.
                let again = eng.and(loose.clone(), set.clone()).unwrap();
                agrees(&f, &set, &again, nvars, &format!("{label} set, loose, conjoined"));
                out.extend([loose, again]);
                let two = eng.and(local_set(&vtree, &mut rng), local_set(&vtree, &mut rng)).unwrap();
                let got = eng.and_in_place(f.clone(), &two).unwrap();
                agrees(&f, &two, &got, nvars, &format!("{label} two sets"));
                out.push(got);
                let any = rand_conj(&vtree, nvars, 3, 3, false, &mut rng);
                let got = eng.and_in_place(f.clone(), &any).unwrap();
                agrees(&f, &any, &got, nvars, &format!("{label} any"));
                let mut minimized = got.clone();
                minimized.minimize().unwrap();
                let mut want = eng.and(f.clone(), any.clone()).unwrap();
                want.minimize().unwrap();
                assert_eq!(minimized.pair_count(), want.pair_count(), "{label}: the minimized sizes differ");
                out.extend([got, minimized]);
            }
        }
    }
    out
}

#[test]
fn and_in_place_is_the_conjunction() {
    cases();
}

/// On implicit levels and on stored ones alike ([`same_as_stored`]).
#[test]
fn and_in_place_on_implicit_levels() {
    same_as_stored(cases);
}

#[test]
fn constants_and_a_true_operand_are_decided_without_a_rewrite() {
    let eng = Engine::new();
    let vtree = Arc::new(Vtree::balanced(4));
    let f = eng.clause(&vtree, [1, -3]).unwrap();
    let same = eng.and_in_place(f.clone(), &eng.one(&vtree)).unwrap();
    assert!(eng.equivalent(&same, &f).unwrap());
    assert!(eng.and_in_place(f.clone(), &eng.zero(&vtree)).unwrap().is_zero());
    assert!(eng.and_in_place(eng.zero(&vtree), &f).unwrap().is_zero());
    let none = eng.and_cube(f.clone(), [-1, 3]).unwrap();
    assert!(none.is_zero());
    let other = Arc::new(Vtree::balanced(4));
    assert!(matches!(eng.and_in_place(f, &eng.one(&other)), Err(crate::OperationError::VtreeMismatch)));
}

/// A cube over the variables a level's left child holds keeps `f`'s nodes
/// under its sibling: the pairs above are rewritten, never rebuilt.
#[test]
fn a_cube_keeps_the_nodes_it_does_not_cut() {
    let eng = Engine::new();
    let vtree = Arc::new(Vtree::balanced(8));
    let f = compile_clauses_on(&eng, &vtree, &[vec![1, 2, 5], vec![-2, 6, 7], vec![3, -8], vec![4, 5, -6]]);
    let (_, right) = vtree.children(vtree.root());
    let before = f.level(right).nodes().len();
    let got = eng.and_cube(f.clone(), [1, -2]).unwrap();
    assert!(got.level(right).nodes().len() <= before);
    agrees(&f, &Tdd::cube(&vtree, [1, -2]).unwrap(), &got, 8, "kept");
}

/// A filter on a column whose nodes hold one code each: every node of the
/// column's block is decided by where it lies, so the kept ones are `f`'s
/// own and no product is appended there.
#[test]
fn a_node_inside_the_filter_is_kept_whole() {
    let eng = Engine::new();
    // Two blocks of three variables, one row per code of the first: the
    // codes of the second block are each the first's plus one.
    let vtree = Arc::new(Vtree::balanced(6));
    let rows: Vec<(u32, u32)> = (0..8).map(|c| (c, (c + 1) % 8)).collect();
    let vars: Vec<VarId> = (1..=6).map(VarId).collect();
    let f = or_of_cubes(&vtree, &vars, |m| rows.iter().any(|&(a, b)| m == (a | (b << 3)) as usize));
    let (left, _) = vtree.children(vtree.root());
    let before = f.level(left).nodes().len();
    // The first block's codes 0..=4: a set over its variables alone.
    let set = or_of_cubes(&vtree, &vars[..3], |m| m <= 4);
    let got = eng.and_in_place(f.clone(), &set).unwrap();
    agrees(&f, &set, &got, 6, "kept whole");
    assert!(got.level(left).nodes().len() <= before, "no product was appended under the filter's block");
}

/// Wide nodes in both operands: a product of two nodes of many pairs finds
/// `g`'s pairs by their sides, where a side of `f`'s pair lies inside one
/// node of `g`, instead of trying each pair against each; the result is
/// still the conjunction, with `g` a set of codes over every variable and
/// over the variables under the root's left child.
#[test]
fn wide_products_find_the_pairs_they_meet_by_their_sides() {
    let eng = Engine::new();
    let mut rng = Lcg::new(0x51de_5eed_2026_1009);
    let nvars = 12u32;
    let vars: Vec<VarId> = (1..=nvars).map(VarId).collect();
    for (shape, vtree) in vtree_shapes(nvars) {
        let (left, _) = vtree.children(vtree.root());
        let low = vars_under(&vtree, left);
        for round in 0..3 {
            let f = Tdd::from_models(&vtree, &vars, &(0..700).map(|_| rng.below(1 << nvars)).collect::<Vec<_>>()).unwrap();
            let all = Tdd::from_models(&vtree, &vars, &(0..400).map(|_| rng.below(1 << nvars)).collect::<Vec<_>>()).unwrap();
            let codes: Vec<u64> = (0..1 + (1u64 << low.len()) / 3).map(|_| rng.below(1 << low.len())).collect();
            let on_left = or_of_cubes(&vtree, &low, |m| codes.contains(&(m as u64)));
            for (name, g) in [("all", &all), ("left", &on_left)] {
                // Each pair against each, by the sides always, and the default.
                for grid in [usize::MAX, 0, 64] {
                    for prune in [true, false] {
                        let label = format!("{shape} round {round} {name} grid {grid} prune {prune}");
                        let got = and_in_place_grid(&eng, f.clone(), g, prune, grid).unwrap();
                        agrees_wide(&f, g, &got, nvars, &label);
                    }
                }
            }
        }
    }
}

