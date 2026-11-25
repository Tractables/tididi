//! An accumulator widened from its own link is the placement's result node
//! for node.

use super::super::place_on;
use crate::vtree::rng::Lcg;
use crate::vtree::{Home, VarId, Vtree, VtreeIdx};
use crate::{Engine, OperationError, Tdd};

fn same(got: &Tdd, want: &Tdd, what: &str) {
    let (a, b) = (got.vtree(), want.vtree());
    assert_eq!(a.num_nodes(), b.num_nodes(), "{what}: vtree size");
    for t in (0..a.num_nodes() as u32).map(VtreeIdx) {
        assert_eq!(a.node(t), b.node(t), "{what}: vtree node {t:?}");
        if a.num_nodes() > 0 && a.home().is_some() {
            assert_eq!(a.home_node(t), b.home_node(t), "{what}: home node of {t:?}");
        }
    }
    assert_eq!(got.output, want.output, "{what}: output");
    let (mut x, mut y) = (Vec::new(), Vec::new());
    for (t, (p, q)) in got.levels.iter().zip(want.levels.iter()).enumerate() {
        assert_eq!(p.is_marginal(), q.is_marginal(), "{what}: level {t}");
        assert_eq!(p.nodes().len(), q.nodes().len(), "{what}: nodes of level {t}");
        for i in 0..p.nodes().len() {
            assert_eq!(p.pairs_read(i, &mut x), q.pairs_read(i, &mut y), "{what}: level {t} node {i}");
        }
    }
}

enum Step {
    And(Vec<i32>),
    Or(Vec<i32>),
}

/// `step` on `f`; with `placed`, `f` is first placed on the projection that
/// adds the step's variables, as the entries did before they widened from
/// the link, so that the entry finds it widened.
fn run(eng: &Engine, placed: bool, mut f: Tdd, step: &Step) -> Result<Tdd, OperationError> {
    let lits = match step {
        Step::And(lits) | Step::Or(lits) => lits,
    };
    let vars = || lits.iter().map(|l| VarId(l.unsigned_abs()));
    let both_ways = lits.iter().any(|l| lits.contains(&-l));
    let vtree = f.vtree().clone();
    if let Some(home) = vtree.home().filter(|_| placed && !vtree.has_regions() && !both_ways)
        && vars().any(|var| vtree.home_has(var))
    {
        let own = vtree.leaf_bottomup().map(|(_, var)| var);
        f = place_on(eng, f, &Vtree::projection(&home, own.chain(vars())).unwrap())?;
    }
    match step {
        Step::And(lits) => eng.and_clause(f, lits),
        Step::Or(lits) => eng.or_cube(f, lits),
    }
}

/// Both ways through `steps` from `f`, compared after each.
fn both_ways(f: Tdd, steps: &[Step], what: &str) -> usize {
    let (fast, placed) = (Engine::new(), Engine::new());
    let (mut a, mut b) = (f.clone(), f);
    let mut widened = 0;
    for (i, step) in steps.iter().enumerate() {
        let before = a.vtree().num_leaves();
        let plain = a.vtree().num_nodes() > 0 && !a.is_zero() && !a.is_one();
        match (run(&fast, false, a, step), run(&placed, true, b, step)) {
            (Ok(x), Ok(y)) => {
                same(&x, &y, &format!("{what}, step {i}"));
                widened += usize::from(plain && x.vtree().num_leaves() > before);
                (a, b) = (x, y);
            }
            (Err(x), Err(y)) => {
                assert_eq!(x, y, "{what}, step {i}");
                return widened;
            }
            (x, y) => panic!("{what}, step {i}: {:?} against {:?}", x.err(), y.err()),
        }
    }
    widened
}

fn literals(n: u32, count: u64, rng: &mut Lcg) -> Vec<i32> {
    (0..1 + rng.below(count))
        .map(|_| {
            let v = 1 + rng.below(u64::from(n)) as i32;
            if rng.coin() { v } else { -v }
        })
        .collect()
}

#[test]
fn and_clause_and_or_cube_widen_as_the_placement_does() {
    let mut rng = Lcg::new(0x7769_6465);
    let mut widened = 0;
    for seed in 0..500u64 {
        let n = 2 + rng.below(15) as u32;
        let home = Home::new(Vtree::random(n, seed));
        let start = 1 + rng.below(u64::from(n)) as i32;
        let f = match seed % 4 {
            0 => Tdd::one(&crate::vtree::Home::unchecked(&Vtree::empty_projection(&home))),
            1 => crate::literal(&crate::vtree::Home::unchecked(&Vtree::projection(&home, [VarId(start as u32)]).unwrap()), -start).unwrap(),
            _ => {
                let lits = literals(n, 3, &mut rng);
                let on = Vtree::projection(&home, lits.iter().map(|l| VarId(l.unsigned_abs()))).unwrap();
                Tdd::clause(&crate::vtree::Home::unchecked(&on), lits).unwrap()
            }
        };
        let steps: Vec<Step> = (0..8)
            .map(|_| match rng.below(3) {
                0 => Step::Or(literals(n, 3, &mut rng)),
                _ => Step::And(literals(n, 4, &mut rng)),
            })
            .collect();
        widened += both_ways(f, &steps, &format!("seed {seed}, n {n}"));
    }
    assert!(widened > 1000, "{widened} widenings");
}

#[test]
fn a_clause_with_seventeen_new_variables_widens_as_the_placement_does() {
    let home = Home::new(Vtree::random(24, 17));
    let f = crate::literal(&crate::vtree::Home::unchecked(&Vtree::projection(&home, [VarId(1), VarId(20)]).unwrap()), 1).unwrap();
    let wide: Vec<i32> = (2..=18).collect();
    let cube: Vec<i32> = (2..=18).map(|v| if v % 3 == 0 { -v } else { v }).collect();
    for steps in [[Step::And(wide.clone()), Step::Or(cube.clone())], [Step::Or(cube), Step::And(wide)]] {
        assert_eq!(both_ways(f.clone(), &steps, "seventeen"), 1);
    }
}

#[test]
fn a_clause_into_a_region_is_refused_both_ways() {
    let home = Home::new(Vtree::balanced(4));
    let mut g = Tdd::clause(&crate::vtree::Home::unchecked(&Vtree::projection(&home, [1, 2, 3].map(VarId)).unwrap()), [1, 2, 3]).unwrap();
    g.marginalize(|v| v.0 <= 2).unwrap();
    assert!(g.vtree().has_regions());
    assert_eq!(both_ways(g.clone(), &[Step::And(vec![1, 4])], "into a region"), 0);
    assert!(matches!(Engine::new().and_clause(g.clone(), &[1, 4]), Err(OperationError::MarginalLevel(_))));
    both_ways(g, &[Step::And(vec![4]), Step::Or(vec![-4, 3])], "beside a region");
}
