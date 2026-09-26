use std::sync::Arc;

use super::*;
use crate::Engine;
use crate::test_helpers::{assert_canonical, assert_same_shape, eval, Lcg};
use crate::vtree::{VarId, Vtree};

/// A vtree shape to spell as vtree text: a variable, or two subtrees.
enum Shape {
    Var(u32),
    Join(Box<Shape>, Box<Shape>),
}

fn join(l: Shape, r: Shape) -> Shape {
    Shape::Join(Box::new(l), Box::new(r))
}

/// Right-linear over `vars`, in order.
fn right_linear(vars: &[u32]) -> Shape {
    match vars {
        [v] => Shape::Var(*v),
        [v, rest @ ..] => join(Shape::Var(*v), right_linear(rest)),
        [] => unreachable!("a block has a variable"),
    }
}

/// Left-linear over `vars`, the first two deepest.
fn left_linear(vars: &[u32]) -> Shape {
    match vars {
        [v] => Shape::Var(*v),
        [rest @ .., v] => join(left_linear(rest), Shape::Var(*v)),
        [] => unreachable!("a block has a variable"),
    }
}

/// Balanced over `vars`, in order.
fn balanced(vars: &[u32]) -> Shape {
    if vars.len() == 1 {
        return Shape::Var(vars[0]);
    }
    let (l, r) = vars.split_at(vars.len() / 2);
    join(balanced(l), balanced(r))
}

fn spell(shape: &Shape, lines: &mut Vec<String>) -> usize {
    let id = match shape {
        Shape::Var(v) => {
            let id = lines.len();
            lines.push(format!("L {id} {v}"));
            return id;
        }
        Shape::Join(l, r) => {
            let (l, r) = (spell(l, lines), spell(r, lines));
            (lines.len(), l, r)
        }
    };
    lines.push(format!("I {} {} {}", id.0, id.1, id.2));
    id.0
}

fn vtree_of(shape: &Shape) -> Arc<Vtree> {
    let mut lines = Vec::new();
    spell(shape, &mut lines);
    let text = format!("vtree {}\n{}\n", lines.len(), lines.join("\n"));
    Arc::new(Vtree::from_text(&text).expect("a well-formed vtree"))
}

/// The shapes every comparison runs on: balanced, right- and left-linear,
/// random, and two blocks of uneven widths joined at the root.
fn shapes(n: u32) -> Vec<(&'static str, Arc<Vtree>)> {
    let vars: Vec<u32> = (1..=n).collect();
    let mut out = vec![
        ("balanced", Arc::new(Vtree::balanced(n))),
        ("right-linear", Arc::new(Vtree::linear(n))),
        ("left-linear", vtree_of(&left_linear(&vars))),
        ("random(7)", Arc::new(Vtree::random(n, 7))),
    ];
    if n >= 3 {
        let (a, b) = vars.split_at(((n as usize) / 3).max(1));
        out.push(("uneven blocks", vtree_of(&join(balanced(a), right_linear(b)))));
        out.push(("uneven blocks, left-linear", vtree_of(&join(left_linear(b), balanced(a)))));
    }
    out
}

/// A random function over `n` variables: `rows` models, drawn over a random
/// subset of the variables so the others are free.
fn random_function(rng: &mut Lcg, vtree: &Arc<Vtree>, n: u32, rows: usize) -> Tdd {
    let vars: Vec<VarId> = (1..=n).filter(|_| rng.below(4) != 0).map(VarId).collect();
    let packed: Vec<u64> = (0..rows).map(|_| rng.next_u64() & ((1u64 << vars.len()) - 1)).collect();
    let f = Tdd::from_models(vtree, &vars, &packed).unwrap();
    assert_canonical(&f);
    f
}

fn truth(f: &Tdd, n: u32) -> Vec<bool> {
    (0..1u64 << n)
        .map(|bits| {
            let asn: Vec<bool> = (0..n).map(|i| (bits >> i) & 1 == 1).collect();
            eval(f, &asn)
        })
        .collect()
}

/// `f ∨ g` the old way, three fills.
fn de_morgan_or(eng: &Engine, f: Tdd, g: Tdd) -> Tdd {
    let nf = eng.negate(f).unwrap();
    let ng = eng.negate(g).unwrap();
    eng.negate(eng.and(nf, ng).unwrap()).unwrap()
}

/// `f ∧ ¬g` through the complement of `g`.
fn and_of_complement(eng: &Engine, f: Tdd, g: Tdd) -> Tdd {
    let ng = eng.negate(g).unwrap();
    let mut h = eng.and(f, ng).unwrap();
    h.minimize().unwrap();
    h
}

/// Check both overlays of `f` and `g` against their complement-based forms:
/// the same function, canonical, and the same diagram.
fn check_pair(eng: &Engine, f: &Tdd, g: &Tdd, n: u32, what: &str) {
    let (tf, tg) = (truth(f, n), truth(g, n));
    let or = overlay_on(eng, Overlay::Or, f.clone(), g.clone()).unwrap();
    assert_canonical(&or);
    let want: Vec<bool> = tf.iter().zip(&tg).map(|(a, b)| *a || *b).collect();
    assert_eq!(truth(&or, n), want, "{what}: or");
    assert_same_shape(&or, &de_morgan_or(eng, f.clone(), g.clone()), &format!("{what}: or"));

    let diff = overlay_on(eng, Overlay::AndNot, f.clone(), g.clone()).unwrap();
    assert_canonical(&diff);
    let want: Vec<bool> = tf.iter().zip(&tg).map(|(a, b)| *a && !*b).collect();
    assert_eq!(truth(&diff, n), want, "{what}: and_not");
    assert_same_shape(&diff, &and_of_complement(eng, f.clone(), g.clone()), &format!("{what}: and_not"));
}

#[test]
fn the_overlay_matches_the_complement_forms_on_random_functions() {
    let eng = &Engine::new();
    let mut rng = Lcg::new(0x0_7e41);
    for n in 1..=9u32 {
        for (name, vtree) in shapes(n) {
            for round in 0..6 {
                let rows_f = 1 + rng.below(1 << n.min(6)) as usize;
                let rows_g = 1 + rng.below(1 << n.min(6)) as usize;
                let f = random_function(&mut rng, &vtree, n, rows_f);
                let g = random_function(&mut rng, &vtree, n, rows_g);
                check_pair(eng, &f, &g, n, &format!("{name}, {n} vars, round {round}"));
            }
        }
    }
}

/// Wide levels: many models over many variables, where a level holds far
/// more lefts and rights than a random clause set gives.
#[test]
fn the_overlay_matches_the_complement_forms_on_wide_levels() {
    let eng = &Engine::new();
    let mut rng = Lcg::new(0x0_3a1d);
    for n in [10u32, 12] {
        for (name, vtree) in shapes(n) {
            for round in 0..3 {
                let (rows_f, rows_g) = (40 + rng.below(200) as usize, 40 + rng.below(200) as usize);
                let f = random_function(&mut rng, &vtree, n, rows_f);
                let g = random_function(&mut rng, &vtree, n, rows_g);
                check_pair(eng, &f, &g, n, &format!("{name}, {n} vars, round {round}"));
                // One operand inside the other, both ways round.
                let both = eng.and(f.clone(), g.clone()).unwrap();
                let mut inner = both;
                inner.minimize().unwrap();
                check_pair(eng, &inner, &f, n, &format!("{name}, {n} vars, round {round}, inner first"));
                check_pair(eng, &f, &inner, n, &format!("{name}, {n} vars, round {round}, inner second"));
            }
        }
    }
}

/// Operands straight out of a conjunction keep unreachable nodes and twins;
/// the overlay reads every node of a level, so it must not be misled by them.
#[test]
fn unreduced_operands_give_the_same_result() {
    let eng = &Engine::new();
    let mut rng = Lcg::new(0x0_51c3);
    for n in [4u32, 7, 9] {
        for (name, vtree) in shapes(n) {
            for round in 0..4 {
                let parts: Vec<Tdd> = (0..4)
                    .map(|_| {
                        let rows = 1 + rng.below(40) as usize;
                        random_function(&mut rng, &vtree, n, rows)
                    })
                    .collect();
                // Neither conjunction is reduced.
                let f = eng.and(parts[0].clone(), parts[1].clone()).unwrap();
                let g = eng.and(parts[2].clone(), parts[3].clone()).unwrap();
                check_pair(eng, &f, &g, n, &format!("{name}, {n} vars, round {round}, unreduced"));
            }
        }
    }
}

#[test]
fn constants_and_equal_operands() {
    let eng = &Engine::new();
    let mut rng = Lcg::new(0x0_c0a5);
    for n in [1u32, 2, 5] {
        for (name, vtree) in shapes(n) {
            let f = random_function(&mut rng, &vtree, n, 3);
            let zero = Tdd::zero(&vtree);
            let one = Tdd::one(&vtree);
            for (a, b, what) in [
                (&f, &zero, "f, false"),
                (&zero, &f, "false, f"),
                (&f, &one, "f, true"),
                (&one, &f, "true, f"),
                (&f, &f, "f, f"),
                (&zero, &zero, "false, false"),
                (&one, &one, "true, true"),
            ] {
                check_pair(eng, a, b, n, &format!("{name}, {n} vars, {what}"));
            }
        }
    }
}

/// Every pair of the four one-variable functions, on the vtree whose root is
/// the leaf itself.
#[test]
fn a_leaf_root_combines_labels() {
    let eng = &Engine::new();
    let vtree = Arc::new(Vtree::balanced(1));
    let fns = [
        Tdd::zero(&vtree),
        Tdd::one(&vtree),
        crate::literal(&vtree, 1).unwrap(),
        crate::literal(&vtree, -1).unwrap(),
    ];
    for (i, f) in fns.iter().enumerate() {
        for (j, g) in fns.iter().enumerate() {
            check_pair(eng, f, g, 1, &format!("leaf root, {i} with {j}"));
        }
    }
}

#[test]
fn the_public_entry_points_route_through_the_overlay() {
    let vtree = Arc::new(Vtree::balanced(3));
    let first = crate::literal(&vtree, 1).unwrap();
    let first_two = Tdd::cube(&vtree, [1, 2]).unwrap();
    let diff = crate::and_not(first.clone(), first_two.clone()).unwrap();
    assert_canonical(&diff);
    assert_eq!(diff.model_count().unwrap(), 2u32.into());
    let union = crate::or(first_two, crate::literal(&vtree, 3).unwrap()).unwrap();
    assert_canonical(&union);
    assert_eq!(union.model_count().unwrap(), 5u32.into());
}

#[test]
fn operands_on_different_vtrees_are_refused() {
    let a = Arc::new(Vtree::balanced(3));
    let b = Arc::new(Vtree::balanced(3));
    let f = crate::literal(&a, 1).unwrap();
    let g = crate::literal(&b, 1).unwrap();
    assert_eq!(crate::and_not(f, g).unwrap_err(), OperationError::VtreeMismatch);
}
