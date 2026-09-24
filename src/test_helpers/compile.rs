//! Turning a formula, or a hand-written pair list, into a diagram.

use std::sync::Arc;

use num_rational::BigRational;


use crate::build::constant_one;
use crate::diagram::{ChildPair, Literal, NodeIdx, Tdd, WeightValue};
use crate::Engine;

use super::r#gen::Lcg;
use crate::vtree::{VarId, Vtree};

/// DIMACS-style literals (`±var`) to `Literal`s.
pub fn literals(clause: &[i32]) -> Vec<Literal> {
    clause.iter().map(|&l| Literal::new(VarId(l.unsigned_abs()), l > 0)).collect()
}

/// `(var, polarity)` pairs to `Literal`s, for tests that carry the polarity
/// separately.
pub fn clause(literals: &[(u32, bool)]) -> Vec<Literal> {
    literals.iter().map(|&(v, positive)| Literal::new(VarId(v), positive)).collect()
}

/// Conjoin DIMACS-style clauses one at a time, minimizing after each, on a
/// fresh engine.
pub fn compile_clauses(vtree: &Arc<Vtree>, clauses: &[Vec<i32>]) -> Tdd {
    compile_clauses_on(&crate::Engine::new(), vtree, clauses)
}

/// Conjoin DIMACS-style clauses as a pairwise tree, neighbours first and then
/// their results, and minimize once: a second route to the diagram
/// [`compile_clauses`] reaches by a left fold.
pub fn compile_clauses_pairwise(vtree: &Arc<Vtree>, clauses: &[Vec<i32>]) -> Tdd {
    let mut queue: Vec<Tdd> = clauses.iter().map(|c| Tdd::clause(vtree, c).unwrap()).collect();
    while queue.len() > 1 {
        let mut next = Vec::with_capacity(queue.len().div_ceil(2));
        let mut it = queue.into_iter();
        while let Some(a) = it.next() {
            next.push(match it.next() {
                Some(b) => crate::and(a, b).unwrap(),
                None => a,
            });
        }
        queue = next;
    }
    let mut tree = queue.pop().unwrap_or_else(|| Tdd::one(vtree));
    tree.minimize().unwrap();
    tree
}

/// The disjunction of one cube per true row of a truth table over `vars`:
/// row `r` is in when `truth(r)`, and takes `vars[i]` positive when bit `i`
/// of `r` is set. Every other variable stays free.
pub fn or_of_cubes(vtree: &Arc<Vtree>, vars: &[VarId], truth: impl Fn(usize) -> bool) -> Tdd {
    let mut f = Tdd::zero(vtree);
    for row in 0..1usize << vars.len() {
        if truth(row) {
            let cube: Vec<Literal> = vars
                .iter()
                .enumerate()
                .map(|(i, &var)| Literal::new(var, (row >> i) & 1 == 1))
                .collect();
            f = crate::or(f, Tdd::cube(vtree, cube).unwrap()).unwrap();
        }
    }
    f
}

/// [`compile_clauses`] on a caller's engine, so a test that installed its own
/// thresholds gets every conjunction and reduction of the fold decided by them.
pub fn compile_clauses_on(eng: &Engine, vtree: &Arc<Vtree>, clauses: &[Vec<i32>]) -> Tdd {
    let mut acc = constant_one(eng, vtree);
    for clause in clauses {
        let cl = clause_to_tdd(eng, vtree, &literals(clause));
        acc = eng.and(acc, cl).expect("compile_clauses_on: allocation refused");
        eng.reduce(&mut acc, crate::reduce::ReductionPlan::default())
            .expect("compile_clauses_on: allocation refused");
    }
    acc
}

/// One input pair from two raw child references, for the hand-built fixtures.
/// Under the bare-is-slot polarity a bare index on a marginal side is a slot
/// reference, so this is also how those fixtures name slots.
pub fn pair(l: u32, r: u32) -> ChildPair {
    ChildPair::new(NodeIdx(l), NodeIdx(r))
}

/// `n/d` as a `BigRational`, for the weighted fixtures.
pub fn rat(n: i64, d: i64) -> BigRational {
    BigRational::new(n.into(), d.into())
}

/// The value an exact [`WeightValue`] carries, in whichever representation it is
/// in. Panics on a logarithmic one, which has no exact rational.
pub fn exact_weight(v: &WeightValue) -> BigRational {
    assert!(!matches!(v, WeightValue::Log(_)), "expected an exact WeightValue");
    v.as_rational().into_owned()
}

/// Conjunction of two diagrams, both cloned, so a fixture survives being
/// conjoined and can be reused for the next case.
pub fn and2(a: &Tdd, b: &Tdd) -> Tdd {
    crate::apply::apply_and(a.clone(), b.clone())
}

/// A cube — a conjunction of literals — as a diagram, folded one unit clause
/// at a time.
pub fn cube(vtree: &Arc<Vtree>, literals: &[(u32, bool)]) -> Tdd {
    let eng = Engine::new();
    let mut acc = clause_to_tdd(&eng, vtree, &clause(&[literals[0]]));
    for &l in &literals[1..] {
        acc = and2(&acc, &clause_to_tdd(&eng, vtree, &clause(&[l])));
    }
    acc
}

/// A random conjunction of clauses whose literals are drawn from `vars`.
///
/// Clause count comes from `1..=nclauses_max` and each clause's width from
/// `1..=width_max`; a variable drawn twice in one clause is skipped, so the
/// widths are a distribution. With `span`, the first and last of `vars` are
/// conjoined in as an extra clause, which forces the support to reach both
/// ends and so roots the result at the vtree root — the same-root shape
/// restriction needs to do anything at all.
pub fn rand_conj_over(
    vtree: &Arc<Vtree>,
    vars: &[u32],
    nclauses_max: u64,
    width_max: u64,
    span: bool,
    rng: &mut Lcg,
) -> Tdd {
    let eng = Engine::new();
    let mut acc: Option<Tdd> = if span && vars.len() >= 2 {
        let ends = clause(&[(vars[0], true), (vars[vars.len() - 1], true)]);
        Some(clause_to_tdd(&eng, vtree, &ends))
    } else {
        None
    };
    let nclauses = 1 + rng.below(nclauses_max) as usize;
    for _ in 0..nclauses {
        let width = 1 + rng.below(width_max) as usize;
        let mut literals: Vec<(u32, bool)> = Vec::new();
        for _ in 0..width {
            let v = vars[(rng.next_u64() as usize) % vars.len()];
            let positive = rng.coin();
            if literals.iter().any(|(u, _)| *u == v) {
                continue;
            }
            literals.push((v, positive));
        }
        literals.sort_by_key(|&(v, _)| v);
        let cl = clause_to_tdd(&eng, vtree, &clause(&literals));
        acc = Some(match acc {
            None => cl,
            Some(a) => and2(&a, &cl),
        });
    }
    acc.expect("a draw of at least one clause")
}

/// [`rand_conj_over`] across every variable of an `nvars` vtree.
pub fn rand_conj(
    vtree: &Arc<Vtree>,
    nvars: u32,
    nclauses_max: u64,
    width_max: u64,
    span: bool,
    rng: &mut Lcg,
) -> Tdd {
    let vars: Vec<u32> = (1..=nvars).collect();
    rand_conj_over(vtree, &vars, nclauses_max, width_max, span, rng)
}

/// Build an unlimited clause fixture independently of the operation's armed engine.
pub(crate) fn clause_to_tdd(_eng: &crate::Engine, vtree: &std::sync::Arc<crate::vtree::Vtree>, clause: &[crate::Literal]) -> crate::Tdd {
    crate::Tdd::clause(vtree, clause).unwrap()
}
