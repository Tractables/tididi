//! Turning a formula, or a hand-written pair list, into a diagram.

use std::sync::Arc;

use num_rational::BigRational;

use crate::apply::conjoin_clause::clause_to_tdd;
use crate::build::constant_one;
use crate::diagram::{InputPair, Literal, NodeIdx, Tdd, TddNodeId, WeightVal};
use crate::engine::Engine;

use super::r#gen::Lcg;
use crate::vtree::{VarId, Vtree, VtreeNode};

/// DIMACS-style literals (`±(var+1)`) to `Literal`s.
pub fn literals(clause: &[i32]) -> Vec<Literal> {
    clause.iter().map(|&l| Literal::new(VarId(l.unsigned_abs() - 1), l > 0)).collect()
}

/// `(var, polarity)` pairs to `Literal`s, for tests that name variables by
/// their 0-based index rather than in DIMACS.
pub fn clause(literals: &[(u32, bool)]) -> Vec<Literal> {
    literals.iter().map(|&(v, positive)| Literal::new(VarId(v), positive)).collect()
}

/// Conjoin DIMACS-style clauses one at a time, minimizing after each, on a
/// fresh engine.
pub fn compile_clauses(vtree: &Arc<Vtree>, clauses: &[Vec<i32>]) -> Tdd {
    compile_clauses_on(&crate::engine::Engine::new(), vtree, clauses)
}

/// [`compile_clauses`] on a caller's engine, so a test that installed its own
/// thresholds gets every conjunction and reduction of the fold decided by them.
pub fn compile_clauses_on(eng: &Engine, vtree: &Arc<Vtree>, clauses: &[Vec<i32>]) -> Tdd {
    let mut acc = constant_one(eng, vtree);
    for clause in clauses {
        let cl = clause_to_tdd(eng, vtree, &literals(clause));
        acc = eng.and(acc, cl).expect("compile_clauses_on: allocation refused");
        crate::reduce::try_minimize(eng, &mut acc, crate::reduce::MinimizeOptions::default())
            .expect("compile_clauses_on: allocation refused");
    }
    acc
}

/// One input pair from two raw child references, for the hand-built fixtures.
/// Under the bare-is-slot polarity a bare index on a marginal side is a slot
/// reference, so this is also how those fixtures name slots.
pub fn pair(l: u32, r: u32) -> InputPair {
    InputPair { left: NodeIdx(l), right: NodeIdx(r) }
}

/// `n/d` as a `BigRational`, for the weighted fixtures.
pub fn rat(n: i64, d: i64) -> BigRational {
    BigRational::new(n.into(), d.into())
}

/// The value an exact [`WeightVal`] carries, in whichever representation it is
/// in. Panics on a logarithmic one, which has no exact rational.
pub fn exact_weight(v: &WeightVal) -> BigRational {
    assert!(!matches!(v, WeightVal::Log(_)), "expected an exact WeightVal");
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
    let vars: Vec<u32> = (0..nvars).collect();
    rand_conj_over(vtree, &vars, nclauses_max, width_max, span, rng)
}

/// Re-home a diagram that depends only on variables under one child of its
/// root so that it is rooted at that child — a genuinely low-rooted diagram.
///
/// Building and applying always root at the vtree root, so re-homing is the
/// only way to reach the differing-root operand shape a tightly-rooted segment
/// would take. The root level must hold a single identity pair, the `g ∧ ⊤`
/// shape a single-region function compiles to.
pub fn reroot_to_child(t: &Tdd, left_child: bool) -> Tdd {
    let root = t.output.vtree;
    let (lc, rc) = match *t.vtree.node(root) {
        VtreeNode::Internal { left, right, .. } => (left, right),
        VtreeNode::Leaf { .. } => panic!("reroot_to_child: the root must be internal"),
    };
    let pairs = t.levels[root.idx()].pairs_of_idx(t.output.local.idx());
    assert_eq!(pairs.len(), 1, "reroot_to_child expects the single-region g ∧ ⊤ shape");
    let p = pairs[0];
    let (child, local) = if left_child { (lc, p.left) } else { (rc, p.right) };
    Tdd::from_levels_unchecked(
        t.vtree.clone(),
        t.levels.clone(),
        TddNodeId { vtree: child, local },
    )
}
