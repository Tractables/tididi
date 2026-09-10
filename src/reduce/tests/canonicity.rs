//! Canonicity: one function over one vtree has one minimized diagram.
//!
//! Every other guarantee the reduction rules make rests on this. It is not
//! checked by comparing model counts — two different diagrams routinely agree
//! on a count — but by building the same function along routes that share no
//! intermediate state, minimizing each, and comparing the results level by
//! level with node numbering normalized away.
//!
//! The routes differ in what they stress. Folding clause by clause conjoins a
//! large accumulator with a tiny operand every step; the tournament conjoins
//! operands of similar size, taking a different path through the apply's
//! product grids; the rotation route puts the diagram over a different vtree
//! and brings it back, so the final diagram is one the restructure primitives
//! rebuilt rather than one the apply produced.

use std::sync::Arc;

use num_bigint::BigInt;
use num_rational::BigRational;

use crate::apply::apply_and;
use crate::build::{clause_to_tdd, constant_one};
use crate::diagram::{Literal, Tdd};
use crate::engine::Engine;
use crate::marginal::{marginalize_closure, weighted_value};
use crate::diagram::RationalWeights;
use crate::query::model_count;
use crate::reduce::minimize;
use crate::restructure::relevel::{
    relevel_after_left_rotation, relevel_after_right_rotation,
};
use crate::restructure::scratch::RestructureScratch;
use crate::test_helpers::{assert_canonical, exact_weight, normalized_levels, Lcg};
use crate::vtree::rotate::{rotate_left, rotate_right};
use crate::vtree::{VarId, Vtree};
use crate::diagram::{Arithmetic, WeightStore};

/// Route 1: fold the clauses left to right into one accumulator.
fn build_by_folding(eng: &Engine, vtree: &Arc<Vtree>, clauses: &[Vec<Literal>]) -> Tdd {
    let mut acc = constant_one(eng, vtree);
    for c in clauses {
        acc = apply_and(acc, clause_to_tdd(eng, vtree, c));
    }
    minimize(&mut acc);
    acc
}

/// Route 2: conjoin the clauses pairwise, halving the operand count each round,
/// so no conjunction sees the lopsided big-against-tiny shape route 1 is made of.
fn build_by_tournament(eng: &Engine, vtree: &Arc<Vtree>, clauses: &[Vec<Literal>]) -> Tdd {
    let mut round: Vec<Tdd> = clauses.iter().map(|c| clause_to_tdd(eng, vtree, c)).collect();
    if round.is_empty() {
        round.push(constant_one(eng, vtree));
    }
    while round.len() > 1 {
        let mut next = Vec::with_capacity(round.len().div_ceil(2));
        let mut it = round.into_iter();
        while let Some(a) = it.next() {
            match it.next() {
                Some(b) => next.push(apply_and(a, b)),
                None => next.push(a),
            }
        }
        round = next;
    }
    let mut out = round.pop().expect("the tournament ends with one diagram");
    minimize(&mut out);
    out
}

/// Route 3: fold the clauses in reverse, then left-rotate the root, rebuild the
/// levels the rotation moved, rotate back, and rebuild again. The vtree ends
/// where it started and the diagram is one the restructure primitives produced.
fn build_by_rotation_round_trip(
    eng: &Engine,
    vtree: &Arc<Vtree>,
    clauses: &[Vec<Literal>],
) -> Option<Tdd> {
    let mut acc = constant_one(eng, vtree);
    for c in clauses.iter().rev() {
        acc = apply_and(acc, clause_to_tdd(eng, vtree, c));
    }
    minimize(&mut acc);

    let mut vt = (**vtree).clone();
    let root = vt.root();
    // A left rotation needs a right child that is internal; a linear-enough
    // vtree at the root has none, and there is nothing to round-trip.
    let left = rotate_left(&mut vt, root)?;
    acc.reseat_vtree(&Arc::new(vt.clone()));
    let mut scratch = RestructureScratch::default();
    relevel_after_left_rotation(&mut acc, &left, &mut scratch, usize::MAX)?;
    minimize(&mut acc);

    let right = rotate_right(&mut vt, root).expect("a left rotation leaves the root right-rotatable");
    acc.reseat_vtree(&Arc::new(vt));
    relevel_after_right_rotation(&mut acc, &right, &mut scratch, usize::MAX)?;
    minimize(&mut acc);
    let nodes = |v: &Vtree| -> Vec<crate::vtree::VtreeNode> {
        (0..v.num_nodes()).map(|i| v.node(crate::vtree::VtreeIdx(i as u32)).clone()).collect()
    };
    assert_eq!(
        nodes(&acc.vtree),
        nodes(vtree),
        "the round trip must land back on the vtree it started from"
    );
    acc.reseat_vtree(vtree);
    Some(acc)
}

/// The weighted mirror: unit weights make every model worth one, so the
/// weighted marginalization of the whole diagram must equal its model count.
/// Reads the integer and semiring value paths against each other on the same
/// structure.
fn weighted_unit_value(eng: &Engine, vtree: &Arc<Vtree>, f: &Tdd) -> BigRational {
    let mut w = f.clone();
    w.set_weights(WeightStore::new(
        RationalWeights::unit(vtree.num_leaves() as usize),
        Arithmetic::ExactRational,
    ));
    marginalize_closure(eng, &mut w, vtree).expect("no wall is installed in a test");
    exact_weight(&weighted_value(&w).expect("a fully marginalized weighted diagram has a value"))
}

/// Random 3-ish-CNFs over a small variable count, as clause literal lists.
fn random_clauses(rng: &mut Lcg, nvars: u32) -> Vec<Vec<Literal>> {
    let nclauses = 1 + rng.below(5) as usize;
    let mut out = Vec::with_capacity(nclauses);
    for _ in 0..nclauses {
        let width = 1 + rng.below(3) as usize;
        let mut seen = vec![false; nvars as usize];
        let mut literals = Vec::new();
        for _ in 0..width {
            let v = rng.below(u64::from(nvars)) as u32;
            if seen[v as usize] {
                continue;
            }
            seen[v as usize] = true;
            literals.push(if rng.coin() { Literal::pos(VarId(v)) } else { Literal::neg(VarId(v)) });
        }
        if literals.is_empty() {
            literals.push(Literal::pos(VarId(0)));
        }
        out.push(literals);
    }
    out
}

/// Three independent routes to the same function, minimized, must agree level
/// by level — and the weighted evaluation of the result must agree with its
/// model count.
#[test]
fn every_route_to_one_function_minimizes_to_the_same_diagram() {
    let eng = Engine::new();
    let mut checked = 0u32;
    let mut rotated = 0u32;
    for &seed in &[0x1234_5678_9abc_def0u64, 0xfeed_face_dead_1234, 0x0bad_c0de_1234_5678] {
        let mut rng = Lcg::new(seed);
        for &nvars in &[3u32, 4, 5, 6] {
            let vtree = Arc::new(Vtree::balanced(nvars));
            for _ in 0..12 {
                let clauses = random_clauses(&mut rng, nvars);
                let folded = build_by_folding(&eng, &vtree, &clauses);
                let tournament = build_by_tournament(&eng, &vtree, &clauses);
                assert_canonical(&folded);
                assert_canonical(&tournament);
                assert_eq!(
                    normalized_levels(&folded),
                    normalized_levels(&tournament),
                    "nvars={nvars} clauses={clauses:?}: folding and the tournament \
                     minimized to different diagrams"
                );
                if let Some(rotated_back) = build_by_rotation_round_trip(&eng, &vtree, &clauses) {
                    assert_canonical(&rotated_back);
                    assert_eq!(
                        normalized_levels(&folded),
                        normalized_levels(&rotated_back),
                        "nvars={nvars} clauses={clauses:?}: the rotation round trip \
                         minimized to a different diagram"
                    );
                    rotated += 1;
                }
                let mc = model_count(&folded);
                assert_eq!(
                    weighted_unit_value(&eng, &vtree, &folded),
                    BigRational::from(BigInt::from(mc)),
                    "nvars={nvars} clauses={clauses:?}: unit weights disagree with the count"
                );
                checked += 1;
            }
        }
    }
    assert!(checked >= 100, "the sweep must actually run: {checked} cases");
    assert!(rotated >= 50, "the rotation route must be exercised, not skipped: {rotated} cases");
}
