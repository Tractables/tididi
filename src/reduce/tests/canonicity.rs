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
use crate::build::constant_one;
use crate::diagram::Tdd;
use crate::Engine;
use crate::marginal::marginalize_closure;

use crate::diagram::RationalWeights;


use crate::restructure::relevel::rebuild_rotated_levels;
use crate::vtree::RotationKind;
use crate::restructure::scratch::RestructureScratch;
use crate::test_helpers::{assert_canonical, assert_same_shape, compile_clauses_pairwise, exact_weight, rand_cnf, rotate_left, rotate_right, CnfShape, Lcg};
use crate::vtree::Vtree;
use crate::diagram::{Arithmetic, WeightStore};

/// Route 1: fold the clauses left to right into one accumulator.
fn build_by_folding(eng: &Engine, vtree: &Arc<Vtree>, clauses: &[Vec<i32>]) -> Tdd {
    let mut acc = constant_one(eng, vtree);
    for c in clauses {
        acc = apply_and(acc, Tdd::clause(vtree, c).unwrap());
    }
    acc.minimize().unwrap();
    acc
}

/// Route 3: fold the clauses in reverse, then left-rotate the root, rebuild the
/// levels the rotation moved, rotate back, and rebuild again. The vtree ends
/// where it started and the diagram is one the restructure primitives produced.
fn build_by_rotation_round_trip(
    eng: &Engine,
    vtree: &Arc<Vtree>,
    clauses: &[Vec<i32>],
) -> Option<Tdd> {
    let mut acc = constant_one(eng, vtree);
    for c in clauses.iter().rev() {
        acc = apply_and(acc, Tdd::clause(vtree, c).unwrap());
    }
    acc.minimize().unwrap();

    let mut vt = (**vtree).clone();
    let root = vt.root();
    // A left rotation needs a right child that is internal; a linear-enough
    // vtree at the root has none, and there is nothing to round-trip.
    let left = rotate_left(&mut vt, root)?;
    acc.vtree = Arc::new(vt.clone());
    let mut scratch = RestructureScratch::default();
    rebuild_rotated_levels(eng.limits(), &mut acc, &left, RotationKind::Left, &mut scratch, usize::MAX).expect("nothing is armed")?;
    acc.minimize().unwrap();

    let right = rotate_right(&mut vt, root).expect("a left rotation leaves the root right-rotatable");
    acc.vtree = Arc::new(vt);
    rebuild_rotated_levels(eng.limits(), &mut acc, &right, RotationKind::Right, &mut scratch, usize::MAX).expect("nothing is armed")?;
    acc.minimize().unwrap();
    let nodes = |v: &Vtree| -> Vec<crate::vtree::VtreeNode> {
        (0..v.num_nodes()).map(|i| v.node(crate::vtree::VtreeIdx(i as u32)).clone()).collect()
    };
    assert_eq!(
        nodes(&acc.vtree),
        nodes(vtree),
        "the round trip must land back on the vtree it started from"
    );
    acc.vtree = Arc::clone(vtree);
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
    )).unwrap();
    marginalize_closure(eng, &mut w).expect("no wall is installed in a test");
    exact_weight(&w.weighted_value().unwrap().expect("a fully marginalized weighted diagram has a value"))
}

/// Random 3-ish-CNFs over a small variable count, as clause literal lists.
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
                let clauses = rand_cnf(&mut rng, nvars, CnfShape { clauses: 5, width: 3 });
                let folded = build_by_folding(&eng, &vtree, &clauses);
                let pairwise = compile_clauses_pairwise(&vtree, &clauses);
                assert_canonical(&folded);
                assert_canonical(&pairwise);
                assert_same_shape(&folded, &pairwise, &format!("nvars={nvars} clauses={clauses:?}: folding and pairwise"));
                if let Some(rotated_back) = build_by_rotation_round_trip(&eng, &vtree, &clauses) {
                    assert_canonical(&rotated_back);
                    assert_same_shape(&folded, &rotated_back, &format!("nvars={nvars} clauses={clauses:?}: folding and the rotation round trip"));
                    rotated += 1;
                }
                let mc = folded.model_count().unwrap();
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
