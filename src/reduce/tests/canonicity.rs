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

use crate::Engine;
use crate::apply::apply_and;
use crate::build::constant_one;
use crate::diagram::{Arithmetic, RationalWeights, Tdd, Weights};
use crate::marginal::marginalize_closure;
use crate::restructure::relevel::{RotationStep, rebuild_levels};
use crate::restructure::scratch::RestructureScratch;
use crate::test_helpers::{
    CnfShape, Lcg, assert_canonical, assert_same_shape, compile_clauses_pairwise, exact_weight,
    rand_cnf,
};
use crate::vtree::rotate::{RotationInfo, rotate_pointers};
use crate::vtree::{Home, RotationKind, VarId, Vtree, VtreeIdx};

/// Route 1: fold `clauses` into one accumulator.
fn build_by_folding<'a>(
    eng: &Engine,
    vtree: &Arc<Vtree>,
    clauses: impl IntoIterator<Item = &'a Vec<i32>>,
) -> Tdd {
    let mut acc = constant_one(eng, vtree);
    for c in clauses {
        acc = apply_and(acc, Tdd::clause(&crate::vtree::Home::unchecked(vtree), c).unwrap());
    }
    acc.minimize().unwrap();
    acc
}

/// Route 3: fold the clauses in reverse, then left-rotate the root of the
/// diagram's vtree, rebuild the levels the rotation moved, rotate back, and
/// rebuild again.
fn build_by_rotation_round_trip(
    eng: &Engine,
    vtree: &Arc<Vtree>,
    clauses: &[Vec<i32>],
) -> Option<Tdd> {
    let mut acc = build_by_folding(eng, vtree, clauses.iter().rev());
    let own = Arc::clone(&acc.vtree);
    // A left rotation needs a right child that is internal; a linear-enough
    // vtree at the root has none, and there is nothing to round-trip.
    if own.num_nodes() == 0 {
        return None;
    }
    let root = own.root();
    let left = rotate_own(&mut acc, root, RotationKind::Left)?;
    let mut scratch = RestructureScratch::default();
    let marginal = acc.has_marginal_level();
    let step = RotationStep { info: &left, dir: RotationKind::Left, crossed: false, max_pairs: usize::MAX };
    rebuild_levels(eng.limits(), &mut acc, step, marginal, &mut scratch).unwrap()?;
    acc.minimize().unwrap();

    let right = rotate_own(&mut acc, root, RotationKind::Right)
        .expect("a left rotation leaves the root right-rotatable");
    let marginal = acc.has_marginal_level();
    let step = RotationStep { info: &right, dir: RotationKind::Right, crossed: false, max_pairs: usize::MAX };
    rebuild_levels(eng.limits(), &mut acc, step, marginal, &mut scratch).unwrap()?;
    acc.minimize().unwrap();
    assert!(
        acc.vtree.same_tree(&own),
        "the round trip must land back on the vtree it started from"
    );
    acc.vtree = own;
    Some(acc)
}

/// Rotate `acc`'s own vtree, a projection of its home, in place, as the
/// rotation search does.
fn rotate_own(acc: &mut Tdd, v: VtreeIdx, kind: RotationKind) -> Option<RotationInfo> {
    let pending = rotate_pointers(Vtree::make_mut(&mut acc.vtree), v, kind)?;
    Some(pending.commit_in_probe(Vtree::make_mut(&mut acc.vtree)))
}

/// The weighted mirror: unit weights make every model worth one, so the
/// weighted marginalization of the whole diagram must equal its model count.
/// Reads the integer and semiring value paths against each other on the same
/// structure.
fn weighted_unit_value(eng: &Engine, vtree: &Arc<Vtree>, f: &Tdd) -> BigRational {
    let weighted = Arc::new(Vtree::clone(vtree).with_weights(Weights::new(
        RationalWeights::unit(vtree.num_leaves() as usize),
        Arithmetic::ExactRational,
    )).unwrap());
    let mut w = crate::test_helpers::embedded(f, &weighted, |v| v).unwrap().0;
    marginalize_closure(eng, &mut w).expect("no wall is installed in a test");
    exact_weight(
        &w.weighted_value()
            .unwrap()
            .expect("a fully marginalized weighted diagram has a value"),
    )
}

/// Random 3-ish-CNFs over a small variable count, as clause literal lists.
/// Three independent routes to the same function, minimized, must agree level
/// by level — and the weighted evaluation of the result must agree with its
/// model count. On implicit levels and on stored ones alike
/// ([`same_as_stored`](crate::test_helpers::same_as_stored)).
#[test]
fn every_route_to_one_function_minimizes_to_the_same_diagram() {
    crate::test_helpers::same_as_stored(every_route_to_one_function);
}

/// The cases of [`every_route_to_one_function_minimizes_to_the_same_diagram`]:
/// the diagrams of every route.
fn every_route_to_one_function() -> Vec<Tdd> {
    let mut out = Vec::new();
    let eng = Engine::new();
    let mut checked = 0u32;
    let mut rotated = 0u32;
    for &seed in &[
        0x1234_5678_9abc_def0u64,
        0xfeed_face_dead_1234,
        0x0bad_c0de_1234_5678,
    ] {
        let mut rng = Lcg::new(seed);
        for &nvars in &[3u32, 4, 5, 6] {
            let vtree = Home::new(Vtree::balanced(nvars));
            for _ in 0..12 {
                let clauses = rand_cnf(
                    &mut rng,
                    nvars,
                    CnfShape {
                        clauses: 5,
                        width: 3,
                    },
                );
                let folded = build_by_folding(&eng, &vtree, &clauses);
                let pairwise = compile_clauses_pairwise(&vtree, &clauses);
                assert_canonical(&folded);
                assert_canonical(&pairwise);
                assert_same_shape(
                    &folded,
                    &pairwise,
                    &format!("nvars={nvars} clauses={clauses:?}: folding and pairwise"),
                );
                if let Some(rotated_back) = build_by_rotation_round_trip(&eng, &vtree, &clauses) {
                    assert_canonical(&rotated_back);
                    assert_same_shape(
                        &folded,
                        &rotated_back,
                        &format!(
                            "nvars={nvars} clauses={clauses:?}: folding and the rotation round trip"
                        ),
                    );
                    rotated += 1;
                    out.push(rotated_back);
                }
                let mc = folded.model_count().unwrap();
                assert_eq!(
                    weighted_unit_value(&eng, &vtree, &folded),
                    BigRational::from(BigInt::from(mc)),
                    "nvars={nvars} clauses={clauses:?}: unit weights disagree with the count"
                );
                checked += 1;
                out.extend([folded, pairwise]);
            }
        }
    }
    assert!(
        checked >= 100,
        "the sweep must actually run: {checked} cases"
    );
    assert!(
        rotated >= 50,
        "the rotation route must be exercised, not skipped: {rotated} cases"
    );
    out
}

/// Quantification rewrites the levels above the quantified leaves and puts
/// only those on the contraction worklist, so contraction reaches the levels
/// below them through the contractions above, and searches them by list
/// (checked against whole searches in debug builds). Quantifying a set at
/// once out of the folded diagram and one variable at a time out of the
/// pairwise one must end on the same canonical diagram.
#[test]
fn listed_twin_searches_reach_the_canonical_diagram() {
    let eng = Engine::new();
    let mut rng = Lcg::new(0x5eed_1157_ed00_0001);
    let before = crate::reduce::contract::tests::listed_searches();
    for &nvars in &[6u32, 8, 10] {
        let vtree = Home::new(Vtree::balanced(nvars));
        for _ in 0..16 {
            let clauses = rand_cnf(
                &mut rng,
                nvars,
                CnfShape {
                    clauses: 2 * nvars as usize,
                    width: 3,
                },
            );
            let quantified: Vec<VarId> = (1..=nvars)
                .filter(|_| rng.below(3) == 0)
                .map(VarId)
                .collect();
            let at_once = crate::test_helpers::minimized(&eng, eng.exists_vars(build_by_folding(&eng, &vtree, &clauses), &quantified));
            let mut one_at_a_time = compile_clauses_pairwise(&vtree, &clauses);
            for &x in &quantified {
                one_at_a_time = crate::test_helpers::minimized(&eng, eng.exists_var(one_at_a_time, x));
            }
            assert_canonical(&at_once);
            assert_same_shape(
                &at_once,
                &one_at_a_time,
                &format!("nvars={nvars} clauses={clauses:?} quantified={quantified:?}"),
            );
        }
    }
    let listed = crate::reduce::contract::tests::listed_searches() - before;
    assert!(listed > 0, "no contraction was searched by list");
}
