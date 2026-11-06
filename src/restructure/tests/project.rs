use std::sync::Arc;

use crate::restructure::EmbedError;
use crate::test_helpers::{assert_canonical, compile_clauses, random_diagrams, test_cases, vtree_shapes};
use crate::vtree::{VarId, Vtree, VtreeError};
use crate::{Engine, OperationError, Tdd};

/// The variables of `vtree` each pattern keeps, in leaf order: every other
/// one, the left subtree of the root, the first, every one but the first,
/// and all of them.
fn patterns(vtree: &Vtree) -> Vec<Vec<VarId>> {
    let leaves: Vec<VarId> = vtree.leaf_bottomup().map(|(_, v)| v).collect();
    let mut out = vec![
        leaves.iter().copied().step_by(2).collect::<Vec<_>>(),
        vec![leaves[0]],
        leaves.clone(),
    ];
    if leaves.len() > 1 {
        out.push(leaves[1..].to_vec());
        let (left, _) = vtree.children(vtree.root());
        out.push(vtree.leaf_bottomup().filter(|&(t, _)| vtree.in_subtree(t, left)).map(|(_, v)| v).collect());
    }
    out
}

/// Project `f` onto `keep` (local ids in `keep`'s order) and check the
/// result against `∃ dropped. f` by embedding it back.
fn check(f: &Tdd, keep: &[VarId]) {
    let local_of = |v: VarId| keep.iter().position(|&k| k == v).map(|i| VarId(i as u32 + 1));
    let g = f.project_to_vars(local_of, keep.len() as u32).unwrap();
    assert_canonical(&g);
    assert_eq!(g.vtree().num_leaves() as usize, keep.len());
    let dropped: Vec<VarId> = f.vtree().leaf_bottomup().map(|(_, v)| v).filter(|v| !keep.contains(v)).collect();
    let expected = f.clone().exists_vars(&dropped).unwrap();
    let (back, _) = g.embed(f.vtree(), |v| keep[v.idx()]).unwrap();
    assert!(back.equivalent(&expected).unwrap(), "keep {keep:?}");
    assert_eq!(g.model_count().unwrap(), f.projected_model_count(keep).unwrap());
}

#[test]
fn a_projection_is_the_quantified_diagram_on_the_restricted_vtree() {
    for (num_vars, clauses) in test_cases() {
        for (_, vtree) in vtree_shapes(num_vars) {
            let f = compile_clauses(&vtree, &clauses);
            for keep in patterns(&vtree) {
                check(&f, &keep);
            }
        }
    }
}

#[test]
fn unreduced_diagrams_project_to_canonical_ones() {
    for f in random_diagrams(7, 24, 3..9) {
        for keep in patterns(f.vtree()) {
            check(&f, &keep);
        }
    }
}

#[test]
fn a_diagram_free_in_the_dropped_variables_keeps_its_size() {
    // x1 ∨ x3 on four variables: x2 and x4 free.
    for (_, vtree) in vtree_shapes(4) {
        let f = compile_clauses(&vtree, &[vec![1, 3]]);
        let keep = [VarId(1), VarId(3)];
        check(&f, &keep);
        let g = f.project_to_vars(|v| keep.iter().position(|&k| k == v).map(|i| VarId(i as u32 + 1)), 2).unwrap();
        assert!(g.pair_count() <= f.pair_count());
    }
}

#[test]
fn false_projects_to_false() {
    let vtree = Arc::new(Vtree::balanced(4));
    let f = compile_clauses(&vtree, &[vec![1], vec![-1]]);
    let g = f.project_to_vars(|v| (v.0 <= 2).then_some(v), 2).unwrap();
    assert!(g.is_zero());
    assert_canonical(&g);
}

#[test]
fn ids_are_checked() {
    let vtree = Arc::new(Vtree::balanced(4));
    let f = compile_clauses(&vtree, &[vec![1, 2]]);
    assert!(matches!(
        f.project_to_vars(Some, 3),
        Err(EmbedError::VariableOutOfRange { variable: VarId(4), num_vars: 3 }),
    ));
    assert!(matches!(
        f.project_to_vars(|_| Some(VarId(1)), 4),
        Err(EmbedError::Vtree(VtreeError::OverlappingVariable(VarId(1)))),
    ));
    assert!(matches!(f.project_to_vars(|_| None, 4), Err(EmbedError::Vtree(VtreeError::Invalid(_)))));
}

#[test]
fn a_stopped_batch_refuses() {
    use crate::limits::{LimitConfig, StopAt, StopRules};

    let vtree = Arc::new(Vtree::balanced(4));
    let f = compile_clauses(&vtree, &[vec![1, 2], vec![3, 4]]);
    let eng = Engine::new();
    let result = {
        let _scope = eng.limits().scope(LimitConfig::none().with_stop_rules(StopRules {
            unconditional: Some(StopAt::WorkUnits(1)), ..StopRules::default()
        }));
        eng.project_to_vars(&f, |v| (v.0 % 2 == 1).then(|| VarId(v.0.div_ceil(2))), 2)
    };
    assert_eq!(result.unwrap_err(), EmbedError::Operation(OperationError::Stopped));
    let g = eng.project_to_vars(&f, |v| (v.0 % 2 == 1).then(|| VarId(v.0.div_ceil(2))), 2).unwrap();
    assert_canonical(&g);
}

mod weighted {
    //! Projections of diagrams that have summed levels out
    //! ([`Engine::project_over`]).

    use std::sync::Arc;

    use num_rational::BigRational;

    use crate::diagram::{Arithmetic, LiteralWeights, RationalWeights, TddBuildError, WeightStore};
    use crate::restructure::EmbedError;
    use crate::test_helpers::{assert_canonical, compile_clauses, test_cases, vtree_shapes};
    use crate::vtree::{VarId, Vtree, VtreeIdx};
    use crate::{Engine, OperationError, Tdd};

    /// Weights that differ by variable, so that a renaming that mixes two
    /// up shows: `x` weighs `1/(x + 1)` and `¬x` weighs `x/(x + 2)`.
    fn weight_of(v: VarId) -> LiteralWeights<BigRational> {
        let x = i64::from(v.0);
        LiteralWeights { negative: BigRational::new(x.into(), (x + 2).into()), positive: BigRational::new(1.into(), (x + 1).into()) }
    }

    /// The weights of `vars`, the variable at position `i` as local `i + 1`.
    fn store_over(vars: &[VarId], arithmetic: Arithmetic) -> WeightStore {
        let table: Vec<_> = vars.iter().map(|&v| weight_of(v)).collect();
        WeightStore::new(RationalWeights::from_literals(&table), arithmetic)
    }

    fn exact(f: &Tdd) -> BigRational {
        f.weighted_value().unwrap().expect("weighted").into_rational()
    }

    /// The leftmost leaf under `t`.
    fn first_leaf(vtree: &Vtree, mut t: VtreeIdx) -> VtreeIdx {
        while !vtree.node(t).is_leaf() {
            t = vtree.children(t).0;
        }
        t
    }

    /// `f` under the weights of its variables with the subtree at `t`
    /// summed out.
    fn summed(f: &Tdd, t: VtreeIdx, arithmetic: Arithmetic) -> Tdd {
        let vars: Vec<VarId> = (1..=f.vtree().num_vars()).map(VarId).collect();
        let mut g = f.clone();
        g.set_weights(store_over(&vars, arithmetic)).unwrap();
        Engine::new().marginalize_levels(&mut g, &[t]).unwrap();
        g
    }

    /// The variables outside `t` and the first leaf of each side of `t`, in
    /// leaf order: what a walk that summed `t` out still needs.
    fn kept_around(vtree: &Vtree, t: VtreeIdx) -> Vec<VarId> {
        let (left, right) = vtree.children(t);
        let reps = [first_leaf(vtree, left), first_leaf(vtree, right)];
        vtree
            .leaf_bottomup()
            .filter(|&(leaf, _)| !vtree.in_subtree(leaf, t) || reps.contains(&leaf))
            .map(|(_, v)| v)
            .collect()
    }

    fn local_in(keep: &[VarId]) -> impl Fn(VarId) -> Option<VarId> + Copy + '_ {
        move |v| keep.iter().position(|&k| k == v).map(|i| VarId(i as u32 + 1))
    }

    /// Project `g`, summed out at `t`, onto `keep` and check the result: the
    /// same value, `t`'s values at its image, and the same value again for
    /// a conjunction with a clause over the variables outside `t`.
    fn check(g: &Tdd, t: VtreeIdx, keep: &[VarId]) {
        let engine = Engine::new();
        let vtree = g.vtree();
        let local_of = local_in(keep);
        let into = Arc::new(vtree.project_to_vars(local_of, keep.len() as u32).unwrap());
        let arithmetic = g.weights().unwrap().arithmetic();
        // Bounded values summed out may leave twins the oracle flags in the
        // source already; exact ones are canonical before and after.
        let exactly = arithmetic == Arithmetic::ExactRational;
        if exactly {
            assert_canonical(g);
        }
        let p = engine.project_over(g, &into, local_of, Some(store_over(keep, arithmetic))).unwrap();
        if exactly {
            assert_canonical(&p);
        }
        assert_eq!(p.vtree().num_leaves() as usize, keep.len());
        let key = |f: &Tdd| crate::diagram::weight_key(&f.weighted_value().unwrap().expect("weighted"));
        assert!(key(&p) == key(g), "the same value");
        // `t`'s image: the node over the variables kept under `t`.
        let under_t = vtree.leaf_bottomup().filter(|&(leaf, v)| vtree.in_subtree(leaf, t) && local_of(v).is_some()).count();
        let mut image = into.leaf_of(local_of(vtree.leaf_var(first_leaf(vtree, t))).unwrap()).unwrap();
        while into.leaf_bottomup().filter(|&(leaf, _)| into.in_subtree(leaf, image)).count() < under_t {
            image = into.node(image).parent().unwrap();
        }
        assert!(p.level(image).is_weight_marginal(), "t's image holds its values");
        assert_eq!(p.level(image).slot_count(), g.level(t).slot_count());
        if !exactly {
            return;
        }
        let outside: Vec<VarId> = vtree.leaf_bottomup().filter(|&(leaf, _)| !vtree.in_subtree(leaf, t)).map(|(_, v)| v).collect();
        if let [a, rest @ ..] = &outside[..] {
            let clause = |x: VarId, y: Option<VarId>| {
                let mut c = vec![x.0 as i32];
                c.extend(y.map(|y| -(y.0 as i32)));
                c
            };
            let y = rest.first().copied();
            let k = engine.clause(vtree, clause(*a, y)).unwrap();
            let local = |v: VarId| local_of(v).unwrap();
            let k_local = engine.clause(&into, clause(local(*a), y.map(local))).unwrap();
            let gk = engine.and(g.clone(), k).unwrap();
            let pk = engine.and(p, k_local).unwrap();
            assert_eq!(exact(&gk), exact(&pk));
        }
    }

    #[test]
    fn a_summed_subtree_keeps_its_level_over_one_leaf_of_each_side() {
        for (num_vars, clauses) in test_cases() {
            for (_, vtree) in vtree_shapes(num_vars) {
                let f = compile_clauses(&vtree, &clauses);
                if f.is_zero() {
                    continue;
                }
                for t in vtree.bottomup().filter(|&t| !vtree.node(t).is_leaf()) {
                    let g = summed(&f, t, Arithmetic::ExactRational);
                    check(&g, t, &kept_around(&vtree, t));
                    // Every variable kept but one on a side of `t` that
                    // keeps another: levels under `t` kept too.
                    let (_, right) = vtree.children(t);
                    if !vtree.node(right).is_leaf() {
                        let gone = vtree.leaf_var(first_leaf(&vtree, right));
                        let rest: Vec<VarId> = vtree.leaf_bottomup().map(|(_, v)| v).filter(|&v| v != gone).collect();
                        check(&g, t, &rest);
                    }
                }
            }
        }
    }

    #[test]
    fn bounded_values_are_carried_as_they_are() {
        for arithmetic in [Arithmetic::ScaledFloat, Arithmetic::SignedLog] {
            for (_, vtree) in vtree_shapes(6) {
                let f = compile_clauses(&vtree, &[vec![1, 2, -3], vec![-2, 4, 5], vec![3, -5, 6], vec![-1, -6]]);
                for t in vtree.bottomup().filter(|&t| !vtree.node(t).is_leaf()) {
                    check(&summed(&f, t, arithmetic), t, &kept_around(&vtree, t));
                }
            }
        }
    }

    #[test]
    fn a_structural_diagram_is_projected_and_takes_the_store() {
        let vtree = Arc::new(Vtree::balanced(4));
        let f = compile_clauses(&vtree, &[vec![1, 2], vec![3, 4]]);
        let keep = [VarId(1), VarId(3)];
        let local_of = local_in(&keep);
        let into = Arc::new(vtree.project_to_vars(local_of, 2).unwrap());
        let engine = Engine::new();
        let plain = engine.project_to_vars(&f, local_of, 2).unwrap();
        let g = engine.project_over(&f, &into, local_of, Some(store_over(&keep, Arithmetic::ExactRational))).unwrap();
        assert!(g.weights().is_some());
        assert_eq!((g.model_count().unwrap(), g.node_count()), (plain.model_count().unwrap(), plain.node_count()));
        let bare = engine.project_over(&f, &into, local_of, None).unwrap();
        assert!(bare.weights().is_none());
        assert_eq!((bare.model_count().unwrap(), bare.node_count()), (plain.model_count().unwrap(), plain.node_count()));
    }

    #[test]
    fn a_variable_dropped_outside_the_summed_levels_is_refused() {
        // Balanced over 8: the subtree over x1 to x4 summed out.
        let vtree = Arc::new(Vtree::balanced(8));
        let f = compile_clauses(&vtree, &[vec![1, 2, 5], vec![3, -4, 6], vec![-5, 7, 8]]);
        let (left, right) = vtree.children(vtree.root());
        assert_eq!(first_leaf(&vtree, left), vtree.leaf_of(VarId(1)).unwrap());
        let g = summed(&f, left, Arithmetic::ExactRational);
        let engine = Engine::new();
        let mut keep = kept_around(&vtree, left);
        keep.retain(|&v| v != VarId(7));
        let local_of = local_in(&keep);
        let into = Arc::new(vtree.project_to_vars(local_of, keep.len() as u32).unwrap());
        let store = store_over(&keep, Arithmetic::ExactRational);
        assert_eq!(engine.project_over(&g, &into, local_of, Some(store)).unwrap_err(), EmbedError::Unsummed { variable: VarId(7) });
        // Nothing kept on the right: the root goes, and with it the right side.
        let keep = [VarId(1), VarId(3)];
        let local_of = local_in(&keep);
        let into = Arc::new(vtree.project_to_vars(local_of, 2).unwrap());
        let store = store_over(&keep, Arithmetic::ExactRational);
        assert!(matches!(
            engine.project_over(&g, &into, local_of, Some(store)).unwrap_err(),
            EmbedError::Unsummed { variable } if vtree.in_subtree(vtree.leaf_of(variable).unwrap(), right),
        ));
    }

    #[test]
    fn values_need_a_store_that_agrees_with_the_source() {
        let vtree = Arc::new(Vtree::balanced(4));
        let f = compile_clauses(&vtree, &[vec![1, 2], vec![-2, 3, 4]]);
        let g = summed(&f, vtree.root(), Arithmetic::ExactRational);
        let keep = kept_around(&vtree, vtree.root());
        let local_of = local_in(&keep);
        let into = Arc::new(vtree.project_to_vars(local_of, keep.len() as u32).unwrap());
        let engine = Engine::new();
        assert!(matches!(
            engine.project_over(&g, &into, local_of, None).unwrap_err(),
            EmbedError::SourceWeights(TddBuildError::WeightedLevelWithoutStore { .. }),
        ));
        let swapped: Vec<VarId> = keep.iter().rev().copied().collect();
        assert_eq!(
            engine.project_over(&g, &into, local_of, Some(store_over(&swapped, Arithmetic::ExactRational))).unwrap_err(),
            EmbedError::SourceWeights(TddBuildError::IncompatibleWeights),
        );
        assert_eq!(
            engine.project_over(&g, &into, local_of, Some(store_over(&keep, Arithmetic::ScaledFloat))).unwrap_err(),
            EmbedError::SourceWeights(TddBuildError::IncompatibleWeights),
        );
        assert_eq!(
            engine.project_over(&g, &into, local_of, Some(store_over(&keep[..1], Arithmetic::ExactRational))).unwrap_err(),
            EmbedError::DestinationWeights(TddBuildError::MissingVariableWeight(VarId(2))),
        );
    }

    #[test]
    fn counts_and_other_vtrees_are_refused() {
        let vtree = Arc::new(Vtree::balanced(4));
        let f = compile_clauses(&vtree, &[vec![1, 2], vec![-2, 3, 4]]);
        let engine = Engine::new();
        let (left, _) = vtree.children(vtree.root());
        let mut counted = f.clone();
        engine.marginalize_levels(&mut counted, &[left]).unwrap();
        let keep = [VarId(1), VarId(2), VarId(3)];
        let local_of = local_in(&keep);
        let into = Arc::new(vtree.project_to_vars(local_of, 3).unwrap());
        assert_eq!(
            engine.project_over(&counted, &into, local_of, None).unwrap_err(),
            EmbedError::Operation(OperationError::MarginalLevel(left)),
        );
        // The restriction is x3 under the root's right; a right-linear vtree
        // over the three puts x2 and x3 together.
        let g = summed(&f, left, Arithmetic::ExactRational);
        let other = Arc::new(Vtree::linear(3));
        assert!(matches!(
            engine.project_over(&g, &other, local_of, Some(store_over(&keep, Arithmetic::ExactRational))).unwrap_err(),
            EmbedError::NotIsomorphic { .. },
        ));
        // A leaf of `into` that no variable is kept as.
        let wide = Arc::new(Vtree::balanced(4));
        assert!(matches!(
            engine.project_over(&g, &wide, local_of, Some(store_over(&[VarId(1), VarId(2), VarId(3), VarId(4)], Arithmetic::ExactRational))).unwrap_err(),
            EmbedError::NotIsomorphic { .. },
        ));
    }
}
