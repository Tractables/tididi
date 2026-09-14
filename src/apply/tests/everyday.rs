use super::*;
use crate::diagram::{Arithmetic, RationalWeights, WeightStore};
use crate::limits::{LimitConfig, StopAt, StopRules};
use crate::{Literal, OperationError};

fn function(tree: &Arc<Vtree>, bits: u16, reverse: bool) -> Tdd {
    let eng = Engine::new();
    let n = tree.num_leaves() as usize;
    let mut result = Tdd::zero(tree);
    for i in 0..1usize << n {
        let row = if reverse { (1 << n) - 1 - i } else { i };
        if bits & (1 << row) != 0 {
            let cube = eng
                .cube(
                    tree,
                    (0..n).map(|v| Literal::new(VarId(v as u32), row & (1 << v) != 0)),
                )
                .unwrap();
            result = eng.or(result, cube).unwrap();
        }
    }
    result
}
fn assignment(row: usize, n: usize) -> Vec<bool> {
    (0..n).map(|v| row & (1 << v) != 0).collect()
}

#[test]
fn comparisons_support_and_witnesses_match_all_two_variable_truth_tables() {
    let eng = Engine::new();
    let tree = Arc::new(Vtree::balanced(2));
    let fs: Vec<_> = (0..16).map(|bits| function(&tree, bits, false)).collect();
    for (bits, f) in fs.iter().enumerate() {
        let other = function(&tree, bits as u16, true);
        assert!(eng.equivalent(f, &other).unwrap());
        let support: Vec<_> = (0..2)
            .filter(|v| (0..4).any(|r| ((bits >> r) & 1) != ((bits >> (r ^ (1 << v))) & 1)))
            .map(VarId)
            .collect();
        assert_eq!(eng.support(f).unwrap(), support);
        match eng.satisfying_assignment(f).unwrap() {
            None => assert_eq!(bits, 0),
            Some(model) => {
                assert_eq!(model.len(), 2);
                assert_eq!(
                    model.iter().map(|l| l.var).collect::<Vec<_>>(),
                    vec![VarId(0), VarId(1)]
                );
                let row = model
                    .iter()
                    .fold(0usize, |a, l| a | ((l.positive as usize) << l.var.0));
                assert_ne!(bits & (1 << row), 0);
            }
        }
        for (other_bits, g) in fs.iter().enumerate() {
            assert_eq!(eng.equivalent(f, g).unwrap(), bits == other_bits);
            assert_eq!(eng.implies(f, g).unwrap(), bits & !other_bits == 0);
            let xor = eng.xor(f.clone(), g.clone()).unwrap();
            assert_canonical(&xor);
            for row in 0..4 {
                assert_eq!(
                    eval(&xor, &assignment(row, 2)),
                    (bits ^ other_bits) & (1 << row) != 0
                );
            }
        }
    }
}

#[test]
fn equivalence_handles_different_construction_and_unminimized_operands() {
    let eng = Engine::new();
    for tree in [
        Vtree::balanced(3),
        Vtree::linear(3),
        Vtree::reverse_linear(3),
    ] {
        let tree = Arc::new(tree);
        for bits in 0..256u16 {
            let dnf = function(&tree, bits, false);
            let mut cnf = Tdd::one(&tree);
            for row in 0..8 {
                if bits & (1 << row) == 0 {
                    let clause = eng
                        .clause(
                            &tree,
                            (0..3).map(|v| Literal::new(VarId(v), row & (1 << v) == 0)),
                        )
                        .unwrap();
                    cnf = eng.and(cnf, clause).unwrap();
                }
            }
            assert!(eng.equivalent(&cnf, &dnf).unwrap(), "{bits}");
            assert_eq!(eng.support(&cnf).unwrap(), eng.support(&dnf).unwrap());
        }
    }
}

#[test]
fn substitution_is_simultaneous_for_cycles_identification_and_functions() {
    let eng = Engine::new();
    for tree in [Vtree::balanced(3), Vtree::linear(3)] {
        let tree = Arc::new(tree);
        let replacements = [
            eng.clause(&tree, [2, 3]).unwrap(),
            eng.literal(&tree, -1).unwrap(),
        ];
        for bits in [0, 1, 42, 85, 150, 254, 255] {
            let f = function(&tree, bits, false);
            for renames in [
                vec![(VarId(0), VarId(1)), (VarId(1), VarId(0))],
                vec![
                    (VarId(0), VarId(1)),
                    (VarId(1), VarId(2)),
                    (VarId(2), VarId(0)),
                ],
                vec![(VarId(0), VarId(1)), (VarId(2), VarId(1))],
            ] {
                let g = eng.rename_vars(f.clone(), &renames).unwrap();
                assert_canonical(&g);
                for row in 0..8 {
                    let destination = assignment(row, 3);
                    let mut source = destination.clone();
                    for &(from, to) in &renames {
                        source[from.idx()] = destination[to.idx()];
                    }
                    assert_eq!(eval(&g, &destination), eval(&f, &source));
                }
            }
            let g = eng
                .substitute(
                    f.clone(),
                    &[(VarId(0), &replacements[0]), (VarId(1), &replacements[1])],
                )
                .unwrap();
            assert_canonical(&g);
            for row in 0..8 {
                let dest = assignment(row, 3);
                assert_eq!(
                    eval(&g, &dest),
                    eval(&f, &[dest[1] || dest[2], !dest[0], dest[2]])
                );
            }
            for constant in [Tdd::one(&tree), Tdd::zero(&tree)] {
                let g = eng.substitute(f.clone(), &[(VarId(0), &constant)]).unwrap();
                for row in 0..8 {
                    let dest = assignment(row, 3);
                    assert_eq!(
                        eval(&g, &dest),
                        eval(&f, &[!constant.is_zero(), dest[1], dest[2]])
                    );
                }
            }
        }
    }
}

#[test]
fn ite_and_existential_conjunction_match_enumeration() {
    let eng = Engine::new();
    let tree = Arc::new(Vtree::balanced(3));
    for (a, b, c) in [(42, 97, 130), (255, 16, 0), (0, 100, 150), (150, 85, 170)] {
        let (f, g, h) = (
            function(&tree, a, false),
            function(&tree, b, false),
            function(&tree, c, false),
        );
        let ite = eng.ite(f.clone(), g.clone(), h).unwrap();
        assert_canonical(&ite);
        for row in 0..8 {
            let mask = 1 << row;
            assert_eq!(
                eval(&ite, &assignment(row, 3)),
                if a & mask != 0 {
                    b & mask != 0
                } else {
                    c & mask != 0
                }
            );
        }
        for how in [
            QuantificationStrategy::Automatic,
            QuantificationStrategy::Structural,
        ] {
            for vars in [
                vec![],
                vec![VarId(0)],
                vec![VarId(0), VarId(2)],
                vec![VarId(0), VarId(0)],
            ] {
                let result = eng.and_exists(f.clone(), g.clone(), &vars, how).unwrap();
                assert_canonical(&result);
                for row in 0..8 {
                    let expected = (0..8).any(|witness| {
                        (0..3).all(|v| {
                            vars.contains(&VarId(v)) || row & (1 << v) == witness & (1 << v)
                        }) && (a & b) & (1 << witness) != 0
                    });
                    assert_eq!(eval(&result, &assignment(row, 3)), expected);
                }
            }
        }
    }
}

#[test]
fn symbolic_reachability_uses_image_rename_and_semantic_convergence() {
    let eng = Engine::new();
    let tree = Arc::new(Vtree::balanced(4)); // two current-state bits, two next-state bits
    let relation = eng
        .or(
            eng.cube(&tree, [-1, -2, 3, -4]).unwrap(),
            eng.cube(&tree, [1, -2, -3, 4]).unwrap(),
        )
        .unwrap();
    let mut reached = eng.cube(&tree, [-1, -2]).unwrap(); // 0 -> 1 -> 2
    let mut steps = 0;
    loop {
        let image = eng
            .and_exists(
                reached.clone(),
                relation.clone(),
                &[VarId(0), VarId(1)],
                QuantificationStrategy::Automatic,
            )
            .unwrap();
        let image = eng
            .rename_vars(image, &[(VarId(2), VarId(0)), (VarId(3), VarId(1))])
            .unwrap();
        let next = eng.or(reached.clone(), image).unwrap();
        steps += 1;
        if eng.equivalent(&next, &reached).unwrap() {
            break;
        }
        assert!(steps <= 3);
        reached = next;
    }
    assert_eq!(steps, 3);
    for row in 0..16 {
        assert_eq!(eval(&reached, &assignment(row, 4)), row & 3 != 3);
    }
    let witness = eng.satisfying_assignment(&reached).unwrap().unwrap();
    assert!(
        eng.implies(&eng.cube(&tree, witness).unwrap(), &reached)
            .unwrap()
    );
}

#[test]
fn sparse_ids_and_single_leaf_diagrams_work() {
    let eng = Engine::new();
    for tree in [
        Vtree::leaf(VarId(19)),
        Vtree::balanced_over(&[VarId(19), VarId(2)]),
    ] {
        let tree = Arc::new(tree);
        let x = eng.literal(&tree, Literal::pos(VarId(19))).unwrap();
        assert_eq!(eng.support(&x).unwrap(), vec![VarId(19)]);
        assert!(
            eng.equivalent(&x, &eng.literal(&tree, 20).unwrap())
                .unwrap()
        );
        let neg = eng.literal(&tree, -20).unwrap();
        let g = eng.substitute(x.clone(), &[(VarId(19), &neg)]).unwrap();
        assert!(eng.equivalent(&g, &neg).unwrap());
        let model = eng.satisfying_assignment(&x).unwrap().unwrap();
        assert_eq!(model.len(), tree.num_leaves() as usize);
        assert!(model.contains(&Literal::pos(VarId(19))));
    }
}

#[test]
fn boolean_queries_ignore_weights_and_substitution_keeps_destination_weights() {
    let eng = Engine::new();
    let tree = Arc::new(Vtree::balanced(2));
    let mut x = eng.literal(&tree, 1).unwrap();
    let mut same = x.clone();
    x.set_weights(WeightStore::new(
        RationalWeights::unit(2),
        Arithmetic::ExactRational,
    ))
    .unwrap();
    same.set_weights(WeightStore::new(
        RationalWeights::unit(2),
        Arithmetic::SignedLog,
    ))
    .unwrap();
    assert!(eng.equivalent(&x, &same).unwrap());
    assert!(eng.implies(&x, &same).unwrap());
    assert_eq!(eng.support(&x).unwrap(), vec![VarId(0)]);
    assert_eq!(
        eng.satisfying_assignment(&x).unwrap(),
        eng.satisfying_assignment(&same).unwrap()
    );
    let result = eng.substitute(x.clone(), &[(VarId(0), &same)]).unwrap();
    assert_eq!(
        result.weights().unwrap().arithmetic(),
        Arithmetic::ExactRational
    );
    assert!(eng.equivalent(&x, &result).unwrap());
    assert_eq!(
        eng.xor(x.clone(), same.clone()).unwrap_err(),
        OperationError::IncompatibleWeights
    );
    let y = eng.literal(&tree, 2).unwrap();
    let xor = eng.xor(x.clone(), y.clone()).unwrap();
    let ite = eng.ite(y.clone(), y, x.clone()).unwrap();
    assert_eq!(
        xor.weights().unwrap().arithmetic(),
        Arithmetic::ExactRational
    );
    assert_eq!(
        ite.weights().unwrap().arithmetic(),
        Arithmetic::ExactRational
    );
    let renamed = eng.rename_vars(x, &[(VarId(0), VarId(1))]).unwrap();
    assert_eq!(
        renamed.weights().unwrap().arithmetic(),
        Arithmetic::ExactRational
    );
}

#[test]
fn invalid_maps_and_marginal_inputs_are_rejected_even_for_constants() {
    let eng = Engine::new();
    let tree = Arc::new(Vtree::balanced(4));
    let x = eng.literal(&tree, 1).unwrap();
    for f in [Tdd::zero(&tree), Tdd::one(&tree)] {
        assert_eq!(
            eng.rename_vars(f.clone(), &[(VarId(0), VarId(4))])
                .unwrap_err(),
            OperationError::VariableNotInVtree(VarId(4))
        );
        assert_eq!(
            eng.rename_vars(f.clone(), &[(VarId(0), VarId(1)), (VarId(0), VarId(2))])
                .unwrap_err(),
            OperationError::DuplicateVariable(VarId(0))
        );
        assert_eq!(
            eng.substitute(f.clone(), &[(VarId(0), &x), (VarId(0), &x)])
                .unwrap_err(),
            OperationError::DuplicateVariable(VarId(0))
        );
        assert_eq!(
            eng.and_exists(f, x.clone(), &[VarId(9)], QuantificationStrategy::Automatic)
                .unwrap_err(),
            OperationError::VariableNotInVtree(VarId(9))
        );
    }
    let mut marginal = x.clone();
    let leaf = tree.leaf_of(VarId(0)).unwrap();
    crate::marginal::marginalize_levels(&eng, &mut marginal, &[leaf]).unwrap();
    let error = OperationError::MarginalLevel(leaf);
    assert_eq!(eng.equivalent(&marginal, &marginal).unwrap_err(), error);
    assert_eq!(eng.implies(&marginal, &x).unwrap_err(), error);
    assert_eq!(eng.support(&marginal).unwrap_err(), error);
    assert_eq!(eng.satisfying_assignment(&marginal).unwrap_err(), error);
    assert_eq!(eng.rename_vars(marginal.clone(), &[]).unwrap_err(), error);
    assert_eq!(
        eng.substitute(x.clone(), &[(VarId(0), &marginal)])
            .unwrap_err(),
        error
    );
    assert_eq!(
        eng.ite(Tdd::zero(&tree), x.clone(), marginal.clone())
            .unwrap_err(),
        error
    );
    assert_eq!(
        eng.xor(Tdd::zero(&tree), marginal.clone()).unwrap_err(),
        error
    );
    assert_eq!(
        eng.and_exists(
            Tdd::zero(&tree),
            marginal,
            &[],
            QuantificationStrategy::Automatic
        )
        .unwrap_err(),
        error
    );
    let other = Tdd::one(&Arc::new(Vtree::balanced(4)));
    assert_eq!(
        eng.equivalent(&x, &other).unwrap_err(),
        OperationError::VtreeMismatch
    );
    assert_eq!(
        eng.substitute(x, &[(VarId(0), &other)]).unwrap_err(),
        OperationError::VtreeMismatch
    );
}

fn exercise(eng: &Engine, op: usize, f: &Tdd, g: &Tdd) -> Result<(), OperationError> {
    match op {
        0 => {
            eng.literal(f.vtree(), 1)?;
        }
        1 => {
            eng.equivalent(f, g)?;
        }
        2 => {
            eng.implies(f, g)?;
        }
        3 => {
            eng.support(f)?;
        }
        4 => {
            eng.satisfying_assignment(f)?;
        }
        5 => {
            eng.ite(f.clone(), g.clone(), g.clone())?;
        }
        6 => {
            eng.xor(f.clone(), g.clone())?;
        }
        7 => {
            eng.and_exists(
                f.clone(),
                g.clone(),
                &[VarId(0)],
                QuantificationStrategy::Automatic,
            )?;
        }
        8 => {
            eng.substitute(f.clone(), &[(VarId(0), g)])?;
        }
        9 => {
            eng.rename_vars(f.clone(), &[(VarId(0), VarId(1))])?;
        }
        _ => unreachable!(),
    }
    Ok(())
}

#[test]
fn every_new_operation_recovers_from_reservation_refusals_and_work_stops() {
    let tree = Arc::new(Vtree::balanced(2));
    let f = Tdd::clause(&tree, [1, 2]);
    let g = Tdd::literal(&tree, 2);
    for op in 0..10 {
        let eng = Engine::new();
        {
            let _scope = eng
                .limits()
                .scope(LimitConfig::none().with_memory_budget_bytes(Some(0)));
            assert_eq!(
                exercise(&eng, op, &f, &g),
                Err(OperationError::OverBudget),
                "operation {op}"
            );
        }
        let mut finished = false;
        for n in 0..4096 {
            let eng = Engine::new();
            eng.limits().refuse_nth_reserve(n);
            match exercise(&eng, op, &f, &g) {
                Ok(()) => {
                    finished = true;
                    break;
                }
                Err(e) => assert_eq!(
                    e,
                    OperationError::OverBudget,
                    "operation {op}, reservation {n}"
                ),
            }
            eng.limits().grant_every_reserve();
            exercise(&eng, op, &f, &g).unwrap();
        }
        assert!(finished, "operation {op} reservation sweep did not finish");
        let eng = Engine::new();
        eng.limits().pin_reduce_poll_stride(Some(1));
        {
            let _scope = eng
                .limits()
                .scope(LimitConfig::none().with_stop_rules(StopRules {
                    unconditional: Some(StopAt::WorkUnits(2)),
                    after_pairs: None,
                }));
            assert_eq!(
                exercise(&eng, op, &f, &g),
                Err(OperationError::Stopped),
                "operation {op}"
            );
        }
        exercise(&eng, op, &f, &g).unwrap();
    }
    assert_eq!(f.model_count(), 3u32.into());
    assert_eq!(g.model_count(), 2u32.into());
}

#[test]
fn composed_operations_propagate_output_caps() {
    let tree = Arc::new(Vtree::balanced(4));
    let f = Tdd::clause(&tree, [1, 3]);
    let g = Tdd::clause(&tree, [2, 4]);
    for op in [0, 5, 6, 7, 8, 9] {
        let eng = Engine::new();
        let _scope = eng
            .limits()
            .scope(LimitConfig::none().with_output_node_cap(Some(0)));
        assert_eq!(
            exercise(&eng, op, &f, &g),
            Err(OperationError::OutputCap),
            "operation {op}"
        );
    }
}

#[test]
fn witness_walk_handles_deep_vtrees_without_recursion() {
    let tree = Arc::new(Vtree::linear(8192));
    let eng = Engine::new();
    let f = eng.literal(&tree, 8192).unwrap();
    let model = eng.satisfying_assignment(&f).unwrap().unwrap();
    assert_eq!(model.len(), 8192);
    assert!(model[..8191].iter().all(|lit| !lit.positive));
    assert_eq!(model[8191], Literal::pos(VarId(8191)));
}

#[test]
fn renaming_changes_the_function_without_renaming_its_weight_table() {
    use crate::diagram::LiteralWeights;
    let eng = Engine::new();
    let tree = Arc::new(Vtree::balanced(2));
    let mut f = eng.literal(&tree, 1).unwrap();
    let weights = RationalWeights::from_literals(&[
        LiteralWeights {
            negative: rat(3, 4),
            positive: rat(1, 4),
        },
        LiteralWeights {
            negative: rat(1, 4),
            positive: rat(3, 4),
        },
    ]);
    f.set_weights(WeightStore::new(weights, Arithmetic::ExactRational))
        .unwrap();
    assert_eq!(
        eng.weighted_value(&f).unwrap().unwrap().into_rational(),
        rat(1, 4)
    );
    let renamed = eng.rename_vars(f, &[(VarId(0), VarId(1))]).unwrap();
    assert_eq!(
        eng.weighted_value(&renamed)
            .unwrap()
            .unwrap()
            .into_rational(),
        rat(3, 4)
    );
}

#[test]
fn renaming_and_literal_substitution_agree_for_all_three_variable_maps() {
    let eng = Engine::new();
    let tree = Arc::new(Vtree::balanced(3));
    let f = function(&tree, 0b00110110, false);
    assert_canonical(&f);
    let literals = [1, 2, 3].map(|v| eng.literal(&tree, v).unwrap());
    for literal in &literals { assert_canonical(literal); }
    for code in 0..27 {
        let targets = [code % 3, code / 3 % 3, code / 9];
        let renames = std::array::from_fn::<_, 3, _>(|i| (VarId(i as u32), VarId(targets[i] as u32)));
        let replacements = std::array::from_fn::<_, 3, _>(|i| (VarId(i as u32), &literals[targets[i]]));
        let renamed = eng.rename_vars(f.clone(), &renames).unwrap();
        let substituted = eng.substitute(f.clone(), &replacements).unwrap();
        assert_canonical(&renamed);
        assert_canonical(&substituted);
        for row in 0..8 {
            let dest = assignment(row, 3);
            let source = targets.map(|i| dest[i]);
            let expected = eval(&f, &source);
            assert_eq!(eval(&renamed, &dest), expected);
            assert_eq!(eval(&substituted, &dest), expected);
        }
    }
}

#[test]
fn rename_validates_each_entry_before_shortcuts_or_literal_construction() {
    let eng = Engine::new();
    let tree = Arc::new(Vtree::balanced(2));
    let maps = [
        (vec![(VarId(0), VarId(0)), (VarId(0), VarId(9))], OperationError::DuplicateVariable(VarId(0))),
        (vec![(VarId(0), VarId(9)), (VarId(0), VarId(0))], OperationError::VariableNotInVtree(VarId(9))),
        (vec![(VarId(8), VarId(9))], OperationError::VariableNotInVtree(VarId(8))),
    ];
    for f in [eng.one(&tree), eng.zero(&tree)] {
        assert_canonical(&f);
        let _limits = eng.limits().scope(LimitConfig::none().with_output_node_cap(Some(0)));
        for (map, error) in &maps {
            assert_eq!(eng.rename_vars(f.clone(), map).unwrap_err(), *error);
        }
    }
}
