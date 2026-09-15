use super::*;
use crate::limits::{LimitConfig, StopAt, StopRules};
use crate::{Literal, OperationError};

/// Exercise constant and empty requests through the public transformation methods.
fn shortcut(eng: &Engine, tree: &Arc<Vtree>, f: Tdd, case: usize) -> Result<Tdd, OperationError> {
    match case {
        0 => eng.and(f.clone(), f),
        1 => eng.or(eng.zero(tree), f),
        2 => eng.or(f, eng.zero(tree)),
        3 => eng.condition(f, [] as [i32; 0]),
        4 => eng.condition(f, [1, -1]),
        5 => eng.condition(eng.zero(tree), [1]),
        6 => eng.condition_var(f, VarId(0), true),
        7 => eng.condition_vars(f, &[], true),
        8 => eng.exists_var(eng.zero(tree), VarId(0)),
        9 => eng.exists_vars(f, &[]),
        10 => eng.exists_vars_with_strategy(f, &[], QuantificationStrategy::Structural),
        11 => eng.restrict_to_care(eng.zero(tree), f).map(|r| r.into_tdd()),
        12 => eng.restrict_to_care(f, eng.zero(tree)).map(|r| r.into_tdd()),
        13 => eng.substitute(f, &[]),
        14 => eng.rename_vars(f, &[]),
        15 => eng.and(f, eng.zero(tree)),
        16 => eng.and(eng.zero(tree), f),
        17 => eng.and(eng.zero(tree), eng.zero(tree)),
        _ => unreachable!(),
    }
}

#[test]
fn constant_and_empty_transforms_honor_entry_stops_and_recover() {
    for n in [1, 3] {
        let tree = Arc::new(Vtree::balanced(n));
        let f = Tdd::literal(&tree, 1).unwrap();
        assert_canonical(&f);
        for case in 0..18 {
            let eng = Engine::new();
            let expected = shortcut(&eng, &tree, f.clone(), case).unwrap();
            {
                let _scope = eng.limits().scope(LimitConfig::none().with_stop_rules(StopRules {
                    unconditional: Some(StopAt::WorkUnits(0)),
                    after_pairs: None,
                }));
                assert_eq!(shortcut(&eng, &tree, f.clone(), case).err(), Some(OperationError::Stopped), "{n} leaves, case {case}");
            }
            let recovered = shortcut(&eng, &tree, f.clone(), case).unwrap();
            assert_canonical(&recovered);
            assert!(eng.equivalent(&recovered, &expected).unwrap());
        }
    }
}

#[test]
fn conditioning_polls_while_reading_an_assignment() {
    let eng = Engine::new();
    let tree = Arc::new(Vtree::balanced(1));
    let f = eng.literal(&tree, 1).unwrap();
    assert_canonical(&f);
    eng.limits().pin_reduce_poll_stride(Some(1));
    let read = std::cell::Cell::new(0);
    {
        let _scope = eng.limits().scope(LimitConfig::none().with_stop_rules(StopRules {
            unconditional: Some(StopAt::WorkUnits(3)),
            after_pairs: None,
        }));
        let assignment = (0..100).map(|_| { read.set(read.get() + 1); Literal::pos(VarId(0)) });
        assert_eq!(eng.condition(f.clone(), assignment).err(), Some(OperationError::Stopped));
        assert!(read.get() < 100);
    }
    assert_canonical(&eng.condition(f, [1]).unwrap());
}

/// Retain an unreachable twin at the root to exercise identity-result canonicity.
fn with_unreachable_twin(mut f: Tdd) -> Tdd {
    let root = f.output().vtree;
    let node = f.levels[root.idx()].nodes[f.output().local.idx()];
    f.levels[root.idx()].nodes.push(node);
    let mut checked = f.clone();
    checked.minimize().unwrap();
    assert_canonical(&checked);
    assert!(checked.level(root).nodes.len() < f.level(root).nodes.len());
    f
}

#[test]
fn empty_substitution_and_renaming_preserve_storage_without_reserving() {
    let eng = Engine::new();
    let tree = Arc::new(Vtree::balanced(4));
    let f = with_unreachable_twin(eng.clause(&tree, [1, 3]).unwrap());
    for rename in [false, true] {
        let input = f.clone();
        let root = input.output().vtree;
        let ptr = input.level(root).nodes.as_ptr();
        let result = {
            let _scope = eng.limits().scope(LimitConfig::none().with_memory_budget_bytes(Some(0)));
            if rename { eng.rename_vars(input, &[]) } else { eng.substitute(input, &[]) }
        }.unwrap();
        assert_eq!(result.level(root).nodes.as_ptr(), ptr);
        assert_eq!(result.level(root).nodes, f.level(root).nodes);
        assert!(eng.equivalent(&result, &f).unwrap());
    }
}

#[test]
fn empty_replacement_maps_still_reject_discarded_structure() {
    let eng = Engine::new();
    let tree = Arc::new(Vtree::balanced(2));
    let mut f = eng.literal(&tree, 1).unwrap();
    eng.marginalize_levels(&mut f, &[tree.root()]).unwrap();
    assert!(matches!(eng.substitute(f.clone(), &[]), Err(OperationError::MarginalLevel(_))));
    assert!(matches!(eng.rename_vars(f, &[]), Err(OperationError::MarginalLevel(_))));
}

#[test]
fn composition_minimizes_nonminimal_operands_on_identity_paths() {
    let eng = Engine::new();
    let tree = Arc::new(Vtree::balanced(4));
    let f = with_unreachable_twin(eng.clause(&tree, [1, 3]).unwrap());
    let mut zero = f.clone();
    zero.output.local = crate::diagram::ZERO;
    for how in [QuantificationStrategy::Automatic, QuantificationStrategy::Structural] {
        for vars in [vec![], vec![VarId(0)]] {
            for input in [&f, &zero] {
                let result = eng.and_exists_with_strategy(input.clone(), input.clone(), &vars, how).unwrap();
                assert_canonical(&result);
                let expected = eng.exists_vars_with_strategy(input.clone(), &vars, how).unwrap();
                assert!(eng.equivalent(&result, &expected).unwrap());
            }
        }
    }
    for condition in [eng.zero(&tree), eng.one(&tree), f.clone()] {
        for otherwise in [zero.clone(), f.clone()] {
            let result = eng.ite(condition.clone(), f.clone(), otherwise.clone()).unwrap();
            assert_canonical(&result);
            for row in 0..16 {
                let assignment: Vec<_> = (0..4).map(|v| row & (1 << v) != 0).collect();
                assert_eq!(eval(&result, &assignment), eval(if eval(&condition, &assignment) { &f } else { &otherwise }, &assignment));
            }
        }
    }
}

#[test]
fn nonempty_substitution_minimizes_false_inputs_with_unreachable_storage() {
    let eng = Engine::new();
    let tree = Arc::new(Vtree::balanced(4));
    let mut f = with_unreachable_twin(eng.literal(&tree, 1).unwrap());
    f.output.local = crate::diagram::ZERO;
    let replacement = eng.literal(&tree, 2).unwrap();
    for result in [
        eng.substitute(f.clone(), &[(VarId(0), &replacement)]).unwrap(),
        eng.rename_vars(f, &[(VarId(0), VarId(1))]).unwrap(),
    ] {
        assert!(result.is_zero());
        assert_canonical(&result);
    }
}
