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
        out.push(vtree.leaf_bottomup().filter(|&(t, _)| under(vtree, t, left)).map(|(_, v)| v).collect());
    }
    out
}

fn under(vtree: &Vtree, mut t: crate::vtree::VtreeIdx, root: crate::vtree::VtreeIdx) -> bool {
    loop {
        if t == root {
            return true;
        }
        match vtree.node(t).parent() {
            Some(p) => t = p,
            None => return false,
        }
    }
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
