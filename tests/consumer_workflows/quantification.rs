use tididi::vtree::VarId;
use tididi::{Engine, Tdd};

use super::support::*;

/// Universal abstraction expressed through the existing public Boolean operations.
fn forall(engine: &Engine, diagram: Tdd, var: VarId) -> Tdd {
    let negated = engine.negate(diagram).unwrap();
    let projected = engine
        .exists_vars(negated, &[var])
        .unwrap();
    engine.negate(projected).unwrap()
}

#[test]
fn alternating_quantifiers_match_all_three_variable_games() {
    let engine = Engine::new();
    let mut differing_orders = 0;
    for (shape, tree) in trees(4).iter().enumerate() {
        // Variables: state, environment action, controller response, and one free bit.
        for table in 0..256usize {
            let accepts = |state: usize, action: usize, response: usize| {
                table & (1usize << (state | (action << 1) | (response << 2))) != 0
            };
            let relation = compile(&engine, tree, &[VarId(0), VarId(1), VarId(2)], |row| {
                table & (1 << row) != 0
            });
            let responsive = forall(
                &engine,
                engine
                    .exists_vars(
                        relation.clone(),
                        &[VarId(2)],
                    )
                    .unwrap(),
                VarId(1),
            );
            let committed = engine
                .exists_vars(
                    forall(&engine, relation, VarId(1)),
                    &[VarId(2)],
                )
                .unwrap();

            let responsive_truth: Vec<_> = (0..16)
                .map(|row| {
                    (0..2).all(|action| (0..2).any(|response| accepts(row & 1, action, response)))
                })
                .collect();
            let committed_truth: Vec<_> = (0..16)
                .map(|row| {
                    (0..2).any(|response| (0..2).all(|action| accepts(row & 1, action, response)))
                })
                .collect();
            assert_truth(
                &engine,
                &responsive,
                &responsive_truth,
                &format!("shape {shape}, game {table:#04x}, forall-exists"),
            );
            assert_truth(
                &engine,
                &committed,
                &committed_truth,
                &format!("shape {shape}, game {table:#04x}, exists-forall"),
            );
            assert_eq!(
                engine.equivalent(&responsive, &committed).unwrap(),
                responsive_truth == committed_truth
            );
            if responsive_truth != committed_truth {
                differing_orders += 1;
            }
        }
    }
    assert!(
        differing_orders > 0,
        "the fixtures must distinguish quantifier order"
    );
}
