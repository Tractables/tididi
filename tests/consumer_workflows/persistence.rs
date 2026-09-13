use std::collections::BTreeMap;
use std::io::Cursor;
use std::sync::Arc;

use num_rational::BigRational;
use tididi::diagram::{Arithmetic, LiteralWeights, RationalWeights, WeightStore};
use tididi::io::{read_tdd, write_tdd};
use tididi::query::evaluate;
use tididi::vtree::VarId;
use tididi::{Engine, Vtree};

use super::support::*;

#[test]
fn named_roots_and_weight_metadata_survive_a_fresh_context() {
    let engine = Engine::new();
    for (shape, tree) in trees(4).iter().enumerate() {
        // The caller stores names and weights separately, in an order different
        // from variable IDs and from the vtree's leaf order.
        let variables = [
            (VarId(2), "rain", fraction(1, 5)),
            (VarId(0), "sprinkler", fraction(1, 3)),
            (VarId(3), "sensor", fraction(3, 4)),
            (VarId(1), "wind", fraction(2, 5)),
        ];
        let mut weights = bernoulli(&vec![fraction(0, 1); 4]);
        let mut metadata = String::new();
        for (var, name, probability) in &variables {
            weights[var.0 as usize] = bernoulli(std::slice::from_ref(probability)).remove(0);
            let weight = &weights[var.0 as usize];
            metadata.push_str(&format!(
                "{}\t{name}\t{}\t{}\n",
                var.0, weight.negative, weight.positive
            ));
        }
        let root_truths = BTreeMap::from([
            (
                "theory",
                (0..16)
                    .map(|row| (bit(row, 2) || bit(row, 0)) && bit(row, 3))
                    .collect::<Vec<_>>(),
            ),
            (
                "query",
                (0..16).map(|row| bit(row, 2) ^ bit(row, 3)).collect(),
            ),
            ("true", vec![true; 16]),
            ("false", vec![false; 16]),
        ]);
        let mut encoded_roots = BTreeMap::new();
        for (name, truth) in &root_truths {
            let mut diagram = compile(
                &engine,
                tree,
                &[VarId(0), VarId(1), VarId(2), VarId(3)],
                |row| truth[row],
            );
            diagram
                .set_weights(WeightStore::new(
                    RationalWeights::from_literals(&weights),
                    Arithmetic::ExactRational,
                ))
                .unwrap();
            let mut bytes = Vec::new();
            write_tdd(&mut bytes, &diagram).unwrap();
            encoded_roots.insert(*name, bytes);
        }

        let restored_tree = Arc::new(Vtree::from_text(&tree.to_text()).unwrap());
        assert!(!Arc::ptr_eq(tree, &restored_tree));
        let mut names = BTreeMap::new();
        let mut restored_weights = vec![None; restored_tree.num_vars() as usize];
        for line in metadata.lines().rev() {
            let fields: Vec<_> = line.split('\t').collect();
            let var = VarId(fields[0].parse().unwrap());
            assert!(names.insert(fields[1].to_owned(), var).is_none());
            restored_weights[var.0 as usize] = Some(LiteralWeights {
                negative: fields[2].parse::<BigRational>().unwrap(),
                positive: fields[3].parse::<BigRational>().unwrap(),
            });
        }
        let restored_weights: Vec<_> = restored_weights.into_iter().map(Option::unwrap).collect();
        for (var, name, _) in &variables {
            assert_eq!(names[*name], *var);
        }
        assert_eq!(restored_weights, weights);
        let algebra = RationalWeights::from_literals(&restored_weights);
        let store = WeightStore::new(algebra.clone(), Arithmetic::ExactRational);
        let mut restored_roots = BTreeMap::new();
        for (name, bytes) in &encoded_roots {
            let mut diagram = read_tdd(&mut Cursor::new(bytes), &restored_tree).unwrap();
            assert!(Arc::ptr_eq(diagram.vtree(), &restored_tree));
            assert!(diagram.weights().is_none()); // the structural format omits weights
            let expected = &root_truths[name];
            assert_truth(
                &engine,
                &diagram,
                expected,
                &format!("shape {shape}, restored {name}"),
            );
            diagram.set_weights(store.clone()).unwrap();
            let expected_mass = mass(expected, &weights);
            assert_eq!(evaluate(&diagram, &algebra), expected_mass);
            assert_eq!(
                engine
                    .weighted_value(&diagram)
                    .unwrap()
                    .unwrap()
                    .as_rational()
                    .as_ref(),
                &expected_mass
            );
            let expected_support: Vec<_> = (0..4)
                .filter(|&var| (0..16).any(|row| expected[row] != expected[row ^ (1 << var)]))
                .map(VarId)
                .collect();
            assert_eq!(engine.support(&diagram).unwrap(), expected_support);
            restored_roots.insert(*name, diagram);
        }

        // Use names to construct a fresh query and combine roots sharing the reloaded vtree.
        let named_query = engine
            .xor(
                engine
                    .literal(&restored_tree, tididi::Literal::pos(names["rain"]))
                    .unwrap(),
                engine
                    .literal(&restored_tree, tididi::Literal::pos(names["sensor"]))
                    .unwrap(),
            )
            .unwrap();
        assert!(
            engine
                .equivalent(&named_query, &restored_roots["query"])
                .unwrap()
        );
        let joint = engine
            .and(
                restored_roots["theory"].clone(),
                restored_roots["query"].clone(),
            )
            .unwrap();
        let joint_truth: Vec<_> = (0..16)
            .map(|row| root_truths["theory"][row] && root_truths["query"][row])
            .collect();
        assert_truth(
            &engine,
            &joint,
            &joint_truth,
            &format!("shape {shape}, restored conjunction"),
        );
        assert_eq!(
            engine
                .weighted_value(&joint)
                .unwrap()
                .unwrap()
                .as_rational()
                .as_ref(),
            &mass(&joint_truth, &weights)
        );
    }
}
