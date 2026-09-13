use num_traits::Zero;
use tididi::diagram::RationalWeights;
use tididi::query::{
    KeepAllColumns, KeepFrontier, ModelCounter, PinSemantics, Retention, evaluate,
};
use tididi::vtree::VarId;
use tididi::{Engine, Literal, Tdd};

use super::support::*;

/// A constrained theory on three variables; the fourth variable is free.
fn theory(row: usize) -> bool {
    (bit(row, 0) || bit(row, 1)) && (!bit(row, 0) || bit(row, 2))
}

/// A query with models both inside and outside the theory.
fn query(row: usize) -> bool {
    bit(row, 1) ^ bit(row, 2)
}

#[test]
fn changing_weights_and_evidence_match_exact_assignment_sums() {
    let engine = Engine::new();
    for (shape, tree) in trees(4).iter().enumerate() {
        let theory_diagram = engine
            .and(
                engine.clause(tree, [1, 2]).unwrap(),
                engine.clause(tree, [-1, 3]).unwrap(),
            )
            .unwrap();
        let query_diagram = engine
            .xor(
                engine.literal(tree, 2).unwrap(),
                engine.literal(tree, 3).unwrap(),
            )
            .unwrap();
        let joint = engine.and(theory_diagram.clone(), query_diagram).unwrap();
        // Return to no evidence, repeat an observation, contradict it, and pin a free variable.
        let observations: &[&[(usize, bool)]] = &[
            &[],
            &[(0, true)],
            &[(0, true)],
            &[(1, false), (2, true)],
            &[(0, true), (0, false)],
            &[(3, false)],
            &[],
        ];
        let compiled: Vec<_> = observations
            .iter()
            .map(|observation| {
                let mut evidence = Tdd::one(tree);
                for &(var, value) in *observation {
                    evidence = engine
                        .and(
                            evidence,
                            engine
                                .literal(tree, Literal::new(VarId(var as u32), value))
                                .unwrap(),
                        )
                        .unwrap();
                }
                (
                    engine
                        .and(theory_diagram.clone(), evidence.clone())
                        .unwrap(),
                    engine.and(joint.clone(), evidence).unwrap(),
                )
            })
            .collect();
        // All combinations of boundary and interior priors, then repeat the first table.
        for setting in (0..27).chain(std::iter::once(0)) {
            let values = [fraction(0, 1), fraction(1, 3), fraction(1, 1)];
            let weights = bernoulli(&[
                values[setting % 3].clone(),
                values[(setting / 3) % 3].clone(),
                values[setting / 9].clone(),
                fraction(2, 5),
            ]);
            let algebra = RationalWeights::from_literals(&weights);
            for (step, (observation, (evidence_diagram, observed_joint))) in
                observations.iter().zip(&compiled).enumerate()
            {
                let context = format!("shape {shape}, weights {setting}, observation {step}");
                let evidence_truth: Vec<_> = (0..16)
                    .map(|row| {
                        theory(row) && observation.iter().all(|&(v, value)| bit(row, v) == value)
                    })
                    .collect();
                let joint_truth: Vec<_> = (0..16)
                    .map(|row| evidence_truth[row] && query(row))
                    .collect();
                let expected_evidence = mass(&evidence_truth, &weights);
                let expected_joint = mass(&joint_truth, &weights);
                let got_evidence = evaluate(evidence_diagram, &algebra);
                let got_joint = evaluate(observed_joint, &algebra);
                assert_eq!(got_evidence, expected_evidence, "{context}, evidence mass");
                assert_eq!(got_joint, expected_joint, "{context}, joint mass");
                assert_eq!(
                    normalized(got_joint, got_evidence),
                    normalized(expected_joint.clone(), expected_evidence.clone()),
                    "{context}, conditional"
                );

                // The same observation through literal weights must retain the consistent prior.
                let mut observed_weights = weights.clone();
                for &(var, value) in *observation {
                    if value {
                        observed_weights[var].negative.set_zero();
                    } else {
                        observed_weights[var].positive.set_zero();
                    }
                }
                let observed_algebra = RationalWeights::from_literals(&observed_weights);
                assert_eq!(
                    evaluate(&theory_diagram, &observed_algebra),
                    expected_evidence,
                    "{context}, weighted evidence"
                );
                assert_eq!(
                    evaluate(&joint, &observed_algebra),
                    expected_joint,
                    "{context}, weighted query"
                );
            }
        }
    }
}

/// Change pins on a retained counter and compare each read with direct enumeration.
fn pin_sequence<R: Retention>(engine: &Engine, diagram: &Tdd, convention: PinSemantics) {
    let mut counter = ModelCounter::<R>::try_new(engine, diagram, 4, convention).unwrap();
    let mut pins = [None; 4];
    for change in [
        None,
        Some((0, Some(true))),
        Some((1, Some(false))),
        Some((0, Some(false))),
        Some((0, Some(false))),
        Some((3, Some(true))),
        Some((1, None)),
        Some((0, None)),
        Some((3, None)),
    ] {
        if let Some((var, value)) = change {
            pins[var] = value;
            counter.set_pin(VarId(var as u32), value);
        }
        let evidence_count = (0..16)
            .filter(|&row| {
                theory(row)
                    && pins
                        .iter()
                        .enumerate()
                        .all(|(v, pin)| pin.is_none_or(|value| bit(row, v) == value))
            })
            .count();
        let expected = match convention {
            PinSemantics::Evidence => evidence_count,
            PinSemantics::Cofactor => {
                evidence_count << pins.iter().filter(|pin| pin.is_some()).count()
            }
            _ => unreachable!("the fixture only selects evidence or cofactor semantics"),
        };
        for _ in 0..2 {
            assert_eq!(
                counter.try_model_count(engine).unwrap(),
                expected.into(),
                "pins {pins:?}, {convention:?}"
            );
        }
    }
}

#[test]
fn incremental_observations_match_enumeration_after_changes_and_resets() {
    let engine = Engine::new();
    for tree in trees(4) {
        let diagram = compile(&engine, &tree, &[VarId(0), VarId(1), VarId(2)], theory);
        for convention in [PinSemantics::Evidence, PinSemantics::Cofactor] {
            pin_sequence::<KeepAllColumns>(&engine, &diagram, convention);
            pin_sequence::<KeepFrontier>(&engine, &diagram, convention);
        }
    }
}
